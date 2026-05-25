"""
Kalman-filter factor ensemble.

Replaces simple weighted blending with a Dynamic Factor Model estimated
via Kalman filter.  The DFM extracts latent economic factors from observed
factor signals, providing:

- Time-varying factor loadings (H) via causal rolling-window EM
- Filtered state estimates (no look-ahead)
- Forward predictions with proper confidence intervals
- Confidence = 1/(1 + sqrt(P_diag)) from Kalman covariance matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

import src.analysis.factors as fmod
from src.analysis.kalman import DynamicFactorModel, ForwardPrediction, KalmanResult


@dataclass
class FactorEnsembleConfig:
    """Configuration for :class:`FactorEnsemble`.

    Attributes:
        n_factors:      Number of latent factors (default=2).
        window_size:    Rolling window for DFM estimation (trading days).
        window_step:    Re-estimation frequency (trading days).
        max_em_iter:    Maximum EM iterations per window.
    """

    n_factors: int = 1
    window_size: int = 1260
    window_step: int = 252
    max_em_iter: int = 20


@dataclass
class EnsembleResult:
    """Output of the KF factor ensemble.

    Attributes:
        composite:        Filtered latent state (mean across factors, daily).
        factor_signals:   Observed factor signals (input to DFM).
        factor_weights:   Factor loadings |H| norm per factor group.
        kalman_result:    Full Kalman filter result.
        forward_pred:     Forward prediction from last filtered state.
        monthly_aggregate: Monthly-mean of filtered composite.
        monthly_confidence: Monthly confidence from KF covariance.
    """

    composite: pd.Series
    factor_signals: pd.DataFrame
    factor_weights: Dict[str, float]
    kalman_result: Optional[KalmanResult] = None
    forward_pred: Optional[ForwardPrediction] = None
    monthly_aggregate: Optional[pd.Series] = None
    monthly_confidence: Optional[pd.Series] = None


class FactorEnsemble:
    """Kalman-filter based factor ensemble.

    Uses a Dynamic Factor Model to extract latent economic state from
    8 observed factor signals.  Provides filtered estimates, forward
    predictions, and proper confidence intervals.

    Args:
        config: :class:`FactorEnsembleConfig`.
    """

    def __init__(self, config: Optional[FactorEnsembleConfig] = None):
        self.config = config or FactorEnsembleConfig()
        self.dfm: Optional[DynamicFactorModel] = None

    def compute_factors(self, asset_prices, macro_indicators=None,
                        yield_short=None, yield_long=None, risk_series=None):
        return fmod._compute_all_factors(
            asset_prices=asset_prices,
            macro_indicators=macro_indicators or {},
            yield_short=yield_short, yield_long=yield_long,
            risk_series=risk_series,
        )

    def run(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> EnsembleResult:
        """Execute KF-based ensemble: compute factors → fit DFM → predict.

        Returns filtered state, forward prediction, and monthly aggregates.
        """
        raw_factors = self.compute_factors(asset_prices, macro_indicators, yield_short, yield_long, risk_series)
        if not raw_factors:
            return EnsembleResult(composite=pd.Series(dtype=float), factor_signals=pd.DataFrame(), factor_weights={})

        factor_df = pd.DataFrame(raw_factors).sort_index().dropna(how="all")
        obs_array = factor_df.values.astype(np.float64)

        self.dfm = DynamicFactorModel(
            n_factors=self.config.n_factors,
            window_size=self.config.window_size,
            window_step=self.config.window_step,
            max_em_iter=self.config.max_em_iter,
        )

        kf_result = self.dfm.fit(obs_array)
        composite = pd.Series(kf_result.filtered_state.mean(axis=1), index=factor_df.index)

        forward_pred = None
        monthly_agg = None
        monthly_conf = None

        if composite is not None and not composite.empty:
            forward_pred = self.dfm.predict(obs_array, steps=3)
            monthly_agg = composite.resample("ME").mean()
            monthly_agg.index = monthly_agg.index + pd.offsets.MonthEnd(0)
            if kf_result.confidence is not None:
                conf_series = pd.Series(kf_result.confidence, index=factor_df.index)
                monthly_conf = conf_series.resample("ME").mean()
                monthly_conf.index = monthly_conf.index + pd.offsets.MonthEnd(0)

        factor_weights = self._loadings_to_weights()

        n_factors = self.config.n_factors
        fwd_conf = forward_pred.confidence[0] if forward_pred is not None else 0.0
        logger.info(f"KF Ensemble: DFM({n_factors}f) → {len(composite)} daily obs, {len(monthly_agg) if monthly_agg is not None else 0} monthly.")
        logger.info(f"  Prediction confidence: {fwd_conf:.1%}")
        if forward_pred is not None:
            logger.info(f"  Forward pred 1mo: {forward_pred.state_mean[0,0]:+.4f} [{forward_pred.conf_lower[0,0]:+.3f}, {forward_pred.conf_upper[0,0]:+.3f}]")
        if factor_weights:
            for name, w in sorted(factor_weights.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"  loading |H[{name}]| = {w:.4f}")

        return EnsembleResult(
            composite=composite,
            factor_signals=factor_df,
            factor_weights=factor_weights,
            kalman_result=kf_result,
            forward_pred=forward_pred,
            monthly_aggregate=monthly_agg,
            monthly_confidence=monthly_conf,
        )

    def _loadings_to_weights(self) -> Dict[str, float]:
        """Convert factor loadings to display weights."""
        if self.dfm is None or self.dfm.H is None:
            return {}
        factor_names = list(DEFAULT_FACTOR_WEIGHTS.keys())
        H = self.dfm.H
        weights = {}
        for i, name in enumerate(factor_names):
            if i < H.shape[0]:
                weights[name] = round(float(np.linalg.norm(H[i])), 4)
        total = sum(weights.values()) or 1.0
        return {k: round(v / total, 4) for k, v in weights.items()}


DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "trend_momentum": 0.20, "mean_reversion": 0.10, "volatility_regime": 0.15,
    "carry_yield_curve": 0.15, "macro_diffusion": 0.15,
    "cross_asset_momentum": 0.10, "global_composite": 0.10, "credit_risk": 0.05,
}


__all__ = ["FactorEnsembleConfig", "EnsembleResult", "FactorEnsemble"]
