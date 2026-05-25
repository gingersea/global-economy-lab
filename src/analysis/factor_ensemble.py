"""
Multi-factor ensemble.

Combines factor signals (from :mod:`src.analysis.factors`) into a single
composite prediction using performance-weighted blending.

Key design principle: **monthly = integral of daily**.

- At daily frequency: individual factor signals are blended in real-time.
- At monthly frequency: daily signals are aggregated (mean) over the month,
  producing inherently higher-accuracy predictions via the law of large numbers.
- Factor weights adapt to recent predictive performance (IR-weighted).

The ensemble works at any frequency — daily, weekly, monthly — depending on
the input signal resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

import src.analysis.factors as fmod


DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "trend_momentum": 0.20,
    "mean_reversion": 0.10,
    "volatility_regime": 0.15,
    "carry_yield_curve": 0.15,
    "macro_diffusion": 0.15,
    "cross_asset_momentum": 0.10,
    "global_composite": 0.10,
    "credit_risk": 0.05,
}


@dataclass
class FactorEnsembleConfig:
    """Configuration for :class:`FactorEnsemble`.

    Attributes:
        factor_weights:    Initial weight per factor (normalised to sum=1).
        adaptation_lookback: Periods for evaluating factor performance.
        min_weight:         Floor for any factor weight after adaptation.
        smoothing:          EMA span for weight updates.
    """

    factor_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FACTOR_WEIGHTS)
    )
    adaptation_lookback: int = 60
    min_weight: float = 0.02
    smoothing: int = 20


@dataclass
class EnsembleResult:
    """Output of the factor ensemble at a single frequency.

    Attributes:
        composite:         Blended factor signal series.
        factor_signals:    DataFrame of individual factor signals.
        factor_weights:    Final weights used after adaptation.
        monthly_aggregate: If daily input, the monthly-mean aggregate
                           (integral of daily signals).
        monthly_confidence: Per-month confidence (signal strength × consistency).
    """

    composite: pd.Series
    factor_signals: pd.DataFrame
    factor_weights: Dict[str, float]
    monthly_aggregate: Optional[pd.Series] = None
    monthly_confidence: Optional[pd.Series] = None


class FactorEnsemble:
    """Unified multi-factor blending engine.

    Computes all factors, blends them with adaptive weights, and produces
    outputs at both daily and monthly resolution.

    Args:
        config: :class:`FactorEnsembleConfig`.
    """

    def __init__(self, config: Optional[FactorEnsembleConfig] = None):
        self.config = config or FactorEnsembleConfig()

    def compute_factors(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
    ) -> Dict[str, pd.Series]:
        """Compute all factor signals.

        Args:
            asset_prices:     Dict ``{asset_name: price_series}`` (daily).
            macro_indicators: Dict ``{indicator_name: series}`` (any freq).
            yield_short:      Short-term yield series.
            yield_long:       Long-term yield series.
            risk_series:      Risk indicator series (VIX, NFCI, etc.).

        Returns:
            Dict mapping factor name to signal series (daily-aligned).
        """
        return fmod._compute_all_factors(
            asset_prices=asset_prices,
            macro_indicators=macro_indicators or {},
            yield_short=yield_short,
            yield_long=yield_long,
            risk_series=risk_series,
        )

    def adapt_weights(
        self,
        factor_signals: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Adapt factor weights based on recent predictive performance.

        Each factor's weight is proportional to its information ratio
        (mean signal × forward return correlation) over the lookback.

        Args:
            factor_signals:  DataFrame of factor signals (columns = factors).
            forward_returns: Optional forward returns for IR calculation.
                             If None, weights stay at config defaults.

        Returns:
            Dict of factor → adapted weight.
        """
        weights = dict(self.config.factor_weights)
        if forward_returns is None or forward_returns.empty or factor_signals.empty:
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}

        common_idx = factor_signals.index.intersection(forward_returns.index)
        if len(common_idx) < self.config.adaptation_lookback // 2:
            return weights

        lookback = min(self.config.adaptation_lookback, len(common_idx))
        recent_signals = factor_signals.loc[common_idx[-lookback:]]
        recent_fwd = forward_returns.loc[common_idx[-lookback:]]

        ir_scores: Dict[str, float] = {}
        for col in recent_signals.columns:
            sig = recent_signals[col].dropna()
            ret = recent_fwd.reindex(sig.index).dropna()
            common = sig.index.intersection(ret.index)
            if len(common) < 10:
                ir_scores[col] = 0.0
                continue
            s = sig.loc[common]
            r = ret.loc[common].shift(-1).dropna()
            common2 = s.index.intersection(r.index)
            if len(common2) < 5:
                ir_scores[col] = 0.0
                continue
            corr = s.loc[common2].corr(r.loc[common2])
            ir_scores[col] = max(0.0, corr) * abs(s.loc[common2].mean())

        total_ir = sum(ir_scores.values()) or 1.0
        adapted = {}
        for name, base_w in weights.items():
            ir = ir_scores.get(name, 0.0)
            adapted[name] = max(self.config.min_weight, 0.5 * base_w + 0.5 * (ir / total_ir))

        total = sum(adapted.values())
        adapted = {k: v / total for k, v in adapted.items()}
        return adapted

    def blend(
        self,
        factor_signals: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Blend factor signals into a single composite.

        Args:
            factor_signals: Dict of factor_name → signal_series.
            weights:        Factor weights (defaults to config).

        Returns:
            Composite signal series in [-1, 1].
        """
        weights = weights or self.config.factor_weights
        if not factor_signals:
            return pd.Series(dtype=float)

        idx = None
        for s in factor_signals.values():
            if s is not None and not s.empty:
                idx = s.index
                break
        if idx is None:
            return pd.Series(dtype=float)

        composite = pd.Series(0.0, index=idx, dtype=float)
        total_w = 0.0
        for name, weight in weights.items():
            if name in factor_signals and factor_signals[name] is not None:
                sig = factor_signals[name].reindex(idx).fillna(0.0)
                composite = composite.add(sig * weight, fill_value=0.0)
                total_w += weight

        if total_w > 0:
            composite = composite / total_w
        return composite.clip(-1, 1)

    def run(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> EnsembleResult:
        """Execute the full ensemble pipeline.

        1. Compute all factor signals at the input frequency.
        2. Adapt weights based on recent performance.
        3. Blend into a composite signal.
        4. If input is daily, aggregate to monthly (integral).

        Args:
            asset_prices:     Dict ``{asset_name: daily_price_series}``.
            macro_indicators: Dict ``{indicator_name: series}``.
            yield_short/yield_long: Yield curve inputs.
            risk_series:      Credit/risk indicator.
            forward_returns:  Optional for weight adaptation.

        Returns:
            :class:`EnsembleResult` with composite, factor signals,
            weights, and monthly aggregate.
        """
        raw_factors = self.compute_factors(
            asset_prices, macro_indicators, yield_short, yield_long, risk_series
        )
        if not raw_factors:
            logger.warning("FactorEnsemble: no factors computed.")
            return EnsembleResult(
                composite=pd.Series(dtype=float),
                factor_signals=pd.DataFrame(),
                factor_weights=self.config.factor_weights,
            )

        factor_df = pd.DataFrame(raw_factors).sort_index()
        factor_df = factor_df.dropna(how="all")

        adapted_weights = self.adapt_weights(factor_df, forward_returns)
        composite = self.blend(raw_factors, adapted_weights)

        monthly_agg = None
        monthly_conf = None
        if composite is not None and not composite.empty:
            monthly_agg = composite.resample("ME").mean()
            monthly_agg.index = monthly_agg.index + pd.offsets.MonthEnd(0)

            factor_monthly = factor_df.resample("ME").mean()
            if not factor_monthly.empty:
                monthly_conf = factor_monthly.std(axis=1)
                monthly_conf = 1.0 / (1.0 + monthly_conf)

        logger.info(
            f"FactorEnsemble: {len(raw_factors)} factors blended → "
            f"{len(composite)} daily obs, "
            f"{len(monthly_agg) if monthly_agg is not None else 0} monthly aggregates."
        )
        return EnsembleResult(
            composite=composite,
            factor_signals=factor_df,
            factor_weights=adapted_weights,
            monthly_aggregate=monthly_agg,
            monthly_confidence=monthly_conf,
        )


__all__ = [
    "DEFAULT_FACTOR_WEIGHTS",
    "FactorEnsembleConfig",
    "EnsembleResult",
    "FactorEnsemble",
]
