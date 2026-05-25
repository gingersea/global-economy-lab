"""
Multi-factor ensemble.

Combines factor signals (from :mod:`src.analysis.factors`) into a single
composite prediction using performance-weighted blending.

Key design principle: **monthly = integral of daily**.

Confidence is defined uniformly across timeframes as **rolling directional
accuracy**: the hit rate of ``sign(prediction) == sign(actual forward return)``
over a configurable lookback window.  Both daily and monthly use the exact
same metric — the only difference is the sampling frequency.
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
        factor_weights:       Initial weight per factor (normalised to sum=1).
        adaptation_lookback:  Periods for evaluating factor performance.
        min_weight:           Floor for any factor weight after adaptation.
        daily_accuracy_window: Trading days for rolling daily accuracy.
        monthly_accuracy_window: Months for rolling monthly accuracy.
    """

    factor_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FACTOR_WEIGHTS)
    )
    adaptation_lookback: int = 60
    min_weight: float = 0.02
    daily_accuracy_window: int = 60
    monthly_accuracy_window: int = 12


@dataclass
class EnsembleResult:
    """Output of the factor ensemble.

    Attributes:
        composite:           Blended factor signal series (daily).
        factor_signals:      DataFrame of individual factor signals (daily).
        factor_weights:      Final weights used after adaptation.
        daily_confidence:    Rolling directional accuracy at daily frequency
                             (``sign(pred_t) == sign(ret_{t+1})`` hit rate).
        monthly_aggregate:   Mean of daily composite over each month (integral).
        monthly_confidence:  Rolling directional accuracy at monthly frequency
                             (same definition as daily, aggregated).
    """

    composite: pd.Series
    factor_signals: pd.DataFrame
    factor_weights: Dict[str, float]
    daily_confidence: Optional[pd.Series] = None
    monthly_aggregate: Optional[pd.Series] = None
    monthly_confidence: Optional[pd.Series] = None


def _rolling_directional_accuracy(
    signal: pd.Series,
    forward_return: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling hit rate: fraction of periods where
    ``sign(signal) == sign(forward_return)``.

    Args:
        signal:          Prediction signal (e.g. composite).
        forward_return:  Actual forward-period return aligned with signal dates.
        window:          Rolling window size.

    Returns:
        Series of hit rates in [0, 1], same index as signal.
    """
    if signal is None or signal.empty or forward_return is None or forward_return.empty:
        return pd.Series(dtype=float)
    common = signal.index.intersection(forward_return.index)
    if len(common) < window:
        return pd.Series(0.5, index=signal.index)
    s = signal.loc[common]
    r = forward_return.loc[common]
    correct = (np.sign(s) == np.sign(r)).astype(float)
    accuracy = correct.rolling(window, min_periods=max(3, window // 4)).mean()
    return accuracy.reindex(signal.index)


class FactorEnsemble:
    """Unified multi-factor blending engine.

    Computes all factors, blends them with adaptive weights, and produces
    outputs at both daily and monthly resolution with **unified confidence**
    (rolling directional accuracy).

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
        return {k: v / total for k, v in adapted.items()}

    def blend(
        self,
        factor_signals: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
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
        """Execute the full ensemble pipeline with unified confidence.

        Confidence = rolling hit rate of sign(pred) == sign(fwd_ret),
        computed at both daily and monthly frequencies.
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

        factor_df = pd.DataFrame(raw_factors).sort_index().dropna(how="all")
        adapted_weights = self.adapt_weights(factor_df, forward_returns)
        composite = self.blend(raw_factors, adapted_weights)

        daily_conf = None
        monthly_agg = None
        monthly_conf = None

        if composite is not None and not composite.empty:
            if forward_returns is not None and not forward_returns.empty:
                fwd_aligned = forward_returns.reindex(composite.index).shift(-1)
                daily_conf = _rolling_directional_accuracy(
                    composite, fwd_aligned, self.config.daily_accuracy_window
                )

            monthly_agg = composite.resample("ME").mean()
            monthly_agg.index = monthly_agg.index + pd.offsets.MonthEnd(0)
            monthly_agg.name = "monthly_composite"

            if daily_conf is not None and not daily_conf.empty:
                monthly_conf = daily_conf.resample("ME").mean()
                monthly_conf.index = monthly_conf.index + pd.offsets.MonthEnd(0)
                monthly_conf.name = "monthly_confidence"

        logger.info(
            f"FactorEnsemble: {len(raw_factors)} factors → "
            f"{len(composite)} daily obs, "
            f"{len(monthly_agg) if monthly_agg is not None else 0} monthly aggregates."
        )
        if daily_conf is not None and not daily_conf.empty:
            logger.info(
                f"  Daily accuracy:   {daily_conf.iloc[-1]:.1%} "
                f"(rolling {self.config.daily_accuracy_window}d)"
            )
        if monthly_conf is not None and not monthly_conf.empty:
            logger.info(
                f"  Monthly accuracy: {monthly_conf.iloc[-1]:.1%} "
                f"(rolling {self.config.monthly_accuracy_window}mo)"
            )

        return EnsembleResult(
            composite=composite,
            factor_signals=factor_df,
            factor_weights=adapted_weights,
            daily_confidence=daily_conf,
            monthly_aggregate=monthly_agg,
            monthly_confidence=monthly_conf,
        )


__all__ = [
    "DEFAULT_FACTOR_WEIGHTS",
    "FactorEnsembleConfig",
    "EnsembleResult",
    "FactorEnsemble",
    "_rolling_directional_accuracy",
]
