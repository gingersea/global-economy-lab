"""
Multi-factor ensemble with correlation-aware factor selection.

Factors are grouped by economic category (momentum, value, volatility,
carry, macro, risk).  Within each group, factors share a weight budget
to prevent correlated signals from dominating the ensemble.

Monthly prediction = multi-month adaptive aggregation of daily signals,
using EMA with variable span based on volatility regime — cycles have
different durations, and the aggregation window should adapt.

Confidence = rolling directional accuracy, unified across timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
        selection_lookback:     Periods for evaluating factor accuracy.
        daily_accuracy_window:  Trading days for rolling daily accuracy.
        monthly_accuracy_window: Months for rolling monthly accuracy.
        adaptive_ema_base:      Base EMA span for monthly aggregation.
    """

    selection_lookback: int = 252
    daily_accuracy_window: int = 60
    monthly_accuracy_window: int = 12
    adaptive_ema_base: float = 2.0


@dataclass
class FactorEval:
    """Per-factor evaluation result.

    Attributes:
        name:        Factor name.
        group:       Economic category group.
        accuracy:    Directional hit rate over lookback.
        selected:    Whether this factor passed accuracy threshold.
        reliability: 'reliable' (>52%), 'marginal' (50-52%), 'unreliable' (<50%).
        weight:      Assigned weight in the blend (0 if not selected).
    """

    name: str
    group: str
    accuracy: float
    selected: bool
    reliability: str
    weight: float


@dataclass
class EnsembleResult:
    """Output of the factor ensemble.

    Attributes:
        composite:           Blended factor signal (daily).
        factor_signals:      Raw individual factor signals (daily).
        factor_weights:      Final weights per selected factor (group-aware).
        factor_evals:        Per-factor accuracy evaluations.
        daily_confidence:    Rolling directional accuracy (daily).
        monthly_aggregate:   Adaptive multi-month aggregate of daily composite.
        monthly_confidence:  Rolling directional accuracy (monthly).
    """

    composite: pd.Series
    factor_signals: pd.DataFrame
    factor_weights: Dict[str, float]
    factor_evals: List[FactorEval] = field(default_factory=list)
    daily_confidence: Optional[pd.Series] = None
    monthly_aggregate: Optional[pd.Series] = None
    monthly_confidence: Optional[pd.Series] = None


def _rolling_directional_accuracy(
    signal: pd.Series,
    forward_return: pd.Series,
    window: int,
) -> pd.Series:
    if signal is None or signal.empty or forward_return is None or forward_return.empty:
        return pd.Series(dtype=float)
    common = signal.index.intersection(forward_return.index)
    if len(common) < window:
        return pd.Series(0.5, index=signal.index)
    s = signal.loc[common]
    r = forward_return.loc[common]
    correct = (np.sign(s) == np.sign(r)).astype(float)
    return correct.rolling(window, min_periods=max(3, window // 4)).mean().reindex(signal.index)


def _adaptive_monthly_aggregate(
    composite: pd.Series,
    vol_regime: Optional[pd.Series] = None,
    ema_base: float = 2.0,
) -> pd.Series:
    """Monthly aggregate with adaptive span.

    Low vol (regimes persist) → longer span. High vol (regimes change
    fast) → shorter span.  Current month mixes with prior months via
    variable-speed EMA.
    """
    if composite is None or composite.empty:
        return pd.Series(dtype=float)
    monthly_mean = composite.resample("ME").mean()
    monthly_mean.index = monthly_mean.index + pd.offsets.MonthEnd(0)
    if vol_regime is None or vol_regime.empty:
        return monthly_mean
    vol_m = vol_regime.resample("ME").mean()
    vol_m.index = vol_m.index + pd.offsets.MonthEnd(0)
    common = monthly_mean.index.intersection(vol_m.index)
    if len(common) < 3:
        return monthly_mean
    vol_c = vol_m.loc[common].clip(-0.8, 0.8)
    span = (ema_base + vol_c * 2.0).clip(1.0, 4.0)
    result = monthly_mean.loc[common].copy()
    ema = float(result.iloc[0])
    for i in range(1, len(common)):
        alpha = 2.0 / (span.iloc[i] + 1.0)
        ema = alpha * float(result.iloc[i]) + (1.0 - alpha) * ema
        result.iloc[i] = ema
    return result


def _group_aware_weights(factor_evals: List[FactorEval]) -> Dict[str, float]:
    """Group-aware weight allocation.

    Each economic group gets one weight budget.  The best factor in each
    group claims that budget.  This prevents correlated factors (e.g. two
    momentum factors) from double-counting.
    """
    groups = fmod.FACTOR_GROUPS
    eval_by_name = {e.name: e for e in factor_evals if e.selected}
    if not eval_by_name:
        return {}

    group_scores: Dict[str, float] = {}
    for gname, fnames in groups.items():
        gfs = [eval_by_name[n] for n in fnames if n in eval_by_name]
        if gfs:
            group_scores[gname] = max(e.accuracy for e in gfs)

    if not group_scores:
        return {}

    total = sum(v - 0.5 for v in group_scores.values()) or 1.0
    gw = {g: max(0.0, (s - 0.5) / total) for g, s in group_scores.items()}
    total_gw = sum(gw.values()) or 1.0
    gw = {g: w / total_gw for g, w in gw.items()}

    weights: Dict[str, float] = {}
    for gname, fnames in groups.items():
        w = gw.get(gname, 0.0)
        if w <= 0:
            continue
        gfs = [eval_by_name[n] for n in fnames if n in eval_by_name]
        if not gfs:
            continue
        best = max(gfs, key=lambda e: e.accuracy)
        weights[best.name] = round(w, 4)
    return weights


class FactorEnsemble:
    """Multi-factor ensemble with correlation-aware grouping.

    Factors grouped by economic category.  Within-group factors share
    a weight budget; across-group weights proportional to group accuracy.
    Monthly aggregation uses adaptive EMA span based on volatility regime.

    Args:
        config: :class:`FactorEnsembleConfig`.
    """

    def __init__(self, config: Optional[FactorEnsembleConfig] = None):
        self.config = config or FactorEnsembleConfig()

    def compute_factors(self, asset_prices, macro_indicators=None,
                        yield_short=None, yield_long=None, risk_series=None):
        return fmod._compute_all_factors(
            asset_prices=asset_prices,
            macro_indicators=macro_indicators or {},
            yield_short=yield_short, yield_long=yield_long,
            risk_series=risk_series,
        )

    def select_factors(
        self,
        factor_signals: Dict[str, pd.Series],
        forward_returns: Optional[pd.Series] = None,
    ) -> tuple[List[FactorEval], Dict[str, float]]:
        """Evaluate each factor's directional accuracy.
        Factors ≤ 50% are marked unreliable and excluded.
        Weights allocated by economic group to prevent correlation double-counting.
        """
        groups = fmod.FACTOR_GROUPS
        evals: List[FactorEval] = []

        if forward_returns is None or forward_returns.empty:
            for name in factor_signals:
                g = next((g for g, fs in groups.items() if name in fs), "other")
                evals.append(FactorEval(name=name, group=g, accuracy=0.5, selected=False, reliability="marginal", weight=0.0))
            return evals, {}

        lb = self.config.selection_lookback
        for name, sig in factor_signals.items():
            g = next((gn for gn, fs in groups.items() if name in fs), "other")
            if sig is None or sig.empty:
                evals.append(FactorEval(name=name, group=g, accuracy=0.5, selected=False, reliability="marginal", weight=0.0))
                continue
            common = sig.index.intersection(forward_returns.index)
            if len(common) < lb // 4:
                evals.append(FactorEval(name=name, group=g, accuracy=0.5, selected=False, reliability="marginal", weight=0.0))
                continue
            s = sig.loc[common[-lb:]]
            r = forward_returns.loc[common[-lb:]].shift(-1)
            c2 = s.index.intersection(r.dropna().index)
            if len(c2) < 20:
                evals.append(FactorEval(name=name, group=g, accuracy=0.5, selected=False, reliability="marginal", weight=0.0))
                continue
            acc = round(float((np.sign(s.loc[c2]) == np.sign(r.loc[c2])).mean()), 4)
            rel = "reliable" if acc >= 0.52 else ("marginal" if acc >= 0.50 else "unreliable")
            selected = acc >= 0.50
            evals.append(FactorEval(name=name, group=g, accuracy=acc, selected=selected, reliability=rel, weight=0.0))

        weights = _group_aware_weights(evals)
        for e in evals:
            if e.name in weights:
                e.weight = weights[e.name]

        n_sel = len(weights)
        n_rel = sum(1 for e in evals if e.reliability == "reliable")
        n_unr = sum(1 for e in evals if e.reliability == "unreliable")
        logger.info(f"FactorEnsemble: {n_sel}/{len(factor_signals)} factors selected (reliable: {n_rel}, unreliable: {n_unr})")
        return evals, weights

    def blend(self, factor_signals, weights=None, factor_evals=None):
        weights = dict(weights or {})
        if not weights and factor_signals:
            eq_w = 1.0 / max(len(factor_signals), 1)
            weights = {k: eq_w for k in factor_signals}
        idx = None
        for name in weights:
            if name in factor_signals and factor_signals[name] is not None and not factor_signals[name].empty:
                idx = factor_signals[name].index
                break
        if idx is None:
            return pd.Series(dtype=float)
        composite = pd.Series(0.0, index=idx, dtype=float)
        for name, w in weights.items():
            if name not in factor_signals or factor_signals[name] is None:
                continue
            composite = composite.add(factor_signals[name].reindex(idx).fillna(0.0) * w, fill_value=0.0)
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
        raw = self.compute_factors(asset_prices, macro_indicators, yield_short, yield_long, risk_series)
        if not raw:
            return EnsembleResult(composite=pd.Series(dtype=float), factor_signals=pd.DataFrame(), factor_weights={})

        factor_df = pd.DataFrame(raw).sort_index().dropna(how="all")
        factor_evals, weights = self.select_factors(raw, forward_returns)

        if not weights:
            eq_w = 1.0 / max(len(raw), 1)
            weights = {k: eq_w for k in raw}

        composite = self.blend(raw, weights, factor_evals)

        daily_conf = None
        monthly_agg = None
        monthly_conf = None

        if composite is not None and not composite.empty:
            if forward_returns is not None and not forward_returns.empty:
                fwd_aligned = forward_returns.reindex(composite.index).shift(-1)
                daily_conf = _rolling_directional_accuracy(composite, fwd_aligned, self.config.daily_accuracy_window)

            vol_signal = raw.get("volatility_regime")
            monthly_agg = _adaptive_monthly_aggregate(composite, vol_signal, self.config.adaptive_ema_base)

            if daily_conf is not None and not daily_conf.empty:
                monthly_conf = daily_conf.resample("ME").mean()
                monthly_conf.index = monthly_conf.index + pd.offsets.MonthEnd(0)

        logger.info(f"FactorEnsemble: {len(weights)} factors → {len(composite)} daily obs, {len(monthly_agg) if monthly_agg is not None else 0} monthly aggregates.")
        if daily_conf is not None and not daily_conf.empty:
            logger.info(f"  Daily accuracy:   {daily_conf.iloc[-1]:.1%}")
        if monthly_conf is not None and not monthly_conf.empty:
            logger.info(f"  Monthly accuracy: {monthly_conf.iloc[-1]:.1%}")
        for e in sorted(factor_evals, key=lambda x: -x.accuracy):
            tag = "✓" if e.selected else "✗"
            logger.info(f"  {tag} [{e.group:12s}] {e.name:25s} acc={e.accuracy:.1%} ({e.reliability}) w={e.weight:.3f}")

        return EnsembleResult(
            composite=composite, factor_signals=factor_df,
            factor_weights=weights, factor_evals=factor_evals,
            daily_confidence=daily_conf,
            monthly_aggregate=monthly_agg,
            monthly_confidence=monthly_conf,
        )


__all__ = [
    "DEFAULT_FACTOR_WEIGHTS",
    "FactorEnsembleConfig",
    "FactorEval",
    "EnsembleResult",
    "FactorEnsemble",
    "_rolling_directional_accuracy",
    "_adaptive_monthly_aggregate",
    "_group_aware_weights",
]
