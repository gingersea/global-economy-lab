"""
Macro-economic regime predictor — Kalman filter based.

Uses the DFM's filtered latent state (aggregated to monthly) to
label economic regimes.  Confidence comes from the Kalman filter's
state covariance matrix — statistically rigorous uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from src.analysis.factor_ensemble import EnsembleResult, FactorEnsemble, FactorEnsembleConfig


@dataclass
class MacroPrediction:
    """Monthly macro prediction from Kalman filter.

    Attributes:
        timestamp:            Reference date.
        composite_signal:     Filtered state (monthly mean of daily composite).
        confidence:           State estimation confidence from KF covariance.
        predicted_regime:     Mapped regime label.
        regime_probabilities: Confidence-weighted regime probabilities.
        forward_pred:         Forward prediction from DFM (mean, CI).
    """

    timestamp: pd.Timestamp
    composite_signal: float
    confidence: float
    predicted_regime: str
    regime_probabilities: Dict[str, float]
    forward_pred_mean: float = 0.0
    forward_pred_lower: float = 0.0
    forward_pred_upper: float = 0.0
    factor_contributions: Dict[str, float] = field(default_factory=dict)


def _signal_to_regime(signal: float, confidence: float) -> tuple[str, Dict[str, float]]:
    abs_s = abs(signal)
    if signal > 0.3:
        probs = {"recovery": 0.45 + 0.2 * abs_s, "overheat": 0.35, "stagflation": 0.12, "recession": 0.08 - 0.2 * abs_s}
        regime = "recovery"
    elif signal > 0:
        probs = {"recovery": 0.25, "overheat": 0.45 + 0.2 * abs_s, "stagflation": 0.20, "recession": 0.10 - 0.2 * abs_s}
        regime = "overheat"
    elif signal > -0.3:
        probs = {"recovery": 0.10 - 0.2 * abs_s, "overheat": 0.20, "stagflation": 0.45 + 0.2 * abs_s, "recession": 0.25}
        regime = "stagflation"
    else:
        probs = {"recovery": 0.08 - 0.2 * abs_s, "overheat": 0.12, "stagflation": 0.25, "recession": 0.55 + 0.2 * abs_s}
        regime = "recession"
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    cf = min(1.0, confidence * 2.0)
    for k in probs:
        probs[k] = round(probs[k] * cf + 0.25 * (1 - cf), 4)
    total = sum(probs.values())
    return regime, {k: round(v / total, 4) for k, v in probs.items()}


class MacroRegimePredictor:
    """Predict economic regime from Kalman-filtered latent state.

    Args:
        ensemble_config: Factor ensemble (KF-based) configuration.
    """

    def __init__(self, ensemble_config: Optional[FactorEnsembleConfig] = None):
        self.ensemble_config = ensemble_config or FactorEnsembleConfig()
        self.ensemble = FactorEnsemble(self.ensemble_config)

    def predict(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> MacroPrediction:
        result = self.ensemble.run(asset_prices, macro_indicators, yield_short, yield_long, risk_series, forward_returns)

        if result.monthly_aggregate is None or result.monthly_aggregate.empty:
            return MacroPrediction(timestamp=pd.Timestamp.now(), composite_signal=0.0, confidence=0.5, predicted_regime="unknown", regime_probabilities={r: 0.25 for r in ["recovery", "overheat", "stagflation", "recession"]})

        last_signal = float(result.monthly_aggregate.iloc[-1])
        confidence = float(result.forward_pred.confidence[0]) if result.forward_pred is not None else 0.3
        regime, probs = _signal_to_regime(last_signal, confidence)

        fwd_mean = fwd_lower = fwd_upper = 0.0
        if result.forward_pred is not None:
            fwd_mean = float(result.forward_pred.state_mean[0, 0])
            fwd_lower = float(result.forward_pred.conf_lower[0, 0])
            fwd_upper = float(result.forward_pred.conf_upper[0, 0])

        return MacroPrediction(
            timestamp=result.monthly_aggregate.index[-1],
            composite_signal=round(last_signal, 4),
            confidence=round(confidence, 4),
            predicted_regime=regime,
            regime_probabilities=probs,
            forward_pred_mean=round(fwd_mean, 4),
            forward_pred_lower=round(fwd_lower, 4),
            forward_pred_upper=round(fwd_upper, 4),
        )


__all__ = ["MacroPrediction", "MacroRegimePredictor", "_signal_to_regime"]
