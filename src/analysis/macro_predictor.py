"""
Macro-economic regime predictor — factor-based.

Uses the multi-factor ensemble at monthly frequency.
The monthly prediction is the **integral of daily factor signals**,
aggregated to month-end for inherently higher accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.factor_ensemble import (
    EnsembleResult,
    FactorEnsemble,
    FactorEnsembleConfig,
)
from src.analysis.regime_labels import REGIME_LABELS, RegimeConfig, label_regimes


@dataclass
class MacroPredictorConfig:
    """Configuration for :class:`MacroRegimePredictor`.

    Attributes:
        regime_config:    Regime labelling parameters.
        ensemble_config:  Factor ensemble configuration.
        horizons:         Forecast horizons (months ahead).
    """

    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    ensemble_config: FactorEnsembleConfig = field(default_factory=FactorEnsembleConfig)
    horizons: tuple = (1, 2, 3)


@dataclass
class MacroPrediction:
    """Monthly macro prediction result.

    Attributes:
        timestamp:            Reference date.
        composite_signal:     Monthly factor composite (integral of daily).
        confidence:           Signal strength × factor consistency.
        predicted_regime:     Mapped regime label from composite signal.
        regime_probabilities: Confidence-weighted regime distribution.
        factor_weights:       Adapted factor weights.
    """

    timestamp: pd.Timestamp
    composite_signal: float
    confidence: float
    predicted_regime: str
    regime_probabilities: Dict[str, float]
    factor_weights: Dict[str, float]
    factor_contributions: Dict[str, float] = field(default_factory=dict)


def _signal_to_regime(signal: float, confidence: float) -> tuple[str, Dict[str, float]]:
    """Map a composite factor signal to a regime label with probabilities.

    Signal interpretation:
    - Strong positive (> 0.3): recovery (growth +, inflation -)
    - Moderate positive (0 to 0.3): overheat (growth +, inflation +)
    - Moderate negative (-0.3 to 0): stagflation (growth -, inflation +)
    - Strong negative (< -0.3): recession (growth -, inflation -)
    """
    abs_s = abs(signal)
    if signal > 0.3:
        probs = {
            "recovery": 0.45 + 0.2 * abs_s,
            "overheat": 0.35,
            "stagflation": 0.12,
            "recession": 0.08 - 0.2 * abs_s,
        }
        regime = "recovery"
    elif signal > 0:
        probs = {
            "recovery": 0.25,
            "overheat": 0.45 + 0.2 * abs_s,
            "stagflation": 0.20,
            "recession": 0.10 - 0.2 * abs_s,
        }
        regime = "overheat"
    elif signal > -0.3:
        probs = {
            "recovery": 0.10 - 0.2 * abs_s,
            "overheat": 0.20,
            "stagflation": 0.45 + 0.2 * abs_s,
            "recession": 0.25,
        }
        regime = "stagflation"
    else:
        probs = {
            "recovery": 0.08 - 0.2 * abs_s,
            "overheat": 0.12,
            "stagflation": 0.25,
            "recession": 0.55 + 0.2 * abs_s,
        }
        regime = "recession"

    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    conf_factor = min(1.0, confidence * 2.0)
    for k in probs:
        probs[k] = round(probs[k] * conf_factor + 0.25 * (1 - conf_factor), 4)
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    return regime, probs


class MacroRegimePredictor:
    """Predict economic regime from aggregated daily factor signals.

    The monthly prediction is the **integral** of daily factor composite
    over the month, producing higher signal-to-noise than any single
    monthly indicator.

    Args:
        config: :class:`MacroPredictorConfig`.
    """

    def __init__(self, config: Optional[MacroPredictorConfig] = None):
        self.config = config or MacroPredictorConfig()
        self.ensemble = FactorEnsemble(self.config.ensemble_config)

    def predict(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> MacroPrediction:
        """Run factor ensemble → aggregate to monthly → map to regime.
        Confidence = rolling directional accuracy from ensemble (monthly)."""
        result = self.ensemble.run(
            asset_prices=asset_prices,
            macro_indicators=macro_indicators,
            yield_short=yield_short,
            yield_long=yield_long,
            risk_series=risk_series,
            forward_returns=forward_returns,
        )

        if result.monthly_aggregate is None or result.monthly_aggregate.empty:
            logger.warning("MacroRegimePredictor: no monthly aggregate.")
            return MacroPrediction(
                timestamp=pd.Timestamp.now(),
                composite_signal=0.0,
                confidence=0.0,
                predicted_regime="unknown",
                regime_probabilities={r: 0.25 for r in ["recovery", "overheat", "stagflation", "recession"]},
                factor_weights=result.factor_weights,
            )

        last_signal = float(result.monthly_aggregate.iloc[-1])
        confidence = 0.5
        if result.monthly_confidence is not None and not result.monthly_confidence.empty:
            confidence = float(result.monthly_confidence.iloc[-1])
        regime, probs = _signal_to_regime(last_signal, confidence)

        contributions = {}
        if not result.factor_signals.empty:
            monthly_factors = result.factor_signals.resample("ME").mean()
            if not monthly_factors.empty:
                last_row = monthly_factors.iloc[-1]
                for col in last_row.index:
                    contributions[col] = round(float(last_row[col]), 4)

        return MacroPrediction(
            timestamp=result.monthly_aggregate.index[-1],
            composite_signal=round(last_signal, 4),
            confidence=round(confidence, 4),
            predicted_regime=regime,
            regime_probabilities=probs,
            factor_weights=result.factor_weights,
            factor_contributions=contributions,
        )


__all__ = [
    "MacroPredictorConfig",
    "MacroPrediction",
    "MacroRegimePredictor",
    "_signal_to_regime",
]
