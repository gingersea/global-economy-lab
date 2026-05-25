"""
Prediction hub — unified factor-based prediction orchestrator.

Runs the factor ensemble once per predictor, produces:
- Daily prediction (latest day signals + rolling accuracy)
- Monthly prediction (integral of daily → regime + rolling accuracy)

Confidence is unified across timeframes: rolling directional accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.analysis.factor_ensemble import FactorEnsembleConfig
from src.analysis.macro_predictor import (
    MacroPrediction,
    MacroRegimePredictor,
    MacroPredictorConfig,
)
from src.analysis.daily_predictor import DailyAssetPredictor, DailyPrediction


@dataclass
class HubPredictionResult:
    """Aggregated prediction result.

    Attributes:
        timestamp:         When generated.
        daily_prediction:  Latest daily signals + rolling accuracy.
        macro_prediction:  Monthly aggregate → regime + rolling accuracy.
        summary:           Human-readable summary.
    """

    timestamp: str
    daily_prediction: Optional[DailyPrediction] = None
    macro_prediction: Optional[MacroPrediction] = None
    summary: Dict = field(default_factory=dict)


class PredictionHub:
    """Unified factor-based prediction orchestrator.

    Both daily and monthly predictions share the same confidence definition:
    rolling directional accuracy of sign(pred) vs sign(actual).

    Args:
        ensemble_config: Factor ensemble configuration.
        macro_config:    Macro predictor config.
        output_dir:      Where to write artefacts.
    """

    def __init__(
        self,
        ensemble_config: Optional[FactorEnsembleConfig] = None,
        macro_config: Optional[MacroPredictorConfig] = None,
        output_dir: str = "data/_meta/predictions",
    ):
        self.ensemble_config = ensemble_config or FactorEnsembleConfig()
        self.macro_config = macro_config or MacroPredictorConfig(
            ensemble_config=self.ensemble_config,
        )
        self.macro_predictor = MacroRegimePredictor(self.macro_config)
        self.daily_predictor = DailyAssetPredictor(self.ensemble_config)
        self.output_dir = Path(output_dir)

    def run(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> HubPredictionResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        daily_pred = self.daily_predictor.predict(
            asset_prices, macro_indicators, yield_short, yield_long,
            risk_series, forward_returns,
        )

        macro_pred = self.macro_predictor.predict(
            asset_prices, macro_indicators, yield_short, yield_long,
            risk_series, forward_returns,
        )

        summary = {
            "timestamp": str(date.today()),
            "confidence_definition": "rolling directional accuracy: sign(pred) == sign(actual forward return)",
            "daily": {
                "composite_signal": daily_pred.composite_signal,
                "confidence": daily_pred.confidence,
                "confidence_window": f"{self.ensemble_config.daily_accuracy_window}d",
                "asset_signals": daily_pred.asset_signals,
                "factor_weights": daily_pred.factor_weights,
            },
            "monthly": {
                "composite_signal": macro_pred.composite_signal,
                "confidence": macro_pred.confidence,
                "confidence_window": f"{self.ensemble_config.monthly_accuracy_window}mo",
                "predicted_regime": macro_pred.predicted_regime,
                "regime_probabilities": macro_pred.regime_probabilities,
                "factor_contributions": macro_pred.factor_contributions,
            },
        }

        (self.output_dir / "prediction_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        logger.success(f"PredictionHub: artefacts written to {self.output_dir}")

        return HubPredictionResult(
            timestamp=str(date.today()),
            daily_prediction=daily_pred,
            macro_prediction=macro_pred,
            summary=summary,
        )


__all__ = [
    "HubPredictionResult",
    "PredictionHub",
]
