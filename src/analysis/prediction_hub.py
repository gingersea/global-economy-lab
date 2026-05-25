"""
Prediction hub — Kalman filter based.

Runs the DFM ensemble once, produces daily + monthly predictions
with proper KF confidence intervals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.analysis.factor_ensemble import FactorEnsemble, FactorEnsembleConfig
from src.analysis.macro_predictor import MacroRegimePredictor, MacroPrediction
from src.analysis.daily_predictor import DailyAssetPredictor, DailyPrediction


@dataclass
class HubPredictionResult:
    timestamp: str
    daily_prediction: Optional[DailyPrediction] = None
    macro_prediction: Optional[MacroPrediction] = None
    summary: Dict = field(default_factory=dict)


class PredictionHub:
    def __init__(self, ensemble_config=None, output_dir="data/_meta/predictions"):
        self.config = ensemble_config or FactorEnsembleConfig()
        self.macro_predictor = MacroRegimePredictor(self.config)
        self.daily_predictor = DailyAssetPredictor(self.config)
        self.output_dir = Path(output_dir)

    def run(
        self,
        asset_prices, macro_indicators=None, yield_short=None,
        yield_long=None, risk_series=None, forward_returns=None,
    ) -> HubPredictionResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        daily_pred = self.daily_predictor.predict(asset_prices, macro_indicators, yield_short, yield_long, risk_series, forward_returns)
        macro_pred = self.macro_predictor.predict(asset_prices, macro_indicators, yield_short, yield_long, risk_series, forward_returns)

        summary = {
            "timestamp": str(date.today()),
            "method": "Kalman Filter / Dynamic Factor Model",
            "confidence_type": "State covariance (1/(1+sqrt(P_diag)))",
            "daily": {
                "composite_signal": daily_pred.composite_signal,
                "confidence": daily_pred.confidence,
                "forward_pred": daily_pred.forward_pred_mean,
                "forward_CI": [daily_pred.forward_pred_lower, daily_pred.forward_pred_upper],
                "asset_signals": daily_pred.asset_signals,
            },
            "monthly": {
                "composite_signal": macro_pred.composite_signal,
                "confidence": macro_pred.confidence,
                "predicted_regime": macro_pred.predicted_regime,
                "regime_probabilities": macro_pred.regime_probabilities,
                "forward_pred": macro_pred.forward_pred_mean,
                "forward_CI": [macro_pred.forward_pred_lower, macro_pred.forward_pred_upper],
            },
        }

        (self.output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        logger.success(f"PredictionHub: artefacts written to {self.output_dir}")
        return HubPredictionResult(timestamp=str(date.today()), daily_prediction=daily_pred, macro_prediction=macro_pred, summary=summary)


__all__ = ["HubPredictionResult", "PredictionHub"]
