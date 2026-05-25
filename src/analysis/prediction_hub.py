"""
Prediction hub — unified factor-based prediction orchestrator.

Runs the factor ensemble once, produces:
- Daily prediction (latest day signals)
- Monthly prediction (integral of daily → regime mapping)
- Factor decomposition and weight reports

The monthly prediction IS the daily integral — not a separate system.
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
from src.analysis.macro_predictor import (
    MacroPrediction,
    MacroRegimePredictor,
    MacroPredictorConfig,
    _signal_to_regime,
)
from src.analysis.daily_predictor import DailyAssetPredictor, DailyPrediction


@dataclass
class HubPredictionResult:
    """Aggregated prediction result.

    Attributes:
        timestamp:           When generated.
        daily_prediction:    Latest daily signals.
        macro_prediction:    Monthly aggregate → regime.
        ensemble_result:     Full factor ensemble output.
        summary:             Human-readable summary.
    """

    timestamp: str
    daily_prediction: Optional[DailyPrediction] = None
    macro_prediction: Optional[MacroPrediction] = None
    ensemble_result: Optional = None
    summary: Dict = field(default_factory=dict)


class PredictionHub:
    """Unified factor-based prediction orchestrator.

    Runs the ensemble once → extracts daily + monthly predictions.
    Monthly = integral(daily) — the key architectural principle.

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
        self.ensemble = FactorEnsemble(self.ensemble_config)
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
    ) -> HubPredictionResult:
        """Execute the unified prediction pipeline.

        Args:
            asset_prices:     Daily price series per asset.
            macro_indicators: Macro indicator series (any frequency).
            yield_short:      2Y yield.
            yield_long:       10Y yield.
            risk_series:      VIX / NFCI.

        Returns:
            :class:`HubPredictionResult`.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        daily_pred = self.daily_predictor.predict(
            asset_prices, macro_indicators, yield_short, yield_long, risk_series
        )

        macro_pred = self.macro_predictor.predict(
            asset_prices, macro_indicators, yield_short, yield_long, risk_series
        )

        ensemble_result = self.ensemble.run(
            asset_prices, macro_indicators, yield_short, yield_long, risk_series
        )

        self._write_outputs(ensemble_result, daily_pred, macro_pred)

        summary = {
            "timestamp": str(date.today()),
            "daily": {
                "composite_signal": daily_pred.composite_signal,
                "confidence": daily_pred.confidence,
                "asset_signals": daily_pred.asset_signals,
                "factor_weights": daily_pred.factor_weights,
            },
            "monthly": {
                "composite_signal": macro_pred.composite_signal,
                "confidence": macro_pred.confidence,
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
            ensemble_result=ensemble_result,
            summary=summary,
        )

    def _write_outputs(self, ensemble_result, daily_pred, macro_pred):
        if not ensemble_result.factor_signals.empty:
            ensemble_result.factor_signals.to_csv(self.output_dir / "factor_signals_daily.csv")
        if ensemble_result.monthly_aggregate is not None and not ensemble_result.monthly_aggregate.empty:
            monthly_df = ensemble_result.monthly_aggregate.to_frame("monthly_composite")
            if ensemble_result.monthly_confidence is not None:
                monthly_df["confidence"] = ensemble_result.monthly_confidence
            monthly_df.to_csv(self.output_dir / "monthly_aggregate.csv")

        macro_probs = pd.DataFrame([macro_pred.regime_probabilities])
        macro_probs.index = [str(macro_pred.timestamp.date())]
        macro_probs.to_csv(self.output_dir / "regime_probabilities.csv")

        daily_sigs = pd.DataFrame([{
            "asset": k, "signal": v,
            "date": str(daily_pred.date.date()),
        } for k, v in daily_pred.asset_signals.items()])
        if not daily_sigs.empty:
            daily_sigs.to_csv(self.output_dir / "daily_signals.csv", index=False)


__all__ = [
    "HubPredictionResult",
    "PredictionHub",
]
