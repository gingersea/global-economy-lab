"""
Prediction hub — bridges the macro monthly and daily short-term predictors.

Orchestrates a two-step prediction pipeline:
1. MacroRegimePredictor → predicted regime + probabilities (1–3mo ahead).
2. DailyAssetPredictor → daily directional signals conditioned on the
   predicted regime.

Outputs are written to ``data/_meta/predictions/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.analysis.macro_predictor import (
    MacroPrediction,
    MacroPredictorConfig,
    MacroRegimePredictor,
)
from src.analysis.daily_predictor import (
    DailyAssetPredictor,
    DailyPredictorConfig,
    DailyPrediction,
)
from config.backtest import DEFAULT_REGIME_WEIGHTS


@dataclass
class HubPredictionResult:
    """Aggregated prediction result from the hub.

    Attributes:
        timestamp:           When the prediction was generated.
        macro_prediction:    Full :class:`MacroPrediction`.
        daily_prediction:    Full :class:`DailyPrediction`.
        daily_series:        Optional full daily signal history DataFrame.
        regime_signal_map:   How the predicted regime maps to asset weights.
        summary:             Human-readable summary dict.
    """

    timestamp: str
    macro_prediction: Optional[MacroPrediction] = None
    daily_prediction: Optional[DailyPrediction] = None
    daily_series: Optional[pd.DataFrame] = None
    regime_signal_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)


class PredictionHub:
    """Two-level prediction orchestrator.

    Args:
        macro_config:  Config for the macro predictor.
        daily_config:  Config for the daily predictor.
        output_dir:    Where to write prediction artefacts.
    """

    def __init__(
        self,
        macro_config: Optional[MacroPredictorConfig] = None,
        daily_config: Optional[DailyPredictorConfig] = None,
        output_dir: str = "data/_meta/predictions",
    ):
        self.macro_predictor = MacroRegimePredictor(macro_config)
        self.daily_predictor = DailyAssetPredictor(daily_config)
        self.output_dir = Path(output_dir)

    def run(
        self,
        panel: pd.DataFrame,
        asset_dfs: Dict[str, pd.DataFrame],
        regime_col: str = "regime",
        price_col: str = "close",
    ) -> HubPredictionResult:
        """Execute the full two-level prediction pipeline.

        Args:
            panel:      Monthly panel with regime labels.
            asset_dfs:  Dict ``{asset_key: daily_df}``.
            regime_col: Name of the regime label column in panel.
            price_col:  Price column in asset DataFrames.

        Returns:
            :class:`HubPredictionResult` with macro and daily predictions.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        macro_pred = self.macro_predictor.predict(panel, regime_col=regime_col)
        predicted_regime = macro_pred.predicted_regimes.get(1, "unknown")
        logger.info(
            f"PredictionHub: macro → {predicted_regime} "
            f"(confidence={macro_pred.confidence.get(1, 0):.2%})"
        )

        daily_pred = self.daily_predictor.predict(asset_dfs, predicted_regime, price_col=price_col)
        logger.info(f"PredictionHub: daily signals for {len(daily_pred.signals)} assets.")

        regime_series = pd.Series(predicted_regime, index=asset_dfs[next(iter(asset_dfs))].index[-1:])

        daily_series = None
        try:
            full_regime = pd.Series(predicted_regime, index=pd.DatetimeIndex(
                [asset_dfs[k].index[-1] for k in asset_dfs if asset_dfs[k] is not None]
            ))
            daily_series = self.daily_predictor.predict_series(
                asset_dfs, full_regime, price_col=price_col
            )
        except Exception:
            logger.warning("PredictionHub: daily series generation skipped.")

        regime_signal_map = self._build_regime_signal_map(macro_pred)
        summary = self._build_summary(macro_pred, daily_pred, regime_signal_map)

        self._write_outputs(macro_pred, daily_pred, daily_series, summary)

        return HubPredictionResult(
            timestamp=str(date.today()),
            macro_prediction=macro_pred,
            daily_prediction=daily_pred,
            daily_series=daily_series,
            regime_signal_map=regime_signal_map,
            summary=summary,
        )

    def rolling_backtest(
        self,
        panel: pd.DataFrame,
        asset_dfs: Dict[str, pd.DataFrame],
        regime_col: str = "regime",
    ) -> pd.DataFrame:
        """Rolling out-of-sample evaluation of the macro predictor.

        Returns:
            DataFrame from :meth:`MacroRegimePredictor.predict_rolling`.
        """
        return self.macro_predictor.predict_rolling(panel, regime_col=regime_col)

    def _build_regime_signal_map(self, macro_pred: MacroPrediction) -> Dict[str, Dict[str, float]]:
        regime = macro_pred.predicted_regimes.get(1, "unknown")
        return {
            "predicted_regime": regime,
            "confidence": macro_pred.confidence.get(1, 0.0),
            "regime_weights": DEFAULT_REGIME_WEIGHTS.get(regime, DEFAULT_REGIME_WEIGHTS["unknown"]),
            "probabilities": {
                r: round(float(macro_pred.probabilities.loc[1, r]), 4)
                for r in ["recovery", "overheat", "stagflation", "recession"]
                if r in macro_pred.probabilities.columns
            },
        }

    def _build_summary(
        self,
        macro_pred: MacroPrediction,
        daily_pred: DailyPrediction,
        regime_signal_map: Dict,
    ) -> Dict:
        return {
            "timestamp": str(date.today()),
            "reference_date": str(macro_pred.timestamp.date()),
            "current_regime": macro_pred.current_regime,
            "predicted_regime_1m": macro_pred.predicted_regimes.get(1),
            "predicted_regime_2m": macro_pred.predicted_regimes.get(2),
            "predicted_regime_3m": macro_pred.predicted_regimes.get(3),
            "confidence_1m": macro_pred.confidence.get(1),
            "confidence_2m": macro_pred.confidence.get(2, 0),
            "confidence_3m": macro_pred.confidence.get(3, 0),
            "daily_signals": daily_pred.signals,
            "daily_confidence": daily_pred.confidence,
            "indicators": macro_pred.indicators,
            "regime_signal_map": regime_signal_map,
        }

    def _write_outputs(
        self,
        macro_pred: MacroPrediction,
        daily_pred: DailyPrediction,
        daily_series: Optional[pd.DataFrame],
        summary: Dict,
    ) -> None:
        if macro_pred.probabilities is not None and not macro_pred.probabilities.empty:
            macro_pred.probabilities.to_csv(self.output_dir / "macro_probabilities.csv")

        if macro_pred.transition_matrix is not None and not macro_pred.transition_matrix.empty:
            macro_pred.transition_matrix.to_csv(self.output_dir / "transition_matrix.csv")

        signals_df = pd.DataFrame([{
            "asset": k,
            "signal": v,
            "confidence": daily_pred.confidence.get(k, 0.0),
            "predicted_regime": daily_pred.predicted_regime,
        } for k, v in daily_pred.signals.items()])
        if not signals_df.empty:
            signals_df.to_csv(self.output_dir / "daily_signals.csv", index=False)

        if daily_series is not None and not daily_series.empty:
            daily_series.to_csv(self.output_dir / "daily_signal_series.csv")

        (self.output_dir / "prediction_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        logger.success(f"PredictionHub: artefacts written to {self.output_dir}")


__all__ = [
    "HubPredictionResult",
    "PredictionHub",
]
