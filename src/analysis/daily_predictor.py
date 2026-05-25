"""
Daily short-term asset predictor — factor-based.

Uses the same multi-factor ensemble as the macro predictor.
Produces daily directional signals per asset by running the factor
ensemble at daily frequency, plus per-asset factor decompositions.

The daily ensemble is the **same factor system** as the monthly predictor —
just at higher resolution.  Monthly = integral(daily).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.factor_ensemble import FactorEnsemble, FactorEnsembleConfig


@dataclass
class DailyPrediction:
    """Single-day prediction for all assets.

    Attributes:
        date:              Prediction date.
        composite_signal:  Ensemble composite at this day.
        confidence:        Signal strength.
        factor_signals:    Per-factor signal values at this day.
        asset_signals:     Per-asset directional signals.
        factor_weights:    Adapted factor weights.
    """

    date: pd.Timestamp
    composite_signal: float
    confidence: float
    factor_signals: Dict[str, float] = field(default_factory=dict)
    asset_signals: Dict[str, float] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)


class DailyAssetPredictor:
    """Generate daily directional signals from the factor ensemble.

    Args:
        config: :class:`FactorEnsembleConfig` for the underlying ensemble.
    """

    def __init__(self, config: Optional[FactorEnsembleConfig] = None):
        self.config = config or FactorEnsembleConfig()
        self.ensemble = FactorEnsemble(self.config)

    def predict(
        self,
        asset_prices: Dict[str, pd.Series],
        macro_indicators: Optional[Dict[str, pd.Series]] = None,
        yield_short: Optional[pd.Series] = None,
        yield_long: Optional[pd.Series] = None,
        risk_series: Optional[pd.Series] = None,
    ) -> DailyPrediction:
        """Run factor ensemble at daily frequency, extract latest day.

        Args:
            asset_prices:     Daily price series per asset.
            macro_indicators: Macro indicator series.
            yield_short:      2Y yield.
            yield_long:       10Y yield.
            risk_series:      VIX / NFCI.

        Returns:
            :class:`DailyPrediction` with latest signals.
        """
        result = self.ensemble.run(
            asset_prices=asset_prices,
            macro_indicators=macro_indicators,
            yield_short=yield_short,
            yield_long=yield_long,
            risk_series=risk_series,
        )

        if result.composite.empty:
            return DailyPrediction(
                date=pd.Timestamp.now(),
                composite_signal=0.0,
                confidence=0.0,
            )

        last_date = result.composite.index[-1]
        last_signal = float(result.composite.iloc[-1])
        confidence = abs(last_signal)

        factor_vals = {}
        if not result.factor_signals.empty:
            last_row = result.factor_signals.iloc[-1]
            factor_vals = {col: round(float(last_row[col]), 4) for col in last_row.index}

        asset_sigs = {}
        for asset, px in asset_prices.items():
            if px is None or px.empty:
                continue
            aligned = px.reindex(result.composite.index)
            if aligned.dropna().empty:
                continue
            asset_ret = aligned.pct_change(fill_method=None).tail(20)
            asset_mom = np.tanh(asset_ret.mean() * 100) if not asset_ret.empty else 0.0
            asset_sigs[asset] = round(
                float(last_signal * 0.7 + asset_mom * 0.3), 4
            )

        return DailyPrediction(
            date=last_date,
            composite_signal=round(last_signal, 4),
            confidence=round(confidence, 4),
            factor_signals=factor_vals,
            asset_signals=asset_sigs,
            factor_weights=result.factor_weights,
        )


__all__ = [
    "DailyPrediction",
    "DailyAssetPredictor",
]
