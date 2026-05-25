"""
Daily short-term asset predictor — Kalman filter based.

Uses the DFM's filtered latent state as the composite signal source.
Confidence from Kalman state covariance.  Asset-level signals are
composite × per-asset momentum overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.factor_ensemble import FactorEnsemble, FactorEnsembleConfig


@dataclass
class DailyPrediction:
    date: pd.Timestamp
    composite_signal: float
    confidence: float
    factor_signals: Dict[str, float] = field(default_factory=dict)
    asset_signals: Dict[str, float] = field(default_factory=dict)
    forward_pred_mean: float = 0.0
    forward_pred_lower: float = 0.0
    forward_pred_upper: float = 0.0


class DailyAssetPredictor:
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
        forward_returns: Optional[pd.Series] = None,
    ) -> DailyPrediction:
        result = self.ensemble.run(asset_prices, macro_indicators, yield_short, yield_long, risk_series, forward_returns)

        if result.composite.empty:
            return DailyPrediction(date=pd.Timestamp.now(), composite_signal=0.0, confidence=0.5)

        last_date = result.composite.index[-1]
        last_signal = float(result.composite.iloc[-1])
        confidence = float(result.forward_pred.confidence[0]) if result.forward_pred is not None else 0.3

        factor_vals = {}
        if not result.factor_signals.empty:
            last_row = result.factor_signals.iloc[-1]
            factor_vals = {str(col): round(float(last_row[col]), 4) for col in last_row.index}

        asset_sigs = {}
        for asset, px in asset_prices.items():
            if px is None or px.empty:
                continue
            aligned = px.reindex(result.composite.index)
            if aligned.dropna().empty:
                continue
            ret = aligned.pct_change(fill_method=None).tail(20)
            mom = np.tanh(ret.mean() * 100) if not ret.empty else 0.0
            asset_sigs[asset] = round(float(last_signal * 0.7 + mom * 0.3), 4)

        fwd_mean = fwd_lower = fwd_upper = 0.0
        if result.forward_pred is not None:
            fwd_mean = float(result.forward_pred.state_mean[0, 0])
            fwd_lower = float(result.forward_pred.conf_lower[0, 0])
            fwd_upper = float(result.forward_pred.conf_upper[0, 0])

        return DailyPrediction(
            date=last_date, composite_signal=round(last_signal, 4),
            confidence=round(confidence, 4), factor_signals=factor_vals,
            asset_signals=asset_sigs,
            forward_pred_mean=round(fwd_mean, 4),
            forward_pred_lower=round(fwd_lower, 4),
            forward_pred_upper=round(fwd_upper, 4),
        )


__all__ = ["DailyPrediction", "DailyAssetPredictor"]
