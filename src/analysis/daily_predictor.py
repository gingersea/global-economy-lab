"""
Daily short-term asset return predictor.

Generates daily directional signals for each asset, conditioned on the
predicted macro regime from :mod:`src.analysis.macro_predictor`.

Uses a blend of momentum, mean-reversion, and volatility-adjusted signals.
Regime conditioning applies different signal weights based on which macro
environment is predicted (e.g., trend-following in recovery, mean-reversion
in recession).

Pure pandas + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger


REGIME_SIGNAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "recovery": {
        "momentum": 0.50,
        "mean_reversion": 0.10,
        "volatility_adjusted": 0.30,
        "trend_strength": 0.10,
    },
    "overheat": {
        "momentum": 0.40,
        "mean_reversion": 0.15,
        "volatility_adjusted": 0.30,
        "trend_strength": 0.15,
    },
    "stagflation": {
        "momentum": 0.20,
        "mean_reversion": 0.30,
        "volatility_adjusted": 0.25,
        "trend_strength": 0.25,
    },
    "recession": {
        "momentum": 0.15,
        "mean_reversion": 0.35,
        "volatility_adjusted": 0.25,
        "trend_strength": 0.25,
    },
    "unknown": {
        "momentum": 0.30,
        "mean_reversion": 0.20,
        "volatility_adjusted": 0.30,
        "trend_strength": 0.20,
    },
}


@dataclass
class DailyPredictorConfig:
    """Configuration for :class:`DailyAssetPredictor`.

    Attributes:
        momentum_lookback:       Days for momentum calculation.
        mean_rev_lookback:       Days for mean-reversion signal.
        vol_lookback:            Days for volatility estimation.
        trend_strength_lookback: Days for trend strength (fraction of
                                 positive days).
        signal_smoothing:        EMA span for smoothing raw signals.
        regime_weights:          Per-regime signal blending weights.
    """

    momentum_lookback: int = 20
    mean_rev_lookback: int = 5
    vol_lookback: int = 20
    trend_strength_lookback: int = 60
    signal_smoothing: int = 3
    regime_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: dict(REGIME_SIGNAL_WEIGHTS)
    )


@dataclass
class DailyPrediction:
    """Container for one day's prediction across assets.

    Attributes:
        date:              Prediction date.
        predicted_regime:  The macro regime used to condition signals.
        signals:           Dict ``{asset_key: composite_signal}`` where signal
                           is in [-1, 1] (1 = strongest bullish).
        components:        Dict ``{asset_key: {component_name: signal}}``
                           with raw component signals before blending.
        confidence:        Dict ``{asset_key: float}`` — signal strength ×
                           regime confidence.
    """

    date: pd.Timestamp
    predicted_regime: str
    signals: Dict[str, float]
    components: Dict[str, Dict[str, float]]
    confidence: Dict[str, float]


def momentum_signal(
    prices: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Compute a momentum signal: (price / SMA) - 1, scaled to [-1, 1].

    Args:
        prices:   Daily price series.
        lookback: SMA window.

    Returns:
        Series of momentum scores in approximately [-1, 1].
    """
    if prices is None or len(prices) < lookback:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    sma = prices.rolling(lookback, min_periods=lookback // 2).mean()
    raw = (prices / sma) - 1.0
    return np.tanh(raw * 10.0)


def mean_reversion_signal(
    prices: pd.Series,
    lookback: int = 5,
) -> pd.Series:
    """Compute a mean-reversion signal: negative of short-term deviation.

    A large positive deviation → negative signal (expect revert down).
    A large negative deviation → positive signal (expect revert up).

    Args:
        prices:   Daily price series.
        lookback: Short SMA window.

    Returns:
        Series of mean-reversion scores. Positive = oversold (bullish).
    """
    if prices is None or len(prices) < lookback:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    sma = prices.rolling(lookback, min_periods=lookback // 2).mean()
    dev = (prices / sma) - 1.0
    return -np.tanh(dev * 15.0)


def volatility_adjusted_signal(
    prices: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Volatility-adjusted trend: momentum divided by recent volatility.

    High volatility → dampen the momentum signal.
    Low volatility → amplify the momentum signal.

    Args:
        prices:   Daily price series.
        lookback: Window for both momentum and volatility.

    Returns:
        Series of volatility-adjusted scores.
    """
    if prices is None or len(prices) < lookback:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    mom = momentum_signal(prices, lookback)
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(lookback, min_periods=lookback // 2).std()
    mean_vol = vol.mean()
    if mean_vol == 0 or np.isnan(mean_vol):
        return mom
    adj_factor = mean_vol / vol.replace(0, mean_vol)
    return (mom * adj_factor).clip(-1, 1)


def trend_strength_signal(
    prices: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Fraction of positive days over the lookback, centered to [-0.5, 0.5].

    Values > 0 indicate a strong uptrend; values < 0 indicate a downtrend.

    Args:
        prices:   Daily price series.
        lookback: Window for counting positive days.

    Returns:
        Series of trend-strength scores.
    """
    if prices is None or len(prices) < 5:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    returns = prices.pct_change(fill_method=None)
    frac_positive = returns.rolling(lookback, min_periods=5).apply(
        lambda x: (x > 0).mean(), raw=False
    )
    return frac_positive - 0.5


class DailyAssetPredictor:
    """Generate daily directional signals for multiple assets.

    Blends momentum, mean-reversion, volatility-adjusted, and trend-strength
    signals, with weights determined by the predicted macro regime.

    Args:
        config: :class:`DailyPredictorConfig`.
    """

    def __init__(self, config: Optional[DailyPredictorConfig] = None):
        self.config = config or DailyPredictorConfig()

    def compute_raw_signals(
        self,
        prices: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Compute all four raw signal components for a single asset.

        Args:
            prices: Daily price series.

        Returns:
            Dict mapping component name to signal Series.
        """
        cfg = self.config
        if prices is None or prices.empty:
            empty_idx = pd.DatetimeIndex([])
            return {
                "momentum": pd.Series(dtype=float),
                "mean_reversion": pd.Series(dtype=float),
                "volatility_adjusted": pd.Series(dtype=float),
                "trend_strength": pd.Series(dtype=float),
            }
        return {
            "momentum": momentum_signal(prices, cfg.momentum_lookback),
            "mean_reversion": mean_reversion_signal(prices, cfg.mean_rev_lookback),
            "volatility_adjusted": volatility_adjusted_signal(prices, cfg.vol_lookback),
            "trend_strength": trend_strength_signal(prices, cfg.trend_strength_lookback),
        }

    def blend_signals(
        self,
        raw: Dict[str, pd.Series],
        regime: str,
        smoothing: bool = True,
    ) -> pd.Series:
        """Blend raw signal components using regime-specific weights.

        Args:
            raw:        Dict of component name → signal Series.
            regime:     The macro regime to use for weight selection.
            smoothing:  Apply EMA smoothing to the blended signal.

        Returns:
            Single composite signal Series in [-1, 1].
        """
        weights = self.config.regime_weights.get(regime, self.config.regime_weights["unknown"])
        if not raw:
            return pd.Series(dtype=float)

        first_key = next(iter(raw.keys()))
        idx = raw[first_key].index
        blended = pd.Series(0.0, index=idx, dtype=float)
        total_w = 0.0
        for name, weight in weights.items():
            if name in raw and not raw[name].empty:
                blended = blended.add(raw[name].fillna(0.0) * weight, fill_value=0.0)
                total_w += weight

        if total_w > 0:
            blended = blended / total_w
        blended = blended.clip(-1, 1)

        if smoothing and self.config.signal_smoothing > 1 and len(blended) > 1:
            blended = blended.ewm(span=self.config.signal_smoothing, min_periods=1).mean()
        return blended

    def predict(
        self,
        asset_dfs: Dict[str, pd.DataFrame],
        predicted_regime: str,
        price_col: str = "close",
    ) -> DailyPrediction:
        """Generate a single-day prediction for all assets.

        Uses the **last row** of each asset DataFrame as the prediction date.

        Args:
            asset_dfs:        Dict ``{asset_key: daily_df}``.
            predicted_regime: The macro regime to condition signals on.
            price_col:        Price column name in each DataFrame.

        Returns:
            :class:`DailyPrediction` with composite signals and components.
        """
        if not asset_dfs:
            raise ValueError("DailyAssetPredictor.predict: asset_dfs is empty.")

        date = None
        signals: Dict[str, float] = {}
        components: Dict[str, Dict[str, float]] = {}
        confidence: Dict[str, float] = {}

        for key, df in asset_dfs.items():
            if df is None or df.empty:
                continue
            if isinstance(df.index, pd.DatetimeIndex):
                px = df[price_col] if price_col in df.columns else df.iloc[:, 0]
            else:
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"])).sort_index()
                px = df[price_col] if price_col in df.columns else df.select_dtypes(include="number").iloc[:, 0]

            raw = self.compute_raw_signals(px)
            blended = self.blend_signals(raw, predicted_regime)

            if date is None and not blended.empty:
                date = blended.index[-1]

            if not blended.empty:
                last_signal = float(blended.iloc[-1])
                signals[key] = round(last_signal, 4)
                components[key] = {
                    name: round(float(s.iloc[-1]), 4)
                    for name, s in raw.items()
                    if not s.empty and len(s) > 0
                }
                confidence[key] = round(abs(last_signal), 4)

        if date is None:
            date = pd.Timestamp.now()

        return DailyPrediction(
            date=date,
            predicted_regime=predicted_regime,
            signals=signals,
            components=components,
            confidence=confidence,
        )

    def predict_series(
        self,
        asset_dfs: Dict[str, pd.DataFrame],
        regime_series: pd.Series,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Generate daily signals over a date range, conditioned on a
        regime time series.

        Args:
            asset_dfs:      Dict ``{asset_key: daily_df}``.
            regime_series:  Series with DatetimeIndex and regime labels.
            price_col:      Price column name.

        Returns:
            DataFrame with columns like ``sp500_signal``, ``gold_signal``, etc.
        """
        all_dates = pd.DatetimeIndex([])
        for df in asset_dfs.values():
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    all_dates = all_dates.union(df.index)
                elif "date" in df.columns:
                    all_dates = all_dates.union(pd.DatetimeIndex(pd.to_datetime(df["date"])))

        if len(all_dates) == 0:
            return pd.DataFrame()

        results = pd.DataFrame(index=all_dates.sort_values())
        for key, df in asset_dfs.items():
            if df is None or df.empty:
                continue
            if isinstance(df.index, pd.DatetimeIndex):
                px = df[price_col] if price_col in df.columns else df.iloc[:, 0]
            else:
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"])).sort_index()
                px = df[price_col] if price_col in df.columns else df.select_dtypes(include="number").iloc[:, 0]

            raw = self.compute_raw_signals(px)

            regime_per_day = regime_series.reindex(px.index, method="ffill").fillna("unknown")
            blended = pd.Series(np.nan, index=px.index)
            for reg in regime_per_day.unique():
                mask = regime_per_day == reg
                if mask.sum() == 0:
                    continue
                reg_blend = self.blend_signals(
                    {k: v.loc[mask] for k, v in raw.items()},
                    reg,
                    smoothing=False,
                )
                blended.loc[mask] = reg_blend

            results[f"{key}_signal"] = blended

        results = results.dropna(how="all")
        return results


__all__ = [
    "REGIME_SIGNAL_WEIGHTS",
    "DailyPredictorConfig",
    "DailyPrediction",
    "DailyAssetPredictor",
    "momentum_signal",
    "mean_reversion_signal",
    "volatility_adjusted_signal",
    "trend_strength_signal",
]
