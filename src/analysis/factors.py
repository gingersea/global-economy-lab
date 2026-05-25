"""
Unified multi-factor library.

Every factor is a pure function: ``(prices_or_indicators, **params) → pd.Series``.
Factors work at any frequency (daily, weekly, monthly) — the caller controls
resolution via the input data.  All outputs are standardised to approximately
[-1, 1] for ensemble compatibility.

Factors cover:
- Price momentum (multi-timeframe)
- Mean reversion
- Volatility regime
- Carry / yield curve
- Macro diffusion (breadth of improving indicators)
- Cross-asset momentum (relative strength)
- Global composite trend
- Credit / risk appetite
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def zscore(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling z-score: (value - rolling_mean) / rolling_std."""
    if series is None or len(series) < lookback:
        return pd.Series(0.0, index=series.index if series is not None else pd.DatetimeIndex([]))
    mean = series.rolling(lookback, min_periods=lookback // 2).mean()
    std = series.rolling(lookback, min_periods=lookback // 2).std()
    std = std.replace(0.0, 1e-8)
    return ((series - mean) / std).clip(-3, 3) / 3.0


def trend_momentum(prices: pd.Series, fast: int = 20, medium: int = 60, slow: int = 120) -> pd.Series:
    """Multi-timeframe momentum: weighted average of fast/medium/slow trend.

    Returns signal in [-1, 1] where positive = uptrend.
    """
    if prices is None or len(prices) < max(fast, medium, slow):
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    f = np.tanh((prices / prices.rolling(fast, min_periods=fast // 2).mean() - 1) * 10)
    m = np.tanh((prices / prices.rolling(medium, min_periods=medium // 2).mean() - 1) * 8)
    s = np.tanh((prices / prices.rolling(slow, min_periods=slow // 2).mean() - 1) * 6)
    return (0.50 * f + 0.30 * m + 0.20 * s).clip(-1, 1)


def mean_reversion(prices: pd.Series, lookback: int = 10) -> pd.Series:
    """Short-term mean reversion: negative of deviation from SMA.

    Positive = oversold (bullish), negative = overbought (bearish).
    """
    if prices is None or len(prices) < lookback:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    sma = prices.rolling(lookback, min_periods=lookback // 2).mean()
    dev = (prices / sma) - 1.0
    return -np.tanh(dev * 15.0).clip(-1, 1)


def volatility_regime(prices: pd.Series, lookback: int = 20, vol_lookback: int = 252) -> pd.Series:
    """Volatility regime: current vol vs historical vol percentile.

    Positive = low vol (risk-on), negative = high vol (risk-off).
    """
    if prices is None or len(prices) < max(lookback, vol_lookback):
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    returns = prices.pct_change(fill_method=None)
    current_vol = returns.rolling(lookback, min_periods=lookback // 2).std()
    hist_vol = returns.rolling(vol_lookback, min_periods=lookback).std()
    hist_vol = hist_vol.replace(0.0, 1e-8)
    ratio = current_vol / hist_vol
    return (1.0 - np.tanh(ratio * 3)).clip(-1, 1)


def carry_yield_curve(
    short_rate: Union[pd.Series, float],
    long_rate: Union[pd.Series, float],
    lookback: int = 252,
) -> pd.Series:
    """Yield curve slope as a factor: steep = expansionary, flat/inverted = restrictive.

    Normalised as z-score of (long - short) spread.
    """
    if isinstance(short_rate, (int, float)):
        short_rate = pd.Series(float(short_rate), index=long_rate.index if hasattr(long_rate, 'index') else None)
    if isinstance(long_rate, (int, float)):
        long_rate = pd.Series(float(long_rate), index=short_rate.index)
    spread = long_rate - short_rate
    return zscore(spread, lookback)


def macro_diffusion(
    indicators: Dict[str, pd.Series],
    lookback: int = 6,
) -> pd.Series:
    """Fraction of macro indicators that are improving over *lookback* months.

    1.0 = all improving (expansion), 0.0 = all deteriorating (contraction).
    Centered to [-1, 1].
    """
    if not indicators:
        return pd.Series(dtype=float)
    idx = None
    for s in indicators.values():
        if s is not None and not s.empty:
            idx = s.index
            break
    if idx is None:
        return pd.Series(dtype=float)
    improving = pd.DataFrame(index=idx)
    for name, series in indicators.items():
        if series is None or series.empty:
            continue
        improving[name] = (series - series.shift(lookback)) > 0
    fraction = improving.mean(axis=1)
    return (fraction * 2.0 - 1.0).clip(-1, 1)


def cross_asset_momentum(
    asset_prices: Dict[str, pd.Series],
    lookback: int = 60,
) -> pd.DataFrame:
    """Relative momentum across assets.  Returns a DataFrame with one column
    per asset, where values are the asset's return rank (0-1) among peers.

    High rank = strong relative momentum.
    """
    if not asset_prices:
        return pd.DataFrame()
    idx = None
    for s in asset_prices.values():
        if s is not None and not s.empty:
            idx = s.index
            break
    if idx is None:
        return pd.DataFrame()
    returns = pd.DataFrame(index=idx)
    for key, px in asset_prices.items():
        if px is None or px.empty:
            continue
        rets = px.pct_change(fill_method=None)
        returns[key] = rets.rolling(lookback, min_periods=lookback // 2).sum()
    n_assets = len(returns.columns)
    if n_assets < 2:
        return pd.DataFrame(0.0, index=idx, columns=returns.columns)
    ranks = returns.rank(axis=1, method="min") / (n_assets + 1)
    return (ranks * 2.0 - 1.0).clip(-1, 1)


def global_composite_trend(
    regional_indicators: Dict[str, pd.Series],
    lookback: int = 60,
) -> pd.Series:
    """Composite global trend from multiple regional indicators.

    Each indicator is z-scored, then averaged.  Higher = stronger global growth.
    """
    if not regional_indicators:
        return pd.Series(dtype=float)
    idx = None
    for s in regional_indicators.values():
        if s is not None and not s.empty:
            idx = s.index
            break
    if idx is None:
        return pd.Series(dtype=float)
    composite = pd.DataFrame(index=idx)
    for name, series in regional_indicators.items():
        if series is None or series.empty:
            continue
        composite[name] = zscore(series, lookback)
    result = composite.mean(axis=1)
    return result.clip(-1, 1)


def credit_risk(risk_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Credit/risk appetite factor.  Uses NFCI, VIX, or credit spreads.

    Higher risk indicator → negative signal (risk-off).
    """
    if risk_series is None or risk_series.empty:
        return pd.Series(dtype=float)
    z = zscore(risk_series, lookback)
    return -z.clip(-1, 1)


def _compute_all_factors(
    asset_prices: Dict[str, pd.Series],
    macro_indicators: Dict[str, pd.Series],
    yield_short: Optional[pd.Series] = None,
    yield_long: Optional[pd.Series] = None,
    risk_series: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """Compute all factor signals for the given inputs.

    Returns a dict of factor_name → signal_series.
    Asset-level factors are averaged across assets.
    """
    factors: Dict[str, pd.Series] = {}

    asset_factors = {}
    for asset, px in asset_prices.items():
        if px is None or px.empty:
            continue
        asset_factors[asset] = {
            "trend_momentum": trend_momentum(px),
            "mean_reversion": mean_reversion(px),
            "volatility_regime": volatility_regime(px),
        }

    if asset_factors:
        for fname in ["trend_momentum", "mean_reversion", "volatility_regime"]:
            series_list = [af[fname] for af in asset_factors.values() if not af[fname].empty]
            if series_list:
                factors[fname] = pd.concat(series_list, axis=1).mean(axis=1)

    cross_mom = cross_asset_momentum(asset_prices)
    if not cross_mom.empty:
        factors["cross_asset_momentum"] = cross_mom.mean(axis=1)

    if yield_short is not None and yield_long is not None:
        factors["carry_yield_curve"] = carry_yield_curve(yield_short, yield_long)

    if macro_indicators:
        factors["macro_diffusion"] = macro_diffusion(macro_indicators)
        factors["global_composite"] = global_composite_trend(macro_indicators)

    if risk_series is not None and not risk_series.empty:
        factors["credit_risk"] = credit_risk(risk_series)

    return factors


__all__ = [
    "zscore",
    "trend_momentum",
    "mean_reversion",
    "volatility_regime",
    "carry_yield_curve",
    "macro_diffusion",
    "cross_asset_momentum",
    "global_composite_trend",
    "credit_risk",
    "_compute_all_factors",
]
