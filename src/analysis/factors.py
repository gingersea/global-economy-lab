"""
Unified multi-factor library with economic rationale.

Every factor is grounded in academic literature.  Factors are grouped
into economic categories to handle cross-factor correlations.

Categories:
- Momentum:    Trend-following across timeframes (Jegadeesh & Titman 1993)
- Value/Mean-Reversion: Short-term reversal (De Bondt & Thaler 1985; Poterba & Summers 1988)
- Volatility:  Risk regime detection (Schwert 1989; Bollerslev 1986)
- Carry:       Yield curve & rate expectations (Campbell & Shiller 1991; Fama & Bliss 1987)
- Macro:       Diffusion index & global trends (Stock & Watson 2002; Ludvigson & Ng 2009)
- Risk:        Credit & financial conditions (Gilchrist & Zakrajsek 2012; Adrian, Crump & Moench 2015)

All factors: (prices_or_indicators, **params) → pd.Series in [-1, 1].
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


FACTOR_GROUPS: Dict[str, List[str]] = {
    "momentum":     ["trend_momentum", "cross_asset_momentum"],
    "value":        ["mean_reversion"],
    "volatility":   ["volatility_regime"],
    "carry":        ["carry_yield_curve"],
    "macro":        ["macro_diffusion", "global_composite"],
    "risk":         ["credit_risk"],
    "liquidity":    ["liquidity_growth"],
    "leading":      ["leading_indicator"],
    "policy":       ["policy_uncertainty"],
    "inflation":    ["inflation_expectations"],
}

FACTOR_LITERATURE: Dict[str, str] = {
    "trend_momentum": "Jegadeesh & Titman (1993) — Returns to buying winners and selling losers. "
                      "Multi-timeframe extension per Moskowitz, Ooi & Pedersen (2012).",
    "mean_reversion": "De Bondt & Thaler (1985) — Does the stock market overreact? "
                      "Short-term reversal per Jegadeesh (1990).",
    "volatility_regime": "Schwert (1989) — Why does stock market volatility change over time? "
                         "Volatility clustering per Bollerslev (1986) GARCH.",
    "carry_yield_curve": "Campbell & Shiller (1991) — Yield spreads and interest rate movements. "
                         "Fama & Bliss (1987) — The information in long-maturity forward rates.",
    "macro_diffusion": "Stock & Watson (2002) — Macroeconomic forecasting using diffusion indexes. "
                       "Breadth of improving indicators signals expansion/contraction.",
    "cross_asset_momentum": "Asness, Moskowitz & Pedersen (2013) — Value and momentum everywhere. "
                            "Cross-asset relative strength across equities, bonds, commodities, FX.",
    "global_composite": "Ludvigson & Ng (2009) — Macro factors in bond risk premia. "
                        "Multi-region composite trend from macro indicators.",
    "credit_risk": "Gilchrist & Zakrajsek (2012) — Credit spreads and business cycle fluctuations. "
                   "NFCI: Adrian, Crump & Moench (2015) — Financial conditions indexes.",
    "liquidity_growth": "Friedman & Schwartz (1963) — A monetary history. "
                        "M2 growth as liquidity signal per Jensen, Johnson & Mercer (1996).",
    "leading_indicator": "Stock & Watson (1989) — New indexes of coincident and leading economic indicators. "
                         "Building permits as housing leading indicator per Leamer (2007).",
    "policy_uncertainty": "Baker, Bloom & Davis (2016) — Measuring economic policy uncertainty. "
                          "EPU index as political/regulatory risk factor.",
    "inflation_expectations": "Faust & Wright (2013) — Forecasting inflation. "
                              "Breakeven inflation as market-implied expectations.",
}


def zscore(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling z-score: (value - rolling_mean) / rolling_std."""
    if series is None or len(series) < lookback:
        return pd.Series(0.0, index=series.index if series is not None else pd.DatetimeIndex([]))
    mean = series.rolling(lookback, min_periods=lookback // 2).mean()
    std = series.rolling(lookback, min_periods=lookback // 2).std()
    std = std.replace(0.0, 1e-8)
    return ((series - mean) / std).clip(-3, 3) / 3.0


def trend_momentum(prices: pd.Series, fast: int = 20, medium: int = 60, slow: int = 120) -> pd.Series:
    """Multi-timeframe momentum (Jegadeesh & Titman 1993).

    Weighted average of fast/medium/slow trend signals.
    Positive = uptrend across timeframes.
    """
    if prices is None or len(prices) < max(fast, medium, slow):
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    f = np.tanh((prices / prices.rolling(fast, min_periods=fast // 2).mean() - 1) * 10)
    m = np.tanh((prices / prices.rolling(medium, min_periods=medium // 2).mean() - 1) * 8)
    s = np.tanh((prices / prices.rolling(slow, min_periods=slow // 2).mean() - 1) * 6)
    return (0.50 * f + 0.30 * m + 0.20 * s).clip(-1, 1)


def mean_reversion(prices: pd.Series, lookback: int = 10) -> pd.Series:
    """Short-term mean reversion (De Bondt & Thaler 1985).

    Negative of deviation from SMA. Positive = oversold (bullish reversal expected).
    """
    if prices is None or len(prices) < lookback:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    sma = prices.rolling(lookback, min_periods=lookback // 2).mean()
    dev = (prices / sma) - 1.0
    return -np.tanh(dev * 15.0).clip(-1, 1)


def volatility_regime(prices: pd.Series, lookback: int = 20, vol_lookback: int = 252) -> pd.Series:
    """Volatility regime detection (Schwert 1989; Bollerslev 1986).

    Current vol vs historical vol percentile. Positive = low vol (risk-on).
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
    """Yield curve slope (Campbell & Shiller 1991; Fama & Bliss 1987).

    Steep curve = expansionary expectations. Flat/inverted = restrictive.
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
    """Diffusion index (Stock & Watson 2002).

    Fraction of macro indicators improving over lookback.
    1.0 = all improving (broad expansion), 0.0 = all deteriorating (broad contraction).
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
    """Cross-asset relative momentum (Asness, Moskowitz & Pedersen 2013).

    Returns per-asset rank (0-1) among peers based on recent returns.
    High rank = strong relative momentum across asset classes.
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
    """Global composite trend (Ludvigson & Ng 2009).

    Multi-region macro indicator composite via z-score averaging.
    Higher = stronger synchronized global growth.
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
    """Credit/risk appetite (Gilchrist & Zakrajsek 2012).

    Higher risk indicator → tighter financial conditions → negative signal (risk-off).
    Uses NFCI, VIX, or credit spread as input.
    """
    if risk_series is None or risk_series.empty:
        return pd.Series(dtype=float)
    z = zscore(risk_series, lookback)
    return -z.clip(-1, 1)


def liquidity_growth(m2_series: pd.Series, lookback: int = 12) -> pd.Series:
    """Liquidity factor from M2 money supply growth (Friedman & Schwartz 1963).

    Year-over-year M2 growth rate, z-scored.  Higher = more liquidity = risk-on.
    """
    if m2_series is None or len(m2_series) < lookback:
        return pd.Series(dtype=float)
    yoy = m2_series.pct_change(lookback, fill_method=None) * 100.0
    return zscore(yoy, lookback * 10)


def leading_indicator(permit_series: pd.Series, lookback: int = 12) -> pd.Series:
    """Housing leading indicator from building permits (Leamer 2007).

    YoY change in building permits, z-scored.  Higher = expanding construction = growth ahead.
    """
    if permit_series is None or len(permit_series) < lookback:
        return pd.Series(dtype=float)
    yoy = permit_series.pct_change(lookback, fill_method=None) * 100.0
    return zscore(yoy, lookback * 10)


def policy_uncertainty(epu_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Economic Policy Uncertainty (Baker, Bloom & Davis 2016).

    Higher EPU → more uncertainty → negative signal (risk-off).
    """
    if epu_series is None or epu_series.empty:
        return pd.Series(dtype=float)
    z = zscore(epu_series, lookback)
    return -z.clip(-1, 1)


def inflation_expectations(breakeven_series: pd.Series, lookback: int = 60) -> pd.Series:
    """Market-implied inflation expectations (Faust & Wright 2013).

    10Y breakeven rate: rising → inflation fears, falling → disinflation.
    Centered so 2-2.5% is neutral (z-scored).
    """
    if breakeven_series is None or len(breakeven_series) < lookback:
        return pd.Series(dtype=float)
    return zscore(breakeven_series, lookback)


def _compute_all_factors(
    asset_prices: Dict[str, pd.Series],
    macro_indicators: Dict[str, pd.Series],
    yield_short: Optional[pd.Series] = None,
    yield_long: Optional[pd.Series] = None,
    risk_series: Optional[pd.Series] = None,
    m2_series: Optional[pd.Series] = None,
    permit_series: Optional[pd.Series] = None,
    epu_series: Optional[pd.Series] = None,
    breakeven_series: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
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

    if m2_series is not None and not m2_series.empty:
        factors["liquidity_growth"] = liquidity_growth(m2_series)

    if permit_series is not None and not permit_series.empty:
        factors["leading_indicator"] = leading_indicator(permit_series)

    if epu_series is not None and not epu_series.empty:
        factors["policy_uncertainty"] = policy_uncertainty(epu_series)

    if breakeven_series is not None and not breakeven_series.empty:
        factors["inflation_expectations"] = inflation_expectations(breakeven_series)

    return factors


__all__ = [
    "FACTOR_GROUPS",
    "FACTOR_LITERATURE",
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
