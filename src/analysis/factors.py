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
    "momentum":     ["trend_momentum", "cross_asset_momentum", "micro_momentum"],
    "value":        ["mean_reversion"],
    "volatility":   ["volatility_regime", "intraweek_volatility"],
    "microstructure": ["gap_signal", "close_momentum", "close_location"],
    "carry":        ["carry_yield_curve"],
    "macro":        ["macro_diffusion", "global_composite"],
    "risk":         ["credit_risk"],
    "liquidity":    ["liquidity_growth"],
    "leading":      ["leading_indicator"],
    "policy":       ["policy_uncertainty"],
    "inflation":    ["inflation_expectations"],
    "cross_market": ["cross_market_sync"],
    "structural":   ["productivity_trend", "defense_burden", "investment_cycle"],
}

STABLE_FACTORS = [
    "trend_momentum",
    "mean_reversion",
    "volatility_regime",
    "cross_asset_momentum",
    "carry_yield_curve",
    "inflation_expectations",
]

UNSTABLE_FACTORS = [
    "credit_risk",
    "policy_uncertainty",
    "macro_diffusion",
    "global_composite",
    "liquidity_growth",
    "leading_indicator",
    "cross_market_sync",
]

FACTOR_LITERATURE: Dict[str, str] = {
    "trend_momentum": "Jegadeesh & Titman (1993) — Returns to buying winners and selling losers. "
                      "Multi-timeframe extension per Moskowitz, Ooi & Pedersen (2012).",
    "micro_momentum": "Jegadeesh & Titman (1993) momentum at a 3-7 day horizon; "
                      "Lehmann (1990) — very short-window momentum straddles continuation "
                      "and reversal, so it modulates confidence rather than vetoing direction. "
                      "Added 2026W33 to catch weekly turns the 20/60/120-day trend misses.",
    "intraweek_volatility": "Schwert (1989) volatility clustering at the weekly scale. "
                            "5-day realized vol vs 60-day baseline: a short-term vol spike "
                            "marks chop / turning points, calm vol supports continuation.",
    "gap_signal": "French (1980) — Stock returns and the weekend effect. "
                  "The weekend/holiday gap (last close → next open) prices overnight news; "
                  "persistent up-gaps = bullish news absorption, down-gaps = bearish.",
    "close_momentum": "Kraus & Stoll (1972) — Price impacts of block trading / auction mechanics. "
                      "Daily close-to-open return measures whether the final auction absorbed "
                      "buying or selling pressure; persistent close strength = bullish absorption. "
                      "Verified 2026W34: IC +0.151 full-period, +0.134 since 2024, 56.5% hit rate.",
    "close_location": "Osler (2003) / intraday price-range location. "
                      "Close position within the day's high-low range reflects who controlled "
                      "the close — near-high closes = accumulation, near-low closes = distribution. "
                      "Verified 2026W34: IC +0.102 full-period, +0.120 since 2024, 58.4% hit rate.",
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
    "cross_market_sync": "Longin & Solnik (2001) — Extreme correlation of international equity markets. "
                         "Cross-market synchronization as global risk appetite signal.",
    "productivity_trend": "Solow (1957) — Technical change and the aggregate production function. "
                          "Gordon (2016) — The rise and fall of American growth. "
                          "Labor productivity growth as structural economic driver.",
    "defense_burden": "Ramey (2011) — Identifying government spending shocks. "
                      "High defense/GDP → unified national purpose → lower policy uncertainty. "
                      "Declining defense burden since 1950s correlates with rising EPU (r=-0.62).",
    "investment_cycle": "Keynes (1936) — The General Theory. "
                        "Private investment as the 'animal spirits' cycle. "
                        "Investment growth leads productivity and employment.",
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


def micro_momentum(prices: pd.Series, window: int = 5) -> pd.Series:
    """5-day micro momentum for short-term turn detection (2026W33).

    Deviation of price from its 5-day SMA, tanh-amplified (the same
    ``tanh`` scaling idiom as trend_momentum).  Positive = short-term
    up-drift.  Operates on a 3-7 day horizon the 20/60/120-day trend
    factor cannot see — catches the weekly direction reversals that
    blind-sided the model during W30-W31 (P99 EPU, hit rate 22-27%).
    """
    if prices is None or len(prices) < window:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    sma = prices.rolling(window, min_periods=window // 2).mean()
    dev = (prices / sma) - 1.0
    # 5-day deviations are small; *30 makes a 1% 5-day move ≈ 0.29,
    # a 3% move ≈ 0.72 — sensitive but not saturated at normal range.
    return np.tanh(dev * 30.0).clip(-1, 1)


def intraweek_volatility(prices: pd.Series, window: int = 5, base_window: int = 60) -> pd.Series:
    """Intra-week volatility ratio: 5-day vol vs 60-day baseline (2026W33).

    Elevated 5-day vol relative to the 60-day baseline marks chop and
    short-term turning points (Schwert 1989 clustering at weekly scale);
    calm 5-day vol supports continuation of the current drift.
    Convention matches volatility_regime: positive = low vol (risk-on).
    """
    if prices is None or len(prices) < max(window, base_window):
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    returns = prices.pct_change(fill_method=None)
    short_vol = returns.rolling(window, min_periods=window // 2).std()
    base_vol = returns.rolling(base_window, min_periods=base_window // 2).std()
    base_vol = base_vol.replace(0.0, 1e-8)
    ratio = short_vol / base_vol
    return (1.0 - np.tanh(ratio * 3)).clip(-1, 1)


def gap_signal(prices: pd.Series, open_prices: Optional[pd.Series] = None) -> pd.Series:
    """Weekend/holiday gap effect — overnight news absorption (2026W33).

    A trading pause of >= 2 calendar days (weekends, holidays) opens a
    window for overnight/weekend news.  The gap from the last close to
    the next open measures how that news is priced in.  Persistent
    up-gaps = bullish absorption (continuation); persistent down-gaps =
    bearish absorption (reversal risk).  Non-pause days carry 0, so the
    trailing mean of this series is the average realized weekend gap.

    When true open prices are not available (predict() passes close
    prices only), the close-to-close return across the pause is used as
    a proxy — it includes the first day's intraday drift but is dominated
    by the gap component.
    """
    if prices is None or len(prices) < 3:
        return pd.Series(0.0, index=prices.index if prices is not None else pd.DatetimeIndex([]))
    idx = prices.index
    if isinstance(idx, pd.DatetimeIndex):
        cal_gap = pd.Series(idx).diff().dt.days.fillna(0)
        pause = pd.Series((cal_gap >= 2).values, index=idx)
        pause.iloc[0] = False
    else:
        pause = pd.Series(False, index=idx)

    close_prev = prices.shift(1)
    if open_prices is not None and len(open_prices) == len(prices):
        op = open_prices.reindex(idx)
        gap = op / close_prev - 1.0
    else:
        gap = prices / close_prev - 1.0
    sig = np.where(pause, np.tanh(gap * 120.0), 0.0)
    return pd.Series(sig, index=idx).fillna(0.0).clip(-1, 1)


def close_momentum(
    ohlc: Union[pd.DataFrame, pd.Series],
    open_prices: Optional[pd.Series] = None,
    close_col: str = "Close",
    open_col: str = "Open",
    window: int = 5,
) -> pd.Series:
    """Intraday close momentum — close vs open absorption (2026W34).

    The daily close-to-open return: how much of the session's move was
    printed in the final auction relative to the opening print.
    Persistently positive values mean buyers controlled the close
    (bullish absorption into the auction); persistently negative means
    sellers pressed into the close.  Averaged over a trailing ``window``
    (default 5) so a single session does not dominate.

    Verified 2026W34 exploration: full-period IC +0.151 (2024+: +0.134,
    directional hit rate 56.5%) at the 1y-forward weekly horizon — the
    strongest of the intraday micro-factors, and near-independent of the
    momentum family (corr with micro_momentum ≈ 0.08-0.13).

    Accepts an OHLC DataFrame (columns ``close_col``/``open_col``) or a
    close-price Series plus explicit ``open_prices``; returns a Series
    aligned to the input index.
    """
    if ohlc is None:
        return pd.Series(dtype=float)
    if isinstance(ohlc, pd.DataFrame):
        if close_col not in ohlc.columns or open_col not in ohlc.columns:
            return pd.Series(dtype=float)
        close = ohlc[close_col].astype(float)
        op = ohlc[open_col].astype(float)
    else:
        close = ohlc.astype(float)
        op = None
        if open_prices is not None and len(open_prices) == len(close):
            op = open_prices.reindex(close.index).astype(float)
    if op is None or len(close) < 2:
        return pd.Series(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        day_ret = close / op - 1.0
    day_ret = day_ret.replace([np.inf, -np.inf], np.nan)
    return day_ret.rolling(window, min_periods=2).mean().clip(-1, 1).fillna(0.0)


def close_location(
    ohlc: Union[pd.DataFrame, pd.Series],
    high_prices: Optional[pd.Series] = None,
    low_prices: Optional[pd.Series] = None,
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    window: int = 5,
) -> pd.Series:
    """Position of the close within the day's high-low range (2026W34).

    (Close - Low) / (High - Low): the relative location of the closing
    price inside each day's traded range.  Closes near the high = buyers
    absorbed selling into the close (accumulation); closes near the low =
    distribution.  Averaged over a trailing ``window`` (default 5).

    Raw series is in [0, 1] (centered to [-1, 1] via ``2*x - 1`` per the
    module contract); the z-score layer downstream centers on the annual
    distribution, and affine transforms leave z-scores invariant.

    Verified 2026W34 exploration: full-period IC +0.102 (2024+: +0.120,
    directional hit rate 58.4%).

    Accepts an OHLC DataFrame (columns ``close_col``/``high_col``/
    ``low_col``) or a close-price Series plus explicit ``high_prices``
    and ``low_prices``; returns a Series aligned to the input index.
    """
    if ohlc is None:
        return pd.Series(dtype=float)
    if isinstance(ohlc, pd.DataFrame):
        if close_col not in ohlc.columns or high_col not in ohlc.columns or low_col not in ohlc.columns:
            return pd.Series(dtype=float)
        close = ohlc[close_col].astype(float)
        high = ohlc[high_col].astype(float)
        low = ohlc[low_col].astype(float)
    else:
        close = ohlc.astype(float)
        high = low = None
        if high_prices is not None and low_prices is not None:
            high = high_prices.reindex(close.index).astype(float)
            low = low_prices.reindex(close.index).astype(float)
    if high is None or low is None or len(close) < 2:
        return pd.Series(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = (high - low).replace(0.0, np.nan)
        loc = (close - low) / rng
    loc = loc.replace([np.inf, -np.inf], np.nan)
    return (loc.rolling(window, min_periods=2).mean() * 2.0 - 1.0).clip(-1, 1).fillna(0.0)


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


def cross_market_sync(
    asset_returns: Dict[str, pd.Series],
    lookback: int = 60,
) -> pd.Series:
    """Cross-market synchronization (Longin & Solnik 2001).

    Average pairwise correlation of global asset returns over a rolling window.
    High sync = global risk-on / contagion.  Low sync = divergence / risk-off.

    Returns signal in [-1, 1] where positive = synchronized (risk-on).
    """
    if not asset_returns or len(asset_returns) < 2:
        return pd.Series(dtype=float)
    ret_df = pd.DataFrame(asset_returns).dropna(how='all')
    if ret_df.shape[1] < 2:
        return pd.Series(dtype=float)
    rolling_corr = ret_df.rolling(lookback, min_periods=lookback // 2).corr()
    n_assets = ret_df.shape[1]
    sync = pd.Series(0.0, index=ret_df.index, dtype=float)
    for t in range(lookback, len(ret_df)):
        try:
            corr_mat = rolling_corr.iloc[t * n_assets:(t + 1) * n_assets]
            if corr_mat.shape[0] < 2:
                continue
            avg_corr = (corr_mat.values.sum() - n_assets) / (n_assets * (n_assets - 1))
            sync.iloc[t] = avg_corr
        except Exception:
            continue
    sync = sync.fillna(0)
    return np.tanh(sync * 3).clip(-1, 1)


def productivity_trend(prod_series: pd.Series, lookback: int = 10) -> pd.Series:
    """Labor productivity growth trend (Solow 1957; Gordon 2016).

    Year-over-year productivity growth rate, z-scored over long history.
    Positive = productivity acceleration = structural bull case.
    """
    if prod_series is None or len(prod_series) < lookback:
        return pd.Series(dtype=float)
    growth = prod_series.pct_change(4, fill_method=None) * 100.0
    return zscore(growth, lookback * 4)


def defense_burden(defense_gdp_series: pd.Series, lookback: int = 40) -> pd.Series:
    """Defense burden factor (Ramey 2011).

    Defense/GDP ratio, z-scored.  High defense spending historically
    correlates with LOW policy uncertainty (r=-0.62) — unified national
    purpose reduces domestic policy noise.

    Signal: higher defense burden → lower EPU → more predictable policy.
    """
    if defense_gdp_series is None or len(defense_gdp_series) < lookback:
        return pd.Series(dtype=float)
    return -zscore(defense_gdp_series, lookback)


def investment_cycle(invest_series: pd.Series, lookback: int = 10) -> pd.Series:
    """Private investment cycle (Keynes 1936).

    Year-over-year real private investment growth, z-scored.
    Investment leads the business cycle — rising investment = expansion ahead.
    """
    if invest_series is None or len(invest_series) < lookback:
        return pd.Series(dtype=float)
    growth = invest_series.pct_change(4, fill_method=None) * 100.0
    return zscore(growth, lookback * 4)


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
    prod_series: Optional[pd.Series] = None,
    defense_series: Optional[pd.Series] = None,
    invest_series: Optional[pd.Series] = None,
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

    if len(asset_prices) >= 3:
        asset_rets = {k: px.pct_change(fill_method=None) for k, px in asset_prices.items() if px is not None and not px.empty}
        factors["cross_market_sync"] = cross_market_sync(asset_rets)

    if prod_series is not None and not prod_series.empty:
        factors["productivity_trend"] = productivity_trend(prod_series)

    if defense_series is not None and not defense_series.empty:
        factors["defense_burden"] = defense_burden(defense_series)

    if invest_series is not None and not invest_series.empty:
        factors["investment_cycle"] = investment_cycle(invest_series)

    return factors


__all__ = [
    "FACTOR_GROUPS",
    "FACTOR_LITERATURE",
    "zscore",
    "trend_momentum",
    "mean_reversion",
    "volatility_regime",
    "micro_momentum",
    "intraweek_volatility",
    "gap_signal",
    "close_momentum",
    "close_location",
    "carry_yield_curve",
    "macro_diffusion",
    "cross_asset_momentum",
    "global_composite_trend",
    "credit_risk",
    "cross_market_sync",
    "_compute_all_factors",
    "STABLE_FACTORS",
    "UNSTABLE_FACTORS",
]
