"""
Performance metrics for the phase-1 monthly backtest.

All functions accept a monthly return series (``pd.Series``).  Inputs
are coerced to ``float64`` and ``NaN`` rows are dropped at the entry
point of every function so callers don't have to pre-clean.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


MONTHS_PER_YEAR = 12


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or len(returns) == 0:
        return pd.Series(dtype="float64")
    return pd.to_numeric(returns, errors="coerce").dropna().astype("float64")


def annualised_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR) of a monthly return series."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    years = len(r) / MONTHS_PER_YEAR
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of monthly returns (sample stdev)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio with an optional risk-free rate.

    The risk-free rate is expressed *annually* and converted to a
    geometric monthly rate before subtraction.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    rf_monthly = (1.0 + risk_free_annual) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    excess = r - rf_monthly
    std = float(excess.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std) * np.sqrt(MONTHS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the cumulative equity curve, as a negative
    decimal (e.g. ``-0.35`` ⇒ -35 %)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by absolute max-drawdown (positive ⇒ better)."""
    mdd = max_drawdown(returns)
    cagr = annualised_return(returns)
    if mdd == 0.0 or np.isnan(mdd) or np.isnan(cagr):
        return float("nan")
    return cagr / abs(mdd)


def turnover(weights: pd.DataFrame) -> float:
    """Average **one-way** monthly turnover of a weight schedule.

    Defined as ``½ × mean(|w_t - w_{t-1}|.sum(axis=1))`` so that a full
    rotation (sell everything, buy something new) counts as 1.0.
    """
    if weights is None or weights.empty or len(weights) < 2:
        return float("nan")
    diffs = weights.fillna(0.0).diff().abs().sum(axis=1)
    return float(diffs.iloc[1:].mean()) * 0.5


def summary_metrics(
    returns: pd.Series,
    weights: Optional[pd.DataFrame] = None,
    risk_free_annual: float = 0.0,
) -> Dict[str, float]:
    """One-shot dictionary of the headline metrics used in the report."""
    out = {
        "ann_return": annualised_return(returns),
        "ann_volatility": annualised_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_annual=risk_free_annual),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "months": int(_clean(returns).shape[0]),
    }
    if weights is not None:
        out["turnover"] = turnover(weights)
    return out


__all__ = [
    "MONTHS_PER_YEAR",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "turnover",
    "summary_metrics",
]
