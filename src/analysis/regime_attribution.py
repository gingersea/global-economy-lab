"""
Per-regime and event-window attribution.

Addresses §5.3 (per-cycle heatmap) and §5.4 (key event windows) from
``docs/phase1_execution_plan.md``.  Pure pandas — no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


KEY_EVENTS: Dict[str, tuple[str, str, str]] = {
    "2008 GFC": ("2008-09-01", "2009-03-01", "Global Financial Crisis"),
    "2020 COVID": ("2020-02-01", "2020-06-01", "COVID-19 pandemic shock"),
    "2022 Inflation": ("2022-01-01", "2022-12-01", "Inflation & rate hike cycle"),
}


@dataclass
class EventWindow:
    """A named event with start/end dates for slicing."""

    name: str
    start: str
    end: str
    description: str = ""


def regime_heatmap(
    panel: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    regime_col: str = "regime",
) -> pd.DataFrame:
    """Per-regime mean monthly return for each asset.

    Args:
        panel:   Monthly panel with a regime label column and ``*_ret`` columns.
        returns: Optional explicit returns DataFrame (otherwise derived from panel).
        regime_col: Name of the regime column.

    Returns:
        DataFrame indexed by regime, with one column per asset.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()

    if returns is None:
        ret_cols = [c for c in panel.columns if c.endswith("_ret")]
        returns = panel[ret_cols]

    if regime_col not in panel.columns:
        return pd.DataFrame()

    regimes = panel[regime_col].dropna()
    common_idx = regimes.index.intersection(returns.index)
    if len(common_idx) == 0:
        return pd.DataFrame()

    regimes = regimes.loc[common_idx]
    rets = returns.loc[common_idx]

    heatmap_rows = []
    for regime_name in sorted(regimes.unique()):
        mask = regimes == regime_name
        if mask.sum() == 0:
            continue
        mean_rets = rets.loc[mask].mean()
        heatmap_rows.append({"regime": regime_name, **mean_rets.to_dict()})

    heatmap = pd.DataFrame(heatmap_rows).set_index("regime")
    heatmap.index.name = "regime"
    return heatmap


def event_window_returns(
    panel: pd.DataFrame,
    events: Optional[Dict[str, tuple[str, str, str]]] = None,
    start: str = "2008-01-01",
) -> pd.DataFrame:
    """Compute cumulative returns for each asset over key event windows.

    Args:
        panel:  Monthly panel with ``*_ret`` columns.
        events: Dict ``{name: (start, end, description)}``.  Defaults to
                :data:`KEY_EVENTS`.
        start:  Earliest date to include asset returns from.

    Returns:
        DataFrame indexed by event name, with one column per asset (cumulative
        return over the window as a decimal).
    """
    events = events or KEY_EVENTS
    ret_cols = [c for c in panel.columns if c.endswith("_ret")]
    if not ret_cols:
        return pd.DataFrame()

    panel = panel.loc[panel.index >= pd.Timestamp(start)]
    rows = []
    for name, (ev_start, ev_end, desc) in events.items():
        window = panel.loc[
            (panel.index >= pd.Timestamp(ev_start))
            & (panel.index <= pd.Timestamp(ev_end)),
            ret_cols,
        ]
        if window.empty:
            continue
        cum = (1.0 + window).prod() - 1.0
        rows.append({"event": name, "description": desc, **cum.to_dict()})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("event")


def regime_asset_table(
    panel: pd.DataFrame,
    regime_col: str = "regime",
) -> pd.DataFrame:
    """Build a wide summary table: regime × asset (mean return, volatility, win rate).

    Args:
        panel:      Monthly panel with regime labels and ``*_ret`` columns.
        regime_col: Name of the regime label column.

    Returns:
        MultiIndex-column DataFrame: level-0 = metric, level-1 = asset.
    """
    if panel is None or panel.empty or regime_col not in panel.columns:
        return pd.DataFrame()

    ret_cols = [c for c in panel.columns if c.endswith("_ret")]
    if not ret_cols:
        return pd.DataFrame()

    valid = panel[[regime_col] + ret_cols].dropna(subset=[regime_col])
    regimes = valid[regime_col].unique()

    frames: dict[str, pd.DataFrame] = {}
    for regime_name in sorted(regimes):
        mask = valid[regime_col] == regime_name
        sub = valid.loc[mask, ret_cols]
        if sub.empty or len(sub) < 2:
            continue

        row = {}
        for col in ret_cols:
            r = sub[col].dropna()
            if len(r) < 2:
                continue
            row[(col, "mean_return")] = float(r.mean())
            row[(col, "volatility")] = float(r.std(ddof=1))
            row[(col, "win_rate")] = float((r > 0).mean())
            row[(col, "months")] = int(len(r))
        if row:
            frames[regime_name] = pd.Series(row)

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames).T
    result.index.name = "regime"
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["asset", "metric"])
    return result


def calendar_year_returns(
    panel: pd.DataFrame,
    asset_col: str = "sp500_ret",
) -> pd.Series:
    """Annual calendar-year returns for a single asset column.

    Args:
        panel:     Monthly panel with ``*_ret`` columns, month-end indexed.
        asset_col: Which return column to use.

    Returns:
        Series indexed by year (int), values are decimal total return.
    """
    if panel is None or panel.empty or asset_col not in panel.columns:
        return pd.Series(dtype="float64")

    rets = panel[asset_col].dropna()
    if rets.empty:
        return pd.Series(dtype="float64")

    yearly = rets.groupby(rets.index.year).apply(lambda g: (1.0 + g).prod() - 1.0)
    yearly.index = yearly.index.astype(int)
    yearly.index.name = "year"
    return yearly


__all__ = [
    "KEY_EVENTS",
    "EventWindow",
    "regime_heatmap",
    "event_window_returns",
    "regime_asset_table",
    "calendar_year_returns",
]
