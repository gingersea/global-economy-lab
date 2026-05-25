"""
Monthly research panel builder.

Implements step 4 of ``docs/phase1_execution_plan.md``: take the raw
fetcher outputs (日度资产 + 月度宏观) and align them onto a single
**month-end** wide DataFrame that downstream regime labels and the
backtest can consume.

Conventions
-----------
* Index: ``DatetimeIndex`` at month-end (``freq="ME"``-aligned).
* Asset columns: monthly *simple* return (e.g. ``sp500_ret``).
* Macro level columns: month-end last value (e.g. ``us_pmi``).
* Macro transform columns: year-over-year % change (``_yoy``),
  month-over-month % change (``_mom``) or difference (``_diff``).
* All values are stored as ``float64``; missing observations stay
  ``NaN`` (no forward-fill by default, leaving the choice to callers).

Everything is pure pandas; no I/O.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


VALID_TRANSFORMS = {"level", "yoy", "mom", "diff", "yoy_diff"}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted with a ``DatetimeIndex`` (cheap no-op when
    already correct)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "date" in df.columns:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.set_index("date").sort_index()
    return df.set_index(pd.to_datetime(df.index)).sort_index()


def daily_to_monthly_return(
    df: pd.DataFrame,
    price_col: str = "close",
    method: str = "simple",
) -> pd.Series:
    """Convert a daily price series to a **month-end simple return** series.

    Args:
        df:        Daily DataFrame with at least one price column.
        price_col: Column holding the price.  Falls back to ``"Close"``
                   or the first numeric column if missing.
        method:    ``"simple"`` (default) or ``"log"``.

    Returns:
        ``pd.Series`` indexed by month-end (``MonthEnd``) holding the
        monthly return.  Empty Series if input is empty.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    df = _ensure_datetime_index(df)
    if price_col not in df.columns:
        for alt in ("Close", "close", "Adj Close", "adj_close"):
            if alt in df.columns:
                price_col = alt
                break
        else:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                return pd.Series(dtype="float64")
            price_col = numeric_cols[0]

    px = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if px.empty:
        return pd.Series(dtype="float64")

    monthly_px = px.resample("ME").last()
    if method == "log":
        ret = np.log(monthly_px / monthly_px.shift(1))
    else:
        ret = monthly_px.pct_change(fill_method=None)
    return ret.rename("ret")


def macro_to_monthly(
    df: pd.DataFrame,
    value_col: str = "value",
    transform: str = "level",
) -> pd.Series:
    """Convert a macro series to a month-end series with optional transform.

    Args:
        df:        DataFrame whose values are observed at any frequency
                   ≤ monthly (daily, weekly, monthly, quarterly).
        value_col: Source value column.
        transform: One of ``VALID_TRANSFORMS``:

                   * ``level``    – last observation in the month.
                   * ``yoy``      – ``pct_change(12)`` × 100, in %.
                   * ``mom``      – ``pct_change(1)`` × 100, in %.
                   * ``diff``     – arithmetic difference vs prior month.
                   * ``yoy_diff`` – 12-month arithmetic difference.

    Returns:
        ``pd.Series`` indexed by month-end.
    """
    if transform not in VALID_TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{transform}'. Expected one of {VALID_TRANSFORMS}."
        )
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    df = _ensure_datetime_index(df)
    if value_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return pd.Series(dtype="float64")
        value_col = numeric_cols[0]

    monthly = pd.to_numeric(df[value_col], errors="coerce").resample("ME").last()

    if transform == "level":
        out = monthly
    elif transform == "yoy":
        out = monthly.pct_change(12, fill_method=None) * 100.0
    elif transform == "mom":
        out = monthly.pct_change(1, fill_method=None) * 100.0
    elif transform == "diff":
        out = monthly.diff(1)
    else:  # yoy_diff
        out = monthly.diff(12)
    return out


def build_monthly_panel(
    assets: Optional[Mapping[str, pd.DataFrame]] = None,
    macros: Optional[Mapping[str, Tuple[pd.DataFrame, str]]] = None,
    asset_price_col: str = "close",
    macro_value_col: str = "value",
) -> pd.DataFrame:
    """Assemble a wide monthly research panel.

    Args:
        assets:          Mapping ``{asset_key: daily_df}``.  Each output
                         becomes a column ``<asset_key>_ret`` holding
                         the monthly simple return.
        macros:          Mapping ``{macro_key: (df, transform)}``.  The
                         transform must be one of :data:`VALID_TRANSFORMS`.
                         The output column is named
                         ``<macro_key>`` for ``level`` and
                         ``<macro_key>_<transform>`` otherwise.
        asset_price_col: Price column inside each asset DataFrame.
        macro_value_col: Value column inside each macro DataFrame.

    Returns:
        Wide DataFrame indexed by month-end.  Returns an empty
        DataFrame if both inputs are empty.

    Notes:
        Callers are responsible for any further filling / lagging.  See
        :func:`apply_signal_lag` for the conventional 1-month shift used
        in backtests to avoid look-ahead bias.
    """
    assets = dict(assets or {})
    macros = dict(macros or {})

    series_map: Dict[str, pd.Series] = {}

    for key, df in assets.items():
        ret = daily_to_monthly_return(df, price_col=asset_price_col)
        if not ret.empty:
            series_map[f"{key}_ret"] = ret

    for key, spec in macros.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            df, transform = spec
        else:  # allow passing just a DataFrame → default to level
            df, transform = spec, "level"
        s = macro_to_monthly(df, value_col=macro_value_col, transform=transform)
        if s.empty:
            continue
        colname = key if transform == "level" else f"{key}_{transform}"
        series_map[colname] = s

    if not series_map:
        logger.warning("build_monthly_panel: no usable input series.")
        return pd.DataFrame()

    panel = pd.concat(series_map, axis=1, sort=True).sort_index()
    panel.index.name = "month_end"
    return panel


def apply_signal_lag(panel: pd.DataFrame, lag: int = 1, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Shift macro / signal columns forward by *lag* months.

    This implements the "上月末已知信号生成当月持仓" rule from
    ``docs/phase1_execution_plan.md`` §3.4.

    Args:
        panel:   Wide monthly panel.
        lag:     Number of months to shift (positive ⇒ delay).
        columns: Subset of columns to shift.  Defaults to every column
                 that is **not** a return column (``*_ret``) – because
                 returns are realised contemporaneously and must not be
                 shifted.

    Returns:
        New DataFrame with the requested columns shifted; non-shifted
        columns are copied as-is.
    """
    if panel is None or panel.empty:
        return panel
    if columns is None:
        columns = [c for c in panel.columns if not c.endswith("_ret")]
    out = panel.copy()
    for c in columns:
        if c in out.columns:
            out[c] = out[c].shift(lag)
    return out


__all__ = [
    "VALID_TRANSFORMS",
    "daily_to_monthly_return",
    "macro_to_monthly",
    "build_monthly_panel",
    "apply_signal_lag",
]
