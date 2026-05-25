"""
Data quality checks for the Global Economy Lab.

Implements the validation rules listed in
``docs/phase1_execution_plan.md`` §2.4:

* 日期覆盖（coverage start/end, span in days）
* 缺失率（NaN ratio per value column）
* 重复日期（duplicate index rows）
* 异常跳变（z-score of period-over-period change above a threshold）

Every check is **pure**: it takes a ``pd.DataFrame`` and returns a small
dict / DataFrame.  Nothing touches the network or disk, so the module
is fully unit-testable with synthetic data.

The :func:`build_quality_report` helper aggregates per-series reports
into a single DataFrame that can be written to ``data/_meta/`` for
future audits.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from loguru import logger


# Default z-score above which a period-over-period change is flagged
# as an "abnormal jump".  5σ is intentionally conservative – it filters
# real regime shifts (e.g. 2008 Q4 returns) but still highlights bad
# data such as a missing decimal point.
DEFAULT_JUMP_Z = 5.0


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with a sorted ``DatetimeIndex``.

    If the index is not already a ``DatetimeIndex`` the function tries
    to coerce it; failing that, it looks for a ``date`` column.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "date" in df.columns:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.set_index("date").sort_index()
    try:
        return df.set_index(pd.to_datetime(df.index)).sort_index()
    except Exception:  # noqa: BLE001
        return df


def check_series(
    df: pd.DataFrame,
    name: str,
    value_col: str = "value",
    jump_z: float = DEFAULT_JUMP_Z,
) -> Dict[str, object]:
    """Run a single-series quality check.

    Args:
        df:        Input DataFrame indexed by date (or with a ``date``
                   column).  An empty / missing column is tolerated and
                   reported.
        name:      Logical name for this series (used in the report).
        value_col: Column to scan for missing values & jumps.  If the
                   column does not exist the function falls back to the
                   first numeric column.
        jump_z:    Threshold for the |z-score| of pct-change above which
                   an observation is counted as an abnormal jump.

    Returns:
        Dict with keys: ``name``, ``rows``, ``coverage_start``,
        ``coverage_end``, ``span_days``, ``missing_pct``,
        ``duplicate_dates``, ``abnormal_jumps``, ``status``.
    """
    base: Dict[str, object] = {
        "name": name,
        "rows": 0,
        "coverage_start": None,
        "coverage_end": None,
        "span_days": 0,
        "missing_pct": np.nan,
        "duplicate_dates": 0,
        "abnormal_jumps": 0,
        "status": "empty",
    }
    if df is None or df.empty:
        logger.warning(f"check_series: '{name}' is empty.")
        return base

    df = _ensure_datetime_index(df)

    # Resolve the value column with a graceful fallback.
    if value_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            base["status"] = "no_numeric_column"
            return base
        value_col = numeric_cols[0]

    series = pd.to_numeric(df[value_col], errors="coerce")

    base["rows"] = int(len(df))
    base["coverage_start"] = str(df.index.min().date())
    base["coverage_end"] = str(df.index.max().date())
    base["span_days"] = int((df.index.max() - df.index.min()).days)
    base["missing_pct"] = round(float(series.isna().mean()) * 100.0, 4)
    base["duplicate_dates"] = int(df.index.duplicated().sum())

    # Abnormal jumps: |z-score of pct change| > jump_z.
    pct = series.pct_change(fill_method=None)
    if pct.notna().sum() >= 3:
        std = pct.std(ddof=0)
        if std and not np.isnan(std):
            z = (pct - pct.mean()) / std
            base["abnormal_jumps"] = int((z.abs() > jump_z).sum())

    base["status"] = "ok"
    return base


def check_panel(
    panels: Mapping[str, pd.DataFrame],
    value_cols: Optional[Mapping[str, str]] = None,
    jump_z: float = DEFAULT_JUMP_Z,
) -> pd.DataFrame:
    """Run :func:`check_series` over a mapping of DataFrames.

    Args:
        panels:     Mapping ``{source_name: DataFrame}``.
        value_cols: Optional mapping ``{source_name: value_column}``.
                    Sources missing from this map default to ``"value"``
                    (macro convention); falling back to ``"close"`` if
                    that column is absent in the DataFrame.
        jump_z:     Forwarded to :func:`check_series`.

    Returns:
        Long-form DataFrame, one row per source, sorted by source name.
    """
    value_cols = dict(value_cols or {})
    rows = []
    for src_name, df in panels.items():
        col = value_cols.get(src_name, "value")
        if df is not None and not df.empty and col not in df.columns and "close" in df.columns:
            col = "close"
        rows.append(check_series(df, src_name, value_col=col, jump_z=jump_z))
    report = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    return report


def summarize_report(report: pd.DataFrame) -> Dict[str, object]:
    """Compute aggregate summary stats from a quality report.

    Returns:
        Dict with totals and worst-offender keys, useful for logging or
        embedding in a manifest file.
    """
    if report.empty:
        return {"sources": 0, "ok": 0, "issues": 0}
    issues_mask = (
        (report["status"] != "ok")
        | (report["missing_pct"].fillna(100) > 20.0)
        | (report["duplicate_dates"].fillna(0) > 0)
        | (report["abnormal_jumps"].fillna(0) > 0)
    )
    worst_missing = (
        report.loc[report["missing_pct"].fillna(0).idxmax(), "name"]
        if not report["missing_pct"].isna().all()
        else None
    )
    return {
        "sources": int(len(report)),
        "ok": int((~issues_mask).sum()),
        "issues": int(issues_mask.sum()),
        "worst_missing_source": worst_missing,
        "max_missing_pct": float(report["missing_pct"].fillna(0).max()),
        "total_abnormal_jumps": int(report["abnormal_jumps"].fillna(0).sum()),
        "total_duplicate_dates": int(report["duplicate_dates"].fillna(0).sum()),
    }


def write_report(report: pd.DataFrame, out_path) -> None:
    """Persist a quality report to CSV (creates parent dirs as needed)."""
    from pathlib import Path

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(p, index=False)
    logger.info(f"data_quality: wrote {len(report)} rows to {p}")


__all__: Iterable[str] = [
    "DEFAULT_JUMP_Z",
    "check_series",
    "check_panel",
    "summarize_report",
    "write_report",
]
