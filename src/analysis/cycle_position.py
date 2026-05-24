"""
Economic cycle positioning analysis.

Classifies the current (or historical) economic regime into one of four
classic quadrants based on growth and inflation dynamics:

    ┌─────────────────┬─────────────────┐
    │  Recovery       │  Overheat       │
    │ (↑ PMI, ↓ CPI)  │ (↑ PMI, ↑ CPI) │
    ├─────────────────┼─────────────────┤
    │  Deflation/     │  Stagflation    │
    │  Recession      │ (↓ PMI, ↑ CPI) │
    │ (↓ PMI, ↓ CPI)  │                 │
    └─────────────────┴─────────────────┘

The ``growth`` axis uses PMI as a proxy.  The ``inflation`` axis uses CPI
YoY change.  A third axis – ``liquidity`` – is approximated by the
central bank policy rate direction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# Canonical phase labels (also used as plot annotations)
PHASE_LABELS: dict[str, str] = {
    "recovery": "Recovery 复苏",
    "overheat": "Overheat 过热",
    "stagflation": "Stagflation 滞胀",
    "recession": "Recession 衰退",
    "unknown": "Unknown 未知",
}

_PMI_EXPANSION_THRESHOLD = 50.0   # PMI > 50 ⇒ expansion / growth


def _classify_phase(pmi: float, cpi_yoy: float) -> str:
    """Classify a single observation into an economic phase.

    Args:
        pmi:     Manufacturing PMI value.
        cpi_yoy: CPI year-over-year percentage change.

    Returns:
        One of ``"recovery"``, ``"overheat"``, ``"stagflation"``,
        ``"recession"``.
    """
    growing = pmi >= _PMI_EXPANSION_THRESHOLD
    inflating = cpi_yoy >= 2.0  # Use 2 % as the neutral CPI boundary
    if growing and not inflating:
        return "recovery"
    if growing and inflating:
        return "overheat"
    if not growing and inflating:
        return "stagflation"
    return "recession"


def get_cycle_position(
    pmi: pd.DataFrame,
    cpi: pd.DataFrame,
    rate: Optional[pd.DataFrame] = None,
    pmi_col: str = "value",
    cpi_col: str = "value",
    rate_col: str = "value",
    window: int = 12,
) -> pd.DataFrame:
    """Compute the economic cycle phase for each period.

    Args:
        pmi:      DataFrame with PMI values (monthly).  Index must be
                  a :class:`pd.DatetimeIndex`.
        cpi:      DataFrame with CPI level values (monthly).  Used to
                  compute the YoY percentage change internally.
        rate:     Optional DataFrame with policy rate values.  Used to
                  annotate the liquidity direction.
        pmi_col:  Column name for the PMI series.
        cpi_col:  Column name for the CPI level series.
        rate_col: Column name for the policy rate series.
        window:   Rolling window (months) for smoothing PMI before
                  phase classification.

    Returns:
        A DataFrame indexed by date with columns:

        * ``pmi``          – raw PMI value
        * ``pmi_smooth``   – rolling-smoothed PMI
        * ``cpi``          – raw CPI level
        * ``cpi_yoy``      – CPI year-over-year change (%)
        * ``phase``        – string label (``"recovery"`` / ``"overheat"`` /
                              ``"stagflation"`` / ``"recession"``)
        * ``phase_label``  – human-readable label (Chinese + English)
        * ``rate``         – policy rate (if provided)
        * ``rate_direction`` – ``"easing"`` | ``"tightening"`` | ``"neutral"``
    """
    if pmi.empty or cpi.empty:
        logger.warning("get_cycle_position: PMI or CPI data is empty.")
        return pd.DataFrame()

    # ── Align series on a common monthly frequency ────────────────────────────
    pmi_series = (
        pmi[pmi_col]
        .resample("ME")
        .last()
        .rename("pmi")
    )
    cpi_series = (
        cpi[cpi_col]
        .resample("ME")
        .last()
        .rename("cpi")
    )

    result = pd.concat([pmi_series, cpi_series], axis=1).dropna()

    # ── Smooth PMI ────────────────────────────────────────────────────────────
    result["pmi_smooth"] = result["pmi"].rolling(window=window, min_periods=1).mean()

    # ── CPI year-over-year % change ───────────────────────────────────────────
    result["cpi_yoy"] = result["cpi"].pct_change(12) * 100

    # ── Phase classification ──────────────────────────────────────────────────
    result["phase"] = result.apply(
        lambda row: _classify_phase(row["pmi_smooth"], row["cpi_yoy"])
        if not np.isnan(row["cpi_yoy"])
        else "unknown",
        axis=1,
    )
    result["phase_label"] = result["phase"].map(PHASE_LABELS)

    # ── Optional liquidity axis ───────────────────────────────────────────────
    if rate is not None and not rate.empty:
        rate_series = (
            rate[rate_col]
            .resample("ME")
            .last()
            .rename("rate")
        )
        result = result.join(rate_series, how="left")
        result["rate_direction"] = (
            result["rate"]
            .diff()
            .apply(
                lambda x: "tightening"
                if x > 0.05
                else ("easing" if x < -0.05 else "neutral")
            )
        )
    else:
        result["rate"] = np.nan
        result["rate_direction"] = "unknown"

    return result


def get_current_phase(
    pmi: pd.DataFrame,
    cpi: pd.DataFrame,
    rate: Optional[pd.DataFrame] = None,
    **kwargs,
) -> dict:
    """Return the most recent economic cycle phase.

    Args:
        pmi:    PMI DataFrame (see :func:`get_cycle_position`).
        cpi:    CPI DataFrame.
        rate:   Optional policy rate DataFrame.
        **kwargs: Forwarded to :func:`get_cycle_position`.

    Returns:
        Dict with keys ``"date"``, ``"phase"``, ``"phase_label"``,
        ``"pmi"``, ``"cpi_yoy"``, ``"rate"``, ``"rate_direction"``.
    """
    df = get_cycle_position(pmi, cpi, rate, **kwargs)
    if df.empty:
        return {"phase": "unknown", "phase_label": PHASE_LABELS["unknown"]}
    last = df.iloc[-1]
    return {
        "date": str(last.name.date()),
        "phase": last["phase"],
        "phase_label": last["phase_label"],
        "pmi": round(float(last["pmi"]), 2),
        "cpi_yoy": round(float(last["cpi_yoy"]), 2)
        if not np.isnan(last["cpi_yoy"])
        else None,
        "rate": round(float(last["rate"]), 4)
        if "rate" in last and not np.isnan(last["rate"])
        else None,
        "rate_direction": last.get("rate_direction", "unknown"),
    }
