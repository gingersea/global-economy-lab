"""
Rule-based four-state economic regime labels.

Extends the simpler :mod:`src.analysis.cycle_position` (which works on
raw PMI + CPI levels) to the **monthly research panel** consumed by the
phase-1 backtest.  The four states match the plan in
``docs/phase1_execution_plan.md`` §3.3:

* ``recovery``    – growth improving, inflation contained
* ``overheat``    – growth strong, inflation strong
* ``stagflation`` – growth weak, inflation strong
* ``recession``   – growth weak, inflation easing / financial stress

Inputs are column names on a panel built by
:func:`src.analysis.research_panel.build_monthly_panel`.  The default
columns mirror the source-registry keys but every column can be
overridden, keeping the function easy to unit-test with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger


REGIME_LABELS = {
    "recovery": "Recovery 复苏",
    "overheat": "Overheat 过热",
    "stagflation": "Stagflation 滞胀",
    "recession": "Recession 衰退",
    "unknown": "Unknown 未知",
}


@dataclass
class RegimeConfig:
    """Column-name and threshold configuration for :func:`label_regimes`.

    Tweaking thresholds is intentionally cheap to encourage robustness
    analysis (e.g. *what if PMI cutoff is 49 instead of 50?*).

    When PMI is unavailable (e.g. ``us_pmi`` column missing from panel),
    the config can be pointed at an alternative growth indicator such as
    industrial production YoY (``us_industrial_production_yoy``), with a
    different expansion threshold (e.g. ``0.0`` for positive YoY).
    """

    growth_col: str = "us_pmi"
    growth_threshold: float = 50.0
    growth_smooth_window: int = 3

    cpi_yoy_col: str = "us_cpi_yoy"
    cpi_high: float = 2.5  # percent YoY – above this counts as inflationary
    cpi_smooth_window: int = 3

    unemp_diff_col: str = "us_unemployment_diff"
    curve_col: str = "yield_curve_10y2y"  # optional, computed if absent
    nfci_col: str = "us_nfci"             # optional

    @property
    def pmi_col(self) -> str:  # noqa: D401
        """Legacy alias for :attr:`growth_col`."""
        return self.growth_col

    @property
    def pmi_expansion(self) -> float:  # noqa: D401
        """Legacy alias for :attr:`growth_threshold`."""
        return self.growth_threshold

    @property
    def pmi_smooth_window(self) -> int:  # noqa: D401
        """Legacy alias for :attr:`growth_smooth_window`."""
        return self.growth_smooth_window


def _smooth(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean with sane defaults (no extra deps)."""
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _classify_row(growing: bool, inflating: bool) -> str:
    if growing and not inflating:
        return "recovery"
    if growing and inflating:
        return "overheat"
    if (not growing) and inflating:
        return "stagflation"
    return "recession"


def label_regimes(
    panel: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """Add regime labels to a monthly research panel.

    Args:
        panel:  Monthly wide panel produced by
                :func:`src.analysis.research_panel.build_monthly_panel`.
                Must contain a PMI level column and a CPI YoY column
                (see :class:`RegimeConfig` for defaults).
        config: Optional :class:`RegimeConfig` override.

    Returns:
        DataFrame indexed identically to *panel* with new columns:

        * ``growth_signal``    – smoothed PMI value used for classification
        * ``inflation_signal`` – smoothed CPI YoY (%)
        * ``unemployment_rising`` – ``True`` when 3-month change in
          unemployment > 0 (additional recession confirmation)
        * ``yield_curve``      – 10Y-2Y spread (if both columns present)
        * ``regime``           – one of the four labels (or
          ``"unknown"`` when inputs are NaN)
        * ``regime_label``     – human-readable bilingual label

    Notes:
        Rows where either the growth or inflation signal is NaN are
        labelled ``"unknown"`` rather than dropped – this lets the
        downstream backtest decide whether to skip them or hold the
        previous allocation.
    """
    cfg = config or RegimeConfig()
    if panel is None or panel.empty:
        logger.warning("label_regimes: empty panel.")
        return panel.copy() if panel is not None else pd.DataFrame()

    out = panel.copy()

    if cfg.growth_col not in out.columns:
        raise KeyError(
            f"label_regimes: panel is missing growth column '{cfg.growth_col}'. "
            f"Available: {list(out.columns)}"
        )
    if cfg.cpi_yoy_col not in out.columns:
        raise KeyError(
            f"label_regimes: panel is missing CPI YoY column "
            f"'{cfg.cpi_yoy_col}'. Available: {list(out.columns)}"
        )

    out["growth_signal"] = _smooth(out[cfg.growth_col], cfg.growth_smooth_window)
    out["inflation_signal"] = _smooth(out[cfg.cpi_yoy_col], cfg.cpi_smooth_window)

    # Optional unemployment confirmation signal (3-month change).
    if cfg.unemp_diff_col in out.columns:
        out["unemployment_rising"] = (
            out[cfg.unemp_diff_col].rolling(3, min_periods=1).sum() > 0.0
        )
    else:
        out["unemployment_rising"] = pd.NA

    # Optional yield curve spread (10Y-2Y).  Computed if the two raw
    # columns are present and the curve column itself is absent.
    if cfg.curve_col not in out.columns and {
        "us_treasury_10y",
        "us_treasury_2y",
    }.issubset(out.columns):
        out[cfg.curve_col] = out["us_treasury_10y"] - out["us_treasury_2y"]

    growing = out["growth_signal"] >= cfg.growth_threshold
    inflating = out["inflation_signal"] >= cfg.cpi_high

    regimes = []
    for g, i, g_nan, i_nan in zip(
        growing.fillna(False),
        inflating.fillna(False),
        out["growth_signal"].isna(),
        out["inflation_signal"].isna(),
    ):
        if g_nan or i_nan:
            regimes.append("unknown")
        else:
            regimes.append(_classify_row(bool(g), bool(i)))
    out["regime"] = regimes
    out["regime_label"] = out["regime"].map(REGIME_LABELS)

    return out


# ─────────────────────────────────────────────────────────────────────
# Regime-aware default target weights for the phase-1 backtest.
# Imported from the config module (single source of truth).
# ─────────────────────────────────────────────────────────────────────

from config.backtest import DEFAULT_REGIME_WEIGHTS  # noqa: E402


def regime_target_weights(
    panel: pd.DataFrame,
    weights: Optional[dict] = None,
    regime_col: str = "regime",
    assets: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Map each row's regime label to a target-weight vector.

    Args:
        panel:      Output of :func:`label_regimes` (must contain
                    *regime_col*).
        weights:    Mapping ``{regime: {asset_col: weight}}``.  Defaults
                    to :data:`DEFAULT_REGIME_WEIGHTS`.
        regime_col: Column holding the regime label.
        assets:     Optional restriction to a subset of assets.  Missing
                    assets are zero-filled and the row is renormalised.

    Returns:
        DataFrame indexed identically to *panel* with one column per
        asset, every row summing to 1.0 (or 0.0 if no weights were
        available for that regime).
    """
    w_table = weights or DEFAULT_REGIME_WEIGHTS
    if panel is None or panel.empty or regime_col not in panel.columns:
        return pd.DataFrame(index=panel.index if panel is not None else None)

    # Establish the union of asset columns across regimes (or the
    # caller-provided subset).
    if assets is None:
        asset_set = sorted({a for w in w_table.values() for a in w.keys()})
    else:
        asset_set = list(assets)

    rows = []
    for reg in panel[regime_col]:
        spec = w_table.get(reg, {})
        row = {a: float(spec.get(a, 0.0)) for a in asset_set}
        total = sum(row.values())
        if total > 0:
            row = {a: v / total for a, v in row.items()}
        rows.append(row)

    weights_df = pd.DataFrame(rows, index=panel.index, columns=asset_set)
    return weights_df


__all__ = [
    "REGIME_LABELS",
    "RegimeConfig",
    "label_regimes",
    "DEFAULT_REGIME_WEIGHTS",
    "regime_target_weights",
]
