"""
Regime-sensitivity analysis tool (§8 improvement #3).

Grids over (growth_threshold, cpi_high) pairs, re-labels regimes, and
computes per-regime mean returns to assess label robustness.

Usage:
    python scripts/regime_sensitivity.py --start-date 2008-01-01
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.backtest import (
    ASSET_KEYS,
    DEFAULT_REGIME_WEIGHTS,
    GROWTH_FALLBACK_COL,
    GROWTH_FALLBACK_THRESHOLD,
    MACRO_SPECS,
)
from config.data_sources import ALL_SOURCES, DataSourceConfig
from src.analysis import regime_labels as rl
from src.analysis import research_panel as rp
from src.backtest import MonthlyBacktest, equal_weight_target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--output-dir", default="data/_meta/sensitivity")
    return parser.parse_args()


def _build_fetcher(cfg: DataSourceConfig):
    mod = importlib.import_module(cfg.fetcher_module)
    cls = getattr(mod, cfg.fetcher_class)
    return cls(**cfg.fetch_kwargs)


def _safe_fetch(key: str, start: str, end: str) -> Optional[pd.DataFrame]:
    cfg = ALL_SOURCES.get(key)
    if cfg is None:
        return None
    try:
        fetcher = _build_fetcher(cfg)
        df = fetcher.fetch(start_date=start, end_date=end)
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def load_panel(start: str, end: str):
    assets = {k: df for k in ASSET_KEYS if (df := _safe_fetch(k, start, end)) is not None}
    macros = {k: (df, MACRO_SPECS[k]) for k in MACRO_SPECS
              if (df := _safe_fetch(k, start, end)) is not None}
    panel = rp.build_monthly_panel(assets=assets, macros=macros)
    return rp.apply_signal_lag(panel, lag=1)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.start_date, args.end_date)
    if panel.empty:
        logger.error("Empty panel — aborting.")
        return 1

    growth_col = "us_pmi" if "us_pmi" in panel.columns else GROWTH_FALLBACK_COL
    if growth_col not in panel.columns:
        logger.error(f"No growth column ({growth_col}) in panel.")
        return 1
    if "us_cpi_yoy" not in panel.columns:
        logger.error("No CPI YoY column in panel.")
        return 1

    growth_thresholds = np.linspace(
        -2.0 if growth_col == GROWTH_FALLBACK_COL else 48.0,
        2.0 if growth_col == GROWTH_FALLBACK_COL else 52.0,
        5,
    )
    cpi_thresholds = np.linspace(1.5, 3.5, 5)

    ret_cols = [c for c in panel.columns if c.endswith("_ret")]
    if not ret_cols:
        logger.error("No return columns in panel.")
        return 1

    records = []
    distribution_rows = []
    for gt in growth_thresholds:
        for ct in cpi_thresholds:
            cfg = rl.RegimeConfig(
                growth_col=growth_col,
                growth_threshold=float(gt),
                cpi_high=float(ct),
            )
            try:
                labelled = rl.label_regimes(panel, config=cfg)
            except Exception:
                continue

            counts = labelled["regime"].value_counts().to_dict()
            total = sum(counts.values()) or 1
            for regime_name in ["recovery", "overheat", "stagflation", "recession"]:
                distribution_rows.append({
                    "growth_threshold": round(float(gt), 2),
                    "cpi_high": round(float(ct), 2),
                    "regime": regime_name,
                    "pct": round(counts.get(regime_name, 0) / total * 100, 1),
                })

            labelled_panel = panel.copy()
            labelled_panel["regime"] = labelled["regime"]

            regime_means: Dict[str, Dict[str, float]] = {}
            for rn in labelled["regime"].unique():
                mask = labelled["regime"] == rn
                sub = labelled_panel.loc[mask, ret_cols]
                if sub.empty:
                    continue
                regime_means[rn] = sub.mean().to_dict()

            # Try regime rotation with these thresholds
            try:
                weights = rl.regime_target_weights(labelled, assets=ret_cols)
                returns = panel[ret_cols].dropna(how="all")
                bt = MonthlyBacktest(returns, cost_bps=10.0)
                res = bt.run(weights, name="regime")
                metrics = res.metrics
            except Exception:
                metrics = {"ann_return": np.nan, "sharpe": np.nan}

            records.append({
                "growth_threshold": round(float(gt), 2),
                "cpi_high": round(float(ct), 2),
                "ann_return": metrics.get("ann_return", np.nan),
                "sharpe": metrics.get("sharpe", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                **{f"{rn}_mean_sp500": regime_means.get(rn, {}).get("sp500_ret", np.nan)
                   for rn in ["recovery", "overheat", "stagflation", "recession"]},
            })

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(out_dir / "sensitivity_metrics.csv", index=False)

    dist_df = pd.DataFrame(distribution_rows)
    dist_df.to_csv(out_dir / "sensitivity_distribution.csv", index=False)

    # Chart: heatmap of annualised return
    try:
        import matplotlib.pyplot as plt

        pivot = metrics_df.pivot_table(
            index="growth_threshold", columns="cpi_high", values="ann_return"
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
        ax.set_xlabel("CPI YoY threshold (%)")
        ax.set_ylabel("Growth threshold")
        ax.set_title("Regime Rotation CAGR by Threshold Pair")
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="black" if 0.04 < val < 0.10 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8, label="CAGR")
        fig.tight_layout()
        fig.savefig(out_dir / "sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        logger.warning("Failed to save sensitivity heatmap.")

    logger.success(f"Sensitivity analysis written to {out_dir}")
    logger.info(
        f"\nBest CAGR: {metrics_df['ann_return'].max():.4f} "
        f"(growth={metrics_df.loc[metrics_df['ann_return'].idxmax(), 'growth_threshold']}, "
        f"cpi={metrics_df.loc[metrics_df['ann_return'].idxmax(), 'cpi_high']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
