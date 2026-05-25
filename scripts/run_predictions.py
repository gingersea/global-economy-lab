"""
Two-dimension prediction driver.

Runs the macro → daily prediction pipeline and outputs reports.

Usage:
    python scripts/run_predictions.py --start-date 2008-01-01
    python scripts/run_predictions.py --start-date 2008-01-01 --rolling-eval
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.backtest import ASSET_KEYS, MACRO_SPECS, GROWTH_FALLBACK_COL, GROWTH_FALLBACK_THRESHOLD
from config.data_sources import ALL_SOURCES, DataSourceConfig
from src.analysis import regime_labels as rl
from src.analysis import research_panel as rp
from src.analysis.prediction_hub import PredictionHub


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--output-dir", default="data/_meta/predictions")
    parser.add_argument("--rolling-eval", action="store_true", help="Run rolling out-of-sample macro evaluation.")
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


def load_data(start: str, end: str):
    logger.info("Loading asset series …")
    assets = {k: df for k in ASSET_KEYS if (df := _safe_fetch(k, start, end)) is not None}
    logger.info(f"  → {len(assets)}/{len(ASSET_KEYS)} assets available.")

    logger.info("Loading macro series …")
    macros = {k: (df, MACRO_SPECS[k]) for k in MACRO_SPECS
              if (df := _safe_fetch(k, start, end)) is not None}
    logger.info(f"  → {len(macros)}/{len(MACRO_SPECS)} macros available.")

    panel = rp.build_monthly_panel(assets=assets, macros=macros)
    lagged = rp.apply_signal_lag(panel, lag=1)

    growth_col = "us_pmi" if "us_pmi" in lagged.columns else GROWTH_FALLBACK_COL
    growth_threshold = 50.0 if growth_col == "us_pmi" else GROWTH_FALLBACK_THRESHOLD
    if growth_col in lagged.columns and "us_cpi_yoy" in lagged.columns:
        cfg = rl.RegimeConfig(growth_col=growth_col, growth_threshold=growth_threshold)
        labelled = rl.label_regimes(lagged, config=cfg)
        panel["regime"] = labelled["regime"]
        panel["regime_label"] = labelled["regime_label"]
    else:
        logger.warning("Cannot label regimes — missing growth/CPI columns.")

    return panel, assets


def main():
    args = parse_args()
    logger.info("=" * 64)
    logger.info("Two-Dimension Prediction Driver")
    logger.info(f"  Period: {args.start_date} → {args.end_date}")
    logger.info("=" * 64)

    panel, asset_dfs = load_data(args.start_date, args.end_date)
    if panel.empty or "regime" not in panel.columns:
        logger.error("Panel has no regime labels. Cannot predict.")
        return 1

    hub = PredictionHub(output_dir=args.output_dir)

    if args.rolling_eval:
        logger.info("Running rolling macro evaluation …")
        rolling = hub.rolling_backtest(panel, asset_dfs)
        if not rolling.empty:
            rolling.to_csv(Path(args.output_dir) / "rolling_eval.csv")
            acc = rolling["correct"].mean()
            logger.info(f"Rolling accuracy: {acc:.2%} ({rolling['correct'].sum()}/{len(rolling)})")

    logger.info("Running prediction pipeline …")
    result = hub.run(panel, asset_dfs)

    logger.info(f"\n{'='*64}")
    logger.info("Prediction Summary")
    logger.info(f"  Reference date : {result.summary['reference_date']}")
    logger.info(f"  Current regime : {result.summary['current_regime']}")
    logger.info(f"  Predicted 1mo  : {result.summary['predicted_regime_1m']} "
                f"(conf={result.summary.get('confidence_1m', 0):.2%})")
    logger.info(f"  Predicted 2mo  : {result.summary.get('predicted_regime_2m')} "
                f"(conf={result.summary.get('confidence_2m', 0):.2%})")
    logger.info(f"  Predicted 3mo  : {result.summary.get('predicted_regime_3m')} "
                f"(conf={result.summary.get('confidence_3m', 0):.2%})")
    logger.info("  Daily signals:")
    for asset, sig in result.daily_prediction.signals.items():
        direction = "▲" if sig > 0.05 else ("▼" if sig < -0.05 else "─")
        logger.info(f"    {direction} {asset}: {sig:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
