"""
Phase-1 closed-loop driver.

Loads cached fetcher outputs → builds the monthly research panel → runs
the data-quality report → labels economic regimes → executes the three
benchmark backtests (equal-weight, S&P 500 buy-and-hold, regime
rotation) → writes summary CSV / JSON artefacts under
``data/_meta/phase1/``.

The script is **best-effort**: every step that depends on an external
data source is wrapped in ``try / except`` so that a partially-populated
cache still produces a meaningful report.  This keeps the phase-1
deliverable runnable on a freshly-cloned machine that has cached at
least the US equity + macro core (the minimum closed-loop universe).

Usage
-----

.. code-block:: bash

    python scripts/bootstrap_history.py --start-date 2008-01-01
    python scripts/run_phase1_backtest.py --start-date 2008-01-01 \
        --output-dir data/_meta/phase1
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from loguru import logger

# Make the project root importable regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.data_sources import ALL_SOURCES, DataSourceConfig  # noqa: E402
from src.analysis import data_quality as dq  # noqa: E402
from src.analysis import research_panel as rp  # noqa: E402
from src.analysis import regime_labels as rl  # noqa: E402
from src.backtest import (  # noqa: E402
    MonthlyBacktest,
    buy_and_hold_target,
    equal_weight_target,
)


# Asset / macro universe needed by the phase-1 closed loop.  Keys must
# match registry keys in ``config/data_sources.py``.
ASSET_KEYS = ["sp500", "gold", "crude_oil_wti", "dxy", "us_treasury_10y"]
MACRO_SPECS: Dict[str, str] = {
    "us_pmi": "level",
    "us_cpi": "yoy",
    "us_unemployment": "diff",
    "us_treasury_2y": "level",
    "us_treasury_10y": "level",  # also kept as a level macro for the curve
    "us_nfci": "level",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default=str(date.today()))
    parser.add_argument(
        "--output-dir",
        default="data/_meta/phase1",
        help="Where to write report CSV/JSON files.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Round-trip trading cost in basis points (default 0).",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="Annual risk-free rate used in Sharpe (default 0.0).",
    )
    return parser.parse_args()


def _build_fetcher(cfg: DataSourceConfig):
    mod = importlib.import_module(cfg.fetcher_module)
    cls = getattr(mod, cfg.fetcher_class)
    return cls(**cfg.fetch_kwargs)


def _safe_fetch(
    key: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """Pull a single source through its fetcher (which reads from cache
    when available).  Returns ``None`` on any error."""
    cfg = ALL_SOURCES.get(key)
    if cfg is None:
        logger.warning(f"_safe_fetch: '{key}' is not in ALL_SOURCES.")
        return None
    try:
        fetcher = _build_fetcher(cfg)
        df = fetcher.fetch(start_date=start, end_date=end)
        if df is None or df.empty:
            logger.warning(f"_safe_fetch: '{key}' returned no rows.")
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error(f"_safe_fetch: '{key}' failed: {exc}")
        return None


def load_inputs(
    start: str, end: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Return ``(asset_dfs, macro_dfs)`` mappings for the phase-1 set."""
    logger.info("Loading asset series …")
    assets = {k: df for k in ASSET_KEYS if (df := _safe_fetch(k, start, end)) is not None}
    logger.info(f"  → {len(assets)}/{len(ASSET_KEYS)} assets available.")

    logger.info("Loading macro series …")
    macros = {k: df for k in MACRO_SPECS if (df := _safe_fetch(k, start, end)) is not None}
    logger.info(f"  → {len(macros)}/{len(MACRO_SPECS)} macros available.")

    return assets, macros


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("Phase-1 closed-loop driver")
    logger.info(f"  Period   : {args.start_date} → {args.end_date}")
    logger.info(f"  Output   : {out_dir}")
    logger.info("=" * 64)

    assets, macros = load_inputs(args.start_date, args.end_date)

    # ── 1. Data quality report ─────────────────────────────────────
    panels = {**assets, **macros}
    quality = dq.check_panel(panels)
    dq.write_report(quality, out_dir / "data_quality.csv")
    summary = dq.summarize_report(quality)
    logger.info(f"Quality summary: {summary}")

    if not assets or "sp500" not in assets:
        logger.error("S&P 500 is required for the phase-1 closed loop. Aborting.")
        (out_dir / "phase1_summary.json").write_text(
            json.dumps({"status": "missing_sp500", "quality": summary}, indent=2)
        )
        return 1

    # ── 2. Monthly research panel ──────────────────────────────────
    macro_inputs = {k: (df, MACRO_SPECS[k]) for k, df in macros.items()}
    panel = rp.build_monthly_panel(assets=assets, macros=macro_inputs)
    panel.to_csv(out_dir / "monthly_panel.csv")
    logger.info(f"Monthly panel: {panel.shape[0]} rows × {panel.shape[1]} cols.")

    # ── 3. Regime labels (signals lagged 1 month) ──────────────────
    lagged = rp.apply_signal_lag(panel, lag=1)
    labelled = (
        rl.label_regimes(lagged)
        if {"us_pmi", "us_cpi_yoy"}.issubset(lagged.columns)
        else None
    )
    if labelled is not None:
        labelled[["regime", "regime_label"]].to_csv(out_dir / "regime_labels.csv")
        regime_counts = labelled["regime"].value_counts().to_dict()
    else:
        regime_counts = {}
        logger.warning("PMI / CPI YoY missing — skipping regime labels.")

    # ── 4. Backtests ───────────────────────────────────────────────
    return_cols = [c for c in panel.columns if c.endswith("_ret")]
    returns = panel[return_cols].dropna(how="all")
    if returns.empty:
        logger.error("No usable return columns. Aborting backtests.")
        return 1

    bt = MonthlyBacktest(
        returns,
        cost_bps=args.cost_bps,
        risk_free_annual=args.risk_free,
    )
    results = {}
    results["equal_weight"] = bt.run(
        equal_weight_target(returns), name="equal_weight"
    )
    if "sp500_ret" in returns.columns:
        results["sp500_bh"] = bt.run(
            buy_and_hold_target(returns, "sp500_ret"), name="sp500_bh"
        )
    if labelled is not None:
        weights = rl.regime_target_weights(labelled, assets=return_cols)
        results["regime_rotation"] = bt.run(weights, name="regime_rotation")

    # ── 5. Reports ─────────────────────────────────────────────────
    metrics_rows = []
    equity_curves = pd.DataFrame()
    for name, res in results.items():
        metrics_rows.append({"strategy": name, **res.metrics})
        equity_curves[name] = res.equity
    metrics_df = pd.DataFrame(metrics_rows).set_index("strategy")
    metrics_df.to_csv(out_dir / "backtest_metrics.csv")
    equity_curves.to_csv(out_dir / "equity_curves.csv")
    logger.info("\n" + metrics_df.to_string())

    overall = {
        "status": "ok",
        "period": {"start": args.start_date, "end": args.end_date},
        "quality": summary,
        "regime_counts": regime_counts,
        "metrics": metrics_df.reset_index().to_dict(orient="records"),
    }
    (out_dir / "phase1_summary.json").write_text(json.dumps(overall, indent=2, default=str))
    logger.success(f"Phase-1 artefacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
