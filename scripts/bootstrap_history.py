"""
Bootstrap ≥15 years of historical data for every registered source.

This is a one-shot helper to populate the local data cache with enough
history to support the modelling roadmap (regime detection, factor
models, event studies).  Default start date is 2008-01-01 to cover the
Global Financial Crisis through the post-COVID inflation cycle.

Usage:
    python scripts/bootstrap_history.py [--start-date YYYY-MM-DD]
                                        [--category CATEGORY]
                                        [--region REGION]
                                        [--limit N]

Outputs are cached as parquet under ``data/processed/`` and a fetch
manifest is appended to ``data/_meta/manifest.jsonl``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import date
from pathlib import Path

# Make the project root importable regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger  # noqa: E402

from config.data_sources import ALL_SOURCES, DataSourceConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default="2008-01-01",
        help="Earliest date to back-fill (default: 2008-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=str(date.today()),
        help="Latest date to fetch (default: today)",
    )
    parser.add_argument(
        "--category",
        default="",
        help="Only fetch sources whose category matches (macro / equity / "
        "bond / commodity / fx / sentiment).",
    )
    parser.add_argument(
        "--region",
        default="",
        help="Only fetch sources whose region matches (US / CN / EU / JP / "
        "HK / Global).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of sources to process (0 = no limit).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore local cache and re-fetch every source.",
    )
    return parser.parse_args()


def _filter_sources(
    category: str, region: str
) -> dict[str, DataSourceConfig]:
    items = ALL_SOURCES
    if category:
        items = {k: v for k, v in items.items() if v.category == category}
    if region:
        items = {k: v for k, v in items.items() if v.region == region}
    return items


def _build_fetcher(cfg: DataSourceConfig):
    """Instantiate the fetcher class declared by a :class:`DataSourceConfig`."""
    mod = importlib.import_module(cfg.fetcher_module)
    cls = getattr(mod, cfg.fetcher_class)
    return cls(**cfg.fetch_kwargs)


def main() -> int:
    """Entry point – bootstrap ≥15y of history."""
    args = parse_args()
    sources = _filter_sources(args.category, args.region)
    if args.limit:
        sources = dict(list(sources.items())[: args.limit])

    logger.info("=" * 60)
    logger.info("Bootstrap historical data")
    logger.info(f"  Period   : {args.start_date} → {args.end_date}")
    logger.info(f"  Sources  : {len(sources)} (of {len(ALL_SOURCES)} total)")
    logger.info(f"  Category : {args.category or 'ALL'}")
    logger.info(f"  Region   : {args.region or 'ALL'}")
    logger.info("=" * 60)

    ok = 0
    failed = 0
    for key, cfg in sources.items():
        logger.info(f"── {key} [{cfg.category}/{cfg.region}] ──")
        try:
            fetcher = _build_fetcher(cfg)
            df = fetcher.fetch(
                start_date=args.start_date,
                end_date=args.end_date,
                force_refresh=args.force_refresh,
            )
            n = 0 if df is None else len(df)
            if n == 0:
                failed += 1
                logger.warning(f"  ✗ {key}: 0 rows (backends exhausted)")
            else:
                ok += 1
                logger.success(f"  ✓ {key}: {n} rows")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error(f"  ✗ {key}: {exc}")

    logger.info("─" * 60)
    logger.info(f"Done. OK={ok}  FAILED={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
