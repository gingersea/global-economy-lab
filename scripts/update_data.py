"""
Update all configured data sources and save to local cache.

Run this script periodically (e.g., daily via cron) to keep your
local data up-to-date for offline analysis.

Usage:
    python scripts/update_data.py [--start-date YYYY-MM-DD] [--sources src1,src2]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the project root importable regardless of where the script is called from.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update all data sources for the Global Economy Lab."
    )
    parser.add_argument(
        "--start-date",
        default=str(date.today() - timedelta(days=365 * 5)),
        help="Earliest date to fetch (default: 5 years ago, ISO-8601)",
    )
    parser.add_argument(
        "--end-date",
        default=str(date.today()),
        help="Latest date to fetch (default: today)",
    )
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated list of source keys to update (default: all)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore local cache and re-fetch all data",
    )
    return parser.parse_args()


def update_equities(
    start_date: str, end_date: str, force_refresh: bool
) -> None:
    """Update major equity index price data."""
    from config.settings import get_settings
    from src.data_fetcher.equities import EquitiesFetcher

    tickers = get_settings().default_equity_tickers
    for ticker in tickers:
        logger.info(f"Updating equity: {ticker}")
        try:
            f = EquitiesFetcher(ticker=ticker)
            df = f.fetch(
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            logger.success(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ✗ {ticker}: {exc}")


def update_commodities(
    start_date: str, end_date: str, force_refresh: bool
) -> None:
    """Update commodity price data."""
    from src.data_fetcher.commodities import CommoditiesFetcher

    for ticker in ["GC=F", "SI=F", "CL=F"]:
        logger.info(f"Updating commodity: {ticker}")
        try:
            f = CommoditiesFetcher(ticker=ticker)
            df = f.fetch(
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            logger.success(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ✗ {ticker}: {exc}")


def update_fx(start_date: str, end_date: str, force_refresh: bool) -> None:
    """Update FX / currency data."""
    from src.data_fetcher.currencies import CurrenciesFetcher

    for ticker in ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "USDCNY=X"]:
        logger.info(f"Updating FX: {ticker}")
        try:
            f = CurrenciesFetcher(ticker=ticker)
            df = f.fetch(
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            logger.success(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ✗ {ticker}: {exc}")


def update_bonds(start_date: str, end_date: str, force_refresh: bool) -> None:
    """Update US Treasury yield data."""
    from src.data_fetcher.bonds import BondsFetcher

    for series_id in ["DGS2", "DGS10"]:
        logger.info(f"Updating bond yield: {series_id}")
        try:
            f = BondsFetcher(series_id=series_id)
            df = f.fetch(
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            logger.success(f"  ✓ {series_id}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ✗ {series_id}: {exc}")


def update_macro(start_date: str, end_date: str, force_refresh: bool) -> None:
    """Update macro-economic indicators (requires FRED API key)."""
    from src.data_fetcher.macro_economic import MacroEconomicFetcher

    for series_id in ["CPIAUCSL", "NAPM", "FEDFUNDS", "GDPC1"]:
        logger.info(f"Updating macro: {series_id}")
        try:
            f = MacroEconomicFetcher(series_id=series_id)
            df = f.fetch(
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            logger.success(f"  ✓ {series_id}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ✗ {series_id}: {exc}")


def update_sentiment(start_date: str, end_date: str, force_refresh: bool) -> None:
    """Update sentiment indicators (VIX)."""
    from src.data_fetcher.sentiment import SentimentFetcher

    logger.info("Updating sentiment: ^VIX")
    try:
        f = SentimentFetcher(ticker="^VIX")
        df = f.fetch(
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh,
        )
        logger.success(f"  ✓ ^VIX: {len(df)} rows")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"  ✗ ^VIX: {exc}")


_UPDATER_MAP = {
    "equities": update_equities,
    "commodities": update_commodities,
    "fx": update_fx,
    "bonds": update_bonds,
    "macro": update_macro,
    "sentiment": update_sentiment,
}


def main() -> None:
    """Entry point: update all (or selected) data sources."""
    args = parse_args()

    from config.settings import get_settings

    settings = get_settings()
    logger.info("=" * 60)
    logger.info("Global Economy Lab – Data Update")
    logger.info(f"  Period : {args.start_date} → {args.end_date}")
    logger.info(f"  Cache  : {settings.processed_data_dir}")
    logger.info("=" * 60)

    requested = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else list(_UPDATER_MAP.keys())
    )

    unknown = [s for s in requested if s not in _UPDATER_MAP]
    if unknown:
        logger.error(
            f"Unknown source(s): {unknown}. "
            f"Choose from: {list(_UPDATER_MAP.keys())}"
        )
        sys.exit(1)

    for source_name in requested:
        logger.info(f"\n── {source_name.upper()} ──")
        _UPDATER_MAP[source_name](
            args.start_date, args.end_date, args.force_refresh
        )

    logger.info("\n✓ Update complete.")


if __name__ == "__main__":
    main()
