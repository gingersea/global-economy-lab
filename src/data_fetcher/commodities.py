"""
Commodities data fetcher.

Retrieves spot/futures prices for gold, silver, crude oil, and other
commodities via yfinance.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.base import BaseFetcher


class CommoditiesFetcher(BaseFetcher):
    """Fetcher for commodity futures/spot price data via yfinance.

    Args:
        ticker: The yfinance commodity ticker symbol.
                - ``"GC=F"``  – Gold futures (COMEX)
                - ``"SI=F"``  – Silver futures
                - ``"CL=F"``  – WTI Crude Oil front-month
                - ``"BZ=F"``  – Brent Crude futures
                - ``"NG=F"``  – Natural Gas
                - ``"HG=F"``  – Copper
    """

    def __init__(self, ticker: str) -> None:
        name = ticker.replace("=", "_").replace("^", "")
        super().__init__(name=f"commodity_{name}")
        self.ticker = ticker

    # ── Abstract implementation ───────────────────────────────────────────────

    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch commodity price data from yfinance."""
        try:
            import yfinance as yf  # type: ignore[import]

            tkr = yf.Ticker(self.ticker)
            df: pd.DataFrame = tkr.history(
                start=start_date, end=end_date, auto_adjust=True
            )
            if df.empty:
                logger.warning(
                    f"[{self.name}] yfinance returned empty data "
                    f"for {self.ticker!r}"
                )
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df["ticker"] = self.ticker
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[{self.name}] yfinance fetch failed for "
                f"{self.ticker!r}: {exc}"
            )
            raise


# ── Convenience functions ─────────────────────────────────────────────────────


def get_gold_price(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch gold futures price (GC=F / XAUUSD proxy).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    fetcher = CommoditiesFetcher(ticker="GC=F")
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )


def get_silver_price(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch silver futures price (SI=F).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    fetcher = CommoditiesFetcher(ticker="SI=F")
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )


def get_crude_oil_price(
    benchmark: str = "WTI",
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch crude oil futures price.

    Args:
        benchmark:  ``"WTI"`` (CL=F, default) or ``"Brent"`` (BZ=F).
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    ticker_map = {"WTI": "CL=F", "Brent": "BZ=F"}
    ticker = ticker_map.get(benchmark, "CL=F")
    fetcher = CommoditiesFetcher(ticker=ticker)
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )
