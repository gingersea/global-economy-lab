"""
Currencies / FX data fetcher.

Retrieves US Dollar Index and major currency pairs via yfinance.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.base import BaseFetcher


_DEFAULT_PAIRS: List[str] = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "USDCAD=X",
    "AUDUSD=X",
    "USDCNY=X",
    "USDCHF=X",
]


class CurrenciesFetcher(BaseFetcher):
    """Fetcher for FX / currency pair data via yfinance.

    Args:
        ticker: The yfinance FX ticker.
                - ``"DX-Y.NYB"`` – US Dollar Index
                - ``"EURUSD=X"`` – EUR/USD
                - ``"USDJPY=X"`` – USD/JPY
                - ``"USDCNY=X"`` – USD/CNY
    """

    def __init__(self, ticker: str) -> None:
        name = ticker.replace("=", "_").replace(".", "_").replace("-", "_")
        super().__init__(name=f"fx_{name}")
        self.ticker = ticker

    # ── Abstract implementation ───────────────────────────────────────────────

    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch FX data from yfinance."""
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
                f"[{self.name}] yfinance FX fetch failed for "
                f"{self.ticker!r}: {exc}"
            )
            raise


# ── Convenience functions ─────────────────────────────────────────────────────


def get_dxy(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch US Dollar Index (DX-Y.NYB).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    fetcher = CurrenciesFetcher(ticker="DX-Y.NYB")
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )


def get_major_pairs(
    pairs: Optional[List[str]] = None,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch closing prices for a list of FX pairs.

    Args:
        pairs:      List of yfinance FX tickers. Defaults to the
                    standard major pairs (EUR, JPY, GBP, CAD, AUD, CNY, CHF).
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        Dict mapping each ticker to its OHLCV DataFrame.
    """
    pairs = pairs or _DEFAULT_PAIRS
    result: dict[str, pd.DataFrame] = {}
    for ticker in pairs:
        fetcher = CurrenciesFetcher(ticker=ticker)
        df = fetcher.fetch(
            start_date=start_date, end_date=end_date, use_cache=use_cache
        )
        result[ticker] = df
    return result
