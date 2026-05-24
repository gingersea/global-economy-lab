"""
Market sentiment data fetcher.

Provides VIX (fear index) and placeholders for social-media /
options-flow sentiment indicators.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.base import BaseFetcher


class SentimentFetcher(BaseFetcher):
    """Fetcher for market sentiment data.

    Args:
        ticker: The yfinance ticker for the sentiment instrument.
                - ``"^VIX"``   – CBOE Volatility Index
                - ``"^SKEW"``  – CBOE Skew Index
                - ``"^VVIX"``  – Volatility of VIX
    """

    def __init__(self, ticker: str) -> None:
        name = ticker.replace("^", "").replace("=", "_")
        super().__init__(name=f"sentiment_{name}")
        self.ticker = ticker

    # ── Abstract implementation ───────────────────────────────────────────────

    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch sentiment data from yfinance."""
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
                f"[{self.name}] yfinance sentiment fetch failed for "
                f"{self.ticker!r}: {exc}"
            )
            raise


# ── Convenience functions ─────────────────────────────────────────────────────


def get_vix(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch CBOE Volatility Index (VIX, ticker ``^VIX``).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    fetcher = SentimentFetcher(ticker="^VIX")
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )


def get_put_call_ratio(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,  # noqa: ARG001
) -> pd.DataFrame:
    """Placeholder: Fetch CBOE equity put/call ratio.

    .. note::
        The CBOE put/call ratio is not available directly through
        yfinance or FRED.  This function returns an empty DataFrame
        as a placeholder.  To implement it, integrate CBOE's public
        data files (https://www.cboe.com/us/options/market_statistics/).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Retained for API consistency; unused.

    Returns:
        Empty DataFrame (pending implementation).
    """
    logger.info(
        "get_put_call_ratio: placeholder – "
        "integrate CBOE data files to enable this function."
    )
    return pd.DataFrame()


def get_social_media_sentiment(
    keyword: str = "S&P 500",
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,  # noqa: ARG001
) -> pd.DataFrame:
    """Placeholder: Fetch social-media sentiment for a keyword.

    .. note::
        Connect to the Reddit API (PRAW), Twitter/X API, or a
        sentiment data provider (e.g., StockTwits) to implement this.

    Args:
        keyword:    Search term or asset name.
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Retained for API consistency; unused.

    Returns:
        Empty DataFrame (pending implementation).
    """
    logger.info(
        f"get_social_media_sentiment({keyword!r}): placeholder – "
        "connect to a social media API to enable this function."
    )
    return pd.DataFrame()
