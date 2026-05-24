"""
Equities data fetcher.

Supports major global equity indices and individual stocks via
yfinance (US, HK, Global) and akshare (A-shares).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.base import BaseFetcher


class EquitiesFetcher(BaseFetcher):
    """Fetcher for equity price data.

    Args:
        ticker: The ticker symbol.  Examples:
                - ``"^GSPC"``      – S&P 500
                - ``"000300.SH"``  – CSI 300 (沪深300)
                - ``"^HSI"``       – Hang Seng
                - ``"AAPL"``       – Apple (US)
        market: Optional market hint (``"US"``, ``"CN"``, ``"HK"``).
                Defaults to auto-detection based on the ticker.
    """

    def __init__(self, ticker: str, market: Optional[str] = None) -> None:
        name = ticker.replace("^", "").replace(".", "_").replace("=", "_")
        super().__init__(name=f"equity_{name}")
        self.ticker = ticker
        self.market = market or self._detect_market(ticker)

    # ── Abstract implementation ───────────────────────────────────────────────

    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch equity OHLCV data from the appropriate backend."""
        if self.market == "CN_SHARE" and not self.ticker.endswith(".SH"):
            return self._fetch_akshare(start_date, end_date)
        return self._fetch_yfinance(start_date, end_date)

    # ── Backend implementations ───────────────────────────────────────────────

    def _fetch_yfinance(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance."""
        try:
            import yfinance as yf  # type: ignore[import]

            tkr = yf.Ticker(self.ticker)
            df: pd.DataFrame = tkr.history(
                start=start_date, end=end_date, auto_adjust=True
            )
            if df.empty:
                logger.warning(
                    f"[{self.name}] yfinance returned empty data "
                    f"for ticker {self.ticker!r}"
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

    def _fetch_akshare(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch A-share data using akshare."""
        try:
            import akshare as ak  # type: ignore[import]
        except ImportError:
            logger.error("akshare is not installed – cannot fetch A-share data.")
            return pd.DataFrame()
        try:
            # akshare dates use YYYYMMDD format
            start_fmt = start_date.replace("-", "")
            end_fmt = end_date.replace("-", "")
            symbol = self.ticker.replace(".SZ", "").replace(".SH", "")
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust="qfq",
            )
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "Open",
                    "收盘": "Close",
                    "最高": "High",
                    "最低": "Low",
                    "成交量": "Volume",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["ticker"] = self.ticker
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[{self.name}] akshare fetch failed for {self.ticker!r}: {exc}"
            )
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_market(ticker: str) -> str:
        """Heuristically determine the market from the ticker format."""
        ticker_upper = ticker.upper()
        if ticker_upper.endswith(".SH") or ticker_upper.endswith(".SZ"):
            return "CN_SHARE"
        if ticker_upper.endswith(".HK"):
            return "HK"
        return "US"


# ── Convenience functions ─────────────────────────────────────────────────────


def get_index_data(
    ticker: str,
    market: Optional[str] = None,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch price data for a market index or stock.

    Args:
        ticker:     Ticker symbol (e.g. ``"^GSPC"``, ``"000300.SH"``).
        market:     Optional market hint (``"US"``, ``"CN"``, ``"HK"``).
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date. Defaults to today.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with OHLCV columns indexed by date.
    """
    fetcher = EquitiesFetcher(ticker=ticker, market=market)
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )


def get_sp500(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch S&P 500 Index (``^GSPC``) historical data.

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.
    """
    return get_index_data(
        ticker="^GSPC",
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )


def get_csi300(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch CSI 300 (沪深300, ``000300.SH``) historical data.

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.
    """
    return get_index_data(
        ticker="000300.SH",
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )
