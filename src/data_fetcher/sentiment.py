"""
Market sentiment data fetcher.

Provides VIX (fear index) and placeholders for social-media /
options-flow sentiment indicators.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
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


_PCR_CACHE_PATH = Path(__file__).parent.parent.parent / "data/processed/cboe_pcr_equity_all.parquet"


def get_put_call_ratio(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch CBOE equity put/call ratio from public CSV.

    Data source: https://cdn.cboe.com/api/global/us_indices/daily_pcr_equity_all.csv

    Caches results to ``data/processed/`` as Parquet.
    Gracefully degrades to an empty DataFrame on any failure (network,
    CBOE rate-limiting, parse error).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available and fresh.

    Returns:
        DataFrame with columns ``date`` (datetime) and ``put_call_ratio`` (float).
        Empty DataFrame if data is unavailable.
    """
    end_date = end_date or str(date.today())

    # Check cache first
    if use_cache and _PCR_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(_PCR_CACHE_PATH)
            if not cached.empty and "date" in cached.columns:
                cached["date"] = pd.to_datetime(cached["date"])
                cached.set_index("date", inplace=True)
                # Filter to requested range
                mask = (cached.index >= start_date) & (cached.index <= end_date)
                result = cached.loc[mask].reset_index()
                if not result.empty:
                    logger.info(
                        f"get_put_call_ratio: {len(result)} rows from cache "
                        f"({start_date} → {end_date})"
                    )
                    return result
        except Exception as exc:
            logger.warning(f"get_put_call_ratio: cache read failed ({exc}), re-fetching")

    # Fetch from CBOE
    try:
        import requests
        url = "https://cdn.cboe.com/api/global/us_indices/daily_pcr_equity_all.csv"
        logger.info(f"get_put_call_ratio: fetching from {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # CBOE CSV format: DATE, P/C Ratio
        from io import StringIO
        raw = resp.text
        df = pd.read_csv(StringIO(raw))

        # Normalize columns
        date_col = None
        pcr_col = None
        for col in df.columns:
            col_lower = col.strip().lower()
            if "date" in col_lower:
                date_col = col
            elif "p/c" in col_lower or "put" in col_lower or "ratio" in col_lower:
                pcr_col = col

        if date_col is None or pcr_col is None:
            logger.warning(
                f"get_put_call_ratio: unexpected CSV columns {list(df.columns)}"
            )
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df[date_col])
        df["put_call_ratio"] = pd.to_numeric(df[pcr_col], errors="coerce")
        df = df[["date", "put_call_ratio"]].dropna()
        df = df.sort_values("date")

        # Cache full dataset
        df.set_index("date").to_parquet(_PCR_CACHE_PATH)
        logger.info(f"get_put_call_ratio: cached {len(df)} rows to {_PCR_CACHE_PATH}")

        # Filter to requested range
        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        result = df.loc[mask].reset_index(drop=True)
        logger.info(f"get_put_call_ratio: {len(result)} rows ({start_date} → {end_date})")
        return result

    except Exception as exc:
        logger.warning(f"get_put_call_ratio: fetch failed — {exc}. Returning empty DataFrame.")
        # Try to return whatever cached data we have
        if _PCR_CACHE_PATH.exists():
            try:
                cached = pd.read_parquet(_PCR_CACHE_PATH)
                if not cached.empty and "date" in cached.columns:
                    cached["date"] = pd.to_datetime(cached["date"])
                    cached.set_index("date", inplace=True)
                    mask = (cached.index >= start_date) & (cached.index <= end_date)
                    result = cached.loc[mask].reset_index()
                    if not result.empty:
                        logger.info(
                            f"get_put_call_ratio: fallback to cache, {len(result)} rows"
                        )
                        return result
            except Exception:
                pass
        return pd.DataFrame()


def get_treasury_spread(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch US Treasury 10Y-2Y yield spread from FRED.

    Uses the existing ``src.data_fetcher.bonds.get_yield_curve()``
    which fetches DGS10 and DGS2 from FRED (public CSV, no API key).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with columns ``date`` and ``spread_10y2y`` (float).
        Empty DataFrame if data is unavailable.
    """
    try:
        from src.data_fetcher.bonds import get_yield_curve
        df = get_yield_curve(
            country="US",
            start_date=start_date,
            end_date=end_date or str(date.today()),
            use_cache=use_cache,
        )
        if df is None or df.empty:
            logger.warning("get_treasury_spread: yield curve data is empty")
            return pd.DataFrame()

        result = pd.DataFrame({
            "date": pd.to_datetime(df.index),
            "spread_10y2y": df["spread_2s10s"].values,
        }).dropna()
        logger.info(f"get_treasury_spread: {len(result)} rows ({start_date} → {end_date})")
        return result
    except Exception as exc:
        logger.warning(f"get_treasury_spread: failed — {exc}. Returning empty DataFrame.")
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
