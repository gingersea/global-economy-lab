"""
Stooq CSV backend.

Stooq.com offers free, key-less daily CSV downloads for many global
instruments – useful as a fallback to yfinance.

URL pattern:
    https://stooq.com/q/d/l/?s=<SYMBOL>&i=d
For US tickers, Stooq expects a ``".us"`` suffix (e.g. ``"aapl.us"``).
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.backends import BackendError
from src.data_fetcher.backends._http import http_get


def fetch_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a daily OHLCV history from Stooq.

    Args:
        series_id:  Stooq symbol (e.g. ``"spx"``, ``"aapl.us"``, ``"eurusd"``).
        start_date: Optional ISO-8601 lower bound.
        end_date:   Optional ISO-8601 upper bound.

    Returns:
        DataFrame with ``["open", "high", "low", "close", "volume",
        "value", "series_id", "source"]`` where ``value`` is the close.
    """
    params = {"s": series_id, "i": "d"}
    if start_date:
        params["d1"] = start_date.replace("-", "")
    if end_date:
        params["d2"] = end_date.replace("-", "")

    resp = http_get("https://stooq.com/q/d/l/", params=params)
    text = resp.text
    if not text or "Date" not in text.split("\n", 1)[0]:
        raise BackendError(
            f"stooq: unexpected response for {series_id}: {text[:120]!r}"
        )

    df = pd.read_csv(StringIO(text))
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise BackendError(
            f"stooq: missing required columns; got {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["value"] = df["close"]
    df["series_id"] = series_id
    df["source"] = "stooq"

    logger.debug(f"stooq: fetched {len(df)} rows for {series_id}")
    return df
