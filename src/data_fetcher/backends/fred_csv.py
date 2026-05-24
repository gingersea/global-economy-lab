"""
FRED CSV public-endpoint backend.

Uses the unauthenticated CSV download URL exposed by FRED for any series:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

This endpoint requires **no API key** and returns the full history of
the series in a simple two-column CSV (``DATE,<series_id>``).

Returned DataFrame schema (see ``backends/__init__.py``):

- index ``date`` (``DatetimeIndex``)
- ``value``      – float (NaN for missing observations, represented as ".")
- ``series_id``  – the requested series id
- ``source``     – ``"fred_csv"``
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.backends import BackendError
from src.data_fetcher.backends._http import http_get


BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a FRED time series from the public CSV endpoint.

    Args:
        series_id:  FRED series identifier (e.g. ``"GDPC1"``).
        start_date: Optional ISO-8601 lower bound (inclusive).
        end_date:   Optional ISO-8601 upper bound (inclusive).

    Returns:
        Standardised DataFrame with columns ``["value", "series_id", "source"]``.

    Raises:
        BackendError: If the download or parsing fails.
    """
    if not series_id:
        raise BackendError("fred_csv backend requires a non-empty series_id")

    params = {"id": series_id}
    if start_date:
        params["cosd"] = start_date
    if end_date:
        params["coed"] = end_date

    resp = http_get(BASE_URL, params=params)
    text = resp.text

    # Defensive: occasionally the endpoint returns an HTML error page with
    # 200 OK; verify the first line really looks like a CSV header.
    first_line = text.splitlines()[0] if text else ""
    if "," not in first_line or "<html" in text[:200].lower():
        raise BackendError(
            f"fred_csv: unexpected response for series {series_id!r}: "
            f"{first_line[:120]!r}"
        )

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise BackendError(
            f"fred_csv: could not parse CSV for {series_id}: {exc}"
        ) from exc

    if df.empty or df.shape[1] < 2:
        raise BackendError(
            f"fred_csv: empty / malformed CSV for series {series_id!r}"
        )

    # First column is always the date column; the second is the series.
    date_col = df.columns[0]
    value_col = df.columns[1]

    # FRED uses "." to represent missing observations.
    df[value_col] = pd.to_numeric(
        df[value_col].replace(".", pd.NA), errors="coerce"
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"

    out = pd.DataFrame(
        {
            "value": df[value_col].astype("float64"),
            "series_id": series_id,
            "source": "fred_csv",
        },
        index=df.index,
    )

    # Apply client-side date filtering (the endpoint generally honours
    # cosd/coed but be defensive).
    if start_date:
        out = out.loc[out.index >= pd.Timestamp(start_date)]
    if end_date:
        out = out.loc[out.index <= pd.Timestamp(end_date)]

    logger.debug(
        f"fred_csv: fetched {len(out)} rows for {series_id} "
        f"({start_date or '…'} → {end_date or '…'})"
    )
    return out
