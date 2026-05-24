"""
World Bank Open Data (WDI) backend.

REST API: https://api.worldbank.org/v2/  (no API key required).

Series identifier format: ``"<COUNTRY>:<INDICATOR>"`` – for example
``"US:NY.GDP.MKTP.CD"`` for US nominal GDP in USD or
``"CN:FP.CPI.TOTL.ZG"`` for China CPI inflation %.
"""

from __future__ import annotations

import json
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
    """Fetch a World Bank WDI indicator for one country.

    Args:
        series_id:  ``"<COUNTRY_ISO2_OR_ISO3>:<INDICATOR_CODE>"``.
        start_date: Optional ISO-8601 lower bound (year is used).
        end_date:   Optional ISO-8601 upper bound (year is used).

    Returns:
        Standardised DataFrame ``["value", "series_id", "source"]``
        indexed by ``date`` at calendar-year-end.
    """
    if ":" not in series_id:
        raise BackendError(
            f"worldbank: expected 'COUNTRY:INDICATOR' format, got {series_id!r}"
        )
    country, indicator = series_id.split(":", 1)
    start_year = pd.Timestamp(start_date).year if start_date else 1960
    end_year = pd.Timestamp(end_date).year if end_date else pd.Timestamp.today().year

    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    )
    params = {
        "format": "json",
        "per_page": "20000",
        "date": f"{start_year}:{end_year}",
    }
    resp = http_get(url, params=params)
    try:
        payload = json.loads(resp.text)
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"worldbank: JSON parse failed: {exc}") from exc

    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise BackendError(
            f"worldbank: empty response for {series_id} "
            f"({start_year}-{end_year})"
        )

    records = []
    for row in payload[1]:
        year = row.get("date")
        value = row.get("value")
        if year is None or value is None:
            continue
        records.append((pd.Timestamp(f"{year}-12-31"), float(value)))
    if not records:
        raise BackendError(f"worldbank: no observations for {series_id}")

    df = pd.DataFrame(records, columns=["date", "value"]).sort_values("date")
    df = df.set_index("date")
    df["series_id"] = series_id
    df["source"] = "worldbank"

    logger.debug(f"worldbank: fetched {len(df)} obs for {series_id}")
    return df
