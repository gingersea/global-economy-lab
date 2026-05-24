"""
US Treasury Daily Treasury Par Yield Curve backend.

Source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates

Public XML feed (no API key, no rate-limit) for daily nominal Par yields
on every maturity from 1 month to 30 years, dating back to 1990.

This backend wraps the underlying XML feed and returns a standardised
DataFrame for a single maturity, identified by series_id such as
``"BC_2YEAR"`` or ``"BC_10YEAR"``.  The full list of available maturities
is exposed via :data:`MATURITY_TO_SERIES`.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.backends import BackendError
from src.data_fetcher.backends._http import http_get


_BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
)  # noqa: E501 – documented base URL (kept for reference)

# Map shorthand maturity → Treasury XML column / CSV header name.
MATURITY_TO_SERIES: Dict[str, str] = {
    "1m": "1 Mo",
    "2m": "2 Mo",
    "3m": "3 Mo",
    "4m": "4 Mo",
    "6m": "6 Mo",
    "1y": "1 Yr",
    "2y": "2 Yr",
    "3y": "3 Yr",
    "5y": "5 Yr",
    "7y": "7 Yr",
    "10y": "10 Yr",
    "20y": "20 Yr",
    "30y": "30 Yr",
}


def _csv_url(year: int) -> str:
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/daily-treasury-rates.csv/"
        f"{year}/all?type=daily_treasury_yield_curve&"
        f"field_tdr_date_value={year}&page&_format=csv"
    )


def _fetch_year_csv(year: int) -> pd.DataFrame:
    """Fetch one calendar year of the Treasury par yield curve CSV."""
    url = _csv_url(year)
    resp = http_get(url)
    text = resp.text
    if not text or "Date" not in text.splitlines()[0]:
        raise BackendError(
            f"treasury_gov: unexpected CSV response for year {year}: "
            f"{text[:120]!r}"
        )
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    if "Date" not in df.columns:
        raise BackendError(
            f"treasury_gov: missing 'Date' column for year {year}; "
            f"columns={list(df.columns)}"
        )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index.name = "date"
    return df


def fetch_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a Treasury par-yield series for the requested maturity.

    Args:
        series_id:  Maturity shorthand (see :data:`MATURITY_TO_SERIES`).
                    Also accepts the FRED-style ``"DGS2"``/``"DGS10"``
                    aliases which are mapped onto the matching maturity.
        start_date: Optional ISO-8601 lower bound (inclusive).
        end_date:   Optional ISO-8601 upper bound (inclusive).

    Returns:
        Standardised DataFrame ``["value", "series_id", "source"]``.
    """
    maturity = _resolve_maturity(series_id)
    column = MATURITY_TO_SERIES[maturity]

    start = pd.Timestamp(start_date) if start_date else pd.Timestamp("1990-01-01")
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())

    years: List[int] = list(range(start.year, end.year + 1))
    frames: List[pd.DataFrame] = []
    for year in years:
        try:
            frames.append(_fetch_year_csv(year))
        except BackendError as exc:
            logger.warning(f"treasury_gov: year {year} skipped – {exc}")

    if not frames:
        raise BackendError(
            f"treasury_gov: no data fetched for {series_id} between {start} and {end}"
        )

    full = pd.concat(frames).sort_index()
    if column not in full.columns:
        raise BackendError(
            f"treasury_gov: column {column!r} missing; available={list(full.columns)}"
        )

    series = pd.to_numeric(full[column], errors="coerce").dropna()
    series = series.loc[(series.index >= start) & (series.index <= end)]

    out = pd.DataFrame(
        {
            "value": series.astype("float64"),
            "series_id": series_id,
            "source": "treasury_gov",
        }
    )
    out.index.name = "date"
    logger.debug(
        f"treasury_gov: fetched {len(out)} rows for {series_id} "
        f"({start.date()} → {end.date()})"
    )
    return out


def _resolve_maturity(series_id: str) -> str:
    """Translate FRED-style DGS ids and case variants to a maturity key."""
    sid = series_id.strip().lower()
    if sid in MATURITY_TO_SERIES:
        return sid
    # FRED aliases
    aliases = {
        "dgs1mo": "1m",
        "dgs3mo": "3m",
        "dgs6mo": "6m",
        "dgs1": "1y",
        "dgs2": "2y",
        "dgs3": "3y",
        "dgs5": "5y",
        "dgs7": "7y",
        "dgs10": "10y",
        "dgs20": "20y",
        "dgs30": "30y",
    }
    if sid in aliases:
        return aliases[sid]
    raise BackendError(
        f"treasury_gov: unknown maturity / series id {series_id!r}. "
        f"Choose from {list(MATURITY_TO_SERIES)} or DGS*."
    )
