"""
Bond / yield-curve data fetcher.

Retrieves government bond yields from FRED (US Treasuries) and
provides spread calculations (2s10s, etc.).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.base import BaseFetcher
from config.settings import get_settings


# FRED series IDs for key maturities
_US_YIELD_SERIES: dict[str, str] = {
    "1m": "DGS1MO",
    "3m": "DGS3MO",
    "6m": "DGS6MO",
    "1y": "DGS1",
    "2y": "DGS2",
    "3y": "DGS3",
    "5y": "DGS5",
    "7y": "DGS7",
    "10y": "DGS10",
    "20y": "DGS20",
    "30y": "DGS30",
}


class BondsFetcher(BaseFetcher):
    """Fetcher for government bond / treasury yield data.

    Args:
        series_id: FRED series identifier (e.g. ``"DGS10"`` for the
                   US 10-year Treasury CMT).
        country:   Country code – currently only ``"US"`` is supported
                   via FRED.
    """

    def __init__(
        self,
        series_id: str = "DGS10",
        country: str = "US",
    ) -> None:
        super().__init__(name=f"bond_{series_id}")
        self.series_id = series_id
        self.country = country

    # ── Abstract implementation ───────────────────────────────────────────────

    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch yield series from FRED."""
        settings = get_settings()
        if not settings.fred_api_key:
            logger.warning("FRED_API_KEY is not set – skipping bond fetch.")
            return pd.DataFrame()
        try:
            from fredapi import Fred  # type: ignore[import]

            fred = Fred(api_key=settings.fred_api_key)
            series: pd.Series = fred.get_series(
                self.series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            df = series.to_frame(name="yield_pct")
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df["series_id"] = self.series_id
            df["country"] = self.country
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[{self.name}] FRED bond fetch failed "
                f"for {self.series_id}: {exc}"
            )
            raise


# ── Convenience functions ─────────────────────────────────────────────────────


def get_us_treasury_yield(
    maturity: str = "10y",
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch US Treasury yield for a given maturity.

    Args:
        maturity:   One of ``"1m"``, ``"3m"``, ``"6m"``, ``"1y"``,
                    ``"2y"``, ``"3y"``, ``"5y"``, ``"7y"``, ``"10y"``,
                    ``"20y"``, ``"30y"``.
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date. Defaults to today.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with ``yield_pct`` column, indexed by date.
    """
    if maturity not in _US_YIELD_SERIES:
        raise ValueError(
            f"Unsupported maturity {maturity!r}. "
            f"Choose from {list(_US_YIELD_SERIES.keys())}"
        )
    series_id = _US_YIELD_SERIES[maturity]
    fetcher = BondsFetcher(series_id=series_id, country="US")
    return fetcher.fetch(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )


def get_yield_curve(
    country: str = "US",
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch 2-year and 10-year yields and compute the spread (2s10s).

    Args:
        country:    Country code.  Only ``"US"`` is supported currently.
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with columns ``["2y", "10y", "spread_2s10s"]``.
    """
    if country != "US":
        logger.warning(
            f"get_yield_curve: country {country!r} not yet supported. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame()

    df_2y = get_us_treasury_yield(
        "2y", start_date=start_date, end_date=end_date, use_cache=use_cache
    )
    df_10y = get_us_treasury_yield(
        "10y", start_date=start_date, end_date=end_date, use_cache=use_cache
    )

    if df_2y.empty or df_10y.empty:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "2y": df_2y["yield_pct"],
            "10y": df_10y["yield_pct"],
        }
    ).dropna()
    result["spread_2s10s"] = result["10y"] - result["2y"]
    return result
