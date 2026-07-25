"""
Macro-economic data fetcher.

Retrieves GDP, CPI, PMI, interest rates, and other macro indicators
from FRED (via fredapi) and Chinese data from akshare.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.base import BaseFetcher
from config.settings import get_settings


class MacroEconomicFetcher(BaseFetcher):
    """Fetcher for macro-economic time series.

    Supports FRED series (when ``source="fred"``) and a limited set
    of Chinese indicators via akshare (``source="akshare"``).
    """

    def __init__(
        self,
        series_id: str = "",
        source: str = "fred",
        indicator: str = "",
    ) -> None:
        """
        Args:
            series_id: FRED series identifier (e.g. ``"GDPC1"``).
            source:    Data source backend – ``"fred"`` or ``"akshare"``.
            indicator: For akshare sources, the specific indicator name
                       (e.g. ``"china_cpi"``).
        """
        name = series_id or indicator or "macro"
        super().__init__(name=name)
        self.series_id = series_id
        self.source = source
        self.indicator = indicator

    # ── Abstract implementation ───────────────────────────────────────────────

    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Route to the appropriate backend."""
        if self.source == "fred":
            # Prefer the no-key CSV endpoint and fall back to the authenticated
            # API only if explicitly configured with a key.
            try:
                return self._fetch_fred_csv(start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[{self.name}] fred_csv backend failed ({exc}); "
                    "trying fredapi fallback."
                )
                return self._fetch_fred(start_date, end_date)
        elif self.source == "fred_csv":
            return self._fetch_fred_csv(start_date, end_date)
        elif self.source == "fredapi":
            return self._fetch_fred(start_date, end_date)
        elif self.source == "akshare":
            return self._fetch_akshare(start_date, end_date)
        elif self.source == "worldbank":
            from src.data_fetcher.backends import worldbank

            return worldbank.fetch_series(
                self.series_id, start_date=start_date, end_date=end_date
            )
        elif self.source == "ecb_sdw":
            from src.data_fetcher.backends import ecb_sdw

            return ecb_sdw.fetch_series(
                self.series_id, start_date=start_date, end_date=end_date
            )
        else:
            raise ValueError(f"Unknown macro source: {self.source!r}")

    # ── Backend implementations ───────────────────────────────────────────────

    def _fetch_fred_csv(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch a FRED series via the public, key-less CSV endpoint."""
        from src.data_fetcher.backends import fred_csv

        df = fred_csv.fetch_series(
            self.series_id, start_date=start_date, end_date=end_date
        )
        # Preserve original column for backwards compatibility with consumers
        # that previously expected just ["value", "series_id"].
        return df

    def _fetch_fred(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch a FRED series using fredapi."""
        settings = get_settings()
        if not settings.fred_api_key:
            logger.warning(
                "FRED_API_KEY is not set – skipping FRED fetch."
            )
            return pd.DataFrame()
        try:
            from fredapi import Fred  # type: ignore[import]

            fred = Fred(api_key=settings.fred_api_key)
            series: pd.Series = fred.get_series(
                self.series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            df = series.to_frame(name="value")
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df["series_id"] = self.series_id
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[{self.name}] FRED fetch failed for {self.series_id}: {exc}"
            )
            raise

    def _fetch_akshare(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch Chinese macro data from akshare."""
        try:
            import akshare as ak  # type: ignore[import]
        except ImportError:
            logger.error("akshare is not installed.")
            return pd.DataFrame()

        try:
            if self.indicator == "china_cpi":
                df = ak.macro_china_cpi_yearly()
                df = df.rename(columns={"date": "date", "cpi": "value"})
                df["date"] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index("date").sort_index()
                # filter by date range
                df = df.loc[start_date:end_date]
                return df[["value"]] if "value" in df.columns else df.iloc[:, :1]
            else:
                logger.warning(
                    f"Unsupported akshare indicator: {self.indicator!r}"
                )
                return pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[{self.name}] akshare fetch failed for "
                f"{self.indicator}: {exc}"
            )
            raise


# ── Convenience functions ─────────────────────────────────────────────────────


def get_us_gdp(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch US Real GDP (FRED: GDPC1).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date. Defaults to today.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with columns ``["value", "series_id"]`` indexed by date.
    """
    fetcher = MacroEconomicFetcher(series_id="GDPC1")
    return fetcher.fetch(start_date=start_date, end_date=end_date, use_cache=use_cache)


def get_us_cpi(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch US CPI (FRED: CPIAUCSL).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with columns ``["value", "series_id"]``.
    """
    fetcher = MacroEconomicFetcher(series_id="CPIAUCSL")
    return fetcher.fetch(start_date=start_date, end_date=end_date, use_cache=use_cache)


def get_china_cpi(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch China CPI year-over-year from akshare.

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame indexed by date.
    """
    fetcher = MacroEconomicFetcher(source="akshare", indicator="china_cpi")
    return fetcher.fetch(start_date=start_date, end_date=end_date, use_cache=use_cache)


def get_us_pmi(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch ISM Manufacturing PMI.

    Primary source: akshare (``macro_usa_ism_pmi``).  Falls back to FRED
    series ``NAPM`` when akshare is unavailable or the data is too stale.

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date. Defaults to today.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with columns ``["value", "series_id"]`` indexed by date.
    """
    # ── Primary: akshare ISM Manufacturing PMI ───────────────────────────
    try:
        import akshare as ak  # type: ignore[import]

        raw = ak.macro_usa_ism_pmi()
        if raw is not None and not raw.empty:
            # Columns: 商品, 日期, 今值, 预测值, 前值
            date_col = "日期" if "日期" in raw.columns else raw.columns[1]
            value_col = "今值" if "今值" in raw.columns else raw.columns[2]
            df = raw[[date_col, value_col]].copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "date", value_col: "value"})
            df = df.dropna(subset=["value"]).set_index("date").sort_index()
            df["series_id"] = "ISM_PMI"
            # Apply date filtering
            if start_date:
                df = df.loc[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df.loc[df.index <= pd.Timestamp(end_date)]
            if not df.empty:
                logger.info(
                    f"[PMI] akshare: {len(df)} rows "
                    f"({df.index[0].date()} → {df.index[-1].date()})"
                )
                return df
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[PMI] akshare backend failed: {exc}")

    # ── Fallback: FRED NAPM (requires API key; public CSV returns 404) ──
    fetcher = MacroEconomicFetcher(series_id="NAPM")
    return fetcher.fetch(start_date=start_date, end_date=end_date, use_cache=use_cache)


def get_global_pmi(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch PMI for major economies (US, Eurozone placeholder).

    Returns a dict mapping region name to a DataFrame of PMI values.

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.
    """
    results: dict[str, pd.DataFrame] = {}
    # US PMI from FRED
    results["US"] = get_us_pmi(
        start_date=start_date, end_date=end_date, use_cache=use_cache
    )
    # Future: add EU PMI (e.g., from FRED series EURPMNFCT or external source)
    logger.info(
        "get_global_pmi: EU PMI placeholder – add an akshare or FRED source to expand."
    )
    return results


def get_fed_funds_rate(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch US Federal Funds Effective Rate (FRED: FEDFUNDS).

    Args:
        start_date: ISO-8601 start date.
        end_date:   ISO-8601 end date.
        use_cache:  Return cached data when available.

    Returns:
        DataFrame with columns ``["value", "series_id"]``.
    """
    fetcher = MacroEconomicFetcher(series_id="FEDFUNDS")
    return fetcher.fetch(start_date=start_date, end_date=end_date, use_cache=use_cache)
