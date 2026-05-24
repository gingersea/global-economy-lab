"""
Tests for the key-free data backends.

All tests mock the underlying HTTP call so they never touch the network
and are safe to run in any CI environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Make the project root importable.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── FRED CSV backend ──────────────────────────────────────────────────────────


class TestFredCsvBackend:
    """Verify the parsing logic of the FRED public-CSV backend."""

    _SAMPLE_CSV = (
        "DATE,GDPC1\n"
        "2008-01-01,15000.0\n"
        "2008-04-01,15100.0\n"
        "2008-07-01,.\n"  # FRED uses "." for missing observations
        "2008-10-01,14900.0\n"
    )

    def _mock_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_standardised_schema(self) -> None:
        from src.data_fetcher.backends import fred_csv

        with patch.object(
            fred_csv,
            "http_get",
            return_value=self._mock_response(self._SAMPLE_CSV),
        ):
            df = fred_csv.fetch_series("GDPC1")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["value", "series_id", "source"]
        assert df.index.name == "date"
        assert df["series_id"].iloc[0] == "GDPC1"
        assert df["source"].iloc[0] == "fred_csv"

    def test_missing_value_dot_is_nan(self) -> None:
        from src.data_fetcher.backends import fred_csv

        with patch.object(
            fred_csv,
            "http_get",
            return_value=self._mock_response(self._SAMPLE_CSV),
        ):
            df = fred_csv.fetch_series("GDPC1")

        # 4 rows total, one is NaN
        assert len(df) == 4
        assert int(df["value"].isna().sum()) == 1

    def test_date_filtering_applied(self) -> None:
        from src.data_fetcher.backends import fred_csv

        with patch.object(
            fred_csv,
            "http_get",
            return_value=self._mock_response(self._SAMPLE_CSV),
        ):
            df = fred_csv.fetch_series(
                "GDPC1", start_date="2008-06-01", end_date="2008-12-31"
            )
        assert df.index.min() >= pd.Timestamp("2008-06-01")
        assert df.index.max() <= pd.Timestamp("2008-12-31")

    def test_empty_series_id_raises(self) -> None:
        from src.data_fetcher.backends import fred_csv, BackendError

        with pytest.raises(BackendError):
            fred_csv.fetch_series("")

    def test_html_error_response_raises(self) -> None:
        from src.data_fetcher.backends import fred_csv, BackendError

        bogus = "<html><body>error</body></html>"
        with patch.object(
            fred_csv, "http_get", return_value=self._mock_response(bogus)
        ):
            with pytest.raises(BackendError):
                fred_csv.fetch_series("GDPC1")


# ── Treasury.gov backend ──────────────────────────────────────────────────────


class TestTreasuryGovBackend:
    """Verify the Daily Treasury Par Yield CSV parser."""

    _SAMPLE_CSV = (
        "Date,1 Mo,2 Mo,3 Mo,4 Mo,6 Mo,1 Yr,2 Yr,3 Yr,5 Yr,7 Yr,10 Yr,20 Yr,30 Yr\n"
        "01/03/2023,4.17,4.42,4.53,4.69,4.76,4.69,4.41,4.22,3.96,3.92,3.79,4.10,3.88\n"
        "01/04/2023,4.20,4.40,4.55,4.68,4.74,4.66,4.37,4.18,3.91,3.87,3.74,4.06,3.83\n"
    )

    def _mock_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = 200
        return resp

    def test_fetches_two_year_yield(self) -> None:
        from src.data_fetcher.backends import treasury_gov

        with patch.object(
            treasury_gov,
            "http_get",
            return_value=self._mock_response(self._SAMPLE_CSV),
        ):
            df = treasury_gov.fetch_series(
                "DGS2", start_date="2023-01-01", end_date="2023-01-31"
            )
        assert not df.empty
        assert list(df.columns) == ["value", "series_id", "source"]
        assert df["source"].iloc[0] == "treasury_gov"
        # 2y yield on 01/03/2023 was 4.41
        assert abs(df["value"].iloc[0] - 4.41) < 1e-6

    def test_alias_resolution(self) -> None:
        from src.data_fetcher.backends.treasury_gov import _resolve_maturity

        assert _resolve_maturity("dgs10") == "10y"
        assert _resolve_maturity("2y") == "2y"
        assert _resolve_maturity("DGS30") == "30y"

    def test_unknown_maturity_raises(self) -> None:
        from src.data_fetcher.backends import treasury_gov, BackendError

        with pytest.raises(BackendError):
            treasury_gov.fetch_series("BOGUS")


# ── World Bank backend ────────────────────────────────────────────────────────


class TestWorldBankBackend:
    """Verify World Bank WDI JSON parsing."""

    _PAYLOAD = (
        '[{"page":1,"pages":1,"per_page":50000,"total":3},'
        '[{"indicator":{"id":"NY.GDP.MKTP.CD","value":"GDP"},'
        '"country":{"id":"US","value":"United States"},'
        '"countryiso3code":"USA","date":"2022","value":25462700000000,'
        '"unit":"","obs_status":"","decimal":0},'
        '{"indicator":{"id":"NY.GDP.MKTP.CD","value":"GDP"},'
        '"country":{"id":"US","value":"United States"},'
        '"countryiso3code":"USA","date":"2021","value":23315080560000,'
        '"unit":"","obs_status":"","decimal":0},'
        '{"indicator":{"id":"NY.GDP.MKTP.CD","value":"GDP"},'
        '"country":{"id":"US","value":"United States"},'
        '"countryiso3code":"USA","date":"2020","value":null,'
        '"unit":"","obs_status":"","decimal":0}]]'
    )

    def test_parses_payload(self) -> None:
        from src.data_fetcher.backends import worldbank

        resp = MagicMock()
        resp.text = self._PAYLOAD
        with patch.object(worldbank, "http_get", return_value=resp):
            df = worldbank.fetch_series("US:NY.GDP.MKTP.CD")
        # Null values are skipped → 2 records
        assert len(df) == 2
        assert df["source"].iloc[0] == "worldbank"
        assert df["series_id"].iloc[0] == "US:NY.GDP.MKTP.CD"

    def test_bad_series_id_format(self) -> None:
        from src.data_fetcher.backends import worldbank, BackendError

        with pytest.raises(BackendError):
            worldbank.fetch_series("missing_colon")


# ── BondsFetcher integration with fred_csv backend ────────────────────────────


class TestBondsFetcherNoKey:
    """The bonds fetcher should work without a FRED API key."""

    def test_fetches_via_fred_csv_without_key(self, tmp_path: Path) -> None:
        from src.data_fetcher.bonds import BondsFetcher
        from src.data_fetcher.backends import fred_csv
        from config.settings import get_settings

        sample = pd.DataFrame(
            {
                "value": [4.0, 4.1, 4.2],
                "series_id": "DGS10",
                "source": "fred_csv",
            },
            index=pd.DatetimeIndex(
                ["2023-01-03", "2023-01-04", "2023-01-05"], name="date"
            ),
        )

        settings = get_settings()
        (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(settings, "fred_api_key", ""),
            patch.object(
                type(settings),
                "processed_data_dir",
                new_callable=lambda: property(lambda self: tmp_path / "processed"),
            ),
            patch.object(fred_csv, "fetch_series", return_value=sample),
        ):
            f = BondsFetcher(series_id="DGS10", source="fred_csv")
            df = f.fetch(
                start_date="2023-01-01",
                end_date="2023-01-31",
                use_cache=False,
            )

        assert not df.empty
        assert "yield_pct" in df.columns
        assert df["country"].iloc[0] == "US"


# ── MacroEconomicFetcher integration with fred_csv backend ────────────────────


class TestMacroFetcherNoKey:
    """``MacroEconomicFetcher`` should default to ``fred_csv`` when no key is set."""

    def test_source_fred_uses_csv_first(self, tmp_path: Path) -> None:
        from src.data_fetcher.macro_economic import MacroEconomicFetcher
        from src.data_fetcher.backends import fred_csv
        from config.settings import get_settings

        sample = pd.DataFrame(
            {
                "value": [100.0, 101.0],
                "series_id": "CPIAUCSL",
                "source": "fred_csv",
            },
            index=pd.DatetimeIndex(
                ["2023-01-01", "2023-02-01"], name="date"
            ),
        )

        settings = get_settings()
        (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(settings, "fred_api_key", ""),
            patch.object(
                type(settings),
                "processed_data_dir",
                new_callable=lambda: property(lambda self: tmp_path / "processed"),
            ),
            patch.object(fred_csv, "fetch_series", return_value=sample),
        ):
            f = MacroEconomicFetcher(series_id="CPIAUCSL", source="fred")
            df = f._fetch_remote("2023-01-01", "2023-12-31")

        assert not df.empty
        assert df["source"].iloc[0] == "fred_csv"


# ── Data quality schema validation ────────────────────────────────────────────


class TestSchemaValidation:
    """Light-weight schema invariants for the standardised backend output."""

    def test_fred_csv_schema_invariants(self) -> None:
        from src.data_fetcher.backends import fred_csv

        sample_csv = "DATE,X\n2023-01-01,1.0\n2023-02-01,2.0\n2023-03-01,3.0\n"
        resp = MagicMock()
        resp.text = sample_csv
        with patch.object(fred_csv, "http_get", return_value=resp):
            df = fred_csv.fetch_series("X")

        # Required columns
        for col in ("value", "series_id", "source"):
            assert col in df.columns

        # Unique sorted DatetimeIndex
        assert df.index.is_unique
        assert df.index.is_monotonic_increasing

        # No future dates
        assert df.index.max() <= pd.Timestamp.now(tz="UTC").tz_localize(
            None
        ) + pd.Timedelta(days=1)

        # Values must be numeric
        assert pd.api.types.is_float_dtype(df["value"])


# ── Registry helpers ──────────────────────────────────────────────────────────


class TestRegistryHelpers:
    """The new category / region helpers should filter correctly."""

    def test_sources_by_category_macro(self) -> None:
        from config.data_sources import sources_by_category

        macro = sources_by_category("macro")
        assert "us_cpi" in macro
        assert all(v.category == "macro" for v in macro.values())

    def test_sources_by_region_us(self) -> None:
        from config.data_sources import sources_by_region

        us = sources_by_region("US")
        assert "us_gdp" in us
        assert all(v.region == "US" for v in us.values())

    def test_requires_api_key_alias_false(self) -> None:
        """The legacy ``requires_api_key`` alias must be False everywhere."""
        from config.data_sources import ALL_SOURCES

        for key, cfg in ALL_SOURCES.items():
            assert cfg.requires_api_key is False, (
                f"{key} still flagged as requiring an API key"
            )
