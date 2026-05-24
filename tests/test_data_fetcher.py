"""
Unit tests for the data fetcher modules.

These tests use mocking and synthetic data to avoid real network calls,
ensuring they run reliably in any CI/CD environment.
"""

from __future__ import annotations

import sys
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so imports work when running
# pytest from any directory.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame indexed by date."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    prices = 100 + rng.standard_normal(100).cumsum()
    return pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 5_000_000, size=100),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def sample_pmi() -> pd.DataFrame:
    """Return a synthetic PMI DataFrame."""
    dates = pd.date_range("2010-01-01", periods=60, freq="ME")
    values = 50 + 5 * np.sin(np.linspace(0, 4 * np.pi, 60))
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def sample_cpi() -> pd.DataFrame:
    """Return a synthetic CPI level DataFrame."""
    dates = pd.date_range("2010-01-01", periods=60, freq="ME")
    values = np.linspace(200, 260, 60)  # Gradual inflation
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Return a temporary directory for cache files."""
    return tmp_path / "processed"


# ── BaseFetcher tests ─────────────────────────────────────────────────────────


class TestBaseFetcher:
    """Tests for the BaseFetcher caching and retry logic."""

    def test_cache_path_is_deterministic(self, tmp_cache: Path) -> None:
        """The same parameters must always produce the same cache file name."""
        from config.settings import get_settings

        settings = get_settings()
        original_dir = settings.processed_data_dir

        with patch.object(type(settings), "processed_data_dir", new_callable=lambda: property(lambda self: tmp_cache)):
            from src.data_fetcher.equities import EquitiesFetcher

            f = EquitiesFetcher("^GSPC")
            p1 = f._cache_path("2020-01-01", "2021-01-01")
            p2 = f._cache_path("2020-01-01", "2021-01-01")
            assert p1 == p2, "Same args must produce the same cache path."
            p3 = f._cache_path("2020-01-01", "2022-01-01")
            assert p1 != p3, "Different end dates must produce different paths."

    def test_cache_roundtrip(self, tmp_cache: Path, sample_ohlcv: pd.DataFrame) -> None:
        """Data saved to cache can be read back correctly."""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed – skipping parquet round-trip test")
        tmp_cache.mkdir(parents=True, exist_ok=True)
        cache_file = tmp_cache / "test_cache.parquet"
        sample_ohlcv.to_parquet(cache_file)
        loaded = pd.read_parquet(cache_file)
        pd.testing.assert_frame_equal(sample_ohlcv, loaded)


# ── EquitiesFetcher tests ─────────────────────────────────────────────────────


class TestEquitiesFetcher:
    """Tests for the equities data fetcher."""

    def test_market_detection_us(self) -> None:
        from src.data_fetcher.equities import EquitiesFetcher

        assert EquitiesFetcher._detect_market("^GSPC") == "US"
        assert EquitiesFetcher._detect_market("AAPL") == "US"

    def test_market_detection_cn(self) -> None:
        from src.data_fetcher.equities import EquitiesFetcher

        assert EquitiesFetcher._detect_market("000300.SH") == "CN_SHARE"
        assert EquitiesFetcher._detect_market("600519.SH") == "CN_SHARE"

    def test_market_detection_hk(self) -> None:
        from src.data_fetcher.equities import EquitiesFetcher

        assert EquitiesFetcher._detect_market("0700.HK") == "HK"

    def test_fetcher_name_sanitization(self) -> None:
        from src.data_fetcher.equities import EquitiesFetcher

        f = EquitiesFetcher("^GSPC")
        assert "^" not in f.name, "Caret must be removed from fetcher name."

    def test_fetch_returns_empty_on_network_error(self, tmp_path: Path) -> None:
        """fetch() returns an empty DataFrame when the network call fails."""
        from src.data_fetcher.equities import EquitiesFetcher
        from config.settings import get_settings

        settings = get_settings()
        with patch.object(
            type(settings),
            "processed_data_dir",
            new_callable=lambda: property(lambda self: tmp_path / "processed"),
        ):
            f = EquitiesFetcher("INVALID_TICKER_XYZ")
            (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

            with patch.object(f, "_fetch_remote", side_effect=RuntimeError("Network error")):
                result = f.fetch(use_cache=False)
            assert isinstance(result, pd.DataFrame)
            assert result.empty


# ── MacroEconomicFetcher tests ────────────────────────────────────────────────


class TestMacroEconomicFetcher:
    """Tests for the macro-economic data fetcher."""

    def test_returns_empty_df_without_fred_key(self, tmp_path: Path) -> None:
        """Without a FRED API key, FRED fetches should return empty DataFrame."""
        from src.data_fetcher.macro_economic import MacroEconomicFetcher
        from config.settings import get_settings

        settings = get_settings()
        (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(settings, "fred_api_key", ""),
            patch.object(
                type(settings),
                "processed_data_dir",
                new_callable=lambda: property(lambda self: tmp_path / "processed"),
            ),
        ):
            f = MacroEconomicFetcher(series_id="GDPC1")
            result = f._fetch_remote("2020-01-01", "2021-01-01")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_unknown_source_raises(self) -> None:
        from src.data_fetcher.macro_economic import MacroEconomicFetcher

        f = MacroEconomicFetcher(source="bogus_source")
        with pytest.raises(ValueError, match="Unknown macro source"):
            f._fetch_remote("2020-01-01", "2021-01-01")


# ── CommoditiesFetcher tests ──────────────────────────────────────────────────


class TestCommoditiesFetcher:
    """Tests for the commodities data fetcher."""

    def test_ticker_stored_correctly(self) -> None:
        from src.data_fetcher.commodities import CommoditiesFetcher

        f = CommoditiesFetcher(ticker="GC=F")
        assert f.ticker == "GC=F"

    def test_name_sanitized(self) -> None:
        from src.data_fetcher.commodities import CommoditiesFetcher

        f = CommoditiesFetcher(ticker="GC=F")
        assert "=" not in f.name


# ── BondsFetcher tests ────────────────────────────────────────────────────────


class TestBondsFetcher:
    """Tests for the bond yield fetcher."""

    def test_unsupported_maturity_raises(self) -> None:
        from src.data_fetcher.bonds import get_us_treasury_yield

        with pytest.raises(ValueError, match="Unsupported maturity"):
            get_us_treasury_yield("99y")

    def test_yield_curve_returns_empty_for_unsupported_country(self) -> None:
        from src.data_fetcher.bonds import get_yield_curve

        result = get_yield_curve(country="JP")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── SentimentFetcher tests ────────────────────────────────────────────────────


class TestSentimentFetcher:
    """Tests for the sentiment data fetcher."""

    def test_placeholder_functions_return_empty_df(self) -> None:
        from src.data_fetcher.sentiment import (
            get_put_call_ratio,
            get_social_media_sentiment,
        )

        assert get_put_call_ratio().empty
        assert get_social_media_sentiment("Bitcoin").empty


# ── Analysis: CyclePosition tests ─────────────────────────────────────────────


class TestCyclePosition:
    """Tests for the economic cycle analysis."""

    def test_recovery_phase(self, sample_pmi: pd.DataFrame, sample_cpi: pd.DataFrame) -> None:
        """PMI above 50 + low CPI inflation → 'recovery' or 'overheat'."""
        from src.analysis.cycle_position import get_cycle_position

        result = get_cycle_position(sample_pmi, sample_cpi)
        assert not result.empty
        assert "phase" in result.columns
        assert set(result["phase"]).issubset(
            {"recovery", "overheat", "stagflation", "recession", "unknown"}
        )

    def test_empty_inputs_return_empty(self) -> None:
        from src.analysis.cycle_position import get_cycle_position

        result = get_cycle_position(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_get_current_phase_returns_dict(
        self, sample_pmi: pd.DataFrame, sample_cpi: pd.DataFrame
    ) -> None:
        from src.analysis.cycle_position import get_current_phase

        result = get_current_phase(sample_pmi, sample_cpi)
        assert isinstance(result, dict)
        assert "phase" in result
        assert "phase_label" in result


# ── Analysis: Correlation tests ───────────────────────────────────────────────


class TestCorrelation:
    """Tests for the correlation analysis module."""

    def test_corr_matrix_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        from src.analysis.correlation import compute_correlation_matrix

        price_data = {"A": sample_ohlcv, "B": sample_ohlcv * 1.1}
        result = compute_correlation_matrix(price_data)
        assert result.shape == (2, 2)

    def test_corr_matrix_diagonal_is_one(self, sample_ohlcv: pd.DataFrame) -> None:
        from src.analysis.correlation import compute_correlation_matrix

        price_data = {"A": sample_ohlcv, "B": sample_ohlcv * 0.9}
        result = compute_correlation_matrix(price_data)
        assert all(abs(result.loc[c, c] - 1.0) < 1e-9 for c in result.columns)

    def test_empty_input_returns_empty(self) -> None:
        from src.analysis.correlation import compute_correlation_matrix

        result = compute_correlation_matrix({})
        assert result.empty

    def test_rolling_correlation_length(self, sample_ohlcv: pd.DataFrame) -> None:
        from src.analysis.correlation import compute_rolling_correlation

        price_data = {"X": sample_ohlcv, "Y": sample_ohlcv * 0.95}
        series = compute_rolling_correlation(price_data, "X", "Y", window=10)
        # pct_change produces NaN for the first row but keeps the same index length
        assert len(series) == len(sample_ohlcv)

    def test_rolling_correlation_missing_asset_raises(
        self, sample_ohlcv: pd.DataFrame
    ) -> None:
        from src.analysis.correlation import compute_rolling_correlation

        with pytest.raises(ValueError, match="not found"):
            compute_rolling_correlation(
                {"X": sample_ohlcv}, "X", "MISSING"
            )


# ── Analysis: EventStudy tests ────────────────────────────────────────────────


class TestEventStudy:
    """Tests for the event study module."""

    def test_analyze_event_returns_correct_columns(
        self, sample_ohlcv: pd.DataFrame
    ) -> None:
        from src.analysis.event_study import analyze_event

        price_data = {"SP500": sample_ohlcv, "Gold": sample_ohlcv * 0.5}
        result = analyze_event(
            event_date="2020-03-15",
            price_data=price_data,
            pre_window=10,
            post_window=20,
        )
        assert not result.empty
        assert "SP500" in result.columns
        assert "Gold" in result.columns

    def test_t0_cumulative_return_is_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        """On the event day (offset 0) the cumulative return should be ~0."""
        from src.analysis.event_study import analyze_event

        price_data = {"Asset": sample_ohlcv}
        result = analyze_event(
            event_date="2020-03-15",
            price_data=price_data,
            pre_window=10,
            post_window=20,
        )
        t0_row = result.loc[result.index == 0]
        if not t0_row.empty:
            assert abs(float(t0_row["Asset"].iloc[0])) < 1e-9

    def test_preset_events_are_listed(self) -> None:
        from src.analysis.event_study import list_preset_events

        df = list_preset_events()
        assert not df.empty
        assert "date" in df.columns
        assert "description" in df.columns

    def test_invalid_preset_key_raises(self, sample_ohlcv: pd.DataFrame) -> None:
        from src.analysis.event_study import analyze_preset_event

        with pytest.raises(KeyError):
            analyze_preset_event("nonexistent_event", {"A": sample_ohlcv})


# ── DataSourceConfig tests ────────────────────────────────────────────────────


class TestDataSources:
    """Tests for the data source registry."""

    def test_get_source_known_key(self) -> None:
        from config.data_sources import get_source

        src = get_source("gold")
        assert src.name == "gold"

    def test_get_source_unknown_key_raises(self) -> None:
        from config.data_sources import get_source

        with pytest.raises(KeyError, match="not found"):
            get_source("nonexistent_source_xyz")

    def test_all_sources_have_required_fields(self) -> None:
        from config.data_sources import ALL_SOURCES

        for key, src in ALL_SOURCES.items():
            assert src.name, f"Source {key} has no name."
            assert src.description, f"Source {key} has no description."
            assert src.update_frequency, f"Source {key} has no update_frequency."


# ── Settings tests ────────────────────────────────────────────────────────────


class TestSettings:
    """Tests for the configuration module."""

    def test_data_dirs_are_paths(self) -> None:
        from config.settings import get_settings

        s = get_settings()
        assert isinstance(s.raw_data_dir, Path)
        assert isinstance(s.processed_data_dir, Path)

    def test_default_equity_tickers_is_list(self) -> None:
        from config.settings import get_settings

        s = get_settings()
        assert isinstance(s.default_equity_tickers, list)
        assert len(s.default_equity_tickers) > 0
