"""
Global configuration for the Global Economy Lab project.

Reads settings from environment variables (via a .env file) and
exposes typed configuration objects to all other modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root resolution ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application-wide settings, resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    fred_api_key: str = Field(
        default="",
        description="FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)",
    )
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage API key")
    quandl_api_key: str = Field(default="", description="Quandl / Nasdaq Data Link key")

    # ── Data paths ───────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        description="Root directory for all local data storage",
    )

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw (unprocessed) data."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Path to cleaned / processed data cache."""
        return self.data_dir / "processed"

    # ── HTTP / retry settings ─────────────────────────────────────────────────
    request_timeout: int = Field(
        default=30, description="HTTP request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries on transient failures"
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Loguru log level")

    # ── Default asset lists ───────────────────────────────────────────────────
    default_equity_tickers: List[str] = Field(
        default=[
            "^GSPC",   # S&P 500
            "^NDX",    # NASDAQ-100
            "000300.SH",  # CSI 300 (沪深300)
            "^HSI",    # Hang Seng
            "^N225",   # Nikkei 225
        ],
        description="Default equity index tickers",
    )

    default_commodity_tickers: List[str] = Field(
        default=[
            "GC=F",   # Gold futures
            "SI=F",   # Silver futures
            "CL=F",   # WTI Crude Oil futures
        ],
        description="Default commodity tickers",
    )

    default_fx_tickers: List[str] = Field(
        default=[
            "DX-Y.NYB",  # US Dollar Index
            "EURUSD=X",
            "USDJPY=X",
            "USDCNY=X",
        ],
        description="Default FX tickers",
    )


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return _settings


# Module-level singleton
_settings = Settings()

# Ensure data directories exist when module is imported
_settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
