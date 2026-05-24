"""
Data source registry for the Global Economy Lab.

Each entry describes one data source: its name, description, update
frequency, the fetcher class / function responsible for it, and any
extra kwargs forwarded to that fetcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DataSourceConfig:
    """Metadata for a single data source."""

    name: str
    description: str
    update_frequency: str  # e.g. "daily", "monthly", "quarterly"
    fetcher_module: str    # dotted import path to the module
    fetcher_class: str     # class name inside that module
    fetch_kwargs: Dict[str, Any] = field(default_factory=dict)
    requires_api_key: bool = False
    api_key_env_var: Optional[str] = None
    notes: str = ""


# ── Macro-economic data sources ───────────────────────────────────────────────

US_GDP = DataSourceConfig(
    name="us_gdp",
    description="US Real GDP (quarterly), sourced from FRED series GDPC1",
    update_frequency="quarterly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "GDPC1"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
)

US_CPI = DataSourceConfig(
    name="us_cpi",
    description="US Consumer Price Index (monthly), FRED series CPIAUCSL",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "CPIAUCSL"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
)

US_CORE_CPI = DataSourceConfig(
    name="us_core_cpi",
    description="US Core CPI (excluding food & energy), FRED series CPILFESL",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "CPILFESL"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
)

US_PMI = DataSourceConfig(
    name="us_pmi_manufacturing",
    description="ISM Manufacturing PMI (US), FRED series MANEMP proxy or external",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "NAPM"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
    notes="FRED series NAPM is the ISM Manufacturing PMI.",
)

US_FED_FUNDS_RATE = DataSourceConfig(
    name="us_fed_funds_rate",
    description="US Federal Funds Effective Rate, FRED series FEDFUNDS",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "FEDFUNDS"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
)

CHINA_CPI = DataSourceConfig(
    name="china_cpi",
    description="China CPI YoY (%) via akshare",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"source": "akshare", "indicator": "china_cpi"},
    requires_api_key=False,
)

# ── Equity data sources ───────────────────────────────────────────────────────

SP500 = DataSourceConfig(
    name="sp500",
    description="S&P 500 Index daily price (^GSPC via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "^GSPC"},
)

CSI300 = DataSourceConfig(
    name="csi300",
    description="CSI 300 Index (沪深300) daily price via yfinance / akshare",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "000300.SH"},
)

HANG_SENG = DataSourceConfig(
    name="hang_seng",
    description="Hang Seng Index daily price (^HSI via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "^HSI"},
)

# ── Bond data sources ─────────────────────────────────────────────────────────

US_TREASURY_2Y = DataSourceConfig(
    name="us_treasury_2y",
    description="US 2-Year Treasury Constant Maturity Rate, FRED DGS2",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.bonds",
    fetcher_class="BondsFetcher",
    fetch_kwargs={"series_id": "DGS2"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
)

US_TREASURY_10Y = DataSourceConfig(
    name="us_treasury_10y",
    description="US 10-Year Treasury Constant Maturity Rate, FRED DGS10",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.bonds",
    fetcher_class="BondsFetcher",
    fetch_kwargs={"series_id": "DGS10"},
    requires_api_key=True,
    api_key_env_var="FRED_API_KEY",
)

# ── Commodity data sources ────────────────────────────────────────────────────

GOLD = DataSourceConfig(
    name="gold",
    description="Gold spot price (GC=F / XAUUSD via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.commodities",
    fetcher_class="CommoditiesFetcher",
    fetch_kwargs={"ticker": "GC=F"},
)

SILVER = DataSourceConfig(
    name="silver",
    description="Silver spot price (SI=F via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.commodities",
    fetcher_class="CommoditiesFetcher",
    fetch_kwargs={"ticker": "SI=F"},
)

CRUDE_OIL_WTI = DataSourceConfig(
    name="crude_oil_wti",
    description="WTI Crude Oil front-month futures (CL=F via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.commodities",
    fetcher_class="CommoditiesFetcher",
    fetch_kwargs={"ticker": "CL=F"},
)

# ── FX / Currency data sources ────────────────────────────────────────────────

DXY = DataSourceConfig(
    name="dxy",
    description="US Dollar Index (DX-Y.NYB via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.currencies",
    fetcher_class="CurrenciesFetcher",
    fetch_kwargs={"ticker": "DX-Y.NYB"},
)

EURUSD = DataSourceConfig(
    name="eurusd",
    description="EUR/USD exchange rate via yfinance",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.currencies",
    fetcher_class="CurrenciesFetcher",
    fetch_kwargs={"ticker": "EURUSD=X"},
)

# ── Sentiment data sources ────────────────────────────────────────────────────

VIX = DataSourceConfig(
    name="vix",
    description="CBOE Volatility Index (^VIX via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.sentiment",
    fetcher_class="SentimentFetcher",
    fetch_kwargs={"ticker": "^VIX"},
)

# ── Registry ──────────────────────────────────────────────────────────────────

ALL_SOURCES: Dict[str, DataSourceConfig] = {
    # Macro
    "us_gdp": US_GDP,
    "us_cpi": US_CPI,
    "us_core_cpi": US_CORE_CPI,
    "us_pmi": US_PMI,
    "us_fed_funds_rate": US_FED_FUNDS_RATE,
    "china_cpi": CHINA_CPI,
    # Equities
    "sp500": SP500,
    "csi300": CSI300,
    "hang_seng": HANG_SENG,
    # Bonds
    "us_treasury_2y": US_TREASURY_2Y,
    "us_treasury_10y": US_TREASURY_10Y,
    # Commodities
    "gold": GOLD,
    "silver": SILVER,
    "crude_oil_wti": CRUDE_OIL_WTI,
    # FX
    "dxy": DXY,
    "eurusd": EURUSD,
    # Sentiment
    "vix": VIX,
}


def get_source(name: str) -> DataSourceConfig:
    """Return a DataSourceConfig by its registry key.

    Args:
        name: Registry key of the data source.

    Returns:
        The matching DataSourceConfig.

    Raises:
        KeyError: If the name is not found in the registry.
    """
    if name not in ALL_SOURCES:
        raise KeyError(
            f"Data source '{name}' not found. "
            f"Available sources: {list(ALL_SOURCES.keys())}"
        )
    return ALL_SOURCES[name]
