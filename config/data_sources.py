"""
Data source registry for the Global Economy Lab.

Each entry describes one data source: its name, description, update
frequency, the fetcher class / function responsible for it, and any
extra kwargs forwarded to that fetcher.

The registry now also carries:

- ``category`` – one of ``macro``, ``equity``, ``bond``, ``commodity``,
  ``fx``, ``sentiment``.
- ``region``   – ``US``, ``CN``, ``EU``, ``JP``, ``Global`` …
- ``backends`` – ordered list of backend identifiers tried by the fetcher
  (e.g. ``["fred_csv", "fredapi"]``).  Purely informational – the actual
  fallback chain is implemented inside each fetcher.
- ``prefers_api_key`` – ``True`` when an API key *improves* coverage / rate
  limits but is **not** required.  No source in this registry strictly
  requires a key any more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataSourceConfig:
    """Metadata for a single data source."""

    name: str
    description: str
    update_frequency: str  # e.g. "daily", "monthly", "quarterly"
    fetcher_module: str    # dotted import path to the module
    fetcher_class: str     # class name inside that module
    fetch_kwargs: Dict[str, Any] = field(default_factory=dict)
    category: str = "macro"
    region: str = "Global"
    backends: List[str] = field(default_factory=list)
    prefers_api_key: bool = False
    api_key_env_var: Optional[str] = None
    notes: str = ""
    publication_lag_days: int = 0

    # Back-compat alias – some early code reads ``requires_api_key``.
    @property
    def requires_api_key(self) -> bool:  # noqa: D401 – legacy alias
        """Deprecated alias for :attr:`prefers_api_key`.

        Returns ``False`` because none of the registered sources strictly
        require a key any longer (FRED CSV, Treasury CSV, ECB SDW, World
        Bank, akshare and yfinance are all key-less).
        """
        return False


# ── Macro-economic data sources ───────────────────────────────────────────────

US_GDP = DataSourceConfig(
    name="us_gdp",
    description="US Real GDP (quarterly), FRED series GDPC1",
    update_frequency="quarterly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "GDPC1"},
    category="macro",
    region="US",
    backends=["fred_csv", "fredapi"],
    prefers_api_key=False,
    api_key_env_var="FRED_API_KEY",
    publication_lag_days=30,
)

US_CPI = DataSourceConfig(
    name="us_cpi",
    description="US Consumer Price Index (monthly), FRED CPIAUCSL",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "CPIAUCSL"},
    category="macro",
    region="US",
    backends=["fred_csv", "fredapi"],
    publication_lag_days=15,
)

US_CORE_CPI = DataSourceConfig(
    name="us_core_cpi",
    description="US Core CPI (ex food & energy), FRED CPILFESL",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "CPILFESL"},
    category="macro",
    region="US",
    backends=["fred_csv", "fredapi"],
    publication_lag_days=15,
)

US_UNEMPLOYMENT = DataSourceConfig(
    name="us_unemployment",
    description="US Unemployment Rate, FRED UNRATE",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "UNRATE"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=7,
)

US_INDPRO = DataSourceConfig(
    name="us_industrial_production",
    description="US Industrial Production Index, FRED INDPRO",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "INDPRO"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=15,
)

US_PMI = DataSourceConfig(
    name="us_pmi_manufacturing",
    description="ISM Manufacturing PMI (US), FRED NAPM",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "NAPM"},
    category="macro",
    region="US",
    backends=["fred_csv", "fredapi"],
    notes="FRED series NAPM is the ISM Manufacturing PMI.",
    publication_lag_days=1,
)

US_FED_FUNDS_RATE = DataSourceConfig(
    name="us_fed_funds_rate",
    description="US Federal Funds Effective Rate, FRED FEDFUNDS",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "FEDFUNDS"},
    category="macro",
    region="US",
    backends=["fred_csv", "fredapi"],
    publication_lag_days=0,
)

US_NFCI = DataSourceConfig(
    name="us_nfci",
    description="Chicago Fed National Financial Conditions Index, FRED NFCI",
    update_frequency="weekly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "NFCI"},
    category="sentiment",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=7,
)

US_M2 = DataSourceConfig(
    name="us_m2",
    description="US M2 Money Supply, FRED M2SL (liquidity factor)",
    update_frequency="weekly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "M2SL"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=7,
)

US_PERMIT = DataSourceConfig(
    name="us_building_permits",
    description="US Building Permits, FRED PERMIT (leading indicator)",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "PERMIT"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=30,
)

US_BREAKEVEN_10Y = DataSourceConfig(
    name="us_breakeven_10y",
    description="10-Year Breakeven Inflation Rate, FRED T10YIE",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "T10YIE"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=0,
)

US_EPU = DataSourceConfig(
    name="us_epu",
    description="US Economic Policy Uncertainty Index, FRED USEPUINDXD",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "USEPUINDXD"},
    category="sentiment",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=0,
)

US_PROD = DataSourceConfig(
    name="us_labor_productivity",
    description="US Labor Productivity (nonfarm), FRED OPHNFB",
    update_frequency="quarterly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "OPHNFB"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=60,
)

US_DEFENSE_GDP = DataSourceConfig(
    name="us_defense_gdp",
    description="US Defense/GDP ratio, FRED A824RE1Q156NBEA",
    update_frequency="quarterly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "A824RE1Q156NBEA"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=90,
)

US_INVEST = DataSourceConfig(
    name="us_private_investment",
    description="US Real Private Investment, FRED GPDIC1",
    update_frequency="quarterly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"series_id": "GPDIC1"},
    category="macro",
    region="US",
    backends=["fred_csv"],
    publication_lag_days=90,
)

CHINA_CPI = DataSourceConfig(
    name="china_cpi",
    description="China CPI YoY (%) via akshare",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"source": "akshare", "indicator": "china_cpi"},
    category="macro",
    region="CN",
    backends=["akshare"],
)

# Global / cross-country macro via World Bank (no key, deep history).
WORLD_GDP_USD = DataSourceConfig(
    name="world_gdp_usd",
    description="World GDP, current US$ – World Bank WDI NY.GDP.MKTP.CD",
    update_frequency="annual",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"source": "worldbank", "series_id": "WLD:NY.GDP.MKTP.CD"},
    category="macro",
    region="Global",
    backends=["worldbank"],
)

# Euro-area HICP all-items from ECB SDW.
EA_HICP = DataSourceConfig(
    name="ea_hicp",
    description="Euro-area HICP all-items, ECB SDW ICP/M.U2.N.000000.4.ANR",
    update_frequency="monthly",
    fetcher_module="src.data_fetcher.macro_economic",
    fetcher_class="MacroEconomicFetcher",
    fetch_kwargs={"source": "ecb_sdw", "series_id": "ICP/M.U2.N.000000.4.ANR"},
    category="macro",
    region="EU",
    backends=["ecb_sdw"],
)


# ── Equity data sources ───────────────────────────────────────────────────────

SP500 = DataSourceConfig(
    name="sp500",
    description="S&P 500 Index daily price (^GSPC via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "^GSPC"},
    category="equity",
    region="US",
    backends=["yfinance", "stooq"],
)

CSI300 = DataSourceConfig(
    name="csi300",
    description="CSI 300 Index (沪深300) daily price via yfinance / akshare",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "000300.SH"},
    category="equity",
    region="CN",
    backends=["yfinance", "akshare"],
)

HANG_SENG = DataSourceConfig(
    name="hang_seng",
    description="Hang Seng Index daily price (^HSI via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "^HSI"},
    category="equity",
    region="HK",
    backends=["yfinance", "stooq"],
)

NIKKEI_225 = DataSourceConfig(
    name="nikkei_225",
    description="Nikkei 225 daily price (^N225 via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.equities",
    fetcher_class="EquitiesFetcher",
    fetch_kwargs={"ticker": "^N225"},
    category="equity",
    region="JP",
    backends=["yfinance", "stooq"],
)

# ── Bond data sources ─────────────────────────────────────────────────────────

US_TREASURY_2Y = DataSourceConfig(
    name="us_treasury_2y",
    description="US 2-Year Treasury Constant Maturity Rate (DGS2)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.bonds",
    fetcher_class="BondsFetcher",
    fetch_kwargs={"series_id": "DGS2"},
    category="bond",
    region="US",
    backends=["fred_csv", "treasury_gov", "fredapi"],
)

US_TREASURY_10Y = DataSourceConfig(
    name="us_treasury_10y",
    description="US 10-Year Treasury Constant Maturity Rate (DGS10)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.bonds",
    fetcher_class="BondsFetcher",
    fetch_kwargs={"series_id": "DGS10"},
    category="bond",
    region="US",
    backends=["fred_csv", "treasury_gov", "fredapi"],
)

US_TREASURY_30Y = DataSourceConfig(
    name="us_treasury_30y",
    description="US 30-Year Treasury Constant Maturity Rate (DGS30)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.bonds",
    fetcher_class="BondsFetcher",
    fetch_kwargs={"series_id": "DGS30"},
    category="bond",
    region="US",
    backends=["fred_csv", "treasury_gov", "fredapi"],
)


# ── Commodity data sources ────────────────────────────────────────────────────

GOLD = DataSourceConfig(
    name="gold",
    description="Gold spot price (GC=F via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.commodities",
    fetcher_class="CommoditiesFetcher",
    fetch_kwargs={"ticker": "GC=F"},
    category="commodity",
    region="Global",
    backends=["yfinance", "stooq"],
)

SILVER = DataSourceConfig(
    name="silver",
    description="Silver spot price (SI=F via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.commodities",
    fetcher_class="CommoditiesFetcher",
    fetch_kwargs={"ticker": "SI=F"},
    category="commodity",
    region="Global",
    backends=["yfinance", "stooq"],
)

CRUDE_OIL_WTI = DataSourceConfig(
    name="crude_oil_wti",
    description="WTI Crude Oil front-month futures (CL=F via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.commodities",
    fetcher_class="CommoditiesFetcher",
    fetch_kwargs={"ticker": "CL=F"},
    category="commodity",
    region="Global",
    backends=["yfinance", "stooq"],
)


# ── FX / Currency data sources ────────────────────────────────────────────────

DXY = DataSourceConfig(
    name="dxy",
    description="US Dollar Index (DX-Y.NYB via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.currencies",
    fetcher_class="CurrenciesFetcher",
    fetch_kwargs={"ticker": "DX-Y.NYB"},
    category="fx",
    region="US",
    backends=["yfinance", "stooq"],
)

EURUSD = DataSourceConfig(
    name="eurusd",
    description="EUR/USD exchange rate via yfinance",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.currencies",
    fetcher_class="CurrenciesFetcher",
    fetch_kwargs={"ticker": "EURUSD=X"},
    category="fx",
    region="EU",
    backends=["yfinance", "ecb_sdw", "stooq"],
)


# ── Sentiment data sources ────────────────────────────────────────────────────

VIX = DataSourceConfig(
    name="vix",
    description="CBOE Volatility Index (^VIX via yfinance)",
    update_frequency="daily",
    fetcher_module="src.data_fetcher.sentiment",
    fetcher_class="SentimentFetcher",
    fetch_kwargs={"ticker": "^VIX"},
    category="sentiment",
    region="US",
    backends=["yfinance", "stooq"],
)


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_SOURCES: Dict[str, DataSourceConfig] = {
    # Macro
    "us_gdp": US_GDP,
    "us_cpi": US_CPI,
    "us_core_cpi": US_CORE_CPI,
    "us_unemployment": US_UNEMPLOYMENT,
    "us_industrial_production": US_INDPRO,
    "us_pmi": US_PMI,
    "us_fed_funds_rate": US_FED_FUNDS_RATE,
    "us_nfci": US_NFCI,
    "china_cpi": CHINA_CPI,
    "world_gdp_usd": WORLD_GDP_USD,
    "ea_hicp": EA_HICP,
    "us_m2": US_M2,
    "us_building_permits": US_PERMIT,
    "us_breakeven_10y": US_BREAKEVEN_10Y,
    "us_epu": US_EPU,
    "us_labor_productivity": US_PROD,
    "us_defense_gdp": US_DEFENSE_GDP,
    "us_private_investment": US_INVEST,
    # Equities
    "sp500": SP500,
    "csi300": CSI300,
    "hang_seng": HANG_SENG,
    "nikkei_225": NIKKEI_225,
    # Bonds
    "us_treasury_2y": US_TREASURY_2Y,
    "us_treasury_10y": US_TREASURY_10Y,
    "us_treasury_30y": US_TREASURY_30Y,
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


def sources_by_category(category: str) -> Dict[str, DataSourceConfig]:
    """Return all sources whose ``category`` matches.

    Args:
        category: One of ``macro``, ``equity``, ``bond``, ``commodity``,
                  ``fx``, ``sentiment``.

    Returns:
        Mapping of registry key → :class:`DataSourceConfig`.
    """
    return {k: v for k, v in ALL_SOURCES.items() if v.category == category}


def sources_by_region(region: str) -> Dict[str, DataSourceConfig]:
    """Return all sources whose ``region`` matches the requested code."""
    return {k: v for k, v in ALL_SOURCES.items() if v.region == region}
