"""
Data fetcher sub-package.
"""

from src.data_fetcher.base import BaseFetcher
from src.data_fetcher.bonds import BondsFetcher, get_yield_curve
from src.data_fetcher.commodities import (
    CommoditiesFetcher,
    get_gold_price,
    get_silver_price,
    get_crude_oil_price,
)
from src.data_fetcher.currencies import CurrenciesFetcher, get_dxy, get_major_pairs
from src.data_fetcher.equities import EquitiesFetcher, get_index_data
from src.data_fetcher.macro_economic import (
    MacroEconomicFetcher,
    get_us_gdp,
    get_us_cpi,
    get_china_cpi,
    get_us_pmi,
    get_global_pmi,
    get_fed_funds_rate,
)
from src.data_fetcher.sentiment import SentimentFetcher, get_vix

__all__ = [
    "BaseFetcher",
    "BondsFetcher",
    "get_yield_curve",
    "CommoditiesFetcher",
    "get_gold_price",
    "get_silver_price",
    "get_crude_oil_price",
    "CurrenciesFetcher",
    "get_dxy",
    "get_major_pairs",
    "EquitiesFetcher",
    "get_index_data",
    "MacroEconomicFetcher",
    "get_us_gdp",
    "get_us_cpi",
    "get_china_cpi",
    "get_us_pmi",
    "get_global_pmi",
    "get_fed_funds_rate",
    "SentimentFetcher",
    "get_vix",
]
