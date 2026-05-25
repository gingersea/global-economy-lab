"""
Phase-1 backtest configuration.

Single source of truth for asset universe, macro specifications, and
default regime weights.  Imported by both the driver script and the
regime-labels module — "adding an asset" only requires changes here.
"""

from __future__ import annotations

from typing import Dict


ASSET_KEYS = ["sp500", "gold", "crude_oil_wti", "dxy", "us_treasury_10y", "hang_seng"]

MACRO_SPECS: Dict[str, str] = {
    "us_pmi": "level",
    "us_cpi": "yoy",
    "us_unemployment": "diff",
    "us_treasury_2y": "level",
    "us_treasury_10y": "level",
    "us_nfci": "level",
    "us_industrial_production": "yoy",
    "us_m2": "level",
    "us_building_permits": "level",
    "us_breakeven_10y": "level",
    "us_epu": "level",
    "china_cpi": "level",
    "hang_seng": "level",
}

DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "recovery": {
        "sp500_ret": 0.55,
        "crude_oil_wti_ret": 0.20,
        "gold_ret": 0.10,
        "dxy_ret": 0.05,
        "us_treasury_10y_ret": 0.10,
    },
    "overheat": {
        "sp500_ret": 0.20,
        "crude_oil_wti_ret": 0.30,
        "gold_ret": 0.20,
        "dxy_ret": 0.20,
        "us_treasury_10y_ret": 0.10,
    },
    "stagflation": {
        "sp500_ret": 0.10,
        "crude_oil_wti_ret": 0.15,
        "gold_ret": 0.40,
        "dxy_ret": 0.25,
        "us_treasury_10y_ret": 0.10,
    },
    "recession": {
        "sp500_ret": 0.15,
        "crude_oil_wti_ret": 0.05,
        "gold_ret": 0.30,
        "dxy_ret": 0.20,
        "us_treasury_10y_ret": 0.30,
    },
    "unknown": {
        "sp500_ret": 0.20,
        "crude_oil_wti_ret": 0.20,
        "gold_ret": 0.20,
        "dxy_ret": 0.20,
        "us_treasury_10y_ret": 0.20,
    },
}

GROWTH_FALLBACK_COL = "us_industrial_production_yoy"
GROWTH_FALLBACK_THRESHOLD = 0.0
