"""
Dashboard data preparation.

Aggregates data from all fetchers and analysis modules into a
pre-computed bundle that the Streamlit dashboard or a Jupyter notebook
can consume without making additional network requests.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger


def build_dashboard_bundle(
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> dict:
    """Fetch and pre-compute all data required by the dashboard.

    This function tolerates partial failures: if any individual data
    source is unavailable, it is skipped and the rest of the bundle
    is still populated.

    Args:
        start_date: ISO-8601 start date for historical data.
        end_date:   ISO-8601 end date.  Defaults to today.
        use_cache:  Pass-through to all underlying fetchers.

    Returns:
        A dict with the following keys (any of which may be an empty
        DataFrame if the source was unavailable):

        * ``"equities"``   – Dict of OHLCV DataFrames (SP500, CSI300, HSI)
        * ``"gold"``       – Gold futures DataFrame
        * ``"vix"``        – VIX DataFrame
        * ``"us_10y"``     – US 10-year Treasury yield DataFrame
        * ``"us_cpi"``     – US CPI DataFrame
        * ``"us_pmi"``     – US PMI DataFrame
        * ``"cycle"``      – Cycle-position DataFrame (empty if PMI/CPI missing)
        * ``"corr_matrix"``– Correlation matrix DataFrame
        * ``"meta"``       – Dict with run metadata
    """
    end_date = end_date or str(date.today())
    bundle: dict = {
        "equities": {},
        "gold": pd.DataFrame(),
        "vix": pd.DataFrame(),
        "us_10y": pd.DataFrame(),
        "us_cpi": pd.DataFrame(),
        "us_pmi": pd.DataFrame(),
        "cycle": pd.DataFrame(),
        "corr_matrix": pd.DataFrame(),
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
            "built_at": str(date.today()),
        },
    }

    # ── Equities ──────────────────────────────────────────────────────────────
    from src.data_fetcher.equities import get_index_data

    for name, ticker in [
        ("SP500", "^GSPC"),
        ("CSI300", "000300.SH"),
        ("HSI", "^HSI"),
    ]:
        try:
            df = get_index_data(
                ticker, start_date=start_date, end_date=end_date, use_cache=use_cache
            )
            bundle["equities"][name] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Dashboard: failed to fetch {name}: {exc}")

    # ── Gold ──────────────────────────────────────────────────────────────────
    try:
        from src.data_fetcher.commodities import get_gold_price

        bundle["gold"] = get_gold_price(
            start_date=start_date, end_date=end_date, use_cache=use_cache
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dashboard: failed to fetch gold: {exc}")

    # ── VIX ───────────────────────────────────────────────────────────────────
    try:
        from src.data_fetcher.sentiment import get_vix

        bundle["vix"] = get_vix(
            start_date=start_date, end_date=end_date, use_cache=use_cache
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dashboard: failed to fetch VIX: {exc}")

    # ── US 10-year yield ──────────────────────────────────────────────────────
    try:
        from src.data_fetcher.bonds import get_us_treasury_yield

        bundle["us_10y"] = get_us_treasury_yield(
            "10y", start_date=start_date, end_date=end_date, use_cache=use_cache
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dashboard: failed to fetch US 10y yield: {exc}")

    # ── Macro: CPI & PMI ──────────────────────────────────────────────────────
    try:
        from src.data_fetcher.macro_economic import get_us_cpi, get_us_pmi

        bundle["us_cpi"] = get_us_cpi(
            start_date=start_date, end_date=end_date, use_cache=use_cache
        )
        bundle["us_pmi"] = get_us_pmi(
            start_date=start_date, end_date=end_date, use_cache=use_cache
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dashboard: failed to fetch macro data: {exc}")

    # ── Cycle position ────────────────────────────────────────────────────────
    if not bundle["us_pmi"].empty and not bundle["us_cpi"].empty:
        try:
            from src.analysis.cycle_position import get_cycle_position

            bundle["cycle"] = get_cycle_position(
                pmi=bundle["us_pmi"], cpi=bundle["us_cpi"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Dashboard: cycle position calculation failed: {exc}")

    # ── Correlation matrix ────────────────────────────────────────────────────
    price_data = dict(bundle["equities"])
    if not bundle["gold"].empty:
        price_data["Gold"] = bundle["gold"]
    if price_data:
        try:
            from src.analysis.correlation import compute_correlation_matrix

            bundle["corr_matrix"] = compute_correlation_matrix(price_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Dashboard: correlation matrix failed: {exc}")

    return bundle
