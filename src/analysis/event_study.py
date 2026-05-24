"""
Event study / impact analysis.

Provides tools to analyze how a set of assets reacted around a
specific historical event date (T0).  Results are expressed as
cumulative returns relative to T0 over a configurable window.

Pre-configured events:
- 2020-03-09: COVID-19 circuit-breaker (新冠熔断)
- 2022-02-24: Russia–Ukraine war start (俄乌冲突)
- 2022-03-16: US Fed begins rate-hike cycle (美联储开始加息)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class HistoricalEvent:
    """Metadata for a pre-configured historical event."""

    name: str
    date: str               # ISO-8601 date of T0
    description: str
    tags: List[str] = field(default_factory=list)


# ── Pre-configured events ─────────────────────────────────────────────────────

PRESET_EVENTS: Dict[str, HistoricalEvent] = {
    "covid_crash": HistoricalEvent(
        name="covid_crash",
        date="2020-03-09",
        description="COVID-19 circuit-breaker / global market sell-off (新冠熔断)",
        tags=["pandemic", "equity", "VIX"],
    ),
    "russia_ukraine": HistoricalEvent(
        name="russia_ukraine",
        date="2022-02-24",
        description="Russia invades Ukraine (俄乌冲突爆发)",
        tags=["geopolitics", "commodities", "oil"],
    ),
    "fed_hike_2022": HistoricalEvent(
        name="fed_hike_2022",
        date="2022-03-16",
        description="US Fed begins 2022 rate-hike cycle (美联储开始加息)",
        tags=["rates", "bonds", "USD"],
    ),
}


def analyze_event(
    event_date: str,
    price_data: Dict[str, pd.DataFrame],
    pre_window: int = 30,
    post_window: int = 90,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute cumulative returns around a single event date.

    For each asset, this function:
    1. Identifies the closest available trading day on or after T0.
    2. Sets that day's closing price as the base (return = 0 %).
    3. Computes cumulative returns for ``[T0 - pre_window, T0 + post_window]``.

    Args:
        event_date:  ISO-8601 string of the event date (T0).
        price_data:  Dict mapping asset name → OHLCV DataFrame with
                     DatetimeIndex.
        pre_window:  Number of calendar days to look back from T0.
        post_window: Number of calendar days to look forward from T0.
        price_col:   Price column used for return calculations.

    Returns:
        Wide DataFrame where each column is one asset's cumulative
        return (as a fraction, not %) and the index is the offset in
        calendar days from T0 (negative = before event).
    """
    event_ts = pd.Timestamp(event_date)
    start = event_ts - pd.Timedelta(days=pre_window)
    end = event_ts + pd.Timedelta(days=post_window)

    results: Dict[str, pd.Series] = {}

    for name, df in price_data.items():
        if df.empty:
            logger.warning(
                f"analyze_event: {name!r} data is empty – skipped."
            )
            continue

        if price_col not in df.columns:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                continue
            col = numeric_cols[0]
        else:
            col = price_col

        series = df[col].sort_index()
        window_data = series.loc[start:end].dropna()

        if window_data.empty:
            logger.warning(
                f"analyze_event: {name!r} has no data in window "
                f"[{start.date()} – {end.date()}]."
            )
            continue

        # Find the base price: closest date >= event_ts
        future_dates = window_data.index[window_data.index >= event_ts]
        if future_dates.empty:
            base_date = window_data.index[-1]
        else:
            base_date = future_dates[0]

        base_price = window_data.loc[base_date]
        if base_price == 0:
            logger.warning(
                f"analyze_event: {name!r} base price at {base_date.date()} is 0."
            )
            continue

        cumulative = (window_data / base_price) - 1.0
        # Reindex to day offsets for easy cross-asset comparison
        day_offset = (cumulative.index - event_ts).days
        cumulative.index = day_offset
        results[name] = cumulative

    if not results:
        return pd.DataFrame()

    result_df = pd.concat(results.values(), axis=1, keys=results.keys())
    result_df.index.name = "days_from_event"
    return result_df


def analyze_preset_event(
    event_key: str,
    price_data: Dict[str, pd.DataFrame],
    pre_window: int = 30,
    post_window: int = 90,
    price_col: str = "Close",
) -> tuple[HistoricalEvent, pd.DataFrame]:
    """Wrapper around :func:`analyze_event` using a pre-configured event.

    Args:
        event_key:   Key in :data:`PRESET_EVENTS`
                     (``"covid_crash"``, ``"russia_ukraine"``,
                     ``"fed_hike_2022"``).
        price_data:  Dict mapping asset name → OHLCV DataFrame.
        pre_window:  Days before T0 to include.
        post_window: Days after T0 to include.
        price_col:   Price column used for returns.

    Returns:
        Tuple of ``(HistoricalEvent, cumulative_returns_DataFrame)``.
    """
    if event_key not in PRESET_EVENTS:
        raise KeyError(
            f"Unknown preset event {event_key!r}. "
            f"Available: {list(PRESET_EVENTS.keys())}"
        )
    event = PRESET_EVENTS[event_key]
    df = analyze_event(
        event_date=event.date,
        price_data=price_data,
        pre_window=pre_window,
        post_window=post_window,
        price_col=price_col,
    )
    return event, df


def list_preset_events() -> pd.DataFrame:
    """Return a summary table of all pre-configured historical events.

    Returns:
        DataFrame with columns ``["name", "date", "description", "tags"]``.
    """
    rows = [
        {
            "name": e.name,
            "date": e.date,
            "description": e.description,
            "tags": ", ".join(e.tags),
        }
        for e in PRESET_EVENTS.values()
    ]
    return pd.DataFrame(rows)
