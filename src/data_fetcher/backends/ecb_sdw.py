"""
ECB Statistical Data Warehouse (SDW) backend.

CSV download endpoint:
    https://sdw-wsrest.ecb.europa.eu/service/data/<FLOW>/<KEY>?format=csvdata

No API key required.  Series identifier format used by this backend:
``"<FLOW>/<KEY>"``, e.g. ``"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"`` for
the euro-area 10y AAA government bond yield, or
``"EXR/D.USD.EUR.SP00.A"`` for daily EUR/USD reference rates.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
from loguru import logger

from src.data_fetcher.backends import BackendError
from src.data_fetcher.backends._http import http_get


def fetch_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch an ECB SDW time series via the public CSV endpoint.

    Args:
        series_id:  ``"<FLOW>/<KEY>"`` as documented above.
        start_date: Optional ISO-8601 lower bound.
        end_date:   Optional ISO-8601 upper bound.

    Returns:
        Standardised DataFrame ``["value", "series_id", "source"]``.
    """
    if "/" not in series_id:
        raise BackendError(
            f"ecb_sdw: expected 'FLOW/KEY' format, got {series_id!r}"
        )
    flow, key = series_id.split("/", 1)
    url = f"https://sdw-wsrest.ecb.europa.eu/service/data/{flow}/{key}"
    params = {"format": "csvdata"}
    if start_date:
        params["startPeriod"] = start_date
    if end_date:
        params["endPeriod"] = end_date

    resp = http_get(url, params=params, headers={"Accept": "text/csv"})
    text = resp.text
    if not text or "TIME_PERIOD" not in text.split("\n", 1)[0]:
        raise BackendError(
            f"ecb_sdw: unexpected response for {series_id}: {text[:120]!r}"
        )
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"ecb_sdw: CSV parse failed: {exc}") from exc

    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise BackendError(
            f"ecb_sdw: missing required columns; got {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    out = pd.DataFrame(
        {
            "value": pd.to_numeric(df["OBS_VALUE"], errors="coerce").astype(
                "float64"
            ),
            "series_id": series_id,
            "source": "ecb_sdw",
        }
    )
    out = out.dropna(subset=["value"])
    logger.debug(f"ecb_sdw: fetched {len(out)} rows for {series_id}")
    return out
