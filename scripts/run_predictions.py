"""
Unified prediction driver — Kalman-based 2-factor model.

Extracts direction + magnitude latent factors from 12 factor signals,
outputs {-1, 0, +1} signal with confidence and fuzzy magnitude.

Usage:
    python scripts/run_predictions.py --start-date 1976-01-01
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.backtest import ASSET_KEYS, MACRO_SPECS
from config.data_sources import ALL_SOURCES, DataSourceConfig
from src.analysis.factors import _compute_all_factors
from src.analysis.kalman import UnifiedPredictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="1976-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    return parser.parse_args()


def _build_fetcher(cfg: DataSourceConfig):
    mod = importlib.import_module(cfg.fetcher_module)
    cls = getattr(mod, cfg.fetcher_class)
    return cls(**cfg.fetch_kwargs)


def _safe_fetch(key: str, start: str, end: str) -> Optional[pd.DataFrame]:
    cfg = ALL_SOURCES.get(key)
    if cfg is None:
        return None
    try:
        fetcher = _build_fetcher(cfg)
        df = fetcher.fetch(start_date=start, end_date=end)
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def _series_from_df(df: pd.DataFrame, col: str = "close") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"])
    else:
        idx = pd.to_datetime(df.index)
    if col in df.columns:
        vals = df[col]
    else:
        numeric_cols = df.select_dtypes(include="number").columns
        vals = df[numeric_cols[0]] if len(numeric_cols) > 0 else pd.Series(dtype=float)
    return pd.Series(vals.values, index=idx, dtype=float).sort_index().dropna()


def main():
    args = parse_args()
    logger.info("=" * 64)
    logger.info("Unified Predictor — 2-Factor Kalman DFM")
    logger.info(f"  Period: {args.start_date} → {args.end_date}")
    logger.info(f"  Assets: {ASSET_KEYS}")
    logger.info(f"  Method: direction + magnitude latent factors")
    logger.info("=" * 64)

    asset_prices: Dict[str, pd.Series] = {}
    for key in ASSET_KEYS:
        df = _safe_fetch(key, args.start_date, args.end_date)
        px = _series_from_df(df)
        if not px.empty:
            asset_prices[key] = px
    logger.info(f"Loaded {len(asset_prices)}/{len(ASSET_KEYS)} asset price series.")

    macro_indicators: Dict[str, pd.Series] = {}
    yield_short = yield_long = risk_series = None
    m2_series = permit_series = epu_series = breakeven_series = None

    for key, _ in MACRO_SPECS.items():
        s = _series_from_df(_safe_fetch(key, args.start_date, args.end_date), col="value")
        if s.empty:
            continue
        if key == "us_treasury_2y": yield_short = s
        elif key == "us_treasury_10y": yield_long = s
        elif key == "us_nfci": risk_series = s
        elif key == "us_m2": m2_series = s
        elif key == "us_building_permits": permit_series = s
        elif key == "us_epu": epu_series = s
        elif key == "us_breakeven_10y": breakeven_series = s
        else: macro_indicators[key] = s

    raw_factors = _compute_all_factors(
        asset_prices, macro_indicators, yield_short, yield_long,
        risk_series, m2_series, permit_series, epu_series, breakeven_series,
    )
    factor_df = pd.DataFrame(raw_factors).sort_index().dropna(how="all")
    obs_array = factor_df.values.astype(np.float64)
    logger.info(f"Computed {len(factor_df.columns)} factor signals ({len(obs_array)} obs).")

    epu_pct = 50.0
    if epu_series is not None and not epu_series.empty:
        epu_annual = epu_series.resample("YE").mean()
        epu_pct = float((epu_annual.values < epu_annual.iloc[-1]).mean() * 100)
    logger.info(f"EPU percentile: {epu_pct:.1f}%")

    predictor = UnifiedPredictor()
    prediction = predictor.predict(obs_array, epu_percentile=epu_pct)

    fwd_pred = prediction.forward_pred
    fwd_mean = fwd_pred.state_mean[0, 0] if fwd_pred is not None else 0
    fwd_lo = fwd_pred.conf_lower[0, 0] if fwd_pred is not None else 0
    fwd_hi = fwd_pred.conf_upper[0, 0] if fwd_pred is not None else 0

    signal_symbol = {1: "▲ BULL", 0: "─ NEUTRAL", -1: "▼ BEAR"}

    logger.info(f"\n{'='*64}")
    logger.info("UNIFIED PREDICTION")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  Signal:       {signal_symbol[prediction.signal]} ({prediction.signal:+d})")
    logger.info(f"  Magnitude:    {prediction.magnitude:+.1%} (fuzzy)")
    logger.info(f"  Confidence:   {prediction.confidence:.1%}")
    logger.info(f"  EPU gate:     {'⚠️ NOISY — signal suppressed' if epu_pct > 75 else '✓ CLEAR — signal active'}")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  Latent factors:")
    logger.info(f"    Direction:  {prediction.direction_factor:+.4f}")
    logger.info(f"    Magnitude:  {prediction.magnitude_factor:+.4f}")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  Forward pred 1-step: {fwd_mean:+.4f}  [{fwd_lo:+.4f}, {fwd_hi:+.4f}]")
    if prediction.factor_loadings is not None and not prediction.factor_loadings.empty:
        logger.info(f"  {'─' * 40}")
        logger.info(f"  Factor loadings (|H| per factor):")
        ld = prediction.factor_loadings
        for i in range(min(len(ld), 8)):
            row_norm = np.linalg.norm(ld.iloc[i].values)
            logger.info(f"    {factor_df.columns[i]:28s} |H|={row_norm:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
