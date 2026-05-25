"""
Unified multi-factor prediction driver.

Runs the factor ensemble once → generates daily + monthly predictions.
Monthly = integral of daily factor signals — inheriting higher accuracy.

Usage:
    python scripts/run_predictions.py --start-date 2008-01-01
    python scripts/run_predictions.py --start-date 2000-01-01  # extended history
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.backtest import ASSET_KEYS, MACRO_SPECS
from config.data_sources import ALL_SOURCES, DataSourceConfig
from src.analysis.prediction_hub import PredictionHub
from src.analysis.reliability import ReliabilityFilter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--output-dir", default="data/_meta/predictions")
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
    logger.info("Unified Multi-Factor Prediction Driver")
    logger.info(f"  Period: {args.start_date} → {args.end_date}")
    logger.info(f"  Assets: {ASSET_KEYS}")
    logger.info(f"  Principle: monthly = integral(daily factor signals)")
    logger.info("=" * 64)

    asset_prices: Dict[str, pd.Series] = {}
    for key in ASSET_KEYS:
        df = _safe_fetch(key, args.start_date, args.end_date)
        px = _series_from_df(df)
        if not px.empty:
            asset_prices[key] = px
    logger.info(f"Loaded {len(asset_prices)}/{len(ASSET_KEYS)} asset price series.")

    macro_indicators: Dict[str, pd.Series] = {}
    yield_short = None
    yield_long = None
    risk_series = None

    for key, transform in MACRO_SPECS.items():
        df = _safe_fetch(key, args.start_date, args.end_date)
        s = _series_from_df(df, col="value")
        if s.empty:
            continue
        if key == "us_treasury_2y":
            yield_short = s
        elif key == "us_treasury_10y":
            yield_long = s
        elif key in ("us_nfci",):
            risk_series = s
        else:
            macro_indicators[key] = s
    logger.info(f"Loaded {len(macro_indicators)} macro indicators.")

    fwd_returns = None
    if asset_prices:
        returns_list = []
        for px in asset_prices.values():
            if px is not None and not px.empty:
                r = px.pct_change(fill_method=None)
                if not r.empty:
                    returns_list.append(r)
        if returns_list:
            fwd_returns = pd.concat(returns_list, axis=1).mean(axis=1)
            logger.info(f"Computed equal-weighted forward returns ({len(fwd_returns)} obs).")

    hub = PredictionHub()
    result = hub.run(
        asset_prices=asset_prices,
        macro_indicators=macro_indicators,
        yield_short=yield_short,
        yield_long=yield_long,
        risk_series=risk_series,
        forward_returns=fwd_returns,
    )

    mp = result.macro_prediction
    dp = result.daily_prediction
    logger.info(f"\n{'='*64}")
    logger.info("Prediction Results (Kalman Filter / DFM)")

    # Reliability filter using EPU data
    epu_df = _safe_fetch("us_epu", args.start_date, args.end_date)
    epu_s = _series_from_df(epu_df, col="value")
    rf = ReliabilityFilter()
    rf.fit(
        epu_s if not epu_s.empty else pd.Series([0]),
        fwd_returns if fwd_returns is not None and not fwd_returns.empty else pd.Series([0]),
        risk_series if risk_series is not None and not risk_series.empty else pd.Series([0]),
    )
    rel = rf.assess(
        epu_value=float(epu_s.iloc[-1]) if not epu_s.empty else 0,
        vol_value=float(fwd_returns.tail(60).std()) if fwd_returns is not None and not fwd_returns.empty else 0,
    )
    reliability_tag = "✓ 可靠" if rel.is_reliable else "⚠️ 不可靠"
    logger.info(f"  Reliability: {reliability_tag} | noise={rel.noise_level} | driver={rel.primary_driver}")
    logger.info(f"  Expected accuracy: {rf.expected_accuracy(rel):.0%} (empirical)")
    logger.info(f"  EPU percentile: {rel.details.get('epu_percentile', '?')}%")
    logger.info(f"  KF confidence: {dp.confidence:.2%}")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  MONTHLY (integral of KF-filtered state):")
    logger.info(f"    Composite signal: {mp.composite_signal:+.4f}")
    logger.info(f"    KF confidence:    {mp.confidence:.2%}")
    logger.info(f"    Predicted regime: {mp.predicted_regime}")
    logger.info(f"    Regime probs:     {mp.regime_probabilities}")
    logger.info(f"    Forward pred 1mo: {mp.forward_pred_mean:+.4f}  [{mp.forward_pred_lower:+.4f}, {mp.forward_pred_upper:+.4f}]")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  DAILY (latest KF-filtered state):")
    logger.info(f"    Composite signal: {dp.composite_signal:+.4f}")
    logger.info(f"    KF confidence:    {dp.confidence:.2%}")
    logger.info(f"    Forward pred 1d:  {dp.forward_pred_mean:+.4f}  [{dp.forward_pred_lower:+.4f}, {dp.forward_pred_upper:+.4f}]")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  ASSET SIGNALS (daily):")
    for asset, sig in sorted(dp.asset_signals.items(), key=lambda x: -abs(x[1])):
        direction = "▲" if sig > 0.02 else ("▼" if sig < -0.02 else "─")
        logger.info(f"    {direction} {asset:20s} {sig:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
