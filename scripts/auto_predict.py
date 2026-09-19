#!/usr/bin/env python3
"""
Global Economy Lab — Automated Prediction Pipeline
===================================================
Modes:
  data-only  : Update data + run predictions (JSON only, no HTML, no deploy)
  summary    : Generate weekly summary HTML + archive (Friday)
  full       : Full pipeline: data → predict → HTML → archive (Saturday)

All output archived to output/archive/YYYY-MM-DD/, never deleted.

2026W28 improvements (July 2026):
  - VIX daily-frequency proxy for EPU (^VIX from yfinance)
  - Per-group EPU thresholds (EM P65 vs DM P75)
  - Overheat downgrade rules (z > +2.5σ → NEUT, z > +3.5σ → BEAR)
  - Auto ISO week calculation (Sat/Sun → next week)
  - actual_accuracy field tracking directional hit rate vs prior prediction
  - Dynamic end_date (today, not hardcoded)
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.high_confidence import (
    HighConfidencePredictor,
    EMERGING_MARKETS,
    DEVELOPED_MARKETS,
    GROUP_EPU_THRESHOLDS,
)
from src.data_fetcher.equities import EquitiesFetcher
from src.data_fetcher.macro_economic import MacroEconomicFetcher

# ── Market definitions ───────────────────────────────────────
MARKET_TICKERS = {
    "US": ("^GSPC", "美国 SP500"),
    "DE": ("^GDAXI", "德国 DAX"),
    "JP": ("^N225", "日经 225"),
    "GB": ("^FTSE", "英国 FTSE"),
    "FR": ("^FCHI", "法国 CAC40"),
    "IT": ("FTSEMIB.MI", "意大利 MIB"),
    "CA": ("^GSPTSE", "加拿大 TSX"),
    "BR": ("^BVSP", "巴西 Bovespa"),
    "KR": ("^KS11", "韩国 KOSPI"),
    "IN": ("^NSEI", "印度 NIFTY"),
    "AU": ("^AXJO", "澳洲 ASX"),
    "CN": ("000001.SS", "中国 A股"),
    "HK": ("^HSI", "恒生指数"),
}

# ── Market-specific parameters ─────────────────────────────
# HK: China EPU (verified +10pp improvement vs US EPU, now 77%).
#     AR(1) 0.60 → 0.75 (2026W32): preserve HSI's strong trend signal
#     (July 2026: +12%/mo rally that the 0.60 decay dampened).
# CN: REVERTED to US EPU — China EPU caused -10pp regression (65→55%).
#     A-shares dominated by retail flow/policy directives, not EPU.
#     Higher signal threshold + lower AR to reduce false signals.
# IN: only 19y data, 0 normal EPU years → single-model fallback.
# BR: high volatility, uses faster AR(1) decay.
MARKET_PARAMS = {
    "HK": {"ar_coeff": 0.75, "signal_threshold": 0.05, "epu_label": "china", "min_confidence": 0.50},
    "CN": {"ar_coeff": 0.55, "signal_threshold": 0.06, "epu_label": "us"},
    "BR": {"ar_coeff": 0.65, "signal_threshold": 0.05, "epu_label": "us"},
    "IN": {"ar_coeff": 0.65, "signal_threshold": 0.03, "epu_label": "us"},
    # GB (英国 FTSE): 月度回测 ~51%, below the global 0.55 bar but above
    # random — publish as low-confidence reference.
    "GB": {"min_confidence": 0.50},
}

TIER_LABELS = {1: "★★★ 可操作", 2: "★★ 参考(1年)", 3: "☆ 不可用"}


# ── ISO week calculation ─────────────────────────────────────
def iso_week_for_prediction(d: date = None):
    """Return (year, week_num, monday, sunday) for the prediction period.

    Rules:
      - Mon-Fri: predict CURRENT ISO week
      - Saturday: predict NEXT ISO week (current week is finishing)
      - Sunday: predict NEXT ISO week (new week starts tomorrow)

    This ensures predictions are always forward-looking.
    """
    if d is None:
        d = date.today()

    iso = d.isocalendar()
    weekday = d.weekday()  # 0=Mon ... 6=Sun

    if weekday >= 5:  # Saturday (5) or Sunday (6)
        # Advance to next Monday, then get its ISO week
        days_to_monday = 7 - weekday
        next_monday = d + timedelta(days=days_to_monday)
        iso = next_monday.isocalendar()

    year, week = iso[0], iso[1]
    # Compute Monday of the target ISO week
    # ISO week 1 is the week containing Jan 4
    jan4 = date(year, 1, 4)
    jan4_iso = jan4.isocalendar()
    jan4_monday = jan4 - timedelta(days=jan4.weekday())
    monday = jan4_monday + timedelta(weeks=week - 1)
    sunday = monday + timedelta(days=6)

    return year, week, monday, sunday


# ── Data loading ─────────────────────────────────────────────
def load_prices(today: date = None):
    """Load price data for all markets.  Uses dynamic end_date."""
    if today is None:
        today = date.today()
    end_str = today.strftime("%Y-%m-%d")

    prices = {}
    for name, (ticker, _) in MARKET_TICKERS.items():
        try:
            f = EquitiesFetcher(ticker=ticker)
            df = f.fetch(start_date="1985-01-01", end_date=end_str)
            if df is not None and not df.empty:
                idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(
                    df["date"] if "date" in df.columns else df.index
                )
                px = df["close"] if "close" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
                prices[name] = pd.Series(px.values, index=idx, dtype=float).sort_index()
        except Exception as e:
            logger.warning(f"  {name} ({ticker}): {e}")
    return prices


def load_ohlc(today: date = None):
    """Load OHLCV frames for all markets (2026W34 intraday factors).

    The EquitiesFetcher parquet cache is keyed on (start, end), so this
    second pass re-reads the same cached file — no extra network cost.
    Returns market → DataFrame with uppercase OHLC columns.
    """
    if today is None:
        today = date.today()
    end_str = today.strftime("%Y-%m-%d")

    ohlc = {}
    for name, (ticker, _) in MARKET_TICKERS.items():
        try:
            f = EquitiesFetcher(ticker=ticker)
            df = f.fetch(start_date="1985-01-01", end_date=end_str)
            if df is not None and not df.empty:
                idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(
                    df["date"] if "date" in df.columns else df.index
                )
                cols = {c: c for c in ("Open", "High", "Low", "Close") if c in df.columns}
                if len(cols) >= 3:
                    ohlc[name] = df[list(cols)].copy()
                    ohlc[name].index = idx
                    ohlc[name] = ohlc[name].sort_index()
        except Exception as e:
            logger.warning(f"  {name} ({ticker}) OHLC: {e}")
    return ohlc


def load_epu():
    """Load US EPU from processed CSV."""
    epu_path = _PROJECT_ROOT / "data/processed/epu_historical_monthly.csv"
    if not epu_path.exists():
        logger.warning("EPU data missing, using default")
        return pd.Series([100.0], dtype=float)
    epu = pd.read_csv(epu_path, index_col=0, parse_dates=True)["value"]
    return epu.resample("YE").mean()


def load_china_epu():
    """Load China EPU (CHNMAINLANDEPU) from FRED."""
    try:
        f = MacroEconomicFetcher(series_id="CHNMAINLANDEPU")
        df = f.fetch(start_date="1985-01-01", end_date=date.today().strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            epu = df["value"] if "value" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
            return pd.Series(epu.values, index=pd.to_datetime(df.index), dtype=float).resample("YE").mean()
    except Exception as e:
        logger.warning(f"China EPU load failed: {e}")
    return pd.Series(dtype=float)


def load_vix():
    """Load VIX data — primary from FRED VIXCLS, fallback to yfinance.

    Returns VIX daily series (pd.Series of closing values), or None on failure.
    VIXCLS is the official CBOE VIX daily series published by FRED.
    """
    # ── Primary: FRED VIXCLS ──────────────────────────────────────────────
    try:
        f = MacroEconomicFetcher(series_id="VIXCLS")
        df = f.fetch(start_date="1990-01-01", end_date=date.today().strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            vix_series = df["value"] if "value" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
            vix_series = pd.Series(vix_series.values, index=pd.to_datetime(df.index), dtype=float).dropna()
            if len(vix_series) >= 252:
                logger.info(f"VIX loaded from FRED VIXCLS: {len(vix_series)} daily obs, latest={vix_series.iloc[-1]:.1f}")
                return vix_series
            else:
                logger.warning(f"VIXCLS insufficient history ({len(vix_series)} < 252)")
    except Exception as e:
        logger.warning(f"VIXCLS load failed: {e}")

    # ── Fallback: yfinance ─────────────────────────────────────────────────
    try:
        vix = yf.Ticker("^VIX")
        df = vix.history(period="max")
        if df is None or df.empty:
            logger.warning("VIX (yfinance fallback): no data returned")
            return None
        close = df["Close"].dropna()
        if len(close) < 252:
            logger.warning(f"VIX (yfinance fallback): insufficient history ({len(close)} < 252)")
            return None
        logger.info(f"VIX loaded from yfinance: {len(close)} daily obs, latest={close.iloc[-1]:.1f}")
        return close
    except Exception as e:
        logger.warning(f"VIX (yfinance fallback) load failed: {e}")
        return None


def load_raw_epu_monthly():
    """Load raw monthly EPU (not annualized) for EPU percentile calculation.

    Returns the actual monthly EPU values used for accurate percentile.
    """
    epu_path = _PROJECT_ROOT / "data/processed/epu_historical_monthly.csv"
    if not epu_path.exists():
        return pd.Series([100.0], dtype=float)
    epu = pd.read_csv(epu_path, index_col=0, parse_dates=True)["value"]
    return epu


# ── Actual accuracy tracking ─────────────────────────────────
def compute_actual_accuracy(predictions, prior_json_path: Path, today: date):
    """Compare prior prediction direction with actual returns.

    Args:
        predictions:     Current prediction objects.
        prior_json_path: Path to prior week's JSON.
        today:          Today's date for computing actual returns.

    Returns:
        Dict with 'hit_rate', 'hits', 'misses', 'details' per market.
    """
    if not prior_json_path.exists():
        logger.info("No prior prediction JSON — skipping actual_accuracy")
        return {"hit_rate": None, "hits": 0, "misses": 0, "markets": {}}

    try:
        with open(prior_json_path) as f:
            prior = json.load(f)
    except Exception as e:
        logger.warning(f"Cannot read prior JSON: {e}")
        return {"hit_rate": None, "hits": 0, "misses": 0, "markets": {}}

    prior_markets = {m["market"]: m for m in prior.get("markets", [])}
    if not prior_markets:
        logger.info("Prior JSON has no market predictions")
        return {"hit_rate": None, "hits": 0, "misses": 0, "markets": {}}

    # Parse prediction period from prior JSON to get start date
    prior_period = prior.get("prediction_period", "")
    # Try to parse "2026年第27周 (6/29-7/5)"
    import re
    m = re.search(r'\((\d+)/(\d+)-(\d+)/(\d+)\)', prior_period)
    if m:
        # Extract month/day ranges (handle single-digit)
        m1, d1 = int(m.group(1)), int(m.group(2))
        m2, d2 = int(m.group(3)), int(m.group(4))
        year = prior.get("generated", "2026")[:4]
        start_date = date(int(year), m1, d1)
        prior_end_date = date(int(year), m2, d2)
    else:
        # Fallback (阶段2 月频): use ~1 calendar month (~21 trading days) ago
        start_date = today - timedelta(days=31)
        prior_end_date = today

    logger.info(f"Actual accuracy: prior period ~{start_date} → ~{prior_end_date} (today={today})")

    hits = 0
    misses = 0
    market_details = {}

    for mkt_code, ticker_info in MARKET_TICKERS.items():
        if mkt_code not in prior_markets:
            continue
        prior_pred = prior_markets[mkt_code]
        prior_signal = prior_pred.get("signal", "NEUT")

        # Fetch actual return over the prediction period
        try:
            ticker = ticker_info[0]
            eq = EquitiesFetcher(ticker=ticker)
            df = eq.fetch(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=today.strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                px = df["close"] if "close" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
                if isinstance(df.index, pd.DatetimeIndex):
                    px.index = df.index
                else:
                    px.index = pd.to_datetime(df["date"] if "date" in df.columns else df.index)

                px = px.sort_index()
                if len(px) >= 2:
                    actual_ret = float((px.iloc[-1] / px.iloc[0]) - 1)
                    actual_dir = 1 if actual_ret > 0.005 else (-1 if actual_ret < -0.005 else 0)
                    prior_dir = 1 if prior_signal == "BULL" else (-1 if prior_signal == "BEAR" else 0)

                    if prior_dir == 0:
                        hit = None  # neutral prediction, no score
                    elif actual_dir == prior_dir:
                        hits += 1
                        hit = True
                    else:
                        misses += 1
                        hit = False

                    market_details[mkt_code] = {
                        "prior_signal": prior_signal,
                        "actual_ret": round(actual_ret, 4),
                        "actual_dir": "UP" if actual_dir > 0 else ("DOWN" if actual_dir < 0 else "FLAT"),
                        "hit": hit,
                    }
        except Exception as e:
            logger.debug(f"  actual_accuracy {mkt_code}: {e}")
            market_details[mkt_code] = {"prior_signal": prior_signal, "error": str(e)}

    total = hits + misses
    hit_rate = round(hits / total, 4) if total > 0 else None

    logger.info(f"Actual accuracy: {hits}/{total} = {hit_rate:.1%}" if hit_rate is not None else "No directional predictions")

    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
        "total_scored": total,
        "period_start": start_date.strftime("%Y-%m-%d"),
        "period_end": today.strftime("%Y-%m-%d"),
        "markets": market_details,
    }


# ── Prediction ───────────────────────────────────────────────
def run_predictions(vix_series=None):
    """Returns (predictions_list, epu_values, is_high_epu, vix_info, epu_percentile, predictor)."""
    prices = load_prices()
    epu_a = load_epu()
    china_epu_a = load_china_epu()
    raw_epu_monthly = load_raw_epu_monthly()

    epu_now = float(epu_a.iloc[-1]) if len(epu_a) > 0 else 100.0
    china_epu_now = float(china_epu_a.iloc[-1]) if len(china_epu_a) > 0 else epu_now

    # Accurate EPU percentile from monthly data (not random noise)
    raw_epu_vals = raw_epu_monthly.dropna().values
    epu_percentile = float(
        (raw_epu_vals < epu_now).mean() * 100
    ) if len(raw_epu_vals) > 0 else 50.0

    logger.info(f"Loaded {len(prices)} markets, US EPU={epu_now:.0f}, China EPU={china_epu_now:.0f}")

    # Initialize predictor with per-group thresholds
    predictor = HighConfidencePredictor(
        min_confidence=0.55,
        group_thresholds=GROUP_EPU_THRESHOLDS,
    )
    predictor.fit_epu(epu_a, label="us")
    if len(china_epu_a) > 0:
        predictor.fit_epu(china_epu_a, label="china")

    # Fit VIX if available
    vix_info = None
    if vix_series is not None and len(vix_series) > 0:
        predictor.fit_vix(vix_series)
        vix_info = {
            "current": predictor._vix_now,
            "p80": predictor._vix_p80,
            "p20": predictor._vix_p20,
            "trend_5d": predictor._vix_trend_5d,
            "trend_20d": predictor._vix_trend_20d,
            "active": predictor._vix_active,
        }

    # Load economic cycle position
    cycle_phase = "unknown"
    cycle_label = "Unknown 未知"
    cycle_pmi = None
    cycle_cpi_yoy = None
    try:
        from src.data_fetcher.macro_economic import get_us_pmi, get_us_cpi, get_fed_funds_rate
        from src.analysis.cycle_position import get_current_phase
        pmi = get_us_pmi()
        cpi = get_us_cpi()
        ffr = get_fed_funds_rate()
        cycle = get_current_phase(pmi, cpi, ffr)
        cycle_phase = cycle.get("phase", "unknown")
        cycle_label = cycle.get("phase_label", "Unknown 未知")
        cycle_pmi = cycle.get("pmi")
        cycle_cpi_yoy = cycle.get("cpi_yoy")
        logger.info(
            f"Economic cycle: {cycle_label} "
            f"(PMI={cycle_pmi}, CPI YoY={cycle_cpi_yoy}%)"
        )
    except Exception as e:
        logger.warning(f"Cycle position skipped: {e}")

    # Set per-market parameters
    for market, params in MARKET_PARAMS.items():
        predictor.set_market_params(market, **params)

    # Fit per-market models
    for market, px in sorted(prices.items()):
        try:
            epu_label = MARKET_PARAMS.get(market, {}).get("epu_label", "us")
            epu_data = china_epu_a if epu_label == "china" else epu_a
            result = predictor.fit_market(market, px, epu_data, epu_label=epu_label)
            if result:
                ar = MARKET_PARAMS.get(market, {}).get("ar_coeff", 0.75)
                group = "EM" if market in EMERGING_MARKETS else "DM"
                epu_pct = result.get("group_pct", predictor.epu_threshold_pct)
                logger.info(
                    f"  {market:6s} [{group}]: acc={result['accuracy']:.0%} "
                    f"(n={result['n_years']}m) EPU={epu_label} P{epu_pct:.0f} AR(1)={ar:.2f}"
                )
        except Exception as e:
            logger.warning(f"  {market}: fit failed — {e}")

    predictions = predictor.predict_all(
        prices, epu_now,
        epu_values={"us": epu_now, "china": china_epu_now},
        cycle_phase=cycle_phase,
    )

    # Compute rolling 5-year backtest for each market
    for p in predictions:
        px = prices.get(p.market)
        if px is not None:
            try:
                epu_label = MARKET_PARAMS.get(p.market, {}).get("epu_label", "us")
                epu_data = china_epu_a if epu_label == "china" else epu_a
                roll_acc = predictor.compute_rolling_backtest(
                    p.market, px, epu_data, window_years=5
                )
                p.rolling_accuracy = roll_acc
            except Exception as e:
                logger.debug(f"  rolling_accuracy {p.market}: {e}")

    # Top-level regime label reflects the EPU regime that drives model
    # selection (2026W34 fix).  VIX state is surfaced separately in the JSON
    # "vix" field.
    is_high = predictor._is_high_epu(epu_now, "us")

    return predictions, {"us": epu_now, "china": china_epu_now}, is_high, vix_info, epu_percentile, predictor, {
        "cycle_phase": cycle_phase,
        "cycle_label": cycle_label,
        "cycle_pmi": cycle_pmi,
        "cycle_cpi_yoy": cycle_cpi_yoy,
    }


# ── JSON output ──────────────────────────────────────────────
def build_json(predictions, epu_values, is_high_epu, gen_time, week_info,
               actual_accuracy=None, vix_info=None, epu_percentile=50.0, predictor=None,
               cycle_info=None):
    """Convert predictions to structured JSON with all metadata."""
    epu_val = epu_values["us"]
    china_epu_val = epu_values.get("china", epu_val)

    tier1 = [p for p in predictions if p.tier == 1]
    tier2 = [p for p in predictions if p.tier == 2]

    def _market(p):
        _, cn_name = MARKET_TICKERS.get(p.market, (p.market, p.market))
        last_date = ""
        if hasattr(p, 'last_date'):
            last_date = str(p.last_date) if p.last_date else ""
        elif hasattr(p, 'price_date'):
            last_date = str(p.price_date) if p.price_date else ""

        flat_market = getattr(p, 'flat_market', False)
        low_confidence = getattr(p, 'low_confidence', False)
        rolling_acc = getattr(p, 'rolling_accuracy', None)
        group = "EM" if p.market in EMERGING_MARKETS else "DM"

        return {
            "market": p.market,
            "name_cn": cn_name,
            "signal": "BULL" if p.signal > 0 else ("BEAR" if p.signal < 0 else "NEUT"),
            "pred_ret": round(float(p.expected_ret), 4),
            "accuracy": round(float(p.confidence), 4),
            "tier": p.tier,
            "tm_z": round(float(p.factors.get("tm_z", 0)), 2),
            "mr_z": round(float(p.factors.get("mr_z", 0)), 2),
            "vr_z": round(float(p.factors.get("vr_z", 0)), 2),
            "detail": getattr(p, 'detail', ''),
            "regime": getattr(p, 'regime', 'normal'),
            "last_date": last_date,
            "flat_market": flat_market,
            "low_confidence": low_confidence,
            "rolling_accuracy": rolling_acc,
            "group": group,
        }

    # Count signals
    bulls = sum(1 for p in predictions if p.tier in (1, 2) and p.signal > 0)
    bears = sum(1 for p in predictions if p.tier in (1, 2) and p.signal < 0)
    neuts = sum(1 for p in predictions if p.tier in (1, 2) and p.signal == 0)

    regime_shifted = sum(1 for p in predictions if getattr(p, 'regime_shift', False))
    regime_warned = sum(1 for p in predictions if getattr(p, 'regime_unfamiliar', False))

    result = {
        "generated": gen_time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "generated_display": gen_time.strftime("%Y-%m-%d %H:%M CST"),
        "prediction_period": f"{gen_time.year}年{gen_time.month}月",
        "epu": {
            "value": round(epu_val, 1),
            "regime": "HIGH_EPU" if is_high_epu else "NORMAL",
            "percentile": round(epu_percentile, 1),
            "china_epu": round(china_epu_val, 1) if china_epu_val != epu_val else None,
        },
        "vix": {
            "current": vix_info["current"] if vix_info else None,
            "p80": vix_info["p80"] if vix_info else None,
            "p20": vix_info["p20"] if vix_info else None,
            "trend_5d": round(vix_info["trend_5d"], 4) if vix_info and vix_info.get("trend_5d") is not None else None,
            "trend_20d": round(vix_info["trend_20d"], 4) if vix_info and vix_info.get("trend_20d") is not None else None,
        },
        "cycle": {
            "phase": cycle_info["cycle_phase"] if cycle_info else "unknown",
            "label": cycle_info["cycle_label"] if cycle_info else "Unknown 未知",
            "pmi": cycle_info["cycle_pmi"] if cycle_info else None,
            "cpi_yoy": cycle_info["cycle_cpi_yoy"] if cycle_info else None,
        },
        "tier1_count": len(tier1),
        "tier2_count": len(tier2),
        "signal_distribution": {"BULL": bulls, "BEAR": bears, "NEUT": neuts},
        "regime_shifted_count": regime_shifted,
        "regime_warned_count": regime_warned,
        "actual_accuracy": actual_accuracy,
        "markets": [_market(p) for p in predictions if p.tier in (1, 2)],
    }
    return result


# ── HTML generation ──────────────────────────────────────────
CSS = """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e0;font-family:'Inter',-apple-system,sans-serif;min-height:100vh}
nav{background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);border-bottom:1px solid rgba(255,255,255,0.08);padding:16px 32px;position:sticky;top:0;z-index:100}
nav .inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:28px}
nav a{color:#888;text-decoration:none;font-size:14px;transition:color 0.2s}
nav a:hover,nav a.active{color:#fff}
nav .home{font-size:18px;font-weight:700;color:#f7971e!important}
.container{max-width:1100px;margin:0 auto;padding:32px 24px 80px}
.page-header{text-align:center;padding:40px 0 32px}
.page-header .icon{font-size:42px;margin-bottom:12px}
.page-header h1{font-size:32px;font-weight:800;margin-bottom:8px}
.page-header .sub{color:#666;font-size:15px}
footer{text-align:center;padding:32px 0;color:#444;font-size:12px;border-top:1px solid rgba(255,255,255,0.05);margin-top:40px}
.card{background:rgba(255,255,255,0.03);border-radius:16px;padding:28px;margin-bottom:24px;border:1px solid rgba(255,255,255,0.06)}
.back-link{display:inline-block;color:#888;text-decoration:none;font-size:13px;margin-bottom:16px}
.back-link:hover{color:#fff}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:24px}
.metric{background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;text-align:center}
.metric .val{font-size:28px;font-weight:800;margin-bottom:4px}
.metric .lbl{font-size:12px;color:#666;text-transform:uppercase}
.predict-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:16px}
.predict-table th{color:#888;font-size:10px;text-transform:uppercase;text-align:left;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)}
.predict-table td{padding:10px 10px;border-bottom:1px solid rgba(255,255,255,0.04)}
.predict-table tr:hover td{background:rgba(255,255,255,0.02)}
.signal-bull{color:#4caf50;font-weight:700}
.signal-bear{color:#f44336;font-weight:700}
.signal-neut{color:#ff9800;font-weight:700}
.overheat-warn{color:#ff6d00;font-weight:700;font-size:10px;vertical-align:middle;margin-left:4px}
.signal-overheat{color:#ff6d00;font-weight:700}
.tier-badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;margin-right:4px}
.tier1{background:rgba(76,175,80,0.15);color:#4caf50}
.tier2{background:rgba(255,152,0,0.15);color:#ff9800}
.change-up{color:#4caf50}
.change-down{color:#f44336}
.insight-box{margin-top:16px;padding:20px;background:rgba(255,255,255,0.04);border-radius:12px;border-left:3px solid #f7971e}
.insight-box p{margin:0;font-size:14px;color:#aaa;line-height:1.8}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){.grid-2{grid-template-columns:1fr}}
.ts-stamp{display:inline-block;background:rgba(255,255,255,0.05);border-radius:6px;padding:3px 8px;font-size:11px;color:#666;font-family:monospace}
.hk-section{background:linear-gradient(135deg,rgba(247,151,30,0.08),rgba(247,151,30,0.02));border:1px solid rgba(247,151,30,0.2);border-radius:16px;padding:24px;margin-bottom:24px}
.hk-section h3{color:#f7971e;font-size:18px;margin-bottom:16px}
.group-badge{display:inline-block;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600;margin-left:4px}
.group-em{background:rgba(255,152,0,0.15);color:#ff9800}
.group-dm{background:rgba(33,150,243,0.15);color:#2196f3}
</style>"""


def build_nav():
    return """<nav>
    <div class="inner">
        <a href="/" class="home">🦊 GingerFamily</a>
        <a href="/tech.html">⏱ 时频技术</a>
        <a href="/ai.html">🤖 AI 前沿</a>
        <a href="/sci.html">🔬 科学探索</a>
        <a href="/market.html">📈 财经情报</a>
        <a href="/prediction.html">🔮 全球预测</a>
    </div>
</nav>"""


def build_page(title, body, extra_css=""):
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · GingerFamily.CN</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
{CSS}{extra_css}</head><body>
{build_nav()}
{body}
<footer>GingerFamily.CN · Global Economy Lab 驱动 · 数据仅供参考不构成投资建议</footer>
</body></html>"""


def _signal_html(signal_str, overheat=False, low_confidence=False):
    if overheat:
        if signal_str == "BEAR":
            return '<span class="signal-overheat">▼ BEAR (过热)</span>'
        return '<span class="signal-overheat">─ NEUT (过热)</span>'
    if signal_str == "BULL":
        return '<span class="signal-bull">▲ BULL</span>'
    elif signal_str == "BEAR":
        return '<span class="signal-bear">▼ BEAR</span>'
    elif low_confidence:
        return '<span class="signal-neut" style="font-style:italic;color:#888">─ NEUT <span style="font-size:10px;color:#666">低信念</span></span>'
    else:
        return '<span class="signal-neut">─ NEUT</span>'


def _tier_badge(tier):
    if tier == 1:
        return '<span class="tier-badge tier1">T1</span>'
    return '<span class="tier-badge tier2">T2</span>'


def _group_badge(group):
    if group == "EM":
        return '<span class="group-badge group-em">EM</span>'
    return '<span class="group-badge group-dm">DM</span>'


def _ret_color(val):
    c = "4caf50" if val >= 0 else "f44336"
    return f'<span style="color:#{c}">{val:+.1%}</span>'


def build_prediction_html(data, week_info):
    """Generate full prediction page HTML."""
    year, week_num, mon, sun = week_info
    gen_time = data["generated_display"]
    period = data.get("prediction_period", f"{year}年{mon.month}月")
    epu = data["epu"]
    markets = data["markets"]
    vix_data = data.get("vix", {})
    actual_acc = data.get("actual_accuracy", {})

    tier1 = [m for m in markets if m["tier"] == 1]
    tier2 = [m for m in markets if m["tier"] == 2]
    hk_markets = [m for m in markets if m["market"] == "HK"]

    # Signal stats
    sdist = data.get("signal_distribution", {"BULL": 0, "BEAR": 0, "NEUT": 0})

    # Market table rows
    def market_row(m):
        overheat = m.get("overheat", False)
        low_conf = m.get("low_confidence", False)
        regime_shift = m.get("regime_shift", False)
        regime_unfamiliar = m.get("regime_unfamiliar", False)
        roll5y = m.get("rolling_accuracy")
        full_acc = m.get("accuracy", 0)
        if roll5y is not None:
            degraded = roll5y < full_acc * 0.7
            r5_color = "#ff9800" if degraded else "#888"
            r5_style = "font-weight:700" if degraded else ""
            r5_cell = f'<span style="color:{r5_color};{r5_style}">{roll5y:.1%}</span>'
        else:
            r5_cell = '<span style="color:#555">N/A</span>'
        # Regime distance cell
        rd = m.get("regime_distance")
        if rd is not None:
            if regime_shift:
                rd_html = f'<span style="color:#f44336;font-weight:700">{rd:.1f} 🔴</span>'
            elif regime_unfamiliar:
                rd_html = f'<span style="color:#ff9800;font-weight:700">{rd:.1f} ⚠</span>'
            else:
                rd_html = f'<span style="color:#666">{rd:.1f}</span>'
        else:
            rd_html = '<span style="color:#555">-</span>'
        sent = m.get("sentiment_factor", "N/A")
        if sent == "BULLISH":
            sent_cell = '<span style="color:#4caf50;font-size:10px">↗</span>'
        elif sent == "BEARISH":
            sent_cell = '<span style="color:#f44336;font-size:10px">↘</span>'
        elif sent == "NEUTRAL":
            sent_cell = '<span style="color:#888;font-size:10px">─</span>'
        else:
            sent_cell = '<span style="color:#555;font-size:10px">N/A</span>'
        # Signal rendering with regime_shift
        sig_html = _signal_html(m["signal"], overheat, low_conf)
        if regime_shift:
            sig_html = '<span style="color:#f44336;font-weight:700">─ NEUT <span style="font-size:9px;color:#f44336">⚠制度偏离</span></span>'
        return (
            f'<tr>'
            f'<td style="padding:6px 8px;font-size:13px">{_tier_badge(m["tier"])}'
            f'{_group_badge(m.get("group","DM"))} {m["name_cn"]}</td>'
            f'<td style="padding:6px 8px">{sig_html}</td>'
            f'<td style="padding:6px 8px;text-align:right">{_ret_color(m["pred_ret"])}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:11px">{r5_cell}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m.get("vr_z",0):+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m.get("mm_z",0):+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m.get("iv_z",0):+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m.get("gs_z",0):+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px">{rd_html}</td>'
            f'<td style="padding:6px 8px;text-align:center">{sent_cell}</td>'
            f'<td style="padding:6px 8px;text-align:left;font-size:10px;color:#555;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{m.get("detail","")}">{m.get("detail","")}</td>'
            f'</tr>'
        )

    tier1_rows = "\n".join(market_row(m) for m in tier1)
    tier2_rows = "\n".join(market_row(m) for m in tier2)

    # HK section
    hk_html = ""
    if hk_markets:
        hk_rows = "\n".join(market_row(m) for m in hk_markets)
        hk_html = f"""
    <div class="hk-section">
        <h3>🇭🇰 港股预测</h3>
        <table class="predict-table">
            <thead><tr>
                <th>市场</th><th>信号</th><th style="text-align:right">预测年收益</th>
                <th style="text-align:right">准确率</th><th style="text-align:right">5年</th><th style="text-align:right">趋势(σ)</th>
                <th style="text-align:right">回归(σ)</th><th style="text-align:right">波动(σ)</th><th style="text-align:right">微动量(σ)</th><th style="text-align:right">周内波动(σ)</th><th style="text-align:right">跳空(σ)</th><th style="text-align:right">制度σ</th><th style="text-align:center">情绪</th><th>详情</th>
            </tr></thead>
            <tbody>{hk_rows}</tbody>
        </table>
    </div>"""

    # VIX section
    vix_html = ""
    if vix_data.get("current") is not None:
        vix_cur = vix_data["current"]
        vix_5d = vix_data.get("trend_5d", 0) or 0
        vix_20d = vix_data.get("trend_20d", 0) or 0
        vix_color = "#f44336" if vix_cur > (vix_data.get("p80") or 25) else "#4caf50"
        vix_html = f"""
    <div class="metric">
        <div class="lbl">VIX (恐慌指数)</div>
        <div class="val" style="color:{vix_color}">{vix_cur:.1f}</div>
        <div style="font-size:10px;color:#888;margin-top:4px">
            5d {vix_5d:+.1%} · 20d {vix_20d:+.1%} · P80={vix_data.get('p80',0):.0f}
        </div>
    </div>"""

    # Actual accuracy section
    acc_html = ""
    if actual_acc and actual_acc.get("hit_rate") is not None:
        hr = actual_acc["hit_rate"]
        hits = actual_acc.get("hits", 0)
        total = actual_acc.get("total_scored", 0)
        acc_color = "#4caf50" if hr >= 0.7 else ("#ff9800" if hr >= 0.5 else "#f44336")
        acc_html = f"""
    <div class="metric">
        <div class="lbl">上期实际命中率</div>
        <div class="val" style="color:{acc_color}">{hr:.0%}</div>
        <div style="font-size:10px;color:#888;margin-top:4px">{hits}/{total} 方向正确</div>
    </div>"""

    body = f"""<div class="container">
    <a href="/" class="back-link">← 首页</a>
    <div class="page-header">
        <div class="icon">🔮</div>
        <h1 style="background:linear-gradient(135deg,#f7971e,#ffd200);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">全球市场预测</h1>
        <p class="sub">{len(markets)} 市场 · 3-Factor 模型 · 分市场制度切换 + VIX 辅助 ｜ 预测周期 {period} ｜ <span class="ts-stamp">生成 {gen_time}</span></p>
    </div>

    <div class="metric-grid">
        <div class="metric">
            <div class="lbl">EPU 政策不确定性</div>
            <div class="val" style="color:{'#f44336' if epu['regime']=='HIGH_EPU' else '#4caf50'}">{epu['value']:.0f} <span style="font-size:11px;color:#888">P{epu['percentile']}</span></div>
            <div style="font-size:10px;color:#888;margin-top:4px">{'⚠ 高不确定性' if epu['regime']=='HIGH_EPU' else '✓ 正常'}</div>
        </div>
        {vix_html}
        <div class="metric">
            <div class="lbl">Tier 1 高置信度</div>
            <div class="val" style="color:#4caf50">{len(tier1)} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">准确率 ≥70% · 可操作</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 2 参考</div>
            <div class="val" style="color:#ff9800">{len(tier2)} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">55-70% · 1年以内</div>
        </div>
        {acc_html}
        <div class="metric">
            <div class="lbl">信号分布</div>
            <div class="val" style="font-size:16px">
                <span style="color:#4caf50">▲{sdist.get('BULL',0)}</span>
                <span style="color:#ff9800;margin-left:8px">─{sdist.get('NEUT',0)}</span>
                <span style="color:#f44336;margin-left:8px">▼{sdist.get('BEAR',0)}</span>
            </div>
            <div style="font-size:10px;color:#888;margin-top:4px">BULL / NEUT / BEAR</div>
        </div>
    </div>

    {hk_html}

    <h3 style="font-size:15px;color:#4caf50;margin-bottom:16px;">★★★ Tier 1 — 高置信度 (≥70%)</h3>
    <p style="font-size:11px;color:#666;margin-top:-8px;margin-bottom:8px">回测=全量历史 | 5年=近5年滚动</p>
    <table class="predict-table">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">预测年收益</th>
            <th style="text-align:right">回测</th><th style="text-align:right">5年</th><th style="text-align:right">趋势(σ)</th><th style="text-align:right">回归(σ)</th><th style="text-align:right">波动(σ)</th><th style="text-align:right">微动量(σ)</th><th style="text-align:right">周内波动(σ)</th><th style="text-align:right">跳空(σ)</th><th style="text-align:right">制度σ</th><th style="text-align:center">情绪</th><th>详情</th>
        </tr></thead>
        <tbody>{tier1_rows}</tbody>
    </table>

    {'<h3 style="font-size:15px;color:#ff9800;margin-top:32px;margin-bottom:16px;">★★ Tier 2 — 参考级别 (55-70%)</h3><table class="predict-table"><thead><tr><th>市场</th><th>信号</th><th style="text-align:right">预测年收益</th><th style="text-align:right">回测</th><th style="text-align:right">5年</th><th style="text-align:right">趋势(σ)</th><th style="text-align:right">回归(σ)</th><th style="text-align:right">波动(σ)</th><th style="text-align:right">微动量(σ)</th><th style="text-align:right">周内波动(σ)</th><th style="text-align:right">跳空(σ)</th><th style="text-align:right">制度σ</th><th style="text-align:center">情绪</th><th>详情</th></tr></thead><tbody>' + tier2_rows + '</tbody></table>' if tier2_rows else ''}

    <div class="insight-box" style="margin-top:32px">
        <p>💡 <strong>模型说明：</strong>3-factor (趋势+回归+波动率) 线性模型 + 分市场制度切换 (EM P65 / DM P75)。
        VIX 作为 EPU 日频代理辅助判断不确定性。过热检测：因子 z-score &gt; +2.5σ → NEUT，&gt; +3.5σ → BEAR。
        Tier 1 市场（≥70% 回测准确率）可作配置参考，Tier 2 市场仅作观察。
        数据来源：yfinance / FRED / akshare · 模型：Global Economy Lab</p>
    </div>
</div>"""

    return build_page(f"全球市场预测 ({gen_time})", body)


def build_market_section_html(data, week_info):
    """Generate market.html insertion snippet."""
    year, week_num, mon, sun = week_info
    gen_time = data["generated_display"]
    period = data.get("prediction_period", f"{year}年{mon.month}月")
    markets = data["markets"]
    epu = data["epu"]
    vix_data = data.get("vix", {})

    tier1 = [m for m in markets if m["tier"] == 1]

    # Top 5 for summary card
    top5 = sorted(tier1, key=lambda m: m["pred_ret"], reverse=True)[:5]

    metric_cards = "\n".join(
        f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">'
        f'<div style="font-size:11px;color:#666;margin-bottom:4px">{m["name_cn"]}</div>'
        f'<div style="font-size:20px;font-weight:700;color:#4caf50">{_signal_html(m["signal"], m.get("overheat"))} {m["pred_ret"]:+.1%}</div>'
        f'<div style="font-size:10px;color:#888;margin-top:4px">准确率 {m["accuracy"]:.1%}</div>'
        f'</div>'
        for m in top5
    )

    # VIX indicator
    vix_indicator = ""
    if vix_data.get("current") is not None:
        vix_cur = vix_data["current"]
        vix_5d = vix_data.get("trend_5d", 0) or 0
        vix_indicator = f"""
                 <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                     <div style="font-size:11px;color:#666;margin-bottom:4px">VIX 恐慌指数</div>
                     <div style="font-size:22px;font-weight:700;color:{'#f44336' if vix_cur > (vix_data.get('p80') or 25) else '#4caf50'}">{vix_cur:.1f} <span style="font-size:11px;color:#888">5d {vix_5d:+.1%}</span></div>
                 </div>"""

    # Market table rows (compact)
    def compact_row(m):
        overheat = m.get("overheat", False)
        low_conf = m.get("low_confidence", False)
        roll5y = m.get("rolling_accuracy")
        full_acc = m.get("accuracy", 0)
        if roll5y is not None:
            degraded = roll5y < full_acc * 0.7
            r5c = "#ff9800" if degraded else "#888"
            r5s = "font-weight:700;" if degraded else ""
            r5_cell = f'<span style="color:{r5c};{r5s}">{roll5y:.1%}</span>'
        else:
            r5_cell = '<span style="color:#555">N/A</span>'
        return (
            f'<tr>'
            f'<td style="padding:6px 8px;font-size:13px">{_group_badge(m.get("group","DM"))} {m["name_cn"]}</td>'
            f'<td style="padding:6px 8px">{_signal_html(m["signal"], overheat, low_conf)}</td>'
            f'<td style="padding:6px 8px;text-align:right">{_ret_color(m["pred_ret"])}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:11px">{r5_cell}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:10px;color:#555;font-family:monospace">{m.get("last_date","")[:10]}</td>'
            f'</tr>'
        )

    all_rows = "\n".join(compact_row(m) for m in markets)

    hk_section = ""
    hk_markets = [m for m in markets if m["market"] == "HK"]
    if hk_markets:
        hk_section = f"""
            <div style="margin-top:20px;background:linear-gradient(135deg,rgba(247,151,30,0.08),rgba(247,151,30,0.02));border:1px solid rgba(247,151,30,0.2);border-radius:12px;padding:16px">
                <div style="font-size:13px;color:#f7971e;margin-bottom:8px">🇭🇰 港股</div>
                <div style="font-size:18px;font-weight:700;color:#4caf50">{hk_markets[0]['name_cn']}: {_signal_html(hk_markets[0]['signal'], hk_markets[0].get('overheat'))} {hk_markets[0]['pred_ret']:+.1%} · 准确率 {hk_markets[0]['accuracy']:.1%}</div>
            </div>"""

    return f"""<!-- 全球市场预测 Section → 插入 market.html 的 insight-box 之后 -->

<div style="margin-top:36px">
    <a href="/prediction.html" style="text-decoration:none">
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:18px;padding:32px;color:#e0e0e0;box-shadow:0 8px 32px rgba(0,0,0,0.3);border:1px solid rgba(247,151,30,0.2)">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
                <span style="font-size:24px">🔮</span>
                <div>
                    <h3 style="color:#f7971e;font-size:18px;margin:0">全球市场预测</h3>
                    <span style="font-size:11px;color:#666;font-family:monospace">预测周期 {period} ｜ 生成 {gen_time}</span>
                </div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px">
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">EPU 政策不确定性</div>
                    <div style="font-size:22px;font-weight:700;color:{'#f44336' if epu['regime']=='HIGH_EPU' else '#4caf50'}">{epu['value']:.0f} <span style="font-size:11px;color:#888">P{epu['percentile']}</span></div>
                </div>
                {vix_indicator}
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">Tier 1 高置信度</div>
                    <div style="font-size:22px;font-weight:700;color:#4caf50">{len(tier1)} 市场</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">制度模式</div>
                    <div style="font-size:18px;font-weight:700;color:#ff9800">{'高不确定' if epu['regime']=='HIGH_EPU' else '正常'}</div>
                </div>
            </div>
            {hk_section}
            {metric_cards}

            <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:16px">
                <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                    <th style="text-align:left;padding:6px 8px;color:#888;font-size:10px">市场</th>
                    <th style="text-align:left;padding:6px 8px;color:#888;font-size:10px">信号</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">预测年收益</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">回测</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">5年</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">数据至</th>
                </tr></thead>
                <tbody>{all_rows}</tbody>
            </table>

            <div style="text-align:right;margin-top:16px">
                <span style="color:#f7971e;font-size:13px">查看完整分析 →</span>
            </div>
        </div>
    </a>
</div>"""


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["data-only", "summary", "full"],
                        default="full", help="Pipeline mode")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--prior-json", default=None,
                        help="Path to prior prediction JSON for actual_accuracy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _PROJECT_ROOT / "output"
    today = date.today()
    archive_dir = output_dir / "archive" / today.strftime("%Y-%m-%d")
    archive_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_time = datetime.now()
    week_info = iso_week_for_prediction(gen_time.date())
    year, week_num, mon, sun = week_info
    # 阶段2 (月频): the prediction_period is monthly-semantic even though the
    # data-only pipeline still runs on a weekly cadence.
    period_label = f"{gen_time.year}年{gen_time.month}月"

    logger.info("=" * 64)
    logger.info(f"Global Economy Lab — Auto Predict [{args.mode}]")
    logger.info(f"  Prediction period: {period_label}")
    logger.info(f"  Generated at: {gen_time.strftime('%Y-%m-%d %H:%M CST')}")
    logger.info(f"  Weekday: {gen_time.strftime('%A')} (ISO: {gen_time.date().isocalendar()})")
    logger.info("=" * 64)

    # Step 0: Load VIX (daily EPU proxy)
    logger.info("Step 0: Loading VIX...")
    vix_series = load_vix()

    # Step 1: Run predictions
    logger.info("Step 1: Running predictions...")
    predictions, epu_values, is_high_epu, vix_info, epu_percentile, predictor, cycle_info = run_predictions(
        vix_series=vix_series
    )
    epu_val = epu_values["us"]
    china_epu_val = epu_values.get("china", epu_val)

    # Step 2: Compute actual accuracy from prior prediction
    logger.info("Step 2: Computing actual accuracy...")
    prior_path = args.prior_json
    if prior_path is None:
        # Auto-detect: look in previous archive directory
        prev_dirs = sorted([
            d for d in (output_dir / "archive").iterdir()
            if d.is_dir() and d.name < today.strftime("%Y-%m-%d")
        ])
        if prev_dirs:
            candidate = prev_dirs[-1] / "weekly_prediction.json"
            if candidate.exists():
                prior_path = str(candidate)
                logger.info(f"  Found prior prediction: {candidate}")

    if prior_path:
        prior_path = Path(prior_path)

    actual_accuracy = compute_actual_accuracy(
        predictions, prior_path or Path("/nonexistent"), today
    ) if prior_path else {"hit_rate": None, "hits": 0, "misses": 0, "markets": {}}

    # Step 3: Build JSON
    logger.info("Step 3: Building JSON output...")
    data = build_json(
        predictions, epu_values, is_high_epu, gen_time, week_info,
        actual_accuracy=actual_accuracy,
        vix_info=vix_info,
        epu_percentile=epu_percentile,
        predictor=predictor,
        cycle_info=cycle_info,
    )

    json_path = output_dir / "weekly_prediction.json"
    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"  → {json_path}")

    # Archive JSON
    with open(archive_dir / "weekly_prediction.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"  → {archive_dir / 'weekly_prediction.json'}")

    if args.mode == "data-only":
        # Print summary even in data-only mode
        tier1 = [m for m in data["markets"] if m["tier"] == 1]
        tier2 = [m for m in data["markets"] if m["tier"] == 2]
        sdist = data.get("signal_distribution", {"BULL": 0, "BEAR": 0, "NEUT": 0})

        logger.info("\n" + "=" * 64)
        logger.info(f"SUMMARY — {period_label}")
        logger.info(f"  US EPU: {epu_val:.0f} (P{epu_percentile:.0f}, {'HIGH' if is_high_epu else 'NORMAL'})")
        if cycle_info and cycle_info.get("cycle_phase") != "unknown":
            logger.info(f"  Cycle: {cycle_info['cycle_label']} (PMI={cycle_info.get('cycle_pmi')}, CPI YoY={cycle_info.get('cycle_cpi_yoy')}%)")
        if vix_info and vix_info["current"]:
            logger.info(f"  VIX: {vix_info['current']:.1f} (5d: {vix_info.get('trend_5d',0):+.1%})")
        logger.info(f"  Signals: {sdist.get('BULL',0)} BULL / {sdist.get('NEUT',0)} NEUT / {sdist.get('BEAR',0)} BEAR")
        flat_count = sum(1 for m in data["markets"] if m.get("flat_market"))
        low_conf_count = sum(1 for m in data["markets"] if m.get("low_confidence"))
        if flat_count or low_conf_count:
            parts = []
            if flat_count:
                parts.append(f"平盘→NEUT:{flat_count}")
            if low_conf_count:
                parts.append(f"低信念→NEUT:{low_conf_count}")
            logger.info(f"  Overrides: {' | '.join(parts)}")
        logger.info(f"  Tier 1 (≥70%): {len(tier1)} markets")
        for m in tier1:
            tags = []
            if m.get("flat_market"):
                tags.append("平盘→NEUT")
            tag_str = f" [{'|'.join(tags)}]" if tags else ""
            logger.info(f"    {m['name_cn']:12s} {m['signal']:5s}{tag_str:20s} {m['pred_ret']:+.1%} acc={m['accuracy']:.1%}")
        if tier2:
            logger.info(f"  Tier 2 (55-70%): {len(tier2)} markets")
            for m in tier2:
                tags = []
                if m.get("flat_market"):
                    tags.append("平盘→NEUT")
                tag_str = f" [{'|'.join(tags)}]" if tags else ""
                logger.info(f"    {m['name_cn']:12s} {m['signal']:5s}{tag_str:20s} {m['pred_ret']:+.1%} acc={m['accuracy']:.1%}")
        if actual_accuracy.get("hit_rate") is not None:
            logger.info(f"  Actual accuracy (prior month): {actual_accuracy['hit_rate']:.1%} ({actual_accuracy['hits']}/{actual_accuracy['total_scored']})")
        logger.info(f"\n  Output: {output_dir}")
        logger.info(f"  Archive: {archive_dir}")
        logger.info("=" * 64)
        return 0

    # Step 4: Generate HTML
    logger.info("Step 4: Generating HTML...")

    # Full prediction page (latest + week-specific)
    html = build_prediction_html(data, week_info)

    # Latest (for /prediction.html nav link)
    html_path = output_dir / "weekly_prediction.html"
    latest_path = output_dir / "prediction.html"
    with open(html_path, "w") as f:
        f.write(html)
    with open(latest_path, "w") as f:
        f.write(html)
    logger.info(f"  → {html_path} ({len(html):,} bytes)")
    logger.info(f"  → {latest_path} (latest)")

    # Week-specific (for /prediction202623.html permanent URL)
    week_filename = f"prediction{year}{week_num:02d}.html"
    week_path = output_dir / week_filename
    with open(week_path, "w") as f:
        f.write(html)
    logger.info(f"  → {week_path} (week-specific)")

    # Archive
    with open(archive_dir / "weekly_prediction.html", "w") as f:
        f.write(html)
    with open(archive_dir / week_filename, "w") as f:
        f.write(html)

    # Market section snippet
    section = build_market_section_html(data, week_info)
    section_path = output_dir / "market_prediction_section.html"
    with open(section_path, "w") as f:
        f.write(section)
    logger.info(f"  → {section_path} ({len(section):,} bytes)")

    # Archive section
    with open(archive_dir / "market_prediction_section.html", "w") as f:
        f.write(section)

    # Step 5: Summary
    tier1 = [m for m in data["markets"] if m["tier"] == 1]
    tier2 = [m for m in data["markets"] if m["tier"] == 2]
    sdist = data.get("signal_distribution", {"BULL": 0, "BEAR": 0, "NEUT": 0})

    logger.info("\n" + "=" * 64)
    logger.info(f"SUMMARY — {period_label}")
    logger.info(f"  US EPU: {epu_val:.0f} (P{epu_percentile:.0f}, {'HIGH' if is_high_epu else 'NORMAL'})")
    if vix_info and vix_info["current"]:
        logger.info(f"  VIX: {vix_info['current']:.1f} (5d: {vix_info.get('trend_5d',0):+.1%}, 20d: {vix_info.get('trend_20d',0):+.1%})")
    logger.info(f"  Signals: {sdist.get('BULL',0)} BULL / {sdist.get('NEUT',0)} NEUT / {sdist.get('BEAR',0)} BEAR")
    if china_epu_val != epu_val:
        logger.info(f"  China EPU: {china_epu_val:.0f} (HK uses for regime)")
    flat_count = sum(1 for m in data["markets"] if m.get("flat_market"))
    low_conf_count = sum(1 for m in data["markets"] if m.get("low_confidence"))
    if flat_count or low_conf_count:
        parts = []
        if flat_count:
            parts.append(f"平盘→NEUT:{flat_count}")
        if low_conf_count:
            parts.append(f"低信念→NEUT:{low_conf_count}")
        logger.info(f"  Overrides: {' | '.join(parts)}")
    logger.info(f"  Tier 1 (≥70%): {len(tier1)} markets")
    for m in tier1:
        tags = []
        if m.get("flat_market"):
            tags.append("平盘→NEUT")
        tag_str = f" [{'|'.join(tags)}]" if tags else ""
        logger.info(f"    [{m.get('group','DM')}] {m['name_cn']:12s} {m['signal']:5s}{tag_str:20s} {m['pred_ret']:+.1%} acc={m['accuracy']:.1%}")
    if tier2:
        logger.info(f"  Tier 2 (55-70%): {len(tier2)} markets")
        for m in tier2:
            tags = []
            if m.get("flat_market"):
                tags.append("平盘→NEUT")
            tag_str = f" [{'|'.join(tags)}]" if tags else ""
            logger.info(f"    [{m.get('group','DM')}] {m['name_cn']:12s} {m['signal']:5s}{tag_str:20s} {m['pred_ret']:+.1%} acc={m['accuracy']:.1%}")
    if actual_accuracy.get("hit_rate") is not None:
        logger.info(f"  Actual accuracy (prior month): {actual_accuracy['hit_rate']:.1%} ({actual_accuracy['hits']}/{actual_accuracy['total_scored']})")
    logger.info(f"\n  Output: {output_dir}")
    logger.info(f"  Archive: {archive_dir}")
    logger.info("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
