"""
High-confidence prediction driver.

Only outputs predictions for markets with >= 70% direction accuracy.
Below 55%: not published.  55-70%: reference only.

Usage:
    python scripts/run_high_confidence.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.high_confidence import HighConfidencePredictor
from src.data_fetcher.equities import EquitiesFetcher
from src.data_fetcher.macro_economic import MacroEconomicFetcher


MARKET_TICKERS = {
    "US": "^GSPC", "DE": "^GDAXI", "JP": "^N225", "GB": "^FTSE",
    "FR": "^FCHI", "IT": "FTSEMIB.MI", "CA": "^GSPTSE", "BR": "^BVSP",
    "KR": "^KS11", "IN": "^NSEI", "AU": "^AXJO", "CN": "000001.SS",
    "HK": "^HSI",
}
TIER_LABELS = {1: "★★★ 可操作", 2: "★★ 参考(1年)", 3: "☆ 不可用"}

# ── Market-specific parameters (2026W27) ──────────────────────────
# HK: China EPU (verified +10pp improvement vs US EPU, now 77%)
# CN: REVERTED to US EPU — China EPU caused -10pp regression (65→55%).
#     A-shares dominated by retail flow/policy directives, not EPU.
#     Higher signal threshold + lower AR to reduce false signals.
# IN: only 19y data, 0 normal EPU years → single-model fallback.
MARKET_PARAMS = {
    "HK": {"ar_coeff": 0.60, "signal_threshold": 0.05, "epu_label": "china"},
    "CN": {"ar_coeff": 0.55, "signal_threshold": 0.06, "epu_label": "us"},
    "BR": {"ar_coeff": 0.65, "signal_threshold": 0.05, "epu_label": "us"},
    "IN": {"ar_coeff": 0.65, "signal_threshold": 0.03, "epu_label": "us"},
}


def load_prices():
    prices = {}
    for name, ticker in MARKET_TICKERS.items():
        try:
            f = EquitiesFetcher(ticker=ticker)
            df = f.fetch(start_date="1985-01-01", end_date="2026-07-05")
            if df is not None and not df.empty:
                idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(
                    df["date"] if "date" in df.columns else df.index
                )
                px = df["close"] if "close" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
                prices[name] = pd.Series(px.values, index=idx, dtype=float).sort_index()
        except Exception:
            pass
    return prices


def load_epu():
    epu = pd.read_csv("data/processed/epu_historical_monthly.csv", index_col=0, parse_dates=True)["value"]
    return epu.resample("YE").mean()


def load_china_epu():
    """Load China EPU (CHNMAINLANDEPU) from FRED."""
    try:
        f = MacroEconomicFetcher(series_id="CHNMAINLANDEPU")
        df = f.fetch(start_date="1985-01-01", end_date="2026-07-05")
        if df is not None and not df.empty:
            epu = df["value"] if "value" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
            return pd.Series(epu.values, index=pd.to_datetime(df.index), dtype=float).resample("YE").mean()
    except Exception as e:
        logger.warning(f"China EPU load failed: {e}")
    return pd.Series(dtype=float)


def main():
    logger.info("=" * 70)
    logger.info("High-Confidence Predictor")
    logger.info("  Threshold: >=70% published | 55-70% reference | <55% suppressed")
    logger.info("  Per-market: AR coeffs + China EPU for HK/CN")
    logger.info("=" * 70)

    prices = load_prices()
    epu_a = load_epu()
    china_epu_a = load_china_epu()
    epu_now = float(epu_a.iloc[-1])
    china_epu_now = float(china_epu_a.iloc[-1]) if len(china_epu_a) > 0 else epu_now
    logger.info(f"Loaded {len(prices)} markets, US EPU={epu_now:.0f}, China EPU={china_epu_now:.0f}")

    predictor = HighConfidencePredictor(min_confidence=0.55)
    predictor.fit_epu(epu_a, label="us")
    if len(china_epu_a) > 0:
        predictor.fit_epu(china_epu_a, label="china")

    # Set per-market parameters
    for market, params in MARKET_PARAMS.items():
        predictor.set_market_params(market, **params)

    logger.info("Fitting per-market regime models...")
    fitted = 0
    for market, px in sorted(prices.items()):
        epu_label = MARKET_PARAMS.get(market, {}).get("epu_label", "us")
        epu_data = china_epu_a if epu_label == "china" else epu_a
        result = predictor.fit_market(market, px, epu_data, epu_label=epu_label)
        if result:
            fitted += 1
            n_years = result['n_years']
            rmodels = result['regime_models']
            normal_n = rmodels.get('normal', {}).get('n', 0)
            high_n = rmodels.get('high_epu', {}).get('n', 0)
            ar = MARKET_PARAMS.get(market, {}).get("ar_coeff", 0.75)
            logger.info(
                f"  {market:6s}: acc={result['accuracy']:.0%} "
                f"(n={n_years}y, normal={normal_n}y, high={high_n}y) "
                f"EPU={epu_label} AR(1)={ar:.2f}"
            )
    logger.info(f"Fitted {fitted}/{len(prices)} markets >=55%")

    predictions = predictor.predict_all(
        prices, epu_now,
        epu_values={"us": epu_now, "china": china_epu_now},
    )

    print()
    print("=" * 70)
    print("HIGH-CONFIDENCE PREDICTIONS (2026)")
    us_regime = "HIGH" if predictor._is_high_epu(epu_now, "us") else "NORMAL"
    cn_regime = "HIGH" if predictor._is_high_epu(china_epu_now, "china") else "NORMAL" if china_epu_now > 0 else "N/A"
    print(f"US EPU: {epu_now:.0f} → {us_regime}  |  China EPU: {china_epu_now:.0f} → {cn_regime}")
    print("=" * 70)
    print()

    tier1 = [p for p in predictions if p.tier == 1]
    tier2 = [p for p in predictions if p.tier == 2]

    if tier1:
        print("=== Tier 1 — 可操作 (≥70%) ===")
        print(f'{"Market":>6s} {"Signal":>6s} {"ExpRet":>8s} {"Acc":>6s} {"tm(σ)":>8s} {"mr(σ)":>8s} {"vr(σ)":>8s} {"Detail"}')
        print("-" * 100)
        for p in tier1:
            s = "BULL" if p.signal > 0 else ("BEAR" if p.signal < 0 else "NEUT")
            print(
                f"{p.market:>6s} {s:>6s} {p.expected_ret:>+7.1%} {p.confidence:>5.0%} "
                f"{p.factors.get('tm_z',0):>+7.2f}σ {p.factors.get('mr_z',0):>+7.2f}σ "
                f"{p.factors.get('vr_z',0):>+7.2f}σ  {p.detail}"
            )
        print()

    if tier2:
        print("=== Tier 2 — 参考 (55-70%, 1年以内) ===")
        print(f'{"Market":>6s} {"Signal":>6s} {"ExpRet":>8s} {"Acc":>6s} {"Detail"}')
        print("-" * 60)
        for p in tier2:
            s = "BULL" if p.signal > 0 else ("BEAR" if p.signal < -0.03 else "NEUT")
            print(
                f"{p.market:>6s} {s:>6s} {p.expected_ret:>+7.1%} {p.confidence:>5.0%}  {p.detail}"
            )
        print()

    print("=== Summary ===")
    print(f"  Tier 1 (≥70%):  {len(tier1)} markets — {', '.join(p.market for p in tier1)}" if tier1 else "  Tier 1: none")
    print(f"  Tier 2 (55-70%): {len(tier2)} markets — {', '.join(p.market for p in tier2)}" if tier2 else "  Tier 2: none")
    us_high = predictor._is_high_epu(epu_now, "us")
    cn_high = predictor._is_high_epu(china_epu_now, "china") if china_epu_now > 0 else False
    print(f"  US EPU regime: {'HIGH→趋势延续' if us_high else 'NORMAL→均值回归'}")
    if china_epu_now > 0:
        print(f"  China EPU regime: {'HIGH→趋势延续' if cn_high else 'NORMAL→均值回归'} (HK/CN使用)")
    cn_hk_high = sum(1 for p in predictions if p.market in ("CN","HK") and "high_epu" in p.regime)
    us_high_count = sum(1 for p in predictions if p.market not in ("CN","HK") and "high_epu" in p.regime)
    print(f"  US EPU>P75 市场: {us_high_count}/{len(predictions) - (2 if 'CN' in [m.market for m in predictions] and 'HK' in [m.market for m in predictions] else 0)}")
    print(f"  China EPU>P75 市场: {cn_hk_high}/2")

    return 0


if __name__ == "__main__":
    sys.exit(main())
