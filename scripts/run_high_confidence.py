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
}
TIER_LABELS = {1: "★★★ 可操作", 2: "★★ 参考(1年)", 3: "☆ 不可用"}


def load_prices():
    prices = {}
    for name, ticker in MARKET_TICKERS.items():
        try:
            f = EquitiesFetcher(ticker=ticker)
            df = f.fetch(start_date="1985-01-01", end_date="2026-05-25")
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


def main():
    logger.info("=" * 70)
    logger.info("High-Confidence Predictor")
    logger.info("  Threshold: >=70% published | 55-70% reference | <55% suppressed")
    logger.info("=" * 70)

    prices = load_prices()
    epu_a = load_epu()
    epu_now = float(epu_a.iloc[-1])
    logger.info(f"Loaded {len(prices)} markets, EPU={epu_now:.0f}")

    predictor = HighConfidencePredictor(min_confidence=0.55)
    predictor.fit_epu(epu_a)

    logger.info("Fitting per-market regime models...")
    fitted = 0
    for market, px in sorted(prices.items()):
        result = predictor.fit_market(market, px, epu_a)
        if result:
            fitted += 1
            logger.info(
                f"  {market:6s}: acc={result['accuracy']:.0%} "
                f"(n={result['n_years']}y, "
                f"normal={result['regime_models'].get('normal',{}).get('n',0)}y, "
                f"high={result['regime_models'].get('high_epu',{}).get('n',0)}y)"
            )
    logger.info(f"Fitted {fitted}/{len(prices)} markets >=55%")

    predictions = predictor.predict_all(prices, epu_now)

    print()
    print("=" * 70)
    print("HIGH-CONFIDENCE PREDICTIONS (2026)")
    print(f"EPU: {epu_now:.0f} → regime={'HIGH' if predictor._is_high_epu(epu_now) else 'NORMAL'}")
    print("=" * 70)
    print()

    tier1 = [p for p in predictions if p.tier == 1]
    tier2 = [p for p in predictions if p.tier == 2]

    if tier1:
        print("=== Tier 1 — 可操作 (≥70%) ===")
        print(f'{"Market":>6s} {"Signal":>6s} {"ExpRet":>8s} {"Acc":>6s} {"tm(σ)":>8s} {"mr(σ)":>8s} {"Detail"}')
        print("-" * 85)
        for p in tier1:
            s = "BULL" if p.signal > 0 else ("BEAR" if p.signal < 0 else "NEUT")
            print(
                f"{p.market:>6s} {s:>6s} {p.expected_ret:>+7.1%} {p.confidence:>5.0%} "
                f"{p.factors.get('tm_z',0):>+7.2f}σ {p.factors.get('mr_z',0):>+7.2f}σ  {p.detail}"
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
    print(f'  EPU regime: {"EPU高→趋势延续模式" if predictor._is_high_epu(epu_now) else "EPU正常→均值回归模式"}')
    print(f"  EPU>P75 准确率高于 EPU<P75: {sum(1 for p in predictions if p.regime=='high_epu')}/{len(predictions)} 市场使用高EPU系数")

    return 0


if __name__ == "__main__":
    sys.exit(main())
