"""
High-confidence predictor.

Only operates on markets with proven directional accuracy >= 70%.
Applies EPU regime-switching: separate 3-factor models (trend_momentum,
mean_reversion, volatility_regime) for EPU-normal and EPU-high periods.

2026 阶段2 (月频重构 + 框架简化):
  The prediction target moved from the longer-horizon/annual return to a
  forward ~1-month (21-session) close-to-close return, annualized ×12 so the
  ``pred`` scale stays in familiar annual-percentage terms.  The 12+ override
  layers accumulated during W23-W34 (overheat, regime distance, VIX-EPU
  conflict, sentiment/flow, micro-period factors, intraday micro-evidence,
  VIX risk-off spike, cycle gating) are REMOVED.  ``predict()`` returns to a
  clean, explainable 4-layer framework:
    1) 3-factor baseline (tm + mr + vr)
    2) EPU regime switch (high_epu / normal, per-group threshold)
    3) low-conviction gate (incl. the final_low_confidence honest-NEUT)
    4) flat-market threshold (|recent monthly return| < 0.5% → NEUT)

Output tiers:
  Tier 1 (>=70%): Actionable prediction
  Tier 2 (55-70%): Reference only, 1-year horizon
  Tier 3 (<55%): Not published

2026W28 improvement (July 2026):
  - Market groups: Emerging (CN,HK,BR,IN,KR) vs Developed (US,DE,JP,GB,FR,IT,CA,AU)
    with distinct EPU sensitivity per group.  Emerging markets have lower EPU
    threshold (P65) since their policy uncertainty is structurally higher.
  - Overheat downgrade rules: break the BULL-everything bias in HIGH_EPU.
    Any factor z-score > +2.5σ → force NEUT; > +3.5σ → force BEAR.
  - VIX integration: daily-frequency proxy for EPU; VIX > P80 → high-uncertainty,
    VIX < P20 → low-uncertainty.  Replaces stale monthly EPU when available.
  - Removed blind "EPU高→趋势延续模式" default.  Overheat rules take priority
    to prevent bubble-chasing predictions.

2026W32 correction (Aug 2026) — de-blanketing the reactive downgrade gates.
Out-of-sample hit rate fell 50% (W28) → 27% (W29) → 22% (W30) → 22% (W31)
as W28-W31 each added a blanket "downgrade to NEUT" rule.  W31 ended with
13/13 markets NEUT (zero actionable signal) while 9/13 markets rose.
Evidence from W31: the raw 3-factor model predicted +7%..+28% annual for
11/13 markets (directionally correct ~65%), but the stacked gates zeroed
every signal.  Fixes (each gate now requires genuine evidence, not a veto):
  - VIX-EPU conflict: now a confidence discount (×0.85), NOT a NEUT veto.
    A calm VIX during monthly-EPU P99 means the lagging EPU read overstates
    risk — that is information, not a reason to silence the model.
  - Removed the unconditional "extreme EPU |pred|<15% → NEUT" gate (it was
    a redundant double-kill on top of the low-conviction gate).
  - Low-conviction return threshold 15% → 8% annual: 8-15% expected annual
    returns ARE meaningful signals; the 15% bar sat above the median pred.
  - Cycle tilt (stagflation/recession kills BULL) now applies only to
    weak-trend signals (max |z| < 1.5).  Strong-momentum BULLs survive.
  - Regime-distance force-NEUT threshold 2.5σ → 3.5σ (warn stays at 1.5σ):
    the 2.5σ bar forced NEUT on exactly the strongest-trend markets.
  - Current factor window: trailing 252-day mean instead of partial
    calendar-year mean, so intra-year trend reversals (e.g. HSI's +12%
    July rally after a weak H1) are visible to the model.

2026W33 — micro-period factors (3-7 day horizon):
  The trend model (20/60/120-day) was blind to weekly turns during W30-W31
  (22-27% hit rate in P99 EPU).  Three new factors — micro_momentum (5d),
  intraweek_volatility (5d vs 60d), and gap_signal (weekend gaps) — are
  computed as a confidence MODULATOR, not added to the regression model:
  when their combined score agrees with the model's direction confidence
  gets ×1.06, when it disagrees confidence is discounted ×0.90.  No signal
  flips, no NEUT forcing — a strong trend survives a weak micro cross-current.

2026W34 — intraday micro-factors with directional power:
  close_momentum (daily close-to-open return, 5d trailing mean) and
  close_location (close position in the day's high-low range, 5d trailing
  mean) are the first micro-factors with VERIFIED directional IC
  (full-period +0.151 / +0.102; 2024+ +0.134 / +0.120; hit rates 56.5% /
  58.4%) and near-zero correlation with the momentum family (≈0.08-0.13
  vs micro_momentum).  They therefore may now influence DIRECTION, not
  just confidence — but only on hard evidence and only in weak states:
  both z-scores must agree and clear |z| ≥ 1.0 (MICRO_EVIDENCE_Z_THRESHOLD).
  A NEUT signal may be lifted to a weak BULL/BEAR (MICRO_EVIDENCE_LIFT_RET
  = 6% expected annual); a weak BULL/BEAR (all 3-factor |z| < 1.5 and
  |pred| < MICRO_EVIDENCE_STRONG_RET = 12%) is pulled to NEUT when both
  oppose.  Strong 3-factor signals (any |z| ≥ 1.5 or |pred| ≥ 12%) are
  never touched; BULL↔BEAR flips are structurally impossible.  The W33
  micro_score blend (mm 0.50 / gs 0.30 / iv 0.20) is unchanged — cm/cl are
  excluded from the confidence modulator and act on direction only through
  this gate.

2026W34 (W34 weekly correction) — risk-off VIX spike:
  W34 hit 0/6 BULL (0.0 actual accuracy): all six BULL signals reversed
  (US -0.79%, DE -1.17%, IT -1.99%, CA -0.11%, IN -1.45%, AU -2.26%) while
  VIX spiked +11.2% in 5 days (15.84) — but stayed below P80 (24.2), so the
  "rising VIX" regime gate never fired.  Fix: a fast VIX spike (> +10% / 5d)
  is now detected at ANY level (VIX_RISK_OFF_5D_THRESHOLD) and discounts
  surviving BULL confidence ×0.70 (VIX_RISK_OFF_BULL_DISCOUNT) — directional,
  not a NEUT veto (a hard veto would re-import W31's 13/13 over-suppression,
  since VIX 5d > +5% occurs on ~27% of calm days).

Previous improvements retained:
  - Per-market EPU assignment: HK uses China EPU (CHNMAINLANDEPU)
  - Market-specific AR(1) decay and signal thresholds
  - CN reverted to US EPU (China EPU caused -10pp regression)
  - Single-model fallback when regime-switching degrades accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.factors import (
    trend_momentum,
    mean_reversion,
    volatility_regime,
)
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Market groups (per-group EPU sensitivity)
# ---------------------------------------------------------------------------
EMERGING_MARKETS = {"CN", "HK", "BR", "IN", "KR"}
DEVELOPED_MARKETS = {"US", "DE", "JP", "GB", "FR", "IT", "CA", "AU"}

# Default EPU thresholds per group
GROUP_EPU_THRESHOLDS = {
    "emerging": 65.0,   # P65: EM structurally higher EPU, less sensitive
    "developed": 75.0,  # P75: DM stable-policy baseline
}

# 2026 阶段2 (月频重构): the model now trains on ~21-trading-day monthly
# returns (annualized ×12 so pred stays in the familiar annual-percentage
# scale and the display "预测年收益" stays consistent).  All the extra override
# layers stacked during W23-W34 (overheat, regime distance, VIX-EPU conflict,
# sentiment/flow, micro-period, intraday micro-evidence, VIX risk-off, cycle)
# are removed — predict() returns to a clean, explainable 4-layer framework:
#   3-factor base + EPU regime switch + low-conviction gate + flat-market gate.

# Trading days per calendar month (used for the 21-session forward target).
TRADING_DAYS_PER_MONTH = 21
# Annualization factor applied to monthly returns so pred keeps a familiar
# annual-percentage scale (a 1% monthly move → ~12% annual-comparable).
MONTHLY_ANNUALIZATION = 12.0

# Flat market threshold (P0: 2026W29) — |recent monthly return| < FLAT bar → NEUT
FLAT_MARKET_THRESHOLD = 0.005   # |return| < 0.5% → force NEUT

# Low-conviction thresholds (P1: 2026W30, tuned 2026W32)
# When predicted return magnitude and all factor z-scores are
# below these thresholds, BULL/BEAR → NEUT (signal quality gating).
# 8-15% annual expected returns are meaningful.
LOW_CONVICTION_RET_THRESHOLD = 0.08    # |pred_ret| < 8% annual → low conviction
LOW_CONVICTION_Z_THRESHOLD = 1.5        # all |z-score| < 1.5 → no clear signal

# Current-factor window (阶段2月频): the "now" factor state is a trailing
# one-month (21 trading day) mean of the daily factor series, consistent with
# the monthly training aggregation.
CURRENT_FACTOR_WINDOW_DAYS = TRADING_DAYS_PER_MONTH



def _market_group(market: str) -> str:
    """Return 'emerging' or 'developed' for a given market code."""
    if market in EMERGING_MARKETS:
        return "emerging"
    return "developed"


@dataclass
class MarketPrediction:
    """Single market prediction with confidence.

    Attributes:
        market:        Market identifier.
        signal:        -1 (bear), 0 (neutral), +1 (bull).
        expected_ret:  Expected annual return (decimal).
        confidence:    Directional accuracy from backtest (0-1).
        tier:          1 (>=70%), 2 (55-70%), 3 (<55%).
        regime:        'normal' or 'high_epu' (+ optional suffix).
        factors:       Current factor values and z-scores.
        detail:        Human-readable summary.
        overheat:      True if overheat rule was triggered.
    """

    market: str
    signal: int
    expected_ret: float
    confidence: float
    tier: int
    regime: str
    factors: Dict[str, float] = field(default_factory=dict)
    detail: str = ""
    flat_market: bool = False
    low_confidence: bool = False
    rolling_accuracy: Optional[float] = None


class HighConfidencePredictor:
    """EPU regime-aware predictor for high-confidence markets.

    Supports per-market EPU assignment (e.g., HK/CN use China EPU),
    market-specific AR(1) decay coefficients, signal thresholds,
    and market-group-based EPU sensitivity.

    Args:
        epu_threshold_pct:  Default EPU percentile for regime switch (P75).
        min_confidence:     Minimum direction accuracy to publish (default 0.70).
        min_window:         Minimum years per regime to fit model.
        group_thresholds:   Dict overriding per-group EPU thresholds.
    """

    def __init__(
        self,
        epu_threshold_pct: float = 75.0,
        min_confidence: float = 0.70,
        min_window: int = 6,
        group_thresholds: Dict[str, float] = None,
    ):
        self.epu_threshold_pct = epu_threshold_pct
        self.min_confidence = min_confidence
        self.min_window = min_window
        self.group_thresholds = group_thresholds or GROUP_EPU_THRESHOLDS
        self._models: Dict[str, Dict] = {}
        self._epu_history: Dict[str, np.ndarray] = {}   # label → values
        self._epu_threshold: Dict[str, float] = {}       # label → threshold
        self._market_epu_label: Dict[str, str] = {}       # market → 'us' | 'china'
        self._market_ar_coeffs: Dict[str, float] = {}     # market → AR(1) decay
        self._market_signal_threshold: Dict[str, float] = {}  # market → signal cutoff
        # Per-market minimum publish accuracy (阶段2 P2).  Defaults to
        # self.min_confidence; used to keep otherwise-sufficiently-accurate
        # markets (e.g. HK/GB ~0.50-0.54) in the published set.
        self._market_min_confidence: Dict[str, float] = {}
        # VIX support
        self._vix_history: Optional[np.ndarray] = None
        self._vix_p80: Optional[float] = None
        self._vix_p20: Optional[float] = None
        self._vix_now: Optional[float] = None
        self._vix_trend_5d: Optional[float] = None   # 5-day change (fraction)
        self._vix_trend_20d: Optional[float] = None  # 20-day change (fraction)
        self._vix_active: bool = False
        self._vix_median: float = 20.0
        # Regime detection (P3: 2026W30)
        self._factor_means: Dict[str, np.ndarray] = {}   # market → (3,) mean vector
        self._factor_cov: Dict[str, np.ndarray] = {}      # market → (3,3) covariance

    # ── EPU / VIX fitting ───────────────────────────────────────
    def fit_epu(self, epu_annual: pd.Series, label: str = "us"):
        """Store EPU distribution for percentile calculation.

        Args:
            epu_annual: Annual mean EPU values.
            label:      Identifier for this EPU series (e.g. 'us', 'china').
        """
        self._epu_history[label] = epu_annual.dropna().values
        self._epu_threshold[label] = np.percentile(
            self._epu_history[label], self.epu_threshold_pct
        )

    def fit_vix(self, vix_series: pd.Series):
        """Fit VIX distribution as a daily EPU proxy.

        VIX above its historical P80 → high uncertainty (like HIGH_EPU).
        VIX below P20 → low uncertainty (like normal).

        Stores 5d/20d trend for directional signal.

        Args:
            vix_series: Daily VIX close prices.
        """
        if vix_series is None or len(vix_series) < 252:
            logger.warning("VIX data insufficient for fitting (< 252 obs)")
            return
        clean = vix_series.dropna().values
        if len(clean) < 252:
            return
        self._vix_history = clean
        self._vix_p80 = float(np.percentile(clean, 80))
        self._vix_p20 = float(np.percentile(clean, 20))
        self._vix_now = float(clean[-1])
        # 5-day and 20-day trends (positive = VIX rising = uncertainty increasing)
        if len(clean) >= 5:
            self._vix_trend_5d = float(clean[-1] / clean[-6] - 1) if clean[-6] > 0 else 0
        if len(clean) >= 20:
            self._vix_trend_20d = float(clean[-1] / clean[-21] - 1) if clean[-21] > 0 else 0
        self._vix_active = True
        self._vix_median = float(np.median(clean))
        logger.info(
            f"VIX fitted: now={self._vix_now:.1f} P80={self._vix_p80:.1f} "
            f"P20={self._vix_p20:.1f} 5d={self._vix_trend_5d:+.1%} 20d={self._vix_trend_20d:+.1%}"
        )

    # ── Market config ───────────────────────────────────────────
    def set_market_params(
        self,
        market: str,
        ar_coeff: float = None,
        signal_threshold: float = None,
        epu_label: str = None,
        min_confidence: float = None,
    ):
        """Set per-market model parameters.

        Args:
            market:           Market identifier (e.g. 'HK', 'CN').
            ar_coeff:         AR(1) decay coefficient (default 0.75).
                              Lower = faster mean-reversion (EM markets).
            signal_threshold: Return cutoff for BULL/BEAR signal (default 0.03).
                              Higher = more conservative signals.
            epu_label:        Which EPU to use for regime switching ('us', 'china').
            min_confidence:   Minimum directional accuracy to publish this
                              market (overrides the global min_confidence).
        """
        if ar_coeff is not None:
            self._market_ar_coeffs[market] = ar_coeff
        if signal_threshold is not None:
            self._market_signal_threshold[market] = signal_threshold
        if epu_label is not None:
            self._market_epu_label[market] = epu_label
        if min_confidence is not None:
            self._market_min_confidence[market] = min_confidence

    def _min_confidence_for(self, market: str) -> float:
        """Publish threshold for a market (per-market override, else global)."""
        return self._market_min_confidence.get(market, self.min_confidence)

    def _tier_for(self, market: str, confidence: float) -> int:
        """Map accuracy to a publish tier.

        Tier 1 requires strict >=70%.  Tier 2's lower bound is the per-market
        publish threshold (0.55 default; 0.50 for markets like HK/GB that are
        explicitly kept in the set even though they sit just above random).
        """
        if confidence >= 0.70:
            return 1
        if confidence >= self._min_confidence_for(market):
            return 2
        return 3

    # ── Regime detection ────────────────────────────────────────
    def _epu_threshold_for_market(self, market: str) -> float:
        """EPU percentile threshold for this market group."""
        group = _market_group(market)
        return self.group_thresholds.get(group, self.epu_threshold_pct)

    def _is_high_epu(self, epu_value: float, label: str = "us") -> bool:
        if label not in self._epu_threshold:
            return False
        return epu_value > self._epu_threshold[label]

    def _get_regime(self, market: str, epu_value: float, epu_label: str) -> Tuple[str, str]:
        """Determine regime for a market from EPU (阶段2: EPU-only switching).

        The VIX-EPU conflict and fast-spike risk-off override layers are
        removed (2026 阶段2).  Regime selection is driven purely by the
        market's EPU value vs its group threshold — a clean, explainable
        high_epu / normal switch.  VIX is still fitted and surfaced for
        display, but no longer overrides or discounts regime decisions.

        Returns (regime_key, source) where:
          regime_key: 'high_epu' / 'normal' (+ _<label> for non-US EPU)
          source:     'epu'
        """
        is_high = self._is_high_epu(epu_value, epu_label)
        base = f"high_epu_{epu_label}" if (is_high and epu_label != "us") else \
               "high_epu" if is_high else \
               f"normal_{epu_label}" if epu_label != "us" else "normal"
        return base, "epu"

    # ── Model fitting ───────────────────────────────────────────
    def fit_market(
        self,
        market: str,
        prices: pd.Series,
        epu_annual: pd.Series,
        epu_label: str = None,
        ohlc: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """Fit per-regime 3-factor model (tm, mr, vr) for a single market.

        Uses the market group's EPU threshold for regime switching.

        阶段2 (月频): the model is refit on monthly-aggregated factors with a
        forward ~1-month (annualized) return label, so the prediction target
        is the coming month's direction rather than a longer-horizon return.

        Args:
            market:     Market identifier.
            prices:     Daily price series.
            epu_annual: Annual mean EPU values (US or China).
            epu_label:  Which EPU to use for regime ('us' or 'china').
                        If None, auto-assigned from _market_epu_label or default 'us'.
            ohlc:       Accepted for backward compatibility only; the 2026W34
                        intraday micro-factor distributions are removed in
                        阶段2 and this argument is no longer used.

        Returns dict with model parameters, or None if market fails criteria.
        """
        if epu_label is None:
            epu_label = self._market_epu_label.get(market, "us")
        self._market_epu_label[market] = epu_label

        # Per-group EPU threshold percentile
        group_pct = self._epu_threshold_for_market(market)
        epu_thresh = self._epu_threshold.get(
            epu_label,
            np.percentile(epu_annual.dropna().values, group_pct)
        )
        if epu_label not in self._epu_threshold:
            # Use per-group threshold when computing for the first time
            epu_thresh = np.percentile(epu_annual.dropna().values, group_pct)
            self._epu_threshold[epu_label] = epu_thresh

        if prices is None or prices.empty:
            return None

        daily_r = prices.pct_change(fill_method=None)
        # 阶段2 (月频): training label is the forward ~1-month (calendar month)
        # close-to-close return, annualized ×12 so pred stays on the familiar
        # annual-percentage scale and the low-conviction / display thresholds
        # keep their meaning.  Direction is unchanged by the scaling.
        monthly_r = daily_r.resample("ME").apply(lambda x: np.prod(1 + x) - 1)
        monthly_r = (monthly_r * MONTHLY_ANNUALIZATION).dropna()
        if len(monthly_r) < 10:
            return None

        # 阶段2 (月频窗口): 3-factor features aggregated at monthly frequency.
        tm = trend_momentum(prices).resample("ME").mean().reindex(monthly_r.index)
        mr = mean_reversion(prices).resample("ME").mean().reindex(monthly_r.index)
        vr = volatility_regime(prices).resample("ME").mean().reindex(monthly_r.index)
        fwd = monthly_r.shift(-1).dropna()

        valid = tm.dropna().index.intersection(mr.dropna().index).intersection(
            vr.dropna().index).intersection(fwd.index).intersection(epu_annual.dropna().index)
        if len(valid) < 10:
            return None

        tm_v = tm.loc[valid]
        mr_v = mr.loc[valid]
        vr_v = vr.loc[valid]
        fwd_v = fwd.loc[valid]
        epu_v = epu_annual.loc[valid]

        high_epu = epu_v > epu_thresh
        n_normal = (~high_epu).sum()
        n_high = high_epu.sum()

        regime_models = {}
        overall_preds = []

        for label, mask in [("normal", ~high_epu), ("high_epu", high_epu)]:
            if mask.sum() < self.min_window:
                continue
            X = pd.DataFrame({"tm": tm_v[mask], "mr": mr_v[mask], "vr": vr_v[mask]}).fillna(0)
            y = fwd_v[mask]
            m = LinearRegression()
            m.fit(X.values, y)
            yp = m.predict(X)
            acc = float((np.sign(yp) == np.sign(y.values)).mean())
            regime_models[label] = {
                "model": m,
                "accuracy": acc,
                "n": int(mask.sum()),
                "tm_coef": float(m.coef_[0]),
                "mr_coef": float(m.coef_[1]),
                "vr_coef": float(m.coef_[2]),
            }

        # Evaluate overall accuracy by predicting each point with correct regime model
        for i in range(len(valid)):
            is_high = high_epu.iloc[i]
            regime_key = "high_epu" if is_high else "normal"
            if regime_key not in regime_models:
                continue
            m = regime_models[regime_key]["model"]
            x = np.array([[tm_v.iloc[i], mr_v.iloc[i], vr_v.iloc[i]]])
            overall_preds.append(float(m.predict(x)[0]))

        if len(overall_preds) < 10:
            return None

        overall_acc = float(
            (np.sign(overall_preds) == np.sign(fwd_v.values[:len(overall_preds)])).mean()
        )

        # If regime-switched accuracy is lower than single-model, use single-model
        X_all = pd.DataFrame({"tm": tm_v, "mr": mr_v, "vr": vr_v}).fillna(0)
        m_single = LinearRegression()
        m_single.fit(X_all.values, fwd_v)
        yp_single = m_single.predict(X_all.values)
        single_acc = float((np.sign(yp_single) == np.sign(fwd_v.values)).mean())

        if single_acc > overall_acc:
            regime_acc = overall_acc  # save before overwrite for logging
            overall_acc = single_acc
            logger.info(
                f"  {market}: single-model ({single_acc:.0%}) > "
                f"regime-switched ({regime_acc:.0%}) — using single"
            )

        threshold = self._min_confidence_for(market)
        if overall_acc < threshold:
            logger.info(
                f"  {market}: accuracy {overall_acc:.0%} < "
                f"{threshold:.0%} threshold — skipping"
            )
            return None

        result = {
            "market": market,
            "accuracy": overall_acc,
            "n_years": len(valid),
            "regime_models": regime_models,
            "tm_mean": float(tm_v.mean()),
            "mr_mean": float(mr_v.mean()),
            "vr_mean": float(vr_v.mean()),
            "tm_std": float(tm_v.std()),
            "mr_std": float(mr_v.std()),
            "vr_std": float(vr_v.std()),
            "epu_threshold": epu_thresh,
            "group": _market_group(market),
            "group_pct": group_pct,
        }
        self._models[market] = result
        return result

    # ── Flat market check (P0: 2026W29) ─────────────────────────
    def _check_flat_market(self, prices: pd.Series) -> Tuple[bool, str]:
        """Check if the market has been effectively flat recently.

        Uses a ~1-month (21 trading day) close-to-close window, aligned with
        the 阶段2 月度 flat-check semantics in ``predict()``.  When
        |monthly return| < FLAT_MARKET_THRESHOLD the market shows negligible
        directional movement — BULL/BEAR signals under these conditions are
        noise rather than actionable predictions.

        Returns (is_flat, reason).
        """
        window = TRADING_DAYS_PER_MONTH  # ~21 trading days ≈ 1 month
        if prices is None or len(prices) < window + 1:
            return False, ""
        recent_ret = float(prices.iloc[-1] / prices.iloc[-(window + 1)] - 1)
        if abs(recent_ret) < FLAT_MARKET_THRESHOLD:
            return True, f"平盘 (|月度ret|={recent_ret:.2%} < {FLAT_MARKET_THRESHOLD:.1%}) → NEUT"
        return False, ""

    def _check_low_conviction(
        self, pred_ret: float, factors: Dict[str, float], signal: int, market: str
    ) -> Tuple[int, bool]:
        """Override BULL/BEAR → NEUT when conviction is too weak.

        Triggered when |pred_ret| < LOW_CONVICTION_RET_THRESHOLD AND
        all |z-scores| < LOW_CONVICTION_Z_THRESHOLD AND signal is not
        already NEUT.

        Returns (new_signal, triggered).
        """
        if signal == 0:
            return signal, False

        max_z = max(abs(factors.get("tm_z", 0)),
                    abs(factors.get("mr_z", 0)),
                    abs(factors.get("vr_z", 0)))

        if abs(pred_ret) < LOW_CONVICTION_RET_THRESHOLD and max_z < LOW_CONVICTION_Z_THRESHOLD:
            sig_label = "BULL" if signal > 0 else "BEAR"
            logger.info(
                f"  {market}: low-conviction override ({sig_label} → NEUT) "
                f"— |ret|={pred_ret:.3%} all |z|<{LOW_CONVICTION_Z_THRESHOLD}"
            )
            return 0, True

        return signal, False

    # ── Prediction ──────────────────────────────────────────────
    def predict(
        self,
        market: str,
        prices: pd.Series,
        epu_value: float,
        cycle_phase: str = "unknown",
    ) -> MarketPrediction:
        """Generate prediction for a single market.

        Uses the appropriate regime model based on current EPU,
        with market-specific AR(1) decay and signal thresholds.

        阶段2 (月频 + 简化框架): the override stack built up during W23-W34
        (overheat, regime distance, VIX-EPU conflict, sentiment/flow,
        micro-period, intraday micro-evidence, VIX risk-off, cycle) is removed.
        predict() returns to a clean, explainable 4-layer framework:
          1) 3-factor baseline (tm + mr + vr)
          2) EPU regime switch (high_epu / normal, per-group threshold)
          3) low-conviction gate (incl. the final_low_confidence honest-NEUT)
          4) flat-market threshold (|recent monthly return| < 0.5% → NEUT)
        """
        if market not in self._models:
            return MarketPrediction(
                market=market, signal=0, expected_ret=0.0,
                confidence=0.0, tier=3, regime="unknown",
                detail="not fitted",
            )

        cfg = self._models[market]
        epu_label = self._market_epu_label.get(market, "us")
        regime_key, regime_source = self._get_regime(market, epu_value, epu_label)

        # Map regime key back to model regime key
        model_regime = "high_epu" if "high_epu" in regime_key else "normal"

        if model_regime not in cfg["regime_models"]:
            return MarketPrediction(
                market=market, signal=0, expected_ret=0.0,
                confidence=cfg["accuracy"],
                tier=1 if cfg["accuracy"] >= 0.70 else 2,
                regime="unknown",
                detail=f"no model for {model_regime}",
            )

        rm = cfg["regime_models"][model_regime]
        m = rm["model"]

        # Current factor values — trailing-1y window (2026W32).
        # Calendar-YTD means lag intra-year reversals (HSI's July +12%
        # rally was invisible in the H1-2026 annual mean), so the "now"
        # factor state uses the trailing 252 trading days instead.
        tm_daily = trend_momentum(prices)
        mr_daily = mean_reversion(prices)
        vr_daily = volatility_regime(prices)

        def _trailing_mean(series: pd.Series, window: int = CURRENT_FACTOR_WINDOW_DAYS) -> float:
            vals = series.dropna()
            if vals.empty:
                return 0.0
            window = min(window, len(vals))
            return float(vals.iloc[-window:].mean())

        tm_now = _trailing_mean(tm_daily)
        mr_now = _trailing_mean(mr_daily)
        vr_now = _trailing_mean(vr_daily)
        tm_z = (tm_now - cfg["tm_mean"]) / cfg["tm_std"] if cfg["tm_std"] > 0 else 0
        mr_z = (mr_now - cfg["mr_mean"]) / cfg["mr_std"] if cfg["mr_std"] > 0 else 0
        vr_z = (vr_now - cfg["vr_mean"]) / cfg["vr_std"] if cfg["vr_std"] > 0 else 0

        # Market-specific AR(1) decay coefficient
        ar_coeff = self._market_ar_coeffs.get(market, 0.75)
        tm_f = cfg["tm_mean"] + ar_coeff * (tm_now - cfg["tm_mean"])
        mr_f = cfg["mr_mean"] + ar_coeff * (mr_now - cfg["mr_mean"])
        vr_f = cfg["vr_mean"] + ar_coeff * (vr_now - cfg["vr_mean"])
        pred = float(m.predict([[tm_f, mr_f, vr_f]])[0])

        # Market-specific signal threshold
        sig_thresh = self._market_signal_threshold.get(market, 0.03)
        signal = 1 if pred > sig_thresh else (-1 if pred < -sig_thresh else 0)
        tier = self._tier_for(market, cfg["accuracy"])

        factors = {
            "tm": round(tm_now, 4), "tm_z": round(tm_z, 2),
            "mr": round(mr_now, 4), "mr_z": round(mr_z, 2),
            "vr": round(vr_now, 4), "vr_z": round(vr_z, 2),
        }

        confidence = cfg["accuracy"]
        expected_ret = float(pred)
        flat_market = False
        low_confidence = False
        final_low_confidence = False

        # ── P0: Flat market check (overrides BULL/BEAR to NEUT) ──
        # 阶段2 (月频): the flat check uses a monthly-window return so the
        # "national" ±0.5% floor reflects a full month of movement.
        if signal != 0:
            is_flat, flat_reason = self._check_flat_market(prices)
            if is_flat:
                signal = 0
                flat_market = True
                logger.info(
                    f"  {market}: flat market override → NEUT — {flat_reason}"
                )

        # ── P1: Low-conviction check (overrides weak BULL/BEAR → NEUT) ──
        if not flat_market and signal != 0:
            signal, low_confidence = self._check_low_conviction(pred, factors, signal, market)

        # ── Final honest-NEUT check (阶段1 P0, kept) ─────────────
        # Re-check low conviction after the earlier gate: ensures that if
        # |pred| < LOW_CONVICTION_RET_THRESHOLD AND all 3-factor |z| <
        # LOW_CONVICTION_Z_THRESHOLD, the final signal is unconditionally
        # NEUT — never a default BULL from a tiny intercept.
        if signal != 0:
            final_max_z = max(abs(tm_z), abs(mr_z), abs(vr_z))
            if abs(pred) < LOW_CONVICTION_RET_THRESHOLD and final_max_z < LOW_CONVICTION_Z_THRESHOLD:
                original_signal_label = "BULL" if signal > 0 else "BEAR"
                signal = 0
                final_low_confidence = True
                logger.info(
                    f"  {market}: final low-conviction override ({original_signal_label} → NEUT) "
                    f"— |ret|={pred:.3%} all |z|<{LOW_CONVICTION_Z_THRESHOLD} "
                    f"(tm_z={tm_z:+.2f} mr_z={mr_z:+.2f} vr_z={vr_z:+.2f})"
                )

        # ── Build detail string ──────────────────────────────
        detail_parts = []
        if flat_market:
            detail_parts.append(f"平盘→NEUT (|月度ret| < {FLAT_MARKET_THRESHOLD:.1%})")
        if low_confidence:
            detail_parts.append("低信念→NEUT")
        if final_low_confidence:
            detail_parts.append("低信念→无法预测")

        # Regime description
        if "high_epu" in regime_key:
            epu_src = "中EPU高" if epu_label == "china" else "EPU高"
            detail_parts.append(f"{epu_src}→高政策不确定性")
        else:
            epu_src = "中EPU正常" if epu_label == "china" else "EPU正常"
            detail_parts.append(f"{epu_src}→均值回归模式")

        # Factor descriptions
        if tm_z < -0.5:
            detail_parts.append("趋势极弱")
        elif tm_z > 0.5:
            detail_parts.append("趋势偏强")
        if mr_z > 0.5:
            detail_parts.append("深度超卖")
        elif mr_z < -0.5:
            detail_parts.append("明显超买")
        if vr_z < -0.5:
            detail_parts.append("高波动")
        elif vr_z > 0.5:
            detail_parts.append("低波动")

        # Add VIX context when available (informational only — no gating)
        if self._vix_active and self._vix_now is not None:
            if self._vix_trend_5d is not None and abs(self._vix_trend_5d) > 0.03:
                direction = "↓" if self._vix_trend_5d < 0 else "↑"
                detail_parts.append(f"VIX={self._vix_now:.1f}{direction}")

        # Tier from accuracy (no confidence discounting remains — clean model).
        # Uses the per-market publish threshold so HK/GB (@0.50) show as Tier 2.
        tier = self._tier_for(market, confidence)

        return MarketPrediction(
            market=market,
            signal=signal,
            expected_ret=round(expected_ret, 4),
            confidence=round(confidence, 4),
            tier=tier,
            regime=f"{regime_key}_{epu_label}" if (epu_label != "us" and regime_key not in ("normal", "high_epu")) else regime_key,
            factors=factors,
            detail=" | ".join(detail_parts) if detail_parts else "neutral",
            flat_market=flat_market,
            low_confidence=low_confidence or final_low_confidence,
        )

    def compute_rolling_backtest(
        self,
        market: str,
        prices: pd.Series,
        epu_annual: pd.Series,
        window_years: int = 5,
    ) -> Optional[float]:
        """Compute directional accuracy over the most recent window_years.

        Re-fits the 3-factor model on only the last window_years of data
        to assess whether the full-history backtest is masking recent
        performance degradation.

        Returns accuracy as a float (0-1), or None if insufficient data.
        """
        if market not in self._models:
            return None

        if prices is None or prices.empty:
            return None

        daily_r = prices.pct_change(fill_method=None)
        # 阶段2 (月频): same monthly aggregation + annualization as fit_market,
        # restricted to the most recent window_years (converted to months).
        monthly_r = daily_r.resample("ME").apply(lambda x: np.prod(1 + x) - 1)
        monthly_r = (monthly_r * MONTHLY_ANNUALIZATION).dropna()
        months = max(int(window_years * 12), 1)
        if len(monthly_r) < months:
            return None

        # Slice to most recent window_years (in months)
        recent_monthly = monthly_r.iloc[-months:]

        cfg = self._models[market]
        epu_label = self._market_epu_label.get(market, "us")
        epu_thresh = self._epu_threshold.get(epu_label)
        if epu_thresh is None:
            return None

        tm = trend_momentum(prices).resample("ME").mean().reindex(monthly_r.index)
        mr = mean_reversion(prices).resample("ME").mean().reindex(monthly_r.index)
        vr = volatility_regime(prices).resample("ME").mean().reindex(monthly_r.index)
        fwd = monthly_r.shift(-1).dropna()

        valid = tm.dropna().index.intersection(mr.dropna().index).intersection(
            vr.dropna().index).intersection(fwd.index).intersection(epu_annual.dropna().index)
        valid = valid.intersection(recent_monthly.index)

        if len(valid) < 4:
            return None

        tm_v = tm.loc[valid]
        mr_v = mr.loc[valid]
        vr_v = vr.loc[valid]
        fwd_v = fwd.loc[valid]
        epu_v = epu_annual.loc[valid]

        high_mask = epu_v > epu_thresh
        preds = []
        actuals = []

        for i in range(len(valid)):
            regime_key = "high_epu" if high_mask.iloc[i] else "normal"
            if regime_key not in cfg["regime_models"]:
                continue
            m = cfg["regime_models"][regime_key]["model"]
            x = np.array([[tm_v.iloc[i], mr_v.iloc[i], vr_v.iloc[i]]])
            preds.append(float(m.predict(x)[0]))
            actuals.append(float(fwd_v.iloc[i]))

        if len(preds) < 3:
            return None

        acc = float((np.sign(preds) == np.sign(actuals)).mean())
        return round(acc, 4)

    def predict_all(
        self,
        market_prices: Dict[str, pd.Series],
        epu_value: float,
        epu_values: Dict[str, float] = None,
        cycle_phase: str = "unknown",
    ) -> List[MarketPrediction]:
        """Generate predictions for all fitted markets.

        Args:
            market_prices: Market → price series mapping.
            epu_value:     Default US EPU value.
            epu_values:    Per-label EPU values (e.g. {'us': 350, 'china': 376}).
                           Falls back to epu_value if a label is not found.
            cycle_phase:   Accepted for backward compatibility; 阶段2 removed
                           the cycle-gate override so it no longer affects
                           the prediction.
        """
        if epu_values is None:
            epu_values = {"us": epu_value}
        results = []
        for market in sorted(self._models.keys()):
            px = market_prices.get(market)
            if px is None:
                continue
            epu_label = self._market_epu_label.get(market, "us")
            epu_val = epu_values.get(epu_label, epu_value)
            pred = self.predict(market, px, epu_val, cycle_phase=cycle_phase)
            results.append(pred)
        results.sort(key=lambda x: -x.confidence)
        return results


__all__ = ["MarketPrediction", "HighConfidencePredictor",
           "EMERGING_MARKETS", "DEVELOPED_MARKETS", "GROUP_EPU_THRESHOLDS",
           "FLAT_MARKET_THRESHOLD",
           "LOW_CONVICTION_RET_THRESHOLD", "LOW_CONVICTION_Z_THRESHOLD",
           "CURRENT_FACTOR_WINDOW_DAYS",
           "TRADING_DAYS_PER_MONTH", "MONTHLY_ANNUALIZATION"]
