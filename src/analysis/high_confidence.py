"""
High-confidence predictor.

Only operates on markets with proven directional accuracy >= 70%.
Applies EPU regime-switching: separate 3-factor models (trend_momentum,
mean_reversion, volatility_regime) for EPU-normal and EPU-high periods.

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
    micro_momentum,
    intraweek_volatility,
    gap_signal,
    close_momentum,
    close_location,
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

# Overheat thresholds (z-score)
OVERHEAT_NEUT_THRESHOLD = 2.5   # factor > +2.5σ → force NEUT
OVERHEAT_BEAR_THRESHOLD = 3.5   # factor > +3.5σ → force BEAR

# Flat market threshold (P0: 2026W29)
FLAT_MARKET_THRESHOLD = 0.005   # |10d return| < 0.5% → force NEUT

# Low-conviction thresholds (P1: 2026W30, tuned 2026W32)
# When predicted return magnitude and all factor z-scores are
# below these thresholds, BULL/BEAR → NEUT (signal quality gating).
# 2026W32: return bar 15% → 8% annual.  The 15% bar sat above the median
# predicted return (median ≈ 12%), so it zeroed most valid BULL signals
# (W31 ended 13/13 NEUT).  8-15% annual expected returns are meaningful.
LOW_CONVICTION_RET_THRESHOLD = 0.08    # |pred_ret| < 8% annual → low conviction
LOW_CONVICTION_Z_THRESHOLD = 1.5        # all |z-score| < 1.5 → no clear signal

# Regime (regime) detection via Mahalanobis distance (P3: 2026W30)
# Measures how far current 3-factor vector is from training distribution.
# Larger distance → more unfamiliar regime → lower confidence / force NEUT.
REGIME_DISTANCE_WARN = 1.5    # D > 1.5 → lower confidence, mark "unfamiliar regime"
# 2026W32: force-NEUT bar 2.5σ → 3.5σ.  At 2.5σ the rule forced NEUT on the
# strongest-trend markets (KR D=3.9, JP D=3.2 both rose the following week).
# D in (2.5, 3.5] now only discounts confidence.
REGIME_DISTANCE_NEUT = 3.5    # D > 3.5 → force NEUT, mark "extreme regime shift"

# Current-factor window (2026W32): use trailing 252 trading days instead of
# the current partial calendar year for the "now" factor values.  Calendar-YTD
# means lag intra-year reversals (HSI: daily tm +0.32 vs YTD mean -0.06 → the
# July +12% rally was invisible to the model).  Trailing-1y keeps the scale
# consistent with the annual training means while reacting to reversals.
CURRENT_FACTOR_WINDOW_DAYS = 252

# Micro-period factors (2026W33): 3-7 day horizon signals to catch short-term
# turns the 20/60/120-day trend model misses.  W30-W31 hit 22-27% in P99 EPU
# because the model kept predicting trend continuation while markets turned;
# these factors are the weekly-timing layer on top of the annual model.
#
# They are used ONLY as a confidence modulator, never a signal veto: when the
# micro score agrees with the model's direction confidence gets a small boost,
# when it disagrees confidence is discounted.  A strong trend still survives a
# weak micro cross-current (no NEUT forcing — W32's lesson).
MICRO_FACTOR_WINDOW_DAYS = 60      # trailing mean window for micro factors
MICRO_ALIGN_THRESHOLD = 0.04       # |micro_score| needed to trigger modulation
MICRO_ALIGN_BOOST = 0.06           # aligned → confidence ×1.06
MICRO_CONFLICT_DISCOUNT = 0.10     # conflicting → confidence ×0.90

# Intraday micro-factors with directional power (2026W34).
# close_momentum / close_location are the first micro-factors with verified
# directional IC (full-period +0.151 / +0.102, 2024+ +0.134 / +0.120) and
# near-zero correlation with the momentum family, so they may influence
# DIRECTION — but only on hard evidence and only in weak states:
#   - both z-scores must agree and clear MICRO_EVIDENCE_Z_THRESHOLD;
#   - NEUT may be lifted to a weak BULL/BEAR (MICRO_EVIDENCE_LIFT_RET annual);
#   - a weak BULL/BEAR (all 3-factor |z| < 1.5 and |pred| <
#     MICRO_EVIDENCE_STRONG_RET) is pulled to NEUT when both oppose;
#   - strong 3-factor signals (any |z| >= 1.5 or |pred| >= STRONG_RET) are
#     never touched.
# BULL↔BEAR flips remain structurally impossible.
MICRO_EVIDENCE_Z_THRESHOLD = 1.0   # each |cm_z|, |cl_z| must clear this
MICRO_EVIDENCE_LIFT_RET = 0.06     # NEUT → weak directional expected annual
MICRO_EVIDENCE_STRONG_RET = 0.12   # |pred| >= this → strong, never touched

# Risk-off VIX spike detection (2026W34).
# A fast VIX spike is a directional risk-off signal at ANY level.  The old
# logic only flagged "rising VIX" when VIX was already > P80, so a +11%
# 5-day spike from a calm base (VIX 15.8 < P80 24.2) was invisible — and
# W34's 6 BULL calls all reversed (0/6).  Calibrated against VIX history
# (through 2026-05-22): 5-day change P80 = +9.3%, P85 = +12.1%, P90 = +16.8%;
# a +10% 5-day spike occurs on ~15% of sub-P80 ("calm") days, so it is a
# real but not rare condition.  We discount surviving BULL confidence ×0.70
# (directional, NOT a NEUT veto): a hard veto at +5% would trigger on ~27%
# of calm days and re-import W31's 13/13 over-suppression.
VIX_RISK_OFF_5D_THRESHOLD = 0.10   # 5-day VIX change > +10% → risk-off
VIX_RISK_OFF_BULL_DISCOUNT = 0.70  # BULL confidence ×0.70 during risk-off



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
    overheat: bool = False
    flat_market: bool = False
    vix_epu_conflict: bool = False
    low_confidence: bool = False
    rolling_accuracy: Optional[float] = None
    regime_distance: Optional[float] = None
    regime_shift: bool = False
    regime_unfamiliar: bool = False
    micro_effect: Optional[str] = None   # 'align' | 'conflict' | None (2026W33)
    micro_evidence: Optional[str] = None # 'lift' | 'downgrade' | None (2026W34)


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
        # VIX support
        self._vix_history: Optional[np.ndarray] = None
        self._vix_p80: Optional[float] = None
        self._vix_p20: Optional[float] = None
        self._vix_now: Optional[float] = None
        self._vix_trend_5d: Optional[float] = None   # 5-day change (fraction)
        self._vix_trend_20d: Optional[float] = None  # 20-day change (fraction)
        self._vix_active: bool = False
        self._vix_median: float = 20.0
        # Sentiment / flow factor (P2: 2026W30)
        self._pcr_history: Optional[pd.Series] = None   # put/call ratio daily series
        self._spread_history: Optional[pd.Series] = None  # 10Y-2Y spread daily series
        self._pcr_p80: Optional[float] = None
        self._pcr_p20: Optional[float] = None
        self._sentiment_active: bool = False
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

    def fit_sentiment(self, pcr_df: pd.DataFrame = None, spread_df: pd.DataFrame = None):
        """Fit sentiment/flow data as optional 4th factor.

        Args:
            pcr_df:    DataFrame with 'put_call_ratio' column, date-indexed.
            spread_df: DataFrame with 'spread_10y2y' column, date-indexed.
        """
        active = False

        if pcr_df is not None and not pcr_df.empty:
            try:
                pcr_vals = pcr_df["put_call_ratio"].dropna()
                if len(pcr_vals) >= 252:
                    self._pcr_history = pcr_vals
                    self._pcr_p80 = float(np.percentile(pcr_vals, 80))
                    self._pcr_p20 = float(np.percentile(pcr_vals, 20))
                    active = True
                    current_pcr = float(pcr_vals.iloc[-1])
                    logger.info(
                        f"Sentiment PCR fitted: now={current_pcr:.3f} "
                        f"P80={self._pcr_p80:.3f} P20={self._pcr_p20:.3f}"
                    )
            except Exception as exc:
                logger.warning(f"Sentiment PCR fit failed: {exc}")

        if spread_df is not None and not spread_df.empty:
            try:
                spread_vals = spread_df["spread_10y2y"].dropna()
                if len(spread_vals) >= 252:
                    self._spread_history = spread_vals
                    active = True
                    current_spread = float(spread_vals.iloc[-1])
                    logger.info(
                        f"Sentiment spread fitted: now={current_spread:+.3%}"
                    )
            except Exception as exc:
                logger.warning(f"Sentiment spread fit failed: {exc}")

        self._sentiment_active = active
        if active:
            logger.info("Sentiment/flow factor ACTIVE")
        else:
            logger.info("Sentiment/flow factor NOT active (insufficient data)")

    def _compute_sentiment_factor(self) -> Tuple[float, str]:
        """Compute sentiment/flow z-score from PCR and yield spread.

        Returns (z_score, direction_label).
          direction_label: 'BULLISH', 'BEARISH', or 'NEUTRAL'
          z_score > 0 → bullish tailwind, z_score < 0 → bearish headwind

        If sentiment data is inactive or unavailable, returns (0.0, 'N/A').
        """
        if not self._sentiment_active:
            return 0.0, "N/A"

        signals = []

        # Put/call ratio signal
        if self._pcr_history is not None and self._pcr_p80 is not None:
            current_pcr = float(self._pcr_history.iloc[-1])
            if current_pcr > self._pcr_p80:
                signals.append(-1.0)  # High PCR = fear = bearish
            elif current_pcr < self._pcr_p20:
                signals.append(1.0)   # Low PCR = greed = bullish
            else:
                signals.append(0.0)

        # Yield spread signal
        if self._spread_history is not None:
            current_spread = float(self._spread_history.iloc[-1])
            if current_spread < 0:
                signals.append(-1.0)  # Inverted = recession signal = bearish
            elif current_spread > 0.015:  # > 1.5% = steep = growth signal
                signals.append(1.0)       # Normal steep curve = bullish
            else:
                signals.append(0.0)

        if not signals:
            return 0.0, "N/A"

        z_score = float(np.mean(signals))  # -1 to +1 scale

        if z_score > 0.2:
            direction = "BULLISH"
        elif z_score < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return z_score, direction

    # ── Market config ───────────────────────────────────────────
    def set_market_params(
        self,
        market: str,
        ar_coeff: float = None,
        signal_threshold: float = None,
        epu_label: str = None,
    ):
        """Set per-market model parameters.

        Args:
            market:           Market identifier (e.g. 'HK', 'CN').
            ar_coeff:         AR(1) decay coefficient (default 0.75).
                              Lower = faster mean-reversion (EM markets).
            signal_threshold: Return cutoff for BULL/BEAR signal (default 0.03).
                              Higher = more conservative signals.
            epu_label:        Which EPU to use for regime switching ('us', 'china').
        """
        if ar_coeff is not None:
            self._market_ar_coeffs[market] = ar_coeff
        if signal_threshold is not None:
            self._market_signal_threshold[market] = signal_threshold
        if epu_label is not None:
            self._market_epu_label[market] = epu_label

    # ── Regime detection ────────────────────────────────────────
    def _epu_threshold_for_market(self, market: str) -> float:
        """EPU percentile threshold for this market group."""
        group = _market_group(market)
        return self.group_thresholds.get(group, self.epu_threshold_pct)

    def _is_high_epu(self, epu_value: float, label: str = "us") -> bool:
        if label not in self._epu_threshold:
            return False
        return epu_value > self._epu_threshold[label]

    def _get_regime(self, market: str, epu_value: float, epu_label: str) -> Tuple[str, str, bool, bool]:
        """Determine regime for a market, considering EPU + VIX.

        Returns (regime_key, source, conflict, vix_risk_off) where:
          regime_key: 'high_epu', 'normal', 'high_epu_vix', etc.
          source: 'epu', 'vix', 'vix_rising', 'vix_low', 'epu+matched'
          conflict: True when VIX and EPU disagree on regime direction
                    (VIX > P80 but EPU normal, or VIX < P20 but EPU high).
                    When True, the caller should downgrade to NEUT for safety.
          vix_risk_off: True when VIX is spiking fast (> +10% / 5d) at ANY
                    level — a directional risk-off signal that discounts
                    surviving BULL confidence (2026W34), not a NEUT veto.
        """
        is_high = self._is_high_epu(epu_value, epu_label)
        base = f"high_epu_{epu_label}" if (is_high and epu_label != "us") else \
               "high_epu" if is_high else \
               f"normal_{epu_label}" if epu_label != "us" else "normal"

        conflict = False
        vix_risk_off = False

        # VIX override if available: VIX > P80 = high uncertainty
        if self._vix_active and self._vix_now is not None:
            # 2026W34: fast-rising VIX is a risk-off signal at ANY level.
            # Previously rising VIX was only flagged once VIX had already
            # crossed P80, so a +11% 5-day spike from a calm base was ignored.
            # Hoisted above the level gate.
            if (self._vix_trend_5d is not None
                    and self._vix_trend_5d > VIX_RISK_OFF_5D_THRESHOLD):
                vix_risk_off = True

            vix_says_high = self._vix_now > self._vix_p80
            vix_says_low = self._vix_now < self._vix_p20

            if vix_says_high:
                # VIX says high uncertainty
                if self._vix_trend_5d is not None and self._vix_trend_5d > 0.05:
                    regime_key = "high_epu_vix_rising"
                    source = "vix_rising"
                else:
                    regime_key = "high_epu_vix"
                    source = "vix"
                # P1: Conflict check — VIX says high but EPU says normal
                if not is_high:
                    conflict = True
            elif vix_says_low:
                # P1: Conflict check — VIX says low but EPU says high
                if is_high:
                    conflict = True
                    regime_key = base  # use EPU regime for model selection
                    source = "epu"
                else:
                    regime_key = "normal_vix"
                    source = "vix_low"
            else:
                # VIX in middle zone — trust EPU, but detect EPU/VIX disconnect
                regime_key = base
                if vix_risk_off:
                    # Fast spike, calm level: keep the EPU model but trace the
                    # risk-off source (model selection unchanged).
                    source = "vix_rising"
                elif is_high == vix_says_high:
                    source = "epu+matched"
                else:
                    source = "epu"
                if is_high and hasattr(self, "_vix_median") and self._vix_now < self._vix_median:
                    conflict = True
        else:
            regime_key = base
            source = "epu"

        return regime_key, source, conflict, vix_risk_off

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

        Args:
            market:     Market identifier.
            prices:     Daily price series.
            epu_annual: Annual mean EPU values (US or China).
            epu_label:  Which EPU to use for regime ('us' or 'china').
                        If None, auto-assigned from _market_epu_label or default 'us'.
            ohlc:       Optional daily OHLC frame used to fit the 2026W34
                        intraday micro-factor distributions (close_momentum /
                        close_location).  When omitted these stay neutral
                        (mean 0, std 1) so z-scores read 0 — legacy callers
                        that pass close prices only are unaffected.

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
        annual_r = daily_r.resample("YE").apply(lambda x: np.prod(1 + x) - 1).dropna()
        if len(annual_r) < 10:
            return None

        tm = trend_momentum(prices).resample("YE").mean().reindex(annual_r.index)
        mr = mean_reversion(prices).resample("YE").mean().reindex(annual_r.index)
        vr = volatility_regime(prices).resample("YE").mean().reindex(annual_r.index)
        # Micro-period factors (2026W33): annual means only — they modulate
        # confidence, so they must NOT enter the regression training set or
        # the regime-distance distribution (keeps the 3-factor model intact).
        mm = micro_momentum(prices).resample("YE").mean().reindex(annual_r.index)
        iv = intraweek_volatility(prices).resample("YE").mean().reindex(annual_r.index)
        gs = gap_signal(prices).resample("YE").mean().reindex(annual_r.index)
        # Intraday micro-factors (2026W34): annual means from the OHLC frame.
        # Same convention as mm/iv/gs — distribution for z-scores only, never
        # entering the regression model.
        cm = close_momentum(ohlc).resample("YE").mean().reindex(annual_r.index) if ohlc is not None else pd.Series(0.0, index=annual_r.index)
        cl = close_location(ohlc).resample("YE").mean().reindex(annual_r.index) if ohlc is not None else pd.Series(0.0, index=annual_r.index)
        fwd = annual_r.shift(-1).dropna()

        valid = tm.dropna().index.intersection(mr.dropna().index).intersection(
            vr.dropna().index).intersection(fwd.index).intersection(epu_annual.dropna().index)
        if len(valid) < 10:
            return None

        tm_v = tm.loc[valid]
        mr_v = mr.loc[valid]
        vr_v = vr.loc[valid]
        mm_v = mm.loc[valid]
        iv_v = iv.loc[valid]
        gs_v = gs.loc[valid]
        cm_v = cm.loc[valid]
        cl_v = cl.loc[valid]
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

        if overall_acc < self.min_confidence:
            logger.info(
                f"  {market}: accuracy {overall_acc:.0%} < "
                f"{self.min_confidence:.0%} threshold — skipping"
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
            # Micro-factor training distribution (2026W33) — for display
            # z-scores only; never used in the regression model.
            "mm_mean": float(mm_v.mean()) if mm_v.notna().any() else 0.0,
            "mm_std": float(mm_v.std()) if mm_v.notna().sum() >= 2 else 1.0,
            "iv_mean": float(iv_v.mean()) if iv_v.notna().any() else 0.0,
            "iv_std": float(iv_v.std()) if iv_v.notna().sum() >= 2 else 1.0,
            "gs_mean": float(gs_v.mean()) if gs_v.notna().any() else 0.0,
            "gs_std": float(gs_v.std()) if gs_v.notna().sum() >= 2 else 1.0,
            "cm_mean": float(cm_v.mean()) if cm_v.notna().any() else 0.0,
            "cm_std": float(cm_v.std()) if cm_v.notna().sum() >= 2 else 1.0,
            "cl_mean": float(cl_v.mean()) if cl_v.notna().any() else 0.0,
            "cl_std": float(cl_v.std()) if cl_v.notna().sum() >= 2 else 1.0,
            "epu_threshold": epu_thresh,
            "group": _market_group(market),
            "group_pct": group_pct,
        }
        # Store training factor distribution for regime distance detection
        X_train = np.column_stack([tm_v, mr_v, vr_v])
        self._factor_means[market] = X_train.mean(axis=0)
        cov = np.cov(X_train.T)
        cov += 1e-6 * np.eye(3)  # regularization for numerical stability
        self._factor_cov[market] = cov
        self._models[market] = result
        return result

    # ── Overheat check ──────────────────────────────────────────
    def _check_overheat(self, factors: Dict[str, float]) -> Tuple[bool, Optional[int], str]:
        """Check factor z-scores for overheat conditions.

        Returns (triggered, override_signal, reason).

        Rules:
          - Any factor z-score > +3.5σ → BEAR (extreme overheat, reversal imminent)
          - Any factor z-score > +2.5σ → NEUT (overheat, trend too extended)
          - Otherwise → no override

        These rules take priority over regime-based prediction to prevent
        bubble-chasing in overheated markets.
        """
        tm_z = factors.get("tm_z", 0)
        mr_z = factors.get("mr_z", 0)
        vr_z = factors.get("vr_z", 0)

        max_z = max(tm_z, mr_z, vr_z)
        factor_names = []
        if tm_z > OVERHEAT_BEAR_THRESHOLD:
            factor_names.append(f"tm={tm_z:.1f}σ")
        if mr_z > OVERHEAT_BEAR_THRESHOLD:
            factor_names.append(f"mr={mr_z:.1f}σ")
        if vr_z > OVERHEAT_BEAR_THRESHOLD:
            factor_names.append(f"vr={vr_z:.1f}σ")

        if factor_names:
            return True, -1, f"🔥 EXTREME OVERHEAT ({', '.join(factor_names)} > +3.5σ) → BEAR"

        factor_names = []
        if tm_z > OVERHEAT_NEUT_THRESHOLD:
            factor_names.append(f"tm={tm_z:.1f}σ")
        if mr_z > OVERHEAT_NEUT_THRESHOLD:
            factor_names.append(f"mr={mr_z:.1f}σ")
        if vr_z > OVERHEAT_NEUT_THRESHOLD:
            factor_names.append(f"vr={vr_z:.1f}σ")

        if factor_names:
            return True, 0, f"⚠ OVERHEAT ({', '.join(factor_names)} > +2.5σ) → NEUT"

        return False, None, ""

    # ── Flat market check (P0: 2026W29) ─────────────────────────
    def _check_flat_market(self, prices: pd.Series) -> Tuple[bool, str]:
        """Check if the market has been effectively flat recently.

        When |10-day return| < FLAT_MARKET_THRESHOLD, the market shows
        negligible directional movement.  BULL/BEAR signals in such
        conditions are noise rather than actionable predictions.

        Returns (is_flat, reason).
        """
        if prices is None or len(prices) < 11:
            return False, ""
        recent_ret = float(prices.iloc[-1] / prices.iloc[-11] - 1)
        if abs(recent_ret) < FLAT_MARKET_THRESHOLD:
            return True, f"平盘 (|10d ret|={recent_ret:.2%} < {FLAT_MARKET_THRESHOLD:.1%}) → NEUT"
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

    def compute_regime_distance(
        self, market: str, tm: float, mr: float, vr: float
    ) -> Optional[float]:
        """Mahalanobis distance of current factors from training distribution.

        Returns None if training distribution is unavailable for this market.
        Falls back to normalized Euclidean distance if covariance is singular.
        """
        if market not in self._factor_means or market not in self._factor_cov:
            return None
        mean = self._factor_means[market]
        cov = self._factor_cov[market]
        point = np.array([tm, mr, vr])
        diff = point - mean
        try:
            inv_cov = np.linalg.inv(cov)
            D2 = diff @ inv_cov @ diff
            return float(np.sqrt(max(D2, 0)))
        except np.linalg.LinAlgError:
            std = np.sqrt(np.diag(cov))
            std[std < 1e-10] = 1.0
            return float(np.sqrt(np.sum((diff / std) ** 2)) / np.sqrt(3))

    # ── Prediction ──────────────────────────────────────────────
    def predict(
        self,
        market: str,
        prices: pd.Series,
        epu_value: float,
        cycle_phase: str = "unknown",
        ohlc: Optional[pd.DataFrame] = None,
    ) -> MarketPrediction:
        """Generate prediction for a single market.

        Uses the appropriate regime model based on current EPU,
        with market-specific AR(1) decay and signal thresholds.
        Applies overheat downgrade rules after model prediction.

        ``ohlc`` is optional; when omitted the 2026W34 intraday factors
        (cm/cl) stay neutral (z = 0) so behavior matches legacy callers.
        """
        if market not in self._models:
            return MarketPrediction(
                market=market, signal=0, expected_ret=0.0,
                confidence=0.0, tier=3, regime="unknown",
                detail="not fitted",
            )

        cfg = self._models[market]
        epu_label = self._market_epu_label.get(market, "us")
        regime_key, regime_source, vix_epu_conflict, vix_risk_off = self._get_regime(market, epu_value, epu_label)

        # Map vix-based regimes back to model regime keys
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

        # ── Micro-period factors (2026W33) ─────────────────────
        # 3-7 day horizon signals: 5-day momentum, intra-week vol ratio,
        # and the weekend-gap effect.  Current value = trailing 60-day mean
        # (MICRO_FACTOR_WINDOW_DAYS): the daily 5-day factors are far noisier
        # than the trend factors, so a short window captures live short-term
        # state while staying robust to single-day spikes.
        mm_daily = micro_momentum(prices)
        iv_daily = intraweek_volatility(prices)
        gs_daily = gap_signal(prices)
        mm_now = _trailing_mean(mm_daily, MICRO_FACTOR_WINDOW_DAYS)
        iv_now = _trailing_mean(iv_daily, MICRO_FACTOR_WINDOW_DAYS)
        gs_now = _trailing_mean(gs_daily, MICRO_FACTOR_WINDOW_DAYS)
        mm_z = (mm_now - cfg["mm_mean"]) / cfg["mm_std"] if cfg["mm_std"] > 0 else 0
        iv_z = (iv_now - cfg["iv_mean"]) / cfg["iv_std"] if cfg["iv_std"] > 0 else 0
        gs_z = (gs_now - cfg["gs_mean"]) / cfg["gs_std"] if cfg["gs_std"] > 0 else 0
        # Intraday micro-factors (2026W34): daily 5-day factors from the OHLC
        # frame, trailing-60d mean (same convention as mm/iv/gs), z-scored
        # against the annual distribution stored by fit_market.
        if ohlc is not None:
            cm_daily = close_momentum(ohlc)
            cl_daily = close_location(ohlc)
        else:
            cm_daily = pd.Series(0.0, index=prices.index)
            cl_daily = pd.Series(0.0, index=prices.index)
        cm_now = _trailing_mean(cm_daily, MICRO_FACTOR_WINDOW_DAYS)
        cl_now = _trailing_mean(cl_daily, MICRO_FACTOR_WINDOW_DAYS)
        cm_z = (cm_now - cfg["cm_mean"]) / cfg["cm_std"] if cfg["cm_std"] > 0 else 0
        cl_z = (cl_now - cfg["cl_mean"]) / cfg["cl_std"] if cfg["cl_std"] > 0 else 0
        # Combined micro score (2026W33 formula, restored 2026W34): mm leads,
        # gap captures weekend news, iv is an amplifier.  The 2026W34 intraday
        # factors (cm/cl) are deliberately EXCLUDED from the blend: they live
        # on a different raw scale (daily returns vs ratios) so weighting them
        # diluted the W33 factors, and their directional power is applied only
        # through the evidence gate below — never through the confidence
        # modulator.
        micro_score = round(0.50 * mm_now + 0.30 * gs_now + 0.20 * iv_now, 4)

        # Market-specific AR(1) decay coefficient
        ar_coeff = self._market_ar_coeffs.get(market, 0.75)
        tm_f = cfg["tm_mean"] + ar_coeff * (tm_now - cfg["tm_mean"])
        mr_f = cfg["mr_mean"] + ar_coeff * (mr_now - cfg["mr_mean"])
        vr_f = cfg["vr_mean"] + ar_coeff * (vr_now - cfg["vr_mean"])
        pred = float(m.predict([[tm_f, mr_f, vr_f]])[0])

        # Market-specific signal threshold
        sig_thresh = self._market_signal_threshold.get(market, 0.03)
        signal = 1 if pred > sig_thresh else (-1 if pred < -sig_thresh else 0)
        tier = 1 if cfg["accuracy"] >= 0.70 else (2 if cfg["accuracy"] >= 0.55 else 3)

        factors = {
            "tm": round(tm_now, 4), "tm_z": round(tm_z, 2),
            "mr": round(mr_now, 4), "mr_z": round(mr_z, 2),
            "vr": round(vr_now, 4), "vr_z": round(vr_z, 2),
            "mm": round(mm_now, 4), "mm_z": round(mm_z, 2),
            "iv": round(iv_now, 4), "iv_z": round(iv_z, 2),
            "gs": round(gs_now, 4), "gs_z": round(gs_z, 2),
            "cm": round(cm_now, 4), "cm_z": round(cm_z, 2),
            "cl": round(cl_now, 4), "cl_z": round(cl_z, 2),
            "micro_score": micro_score,
        }

        # ── Regime (regime) distance check (P3: 2026W30) ──
        regime_dist = self.compute_regime_distance(market, tm_now, mr_now, vr_now)
        regime_shift = False
        regime_unfamiliar = False
        confidence = cfg["accuracy"]
        micro_effect = None   # 2026W33: 'align' | 'conflict' | None
        micro_evidence = None  # 2026W34: 'lift' | 'downgrade' | None

        if regime_dist is not None and regime_dist > REGIME_DISTANCE_NEUT:
            regime_shift = True
            logger.info(
                f"  {market}: regime shift (D={regime_dist:.1f} > "
                f"{REGIME_DISTANCE_NEUT}) — forcing NEUT"
            )
            detail_parts = [f"⚠ 制度极端偏离(D={regime_dist:.1f}σ)→NEUT"]
            # Build regime description for detail
            if "vix_rising" in regime_key:
                detail_parts.append("VIX急升→不确定性加剧")
            elif "vix" in regime_key:
                detail_parts.append("VIX高位→高不确定")
            elif "high_epu" in regime_key:
                epu_src = "中EPU高" if epu_label == "china" else "EPU高"
                detail_parts.append(f"{epu_src}→高政策不确定性")
            else:
                epu_src = "中EPU正常" if epu_label == "china" else "EPU正常"
                detail_parts.append(f"{epu_src}→均值回归模式")
            return MarketPrediction(
                market=market,
                signal=0,
                expected_ret=round(float(pred), 4),
                confidence=round(confidence, 4),
                tier=tier,
                regime=f"{regime_key}_{epu_label}" if (epu_label != "us" and regime_key not in ("normal", "high_epu")) else regime_key,
                factors=factors,
                detail=" | ".join(detail_parts),
                regime_distance=regime_dist,
                regime_shift=True,
                regime_unfamiliar=False,
            )
        elif regime_dist is not None and regime_dist > REGIME_DISTANCE_WARN:
            regime_unfamiliar = True
            confidence = confidence * (REGIME_DISTANCE_WARN / regime_dist)
            logger.info(
                f"  {market}: unfamiliar regime (D={regime_dist:.1f}) — "
                f"confidence: {cfg['accuracy']:.0%} → {confidence:.0%}"
            )

        # ── Overheat check (overrides model signal) ──────────
        overheated, over_signal, over_reason = self._check_overheat(factors)
        if overheated:
            original_signal = signal
            signal = over_signal
            logger.info(
                f"  {market}: overheat override "
                f"({['BEAR','NEUT','BULL'][original_signal + 1]} → "
                f"{['BEAR','NEUT','BULL'][signal + 1]}) — {over_reason}"
            )

        # ── P1: VIX-EPU conflict check (2026W32: discount, not veto) ──
        # A calm VIX while monthly EPU reads P99 means the lagging EPU
        # overstates risk.  The signal survives at reduced confidence;
        # zeroing it (W31 behavior) silenced 11/13 markets.  ×0.85 is a
        # visible warning — heavier discounts stack with the regime-distance
        # haircut and suppress most markets (W31: 6 markets dropped below
        # the 55% publication bar for one global VIX/EPU condition).
        conflict_override = False
        if vix_epu_conflict and signal != 0:
            original_signal = signal
            conflict_override = True
            confidence *= 0.85
            logger.info(
                f"  {market}: VIX-EPU conflict — confidence "
                f"{cfg['accuracy']:.0%} → {confidence:.0%} "
                f"(VIX={self._vix_now:.1f} vs EPU={epu_value:.0f} disagree)"
            )

        # ── P0: Flat market check (overrides BULL/BEAR to NEUT) ──
        flat_market = False
        if not overheated and signal != 0:
            is_flat, flat_reason = self._check_flat_market(prices)
            if is_flat:
                original_signal = signal
                signal = 0
                flat_market = True
                logger.info(
                    f"  {market}: flat market override "
                    f"({['BEAR','NEUT','BULL'][original_signal + 1]} → NEUT) — {flat_reason}"
                )

        # ── Sentiment/flow regulator (4th factor) ────────────
        sentiment_direction = "N/A"
        sentiment_overridden = False
        if self._sentiment_active and signal != 0:
            sz, sd = self._compute_sentiment_factor()
            sentiment_direction = sd
            model_dir = 1 if signal > 0 else -1
            sent_dir = 1 if sz > 0.2 else (-1 if sz < -0.2 else 0)
            if sent_dir != 0 and sent_dir != model_dir:
                original_signal = signal
                signal = 0
                sentiment_overridden = True
                logger.info(
                    f"  {market}: sentiment regulator override "
                    f"({['BEAR','NEUT','BULL'][original_signal + 1]} → NEUT) — "
                    f"sentiment={sd} vs model={'BULL' if model_dir > 0 else 'BEAR'}"
                )

        # ── Cycle phase (2026W32: informational, not a veto) ──
        # W31 evidence: the stagflation label (US PMI 48.7, CPI 3.0%) is a
        # coarse US-macro read applied globally; all 7 BULL signals it
        # suppressed in W31 rose the same week (+0.2%..+1.6%).  Removing it
        # would have scored 7/9 (78%) on W31 vs 0 actionable signals.
        cycle_tilt_applied = False

        # ── P1: Low-conviction check (overrides weak BULL/BEAR → NEUT) ──
        low_confidence = False
        if not overheated and not flat_market and not sentiment_overridden and signal != 0 and not cycle_tilt_applied:
            signal, low_confidence = self._check_low_conviction(pred, factors, signal, market)

        # ── Micro-factor confidence modulator (2026W33) ────────
        # Short-term (3-7 day) signals adjust confidence in the model's
        # direction.  Alignment → small boost; divergence → discount.
        # Never flips the signal and never forces NEUT: a strong trend
        # survives a weak micro cross-current (W32 de-blanketing lesson).
        if signal != 0 and not overheated and not sentiment_overridden:
            if micro_score > MICRO_ALIGN_THRESHOLD:
                micro_effect = "align" if signal > 0 else "conflict"
            elif micro_score < -MICRO_ALIGN_THRESHOLD:
                micro_effect = "align" if signal < 0 else "conflict"
            if micro_effect == "align":
                confidence = min(confidence * (1.0 + MICRO_ALIGN_BOOST), 0.99)
                logger.info(
                    f"  {market}: micro-factor alignment (score={micro_score:+.3f}) "
                    f"— confidence {cfg['accuracy']:.0%} → {confidence:.0%}"
                )
            elif micro_effect == "conflict":
                confidence = confidence * (1.0 - MICRO_CONFLICT_DISCOUNT)
                logger.info(
                    f"  {market}: micro-factor divergence (score={micro_score:+.3f}) "
                    f"— confidence {cfg['accuracy']:.0%} → {confidence:.0%}"
                )

        # ── Intraday micro-factor evidence gate (2026W34) ─────
        # close_momentum / close_location carry VERIFIED directional IC, so on
        # hard evidence they may influence direction — but only in weak states.
        # Both z-scores must agree and each clear MICRO_EVIDENCE_Z_THRESHOLD.
        #   - a weak NEUT (low-conviction / below-threshold |pred|) may be
        #     lifted to a weak BULL/BEAR (MICRO_EVIDENCE_LIFT_RET) — only in
        #     the direction of the model's own raw lean sign(pred), so the
        #     evidence corroborates a weak view, never invents or flips one;
        #   - a weak BULL/BEAR (all 3-factor |z| < 1.5 AND |pred| <
        #     MICRO_EVIDENCE_STRONG_RET) → NEUT when evidence opposes.
        # Strong 3-factor signals are never touched, and a signal zeroed by
        # the overheat / flat / sentiment gates is not resurrected.  BULL↔BEAR
        # flips remain impossible.
        expected_ret = float(pred)
        if not overheated and not flat_market and not sentiment_overridden:
            cm_sign = 1 if cm_z >= MICRO_EVIDENCE_Z_THRESHOLD else (-1 if cm_z <= -MICRO_EVIDENCE_Z_THRESHOLD else 0)
            cl_sign = 1 if cl_z >= MICRO_EVIDENCE_Z_THRESHOLD else (-1 if cl_z <= -MICRO_EVIDENCE_Z_THRESHOLD else 0)
            if cm_sign != 0 and cm_sign == cl_sign:
                strong_z = max(abs(tm_z), abs(mr_z), abs(vr_z)) >= LOW_CONVICTION_Z_THRESHOLD
                strong_ret = abs(pred) >= MICRO_EVIDENCE_STRONG_RET
                pred_lean = 1 if pred > 0 else (-1 if pred < 0 else 0)
                if signal == 0 and not strong_z and not strong_ret and pred_lean != 0 and cm_sign == pred_lean:
                    signal = cm_sign
                    expected_ret = cm_sign * MICRO_EVIDENCE_LIFT_RET
                    low_confidence = False
                    micro_evidence = "lift"
                    logger.info(
                        f"  {market}: micro evidence lift (cm_z={cm_z:+.1f}σ "
                        f"cl_z={cl_z:+.1f}σ) → {'BULL' if signal > 0 else 'BEAR'}"
                    )
                elif signal != 0 and not strong_z and not strong_ret and cm_sign != (1 if signal > 0 else -1):
                    signal = 0
                    micro_evidence = "downgrade"
                    logger.info(
                        f"  {market}: micro evidence downgrade (cm_z={cm_z:+.1f}σ "
                        f"cl_z={cl_z:+.1f}σ oppose) → NEUT"
                    )

        # ── 2026W34: risk-off BULL discount ──────────────────
        # A fast VIX spike (> +10% / 5d) is a directional risk-off signal.
        # W34: VIX +11.2% in 5d (still < P80 24.2) while all 6 BULLs reversed
        # (0/6).  Discount surviving BULL confidence ×0.70 — enough to push
        # marginal BULLs under the 55% publication bar — but NOT a NEUT veto:
        # a hard veto would re-import W31's 13/13 over-suppression.
        vix_risk_off_applied = False
        if vix_risk_off and signal == 1:
            vix_risk_off_applied = True
            confidence *= VIX_RISK_OFF_BULL_DISCOUNT
            logger.info(
                f"  {market}: VIX risk-off (5d {self._vix_trend_5d:+.1%}) — "
                f"BULL confidence ×{VIX_RISK_OFF_BULL_DISCOUNT:.2f} → {confidence:.0%}"
            )

        # ── Build detail string ──────────────────────────────
        detail_parts = []
        if overheated:
            detail_parts.append(over_reason)
        if conflict_override:
            detail_parts.append(f"⚠ VIX-EPU冲突→置信度×0.85 (VIX={self._vix_now:.1f}, EPU={epu_value:.0f})")
        if vix_risk_off_applied:
            detail_parts.append(
                f"⚠ VIX急升(5d {self._vix_trend_5d:+.1%})→BULL置信度×{VIX_RISK_OFF_BULL_DISCOUNT:.2f}"
            )
        if flat_market:
            detail_parts.append(f"平盘→NEUT (|10d ret| < {FLAT_MARKET_THRESHOLD:.1%})")
        if sentiment_overridden:
            detail_parts.append(f"情绪{'-'.join(sentiment_direction.split('/'))}→NEUT")
        if low_confidence:
            detail_parts.append("低信念→NEUT")
        if cycle_tilt_applied:
            detail_parts.append(f"周期{cycle_phase}→NEUT")

        # Regime description (no longer defaulting to trend-continuation)
        if "vix_rising" in regime_key:
            detail_parts.append("VIX急升→不确定性加剧")
        elif "vix" in regime_key:
            detail_parts.append("VIX高位→高不确定")
        elif "high_epu" in regime_key:
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

        # Micro-factor modulation + descriptions (2026W33)
        if micro_effect == "align":
            detail_parts.append(f"微周期共振→置信度×{1.0 + MICRO_ALIGN_BOOST:.2f}")
        elif micro_effect == "conflict":
            detail_parts.append(f"微周期背离→置信度×{1.0 - MICRO_CONFLICT_DISCOUNT:.2f}")
        if abs(mm_z) >= 1.0 or abs(iv_z) >= 1.0 or abs(gs_z) >= 1.0:
            detail_parts.append(f"微: 动量{mm_z:+.1f}σ 波动{iv_z:+.1f}σ 跳空{gs_z:+.1f}σ")

        # Intraday micro-factor evidence (2026W34)
        if micro_evidence == "lift":
            detail_parts.append(f"微证据提升→{'BULL' if signal > 0 else 'BEAR'}")
        elif micro_evidence == "downgrade":
            detail_parts.append("微证据背离→NEUT")

        # Add VIX context when available
        if self._vix_active and self._vix_now is not None:
            if self._vix_trend_5d is not None and abs(self._vix_trend_5d) > 0.03:
                direction = "↓" if self._vix_trend_5d < 0 else "↑"
                detail_parts.append(f"VIX={self._vix_now:.1f}{direction}")

        # ── Regime detail for non-shift paths ──
        if regime_unfamiliar and regime_dist is not None:
            detail_parts.append(f"⚠ 陌生制度(D={regime_dist:.1f}σ)")

        # Tier from post-discount confidence (2026W32): a market whose
        # confidence was discounted below the publication bars must not
        # be presented as "Tier 1 actionable".
        if confidence >= 0.70:
            tier = 1
        elif confidence >= 0.55:
            tier = 2
        else:
            tier = 3

        return MarketPrediction(
            market=market,
            signal=signal,
            expected_ret=round(expected_ret, 4),
            confidence=round(confidence, 4),
            tier=tier,
            regime=f"{regime_key}_{epu_label}" if (epu_label != "us" and regime_key not in ("normal", "high_epu")) else regime_key,
            factors=factors,
            detail=" | ".join(detail_parts) if detail_parts else "neutral",
            overheat=overheated,
            flat_market=flat_market,
            vix_epu_conflict=conflict_override,
            low_confidence=low_confidence,
            regime_distance=round(regime_dist, 2) if regime_dist is not None else None,
            regime_shift=regime_shift,
            regime_unfamiliar=regime_unfamiliar,
            micro_effect=micro_effect,
            micro_evidence=micro_evidence,
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
        annual_r = daily_r.resample("YE").apply(lambda x: np.prod(1 + x) - 1).dropna()
        if len(annual_r) < window_years:
            return None

        # Slice to most recent window_years
        recent_annual = annual_r.iloc[-window_years:]

        cfg = self._models[market]
        epu_label = self._market_epu_label.get(market, "us")
        epu_thresh = self._epu_threshold.get(epu_label)
        if epu_thresh is None:
            return None

        tm = trend_momentum(prices).resample("YE").mean().reindex(annual_r.index)
        mr = mean_reversion(prices).resample("YE").mean().reindex(annual_r.index)
        vr = volatility_regime(prices).resample("YE").mean().reindex(annual_r.index)
        fwd = annual_r.shift(-1).dropna()

        valid = tm.dropna().index.intersection(mr.dropna().index).intersection(
            vr.dropna().index).intersection(fwd.index).intersection(epu_annual.dropna().index)
        valid = valid.intersection(recent_annual.index)

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
        ohlc: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[MarketPrediction]:
        """Generate predictions for all fitted markets.

        Args:
            market_prices: Market → price series mapping.
            epu_value:     Default US EPU value.
            epu_values:    Per-label EPU values (e.g. {'us': 350, 'china': 376}).
                           Falls back to epu_value if a label is not found.
            ohlc:          Optional Market → OHLC frame mapping for the
                           2026W34 intraday factors.  Omitted markets stay
                           neutral (cm/cl z = 0), preserving legacy behavior.
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
            market_ohlc = ohlc.get(market) if ohlc is not None else None
            pred = self.predict(market, px, epu_val, cycle_phase=cycle_phase, ohlc=market_ohlc)
            results.append(pred)
        results.sort(key=lambda x: -x.confidence)
        return results


__all__ = ["MarketPrediction", "HighConfidencePredictor",
           "EMERGING_MARKETS", "DEVELOPED_MARKETS", "GROUP_EPU_THRESHOLDS",
           "OVERHEAT_NEUT_THRESHOLD", "OVERHEAT_BEAR_THRESHOLD",
           "FLAT_MARKET_THRESHOLD",
           "LOW_CONVICTION_RET_THRESHOLD", "LOW_CONVICTION_Z_THRESHOLD",
           "REGIME_DISTANCE_WARN", "REGIME_DISTANCE_NEUT",
           "CURRENT_FACTOR_WINDOW_DAYS",
           "MICRO_FACTOR_WINDOW_DAYS", "MICRO_ALIGN_THRESHOLD",
           "MICRO_ALIGN_BOOST", "MICRO_CONFLICT_DISCOUNT",
           "MICRO_EVIDENCE_Z_THRESHOLD", "MICRO_EVIDENCE_LIFT_RET",
           "MICRO_EVIDENCE_STRONG_RET"]
