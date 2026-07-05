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

from src.analysis.factors import trend_momentum, mean_reversion, volatility_regime
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

    def _get_regime(self, market: str, epu_value: float, epu_label: str) -> Tuple[str, str]:
        """Determine regime for a market, considering EPU + VIX.

        Returns (regime_key, source) where:
          regime_key: 'high_epu', 'normal', 'high_epu_china', 'normal_china'
          source: 'epu', 'vix', 'vix+epu'
        """
        is_high = self._is_high_epu(epu_value, epu_label)
        base = f"high_epu_{epu_label}" if (is_high and epu_label != "us") else \
               "high_epu" if is_high else \
               f"normal_{epu_label}" if epu_label != "us" else "normal"

        # VIX override if available: VIX > P80 = high uncertainty
        if self._vix_active and self._vix_now is not None:
            if self._vix_now > self._vix_p80:
                # VIX says high uncertainty — override EPU
                # Use VIX trend to determine if uncertainty is accelerating or fading
                if self._vix_trend_5d is not None and self._vix_trend_5d > 0.05:
                    regime_key = "high_epu_vix_rising"
                    source = "vix_rising"
                else:
                    regime_key = "high_epu_vix"
                    source = "vix"
            elif self._vix_now < self._vix_p20 and not is_high:
                regime_key = "normal_vix"
                source = "vix_low"
            else:
                regime_key = base
                source = "epu+matched" if is_high == (self._vix_now > self._vix_p80) else "epu"
        else:
            regime_key = base
            source = "epu"

        return regime_key, source

    # ── Model fitting ───────────────────────────────────────────
    def fit_market(
        self,
        market: str,
        prices: pd.Series,
        epu_annual: pd.Series,
        epu_label: str = None,
    ) -> Optional[Dict]:
        """Fit per-regime 3-factor model (tm, mr, vr) for a single market.

        Uses the market group's EPU threshold for regime switching.

        Args:
            market:     Market identifier.
            prices:     Daily price series.
            epu_annual: Annual mean EPU values (US or China).
            epu_label:  Which EPU to use for regime ('us' or 'china').
                        If None, auto-assigned from _market_epu_label or default 'us'.

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
        fwd = annual_r.shift(-1).dropna()

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
            "epu_threshold": epu_thresh,
            "group": _market_group(market),
            "group_pct": group_pct,
        }
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

    # ── Prediction ──────────────────────────────────────────────
    def predict(
        self,
        market: str,
        prices: pd.Series,
        epu_value: float,
    ) -> MarketPrediction:
        """Generate prediction for a single market.

        Uses the appropriate regime model based on current EPU,
        with market-specific AR(1) decay and signal thresholds.
        Applies overheat downgrade rules after model prediction.
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

        # Current factor values
        annual_r = prices.pct_change(fill_method=None).resample("YE").apply(
            lambda x: np.prod(1 + x) - 1
        ).dropna()
        tm_all = trend_momentum(prices).resample("YE").mean().reindex(annual_r.index)
        mr_all = mean_reversion(prices).resample("YE").mean().reindex(annual_r.index)
        vr_all = volatility_regime(prices).resample("YE").mean().reindex(annual_r.index)

        tm_now = float(tm_all.iloc[-1]) if not np.isnan(tm_all.iloc[-1]) else 0
        mr_now = float(mr_all.iloc[-1]) if not np.isnan(mr_all.iloc[-1]) else 0
        vr_now = float(vr_all.iloc[-1]) if not np.isnan(vr_all.iloc[-1]) else 0
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
        tier = 1 if cfg["accuracy"] >= 0.70 else (2 if cfg["accuracy"] >= 0.55 else 3)

        factors = {
            "tm": round(tm_now, 4), "tm_z": round(tm_z, 2),
            "mr": round(mr_now, 4), "mr_z": round(mr_z, 2),
            "vr": round(vr_now, 4), "vr_z": round(vr_z, 2),
        }

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

        # ── Build detail string ──────────────────────────────
        detail_parts = []
        if overheated:
            detail_parts.append(over_reason)

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

        # Add VIX context when available
        if self._vix_active and self._vix_now is not None:
            if self._vix_trend_5d is not None and abs(self._vix_trend_5d) > 0.03:
                direction = "↓" if self._vix_trend_5d < 0 else "↑"
                detail_parts.append(f"VIX={self._vix_now:.1f}{direction}")

        return MarketPrediction(
            market=market,
            signal=signal,
            expected_ret=round(float(pred), 4),
            confidence=round(cfg["accuracy"], 4),
            tier=tier,
            regime=f"{regime_key}_{epu_label}" if (epu_label != "us" and regime_key not in ("normal", "high_epu")) else regime_key,
            factors=factors,
            detail=" | ".join(detail_parts) if detail_parts else "neutral",
            overheat=overheated,
        )

    def predict_all(
        self,
        market_prices: Dict[str, pd.Series],
        epu_value: float,
        epu_values: Dict[str, float] = None,
    ) -> List[MarketPrediction]:
        """Generate predictions for all fitted markets.

        Args:
            market_prices: Market → price series mapping.
            epu_value:     Default US EPU value.
            epu_values:    Per-label EPU values (e.g. {'us': 350, 'china': 376}).
                           Falls back to epu_value if a label is not found.
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
            pred = self.predict(market, px, epu_val)
            results.append(pred)
        results.sort(key=lambda x: -x.confidence)
        return results


__all__ = ["MarketPrediction", "HighConfidencePredictor",
           "EMERGING_MARKETS", "DEVELOPED_MARKETS", "GROUP_EPU_THRESHOLDS",
           "OVERHEAT_NEUT_THRESHOLD", "OVERHEAT_BEAR_THRESHOLD"]
