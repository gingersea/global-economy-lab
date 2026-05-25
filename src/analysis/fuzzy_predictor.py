"""
Fuzzy magnitude prediction: state → {-1, 0, +1} signal.

Uses state percentile to assign a trinary signal, with
reliability filtering based on EPU regime.

Empirically calibrated on 50 years of annual data:
- EPU normal: state quartile maps to directional signal
- EPU high:   signals are unreliable, output 0 (neutral/uncertain)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FuzzySignal:
    """Trinary prediction signal with confidence.

    Attributes:
        signal:     -1 (bearish), 0 (neutral/uncertain), +1 (bullish).
        confidence: 0-1 score of how confident the signal is.
        regime:     'normal' or 'noisy' (EPU-driven).
        detail:     Human-readable description with direction + magnitude hint.
        expected_return_range: (P25, median, P75) of historical returns in this bucket.
    """

    signal: int
    confidence: float
    regime: str
    detail: str
    expected_return_range: tuple


class FuzzyPredictor:
    """Map state percentile to {-1, 0, +1} with EPU gating.

    Calibrated on historical state→return quantile distributions.
    """

    # Historical calibration: state_percentile → (signal, detail, median_return, p25, p75)
    # EPU normal regime
    NORMAL_MAP = {
        (0.00, 0.25): (-1, "偏空", -0.046, -0.072, +0.036),
        (0.25, 0.50): ( 0, "中性偏弱", +0.022, -0.030, +0.057),
        (0.50, 0.75): (+1, "偏多", +0.094, +0.069, +0.149),
        (0.75, 1.00): (+1, "强多", +0.061, +0.006, +0.110),
    }

    # EPU noisy regime — all signals are unreliable
    NOISY_MAP = {
        (0.00, 0.25): ( 0, "噪声偏空(不可靠)", +0.110, +0.052, +0.223),
        (0.25, 0.50): ( 0, "噪声(不可靠)", +0.157, +0.071, +0.198),
        (0.50, 0.75): ( 0, "噪声偏多(不可靠)", +0.092, +0.025, +0.158),
        (0.75, 1.00): ( 0, "噪声强多(不可靠)", +0.093, +0.058, +0.117),
    }

    def __init__(self):
        self._state_history: Optional[np.ndarray] = None

    def predict(
        self,
        state_value: float,
        state_percentile: float,
        is_high_epu: bool,
    ) -> FuzzySignal:
        """Generate fuzzy signal from state and EPU regime.

        Args:
            state_value:      Raw KF state value.
            state_percentile: State percentile relative to history (0-1).
            is_high_epu:      True if EPU is in high-noise regime.

        Returns:
            :class:`FuzzySignal` with -1/0/+1 and confidence.
        """
        regime = "noisy" if is_high_epu else "normal"
        signal_map = self.NOISY_MAP if is_high_epu else self.NORMAL_MAP

        signal = 0
        detail = "未知"
        p25 = p50 = p75 = 0.0

        for (lo, hi), (sig, desc, med, p25v, p75v) in signal_map.items():
            if lo <= state_percentile < hi or (hi == 1.0 and state_percentile >= lo):
                signal = sig
                if is_high_epu:
                    detail = f"{desc} (EPU噪声)"
                else:
                    detail = desc
                p25, p50, p75 = p25v, med, p75v
                break

        if is_high_epu:
            confidence = 0.25
        else:
            pct_distance = abs(state_percentile - 0.50) * 2
            confidence = 0.40 + pct_distance * 0.30

        return FuzzySignal(
            signal=signal,
            confidence=round(min(confidence, 0.85), 4),
            regime=regime,
            detail=detail,
            expected_return_range=(round(p25, 3), round(p50, 3), round(p75, 3)),
        )

    def predict_json(self, state_value: float, state_percentile: float, is_high_epu: bool) -> dict:
        """Return prediction as JSON-serializable dict."""
        fs = self.predict(state_value, state_percentile, is_high_epu)
        return {
            "signal": fs.signal,
            "confidence": fs.confidence,
            "regime": fs.regime,
            "detail": fs.detail,
            "expected_return": {
                "pessimistic": fs.expected_return_range[0],
                "median": fs.expected_return_range[1],
                "optimistic": fs.expected_return_range[2],
            },
        }


__all__ = ["FuzzySignal", "FuzzyPredictor"]
