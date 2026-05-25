"""
Unit tests for the unified multi-factor prediction modules.

* ``src.analysis.factors``
* ``src.analysis.factor_ensemble``
* ``src.analysis.macro_predictor`` (factor-based)
* ``src.analysis.daily_predictor`` (factor-based)
* ``src.analysis.prediction_hub``

All tests use synthetic, deterministic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis import factors as fmod
from src.analysis import factor_ensemble as fe
from src.analysis import macro_predictor as mp
from src.analysis import daily_predictor as dp
from src.analysis import prediction_hub as hub


def _synth_prices(n: int = 500, start: float = 100.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.012, n)
    px = start * np.cumprod(1.0 + rets)
    return pd.Series(px, index=dates, dtype=float)


def _synth_macro(n: int = 500, seed: int = 43) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    vals = np.cumsum(rng.normal(0, 0.3, n)) + 50
    return pd.Series(vals, index=dates, dtype=float)


class TestFactors:
    def test_trend_momentum_positive(self):
        px = pd.Series(np.linspace(100, 200, 300))
        sig = fmod.trend_momentum(px, fast=10, medium=40, slow=100)
        assert sig.iloc[-1] > 0

    def test_trend_momentum_negative(self):
        px = pd.Series(np.linspace(200, 100, 300))
        sig = fmod.trend_momentum(px, fast=10, medium=40, slow=100)
        assert sig.iloc[-1] < 0

    def test_mean_reversion_oversold(self):
        px = pd.Series([100.0] * 20 + [80.0, 80.0])
        sig = fmod.mean_reversion(px, lookback=10)
        assert sig.iloc[-1] > 0

    def test_volatility_regime_range(self):
        px = _synth_prices(300)
        sig = fmod.volatility_regime(px, lookback=20, vol_lookback=100)
        assert sig.max() <= 1.0
        assert sig.min() >= -1.0

    def test_carry_yield_curve(self):
        short = pd.Series(np.linspace(1, 5, 200))
        long = pd.Series(np.linspace(2, 4, 200))
        sig = fmod.carry_yield_curve(short, long)
        assert len(sig) == 200

    def test_macro_diffusion(self):
        indicators = {
            "a": pd.Series(np.linspace(1, 2, 100)),
            "b": pd.Series(np.linspace(2, 1, 100)),
        }
        sig = fmod.macro_diffusion(indicators, lookback=20)
        assert sig.max() <= 1.0
        assert sig.min() >= -1.0

    def test_cross_asset_momentum(self):
        prices = {
            "sp500": _synth_prices(200, seed=1),
            "gold": _synth_prices(200, seed=2),
        }
        ranks = fmod.cross_asset_momentum(prices, lookback=40)
        assert ranks.shape == (200, 2)

    def test_credit_risk(self):
        risk = pd.Series(np.random.randn(300).cumsum() + 0)
        sig = fmod.credit_risk(risk)
        assert sig.max() <= 1.0
        assert sig.min() >= -1.0


class TestFactorEnsemble:
    def test_compute_factors(self):
        prices = {"sp500": _synth_prices(200)}
        e = fe.FactorEnsemble()
        factors = e.compute_factors(prices)
        assert len(factors) >= 3

    def test_blend_produces_series(self):
        prices = {"sp500": _synth_prices(200)}
        e = fe.FactorEnsemble()
        raw = e.compute_factors(prices)
        blended = e.blend(raw)
        assert len(blended) == 200
        assert blended.max() <= 1.0
        assert blended.min() >= -1.0

    def test_run_produces_monthly_aggregate(self):
        prices = {"sp500": _synth_prices(500)}
        e = fe.FactorEnsemble()
        result = e.run(prices)
        assert result.monthly_aggregate is not None
        assert not result.monthly_aggregate.empty
        assert len(result.monthly_aggregate) > 0

    def test_select_factors_with_returns(self):
        prices = {"sp500": _synth_prices(300)}
        e = fe.FactorEnsemble()
        raw = e.compute_factors(prices)
        returns = _synth_prices(300).pct_change(fill_method=None).shift(-1)
        evals, weights = e.select_factors(raw, returns)
        assert isinstance(evals, list)
        assert isinstance(weights, dict)
        for e_item in evals:
            assert e_item.group in fmod.FACTOR_GROUPS or e_item.group == "other"


class TestMacroPredictor:
    def test_predict_returns_prediction(self):
        prices = {"sp500": _synth_prices(500)}
        pred = mp.MacroRegimePredictor()
        result = pred.predict(prices)
        assert result.predicted_regime in {"recovery", "overheat", "stagflation", "recession"}
        assert 0 <= result.confidence <= 1

    def test_signal_to_regime_strong_positive(self):
        regime, probs = mp._signal_to_regime(0.5, 0.9)
        assert regime == "recovery"
        assert probs["recovery"] > probs["recession"]

    def test_signal_to_regime_strong_negative(self):
        regime, probs = mp._signal_to_regime(-0.5, 0.9)
        assert regime == "recession"
        assert probs["recession"] > probs["recovery"]


class TestDailyPredictor:
    def test_predict_returns_daily(self):
        prices = {"sp500": _synth_prices(200)}
        pred = dp.DailyAssetPredictor()
        result = pred.predict(prices)
        assert "sp500" in result.asset_signals
        assert 0 <= result.confidence <= 1

    def test_factor_signals_populated(self):
        prices = {"sp500": _synth_prices(200), "gold": _synth_prices(200, seed=10)}
        pred = dp.DailyAssetPredictor()
        result = pred.predict(prices)
        assert len(result.factor_signals) > 0
        assert len(result.factor_weights) > 0


class TestPredictionHub:
    def test_run_produces_result(self):
        prices = {"sp500": _synth_prices(300)}
        h = hub.PredictionHub()
        result = h.run(prices)
        assert result.daily_prediction is not None
        assert result.macro_prediction is not None
        assert result.macro_prediction.predicted_regime in {"recovery", "overheat", "stagflation", "recession"}

    def test_output_files_written(self, tmp_path):
        prices = {"sp500": _synth_prices(300)}
        h = hub.PredictionHub(output_dir=str(tmp_path))
        h.run(prices)
        assert (tmp_path / "prediction_summary.json").exists()

    def test_monthly_and_daily_confidence_in_range(self):
        prices = {"sp500": _synth_prices(500)}
        fwd = prices["sp500"].pct_change(fill_method=None)
        h = hub.PredictionHub()
        result = h.run(prices, forward_returns=fwd)
        dc = result.daily_prediction.confidence
        mc = result.macro_prediction.confidence
        assert 0 <= dc <= 1
        assert 0 <= mc <= 1
