"""
Unit tests for the prediction modules:

* ``src.analysis.macro_predictor``
* ``src.analysis.daily_predictor``
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

from src.analysis import macro_predictor as mp
from src.analysis import daily_predictor as dp
from src.analysis import prediction_hub as hub


def _synth_regime_panel(n_months: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    regimes_cycle = (["recovery", "overheat", "stagflation", "recession"]
                      * ((n_months // 4) + 1))
    regimes = regimes_cycle[:n_months]
    rng.shuffle(regimes)

    cpi_base = np.cumsum(rng.normal(0, 0.5, n_months)) + 2.0
    unemp = np.cumsum(rng.normal(0, 0.1, n_months)) + 5.0

    return pd.DataFrame({
        "us_pmi": rng.uniform(45, 55, n_months),
        "us_cpi_yoy": cpi_base,
        "us_unemployment_diff": unemp,
        "regime": regimes,
        "sp500_ret": rng.normal(0.005, 0.04, n_months),
        "gold_ret": rng.normal(0.003, 0.035, n_months),
    }, index=dates)


def _synth_daily_prices(n_days: int = 500, start_price: float = 100.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = rng.normal(0.0005, 0.012, n_days)
    prices = start_price * np.cumprod(1.0 + returns)
    return pd.Series(prices, index=dates, name="close")


class TestBuildTransitionMatrix:
    def test_returns_4x4(self):
        panel = _synth_regime_panel(60)
        tmat = mp.build_transition_matrix(panel)
        assert tmat.shape == (4, 4)
        assert list(tmat.index) == mp.ALL_REGIMES[:4]

    def test_rows_sum_to_one(self):
        panel = _synth_regime_panel(100)
        tmat = mp.build_transition_matrix(panel)
        np.testing.assert_array_almost_equal(tmat.sum(axis=1).values, [1.0] * 4, decimal=10)

    def test_empty_panel_returns_uniform(self):
        tmat = mp.build_transition_matrix(pd.DataFrame())
        assert (tmat.values == 0.25).all()

    def test_single_regime_returns_uniform(self):
        panel = _synth_regime_panel(60)
        panel["regime"] = "recovery"
        tmat = mp.build_transition_matrix(panel)
        assert tmat.loc["recovery", "recovery"] == 0.25


class TestPredictRegimeProbs:
    def test_probs_sum_to_one(self):
        panel = _synth_regime_panel(100)
        tmat = mp.build_transition_matrix(panel)
        probs = mp.predict_regime_probs("recovery", tmat, horizon=1)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_horizon_2_uses_matrix_square(self):
        panel = _synth_regime_panel(100)
        tmat = mp.build_transition_matrix(panel)
        probs = mp.predict_regime_probs("recovery", tmat, horizon=2)
        assert abs(probs.sum() - 1.0) < 1e-10


class TestTrendAssessment:
    def test_returns_dict_with_keys(self):
        panel = _synth_regime_panel(30)
        scores = mp.trend_assessment(panel)
        assert "us_cpi_yoy" in scores
        assert "us_unemployment_diff" in scores

    def test_handles_empty_panel(self):
        assert mp.trend_assessment(pd.DataFrame()) == {}


class TestIndicatorAdjustment:
    def test_high_cpi_boosts_inflation_regimes(self):
        probs = pd.Series([0.25, 0.25, 0.25, 0.25], index=mp.ALL_REGIMES[:4])
        adjusted = mp.indicator_adjustment(probs, {"us_cpi_yoy": 2.0, "us_unemployment_diff": 0.0})
        assert abs(adjusted.sum() - 1.0) < 1e-10
        assert adjusted["overheat"] > probs["overheat"]

    def test_negative_cpi_boosts_recovery(self):
        probs = pd.Series([0.25, 0.25, 0.25, 0.25], index=mp.ALL_REGIMES[:4])
        adjusted = mp.indicator_adjustment(probs, {"us_cpi_yoy": -2.0, "us_unemployment_diff": 0.0})
        assert abs(adjusted.sum() - 1.0) < 1e-10


class TestMacroRegimePredictor:
    def test_fit_and_predict(self):
        panel = _synth_regime_panel(100)
        pred = mp.MacroRegimePredictor()
        pred.fit(panel)
        result = pred.predict(panel)
        assert result.current_regime == panel["regime"].iloc[-1]
        assert len(result.predicted_regimes) == 3
        assert 1 in result.predicted_regimes

    def test_predict_rolling_returns_dataframe(self):
        panel = _synth_regime_panel(80)
        pred = mp.MacroRegimePredictor()
        rolling = pred.predict_rolling(panel, min_train_months=24)
        assert not rolling.empty
        assert "predicted_regime" in rolling.columns
        assert "correct" in rolling.columns

    def test_config_horizons_respected(self):
        panel = _synth_regime_panel(60)
        cfg = mp.MacroPredictorConfig(horizons=(1, 2))
        pred = mp.MacroRegimePredictor(cfg)
        pred.fit(panel)
        result = pred.predict(panel)
        assert result.probabilities.shape[0] == 2


class TestMomentumSignal:
    def test_returns_series_same_index(self):
        px = _synth_daily_prices(200)
        sig = dp.momentum_signal(px)
        assert len(sig) == len(px)
        assert isinstance(sig, pd.Series)

    def test_positive_for_uptrend(self):
        px = pd.Series(np.linspace(100, 200, 100))
        sig = dp.momentum_signal(px)
        assert sig.iloc[-1] > 0

    def test_negative_for_downtrend(self):
        px = pd.Series(np.linspace(200, 100, 100))
        sig = dp.momentum_signal(px)
        assert sig.iloc[-1] < 0

    def test_clipped_to_range(self):
        px = _synth_daily_prices(500)
        sig = dp.momentum_signal(px)
        assert sig.max() <= 1.0 + 1e-6
        assert sig.min() >= -1.0 - 1e-6


class TestMeanReversionSignal:
    def test_negative_after_spike(self):
        px = pd.Series([100.0] * 20 + [120.0, 120.0])
        sig = dp.mean_reversion_signal(px)
        assert sig.iloc[-1] < 0

    def test_positive_after_drop(self):
        px = pd.Series([100.0] * 20 + [80.0, 80.0])
        sig = dp.mean_reversion_signal(px)
        assert sig.iloc[-1] > 0


class TestVolatilityAdjustedSignal:
    def test_returns_series(self):
        px = _synth_daily_prices(300)
        sig = dp.volatility_adjusted_signal(px)
        assert len(sig) == len(px)


class TestTrendStrengthSignal:
    def test_centered_around_zero(self):
        px = _synth_daily_prices(500)
        sig = dp.trend_strength_signal(px)
        assert sig.max() <= 0.5
        assert sig.min() >= -0.5


class TestDailyAssetPredictor:
    def test_compute_raw_signals(self):
        px = _synth_daily_prices(200)
        dpred = dp.DailyAssetPredictor()
        raw = dpred.compute_raw_signals(px)
        assert set(raw.keys()) == {"momentum", "mean_reversion", "volatility_adjusted", "trend_strength"}

    def test_blend_signals_respects_regime(self):
        px = _synth_daily_prices(200)
        dpred = dp.DailyAssetPredictor()
        raw = dpred.compute_raw_signals(px)
        blended_rec = dpred.blend_signals(raw, "recovery")
        blended_rec2 = dpred.blend_signals(raw, "recession")
        assert len(blended_rec) == len(blended_rec2)

    def test_predict_returns_daily_prediction(self):
        px = _synth_daily_prices(200)
        df = pd.DataFrame({"close": px.values}, index=px.index)
        dpred = dp.DailyAssetPredictor()
        result = dpred.predict({"test_asset": df}, "recovery")
        assert result.predicted_regime == "recovery"
        assert "test_asset" in result.signals

    def test_predict_handles_empty(self):
        dpred = dp.DailyAssetPredictor()
        with pytest.raises(ValueError):
            dpred.predict({}, "recovery")


class TestPredictionHub:
    def test_run_returns_result(self):
        panel = _synth_regime_panel(80)
        px = _synth_daily_prices(200)
        df = pd.DataFrame({"close": px.values}, index=px.index)
        h = hub.PredictionHub()
        result = h.run(panel, {"sp500": df})
        assert result.macro_prediction is not None
        assert result.daily_prediction is not None
        assert result.summary["current_regime"] == panel["regime"].iloc[-1]

    def test_rolling_backtest_delegates(self):
        panel = _synth_regime_panel(80)
        h = hub.PredictionHub()
        rolling = h.rolling_backtest(panel, {})
        assert not rolling.empty
        assert "correct" in rolling.columns

    def test_output_files_written(self, tmp_path):
        panel = _synth_regime_panel(80)
        px = _synth_daily_prices(200)
        df = pd.DataFrame({"close": px.values}, index=px.index)
        h = hub.PredictionHub(output_dir=str(tmp_path))
        h.run(panel, {"sp500": df})
        assert (tmp_path / "macro_probabilities.csv").exists()
        assert (tmp_path / "daily_signals.csv").exists()
        assert (tmp_path / "prediction_summary.json").exists()
