"""
Unit tests for the phase-1 closed-loop modules:

* ``src.analysis.data_quality``
* ``src.analysis.research_panel``
* ``src.analysis.regime_labels``
* ``src.backtest`` (engine + metrics + strategies)

All tests use synthetic, deterministic data – no network or filesystem
access, so they run reliably in any CI sandbox.
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

from src.analysis import data_quality as dq
from src.analysis import research_panel as rp
from src.analysis import regime_labels as rl
from src.backtest import (
    MonthlyBacktest,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    turnover,
    summary_metrics,
    equal_weight_target,
    buy_and_hold_target,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def daily_prices() -> pd.DataFrame:
    """3-year deterministic daily price series (drift + small noise)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    rets = rng.normal(0.0005, 0.01, size=len(idx))
    px = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": px}, index=idx)


@pytest.fixture
def monthly_macro() -> pd.DataFrame:
    """Monthly macro series with a small upward trend (e.g. CPI level)."""
    idx = pd.date_range("2015-01-31", "2022-12-31", freq="ME")
    # Slow ~3 %/yr level growth
    level = 100.0 * (1.0 + 0.0025) ** np.arange(len(idx))
    return pd.DataFrame({"value": level}, index=idx)


# ─────────────────────────────────────────────────────────────────────
# data_quality
# ─────────────────────────────────────────────────────────────────────


def test_check_series_basic(monthly_macro):
    rep = dq.check_series(monthly_macro, name="cpi")
    assert rep["status"] == "ok"
    assert rep["rows"] == len(monthly_macro)
    assert rep["coverage_start"] == "2015-01-31"
    assert rep["missing_pct"] == 0.0
    assert rep["duplicate_dates"] == 0
    assert rep["abnormal_jumps"] == 0


def test_check_series_detects_missing_and_duplicates(monthly_macro):
    df = monthly_macro.copy()
    df.iloc[5, 0] = np.nan
    # Inject a duplicate date by appending the first row again.
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup])
    rep = dq.check_series(df, name="cpi")
    assert rep["missing_pct"] > 0
    assert rep["duplicate_dates"] == 1


def test_check_series_detects_jump(monthly_macro):
    df = monthly_macro.copy()
    df.iloc[20, 0] *= 5.0  # 400 % spike
    rep = dq.check_series(df, name="cpi", jump_z=3.0)
    assert rep["abnormal_jumps"] >= 1


def test_check_series_handles_empty():
    rep = dq.check_series(pd.DataFrame(), name="empty")
    assert rep["status"] == "empty"
    assert rep["rows"] == 0


def test_check_panel_and_summary(daily_prices, monthly_macro):
    panels = {"sp500": daily_prices, "us_cpi": monthly_macro, "missing": pd.DataFrame()}
    report = dq.check_panel(panels)
    assert set(report["name"]) == {"sp500", "us_cpi", "missing"}
    s = dq.summarize_report(report)
    assert s["sources"] == 3
    # sp500 + us_cpi should be ok
    assert s["ok"] >= 2


# ─────────────────────────────────────────────────────────────────────
# research_panel
# ─────────────────────────────────────────────────────────────────────


def test_daily_to_monthly_return(daily_prices):
    ret = rp.daily_to_monthly_return(daily_prices, price_col="close")
    assert isinstance(ret, pd.Series)
    # First month return is NaN (no prior month).
    assert pd.isna(ret.iloc[0])
    assert ret.notna().sum() >= 30
    # Month-end frequency
    assert ret.index.is_monotonic_increasing
    assert all(d == d + pd.offsets.MonthEnd(0) for d in ret.index)


def test_macro_to_monthly_yoy(monthly_macro):
    yoy = rp.macro_to_monthly(monthly_macro, transform="yoy")
    # ~3 %/yr level growth ⇒ YoY around 3 % once 12 months elapse.
    last = yoy.dropna().iloc[-1]
    assert 1.5 < last < 5.0


def test_macro_to_monthly_invalid_transform(monthly_macro):
    with pytest.raises(ValueError):
        rp.macro_to_monthly(monthly_macro, transform="bogus")


def test_build_monthly_panel(daily_prices, monthly_macro):
    panel = rp.build_monthly_panel(
        assets={"sp500": daily_prices},
        macros={"us_cpi": (monthly_macro, "yoy"), "us_cpi_level": (monthly_macro, "level")},
    )
    assert "sp500_ret" in panel.columns
    assert "us_cpi_yoy" in panel.columns
    assert "us_cpi_level" in panel.columns
    assert panel.index.name == "month_end"


def test_apply_signal_lag_shifts_only_signals():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "sp500_ret": [0.01, 0.02, -0.01, 0.03, 0.00, 0.04],
            "us_cpi_yoy": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        },
        index=idx,
    )
    lagged = rp.apply_signal_lag(df, lag=1)
    # Return column unchanged
    pd.testing.assert_series_equal(lagged["sp500_ret"], df["sp500_ret"])
    # Signal column shifted: first NaN, second equals original first.
    assert pd.isna(lagged["us_cpi_yoy"].iloc[0])
    assert lagged["us_cpi_yoy"].iloc[1] == df["us_cpi_yoy"].iloc[0]


# ─────────────────────────────────────────────────────────────────────
# regime_labels
# ─────────────────────────────────────────────────────────────────────


def test_label_regimes_all_four_states():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    # PMI: high, high, low, low ; CPI YoY: low, high, high, low
    panel = pd.DataFrame(
        {
            "us_pmi": [55.0, 55.0, 45.0, 45.0],
            "us_cpi_yoy": [1.0, 4.0, 4.0, 1.0],
        },
        index=idx,
    )
    cfg = rl.RegimeConfig(growth_smooth_window=1, cpi_smooth_window=1)
    out = rl.label_regimes(panel, cfg)
    assert list(out["regime"]) == [
        "recovery",
        "overheat",
        "stagflation",
        "recession",
    ]
    # Bilingual labels populated.
    assert all(isinstance(x, str) and "/" not in x for x in out["regime_label"])


def test_label_regimes_unknown_on_nan():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.DataFrame(
        {"us_pmi": [np.nan, 55.0], "us_cpi_yoy": [2.0, 1.0]},
        index=idx,
    )
    cfg = rl.RegimeConfig(growth_smooth_window=1, cpi_smooth_window=1)
    out = rl.label_regimes(panel, cfg)
    assert out["regime"].iloc[0] == "unknown"
    assert out["regime"].iloc[1] in {"recovery", "overheat"}


def test_regime_target_weights_sum_to_one():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    panel = pd.DataFrame(
        {"regime": ["recovery", "overheat", "stagflation", "recession"]},
        index=idx,
    )
    w = rl.regime_target_weights(panel)
    # Every row sums to ~1.
    sums = w.sum(axis=1)
    assert np.allclose(sums.values, 1.0, atol=1e-9)


def test_regime_target_weights_subset_renormalises():
    idx = pd.date_range("2020-01-31", periods=1, freq="ME")
    panel = pd.DataFrame({"regime": ["overheat"]}, index=idx)
    w = rl.regime_target_weights(
        panel,
        assets=["sp500_ret", "gold_ret"],
    )
    assert set(w.columns) == {"sp500_ret", "gold_ret"}
    assert np.isclose(w.sum(axis=1).iloc[0], 1.0)


# ─────────────────────────────────────────────────────────────────────
# backtest – metrics
# ─────────────────────────────────────────────────────────────────────


def test_metrics_on_constant_return():
    # 1 % per month, 12 months ⇒ CAGR ≈ 12.68 %.
    r = pd.Series([0.01] * 12)
    assert np.isclose(annualised_return(r), 1.01 ** 12 - 1)
    # Effectively zero volatility (float noise tolerated).
    assert annualised_volatility(r) < 1e-10
    # Constant positive returns ⇒ no drawdown.
    assert max_drawdown(r) == 0.0


def test_metrics_drawdown_and_sharpe():
    # Build a deterministic sequence: +5 %, -10 %, +5 %, -10 %, ...
    r = pd.Series([0.05, -0.10] * 12)
    mdd = max_drawdown(r)
    assert mdd < 0
    sr = sharpe_ratio(r)
    assert not np.isnan(sr)


def test_turnover_zero_for_constant_weights():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    w = pd.DataFrame({"a": [0.5] * 6, "b": [0.5] * 6}, index=idx)
    assert turnover(w) == 0.0


def test_turnover_full_rotation_one():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    w = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]}, index=idx)
    assert np.isclose(turnover(w), 1.0)


# ─────────────────────────────────────────────────────────────────────
# backtest – engine
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def two_asset_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    return pd.DataFrame(
        {
            "sp500_ret": rng.normal(0.008, 0.04, size=len(idx)),
            "gold_ret": rng.normal(0.004, 0.03, size=len(idx)),
        },
        index=idx,
    )


def test_engine_equal_weight_matches_average(two_asset_returns):
    weights = equal_weight_target(two_asset_returns)
    bt = MonthlyBacktest(two_asset_returns).run(weights, name="ew")
    expected = two_asset_returns.mean(axis=1)
    pd.testing.assert_series_equal(
        bt.returns.rename(None).reset_index(drop=True),
        expected.rename(None).reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )
    assert bt.equity.iloc[0] == pytest.approx(1.0 + expected.iloc[0])


def test_engine_buy_and_hold_isolates_asset(two_asset_returns):
    weights = buy_and_hold_target(two_asset_returns, asset="sp500_ret")
    bt = MonthlyBacktest(two_asset_returns).run(weights, name="bh")
    pd.testing.assert_series_equal(
        bt.returns.rename(None).reset_index(drop=True),
        two_asset_returns["sp500_ret"].rename(None).reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_engine_cost_reduces_return(two_asset_returns):
    idx = two_asset_returns.index
    # Alternating full-rotation weights – maximum turnover.
    w_rotate = pd.DataFrame(
        {
            "sp500_ret": [1.0, 0.0] * (len(idx) // 2),
            "gold_ret": [0.0, 1.0] * (len(idx) // 2),
        },
        index=idx,
    )
    no_cost = MonthlyBacktest(two_asset_returns, cost_bps=0).run(w_rotate)
    with_cost = MonthlyBacktest(two_asset_returns, cost_bps=50).run(w_rotate)
    assert with_cost.returns.sum() < no_cost.returns.sum()


def test_engine_raises_on_empty():
    with pytest.raises(ValueError):
        MonthlyBacktest(pd.DataFrame())


def test_engine_raises_on_no_common_columns(two_asset_returns):
    idx = two_asset_returns.index
    bad_weights = pd.DataFrame({"foobar": [1.0] * len(idx)}, index=idx)
    with pytest.raises(ValueError):
        MonthlyBacktest(two_asset_returns).run(bad_weights)


def test_summary_metrics_keys(two_asset_returns):
    weights = equal_weight_target(two_asset_returns)
    bt = MonthlyBacktest(two_asset_returns).run(weights)
    keys = set(bt.metrics.keys())
    assert {"ann_return", "ann_volatility", "sharpe", "max_drawdown",
            "calmar", "months", "turnover", "total_return"}.issubset(keys)


# ─────────────────────────────────────────────────────────────────────
# Integration: end-to-end mini phase-1 closed loop
# ─────────────────────────────────────────────────────────────────────


def test_end_to_end_pipeline():
    """Build a small panel → label regimes → run all three benchmarks."""
    rng = np.random.default_rng(123)
    # 5 years of daily prices for two assets
    daily_idx = pd.date_range("2018-01-01", "2022-12-31", freq="B")
    sp = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(daily_idx))))},
        index=daily_idx,
    )
    gold = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, len(daily_idx))))},
        index=daily_idx,
    )
    bond = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.005, len(daily_idx))))},
        index=daily_idx,
    )
    oil = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.020, len(daily_idx))))},
        index=daily_idx,
    )
    dxy = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.00005, 0.004, len(daily_idx))))},
        index=daily_idx,
    )

    # Macro (monthly)
    macro_idx = pd.date_range("2017-01-31", "2022-12-31", freq="ME")
    pmi = pd.DataFrame(
        {"value": 50 + 5 * np.sin(np.linspace(0, 6, len(macro_idx)))},
        index=macro_idx,
    )
    cpi = pd.DataFrame(
        {"value": 100 * (1.0 + 0.002) ** np.arange(len(macro_idx))},
        index=macro_idx,
    )

    panel = rp.build_monthly_panel(
        assets={
            "sp500": sp,
            "gold": gold,
            "us_treasury_10y": bond,
            "crude_oil_wti": oil,
            "dxy": dxy,
        },
        macros={
            "us_pmi": (pmi, "level"),
            "us_cpi": (cpi, "yoy"),
        },
    )
    assert not panel.empty

    # Signal lag of 1 month – we are about to derive weights from signals.
    lagged = rp.apply_signal_lag(panel, lag=1)
    labelled = rl.label_regimes(lagged)
    assert "regime" in labelled.columns

    returns_cols = [c for c in panel.columns if c.endswith("_ret")]
    returns = panel[returns_cols]

    # Three benchmarks all run successfully.
    bt = MonthlyBacktest(returns, cost_bps=10)
    ew = bt.run(equal_weight_target(returns), name="equal_weight")
    bh = bt.run(buy_and_hold_target(returns, "sp500_ret"), name="sp500_bh")
    regime_weights = rl.regime_target_weights(labelled, assets=returns_cols)
    rotation = bt.run(regime_weights, name="regime_rotation")

    for res in (ew, bh, rotation):
        assert len(res.returns) > 12
        assert res.equity.iloc[-1] > 0
        assert "ann_return" in res.metrics
