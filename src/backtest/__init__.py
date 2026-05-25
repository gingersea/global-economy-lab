"""
Phase-1 monthly backtest engine.

Implements the minimum closed-loop backtest described in
``docs/phase1_execution_plan.md`` §3.  See package docstring in
``src/backtest/__init__.py`` for an overview.
"""

from __future__ import annotations

from .engine import MonthlyBacktest, BacktestResult
from .metrics import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    turnover,
    summary_metrics,
)
from .strategies import equal_weight_target, buy_and_hold_target

__all__ = [
    "MonthlyBacktest",
    "BacktestResult",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "turnover",
    "summary_metrics",
    "equal_weight_target",
    "buy_and_hold_target",
]
