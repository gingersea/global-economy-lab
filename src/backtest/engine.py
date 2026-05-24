"""
Monthly rebalanced backtest engine.

A *minimal* engine — by design.  The phase-1 plan calls for a
demonstration that the research panel + regime labels translate into
explainable asset-allocation performance, **not** a production trading
simulator.

Mechanics
---------
* Inputs are aligned on ``returns.index`` (assumed month-end).
* At month *t* the portfolio holds weights ``weights.loc[t]``.
* The realised gross return for month *t* is
  ``(weights_t * returns_t).sum()``.
* Optionally subtract a per-rebalance trading cost equal to
  ``cost_bps × turnover_t × 1e-4``.
* The equity curve starts at 1.0.

No look-ahead is enforced *by the engine itself*: callers must shift
their signals before passing them in (see
:func:`src.analysis.research_panel.apply_signal_lag`).  An optional
:meth:`MonthlyBacktest.assert_no_lookahead` helper validates the most
common form of mistake (weights changing on the same row as the return
they react to).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .metrics import summary_metrics


@dataclass
class BacktestResult:
    """Output of :meth:`MonthlyBacktest.run`."""

    returns: pd.Series                     # monthly portfolio return (net of costs)
    equity: pd.Series                      # cumulative equity curve, starts at 1.0
    weights: pd.DataFrame                  # weights actually used each month
    metrics: Dict[str, float] = field(default_factory=dict)
    name: str = "portfolio"

    @property
    def cagr(self) -> float:
        return self.metrics.get("ann_return", float("nan"))

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown", float("nan"))


class MonthlyBacktest:
    """Vectorised monthly rebalance backtest.

    Args:
        returns: Wide DataFrame of monthly asset returns (one column
                 per asset).  Index must be month-end-aligned.
        cost_bps: Round-trip trading cost in basis points applied to the
                  monthly turnover.  ``10`` ≈ 0.10 % per full rotation.
                  Defaults to ``0`` (phase-1 plan §3.4 baseline).
        risk_free_annual: Annual risk-free rate used in Sharpe.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        cost_bps: float = 0.0,
        risk_free_annual: float = 0.0,
    ) -> None:
        if returns is None or returns.empty:
            raise ValueError("MonthlyBacktest: returns DataFrame is empty.")
        self.returns = returns.sort_index().astype("float64")
        self.cost_bps = float(cost_bps)
        self.risk_free_annual = float(risk_free_annual)

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        weights: pd.DataFrame,
        name: str = "portfolio",
    ) -> BacktestResult:
        """Run the backtest with a pre-computed weight schedule."""
        if weights is None or weights.empty:
            raise ValueError("MonthlyBacktest.run: weights DataFrame is empty.")

        aligned_weights, aligned_returns = self._align(weights, self.returns)

        # Gross return per month: dot product of weight row and return row.
        # NaN weights / returns are treated as 0 contribution.
        w = aligned_weights.fillna(0.0)
        r = aligned_returns.fillna(0.0)
        gross = (w * r).sum(axis=1)

        # Turnover-based trading cost (one-way), subtracted from the
        # month it occurs.  ``½`` is already inside the cost factor for
        # symmetry with metrics.turnover.
        if self.cost_bps > 0.0 and len(w) > 1:
            delta = w.diff().abs().sum(axis=1).fillna(0.0)
            tc = (delta * (self.cost_bps * 1e-4) * 0.5)
        else:
            tc = pd.Series(0.0, index=gross.index)

        net = gross - tc
        equity = (1.0 + net).cumprod()

        metrics = summary_metrics(
            net,
            weights=aligned_weights,
            risk_free_annual=self.risk_free_annual,
        )
        metrics["total_return"] = float((1.0 + net).prod() - 1.0)

        return BacktestResult(
            returns=net.rename(name),
            equity=equity.rename(name),
            weights=aligned_weights,
            metrics=metrics,
            name=name,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _align(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align *weights* and *returns* on their common index/columns.

        Missing assets are filled with ``NaN`` in the appropriate frame
        — the run loop then treats them as zero contribution.
        """
        common_cols = [c for c in weights.columns if c in returns.columns]
        if not common_cols:
            raise ValueError(
                "MonthlyBacktest: weights and returns share no asset columns. "
                f"weights={list(weights.columns)}  returns={list(returns.columns)}"
            )
        common_idx = weights.index.intersection(returns.index)
        if len(common_idx) == 0:
            raise ValueError("MonthlyBacktest: weights and returns share no dates.")
        w = weights.loc[common_idx, common_cols].sort_index()
        r = returns.loc[common_idx, common_cols].sort_index()
        return w, r

    @staticmethod
    def assert_no_lookahead(
        weights: pd.DataFrame,
        signals: pd.DataFrame,
        signal_cols: Optional[list[str]] = None,
    ) -> None:
        """Best-effort sanity check that *weights* do not move on the
        same row as the underlying signal."""
        if weights is None or signals is None:
            return
        cols = signal_cols or [c for c in signals.columns if c in weights.columns]
        if not cols:
            return
        for c in cols:
            same_day_change = (
                weights[c].diff().abs() > 0
            ) & signals[c].diff().abs() > 0
            if bool(same_day_change.any()):
                logger.warning(
                    f"assert_no_lookahead: weights and signal '{c}' changed on "
                    f"the same row {int(same_day_change.sum())} times — verify "
                    f"that signals are shifted by ≥ 1 period."
                )
