"""
Phase-1 chart generators for reports and notebooks.

All functions accept DataFrames / Series and return matplotlib ``Figure``
objects or plotly ``Figure`` objects — callers choose the backend via
the ``engine`` parameter (``"matplotlib"`` or ``"plotly"``).

Outputs cover §5.3 (equity curves, drawdown curves, per-cycle heatmap)
and §5.4 (event-window analysis) from the execution plan.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def equity_curve_plot(
    equity_curves: pd.DataFrame,
    title: str = "Equity Curves",
    engine: str = "matplotlib",
):
    """Plot cumulative equity curves for one or more strategies.

    Args:
        equity_curves: DataFrame with one column per strategy.
        title:         Chart title.
        engine:        ``"matplotlib"`` or ``"plotly"``.

    Returns:
        matplotlib ``Figure`` or plotly ``Figure``.
    """
    if engine == "plotly":
        return _equity_plotly(equity_curves, title)
    return _equity_mpl(equity_curves, title)


def drawdown_plot(
    equity_curves: pd.DataFrame,
    title: str = "Drawdown Curves",
    engine: str = "matplotlib",
):
    """Plot drawdown curves for one or more strategies.

    Args:
        equity_curves: DataFrame with one column per strategy (cumulative equity).
        title:         Chart title.
        engine:        ``"matplotlib"`` or ``"plotly"``.

    Returns:
        matplotlib ``Figure`` or plotly ``Figure``.
    """
    dd = equity_curves / equity_curves.cummax() - 1.0
    if engine == "plotly":
        return _drawdown_plotly(dd, title)
    return _drawdown_mpl(dd, title)


def regime_heatmap_plot(
    heatmap: pd.DataFrame,
    title: str = "Mean Monthly Return by Regime",
    engine: str = "matplotlib",
):
    """Plot a regime × asset heatmap of mean returns.

    Args:
        heatmap: Output of :func:`src.analysis.regime_attribution.regime_heatmap`.
        title:   Chart title.
        engine:  ``"matplotlib"`` or ``"plotly"``.

    Returns:
        matplotlib ``Figure`` or plotly ``Figure``.
    """
    if engine == "plotly":
        return _heatmap_plotly(heatmap, title)
    return _heatmap_mpl(heatmap, title)


def event_window_bar(
    event_returns: pd.DataFrame,
    title: str = "Cumulative Return by Event Window",
    engine: str = "matplotlib",
):
    """Grouped bar chart of cumulative returns per event window.

    Args:
        event_returns: Output of
                       :func:`src.analysis.regime_attribution.event_window_returns`.
        title:  Chart title.
        engine: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        matplotlib ``Figure`` or plotly ``Figure``.
    """
    if engine == "plotly":
        return _event_bar_plotly(event_returns, title)
    return _event_bar_mpl(event_returns, title)


def annual_return_bar(
    annual_rets: pd.Series,
    title: str = "Calendar-Year Returns",
    engine: str = "matplotlib",
):
    """Bar chart of calendar-year returns.

    Args:
        annual_rets: Output of
                     :func:`src.analysis.regime_attribution.calendar_year_returns`.
        title:  Chart title.
        engine: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        matplotlib ``Figure`` or plotly ``Figure``.
    """
    if engine == "plotly":
        return _annual_bar_plotly(annual_rets, title)
    return _annual_bar_mpl(annual_rets, title)


def _equity_mpl(equity_curves, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in equity_curves.columns:
        ax.plot(equity_curves.index, equity_curves[col], label=col, linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Equity (log scale)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _drawdown_mpl(drawdowns, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for col in drawdowns.columns:
        ax.fill_between(
            drawdowns.index,
            0,
            drawdowns[col],
            alpha=0.3,
            label=col,
        )
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _heatmap_mpl(heatmap, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(heatmap.columns) * 1.8),
                                    max(4, len(heatmap) * 0.6)))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(heatmap)))
    ax.set_yticklabels(heatmap.index, fontsize=9)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean monthly return")
    for i in range(len(heatmap)):
        for j in range(len(heatmap.columns)):
            val = heatmap.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 0.03 else "white")
    fig.tight_layout()
    return fig


def _event_bar_mpl(event_returns, title):
    import matplotlib.pyplot as plt

    ret_cols = [c for c in event_returns.columns if c != "description"]
    if not ret_cols:
        fig, ax = plt.subplots()
        return fig
    x = np.arange(len(event_returns))
    width = 0.8 / len(ret_cols)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(ret_cols):
        ax.bar(x + i * width, event_returns[col].values, width, label=col)
    ax.set_xticks(x + width * (len(ret_cols) - 1) / 2)
    ax.set_xticklabels(event_returns.index, fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _annual_bar_mpl(annual_rets, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in annual_rets.values]
    ax.bar(annual_rets.index.astype(str), annual_rets.values, color=colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Total return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _equity_plotly(equity_curves, title):
    import plotly.graph_objects as go

    fig = go.Figure()
    for col in equity_curves.columns:
        fig.add_trace(go.Scatter(
            x=equity_curves.index, y=equity_curves[col],
            mode="lines", name=col,
        ))
    fig.update_layout(title=title, yaxis_type="log", yaxis_title="Equity",
                      hovermode="x unified")
    return fig


def _drawdown_plotly(drawdowns, title):
    import plotly.graph_objects as go

    fig = go.Figure()
    for col in drawdowns.columns:
        fig.add_trace(go.Scatter(
            x=drawdowns.index, y=drawdowns[col],
            mode="lines", fill="tozeroy", name=col,
        ))
    fig.update_layout(title=title, yaxis_title="Drawdown",
                      yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def _heatmap_plotly(heatmap, title):
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=heatmap.values, x=heatmap.columns.tolist(), y=heatmap.index.tolist(),
        colorscale="RdYlGn", zmid=0, text=np.round(heatmap.values, 3),
        texttemplate="%{text}", textfont={"size": 10},
    ))
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Regime")
    return fig


def _event_bar_plotly(event_returns, title):
    import plotly.graph_objects as go

    ret_cols = [c for c in event_returns.columns if c != "description"]
    fig = go.Figure()
    for col in ret_cols:
        fig.add_trace(go.Bar(name=col, x=event_returns.index, y=event_returns[col]))
    fig.update_layout(title=title, yaxis_title="Cumulative return",
                      yaxis_tickformat=".0%", barmode="group")
    return fig


def _annual_bar_plotly(annual_rets, title):
    import plotly.graph_objects as go

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in annual_rets.values]
    fig = go.Figure(data=go.Bar(
        x=annual_rets.index.astype(str), y=annual_rets.values,
        marker_color=colors,
    ))
    fig.update_layout(title=title, yaxis_title="Total return",
                      yaxis_tickformat=".0%")
    return fig


__all__ = [
    "equity_curve_plot",
    "drawdown_plot",
    "regime_heatmap_plot",
    "event_window_bar",
    "annual_return_bar",
]
