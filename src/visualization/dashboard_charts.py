"""
Reusable chart generation functions for the Global Economy Lab.

Supports both interactive (Plotly) and static (Matplotlib) outputs.
All public functions accept a DataFrame produced by one of the
analysis modules and return the corresponding figure object.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_plotly():
    """Lazy import plotly – raise a friendly error if not installed."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        return go, px
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive charts.  "
            "Install it with: pip install plotly"
        ) from exc


def _require_matplotlib():
    """Lazy import matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        return plt, mticker
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for static charts.  "
            "Install it with: pip install matplotlib"
        ) from exc


# ── Cycle heatmap ─────────────────────────────────────────────────────────────


def plot_cycle_heatmap(
    cycle_df: pd.DataFrame,
    backend: str = "plotly",
    title: str = "Economic Cycle Phase",
    figsize: tuple = (14, 4),
):
    """Plot a timeline bar coloured by economic cycle phase.

    Args:
        cycle_df: Output of :func:`src.analysis.cycle_position.get_cycle_position`.
                  Must contain columns ``"phase"`` and (optionally)
                  ``"pmi_smooth"`` and ``"cpi_yoy"``.
        backend:  ``"plotly"`` (interactive) or ``"matplotlib"`` (static).
        title:    Chart title.
        figsize:  Figure size for matplotlib backend.

    Returns:
        A :class:`plotly.graph_objects.Figure` or
        :class:`matplotlib.figure.Figure` depending on *backend*.
    """
    _PHASE_COLORS = {
        "recovery": "#4caf50",       # green
        "overheat": "#ff9800",       # orange
        "stagflation": "#f44336",    # red
        "recession": "#9e9e9e",      # grey
        "unknown": "#e0e0e0",
    }

    if cycle_df.empty:
        raise ValueError("cycle_df is empty.")

    if backend == "plotly":
        go, px = _require_plotly()
        fig = go.Figure()
        for phase, color in _PHASE_COLORS.items():
            mask = cycle_df["phase"] == phase
            sub = cycle_df[mask]
            if sub.empty:
                continue
            label_col = "phase_label" if "phase_label" in sub.columns else "phase"
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=sub.get("pmi_smooth", sub["phase"].map({"recovery": 55, "overheat": 55, "stagflation": 45, "recession": 45, "unknown": 50})),
                    mode="markers",
                    marker=dict(color=color, size=6),
                    name=sub[label_col].iloc[0] if len(sub) > 0 else phase,
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="PMI (smoothed)",
            template="plotly_white",
        )
        return fig

    else:  # matplotlib
        plt, _ = _require_matplotlib()
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize)
        for phase, color in _PHASE_COLORS.items():
            mask = cycle_df["phase"] == phase
            sub = cycle_df[mask]
            if sub.empty:
                continue
            pmi_col = "pmi_smooth" if "pmi_smooth" in sub.columns else "pmi"
            ax.scatter(sub.index, sub[pmi_col], c=color, s=20, label=phase)
        ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("PMI (smoothed)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig


# ── Multi-asset cumulative returns ────────────────────────────────────────────


def plot_asset_returns(
    price_data: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    price_col: str = "Close",
    backend: str = "plotly",
    title: str = "Cumulative Returns",
    figsize: tuple = (12, 6),
):
    """Plot cumulative returns for multiple assets on a single chart.

    Args:
        price_data:  Dict mapping asset name → OHLCV DataFrame.
        start_date:  Optional ISO-8601 filter start.
        end_date:    Optional ISO-8601 filter end.
        price_col:   Price column used for returns.
        backend:     ``"plotly"`` or ``"matplotlib"``.
        title:       Chart title.
        figsize:     Figure size (matplotlib only).

    Returns:
        Figure object.
    """
    cum_returns: Dict[str, pd.Series] = {}
    for name, df in price_data.items():
        if df.empty:
            continue
        col = price_col if price_col in df.columns else df.select_dtypes(include="number").columns[0]
        series = df[col]
        if start_date:
            series = series.loc[start_date:]
        if end_date:
            series = series.loc[:end_date]
        series = series.dropna()
        if series.empty:
            continue
        cum_returns[name] = (series / series.iloc[0]) - 1.0

    if not cum_returns:
        raise ValueError("No valid price data to plot.")

    if backend == "plotly":
        go, _ = _require_plotly()
        fig = go.Figure()
        for name, series in cum_returns.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series * 100,
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    else:
        plt, _ = _require_matplotlib()
        fig, ax = plt.subplots(figsize=figsize)
        for name, series in cum_returns.items():
            ax.plot(series.index, series * 100, label=name)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        fig.tight_layout()
        return fig


# ── Correlation matrix heatmap ────────────────────────────────────────────────


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    backend: str = "plotly",
    title: str = "Asset Correlation Matrix",
    figsize: tuple = (8, 7),
):
    """Render a correlation matrix as a colour-coded heatmap.

    Args:
        corr_matrix: Square DataFrame of correlation coefficients,
                     as returned by :func:`src.analysis.correlation.compute_correlation_matrix`.
        backend:     ``"plotly"`` or ``"matplotlib"``.
        title:       Chart title.
        figsize:     Figure size (matplotlib only).

    Returns:
        Figure object.
    """
    if corr_matrix.empty:
        raise ValueError("corr_matrix is empty.")

    if backend == "plotly":
        go, _ = _require_plotly()
        labels = list(corr_matrix.columns)
        z = corr_matrix.values.tolist()
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            xaxis_side="bottom",
        )
        return fig

    else:
        import numpy as np
        plt, _ = _require_matplotlib()
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("RdBu")
        im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        labels = list(corr_matrix.columns)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        fig.tight_layout()
        return fig


# ── Event impact chart ────────────────────────────────────────────────────────


def plot_event_impact(
    event_returns: pd.DataFrame,
    event_date: str = "T0",
    backend: str = "plotly",
    title: str = "Event Impact: Cumulative Returns",
    figsize: tuple = (12, 6),
    annotate_t0: bool = True,
):
    """Plot cumulative returns around an event date (T0 = 0).

    Args:
        event_returns: Output of :func:`src.analysis.event_study.analyze_event`.
                       Index is ``days_from_event``; columns are asset names.
        event_date:    Label for the T0 annotation (ISO-8601 string or free text).
        backend:       ``"plotly"`` or ``"matplotlib"``.
        title:         Chart title.
        figsize:       Figure size (matplotlib only).
        annotate_t0:   Whether to draw a vertical line at T0.

    Returns:
        Figure object.
    """
    if event_returns.empty:
        raise ValueError("event_returns is empty.")

    if backend == "plotly":
        go, _ = _require_plotly()
        fig = go.Figure()
        for col in event_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=event_returns.index,
                    y=event_returns[col] * 100,
                    mode="lines",
                    name=col,
                )
            )
        if annotate_t0:
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="red",
                annotation_text=f"T0: {event_date}",
            )
        fig.update_layout(
            title=title,
            xaxis_title="Days from Event",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    else:
        plt, _ = _require_matplotlib()
        fig, ax = plt.subplots(figsize=figsize)
        for col in event_returns.columns:
            ax.plot(event_returns.index, event_returns[col] * 100, label=col)
        if annotate_t0:
            ax.axvline(0, color="red", linestyle="--", linewidth=1, label=f"T0: {event_date}")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Days from Event")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        fig.tight_layout()
        return fig
