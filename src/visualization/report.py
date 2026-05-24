"""
Automated Markdown / HTML report generator.

Assembles key charts and textual commentary into a structured report
that can be saved to disk or rendered inline in a Jupyter notebook.
"""

from __future__ import annotations

import base64
import io
import textwrap
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def _fig_to_base64_png(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string.

    Args:
        fig: A :class:`matplotlib.figure.Figure` instance.

    Returns:
        Base64-encoded PNG data URI (``data:image/png;base64,...``).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_markdown_report(
    cycle_df: Optional[pd.DataFrame] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
    price_data: Optional[dict] = None,
    event_returns: Optional[pd.DataFrame] = None,
    event_label: str = "",
    title: str = "Global Economy Lab – Snapshot Report",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a Markdown report with embedded static charts.

    If matplotlib is available, charts are rendered as static PNGs and
    embedded inline via base64 data URIs.

    Args:
        cycle_df:     Cycle-position DataFrame (from cycle_position module).
        corr_matrix:  Correlation matrix DataFrame (from correlation module).
        price_data:   Dict of OHLCV DataFrames for the returns chart.
        event_returns: Cumulative returns from an event study.
        event_label:  Human-readable label for the event.
        title:        Report title.
        start_date:   Analysis start date (for annotation only).
        end_date:     Analysis end date. Defaults to today.
        output_path:  If provided, write the report to this file path.

    Returns:
        The full Markdown string.
    """
    end_date = end_date or str(date.today())
    lines: list[str] = [
        f"# {title}",
        "",
        f"*Generated on {date.today()}  |  Period: {start_date} → {end_date}*",
        "",
        "---",
        "",
    ]

    # ── Economic cycle section ────────────────────────────────────────────────
    if cycle_df is not None and not cycle_df.empty:
        lines += [
            "## Economic Cycle Position",
            "",
        ]
        try:
            from src.visualization.dashboard_charts import plot_cycle_heatmap

            fig = plot_cycle_heatmap(cycle_df, backend="matplotlib")
            img_uri = _fig_to_base64_png(fig)
            lines += [
                f"![Cycle Heatmap]({img_uri})",
                "",
            ]
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Report: could not render cycle chart: {exc}")

        latest = cycle_df.iloc[-1]
        phase_label = latest.get("phase_label", latest.get("phase", "N/A"))
        lines += [
            f"> **Current phase (latest observation):** {phase_label}",
            "",
        ]

    # ── Asset returns section ─────────────────────────────────────────────────
    if price_data:
        lines += [
            "## Asset Performance",
            "",
        ]
        try:
            from src.visualization.dashboard_charts import plot_asset_returns

            fig = plot_asset_returns(
                price_data,
                start_date=start_date,
                end_date=end_date,
                backend="matplotlib",
            )
            img_uri = _fig_to_base64_png(fig)
            lines += [
                f"![Asset Returns]({img_uri})",
                "",
            ]
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Report: could not render returns chart: {exc}")

    # ── Correlation matrix section ────────────────────────────────────────────
    if corr_matrix is not None and not corr_matrix.empty:
        lines += [
            "## Asset Correlation Matrix",
            "",
        ]
        try:
            from src.visualization.dashboard_charts import plot_correlation_matrix

            fig = plot_correlation_matrix(corr_matrix, backend="matplotlib")
            img_uri = _fig_to_base64_png(fig)
            lines += [
                f"![Correlation Matrix]({img_uri})",
                "",
            ]
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Report: could not render correlation matrix: {exc}")

        lines += [
            "### Correlation Table",
            "",
            corr_matrix.round(3).to_markdown(),
            "",
        ]

    # ── Event study section ───────────────────────────────────────────────────
    if event_returns is not None and not event_returns.empty:
        lines += [
            f"## Event Study: {event_label or 'N/A'}",
            "",
        ]
        try:
            from src.visualization.dashboard_charts import plot_event_impact

            fig = plot_event_impact(
                event_returns,
                event_date=event_label,
                backend="matplotlib",
            )
            img_uri = _fig_to_base64_png(fig)
            lines += [
                f"![Event Impact]({img_uri})",
                "",
            ]
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Report: could not render event chart: {exc}")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "*Report generated by [Global Economy Lab](https://github.com/your-org/global-economy-lab)*",
    ]

    report = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {output_path}")

    return report


def generate_html_report(
    output_path: Optional[Path] = None,
    **kwargs,
) -> str:
    """Generate an HTML report by converting the Markdown output.

    Requires the ``markdown`` package (``pip install markdown``).

    Args:
        output_path: Optional file path for the HTML output.
        **kwargs:    Forwarded to :func:`generate_markdown_report`.

    Returns:
        HTML string.
    """
    md = generate_markdown_report(**kwargs)
    try:
        import markdown  # type: ignore[import]

        html_body = markdown.markdown(md, extensions=["tables"])
    except ImportError:
        logger.warning(
            "markdown package not installed – returning raw Markdown wrapped in <pre>."
        )
        html_body = f"<pre>{md}</pre>"

    html = textwrap.dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Global Economy Lab Report</title>
            <style>
                body {{ font-family: sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; }}
                img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 6px 12px; }}
                th {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
        {html_body}
        </body>
        </html>
        """
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved to {output_path}")

    return html
