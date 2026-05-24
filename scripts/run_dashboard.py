"""
Launch the Streamlit dashboard for the Global Economy Lab.

Usage:
    python scripts/run_dashboard.py [--port 8501] [--start-date YYYY-MM-DD]

The dashboard shows:
- Economic cycle position (PMI + CPI quadrant)
- Multi-asset cumulative returns
- Correlation heatmap
- VIX / sentiment gauge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the Global Economy Lab Streamlit dashboard."
    )
    parser.add_argument("--port", type=int, default=8501, help="Port to run on.")
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Start date for historical data (ISO-8601).",
    )
    return parser.parse_args()


def check_streamlit() -> bool:
    """Return True if streamlit is installed."""
    try:
        import streamlit  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False


def write_app_file(start_date: str) -> Path:
    """Write a temporary Streamlit app file and return its path."""
    app_code = f'''\
"""
Global Economy Lab – Streamlit Dashboard (auto-generated).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from loguru import logger

st.set_page_config(
    page_title="Global Economy Lab",
    page_icon="🌍",
    layout="wide",
)

START_DATE = "{start_date}"

@st.cache_data(ttl=3600)
def load_data():
    from src.analysis.dashboard_data import build_dashboard_bundle
    return build_dashboard_bundle(start_date=START_DATE, use_cache=True)

st.title("🌍 Global Economy Lab Dashboard")
st.markdown("*Real-time economic cycle and asset performance monitor.*")

with st.spinner("Loading data …"):
    bundle = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
st.sidebar.markdown(f"**Data from:** {{START_DATE}}")
meta = bundle.get("meta", {{}})
st.sidebar.markdown(f"**Last updated:** {{meta.get('built_at', 'N/A')}}")

# ── Economic Cycle ────────────────────────────────────────────────────────────
cycle_df = bundle.get("cycle", pd.DataFrame())
st.header("Economic Cycle Position")
if not cycle_df.empty:
    try:
        from src.visualization.dashboard_charts import plot_cycle_heatmap
        fig = plot_cycle_heatmap(cycle_df, backend="plotly")
        st.plotly_chart(fig, use_container_width=True)
        latest = cycle_df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Phase", latest.get("phase_label", latest.get("phase", "N/A")))
        col2.metric("PMI (smoothed)", f"{{latest.get('pmi_smooth', latest.get('pmi', 0)):.1f}}")
        if "cpi_yoy" in latest and latest["cpi_yoy"] == latest["cpi_yoy"]:
            col3.metric("CPI YoY", f"{{latest['cpi_yoy']:.1f}}%")
    except Exception as e:
        st.warning(f"Cycle chart unavailable: {{e}}")
else:
    st.info("Cycle data not available. Check your FRED_API_KEY in .env.")

# ── Asset Returns ─────────────────────────────────────────────────────────────
st.header("Asset Performance")
equities = bundle.get("equities", {{}})
gold_df = bundle.get("gold", pd.DataFrame())
price_data = dict(equities)
if not gold_df.empty:
    price_data["Gold"] = gold_df

if price_data:
    try:
        from src.visualization.dashboard_charts import plot_asset_returns
        fig = plot_asset_returns(price_data, start_date=START_DATE, backend="plotly")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Returns chart unavailable: {{e}}")
else:
    st.info("No equity/commodity data loaded.")

# ── Correlation Matrix ────────────────────────────────────────────────────────
corr = bundle.get("corr_matrix", pd.DataFrame())
if not corr.empty:
    st.header("Asset Correlation Matrix")
    try:
        from src.visualization.dashboard_charts import plot_correlation_matrix
        fig = plot_correlation_matrix(corr, backend="plotly")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Correlation chart unavailable: {{e}}")

# ── VIX ───────────────────────────────────────────────────────────────────────
vix_df = bundle.get("vix", pd.DataFrame())
if not vix_df.empty and "Close" in vix_df.columns:
    st.header("Market Sentiment (VIX)")
    st.line_chart(vix_df[["Close"]].rename(columns={{"Close": "VIX"}}))

st.markdown("---")
st.markdown("*Global Economy Lab | [GitHub](https://github.com/your-org/global-economy-lab)*")
'''
    app_path = _PROJECT_ROOT / "scripts" / "_dashboard_app.py"
    app_path.write_text(app_code)
    return app_path


def main() -> None:
    """Entry point."""
    args = parse_args()

    if not check_streamlit():
        print(
            "ERROR: streamlit is not installed.\n"
            "Install it with:  pip install streamlit\n"
            "Then re-run this script."
        )
        sys.exit(1)

    import subprocess

    app_path = write_app_file(args.start_date)
    print(
        f"Starting Streamlit dashboard on http://localhost:{args.port} …\n"
        "Press Ctrl+C to stop."
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(args.port),
                "--server.headless",
                "true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
