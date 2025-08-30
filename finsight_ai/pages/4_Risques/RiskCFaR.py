"""⚖️ Risk & CFaR - Analyse simple des risques quotidiens"""

import os
import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st


# Dynamic local imports (avoid relative import issues when Streamlit loads pages)
_THIS_DIR = Path(__file__).parent

def _load_local_module(module_filename: str, module_name: str):
    module_path = _THIS_DIR / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_q = _load_local_module("queries.py", "risques_queries")
_viz = _load_local_module("visualization.py", "risques_visualization")


def main() -> None:
    st.subheader("Risk & CFaR")

    with st.expander("How does it work?", expanded=False):
        st.markdown(
            """
            ### 🎯 What are we analyzing?

            This page calculates **simple risk indicators** based on the variability of your bank's daily flows.

            ### 📊 Where does the data come from?

            - **Daily series**: We aggregate transactions by day (via SQL `GROUP BY tx_step`)
            - **Amount**: Sum of amounts per day (`SUM(amount)`)
            - **Volume**: Number of transactions per day (`COUNT(*)`)
            - **Bank filter**: Optional, to analyze a specific bank

            ### 📈 The 3 indicators explained simply

            **1️⃣ Volatility** 📊
            - Standard deviation of daily values → how much your flows vary from day to day

            **2️⃣ VaR α (Value-at-Risk)** ⚠️
            - Minimum amount you risk losing in the α% worst days

            **3️⃣ CFaR α (Cash-Flow-at-Risk)** 💸
            - Same principle as VaR, applied to cash flows

            ### 🔍 How to interpret?
            - High volatility → flows are unpredictable
            - High VaR/CFaR → adverse scenarios have significant impact

            ### ⚖️ Warning - Important limitations
            - Not regulatory; synthetic data; daily granularity only
            """
        )

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        period_choice = st.selectbox("Period", [
            "Last 90 days", "Last 180 days", "Last 365 days", "Last 540 days", "Last 720 days"
        ], index=0)
        import re as _re
        _m = _re.search(r"(\d+)", period_choice)
        history_days = int(_m.group(1)) if _m else 90
    with c2:
        series_choice = st.selectbox("Series", ["Amount", "Volume"], index=0)
    with c3:
        bank_options = ["All"] + list(_q.BANK_MAPPING.values())
        bank_label = st.selectbox("Bank (optional)", bank_options, index=0)

    c4, c5 = st.columns(2)
    with c4:
        alpha = st.slider("Level (α)", 0.80, 0.99, 0.95, 0.01)
    with c5:
        st.caption("α=0.95 → 95% VaR (worst 5% of days)")

    df = _q.load_daily_series(history_days, series_choice, bank_label)
    if df.empty:
        st.info("No data available.")
        return

    y_col = "value"
    x = df[y_col].astype(float).values
    vol = float(np.std(x))
    var = float(np.quantile(x, 1 - alpha))
    cfar = var

    _viz.render_kpis(vol, var, cfar, alpha)
    _viz.render_time_series(df, y_col, series_choice, history_days, var)
    _viz.render_histogram(df, y_col, series_choice, var)


if __name__ == "__main__":
    main() 