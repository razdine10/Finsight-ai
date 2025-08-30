"""Analysis hub page: exploration, fraud detection, anomalies."""

import os
import sys
from typing import Optional
from pathlib import Path
import importlib.util

import streamlit as st

# Allow importing src.db for any direct DB access if needed by submodules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.db import query_df  # noqa: E402


_THIS_DIR = Path(__file__).parent

def _load_local(module_filename: str, module_name: str):
    path = _THIS_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_q = _load_local("queries.py", "analyse_queries")
_viz = _load_local("visualization.py", "analyse_visualization")


def main() -> None:
    # removed page header

    LABEL_EXPLORATION = "📊 Exploration"
    LABEL_FRAUD = "⚠️ Fraud detection"
    LABEL_ANOMALIES = "🧪 Anomalies"
    section = st.radio(
        "Choose your analysis:",
        [LABEL_EXPLORATION, LABEL_FRAUD, LABEL_ANOMALIES],
        horizontal=True,
        label_visibility="visible",
    )
    st.markdown("---")

    if section == LABEL_EXPLORATION:
        _render_exploration()
    elif section == LABEL_FRAUD:
        _render_fraud()
    else:
        _render_anomalies()


def _render_exploration() -> None:
    # removed exploration caption
    # Controls
    colp, colt, colb = st.columns(3)
    period = colp.selectbox("Period", ["Last 7 days", "Last 30 days", "Last 90 days"], index=1)
    type_options = ["All", "Transfer", "Deposit", "Withdrawal", "Cash", "Remittance"]
    tx_type_label = colt.selectbox("Transaction type", type_options, index=0)
    bank_options = ["All"] + list(_q.BANK_MAPPING.values())
    bank_label = colb.selectbox("Bank (origin)", bank_options, index=0)

    min_b, max_b = _q.get_amount_bounds(period, tx_type_label, bank_label, None)
    min_amount, max_amount = st.slider(
        "Amount (min / max)", min_value=int(min_b), max_value=int(max_b), value=(int(min_b), int(max_b))
    )
    suspicious_only = st.checkbox("Show suspicious only", value=False)

    df = _q.load_transactions(period, tx_type_label, bank_label, min_amount, max_amount, suspicious_only)

    _viz.render_exploration_kpis(df)
    st.markdown("---")
    _viz.render_exploration_charts(df)
    st.markdown("---")
    _viz.render_exploration_table(df)


def _render_fraud() -> None:
    st.caption("Supervised ML — RandomForest")
    _viz.render_fraud_detector()


def _render_anomalies() -> None:
    st.caption("Unsupervised — IsolationForest")
    _viz.render_anomalies_detector()


if __name__ == "__main__":
    main() 