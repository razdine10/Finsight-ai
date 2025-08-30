"""Visualization logic for Analysis sections (exploration, fraud, anomalies)."""

import os
import base64
from typing import Optional, List
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# dynamically load local queries to avoid name collisions
_THIS_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("analyse_queries_for_viz", str(_THIS_DIR / "queries.py"))
_q = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_q)  # type: ignore[attr-defined]


def _inject_local_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


# -------------------- Exploration --------------------

def render_exploration_kpis(df: pd.DataFrame) -> None:
    st.markdown("\n")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Transactions", f"{len(df):,}")
    with k2:
        st.metric("Total amount", f"${df['amount'].sum():,.0f}")
    with k3:
        avg = df["amount"].mean() if not df.empty else 0
        st.metric("Average amount", f"${avg:,.2f}")
    with k4:
        mx = df["amount"].max() if not df.empty else 0
        st.metric("Max amount", f"${mx:,.2f}")


def render_exploration_charts(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily number of transactions (tx_step)")
        if df.empty:
            st.info("No data")
        else:
            daily = (
                df.groupby("day", as_index=False)["transaction_id"].count()
                .rename(columns={"transaction_id": "count"})
                .sort_values("day")
            )
            fig_line = px.line(daily, x="day", y="count")
            fig_line.update_traces(mode="markers+lines", hovertemplate="<b>Day</b>: %{x}<br><b>Transactions</b>: %{y:,}<extra></extra>")
            fig_line.update_layout(height=360, margin=dict(t=30, r=10, b=10, l=10), hovermode='x unified')
            fig_line.update_layout(xaxis=dict(unifiedhovertitle=dict(text='<b>Day %{x}</b>')))
            fig_line.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash='dot')
            fig_line.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash='dot')
            st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        st.subheader("Amount distribution")
        if df.empty:
            st.info("No data")
        else:
            fig_hist = px.histogram(df, x="amount")
            fig_hist.update_traces(hovertemplate="<b>Amount</b>: $%{x:,.2f}<br><b>Count</b>: %{y:,}<extra></extra>")
            fig_hist.update_layout(height=360, margin=dict(t=30, r=10, b=10, l=10), hovermode='x unified')
            fig_hist.update_layout(xaxis=dict(unifiedhovertitle=dict(text='<b>Amount: $%{x:,.2f}</b>')))
            fig_hist.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash='dot')
            fig_hist.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash='dot')
            st.plotly_chart(fig_hist, use_container_width=True)


def render_exploration_table(df: pd.DataFrame) -> None:
    st.markdown("<h3 style='text-align: center;'>Transactions</h3>", unsafe_allow_html=True)
    search = st.text_input("Search (origin_account)", "")
    if search:
        s = search.lower()
        df = df[df["origin_account"].astype(str).str.lower().str.contains(s)]

    total = len(df)
    page_size = 1000
    total_pages = max(1, (total - 1) // page_size + 1)

    current_page = int(st.session_state.get("tx_page", 1))
    current_page = max(1, min(current_page, total_pages))

    start = (current_page - 1) * page_size
    end = min(start + page_size, total)

    display_cols = [
        "day", "origin_account", "dest_account",
        "tx_type", "amount", "is_suspicious",
    ]

    if total > 0:
        st.dataframe(df[display_cols].iloc[start:end], use_container_width=True, hide_index=True)
    else:
        st.info("No transactions for these filters.")

    col_page, _ = st.columns([1, 6])
    with col_page:
        st.number_input("Page", min_value=1, max_value=total_pages, value=current_page, step=1, key="tx_page")
    st.caption(f"Showing {min(total, start + 1)}-{end} of {total:,} transactions")


# -------------------- Fraud detection --------------------

def render_fraud_detector() -> None:
    _inject_local_css()
    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            ## 🎯 Goal (beginner friendly)
            We train on past transactions that are labeled normal or suspicious. The model then assigns a probability score to new transactions so you can review the riskiest first.

            ## 🧠 What the model looks at (signals)
            - Amount: unusually large values can be riskier in context
            - Transaction type: withdrawal, transfer, deposit… (one-hot encoded)
            - Account activity: how frequently the account operates and how diverse its counterparties are
            - Time: a simple time position signal to account for calendar patterns

            Practically, we build a feature matrix X and a label y (1 = suspicious, 0 = normal).

            ## ⚙️ Model and training
            - Model: RandomForest (an ensemble of decision trees) — robust and easy to tune
            - Class balancing: fraud is rare, so we re-balance to avoid ignoring the 1s
            - Temporal validation: train on older data and validate on more recent data, like in real life

            ## 📊 Reading the results
            - ROC AUC and PR AUC: closer to 1 means better separation between normal and suspicious
            - Decision threshold: above the threshold = review first. Lower threshold → more alerts (more false positives); higher threshold → fewer alerts (risk of false negatives)
            - Table: we show the score, useful features (e.g., amount_z = amount vs account’s usual behavior), and a short explanation

            ## 📝 Takeaway
            This is a decision-support tool: it helps you rank transactions by risk, it is not a legal verdict.
            """
        )
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve, average_precision_score)
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        st.error("scikit-learn is not installed. Run: pip install scikit-learn")
        return

    c1, c2 = st.columns(2)
    with c1:
        period_choice = st.selectbox("Period", ["Last 30 days", "Last 60 days", "Last 90 days"], index=0)
        import re
        m = re.search(r"(\d+)", period_choice)
        period_days = int(m.group(1)) if m else 30
    with c2:
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

    df = _q.get_recent_transactions(period_days)
    if df.empty:
        st.info("No data available.")
        return

    df_feat = _fraud_build_features(df)
    y = df_feat["is_suspicious"].astype(int)
    X = df_feat.drop(columns=["is_suspicious"])

    split_idx = int(0.7 * len(df_feat))
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    categorical_cols = ["tx_type"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)], remainder="passthrough")
    model = Pipeline([("prep", preproc), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))])

    if st.button("Train model"):
        model.fit(X_train, y_train)
        st.session_state["fraud_model"] = model
        st.success("Model trained.")

    if "fraud_model" not in st.session_state:
        st.info("Click 'Train model' to run the ML scoring.")
        return

    model = st.session_state["fraud_model"]
    y_prob = model.predict_proba(X_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, y_prob))
    pr_auc = float(average_precision_score(y_valid, y_prob))

    prec, rec, thr = precision_recall_curve(y_valid, y_prob)
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("ROC AUC", f"{auc:.3f}")
    with k2: st.metric("PR AUC", f"{pr_auc:.3f}")
    with k3: st.metric("Above threshold", f"{int((y_prob >= threshold).sum()):,}")

    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    col_roc, col_pr = st.columns(2)
    with col_roc:
        fig_roc = px.line(x=fpr, y=tpr, labels={"x": "FPR", "y": "TPR"}, title="ROC Curve")
        fig_roc.update_traces(
            mode="markers+lines",
            hovertemplate="<b>FPR</b>: %{x:.3f}<br><b>TPR</b>: %{y:.3f}<br><b>ROC AUC</b>: " + f"{auc:.3f}" + "<extra></extra>"
        )
        fig_roc.update_layout(hovermode="x unified")
        fig_roc.update_layout(xaxis=dict(unifiedhovertitle=dict(text="<b>%{x:.3f}</b> FPR")))
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_pr:
        fig_pr = px.line(x=rec, y=prec, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall Curve")
        fig_pr.update_traces(
            mode="markers+lines",
            hovertemplate="<b>Recall</b>: %{x:.3f}<br><b>Precision</b>: %{y:.3f}<br><b>PR AUC</b>: " + f"{pr_auc:.3f}" + "<extra></extra>"
        )
        fig_pr.update_layout(hovermode="x unified")
        fig_pr.update_layout(xaxis=dict(unifiedhovertitle=dict(text="<b>%{x:.3f}</b> Recall")))
        st.plotly_chart(fig_pr, use_container_width=True)

    df_all = df_feat.copy()
    df_all["score"] = model.predict_proba(df_all.drop(columns=["is_suspicious"]))[:, 1]
    only_risky = st.checkbox("Show only above threshold", value=True)
    merge_cols = [c for c in ["score", "amount_z", "pair_freq_norm", "activity", "pair_count"] if c in df_all.columns]
    df_view = df.merge(df_all[merge_cols], left_index=True, right_index=True, how="left")
    if "amount_z" in df_view.columns:
        def _summary_row(row):
            z = row.get("amount_z", np.nan)
            level = "very high" if z >= 2.0 else "high" if z >= 1.0 else "normal"
            return f"Amount {level} vs history"
        df_view["summary"] = df_view.apply(_summary_row, axis=1)
    if only_risky:
        df_view = df_view[df_view["score"] >= threshold]
    df_view = df_view.sort_values("score", ascending=False)

    display_cols = [
        "transaction_id", "day", "origin_account", "dest_account",
        "tx_type", "amount", "score", "amount_z", "pair_freq_norm", "activity", "summary"
    ]
    st.dataframe(df_view[[c for c in display_cols if c in df_view.columns]].head(1000), use_container_width=True)
    _render_csv_download(df_view[[c for c in display_cols if c in df_view.columns]], "fraud_scores.csv")


def _fraud_build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    if "day" in df_feat.columns:
        day_max = df_feat["day"].max()
        df_feat["day_norm"] = (df_feat["day"] / day_max) if day_max > 0 else 0.0
    tx_count = df_feat.groupby("origin_account")["transaction_id"].transform("count").astype(float)
    amt_sum = df_feat.groupby("origin_account")["amount"].transform("sum").astype(float)
    amt_mean = df_feat.groupby("origin_account")["amount"].transform("mean").astype(float)
    amt_std = df_feat.groupby("origin_account")["amount"].transform("std").fillna(0.0).astype(float)
    uniq_dest = df_feat.groupby("origin_account")["dest_account"].transform("nunique").astype(float)
    df_feat["activity"] = (tx_count / (tx_count.max() or 1.0)).astype(float)
    df_feat["acct_amt_sum_norm"] = (amt_sum / (amt_sum.max() or 1.0)).astype(float)
    df_feat["acct_uniq_dest_norm"] = (uniq_dest / (uniq_dest.max() or 1.0)).astype(float)
    df_feat["amount_z"] = (df_feat["amount"].astype(float) - amt_mean) / (amt_std + 1e-6)
    pair_count = df_feat.groupby(["origin_account", "dest_account"])['transaction_id'].transform('count').astype(float)
    df_feat["pair_freq_norm"] = (pair_count / (tx_count + 1e-6)).astype(float)
    keep = [
        "day_norm", "amount", "tx_type", "activity",
        "acct_amt_sum_norm", "acct_uniq_dest_norm", "amount_z", "pair_freq_norm", "is_suspicious",
    ]
    keep = [c for c in keep if c in df_feat.columns]
    return df_feat[keep]


# -------------------- Anomalies --------------------

def render_anomalies_detector() -> None:
    _inject_local_css()
    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            ## 🎯 Goal (beginner friendly)
            Spot data points that look unusual without needing labels, by analyzing the shape of the data.

            ## 🧠 Signals we use
            - Transaction type: one-hot encoded
            - Normalized amount: compare each amount to the overall scale in the period

            ## ⚙️ Model
            - IsolationForest: think of many small random cuts in the data space; a point isolated with very few cuts is more unusual → higher score
            - 'Contamination' setting: target share of anomalies (e.g. 2% ≈ 2 out of 100)

            ## 📊 Outputs and reading
            - Score histogram: the right tail = cases to prioritize
            - Sorted table: top anomalies with a brief summary (e.g., amount unusually high vs period)

            ## 📝 Takeaway
            "Anomalous" ≠ fraud. It’s a signal for investigation, to help you prioritize your work.
            """
        )
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        st.error("scikit-learn is not installed. Run: pip install scikit-learn")
        return

    c1, c2 = st.columns(2)
    with c1:
        period_option = st.selectbox("Period", ["Last 30 days", "Last 60 days", "Last 90 days"], index=0)
        import re
        m = re.search(r"(\d+)", period_option)
        period_days = int(m.group(1)) if m else 30
    with c2:
        contamination = st.slider("Expected anomaly rate", 0.001, 0.20, 0.02, 0.001)

    df = _q.get_recent_transactions(period_days)
    if df.empty:
        st.info("No data available.")
        return

    X = _anomaly_build_features(df)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X)
    scores = -iso.score_samples(X)

    df_view = df.copy()
    df_view["anomaly_score"] = scores
    a = df_view["amount"].astype(float)
    a_min, a_max = a.min(), a.max()
    df_view["amount_norm"] = (a - a_min) / (a_max - a_min) if a_max > a_min else 0.0

    def _row_summary(row):
        an = float(row.get("amount_norm", 0))
        level = "very high" if an >= 0.95 else "high" if an >= 0.85 else "moderate" if an >= 0.70 else "normal"
        return f"Amount {level} for the period"
    df_view["summary"] = df_view.apply(_row_summary, axis=1)

    st.subheader("Top anomalies")
    existing_cols = [c for c in ["transaction_id", "day", "origin_account", "dest_account", "tx_type", "amount", "amount_norm", "anomaly_score", "summary"] if c in df_view.columns]
    st.dataframe(df_view.sort_values("anomaly_score", ascending=False)[existing_cols].head(1000), use_container_width=True)

    fig = px.histogram(df_view, x="anomaly_score", nbins=50, title="Anomaly score distribution")
    fig.update_traces(hovertemplate="<b>Score</b>: %{x:.3f}<br><b>Count</b>: %{y:,}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_layout(xaxis=dict(unifiedhovertitle=dict(text="<b>Score: %{x:.3f}</b>")))
    st.plotly_chart(fig, use_container_width=True)

    _render_csv_download(df_view[existing_cols], "anomalies.csv")


def _anomaly_build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for t in sorted(X["tx_type"].unique()):
        X[f"tx_{t}"] = (X["tx_type"] == t).astype(int)
    a = X["amount"].astype(float)
    a_min, a_max = a.min(), a.max()
    X["amount_norm"] = (a - a_min) / (a_max - a_min) if a_max > a_min else 0.0
    keep = [c for c in X.columns if c.startswith("tx_") and c != "tx_type"] + ["amount_norm"]
    return X[keep]


# -------------------- Shared --------------------

def _render_csv_download(df: pd.DataFrame, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    assets_logo = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "img", "csv_logo.png")
    logo_html = ""
    if os.path.exists(assets_logo):
        with open(assets_logo, "rb") as f:
            logo_html = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    st.markdown(
        f"""
        <a href="data:text/csv;base64,{base64.b64encode(csv_bytes).decode()}" 
           download="{filename}" 
           class="csv-download-btn"
           title="Download CSV">
            {f'<img src="{logo_html}" class="csv-logo" alt="CSV"/>' if logo_html else ''}
            Download CSV
        </a>
        """,
        unsafe_allow_html=True,
    ) 