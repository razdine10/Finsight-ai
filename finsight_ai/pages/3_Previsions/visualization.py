"""Visualization utilities for 5_Previsions (Forecasting and Fraud Loss)."""

import os
import base64
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _inject_local_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def create_forecast_figure(
    df: pd.DataFrame,
    merged_valid: pd.DataFrame,
    fc_future: pd.DataFrame,
    y_label: str,
    test_start: int,
    horizon_start: int,
) -> go.Figure:
    """Build the standard forecasting figure (historical + test + future + CI)."""
    fig = go.Figure()
    # Historical
    fig.add_trace(
        go.Scatter(
            x=df["day"], y=df["value"], mode="lines", name="Historical",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="<b>Day</b>: %{x}<br><b>" + y_label + "</b>: %{y:,.0f}<extra></extra>",
        )
    )
    # Forecast on validation window
    fig.add_trace(
        go.Scatter(
            x=merged_valid["day"], y=merged_valid["yhat"], mode="lines",
            name="Forecast (test)", line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="<b>Day</b>: %{x}<br><b>" + y_label + "</b>: %{y:,.0f}<extra></extra>",
        )
    )
    # Confidence interval for future horizon if available
    if {"yhat_lower", "yhat_upper"}.issubset(fc_future.columns) and fc_future[["yhat_lower", "yhat_upper"]].notna().all().all():
        fig.add_trace(go.Scatter(x=fc_future["day"], y=fc_future["yhat_lower"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=fc_future["day"], y=fc_future["yhat_upper"], fill="tonexty",
                fillcolor="rgba(44,127,184,0.15)", line=dict(width=0),
                name="Confidence interval", hoverinfo="skip"
            )
        )
    # Future horizon
    fig.add_trace(
        go.Scatter(x=fc_future["day"], y=fc_future["yhat"], mode="lines",
                   name="Forecast (horizon)", line=dict(color="#2c7fb8", width=2),
                   hovertemplate="<b>Day</b>: %{x}<br><b>" + y_label + "</b>: %{y:,.0f}<extra></extra>")
    )

    fig.add_vline(x=test_start, line_width=1, line_dash="dot", line_color="#999",
                  annotation_text="Test start", annotation_position="top left")
    fig.add_vline(x=horizon_start, line_width=1, line_dash="dot", line_color="#666",
                  annotation_text="Horizon", annotation_position="top left")

    fig.update_layout(xaxis_title="Day (tx_step)", yaxis_title=y_label, hovermode="x unified")
    fig.update_layout(xaxis=dict(unifiedhovertitle=dict(text="<b>Day %{x}</b>")))
    fig.update_traces(mode="lines")
    return fig


def render_csv_download(df: pd.DataFrame, filename: str) -> None:
    """Render a centered CSV download button with optional logo, using local CSS."""
    _inject_local_css()
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Optional logo resolution (Downloads/csv.png then local csv_logo.png)
    download_logo = os.path.expanduser(os.path.join("~", "Downloads", "csv.png"))
    local_logo = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "img", "csv_logo.png")
    logo_path = download_logo if os.path.exists(download_logo) else local_logo
    logo_html = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_html = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    col_l, col_c, col_r = st.columns([3, 2, 3])
    with col_c:
        if logo_html:
            st.markdown(
                f"""
                <a href="data:text/csv;base64,{base64.b64encode(csv_bytes).decode()}"
                   download="{filename}"
                   class="csv-download-btn"
                   title="Download CSV">
                    <img src="{logo_html}" class="csv-logo" alt="CSV"/>
                    Download CSV
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.download_button(label="📊 Download CSV", data=csv_bytes, file_name=filename, mime="text/csv")


def render_chart_help() -> None:
    """Show the small help panel explaining chart elements."""
    st.info(
        """
        **💡 How to read this chart:**
        - **Historical (blue):** observed daily values
        - **Forecast (test, orange dashed):** model prediction on the validation window
        - **Forecast (horizon, blue):** future prediction after the "Horizon" marker
        - **Confidence interval (dark area):** uncertainty band around the forecast
        - **Vertical dotted lines:** "Test start" and "Horizon"
        """
    ) 