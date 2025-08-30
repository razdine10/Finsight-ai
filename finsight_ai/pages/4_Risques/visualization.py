"""Visualization for Risk & CFaR (6_Risques). Pure UI/Plotly functions."""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _hover_style(fig) -> None:
    fig.update_layout(
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(139, 92, 246, 0.9)",
            bordercolor="#8B5CF6",
            font=dict(color="white", size=12),
        ),
    )


def render_kpis(vol: float, var: float, cfar: float, alpha: float) -> None:
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Volatility", f"{vol:,.2f}")
    with k2:
        st.metric(f"VaR {int(alpha*100)}%", f"{var:,.2f}")
    with k3:
        st.metric(f"CFaR {int(alpha*100)}%", f"{cfar:,.2f}")


def render_time_series(df: pd.DataFrame, y_col: str, series_choice: str, history_days: int, var: float) -> None:
    st.subheader("📈 Temporal evolution")
    st.caption("👀 **What we see:** How your flows vary over time. Peaks and valleys show volatility.")

    fig = px.line(
        df,
        x="day",
        y=y_col,
        title=f"Evolution of daily {series_choice.lower()} over {history_days} days",
        labels={"day": "Day (tx_step)", y_col: f"Daily {series_choice.lower()}"},
    )

    mean_val = float(df[y_col].mean())
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Mean: {mean_val:,.0f}",
    )
    fig.add_hline(
        y=var,
        line_dash="dot",
        line_color="red",
        annotation_text=f"VaR: {var:,.0f}",
    )
    fig.update_traces(
        mode="markers+lines",
        hovertemplate="<b>Day</b>: %{x}<br><b>" + series_choice + "</b>: %{y:,.0f}<extra></extra>",
        line=dict(width=2.5),
    )
    _hover_style(fig)
    fig.update_xaxes(
        showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash="solid",
        title_text="Day (tx_step)",
    )
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash="solid")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        **💡 How to read this chart:**
        - **Blue line:** Your daily flows
        - **Orange dashed line:** Average over the period
        - **Red dotted line:** VaR (risk threshold)
        - **The more the blue line moves** → the higher the volatility
        - **Points below the red line** → worst days bucket per VaR
        """
    )


def render_histogram(df: pd.DataFrame, y_col: str, series_choice: str, var: float) -> None:
    st.subheader("📊 Daily distribution")
    st.caption("👀 **What we see:** Distribution of your flows. The shape tells you if it's regular or not.")

    hist = px.histogram(
        df,
        x=y_col,
        nbins=30,
        title=f"Distribution of daily {series_choice.lower()} values",
        labels={y_col: f"Daily {series_choice.lower()}", "count": "Number of days"},
    )
    mean_val = float(df[y_col].mean())
    hist.add_vline(x=var, line_dash="dot", line_color="red", annotation_text="VaR")
    hist.add_vline(x=mean_val, line_dash="dash", line_color="orange", annotation_text="Mean")
    hist.update_traces(hovertemplate="<b>Range</b>: %{x}<br><b>Days</b>: %{y}<extra></extra>")
    _hover_style(hist)
    hist.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash="solid")
    hist.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6", spikedash="solid")
    st.plotly_chart(hist, use_container_width=True)

    st.info(
        """
        **💡 How to read this histogram:**
        - **Each bar:** Number of days with a given flow level
        - **Orange line:** Average value
        - **Red line:** VaR (risk threshold)
        - **"Normal" shape (bell)** → regular and predictable flows
        - **Spread or asymmetric shape** → volatile flows, exceptional events
        - **Bars left of the red line** → worst days bucket per VaR
        """
    ) 