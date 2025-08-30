"""Visualization utilities for 7_Client pages."""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def create_network_plot(G, features_df: pd.DataFrame, communities: Dict[str, int]) -> go.Figure:
    if len(G) == 0:
        return go.Figure()
    import networkx as nx
    try:
        pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    except Exception:
        pos = {node: (np.random.random(), np.random.random()) for node in G.nodes}

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=0.8, color="rgba(100,100,100,0.3)"),
        showlegend=False,
    )

    node_x, node_y, node_color, node_size, node_text, node_hover = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        community_id = communities.get(node, 0)
        node_color.append(community_id)
        pagerank_val = float(features_df.loc[features_df.account_id == node, "pagerank"].values[0]) if not features_df.empty and len(features_df.loc[features_df.account_id == node]) > 0 else 0.001
        size = max(15, min(40, pagerank_val * 2000))
        node_size.append(size)
        node_text.append(str(node) if size > 25 else "")
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg
        node_hover.append([str(node), int(community_id), int(total_deg), float(pagerank_val), int(in_deg), int(out_deg)])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        customdata=node_hover,
        hovertemplate=(
            "<b>Account</b>: %{customdata[0]}<br>"
            "<b>Community</b>: %{customdata[1]}<br>"
            "<b>Connections</b>: %{customdata[2]}<br>"
            "<b>PageRank</b>: %{customdata[3]:.4f}<br>"
            "<b>In</b>: %{customdata[4]} | <b>Out</b>: %{customdata[5]}<extra></extra>"
        ),
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Community", thickness=15, len=0.7),
            line=dict(width=1, color="white"),
            opacity=0.8,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Transaction Network (size = PageRank, color = community)",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        hovermode="closest",
        hoverlabel=dict(bgcolor="rgba(17,24,39,0.95)", bordercolor="#8B5CF6", font=dict(size=12)),
    )
    return fig


def render_balance_chart(ts_df: pd.DataFrame) -> None:
    if ts_df.empty:
        st.info("No recent transactions")
        return
    fig2 = go.Figure()
    ts_df = ts_df.copy()
    ts_df['variation'] = ts_df['balance'].diff().fillna(0)
    fig2.add_trace(go.Scatter(
        x=ts_df['day'],
        y=ts_df['balance'],
        mode='lines',
        name='Balance',
        line=dict(color='#8B5CF6', width=4),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.3)',
        hovertemplate='<b>Day %{x}</b><br><b>Balance</b>: $%{y:,.2f}<br><extra></extra>'
    ))
    significant_changes = ts_df[abs(ts_df['variation']) > ts_df['variation'].std() * 1.5]
    if not significant_changes.empty:
        colors = ['#10B981' if v > 0 else '#EF4444' for v in significant_changes['variation']]
        fig2.add_trace(go.Scatter(
            x=significant_changes['day'],
            y=significant_changes['balance'],
            mode='markers',
            name='Significant variations',
            marker=dict(size=12, color=colors, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate='<b>Day %{x}</b><br><b>Balance</b>: $%{y:,.2f}<br><b>Variation</b>: %{customdata:+,.0f}<extra></extra>',
            customdata=significant_changes['variation'],
            showlegend=False,
        ))
    fig2.update_layout(
        xaxis_title="Day",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        barmode='relative',
        showlegend=False,
    )
    fig2.update_xaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1)
    fig2.update_yaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1)
    st.plotly_chart(fig2, use_container_width=True) 