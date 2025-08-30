import os
import base64
from typing import Optional, List

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

from constants import (
    BANK_COLORS,
    BANK_LOGO_MAPPING,
    DEFAULT_CHART_HEIGHT,
    BANK_CHART_HEIGHT,
    KPI_CARD_HEIGHT
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'img')


def _apply_neon_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply a dark futuristic theme with neon accents to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#F9FAFB"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="#1F2937",
            borderwidth=1,
            font=dict(color="#F9FAFB"),
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#8B5CF6",
            font_color="#F9FAFB",
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#1F2937", zeroline=False, linecolor="#1F2937"
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#1F2937", zeroline=False, linecolor="#1F2937"
    )
    return fig


def create_kpi_cards_bi(kpis):
    """Render four KPIs with neutral colors (no neon)."""

    LABEL_COLOR = "#9CA3AF"
    VALUE_COLOR = "#F9FAFB"

    def _block(title: str, value: str) -> None:
        st.markdown(
            f"""
            <div style='text-align:center;padding:8px 0;'>
              <div style='font-size:12px;color:{LABEL_COLOR};letter-spacing:0.06em;'>
                {title}
              </div>
              <div style='font-size:36px;font-weight:900;color:{VALUE_COLOR};'>
                {value}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _block("Transactions", f"{int(kpis.get('total_transactions', 0)):,}")
    with col2:
        _block("Total amount", f"${float(kpis.get('montant_total', 0)):,.0f}")
    with col3:
        _block("% suspicious", f"{float(kpis.get('pourcentage_suspect', 0)):.2f}%")
    with col4:
        _block("Estimated losses", f"${float(kpis.get('pertes_estimees', 0)):,.0f}")


def create_daily_volume_chart(df_daily):
    """Create daily volume evolution chart with dual y-axis (neutral palette)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Total Amount per Day", "Number of Transactions")
    )

    # Rich indigo for amount
    fig.add_trace(
        go.Scatter(x=df_daily['day'], y=df_daily['daily_volume'],
                  name='Amount', line=dict(color='#4F46E5', width=2.4),
                  hovertemplate='Amount: $%{y:,}<extra></extra>'),
        row=1, col=1
    )

    # Warm amber for counts
    fig.add_trace(
        go.Scatter(x=df_daily['day'], y=df_daily['nb_transactions'],
                  name='Transactions', line=dict(color='#F59E0B', width=2.2),
                  hovertemplate='Transactions: %{y:,}<extra></extra>'),
        row=2, col=1
    )

    fig.update_layout(
        height=DEFAULT_CHART_HEIGHT,
        hovermode='x unified',
        showlegend=False,
        title="",
    )

    
    fig.update_layout(xaxis=dict(unifiedhovertitle=dict(text='Day: %{x}')))
    fig.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6",
                     spikedash="dot")
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6",
                     spikedash="dot")

    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return _apply_neon_plotly_theme(fig)


def create_simple_bar_chart(df, x_col, y_col, title, x_title=None, 
                           y_title=None):
    """Create a bar chart with one distinct color per category (violet theme)."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        text=y_col,
        color=x_col,
        color_discrete_sequence=["#8B5CF6", "#6366F1", "#A78BFA", "#7C3AED", "#4F46E5", "#9333EA"],
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside',
                      hovertemplate=f"{y_title or y_col}: %{{y:,}}<extra></extra>")
    fig.update_layout(
        yaxis_title=y_title or y_col,
        xaxis_title=x_title or x_col,
        height=420,
        showlegend=False,
        hoverlabel=dict(bgcolor="#111827", bordercolor="#8B5CF6", font_color="#F9FAFB"),
    )
    return _apply_neon_plotly_theme(fig)


def create_bank_bar_with_logos(df_banks, y_col='nb_transactions',
                              title='Transactions by Bank',
                              y_title='Transactions'):
    """Create bar chart with bank logos and custom colors"""
    if df_banks.empty:
        return go.Figure()
    
    
    distinct_palette = [
        '#2563EB',  # blue
        '#10B981',  # green
        '#EF4444',  # red
        '#F59E0B',  # orange
        '#14B8A6',  # teal
        '#8B5CF6',  # violet
        '#EC4899',  # pink
        '#06B6D4',  # cyan
        '#84CC16',  # lime
        '#F97316',  # amber
    ]
    y_max = df_banks[y_col].max()

    colors = []
    idx = 0
    for bank in df_banks['bank']:
        if bank == 'La Banque Postale':
            colors.append('#FFD320')
        else:
            colors.append(distinct_palette[idx % len(distinct_palette)])
            idx += 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_banks['bank'],
        y=df_banks[y_col],
        text=[f"{val:,}" for val in df_banks[y_col]],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.6)', width=1)
        ),
        name=y_title,
        hovertemplate='<b>%{x}</b><br>' + y_title + ': %{y:,}<extra></extra>'
    ))
    
    _configure_bank_chart_layout(fig, title, y_title, y_max)
    _add_bank_logos(fig, df_banks, y_col, y_max)
    
    return fig


def create_daily_by_bank_subplots(df_by_bank, banks_filter=None):
    """Create daily evolution subplots by bank with dual y-axis"""
    if df_by_bank.empty:
        return go.Figure()

    if banks_filter:
        df_by_bank = df_by_bank[df_by_bank['bank'].isin(banks_filter)]
        if df_by_bank.empty:
            return go.Figure()

    banks = df_by_bank['bank'].unique().tolist()
    num_banks = len(banks)

    specs = [[{"secondary_y": True}] for _ in range(num_banks)]
    fig = make_subplots(
        rows=num_banks, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=tuple(banks), specs=specs
    )

    for idx, bank in enumerate(banks, start=1):
        bank_data = df_by_bank[df_by_bank['bank'] == bank].sort_values('day')
        
        
        fig.add_trace(
            go.Scatter(
                x=bank_data['day'],
                y=bank_data['daily_volume'],
                mode='lines',
                line=dict(width=2.2, color='#4F46E5'),
                name=f"{bank} - Amount",
                hovertemplate='Amount: $%{y:,}<extra></extra>',
            ),
            row=idx, col=1, secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=bank_data['day'],
                y=bank_data['nb_transactions'],
                mode='lines',
                line=dict(width=2.0, color='#F59E0B'),
                name=f"{bank} - Transactions",
                hovertemplate='Transactions: %{y:,}<extra></extra>',
            ),
            row=idx, col=1, secondary_y=True,
        )
        
        fig.update_yaxes(title_text='Amount', row=idx, col=1, 
                        secondary_y=False)
        fig.update_yaxes(title_text='Transactions', row=idx, col=1,
                        secondary_y=True)

    fig.update_layout(
        height=max(360, 260 * num_banks),
        hovermode='x unified',
        hoversubplots='axis',
        showlegend=False,
        title=''
    )
    fig.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6",
                     spikedash='dot')
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#8B5CF6",
                     spikedash='dot')
    fig.update_xaxes(title_text='Day', row=num_banks, col=1)
    
    return fig


def create_type_distribution_by_bank_chart(df_type_bank: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart for type distribution by bank."""
    if df_type_bank.empty:
        return go.Figure()
    fig = px.bar(
        df_type_bank,
        x='transaction_type', y='nb_transactions', color='bank',
        barmode='group', text='nb_transactions',
        title=''
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(xaxis_title='Type', yaxis_title='Transactions',
                     height=520)
    return _apply_neon_plotly_theme(fig)


def _get_kpi_css():
    """Get CSS for KPI cards"""
    return f"""
    <style>
    .kpi-card {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: box-shadow 0.2s ease;
        height: {KPI_CARD_HEIGHT}px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    
    .kpi-card:hover {{
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}
    
    .kpi-title {{
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .kpi-value {{
        font-size: 1.875rem;
        font-weight: 700;
        margin: 0;
    }}
    
    .kpi-transactions {{ color: #3b82f6; }}
    .kpi-amount {{ color: #059669; }}
    .kpi-suspicious {{ color: #dc2626; }}
    .kpi-losses {{ color: #ea580c; }}
    </style>
    """


def _render_kpi_card(title, value, css_class):
    """Render a single KPI card"""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def _configure_bank_chart_layout(fig, title, y_title, y_max):
    fig.update_layout(
        title="",
        yaxis_title=y_title,
        height=420,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#F9FAFB"),
        hoverlabel=dict(bgcolor="#111827", bordercolor="#8B5CF6", font_color="#F9FAFB"),
    )
    fig.update_yaxes(range=[0, y_max * 1.25])
    fig.update_xaxes(title="Bank")
    fig.update_yaxes(title=y_title)
    # Remove gridlines for better logo visibility
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(legend=dict(bgcolor="rgba(15,23,42,0.6)", font=dict(color="#F9FAFB")))
    return fig


def _add_bank_logos(fig, df_banks, y_col, y_max):
    """Add bank logos to the chart with white circular background"""
    for i, row in enumerate(df_banks.itertuples(index=False)):
        bank_name = getattr(row, 'bank')
        logo_file = BANK_LOGO_MAPPING.get(bank_name)
        
        if not logo_file:
            continue
            
        logo_path = os.path.join(ASSETS_DIR, logo_file)
        if not os.path.exists(logo_path) or os.path.getsize(logo_path) == 0:
            continue
            
        base64_image = _encode_image_to_base64(logo_path)
        if not base64_image:
            continue

        value = getattr(row, y_col)
        size_y, size_x, offset = _get_logo_dimensions(bank_name, y_max)
        logo_y_position = value + (y_max * offset)

        # Add white circular background
        fig.add_shape(
            type="circle",
            x0=i - size_x/2, y0=logo_y_position - size_y/2,
            x1=i + size_x/2, y1=logo_y_position + size_y/2,
            fillcolor="white",
            line=dict(color="white", width=2),
            layer="below"
        )

        # Add logo on top
        fig.add_layout_image(dict(
            source=base64_image,
            x=i,
            y=logo_y_position,
            xref="x",
            yref="y",
            sizex=size_x * 0.8,  # Slightly smaller to fit in circle
            sizey=size_y * 0.8,
            xanchor="center",
            yanchor="middle",
            layer="above",
            opacity=0.95
        ))


def _get_logo_dimensions(bank_name, y_max):
    """Get logo dimensions based on bank name"""
    if bank_name == 'Banque Populaire':
        return y_max * 0.155, 0.92, 0.185
    else:
        return y_max * 0.15, 0.9, 0.18


def _encode_image_to_base64(image_path):
    """Encode an image to base64 for Plotly"""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return "" 