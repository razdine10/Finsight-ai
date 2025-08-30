"""Global Dashboard - BI Overview"""

import streamlit as st 
import pandas as pd

from pathlib import Path
import importlib.util
from typing import Dict


def _load_local_module(filename: str, alias: str):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

queries = _load_local_module("queries.py", "_dashboard_queries")
viz = _load_local_module("visualization.py", "_dashboard_viz")



def main() -> None:
    """Main dashboard function"""
    st.set_page_config(
        page_title="Finsight AI - Global Dashboard",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    _render_kpis()
    st.markdown("---")
    
    _render_daily_evolution()
    st.markdown("---")
    
    _render_bank_distribution()
    st.markdown("---")
    
    _render_type_distribution()



def _render_kpis() -> None:
    """Render KPI cards section"""
    kpis = queries.get_global_kpis()
    viz.create_kpi_cards_bi(kpis)



def _render_daily_evolution() -> None:
    """Render daily evolution section"""
    st.subheader("Daily Transaction Volume and Amount")
    show_by_bank = st.toggle("Detail by bank", value=False,
                            key="daily_detail_by_bank")
    
    if not show_by_bank:
        df_daily = queries.get_daily_evolution()
        fig_daily = viz.create_daily_volume_chart(df_daily)
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        _render_daily_by_bank()



def _render_daily_by_bank() -> None:
    """Render daily evolution by bank"""
    banks_df = queries.get_banks_by_activity(limit=10)
    bank_list = banks_df['bank'].tolist()
    default_selection = bank_list[:3]
    
    selected_banks = st.multiselect("Banks to display", 
                                   options=bank_list,
                                   default=default_selection)
    
    df_daily_bank = queries.get_daily_evolution_by_bank(top_n=10)
    if df_daily_bank.empty:
        st.info("Not enough data for bank view.")
    else:
        fig_multi = viz.create_daily_by_bank_subplots(df_daily_bank,
                                                 banks_filter=selected_banks)
        st.plotly_chart(fig_multi, use_container_width=True)



def _render_bank_distribution() -> None:
    """Render bank distribution section"""
    st.subheader("Distribution by Bank")
    df_bank = queries.get_bank_distribution()
    
    if not df_bank.empty:
        fig_bank = viz.create_bank_bar_with_logos(
            df_bank,
            y_col='nb_transactions',
            title='Transactions by Bank',
            y_title='Transactions'
        )
        st.plotly_chart(fig_bank, use_container_width=True)
    else:
        st.info("No bank data available.")



def _render_type_distribution() -> None:
    """Render transaction type distribution section"""
    st.subheader("Distribution by Transaction Type")
    detail_par_banque = st.toggle("Detail by bank", value=False,
                                 key="type_detail_by_bank")

    if not detail_par_banque:
        _render_simple_type_distribution()
    else:
        _render_type_distribution_by_bank()



def _render_simple_type_distribution():
    """Render simple type distribution"""
    df_type = queries.get_type_distribution()
    if not df_type.empty:
        fig_type = viz.create_simple_bar_chart(
            df_type, 'transaction_type', 'nb_transactions',
            "", x_title="Type", y_title="Transactions"
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("No type data available.")



def _render_type_distribution_by_bank():
    """Render type distribution grouped by bank"""
    df_type_bank = queries.get_type_distribution_by_bank()
    if not df_type_bank.empty:
        fig = viz.create_type_distribution_by_bank_chart(df_type_bank)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bank data available for types.")


if __name__ == "__main__":
    main() 