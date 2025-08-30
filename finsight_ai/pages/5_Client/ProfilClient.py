"""👤 Profil Client - Vue 360° (simple et cohérente schéma Supabase)

Cette page fournit une vue 360° d'un compte:
- Sélection d'un compte (ID + Banque + Client)
- Bloc KYC (banque, client, pays, solde initial, période d'activité)
- KPIs d'activité (transactions, montants, % suspect, contreparties)
- Graphiques (transactions/jour, montants/jour)
- Tables (transactions récentes, alertes)

Cohérence avec supabase_schema.sql:
- accounts: account_id, party_id, init_balance, start_date, end_date, country, business, model_id
- transactions: transaction_id, account_id, counter_party, tx_type, amount, tx_step, is_suspicious
- alerts: account_id, alert_type, alert_score, created_at, status
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st

_THIS_DIR = Path(__file__).parent


def _load_local(module_filename: str, module_name: str):
    path = _THIS_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_q = _load_local("queries.py", "client_queries")
_viz = _load_local("visualization.py", "client_visualization")


BANK_MAPPING = {
    0: "BNP Paribas",
    1: "Société Générale",
    2: "Crédit Agricole",
    3: "Banque Populaire",
    4: "Caisse d'Épargne",
    5: "Crédit Mutuel",
    6: "La Banque Postale",
    7: "HSBC France",
    8: "LCL (Crédit Lyonnais)",
    9: "Boursorama Banque",
}

TX_TYPE_MAPPING = {
    "W": "Withdrawal",
    "T": "Transfer",
    "D": "Deposit",
    "C": "Cash",
    "R": "Remittance",
}


def _format_money(x: float) -> str:
    return f"${float(x):,.0f}"


def main() -> None:

    account_id = st.number_input("Enter an account_id:", min_value=0, step=1, value=19)
    if account_id == 0:
        st.info("Enter a valid account identifier")
        return

    with st.spinner("Loading profile..."):
        profile = _q.get_account_profile(account_id)
        if not profile["kyc"]:
            st.error("No KYC data for this account")
            return

    kyc = profile["kyc"]
    bank_name = BANK_MAPPING.get(int(kyc.get("model_id", -1)), "Other")

    stats = profile["stats"]

    st.markdown("### 📋 KYC Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💳 Account", str(kyc.get("account_id", "N/A")))
    with c2:
        st.metric("🏦 Bank", bank_name)
    with c3:
        st.metric("💰 Initial balance", _format_money(kyc.get("init_balance", 0)))
    with c4:
        current_balance = float(kyc.get("init_balance", 0)) + float(stats.get("net_balance_change", 0))
        st.metric("🏛️ Current balance", _format_money(current_balance))

    st.markdown("### 📊 Account activity")
    total_tx = int(stats.get("total_transactions", 0) or 0)
    suspicious = int(stats.get("suspicious_count", 0) or 0)
    suspicious_pct = (suspicious / total_tx * 100.0) if total_tx else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Transactions", str(total_tx))
    with k2:
        st.metric("Total amount", _format_money(stats.get("total_amount", 0)))
    with k3:
        st.metric("Average", _format_money(stats.get("avg_amount", 0)))
    with k4:
        st.metric("% suspicious", f"{suspicious_pct:.1f}%")

    ts_df = _q.get_account_time_series(account_id)

    st.markdown("### 📈 Evolution (complete history)")
    _viz.render_balance_chart(ts_df)

    st.markdown("### 📋 Transaction list")
    recent = profile["recent"].copy()
    if not recent.empty:
        recent["tx_type"] = recent["tx_type"].map(TX_TYPE_MAPPING).fillna(recent["tx_type"])
        if "tx_step" in recent.columns:
            recent = recent.sort_values(by=["tx_step", "transaction_id"], ascending=[True, True])
        else:
            recent = recent.sort_values(by=["day", "transaction_id"], ascending=[True, True])
        display_cols = ["transaction_id", "tx_step", "tx_type", "amount", "is_suspicious"]
        available_cols = [c for c in display_cols if c in recent.columns]
        if available_cols:
            recent_display = recent[available_cols].head(10)
            if "is_suspicious" in recent_display.columns:
                recent_display["is_suspicious"] = recent_display["is_suspicious"].apply(lambda x: "🚨" if x else "✅")
            column_names = {"transaction_id": "TX ID", "tx_step": "Day", "tx_type": "Type", "amount": "Amount", "is_suspicious": "Status"}
            recent_display = recent_display.rename(columns=column_names)
            st.dataframe(recent_display, use_container_width=True, hide_index=True)
        else:
            st.info("No transaction data to display")
    else:
        st.info("No recent transactions")


if __name__ == "__main__":
    main() 