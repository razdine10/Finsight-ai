"""SQL queries for Analysis (exploration, fraud, anomalies)."""

import os
import sys
from typing import Optional

import pandas as pd
import streamlit as st

SCHEMA_NAME = "aml"
CACHE_TTL = 300

# Access db helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.db import query_df  # noqa: E402

# Shared mappings
TYPE_LABELS = {
    "W": "Withdrawal",
    "T": "Transfer",
    "D": "Deposit",
    "C": "Cash",
    "R": "Remittance",
}

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


@st.cache_data(ttl=CACHE_TTL)
def get_amount_bounds(period: str, tx_type_label: str, bank_label: str, days_custom: Optional[int]):
    if period == "Custom" and days_custom:
        days = int(days_custom)
    else:
        # Extract the first integer found in label like "Last 30 days"
        import re
        m = re.search(r"(\d+)", period)
        days = int(m.group(1)) if m else 30

    try:
        max_step = query_df(f"SELECT MAX(tx_step) AS s FROM {SCHEMA_NAME}.transactions").iloc[0]["s"] or 2000
    except Exception:
        return 0, 100000
    min_step = max(1, int(max_step) - days + 1)

    where = [f"t.tx_step >= {min_step}"]
    if tx_type_label and tx_type_label != "All":
        code = {v: k for k, v in TYPE_LABELS.items()}.get(tx_type_label)
        if code:
            where.append(f"t.tx_type = '{code}'")
    if bank_label and bank_label != "All":
        where.append(
            "CASE COALESCE(a.model_id, -1) "
            "WHEN 0 THEN 'BNP Paribas' WHEN 1 THEN 'Société Générale' "
            "WHEN 2 THEN 'Crédit Agricole' WHEN 3 THEN 'Banque Populaire' "
            "WHEN 4 THEN 'Caisse d''Épargne' WHEN 5 THEN 'Crédit Mutuel' "
            "WHEN 6 THEN 'La Banque Postale' WHEN 7 THEN 'HSBC France' "
            "WHEN 8 THEN 'LCL (Crédit Lyonnais)' WHEN 9 THEN 'Boursorama Banque' "
            "ELSE 'Other' END = '" + bank_label + "'"
        )

    where_clause = "WHERE " + " AND ".join(where)

    sql = f"""
    SET search_path={SCHEMA_NAME},public;
    SELECT MIN(t.amount) AS min_amount, MAX(t.amount) AS max_amount
    FROM {SCHEMA_NAME}.transactions t
    LEFT JOIN {SCHEMA_NAME}.accounts a ON a.account_id = t.account_id
    {where_clause};
    """
    try:
        b = query_df(sql)
        if b.empty:
            return 0, 100000
        return float(b.iloc[0]["min_amount"] or 0), float(b.iloc[0]["max_amount"] or 100000)
    except Exception:
        return 0, 100000


@st.cache_data(ttl=CACHE_TTL)
def load_transactions(
    period: str,
    tx_type_label: str,
    bank_label: str,
    min_amount: int,
    max_amount: int,
    suspicious_only: bool,
    days_custom: Optional[int] = None,
) -> pd.DataFrame:
    if period == "Custom" and days_custom:
        days = int(days_custom)
    else:
        # Extract the first integer found in label like "Last 30 days"
        import re
        m = re.search(r"(\d+)", period)
        days = int(m.group(1)) if m else 30

    max_step_sql = f"SELECT MAX(tx_step) AS s FROM {SCHEMA_NAME}.transactions"
    max_step = query_df(max_step_sql).iloc[0]["s"] or 2000
    min_step = max(1, int(max_step) - days + 1)

    where = [f"t.tx_step >= {min_step}"]

    if tx_type_label and tx_type_label != "All":
        code = {v: k for k, v in TYPE_LABELS.items()}.get(tx_type_label)
        if code:
            where.append(f"t.tx_type = '{code}'")

    if bank_label and bank_label != "All":
        where.append(
            "CASE COALESCE(a.model_id, -1) "
            "WHEN 0 THEN 'BNP Paribas' WHEN 1 THEN 'Société Générale' "
            "WHEN 2 THEN 'Crédit Agricole' WHEN 3 THEN 'Banque Populaire' "
            "WHEN 4 THEN 'Caisse d''Épargne' WHEN 5 THEN 'Crédit Mutuel' "
            "WHEN 6 THEN 'La Banque Postale' WHEN 7 THEN 'HSBC France' "
            "WHEN 8 THEN 'LCL (Crédit Lyonnais)' WHEN 9 THEN 'Boursorama Banque' "
            "ELSE 'Other' END = '" + bank_label + "'"
        )

    where.append(f"t.amount >= {int(min_amount)}")
    where.append(f"t.amount <= {int(max_amount)}")

    if suspicious_only:
        where.append("t.is_suspicious = true")

    where_clause = "WHERE " + " AND ".join(where) if where else ""

    bank_case = (
        "CASE COALESCE(a.model_id, -1) "
        "WHEN 0 THEN 'BNP Paribas' WHEN 1 THEN 'Société Générale' "
        "WHEN 2 THEN 'Crédit Agricole' WHEN 3 THEN 'Banque Populaire' "
        "WHEN 4 THEN 'Caisse d''Épargne' WHEN 5 THEN 'Crédit Mutuel' "
        "WHEN 6 THEN 'La Banque Postale' WHEN 7 THEN 'HSBC France' "
        "WHEN 8 THEN 'LCL (Crédit Lyonnais)' WHEN 9 THEN 'Boursorama Banque' "
        "ELSE 'Other' END"
    )

    sql = f"""
    SET search_path={SCHEMA_NAME},public;
    SELECT
        t.transaction_id,
        t.tx_step                  AS day,
        t.account_id               AS origin_account,
        t.counter_party            AS dest_account,
        {bank_case}               AS origin_bank,
        CASE COALESCE(t.tx_type, '?')
            WHEN 'W' THEN 'Withdrawal'
            WHEN 'T' THEN 'Transfer'
            WHEN 'D' THEN 'Deposit'
            WHEN 'C' THEN 'Cash'
            WHEN 'R' THEN 'Remittance'
            ELSE 'Other'
        END AS tx_type,
        t.amount,
        t.is_suspicious
    FROM {SCHEMA_NAME}.transactions t
    LEFT JOIN {SCHEMA_NAME}.accounts a ON a.account_id = t.account_id
    {where_clause}
    ORDER BY t.tx_step DESC, t.transaction_id DESC
    LIMIT 50000;
    """

    try:
        df = query_df(sql)
    except Exception:
        return pd.DataFrame(columns=[
            "day", "origin_account", "dest_account",
            "tx_type", "amount", "is_suspicious",
        ])

    if not df.empty:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["is_suspicious"] = df["is_suspicious"].astype(bool)
    return df


@st.cache_data(ttl=CACHE_TTL)
def get_recent_transactions(days: int) -> pd.DataFrame:
    sql = f"""
    SET search_path={SCHEMA_NAME},public;
    WITH mx AS (SELECT MAX(tx_step) AS s FROM {SCHEMA_NAME}.transactions)
    SELECT t.transaction_id,
           t.tx_step AS day,
           t.account_id AS origin_account,
           t.counter_party AS dest_account,
           CASE COALESCE(t.tx_type,'?')
                WHEN 'W' THEN 'Withdrawal'
                WHEN 'T' THEN 'Transfer'
                WHEN 'D' THEN 'Deposit'
                WHEN 'C' THEN 'Cash'
                WHEN 'R' THEN 'Remittance'
                ELSE 'Other' END AS tx_type,
           t.amount,
           t.is_suspicious
    FROM {SCHEMA_NAME}.transactions t
    CROSS JOIN mx
    WHERE t.tx_step >= GREATEST(1, mx.s - {days} + 1)
    ;
    """
    try:
        df = query_df(sql)
        df.set_index(pd.RangeIndex(len(df)), inplace=True)
        return df
    except Exception:
        return pd.DataFrame() 