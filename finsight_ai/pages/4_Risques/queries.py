"""SQL queries for Risk & CFaR page (6_Risques)."""

from typing import Dict

import pandas as pd
import streamlit as st

from src.db import query_df

SCHEMA_NAME: str = "aml"
CACHE_TTL: int = 300

BANK_MAPPING: Dict[int, str] = {
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
def load_daily_series(history_days: int, series_choice: str, bank_label: str) -> pd.DataFrame:
    select_expr = "SUM(t.amount) AS value" if series_choice == "Amount" else "COUNT(*) AS value"

    bank_case = (
        "CASE COALESCE(a.model_id, -1) "
        "WHEN 0 THEN 'BNP Paribas' WHEN 1 THEN 'Société Générale' "
        "WHEN 2 THEN 'Crédit Agricole' WHEN 3 THEN 'Banque Populaire' "
        "WHEN 4 THEN 'Caisse d''Épargne' WHEN 5 THEN 'Crédit Mutuel' "
        "WHEN 6 THEN 'La Banque Postale' WHEN 7 THEN 'HSBC France' "
        "WHEN 8 THEN 'LCL (Crédit Lyonnais)' WHEN 9 THEN 'Boursorama Banque' "
        "ELSE 'Other' END"
    )
    where_bank = ""
    if bank_label and bank_label != "All":
        where_bank = f"AND {bank_case} = '{bank_label}'"

    sql = f"""
    SET search_path={SCHEMA_NAME},public;
    WITH mx AS (SELECT MAX(tx_step) AS s FROM {SCHEMA_NAME}.transactions)
    SELECT t.tx_step AS day,
           {select_expr}
    FROM {SCHEMA_NAME}.transactions t
    LEFT JOIN {SCHEMA_NAME}.accounts a ON a.account_id = t.account_id
    CROSS JOIN mx
    WHERE t.tx_step >= GREATEST(1, mx.s - {int(history_days)} + 1)
      {where_bank}
    GROUP BY 1
    ORDER BY 1;
    """
    try:
        df = query_df(sql)
    except Exception as exc:  # pragma: no cover - defensive
        print("load_daily_series (risk) error:", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    base = pd.Timestamp("2020-01-01")
    df["date"] = base + pd.to_timedelta(df["day"].astype(int) - 1, unit="D")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df[["day", "date", "value"]] 