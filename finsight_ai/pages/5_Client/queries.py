from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from src.db import query_df

SCHEMA_NAME: str = "aml"
CACHE_TTL: int = 300


@st.cache_data(ttl=CACHE_TTL)
def get_transaction_network(days: int = 30, min_amount: float = 1000, limit_edges: int = 1000) -> pd.DataFrame:
    sql = f"""
    WITH recent_tx AS (
        SELECT * FROM {SCHEMA_NAME}.transactions 
        WHERE tx_step >= (SELECT MAX(tx_step) - :days FROM {SCHEMA_NAME}.transactions)
          AND amount >= :min_amount
          AND account_id != counter_party
          AND account_id IS NOT NULL 
          AND counter_party IS NOT NULL
    ),
    all_connections AS (
        -- outgoing (account_id -> counter_party)
        SELECT 
            account_id AS source,
            counter_party AS target,
            COUNT(*) AS tx_count,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount,
            MAX(amount) AS max_amount
        FROM recent_tx
        GROUP BY account_id, counter_party
        
        UNION ALL
        
        -- incoming (counter_party -> account_id)
        SELECT 
            counter_party AS source,
            account_id AS target,
            COUNT(*) AS tx_count,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount,
            MAX(amount) AS max_amount
        FROM recent_tx
        GROUP BY counter_party, account_id
    )
    SELECT 
        source,
        target,
        SUM(tx_count) AS tx_count,
        SUM(total_amount) AS total_amount,
        AVG(avg_amount) AS avg_amount,
        MAX(max_amount) AS max_amount
    FROM all_connections
    GROUP BY source, target
    ORDER BY total_amount DESC
    LIMIT :limit_edges
    """
    return query_df(sql, {"days": days, "min_amount": min_amount, "limit_edges": limit_edges})


@st.cache_data(ttl=CACHE_TTL)
def get_all_accounts(limit: int = 1000) -> pd.DataFrame:
    sql = f"""
    SELECT a.account_id,
           a.model_id,
           a.init_balance,
           p.name AS party_name,
           p.type AS party_type
    FROM {SCHEMA_NAME}.accounts a
    LEFT JOIN {SCHEMA_NAME}.parties p ON p.party_id = a.party_id
    ORDER BY a.account_id
    LIMIT :limit
    """
    return query_df(sql, {"limit": limit})


@st.cache_data(ttl=CACHE_TTL)
def get_account_profile(account_id: int) -> Dict[str, Any]:
    kyc_sql = f"""
    SELECT a.account_id,
           a.model_id,
           a.init_balance,
           a.start_date,
           a.end_date,
           a.country,
           a.business,
           p.name AS party_name,
           p.type AS party_type,
           p.country AS party_country
    FROM {SCHEMA_NAME}.accounts a
    LEFT JOIN {SCHEMA_NAME}.parties p ON p.party_id = a.party_id
    WHERE a.account_id = :account_id
    """

    stats_sql = f"""
    SELECT COUNT(*)                                 AS total_transactions,
           COALESCE(SUM(amount), 0)                 AS total_amount,
           COALESCE(AVG(amount), 0)                 AS avg_amount,
           COALESCE(MAX(amount), 0)                 AS max_amount,
           COALESCE(MIN(amount), 0)                 AS min_amount,
           COUNT(DISTINCT counter_party)            AS unique_counterparties,
           SUM(CASE WHEN is_suspicious THEN 1 ELSE 0 END) AS suspicious_count,
           COALESCE(SUM(CASE WHEN is_suspicious THEN amount ELSE 0 END), 0)
             AS suspicious_amount,
           COALESCE(SUM(CASE 
             WHEN tx_type IN ('D', 'C') THEN amount
             WHEN tx_type IN ('W', 'T', 'R') THEN -amount
             ELSE 0 END), 0) AS net_balance_change,
           COALESCE(MIN(tx_step), 0)                AS first_tx_day,
           COALESCE(MAX(tx_step), 0)                AS last_tx_day
    FROM {SCHEMA_NAME}.transactions
    WHERE account_id = :account_id
    """

    recent_sql = f"""
    SELECT transaction_id,
           tx_step AS day,
           amount,
           tx_type,
           counter_party,
           is_suspicious
    FROM {SCHEMA_NAME}.transactions
    WHERE account_id = :account_id
    ORDER BY tx_step DESC, transaction_id DESC
    LIMIT 100
    """

    alerts_sql = f"""
    SELECT alert_type,
           alert_score,
           status,
           created_at::date AS day
    FROM {SCHEMA_NAME}.alerts
    WHERE account_id = :account_id
    ORDER BY created_at DESC
    LIMIT 20
    """

    kyc_df = query_df(kyc_sql, {"account_id": account_id})
    stats_df = query_df(stats_sql, {"account_id": account_id})
    recent_df = query_df(recent_sql, {"account_id": account_id})
    alerts_df = query_df(alerts_sql, {"account_id": account_id})

    return {
        "kyc": kyc_df.iloc[0].to_dict() if not kyc_df.empty else {},
        "stats": stats_df.iloc[0].to_dict() if not stats_df.empty else {},
        "recent": recent_df,
        "alerts": alerts_df,
    }


@st.cache_data(ttl=CACHE_TTL)
def get_account_time_series(account_id: int) -> pd.DataFrame:
    sql = f"""
    WITH acct AS (
        SELECT init_balance
        FROM {SCHEMA_NAME}.accounts
        WHERE account_id = :account_id
    ),
    daily_tx AS (
        SELECT
            tx_step AS day,
            SUM(CASE
                WHEN tx_type IN ('D','C') THEN amount
                WHEN tx_type IN ('W','T','R') THEN -amount
                ELSE 0
            END) AS net_change
        FROM {SCHEMA_NAME}.transactions
        WHERE account_id = :account_id
        GROUP BY tx_step
    ),
    baseline AS (
        SELECT (MIN(day) - 1) AS day, 0::numeric AS net_change
        FROM daily_tx
    ),
    daily AS (
        SELECT * FROM baseline
        UNION ALL
        SELECT * FROM daily_tx
    )
    SELECT
        d.day,
        (SELECT init_balance FROM acct) + COALESCE(SUM(d.net_change) OVER (
            ORDER BY d.day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 0) AS balance,
        d.net_change
    FROM daily d
    ORDER BY d.day
    """
    return query_df(sql, {"account_id": account_id})


@st.cache_data(ttl=CACHE_TTL)
def load_training_data(days: int = 2000) -> pd.DataFrame:
    sql = f"""
    SELECT transaction_id, account_id, tx_step AS day, amount, tx_type, counter_party, is_suspicious
    FROM {SCHEMA_NAME}.transactions
    WHERE tx_step >= (SELECT MAX(tx_step) - :days FROM {SCHEMA_NAME}.transactions)
    ORDER BY tx_step
    """
    return query_df(sql, {"days": days})


@st.cache_data(ttl=CACHE_TTL)
def get_recent_account_transactions(account_id: int, limit: int = 50) -> pd.DataFrame:
    sql = f"""
    SELECT transaction_id, tx_step AS day, amount, tx_type, counter_party, is_suspicious
    FROM {SCHEMA_NAME}.transactions
    WHERE account_id = :account_id
    ORDER BY tx_step DESC, transaction_id DESC
    LIMIT :limit
    """
    return query_df(sql, {"account_id": account_id, "limit": limit}) 