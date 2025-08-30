"""SQL queries for the Global Dashboard"""

import os
import sys
import pandas as pd 
import streamlit as st


from constants import (
    BANK_MAPPING,
    TRANSACTION_TYPE_MAPPING,
    SCHEMA_NAME,
    CACHE_TTL
)
def build_type_case_statement() -> str:
    cases = []
    for type_code, type_name in TRANSACTION_TYPE_MAPPING.items():
        cases.append(f"WHEN '{type_code}' THEN '{type_name}'")
    return f"""CASE COALESCE(tx_type, '?')
        {' '.join(cases)}
        ELSE 'Other'
    END"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../../")))
from src.db import query_df


@st.cache_data(ttl=CACHE_TTL)
def get_global_kpis() -> dict:
    """Get global KPIs: transactions, amount, alerts, suspicious %, losses"""
    try:
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        WITH base AS (
            SELECT COUNT(*) AS total_tx,
                   SUM(amount)::numeric AS total_amount,
                   COUNT(*) FILTER (WHERE is_suspicious = true)::numeric 
                   AS suspect_tx,
                   SUM(amount) FILTER (WHERE is_suspicious = true)::numeric 
                   AS losses
            FROM {SCHEMA_NAME}.transactions
        ), alerts AS (
            SELECT COUNT(*) AS open_alerts
            FROM {SCHEMA_NAME}.alerts
            WHERE LOWER(COALESCE(status,'a')) IN ('a','open','active','o')
        )
        SELECT b.total_tx AS total_transactions,
               ROUND(COALESCE(b.total_amount,0), 2) AS montant_total,
               a.open_alerts AS alertes_ouvertes,
               CASE WHEN b.total_tx = 0 THEN 0
                    ELSE ROUND((COALESCE(b.suspect_tx,0) * 100.0 / 
                               b.total_tx), 2)
               END AS pourcentage_suspect,
               COALESCE(ROUND(b.losses, 2), 0) AS pertes_estimees
        FROM base b CROSS JOIN alerts a;
        """
        row = query_df(sql).iloc[0].to_dict()
        return {k: (v.item() if hasattr(v, 'item') else v) 
                for k, v in row.items()}
    except Exception as e:
        print(f"Error KPIs: {e}")
        return {
            'total_transactions': 0,
            'montant_total': 0.0,
            'alertes_ouvertes': 0,
            'pourcentage_suspect': 0.0,
            'pertes_estimees': 0.0
        }


@st.cache_data(ttl=CACHE_TTL)
def get_daily_evolution() -> pd.DataFrame:
    """Get daily evolution: transaction count and volume per day"""
    try:
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        SELECT tx_step AS day,
               COUNT(*) AS nb_transactions,
               SUM(amount) AS daily_volume
        FROM {SCHEMA_NAME}.transactions
        GROUP BY tx_step
        ORDER BY tx_step;
        """
        return query_df(sql)
    except Exception as e:
        print(f"Error daily evolution: {e}")
        return pd.DataFrame(columns=['day', 'nb_transactions', 'daily_volume'])


def get_daily_evolution_by_bank(top_n: int = 5) -> pd.DataFrame:
    """Get daily evolution by bank for top N banks"""
    try:
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        WITH mapped AS (
            SELECT 
                t.tx_step AS day,
                CASE COALESCE(a.model_id, -1)
                    WHEN 0 THEN 'BNP Paribas'
                    WHEN 1 THEN 'Société Générale'
                    WHEN 2 THEN 'Crédit Agricole'
                    WHEN 3 THEN 'Banque Populaire'
                    WHEN 4 THEN 'Caisse d''Épargne'
                    WHEN 5 THEN 'Crédit Mutuel'
                    WHEN 6 THEN 'La Banque Postale'
                    WHEN 7 THEN 'HSBC France'
                    WHEN 8 THEN 'LCL (Crédit Lyonnais)'
                    WHEN 9 THEN 'Boursorama Banque'
                    ELSE 'Other'
                END AS bank,
                t.amount
            FROM {SCHEMA_NAME}.transactions t
            LEFT JOIN {SCHEMA_NAME}.accounts a ON t.account_id = a.account_id
        ), top_banks AS (
            SELECT bank, COUNT(*) AS nb
            FROM mapped
            GROUP BY bank
            ORDER BY nb DESC
            LIMIT {top_n}
        )
        SELECT m.day,
               m.bank,
               COUNT(*) AS nb_transactions,
               SUM(m.amount) AS daily_volume
        FROM mapped m
        JOIN top_banks tb ON tb.bank = m.bank
        GROUP BY m.day, m.bank
        ORDER BY m.day, m.bank;
        """
        return query_df(sql)
    except Exception as e:
        print(f"Error daily evolution by bank: {e}")
        return pd.DataFrame(columns=["day", "bank", "nb_transactions", 
                                   "daily_volume"])


def get_bank_distribution() -> pd.DataFrame:
    """Get transaction distribution by bank"""
    try:
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        WITH mapped AS (
            SELECT 
                CASE COALESCE(a.model_id, -1)
                    WHEN 0 THEN 'BNP Paribas'
                    WHEN 1 THEN 'Société Générale'
                    WHEN 2 THEN 'Crédit Agricole'
                    WHEN 3 THEN 'Banque Populaire'
                    WHEN 4 THEN 'Caisse d''Épargne'
                    WHEN 5 THEN 'Crédit Mutuel'
                    WHEN 6 THEN 'La Banque Postale'
                    WHEN 7 THEN 'HSBC France'
                    WHEN 8 THEN 'LCL (Crédit Lyonnais)'
                    WHEN 9 THEN 'Boursorama Banque'
                    ELSE 'Other'
                END AS bank,
                t.transaction_id,
                t.amount
            FROM {SCHEMA_NAME}.transactions t
            LEFT JOIN {SCHEMA_NAME}.accounts a ON t.account_id = a.account_id
        )
        SELECT bank,
               COUNT(transaction_id) AS nb_transactions,
               SUM(amount) AS volume
        FROM mapped
        GROUP BY bank
        ORDER BY nb_transactions DESC
        LIMIT 10;
        """
        return query_df(sql)
    except Exception as e:
        print(f"Error bank distribution: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def get_type_distribution() -> pd.DataFrame:
    """Get transaction distribution by type"""
    try:
        type_cases = build_type_case_statement()
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        WITH mapped AS (
            SELECT 
                {type_cases} AS transaction_type,
                amount
            FROM {SCHEMA_NAME}.transactions
        )
        SELECT transaction_type,
               COUNT(*) AS nb_transactions,
               SUM(amount) AS volume
        FROM mapped
        GROUP BY transaction_type
        ORDER BY nb_transactions DESC;
        """
        return query_df(sql)
    except Exception as e:
        print(f"Error type distribution: {e}")
        return pd.DataFrame()


def get_type_distribution_by_bank() -> pd.DataFrame:
    """Get transaction distribution by type and bank"""
    try:
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        WITH base AS (
            SELECT 
                CASE COALESCE(t.tx_type, '?')
                    WHEN 'W' THEN 'Withdrawal'
                    WHEN 'T' THEN 'Transfer'
                    WHEN 'D' THEN 'Deposit'
                    WHEN 'C' THEN 'Cash'
                    WHEN 'R' THEN 'Remittance'
                    ELSE 'Other'
                END AS transaction_type,
                CASE COALESCE(a.model_id, -1)
                    WHEN 0 THEN 'BNP Paribas'
                    WHEN 1 THEN 'Société Générale'
                    WHEN 2 THEN 'Crédit Agricole'
                    WHEN 3 THEN 'Banque Populaire'
                    WHEN 4 THEN 'Caisse d''Épargne'
                    WHEN 5 THEN 'Crédit Mutuel'
                    WHEN 6 THEN 'La Banque Postale'
                    WHEN 7 THEN 'HSBC France'
                    WHEN 8 THEN 'LCL (Crédit Lyonnais)'
                    WHEN 9 THEN 'Boursorama Banque'
                    ELSE 'Other'
                END AS bank,
                t.amount
            FROM {SCHEMA_NAME}.transactions t
            LEFT JOIN {SCHEMA_NAME}.accounts a ON t.account_id = a.account_id
        )
        SELECT bank, transaction_type,
               COUNT(*) AS nb_transactions,
               SUM(amount) AS volume
        FROM base
        GROUP BY bank, transaction_type
        ORDER BY bank, transaction_type;
        """
        return query_df(sql)
    except Exception as e:
        print(f"Error type distribution by bank: {e}")
        return pd.DataFrame()


def get_banks_by_activity(limit: int = 20) -> pd.DataFrame:
    """Get banks ordered by transaction count"""
    try:
        sql = f"""
        SET search_path={SCHEMA_NAME},public;
        WITH mapped AS (
            SELECT 
                CASE COALESCE(a.model_id, -1)
                    WHEN 0 THEN 'BNP Paribas'
                    WHEN 1 THEN 'Société Générale'
                    WHEN 2 THEN 'Crédit Agricole'
                    WHEN 3 THEN 'Banque Populaire'
                    WHEN 4 THEN 'Caisse d''Épargne'
                    WHEN 5 THEN 'Crédit Mutuel'
                    WHEN 6 THEN 'La Banque Postale'
                    WHEN 7 THEN 'HSBC France'
                    WHEN 8 THEN 'LCL (Crédit Lyonnais)'
                    WHEN 9 THEN 'Boursorama Banque'
                    ELSE 'Other'
                END AS bank
            FROM {SCHEMA_NAME}.transactions t
            LEFT JOIN {SCHEMA_NAME}.accounts a ON t.account_id = a.account_id
        )
        SELECT bank, COUNT(*) AS nb_transactions
        FROM mapped
        GROUP BY bank
        ORDER BY nb_transactions DESC
        LIMIT {limit};
        """
        return query_df(sql)
    except Exception as e:
        print(f"Error get_banks_by_activity: {e}")
        return pd.DataFrame(columns=['bank', 'nb_transactions'])


def _build_bank_case_statement_deprecated():
    """Build SQL CASE statement for bank mapping"""
    cases = []
    for model_id, bank_name in BANK_MAPPING.items():
        cases.append(f"WHEN {model_id} THEN '{bank_name}'")
    
    return f"""CASE COALESCE(a.model_id, -1)
        {' '.join(cases)}
        ELSE 'Other'
    END"""


def _build_type_case_statement_deprecated():
    """Build SQL CASE statement for transaction type mapping"""
    cases = []
    for type_code, type_name in TRANSACTION_TYPE_MAPPING.items():
        cases.append(f"WHEN '{type_code}' THEN '{type_name}'")
    
    return f"""CASE COALESCE(tx_type, '?')
        {' '.join(cases)}
        ELSE 'Other'
    END"""


def _get_default_kpis_deprecated():
    """Default KPIs when database is unavailable"""
    return {
        'total_transactions': 0,
        'montant_total': 0.0,
        'alertes_ouvertes': 0,
        'pourcentage_suspect': 0.0,
        'pertes_estimees': 0.0
    }


def _get_default_daily_evolution_deprecated():
    """Default daily evolution when database is unavailable"""
    days = list(range(1, 31))
    return pd.DataFrame({
        'day': days,
        'nb_transactions': np.random.randint(800, 2200, len(days)),
        'daily_volume': np.random.randint(25000, 70000, len(days))
    }) 