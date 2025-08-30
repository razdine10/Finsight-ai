import os
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Attempt to import streamlit to read secrets when running on Streamlit Cloud
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - streamlit not always available
    st = None  # type: ignore

_ENGINE: Optional[Engine] = None


def _build_database_url() -> Optional[str]:
    # 1) Streamlit secrets: DATABASE_URL directly
    if st is not None:
        try:
            secrets = st.secrets  # type: ignore[attr-defined]
            if "DATABASE_URL" in secrets:
                url = str(secrets["DATABASE_URL"]).strip()
                if url:
                    if url.startswith("postgres://"):
                        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
                    if url.startswith("postgresql://") and "+psycopg2" not in url:
                        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
                    return url

            # 2) Streamlit secrets: [postgres] mapping
            if "postgres" in secrets:
                pg = secrets["postgres"]
                host = str(pg.get("host", "")).strip()
                database = str(pg.get("dbname", pg.get("database", "postgres"))).strip()
                user = str(pg.get("user", "")).strip()
                password = str(pg.get("password", "")).strip()
                port = str(pg.get("port", 5432)).strip()
                if host and user and password:
                    return (
                        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
                    )
        except Exception:
            # If secrets access fails for any reason, continue with env vars
            pass

    # 3) Environment variable: DATABASE_URL
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        if url.startswith("postgresql://") and "+psycopg2" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    # 4) Environment variables: PG* family
    host = os.getenv("PGHOST")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    port = os.getenv("PGPORT", "5432")

    if all([host, database, user, password]):
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # 5) Fallback to local default (developer machines)
    return "postgresql+psycopg2://localhost:5432/finsight_amlsim"


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    url = _build_database_url()

    connect_args: Dict[str, Any] = {}
    # Enforce SSL for common managed Postgres providers like Supabase/Render/AWS
    if any(hint in url for hint in ["supabase", "aws", "render.com"]):
        connect_args["sslmode"] = "require"

    _ENGINE = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    return _ENGINE


def query_df(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df


def execute(sql: str, params: Optional[Dict[str, Any]] = None) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def test_connection() -> bool:
    try:
        df = query_df("SELECT 1 AS ok")
        return not df.empty and df.iloc[0]["ok"] == 1
    except Exception:
        return False 