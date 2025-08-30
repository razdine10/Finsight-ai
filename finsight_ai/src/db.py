import os
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_ENGINE: Optional[Engine] = None


def _build_database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL")
    if url:
        # Ensure SQLAlchemy driver prefix
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        if url.startswith("postgresql://") and "+psycopg2" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    host = os.getenv("PGHOST")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    port = os.getenv("PGPORT", "5432")

    if all([host, database, user, password]):
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # Fallback to local default
    return "postgresql+psycopg2://localhost:5432/finsight_amlsim"


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    url = _build_database_url()

    connect_args: Dict[str, Any] = {}
    # Supabase usually requires SSL
    if any(hint in url for hint in ["supabase.co", "aws", "render.com"]):
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