import os
import duckdb
import pandas as pd
from pathlib import Path

DEFAULT_DB_PATH = os.path.join("data", "nba_warehouse.duckdb")

def resolve_db_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]  # goes up from src/ to repo root
    db_path = repo_root / "data" / "nba_warehouse.duckdb"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at: {db_path}. "
            "Make sure you committed it to your repo at /data/nba_warehouse.duckdb"
        )

    return str(db_path)

def query_df(sql: str, params=None, db_path: str | None = None) -> pd.DataFrame:
    db_path = db_path or resolve_db_path()
    con = duckdb.connect(db_path, read_only=True)
    try:
        if params is None:
            return con.execute(sql).df()
        return con.execute(sql, params).df()
    finally:
        con.close()
