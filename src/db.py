import os
import duckdb
import pandas as pd

DEFAULT_DB_PATH = os.path.join("data", "nba_warehouse.duckdb")

def resolve_db_path() -> str:
    return os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH)

def query_df(sql: str, params=None, db_path: str | None = None) -> pd.DataFrame:
    db_path = db_path or resolve_db_path()
    con = duckdb.connect(db_path, read_only=True)
    try:
        if params is None:
            return con.execute(sql).df()
        return con.execute(sql, params).df()
    finally:
        con.close()
