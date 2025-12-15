from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd


def season_labels(start_year: int, end_year: int) -> list[str]:
    # 2015 -> "2015-16"
    out = []
    for y in range(start_year, end_year + 1):
        out.append(f"{y}-{str(y+1)[-2:]}")
    return out


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def minutes_str_to_float(x) -> float:
    """
    nba_api often returns MIN as 'mm:ss' or already numeric.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    if ":" in s:
        mm, ss = s.split(":", 1)
        try:
            return float(mm) + float(ss) / 60.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def safe_to_datetime(series: pd.Series) -> pd.Series:
    # Handles mixed formats safely
    return pd.to_datetime(series, errors="coerce", utc=False)


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def normalize_team_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "TEAM_ID" in df.columns:
        df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    return df


def normalize_team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in ["TEAM_ABBREVIATION", "TEAM"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def normalize_player_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "PLAYER_ID" in df.columns:
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    return df


def add_basic_shooting_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 2PM/3PM and PRA if base columns exist.
    """
    if df is None or df.empty:
        return df

    if "FGM" in df.columns and "FG3M" in df.columns and "FG2M" not in df.columns:
        df["FG2M"] = pd.to_numeric(df["FGM"], errors="coerce").fillna(0) - pd.to_numeric(df["FG3M"], errors="coerce").fillna(0)

    for col in ["PTS", "REB", "AST"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(c in df.columns for c in ["PTS", "REB", "AST"]) and "PRA" not in df.columns:
        df["PRA"] = df["PTS"].fillna(0) + df["REB"].fillna(0) + df["AST"].fillna(0)

    return df
