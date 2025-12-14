from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
from nba_api.stats.static import teams as nba_teams_static

LA_TZ = pytz.timezone("America/Los_Angeles")


def season_labels(start_year: int = 2015, end_year: Optional[int] = None):
    """
    Returns NBA season strings like '2024-25' through end_year.
    """
    if end_year is None:
        end_year = dt.datetime.now(dt.timezone.utc).year
    labels = []
    for y in range(start_year, end_year + 1):
        labels.append(f"{y}-{str((y+1) % 100).zfill(2)}")
    return labels


def season_to_api(season_label: str) -> str:
    # nba_api expects '2024-25' format for many endpoints; keep same
    return season_label


def today_la_date() -> dt.date:
    return dt.datetime.now(LA_TZ).date()


def coerce_team_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize TEAM_ID, TEAM_ABBREVIATION and TEAM_NAME types to avoid parquet dtype drift.
    """
    out = df.copy()

    if "TEAM_ID" in out.columns:
        out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce").astype("Int64")

    if "TEAM_ABBREVIATION" in out.columns:
        out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].astype("string")
        out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].fillna(pd.NA)

    if "TEAM_NAME" in out.columns:
        out["TEAM_NAME"] = out["TEAM_NAME"].astype("string")

    return out


def coerce_game_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("GAME_ID", "GAME_DATE", "GAME_DATE_EST"):
        if c in out.columns:
            if "DATE" in c:
                out[c] = pd.to_datetime(out[c], errors="coerce")
            else:
                out[c] = out[c].astype("string")
    return out


def team_id_maps() -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Returns (team_id->abbrev, team_id->full_name) based on static team list.
    """
    t = nba_teams_static.get_teams()
    id_to_abbr = {}
    id_to_name = {}
    for row in t:
        tid = int(row["id"])
        id_to_abbr[tid] = row["abbreviation"]
        id_to_name[tid] = row["full_name"]
    return id_to_abbr, id_to_name


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def minutes_str_to_float(min_str) -> float:
    """
    Converts NBA 'MIN' fields, which can be '32:15' or '32.0' or None, to float minutes.
    """
    if min_str is None or (isinstance(min_str, float) and np.isnan(min_str)):
        return 0.0
    s = str(min_str).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return 0.0
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            m = safe_float(parts[0], 0.0)
            sec = safe_float(parts[1], 0.0)
            return m + sec / 60.0
    return safe_float(s, 0.0)


def add_basic_shooting_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds FG2M, FG2A, FG3M, FG3A if possible; also PRA.
    Expects columns: FGM, FGA, FG3M, FG3A, PTS, REB, AST
    """
    out = df.copy()
    for col in ["FGM", "FGA", "FG3M", "FG3A", "PTS", "REB", "AST"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "FGM" in out.columns and "FG3M" in out.columns:
        out["FG2M"] = (out["FGM"] - out["FG3M"]).clip(lower=0)
    if "FGA" in out.columns and "FG3A" in out.columns:
        out["FG2A"] = (out["FGA"] - out["FG3A"]).clip(lower=0)

    if all(c in out.columns for c in ("PTS", "REB", "AST")):
        out["PRA"] = out["PTS"].fillna(0) + out["REB"].fillna(0) + out["AST"].fillna(0)

    return out
