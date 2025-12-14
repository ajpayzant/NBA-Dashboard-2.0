from __future__ import annotations

import datetime as dt
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.endpoints import (
    leaguestandings,
    leaguedashteamstats,
    leaguedashplayerstats,
)

# ----------------------------
# Helpers
# ----------------------------

def _teams_lookup_df() -> pd.DataFrame:
    """Static NBA teams lookup (TEAM_ID -> TEAM_ABBREVIATION, TEAM_NAME)."""
    t = pd.DataFrame(nba_teams_static.get_teams())
    # nba_api static teams columns: id, full_name, abbreviation, nickname, city, state, year_founded
    t = t.rename(
        columns={
            "id": "TEAM_ID",
            "abbreviation": "TEAM_ABBREVIATION",
            "full_name": "TEAM_NAME",
        }
    )
    t["TEAM_ID"] = pd.to_numeric(t["TEAM_ID"], errors="coerce")
    t["TEAM_ABBREVIATION"] = t["TEAM_ABBREVIATION"].astype(str)
    t["TEAM_NAME"] = t["TEAM_NAME"].astype(str)
    return t[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].dropna()

def _coerce_team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TEAM_ABBREVIATION exists when possible."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # Common alternative column names seen in some endpoints / versions
    rename_map = {}
    for c in out.columns:
        if c.upper() in ("TEAM_ABBREV", "TEAM_ABBR", "ABBREVIATION"):
            rename_map[c] = "TEAM_ABBREVIATION"
        if c.upper() in ("TEAMID", "TEAM_ID"):
            rename_map[c] = "TEAM_ID"
        if c.upper() in ("TEAMNAME", "TEAM_NAME"):
            rename_map[c] = "TEAM_NAME"
    if rename_map:
        out = out.rename(columns=rename_map)

    # If TEAM_ABBREVIATION missing but TEAM_ID exists, merge from lookup
    if "TEAM_ABBREVIATION" not in out.columns:
        if "TEAM_ID" in out.columns:
            out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce")
            lk = _teams_lookup_df()
            out = out.merge(lk, on="TEAM_ID", how="left", suffixes=("", "_LK"))
            # If TEAM_NAME exists in original, keep it; else use lookup.
            if "TEAM_NAME" not in df.columns and "TEAM_NAME_LK" in out.columns:
                out["TEAM_NAME"] = out["TEAM_NAME_LK"]
            if "TEAM_ABBREVIATION_LK" in out.columns:
                out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION_LK"]
            out = out.drop(columns=[c for c in ["TEAM_NAME_LK", "TEAM_ABBREVIATION_LK"] if c in out.columns])

    # Clean dtype
    if "TEAM_ABBREVIATION" in out.columns:
        out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].astype(str)
        out.loc[out["TEAM_ABBREVIATION"].isin(["nan", "None", "NA"]), "TEAM_ABBREVIATION"] = np.nan

    return out

def _safe_merge_team(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge standings/team stats safely.
    Prefer TEAM_ABBREVIATION if present on both sides; else use TEAM_ID.
    """
    left2 = _coerce_team_abbr(left)
    right2 = _coerce_team_abbr(right)

    # Prefer ABBR merge if both have it
    if "TEAM_ABBREVIATION" in left2.columns and "TEAM_ABBREVIATION" in right2.columns:
        l = left2.copy()
        r = right2.copy()
        l["TEAM_ABBREVIATION"] = l["TEAM_ABBREVIATION"].astype(str)
        r["TEAM_ABBREVIATION"] = r["TEAM_ABBREVIATION"].astype(str)
        return l.merge(r, on="TEAM_ABBREVIATION", how=how, suffixes=("", "_R"))

    # Else use TEAM_ID if possible
    if "TEAM_ID" in left2.columns and "TEAM_ID" in right2.columns:
        l = left2.copy()
        r = right2.copy()
        l["TEAM_ID"] = pd.to_numeric(l["TEAM_ID"], errors="coerce")
        r["TEAM_ID"] = pd.to_numeric(r["TEAM_ID"], errors="coerce")
        return l.merge(r, on="TEAM_ID", how=how, suffixes=("", "_R"))

    # If we cannot merge, just return left (app will show partial table)
    return left2


# ----------------------------
# Public fetchers used by app.py
# ----------------------------

def fetch_league_team_summary(season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      team_table: Standings + Team Stats (per game)
      opp_table:  Opponent allowed stats (per game), if available
    """
    # Standings
    st = leaguestandings.LeagueStandings(season=season, season_type="Regular Season").get_data_frames()[0]
    st = st.copy()

    # Normalize expected cols
    # LeagueStandings usually has: TeamID, TeamName, TeamAbbreviation, WINS, LOSSES, WinPCT, Conference, ...
    st = st.rename(
        columns={
            "TeamID": "TEAM_ID",
            "TeamName": "TEAM_NAME",
            "TeamAbbreviation": "TEAM_ABBREVIATION",
            "WINS": "W",
            "LOSSES": "L",
            "WinPCT": "WIN_PCT",
            "Conference": "CONFERENCE",
        }
    )
    st = _coerce_team_abbr(st)

    # Team stats (base) and opponent stats
    # leaguedashteamstats returns multiple measures; we grab PerGame where possible
    td = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    td = td.copy()
    td = td.rename(columns={"TEAM_ID": "TEAM_ID", "TEAM_NAME": "TEAM_NAME", "TEAM_ABBREVIATION": "TEAM_ABBREVIATION"})
    td = _coerce_team_abbr(td)

    # Advanced team stats (optional but often present in same endpoint if you call again; keeping minimal)
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]
    adv = adv.copy()
    adv = _coerce_team_abbr(adv)

    # Opponent (allowed) stats
    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent",
    ).get_data_frames()[0]
    opp = opp.copy()
    opp = _coerce_team_abbr(opp)

    # Merge standings + base + adv
    merged = _safe_merge_team(st, td, how="left")
    merged = _safe_merge_team(merged, adv, how="left")

    # Clean up / keep your existing table behavior: if W/L exist, enforce ints
    if "W" in merged.columns:
        merged["W"] = pd.to_numeric(merged["W"], errors="coerce").fillna(0).astype(int)
    if "L" in merged.columns:
        merged["L"] = pd.to_numeric(merged["L"], errors="coerce").fillna(0).astype(int)
    if "WIN_PCT" in merged.columns:
        merged["WIN_PCT"] = pd.to_numeric(merged["WIN_PCT"], errors="coerce")

    # Build opp table matched by TEAM_ID/ABBR safely
    opp_merged = _safe_merge_team(st, opp, how="left")

    return merged, opp_merged


def fetch_league_player_leaders(season: str) -> pd.DataFrame:
    """Leaguewide player leaders (per-game) used by League tab."""
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0].copy()

    # Normalize team col name for UI
    if "TEAM_ABBREVIATION" not in df.columns and "TEAM_ABBREV" in df.columns:
        df = df.rename(columns={"TEAM_ABBREV": "TEAM_ABBREVIATION"})
    return df
