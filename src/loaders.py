from __future__ import annotations

import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguestandings,
    leaguedashteamstats,
    leaguedashplayerstats,
)


# ----------------------------
# Team lookup (TEAM_ID <-> ABBR)
# ----------------------------

def _teams_lookup_df() -> pd.DataFrame:
    t = pd.DataFrame(nba_teams_static.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "abbreviation": "TEAM_ABBREVIATION", "full_name": "TEAM_NAME"})
    t["TEAM_ID"] = pd.to_numeric(t["TEAM_ID"], errors="coerce")
    t["TEAM_ABBREVIATION"] = t["TEAM_ABBREVIATION"].astype(str)
    t["TEAM_NAME"] = t["TEAM_NAME"].astype(str)
    return t[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].dropna()


def _ensure_team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TEAM_ABBREVIATION exists (derive from TEAM_ID via static lookup if needed)."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # normalize common alt names
    rename_map = {}
    for c in list(out.columns):
        cu = c.upper()
        if cu in ("TEAMID", "TEAM_ID"):
            rename_map[c] = "TEAM_ID"
        if cu in ("TEAMNAME", "TEAM_NAME"):
            rename_map[c] = "TEAM_NAME"
        if cu in ("TEAMABBREVIATION", "TEAM_ABBREVIATION", "TEAM_ABBR", "TEAM_ABBREV", "ABBREVIATION"):
            rename_map[c] = "TEAM_ABBREVIATION"
    if rename_map:
        out = out.rename(columns=rename_map)

    # if missing ABBR but has TEAM_ID, join lookup
    if "TEAM_ABBREVIATION" not in out.columns and "TEAM_ID" in out.columns:
        out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce")
        lk = _teams_lookup_df()
        out = out.merge(lk, on="TEAM_ID", how="left", suffixes=("", "_LK"))
        if "TEAM_ABBREVIATION_LK" in out.columns:
            out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION_LK"]
        if "TEAM_NAME" not in df.columns and "TEAM_NAME_LK" in out.columns:
            out["TEAM_NAME"] = out["TEAM_NAME_LK"]
        out = out.drop(columns=[c for c in ["TEAM_ABBREVIATION_LK", "TEAM_NAME_LK"] if c in out.columns])

    if "TEAM_ABBREVIATION" in out.columns:
        out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].astype(str)
        out.loc[out["TEAM_ABBREVIATION"].isin(["nan", "None", "NA"]), "TEAM_ABBREVIATION"] = np.nan

    return out


def _safe_merge_team(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """
    Merge two frames safely:
      - Prefer TEAM_ID if present on both (most stable)
      - Else use TEAM_ABBREVIATION if present on both
      - Else return left unchanged
    """
    l = _ensure_team_abbr(left)
    r = _ensure_team_abbr(right)

    if "TEAM_ID" in l.columns and "TEAM_ID" in r.columns:
        l2 = l.copy()
        r2 = r.copy()
        l2["TEAM_ID"] = pd.to_numeric(l2["TEAM_ID"], errors="coerce")
        r2["TEAM_ID"] = pd.to_numeric(r2["TEAM_ID"], errors="coerce")
        return l2.merge(r2, on="TEAM_ID", how=how, suffixes=("", "_R"))

    if "TEAM_ABBREVIATION" in l.columns and "TEAM_ABBREVIATION" in r.columns:
        l2 = l.copy()
        r2 = r.copy()
        l2["TEAM_ABBREVIATION"] = l2["TEAM_ABBREVIATION"].astype(str)
        r2["TEAM_ABBREVIATION"] = r2["TEAM_ABBREVIATION"].astype(str)
        return l2.merge(r2, on="TEAM_ABBREVIATION", how=how, suffixes=("", "_R"))

    return l


# ============================
# Functions app.py IMPORTS
# ============================

def fetch_daily_scoreboard(game_date: dt.date) -> pd.DataFrame:
    """
    One row per matchup:
      - MATCHUP (AWY @ HOME)
      - START_TIME_LOCAL
      - STATUS (Final / Scheduled / In Progress)
      - AWAY_SCORE / HOME_SCORE when available
    """
    # nba_api expects MM/DD/YYYY
    dstr = game_date.strftime("%m/%d/%Y")
    sb = scoreboardv2.ScoreboardV2(game_date=dstr).get_data_frames()

    # ScoreboardV2 frames: [GameHeader, LineScore, SeriesStandings, ...]
    if not sb or len(sb) < 2:
        return pd.DataFrame()

    gh = sb[0].copy()
    ls = sb[1].copy()

    # GameHeader has: GAME_ID, GAME_STATUS_TEXT, GAME_DATE_EST, GAME_SEQUENCE, LIVE_PERIOD, ...
    # LineScore has: GAME_ID, TEAM_ID, TEAM_ABBREVIATION, PTS, etc.
    gh = gh.rename(columns={"GAME_ID": "GAME_ID", "GAME_STATUS_TEXT": "STATUS"})
    ls = _ensure_team_abbr(ls)

    # build away/home rows from LineScore via GAME_ID + HOME_TEAM_ID/VISITOR_TEAM_ID in GameHeader
    # GameHeader usually has HOME_TEAM_ID and VISITOR_TEAM_ID
    for col in ["HOME_TEAM_ID", "VISITOR_TEAM_ID"]:
        if col in gh.columns:
            gh[col] = pd.to_numeric(gh[col], errors="coerce")

    ls["TEAM_ID"] = pd.to_numeric(ls.get("TEAM_ID"), errors="coerce")

    # map scores/abbr
    away = ls.rename(columns={"PTS": "AWAY_SCORE", "TEAM_ABBREVIATION": "AWAY"})[["GAME_ID", "TEAM_ID", "AWAY", "AWAY_SCORE"]]
    home = ls.rename(columns={"PTS": "HOME_SCORE", "TEAM_ABBREVIATION": "HOME"})[["GAME_ID", "TEAM_ID", "HOME", "HOME_SCORE"]]

    out_rows = []
    for _, g in gh.iterrows():
        gid = g.get("GAME_ID")
        home_id = g.get("HOME_TEAM_ID")
        away_id = g.get("VISITOR_TEAM_ID")

        hrow = home[home["GAME_ID"] == gid]
        if pd.notna(home_id) and not hrow.empty:
            hrow = hrow[hrow["TEAM_ID"] == home_id]
        arow = away[away["GAME_ID"] == gid]
        if pd.notna(away_id) and not arow.empty:
            arow = arow[arow["TEAM_ID"] == away_id]

        home_abbr = hrow["HOME"].iloc[0] if not hrow.empty else None
        away_abbr = arow["AWAY"].iloc[0] if not arow.empty else None
        home_pts = hrow["HOME_SCORE"].iloc[0] if not hrow.empty else np.nan
        away_pts = arow["AWAY_SCORE"].iloc[0] if not arow.empty else np.nan

        # start time
        # GameHeader has GAME_DATE_EST (timestamp), or you might have GAME_TIME_EST depending on version
        start_time = None
        if "GAME_DATE_EST" in gh.columns and pd.notna(g.get("GAME_DATE_EST")):
            try:
                # often already datetime-like; keep as string
                start_time = pd.to_datetime(g["GAME_DATE_EST"]).strftime("%Y-%m-%d %I:%M %p")
            except Exception:
                start_time = str(g.get("GAME_DATE_EST"))

        matchup = None
        if away_abbr and home_abbr:
            matchup = f"{away_abbr} @ {home_abbr}"

        out_rows.append(
            {
                "GAME_ID": gid,
                "MATCHUP": matchup,
                "START_TIME_LOCAL": start_time,
                "STATUS": g.get("STATUS"),
                "AWAY": away_abbr,
                "HOME": home_abbr,
                "AWAY_SCORE": away_pts,
                "HOME_SCORE": home_pts,
            }
        )

    out = pd.DataFrame(out_rows)
    # nice formatting: show scores only when they exist
    return out


def fetch_league_team_summary(season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      team_table: Standings + Team Stats (sortable)
      opp_table:  Standings + Opponent allowed stats
    """
    # Standings
    st = leaguestandings.LeagueStandings(season=season, season_type="Regular Season").get_data_frames()[0].copy()
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
    st = _ensure_team_abbr(st)

    # Team Base per game
    base = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0].copy()
    base = _ensure_team_abbr(base)

    # Team Advanced per game
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0].copy()
    adv = _ensure_team_abbr(adv)

    # Opponent allowed per game
    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent",
    ).get_data_frames()[0].copy()
    opp = _ensure_team_abbr(opp)

    # Merge safely (NO blind merge on missing keys)
    merged = _safe_merge_team(st, base, how="left")
    merged = _safe_merge_team(merged, adv, how="left")

    # normalize W/L
    if "W" in merged.columns:
        merged["W"] = pd.to_numeric(merged["W"], errors="coerce").fillna(0).astype(int)
    if "L" in merged.columns:
        merged["L"] = pd.to_numeric(merged["L"], errors="coerce").fillna(0).astype(int)
    if "WIN_PCT" in merged.columns:
        merged["WIN_PCT"] = pd.to_numeric(merged["WIN_PCT"], errors="coerce")

    opp_merged = _safe_merge_team(st, opp, how="left")
    if "W" in opp_merged.columns:
        opp_merged["W"] = pd.to_numeric(opp_merged["W"], errors="coerce").fillna(0).astype(int)
    if "L" in opp_merged.columns:
        opp_merged["L"] = pd.to_numeric(opp_merged["L"], errors="coerce").fillna(0).astype(int)
    if "WIN_PCT" in opp_merged.columns:
        opp_merged["WIN_PCT"] = pd.to_numeric(opp_merged["WIN_PCT"], errors="coerce")

    return merged, opp_merged


def fetch_league_player_stats(season: str) -> pd.DataFrame:
    """
    Leaguewide player table (per game) used for leader tables.
    """
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0].copy()

    # normalize common col names
    if "TEAM_ABBREVIATION" not in df.columns and "TEAM_ABBREV" in df.columns:
        df = df.rename(columns={"TEAM_ABBREV": "TEAM_ABBREVIATION"})

    return df
