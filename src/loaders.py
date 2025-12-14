from __future__ import annotations

import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguedashteamstats,
    leaguestandings,
    LeagueDashPlayerStats,
    teamgamelog,
    boxscoretraditionalv2,
)

from .nba_client import retry_api
from .data_store import CacheSpec, read_or_refresh_parquet


# -----------------------------
# TTLs (tune as you like)
# -----------------------------
TTL_SCOREBOARD = 60 * 20          # 20 minutes
TTL_STANDINGS = 60 * 60 * 6       # 6 hours
TTL_LEAGUE_TABLES = 60 * 60 * 24  # daily
TTL_TEAM_GAMELOG = 60 * 60 * 12   # 12 hours
TTL_BOXSCORE = 60 * 60 * 24 * 7   # 7 days (historical boxscores rarely change)


def _season_labels(start: int = 2015, end: Optional[int] = None) -> list[str]:
    if end is None:
        end = dt.datetime.utcnow().year
    def lab(y: int) -> str:
        return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start - 1, -1)]


def _today_yyyymmdd() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def _teams_df() -> pd.DataFrame:
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = t["TEAM_ID"].astype(int)
    return t[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]]


# ============================================================
# Scoreboard (Schedule + finals)
# ============================================================
def get_daily_scoreboard(date_yyyy_mm_dd: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a *clean* scoreboard table for a given date (UTC date string YYYY-MM-DD).
    Includes matchup and time or final score when available.
    """
    key = f"scoreboard_{date_yyyy_mm_dd}"
    spec = CacheSpec(key=key, ttl_seconds=TTL_SCOREBOARD)

    def _fetch() -> pd.DataFrame:
        d = pd.to_datetime(date_yyyy_mm_dd).strftime("%m/%d/%Y")
        frames = retry_api(scoreboardv2.ScoreboardV2, {"game_date": d, "league_id": "00"})
        if not frames:
            return pd.DataFrame()

        # nba_api ScoreboardV2 returns multiple frames: GameHeader + LineScore + etc.
        # We need LineScore for team abbreviations and points.
        game_header = frames[0] if len(frames) > 0 else pd.DataFrame()
        line_score = frames[1] if len(frames) > 1 else pd.DataFrame()

        if game_header.empty or line_score.empty:
            return pd.DataFrame()

        # Normalize
        gh = game_header.copy()
        ls = line_score.copy()

        # Join home/away line scores into one row per game
        # GameHeader has GAME_ID and GAME_STATUS_TEXT
        needed_gh = ["GAME_ID", "GAME_STATUS_TEXT", "GAME_STATUS_ID", "GAME_DATE_EST", "GAMECODE"]
        for c in needed_gh:
            if c not in gh.columns:
                gh[c] = np.nan

        # LineScore has GAME_ID, TEAM_ABBREVIATION, TEAM_ID, PTS, etc.
        needed_ls = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]
        for c in needed_ls:
            if c not in ls.columns:
                ls[c] = np.nan

        # Determine home/away using MATCHUP in GameHeader if present, else use HOME_TEAM_ID/AWAY_TEAM_ID if present
        if "HOME_TEAM_ID" in gh.columns and "VISITOR_TEAM_ID" in gh.columns:
            home_map = gh[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].copy()
        else:
            # fallback: parse GAMECODE like "20251211/NYKBOS"
            home_map = gh[["GAME_ID"]].copy()
            home_map["HOME_TEAM_ID"] = np.nan
            home_map["VISITOR_TEAM_ID"] = np.nan

        # Build home/away rows
        home = ls.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME_TEAM", "PTS": "HOME_PTS"})
        away = ls.rename(columns={"TEAM_ID": "VISITOR_TEAM_ID", "TEAM_ABBREVIATION": "VISITOR_TEAM", "PTS": "VISITOR_PTS"})

        merged = gh.merge(home_map, on="GAME_ID", how="left")

        # Join home team
        if "HOME_TEAM_ID" in merged.columns:
            merged = merged.merge(
                home[["GAME_ID", "HOME_TEAM_ID", "HOME_TEAM", "HOME_PTS"]],
                on=["GAME_ID", "HOME_TEAM_ID"],
                how="left",
            )
            merged = merged.merge(
                away[["GAME_ID", "VISITOR_TEAM_ID", "VISITOR_TEAM", "VISITOR_PTS"]],
                on=["GAME_ID", "VISITOR_TEAM_ID"],
                how="left",
            )
        else:
            # fallback: pivot line_score
            piv = ls.pivot_table(index="GAME_ID", columns="TEAM_ABBREVIATION", values="PTS", aggfunc="first")
            merged["HOME_TEAM"] = np.nan
            merged["VISITOR_TEAM"] = np.nan
            merged["HOME_PTS"] = np.nan
            merged["VISITOR_PTS"] = np.nan

        # Ensure strings for concat (fixes your earlier UFuncNoLoopError)
        for c in ["VISITOR_TEAM", "HOME_TEAM"]:
            if c in merged.columns:
                merged[c] = merged[c].astype("string").fillna("")

        merged["MATCHUP"] = merged["VISITOR_TEAM"].fillna("").astype(str) + " @ " + merged["HOME_TEAM"].fillna("").astype(str)

        # Display status: if final score available show it, else show status text
        def _status_row(r):
            vp = r.get("VISITOR_PTS", np.nan)
            hp = r.get("HOME_PTS", np.nan)
            if pd.notna(vp) and pd.notna(hp) and (float(vp) > 0 or float(hp) > 0) and str(r.get("GAME_STATUS_TEXT","")):
                # if game finished or in progress, show score too
                return f"{int(vp)}–{int(hp)} ({r.get('GAME_STATUS_TEXT','')})"
            return str(r.get("GAME_STATUS_TEXT", "") or "")

        merged["STATUS"] = merged.apply(_status_row, axis=1)

        out = merged[["GAME_ID", "MATCHUP", "STATUS"]].copy()
        out["DATE"] = date_yyyy_mm_dd
        out = out.sort_values("MATCHUP").reset_index(drop=True)
        return out

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


# ============================================================
# Standings + Team summary tables
# ============================================================
def get_league_standings(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Uses LeagueStandings endpoint. Returns one row per team with W/L/PCT and conference.
    """
    spec = CacheSpec(key=f"standings_{season}", ttl_seconds=TTL_STANDINGS)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(leaguestandings.LeagueStandings, {"season": season, "season_type": "Regular Season"})
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df

        # Keep only NBA teams
        if "TeamID" in df.columns:
            df = df[df["TeamID"].astype(str).str.startswith("161061")].copy()

        # Normalize columns
        colmap = {
            "TeamID": "TEAM_ID",
            "TeamCity": "TEAM_CITY",
            "TeamName": "TEAM_NICK",
            "TeamSlug": "TEAM_SLUG",
            "Conference": "CONF",
            "ConferenceRank": "CONF_RANK",
            "WINS": "W",
            "LOSSES": "L",
            "WinPCT": "W_PCT",
        }
        for k, v in colmap.items():
            if k in df.columns:
                df[v] = df[k]

        # Add official name/abbr from static
        t = _teams_df()
        df = df.merge(t, left_on="TEAM_ID", right_on="TEAM_ID", how="left")

        keep = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "CONF", "W", "L", "W_PCT"]
        for c in keep:
            if c not in df.columns:
                df[c] = np.nan

        df["W"] = pd.to_numeric(df["W"], errors="coerce")
        df["L"] = pd.to_numeric(df["L"], errors="coerce")
        df["W_PCT"] = pd.to_numeric(df["W_PCT"], errors="coerce")
        return df[keep].drop_duplicates(subset=["TEAM_ID"]).reset_index(drop=True)

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


def get_league_team_stats(season: str, measure: str, per_mode: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    LeagueDashTeamStats for Base/Advanced/Opponent.
    measure in {"Base","Advanced","Opponent"}
    per_mode in {"Totals","PerGame"}
    """
    key = f"teamstats_{season}_{measure}_{per_mode}"
    spec = CacheSpec(key=key, ttl_seconds=TTL_LEAGUE_TABLES)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                measure_type_detailed_defense=measure,
                per_mode_detailed=per_mode,
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df

        # NBA teams only
        if "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()

        return df.reset_index(drop=True)

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


def get_league_team_summary(season: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - team_table: standings merged with base per-game + advanced per-game
      - opp_table: opponent per-game (OPP_*) table
    """
    # Standings
    standings = get_league_standings(season, force_refresh=force_refresh)
    base_pg = get_league_team_stats(season, measure="Base", per_mode="PerGame", force_refresh=force_refresh)
    adv_pg = get_league_team_stats(season, measure="Advanced", per_mode="PerGame", force_refresh=force_refresh)
    opp_pg = get_league_team_stats(season, measure="Opponent", per_mode="PerGame", force_refresh=force_refresh)

    # Reduce columns safely (avoid KeyError)
    def _ensure(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    base_keep = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST"]
    adv_keep = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]

    base_min = _ensure(base_pg, base_keep)[base_keep].copy()
    adv_min = _ensure(adv_pg, adv_keep)[adv_keep].copy()

    # Coerce merge key types (fixes ValueError merge dtype mismatch)
    for d in (standings, base_min, adv_min):
        if "TEAM_ID" in d.columns:
            d["TEAM_ID"] = pd.to_numeric(d["TEAM_ID"], errors="coerce").astype("Int64")

    merged = standings.merge(base_min, on=["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"], how="left")
    merged = merged.merge(adv_min, on=["TEAM_ID"], how="left")

    # Opponent table
    if opp_pg is None or opp_pg.empty:
        opp_table = pd.DataFrame()
    else:
        opp_cols = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"] + [c for c in opp_pg.columns if c.startswith("OPP_")]
        opp_table = _ensure(opp_pg, opp_cols)[opp_cols].copy()

        if "TEAM_ID" in opp_table.columns:
            opp_table["TEAM_ID"] = pd.to_numeric(opp_table["TEAM_ID"], errors="coerce").astype("Int64")

        # Attach W/L/PCT for context
        opp_table = opp_table.merge(
            standings[["TEAM_ID", "W", "L", "W_PCT"]],
            on="TEAM_ID",
            how="left",
        )

    return merged.reset_index(drop=True), opp_table.reset_index(drop=True)


# ============================================================
# League-wide player tables
# ============================================================
def get_league_players(
    season: str,
    per_mode: str = "PerGame",
    last_n_games: int = 0,
    min_gp: int = 0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Cached LeagueDashPlayerStats.
    per_mode: "PerGame" or "Per36"
    last_n_games: 0 means season-to-date
    min_gp: filter out small sample players
    """
    key = f"players_{season}_{per_mode}_L{last_n_games}_minGP{min_gp}"
    spec = CacheSpec(key=key, ttl_seconds=TTL_LEAGUE_TABLES)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed=per_mode,
                last_n_games=last_n_games,
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df

        # NBA only
        if "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()

        # Filter min GP
        if min_gp and "GP" in df.columns:
            df["GP"] = pd.to_numeric(df["GP"], errors="coerce")
            df = df[df["GP"] >= min_gp].copy()

        return df.reset_index(drop=True)

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


# ============================================================
# Team gamelog + boxscore
# ============================================================
def get_team_gamelog(team_id: int, season: str, force_refresh: bool = False) -> pd.DataFrame:
    spec = CacheSpec(key=f"team_gamelog_{season}_{team_id}", ttl_seconds=TTL_TEAM_GAMELOG)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(teamgamelog.TeamGameLog, {"team_id": team_id, "season": season, "season_type_all_star": "Regular Season"})
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


def get_game_boxscore_traditional(game_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns player-level traditional boxscore for a game.
    """
    spec = CacheSpec(key=f"boxscore_trad_{game_id}", ttl_seconds=TTL_BOXSCORE)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(boxscoretraditionalv2.BoxScoreTraditionalV2, {"game_id": game_id})
        # frames[0] is usually player stats
        df = frames[0] if frames else pd.DataFrame()
        return df.reset_index(drop=True) if df is not None else pd.DataFrame()

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)
