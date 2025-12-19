# src/loaders.py
import os
import datetime as dt
from typing import Tuple

import pandas as pd
import duckdb

from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguedashteamstats,
    leaguedashplayerstats,
    leaguestandings,
    playergamelog,
    teamgamelog,
    boxscoretraditionalv2,
    commonplayerinfo,
)

# ----------------------------
# DuckDB Path (your Colab export)
# ----------------------------
DEFAULT_DUCKDB_PATH = os.environ.get("NBA_DUCKDB_PATH", "duckdb/nba_warehouse.duckdb")


def _con(db_path: str = DEFAULT_DUCKDB_PATH):
    return duckdb.connect(db_path, read_only=True)


# ----------------------------
# Static Teams
# ----------------------------
def get_teams_static() -> pd.DataFrame:
    """
    Prefer DuckDB raw.dim_team if available; else fallback nba_api static.
    """
    db_path = DEFAULT_DUCKDB_PATH
    if os.path.exists(db_path):
        try:
            con = _con(db_path)
            df = con.execute("SELECT * FROM raw.dim_team ORDER BY TEAM_ABBREVIATION").df()
            con.close()
            if df is not None and not df.empty:
                # Align expected columns used by app
                # TEAM_NAME already exists in your ETL
                return df
        except Exception:
            pass

    # fallback
    rows = nba_teams_static.get_teams()
    df = pd.DataFrame(rows).rename(
        columns={
            "id": "TEAM_ID",
            "abbreviation": "TEAM_ABBREVIATION",
            "full_name": "TEAM_NAME",
            "city": "TEAM_CITY",
            "state": "TEAM_STATE",
            "nickname": "TEAM_NICKNAME",
            "year_founded": "YEAR_FOUNDED",
        }
    )
    keep = ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "TEAM_CITY", "TEAM_STATE", "TEAM_NICKNAME", "YEAR_FOUNDED"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    return df[keep].sort_values("TEAM_ABBREVIATION").reset_index(drop=True)


# ----------------------------
# Scoreboard / Schedule
# ----------------------------
def fetch_daily_scoreboard(game_date: dt.date) -> pd.DataFrame:
    """
    Prefer DuckDB raw.fact_schedule_upcoming_raw if present; else call nba_api scoreboardv2.
    Returns a standardized schedule table.
    """
    db_path = DEFAULT_DUCKDB_PATH
    if os.path.exists(db_path):
        try:
            con = _con(db_path)
            # If your ETL stored upcoming schedule, use it.
            # Columns may vary, so we select safe subset and standardize.
            df = con.execute(
                """
                SELECT *
                FROM raw.fact_schedule_upcoming_raw
                WHERE CAST(GAME_DATE AS DATE) = ?
                """,
                [game_date],
            ).df()
            con.close()

            if df is not None and not df.empty:
                # Try to standardize from likely columns
                # expected: GAME_DATE, GAME_ID, HOME_TEAM_ABBREVIATION, AWAY_TEAM_ABBREVIATION, STATUS
                colmap = {c.upper(): c for c in df.columns}
                def _get(name):
                    return df[colmap[name]] if name in colmap else None

                home = _get("HOME_TEAM_ABBREVIATION")
                away = _get("AWAY_TEAM_ABBREVIATION")
                status = _get("STATUS") or _get("GAME_STATUS_TEXT")

                out = pd.DataFrame()
                out["DATE"] = pd.to_datetime(game_date)
                if away is not None and home is not None:
                    out["MATCHUP"] = away.astype(str) + " @ " + home.astype(str)
                else:
                    # fallback if schedule table uses MATCHUP already
                    if "MATCHUP" in df.columns:
                        out["MATCHUP"] = df["MATCHUP"].astype(str)
                    else:
                        out["MATCHUP"] = None
                out["STATUS"] = status.astype(str) if status is not None else ""
                # optional columns
                if "GAME_ID" in df.columns:
                    out["GAME_ID"] = df["GAME_ID"].astype(str)
                return out[["DATE", "MATCHUP", "STATUS"] + (["GAME_ID"] if "GAME_ID" in out.columns else [])]
        except Exception:
            pass

    # nba_api fallback
    sb = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y")).get_data_frames()
    games = sb[0] if sb else pd.DataFrame()
    if games is None or games.empty:
        return pd.DataFrame(columns=["DATE", "MATCHUP", "STATUS"])

    # standardize scoreboardv2 structure
    out = pd.DataFrame()
    out["DATE"] = pd.to_datetime(game_date)
    # These columns exist in ScoreboardV2 game header
    away = games.get("VISITOR_TEAM_ABBREVIATION")
    home = games.get("HOME_TEAM_ABBREVIATION")
    out["MATCHUP"] = away.astype(str) + " @ " + home.astype(str)
    out["STATUS"] = games.get("GAME_STATUS_TEXT", "").astype(str)
    out["GAME_ID"] = games.get("GAME_ID", "").astype(str)
    return out[["DATE", "MATCHUP", "STATUS", "GAME_ID"]]


# ----------------------------
# League Index / Logs
# ----------------------------
def fetch_league_players_index(season: str) -> pd.DataFrame:
    """
    Provides PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION for the selected season.
    Prefer DuckDB raw.fact_player_game_log_raw (fast, stable).
    """
    db_path = DEFAULT_DUCKDB_PATH
    if os.path.exists(db_path):
        con = _con(db_path)
        df = con.execute(
            """
            SELECT
              CAST(PLAYER_ID AS BIGINT) AS PLAYER_ID,
              ANY_VALUE(PLAYER_NAME) AS PLAYER_NAME,
              CAST(TEAM_ID AS BIGINT) AS TEAM_ID,
              ANY_VALUE(TEAM_ABBREVIATION) AS TEAM_ABBREVIATION
            FROM raw.fact_player_game_log_raw
            WHERE SEASON = ?
            GROUP BY PLAYER_ID, TEAM_ID
            """,
            [season],
        ).df()
        con.close()

        if df is None or df.empty:
            return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"])

        # Collapse to one row per player (latest team in season can change; keep most frequent team)
        # Simple approach: take first occurrence after sorting by name
        df = df.sort_values(["PLAYER_NAME", "PLAYER_ID"]).drop_duplicates(["PLAYER_ID"], keep="first").reset_index(drop=True)
        return df

    # Fallback: use playergamelog league-wide is not supported directly; return empty
    return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"])


def fetch_player_logs(player_id: int, season: str) -> pd.DataFrame:
    db_path = DEFAULT_DUCKDB_PATH
    if os.path.exists(db_path):
        con = _con(db_path)
        df = con.execute(
            """
            SELECT *
            FROM raw.fact_player_game_log_raw
            WHERE CAST(PLAYER_ID AS BIGINT) = ? AND SEASON = ?
            ORDER BY CAST(GAME_DATE AS DATE) DESC, CAST(GAME_ID AS VARCHAR) DESC
            """,
            [int(player_id), season],
        ).df()
        con.close()
        if df is None:
            df = pd.DataFrame()
        return df

    # fallback nba_api
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    return df


def fetch_common_player_info(player_id: int) -> pd.DataFrame:
    # nba_api provides richer metadata than DuckDB warehouse
    try:
        return commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


def fetch_team_gamelog(team_id: int, season: str) -> pd.DataFrame:
    db_path = DEFAULT_DUCKDB_PATH
    if os.path.exists(db_path):
        con = _con(db_path)
        df = con.execute(
            """
            SELECT *
            FROM raw.fact_team_game_log_raw
            WHERE CAST(TEAM_ID AS BIGINT) = ? AND SEASON = ?
            ORDER BY CAST(GAME_DATE AS DATE) DESC, CAST(GAME_ID AS VARCHAR) DESC
            """,
            [int(team_id), season],
        ).df()
        con.close()
        if df is None:
            df = pd.DataFrame()
        return df

    # fallback nba_api
    df = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    return df


def fetch_game_team_boxscore(game_id: str, team_id: int) -> pd.DataFrame:
    """
    Traditional box score returns both teams; filter to requested TEAM_ID.
    """
    try:
        frames = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()
        players = frames[0] if frames else pd.DataFrame()
        if players is None or players.empty:
            return pd.DataFrame()
        players["TEAM_ID"] = pd.to_numeric(players.get("TEAM_ID"), errors="coerce")
        return players[players["TEAM_ID"] == int(team_id)].copy()
    except Exception:
        return pd.DataFrame()


# ----------------------------
# League Tables (nba_api)
# ----------------------------
def fetch_league_standings(season: str) -> pd.DataFrame:
    """
    Returns TEAM_ID, TEAM, CONF, W, L, W%
    """
    try:
        df = leaguestandings.LeagueStandings(season=season).get_data_frames()[0]
        if df is None or df.empty:
            return pd.DataFrame()

        # Standardize from nba_api standings columns
        # Common columns: TeamID, TeamName, Conference, WINS, LOSSES, WinPCT
        col_map = {}
        for c in df.columns:
            cu = c.upper()
            if cu in ("TEAMID", "TEAM_ID"):
                col_map[c] = "TEAM_ID"
            elif cu in ("TEAMNAME", "TEAM", "TEAM_NAME"):
                col_map[c] = "TEAM"
            elif cu in ("CONFERENCE", "CONF"):
                col_map[c] = "CONF"
            elif cu in ("WINS", "W"):
                col_map[c] = "W"
            elif cu in ("LOSSES", "L"):
                col_map[c] = "L"
            elif cu in ("WINPCT", "W_PCT", "W%"):
                col_map[c] = "W%"

        out = df.rename(columns=col_map)
        need = ["TEAM_ID", "TEAM", "CONF", "W", "L", "W%"]
        for c in need:
            if c not in out.columns:
                out[c] = None
        return out[need].copy()
    except Exception:
        return pd.DataFrame()


def fetch_league_team_stats(season: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (base, adv, opp) like your app expects.
    """
    try:
        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Base"
        ).get_data_frames()[0]
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Opponent"
        ).get_data_frames()[0]
        return base, adv, opp
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def fetch_league_players_stats(season: str, per_mode: str = "PerGame", last_n_games: int = 0) -> pd.DataFrame:
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            last_n_games=int(last_n_games),
            measure_type_detailed_defense="Base",
        ).get_data_frames()[0]
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_career_summary(player_id: int) -> pd.DataFrame:
    """
    Keep as stub for now; your previous version expects it.
    If you want, we can wire this to playercareerstats later.
    """
    return pd.DataFrame()
