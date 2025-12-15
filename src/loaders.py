from __future__ import annotations

import datetime as dt
from typing import Tuple

import pandas as pd

from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguestandingsv3,
    leaguedashteamstats,
    leaguedashplayerstats,
    teamgamelog,
    playergamelog,
    boxscoretraditionalv3,
)
from nba_api.stats.static import teams as teams_static

from .utils import (
    safe_to_datetime,
    minutes_str_to_float,
    ensure_columns,
    normalize_team_id,
    normalize_player_id,
    add_basic_shooting_breakouts,
)


def _to_df(endpoint_obj, idx: int = 0) -> pd.DataFrame:
    try:
        dfs = endpoint_obj.get_data_frames()
        if not dfs or len(dfs) <= idx:
            return pd.DataFrame()
        return dfs[idx].copy()
    except Exception:
        return pd.DataFrame()


def fetch_daily_scoreboard(game_date: dt.date) -> pd.DataFrame:
    """
    One row per matchup: matchup, start time, final score if completed.
    """
    ds = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y"), timeout=60)
    games = _to_df(ds, 0)
    # scoreboardv2 can return multiple tables; GameHeader is usually index 0.

    # Normalize expected columns
    games = ensure_columns(
        games,
        [
            "GAME_ID",
            "GAME_DATE_EST",
            "GAME_STATUS_TEXT",
            "HOME_TEAM_ID",
            "VISITOR_TEAM_ID",
            "HOME_TEAM_WINS_LOSSES",
            "VISITOR_TEAM_WINS_LOSSES",
            "HOME_TEAM_SCORE",
            "VISITOR_TEAM_SCORE",
        ],
    )

    # Team metadata map
    tdf = pd.DataFrame(teams_static.get_teams())
    tdf = tdf.rename(columns={"id": "TEAM_ID", "abbreviation": "TEAM_ABBREVIATION", "full_name": "TEAM_NAME"})
    tdf["TEAM_ID"] = pd.to_numeric(tdf["TEAM_ID"], errors="coerce").astype("Int64")

    games["HOME_TEAM_ID"] = pd.to_numeric(games["HOME_TEAM_ID"], errors="coerce").astype("Int64")
    games["VISITOR_TEAM_ID"] = pd.to_numeric(games["VISITOR_TEAM_ID"], errors="coerce").astype("Int64")

    games = games.merge(
        tdf[["TEAM_ID", "TEAM_ABBREVIATION"]].rename(columns={"TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME"}),
        on="HOME_TEAM_ID",
        how="left",
    ).merge(
        tdf[["TEAM_ID", "TEAM_ABBREVIATION"]].rename(columns={"TEAM_ID": "VISITOR_TEAM_ID", "TEAM_ABBREVIATION": "AWAY"}),
        on="VISITOR_TEAM_ID",
        how="left",
    )

    # Parse date
    games["GAME_DATE_EST"] = safe_to_datetime(games["GAME_DATE_EST"])

    # Build display columns
    games["MATCHUP"] = games["AWAY"].astype("string").fillna("?") + " @ " + games["HOME"].astype("string").fillna("?")

    # Start time / status
    games["STATUS"] = games["GAME_STATUS_TEXT"].astype("string")

    # Score display (only if numeric)
    games["HOME_TEAM_SCORE"] = pd.to_numeric(games["HOME_TEAM_SCORE"], errors="coerce")
    games["VISITOR_TEAM_SCORE"] = pd.to_numeric(games["VISITOR_TEAM_SCORE"], errors="coerce")

    def _score_row(r):
        if pd.notna(r["HOME_TEAM_SCORE"]) and pd.notna(r["VISITOR_TEAM_SCORE"]) and (r["HOME_TEAM_SCORE"] > 0 or r["VISITOR_TEAM_SCORE"] > 0):
            return f'{int(r["VISITOR_TEAM_SCORE"])} - {int(r["HOME_TEAM_SCORE"])}'
        return ""

    games["FINAL"] = games.apply(_score_row, axis=1)

    out = games[["GAME_ID", "MATCHUP", "STATUS", "FINAL"]].copy()
    out = out.sort_values(["MATCHUP"], ascending=True).reset_index(drop=True)
    return out


def fetch_league_team_summary(season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - team_table: standings + per-game + ratings
      - opp_table: opponent allowed metrics (from opponent splits if available; here derived from leaguedashteamstats where possible)

    We merge on TEAM_ID only (stable), avoiding TEAM_ABBREVIATION dtype issues.
    """
    # Standings
    st = leaguestandingsv3.LeagueStandingsV3(season=season, timeout=60)
    standings = _to_df(st, 0)
    standings = ensure_columns(standings, ["TeamID", "TeamName", "TeamCity", "WINS", "LOSSES", "WinPCT", "Conference"])
    standings = standings.rename(columns={"TeamID": "TEAM_ID", "TeamName": "TEAM_NAME"})
    standings = normalize_team_id(standings)

    # Team stats per game + ratings
    td = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        timeout=60,
    )
    base = _to_df(td, 0)
    base = ensure_columns(base, ["TEAM_ID", "TEAM_NAME"])
    base = normalize_team_id(base)

    # Advanced table (ratings, net rating)
    td_adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=60,
    )
    adv = _to_df(td_adv, 0)
    adv = ensure_columns(adv, ["TEAM_ID"])
    adv = normalize_team_id(adv)

    # Merge on TEAM_ID only
    merged = standings.merge(base, on="TEAM_ID", how="left", suffixes=("", "_BASE"))
    merged = merged.merge(adv, on="TEAM_ID", how="left", suffixes=("", "_ADV"))

    # Clean/format columns
    merged["WINS"] = pd.to_numeric(merged["WINS"], errors="coerce").fillna(0).astype(int)
    merged["LOSSES"] = pd.to_numeric(merged["LOSSES"], errors="coerce").fillna(0).astype(int)
    merged["WinPCT"] = pd.to_numeric(merged["WinPCT"], errors="coerce")

    # Build final team table columns (keep it wide but sortable)
    cols_keep = []
    for c in [
        "TEAM_NAME",
        "Conference",
        "WINS",
        "LOSSES",
        "WinPCT",
        "PTS",
        "REB",
        "AST",
        "NET_RATING",
        "OFF_RATING",
        "DEF_RATING",
    ]:
        if c in merged.columns:
            cols_keep.append(c)

    team_table = merged[cols_keep].copy()
    team_table = team_table.sort_values("WinPCT", ascending=False).reset_index(drop=True)

    # Opponent table: leaguedashteamstats has opponent columns in some seasons/measures, but not always.
    # We attempt to read common "OPP_*" columns if present; otherwise create a blank but stable table.
    opp_cols = [c for c in merged.columns if c.startswith("OPP_")]
    if opp_cols:
        opp_table = merged[["TEAM_NAME", "Conference"] + opp_cols].copy()
    else:
        opp_table = merged[["TEAM_NAME", "Conference"]].copy()
        # Provide placeholders so UI remains stable
        for c in ["OPP_PTS", "OPP_REB", "OPP_AST"]:
            opp_table[c] = pd.NA

    return team_table, opp_table


def fetch_league_player_stats(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    Leaguewide player table (used for leaders).
    """
    pl = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=per_mode,  # "PerGame" or "Per36"
        measure_type_detailed_defense="Base",
        timeout=60,
    )
    df = _to_df(pl, 0)
    df = ensure_columns(df, ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST"])
    df = normalize_player_id(df)
    df["GP"] = pd.to_numeric(df["GP"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_team_gamelog(season: str, team_id: int) -> pd.DataFrame:
    gl = teamgamelog.TeamGameLog(season=season, team_id=team_id, timeout=60)
    df = _to_df(gl, 0)
    df = ensure_columns(df, ["Game_ID", "GAME_DATE", "MATCHUP", "WL"])
    df["GAME_DATE"] = safe_to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def fetch_player_gamelog(season: str, player_id: int) -> pd.DataFrame:
    gl = playergamelog.PlayerGameLog(season=season, player_id=player_id, timeout=60)
    df = _to_df(gl, 0)
    df = ensure_columns(df, ["Game_ID", "GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "FGM", "FG3M", "OREB", "DREB"])
    df["GAME_DATE"] = safe_to_datetime(df["GAME_DATE"])
    df["MIN"] = df["MIN"].apply(minutes_str_to_float)
    df = add_basic_shooting_breakouts(df)
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def fetch_boxscore_traditional(game_id: str) -> pd.DataFrame:
    bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=60)
    # First table is typically player stats
    df = _to_df(bs, 0)
    df = ensure_columns(df, ["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "FGM", "FG3M", "OREB", "DREB"])
    df["MIN"] = df["MIN"].apply(minutes_str_to_float)
    df = add_basic_shooting_breakouts(df)
    return df
