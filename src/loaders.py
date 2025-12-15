from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import pandas as pd

from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    playergamelog,
    commonplayerinfo,
    leaguestandingsv3,
    teamgamelog,
    boxscoretraditionalv2,
    playercareerstats,
)
from nba_api.stats.static import teams as static_teams


def _to_df(frames: List[pd.DataFrame], idx: int = 0) -> pd.DataFrame:
    if not frames or len(frames) <= idx:
        return pd.DataFrame()
    df = frames[idx]
    return df.copy() if df is not None else pd.DataFrame()


def get_teams_static_df() -> pd.DataFrame:
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = pd.to_numeric(t["TEAM_ID"], errors="coerce").astype("Int64")
    return t[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]].dropna().reset_index(drop=True)


def fetch_daily_scoreboard(game_date: dt.date, timeout: int = 60) -> pd.DataFrame:
    """
    Returns: DATE, GAME_ID, MATCHUP, STATUS_TEXT, STATUS, AWAY_PTS, HOME_PTS
    """
    date_str = game_date.strftime("%m/%d/%Y")
    obj = scoreboardv2.ScoreboardV2(game_date=date_str, league_id="00", timeout=timeout)
    frames = obj.get_data_frames()
    if not frames or len(frames) < 2:
        return pd.DataFrame()

    game_header = _to_df(frames, 0)
    line_score = _to_df(frames, 1)
    if game_header.empty:
        return pd.DataFrame()

    need_gh = ["GAME_ID", "GAME_STATUS_ID", "GAME_STATUS_TEXT", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    for c in need_gh:
        if c not in game_header.columns:
            game_header[c] = pd.NA

    for c in ["GAME_ID", "TEAM_ID", "PTS"]:
        if c not in line_score.columns:
            line_score[c] = pd.NA

    tstatic = get_teams_static_df()
    id2abbr = dict(zip(tstatic["TEAM_ID"].astype(int), tstatic["TEAM_ABBREVIATION"].astype(str)))

    line_score["TEAM_ID"] = pd.to_numeric(line_score["TEAM_ID"], errors="coerce")
    line_score["PTS"] = pd.to_numeric(line_score["PTS"], errors="coerce")
    pts_map = (
        line_score.dropna(subset=["GAME_ID", "TEAM_ID"])
        .assign(TEAM_ID=lambda d: d["TEAM_ID"].astype(int))
        .set_index(["GAME_ID", "TEAM_ID"])["PTS"]
        .to_dict()
    )

    out_rows = []
    for r in game_header.itertuples(index=False):
        gid = str(getattr(r, "GAME_ID"))
        home_id = pd.to_numeric(getattr(r, "HOME_TEAM_ID", pd.NA), errors="coerce")
        away_id = pd.to_numeric(getattr(r, "VISITOR_TEAM_ID", pd.NA), errors="coerce")
        status_id = pd.to_numeric(getattr(r, "GAME_STATUS_ID", pd.NA), errors="coerce")
        status_text = str(getattr(r, "GAME_STATUS_TEXT", "")).strip()

        if pd.isna(home_id) or pd.isna(away_id):
            continue
        home_id = int(home_id)
        away_id = int(away_id)

        home_abbr = id2abbr.get(home_id, str(home_id))
        away_abbr = id2abbr.get(away_id, str(away_id))
        matchup = f"{away_abbr} @ {home_abbr}"

        home_pts = pts_map.get((gid, home_id), pd.NA)
        away_pts = pts_map.get((gid, away_id), pd.NA)

        status_display = status_text if status_text else "—"
        if pd.notna(status_id) and int(status_id) == 3 and pd.notna(home_pts) and pd.notna(away_pts):
            status_display = f"Final: {int(away_pts)}–{int(home_pts)}"
        elif isinstance(status_text, str) and status_text.lower().startswith("final") and pd.notna(home_pts) and pd.notna(away_pts):
            status_display = f"{status_text}: {int(away_pts)}–{int(home_pts)}"

        out_rows.append(
            {
                "DATE": game_date,
                "GAME_ID": gid,
                "MATCHUP": matchup,
                "STATUS_TEXT": status_text,
                "STATUS": status_display,
                "AWAY_PTS": away_pts,
                "HOME_PTS": home_pts,
            }
        )

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows).sort_values(["DATE", "MATCHUP"]).reset_index(drop=True)
    return out


def fetch_league_standings(season: str, timeout: int = 60) -> pd.DataFrame:
    obj = leaguestandingsv3.LeagueStandingsV3(season=season, season_type="Regular Season", timeout=timeout)
    df = _to_df(obj.get_data_frames(), 0)
    if df.empty:
        return df

    wanted = ["TeamID", "TeamName", "TeamCity", "Conference", "WINS", "LOSSES", "WinPCT"]
    for c in wanted:
        if c not in df.columns:
            df[c] = pd.NA

    out = pd.DataFrame(
        {
            "TEAM_ID": pd.to_numeric(df["TeamID"], errors="coerce").astype("Int64"),
            "TEAM": (df["TeamCity"].astype("string").fillna("") + " " + df["TeamName"].astype("string").fillna("")).str.strip(),
            "CONF": df["Conference"].astype("string"),
            "W": pd.to_numeric(df["WINS"], errors="coerce"),
            "L": pd.to_numeric(df["LOSSES"], errors="coerce"),
            "W%": pd.to_numeric(df["WinPCT"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["TEAM_ID"]).drop_duplicates(subset=["TEAM_ID"]).reset_index(drop=True)
    return out


def fetch_league_team_stats(season: str, timeout: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _get(measure: str) -> pd.DataFrame:
        obj = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable="00",
            measure_type_detailed_defense=measure,
            per_mode_detailed="PerGame",
            timeout=timeout,
        )
        df = _to_df(obj.get_data_frames(), 0)
        if df.empty:
            return df
        if "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    base = _get("Base")
    adv = _get("Advanced")
    opp = _get("Opponent")
    return base, adv, opp


def fetch_league_players_stats(season: str, per_mode: str, last_n_games: int, timeout: int = 60) -> pd.DataFrame:
    obj = LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        per_mode_detailed=per_mode,
        last_n_games=int(last_n_games),
        timeout=timeout,
    )
    df = _to_df(obj.get_data_frames(), 0)
    if df.empty:
        return df
    if "TEAM_ID" in df.columns:
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)


def fetch_league_players_index(season: str, timeout: int = 60) -> pd.DataFrame:
    df = fetch_league_players_stats(season, per_mode="PerGame", last_n_games=0, timeout=timeout)
    if df.empty:
        return df
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GP", "MIN"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[keep].copy()
    out["PLAYER_ID"] = pd.to_numeric(out["PLAYER_ID"], errors="coerce").astype("Int64")
    out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["PLAYER_ID"]).drop_duplicates(subset=["PLAYER_ID"])
    return out.sort_values(["TEAM_NAME", "PLAYER_NAME"]).reset_index(drop=True)


def fetch_player_logs(player_id: int, season: str, timeout: int = 60) -> pd.DataFrame:
    obj = playergamelog.PlayerGameLog(
        player_id=int(player_id),
        season=season,
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        timeout=timeout,
    )
    df = _to_df(obj.get_data_frames(), 0)
    if df.empty:
        return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def fetch_common_player_info(player_id: int, timeout: int = 60) -> pd.DataFrame:
    obj = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id), timeout=timeout)
    return _to_df(obj.get_data_frames(), 0)


def fetch_team_gamelog(team_id: int, season: str, timeout: int = 60) -> pd.DataFrame:
    obj = teamgamelog.TeamGameLog(
        team_id=int(team_id),
        season=season,
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        timeout=timeout,
    )
    df = _to_df(obj.get_data_frames(), 0)
    if df.empty:
        return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def fetch_game_team_boxscore(game_id: str, team_id: int, timeout: int = 60) -> pd.DataFrame:
    obj = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=str(game_id), timeout=timeout)
    df = _to_df(obj.get_data_frames(), 0)
    if df.empty:
        return df
    if "TEAM_ID" not in df.columns:
        return pd.DataFrame()
    df = df[df["TEAM_ID"] == int(team_id)].copy()
    return df.reset_index(drop=True)


def fetch_career_totals(player_id: int, timeout: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obj = playercareerstats.PlayerCareerStats(player_id=int(player_id), timeout=timeout)
    frames = obj.get_data_frames()
    season_totals = _to_df(frames, 0)
    career_totals = _to_df(frames, 1)
    return season_totals, career_totals
