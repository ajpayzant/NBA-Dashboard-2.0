from __future__ import annotations

import random
import time
import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as static_teams
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

# ----------------------- API Resilience -----------------------
REQUEST_TIMEOUT = int(60)
MAX_RETRIES = int(6)
BASE_SLEEP = float(1.25)
MIN_SECONDS_BETWEEN_CALLS = float(1.0)

_LAST_CALL_TS = 0.0


def _rate_limit_pause(min_gap: float = MIN_SECONDS_BETWEEN_CALLS) -> None:
    global _LAST_CALL_TS
    now = time.time()
    gap = now - _LAST_CALL_TS
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _LAST_CALL_TS = time.time()


def _retry_api(endpoint_cls, kwargs: dict, timeout: int = REQUEST_TIMEOUT) -> List[pd.DataFrame]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _rate_limit_pause()
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            sleep_s = BASE_SLEEP * (2 ** (attempt - 1))
            sleep_s = min(40.0, sleep_s)
            sleep_s = sleep_s * (0.85 + 0.30 * random.random())
            time.sleep(sleep_s)
    raise last_err


def _to_df(frames: List[pd.DataFrame], idx: int) -> pd.DataFrame:
    if not frames or idx >= len(frames) or frames[idx] is None:
        return pd.DataFrame()
    df = frames[idx]
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


# ----------------------- Static Teams -----------------------
def get_teams_static() -> pd.DataFrame:
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = pd.to_numeric(t["TEAM_ID"], errors="coerce").astype("Int64")
    return t[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]].dropna().reset_index(drop=True)


# ----------------------- League / Players Index -----------------------
def fetch_league_players_index(season: str) -> pd.DataFrame:
    try:
        frames = _retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed="PerGame",
                last_n_games=0,
            ),
        )
        df = _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GP", "MIN"]
    df = _ensure_cols(df, keep)[keep].copy()
    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["PLAYER_ID"]).drop_duplicates(subset=["PLAYER_ID"])
    return df.sort_values(["TEAM_NAME", "PLAYER_NAME"]).reset_index(drop=True)


def fetch_league_standings(season: str) -> pd.DataFrame:
    try:
        frames = _retry_api(
            leaguestandingsv3.LeagueStandingsV3,
            dict(season=season, season_type="Regular Season"),
        )
        df = _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    wanted = ["TeamID", "TeamName", "TeamCity", "TeamAbbreviation", "Conference", "WINS", "LOSSES", "WinPCT"]
    df = _ensure_cols(df, wanted)

    out = pd.DataFrame(
        {
            "TEAM_ID": pd.to_numeric(df["TeamID"], errors="coerce"),
            "TEAM": (df["TeamCity"].astype("string").fillna("") + " " + df["TeamName"].astype("string").fillna("")).str.strip(),
            "CONF": df["Conference"].astype("string"),
            "W": pd.to_numeric(df["WINS"], errors="coerce"),
            "L": pd.to_numeric(df["LOSSES"], errors="coerce"),
            "W%": pd.to_numeric(df["WinPCT"], errors="coerce"),
        }
    )
    out["TEAM_ID"] = out["TEAM_ID"].astype("Int64")
    out = out.dropna(subset=["TEAM_ID"]).drop_duplicates(subset=["TEAM_ID"]).reset_index(drop=True)
    return out


def fetch_league_team_stats(season: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _get(measure: str, per_mode: str) -> pd.DataFrame:
        try:
            frames = _retry_api(
                leaguedashteamstats.LeagueDashTeamStats,
                dict(
                    season=season,
                    season_type_all_star="Regular Season",
                    league_id_nullable="00",
                    measure_type_detailed_defense=measure,
                    per_mode_detailed=per_mode,
                ),
            )
            df = _to_df(frames, 0)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        if "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    base = _get("Base", "PerGame")
    adv = _get("Advanced", "PerGame")
    opp = _get("Opponent", "PerGame")
    return base, adv, opp


def fetch_league_players_stats(season: str, per_mode: str, last_n_games: int) -> pd.DataFrame:
    try:
        frames = _retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed=per_mode,
                last_n_games=int(last_n_games),
            ),
        )
        df = _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "TEAM_ID" in df.columns:
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)


# ----------------------- Scoreboard / Schedule -----------------------
def fetch_daily_scoreboard(game_date: dt.date) -> pd.DataFrame:
    date_str = game_date.strftime("%m/%d/%Y")
    try:
        frames = _retry_api(scoreboardv2.ScoreboardV2, {"game_date": date_str, "league_id": "00"})
    except Exception:
        return pd.DataFrame()

    if not frames or len(frames) < 2:
        return pd.DataFrame()

    game_header = _to_df(frames, 0)
    line_score = _to_df(frames, 1)

    if game_header.empty:
        return pd.DataFrame()

    game_header = _ensure_cols(
        game_header,
        ["GAME_ID", "GAME_STATUS_ID", "GAME_STATUS_TEXT", "HOME_TEAM_ID", "VISITOR_TEAM_ID"],
    )
    line_score = _ensure_cols(line_score, ["GAME_ID", "TEAM_ID", "PTS"])

    tstatic = get_teams_static()
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
        home_id = pd.to_numeric(getattr(r, "HOME_TEAM_ID", np.nan), errors="coerce")
        away_id = pd.to_numeric(getattr(r, "VISITOR_TEAM_ID", np.nan), errors="coerce")
        status_id = pd.to_numeric(getattr(r, "GAME_STATUS_ID", np.nan), errors="coerce")
        status_text = str(getattr(r, "GAME_STATUS_TEXT", "")).strip()

        if pd.isna(home_id) or pd.isna(away_id):
            continue
        home_id = int(home_id)
        away_id = int(away_id)

        home_abbr = id2abbr.get(home_id, str(home_id))
        away_abbr = id2abbr.get(away_id, str(away_id))
        matchup = f"{away_abbr} @ {home_abbr}"

        home_pts = pts_map.get((gid, home_id), np.nan)
        away_pts = pts_map.get((gid, away_id), np.nan)

        status_display = status_text if status_text else "—"
        if pd.notna(status_id) and int(status_id) == 3 and pd.notna(home_pts) and pd.notna(away_pts):
            status_display = f"Final: {int(away_pts)}–{int(home_pts)}"

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

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["DATE", "MATCHUP"]).reset_index(drop=True)
    return out


# ----------------------- Team Game Log (FIXED) -----------------------
def fetch_team_gamelog(team_id: int, season: str) -> pd.DataFrame:
    """
    Returns a TeamGameLog dataframe with consistent columns:
      GAME_ID, GAME_DATE, MATCHUP, WL
    Fixes the common schema mismatch where nba_api returns Game_ID instead of GAME_ID.
    """
    try:
        frames = _retry_api(
            teamgamelog.TeamGameLog,
            dict(
                team_id=int(team_id),
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
            ),
        )
        df = _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    rename_map = {}
    if "Game_ID" in df.columns and "GAME_ID" not in df.columns:
        rename_map["Game_ID"] = "GAME_ID"
    if "GameDate" in df.columns and "GAME_DATE" not in df.columns:
        rename_map["GameDate"] = "GAME_DATE"
    if "Matchup" in df.columns and "MATCHUP" not in df.columns:
        rename_map["Matchup"] = "MATCHUP"
    if "W/L" in df.columns and "WL" not in df.columns:
        rename_map["W/L"] = "WL"

    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ["GAME_ID", "GAME_DATE", "MATCHUP", "WL"]:
        if c not in df.columns:
            df[c] = pd.NA

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return df


def fetch_game_team_boxscore(game_id: str, team_id: int) -> pd.DataFrame:
    """
    BoxScoreTraditionalV2: returns TEAM-level player rows for the given TEAM_ID.
    """
    try:
        frames = _retry_api(boxscoretraditionalv2.BoxScoreTraditionalV2, dict(game_id=str(game_id)))
        df = _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "TEAM_ID" not in df.columns:
        return pd.DataFrame()

    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    df = df[df["TEAM_ID"] == int(team_id)].copy()
    return df.reset_index(drop=True)


# ----------------------- Player -----------------------
def fetch_player_logs(player_id: int, season: str) -> pd.DataFrame:
    try:
        frames = _retry_api(
            playergamelog.PlayerGameLog,
            dict(
                player_id=int(player_id),
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
            ),
        )
        df = _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def fetch_common_player_info(player_id: int) -> pd.DataFrame:
    try:
        frames = _retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": int(player_id)})
        return _to_df(frames, 0)
    except Exception:
        return pd.DataFrame()


def fetch_career_summary(player_id: int) -> pd.DataFrame:
    try:
        frames = _retry_api(playercareerstats.PlayerCareerStats, dict(player_id=int(player_id)))
        season_totals = _to_df(frames, 0)
        career_totals = _to_df(frames, 1) if len(frames) > 1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    use = None
    if career_totals is not None and not career_totals.empty:
        use = career_totals.copy()
    elif season_totals is not None and not season_totals.empty:
        use = season_totals.copy()
        num_cols = use.select_dtypes(include=[np.number]).columns
        use = pd.DataFrame([use[num_cols].sum(numeric_only=True)])
    else:
        return pd.DataFrame()

    for c in ["GP", "MIN", "PTS", "REB", "AST", "FGM", "FG3M", "OREB", "DREB"]:
        if c not in use.columns:
            use[c] = np.nan
        use[c] = pd.to_numeric(use[c], errors="coerce")

    gp = float(use["GP"].iloc[0]) if pd.notna(use["GP"].iloc[0]) else np.nan
    mins = float(use["MIN"].iloc[0]) if pd.notna(use["MIN"].iloc[0]) else np.nan

    def per_game(col):
        v = float(use[col].iloc[0]) if col in use.columns and pd.notna(use[col].iloc[0]) else np.nan
        return v / gp if gp and gp > 0 else np.nan

    def per36(col):
        v = float(use[col].iloc[0]) if col in use.columns and pd.notna(use[col].iloc[0]) else np.nan
        return (v / mins) * 36.0 if mins and mins > 0 else np.nan

    out = pd.DataFrame(
        {
            "GP": [gp],
            "MIN": [mins],
            "PTS_PG": [per_game("PTS")],
            "REB_PG": [per_game("REB")],
            "AST_PG": [per_game("AST")],
            "FGM_PG": [per_game("FGM")],
            "FG3M_PG": [per_game("FG3M")],
            "OREB_PG": [per_game("OREB")],
            "DREB_PG": [per_game("DREB")],
            "PTS_36": [per36("PTS")],
            "REB_36": [per36("REB")],
            "AST_36": [per36("AST")],
            "FGM_36": [per36("FGM")],
            "FG3M_36": [per36("FG3M")],
            "OREB_36": [per36("OREB")],
            "DREB_36": [per36("DREB")],
        }
    )
    return out
