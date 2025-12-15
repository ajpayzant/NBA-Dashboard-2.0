from __future__ import annotations

import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd

from .cache_store import get_or_refresh
from .loaders import (
    get_teams_static_df,
    fetch_daily_scoreboard,
    fetch_league_standings,
    fetch_league_team_stats,
    fetch_league_players_stats,
    fetch_league_players_index,
    fetch_player_logs,
    fetch_common_player_info,
    fetch_team_gamelog,
    fetch_game_team_boxscore,
    fetch_career_totals,
)

# App intent: long-lived stats, short-lived “today” context
CACHE_HOURS = 8
TEAM_CTX_TTL_SECONDS = 300

TTL_LONG = CACHE_HOURS * 3600
TTL_SHORT = TEAM_CTX_TTL_SECONDS


def get_teams_static(force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key="teams_static",
        ttl_seconds=30 * 24 * 3600,
        fetch_fn=get_teams_static_df,
        force_refresh=force_refresh,
    ).df


def get_daily_scoreboard(game_date: dt.date, force_refresh: bool = False) -> pd.DataFrame:
    key = f"scoreboard__{game_date.isoformat()}"
    return get_or_refresh(
        key=key,
        ttl_seconds=TTL_SHORT,
        fetch_fn=lambda: fetch_daily_scoreboard(game_date),
        force_refresh=force_refresh,
    ).df


def get_schedule_range(start_date: dt.date, days: int, force_refresh: bool = False) -> pd.DataFrame:
    all_days = []
    for i in range(int(days)):
        d = start_date + dt.timedelta(days=i)
        df = get_daily_scoreboard(d, force_refresh=force_refresh)
        if df is None or df.empty:
            continue
        all_days.append(df[["DATE", "MATCHUP", "STATUS"]].copy())
    if not all_days:
        return pd.DataFrame()
    sched = pd.concat(all_days, ignore_index=True)
    return sched.sort_values(["DATE", "MATCHUP"]).reset_index(drop=True)


def fetch_league_standings_cached(season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"league_standings__{season}",
        ttl_seconds=TTL_LONG,
        fetch_fn=lambda: fetch_league_standings(season),
        force_refresh=force_refresh,
    ).df


def fetch_league_team_stats_cached(season: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = get_or_refresh(
        key=f"teamstats_base__{season}",
        ttl_seconds=TTL_LONG,
        fetch_fn=lambda: fetch_league_team_stats(season)[0],
        force_refresh=force_refresh,
    ).df
    adv = get_or_refresh(
        key=f"teamstats_adv__{season}",
        ttl_seconds=TTL_LONG,
        fetch_fn=lambda: fetch_league_team_stats(season)[1],
        force_refresh=force_refresh,
    ).df
    opp = get_or_refresh(
        key=f"teamstats_opp__{season}",
        ttl_seconds=TTL_LONG,
        fetch_fn=lambda: fetch_league_team_stats(season)[2],
        force_refresh=force_refresh,
    ).df
    return base, adv, opp


def fetch_league_players_stats_cached(season: str, per_mode: str, last_n_games: int, force_refresh: bool = False) -> pd.DataFrame:
    key = f"playerstats__{season}__{per_mode}__last{int(last_n_games)}"
    ttl = TTL_LONG if int(last_n_games) == 0 else TTL_SHORT
    return get_or_refresh(
        key=key,
        ttl_seconds=ttl,
        fetch_fn=lambda: fetch_league_players_stats(season, per_mode=per_mode, last_n_games=last_n_games),
        force_refresh=force_refresh,
    ).df


def get_league_players_index(season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"player_index__{season}",
        ttl_seconds=TTL_LONG,
        fetch_fn=lambda: fetch_league_players_index(season),
        force_refresh=force_refresh,
    ).df


def get_player_logs(player_id: int, season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"player_gamelog__{season}__{int(player_id)}",
        ttl_seconds=TTL_SHORT,
        fetch_fn=lambda: fetch_player_logs(player_id, season),
        force_refresh=force_refresh,
    ).df


def get_common_player_info(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"player_info__{int(player_id)}",
        ttl_seconds=7 * 24 * 3600,
        fetch_fn=lambda: fetch_common_player_info(player_id),
        force_refresh=force_refresh,
    ).df


def get_team_gamelog(team_id: int, season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"team_gamelog__{season}__{int(team_id)}",
        ttl_seconds=TTL_SHORT,
        fetch_fn=lambda: fetch_team_gamelog(team_id, season),
        force_refresh=force_refresh,
    ).df


def get_game_team_boxscore(game_id: str, team_id: int, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"boxscore_team__{str(game_id)}__{int(team_id)}",
        ttl_seconds=30 * 24 * 3600,
        fetch_fn=lambda: fetch_game_team_boxscore(game_id, team_id),
        force_refresh=force_refresh,
    ).df


def get_career_summary(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Produces the same schema your old app expects:
    GP, MIN, PTS_PG, REB_PG, AST_PG, ... and per-36 equivalents
    """
    raw = get_or_refresh(
        key=f"career_totals_raw__{int(player_id)}",
        ttl_seconds=7 * 24 * 3600,
        fetch_fn=lambda: pd.concat(fetch_career_totals(player_id), axis=0, ignore_index=True),
        force_refresh=force_refresh,
    ).df

    if raw is None or raw.empty:
        return pd.DataFrame()

    use = raw.copy()
    num_cols = use.select_dtypes(include=[np.number]).columns

    if "GP" in use.columns and len(use) >= 1:
        gp_num = pd.to_numeric(use["GP"], errors="coerce").fillna(0)
        use_row = use.loc[[gp_num.idxmax()]].copy()
    else:
        use_row = pd.DataFrame([use[num_cols].sum(numeric_only=True)])

    for c in ["GP", "MIN", "PTS", "REB", "AST", "FGM", "FG3M", "OREB", "DREB"]:
        if c not in use_row.columns:
            use_row[c] = np.nan
        use_row[c] = pd.to_numeric(use_row[c], errors="coerce")

    gp = float(use_row["GP"].iloc[0]) if pd.notna(use_row["GP"].iloc[0]) else np.nan
    mins = float(use_row["MIN"].iloc[0]) if pd.notna(use_row["MIN"].iloc[0]) else np.nan

    def per_game(col: str) -> float:
        v = float(use_row[col].iloc[0]) if col in use_row.columns and pd.notna(use_row[col].iloc[0]) else np.nan
        return v / gp if gp and gp > 0 else np.nan

    def per36(col: str) -> float:
        v = float(use_row[col].iloc[0]) if col in use_row.columns and pd.notna(use_row[col].iloc[0]) else np.nan
        return (v / mins) * 36.0 if mins and mins > 0 else np.nan

    return pd.DataFrame(
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
