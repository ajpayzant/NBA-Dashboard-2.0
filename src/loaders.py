from __future__ import annotations

import datetime as dt
from typing import Tuple, Optional, List

import pandas as pd

from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguestandings,
    leaguedashteamstats,
    leaguedashplayerstats,
    teamgamelog,
    playergamelog,
    commonteamroster,
    playercareerstats,
    boxscoretraditionalv3,
)
from nba_api.stats.library.parameters import SeasonAll

from .utils import team_id_maps, coerce_team_keys, coerce_game_keys, minutes_str_to_float


def _endpoint_df(endpoint_obj, idx: int = 0) -> pd.DataFrame:
    """
    Robustly get a dataframe from an nba_api endpoint object.
    """
    dfs = endpoint_obj.get_data_frames()
    if not dfs:
        return pd.DataFrame()
    if idx >= len(dfs):
        return dfs[0]
    return dfs[idx]


# -------------------------
# Schedule / Scoreboard
# -------------------------
def fetch_daily_scoreboard(game_date: dt.date) -> pd.DataFrame:
    """
    One row per matchup with matchup, time, and final score if completed.
    Uses scoreboardv2; normalizes team abbreviations via static mapping.
    """
    date_str = game_date.strftime("%m/%d/%Y")
    obj = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)

    games = _endpoint_df(obj, 0).copy()
    if games.empty:
        return pd.DataFrame(columns=["GAME_ID", "DATE", "MATCHUP", "STATUS", "START_TIME_LOCAL", "SCORE"])

    id_to_abbr, _ = team_id_maps()

    # ScoreboardV2 games df usually has HOME_TEAM_ID / VISITOR_TEAM_ID and GAME_STATUS_TEXT
    for col in ["HOME_TEAM_ID", "VISITOR_TEAM_ID"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce").astype("Int64")

    games["HOME_ABBR"] = games["HOME_TEAM_ID"].map(lambda x: id_to_abbr.get(int(x), None) if pd.notna(x) else None)
    games["AWAY_ABBR"] = games["VISITOR_TEAM_ID"].map(lambda x: id_to_abbr.get(int(x), None) if pd.notna(x) else None)

    # time columns vary; GAME_STATUS_TEXT contains 'Final' etc.
    status = games.get("GAME_STATUS_TEXT", pd.Series([""] * len(games)))
    start_time = games.get("GAME_STATUS_TEXT", status).astype("string")  # fallback

    # Line score table contains points by team; join to compute score
    lines = _endpoint_df(obj, 1).copy()
    if not lines.empty and "TEAM_ID" in lines.columns and "PTS" in lines.columns:
        lines["TEAM_ID"] = pd.to_numeric(lines["TEAM_ID"], errors="coerce").astype("Int64")
        lines["TEAM_ABBR"] = lines["TEAM_ID"].map(lambda x: id_to_abbr.get(int(x), None) if pd.notna(x) else None)
        pts_map = lines.pivot_table(index="GAME_ID", columns="TEAM_ABBR", values="PTS", aggfunc="max")
    else:
        pts_map = pd.DataFrame()

    out = pd.DataFrame()
    out["GAME_ID"] = games.get("GAME_ID", pd.Series([], dtype="string")).astype("string")
    out["DATE"] = pd.to_datetime(games.get("GAME_DATE_EST", date_str), errors="coerce").dt.date
    out["MATCHUP"] = games["AWAY_ABBR"].astype("string") + " @ " + games["HOME_ABBR"].astype("string")
    out["STATUS"] = status.astype("string")

    # Try to derive local start time
    if "GAME_STATUS_TEXT" in games.columns and "GAME_STATUS_ID" in games.columns:
        # Not perfect, but better than blank. Many Scoreboard fields differ by season.
        out["START_TIME_LOCAL"] = games.get("GAME_STATUS_TEXT", "").astype("string")
    else:
        out["START_TIME_LOCAL"] = ""

    # Compute score if available
    scores = []
    for gid, away, home in zip(out["GAME_ID"], games["AWAY_ABBR"], games["HOME_ABBR"]):
        if pts_map.empty or gid not in pts_map.index:
            scores.append("")
            continue
        try:
            a = pts_map.loc[gid, away] if away in pts_map.columns else None
            h = pts_map.loc[gid, home] if home in pts_map.columns else None
            if pd.notna(a) and pd.notna(h):
                scores.append(f"{int(a)}-{int(h)}")
            else:
                scores.append("")
        except Exception:
            scores.append("")
    out["SCORE"] = scores

    return out


# -------------------------
# Standings + Team Stats (and Opponent)
# -------------------------
def fetch_league_team_summary(season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (team_table, opp_table)
    team_table: standings + per-game team stats + ratings
    opp_table: opponent per-game stats
    """
    # Standings
    st = leaguestandings.LeagueStandings(season=season, timeout=30)
    standings = _endpoint_df(st, 0).copy()
    standings = coerce_team_keys(standings)

    # Team stats
    ts = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        timeout=30,
    )
    base = _endpoint_df(ts, 0).copy()
    base = coerce_team_keys(base)

    # Advanced stats (ratings, net)
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=30,
    )
    adv = _endpoint_df(adv, 0).copy()
    adv = coerce_team_keys(adv)

    # Opponent (Base)
    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent",
        timeout=30,
    )
    opp = _endpoint_df(opp, 0).copy()
    opp = coerce_team_keys(opp)

    # Normalize keys for safe merges
    for d in (standings, base, adv, opp):
        if "TEAM_ABBREVIATION" in d.columns:
            d["TEAM_ABBREVIATION"] = d["TEAM_ABBREVIATION"].astype("string")
        if "TEAM_ID" in d.columns:
            d["TEAM_ID"] = pd.to_numeric(d["TEAM_ID"], errors="coerce").astype("Int64")

    # Select columns
    # Standings varies by endpoint version; handle robustly
    keep_st = []
    for c in ["TeamName", "TEAM_NAME", "TEAM", "TEAM_ID", "TEAM_ABBREVIATION", "WINS", "LOSSES", "WinPCT", "W", "L", "PCT", "Conference"]:
        if c in standings.columns:
            keep_st.append(c)
    st2 = standings[keep_st].copy() if keep_st else standings.copy()

    # Harmonize column names
    rename_map = {
        "TeamName": "TEAM_NAME",
        "TEAM": "TEAM_NAME",
        "W": "WINS",
        "L": "LOSSES",
        "PCT": "WIN_PCT",
        "WinPCT": "WIN_PCT",
        "Conference": "CONF",
    }
    st2 = st2.rename(columns=rename_map)

    if "WIN_PCT" not in st2.columns and "WINS" in st2.columns and "LOSSES" in st2.columns:
        st2["WIN_PCT"] = (pd.to_numeric(st2["WINS"], errors="coerce") / (pd.to_numeric(st2["WINS"], errors="coerce") + pd.to_numeric(st2["LOSSES"], errors="coerce")))

    # Base per-game stats select
    base_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "GP", "PTS", "REB", "AST", "TOV", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT"]
    base_cols = [c for c in base_cols if c in base.columns]
    base2 = base[base_cols].copy()

    # Advanced select
    adv_cols = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    adv_cols = [c for c in adv_cols if c in adv.columns]
    adv2 = adv[adv_cols].copy()

    # Merge standings + base + adv on TEAM_ID (most stable)
    if "TEAM_ID" in st2.columns and "TEAM_ID" in base2.columns:
        merged = st2.merge(base2, on="TEAM_ID", how="left")
    else:
        # fallback: attempt on TEAM_ABBREVIATION
        merged = st2.merge(base2, on="TEAM_ABBREVIATION", how="left")

    if "TEAM_ID" in merged.columns and "TEAM_ID" in adv2.columns:
        merged = merged.merge(adv2, on="TEAM_ID", how="left")

    # Final cleanup types
    for c in ["WINS", "LOSSES"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    if "WIN_PCT" in merged.columns:
        merged["WIN_PCT"] = pd.to_numeric(merged["WIN_PCT"], errors="coerce")

    # Opp table: keep opponent metrics only, but carry team identifiers
    opp_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "GP", "OPP_PTS", "OPP_REB", "OPP_AST", "OPP_TOV", "OPP_FG_PCT", "OPP_FG3_PCT"]
    # Endpoint names vary; detect likely columns
    if "PTS" in opp.columns and "OPP_PTS" not in opp.columns:
        # In 'Opponent' measure, fields are often still PTS/REB/AST but represent opponent values
        # Rename with OPP_ prefix
        ren = {}
        for c in ["PTS", "REB", "AST", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT"]:
            if c in opp.columns:
                ren[c] = f"OPP_{c}"
        opp2 = opp.rename(columns=ren)
    else:
        opp2 = opp.copy()

    opp_keep = ["TEAM_ID", "TEAM_ABBREVIATION", "GP"]
    for c in opp2.columns:
        if c.startswith("OPP_") and c not in opp_keep:
            opp_keep.append(c)
    opp2 = opp2[[c for c in opp_keep if c in opp2.columns]].copy()

    return merged, opp2


# -------------------------
# Leaguewide Player Leaders
# -------------------------
def fetch_league_player_stats(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    obj = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=per_mode,
        timeout=30,
    )
    df = _endpoint_df(obj, 0).copy()
    # normalize core fields
    if "TEAM_ABBREVIATION" in df.columns:
        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype("string")
    if "GP" in df.columns:
        df["GP"] = pd.to_numeric(df["GP"], errors="coerce")
    return df


# -------------------------
# Team tab helpers
# -------------------------
def fetch_team_game_log(team_id: int, season: str) -> pd.DataFrame:
    obj = teamgamelog.TeamGameLog(team_id=team_id, season=season, timeout=30)
    df = _endpoint_df(obj, 0).copy()
    df = coerce_game_keys(df)
    return df


def fetch_team_roster(team_id: int, season: str) -> pd.DataFrame:
    # commonteamroster wants season like '2024-25'
    obj = commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=30)
    df = _endpoint_df(obj, 0).copy()
    # Standard fields: PLAYER, PLAYER_ID, POSITION
    if "PLAYER_ID" in df.columns:
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    return df


def fetch_boxscore_team_players(game_id: str) -> pd.DataFrame:
    obj = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30)
    # DataFrames: typically first is player stats
    df = _endpoint_df(obj, 0).copy()
    return df


# -------------------------
# Player tab helpers
# -------------------------
def fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    obj = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=30)
    df = _endpoint_df(obj, 0).copy()
    df = coerce_game_keys(df)
    return df


def fetch_player_career(player_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obj = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=30)
    season_totals = _endpoint_df(obj, 0).copy()
    career_totals = _endpoint_df(obj, 1).copy() if len(obj.get_data_frames()) > 1 else pd.DataFrame()
    return season_totals, career_totals


def compute_rotation_minutes_table(roster: pd.DataFrame, player_logs: List[pd.DataFrame], team_recent_game_ids: List[str]) -> pd.DataFrame:
    """
    Build rotation table:
      - MIN_L3, MIN_L10, MIN_SEASON, and minutes for each recent game_id
    Requires roster (PLAYER, PLAYER_ID) and list of per-player logs.
    """
    # Build a dict of player_id -> log
    logs_map = {}
    for lg in player_logs:
        if not lg.empty and "Player_ID" in lg.columns:
            pid = lg["Player_ID"].iloc[0]
            logs_map[int(pid)] = lg.copy()

    rows = []
    for _, r in roster.iterrows():
        pid = r.get("PLAYER_ID")
        name = r.get("PLAYER")
        pos = r.get("POSITION")
        if pd.isna(pid):
            continue
        pid = int(pid)
        lg = logs_map.get(pid, pd.DataFrame())

        mins = []
        if not lg.empty and "MIN" in lg.columns:
            mins = [minutes_str_to_float(x) for x in lg["MIN"].tolist()]
        # logs are usually reverse-chronological already
        min_l3 = sum(mins[:3]) / 3 if len(mins) >= 3 else (sum(mins) / max(len(mins), 1) if mins else 0)
        min_l10 = sum(mins[:10]) / 10 if len(mins) >= 10 else (sum(mins) / max(len(mins), 1) if mins else 0)
        min_season = (sum(mins) / max(len(mins), 1)) if mins else 0

        row = {
            "Player": name,
            "Pos": pos,
            "MIN_L3": round(min_l3, 1),
            "MIN_L10": round(min_l10, 1),
            "MIN_SEASON": round(min_season, 1),
        }

        # exact minutes for last 5 games by team game_id
        if not lg.empty and "GAME_ID" in lg.columns and "MIN" in lg.columns:
            lg2 = lg.copy()
            lg2["GAME_ID"] = lg2["GAME_ID"].astype("string")
            lg2["_MINF"] = lg2["MIN"].apply(minutes_str_to_float)
            for gid in team_recent_game_ids:
                val = lg2.loc[lg2["GAME_ID"] == str(gid), "_MINF"]
                row[str(gid)] = round(float(val.iloc[0]), 1) if not val.empty else 0.0
        else:
            for gid in team_recent_game_ids:
                row[str(gid)] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)
