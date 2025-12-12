# app.py — NBA League + Player + Team Dashboard
# Uses nba_api live endpoints with caching. Designed for Streamlit Cloud.

import time
import datetime
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
from zoneinfo import ZoneInfo  # ET cutoff for season-to-date

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    commonplayerinfo,
    leaguegamefinder,
    teamdashboardbygeneralsplits,
    scoreboardv2,
    leaguestandingsv3,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Dashboards", layout="wide")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
TEAM_CTX_TTL_SECONDS = 300
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    """
    Lightweight retry wrapper for nba_api endpoints.
    Keeps calls gentle but resilient without hammering the API.
    """
    last_err = None
    for i in range(retries + 1):
        try:
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(sleep * (i + 1))
    raise last_err

def _season_labels(start=2010, end=None):
    if end is None:
        end = datetime.datetime.utcnow().year
    def lab(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start-1, -1)]

SEASONS = _season_labels(2015, datetime.datetime.utcnow().year)

def _prev_season_label(season_label: str) -> str:
    try:
        y0 = int(season_label.split("-")[0])
        return f"{y0-1}-{str((y0)%100).zfill(2)}"
    except Exception:
        return season_label

# ----------------------- UI Helpers (rank-aware tiles) -----------------------
def inject_rank_tile_css():
    st.markdown(
        """
        <style>
        .rank-tile { border-radius: 14px; padding: 10px 12px; margin: 4px 0; border: 1px solid rgba(0,0,0,0.08); }
        .rank-tile .label { font-size: 0.85rem; opacity: 0.85; margin-bottom: 4px; }
        .rank-tile .value { font-weight: 700; font-size: 1.25rem; line-height: 1.2; }
        .rank-tile .delta { font-size: 0.8rem; margin-top: 2px; opacity: 0.9; }
        .rank-good  { background: rgba(0,170,85,0.16); }
        .rank-mid   { background: rgba(180,180,180,0.16); }
        .rank-bad   { background: rgba(220,30,40,0.16); }
        @media (prefers-color-scheme: dark) {
            .rank-tile { border-color: rgba(255,255,255,0.12); }
            .rank-good  { background: rgba(0,170,85,0.22); }
            .rank-mid   { background: rgba(200,200,200,0.10); }
            .rank-bad   { background: rgba(255,60,70,0.22); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _fmt1(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"

def _rank_tile(col, label, value, rank, total=30, pct=False, decimals=1):
    """
    Top 25% (1–8): green ▲
    Middle (9–22): neutral •
    Bottom 25% (23–30): red ▼
    """
    if pd.isna(rank):
        tier_class, arrow, rank_txt = "rank-mid", "•", "Rank —"
    else:
        r = int(rank)
        if r <= 8:
            tier_class, arrow = "rank-good", "▲"
        elif r >= 23:
            tier_class, arrow = "rank-bad", "▼"
        else:
            tier_class, arrow = "rank-mid", "•"
        rank_txt = f"{arrow} Rank {r}/{total}"

    if pct:
        val_txt = f"{float(value)*100:.{decimals}f}%" if pd.notna(value) else "—"
    else:
        try:
            val_txt = f"{float(value):.{decimals}f}" if pd.notna(value) else "—"
        except Exception:
            val_txt = "—"

    col.markdown(
        f"""
        <div class="rank-tile {tier_class}">
            <div class="label">{label}</div>
            <div class="value">{val_txt}</div>
            <div class="delta">{rank_txt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------- Misc Utils -----------------------
def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: "{:.2f}" for c in num_cols}

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

_punct_re = re.compile(r"[^\w]")
def parse_opp_from_matchup(matchup_str: str):
    if not isinstance(matchup_str, str):
        return None
    parts = matchup_str.split()
    if len(parts) < 3:
        return None
    token = parts[-1].upper().strip()
    token = _punct_re.sub("", token)
    return token

def add_shot_breakouts(df):
    for col in ["MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]:
        if col not in df.columns:
            df[col] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["2PM"] = df["FGM"] - df["FG3M"]
    df["2PA"] = df["FGA"] - df["FG3A"]
    keep_order = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","2PM","2PA",
                  "FG3M","FG3A","FTM","FTA","OREB","DREB"]
    existing = [c for c in keep_order if c in df.columns]
    return df[existing]

def format_record(w, l):
    try:
        return f"{int(w)}–{int(l)}"
    except Exception:
        return "—"

def append_average_row(df: pd.DataFrame, label: str = "Average") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return out
    avg_vals = out[num_cols].mean(numeric_only=True)
    avg_row = {c: np.nan for c in out.columns}
    for c in num_cols:
        avg_row[c] = float(avg_vals.get(c, np.nan))
    if "GAME_DATE" in out.columns:
        avg_row["GAME_DATE"] = pd.NaT
    if "MATCHUP" in out.columns:
        avg_row["MATCHUP"] = label
    if "WL" in out.columns:
        avg_row["WL"] = ""
    out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)
    return out

# ---------- Per-36 helpers ----------
def _safe_num(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def compute_per36_from_logs(logs: pd.DataFrame) -> dict:
    keys = ["PTS","REB","AST","PRA","FG2M","FG3M","FTM","OREB","DREB"]
    if logs is None or logs.empty:
        return {k: np.nan for k in keys}
    need = ["MIN","PTS","REB","AST","FGM","FG3M","FTM","OREB","DREB"]
    for c in need:
        if c not in logs.columns:
            logs[c] = 0
        logs[c] = _safe_num(logs[c])

    sum_min = logs["MIN"].sum()
    sum_pts = logs["PTS"].sum()
    sum_reb = logs["REB"].sum()
    sum_ast = logs["AST"].sum()
    sum_fgm = logs["FGM"].sum()
    sum_fg3 = logs["FG3M"].sum()
    sum_ftm = logs["FTM"].sum()
    sum_oreb= logs["OREB"].sum()
    sum_dreb= logs["DREB"].sum()
    sum_pra = sum_pts + sum_reb + sum_ast
    sum_fg2 = max(0.0, sum_fgm - sum_fg3)

    def per36(x): return 36.0 * (x / sum_min) if sum_min and sum_min > 0 else np.nan
    return {
        "PTS":  per36(sum_pts),
        "REB":  per36(sum_reb),
        "AST":  per36(sum_ast),
        "PRA":  per36(sum_pra),
        "FG2M": per36(sum_fg2),
        "FG3M": per36(sum_fg3),
        "FTM":  per36(sum_ftm),
        "OREB": per36(sum_oreb),
        "DREB": per36(sum_dreb),
    }

def per36_from_career_totals(career_df: pd.DataFrame) -> dict:
    keys = ["PTS","REB","AST","PRA","FG2M","FG3M","FTM","OREB","DREB"]
    if career_df is None or career_df.empty:
        return {k: np.nan for k in keys}
    need = ["MIN","PTS","REB","AST","FGM","FG3M","FTM","OREB","DREB"]
    for c in need:
        if c not in career_df.columns:
            career_df[c] = 0
        career_df[c] = _safe_num(career_df[c])
    sum_min = career_df["MIN"].sum()
    sum_pts = career_df["PTS"].sum()
    sum_reb = career_df["REB"].sum()
    sum_ast = career_df["AST"].sum() if "AST" in career_df.columns else 0.0
    sum_fgm = career_df["FGM"].sum()
    sum_fg3 = career_df["FG3M"].sum()
    sum_ftm = career_df["FTM"].sum()
    sum_oreb= career_df["OREB"].sum()
    sum_dreb= career_df["DREB"].sum()
    sum_pra = sum_pts + sum_reb + sum_ast
    sum_fg2 = max(0.0, sum_fgm - sum_fg3)

    def per36(x): return 36.0 * (x / sum_min) if sum_min and sum_min > 0 else np.nan
    return {
        "PTS":  per36(sum_pts),
        "REB":  per36(sum_reb),
        "AST":  per36(sum_ast),
        "PRA":  per36(sum_pra),
        "FG2M": per36(sum_fg2),
        "FG3M": per36(sum_fg3),
        "FTM":  per36(sum_ftm),
        "OREB": per36(sum_oreb),
        "DREB": per36(sum_dreb),
    }

def _series_from_dict(d: dict, name: str) -> pd.Series:
    s = pd.Series(d, dtype="float64")
    s.name = name
    return s

# ----------------------- Team name helpers -----------------------
def _build_static_maps():
    teams_df = pd.DataFrame(static_teams.get_teams())
    by_full = dict(zip(teams_df["full_name"].astype(str),
                       teams_df["abbreviation"].astype(str)))
    id_by_full = dict(zip(teams_df["full_name"].astype(str),
                          teams_df["id"].astype(int)))

    nick_map = {
        "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
        "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
        "NY Knicks": "NYK", "New York Knicks": "NYK",
        "GS Warriors": "GSW", "Golden State Warriors": "GSW",
        "SA Spurs": "SAS", "San Antonio Spurs": "SAS",
        "NO Pelicans": "NOP", "New Orleans Pelicans": "NOP",
        "OKC Thunder": "OKC", "Oklahoma City Thunder": "OKC",
        "PHX Suns": "PHX", "Phoenix Suns": "PHX",
        "POR Trail Blazers": "POR", "Portland Trail Blazers": "POR",
        "UTA Jazz": "UTA", "Utah Jazz": "UTA",
        "WAS Wizards": "WAS", "Washington Wizards": "WAS",
        "CLE Cavaliers": "CLE", "Cleveland Cavaliers": "CLE",
        "MIN Timberwolves": "MIN", "Minnesota Timberwolves": "MIN",
        "CHA Hornets": "CHA", "Charlotte Hornets": "CHA",
        "BRK Nets": "BKN", "Brooklyn Nets": "BKN",
        "PHI 76ers": "PHI", "Philadelphia 76ers": "PHI",
    }
    alias_map = {
        "PHO":"PHX","BRK":"BKN","NJN":"BKN","NOH":"NOP","NOK":"NOP",
        "CHO":"CHA","CHH":"CHA","SEA":"OKC","WSB":"WAS","VAN":"MEM"
    }

    by_full_cf = {k.casefold(): v for k, v in by_full.items()}
    nick_cf = {k.casefold(): v for k, v in nick_map.items()}
    alias_up = {k.upper(): v.upper() for k, v in alias_map.items()}
    return by_full_cf, nick_cf, alias_up, id_by_full

BY_FULL_CF, NICK_CF, ABBR_ALIAS, TEAMID_BY_FULL = _build_static_maps()

def normalize_abbr(abbr: str | None) -> str | None:
    if not isinstance(abbr, str) or not abbr:
        return None
    a = abbr.upper().strip()
    return ABBR_ALIAS.get(a, a)

def resolve_team_abbrev(team_name: str, team_ctx_row: pd.Series | None = None) -> str | None:
    if team_ctx_row is not None and "TEAM_ABBREVIATION" in team_ctx_row.index:
        v = str(team_ctx_row.get("TEAM_ABBREVIATION", "")).strip().upper()
        if 2 <= len(v) <= 4:
            return normalize_abbr(v)
    if isinstance(team_name, str):
        cf = team_name.casefold().strip()
        if cf in BY_FULL_CF:
            return normalize_abbr(BY_FULL_CF[cf])
        if cf in NICK_CF:
            return normalize_abbr(NICK_CF[cf])
    return None

def resolve_team_id(team_name: str, team_ctx_row: pd.Series | None = None) -> int | None:
    if team_ctx_row is not None and "TEAM_ID" in team_ctx_row.index:
        try:
            return int(team_ctx_row["TEAM_ID"])
        except Exception:
            pass
    return TEAMID_BY_FULL.get(team_name)

# ----------------------- Shared cached accessors -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_season_player_index(season):
    try:
        frames = _retry_api(LeagueDashPlayerStats, {
            "season": season,
            "per_mode_detailed": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        })
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    keep = ["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME","GP","MIN"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    return df[keep].drop_duplicates(subset=["PLAYER_ID"]).sort_values(
        ["TEAM_NAME","PLAYER_NAME"]
    ).reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_logs(player_id, season):
    try:
        frames = _retry_api(playergamelog.PlayerGameLog, {
            "player_id": player_id,
            "season": season,
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        })
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_career(player_id):
    try:
        frames = _retry_api(playercareerstats.PlayerCareerStats, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_common_player_info(player_id):
    try:
        frames = _retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_all_player_logs_all_seasons(player_id, season_labels):
    frames = []
    for s in season_labels:
        df = get_player_logs(player_id, s)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_vs_opponent_games(player_id: int, opp_team_id: int):
    try:
        frames = _retry_api(
            leaguegamefinder.LeagueGameFinder,
            {
                "player_or_team_abbreviation": "P",
                "player_id_nullable": player_id,
                "vs_team_id_nullable": opp_team_id,
                "season_type_nullable": "Regular Season",
                "league_id_nullable": "00",
            },
        )
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    wanted = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","FGM","FGA",
              "FG3M","FG3A","FTM","FTA","OREB","DREB"]
    for c in wanted:
        if c not in df.columns:
            df[c] = 0
    return df[wanted]

# ----------------------- Team context for Player tab -----------------------
@st.cache_data(ttl=TEAM_CTX_TTL_SECONDS, show_spinner=False)
def get_team_context_regular_season_to_date(season: str, cutoff_date_et: str, _refresh_key: int = 0):
    common = dict(
        season=season,
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        date_from_nullable=None,
        date_to_nullable=cutoff_date_et,
        po_round_nullable=None,
    )

    def _safe_frames(ep_cls, kwargs):
        try:
            frames = _retry_api(ep_cls, kwargs)
            return frames[0] if frames else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    adv = _safe_frames(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(common, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame"),
    )
    base = _safe_frames(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(common, measure_type_detailed_defense="Base", per_mode_detailed="Totals"),
    )
    opp = _safe_frames(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(common, measure_type_detailed_defense="Opponent", per_mode_detailed="PerGame"),
    )

    def _nba_only(df):
        if df is None or df.empty or "TEAM_ID" not in df.columns:
            return pd.DataFrame()
        return df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()

    adv = _nba_only(adv)
    base = _nba_only(base)
    opp  = _nba_only(opp)

    for df in (adv, base, opp):
        if not df.empty:
            df.sort_values(["TEAM_ID"], inplace=True)
            df.drop_duplicates(subset=["TEAM_ID"], keep="first", inplace=True)

    adv_cols = ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","PACE","OFF_RATING",
                "DEF_RATING","NET_RATING"]
    for c in adv_cols:
        if c not in adv.columns: adv[c] = np.nan
    adv = adv[adv_cols].copy()

    base_cols = ["TEAM_ID","GP","W","L","W_PCT","MIN"]
    for c in base_cols:
        if c not in base.columns: base[c] = np.nan
    base = base[base_cols].copy()

    opp_cols = [c for c in opp.columns if c.startswith("OPP_")] + ["TEAM_ID"]
    opp = opp[opp_cols].copy() if not opp.empty else pd.DataFrame(columns=["TEAM_ID"])

    for c in ["PACE","OFF_RATING","DEF_RATING","NET_RATING"]:
        adv[c] = pd.to_numeric(adv[c], errors="coerce")
    for c in ["GP","W","L","W_PCT","MIN"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    for c in opp.columns:
        if c != "TEAM_ID":
            opp[c] = pd.to_numeric(opp[c], errors="coerce")

    df = pd.merge(adv, base, on="TEAM_ID", how="inner")
    df = pd.merge(df, opp, on="TEAM_ID", how="left")

    teams_df = pd.DataFrame(static_teams.get_teams())
    abbr_map = dict(zip(teams_df["id"], teams_df["abbreviation"]))
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].fillna(df["TEAM_ID"].map(abbr_map))

    def _fix_row(r):
        bad_def = pd.isna(r["DEF_RATING"]) or not (90 <= float(r["DEF_RATING"]) <= 130)
        bad_pace = pd.isna(r["PACE"]) or not (90 <= float(r["PACE"]) <= 110)
        if not (bad_def or bad_pace):
            return r
        try:
            td = _retry_api(
                teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits,
                dict(
                    team_id=int(r["TEAM_ID"]),
                    season=season,
                    season_type_all_star="Regular Season",
                    league_id_nullable="00",
                    date_from_nullable=None,
                    date_to_nullable=cutoff_date_et,
                    measure_type_detailed_defense="Advanced",
                    per_mode_detailed="PerGame",
                ),
            )
            dash = td[0] if td else pd.DataFrame()
            if not dash.empty:
                for k in ["OFF_RATING","DEF_RATING","NET_RATING","PACE","GP","W","L","W_PCT"]:
                    if k in dash.columns:
                        r[k] = pd.to_numeric(dash.iloc[0][k], errors="coerce")
        except Exception:
            pass
        return r

    if not df.empty:
        df = df.apply(_fix_row, axis=1)

    # Own-team rating ranks
    df["DEF_RANK"] = df["DEF_RATING"].rank(ascending=True,  method="min").astype("Int64")
    df["PACE_RANK"] = df["PACE"].rank(ascending=False, method="min").astype("Int64")
    df["NET_RANK"]  = df["NET_RATING"].rank(ascending=False, method="min").astype("Int64")

    # Opponent ranks
    def _add_opp_rank(col, ascending=True):
        if col in df.columns:
            df[f"{col}_RANK"] = df[col].rank(ascending=ascending, method="min").astype("Int64")

    for col, asc in [
        ("OPP_PTS", True),
        ("OPP_FG_PCT", True),
        ("OPP_FG3_PCT", True),
        ("OPP_FT_PCT", True),
        ("OPP_REB", True),
        ("OPP_OREB", True),
        ("OPP_DREB", True),
        ("OPP_AST", True),
        ("OPP_TOV", False),
        ("OPP_STL", False),
        ("OPP_BLK", False),
        ("OPP_PF", False),
        ("OPP_FGA", True),
        ("OPP_FG3A", True),
        ("OPP_FTA", True),
    ]:
        _add_opp_rank(col, ascending=asc)

    df.sort_values("TEAM_NAME", inplace=True)
    fetched_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return df.reset_index(drop=True), fetched_at, cutoff_date_et

# =====================================================================
# LEAGUE TAB
# =====================================================================
@st.cache_data(ttl=900, show_spinner=True)
def get_daily_scoreboard(date_obj: datetime.date):
    """League schedule / scoreboard for a given date."""
    date_str = date_obj.strftime("%Y-%m-%d")
    try:
        frames = _retry_api(
            scoreboardv2.ScoreboardV2,
            {"game_date": date_str, "league_id": "00"},
            timeout=REQUEST_TIMEOUT,
            retries=2,
            sleep=1.0,
        )
    except Exception:
        return pd.DataFrame()
    if not frames:
        return pd.DataFrame()
    games = frames[0]  # GameHeader
    if games.empty:
        return games
    keep_cols = [
        "GAME_ID","GAMECODE","GAME_STATUS_TEXT","HOME_TEAM_ID","VISITOR_TEAM_ID",
        "HOME_TEAM_ABBREVIATION","VISITOR_TEAM_ABBREVIATION",
        "HOME_TEAM_WINS","HOME_TEAM_LOSSES","VISITOR_TEAM_WINS","VISITOR_TEAM_LOSSES",
        "PTS_HOME","PTS_VISITOR",
    ]
    for c in keep_cols:
        if c not in games.columns:
            games[c] = np.nan
    games = games[keep_cols].copy()
    games["MATCHUP"] = games["VISITOR_TEAM_ABBREVIATION"] + " @ " + games["HOME_TEAM_ABBREVIATION"]
    games["SCORE"] = games["PTS_VISITOR"].fillna(0).astype(int).astype(str) + " - " + \
                     games["PTS_HOME"].fillna(0).astype(int).astype(str)
    games["HOME_REC"] = games["HOME_TEAM_WINS"].astype(str) + "-" + games["HOME_TEAM_LOSSES"].astype(str)
    games["AWAY_REC"] = games["VISITOR_TEAM_WINS"].astype(str) + "-" + games["VISITOR_TEAM_LOSSES"].astype(str)
    return games[["GAME_STATUS_TEXT","MATCHUP","SCORE","AWAY_REC","HOME_REC"]].copy()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def get_league_standings(season: str):
    try:
        frames = _retry_api(
            leaguestandingsv3.LeagueStandingsV3,
            {"season": season, "league_id": "00"},
            timeout=REQUEST_TIMEOUT,
            retries=2,
            sleep=1.0,
        )
    except Exception:
        return pd.DataFrame()
    if not frames:
        return pd.DataFrame()
    df = frames[0]
    if df.empty:
        return df
    # v3 has CONFERENCE and TEAM_NAME, W, L, etc.
    keep = [
        "CONFERENCE","TEAM_NAME","TEAM_ABBREVIATION",
        "W","L","WIN_PCT","HOME_RECORD","ROAD_RECORD","L10",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()
    df["WIN_PCT"] = pd.to_numeric(df["WIN_PCT"], errors="coerce")
    df = df.sort_values(["CONFERENCE","WIN_PCT"], ascending=[True, False]).reset_index(drop=True)
    return df

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def get_league_player_leaders(season: str, per_mode: str, last_n_games: int):
    """League-wide player stats snapshot (PerGame or Per36) for leaderboards."""
    frames = _retry_api(
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
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    # Coerce numeric on key stat columns
    for c in ["PTS","REB","AST","STL","BLK","PLUS_MINUS","USG_PCT","TS_PCT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)

def league_dashboard():
    inject_rank_tile_css()
    st.title("NBA League Overview")

    # Sidebar filters
    with st.sidebar:
        st.header("League Filters")
        season = st.selectbox("Season", SEASONS, index=0, key="league_season")
        today = datetime.date.today()
        date_choice = st.date_input("Schedule date", value=today, key="league_date")
        per_mode = st.selectbox("Leaders – Per mode", ["PerGame", "Per36"], index=0)
        last_n = st.selectbox("Leaders – window", [0, 5, 10, 15], index=0,
                              format_func=lambda x: "Season" if x == 0 else f"Last {x}")

    col_sched, col_stand = st.columns([1.1, 1.4])

    # Schedule / scoreboard
    with col_sched:
        st.subheader("Schedule & Scores")
        with st.spinner("Loading scoreboard..."):
            sched = get_daily_scoreboard(date_choice)
        if sched.empty:
            st.info("No games for this date or scoreboard not available.")
        else:
            st.dataframe(
                sched.style.format({"SCORE": "{}"}),
                use_container_width=True,
                height=_auto_height(sched, max_px=400),
            )

    # Standings
    with col_stand:
        st.subheader(f"{season} Standings")
        with st.spinner("Loading standings..."):
            standings = get_league_standings(season)
        if standings.empty:
            st.info("Standings unavailable for this season.")
        else:
            east = standings[standings["CONFERENCE"] == "East"].copy()
            west = standings[standings["CONFERENCE"] == "West"].copy()
            ce, cw = st.columns(2)
            if not east.empty:
                ce.markdown("**Eastern Conference**")
                ce.dataframe(
                    east[["TEAM_NAME","TEAM_ABBREVIATION","W","L","WIN_PCT","HOME_RECORD","ROAD_RECORD","L10"]]
                    .style.format({"WIN_PCT": "{:.3f}"}),
                    use_container_width=True,
                    height=_auto_height(east, max_px=500),
                )
            if not west.empty:
                cw.markdown("**Western Conference**")
                cw.dataframe(
                    west[["TEAM_NAME","TEAM_ABBREVIATION","W","L","WIN_PCT","HOME_RECORD","ROAD_RECORD","L10"]]
                    .style.format({"WIN_PCT": "{:.3f}"}),
                    use_container_width=True,
                    height=_auto_height(west, max_px=500),
                )

    # League leaders
    st.markdown("---")
    st.subheader("League Leaderboards")

    with st.spinner("Loading league player stats..."):
        leaders_df = get_league_player_leaders(season, per_mode, last_n)

    if leaders_df.empty:
        st.info("Player leader data unavailable.")
        return

    stat_options = ["PTS","REB","AST","STL","BLK","PLUS_MINUS"]
    stat = st.selectbox("Sort by stat", stat_options, index=0)
    top_n = st.slider("Show top N players", min_value=10, max_value=50, value=25, step=5)

    df_view = leaders_df.copy()
    df_view = df_view.sort_values(stat, ascending=False, na_position="last").head(top_n)

    cols_keep = ["PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN"] + [c for c in stat_options if c in df_view.columns]
    df_view = df_view[cols_keep]
    fmt_map = {c: "{:.1f}" for c in df_view.select_dtypes(include=[np.number]).columns if c not in ["GP"]}

    st.dataframe(
        df_view.style.format(fmt_map),
        use_container_width=True,
        height=_auto_height(df_view, max_px=600),
    )

    st.caption(
        "Schedule from ScoreboardV2 (date-specific). Standings from LeagueStandingsV3. "
        "Player leaderboards from LeagueDashPlayerStats with configurable per-mode and recency window."
    )

# =====================================================================
# PLAYER DASHBOARD
# =====================================================================
def player_dashboard():
    inject_rank_tile_css()
    st.title("NBA Player Dashboard")

    # Sidebar filters
    with st.sidebar:
        st.header("Player Filters")
        season = st.selectbox("Season", SEASONS, index=0, key="season_sel")

        col_r1, col_r2 = st.columns([1,1])
        with col_r1:
            if st.button("Refresh team context"):
                st.session_state["team_ctx_refresh_key"] = st.session_state.get("team_ctx_refresh_key", 0) + 1
        with col_r2:
            if st.button("Clear all caches"):
                st.cache_data.clear()
                st.session_state["team_ctx_refresh_key"] = st.session_state.get("team_ctx_refresh_key", 0) + 1

    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    cutoff_date_et = now_et.strftime("%m/%d/%Y")

    refresh_key = st.session_state.get("team_ctx_refresh_key", 0)
    with st.spinner("Loading league team context..."):
        team_ctx, fetched_at, cutoff_used = get_team_context_regular_season_to_date(
            season, cutoff_date_et, refresh_key
        )

    if team_ctx.empty:
        st.error("Unable to load team context for this season.")
        st.stop()

    team_list = team_ctx["TEAM_NAME"].tolist()

    # Player selection
    with st.sidebar:
        with st.spinner("Loading player index..."):
            season_players = get_season_player_index(season)

        q = st.text_input("Search player", key="player_search").strip()
        filtered_players = season_players if not q else season_players[
            season_players["PLAYER_NAME"].str.contains(q, case=False, na=False)
        ]

        if filtered_players.empty:
            st.info("No players match your search.")
            st.stop()

        default_idx = 0
        if "player_sel" in st.session_state:
            if st.session_state["player_sel"] in filtered_players["PLAYER_NAME"].tolist():
                default_idx = filtered_players["PLAYER_NAME"].tolist().index(
                    st.session_state["player_sel"]
                )

        player_name = st.selectbox(
            "Player", filtered_players["PLAYER_NAME"].tolist(),
            index=default_idx, key="player_sel"
        )
        player_row = filtered_players[filtered_players["PLAYER_NAME"] == player_name].iloc[0]
        player_id  = int(player_row["PLAYER_ID"])

        recency_window_choice = st.selectbox(
            "Recency window (for averages/trends)",
            ["Season", 5, 10, 15, 20],
            index=1,
            key="recent_sel",
        )

        view_mode = st.radio(
            "Player view mode",
            ["Overview", "Game Logs", "Per-36 & Windows", "Opponent & vs Team", "Advanced Projections"],
            index=0,
        )

    with st.spinner("Fetching player logs & info..."):
        logs = get_player_logs(player_id, season)
        if logs.empty:
            st.error("No game logs for this player/season.")
            st.stop()
        career_df = get_player_career(player_id)
        cpi = get_common_player_info(player_id)

    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"{player_name} — {season}")
        team_name_disp = (
            cpi["TEAM_NAME"].iloc[0]
            if ("TEAM_NAME" in cpi.columns and not cpi.empty)
            else player_row.get("TEAM_NAME","Unknown")
        )
        pos = (cpi["POSITION"].iloc[0] if ("POSITION" in cpi.columns and not cpi.empty) else "N/A")
        exp = (cpi["SEASON_EXP"].iloc[0] if ("SEASON_EXP" in cpi.columns and not cpi.empty) else "N/A")
        gp = len(logs)
        st.caption(f"**Team:** {team_name_disp} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")
    with right:
        opponent = st.selectbox("Opponent (for matchup context)", team_list, index=0, key="opponent_sel")

    opp_row = team_ctx.loc[team_ctx["TEAM_NAME"] == opponent].iloc[0]
    opp_record = format_record(opp_row.get("W", np.nan), opp_row.get("L", np.nan))

    # Opponent summary tiles (always visible at top)
    st.markdown(f"### Opponent: **{opponent}** ({opp_record})")
    st.caption(f"Opponent metrics last updated: {fetched_at} • Season-to-date through (ET): {cutoff_used}")

    c1, c2, c3 = st.columns(3)
    _rank_tile(c1, "DEF Rating", opp_row.get("DEF_RATING", np.nan), opp_row.get("DEF_RANK", np.nan), total=30)
    _rank_tile(c2, "PACE",       opp_row.get("PACE", np.nan),       opp_row.get("PACE_RANK", np.nan), total=30)
    _rank_tile(c3, "NET Rating", opp_row.get("NET_RATING", np.nan), opp_row.get("NET_RANK", np.nan),  total=30)

    # Ensure basic columns exist
    for col in ["MIN","PTS","REB","AST","FG3M"]:
        if col not in logs.columns:
            logs[col] = 0

    # -----------------------------------------------------------------
    # VIEW: OVERVIEW (season + recency stats, last 5 box scores)
    # -----------------------------------------------------------------
    if view_mode == "Overview":
        # Recent averages based on recency_window_choice
        window_df = logs if recency_window_choice == "Season" else logs.head(int(recency_window_choice))
        recent_avg = window_df[["MIN","PTS","REB","AST","FG3M"]].mean(numeric_only=True)

        st.markdown("### Recent Averages")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MIN", _fmt1(recent_avg.get("MIN", np.nan)))
        m2.metric("PTS", _fmt1(recent_avg.get("PTS", np.nan)))
        m3.metric("REB", _fmt1(recent_avg.get("REB", np.nan)))
        m4.metric("AST", _fmt1(recent_avg.get("AST", np.nan)))
        m5.metric("3PM", _fmt1(recent_avg.get("FG3M", np.nan)))

        # Season averages
        st.markdown("### Season Averages (Per Game)")
        season_pg = logs[["MIN","PTS","REB","AST","FG3M"]].mean(numeric_only=True).to_frame("Season Avg").T
        st.dataframe(
            season_pg.round(2),
            use_container_width=True,
            height=_auto_height(season_pg),
        )

        # Last 5 games table
        st.markdown("### Last 5 Games (Box Scores)")
        cols_base = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","FGM","FGA",
                     "FG3M","FG3A","FTM","FTA","OREB","DREB"]
        last5 = logs[cols_base].head(5).copy()
        last5 = add_shot_breakouts(last5)
        last5 = append_average_row(last5, label="Average (Last 5)")
        num_fmt = {c: "{:.1f}" for c in last5.select_dtypes(include=[np.number]).columns if c != "GAME_DATE"}
        st.dataframe(
            last5.style.format(num_fmt),
            use_container_width=True,
            height=_auto_height(last5),
        )

    # -----------------------------------------------------------------
    # VIEW: GAME LOGS (with trend charts)
    # -----------------------------------------------------------------
    elif view_mode == "Game Logs":
        if "PRA" not in logs.columns:
            logs["PRA"] = logs.get("PTS", 0) + logs.get("REB", 0) + logs.get("AST", 0)

        n_recent_val = recency_window_choice
        trend_df = logs.copy()
        if n_recent_val != "Season":
            trend_df = trend_df.head(int(n_recent_val))
        trend_df = trend_df.sort_values("GAME_DATE")

        st.markdown(f"### Game Logs — {n_recent_val if n_recent_val != 'Season' else 'Season'} Window")
        cols_base = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","FG3M"]
        existing_cols = [c for c in cols_base if c in logs.columns]
        logs_view = logs[existing_cols].copy()
        st.dataframe(
            logs_view,
            use_container_width=True,
            height=_auto_height(logs_view, max_px=700),
        )

        # Trends
        st.markdown("### Trends")
        if "GAME_DATE" in trend_df.columns and len(trend_df) > 0:
            trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in trend_df.columns]
            for s in trend_cols:
                chart = (
                    alt.Chart(trend_df)
                    .mark_line(point=True)
                    .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
                    .properties(height=160)
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No trend data available to chart.")

    # -----------------------------------------------------------------
    # VIEW: PER-36 & WINDOWS (career, prev, current, recency, vs opp)
    # -----------------------------------------------------------------
    elif view_mode == "Per-36 & Windows":
        st.markdown("### Per-36 — Career / Season / Recency / Opponent")

        # Build needed windows once
        y0 = int(season.split("-")[0])
        prev_season_label = f"{y0-1}-{str(y0 % 100).zfill(2)}"
        prev_logs = get_player_logs(player_id, prev_season_label)

        for c in ["MIN","PTS","REB","AST","FGM","FG3M","FTM","OREB","DREB"]:
            if c not in logs.columns: logs[c] = 0
            if c not in prev_logs.columns: prev_logs[c] = 0

        # vs opponent last 5
        opp_team_id = resolve_team_id(opponent, opp_row)
        vs_opp_df = pd.DataFrame()
        if opp_team_id:
            vs_opp_df = get_vs_opponent_games(player_id, opp_team_id)

        if vs_opp_df.empty:
            opp_abbrev = resolve_team_abbrev(opponent, opp_row)
            if "SEASON" in career_df.columns and not career_df.empty:
                season_labels = list(career_df["SEASON"].dropna().unique())
                def _yr(s):
                    try: return int(s.split("-")[0])
                    except: return -1
                season_labels = sorted(season_labels, key=_yr, reverse=True)
            else:
                season_labels = SEASONS
            if opp_abbrev:
                all_logs = get_all_player_logs_all_seasons(player_id, season_labels)
                if not all_logs.empty and "MATCHUP" in all_logs.columns:
                    all_logs = all_logs.copy()
                    all_logs["OPP_ABBR"] = all_logs["MATCHUP"].apply(parse_opp_from_matchup)
                    all_logs["OPP_ABBR"] = all_logs["OPP_ABBR"].apply(
                        lambda x: ABBR_ALIAS.get(x, x) if isinstance(x, str) else x
                    )
                    cols_base = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST",
                                 "FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
                    vs_opp_df = all_logs[all_logs["OPP_ABBR"] == resolve_team_abbrev(opponent, opp_row)][cols_base].copy()

        per36_rows = []
        per36_rows.append(_series_from_dict(per36_from_career_totals(career_df), "Career Per 36"))
        per36_rows.append(_series_from_dict(compute_per36_from_logs(prev_logs),  "Prev Season Per 36"))
        per36_rows.append(_series_from_dict(compute_per36_from_logs(logs),       "Current Season Per 36"))
        per36_rows.append(_series_from_dict(compute_per36_from_logs(logs.head(5)),   "Last 5 Per 36"))
        per36_rows.append(_series_from_dict(compute_per36_from_logs(logs.head(10)),  "Last 10 Per 36"))
        per36_rows.append(_series_from_dict(compute_per36_from_logs(logs.head(15)),  "Last 15 Per 36"))
        per36_rows.append(_series_from_dict(compute_per36_from_logs(logs.head(20)),  "Last 20 Per 36"))
        if not vs_opp_df.empty:
            per36_rows.append(
                _series_from_dict(compute_per36_from_logs(vs_opp_df.head(5)),
                                  f"Last 5 vs {opponent} Per 36")
            )
        else:
            per36_rows.append(
                _series_from_dict({k: np.nan for k in ["PTS","REB","AST","PRA",
                                                       "FG2M","FG3M","FTM","OREB","DREB"]},
                                  f"Last 5 vs {opponent} Per 36")
            )

        per36_df = pd.DataFrame(per36_rows)[
            ["PTS","REB","AST","PRA","FG2M","FG3M","FTM","OREB","DREB"]
        ].round(2)
        st.dataframe(
            per36_df.style.format(numeric_format_map(per36_df)),
            use_container_width=True,
            height=_auto_height(per36_df),
        )

    # -----------------------------------------------------------------
    # VIEW: OPPONENT & VS TEAM (allowed stats + last 5 vs opp)
    # -----------------------------------------------------------------
    elif view_mode == "Opponent & vs Team":
        # Opponent allowed per-game
        st.markdown("#### Opponent Averages Allowed (Per-Game)")

        def _opp_row(cols_labels):
            cols = st.columns(len(cols_labels))
            for (api_col, label), col in zip(cols_labels, cols):
                val = opp_row.get(api_col, np.nan)
                rnk = opp_row.get(f"{api_col}_RANK", np.nan)
                pct = api_col.endswith("_PCT")
                _rank_tile(col, label, val, rnk, total=30, pct=pct, decimals=(1 if pct else 1))

        _opp_row([
            ("OPP_PTS",     "Opp PTS"),
            ("OPP_FGA",     "Opp FGA"),
            ("OPP_FG_PCT",  "Opp FG%"),
            ("OPP_FG3A",    "Opp 3PA"),
            ("OPP_FG3_PCT", "Opp 3P%"),
        ])
        _opp_row([
            ("OPP_FTA",     "Opp FTA"),
            ("OPP_FT_PCT",  "Opp FT%"),
            ("OPP_OREB",    "Opp OREB"),
            ("OPP_DREB",    "Opp DREB"),
            ("OPP_REB",     "Opp REB"),
        ])
        _opp_row([
            ("OPP_AST", "Opp AST"),
            ("OPP_TOV", "Opp TOV"),
            ("OPP_STL", "Opp STL"),
            ("OPP_BLK", "Opp BLK"),
            ("OPP_PF",  "Opp PF"),
        ])

        # Last 5 vs opponent
        st.markdown(f"### Last 5 Games vs {opponent}")
        cols_base = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST",
                     "FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]

        opp_team_id = resolve_team_id(opponent, opp_row)
        vs_opp_df = pd.DataFrame()
        if opp_team_id:
            vs_opp_df = get_vs_opponent_games(player_id, opp_team_id)

        if vs_opp_df.empty:
            st.info(f"No historical games vs {opponent}.")
        else:
            vs_opp5 = add_shot_breakouts(
                vs_opp_df.sort_values("GAME_DATE", ascending=False).head(5).copy()
            )
            vs_opp5 = append_average_row(vs_opp5, label="Average (Last 5 vs Opp)")
            num_fmt2 = {c: "{:.1f}" for c in vs_opp5.select_dtypes(include=[np.number]).columns if c != "GAME_DATE"}
            st.dataframe(
                vs_opp5.style.format(num_fmt2),
                use_container_width=True,
                height=_auto_height(vs_opp5),
            )

        st.caption("Opponent metrics: NBA-only ‘Regular Season’ through today’s ET date (5-min cache).")

    # -----------------------------------------------------------------
    # VIEW: ADVANCED PROJECTIONS (same model, wrapped in expanders)
    # -----------------------------------------------------------------
    elif view_mode == "Advanced Projections":
        st.markdown("### Player Projection — Per-36, Opponent-Adjusted (tunable)")
        st.caption("Advanced model view. Use only if you want to tweak projection parameters.")

        # Build per-36 buckets for projection
        p36_career = pd.Series(per36_from_career_totals(career_df), dtype="float64")
        p36_prev   = pd.Series(compute_per36_from_logs(get_player_logs(player_id, _prev_season_label(season))), dtype="float64")
        p36_season = pd.Series(compute_per36_from_logs(logs), dtype="float64")
        p36_recent_5  = pd.Series(compute_per36_from_logs(logs.head(5)), dtype="float64")
        p36_recent_10 = pd.Series(compute_per36_from_logs(logs.head(10)), dtype="float64")

        # Last 5 vs opponent per-36
        opp_team_id = resolve_team_id(opponent, opp_row)
        vs_opp_df = get_vs_opponent_games(player_id, opp_team_id) if opp_team_id else pd.DataFrame()
        p36_vsopp_L5 = pd.Series(compute_per36_from_logs(vs_opp_df.head(5))) if not vs_opp_df.empty else pd.Series(
            {"PTS":np.nan,"REB":np.nan,"AST":np.nan,"PRA":np.nan,"FG2M":np.nan,"FG3M":np.nan,"FTM":np.nan,"OREB":np.nan,"DREB":np.nan}
        )

        # Controls
        colA, colB, colC, colD = st.columns(4)
        with colA:
            exp_min = st.number_input(
                "Expected MIN",
                min_value=12.0, max_value=44.0,
                value=float(logs["MIN"].mean()) if not logs["MIN"].isna().all() else 32.0,
                step=1.0,
            )
        with colB:
            recent_window = st.selectbox(
                "Recent window",
                options=["Season", 5, 10, 15, 20],
                index=2,
            )
        with colC:
            k_vol = st.slider("Volume sensitivity (k_vol)", 0.0, 1.5, 0.8, 0.05)
        with colD:
            k_eff = st.slider("Efficiency sensitivity (k_eff)", 0.0, 1.0, 0.3, 0.05)

        colE, colF = st.columns(2)
        with colE:
            cap_vol = st.slider("Volume cap (±)", 0.05, 0.35, 0.25, 0.01)
        with colF:
            cap_eff = st.slider("Efficiency cap (±)", 0.02, 0.20, 0.12, 0.01)

        st.caption("Set blend weights (auto-normalized). Vs Opp is fixed to Last-5 vs opponent.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w_career = st.slider("Career", 0.0, 1.0, 0.10, 0.01)
        with c2:
            w_prev   = st.slider("Prev Season", 0.0, 1.0, 0.20, 0.01)
        with c3:
            w_season = st.slider("Current Season", 0.0, 1.0, 0.40, 0.01)
        with c4:
            w_recent = st.slider("Recent", 0.0, 1.0, 0.30, 0.01)
        w_vsopp = 0.10  # fixed

        # pick recent p36
        p36_recent = {
            "Season": p36_season,
            5: p36_recent_5,
            10: p36_recent_10,
            15: pd.Series(compute_per36_from_logs(logs.head(15))),
            20: pd.Series(compute_per36_from_logs(logs.head(20))),
        }[recent_window]

        # normalize weights (include vsopp in total)
        base_sum = max(1e-9, (w_career + w_prev + w_season + w_recent))
        scale_other = max(0.0, 1.0 - w_vsopp)
        w = {
            "career": (w_career / base_sum) * scale_other,
            "prev":   (w_prev   / base_sum) * scale_other,
            "season": (w_season / base_sum) * scale_other,
            "recent": (w_recent / base_sum) * scale_other,
            "vsopp":  w_vsopp,
        }

        opp_cols = [c for c in team_ctx.columns if c.startswith("OPP_")]
        lg_avgs = team_ctx[opp_cols].mean(numeric_only=True)

        def _mult(col, sens, cap):
            ov = opp_row.get(col, np.nan); la = lg_avgs.get(col, np.nan)
            if pd.isna(ov) or pd.isna(la) or la == 0:
                return 1.0
            raw = (float(ov) - float(la)) / float(la)
            adj = float(np.clip(raw * sens, -cap, cap))
            return 1.0 + adj

        lg_pace = float(team_ctx["PACE"].mean(numeric_only=True))
        pace_factor = float(np.clip(((lg_pace + float(opp_row.get("PACE", lg_pace))) / 2) /
                                    max(1e-6, lg_pace), 0.90, 1.10))

        def _season_pct(m_col, a_col):
            if m_col not in logs.columns or a_col not in logs.columns: return np.nan
            m = pd.to_numeric(logs[m_col], errors="coerce").sum()
            a = pd.to_numeric(logs[a_col], errors="coerce").sum()
            return (m / a) if a else np.nan

        fgpct_base = _season_pct("FGM","FGA");   fgpct_base = fgpct_base if pd.notna(fgpct_base) else 0.46
        fg3pct_base= _season_pct("FG3M","FG3A"); fg3pct_base= fg3pct_base if pd.notna(fg3pct_base) else 0.36
        ftpct_base = _season_pct("FTM","FTA");   ftpct_base = ftpct_base if pd.notna(ftpct_base) else 0.78

        m_FGA = _mult("OPP_FGA",  k_vol, cap_vol)
        m_3PA = _mult("OPP_FG3A", k_vol, cap_vol)
        m_FTA = _mult("OPP_FTA",  k_vol, cap_vol)
        m_FGP = _mult("OPP_FG_PCT",  k_eff, cap_eff)
        m_3PP = _mult("OPP_FG3_PCT", k_eff, cap_eff)
        m_FTP = _mult("OPP_FT_PCT",  k_eff, cap_eff)
        m_OREB= _mult("OPP_OREB", k_vol, min(cap_vol, 0.25))
        m_DREB= _mult("OPP_DREB", k_vol, min(cap_vol, 0.25))
        m_AST = _mult("OPP_AST",  k_vol, min(cap_vol, 0.25))
        m_TOV = _mult("OPP_TOV",  k_vol, min(cap_vol, 0.25))

        p36_blend = (
            w["career"] * p36_career.fillna(0) +
            w["prev"]   * p36_prev.fillna(0)   +
            w["season"] * p36_season.fillna(0) +
            w["recent"] * p36_recent.fillna(0) +
            w["vsopp"]  * p36_vsopp_L5.fillna(0)
        ).astype(float)

        fgpct_adj  = float(np.clip(fgpct_base  * m_FGP,  0.38, 0.70))
        fg3pct_adj = float(np.clip(fg3pct_base * m_3PP,  0.25, 0.55))
        ftpct_adj  = float(np.clip(ftpct_base  * m_FTP,  0.60, 0.95))

        fg3m36 = float(p36_blend.get("FG3M", 0))
        fg2m36 = float(p36_blend.get("FG2M", 0))
        ftm36  = float(p36_blend.get("FTM",  0))

        fg3a36_base = fg3m36 / max(1e-6, fg3pct_base)
        fg2a36_base = fg2m36 / max(1e-6, max(0.40, fgpct_base - 0.5*(fg3pct_base-0.35)))
        fga36_base  = fg2a36_base + fg3a36_base
        fta36_base  = ftm36 / max(1e-6, ftpct_base)

        fga36 = fga36_base * m_FGA * pace_factor
        fg3a36= fg3a36_base * m_3PA * pace_factor
        fta36 = fta36_base * m_FTA * pace_factor
        if fga36 > 0:
            fg3a36 = float(min(fga36*0.90, fg3a36))
        fg2a36 = max(0.0, fga36 - fg3a36)

        scale = float(exp_min) / 36.0
        FGA = max(0.0, fga36 * scale)
        FG3A= max(0.0, fg3a36 * scale)
        FG2A= max(0.0, fg2a36 * scale)
        FTA = max(0.0, fta36 * scale)

        FG3M = FG3A * fg3pct_adj
        FG2M = FG2A * np.clip(fgpct_adj * (1 - 0.5*(fg3pct_adj-0.35)), 0.35, 0.75)
        FGM  = FG2M + FG3M
        PTS  = 2*FG2M + 3*FG3M + FTA * ftpct_adj

        OREB = max(0.0, float(p36_blend.get("OREB", 0)) * m_OREB * pace_factor * scale)
        DREB = max(0.0, float(p36_blend.get("DREB", 0)) * m_DREB * pace_factor * scale)
        REB  = OREB + DREB
        AST  = max(0.0, float(p36_blend.get("AST", 0))  * m_AST  * pace_factor * scale)
        TOV  = max(0.0, 0.9 * AST * (m_TOV / max(1e-6, m_AST)))

        proj = {
            "MIN":exp_min,"PTS":PTS,"REB":REB,"AST":AST,"FGM":FGM,"FGA":FGA,
            "3PM":FG3M,"3PA":FG3A,"FTM":FTA*ftpct_adj,"FTA":FTA,
            "OREB":OREB,"DREB":DREB,"TOV":TOV
        }
        st.dataframe(pd.DataFrame([proj]).round(2), use_container_width=True)

        with st.expander("Diagnostics"):
            di = {
                "pace": pace_factor, "m_FGA": m_FGA, "m_3PA": m_3PA, "m_FTA": m_FTA,
                "FG% adj": fgpct_adj, "3P% adj": fg3pct_adj, "FT% adj": ftpct_adj
            }
            st.write(pd.DataFrame([di]).round(3))

# =====================================================================
# TEAM DASHBOARD
# =====================================================================
def team_dashboard():
    inject_rank_tile_css()
    st.title("NBA Team Dashboard")

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
    def get_teams_df():
        t = pd.DataFrame(static_teams.get_teams())
        t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME",
                              "abbreviation": "TEAM_ABBREVIATION"})
        t["TEAM_ID"] = t["TEAM_ID"].astype(int)
        return t[["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION"]]

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_team_traditional(season: str) -> pd.DataFrame:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                measure_type_detailed_defense="Base",
                per_mode_detailed="PerGame",
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        for c in df.columns:
            if c not in ("TEAM_NAME","TEAM_ABBREVIATION"):
                df[c] = pd.to_numeric(df[c], errors="ignore")
        return df.reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_team_advanced(season: str) -> pd.DataFrame:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        for c in df.columns:
            if c not in ("TEAM_NAME","TEAM_ABBREVIATION"):
                df[c] = pd.to_numeric(df[c], errors="ignore")
        return df.reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_team_opponent(season: str) -> pd.DataFrame:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                measure_type_detailed_defense="Opponent",
                per_mode_detailed="PerGame",
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        cols = ["TEAM_ID"] + [c for c in df.columns if c.startswith("OPP_")]
        for c in cols:
            if c != "TEAM_ID" and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[cols].reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_players_pg(season: str, last_n_games: int) -> pd.DataFrame:
        frames = _retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed="PerGame",
                last_n_games=last_n_games,
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_players_per36(season: str, last_n_games: int) -> pd.DataFrame:
        frames = _retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed="Per36",
                last_n_games=last_n_games,
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    def _rank_series(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan]*len(df))
        return df[col].rank(ascending=ascending, method="min")

    def _select_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
        colmap = {
            "TEAM_ABBREVIATION": "TEAM",
            "PLAYER_NAME": "PLAYER_NAME",
            "AGE": "AGE",
            "GP": "GP",
            "MIN": "MIN",
            "PTS": "PTS",
            "REB": "REB",
            "AST": "AST",
            "FGM": "FGM",
            "FGA": "FGA",
            "FG3M": "FG3M",
            "FG3A": "FG3A",
            "FTM": "FTM",
            "FTA": "FTA",
            "OREB": "OREB",
            "DREB": "DREB",
            "STL": "STL",
            "BLK": "BLK",
            "TOV": "TOV",
            "PF": "PF",
            "PLUS_MINUS": "PLUS_MINUS",
        }
        for c in colmap.keys():
            if c not in df.columns:
                df[c] = np.nan
        out = df[list(colmap.keys())].copy()
        out.columns = list(colmap.values())
        return out

    def _auto_height_local(df: pd.DataFrame, row_px=34, header_px=38, max_px=900):
        rows = max(len(df), 1)
        return min(max_px, header_px + row_px * rows + 8)

    def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    # Sidebar filters
    with st.sidebar:
        st.header("Team Filters")
        season = st.selectbox("Season (Team Tab)", _season_labels(2015, dt.datetime.utcnow().year),
                              index=0, key="team_season_sel")
        teams_df = get_teams_df()
        team_name = st.selectbox("Team", sorted(teams_df["TEAM_NAME"].tolist()))
        team_row = teams_df[teams_df["TEAM_NAME"] == team_name].iloc[0]
        team_id = int(team_row["TEAM_ID"])
        team_abbr = team_row["TEAM_ABBREVIATION"]
        roster_view = st.radio("Roster stats view", ["Per-Game", "Per-36"], index=0)

    with st.spinner("Loading league team stats..."):
        trad = fetch_league_team_traditional(season)
        adv = fetch_league_team_advanced(season)
        opp = fetch_league_team_opponent(season)

    if trad.empty or adv.empty:
        st.error("Could not load team stats. Try refreshing or changing the season.")
        st.stop()

    TRAD_WANTED = [
        "TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","GP","W","L","W_PCT",
        "MIN","PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","STL","BLK","TOV","PF","PLUS_MINUS"
    ]
    ADV_WANTED = ["TEAM_ID","OFF_RATING","DEF_RATING","NET_RATING","PACE"]

    trad_g = _ensure_cols(trad, TRAD_WANTED)[TRAD_WANTED].copy()
    adv_g  = _ensure_cols(adv,  ADV_WANTED)[ADV_WANTED].copy()

    merged = pd.merge(trad_g, adv_g, on="TEAM_ID", how="left")
    if not opp.empty:
        merged = pd.merge(merged, opp, on="TEAM_ID", how="left")

    # League ranks
    def _safe_rank(col, ascending):
        return _rank_series(merged, col, ascending=ascending)

    ranks = pd.DataFrame({"TEAM_ID": merged["TEAM_ID"]})
    ranks["PTS"]         = _safe_rank("PTS", ascending=False)
    ranks["NET_RATING"]  = _safe_rank("NET_RATING", ascending=False)
    ranks["OFF_RATING"]  = _safe_rank("OFF_RATING", ascending=False)
    ranks["DEF_RATING"]  = _safe_rank("DEF_RATING", ascending=True)
    ranks["PACE"]        = _safe_rank("PACE", ascending=False)
    ranks["FGA"]         = _safe_rank("FGA", ascending=False)
    ranks["FG_PCT"]      = _safe_rank("FG_PCT", ascending=False)
    ranks["FG3A"]        = _safe_rank("FG3A", ascending=False)
    ranks["FG3_PCT"]     = _safe_rank("FG3_PCT", ascending=False)
    ranks["FTA"]         = _safe_rank("FTA", ascending=False)
    ranks["FT_PCT"]      = _safe_rank("FT_PCT", ascending=False)
    ranks["OREB"]        = _safe_rank("OREB", ascending=False)
    ranks["DREB"]        = _safe_rank("DREB", ascending=False)
    ranks["REB"]         = _safe_rank("REB", ascending=False)
    ranks["AST"]         = _safe_rank("AST", ascending=False)
    ranks["TOV"]         = _safe_rank("TOV", ascending=True)
    ranks["STL"]         = _safe_rank("STL", ascending=False)
    ranks["BLK"]         = _safe_rank("BLK", ascending=False)
    ranks["PF"]          = _safe_rank("PF",  ascending=True)
    ranks["PLUS_MINUS"]  = _safe_rank("PLUS_MINUS", ascending=False)

    # Opponent ranks (team tab)
    def _add_opp_rank_team(col, ascending=True):
        if col in merged.columns:
            merged[f"{col}_RANK"] = merged[col].rank(ascending=ascending, method="min")

    for col, asc in [
        ("OPP_PTS", True),
        ("OPP_FG_PCT", True),
        ("OPP_FG3_PCT", True),
        ("OPP_FT_PCT", True),
        ("OPP_REB", True),
        ("OPP_OREB", True),
        ("OPP_DREB", True),
        ("OPP_AST", True),
        ("OPP_TOV", False),
        ("OPP_STL", False),
        ("OPP_BLK", False),
        ("OPP_PF", False),
        ("OPP_FGA", True),
        ("OPP_FG3A", True),
        ("OPP_FTA", True),
    ]:
        _add_opp_rank_team(col, ascending=asc)

    n_teams = len(merged)

    sel = merged[merged["TEAM_ID"] == team_id]
    if sel.empty:
        st.error("Selected team not found in this season dataset.")
        st.stop()

    tr = sel.iloc[0]
    rr = ranks[ranks["TEAM_ID"] == team_id].iloc[0]
    record = (
        f"{int(tr['W'])}–{int(tr['L'])}"
        if pd.notna(tr.get("W")) and pd.notna(tr.get("L"))
        else "—"
    )

    # Header
    st.subheader(f"{tr['TEAM_NAME']} — {season}")
    c_rec, _, _, _, _ = st.columns(5)
    c_rec.metric("Record", record)

    # Tiles in fixed order
    def tile_row(items):
        cols = st.columns(len(items))
        for (label, key, pct_flag), col in zip(items, cols):
            _rank_tile(col, label, tr.get(key), rr.get(key if key != "NET_RATING" else "NET_RATING"),
                       total=n_teams, pct=pct_flag)

    st.markdown("### Team Snapshot (League Ranks)")
    tile_row([
        ("PTS", "PTS", False),
        ("NET Rating", "NET_RATING", False),
        ("OFF Rating", "OFF_RATING", False),
        ("DEF Rating", "DEF_RATING", False),
        ("PACE", "PACE", False),
    ])
    tile_row([
        ("FGA", "FGA", False),
        ("FG%", "FG_PCT", True),
        ("3PA", "FG3A", False),
        ("3P%", "FG3_PCT", True),
        ("FTA", "FTA", False),
    ])
    tile_row([
        ("FT%", "FT_PCT", True),
        ("OREB", "OREB", False),
        ("DREB", "DREB", False),
        ("REB", "REB", False),
        ("AST", "AST", False),
    ])
    tile_row([
        ("TOV", "TOV", False),
        ("STL", "STL", False),
        ("BLK", "BLK", False),
        ("PF",  "PF",  False),
        ("+/-", "PLUS_MINUS", False),
    ])

    st.caption("Ranks are relative to all NBA teams (1 = best). Tile color and arrow reflect tier (top/middle/bottom).")

    # Opponent averages allowed
    st.markdown("### Opponent Averages Allowed (Per-Game)")

    def _opp_row_team(cols_labels):
        cols = st.columns(len(cols_labels))
        for (api_col, label), col in zip(cols_labels, cols):
            val = tr.get(api_col, np.nan)
            rank = tr.get(f"{api_col}_RANK", np.nan)
            pct = api_col.endswith("_PCT")
            _rank_tile(col, label, val, rank, total=n_teams, pct=pct)

    _opp_row_team([
        ("OPP_PTS",     "Opp PTS"),
        ("OPP_FGA",     "Opp FGA"),
        ("OPP_FG_PCT",  "Opp FG%"),
        ("OPP_FG3A",    "Opp 3PA"),
        ("OPP_FG3_PCT", "Opp 3P%"),
    ])
    _opp_row_team([
        ("OPP_FTA",     "Opp FTA"),
        ("OPP_FT_PCT",  "Opp FT%"),
        ("OPP_OREB",    "Opp OREB"),
        ("OPP_DREB",    "Opp DREB"),
        ("OPP_REB",     "Opp REB"),
    ])
    _opp_row_team([
        ("OPP_AST", "Opp AST"),
        ("OPP_TOV", "Opp TOV"),
        ("OPP_STL", "Opp STL"),
        ("OPP_BLK", "Opp BLK"),
        ("OPP_PF",  "Opp PF"),
    ])

    # Roster tables
    with st.spinner("Loading roster stats..."):
        season_pg = fetch_league_players_pg(season, last_n_games=0)
        last5_pg  = fetch_league_players_pg(season, last_n_games=5)
        last15_pg = fetch_league_players_pg(season, last_n_games=15)
        season_p36 = fetch_league_players_per36(season, last_n_games=0)
        last5_p36  = fetch_league_players_per36(season, last_n_games=5)
        last15_p36 = fetch_league_players_per36(season, last_n_games=15)

    def _prep_roster(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = df[df["TEAM_ID"] == team_id].copy()
        if out.empty:
            return out
        num_like = ["AGE","GP","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A",
                    "FTM","FTA","OREB","DREB","STL","BLK","TOV","PF","PLUS_MINUS"]
        for c in num_like:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = _select_roster_columns(out)
        if "MIN" in out.columns:
            out = out.sort_values("MIN", ascending=False).reset_index(drop=True)
        return out

    def _num_fmt_map(df: pd.DataFrame):
        fmts = {}
        for c in df.columns:
            if c in ("TEAM","PLAYER_NAME"):
                continue
            fmts[c] = "{:.1f}"
        return fmts

    st.markdown("### Roster")

    if roster_view == "Per-Game":
        season_tbl = _prep_roster(season_pg, team_id)
        last5_tbl  = _prep_roster(last5_pg, team_id)
        last15_tbl = _prep_roster(last15_pg, team_id)

        st.markdown("#### Season Per-Game")
        if season_tbl.empty:
            st.info("No season per-game data for this team.")
        else:
            st.dataframe(
                season_tbl.style.format(_num_fmt_map(season_tbl)),
                use_container_width=True,
                height=_auto_height_local(season_tbl),
            )

        st.markdown("#### Last 5 Games (Per-Game)")
        if last5_tbl.empty:
            st.info("No Last 5 per-game data for this team.")
        else:
            st.dataframe(
                last5_tbl.style.format(_num_fmt_map(last5_tbl)),
                use_container_width=True,
                height=_auto_height_local(last5_tbl),
            )

        st.markdown("#### Last 15 Games (Per-Game)")
        if last15_tbl.empty:
            st.info("No Last 15 per-game data for this team.")
        else:
            st.dataframe(
                last15_tbl.style.format(_num_fmt_map(last15_tbl)),
                use_container_width=True,
                height=_auto_height_local(last15_tbl),
            )

    else:  # Per-36
        season_p36_tbl = _prep_roster(season_p36, team_id)
        last5_p36_tbl  = _prep_roster(last5_p36, team_id)
        last15_p36_tbl = _prep_roster(last15_p36, team_id)

        st.markdown("#### Season Per-36")
        if season_p36_tbl.empty:
            st.info("No season Per-36 data for this team.")
        else:
            st.dataframe(
                season_p36_tbl.style.format(_num_fmt_map(season_p36_tbl)),
                use_container_width=True,
                height=_auto_height_local(season_p36_tbl),
            )

        st.markdown("#### Last 5 Games (Per-36)")
        if last5_p36_tbl.empty:
            st.info("No Last 5 Per-36 data for this team.")
        else:
            st.dataframe(
                last5_p36_tbl.style.format(_num_fmt_map(last5_p36_tbl)),
                use_container_width=True,
                height=_auto_height_local(last5_p36_tbl),
            )

        st.markdown("#### Last 15 Games (Per-36)")
        if last15_p36_tbl.empty:
            st.info("No Last 15 Per-36 data for this team.")
        else:
            st.dataframe(
                last15_p36_tbl.style.format(_num_fmt_map(last15_p36_tbl)),
                use_container_width=True,
                height=_auto_height_local(last15_p36_tbl),
            )

    st.caption(
        "Team stats from LeagueDashTeamStats (Traditional, Advanced, Opponent). "
        "Roster stats from LeagueDashPlayerStats (PerGame and Per36) with last_n_games filters. "
        "Use the roster view toggle in the sidebar to switch between Per-Game and Per-36."
    )

# =====================================================================
# MAIN APP TABS
# =====================================================================
def main():
    tab1, tab2, tab3 = st.tabs(["League", "Player", "Team"])
    with tab1:
        league_dashboard()
    with tab2:
        player_dashboard()
    with tab3:
        team_dashboard()

if __name__ == "__main__":
    main()
