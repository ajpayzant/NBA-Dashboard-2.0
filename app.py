# app.py — NBA League / Player / Team Dashboard (Streamlit)
# Data: nba_api (stats.nba.com)
# Goals:
# - League tab: real schedule + sortable standings/team table + leaguewide player leaderboards
# - Player tab: player-only focus, per-game/per-36, season vs last-N windows, interactive boxscore filtering
# - Team tab: tiles + opponent tiles + one interactive roster table (per-game/per-36, season vs last-N)
#
# Streamlit Cloud safe:
# - aggressive retry/backoff + jitter + global rate limiting
# - defensive column handling + dtype-safe merges
# - avoids slider min/max crash, avoids pandas ufunc string concat errors

import time
import random
import re
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    playergamelog,
    commonplayerinfo,
    leaguestandingsv3,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Dashboards", layout="wide")

# ----------------------- Config -----------------------
CACHE_HOURS = 8
TEAM_CTX_TTL_SECONDS = 300

# NBA stats can be slow; Streamlit Cloud can be slower.
REQUEST_TIMEOUT = 60
MAX_RETRIES = 6
BASE_SLEEP = 1.25

# Global rate limit: enforce minimum time between requests
MIN_SECONDS_BETWEEN_CALLS = 1.0

# ----------------------- Global request pacing -----------------------
if "last_api_call_ts" not in st.session_state:
    st.session_state["last_api_call_ts"] = 0.0


def _rate_limit_pause(min_gap: float = MIN_SECONDS_BETWEEN_CALLS) -> None:
    now = time.time()
    last = float(st.session_state.get("last_api_call_ts", 0.0))
    gap = now - last
    if gap < min_gap:
        time.sleep(min_gap - gap)
    st.session_state["last_api_call_ts"] = time.time()


# ----------------------- Robust API wrapper -----------------------
def _retry_api(endpoint_cls, kwargs: dict, timeout: int = REQUEST_TIMEOUT) -> List[pd.DataFrame]:
    """
    Robust wrapper around nba_api endpoint calls.
    - exponential backoff + jitter
    - rate limiting between calls
    - returns list of DataFrames (nba_api standard)
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _rate_limit_pause()
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            # exponential backoff with jitter; grow quickly to survive stats.nba.com throttling
            sleep_s = BASE_SLEEP * (2 ** (attempt - 1))
            sleep_s = min(40.0, sleep_s)  # cap
            sleep_s = sleep_s * (0.85 + 0.30 * random.random())
            time.sleep(sleep_s)
    raise last_err


# ----------------------- Helpers -----------------------
def _season_labels(start=2015, end=None) -> List[str]:
    if end is None:
        end = dt.datetime.now(dt.UTC).year
    def lab(y): return f"{y}-{str((y+1) % 100).zfill(2)}"
    return [lab(y) for y in range(end, start - 1, -1)]


SEASONS = _season_labels(2015, dt.datetime.now(dt.UTC).year)


def _auto_height(df: pd.DataFrame, row_px=34, header_px=38, max_px=900) -> int:
    rows = max(len(df), 1)
    return int(min(max_px, header_px + row_px * rows + 8))


def _fmt1(v) -> str:
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"


def _fmt2(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def numeric_format_map(df: pd.DataFrame, decimals=2) -> Dict[str, str]:
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: f"{{:.{decimals}f}}" for c in num_cols}


_punct_re = re.compile(r"[^\w]")
def parse_opp_from_matchup(matchup_str: str) -> Optional[str]:
    if not isinstance(matchup_str, str):
        return None
    parts = matchup_str.split()
    if len(parts) < 3:
        return None
    token = parts[-1].upper().strip()
    token = _punct_re.sub("", token)
    return token


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


def _rank_tile(col, label, value, rank, total=30, pct=False, decimals=1):
    if pd.isna(rank):
        tier_class, arrow, rank_txt = "rank-mid", "•", "Rank —"
    else:
        r = int(rank)
        if r <= max(1, total // 4):
            tier_class, arrow = "rank-good", "▲"
        elif r >= total - max(1, total // 4) + 1:
            tier_class, arrow = "rank-bad", "▼"
        else:
            tier_class, arrow = "rank-mid", "•"
        rank_txt = f"{arrow} Rank {r}/{total}"

    if pct:
        try:
            val_txt = f"{float(value)*100:.{decimals}f}%" if pd.notna(value) else "—"
        except Exception:
            val_txt = "—"
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


# ----------------------- Static team maps -----------------------
@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_teams_static() -> pd.DataFrame:
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = pd.to_numeric(t["TEAM_ID"], errors="coerce").astype("Int64")
    return t[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]].dropna().reset_index(drop=True)


# ----------------------- Data fetchers (cached) -----------------------
@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_league_players_index(season: str) -> pd.DataFrame:
    """
    Player list (PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, GP, MIN)
    for the selected season.
    """
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
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GP", "MIN"]
    df = _ensure_cols(df, keep)
    df = df[keep].copy()
    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["PLAYER_ID"]).drop_duplicates(subset=["PLAYER_ID"])
    df = df.sort_values(["TEAM_NAME", "PLAYER_NAME"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_player_logs(player_id: int, season: str) -> pd.DataFrame:
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
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_common_player_info(player_id: int) -> pd.DataFrame:
    try:
        frames = _retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": int(player_id)})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TEAM_CTX_TTL_SECONDS, show_spinner=False)
def get_daily_scoreboard(game_date: dt.date) -> pd.DataFrame:
    """
    Returns a schedule-like table for a specific date.
    Handles empty days gracefully.
    """
    date_str = game_date.strftime("%m/%d/%Y")
    try:
        frames = _retry_api(scoreboardv2.ScoreboardV2, {"game_date": date_str, "league_id": "00"})
    except Exception:
        return pd.DataFrame()

    if not frames or len(frames) < 2:
        return pd.DataFrame()

    game_header = frames[0].copy()  # GAME_HEADER
    line_score = frames[1].copy()   # LINE_SCORE

    if game_header.empty:
        return pd.DataFrame()

    # Defensive normalize
    need_h = ["GAME_ID", "GAME_DATE_EST", "GAME_STATUS_TEXT", "LIVE_PERIOD", "LIVE_PC_TIME"]
    game_header = _ensure_cols(game_header, need_h)

    need_l = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY_NAME", "TEAM_NICKNAME", "PTS"]
    line_score = _ensure_cols(line_score, need_l)

    # split home/away from line_score by GAME_ID
    line_score["PTS"] = pd.to_numeric(line_score["PTS"], errors="coerce")

    # game_header has HOME_TEAM_ID/VISITOR_TEAM_ID in some versions; if missing, infer from line_score order
    if "HOME_TEAM_ID" not in game_header.columns or "VISITOR_TEAM_ID" not in game_header.columns:
        # attempt to infer: take the two teams and label first as away, second as home (common in nba_api)
        pairs = (
            line_score.sort_values(["GAME_ID"])
            .groupby("GAME_ID")
            .head(2)
            .groupby("GAME_ID")["TEAM_ID"]
            .apply(list)
            .to_dict()
        )
        game_header["VISITOR_TEAM_ID"] = game_header["GAME_ID"].map(lambda gid: pairs.get(gid, [np.nan, np.nan])[0] if gid in pairs else np.nan)
        game_header["HOME_TEAM_ID"] = game_header["GAME_ID"].map(lambda gid: pairs.get(gid, [np.nan, np.nan])[1] if gid in pairs else np.nan)

    # Build home/away abbreviations and scores
    away = line_score.rename(
        columns={"TEAM_ID": "VISITOR_TEAM_ID", "TEAM_ABBREVIATION": "VISITOR_TEAM_ABBREVIATION", "PTS": "VISITOR_PTS"}
    )[["GAME_ID", "VISITOR_TEAM_ID", "VISITOR_TEAM_ABBREVIATION", "VISITOR_PTS"]]
    home = line_score.rename(
        columns={"TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION", "PTS": "HOME_PTS"}
    )[["GAME_ID", "HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION", "HOME_PTS"]]

    out = game_header[["GAME_ID", "GAME_STATUS_TEXT"]].copy()
    out = out.merge(away, on="GAME_ID", how="left")
    out = out.merge(home, on="GAME_ID", how="left")

    # Safe string concat (fixes numpy/pandas ufunc errors)
    out["VISITOR_TEAM_ABBREVIATION"] = out["VISITOR_TEAM_ABBREVIATION"].astype("string").fillna("")
    out["HOME_TEAM_ABBREVIATION"] = out["HOME_TEAM_ABBREVIATION"].astype("string").fillna("")
    out["MATCHUP"] = out["VISITOR_TEAM_ABBREVIATION"] + " @ " + out["HOME_TEAM_ABBREVIATION"]
    out["VISITOR_PTS"] = pd.to_numeric(out["VISITOR_PTS"], errors="coerce")
    out["HOME_PTS"] = pd.to_numeric(out["HOME_PTS"], errors="coerce")

    # Sort stable
    return out[["MATCHUP", "GAME_STATUS_TEXT", "VISITOR_PTS", "HOME_PTS"]].reset_index(drop=True)


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def fetch_league_standings(season: str) -> pd.DataFrame:
    """
    League standings (wins/losses/conference, etc).
    Uses leaguestandingsv3 which is the most complete for standings.
    """
    try:
        frames = _retry_api(
            leaguestandingsv3.LeagueStandingsV3,
            dict(
                season=season,
                season_type="Regular Season",
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # Common columns (varies by nba_api versions)
    wanted = [
        "TeamID", "TeamName", "TeamCity", "TeamAbbreviation",
        "Conference", "WINS", "LOSSES", "WinPCT",
        "ConferenceRank", "LeagueRank",
    ]
    df = _ensure_cols(df, wanted)

    out = pd.DataFrame({
        "TEAM_ID": pd.to_numeric(df["TeamID"], errors="coerce"),
        "TEAM_NAME": (df["TeamCity"].astype("string").fillna("") + " " + df["TeamName"].astype("string").fillna("")).str.strip(),
        "TEAM_ABBREVIATION": df["TeamAbbreviation"].astype("string"),
        "CONFERENCE": df["Conference"].astype("string"),
        "W": pd.to_numeric(df["WINS"], errors="coerce"),
        "L": pd.to_numeric(df["LOSSES"], errors="coerce"),
        "W_PCT": pd.to_numeric(df["WinPCT"], errors="coerce"),
        "CONF_RANK": pd.to_numeric(df["ConferenceRank"], errors="coerce"),
        "LEAGUE_RANK": pd.to_numeric(df["LeagueRank"], errors="coerce"),
    })
    out["TEAM_ID"] = out["TEAM_ID"].astype("Int64")
    out = out.dropna(subset=["TEAM_ID"]).drop_duplicates(subset=["TEAM_ID"]).reset_index(drop=True)
    return out


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def fetch_league_team_stats(season: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (base_per_game, advanced_per_game, opponent_per_game)
    """
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
            df = frames[0] if frames else pd.DataFrame()
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


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def fetch_league_players_stats(season: str, per_mode: str, last_n_games: int) -> pd.DataFrame:
    """
    LeagueDashPlayerStats for a season + per_mode + last_n_games.
    per_mode: "PerGame" or "Per36"
    last_n_games: 0 means season-to-date
    """
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
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "TEAM_ID" in df.columns:
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)


# ----------------------- League summary builder -----------------------
@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_league_team_summary(season: str) -> pd.DataFrame:
    """
    One sortable standings/team table:
    Rank, Team, W/L/W%, plus PTS/REB/AST, ratings, pace, etc.
    """
    standings = fetch_league_standings(season)
    base, adv, opp = fetch_league_team_stats(season)

    if standings.empty and base.empty and adv.empty:
        return pd.DataFrame()

    # Normalize base
    base_cols = [
        "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
        "PTS", "REB", "AST",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "TOV", "STL", "BLK",
        "PLUS_MINUS",
    ]
    base = _ensure_cols(base, base_cols)
    base = base[base_cols].copy()
    base["TEAM_ID"] = pd.to_numeric(base["TEAM_ID"], errors="coerce").astype("Int64")

    # Normalize advanced
    adv_cols = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    adv = _ensure_cols(adv, adv_cols)
    adv = adv[adv_cols].copy()
    adv["TEAM_ID"] = pd.to_numeric(adv["TEAM_ID"], errors="coerce").astype("Int64")

    # Merge by TEAM_ID only to avoid name-abbrev mismatches across endpoints
    merged = standings.copy()
    if not base.empty:
        merged = merged.merge(base.drop(columns=["TEAM_NAME", "TEAM_ABBREVIATION"], errors="ignore"), on="TEAM_ID", how="left")
    if not adv.empty:
        merged = merged.merge(adv, on="TEAM_ID", how="left")

    # Fill names/abbr from static if needed
    tstatic = get_teams_static()
    merged = merged.merge(tstatic, on="TEAM_ID", how="left", suffixes=("", "_STATIC"))
    merged["TEAM_NAME"] = merged["TEAM_NAME"].fillna(merged["TEAM_NAME_STATIC"])
    merged["TEAM_ABBREVIATION"] = merged["TEAM_ABBREVIATION"].fillna(merged["TEAM_ABBREVIATION_STATIC"])
    merged = merged.drop(columns=["TEAM_NAME_STATIC", "TEAM_ABBREVIATION_STATIC"], errors="ignore")

    # Compute league rank fallback if missing
    if "LEAGUE_RANK" not in merged.columns or merged["LEAGUE_RANK"].isna().all():
        # Rank by W_PCT then W
        merged["LEAGUE_RANK"] = merged[["W_PCT", "W"]].apply(tuple, axis=1).rank(ascending=False, method="min").astype("Int64")

    # Nice ordering
    order = [
        "LEAGUE_RANK", "CONF_RANK", "CONFERENCE", "TEAM_NAME", "TEAM_ABBREVIATION",
        "W", "L", "W_PCT",
        "PTS", "REB", "AST",
        "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "TOV", "STL", "BLK", "PLUS_MINUS",
    ]
    merged = _ensure_cols(merged, order)
    merged = _to_num(merged, [c for c in order if c not in ("CONFERENCE", "TEAM_NAME", "TEAM_ABBREVIATION")])
    merged = merged.sort_values(["LEAGUE_RANK", "TEAM_NAME"]).reset_index(drop=True)

    # Rename for UI
    ren = {
        "LEAGUE_RANK": "RANK",
        "CONF_RANK": "CONF_RANK",
        "CONFERENCE": "CONF",
        "TEAM_NAME": "TEAM",
        "TEAM_ABBREVIATION": "ABBR",
        "W_PCT": "W%",
        "NET_RATING": "NETRTG",
        "OFF_RATING": "OFFRTG",
        "DEF_RATING": "DEFRTG",
        "PLUS_MINUS": "+/-",
        "FG_PCT": "FG%",
        "FG3_PCT": "3P%",
        "FT_PCT": "FT%",
    }
    merged = merged.rename(columns=ren)

    # Round % columns for readability
    for c in ["W%", "FG%", "3P%", "FT%"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged


# ----------------------- League leaders builder -----------------------
def _leader_table(df: pd.DataFrame, stat: str, n: int = 15) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if stat not in df.columns:
        return pd.DataFrame()

    keep = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", stat]
    df = _ensure_cols(df, keep)
    out = df[keep].copy()
    out["GP"] = pd.to_numeric(out["GP"], errors="coerce")
    out["MIN"] = pd.to_numeric(out["MIN"], errors="coerce")
    out[stat] = pd.to_numeric(out[stat], errors="coerce")
    out = out.sort_values(stat, ascending=False).head(n).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    return out


# =====================================================================
# LEAGUE DASHBOARD
# =====================================================================
def league_dashboard():
    st.title("NBA League Dashboard")

    # Controls
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        season = st.selectbox("Season", SEASONS, index=0, key="lg_season")
    with c2:
        date_choice = st.date_input("Schedule date", value=dt.date.today(), key="lg_date")
    with c3:
        if st.button("Hard clear cache", key="lg_clear_cache"):
            st.cache_data.clear()
            st.success("Cache cleared.")

    tabA, tabB = st.tabs(["League Teams", "League Players"])

    with tabA:
        st.subheader("Schedule")
        with st.spinner("Loading scoreboard..."):
            sched = get_daily_scoreboard(date_choice)

        if sched.empty:
            st.info("No games found for this date (or the API returned an empty payload). Try another date.")
        else:
            st.dataframe(
                sched,
                width="stretch",
                height=_auto_height(sched, max_px=380),
            )

        st.divider()
        st.subheader("Standings + Team Stats (sortable)")

        conf = st.radio("Filter", ["All", "East", "West"], horizontal=True, key="lg_conf_filter")
        with st.spinner("Loading standings & team stats..."):
            team_summary = get_league_team_summary(season)

        if team_summary.empty:
            st.error("Could not load league team summary. Try clearing cache and refreshing.")
            return

        # Filter by conference
        if conf != "All" and "CONF" in team_summary.columns:
            team_summary = team_summary[team_summary["CONF"].astype("string").str.upper() == conf.upper()].copy()

        # Formatting
        fmt = numeric_format_map(team_summary, decimals=2)
        for pct_col in ["W%", "FG%", "3P%", "FT%"]:
            if pct_col in fmt:
                fmt[pct_col] = "{:.3f}"
        st.dataframe(
            team_summary.style.format(fmt),
            width="stretch",
            height=_auto_height(team_summary, max_px=820),
        )

    with tabB:
        st.subheader("Leaguewide Player Leaders")

        cL1, cL2, cL3 = st.columns([1, 1, 1])
        with cL1:
            per_mode = st.selectbox("Per mode", ["PerGame", "Per36"], index=0, key="lg_pl_permode")
        with cL2:
            window = st.selectbox("Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="lg_pl_window")
        with cL3:
            top_n = st.slider("Top N", 5, 30, 15, 1, key="lg_pl_topn")

        last_n = 0 if window == "Season" else int(window.split()[-1])

        with st.spinner("Loading player stats..."):
            pstats = fetch_league_players_stats(season, per_mode=per_mode, last_n_games=last_n)

        if pstats.empty:
            st.error("Could not load player stats for this view. Try clearing cache and retry.")
            return

        # Common leader categories
        leader_sets = [
            ("Scoring", ["PTS", "FG3M"]),
            ("Playmaking", ["AST", "TOV"]),
            ("Rebounding", ["REB", "OREB", "DREB"]),
            ("Defense", ["STL", "BLK"]),
        ]

        for section, stats in leader_sets:
            st.markdown(f"### {section}")
            cols = st.columns(len(stats))
            for s, col in zip(stats, cols):
                tbl = _leader_table(pstats, s, n=int(top_n))
                if tbl.empty:
                    col.info(f"No data for {s}.")
                else:
                    # readable rounding
                    num_cols = [c for c in tbl.columns if c not in ("PLAYER_NAME", "TEAM_ABBREVIATION")]
                    tbl[num_cols] = tbl[num_cols].apply(pd.to_numeric, errors="coerce")
                    col.dataframe(
                        tbl.style.format(numeric_format_map(tbl, decimals=2)),
                        width="stretch",
                        height=_auto_height(tbl, max_px=520),
                    )


# =====================================================================
# PLAYER DASHBOARD
# =====================================================================
def player_dashboard():
    st.title("NBA Player Dashboard")

    # Sidebar controls
    with st.sidebar:
        st.header("Player Filters")
        season = st.selectbox("Season", SEASONS, index=0, key="pl_season")

        with st.spinner("Loading players..."):
            pindex = get_league_players_index(season)

        if pindex.empty:
            st.error("Could not load player index for this season.")
            st.stop()

        q = st.text_input("Search player", key="pl_search").strip()
        filt = pindex if not q else pindex[pindex["PLAYER_NAME"].astype("string").str.contains(q, case=False, na=False)]

        if filt.empty:
            st.info("No players match your search.")
            st.stop()

        player_name = st.selectbox("Player", filt["PLAYER_NAME"].tolist(), index=0, key="pl_player")
        prow = filt[filt["PLAYER_NAME"] == player_name].iloc[0]
        player_id = int(prow["PLAYER_ID"])

        per_mode = st.selectbox("View", ["PerGame", "Per36"], index=0, key="pl_permode")
        window = st.selectbox("Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="pl_window")

        if st.button("Hard clear cache", key="pl_clear_cache"):
            st.cache_data.clear()
            st.success("Cache cleared.")

    with st.spinner("Fetching player logs..."):
        logs = get_player_logs(player_id, season)
        cpi = get_common_player_info(player_id)

    if logs.empty:
        st.error("No game logs available for this player/season.")
        st.stop()

    # Player header
    team_name_disp = (
        cpi["TEAM_NAME"].iloc[0]
        if (cpi is not None and not cpi.empty and "TEAM_NAME" in cpi.columns)
        else str(prow.get("TEAM_NAME", ""))
    )
    pos = (
        cpi["POSITION"].iloc[0]
        if (cpi is not None and not cpi.empty and "POSITION" in cpi.columns)
        else "N/A"
    )
    st.subheader(f"{player_name} — {season}")
    st.caption(f"Team: {team_name_disp} | Position: {pos} | Games in log: {len(logs)}")

    # Choose window_df
    if window == "Season":
        window_df = logs.copy()
    else:
        n = int(window.split()[-1])
        window_df = logs.head(n).copy()

    # Build derived PRA and FG2 columns
    for c in ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "STL", "BLK", "TOV", "PF"]:
        if c not in window_df.columns:
            window_df[c] = 0
        window_df[c] = pd.to_numeric(window_df[c], errors="coerce").fillna(0)

    window_df["PRA"] = window_df["PTS"] + window_df["REB"] + window_df["AST"]
    window_df["FG2M"] = (window_df["FGM"] - window_df["FG3M"]).clip(lower=0)
    window_df["FG2A"] = (window_df["FGA"] - window_df["FG3A"]).clip(lower=0)

    # Averages (per-game or per-36)
    sums = window_df[["MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "FTM", "OREB", "DREB", "STL", "BLK", "TOV"]].sum(numeric_only=True)
    games_n = max(1, len(window_df))
    if per_mode == "PerGame":
        av = sums / games_n
    else:
        # Per-36: scale using minutes
        total_min = float(sums.get("MIN", 0.0))
        if total_min <= 0:
            av = sums * np.nan
        else:
            av = (sums / total_min) * 36.0

    st.markdown("### Averages")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("MIN", _fmt1(av.get("MIN", np.nan)) if per_mode == "PerGame" else "—")
    a2.metric("PTS", _fmt1(av.get("PTS", np.nan)))
    a3.metric("REB", _fmt1(av.get("REB", np.nan)))
    a4.metric("AST", _fmt1(av.get("AST", np.nan)))
    a5.metric("PRA", _fmt1(av.get("PRA", np.nan)))
    st.caption("Per-36 view computes totals first then scales by total minutes in the selected window.")

    # ----------------------- Box scores (interactive) -----------------------
    st.markdown("### Box Scores (interactive)")

    # Opponent filter based on MATCHUP parsing (safe)
    logs2 = logs.copy()
    if "MATCHUP" in logs2.columns:
        logs2["OPP_ABBR"] = logs2["MATCHUP"].apply(parse_opp_from_matchup)
    else:
        logs2["OPP_ABBR"] = None

    # Date range filter (safe bounds)
    min_date = logs2["GAME_DATE"].min()
    max_date = logs2["GAME_DATE"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        min_date = dt.datetime.now(dt.UTC) - dt.timedelta(days=365)
        max_date = dt.datetime.now(dt.UTC)

    f1, f2, f3 = st.columns([1.1, 1.1, 1])
    with f1:
        dr = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="pl_date_range",
        )
    with f2:
        opps = sorted([x for x in logs2["OPP_ABBR"].dropna().unique().tolist() if isinstance(x, str) and len(x) in (3, 4)])
        opp_choice = st.selectbox("Opponent (optional)", ["All"] + opps, index=0, key="pl_opp_choice")
    with f3:
        # avoid slider min=max crash
        max_games = int(len(window_df))
        if max_games <= 1:
            show_n = max_games
            st.caption(f"Only {max_games} game available.")
        else:
            min_val = 1 if max_games < 5 else 5
            default_val = min(10, max_games)
            if default_val < min_val:
                default_val = max_games
            show_n = st.slider("Max games to display", min_value=min_val, max_value=max_games, value=default_val, key="pl_show_n")

    # Apply filters to full logs (not just window)
    start_d, end_d = dr
    flt = logs2[(logs2["GAME_DATE"].dt.date >= start_d) & (logs2["GAME_DATE"].dt.date <= end_d)].copy()
    if opp_choice != "All":
        flt = flt[flt["OPP_ABBR"] == opp_choice].copy()

    # Display
    show = flt.head(show_n).copy()
    keep_cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG2A", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "STL", "BLK", "TOV", "PF"]
    show = _ensure_cols(show, keep_cols)[keep_cols]
    for c in keep_cols:
        if c != "GAME_DATE" and c != "MATCHUP" and c != "WL":
            show[c] = pd.to_numeric(show[c], errors="coerce")

    st.dataframe(
        show.style.format({c: "{:.1f}" for c in show.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(show),
    )

    # Trend charts (selected window)
    st.markdown("### Trends (selected window)")
    trend_cols = ["MIN", "PTS", "REB", "AST", "PRA"]
    tdf = window_df[["GAME_DATE"] + trend_cols].copy().sort_values("GAME_DATE")
    for s in trend_cols:
        chart = (
            alt.Chart(tdf)
            .mark_line(point=True)
            .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
            .properties(height=160)
        )
        st.altair_chart(chart, width="stretch")


# =====================================================================
# TEAM DASHBOARD
# =====================================================================
def team_dashboard():
    inject_rank_tile_css()
    st.title("NBA Team Dashboard")

    with st.sidebar:
        st.header("Team Filters")
        season = st.selectbox("Season (Team Tab)", SEASONS, index=0, key="tm_season")

        tdf = get_teams_static()
        team_name = st.selectbox("Team", sorted(tdf["TEAM_NAME"].tolist()), key="tm_team")
        team_row = tdf[tdf["TEAM_NAME"] == team_name].iloc[0]
        team_id = int(team_row["TEAM_ID"])
        team_abbr = str(team_row["TEAM_ABBREVIATION"])

        roster_per_mode = st.selectbox("Roster view", ["PerGame", "Per36"], index=0, key="tm_roster_permode")
        roster_window = st.selectbox("Roster window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="tm_roster_window")

        if st.button("Hard clear cache", key="tm_clear_cache"):
            st.cache_data.clear()
            st.success("Cache cleared.")

    with st.spinner("Loading team stats..."):
        base, adv, opp = fetch_league_team_stats(season)

    if base.empty or adv.empty:
        st.error("Could not load team stats. Try clearing cache.")
        return

    # Merge base+adv+opp by TEAM_ID
    base_cols = [
        "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
        "GP", "W", "L", "W_PCT",
        "MIN", "PTS", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "PLUS_MINUS",
    ]
    adv_cols = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]

    base = _ensure_cols(base, base_cols)[base_cols].copy()
    adv = _ensure_cols(adv, adv_cols)[adv_cols].copy()
    opp_cols = ["TEAM_ID"] + [c for c in opp.columns if c.startswith("OPP_")] if not opp.empty else ["TEAM_ID"]
    opp = _ensure_cols(opp, opp_cols)[opp_cols].copy()

    base["TEAM_ID"] = pd.to_numeric(base["TEAM_ID"], errors="coerce").astype("Int64")
    adv["TEAM_ID"] = pd.to_numeric(adv["TEAM_ID"], errors="coerce").astype("Int64")
    opp["TEAM_ID"] = pd.to_numeric(opp["TEAM_ID"], errors="coerce").astype("Int64")

    merged = base.merge(adv, on="TEAM_ID", how="left").merge(opp, on="TEAM_ID", how="left")

    # League ranks for tiles
    def _rank(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce").rank(ascending=ascending, method="min")

    ranks = {
        "PTS": _rank(merged, "PTS", ascending=False),
        "NET_RATING": _rank(merged, "NET_RATING", ascending=False),
        "OFF_RATING": _rank(merged, "OFF_RATING", ascending=False),
        "DEF_RATING": _rank(merged, "DEF_RATING", ascending=True),
        "PACE": _rank(merged, "PACE", ascending=False),
        "FGA": _rank(merged, "FGA", ascending=False),
        "FG_PCT": _rank(merged, "FG_PCT", ascending=False),
        "FG3A": _rank(merged, "FG3A", ascending=False),
        "FG3_PCT": _rank(merged, "FG3_PCT", ascending=False),
        "FTA": _rank(merged, "FTA", ascending=False),
        "FT_PCT": _rank(merged, "FT_PCT", ascending=False),
        "OREB": _rank(merged, "OREB", ascending=False),
        "DREB": _rank(merged, "DREB", ascending=False),
        "REB": _rank(merged, "REB", ascending=False),
        "AST": _rank(merged, "AST", ascending=False),
        "TOV": _rank(merged, "TOV", ascending=True),
        "STL": _rank(merged, "STL", ascending=False),
        "BLK": _rank(merged, "BLK", ascending=False),
        "PF": _rank(merged, "PF", ascending=True),
        "PLUS_MINUS": _rank(merged, "PLUS_MINUS", ascending=False),
    }
    n_teams = int(len(merged))

    row = merged[merged["TEAM_ID"] == team_id]
    if row.empty:
        st.error("Team not found in this season dataset.")
        return
    tr = row.iloc[0]

    # Header
    record = "—"
    if pd.notna(tr.get("W")) and pd.notna(tr.get("L")):
        record = f"{int(tr['W'])}–{int(tr['L'])}"

    st.subheader(f"{tr['TEAM_NAME']} — {season}")
    st.caption(f"Record: {record}")

    # Tile helper
    def tile_row(items):
        cols = st.columns(len(items))
        for (label, key, pct_flag), col in zip(items, cols):
            r = ranks.get(key, pd.Series([np.nan] * len(merged))).loc[row.index[0]] if key in ranks else np.nan
            _rank_tile(col, label, tr.get(key, np.nan), r, total=n_teams, pct=pct_flag)

    # Tiles in your preferred order
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
        ("PF", "PF", False),
        ("+/-", "PLUS_MINUS", False),
    ])

    # Opponent allowed tiles (rank across league where lower allowed = better for most)
    st.markdown("### Opponent Averages Allowed (Per-Game)")
    if opp.empty:
        st.info("Opponent stats were not available from the API for this season.")
    else:
        # Compute ranks for opponent columns (lower is better for allowed)
        for col in [c for c in merged.columns if c.startswith("OPP_")]:
            asc = True
            # For turnovers/steals/blocks allowed, interpret higher allowed as worse, so still ascending=True.
            merged[f"{col}_RANK"] = pd.to_numeric(merged[col], errors="coerce").rank(ascending=True, method="min")

        def opp_row(cols_labels):
            cols = st.columns(len(cols_labels))
            for (api_col, label), col in zip(cols_labels, cols):
                val = tr.get(api_col, np.nan)
                rnk = tr.get(f"{api_col}_RANK", np.nan)
                pct = api_col.endswith("_PCT")
                _rank_tile(col, label, val, rnk, total=n_teams, pct=pct, decimals=1)

        opp_row([
            ("OPP_PTS", "Opp PTS"),
            ("OPP_FGA", "Opp FGA"),
            ("OPP_FG_PCT", "Opp FG%"),
            ("OPP_FG3A", "Opp 3PA"),
            ("OPP_FG3_PCT", "Opp 3P%"),
        ])
        opp_row([
            ("OPP_FTA", "Opp FTA"),
            ("OPP_FT_PCT", "Opp FT%"),
            ("OPP_OREB", "Opp OREB"),
            ("OPP_DREB", "Opp DREB"),
            ("OPP_REB", "Opp REB"),
        ])
        opp_row([
            ("OPP_AST", "Opp AST"),
            ("OPP_TOV", "Opp TOV"),
            ("OPP_STL", "Opp STL"),
            ("OPP_BLK", "Opp BLK"),
            ("OPP_PF", "Opp PF"),
        ])

    # ----------------------- One interactive roster table -----------------------
    st.markdown("### Roster (interactive)")

    last_n = 0 if roster_window == "Season" else int(roster_window.split()[-1])
    with st.spinner("Loading roster stats..."):
        roster = fetch_league_players_stats(season, per_mode=roster_per_mode, last_n_games=last_n)

    if roster.empty:
        st.info("No roster data returned for this view.")
        return

    # Filter to team
    roster = roster[roster["TEAM_ID"] == team_id].copy()
    if roster.empty:
        st.info("No players returned for this team in this view.")
        return

    # Select readable columns (defensive)
    wanted = [
        "PLAYER_NAME", "AGE", "GP", "MIN",
        "PTS", "REB", "AST",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "STL", "BLK", "TOV", "PF",
        "PLUS_MINUS",
    ]
    roster = _ensure_cols(roster, wanted)[wanted].copy()
    for c in wanted:
        if c not in ("PLAYER_NAME",):
            roster[c] = pd.to_numeric(roster[c], errors="coerce")

    roster = roster.sort_values("MIN", ascending=False).reset_index(drop=True)

    st.dataframe(
        roster.style.format({c: "{:.1f}" for c in roster.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(roster),
    )


# =====================================================================
# MAIN
# =====================================================================
def main():
    tab0, tab1, tab2 = st.tabs(["League", "Player", "Team"])
    with tab0:
        league_dashboard()
    with tab1:
        player_dashboard()
    with tab2:
        team_dashboard()


if __name__ == "__main__":
    main()
