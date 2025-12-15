# app.py — NBA League / Team / Player Dashboard (Streamlit)
# Updated: Team Box Scores auto-cache to disk the first time (no manual preload step).

import time
import random
import re
import datetime as dt
from typing import List, Optional, Tuple

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
    teamgamelog,
    boxscoretraditionalv2,
    playercareerstats,
)

# NEW: disk cache read/write for team boxscores
from src.cache_store import read_team_boxscores, write_team_boxscores

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Dashboard", layout="wide")

# ----------------------- Config -----------------------
CACHE_HOURS = 8
TEAM_CTX_TTL_SECONDS = 300

REQUEST_TIMEOUT = 60
MAX_RETRIES = 6
BASE_SLEEP = 1.25

MIN_SECONDS_BETWEEN_CALLS = 1.0
BOX_SCORE_REFRESH_TTL_SECONDS = 24 * 3600  # refresh disk cache at most daily

if "last_api_call_ts" not in st.session_state:
    st.session_state["last_api_call_ts"] = 0.0


def _rate_limit_pause(min_gap: float = MIN_SECONDS_BETWEEN_CALLS) -> None:
    now = time.time()
    last = float(st.session_state.get("last_api_call_ts", 0.0))
    gap = now - last
    if gap < min_gap:
        time.sleep(min_gap - gap)
    st.session_state["last_api_call_ts"] = time.time()


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


# ----------------------- Helpers -----------------------
def _season_labels(start=2015, end=None) -> List[str]:
    if end is None:
        end = dt.datetime.now(dt.UTC).year

    def lab(y):
        return f"{y}-{str((y + 1) % 100).zfill(2)}"

    return [lab(y) for y in range(end, start - 1, -1)]


SEASONS = _season_labels(2015, dt.datetime.now(dt.UTC).year)


def _auto_height(df: pd.DataFrame, row_px=34, header_px=38, max_px=900) -> int:
    rows = max(len(df), 1)
    return int(min(max_px, header_px + row_px * rows + 8))


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


def numeric_format_map(df: pd.DataFrame, decimals=2):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: f"{{:.{decimals}f}}" for c in num_cols}


def _fmt1(v) -> str:
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"


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
        .rank-tile .value { font-weight: 750; font-size: 1.35rem; line-height: 1.2; }
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
            val_txt = f"{float(value) * 100:.{decimals}f}%" if pd.notna(value) else "—"
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
        unsafe_allow_html=True,
    )


# ----------------------- Static team maps -----------------------
@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_teams_static() -> pd.DataFrame:
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = pd.to_numeric(t["TEAM_ID"], errors="coerce").astype("Int64")
    return t[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]].dropna().reset_index(drop=True)


# ----------------------- Fetchers -----------------------
@st.cache_data(ttl=TEAM_CTX_TTL_SECONDS, show_spinner=False)
def get_team_gamelog(team_id: int, season: str) -> pd.DataFrame:
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
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    if df is not None and not df.empty and "GAME_ID" in df.columns:
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    return pd.DataFrame()


def _parse_min_to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) + float(ss) / 60.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _fetch_single_game_boxscore(game_id: str) -> pd.DataFrame:
    frames = _retry_api(boxscoretraditionalv2.BoxScoreTraditionalV2, {"game_id": str(game_id)})
    if not frames or len(frames) < 2:
        return pd.DataFrame()
    players = frames[0].copy()
    if players is None or players.empty:
        return pd.DataFrame()
    players["GAME_ID"] = str(game_id)
    return players


def ensure_team_boxscores_cached(team_id: int, team_abbr: str, season: str, team_gamelog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Disk-backed cache:
      - Reads existing team-season boxscores from disk if present
      - Fetches ONLY missing games for this team-season
      - Writes merged result to disk
    """
    disk = read_team_boxscores(team_id, season)
    disk_game_ids = set(disk["GAME_ID"].astype(str).unique()) if (disk is not None and not disk.empty and "GAME_ID" in disk.columns) else set()

    glog = team_gamelog_df.copy()
    if glog.empty or "GAME_ID" not in glog.columns:
        return disk if disk is not None else pd.DataFrame()

    glog["GAME_ID"] = glog["GAME_ID"].astype(str)
    wanted_game_ids = glog["GAME_ID"].dropna().astype(str).unique().tolist()

    missing = [gid for gid in wanted_game_ids if gid not in disk_game_ids]
    if not missing:
        return disk

    # Fetch missing in a controlled way. This will run once, then persist.
    pulled = []
    for gid in missing:
        try:
            df = _fetch_single_game_boxscore(gid)
            if df is None or df.empty:
                continue
            pulled.append(df)
        except Exception:
            continue

    if not pulled:
        # Return whatever we had (may be empty)
        return disk if disk is not None else pd.DataFrame()

    new_df = pd.concat(pulled, ignore_index=True)
    # Filter to just this team (players rows contain TEAM_ABBREVIATION)
    if "TEAM_ABBREVIATION" in new_df.columns:
        new_df = new_df[new_df["TEAM_ABBREVIATION"].astype(str) == str(team_abbr)].copy()

    # Merge with existing disk
    if disk is None or disk.empty:
        merged = new_df
    else:
        merged = pd.concat([disk, new_df], ignore_index=True)

    # Basic normalization
    if "PLAYER_ID" in merged.columns:
        merged["PLAYER_ID"] = pd.to_numeric(merged["PLAYER_ID"], errors="coerce").astype("Int64")
    if "MIN" in merged.columns:
        # keep original MIN string and a numeric if needed downstream
        pass

    # De-dupe: player per game unique
    dedupe_cols = [c for c in ["GAME_ID", "PLAYER_ID"] if c in merged.columns]
    if len(dedupe_cols) == 2:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)

    write_team_boxscores(team_id, season, merged)
    return merged


# -----------------------------------------------------------------------------
# NOTE:
# The rest of your app (League tab, Player tab, other Team sections)
# should remain as you already had it working. I’m only including the
# Team tab implementation below to keep this answer focused and safe.
# -----------------------------------------------------------------------------

def team_tab():
    inject_rank_tile_css()
    st.title("NBA Team")

    t1, t2 = st.columns([1.2, 2.0])
    with t1:
        season = st.selectbox("Season", SEASONS, index=0, key="tm_season")
    with t2:
        tdf = get_teams_static()
        team_name = st.selectbox("Team", sorted(tdf["TEAM_NAME"].tolist()), key="tm_team")
        team_row = tdf[tdf["TEAM_NAME"] == team_name].iloc[0]
        team_id = int(team_row["TEAM_ID"])
        team_abbr = str(team_row["TEAM_ABBREVIATION"])

    with st.spinner("Loading team game log..."):
        glog = get_team_gamelog(team_id, season)

    st.subheader("Team Box Scores (Pick a Game)")

    if glog is None or glog.empty or "GAME_ID" not in glog.columns:
        st.info("No game log available for box score selection.")
        return

    glog2 = glog.copy()
    glog2["GAME_DATE"] = pd.to_datetime(glog2["GAME_DATE"], errors="coerce")
    glog2["GAME_DATE_STR"] = glog2["GAME_DATE"].dt.strftime("%Y-%m-%d")
    glog2["LABEL"] = (
        glog2["GAME_DATE_STR"].astype("string")
        + " | "
        + glog2["MATCHUP"].astype("string")
        + " | "
        + glog2.get("WL", "").astype("string")
    )

    gsel = st.selectbox("Select game", glog2["LABEL"].tolist(), index=0, key="tm_game_select")
    game_id = str(glog2[glog2["LABEL"] == gsel]["GAME_ID"].iloc[0])

    # ✅ NEW: Automatically build disk cache (fetch missing games once)
    with st.spinner("Preparing team boxscores cache (one-time per team/season)..."):
        box_all = ensure_team_boxscores_cached(team_id, team_abbr, season, glog2)

    if box_all is None or box_all.empty:
        st.error("Could not load box scores for this team yet. Try again in a moment or switch teams.")
        return

    bx = box_all[box_all["GAME_ID"].astype(str) == str(game_id)].copy()
    if bx.empty:
        st.info("No box score stored for this game yet (it may not have been fetched). Select another game.")
        return

    keep = [
        "PLAYER_NAME",
        "START_POSITION",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "STL",
        "BLK",
        "TO",
        "PF",
        "PLUS_MINUS",
    ]
    bx = _ensure_cols(bx, keep)[keep].copy()
    bx = bx.rename(columns={"TO": "TOV"})

    for c in bx.columns:
        if c not in ("PLAYER_NAME", "START_POSITION", "MIN"):
            bx[c] = pd.to_numeric(bx[c], errors="coerce")

    bx["MIN"] = bx["MIN"].apply(_parse_min_to_float)
    bx["PRA"] = bx["PTS"].fillna(0) + bx["REB"].fillna(0) + bx["AST"].fillna(0)
    bx["FG2M"] = (bx["FGM"].fillna(0) - bx["FG3M"].fillna(0)).clip(lower=0)

    show_cols = [
        "PLAYER_NAME",
        "START_POSITION",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "PRA",
        "FG2M",
        "FG3M",
        "FTM",
        "OREB",
        "DREB",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PLUS_MINUS",
    ]
    bx = _ensure_cols(bx, show_cols)[show_cols].copy()

    st.dataframe(
        bx.style.format({c: "{:.1f}" for c in bx.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(bx, max_px=760),
    )


def main():
    # Keep your existing tab layout. If your current app has League/Team/Player,
    # call those functions here as before. This minimal main keeps the example runnable.
    tab0, tab1 = st.tabs(["Team", "About"])
    with tab0:
        team_tab()
    with tab1:
        st.write("NBA Dashboard")


if __name__ == "__main__":
    main()
