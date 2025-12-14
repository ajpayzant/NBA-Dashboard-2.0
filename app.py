# app.py
# NBA Dashboard (Streamlit) with Disk Parquet Cache Layer
# - Uses src/loaders.py + src/data_store.py + src/nba_client.py
# - Tabs: League (Team + Player Leaders), Team, Player
# - No sidebar: controls live in-page (top header + section controls)

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Disk-cached loaders (from the new files you added) ----------
from src.loaders import (
    get_daily_scoreboard,
    get_league_team_summary,
    get_league_players,
    get_team_gamelog,
    get_game_boxscore_traditional,
)

# ---------- Use same cache layer for player-level endpoints ----------
from src.data_store import CacheSpec, read_or_refresh_parquet, get_cache_dir
from src.nba_client import retry_api

from nba_api.stats.static import players as static_players
from nba_api.stats.endpoints import (
    commonplayerinfo,
    playergamelog,
    playercareerstats,
)

# ============================================================
# App config
# ============================================================
st.set_page_config(
    page_title="NBA Dashboard",
    page_icon="🏀",
    layout="wide",
)

# ============================================================
# Helpers
# ============================================================
def season_labels(start: int = 2015, end: Optional[int] = None) -> List[str]:
    if end is None:
        end = dt.datetime.utcnow().year
    def lab(y: int) -> str:
        return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start - 1, -1)]


SEASONS = season_labels(2015, dt.datetime.utcnow().year)


def safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt_no_decimals(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def to_per36(df: pd.DataFrame, min_col: str = "MIN", cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert counting stats to per-36 based on MIN.
    Leaves percentage columns unchanged.
    """
    if df is None or df.empty or min_col not in df.columns:
        return df

    out = df.copy()
    out[min_col] = pd.to_numeric(out[min_col], errors="coerce")
    mins = out[min_col].replace(0, np.nan)

    if cols is None:
        # Default set of common counting stats if present
        candidates = [
            "PTS", "REB", "AST", "STL", "BLK", "TOV",
            "OREB", "DREB",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "PF",
        ]
        cols = [c for c in candidates if c in out.columns]

    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = (out[c] / mins) * 36.0

    return out


def add_derived_player_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds PRA, FG2M, FG2A where possible.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    for c in ["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        out["PRA"] = out["PTS"].fillna(0) + out["REB"].fillna(0) + out["AST"].fillna(0)

    if all(c in out.columns for c in ["FGM", "FG3M"]):
        out["FG2M"] = out["FGM"].fillna(0) - out["FG3M"].fillna(0)

    if all(c in out.columns for c in ["FGA", "FG3A"]):
        out["FG2A"] = out["FGA"].fillna(0) - out["FG3A"].fillna(0)

    return out


def dataframe_stretch(df: pd.DataFrame):
    # Streamlit deprecates use_container_width. Use width='stretch'.
    st.dataframe(df, width="stretch", hide_index=True)


# ============================================================
# Disk-cached player endpoints (new, but uses the same cache system)
# ============================================================
TTL_PLAYER_INFO = 60 * 60 * 24 * 14    # 14 days
TTL_PLAYER_GAMELOG = 60 * 60 * 12      # 12 hours
TTL_PLAYER_CAREER = 60 * 60 * 24 * 30  # 30 days


def get_player_info(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    spec = CacheSpec(key=f"player_info_{player_id}", ttl_seconds=TTL_PLAYER_INFO)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": player_id})
        df = frames[0] if frames else pd.DataFrame()
        return df.reset_index(drop=True) if df is not None else pd.DataFrame()

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


def get_player_gamelog(player_id: int, season: str, force_refresh: bool = False) -> pd.DataFrame:
    spec = CacheSpec(key=f"player_gamelog_{season}_{player_id}", ttl_seconds=TTL_PLAYER_GAMELOG)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(
            playergamelog.PlayerGameLog,
            {"player_id": player_id, "season": season, "season_type_all_star": "Regular Season"},
        )
        df = frames[0] if frames else pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()

        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


def get_player_career(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    spec = CacheSpec(key=f"player_career_{player_id}", ttl_seconds=TTL_PLAYER_CAREER)

    def _fetch() -> pd.DataFrame:
        frames = retry_api(playercareerstats.PlayerCareerStats, {"player_id": player_id})
        # frame[0] usually SeasonTotalsRegularSeason
        df = frames[0] if frames else pd.DataFrame()
        return df.reset_index(drop=True) if df is not None else pd.DataFrame()

    return read_or_refresh_parquet(spec, _fetch, force_refresh=force_refresh)


# ============================================================
# Header bar (no sidebar)
# ============================================================
def header_bar() -> Dict[str, object]:
    with st.container():
        c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.0, 1.0], vertical_alignment="bottom")

        with c1:
            st.markdown("## 🏀 NBA Dashboard")

        with c2:
            season = st.selectbox("Season", SEASONS, index=0)

        with c3:
            today = dt.date.today()
            date_choice = st.date_input("Scoreboard Date", value=today)

        with c4:
            force_refresh = st.toggle("Force refresh (slow)", value=False)

        # Optional: manual nuke-cache button
        colA, colB = st.columns([1, 6])
        with colA:
            if st.button("Clear parquet cache"):
                import shutil
                shutil.rmtree(get_cache_dir(), ignore_errors=True)
                st.success("Cache cleared. Reloading…")
                st.rerun()

    return {
        "season": season,
        "date_choice": date_choice,
        "force_refresh": force_refresh,
    }


# ============================================================
# League tab
# ============================================================
def league_tab(season: str, date_choice: dt.date, force_refresh: bool):
    subtab_team, subtab_players = st.tabs(["League: Teams", "League: Player Leaders"])

    # ---------------------------
    # Teams subtab
    # ---------------------------
    with subtab_team:
        st.markdown("### Schedule / Scoreboard")

        date_str = pd.to_datetime(date_choice).strftime("%Y-%m-%d")
        sched = get_daily_scoreboard(date_str, force_refresh=force_refresh)

        if sched is None or sched.empty:
            st.info("No games found for this date.")
        else:
            # Ensure one row per matchup: this is already enforced in loaders.get_daily_scoreboard
            view = sched[["MATCHUP", "STATUS"]].copy()
            dataframe_stretch(view)

        st.divider()

        st.markdown("### Standings + Team Stats (sortable)")

        team_table, opp_table = get_league_team_summary(season, force_refresh=force_refresh)

        if team_table is None or team_table.empty:
            st.warning("Standings/team stats are not available right now.")
        else:
            # Remove ID/extra columns if present
            drop_cols = [c for c in ["TEAM_ID", "CONF_RANK"] if c in team_table.columns]
            out = team_table.drop(columns=drop_cols, errors="ignore").copy()

            # Ensure W/L are integers, W_PCT sortable
            if "W" in out.columns:
                out["W"] = pd.to_numeric(out["W"], errors="coerce").fillna(0).astype(int)
            if "L" in out.columns:
                out["L"] = pd.to_numeric(out["L"], errors="coerce").fillna(0).astype(int)
            if "W_PCT" in out.columns:
                out["W_PCT"] = pd.to_numeric(out["W_PCT"], errors="coerce")

            # Default sort: win% desc
            if "W_PCT" in out.columns:
                out = out.sort_values("W_PCT", ascending=False)

            # Conference filter
            conf = st.radio("Filter", ["League", "East", "West"], horizontal=True)
            if conf != "League" and "CONF" in out.columns:
                out = out[out["CONF"].astype(str).str.upper().str.contains(conf.upper())]

            # Rename for readability
            rename = {"TEAM_ABBREVIATION": "Team", "TEAM_NAME": "Team Name", "W_PCT": "Win%"}
            out = out.rename(columns=rename)

            # Keep reasonable columns order
            preferred = [
                "Team", "Team Name", "CONF", "W", "L", "Win%",
                "PTS", "REB", "AST",
                "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
            ]
            cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
            out = out[cols].copy()

            dataframe_stretch(out)

        st.divider()

        st.markdown("### Opponent Allowed Metrics (sortable)")

        if opp_table is None or opp_table.empty:
            st.info("Opponent table unavailable right now.")
        else:
            out2 = opp_table.copy()
            # Normalize W/L to int for display if present
            for c in ["W", "L"]:
                if c in out2.columns:
                    out2[c] = pd.to_numeric(out2[c], errors="coerce").fillna(0).astype(int)

            # Same conference filter
            conf2 = st.radio("Opponent table filter", ["League", "East", "West"], horizontal=True, key="opp_conf")
            if conf2 != "League":
                # Need CONF; if missing, merge from team_table (which has CONF)
                if "CONF" not in out2.columns and team_table is not None and not team_table.empty:
                    out2 = out2.merge(team_table[["TEAM_ID", "CONF"]], on="TEAM_ID", how="left")
                if "CONF" in out2.columns:
                    out2 = out2[out2["CONF"].astype(str).str.upper().str.contains(conf2.upper())]

            # Rename header "TEAM_ABBREVIATION" to "Team"
            out2 = out2.rename(columns={"TEAM_ABBREVIATION": "Team", "TEAM_NAME": "Team Name"})
            # Prefer to show a compact set first
            pref2 = ["Team", "Team Name", "W", "L", "W_PCT"]
            opp_cols = [c for c in out2.columns if c.startswith("OPP_")]
            cols2 = [c for c in pref2 if c in out2.columns] + opp_cols
            out2 = out2[cols2].copy()

            dataframe_stretch(out2)

    # ---------------------------
    # Player leaders subtab
    # ---------------------------
    with subtab_players:
        st.markdown("### Leaguewide Player Leaders")

        c1, c2, c3 = st.columns([1.0, 1.0, 2.0], vertical_alignment="bottom")
        with c1:
            per_mode = st.selectbox("Rate", ["PerGame", "Per36"], index=0)
        with c2:
            last_n = st.selectbox("Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0)
            last_n_games = 0 if last_n == "Season" else int(last_n.split()[-1])
        with c3:
            min_gp = st.slider("Minimum games played", min_value=0, max_value=30, value=10, step=1)

        players_df = get_league_players(
            season=season,
            per_mode=per_mode,
            last_n_games=last_n_games,
            min_gp=min_gp,
            force_refresh=force_refresh,
        )

        if players_df is None or players_df.empty:
            st.warning("Player leader tables unavailable right now.")
            return

        # Rename TEAM_ABBREVIATION column header to just "Team" when present
        if "TEAM_ABBREVIATION" in players_df.columns:
            players_df = players_df.rename(columns={"TEAM_ABBREVIATION": "Team"})

        # Build leader tables for major categories (top N)
        leader_stats = [
            ("Points", "PTS"),
            ("Rebounds", "REB"),
            ("Assists", "AST"),
            ("3PT Made", "FG3M"),
            ("OREB", "OREB"),
            ("DREB", "DREB"),
            ("Steals", "STL"),
            ("Blocks", "BLK"),
            ("Turnovers", "TOV"),
        ]

        top_n = st.slider("Rows per table", min_value=5, max_value=50, value=15, step=5)

        # Compact columns shown
        base_cols = ["PLAYER_NAME", "Team", "GP", "MIN"]
        base_cols = [c for c in base_cols if c in players_df.columns]

        grid = st.columns(2)
        for i, (title, col) in enumerate(leader_stats):
            if col not in players_df.columns:
                continue
            df = players_df.copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values(col, ascending=False).head(top_n)

            show_cols = base_cols + [col]
            show_cols = list(dict.fromkeys(show_cols))  # unique keep order
            block = df[show_cols].copy()
            block = block.rename(columns={"PLAYER_NAME": "Player", col: title})

            with grid[i % 2]:
                st.markdown(f"#### {title}")
                dataframe_stretch(block)


# ============================================================
# Team tab
# ============================================================
def team_tab(season: str, force_refresh: bool):
    st.markdown("### Team Dashboard")

    # Team select (from standings table, so abbreviations are stable)
    team_table, _ = get_league_team_summary(season, force_refresh=force_refresh)
    if team_table is None or team_table.empty:
        st.warning("Team list unavailable right now.")
        return

    team_table = team_table.dropna(subset=["TEAM_ID", "TEAM_ABBREVIATION"]).copy()
    team_table["TEAM_ID"] = pd.to_numeric(team_table["TEAM_ID"], errors="coerce").astype("Int64")
    teams = team_table[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "W", "L", "W_PCT"]].copy()

    teams = teams.sort_values("TEAM_NAME")
    options = [f"{r.TEAM_NAME} ({r.TEAM_ABBREVIATION})" for _, r in teams.iterrows()]
    opt_to_id = {options[i]: int(teams.iloc[i]["TEAM_ID"]) for i in range(len(options))}
    opt_to_abbr = {options[i]: str(teams.iloc[i]["TEAM_ABBREVIATION"]) for i in range(len(options))}

    pick = st.selectbox("Team", options, index=0)
    team_id = opt_to_id[pick]
    team_abbr = opt_to_abbr[pick]

    # Record + last 10
    glog = get_team_gamelog(team_id, season, force_refresh=force_refresh)

    # overall
    rec_row = teams[teams["TEAM_ID"].astype(int) == team_id].head(1)
    w = safe_int(rec_row["W"].iloc[0]) if not rec_row.empty else 0
    l = safe_int(rec_row["L"].iloc[0]) if not rec_row.empty else 0

    # last 10
    last10_w, last10_l = 0, 0
    if glog is not None and not glog.empty and "WL" in glog.columns:
        last10 = glog.head(10)
        last10_w = int((last10["WL"] == "W").sum())
        last10_l = int((last10["WL"] == "L").sum())

    st.markdown(
        f"## {team_abbr} Record: **{w}-{l}**  |  Last 10: **{last10_w}-{last10_l}**"
    )

    st.divider()

    # Team Game Log + Box Score selector
    st.markdown("### Team Box Scores (pick a game)")

    if glog is None or glog.empty or "GAME_ID" not in glog.columns:
        st.info("No game log available for box score selection.")
        return

    # Let user pick a game from gamelog
    gl = glog.copy()
    if "GAME_DATE" in gl.columns:
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce")
        gl["DATE_STR"] = gl["GAME_DATE"].dt.strftime("%Y-%m-%d")
    else:
        gl["DATE_STR"] = ""

    if "MATCHUP" not in gl.columns:
        gl["MATCHUP"] = ""

    gl["LABEL"] = gl["DATE_STR"].fillna("") + " | " + gl["MATCHUP"].fillna("")
    gl = gl.sort_values("GAME_DATE", ascending=False)

    game_label = st.selectbox("Select game", gl["LABEL"].tolist(), index=0)
    game_id = gl.loc[gl["LABEL"] == game_label, "GAME_ID"].iloc[0]

    box = get_game_boxscore_traditional(str(game_id), force_refresh=force_refresh)
    if box is None or box.empty:
        st.warning("Box score unavailable for this game.")
        return

    # Filter just this team
    if "TEAM_ABBREVIATION" in box.columns:
        box_team = box[box["TEAM_ABBREVIATION"] == team_abbr].copy()
    else:
        box_team = box.copy()

    # Make a tidy view
    cols = ["PLAYER_NAME", "MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "STL", "BLK", "TOV", "PF", "+/-"]
    cols = [c for c in cols if c in box_team.columns]
    view = add_derived_player_cols(box_team[cols].copy())
    # Put derived columns near core stats if present
    if "PRA" in view.columns:
        base_order = ["PLAYER_NAME", "MIN", "PTS", "REB", "AST", "PRA"]
        rest = [c for c in view.columns if c not in base_order]
        view = view[base_order + rest]
    dataframe_stretch(view)

    st.divider()

    # Rotation/minutes section (last 3, last 10, season avg, plus last 5 games mins)
    st.markdown("### Minutes Rotation Snapshot")

    if box is None or box.empty or "PLAYER_NAME" not in box.columns or glog is None or glog.empty:
        st.info("Rotation snapshot unavailable.")
        return

    # Build from player game logs for this team by using boxscore minutes for each game in last N games.
    # We'll pull last 15 games and compile minutes by player.
    recent_games = gl.head(15)[["GAME_ID", "DATE_STR"]].copy()
    recent_ids = recent_games["GAME_ID"].astype(str).tolist()

    # Fetch cached boxscores for last games and build minutes matrix
    rows = []
    for gid in recent_ids:
        b = get_game_boxscore_traditional(gid, force_refresh=False)
        if b is None or b.empty:
            continue
        if "TEAM_ABBREVIATION" in b.columns:
            b = b[b["TEAM_ABBREVIATION"] == team_abbr].copy()
        if b.empty:
            continue
        if "MIN" not in b.columns:
            continue
        tmp = b[["PLAYER_NAME", "MIN"]].copy()
        tmp["MIN"] = tmp["MIN"].astype(str).str.split(":").str[0]  # naive parse "MM:SS" -> "MM"
        tmp["MIN"] = pd.to_numeric(tmp["MIN"], errors="coerce").fillna(0)
        tmp["GAME_ID"] = gid
        rows.append(tmp)

    if not rows:
        st.info("Rotation snapshot unavailable (could not compile recent box scores).")
        return

    mins_df = pd.concat(rows, ignore_index=True)
    # Attach dates
    gid_to_date = dict(zip(recent_games["GAME_ID"].astype(str), recent_games["DATE_STR"]))
    mins_df["DATE"] = mins_df["GAME_ID"].map(gid_to_date)

    # Compute season avg minutes from cached boxscores is expensive; approximate via last 15 as "current season trend"
    # If you want true season avg MIN, we'd need a roster player list + PlayerGameLog per player.
    # We'll still provide season avg proxy using last 15, and last 3/10 exact in that window.
    pivot = mins_df.pivot_table(index="PLAYER_NAME", columns="DATE", values="MIN", aggfunc="first").fillna(0)

    # last 5 game dates (most recent first)
    date_cols = [c for c in pivot.columns if isinstance(c, str)]
    date_cols_sorted = sorted(date_cols, reverse=True)
    last5_dates = date_cols_sorted[:5]

    # last 3/10 from mins_df (by recent chronological)
    mins_df_sorted = mins_df.sort_values("DATE", ascending=False)
    # build per-player series in recency order
    def avg_last_k(player: str, k: int) -> float:
        s = mins_df_sorted[mins_df_sorted["PLAYER_NAME"] == player]["MIN"].head(k)
        return float(s.mean()) if len(s) else 0.0

    players_list = pivot.index.tolist()
    rot = pd.DataFrame({"Player": players_list})
    rot["Last 3 Avg MIN"] = rot["Player"].apply(lambda p: round(avg_last_k(p, 3), 1))
    rot["Last 10 Avg MIN"] = rot["Player"].apply(lambda p: round(avg_last_k(p, 10), 1))
    rot["Season Avg MIN (proxy)"] = rot["Player"].apply(lambda p: round(avg_last_k(p, 15), 1))

    # Add last 5 exact minutes columns
    for dcol in last5_dates:
        rot[dcol] = rot["Player"].map(lambda p: float(pivot.loc[p, dcol]) if dcol in pivot.columns else 0.0)

    rot = rot.sort_values("Season Avg MIN (proxy)", ascending=False)
    dataframe_stretch(rot)


# ============================================================
# Player tab
# ============================================================
def player_tab(season: str, force_refresh: bool):
    st.markdown("### Player Dashboard")

    # Player search (static list)
    all_players = static_players.get_players()
    players_df = pd.DataFrame(all_players)
    players_df = players_df[players_df["is_active"] == True].copy()

    # Search box
    query = st.text_input("Search player", value="")
    if query.strip():
        mask = players_df["full_name"].str.contains(query.strip(), case=False, na=False)
        options_df = players_df[mask].copy()
    else:
        options_df = players_df.copy()

    # Limit options for performance
    options_df = options_df.sort_values("full_name").head(200)

    options = options_df["full_name"].tolist()
    if not options:
        st.info("No players found.")
        return

    sel = st.selectbox("Player", options, index=0)
    player_id = int(options_df.loc[options_df["full_name"] == sel, "id"].iloc[0])

    info = get_player_info(player_id, force_refresh=force_refresh)
    team_abbr = ""
    pos = ""
    if info is not None and not info.empty:
        if "TEAM_ABBREVIATION" in info.columns:
            team_abbr = str(info["TEAM_ABBREVIATION"].iloc[0] or "")
        if "POSITION" in info.columns:
            pos = str(info["POSITION"].iloc[0] or "")

    # Bigger headline line
    st.markdown(f"## {sel}  |  **{team_abbr}**  |  **{pos}**")

    st.divider()

    # Window controls
    c1, c2, c3 = st.columns([1.0, 1.0, 2.0], vertical_alignment="bottom")
    with c1:
        rate = st.selectbox("Rate", ["PerGame", "Per36"], index=0)
    with c2:
        window = st.selectbox("Averages Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0)
    with c3:
        box_n = st.slider("Box scores to display", min_value=1, max_value=30, value=10, step=1)

    last_n = 0 if window == "Season" else int(window.split()[-1])

    # Pull player game log (cached)
    glog = get_player_gamelog(player_id, season, force_refresh=force_refresh)
    if glog is None or glog.empty:
        st.warning("No player game log available.")
        return

    # Normalize minutes to numeric minutes for per36 conversion
    if "MIN" in glog.columns:
        # PlayerGameLog MIN is often "MM:SS" string
        mins = glog["MIN"].astype(str).str.split(":").str[0]
        glog["MIN"] = pd.to_numeric(mins, errors="coerce").fillna(0)

    # Add derived fields
    glog = add_derived_player_cols(glog)

    # Window df
    window_df = glog.copy()
    if last_n > 0:
        window_df = window_df.head(last_n).copy()

    # Averages header label
    avg_label = "Averages Season Long" if last_n == 0 else f"Averages Last {last_n} Games"
    st.markdown(f"### {avg_label}")

    # Compute averages for a set of columns if present
    avg_cols = ["MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV"]
    avg_cols = [c for c in avg_cols if c in window_df.columns]

    avg_df = window_df[avg_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_frame().T
    if rate == "Per36":
        # Convert avg_df using MIN
        avg_df = to_per36(avg_df, min_col="MIN", cols=[c for c in avg_cols if c != "MIN"])

    # Round for readability
    avg_df = avg_df.round(1)
    dataframe_stretch(avg_df)

    st.divider()

    # Box scores section (interactive)
    st.markdown("### Box Scores")
    show_df = glog.head(box_n).copy()

    # Add per36 option for display
    if rate == "Per36":
        show_df = to_per36(show_df, min_col="MIN", cols=[c for c in avg_cols if c != "MIN"])

    # Choose columns for display
    cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV"]
    cols = [c for c in cols if c in show_df.columns]

    # Convert date to string
    if "GAME_DATE" in show_df.columns:
        show_df["GAME_DATE"] = pd.to_datetime(show_df["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")

    dataframe_stretch(show_df[cols].round(1))

    st.divider()

    # Window comparison table
    st.markdown("### Window Comparison (Per Game / Per 36)")
    comp_rate = st.selectbox("Comparison rate", ["PerGame", "Per36"], index=0, key="comp_rate")

    # Current season (season-to-date)
    cur = glog.copy()
    cur_avg = cur[avg_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_frame().T
    cur_avg["Label"] = "Current Season"

    # Last 5
    l5 = glog.head(5).copy()
    l5_avg = l5[avg_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_frame().T
    l5_avg["Label"] = "Last 5 Games"

    # Last 20
    l20 = glog.head(20).copy()
    l20_avg = l20[avg_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_frame().T
    l20_avg["Label"] = "Last 20 Games"

    # Career and previous season (from career stats)
    career = get_player_career(player_id, force_refresh=force_refresh)
    prev_avg = None
    career_avg = None
    if career is not None and not career.empty and "SEASON_ID" in career.columns:
        c = career.copy()

        # Career averages proxy: mean of per-game over all seasons with GP>0, weighted by GP
        if all(x in c.columns for x in ["GP", "MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "OREB", "DREB", "STL", "BLK", "TOV"]):
            for col in ["GP", "MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "OREB", "DREB", "STL", "BLK", "TOV"]:
                c[col] = pd.to_numeric(c[col], errors="coerce")

            c = c[c["GP"].fillna(0) > 0].copy()
            if not c.empty:
                w = c["GP"].fillna(0)
                def wavg(col):
                    return float(np.average(c[col].fillna(0), weights=w))

                career_avg = {
                    "MIN": wavg("MIN"),
                    "PTS": wavg("PTS"),
                    "REB": wavg("REB"),
                    "AST": wavg("AST"),
                    "OREB": wavg("OREB"),
                    "DREB": wavg("DREB"),
                    "STL": wavg("STL"),
                    "BLK": wavg("BLK"),
                    "TOV": wavg("TOV"),
                    "FGM": wavg("FGM"),
                    "FGA": wavg("FGA"),
                    "FG3M": wavg("FG3M"),
                    "FG3A": wavg("FG3A"),
                }
                career_avg = add_derived_player_cols(pd.DataFrame([career_avg])).iloc[0].to_dict()
                career_avg["Label"] = "Career"

        # Previous season: take most recent season before current label if present
        # Your season is like "2025-26"; career SEASON_ID often like "2025-26"
        if "SEASON_ID" in c.columns:
            # sort by season string
            c2 = c.sort_values("SEASON_ID")
            prev = c2[c2["SEASON_ID"] != season].tail(1)
            if not prev.empty:
                prev_row = prev.iloc[0].to_dict()
                prev_avg = {
                    "MIN": safe_float(prev_row.get("MIN")),
                    "PTS": safe_float(prev_row.get("PTS")),
                    "REB": safe_float(prev_row.get("REB")),
                    "AST": safe_float(prev_row.get("AST")),
                    "OREB": safe_float(prev_row.get("OREB")),
                    "DREB": safe_float(prev_row.get("DREB")),
                    "STL": safe_float(prev_row.get("STL")),
                    "BLK": safe_float(prev_row.get("BLK")),
                    "TOV": safe_float(prev_row.get("TOV")),
                    "FGM": safe_float(prev_row.get("FGM")),
                    "FGA": safe_float(prev_row.get("FGA")),
                    "FG3M": safe_float(prev_row.get("FG3M")),
                    "FG3A": safe_float(prev_row.get("FG3A")),
                }
                prev_avg = add_derived_player_cols(pd.DataFrame([prev_avg])).iloc[0].to_dict()
                prev_avg["Label"] = "Previous Season"

    blocks = [cur_avg, l5_avg, l20_avg]
    if prev_avg is not None:
        blocks.append(pd.DataFrame([prev_avg]))
    if career_avg is not None:
        blocks.append(pd.DataFrame([career_avg]))

    comp = pd.concat(blocks, ignore_index=True)

    # Ensure columns exist
    comp_cols = ["Label", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB"]
    for c in comp_cols:
        if c not in comp.columns:
            comp[c] = np.nan

    if comp_rate == "Per36":
        comp = to_per36(comp, min_col="MIN", cols=[c for c in comp_cols if c not in ["Label", "MIN"]])

    comp = comp[comp_cols].round(1)
    dataframe_stretch(comp)


# ============================================================
# Main
# ============================================================
def main():
    state = header_bar()
    season = state["season"]
    date_choice = state["date_choice"]
    force_refresh = state["force_refresh"]

    # Tab order: League, Team, Player
    tab_league, tab_team, tab_player = st.tabs(["League", "Team", "Player"])

    with tab_league:
        league_tab(season, date_choice, force_refresh)

    with tab_team:
        team_tab(season, force_refresh)

    with tab_player:
        player_tab(season, force_refresh)


if __name__ == "__main__":
    main()
