# app.py — NBA League / Team / Player Dashboard (Streamlit)
import re
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.cache_store import get_or_refresh
from src.loaders import (
    get_teams_static as _teams_static,
    fetch_daily_scoreboard,
    fetch_league_players_index,
    fetch_player_logs,
    fetch_common_player_info,
    fetch_league_standings,
    fetch_league_team_stats,
    fetch_league_players_stats,
    fetch_team_gamelog,
    fetch_game_team_boxscore,
    fetch_career_summary,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Dashboard", layout="wide")

# ----------------------- Config -----------------------
CACHE_HOURS = 8
TEAM_CTX_TTL_SECONDS = 300  # "fresh-ish" context items

TTL_LEAGUE = CACHE_HOURS * 3600
TTL_TEAM_CTX = TEAM_CTX_TTL_SECONDS
TTL_PLAYER = CACHE_HOURS * 3600

_punct_re = re.compile(r"[^\w]")


# ----------------------- Helpers -----------------------
def _season_labels(start=2015, end=None) -> List[str]:
    if end is None:
        end = dt.datetime.now(dt.UTC).year

    def lab(y):
        return f"{y}-{str((y+1) % 100).zfill(2)}"

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


def numeric_format_map(df: pd.DataFrame, decimals=2) -> Dict[str, str]:
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: f"{{:.{decimals}f}}" for c in num_cols}


def _fmt1(v) -> str:
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"


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
        unsafe_allow_html=True,
    )


# ----------------------- Disk cache wrappers -----------------------
def teams_static_cached(force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key="teams_static",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: _teams_static(),
        force_refresh=force_refresh,
    ).df


def daily_scoreboard_cached(game_date: dt.date, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"scoreboard__{game_date.isoformat()}",
        ttl_seconds=TTL_TEAM_CTX,
        fetch_fn=lambda: fetch_daily_scoreboard(game_date),
        force_refresh=force_refresh,
    ).df


def get_schedule_range(start_date: dt.date, days: int, force_refresh: bool = False) -> pd.DataFrame:
    all_days = []
    for i in range(days):
        d = start_date + dt.timedelta(days=i)
        df = daily_scoreboard_cached(d, force_refresh=force_refresh)
        if df is None or df.empty:
            continue
        all_days.append(df[["DATE", "MATCHUP", "STATUS"]].copy())
    if not all_days:
        return pd.DataFrame()
    sched = pd.concat(all_days, ignore_index=True)
    return sched.sort_values(["DATE", "MATCHUP"]).reset_index(drop=True)


def league_players_index_cached(season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"players_index__{season}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_players_index(season),
        force_refresh=force_refresh,
    ).df


def league_standings_cached(season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"standings__{season}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_standings(season),
        force_refresh=force_refresh,
    ).df


def league_team_stats_cached(season: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = get_or_refresh(
        key=f"teamstats_base__{season}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_team_stats(season)[0],
        force_refresh=force_refresh,
    ).df
    adv = get_or_refresh(
        key=f"teamstats_adv__{season}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_team_stats(season)[1],
        force_refresh=force_refresh,
    ).df
    opp = get_or_refresh(
        key=f"teamstats_opp__{season}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_team_stats(season)[2],
        force_refresh=force_refresh,
    ).df
    return base, adv, opp


def league_players_stats_cached(season: str, per_mode: str, last_n: int, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"playerstats__{season}__{per_mode}__last{int(last_n)}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_players_stats(season, per_mode=per_mode, last_n_games=last_n),
        force_refresh=force_refresh,
    ).df


def player_logs_cached(player_id: int, season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"playerlogs__{season}__{int(player_id)}",
        ttl_seconds=TTL_PLAYER,
        fetch_fn=lambda: fetch_player_logs(player_id, season),
        force_refresh=force_refresh,
    ).df


def player_info_cached(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"playerinfo__{int(player_id)}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_common_player_info(player_id),
        force_refresh=force_refresh,
    ).df


def career_summary_cached(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"career__{int(player_id)}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_career_summary(player_id),
        force_refresh=force_refresh,
    ).df


def team_gamelog_cached(team_id: int, season: str, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"teamgamelog__{season}__{int(team_id)}",
        ttl_seconds=TTL_TEAM_CTX,
        fetch_fn=lambda: fetch_team_gamelog(team_id, season),
        force_refresh=force_refresh,
    ).df


def team_boxscore_cached(game_id: str, team_id: int, force_refresh: bool = False) -> pd.DataFrame:
    return get_or_refresh(
        key=f"boxscore__{str(game_id)}__team{int(team_id)}",
        ttl_seconds=TTL_TEAM_CTX,
        fetch_fn=lambda: fetch_game_team_boxscore(game_id, team_id),
        force_refresh=force_refresh,
    ).df


# ----------------------- League Tables -----------------------
def build_team_standings_table(season: str, force_refresh: bool = False) -> pd.DataFrame:
    standings = league_standings_cached(season, force_refresh=force_refresh)
    base, adv, _opp = league_team_stats_cached(season, force_refresh=force_refresh)

    if standings.empty and base.empty and adv.empty:
        return pd.DataFrame()

    base_cols = [
        "TEAM_ID",
        "PTS",
        "REB",
        "AST",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "TOV",
        "STL",
        "BLK",
        "PLUS_MINUS",
    ]
    base = _ensure_cols(base, base_cols)[base_cols].copy()
    base["TEAM_ID"] = pd.to_numeric(base["TEAM_ID"], errors="coerce").astype("Int64")

    adv_cols = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    adv = _ensure_cols(adv, adv_cols)[adv_cols].copy()
    adv["TEAM_ID"] = pd.to_numeric(adv["TEAM_ID"], errors="coerce").astype("Int64")

    merged = standings.copy()
    merged = merged.merge(base, on="TEAM_ID", how="left")
    merged = merged.merge(adv, on="TEAM_ID", how="left")

    tstatic = teams_static_cached(force_refresh=False)
    merged = merged.merge(tstatic[["TEAM_ID", "TEAM_NAME"]], on="TEAM_ID", how="left", suffixes=("", "_STATIC"))
    merged["TEAM"] = merged["TEAM"].fillna(merged["TEAM_NAME"])
    merged = merged.drop(columns=["TEAM_NAME", "TEAM_NAME_STATIC"], errors="ignore")

    merged["W"] = pd.to_numeric(merged["W"], errors="coerce").fillna(0).astype(int)
    merged["L"] = pd.to_numeric(merged["L"], errors="coerce").fillna(0).astype(int)
    merged["W%"] = pd.to_numeric(merged["W%"], errors="coerce")

    order = [
        "TEAM",
        "CONF",
        "W",
        "L",
        "W%",
        "PTS",
        "REB",
        "AST",
        "NET_RATING",
        "OFF_RATING",
        "DEF_RATING",
        "PACE",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "TOV",
        "STL",
        "BLK",
        "PLUS_MINUS",
    ]
    merged = _ensure_cols(merged, order)[order].copy()
    merged = _to_num(merged, [c for c in order if c not in ("TEAM", "CONF")])

    merged = merged.sort_values(["W%", "W", "TEAM"], ascending=[False, False, True]).reset_index(drop=True)

    merged = merged.rename(
        columns={
            "NET_RATING": "NETRTG",
            "OFF_RATING": "OFFRTG",
            "DEF_RATING": "DEFRTG",
            "FG_PCT": "FG%",
            "FG3_PCT": "3P%",
            "FT_PCT": "FT%",
            "PLUS_MINUS": "+/-",
        }
    )
    return merged


def build_opponent_allowed_table(season: str, force_refresh: bool = False) -> pd.DataFrame:
    standings = league_standings_cached(season, force_refresh=force_refresh)
    _base, _adv, opp = league_team_stats_cached(season, force_refresh=force_refresh)

    if standings.empty and opp.empty:
        return pd.DataFrame()
    if opp.empty or "TEAM_ID" not in opp.columns:
        return pd.DataFrame()

    opp = opp.copy()
    opp["TEAM_ID"] = pd.to_numeric(opp["TEAM_ID"], errors="coerce").astype("Int64")

    preferred = [
        "OPP_PTS",
        "OPP_REB",
        "OPP_AST",
        "OPP_FG_PCT",
        "OPP_FG3_PCT",
        "OPP_FT_PCT",
        "OPP_TOV",
        "OPP_STL",
        "OPP_BLK",
        "OPP_OREB",
        "OPP_DREB",
        "OPP_FGA",
        "OPP_FG3A",
        "OPP_FTA",
        "OPP_PF",
    ]
    have = [c for c in preferred if c in opp.columns]
    opp2 = opp[["TEAM_ID"] + have].copy()

    out = standings[["TEAM_ID", "TEAM", "CONF", "W", "L", "W%"]].copy()
    out = out.merge(opp2, on="TEAM_ID", how="left")

    out["W"] = pd.to_numeric(out["W"], errors="coerce").fillna(0).astype(int)
    out["L"] = pd.to_numeric(out["L"], errors="coerce").fillna(0).astype(int)
    out["W%"] = pd.to_numeric(out["W%"], errors="coerce")

    if "OPP_PTS" in out.columns:
        out = out.sort_values(["OPP_PTS", "TEAM"], ascending=[True, True]).reset_index(drop=True)
    else:
        out = out.sort_values(["W%", "TEAM"], ascending=[False, True]).reset_index(drop=True)

    rename = {
        "OPP_PTS": "Opp PTS",
        "OPP_REB": "Opp REB",
        "OPP_AST": "Opp AST",
        "OPP_FG_PCT": "Opp FG%",
        "OPP_FG3_PCT": "Opp 3P%",
        "OPP_FT_PCT": "Opp FT%",
        "OPP_TOV": "Opp TOV",
        "OPP_STL": "Opp STL",
        "OPP_BLK": "Opp BLK",
        "OPP_OREB": "Opp OREB",
        "OPP_DREB": "Opp DREB",
        "OPP_FGA": "Opp FGA",
        "OPP_FG3A": "Opp 3PA",
        "OPP_FTA": "Opp FTA",
        "OPP_PF": "Opp PF",
    }
    out = out.rename(columns=rename)
    out = out.drop(columns=["TEAM_ID"], errors="ignore")
    return out


def _leader_table(df: pd.DataFrame, stat: str, n: int = 15) -> pd.DataFrame:
    if df is None or df.empty or stat not in df.columns:
        return pd.DataFrame()
    keep = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", stat]
    df = _ensure_cols(df, keep)[keep].copy()
    for c in ["GP", "MIN", stat]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(stat, ascending=False).head(n).reset_index(drop=True)
    df.insert(0, "RANK", np.arange(1, len(df) + 1))
    df = df.rename(columns={"TEAM_ABBREVIATION": "Team"})
    return df


# =====================================================================
# LEAGUE TAB
# =====================================================================
def league_tab():
    st.title("NBA League")

    bar1, bar2, bar3, bar4 = st.columns([1.2, 1.0, 1.0, 1.1])
    with bar1:
        season = st.selectbox("Season", SEASONS, index=0, key="lg_season")
    with bar2:
        start_date = st.date_input("Schedule start", value=dt.date.today(), key="lg_sched_start")
    with bar3:
        days_ahead = st.selectbox("Days ahead", [3, 5, 7, 10, 14], index=2, key="lg_days_ahead")
    with bar4:
        force_refresh = st.button("Hard refresh (league)", key="lg_force_refresh")

    tabA, tabB = st.tabs(["Teams", "Player Leaders"])

    with tabA:
        st.subheader("Schedule (Upcoming + Completed)")
        with st.spinner("Loading schedule..."):
            sched = get_schedule_range(start_date, int(days_ahead), force_refresh=force_refresh)

        if sched.empty:
            st.info("No games found in this date range (or the API returned empty). Try a different start date.")
        else:
            st.dataframe(sched, width="stretch", height=_auto_height(sched, max_px=460))

        st.divider()

        st.subheader("Standings + Team Stats")
        conf = st.radio("Conference filter", ["All", "East", "West"], horizontal=True, key="lg_conf_filter")
        with st.spinner("Loading standings & team stats..."):
            team_tbl = build_team_standings_table(season, force_refresh=force_refresh)

        if team_tbl.empty:
            st.error("Could not load standings/team stats. Try refreshing.")
            return

        if conf != "All":
            team_tbl = team_tbl[team_tbl["CONF"].astype("string").str.upper() == conf.upper()].copy()

        fmt = numeric_format_map(team_tbl, decimals=2)
        for pct_col in ["W%", "FG%", "3P%", "FT%"]:
            if pct_col in fmt:
                fmt[pct_col] = "{:.3f}"
        fmt.pop("W", None)
        fmt.pop("L", None)

        st.dataframe(team_tbl.style.format(fmt), width="stretch", height=_auto_height(team_tbl, max_px=820))

        st.divider()

        st.subheader("Opponent Allowed Metrics (Leaguewide)")
        with st.spinner("Loading opponent tables..."):
            opp_tbl = build_opponent_allowed_table(season, force_refresh=force_refresh)

        if opp_tbl.empty:
            st.info("Opponent allowed metrics are not available for this season from the API.")
        else:
            fmt2 = numeric_format_map(opp_tbl, decimals=2)
            fmt2.pop("W", None)
            fmt2.pop("L", None)
            for col in ["W%", "Opp FG%", "Opp 3P%", "Opp FT%"]:
                if col in fmt2:
                    fmt2[col] = "{:.3f}"
            st.dataframe(opp_tbl.style.format(fmt2), width="stretch", height=_auto_height(opp_tbl, max_px=820))

    with tabB:
        st.subheader("Leaguewide Player Leaders")

        l1, l2, l3, l4 = st.columns([1.0, 1.0, 1.0, 1.2])
        with l1:
            per_mode = st.selectbox("Per mode", ["PerGame", "Per36"], index=0, key="lg_pl_permode")
        with l2:
            window = st.selectbox("Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="lg_pl_window")
        with l3:
            min_gp = st.selectbox("Min GP", [0, 5, 10, 15, 20], index=2, key="lg_pl_mingp")
        with l4:
            top_n = st.slider("Top N", 5, 30, 15, 1, key="lg_pl_topn")

        last_n = 0 if window == "Season" else int(window.split()[-1])
        with st.spinner("Loading player stats..."):
            pstats = league_players_stats_cached(season, per_mode=per_mode, last_n=int(last_n), force_refresh=force_refresh)

        if pstats.empty:
            st.error("Could not load player stats. Try refreshing.")
            return

        if "GP" in pstats.columns and int(min_gp) > 0:
            pstats = pstats[pd.to_numeric(pstats["GP"], errors="coerce") >= int(min_gp)].copy()

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
                    col.dataframe(
                        tbl.style.format(numeric_format_map(tbl, decimals=2)),
                        width="stretch",
                        height=_auto_height(tbl, max_px=520),
                    )


# =====================================================================
# TEAM TAB
# =====================================================================
def team_tab():
    inject_rank_tile_css()
    st.title("NBA Team")

    t1, t2, t3, t4 = st.columns([1.2, 1.3, 1.0, 1.1])
    with t1:
        season = st.selectbox("Season", SEASONS, index=0, key="tm_season")
    with t2:
        tdf = teams_static_cached()
        team_name = st.selectbox("Team", sorted(tdf["TEAM_NAME"].tolist()), key="tm_team")
        team_row = tdf[tdf["TEAM_NAME"] == team_name].iloc[0]
        team_id = int(team_row["TEAM_ID"])
        team_abbr = str(team_row["TEAM_ABBREVIATION"])
    with t3:
        roster_per_mode = st.selectbox("Roster per-mode", ["PerGame", "Per36"], index=0, key="tm_roster_permode")
    with t4:
        roster_window = st.selectbox("Roster window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="tm_roster_window")

    with st.spinner("Loading team stats..."):
        base, adv, opp = league_team_stats_cached(season)

    if base.empty or adv.empty:
        st.error("Could not load team stats. Try refreshing.")
        return

    base_cols = [
        "TEAM_ID",
        "TEAM_NAME",
        "GP",
        "W",
        "L",
        "W_PCT",
        "PTS",
        "REB",
        "AST",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "TOV",
        "STL",
        "BLK",
        "PF",
        "PLUS_MINUS",
        "FGA",
        "FG3A",
        "FTA",
        "OREB",
        "DREB",
    ]
    adv_cols = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]

    base = _ensure_cols(base, base_cols)[base_cols].copy()
    adv = _ensure_cols(adv, adv_cols)[adv_cols].copy()

    base["TEAM_ID"] = pd.to_numeric(base["TEAM_ID"], errors="coerce").astype("Int64")
    adv["TEAM_ID"] = pd.to_numeric(adv["TEAM_ID"], errors="coerce").astype("Int64")

    merged = base.merge(adv, on="TEAM_ID", how="left")

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

    # ----------------------- Team Gamelog (FIXED UI + refresh) -----------------------
    refresh_gamelog = st.button("Refresh Team Game Log", key=f"tm_refresh_gamelog_{team_id}_{season}")

    with st.spinner("Loading team gamelog..."):
        glog = team_gamelog_cached(team_id, season, force_refresh=refresh_gamelog)

    season_record = "—"
    if pd.notna(tr.get("W")) and pd.notna(tr.get("L")):
        season_record = f"{int(tr['W'])}–{int(tr['L'])}"

    last10_record = "—"
    if glog is not None and not glog.empty and "WL" in glog.columns:
        last10 = glog.head(10).copy()
        w10 = int((last10["WL"].astype("string") == "W").sum())
        l10 = int((last10["WL"].astype("string") == "L").sum())
        last10_record = f"{w10}–{l10}"

    st.subheader(f"{tr['TEAM_NAME']} — {season}")
    st.markdown(
        f"<div style='font-size: 1.15rem; font-weight: 700;'>Record: {season_record} &nbsp;&nbsp;|&nbsp;&nbsp; Last 10: {last10_record}</div>",
        unsafe_allow_html=True,
    )

    def tile_row(items):
        cols = st.columns(len(items))
        for (label, key, pct_flag), col in zip(items, cols):
            r = ranks.get(key, pd.Series([np.nan] * len(merged))).loc[row.index[0]] if key in ranks else np.nan
            _rank_tile(col, label, tr.get(key, np.nan), r, total=n_teams, pct=pct_flag)

    tile_row([("PTS", "PTS", False), ("NETRTG", "NET_RATING", False), ("OFFRTG", "OFF_RATING", False), ("DEFRTG", "DEF_RATING", False), ("PACE", "PACE", False)])
    tile_row([("FGA", "FGA", False), ("FG%", "FG_PCT", True), ("3PA", "FG3A", False), ("3P%", "FG3_PCT", True), ("FTA", "FTA", False)])
    tile_row([("FT%", "FT_PCT", True), ("OREB", "OREB", False), ("DREB", "DREB", False), ("REB", "REB", False), ("AST", "AST", False)])
    tile_row([("TOV", "TOV", False), ("STL", "STL", False), ("BLK", "BLK", False), ("PF", "PF", False), ("+/-", "PLUS_MINUS", False)])

    st.divider()

    # ----------------------- Team Box Scores -----------------------
    st.subheader("Team Box Scores (Pick a Game)")

    if glog is None or glog.empty:
        st.info("No game log available for box score selection. Try 'Refresh Team Game Log'.")
    else:
        glog2 = glog.copy()

        # Back-compat normalization in case cached parquet has older casing
        if "GAME_ID" not in glog2.columns and "Game_ID" in glog2.columns:
            glog2 = glog2.rename(columns={"Game_ID": "GAME_ID"})
        if "GAME_DATE" not in glog2.columns and "GameDate" in glog2.columns:
            glog2 = glog2.rename(columns={"GameDate": "GAME_DATE"})

        if "GAME_ID" not in glog2.columns:
            st.info("Game log loaded, but GAME_ID is missing. Click 'Refresh Team Game Log'.")
            return

        glog2["GAME_DATE"] = pd.to_datetime(glog2.get("GAME_DATE", pd.NaT), errors="coerce")
        glog2["GAME_DATE_STR"] = glog2["GAME_DATE"].dt.strftime("%Y-%m-%d")
        glog2["LABEL"] = (
            glog2["GAME_DATE_STR"].astype("string")
            + " | "
            + glog2.get("MATCHUP", "").astype("string")
            + " | "
            + glog2.get("WL", "").astype("string")
        )

        # Default to most recent played game
        label_list = glog2["LABEL"].fillna("").tolist()
        gsel = st.selectbox("Select game", label_list, index=0, key="tm_game_select")

        game_id = str(glog2[glog2["LABEL"] == gsel]["GAME_ID"].iloc[0])

        refresh_box = st.button("Refresh Box Score", key=f"tm_refresh_box_{team_id}_{game_id}")
        with st.spinner("Loading box score..."):
            bx = team_boxscore_cached(game_id, team_id, force_refresh=refresh_box)

        if bx.empty:
            st.info("No box score returned for this game (API may be throttling). Try another game or refresh.")
        else:
            keep = ["PLAYER_NAME", "START_POSITION", "MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "STL", "BLK", "TO", "PF", "PLUS_MINUS"]
            bx = _ensure_cols(bx, keep)[keep].copy()
            bx = bx.rename(columns={"TO": "TOV"})
            for c in bx.columns:
                if c not in ("PLAYER_NAME", "START_POSITION", "MIN"):
                    bx[c] = pd.to_numeric(bx[c], errors="coerce")

            def parse_min(x):
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

            bx["MIN"] = bx["MIN"].apply(parse_min)
            bx["PRA"] = bx["PTS"].fillna(0) + bx["REB"].fillna(0) + bx["AST"].fillna(0)
            bx["FG2M"] = (bx["FGM"].fillna(0) - bx["FG3M"].fillna(0)).clip(lower=0)

            show_cols = ["PLAYER_NAME", "START_POSITION", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "FTM", "OREB", "DREB", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"]
            bx = _ensure_cols(bx, show_cols)[show_cols].copy()
            st.dataframe(
                bx.style.format({c: "{:.1f}" for c in bx.select_dtypes(include=[np.number]).columns}),
                width="stretch",
                height=_auto_height(bx, max_px=760),
            )

    st.divider()

    # ----------------------- Minutes Rotation -----------------------
    st.subheader("Minutes Rotation (Season vs Recent + Last 5 Games)")
    last_n_3 = 3
    last_n_10 = 10
    last_n_user = 0 if roster_window == "Season" else int(roster_window.split()[-1])

    with st.spinner("Loading roster snapshots..."):
        season_roster_pg = league_players_stats_cached(season, per_mode="PerGame", last_n=0)
        last3_pg = league_players_stats_cached(season, per_mode="PerGame", last_n=last_n_3)
        last10_pg = league_players_stats_cached(season, per_mode="PerGame", last_n=last_n_10)

    if season_roster_pg.empty:
        st.info("Roster data not available.")
        return

    season_team = season_roster_pg[season_roster_pg["TEAM_ID"] == team_id].copy()
    last3_team = last3_pg[last3_pg["TEAM_ID"] == team_id].copy() if not last3_pg.empty else pd.DataFrame()
    last10_team = last10_pg[last10_pg["TEAM_ID"] == team_id].copy() if not last10_pg.empty else pd.DataFrame()

    roster_df = league_players_stats_cached(season, per_mode=roster_per_mode, last_n=last_n_user)
    roster_df = roster_df[roster_df["TEAM_ID"] == team_id].copy() if roster_df is not None and not roster_df.empty else pd.DataFrame()

    rot = season_team[["PLAYER_ID", "PLAYER_NAME", "MIN"]].copy()
    rot = rot.rename(columns={"MIN": "MIN_season"})
    if not last3_team.empty:
        tmp = last3_team[["PLAYER_ID", "MIN"]].copy().rename(columns={"MIN": "MIN_last3"})
        rot = rot.merge(tmp, on="PLAYER_ID", how="left")
    else:
        rot["MIN_last3"] = np.nan
    if not last10_team.empty:
        tmp = last10_team[["PLAYER_ID", "MIN"]].copy().rename(columns={"MIN": "MIN_last10"})
        rot = rot.merge(tmp, on="PLAYER_ID", how="left")
    else:
        rot["MIN_last10"] = np.nan

    rot["MIN_season"] = pd.to_numeric(rot["MIN_season"], errors="coerce")
    rot["MIN_last3"] = pd.to_numeric(rot["MIN_last3"], errors="coerce")
    rot["MIN_last10"] = pd.to_numeric(rot["MIN_last10"], errors="coerce")
    rot = rot.sort_values("MIN_season", ascending=False).reset_index(drop=True)

    display_cols = ["PLAYER_NAME", "MIN_last3", "MIN_last10", "MIN_season"]
    rot_disp = _ensure_cols(rot, display_cols)[display_cols].copy()
    rot_disp = rot_disp.rename(columns={"MIN_last3": "MIN (Last 3)", "MIN_last10": "MIN (Last 10)", "MIN_season": "MIN (Season)"})

    st.dataframe(
        rot_disp.style.format({c: "{:.1f}" for c in rot_disp.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(rot_disp, max_px=900),
    )

    st.divider()

    # ----------------------- Roster Stats -----------------------
    st.subheader("Roster Stats (Interactive)")
    if roster_df.empty:
        st.info("No roster data returned for this view.")
        return

    roster_wanted = [
        "PLAYER_NAME", "AGE", "GP", "MIN", "PTS", "REB", "AST",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
    ]
    roster_df = _ensure_cols(roster_df, roster_wanted)[roster_wanted].copy()
    for c in roster_df.columns:
        if c not in ("PLAYER_NAME",):
            roster_df[c] = pd.to_numeric(roster_df[c], errors="coerce")

    roster_df = roster_df.sort_values("MIN", ascending=False).reset_index(drop=True)

    st.dataframe(
        roster_df.style.format({c: "{:.1f}" for c in roster_df.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(roster_df, max_px=900),
    )


# =====================================================================
# PLAYER TAB
# =====================================================================
def player_tab():
    st.title("NBA Player")

    p1, p2, p3, p4 = st.columns([1.2, 1.8, 1.0, 1.0])
    with p1:
        season = st.selectbox("Season", SEASONS, index=0, key="pl_season")
    with p2:
        with st.spinner("Loading players..."):
            pindex = league_players_index_cached(season)
        if pindex.empty:
            st.error("Could not load player index for this season.")
            return
        q = st.text_input("Search player", key="pl_search").strip()
        filt = pindex if not q else pindex[pindex["PLAYER_NAME"].astype("string").str.contains(q, case=False, na=False)]
        if filt.empty:
            st.info("No players match your search.")
            return
        player_name = st.selectbox("Player", filt["PLAYER_NAME"].tolist(), index=0, key="pl_player")
        prow = filt[filt["PLAYER_NAME"] == player_name].iloc[0]
        player_id = int(prow["PLAYER_ID"])
    with p3:
        per_mode = st.selectbox("View", ["PerGame", "Per36"], index=0, key="pl_permode")
    with p4:
        window = st.selectbox("Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="pl_window")

    with st.spinner("Fetching player logs..."):
        logs = player_logs_cached(player_id, season)
        cpi = player_info_cached(player_id)

    if logs.empty:
        st.error("No game logs available for this player/season.")
        return

    team_name_disp = (
        cpi["TEAM_NAME"].iloc[0] if (cpi is not None and not cpi.empty and "TEAM_NAME" in cpi.columns) else str(prow.get("TEAM_NAME", ""))
    )
    pos = cpi["POSITION"].iloc[0] if (cpi is not None and not cpi.empty and "POSITION" in cpi.columns) else "N/A"
    gp = len(logs)

    st.subheader(f"{player_name} — {season}")
    st.markdown(
        f"<div style='font-size: 1.05rem; font-weight: 650;'>Team: {team_name_disp} &nbsp;&nbsp;|&nbsp;&nbsp; Position: {pos} &nbsp;&nbsp;|&nbsp;&nbsp; Games Logged: {gp}</div>",
        unsafe_allow_html=True,
    )

    if window == "Season":
        window_df = logs.copy()
        avg_label = "Averages (Season Long)"
    else:
        n = int(window.split()[-1])
        window_df = logs.head(n).copy()
        avg_label = f"Averages (Last {n} Games)"

    for c in ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "STL", "BLK", "TOV", "PF"]:
        if c not in window_df.columns:
            window_df[c] = 0
        window_df[c] = pd.to_numeric(window_df[c], errors="coerce").fillna(0)

    window_df["PRA"] = window_df["PTS"] + window_df["REB"] + window_df["AST"]
    window_df["FG2M"] = (window_df["FGM"] - window_df["FG3M"]).clip(lower=0)

    sums_cols = ["MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV"]
    sums = window_df[sums_cols].sum(numeric_only=True)
    games_n = max(1, len(window_df))
    if per_mode == "PerGame":
        av = sums / games_n
    else:
        total_min = float(sums.get("MIN", 0.0))
        av = (sums / total_min) * 36.0 if total_min > 0 else sums * np.nan

    st.markdown(f"### {avg_label} — {per_mode}")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("MIN" if per_mode == "PerGame" else "MIN (—)", _fmt1(av.get("MIN", np.nan)) if per_mode == "PerGame" else "—")
    a2.metric("PTS", _fmt1(av.get("PTS", np.nan)))
    a3.metric("REB", _fmt1(av.get("REB", np.nan)))
    a4.metric("AST", _fmt1(av.get("AST", np.nan)))
    a5.metric("PRA", _fmt1(av.get("PRA", np.nan)))
    st.caption("Per-36 computes totals first, then scales by total minutes in the selected window.")

    st.divider()

    st.markdown("### Box Scores (interactive)")

    logs2 = logs.copy()
    if "MATCHUP" in logs2.columns:
        logs2["OPP_ABBR"] = logs2["MATCHUP"].apply(parse_opp_from_matchup)
    else:
        logs2["OPP_ABBR"] = None

    min_date = logs2["GAME_DATE"].min()
    max_date = logs2["GAME_DATE"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        min_date = dt.datetime.now(dt.UTC) - dt.timedelta(days=365)
        max_date = dt.datetime.now(dt.UTC)

    f1, f2, f3 = st.columns([1.2, 1.0, 1.0])
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
        max_games = int(len(logs2))
        if max_games <= 1:
            show_n = max_games
            st.caption(f"Only {max_games} game available.")
        else:
            min_val = 1 if max_games < 5 else 5
            default_val = min(10, max_games)
            if default_val < min_val:
                default_val = max_games
            show_n = st.slider("Max games to display", min_value=min_val, max_value=max_games, value=default_val, key="pl_show_n")

    start_d, end_d = dr
    flt = logs2[(logs2["GAME_DATE"].dt.date >= start_d) & (logs2["GAME_DATE"].dt.date <= end_d)].copy()
    if opp_choice != "All":
        flt = flt[flt["OPP_ABBR"] == opp_choice].copy()

    show = flt.head(show_n).copy()

    for c in ["MIN", "PTS", "REB", "AST", "FGM", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV", "PF", "FGA", "FG3A", "FTM", "FTA"]:
        if c not in show.columns:
            show[c] = 0
        show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0)
    show["PRA"] = show["PTS"] + show["REB"] + show["AST"]
    show["FG2M"] = (show["FGM"] - show["FG3M"]).clip(lower=0)

    keep_cols = [
        "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "PRA",
        "FG2M", "FG3M", "FTM", "OREB", "DREB", "STL", "BLK", "TOV", "PF",
    ]
    show = _ensure_cols(show, keep_cols)[keep_cols].copy()
    st.dataframe(
        show.style.format({c: "{:.1f}" for c in show.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(show),
    )

    st.divider()

    st.markdown("### Window Comparison (Season / Previous / Career / Last 5 / Last 20)")
    cA, cB = st.columns([1.0, 2.0])
    with cA:
        comp_mode = st.selectbox("Comparison per-mode", ["PerGame", "Per36"], index=0, key="pl_comp_mode")
    with cB:
        st.caption("Compares: current season, previous season, career, last 5 games, last 20 games.")

    def get_single_player_dash(season_str: str, per_mode_str: str, last_n: int) -> pd.Series:
        df = league_players_stats_cached(season_str, per_mode=per_mode_str, last_n=last_n)
        if df is None or df.empty or "PLAYER_ID" not in df.columns:
            return pd.Series(dtype="float")
        row = df[df["PLAYER_ID"] == player_id]
        if row.empty:
            return pd.Series(dtype="float")
        return row.iloc[0]

    def prev_season(season_str: str) -> Optional[str]:
        if season_str not in SEASONS:
            return None
        idx = SEASONS.index(season_str)
        if idx + 1 < len(SEASONS):
            return SEASONS[idx + 1]
        return None

    prev = prev_season(season)

    cur_row = get_single_player_dash(season, comp_mode, 0)
    prev_row = get_single_player_dash(prev, comp_mode, 0) if prev else pd.Series(dtype="float")
    last5_row = get_single_player_dash(season, comp_mode, 5)
    last20_row = get_single_player_dash(season, comp_mode, 20)

    car = career_summary_cached(player_id)

    def extract_comp(row: pd.Series, label: str) -> Dict[str, float]:
        if row is None or row.empty:
            return {"WINDOW": label}
        for c in ["MIN", "PTS", "REB", "AST", "FGM", "FG3M", "OREB", "DREB"]:
            if c not in row.index:
                row[c] = np.nan
        out = {
            "WINDOW": label,
            "MIN": float(pd.to_numeric(row.get("MIN", np.nan), errors="coerce")) if pd.notna(row.get("MIN", np.nan)) else np.nan,
            "PTS": float(pd.to_numeric(row.get("PTS", np.nan), errors="coerce")) if pd.notna(row.get("PTS", np.nan)) else np.nan,
            "REB": float(pd.to_numeric(row.get("REB", np.nan), errors="coerce")) if pd.notna(row.get("REB", np.nan)) else np.nan,
            "AST": float(pd.to_numeric(row.get("AST", np.nan), errors="coerce")) if pd.notna(row.get("AST", np.nan)) else np.nan,
            "FG3M": float(pd.to_numeric(row.get("FG3M", np.nan), errors="coerce")) if pd.notna(row.get("FG3M", np.nan)) else np.nan,
            "OREB": float(pd.to_numeric(row.get("OREB", np.nan), errors="coerce")) if pd.notna(row.get("OREB", np.nan)) else np.nan,
            "DREB": float(pd.to_numeric(row.get("DREB", np.nan), errors="coerce")) if pd.notna(row.get("DREB", np.nan)) else np.nan,
        }
        out["PRA"] = (out.get("PTS", np.nan) + out.get("REB", np.nan) + out.get("AST", np.nan)) if pd.notna(out.get("PTS", np.nan)) else np.nan
        fgm = pd.to_numeric(row.get("FGM", np.nan), errors="coerce")
        fg3m = pd.to_numeric(row.get("FG3M", np.nan), errors="coerce")
        out["FG2M"] = float((fgm - fg3m)) if pd.notna(fgm) and pd.notna(fg3m) else np.nan
        return out

    comp_rows = []
    comp_rows.append(extract_comp(cur_row, f"Season {season}"))
    comp_rows.append(extract_comp(prev_row, f"Season {prev}" if prev else "Previous Season (N/A)"))

    if car is not None and not car.empty:
        if comp_mode == "PerGame":
            comp_rows.append(
                {
                    "WINDOW": "Career",
                    "MIN": np.nan,
                    "PTS": float(car["PTS_PG"].iloc[0]),
                    "REB": float(car["REB_PG"].iloc[0]),
                    "AST": float(car["AST_PG"].iloc[0]),
                    "PRA": float(car["PTS_PG"].iloc[0] + car["REB_PG"].iloc[0] + car["AST_PG"].iloc[0]),
                    "FG2M": float(car["FGM_PG"].iloc[0] - car["FG3M_PG"].iloc[0]),
                    "FG3M": float(car["FG3M_PG"].iloc[0]),
                    "OREB": float(car["OREB_PG"].iloc[0]),
                    "DREB": float(car["DREB_PG"].iloc[0]),
                }
            )
        else:
            comp_rows.append(
                {
                    "WINDOW": "Career",
                    "MIN": np.nan,
                    "PTS": float(car["PTS_36"].iloc[0]),
                    "REB": float(car["REB_36"].iloc[0]),
                    "AST": float(car["AST_36"].iloc[0]),
                    "PRA": float(car["PTS_36"].iloc[0] + car["REB_36"].iloc[0] + car["AST_36"].iloc[0]),
                    "FG2M": float(car["FGM_36"].iloc[0] - car["FG3M_36"].iloc[0]),
                    "FG3M": float(car["FG3M_36"].iloc[0]),
                    "OREB": float(car["OREB_36"].iloc[0]),
                    "DREB": float(car["DREB_36"].iloc[0]),
                }
            )
    else:
        comp_rows.append({"WINDOW": "Career"})

    comp_rows.append(extract_comp(last5_row, "Last 5"))
    comp_rows.append(extract_comp(last20_row, "Last 20"))

    comp_df = pd.DataFrame(comp_rows)
    comp_df = _ensure_cols(comp_df, ["WINDOW", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB"])
    for c in comp_df.columns:
        if c != "WINDOW":
            comp_df[c] = pd.to_numeric(comp_df[c], errors="coerce")

    st.dataframe(
        comp_df.style.format({c: "{:.1f}" for c in comp_df.select_dtypes(include=[np.number]).columns}),
        width="stretch",
        height=_auto_height(comp_df, max_px=420),
    )

    trend_cols = ["MIN", "PTS", "REB", "AST", "PRA"]
    tdf = window_df[["GAME_DATE"] + trend_cols].copy().sort_values("GAME_DATE")
    for s in trend_cols:
        chart = alt.Chart(tdf).mark_line(point=True).encode(x="GAME_DATE:T", y=alt.Y(s, title=s)).properties(height=160)
        st.altair_chart(chart, width="stretch")


# =====================================================================
# MAIN
# =====================================================================
def main():
    tab0, tab1, tab2 = st.tabs(["League", "Team", "Player"])
    with tab0:
        league_tab()
    with tab1:
        team_tab()
    with tab2:
        player_tab()


if __name__ == "__main__":
    main()
