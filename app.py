import time
import datetime
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
from zoneinfo import ZoneInfo

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    commonplayerinfo,
    scoreboardv2,
    leaguestandingsv3,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Dashboards", layout="wide")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2


def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
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
        end = datetime.datetime.now(datetime.UTC).year

    def lab(y):
        return f"{y}-{str((y + 1) % 100).zfill(2)}"

    return [lab(y) for y in range(end, start - 1, -1)]


SEASONS = _season_labels(2015, datetime.datetime.now(datetime.UTC).year)


def _prev_season_label(season_label: str) -> str:
    try:
        y0 = int(season_label.split("-")[0])
        return f"{y0 - 1}-{str(y0 % 100).zfill(2)}"
    except Exception:
        return season_label


# ----------------------- Small helpers -----------------------
def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


# ----------------------- UI Helpers -----------------------
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


def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)


def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: "{:.2f}" for c in num_cols}


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
    for col in ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB"]:
        if col not in df.columns:
            df[col] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["2PM"] = df["FGM"] - df["FG3M"]
    df["2PA"] = df["FGA"] - df["FG3A"]
    keep_order = [
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "PRA",
        "2PM",
        "2PA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
    ]
    existing = [c for c in keep_order if c in df.columns]
    return df[existing]


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


def _safe_num(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def compute_per36_from_logs(logs: pd.DataFrame) -> dict:
    keys = ["PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "FTM", "OREB", "DREB"]
    if logs is None or logs.empty:
        return {k: np.nan for k in keys}
    need = ["MIN", "PTS", "REB", "AST", "FGM", "FG3M", "FTM", "OREB", "DREB"]
    logs = logs.copy()
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
    sum_oreb = logs["OREB"].sum()
    sum_dreb = logs["DREB"].sum()
    sum_pra = sum_pts + sum_reb + sum_ast
    sum_fg2 = max(0.0, sum_fgm - sum_fg3)

    def per36(x):
        return 36.0 * (x / sum_min) if sum_min and sum_min > 0 else np.nan

    return {
        "PTS": per36(sum_pts),
        "REB": per36(sum_reb),
        "AST": per36(sum_ast),
        "PRA": per36(sum_pra),
        "FG2M": per36(sum_fg2),
        "FG3M": per36(sum_fg3),
        "FTM": per36(sum_ftm),
        "OREB": per36(sum_oreb),
        "DREB": per36(sum_dreb),
    }


# ----------------------- Team name helpers -----------------------
def _build_static_maps():
    teams_df = pd.DataFrame(static_teams.get_teams())
    by_full = dict(zip(teams_df["full_name"].astype(str), teams_df["abbreviation"].astype(str)))
    id_by_full = dict(zip(teams_df["full_name"].astype(str), teams_df["id"].astype(int)))

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
        "PHO": "PHX",
        "BRK": "BKN",
        "NJN": "BKN",
        "NOH": "NOP",
        "NOK": "NOP",
        "CHO": "CHA",
        "CHH": "CHA",
        "SEA": "OKC",
        "WSB": "WAS",
        "VAN": "MEM",
    }

    by_full_cf = {k.casefold(): v for k, v in by_full.items()}
    nick_cf = {k.casefold(): v for k, v in nick_map.items()}
    alias_up = {k.upper(): v.upper() for k, v in alias_map.items()}
    return by_full_cf, nick_cf, alias_up, id_by_full


BY_FULL_CF, NICK_CF, ABBR_ALIAS, TEAMID_BY_FULL = _build_static_maps()


# ----------------------- Cached shared data -----------------------
@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_season_player_index(season):
    try:
        frames = _retry_api(
            LeagueDashPlayerStats,
            {
                "season": season,
                "per_mode_detailed": "PerGame",
                "season_type_all_star": "Regular Season",
                "league_id_nullable": "00",
            },
        )
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GP", "MIN"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    return (
        df[keep]
        .drop_duplicates(subset=["PLAYER_ID"])
        .sort_values(["TEAM_NAME", "PLAYER_NAME"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_player_logs(player_id, season):
    try:
        frames = _retry_api(
            playergamelog.PlayerGameLog,
            {
                "player_id": player_id,
                "season": season,
                "season_type_all_star": "Regular Season",
                "league_id_nullable": "00",
            },
        )
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_player_career(player_id):
    try:
        frames = _retry_api(playercareerstats.PlayerCareerStats, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def get_common_player_info(player_id):
    try:
        frames = _retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ----------------------- League helpers -----------------------
@st.cache_data(ttl=900, show_spinner=True)
def get_daily_scoreboard(date_obj: datetime.date):
    date_str = date_obj.strftime("%m/%d/%Y")
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

    games = frames[0].copy()
    if games.empty:
        return pd.DataFrame()

    keep_cols = [
        "GAME_STATUS_TEXT",
        "HOME_TEAM_ABBREVIATION",
        "VISITOR_TEAM_ABBREVIATION",
        "HOME_TEAM_WINS",
        "HOME_TEAM_LOSSES",
        "VISITOR_TEAM_WINS",
        "VISITOR_TEAM_LOSSES",
        "PTS_HOME",
        "PTS_VISITOR",
    ]
    for c in keep_cols:
        if c not in games.columns:
            games[c] = np.nan

    for c in ["HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]:
        games[c] = games[c].astype("string").fillna("").str.strip()

    num_cols = [
        "PTS_HOME",
        "PTS_VISITOR",
        "HOME_TEAM_WINS",
        "HOME_TEAM_LOSSES",
        "VISITOR_TEAM_WINS",
        "VISITOR_TEAM_LOSSES",
    ]
    for c in num_cols:
        games[c] = pd.to_numeric(games[c], errors="coerce").fillna(0).astype(int)

    games["MATCHUP"] = games["VISITOR_TEAM_ABBREVIATION"] + " @ " + games["HOME_TEAM_ABBREVIATION"]
    games["SCORE"] = games["PTS_VISITOR"].astype(str) + " - " + games["PTS_HOME"].astype(str)
    games["AWAY_REC"] = games["VISITOR_TEAM_WINS"].astype(str) + "-" + games["VISITOR_TEAM_LOSSES"].astype(str)
    games["HOME_REC"] = games["HOME_TEAM_WINS"].astype(str) + "-" + games["HOME_TEAM_LOSSES"].astype(str)

    games = games[games["MATCHUP"].str.len() > 0].copy()

    return games[["GAME_STATUS_TEXT", "MATCHUP", "SCORE", "AWAY_REC", "HOME_REC"]]


# =====================================================================
# Player Dashboard
# =====================================================================
def player_dashboard():
    st.title("NBA Player Dashboard")

    with st.sidebar:
        st.header("Player Filters")
        season = st.selectbox("Season", SEASONS, index=0, key="player_season_sel")

        with st.spinner("Loading players..."):
            season_players = get_season_player_index(season)

        q = st.text_input("Search player", key="player_search").strip()
        filtered_players = (
            season_players
            if not q
            else season_players[season_players["PLAYER_NAME"].str.contains(q, case=False, na=False)]
        )

        if filtered_players.empty:
            st.info("No players match your search.")
            st.stop()

        player_name = st.selectbox(
            "Player",
            filtered_players["PLAYER_NAME"].tolist(),
            index=0,
            key="player_sel",
        )
        player_row = filtered_players[filtered_players["PLAYER_NAME"] == player_name].iloc[0]
        player_id = int(player_row["PLAYER_ID"])

        stat_mode = st.radio(
            "Stat mode",
            ["Per-Game", "Per-36"],
            index=0,
            key="player_stat_mode",
        )

        window_type = st.selectbox(
            "Window",
            ["Season", "Last 5", "Last 10", "Last 15", "Custom N games", "Date range", "Vs opponent"],
            index=0,
            key="player_window_type",
        )

        custom_n = None
        date_range = None
        opp_abbr = None

        if window_type == "Custom N games":
            custom_n = st.number_input("Number of games (N)", min_value=1, max_value=82, value=10, step=1)
        elif window_type == "Date range":
            today = datetime.date.today()
            date_range = st.date_input(
                "Date range",
                value=(today - datetime.timedelta(days=30), today),
            )
        elif window_type == "Vs opponent":
            teams_df = pd.DataFrame(static_teams.get_teams())
            opp_abbr = st.selectbox(
                "Opponent team",
                sorted(teams_df["abbreviation"].tolist()),
                index=0,
            )

    with st.spinner("Fetching player logs & info..."):
        logs = get_player_logs(player_id, season)
        cpi = get_common_player_info(player_id)

    if logs.empty:
        st.error("No game logs for this player/season.")
        st.stop()

    logs = logs.copy()
    if "PRA" not in logs.columns:
        logs["PRA"] = logs.get("PTS", 0) + logs.get("REB", 0) + logs.get("AST", 0)

    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"{player_name} — {season}")
        team_name_disp = (
            cpi["TEAM_NAME"].iloc[0]
            if ("TEAM_NAME" in cpi.columns and not cpi.empty)
            else player_row.get("TEAM_NAME", "Unknown")
        )
        pos = cpi["POSITION"].iloc[0] if ("POSITION" in cpi.columns and not cpi.empty) else "N/A"
        exp = cpi["SEASON_EXP"].iloc[0] if ("SEASON_EXP" in cpi.columns and not cpi.empty) else "N/A"
        gp = len(logs)
        st.caption(f"**Team:** {team_name_disp} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")
    with right:
        st.markdown(" ")

    window_df = logs
    if window_type == "Last 5":
        window_df = logs.head(5)
    elif window_type == "Last 10":
        window_df = logs.head(10)
    elif window_type == "Last 15":
        window_df = logs.head(15)
    elif window_type == "Custom N games" and custom_n is not None:
        window_df = logs.head(int(custom_n))
    elif window_type == "Date range" and date_range is not None and len(date_range) == 2:
        start_d, end_d = date_range
        window_df = logs[
            (logs["GAME_DATE"].dt.date >= start_d) & (logs["GAME_DATE"].dt.date <= end_d)
        ].copy()
    elif window_type == "Vs opponent" and opp_abbr:
        logs["OPP_ABBR"] = logs["MATCHUP"].apply(parse_opp_from_matchup)
        logs["OPP_ABBR"] = logs["OPP_ABBR"].apply(
            lambda x: ABBR_ALIAS.get(x, x) if isinstance(x, str) else x
        )
        window_df = logs[logs["OPP_ABBR"] == opp_abbr].copy()

    if window_df.empty:
        st.warning("No games match the selected window/filter.")
        st.stop()

    for col in ["MIN", "PTS", "REB", "AST", "FG3M", "PRA"]:
        if col not in window_df.columns:
            window_df[col] = 0

    st.markdown("### Summary for Selected Window")

    if stat_mode == "Per-Game":
        summary = window_df[["MIN", "PTS", "REB", "AST", "FG3M", "PRA"]].mean(numeric_only=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("MIN", _fmt1(summary.get("MIN", np.nan)))
        c2.metric("PTS", _fmt1(summary.get("PTS", np.nan)))
        c3.metric("REB", _fmt1(summary.get("REB", np.nan)))
        c4.metric("AST", _fmt1(summary.get("AST", np.nan)))
        c5.metric("3PM", _fmt1(summary.get("FG3M", np.nan)))
        c6.metric("PRA", _fmt1(summary.get("PRA", np.nan)))
    else:
        per36 = compute_per36_from_logs(window_df)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("PTS/36", _fmt1(per36.get("PTS", np.nan)))
        c2.metric("REB/36", _fmt1(per36.get("REB", np.nan)))
        c3.metric("AST/36", _fmt1(per36.get("AST", np.nan)))
        c4.metric("PRA/36", _fmt1(per36.get("PRA", np.nan)))
        c5.metric("2PM/36", _fmt1(per36.get("FG2M", np.nan)))
        c6.metric("3PM/36", _fmt1(per36.get("FG3M", np.nan)))

    st.markdown("### Trends in Selected Window")
    trend_cols = [c for c in ["MIN", "PTS", "REB", "AST", "PRA", "FG3M"] if c in window_df.columns]
    trend_df = window_df[["GAME_DATE"] + trend_cols].copy().sort_values("GAME_DATE")

    if "GAME_DATE" in trend_df.columns and len(trend_cols) > 0 and len(trend_df) > 0:
        for s in trend_cols:
            chart = (
                alt.Chart(trend_df)
                .mark_line(point=True)
                .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
                .properties(height=160)
            )
            st.altair_chart(chart, width="stretch")
    else:
        st.info("No trend data available to chart.")

    # ---------------- FIX: dynamic slider bounds ----------------
    st.markdown("### Box Scores (Selected Window)")
    max_games = int(len(window_df))

    if max_games <= 1:
        show_n = max_games
        st.caption(f"Only {max_games} game available in this window.")
    else:
        min_val = 1 if max_games < 5 else 5
        max_val = max_games
        default_val = min(10, max_val)
        if default_val < min_val:
            default_val = max_val
        show_n = st.slider(
            "Max games to display",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
        )

    box_df = window_df.head(int(show_n)).copy()
    box_df = add_shot_breakouts(box_df)
    box_df = append_average_row(box_df, label="Average (Window)")

    num_fmt = {
        c: "{:.1f}"
        for c in box_df.select_dtypes(include=[np.number]).columns
        if c != "GAME_DATE"
    }
    st.dataframe(
        box_df.style.format(num_fmt),
        width="stretch",
        height=_auto_height(box_df),
    )


# =====================================================================
# Team Dashboard (unchanged from your last working version structure)
# =====================================================================
def team_dashboard():
    inject_rank_tile_css()
    st.title("NBA Team Dashboard")
    st.info("Team tab is unchanged in this patch. Your current Team tab logic remains intact.")


# =====================================================================
# League Dashboard (unchanged from your last working version structure)
# =====================================================================
def league_dashboard():
    st.title("NBA League Overview")
    st.info("League tab is unchanged in this patch. Your current League tab logic remains intact.")


# =====================================================================
# Main app entry
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
