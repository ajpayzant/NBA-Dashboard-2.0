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
        end = datetime.datetime.utcnow().year

    def lab(y):
        return f"{y}-{str((y + 1) % 100).zfill(2)}"

    return [lab(y) for y in range(end, start - 1, -1)]


SEASONS = _season_labels(2015, datetime.datetime.utcnow().year)


def _prev_season_label(season_label: str) -> str:
    try:
        y0 = int(season_label.split("-")[0])
        return f"{y0 - 1}-{str(y0 % 100).zfill(2)}"
    except Exception:
        return season_label


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


def per36_from_career_totals(career_df: pd.DataFrame) -> dict:
    keys = ["PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "FTM", "OREB", "DREB"]
    if career_df is None or career_df.empty:
        return {k: np.nan for k in keys}
    need = ["MIN", "PTS", "REB", "AST", "FGM", "FG3M", "FTM", "OREB", "DREB"]
    career_df = career_df.copy()
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
    sum_oreb = career_df["OREB"].sum()
    sum_dreb = career_df["DREB"].sum()
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


def _series_from_dict(d: dict, name: str) -> pd.Series:
    s = pd.Series(d, dtype="float64")
    s.name = name
    return s


def _rank_tile(col, label, value, rank, total=30, pct=False, decimals=1):
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
        val_txt = f"{float(value) * 100:.{decimals}f}%" if pd.notna(value) else "—"
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


def normalize_abbr(abbr: str | None) -> str | None:
    if not isinstance(abbr, str) or not abbr:
        return None
    a = abbr.upper().strip()
    return ABBR_ALIAS.get(a, a)


def resolve_team_abbrev(team_name: str) -> str | None:
    if isinstance(team_name, str):
        cf = team_name.casefold().strip()
        if cf in BY_FULL_CF:
            return normalize_abbr(BY_FULL_CF[cf])
        if cf in NICK_CF:
            return normalize_abbr(NICK_CF[cf])
    return None


def resolve_team_id_from_name(team_name: str) -> int | None:
    return TEAMID_BY_FULL.get(team_name)


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
    """League schedule / scoreboard for a given date."""
    # IMPORTANT: ScoreboardV2 expects MM/DD/YYYY
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


@st.cache_data(ttl=3600, show_spinner=True)
def get_league_team_base_pergame(season: str) -> pd.DataFrame:
    """Traditional team stats, PerGame."""
    try:
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
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    for c in df.columns:
        if c not in ("TEAM_NAME", "TEAM_ABBREVIATION"):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=True)
def get_league_team_advanced(season: str) -> pd.DataFrame:
    """League-wide advanced team table for OFF/DEF/NET/PACE."""
    try:
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
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    wanted = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[wanted].copy()
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=True)
def get_standings_full(season: str) -> pd.DataFrame:
    """Simplified standings: TEAM_ID, TEAM_NAME, TEAM_ABBREVIATION, CONF, W, L, WIN_PCT."""
    try:
        frames = _retry_api(
            leaguestandingsv3.LeagueStandingsV3,
            {
                "league_id": "00",
                "season": season,
                "season_type": "Regular Season",
            },
            timeout=REQUEST_TIMEOUT,
            retries=2,
            sleep=1.0,
        )
    except Exception:
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()

    st_df = frames[0].copy()
    if st_df.empty:
        return pd.DataFrame()

    cols = st_df.columns

    def col_like(options):
        for o in options:
            if o in cols:
                return o
        return None

    id_col = col_like(["TeamID", "TEAM_ID", "teamId"])
    team_name_col = col_like(["TeamName", "TEAM_NAME"])
    team_abbr_col = col_like(["TeamSlug", "TeamAbbreviation", "TEAM_ABBREVIATION"])
    conf_col = col_like(["Conference", "TEAM_CONFERENCE"])
    wins_col = col_like(["WINS", "Wins"])
    losses_col = col_like(["LOSSES", "Losses"])
    winpct_col = col_like(["WinPCT", "WIN_PCT"])

    required = [id_col, team_name_col, team_abbr_col, conf_col, wins_col, losses_col]
    if any(c is None for c in required):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "TEAM_ID": pd.to_numeric(st_df[id_col], errors="coerce"),
            "TEAM_NAME": st_df[team_name_col].astype(str),
            "TEAM_ABBREVIATION": st_df[team_abbr_col].astype(str),
            "CONF": st_df[conf_col].astype(str),
            "W": pd.to_numeric(st_df[wins_col], errors="coerce").fillna(0).astype(int),
            "L": pd.to_numeric(st_df[losses_col], errors="coerce").fillna(0).astype(int),
        }
    )
    if winpct_col:
        out["WIN_PCT"] = pd.to_numeric(st_df[winpct_col], errors="coerce")
    else:
        out["WIN_PCT"] = out["W"] / (out["W"] + out["L"]).replace(0, np.nan)
    out["WIN_PCT"] = out["WIN_PCT"].round(3)

    return out.dropna(subset=["TEAM_ID"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=True)
def get_league_team_summary(season: str) -> pd.DataFrame:
    """Combined standings + team PTS/REB/AST + OFF/DEF/NET/PACE."""
    standings = get_standings_full(season)
    base = get_league_team_base_pergame(season)
    adv = get_league_team_advanced(season)

    if standings.empty or base.empty:
        return pd.DataFrame()

    # Align keys
    base_min = base[
        [
            "TEAM_ID",
            "TEAM_NAME",
            "TEAM_ABBREVIATION",
            "PTS",
            "REB",
            "AST",
        ]
    ].copy()
    adv_min = adv[
        [
            "TEAM_ID",
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "PACE",
        ]
    ].copy()

    merged = standings.merge(
        base_min,
        on=["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"],
        how="left",
    )
    merged = merged.merge(adv_min, on="TEAM_ID", how="left")

    merged["RANK"] = merged["WIN_PCT"].rank(ascending=False, method="min").astype(int)
    merged = merged.sort_values("RANK").reset_index(drop=True)
    return merged


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=True)
def get_league_players_stats(season: str, per_mode: str) -> pd.DataFrame:
    """League-wide player stats, per_mode is 'PerGame' or 'Per36'."""
    try:
        frames = _retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed=per_mode,
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)


# =====================================================================
# Player Dashboard
# =====================================================================
def player_dashboard():
    st.title("NBA Player Dashboard")

    # Sidebar: filters
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

    # Load logs and info
    with st.spinner("Fetching player logs & info..."):
        logs = get_player_logs(player_id, season)
        career_df = get_player_career(player_id)
        cpi = get_common_player_info(player_id)

    if logs.empty:
        st.error("No game logs for this player/season.")
        st.stop()

    logs = logs.copy()
    if "PRA" not in logs.columns:
        logs["PRA"] = logs.get("PTS", 0) + logs.get("REB", 0) + logs.get("AST", 0)

    # Header
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

    # Apply window filter
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

    # Summary metrics
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

    # Trend charts for window
    st.markdown("### Trends in Selected Window")
    trend_cols = [c for c in ["MIN", "PTS", "REB", "AST", "PRA", "FG3M"] if c in window_df.columns]
    trend_df = window_df[["GAME_DATE"] + trend_cols].copy()
    trend_df = trend_df.sort_values("GAME_DATE")

    if "GAME_DATE" in trend_df.columns and len(trend_cols) > 0 and len(trend_df) > 0:
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

    # Box score table for window
    st.markdown("### Box Scores (Selected Window)")
    max_games = len(window_df)
    show_n = st.slider("Max games to display", min_value=5, max_value=max_games, value=min(10, max_games))

    box_df = window_df.head(show_n).copy()
    box_df = add_shot_breakouts(box_df)
    box_df = append_average_row(box_df, label="Average (Window)")

    num_fmt = {
        c: "{:.1f}"
        for c in box_df.select_dtypes(include=[np.number]).columns
        if c != "GAME_DATE"
    }
    st.dataframe(
        box_df.style.format(num_fmt),
        use_container_width=True,
        height=_auto_height(box_df),
    )

    # Advanced comparison (career vs windows)
    st.markdown("### Comparison: Career vs Season vs Recent")

    def career_per_game(career_df, cols=("MIN", "PTS", "REB", "AST", "FG3M")):
        if career_df.empty or "GP" not in career_df.columns:
            return pd.Series({c: np.nan for c in cols}, dtype=float)
        needed = list(set(cols) | {"GP"})
        for c in needed:
            if c not in career_df.columns:
                career_df[c] = 0
        total_gp = pd.to_numeric(career_df["GP"], errors="coerce").sum()
        if total_gp == 0:
            return pd.Series({c: np.nan for c in cols}, dtype=float)
        out = {c: pd.to_numeric(career_df[c], errors="coerce").sum() / total_gp for c in cols}
        return pd.Series(out).astype(float)

    for col in ["MIN", "PTS", "REB", "AST", "FG3M"]:
        if col not in logs.columns:
            logs[col] = 0

    metrics_order = ["MIN", "PTS", "REB", "AST", "FG3M"]
    career_pg = career_per_game(career_df, cols=metrics_order)

    prev_season = _prev_season_label(season)
    prev_logs = get_player_logs(player_id, prev_season)
    prev_logs = prev_logs.copy()
    for col in metrics_order:
        if col not in prev_logs.columns:
            prev_logs[col] = 0
    prev_season_pg = prev_logs[metrics_order].mean(numeric_only=True)

    current_season_pg = logs[metrics_order].mean(numeric_only=True)
    l5_pg = logs[metrics_order].head(5).mean(numeric_only=True)
    l20_pg = logs[metrics_order].head(20).mean(numeric_only=True)

    cmp_df = pd.DataFrame(
        {
            "Career Avg": career_pg,
            "Prev Season Avg": prev_season_pg,
            "Current Season Avg": current_season_pg,
            "Last 5 Avg": l5_pg,
            "Last 20 Avg": l20_pg,
        },
        index=metrics_order,
    ).round(2)

    st.dataframe(
        cmp_df.style.format(numeric_format_map(cmp_df)),
        use_container_width=True,
        height=_auto_height(cmp_df),
    )


# =====================================================================
# Team Dashboard
# =====================================================================
def team_dashboard():
    inject_rank_tile_css()
    st.title("NBA Team Dashboard")

    @st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
    def get_teams_df():
        t = pd.DataFrame(static_teams.get_teams())
        t = t.rename(
            columns={
                "id": "TEAM_ID",
                "full_name": "TEAM_NAME",
                "abbreviation": "TEAM_ABBREVIATION",
            }
        )
        t["TEAM_ID"] = t["TEAM_ID"].astype(int)
        return t[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]]

    @st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=True)
    def fetch_league_team_opponent(season: str) -> pd.DataFrame:
        try:
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
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        cols = ["TEAM_ID"] + [c for c in df.columns if c.startswith("OPP_")]
        for c in cols:
            if c != "TEAM_ID" and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[cols].reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=True)
    def fetch_league_players_pg(season: str, last_n_games: int) -> pd.DataFrame:
        try:
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
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=True)
    def fetch_league_players_per36(season: str, last_n_games: int) -> pd.DataFrame:
        try:
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
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    def _rank_series(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan] * len(df))
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
        df = df.copy()
        for c in colmap.keys():
            if c not in df.columns:
                df[c] = np.nan
        out = df[list(colmap.keys())].copy()
        out.columns = list(colmap.values())
        return out

    def _auto_height_local(df: pd.DataFrame, row_px=34, header_px=38, max_px=900):
        rows = max(len(df), 1)
        return min(max_px, header_px + row_px * rows + 8)

    def _num_fmt_map(df: pd.DataFrame):
        fmts = {}
        for c in df.columns:
            if c in ("TEAM", "PLAYER_NAME"):
                continue
            fmts[c] = "{:.1f}"
        return fmts

    # Sidebar: team selection
    with st.sidebar:
        st.header("Team Filters")
        season = st.selectbox(
            "Season",
            _season_labels(2015, dt.datetime.utcnow().year),
            index=0,
            key="team_season_sel",
        )
        teams_df = get_teams_df()
        team_name = st.selectbox("Team", sorted(teams_df["TEAM_NAME"].tolist()))
        team_row = teams_df[teams_df["TEAM_NAME"] == team_name].iloc[0]
        team_id = int(team_row["TEAM_ID"])
        team_abbr = team_row["TEAM_ABBREVIATION"]

    # Load team-level stats
    with st.spinner("Loading league team stats..."):
        trad = get_league_team_base_pergame(season)
        adv = get_league_team_advanced(season)
        opp = fetch_league_team_opponent(season)

    if trad.empty or adv.empty:
        st.error("Could not load team stats. Try refreshing or changing the season.")
        st.stop()

    TRAD_WANTED = [
        "TEAM_ID",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "GP",
        "W",
        "L",
        "W_PCT",
        "MIN",
        "PTS",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PLUS_MINUS",
    ]
    ADV_WANTED = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]

    def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    trad_g = _ensure_cols(trad, TRAD_WANTED)[TRAD_WANTED].copy()
    adv_g = _ensure_cols(adv, ADV_WANTED)[ADV_WANTED].copy()

    merged = pd.merge(trad_g, adv_g, on="TEAM_ID", how="left")
    if not opp.empty:
        merged = pd.merge(merged, opp, on="TEAM_ID", how="left")

    def _safe_rank(col, ascending):
        return _rank_series(merged, col, ascending=ascending)

    ranks = pd.DataFrame({"TEAM_ID": merged["TEAM_ID"]})
    ranks["PTS"] = _safe_rank("PTS", ascending=False)
    ranks["NET_RATING"] = _safe_rank("NET_RATING", ascending=False)
    ranks["OFF_RATING"] = _safe_rank("OFF_RATING", ascending=False)
    ranks["DEF_RATING"] = _safe_rank("DEF_RATING", ascending=True)
    ranks["PACE"] = _safe_rank("PACE", ascending=False)
    ranks["FGA"] = _safe_rank("FGA", ascending=False)
    ranks["FG_PCT"] = _safe_rank("FG_PCT", ascending=False)
    ranks["FG3A"] = _safe_rank("FG3A", ascending=False)
    ranks["FG3_PCT"] = _safe_rank("FG3_PCT", ascending=False)
    ranks["FTA"] = _safe_rank("FTA", ascending=False)
    ranks["FT_PCT"] = _safe_rank("FT_PCT", ascending=False)
    ranks["OREB"] = _safe_rank("OREB", ascending=False)
    ranks["DREB"] = _safe_rank("DREB", ascending=False)
    ranks["REB"] = _safe_rank("REB", ascending=False)
    ranks["AST"] = _safe_rank("AST", ascending=False)
    ranks["TOV"] = _safe_rank("TOV", ascending=True)
    ranks["STL"] = _safe_rank("STL", ascending=False)
    ranks["BLK"] = _safe_rank("BLK", ascending=False)
    ranks["PF"] = _safe_rank("PF", ascending=True)
    ranks["PLUS_MINUS"] = _safe_rank("PLUS_MINUS", ascending=False)

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

    st.subheader(f"{tr['TEAM_NAME']} — {season}")

    c_rec, _, _, _, _ = st.columns(5)
    c_rec.metric("Record", record)

    def tile_row(items):
        cols = st.columns(len(items))
        for (label, key, pct_flag), col in zip(items, cols):
            rank_key = key
            _rank_tile(col, label, tr.get(key), rr.get(rank_key), total=n_teams, pct=pct_flag)

    tile_row(
        [
            ("PTS", "PTS", False),
            ("NET Rating", "NET_RATING", False),
            ("OFF Rating", "OFF_RATING", False),
            ("DEF Rating", "DEF_RATING", False),
            ("PACE", "PACE", False),
        ]
    )
    tile_row(
        [
            ("FGA", "FGA", False),
            ("FG%", "FG_PCT", True),
            ("3PA", "FG3A", False),
            ("3P%", "FG3_PCT", True),
            ("FTA", "FTA", False),
        ]
    )
    tile_row(
        [
            ("FT%", "FT_PCT", True),
            ("OREB", "OREB", False),
            ("DREB", "DREB", False),
            ("REB", "REB", False),
            ("AST", "AST", False),
        ]
    )
    tile_row(
        [
            ("TOV", "TOV", False),
            ("STL", "STL", False),
            ("BLK", "BLK", False),
            ("PF", "PF", False),
            ("+/-", "PLUS_MINUS", False),
        ]
    )

    st.caption(
        "Ranks are relative to all NBA teams (1 = best). Tile color and arrow reflect tier (top/middle/bottom)."
    )

    st.markdown("### Opponent Averages Allowed (Per-Game)")

    def _opp_row_team(cols_labels):
        cols = st.columns(len(cols_labels))
        for (api_col, label), col in zip(cols_labels, cols):
            val = tr.get(api_col, np.nan)
            rank = tr.get(f"{api_col}_RANK", np.nan)
            pct = api_col.endswith("_PCT")
            _rank_tile(col, label, val, rank, total=n_teams, pct=pct)

    _opp_row_team(
        [
            ("OPP_PTS", "Opp PTS"),
            ("OPP_FGA", "Opp FGA"),
            ("OPP_FG_PCT", "Opp FG%"),
            ("OPP_FG3A", "Opp 3PA"),
            ("OPP_FG3_PCT", "Opp 3P%"),
        ]
    )
    _opp_row_team(
        [
            ("OPP_FTA", "Opp FTA"),
            ("OPP_FT_PCT", "Opp FT%"),
            ("OPP_OREB", "Opp OREB"),
            ("OPP_DREB", "Opp DREB"),
            ("OPP_REB", "Opp REB"),
        ]
    )
    _opp_row_team(
        [
            ("OPP_AST", "Opp AST"),
            ("OPP_TOV", "Opp TOV"),
            ("OPP_STL", "Opp STL"),
            ("OPP_BLK", "Opp BLK"),
            ("OPP_PF", "Opp PF"),
        ]
    )

    # Roster section: single interactive roster
    st.markdown("### Roster")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        roster_mode = st.radio(
            "Roster stat view",
            ["Per-Game", "Per-36"],
            index=0,
            key="roster_stat_mode",
        )
    with col_r2:
        roster_window = st.selectbox(
            "Roster window",
            ["Season", "Last 5", "Last 10", "Last 15", "Last 20"],
            index=0,
            key="roster_window_sel",
        )

    last_n_map = {"Season": 0, "Last 5": 5, "Last 10": 10, "Last 15": 15, "Last 20": 20}
    last_n = last_n_map.get(roster_window, 0)

    with st.spinner("Loading roster stats..."):
        if roster_mode == "Per-Game":
            roster_df = fetch_league_players_pg(season, last_n)
        else:
            roster_df = fetch_league_players_per36(season, last_n)

    if roster_df.empty:
        st.info("No roster data available for this team/window.")
        return

    roster_df = roster_df[roster_df["TEAM_ID"] == team_id].copy()
    if roster_df.empty:
        st.info("No players for this team in the selected window.")
        return

    roster_tbl = _select_roster_columns(roster_df)
    if "MIN" in roster_tbl.columns:
        roster_tbl = roster_tbl.sort_values("MIN", ascending=False).reset_index(drop=True)

    st.dataframe(
        roster_tbl.style.format(_num_fmt_map(roster_tbl)),
        use_container_width=True,
        height=_auto_height_local(roster_tbl),
    )

    st.caption(
        "Roster stats from LeagueDashPlayerStats with per-mode and last-n-games filters. "
        "Use the controls above to switch between per-game and per-36, and to change the recency window."
    )


# =====================================================================
# League Dashboard
# =====================================================================
def league_dashboard():
    inject_rank_tile_css()
    st.title("NBA League Overview")

    with st.sidebar:
        st.header("League Filters")
        season = st.selectbox(
            "Season",
            SEASONS,
            index=0,
            key="league_season_sel",
        )
        default_date = datetime.date.today()
        date_choice = st.date_input(
            "Date for schedule",
            value=default_date,
            key="league_date",
        )

    teams_tab, players_tab = st.tabs(["Teams", "Players"])

    # ---------------- Teams tab ----------------
    with teams_tab:
        st.subheader("Daily Schedule & Scoreboard")
        with st.spinner("Loading games for selected date..."):
            sched = get_daily_scoreboard(date_choice)

        if sched.empty:
            st.info("No games found or unable to load scoreboard for this date.")
        else:
            st.dataframe(
                sched,
                use_container_width=True,
                height=_auto_height(sched),
            )

        st.subheader("League Standings & Team Stats")

        with st.spinner("Loading league team summary..."):
            team_summary = get_league_team_summary(season)

        if team_summary.empty:
            st.info("Unable to load league standings / team stats for this season.")
        else:
            conf_filter = st.selectbox(
                "Conference filter",
                ["All", "East", "West"],
                index=0,
                key="league_conf_filter",
            )

            df = team_summary.copy()
            df["CONF_UP"] = df["CONF"].str.upper()

            if conf_filter == "East":
                df = df[df["CONF_UP"].str.contains("EAST")]
            elif conf_filter == "West":
                df = df[df["CONF_UP"].str.contains("WEST")]

            df = df.sort_values("RANK").reset_index(drop=True)

            display_cols = [
                "RANK",
                "TEAM_ABBREVIATION",
                "TEAM_NAME",
                "CONF",
                "W",
                "L",
                "WIN_PCT",
                "PTS",
                "REB",
                "AST",
                "OFF_RATING",
                "DEF_RATING",
                "NET_RATING",
                "PACE",
            ]
            df_disp = df[display_cols].rename(
                columns={
                    "TEAM_ABBREVIATION": "TEAM",
                    "WIN_PCT": "WIN%",
                    "OFF_RATING": "OFF_RTG",
                    "DEF_RATING": "DEF_RTG",
                    "NET_RATING": "NET_RTG",
                }
            )
            st.dataframe(
                df_disp.style.format(
                    {
                        "WIN%": "{:.3f}",
                        "PTS": "{:.1f}",
                        "REB": "{:.1f}",
                        "AST": "{:.1f}",
                        "OFF_RTG": "{:.1f}",
                        "DEF_RTG": "{:.1f}",
                        "NET_RTG": "{:.1f}",
                        "PACE": "{:.1f}",
                    }
                ),
                use_container_width=True,
                height=_auto_height(df_disp),
            )

    # ---------------- Players tab ----------------
    with players_tab:
        st.subheader("Leaguewide Player Leaders")

        with st.spinner("Loading league player stats..."):
            per_mode_choice = st.radio(
                "Stat mode",
                ["Per-Game", "Per-36"],
                index=0,
                key="league_player_mode",
            )
            per_mode = "PerGame" if per_mode_choice == "Per-Game" else "Per36"
            top_n = st.slider("Top N players", min_value=5, max_value=50, value=15, step=1)
            min_gp = st.slider("Minimum games played", min_value=1, max_value=82, value=10, step=1)

            league_players = get_league_players_stats(season, per_mode)

        if league_players.empty:
            st.info("Unable to load league-wide player stats.")
            return

        df = league_players.copy()
        df["GP"] = pd.to_numeric(df["GP"], errors="coerce").fillna(0).astype(int)
        df = df[df["GP"] >= min_gp].copy()

        stat_categories = [
            ("PTS", "Points"),
            ("REB", "Rebounds"),
            ("AST", "Assists"),
            ("FG3M", "3-Pointers Made"),
            ("OREB", "Offensive Rebounds"),
            ("DREB", "Defensive Rebounds"),
            ("STL", "Steals"),
            ("BLK", "Blocks"),
            ("TOV", "Turnovers"),
        ]

        for stat, label in stat_categories:
            if stat not in df.columns:
                continue
            st.markdown(f"#### Top {top_n} — {label} ({per_mode_choice})")
            leaders = (
                df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", stat]]
                .dropna(subset=[stat])
                .copy()
            )
            leaders[stat] = pd.to_numeric(leaders[stat], errors="coerce")
            leaders = leaders.sort_values(stat, ascending=False).head(top_n)
            leaders = leaders.rename(
                columns={
                    "TEAM_ABBREVIATION": "TEAM",
                }
            )
            st.dataframe(
                leaders.style.format(
                    {
                        "MIN": "{:.1f}",
                        stat: "{:.1f}",
                    }
                ),
                use_container_width=True,
                height=_auto_height(leaders),
            )

        st.caption(
            "Player leader tables use LeagueDashPlayerStats with your selected per-mode and filters. "
            "Sort any table by clicking column headers to explore further."
        )


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
