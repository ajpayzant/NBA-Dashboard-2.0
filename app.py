from __future__ import annotations

import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st

from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.static import players as nba_players_static

from src.cache_store import get_or_refresh
from src.utils import (
    season_labels,
    today_la_date,
    team_id_maps,
    minutes_str_to_float,
    add_basic_shooting_breakouts,
)
from src.loaders import (
    fetch_daily_scoreboard,
    fetch_league_team_summary,
    fetch_league_player_stats,
    fetch_team_game_log,
    fetch_team_roster,
    fetch_boxscore_team_players,
    fetch_player_game_log,
    fetch_player_career,
    compute_rotation_minutes_table,
)

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="NBA Dashboard 2.0", layout="wide")


# -----------------------------
# Constants / TTLs
# -----------------------------
TTL_SCOREBOARD = 60 * 10         # 10 minutes
TTL_LEAGUE = 60 * 60 * 12        # 12 hours
TTL_TEAM = 60 * 60 * 12          # 12 hours
TTL_PLAYER = 60 * 60 * 12        # 12 hours


# -----------------------------
# Helpers
# -----------------------------
def _season_default():
    return season_labels(2015)[-1]


def _safe_date_range(date_val):
    """
    Streamlit date_input can return:
      - a single date
      - a tuple (start, end)
    Normalize to (start, end).
    """
    if isinstance(date_val, tuple) or isinstance(date_val, list):
        if len(date_val) == 2:
            return date_val[0], date_val[1]
        if len(date_val) == 1:
            return date_val[0], date_val[0]
    return date_val, date_val


def _header_bar():
    st.markdown("## NBA Dashboard 2.0")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    with c1:
        season = st.selectbox("Season", season_labels(2015), index=len(season_labels(2015)) - 1)

    with c2:
        date_choice = st.date_input("Scoreboard date", value=today_la_date())

    with c3:
        force_refresh = st.checkbox("Force refresh (re-pull API now)", value=False)

    with c4:
        st.caption("Parquet cache reduces repeated NBA API calls when changing filters.")

    return season, date_choice, force_refresh


def _team_list():
    t = nba_teams_static.get_teams()
    # Sort by full name for user selection
    t = sorted(t, key=lambda x: x["full_name"])
    return t


def _player_list():
    # Static list can be huge; we’ll use it for lookup, then filter in UI
    return nba_players_static.get_players()


def _fmt_pct(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return ""


def _leaders_table(df: pd.DataFrame, stat_col: str, min_gp: int, top_n: int = 15):
    if df is None or df.empty or stat_col not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    if "GP" in d.columns:
        d["GP"] = pd.to_numeric(d["GP"], errors="coerce")
        d = d[d["GP"].fillna(0) >= min_gp].copy()

    d[stat_col] = pd.to_numeric(d[stat_col], errors="coerce")
    d = d.sort_values(stat_col, ascending=False).head(top_n)

    keep = []
    for c in ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", stat_col]:
        if c in d.columns:
            keep.append(c)
    out = d[keep].copy()
    if "TEAM_ABBREVIATION" in out.columns:
        out = out.rename(columns={"TEAM_ABBREVIATION": "Team"})
    if "PLAYER_NAME" in out.columns:
        out = out.rename(columns={"PLAYER_NAME": "Player"})
    return out


# -----------------------------
# League tab
# -----------------------------
def league_tab(season: str, date_choice: dt.date, force_refresh: bool):
    sub_team, sub_players = st.tabs(["Team (Leaguewide)", "Players (Leaders)"])

    # ---- Team subtab
    with sub_team:
        st.markdown("### Schedule / Scoreboard")

        sched_res = get_or_refresh(
            key=f"scoreboard__{date_choice.isoformat()}",
            ttl_seconds=TTL_SCOREBOARD,
            fetch_fn=lambda: fetch_daily_scoreboard(date_choice),
            force_refresh=force_refresh,
        )

        sched = sched_res.df
        if sched is None or sched.empty:
            st.info("No games found for this date.")
        else:
            # clean view
            view = sched[["MATCHUP", "STATUS", "SCORE"]].copy()
            st.dataframe(view, width="stretch", hide_index=True)

        st.markdown("### Standings + Team Stats (sortable)")
        team_res = get_or_refresh(
            key=f"league_team_summary__{season}",
            ttl_seconds=TTL_LEAGUE,
            fetch_fn=lambda: fetch_league_team_summary(season)[0],
            force_refresh=force_refresh,
        )
        opp_res = get_or_refresh(
            key=f"league_opp_summary__{season}",
            ttl_seconds=TTL_LEAGUE,
            fetch_fn=lambda: fetch_league_team_summary(season)[1],
            force_refresh=force_refresh,
        )

        team_table = team_res.df
        opp_table = opp_res.df

        if team_table is None or team_table.empty:
            st.warning("Standings/team stats not available right now.")
        else:
            # remove columns you said you don’t want
            drop_cols = [c for c in ["TABLE_ID", "ABBR", "TEAM_ID", "TEAM_ABBREVIATION", "CONF Rank", "Rank"] if c in team_table.columns]
            # Keep TEAM_ABBREVIATION if you want it visible; you asked to remove ABBR but keep team name
            if "TEAM_ABBREVIATION" in team_table.columns:
                # keep, but rename shorter
                team_table = team_table.rename(columns={"TEAM_ABBREVIATION": "Team"})
            if "TEAM_NAME" in team_table.columns:
                team_table = team_table.rename(columns={"TEAM_NAME": "Team Name"})

            # Wins/Losses as ints
            for c in ["WINS", "LOSSES"]:
                if c in team_table.columns:
                    team_table[c] = pd.to_numeric(team_table[c], errors="coerce").fillna(0).astype(int)

            # Default sort by WIN_PCT desc if present
            if "WIN_PCT" in team_table.columns:
                team_table["WIN_PCT"] = pd.to_numeric(team_table["WIN_PCT"], errors="coerce")
                team_table = team_table.sort_values("WIN_PCT", ascending=False)

            # Apply safe drop after renames
            team_table_show = team_table.drop(columns=[c for c in drop_cols if c in team_table.columns], errors="ignore")
            st.dataframe(team_table_show, width="stretch", hide_index=True)

        st.markdown("### Opponent Allowed (sortable)")
        if opp_table is None or opp_table.empty:
            st.warning("Opponent team stats not available right now.")
        else:
            if "TEAM_ABBREVIATION" in opp_table.columns:
                opp_table = opp_table.rename(columns={"TEAM_ABBREVIATION": "Team"})
            st.dataframe(opp_table, width="stretch", hide_index=True)

    # ---- Players subtab
    with sub_players:
        st.markdown("### Leaguewide Player Leaders")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            per_mode = st.selectbox("Per Mode", ["PerGame", "Per36"], index=0)
        with c2:
            min_gp = st.slider("Min games played", min_value=1, max_value=30, value=10, step=1)
        with c3:
            top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)

        players_res = get_or_refresh(
            key=f"league_players__{season}__{per_mode}",
            ttl_seconds=TTL_LEAGUE,
            fetch_fn=lambda: fetch_league_player_stats(season, per_mode),
            force_refresh=force_refresh,
        )
        df = players_res.df

        if df is None or df.empty:
            st.warning("League player stats not available right now.")
            return

        stat_blocks = [
            ("PTS", "Points"),
            ("REB", "Rebounds"),
            ("AST", "Assists"),
            ("FG3M", "3PT Made"),
            ("OREB", "Off Reb"),
            ("DREB", "Def Reb"),
            ("STL", "Steals"),
            ("BLK", "Blocks"),
            ("TOV", "Turnovers"),
        ]

        # Render as multiple tables
        cols = st.columns(3)
        for i, (col, title) in enumerate(stat_blocks):
            with cols[i % 3]:
                st.markdown(f"**{title}**")
                tbl = _leaders_table(df, col, min_gp=min_gp, top_n=top_n)
                if tbl.empty:
                    st.caption("No data")
                else:
                    st.dataframe(tbl, width="stretch", hide_index=True)


# -----------------------------
# Team tab
# -----------------------------
def team_tab(season: str, force_refresh: bool):
    teams = _team_list()
    name_to_id = {t["full_name"]: int(t["id"]) for t in teams}
    name_to_abbr = {t["full_name"]: t["abbreviation"] for t in teams}

    team_name = st.selectbox("Select Team", list(name_to_id.keys()), index=0)
    team_id = name_to_id[team_name]
    team_abbr = name_to_abbr[team_name]

    # Team standings/summary (for record / last10)
    team_summary_res = get_or_refresh(
        key=f"league_team_summary__{season}",
        ttl_seconds=TTL_LEAGUE,
        fetch_fn=lambda: fetch_league_team_summary(season)[0],
        force_refresh=force_refresh,
    )
    team_summary = team_summary_res.df

    record_text = ""
    last10_text = ""
    if team_summary is not None and not team_summary.empty:
        # attempt match by abbreviation or name
        row = None
        if "Team" in team_summary.columns:
            r = team_summary[team_summary["Team"].astype("string") == team_abbr]
            if not r.empty:
                row = r.iloc[0]
        if row is None and "Team Name" in team_summary.columns:
            r = team_summary[team_summary["Team Name"].astype("string") == team_name]
            if not r.empty:
                row = r.iloc[0]
        if row is not None:
            w = int(row.get("WINS", 0)) if "WINS" in row else int(row.get("W", 0)) if "W" in row else 0
            l = int(row.get("LOSSES", 0)) if "LOSSES" in row else int(row.get("L", 0)) if "L" in row else 0
            record_text = f"**Record:** {w}-{l}"
            # Some standings datasets contain LAST10; if not, we compute from team game log below
            if "LAST10" in row:
                last10_text = f"**Last 10:** {row.get('LAST10')}"
    # fallback compute last10 from game log
    log_res = get_or_refresh(
        key=f"team_gamelog__{season}__{team_id}",
        ttl_seconds=TTL_TEAM,
        fetch_fn=lambda: fetch_team_game_log(team_id, season),
        force_refresh=force_refresh,
    )
    tlog = log_res.df

    if last10_text == "" and tlog is not None and not tlog.empty:
        # TeamGameLog includes WL column
        if "WL" in tlog.columns:
            wl = tlog["WL"].astype("string").head(10).tolist()
            w10 = sum(1 for x in wl if x == "W")
            l10 = sum(1 for x in wl if x == "L")
            last10_text = f"**Last 10:** {w10}-{l10}"

    st.markdown(f"### {team_name}")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"<div style='font-size:20px;'>{record_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px;'>{last10_text}</div>", unsafe_allow_html=True)

    with c2:
        if tlog is not None and not tlog.empty:
            view_cols = [c for c in ["GAME_DATE", "MATCHUP", "WL", "PTS"] if c in tlog.columns]
            st.dataframe(tlog[view_cols].head(10), width="stretch", hide_index=True)

    st.markdown("### Team Box Score (Select a Game)")
    if tlog is None or tlog.empty or "GAME_ID" not in tlog.columns:
        st.info("No game log available for box score selection.")
    else:
        tlog2 = tlog.copy()
        # Ensure game date parsed
        if "GAME_DATE" in tlog2.columns:
            tlog2["GAME_DATE"] = pd.to_datetime(tlog2["GAME_DATE"], errors="coerce")
            tlog2 = tlog2.sort_values("GAME_DATE", ascending=False)

        # Build select options
        def opt_label(r):
            gd = r["GAME_DATE"].date().isoformat() if pd.notna(r["GAME_DATE"]) else ""
            mu = r.get("MATCHUP", "")
            wl = r.get("WL", "")
            pts = r.get("PTS", "")
            return f"{gd} | {mu} | {wl} {pts}"

        tlog2["_label"] = tlog2.apply(opt_label, axis=1)
        options = list(tlog2["_label"].head(30))
        sel = st.selectbox("Choose game (last 30)", options, index=0)
        game_id = str(tlog2.loc[tlog2["_label"] == sel, "GAME_ID"].iloc[0])

        bs_res = get_or_refresh(
            key=f"boxscore__{game_id}",
            ttl_seconds=TTL_TEAM,
            fetch_fn=lambda: fetch_boxscore_team_players(game_id),
            force_refresh=force_refresh,
        )
        bs = bs_res.df
        if bs is None or bs.empty:
            st.warning("Box score not available (API may be rate limiting or the endpoint returned empty).")
        else:
            # Filter to the selected team only if TEAM_ABBREVIATION exists
            if "TEAM_ABBREVIATION" in bs.columns:
                bs_team = bs[bs["TEAM_ABBREVIATION"].astype("string") == team_abbr].copy()
            else:
                bs_team = bs.copy()

            show_cols = [c for c in ["PLAYER_NAME", "MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"] if c in bs_team.columns]
            st.dataframe(bs_team[show_cols], width="stretch", hide_index=True)

    st.markdown("### Roster (Interactive Averages)")
    roster_res = get_or_refresh(
        key=f"team_roster__{season}__{team_id}",
        ttl_seconds=TTL_TEAM,
        fetch_fn=lambda: fetch_team_roster(team_id, season),
        force_refresh=force_refresh,
    )
    roster = roster_res.df

    if roster is None or roster.empty:
        st.warning("Roster not available.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        per_mode = st.selectbox("Per Mode", ["PerGame", "Per36"], index=0, key="team_roster_per")
    with c2:
        window = st.selectbox("Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="team_roster_win")

    # For each player, pull player log (cached) and compute averages
    rows = []
    for _, pr in roster.iterrows():
        pid = pr.get("PLAYER_ID")
        pname = pr.get("PLAYER")
        pos = pr.get("POSITION")
        if pd.isna(pid):
            continue
        pid = int(pid)

        plog_res = get_or_refresh(
            key=f"player_gamelog__{season}__{pid}",
            ttl_seconds=TTL_PLAYER,
            fetch_fn=lambda pid=pid: fetch_player_game_log(pid, season),
            force_refresh=force_refresh,
        )
        plog = plog_res.df
        if plog is None or plog.empty:
            continue

        # Ensure numeric
        for c in ["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "OREB", "DREB", "STL", "BLK", "TOV"]:
            if c in plog.columns:
                plog[c] = pd.to_numeric(plog[c], errors="coerce")

        # choose window
        if window == "Season":
            wdf = plog.copy()
        else:
            n = int(window.split()[-1])
            wdf = plog.head(n).copy()

        # minutes
        if "MIN" in wdf.columns:
            mins = wdf["MIN"].apply(minutes_str_to_float)
        else:
            mins = pd.Series([0.0] * len(wdf))

        gp = len(wdf)
        if gp == 0:
            continue

        # per-game
        avg = {}
        for c in ["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "OREB", "DREB", "STL", "BLK", "TOV"]:
            if c in wdf.columns:
                avg[c] = float(wdf[c].mean())
        avg_min = float(mins.mean())

        # per-36 conversion
        if per_mode == "Per36":
            factor = 36.0 / max(avg_min, 1e-9)
            for k in list(avg.keys()):
                avg[k] = avg[k] * factor
            avg_min = 36.0

        row = {"Player": pname, "Pos": pos, "GP": gp, "MIN": round(avg_min, 1)}
        for k in ["PTS", "REB", "AST", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV"]:
            if k in avg:
                row[k] = round(avg[k], 1)
        rows.append(row)

    roster_tbl = pd.DataFrame(rows).sort_values(["MIN"], ascending=False) if rows else pd.DataFrame()
    if roster_tbl.empty:
        st.info("No roster stat rows available (logs may be empty early season).")
    else:
        st.dataframe(roster_tbl, width="stretch", hide_index=True)

    # Minutes rotation table in expander (API expensive)
    with st.expander("Player Minutes Rotation (Last 3 / Last 10 / Season + last 5 team games)"):
        if tlog is None or tlog.empty or "GAME_ID" not in tlog.columns:
            st.info("Team game log unavailable to build rotation view.")
        else:
            tlog3 = tlog.copy()
            if "GAME_DATE" in tlog3.columns:
                tlog3["GAME_DATE"] = pd.to_datetime(tlog3["GAME_DATE"], errors="coerce")
                tlog3 = tlog3.sort_values("GAME_DATE", ascending=False)
            recent_game_ids = [str(x) for x in tlog3["GAME_ID"].head(5).astype("string").tolist()]

            player_logs = []
            # reuse cached player logs
            for _, pr in roster.iterrows():
                pid = pr.get("PLAYER_ID")
                if pd.isna(pid):
                    continue
                pid = int(pid)
                plog_res = get_or_refresh(
                    key=f"player_gamelog__{season}__{pid}",
                    ttl_seconds=TTL_PLAYER,
                    fetch_fn=lambda pid=pid: fetch_player_game_log(pid, season),
                    force_refresh=force_refresh,
                )
                lg = plog_res.df
                if lg is None or lg.empty:
                    continue
                lg = lg.copy()
                lg["Player_ID"] = pid
                player_logs.append(lg)

            rot = compute_rotation_minutes_table(roster, player_logs, recent_game_ids)
            if rot.empty:
                st.info("Rotation table could not be built.")
            else:
                st.dataframe(rot, width="stretch", hide_index=True)


# -----------------------------
# Player tab
# -----------------------------
def player_tab(season: str, force_refresh: bool):
    # Player select with text filter
    all_players = _player_list()

    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search player (type last name, first name, etc.)", value="")
    with c2:
        per_mode = st.selectbox("Per Mode", ["PerGame", "Per36"], index=0, key="player_per")

    filtered = all_players
    if q.strip():
        ql = q.lower().strip()
        filtered = [p for p in all_players if ql in p["full_name"].lower()]

    # limit options for UI
    filtered = filtered[:250] if len(filtered) > 250 else filtered
    if not filtered:
        st.info("No players match your search.")
        return

    player_name = st.selectbox("Select Player", [p["full_name"] for p in filtered], index=0)
    player_id = int([p["id"] for p in filtered if p["full_name"] == player_name][0])

    # window controls
    c3, c4 = st.columns([1, 2])
    with c3:
        window = st.selectbox("Averages Window", ["Season", "Last 5", "Last 10", "Last 15"], index=0, key="player_win")
    with c4:
        st.caption("Box scores can be filtered further below.")

    plog_res = get_or_refresh(
        key=f"player_gamelog__{season}__{player_id}",
        ttl_seconds=TTL_PLAYER,
        fetch_fn=lambda: fetch_player_game_log(player_id, season),
        force_refresh=force_refresh,
    )
    logs = plog_res.df

    if logs is None or logs.empty:
        st.warning("No game logs found for this player in this season.")
        return

    # Ensure date and numeric columns
    if "GAME_DATE" in logs.columns:
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
    for c in ["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "OREB", "DREB", "STL", "BLK", "TOV"]:
        if c in logs.columns:
            logs[c] = pd.to_numeric(logs[c], errors="coerce")

    logs = add_basic_shooting_breakouts(logs)

    # Determine window df
    if window == "Season":
        wdf = logs.copy()
        title = "Averages Season Long"
    else:
        n = int(window.split()[-1])
        wdf = logs.head(n).copy()
        title = f"Averages Last {n} Games"

    # minutes
    if "MIN" in wdf.columns:
        mins = wdf["MIN"].apply(minutes_str_to_float)
    else:
        mins = pd.Series([0.0] * len(wdf))
    avg_min = float(mins.mean()) if len(mins) else 0.0

    # averages
    avg_cols = ["PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV"]
    avg = {"MIN": avg_min}
    for c in avg_cols:
        if c in wdf.columns:
            avg[c] = float(wdf[c].mean())

    if per_mode == "Per36":
        factor = 36.0 / max(avg_min, 1e-9)
        for k in list(avg.keys()):
            if k != "MIN":
                avg[k] = avg[k] * factor
        avg["MIN"] = 36.0

    # Header info
    st.markdown(f"### {player_name}")
    st.markdown(f"<div style='font-size:18px;'><b>{title}</b></div>", unsafe_allow_html=True)

    # Tiles
    tile_cols = st.columns(6)
    show_keys = ["MIN", "PTS", "REB", "AST", "PRA", "FG3M"]
    for i, k in enumerate(show_keys):
        with tile_cols[i]:
            v = avg.get(k, np.nan)
            st.metric(k, f"{v:.1f}" if pd.notna(v) else "—")

    # Box score filters
    st.markdown("### Box Scores (Interactive)")
    logs2 = logs.copy()

    # Opponent choices
    opp_col = None
    for cand in ["MATCHUP", "OPPONENT", "OPP_ABBR"]:
        if cand in logs2.columns:
            opp_col = cand
            break

    if opp_col == "MATCHUP":
        # Parse opponent abbreviation from MATCHUP like "LAL vs. BOS" / "LAL @ BOS"
        def _opp_from_matchup(m):
            s = str(m)
            parts = s.replace("vs.", "@").replace("VS.", "@").split("@")
            if len(parts) == 2:
                # left is team, right is opponent
                return parts[1].strip()
            return ""
        logs2["OPP_ABBR"] = logs2["MATCHUP"].apply(_opp_from_matchup)
    elif opp_col is not None:
        logs2["OPP_ABBR"] = logs2[opp_col].astype("string")
    else:
        logs2["OPP_ABBR"] = "All"

    # Date range
    if "GAME_DATE" in logs2.columns and logs2["GAME_DATE"].notna().any():
        min_d = logs2["GAME_DATE"].min().date()
        max_d = logs2["GAME_DATE"].max().date()
        dr_val = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        start_d, end_d = _safe_date_range(dr_val)
    else:
        start_d, end_d = None, None

    opps = sorted([o for o in logs2["OPP_ABBR"].dropna().unique().tolist() if o])
    opp_choice = st.selectbox("Opponent", ["All"] + opps, index=0)

    # Apply filters
    flt = logs2.copy()
    if start_d is not None and end_d is not None and "GAME_DATE" in flt.columns:
        flt = flt[(flt["GAME_DATE"].dt.date >= start_d) & (flt["GAME_DATE"].dt.date <= end_d)].copy()
    if opp_choice != "All":
        flt = flt[flt["OPP_ABBR"] == opp_choice].copy()

    # Slider show_n with guard
    max_games = int(len(flt))
    if max_games == 0:
        st.info("No games match your filters.")
        return

    min_slider = 1
    max_slider = max(1, max_games)
    default_slider = min(10, max_games)

    show_n = st.slider("Max games to display", min_value=min_slider, max_value=max_slider, value=default_slider, step=1)

    box_df = flt.head(show_n).copy()
    box_df = add_basic_shooting_breakouts(box_df)

    show_cols = [c for c in ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB", "STL", "BLK", "TOV"] if c in box_df.columns]
    st.dataframe(box_df[show_cols], width="stretch", hide_index=True)

    # Window comparison
    st.markdown("### Window Comparison (Per Game / Per 36)")
    # Previous season label
    try:
        y0 = int(season.split("-")[0])
        prev_season = f"{y0-1}-{str(y0 % 100).zfill(2)}"
    except Exception:
        prev_season = None

    # career data
    career_season_totals, career_totals = fetch_player_career(player_id)

    def summarize(df_in: pd.DataFrame, label: str):
        if df_in is None or df_in.empty:
            return None
        df = df_in.copy()
        df = add_basic_shooting_breakouts(df)
        # minutes
        mins = df["MIN"].apply(minutes_str_to_float) if "MIN" in df.columns else pd.Series([0.0]*len(df))
        avg_min = float(mins.mean()) if len(mins) else 0.0

        keys = ["PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB"]
        row = {"Window": label, "MIN": avg_min}
        for k in keys:
            if k in df.columns:
                row[k] = float(pd.to_numeric(df[k], errors="coerce").mean())
        if per_mode == "Per36":
            factor = 36.0 / max(avg_min, 1e-9)
            row["MIN"] = 36.0
            for k in keys:
                if k in row:
                    row[k] = row[k] * factor
        return row

    # Current season windows from logs
    row_season = summarize(logs, "Current Season")
    row_l5 = summarize(logs.head(5), "Last 5")
    row_l20 = summarize(logs.head(20), "Last 20")

    # Previous season from API (if available)
    prev_row = None
    if prev_season:
        prev_logs_res = get_or_refresh(
            key=f"player_gamelog__{prev_season}__{player_id}",
            ttl_seconds=TTL_PLAYER,
            fetch_fn=lambda: fetch_player_game_log(player_id, prev_season),
            force_refresh=force_refresh,
        )
        prev_logs = prev_logs_res.df
        if prev_logs is not None and not prev_logs.empty:
            prev_row = summarize(prev_logs, "Previous Season")

    # Career row using career totals if possible
    career_row = None
    if career_totals is not None and not career_totals.empty:
        # career_totals often has totals not game logs; we can show per-game from totals if columns exist
        ct = career_totals.copy()
        if "GP" in ct.columns and int(ct["GP"].iloc[0]) > 0:
            gp = float(ct["GP"].iloc[0])
            row = {"Window": "Career", "MIN": np.nan}
            # map totals to per-game if present
            for tot, outk in [("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"), ("OREB", "OREB"), ("DREB", "DREB"), ("FG3M", "FG3M"), ("FGM", "FGM"), ("FGA", "FGA")]:
                if tot in ct.columns:
                    row[outk] = float(ct[tot].iloc[0]) / gp
            # build derived
            if "FGM" in row and "FG3M" in row:
                row["FG2M"] = max(row["FGM"] - row["FG3M"], 0)
            if all(k in row for k in ["PTS", "REB", "AST"]):
                row["PRA"] = row["PTS"] + row["REB"] + row["AST"]
            career_row = row

    rows = [r for r in [row_season, prev_row, career_row, row_l5, row_l20] if r is not None]
    comp = pd.DataFrame(rows)
    if not comp.empty:
        # Round numeric
        for c in comp.columns:
            if c != "Window":
                comp[c] = pd.to_numeric(comp[c], errors="coerce").round(1)
        st.dataframe(comp, width="stretch", hide_index=True)


# -----------------------------
# Main
# -----------------------------
def main():
    season, date_choice, force_refresh = _header_bar()

    tab_league, tab_team, tab_player = st.tabs(["League", "Team", "Player"])

    with tab_league:
        league_tab(season, date_choice, force_refresh)

    with tab_team:
        team_tab(season, force_refresh)

    with tab_player:
        player_tab(season, force_refresh)


if __name__ == "__main__":
    main()
