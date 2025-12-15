from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
from nba_api.stats.static import teams as teams_static, players as players_static

from src.cache_store import get_or_refresh, last_updated_ts
from src.loaders import (
    fetch_daily_scoreboard,
    fetch_league_team_summary,
    fetch_league_player_stats,
    fetch_team_gamelog,
    fetch_boxscore_traditional,
    fetch_player_gamelog,
)
from src.utils import season_labels, now_utc, add_basic_shooting_breakouts

# TTLs (seconds)
TTL_SCOREBOARD = 12 * 3600
TTL_LEAGUE = 24 * 3600
TTL_GAMELOG = 24 * 3600
TTL_BOXSCORE = 30 * 24 * 3600  # boxscores don't change

st.set_page_config(page_title="NBA Dashboard", layout="wide")


def _fmt_updated(ts: int | None) -> str:
    if not ts:
        return "Unknown"
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def league_tab(season: str, date_choice: dt.date, force_refresh: bool):
    tab_team, tab_players = st.tabs(["Team (Leaguewide)", "Player Leaders"])

    with tab_team:
        st.subheader("Schedule")

        key_sb = f"scoreboard__{date_choice.isoformat()}"
        sb_res = get_or_refresh(
            key=key_sb,
            ttl_seconds=TTL_SCOREBOARD,
            fetch_fn=lambda: fetch_daily_scoreboard(date_choice),
            force_refresh=force_refresh,
        )
        sb = sb_res.df

        if sb is None or sb.empty:
            st.warning("No schedule data available for this date.")
        else:
            st.dataframe(sb, width="stretch", hide_index=True)

        st.caption(f"Schedule cache: {('refreshed' if sb_res.refreshed else 'cached')} | Last updated: {_fmt_updated(last_updated_ts(key_sb))}")

        st.divider()
        st.subheader("Standings + Team Stats (sortable)")

        team_res = get_or_refresh(
            key=f"league_team_summary__{season}",
            ttl_seconds=TTL_LEAGUE,
            fetch_fn=lambda: fetch_league_team_summary(season)[0],
            force_refresh=force_refresh,
        )
        team_table = team_res.df

        if team_table is None or team_table.empty:
            st.warning("Standings/team stats are not available right now.")
        else:
            # Ensure wins/losses are ints, WinPCT numeric
            if "WINS" in team_table.columns:
                team_table["WINS"] = pd.to_numeric(team_table["WINS"], errors="coerce").fillna(0).astype(int)
            if "LOSSES" in team_table.columns:
                team_table["LOSSES"] = pd.to_numeric(team_table["LOSSES"], errors="coerce").fillna(0).astype(int)
            if "WinPCT" in team_table.columns:
                team_table["WinPCT"] = pd.to_numeric(team_table["WinPCT"], errors="coerce")

            # Default sort by WinPCT desc
            if "WinPCT" in team_table.columns:
                view = team_table.sort_values("WinPCT", ascending=False).reset_index(drop=True)
            else:
                view = team_table.copy()

            # Conference filter
            conf = st.radio("Conference", ["All", "East", "West"], horizontal=True)
            if conf != "All" and "Conference" in view.columns:
                view = view[view["Conference"].astype(str).str.contains(conf, case=False, na=False)].copy()

            st.dataframe(view, width="stretch", hide_index=True)

        st.caption(f"Team table cache: {('refreshed' if team_res.refreshed else 'cached')} | Last updated: {_fmt_updated(last_updated_ts(f'league_team_summary__{season}'))}")

        st.divider()
        st.subheader("Opponent Allowed Metrics (leaguewide)")

        opp_res = get_or_refresh(
            key=f"league_opp_summary__{season}",
            ttl_seconds=TTL_LEAGUE,
            fetch_fn=lambda: fetch_league_team_summary(season)[1],
            force_refresh=force_refresh,
        )
        opp_table = opp_res.df
        if opp_table is None or opp_table.empty:
            st.warning("Opponent table is not available right now.")
        else:
            conf2 = st.radio("Conference (Opponent Table)", ["All", "East", "West"], horizontal=True, key="opp_conf")
            view2 = opp_table.copy()
            if conf2 != "All" and "Conference" in view2.columns:
                view2 = view2[view2["Conference"].astype(str).str.contains(conf2, case=False, na=False)].copy()
            st.dataframe(view2, width="stretch", hide_index=True)

        st.caption(f"Opponent table cache: {('refreshed' if opp_res.refreshed else 'cached')} | Last updated: {_fmt_updated(last_updated_ts(f'league_opp_summary__{season}'))}")

    with tab_players:
        st.subheader("Leaguewide Player Leaders")

        min_gp = st.slider("Minimum Games Played", min_value=1, max_value=40, value=10)

        per_mode = st.radio("Per Mode", ["PerGame", "Per36"], horizontal=True)

        pl_res = get_or_refresh(
            key=f"league_players__{season}__{per_mode}",
            ttl_seconds=TTL_LEAGUE,
            fetch_fn=lambda: fetch_league_player_stats(season, per_mode=per_mode),
            force_refresh=force_refresh,
        )
        df = pl_res.df
        if df is None or df.empty:
            st.warning("Player leaders not available.")
            return

        df = df[df["GP"] >= min_gp].copy()
        if "TEAM_ABBREVIATION" in df.columns:
            df = df.rename(columns={"TEAM_ABBREVIATION": "Team"})

        # Define leader categories
        leader_stats = [
            ("Points", "PTS"),
            ("Rebounds", "REB"),
            ("Assists", "AST"),
            ("3PM", "FG3M"),
            ("2PM", "FG2M"),
            ("OREB", "OREB"),
            ("DREB", "DREB"),
        ]

        for title, col in leader_stats:
            if col not in df.columns:
                continue
            st.markdown(f"### {title}")
            tmp = df[["PLAYER_NAME", "Team", "GP", "MIN", col]].copy() if "Team" in df.columns else df[["PLAYER_NAME", "GP", "MIN", col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.sort_values(col, ascending=False).head(25).reset_index(drop=True)
            st.dataframe(tmp, width="stretch", hide_index=True)

        st.caption(f"Player table cache: {('refreshed' if pl_res.refreshed else 'cached')} | Last updated: {_fmt_updated(last_updated_ts(f'league_players__{season}__{per_mode}'))}")


def team_tab(season: str, force_refresh: bool):
    st.subheader("Team")

    teams = teams_static.get_teams()
    team_names = sorted([t["full_name"] for t in teams])
    name_to_id = {t["full_name"]: t["id"] for t in teams}
    name_to_abbr = {t["full_name"]: t["abbreviation"] for t in teams}

    team_name = st.selectbox("Select Team", team_names)
    team_id = name_to_id[team_name]
    team_abbr = name_to_abbr[team_name]

    # Team game log (cached)
    gl_key = f"team_gamelog__{season}__{team_id}"
    gl_res = get_or_refresh(
        key=gl_key,
        ttl_seconds=TTL_GAMELOG,
        fetch_fn=lambda: fetch_team_gamelog(season, team_id),
        force_refresh=force_refresh,
    )
    gl = gl_res.df

    if gl is None or gl.empty:
        st.warning("No game log available for this team/season yet.")
        st.caption(f"Game log cache: {('refreshed' if gl_res.refreshed else 'cached')}")
        return

    # Record + last 10 record
    wins = (gl["WL"] == "W").sum()
    losses = (gl["WL"] == "L").sum()
    last10 = gl.head(10)
    w10 = (last10["WL"] == "W").sum()
    l10 = (last10["WL"] == "L").sum()

    st.markdown(f"## {team_name} ({team_abbr})")
    st.markdown(f"### Record: **{wins}-{losses}** | Last 10: **{w10}-{l10}**")

    st.divider()
    st.subheader("Team Box Score (select a game)")

    # Game selection
    gl2 = gl.copy()
    gl2["GAME_DATE"] = pd.to_datetime(gl2["GAME_DATE"], errors="coerce")
    gl2 = gl2.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    options = [
        (row["Game_ID"], f'{row["GAME_DATE"].date()} | {row["MATCHUP"]} | {row["WL"]}')
        for _, row in gl2.iterrows()
        if pd.notna(row.get("Game_ID"))
    ]

    if not options:
        st.warning("No game log available for box score selection.")
        return

    game_label = st.selectbox("Choose game", [o[1] for o in options])
    game_id = [gid for gid, lab in options if lab == game_label][0]

    bs_key = f"boxscore_trad__{game_id}"
    bs_res = get_or_refresh(
        key=bs_key,
        ttl_seconds=TTL_BOXSCORE,
        fetch_fn=lambda: fetch_boxscore_traditional(game_id),
        force_refresh=force_refresh,
    )
    bs = bs_res.df
    if bs is None or bs.empty:
        st.warning("Box score not available.")
        return

    # Filter to this team
    if "TEAM_ABBREVIATION" in bs.columns:
        team_bs = bs[bs["TEAM_ABBREVIATION"].astype(str) == team_abbr].copy()
    else:
        team_bs = bs.copy()

    team_bs = add_basic_shooting_breakouts(team_bs)

    st.dataframe(team_bs, width="stretch", hide_index=True)
    st.caption(f"Box score cache: {('refreshed' if bs_res.refreshed else 'cached')} | Last updated: {_fmt_updated(last_updated_ts(bs_key))}")


def player_tab(season: str, force_refresh: bool):
    st.subheader("Player")

    players = players_static.get_active_players()
    names = sorted([p["full_name"] for p in players])
    name_to_id = {p["full_name"]: p["id"] for p in players}

    player_name = st.selectbox("Select Player", names)
    player_id = name_to_id[player_name]

    # Player gamelog (cached)
    pl_key = f"player_gamelog__{season}__{player_id}"
    pl_res = get_or_refresh(
        key=pl_key,
        ttl_seconds=TTL_GAMELOG,
        fetch_fn=lambda: fetch_player_gamelog(season, player_id),
        force_refresh=force_refresh,
    )
    logs = pl_res.df

    if logs is None or logs.empty:
        st.warning("No game log available for this player/season yet.")
        return

    logs = add_basic_shooting_breakouts(logs)

    # Window selector
    window_label = st.selectbox("Averages Window", ["Season", "Last 5", "Last 10", "Last 15", "Last 20"])
    n_map = {"Season": len(logs), "Last 5": 5, "Last 10": 10, "Last 15": 15, "Last 20": 20}
    n = min(n_map[window_label], len(logs))
    window_df = logs.head(n).copy()

    st.markdown(f"### Averages ({'Season Long' if window_label == 'Season' else window_label})")

    avg_cols = ["MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB"]
    avg_cols = [c for c in avg_cols if c in window_df.columns]

    avgs = window_df[avg_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_frame("AVG").T
    st.dataframe(avgs.round(2), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Box Scores (interactive)")

    # Date range filter
    logs2 = logs.copy()
    logs2["GAME_DATE"] = pd.to_datetime(logs2["GAME_DATE"], errors="coerce")
    logs2 = logs2.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    if logs2.empty:
        st.warning("No dated box scores available.")
        return

    min_d = logs2["GAME_DATE"].min().date()
    max_d = logs2["GAME_DATE"].max().date()

    dr = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(dr, tuple) and len(dr) == 2:
        start_d, end_d = dr
    else:
        # Streamlit can return a single date in some edge cases
        start_d, end_d = min_d, max_d

    # Opponent filter from MATCHUP text
    logs2["OPP"] = logs2["MATCHUP"].astype(str).str.replace("@", "vs").str.split("vs").str[-1].str.strip()
    opps = ["All"] + sorted([x for x in logs2["OPP"].dropna().unique().tolist() if x])
    opp_choice = st.selectbox("Opponent", opps)

    flt = logs2[(logs2["GAME_DATE"].dt.date >= start_d) & (logs2["GAME_DATE"].dt.date <= end_d)].copy()
    if opp_choice != "All":
        flt = flt[flt["OPP"] == opp_choice].copy()

    max_games = len(flt)
    if max_games == 0:
        st.warning("No games in this filter window.")
        return

    min_slider = 1
    max_slider = max_games
    default_n = min(10, max_games)

    show_n = st.slider("Max games to display", min_value=min_slider, max_value=max_slider, value=default_n)

    box_df = flt.head(show_n).copy()
    cols_show = [c for c in ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "PRA", "FG2M", "FG3M", "OREB", "DREB"] if c in box_df.columns]
    st.dataframe(box_df[cols_show], width="stretch", hide_index=True)

    st.caption(f"Player game log cache: {('refreshed' if pl_res.refreshed else 'cached')} | Last updated: {_fmt_updated(last_updated_ts(pl_key))}")


def main():
    st.title("NBA Dashboard")

    # Global controls (top bar style)
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        end_year = now_utc().year
        seasons = season_labels(2015, end_year)
        season = st.selectbox("Season", seasons[::-1], index=0)

    with c2:
        date_choice = st.date_input("Scoreboard Date", value=now_utc().date())

    with c3:
        force_refresh = st.toggle("Force Refresh", value=False)

    tab_league, tab_team, tab_player = st.tabs(["League", "Team", "Player"])

    with tab_league:
        league_tab(season, date_choice, force_refresh)

    with tab_team:
        team_tab(season, force_refresh)

    with tab_player:
        player_tab(season, force_refresh)


if __name__ == "__main__":
    main()
