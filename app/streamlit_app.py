# app/streamlit_app.py

import streamlit as st
import pandas as pd

from src import data_loader as dl

st.set_page_config(page_title="NBA Stats App", layout="wide")
st.title("NBA Stats Explorer")

page = st.sidebar.radio(
    "Sections",
    ["Teams", "Players", "Games", "Today"]
)

@st.cache_data
def get_team_full():
    return dl.load_team_full_all()

@st.cache_data
def get_player_all():
    return dl.load_player_all()

@st.cache_data
def get_schedule_all():
    return dl.load_schedule_all()

@st.cache_data
def get_player_gamelog():
    return dl.load_player_gamelog_all()

@st.cache_data
def get_scoreboard_today():
    try:
        return dl.load_scoreboard_today()
    except FileNotFoundError:
        return pd.DataFrame()

if page == "Teams":
    st.header("Teams – Multi-Season")
    df = get_team_full()
    seasons = sorted(df["SEASON"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons)-1)
    df_season = df[df["SEASON"] == season].copy()

    st.subheader(f"Team Summary – {season}")
    st.dataframe(
        df_season[["TEAM_ABBREVIATION", "TEAM_NAME", "W", "L", "W_PCT", "PTS", "REB", "AST"]]
        .sort_values("W_PCT", ascending=False)
        .reset_index(drop=True)
    )

    team = st.selectbox(
        "Team detail",
        df_season["TEAM_ABBREVIATION"].sort_values().unique()
    )
    st.subheader(f"{team} – Full stat row")
    st.dataframe(df_season[df_season["TEAM_ABBREVIATION"] == team])

elif page == "Players":
    st.header("Players – Multi-Season")
    df = get_player_all()
    seasons = sorted(df["SEASON"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons)-1)
    df_season = df[df["SEASON"] == season].copy()

    team_filter = st.selectbox(
        "Filter by team (optional)",
        ["All"] + sorted(df_season["TEAM_ABBREVIATION"].dropna().unique().tolist())
    )
    if team_filter != "All":
        df_season = df_season[df_season["TEAM_ABBREVIATION"] == team_filter]

    st.subheader(f"Players – {season}")
    st.dataframe(
        df_season[
            ["PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN", "PTS", "REB", "AST"]
        ].sort_values("PTS", ascending=False)
        .reset_index(drop=True)
    )

    player_name = st.selectbox(
        "Player detail",
        df_season["PLAYER_NAME"].sort_values().unique()
    )
    st.subheader(f"{player_name} – Full season row")
    st.dataframe(df_season[df_season["PLAYER_NAME"] == player_name])

    gamelog = get_player_gamelog()
    gamelog_p = gamelog[gamelog["PLAYER_NAME"] == player_name].copy()
    gamelog_p = gamelog_p.sort_values("GAME_DATE", ascending=False)

    st.subheader(f"{player_name} – Recent game log")
    cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "STL", "BLK"]
    st.dataframe(gamelog_p[cols].head(20))

elif page == "Games":
    st.header("Games – Schedule & Results")
    sched = get_schedule_all()

    seasons = sorted(sched["SEASON"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons)-1)
    sched_s = sched[sched["SEASON"] == season].copy()

    team_filter = st.selectbox(
        "Team filter (optional)",
        ["All"] + sorted(sched_s["TEAM_NAME"].dropna().unique().tolist())
    )
    if team_filter != "All":
        sched_s = sched_s[sched_s["TEAM_NAME"] == team_filter]

    st.subheader(f"Games – {season}")
    st.dataframe(
        sched_s[["GAME_DATE", "MATCHUP", "TEAM_NAME", "WL", "PTS", "PLUS_MINUS"]]
        .sort_values("GAME_DATE", ascending=False)
        .reset_index(drop=True)
        .head(200)
    )

elif page == "Today":
    st.header("Today's Games / Scoreboard")
    sb = get_scoreboard_today()
    if sb.empty:
        st.info("No scoreboard data found for today (or data not yet scraped).")
    else:
        st.dataframe(sb)
