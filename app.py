import os
import numpy as np
import pandas as pd
import streamlit as st

from src.db import query_df, resolve_db_path
from src import queries as Q
from src.ui import title_block, metric_row, df_download_button

st.set_page_config(page_title="NBA Offline Database App", layout="wide")

# -----------------------------
# Caching layer
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def cached_query(sql: str, params=None) -> pd.DataFrame:
    return query_df(sql, params=params)

def per36(df: pd.DataFrame, stat_cols, min_col="MIN_PG"):
    out = df.copy()
    denom = out[min_col].replace(0, np.nan)
    factor = 36.0 / denom
    for c in stat_cols:
        if c in out.columns:
            out[c] = out[c] * factor
    return out

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# -----------------------------
# Global header
# -----------------------------
st.markdown("# NBA Offline Data App")
st.caption("Runs entirely on a prebuilt DuckDB warehouse (no nba_api calls at runtime).")

with st.expander("Global Settings (optional)"):
    db_path = resolve_db_path()
    st.write("DuckDB Path:", db_path)
    st.write("Tip: set DUCKDB_PATH env var in Streamlit Cloud or locally.")
    st.divider()
    st.write("This expander is the only global UI element; all filters live inside each tab.")

# -----------------------------
# Load small reference sets
# -----------------------------
seasons = cached_query(Q.get_seasons_sql())["SEASON"].tolist()
if not seasons:
    st.error("No seasons found in marts.team_games. Confirm your DuckDB file is present and populated.")
    st.stop()

default_season = seasons[-1]
teams_ref = cached_query(Q.get_team_list_sql())
team_abbrs = teams_ref["TEAM_ABBREVIATION"].tolist()

# -----------------------------
# Tabs
# -----------------------------
tab_league, tab_team, tab_player = st.tabs(["League Data", "Team Data", "Player Data"])

# ============================================================
# TAB 1 — LEAGUE DATA
# ============================================================
with tab_league:
    title_block("League Data", "Leaguewide teams + league leaders, computed from your offline warehouse.")

    sub = st.radio(
        "Choose a leaguewide page",
        ["Teams (Standings + Team Metrics)", "Players (League Leaders)"],
        horizontal=True
    )

    season = st.selectbox("Season", seasons, index=seasons.index(default_season), key="league_season")

    if sub.startswith("Teams"):
        c1, c2 = st.columns([1, 2])
        with c1:
            conf = st.selectbox("Conference Filter", ["All", "East", "West"], key="league_conf")
            show_opp = st.checkbox("Show Opponent Averages table", value=True, key="league_show_opp")

        team_df = cached_query(Q.league_team_table_sql())
        team_df = team_df[team_df["SEASON"] == season].copy()

        # Conference mapping (simple and stable)
        EAST = set(["ATL","BOS","BKN","CHA","CHI","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS"])
        WEST = set([t for t in team_abbrs if t not in EAST])

        if conf == "East":
            team_df = team_df[team_df["TEAM_ABBREVIATION"].isin(EAST)]
        elif conf == "West":
            team_df = team_df[team_df["TEAM_ABBREVIATION"].isin(WEST)]

        team_df = team_df.sort_values(["WIN_PCT","W"], ascending=[False, False]).reset_index(drop=True)
        team_df["CONF_RANK"] = np.arange(1, len(team_df) + 1)

        metric_row([
            ("Teams", len(team_df)),
            ("Season", season),
            ("Top WIN% (filtered)", f"{team_df['WIN_PCT'].max():.3f}" if len(team_df) else "—"),
        ])

        st.subheader("Standings + Team Per-Game & Efficiency")
        show_cols = [
            "CONF_RANK","TEAM_ABBREVIATION","TEAM_NAME","W","L","GP","WIN_PCT",
            "PTS_PG","REB_PG","AST_PG","TOV_PG","OFF_RTG","POSS_PG"
        ]
        st.dataframe(team_df[show_cols], use_container_width=True, hide_index=True)
        df_download_button(team_df[show_cols], f"league_team_metrics_{season}.csv")

        if show_opp:
            opp_df = cached_query(Q.league_team_opponent_table_sql())
            opp_df = opp_df[opp_df["SEASON"] == season].copy()
            if conf == "East":
                opp_df = opp_df[opp_df["TEAM_ABBREVIATION"].isin(EAST)]
            elif conf == "West":
                opp_df = opp_df[opp_df["TEAM_ABBREVIATION"].isin(WEST)]

            st.subheader("Opponent Averages (Allowed Per Game)")
            opp_cols = [
                "TEAM_ABBREVIATION","TEAM_NAME",
                "OPP_PTS_PG","OPP_REB_PG","OPP_AST_PG","OPP_TOV_PG",
                "OPP_FGA_PG","OPP_FG3A_PG","OPP_FTA_PG"
            ]
            st.dataframe(opp_df[opp_cols].sort_values("TEAM_ABBREVIATION"), use_container_width=True, hide_index=True)
            df_download_button(opp_df[opp_cols], f"league_team_opponent_metrics_{season}.csv")

    else:
        # Players — league leaders
        st.subheader("League Leaders")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])

        with c1:
            mode = st.selectbox("Rate", ["Per Game", "Per 36"], key="leaders_mode")
        with c2:
            min_gp = st.number_input("Min Games Played", min_value=0, max_value=82, value=10, step=1, key="leaders_gp")
        with c3:
            min_min = st.number_input("Min Minutes / Game", min_value=0.0, max_value=48.0, value=15.0, step=1.0, key="leaders_min")
        with c4:
            stat = st.selectbox("Leader Stat", ["PTS_PG","REB_PG","AST_PG","STL_PG","BLK_PG","TOV_PG"], key="leaders_stat")

        p = cached_query(Q.league_player_leaders_sql())
        p = p[p["SEASON"] == season].copy()
        p = p[(p["GP"] >= min_gp) & (p["MIN_PG"] >= float(min_min))].copy()

        rate_cols = ["PTS_PG","REB_PG","AST_PG","STL_PG","BLK_PG","TOV_PG"]
        if mode == "Per 36":
            p = per36(p, rate_cols, min_col="MIN_PG")
            # Keep MIN_PG as context
        p = p.sort_values(stat, ascending=False).reset_index(drop=True)

        st.caption("Tip: increase min games/minutes to get cleaner leaderboards.")
        st.dataframe(
            p[["PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN_PG"] + rate_cols].head(50),
            use_container_width=True,
            hide_index=True
        )
        df_download_button(p, f"league_leaders_{season}_{mode.replace(' ','_').lower()}.csv")


# ============================================================
# TAB 2 — TEAM DATA
# ============================================================
with tab_team:
    title_block("Team Data", "Roster, minutes rotation, and full team box scores from your offline warehouse.")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        season = st.selectbox("Season", seasons, index=seasons.index(default_season), key="team_season")
    with c2:
        team_abbr = st.selectbox("Team", sorted(team_abbrs), index=sorted(team_abbrs).index("BOS") if "BOS" in team_abbrs else 0, key="team_abbr")
    with c3:
        stat_mode = st.selectbox("Roster Stats Mode", ["Per Game", "Per 36"], key="team_stat_mode")

    # Roster table from player_latest filtered by team + season
    roster = cached_query(Q.team_roster_latest_sql())
    roster = roster[(roster["SEASON"] == season) & (roster["TEAM_ABBREVIATION"] == team_abbr)].copy()

    if roster.empty:
        st.warning("No roster rows found for this team/season in marts.player_latest.")
    else:
        # Build per-game view for roster using player_games (more stable than latest-only)
        pg = cached_query(Q.player_game_logs_sql())
        pg = pg[(pg["SEASON"] == season) & (pg["TEAM_ABBREVIATION"] == team_abbr)].copy()

        # Aggregate per-game season stats
        agg = (pg.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False)
                 .agg(GP=("GAME_ID","count"),
                      MIN_PG=("MIN","mean"),
                      PTS_PG=("PTS","mean"),
                      REB_PG=("REB","mean"),
                      AST_PG=("AST","mean"),
                      STL_PG=("STL","mean"),
                      BLK_PG=("BLK","mean"),
                      TOV_PG=("TOV","mean")))

        if stat_mode == "Per 36":
            agg = per36(agg, ["PTS_PG","REB_PG","AST_PG","STL_PG","BLK_PG","TOV_PG"], min_col="MIN_PG")

        # Merge minutes rotation from player_latest (already has L5/L10)
        rot = roster[["PLAYER_ID","PLAYER_NAME","MIN_L5","MIN_L10","MIN"]].rename(columns={"MIN":"MIN_LAST"})
        out = agg.merge(rot, on=["PLAYER_ID","PLAYER_NAME"], how="left")

        st.subheader("Roster Stats")
        roster_cols = ["PLAYER_NAME","GP","MIN_PG","PTS_PG","REB_PG","AST_PG","STL_PG","BLK_PG","TOV_PG","MIN_L5","MIN_L10","MIN_LAST"]
        st.dataframe(out.sort_values("MIN_PG", ascending=False)[roster_cols], use_container_width=True, hide_index=True)
        df_download_button(out[roster_cols], f"{team_abbr}_roster_{season}_{stat_mode.replace(' ','_').lower()}.csv")

        # Minutes rotation: last 3/5/10 + last 5 game minutes by date
        st.subheader("Minutes Rotation (Last 3/5/10 + Last 5 Games)")
        # Compute last 3 + last 5 game-by-game minutes
        pg_sorted = pg.sort_values(["PLAYER_ID","GAME_DATE","GAME_ID"])
        # Last N minutes summary
        def last_n_mean(sdf, n):
            return sdf.tail(n)["MIN"].mean() if len(sdf) else np.nan

        # Build last-5 game minutes wide
        last5 = (pg.sort_values(["PLAYER_ID","GAME_DATE","GAME_ID"])
                   .groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False)
                   .apply(lambda g: g.tail(5))
                   .reset_index(drop=True))

        # Rank within each player (1..5) to pivot
        last5["GNUM"] = last5.sort_values(["GAME_DATE","GAME_ID"]).groupby("PLAYER_ID").cumcount() + 1
        # Create date header strings
        last5["GDATE"] = last5["GAME_DATE"].astype(str)

        pivot = last5.pivot_table(index=["PLAYER_ID","PLAYER_NAME"], columns="GNUM", values="MIN", aggfunc="first")
        dates = last5.pivot_table(index=["PLAYER_ID","PLAYER_NAME"], columns="GNUM", values="GDATE", aggfunc="first")

        # Compute last3 and season MIN_PG from agg, add last3 mean
        last3_mean = (pg_sorted.groupby(["PLAYER_ID","PLAYER_NAME"])
                              .apply(lambda g: last_n_mean(g, 3))
                              .reset_index(name="MIN_L3"))

        rot_table = agg[["PLAYER_ID","PLAYER_NAME","MIN_PG"]].merge(last3_mean, on=["PLAYER_ID","PLAYER_NAME"], how="left")
        rot_table = rot_table.merge(roster[["PLAYER_ID","PLAYER_NAME","MIN_L5","MIN_L10"]], on=["PLAYER_ID","PLAYER_NAME"], how="left")

        # Attach last-5 game mins with dynamic headers
        # We'll render as a dataframe with column labels like "1 (2025-12-18)"
        if not pivot.empty:
            col_map = {}
            for col in pivot.columns:
                # pick a representative date for header
                d = dates[col].dropna().iloc[0] if col in dates.columns and dates[col].notna().any() else ""
                col_map[col] = f"MIN_G{safe_int(col)} ({d})"
            pivot_ren = pivot.rename(columns=col_map).reset_index()
            rot_table = rot_table.merge(pivot_ren, on=["PLAYER_ID","PLAYER_NAME"], how="left")

        rot_cols = [c for c in rot_table.columns if c not in ["PLAYER_ID"]]
        st.dataframe(rot_table.sort_values("MIN_PG", ascending=False)[rot_cols], use_container_width=True, hide_index=True)
        df_download_button(rot_table[rot_cols], f"{team_abbr}_minutes_rotation_{season}.csv")

        # Team box score stats by game selector
        st.subheader("Team Box Score (Select a Game)")
        tg = cached_query(Q.team_game_logs_sql())
        tg = tg[(tg["SEASON"] == season) & (tg["TEAM_ABBREVIATION"] == team_abbr)].copy()
        tg = tg.sort_values(["GAME_DATE","GAME_ID"], ascending=[False, False])

        if tg.empty:
            st.info("No team games found for selection.")
        else:
            # Select by date + matchup
            tg["LABEL"] = tg["GAME_DATE"].astype(str) + " — " + tg["MATCHUP"].astype(str)
            label = st.selectbox("Choose game", tg["LABEL"].tolist(), key="team_game_select")
            row = tg[tg["LABEL"] == label].head(1)

            st.dataframe(row.drop(columns=["LABEL"]), use_container_width=True, hide_index=True)
            df_download_button(tg.drop(columns=["LABEL"]), f"{team_abbr}_team_games_{season}.csv", label="Download all team box scores (CSV)")


# ============================================================
# TAB 3 — PLAYER DATA
# ============================================================
with tab_player:
    title_block("Player Data", "Player season/rolling stats + box scores with filters (offline).")

    # Core selectors
    c1, c2, c3 = st.columns([1.1, 1.4, 1.1])
    with c1:
        season = st.selectbox("Season", seasons, index=seasons.index(default_season), key="player_season")
    with c2:
        # Use player_latest list to populate quickly
        pl = cached_query("SELECT DISTINCT PLAYER_NAME FROM marts.player_latest ORDER BY PLAYER_NAME")
        player_name = st.selectbox("Player", pl["PLAYER_NAME"].tolist(), key="player_name")
    with c3:
        stat_mode = st.selectbox("Stats Mode", ["Per Game", "Per 36"], key="player_stat_mode")

    # Pull all games for player in season
    pg = cached_query(Q.player_game_logs_sql())
    pg = pg[(pg["SEASON"] == season) & (pg["PLAYER_NAME"] == player_name)].copy()

    if pg.empty:
        st.warning("No games found for that player/season.")
    else:
        pg = pg.sort_values(["GAME_DATE","GAME_ID"], ascending=[False, False])

        # Rolling windows / splits
        st.subheader("Summary Splits")
        cA, cB, cC, cD = st.columns([1, 1, 1, 1])
        with cA:
            window = st.selectbox("Split Window", ["Season", "Last 5", "Last 10", "Last 20"], key="player_window")
        with cB:
            min_games = st.number_input("Require at least GP", min_value=1, max_value=82, value=1, step=1, key="player_min_gp")
        with cC:
            opp_filter = st.selectbox("Opponent Filter (optional)", ["All"] + sorted(pg["TEAM_ABBREVIATION"].unique().tolist()), key="player_opp")
        with cD:
            show_cols = st.multiselect(
                "Show columns",
                ["PTS","REB","AST","STL","BLK","TOV","MIN","FGA","FGM","FG3A","FG3M","FTA","FTM","PLUS_MINUS"],
                default=["MIN","PTS","REB","AST","TOV","PLUS_MINUS"],
                key="player_cols"
            )

        # Apply opponent filter (by matchup parsing is messy; use TEAM_ABBREVIATION as player team; we’ll filter by opponent from MATCHUP not present here)
        # For now, we support opponent filtering by date range and/or show all; if you want opponent-specific, we can join to team_games by GAME_ID.
        # We'll implement opponent filter properly in the next iteration when you confirm your preferred behavior.

        if window == "Season":
            sdf = pg.copy()
        elif window == "Last 5":
            sdf = pg.head(5).copy()
        elif window == "Last 10":
            sdf = pg.head(10).copy()
        else:
            sdf = pg.head(20).copy()

        gp = len(sdf)
        if gp < int(min_games):
            st.info(f"Split has only {gp} games (min required: {int(min_games)}). Adjust filters.")
        else:
            # Compute per-game
            summ = {
                "GP": gp,
                "TEAM": sdf["TEAM_ABBREVIATION"].mode().iloc[0] if sdf["TEAM_ABBREVIATION"].notna().any() else "",
                "MIN_PG": sdf["MIN"].mean(),
                "PTS_PG": sdf["PTS"].mean(),
                "REB_PG": sdf["REB"].mean(),
                "AST_PG": sdf["AST"].mean(),
                "STL_PG": sdf["STL"].mean() if "STL" in sdf.columns else np.nan,
                "BLK_PG": sdf["BLK"].mean() if "BLK" in sdf.columns else np.nan,
                "TOV_PG": sdf["TOV"].mean() if "TOV" in sdf.columns else np.nan,
            }
            summ_df = pd.DataFrame([summ])

            if stat_mode == "Per 36":
                summ_df = per36(summ_df, ["PTS_PG","REB_PG","AST_PG","STL_PG","BLK_PG","TOV_PG"], min_col="MIN_PG")

            metric_row([
                ("GP", summ_df.loc[0,"GP"]),
                ("MIN", f"{summ_df.loc[0,'MIN_PG']:.1f}"),
                ("PTS", f"{summ_df.loc[0,'PTS_PG']:.1f}"),
                ("REB", f"{summ_df.loc[0,'REB_PG']:.1f}"),
                ("AST", f"{summ_df.loc[0,'AST_PG']:.1f}"),
            ])

            st.dataframe(summ_df, use_container_width=True, hide_index=True)

        st.subheader("Box Scores (Game Log)")
        cX, cY, cZ = st.columns([1.2, 1.2, 1.2])
        with cX:
            n_games = st.slider("Number of games", min_value=5, max_value=min(50, len(pg)), value=min(10, len(pg)), step=1, key="player_ngames")
        with cY:
            date_min = st.date_input("From", value=pg["GAME_DATE"].min(), key="player_date_min")
        with cZ:
            date_max = st.date_input("To", value=pg["GAME_DATE"].max(), key="player_date_max")

        # Apply date range
        mask = (pg["GAME_DATE"] >= pd.to_datetime(date_min).date()) & (pg["GAME_DATE"] <= pd.to_datetime(date_max).date())
        box = pg[mask].copy().head(int(n_games))

        # Render
        cols = ["GAME_DATE","TEAM_ABBREVIATION","MIN"] + [c for c in show_cols if c not in ["MIN"]]
        cols = [c for c in cols if c in box.columns]
        st.dataframe(box[cols], use_container_width=True, hide_index=True)
        df_download_button(box[cols], f"{player_name.replace(' ','_')}_boxscores_{season}.csv")
