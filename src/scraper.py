# src/scraper.py

import os
import time
import random
import inspect
from datetime import datetime

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguedashplayerstats,
    leaguegamefinder,
    scoreboardv2,
    leaguegamelog,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Seasons to scrape for your app.
SEASONS = ["2024-25", "2025-26"]   # you can add/remove seasons here
PER_MODE = "PerGame"

# Base output folder (inside repo)
OUTPUT_BASE_DIR = os.environ.get("NBA_DATA_DIR", "./data")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
print(f"💾 Output directory: {OUTPUT_BASE_DIR}")

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def sleep_short(min_sec: float = 0.6, max_sec: float = 1.2) -> None:
    """Short randomized sleep to avoid hammering the NBA stats API."""
    time.sleep(random.uniform(min_sec, max_sec))


def build_kwargs(endpoint_cls, logical_params):
    """
    Version-agnostic kwargs builder.
    logical_params: list of tuples: (value, [possible_param_names])
    """
    sig = inspect.signature(endpoint_cls)
    valid_names = set(sig.parameters.keys())

    kwargs = {}
    for value, name_options in logical_params:
        for name in name_options:
            if name in valid_names:
                kwargs[name] = value
                break
    return kwargs


# -------------------------------------------------------------------
# STATIC TEAM METADATA
# -------------------------------------------------------------------

teams_list = nba_teams_static.get_teams()
teams_meta_df = pd.DataFrame(teams_list)

teams_meta_df["TEAM_ID"] = teams_meta_df["id"]
teams_meta_df["TEAM_NAME_STATIC"] = teams_meta_df["full_name"]
teams_meta_df["TEAM_ABBREVIATION_STATIC"] = teams_meta_df["abbreviation"]

teams_meta_short = teams_meta_df[["TEAM_ID", "TEAM_NAME_STATIC", "TEAM_ABBREVIATION_STATIC"]].copy()

print("✅ Loaded static team metadata:")
print(teams_meta_short.head(), "\n")


# -------------------------------------------------------------------
# TEAM-LEVEL STATS
# -------------------------------------------------------------------

def fetch_team_base(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """TEAM stats - base (counting box scores)."""
    print(f"📆 Fetching TEAM Base stats for {season} (per_mode={per_mode})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
        ("Base", ["measure_type_detailed_def", "measure_type", "MeasureType"]),
    ]

    kwargs = build_kwargs(leaguedashteamstats.LeagueDashTeamStats, logical_params)
    ep = leaguedashteamstats.LeagueDashTeamStats(**kwargs)
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season

    print("\nteam_base columns:", list(df.columns), "\n")
    return df


def fetch_team_advanced(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """TEAM stats - 'Advanced' flavor."""
    print(f"📈 Fetching TEAM Advanced stats for {season} (per_mode={per_mode})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
        ("Advanced", ["measure_type_detailed_def", "measure_type", "MeasureType"]),
    ]

    kwargs = build_kwargs(leaguedashteamstats.LeagueDashTeamStats, logical_params)
    ep = leaguedashteamstats.LeagueDashTeamStats(**kwargs)
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season

    print("\nteam_adv columns:", list(df.columns), "\n")
    return df


def fetch_team_opponent(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """TEAM opponent stats."""
    print(f"🛡 Fetching TEAM Opponent stats for {season} (per_mode={per_mode})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
        ("Opponent", ["measure_type_detailed_def", "measure_type", "MeasureType"]),
    ]

    kwargs = build_kwargs(leaguedashteamstats.LeagueDashTeamStats, logical_params)
    ep = leaguedashteamstats.LeagueDashTeamStats(**kwargs)
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season

    print("\nteam_opp columns:", list(df.columns), "\n")
    return df


def build_team_frames(
    team_base: pd.DataFrame,
    team_adv: pd.DataFrame,
    team_opp: pd.DataFrame,
    teams_static: pd.DataFrame,
    season_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Attach team abbreviations and merge base + advanced + opponent into a single frame.
    Saves per-season CSVs:
      - team_base_adv_{season}.csv
      - team_opp_{season}.csv
      - team_full_{season}.csv
    """
    id_abbr = teams_static[["TEAM_ID", "TEAM_ABBREVIATION_STATIC"]].rename(
        columns={"TEAM_ABBREVIATION_STATIC": "TEAM_ABBREVIATION"}
    )

    tb = team_base.merge(id_abbr, on="TEAM_ID", how="left")
    ta = team_adv.merge(id_abbr, on="TEAM_ID", how="left")
    to = team_opp.merge(id_abbr, on="TEAM_ID", how="left")

    adv_keep = [c for c in ta.columns if c not in ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "SEASON"]]
    ta_renamed = ta[["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"] + adv_keep].copy()
    ta_renamed = ta_renamed.rename(columns={c: f"{c}_ADV2" for c in adv_keep})

    opp_keep = [c for c in to.columns if c not in ["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"]]
    rename_map_opp = {}
    for c in opp_keep:
        if c == "TEAM_NAME":
            rename_map_opp[c] = "TEAM_NAME_OPP2"
        else:
            rename_map_opp[c] = f"{c}_OPP2"

    to_renamed = to[["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"] + opp_keep].copy()
    to_renamed = to_renamed.rename(columns=rename_map_opp)

    team_base_adv = tb.merge(
        ta_renamed,
        on=["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"],
        how="left"
    )

    team_full = team_base_adv.merge(
        to_renamed,
        on=["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"],
        how="left"
    )

    print("\n🔗 Merging TEAM frames on keys: ['TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON']\n")
    print("==================== TEAM_FULL (Base+Adv+Opponent) ====================")
    print(f"Shape: {team_full.shape}\n")
    print(team_full.head(), "\n")

    path_base_adv = os.path.join(OUTPUT_BASE_DIR, f"team_base_adv_{season_label}.csv")
    path_opp = os.path.join(OUTPUT_BASE_DIR, f"team_opp_{season_label}.csv")
    path_full = os.path.join(OUTPUT_BASE_DIR, f"team_full_{season_label}.csv")

    team_base_adv.to_csv(path_base_adv, index=False)
    to_renamed.to_csv(path_opp, index=False)
    team_full.to_csv(path_full, index=False)

    print("💾 Saved TEAM CSVs:")
    print(f" - {path_base_adv}")
    print(f" - {path_opp}")
    print(f" - {path_full}\n")

    return team_base_adv, to_renamed, team_full


# -------------------------------------------------------------------
# PLAYER-LEVEL SEASON STATS
# -------------------------------------------------------------------

def _fetch_player_variant(season: str,
                          per_mode: str,
                          measure_type_value: str | None = None) -> pd.DataFrame:
    """Internal helper to fetch a variant of LeagueDashPlayerStats."""
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
    ]
    if measure_type_value is not None:
        logical_params.append(
            (measure_type_value, ["measure_type_detailed_def", "measure_type", "MeasureType"])
        )

    kwargs = build_kwargs(leaguedashplayerstats.LeagueDashPlayerStats, logical_params)
    ep = leaguedashplayerstats.LeagueDashPlayerStats(**kwargs)
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season
    return df


def fetch_player_all(season: str) -> pd.DataFrame:
    """
    PLAYER stats (season-wide):
      - Base PerGame
      - Advanced PerGame (_ADV2)
      - Per36 (_P36)
    """
    print(f"👤 Fetching PLAYER Base stats for {season} (per_mode=PerGame)...")
    base_df = _fetch_player_variant(season, "PerGame", measure_type_value=None)
    print(f"Base shape: {base_df.shape}")

    print(f"📊 Fetching PLAYER Advanced stats for {season}...")
    adv_df = _fetch_player_variant(season, "PerGame", measure_type_value="Advanced")
    print(f"Advanced shape: {adv_df.shape}")

    print(f"⏱ Fetching PLAYER Per36 stats for {season}...")
    p36_df = _fetch_player_variant(season, "Per36", measure_type_value=None)
    print(f"Per36 shape: {p36_df.shape}")

    key_cols = ["PLAYER_ID", "SEASON"]

    adv_keep = [c for c in adv_df.columns if c not in key_cols + ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    adv_renamed = adv_df[key_cols + adv_keep].copy()
    adv_renamed = adv_renamed.rename(columns={c: f"{c}_ADV2" for c in adv_keep})

    merged = base_df.merge(adv_renamed, on=key_cols, how="left")

    p36_keep = [c for c in p36_df.columns if c not in key_cols + ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    p36_renamed = p36_df[key_cols + p36_keep].copy()
    p36_renamed = p36_renamed.rename(columns={c: f"{c}_P36" for c in p36_keep})

    merged = merged.merge(p36_renamed, on=key_cols, how="left")

    print("\n==================== PLAYER_ALL (PerGame + Advanced + Per36) ====================")
    print(f"Shape: {merged.shape} \n")
    print(merged.head(), "\n")

    out_path = os.path.join(OUTPUT_BASE_DIR, f"player_all_{season}.csv")
    merged.to_csv(out_path, index=False)
    print(f"💾 Saved PLAYER table to: {out_path}")

    return merged


# -------------------------------------------------------------------
# SCHEDULE, SCOREBOARD, GAME LOGS
# -------------------------------------------------------------------

def fetch_schedule_team(season: str) -> pd.DataFrame:
    """Team-level schedule/results for a season."""
    print(f"📚 Fetching season schedule/results via LeagueGameFinder for {season}...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        ("00", ["league_id_nullable", "league_id"]),
        ("T", ["player_or_team_abbreviation", "player_or_team"]),
    ]

    kwargs = build_kwargs(leaguegamefinder.LeagueGameFinder, logical_params)
    ep = leaguegamefinder.LeagueGameFinder(**kwargs)
    df = ep.get_data_frames()[0].copy()

    cols = ["GAME_ID", "GAME_DATE", "MATCHUP", "WL", "TEAM_ID", "TEAM_NAME", "PTS", "PLUS_MINUS"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found in LeagueGameFinder output.")

    sched = df[cols].copy()
    sched["SEASON"] = season

    print("\n==================== SEASON SCHEDULE (Team-level) ====================")
    print(f"Shape: {sched.shape} \n")
    print(sched.head(), "\n")

    out_path = os.path.join(OUTPUT_BASE_DIR, f"schedule_season_team_{season}.csv")
    sched.to_csv(out_path, index=False)
    print(f"💾 Saved schedule to: {out_path}\n")

    return sched


def fetch_scoreboard_today() -> pd.DataFrame:
    """Fetch today's scoreboard."""
    print("📺 Fetching scoreboard for today...")

    today_str = datetime.today().strftime("%Y-%m-%d")

    logical_params = [
        (today_str, ["game_date", "GameDate"]),
        ("00", ["league_id"]),
    ]

    kwargs = build_kwargs(scoreboardv2.ScoreboardV2, logical_params)
    sleep_short()
    ep = scoreboardv2.ScoreboardV2(**kwargs)

    games = ep.game_header.get_data_frame().copy()

    id_to_abbr = teams_meta_short.set_index("TEAM_ID")["TEAM_ABBREVIATION_STATIC"].to_dict()

    required_cols = ["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_STATUS_TEXT"]
    for c in required_cols:
        if c not in games.columns:
            raise KeyError(f"Expected column '{c}' not in ScoreboardV2.game_header output: {games.columns}")

    sb = games[required_cols].copy()
    sb["HOME_ABBR"] = sb["HOME_TEAM_ID"].map(id_to_abbr)
    sb["AWAY_ABBR"] = sb["VISITOR_TEAM_ID"].map(id_to_abbr)

    print("\n==================== SCOREBOARD TODAY ====================")
    print(f"Shape: {sb.shape} \n")
    print(sb.head(), "\n")

    out_path = os.path.join(OUTPUT_BASE_DIR, "scoreboard_today.csv")
    sb.to_csv(out_path, index=False)
    print(f"💾 Saved scoreboard to: {out_path}\n")

    return sb


def fetch_player_gamelog(season: str,
                         season_type: str = "Regular Season") -> pd.DataFrame:
    """League-wide player game logs for a season."""
    print(f"📓 Fetching PLAYER GAME LOGS for {season} ({season_type})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        (season_type, ["season_type_all_star", "season_type_nullable", "season_type"]),
        ("P", ["player_or_team_abbreviation", "player_or_team"]),
        ("DATE", ["sorter"]),
        ("DESC", ["direction"]),
    ]

    kwargs = build_kwargs(leaguegamelog.LeagueGameLog, logical_params)
    ep = leaguegamelog.LeagueGameLog(**kwargs)
    df = ep.get_data_frames()[0].copy()

    if "SEASON" not in df.columns:
        df["SEASON"] = season

    print("\n==================== PLAYER GAME LOGS ====================")
    print(f"Shape: {df.shape}\n")
    print(df.head(), "\n")

    out_path = os.path.join(OUTPUT_BASE_DIR, f"player_gamelog_{season}.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved PLAYER GAME LOGS to: {out_path}\n")

    return df


# -------------------------------------------------------------------
# BOX SCORES
# -------------------------------------------------------------------

def fetch_boxscores_traditional_for_season(
    schedule_df: pd.DataFrame,
    season: str,
    max_games: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BoxScoreTraditionalV3 for all GAME_IDs in a season."""
    game_ids = (
        schedule_df["GAME_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if max_games is not None:
        game_ids = game_ids[:max_games]

    print(f"📦 Fetching TRADITIONAL box scores for {len(game_ids)} games in {season}...")

    all_player_stats = []
    all_team_stats = []

    def _box_trad_kwargs(game_id: str) -> dict:
        return build_kwargs(
            boxscoretraditionalv3.BoxScoreTraditionalV3,
            [(game_id, ["game_id", "game_id_nullable", "GameID"])],
        )

    for i, gid in enumerate(game_ids, start=1):
        print(f"  [{i}/{len(game_ids)}] GAME_ID = {gid}")
        sleep_short()
        kwargs = _box_trad_kwargs(gid)
        ep = boxscoretraditionalv3.BoxScoreTraditionalV3(**kwargs)

        ps = ep.player_stats.get_data_frame().copy()
        ps["GAME_ID"] = gid
        ps["SEASON"] = season
        all_player_stats.append(ps)

        ts = ep.team_stats.get_data_frame().copy()
        ts["GAME_ID"] = gid
        ts["SEASON"] = season
        all_team_stats.append(ts)

    all_player_stats_df = pd.concat(all_player_stats, ignore_index=True) if all_player_stats else pd.DataFrame()
    all_team_stats_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()

    out_p = os.path.join(OUTPUT_BASE_DIR, f"boxscore_traditional_players_{season}.csv")
    out_t = os.path.join(OUTPUT_BASE_DIR, f"boxscore_traditional_teams_{season}.csv")

    all_player_stats_df.to_csv(out_p, index=False)
    all_team_stats_df.to_csv(out_t, index=False)

    print(f"💾 Saved TRADITIONAL box score PLAYER stats to: {out_p}")
    print(f"💾 Saved TRADITIONAL box score TEAM stats to: {out_t}\n")

    return all_player_stats_df, all_team_stats_df


def fetch_boxscores_advanced_for_season(
    schedule_df: pd.DataFrame,
    season: str,
    max_games: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BoxScoreAdvancedV3 for all GAME_IDs in a season."""
    game_ids = (
        schedule_df["GAME_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if max_games is not None:
        game_ids = game_ids[:max_games]

    print(f"📦 Fetching ADVANCED box scores for {len(game_ids)} games in {season}...")

    all_player_adv = []
    all_team_adv = []

    def _box_adv_kwargs(game_id: str) -> dict:
        return build_kwargs(
            boxscoreadvancedv3.BoxScoreAdvancedV3,
            [(game_id, ["game_id", "game_id_nullable", "GameID"])],
        )

    for i, gid in enumerate(game_ids, start=1):
        print(f"  [{i}/{len(game_ids)}] GAME_ID = {gid}")
        sleep_short()
        kwargs = _box_adv_kwargs(gid)
        ep = boxscoreadvancedv3.BoxScoreAdvancedV3(**kwargs)

        ps = ep.player_stats.get_data_frame().copy()
        ps["GAME_ID"] = gid
        ps["SEASON"] = season
        all_player_adv.append(ps)

        ts = ep.team_stats.get_data_frame().copy()
        ts["GAME_ID"] = gid
        ts["SEASON"] = season
        all_team_adv.append(ts)

    all_player_adv_df = pd.concat(all_player_adv, ignore_index=True) if all_player_adv else pd.DataFrame()
    all_team_adv_df = pd.concat(all_team_adv, ignore_index=True) if all_team_adv else pd.DataFrame()

    out_p = os.path.join(OUTPUT_BASE_DIR, f"boxscore_advanced_players_{season}.csv")
    out_t = os.path.join(OUTPUT_BASE_DIR, f"boxscore_advanced_teams_{season}.csv")

    all_player_adv_df.to_csv(out_p, index=False)
    all_team_adv_df.to_csv(out_t, index=False)

    print(f"💾 Saved ADVANCED box score PLAYER stats to: {out_p}")
    print(f"💾 Saved ADVANCED box score TEAM stats to: {out_t}\n")

    return all_player_adv_df, all_team_adv_df


# -------------------------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------------------------

def _concat_and_save(frames, filename: str) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = os.path.join(OUTPUT_BASE_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"💾 Saved combined '{filename}' with shape {df.shape}")
    return df


def main():
    all_team_base_adv = []
    all_team_opp = []
    all_team_full = []

    all_player_all = []
    all_schedule = []
    all_player_gamelog = []

    all_trad_player_bs = []
    all_trad_team_bs = []
    all_adv_player_bs = []
    all_adv_team_bs = []

    MAX_GAMES_PER_SEASON_FOR_BOX = 100  # can set to None for full-season

    for season in SEASONS:
        print("\n" + "="*80)
        print(f"🚀 STARTING SCRAPE FOR SEASON {season}")
        print("="*80 + "\n")

        team_base = fetch_team_base(season, PER_MODE)
        team_adv = fetch_team_advanced(season, PER_MODE)
        team_opp = fetch_team_opponent(season, PER_MODE)
        team_base_adv_s, team_opp_s, team_full_s = build_team_frames(
            team_base, team_adv, team_opp, teams_meta_short, season
        )
        all_team_base_adv.append(team_base_adv_s)
        all_team_opp.append(team_opp_s)
        all_team_full.append(team_full_s)

        player_all_s = fetch_player_all(season)
        all_player_all.append(player_all_s)

        schedule_season_team_s = fetch_schedule_team(season)
        all_schedule.append(schedule_season_team_s)

        player_gamelog_s = fetch_player_gamelog(season)
        all_player_gamelog.append(player_gamelog_s)

        trad_player_bs_s, trad_team_bs_s = fetch_boxscores_traditional_for_season(
            schedule_season_team_s,
            season=season,
            max_games=MAX_GAMES_PER_SEASON_FOR_BOX,
        )
        adv_player_bs_s, adv_team_bs_s = fetch_boxscores_advanced_for_season(
            schedule_season_team_s,
            season=season,
            max_games=MAX_GAMES_PER_SEASON_FOR_BOX,
        )

        all_trad_player_bs.append(trad_player_bs_s)
        all_trad_team_bs.append(trad_team_bs_s)
        all_adv_player_bs.append(adv_player_bs_s)
        all_adv_team_bs.append(adv_team_bs_s)

    scoreboard_today = fetch_scoreboard_today()

    team_base_adv_all = _concat_and_save(all_team_base_adv, "team_base_adv_all.csv")
    team_opp_all      = _concat_and_save(all_team_opp,      "team_opp_all.csv")
    team_full_all     = _concat_and_save(all_team_full,     "team_full_all.csv")

    player_all        = _concat_and_save(all_player_all,    "player_all_all.csv")
    schedule_all      = _concat_and_save(all_schedule,      "schedule_season_team_all.csv")
    player_gamelog_all= _concat_and_save(all_player_gamelog,"player_gamelog_all.csv")

    trad_player_bs    = _concat_and_save(all_trad_player_bs, "boxscore_traditional_players_all.csv")
    trad_team_bs      = _concat_and_save(all_trad_team_bs,   "boxscore_traditional_teams_all.csv")
    adv_player_bs     = _concat_and_save(all_adv_player_bs,  "boxscore_advanced_players_all.csv")
    adv_team_bs       = _concat_and_save(all_adv_team_bs,    "boxscore_advanced_teams_all.csv")

    print("\n✅ DATA INGEST COMPLETE ACROSS SEASONS.\n")
    print("Available combined tables:")
    print(f" - team_full_all:     {team_full_all.shape}")
    print(f" - player_all:        {player_all.shape}")
    print(f" - schedule_all:      {schedule_all.shape}")
    print(f" - scoreboard_today:  {scoreboard_today.shape}")
    print(f" - player_gamelog_all:{player_gamelog_all.shape}")
    print(f" - trad_player_bs:    {trad_player_bs.shape}")
    print(f" - adv_player_bs:     {adv_player_bs.shape}")


if __name__ == "__main__":
    main()
