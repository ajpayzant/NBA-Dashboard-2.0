"""
NBA DATA SCRAPER FOR DASHBOARD

- Scrapes core data for multiple seasons (team stats, player stats, schedule,
  scoreboard, player gamelogs, box scores).
- Seasons controlled by SEASONS list.
- Uses retry + backoff on all nba_api calls to reduce ReadTimeout issues.
- Heavy endpoints (box scores) support incremental updates based on GAME_ID.

Designed for:
- First run: full scrape for each season in SEASONS.
- Later runs: cheap "top-up" scrapes that only fetch new games for that season.
"""

import os
import time
import random
import inspect
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

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

# =============================================================================
# CONFIG
# =============================================================================

# Seasons you want to maintain in your data folder
SEASONS: List[str] = [
    "2025-26",
]

PER_MODE = "PerGame"        # usually "PerGame" for season stats

# Base output folder (GitHub / Codespaces)
OUTPUT_BASE_DIR = "./data"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# If True, ignore existing CSVs and re-scrape everything
FORCE_REFRESH: bool = False

print(f"💾 Output directory: {OUTPUT_BASE_DIR}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sleep_short(min_sec: float = 0.7, max_sec: float = 1.5) -> None:
    """
    Short randomized sleep to avoid hammering the NBA stats API.
    Called between requests.
    """
    time.sleep(random.uniform(min_sec, max_sec))


def sleep_long(min_sec: float = 3.0, max_sec: float = 6.0) -> None:
    """
    Longer randomized sleep for backoff after failures.
    """
    time.sleep(random.uniform(min_sec, max_sec))


def build_kwargs(endpoint_cls, logical_params):
    """
    Version-agnostic kwargs builder.

    logical_params: list of tuples:
      (value, [possible_param_names])

    It inspects the endpoint class signature and only uses param
    names that actually exist for the installed nba_api version.
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


def call_endpoint(endpoint_cls, logical_params, label: str = "", max_retries: int = 4):
    """
    Generic wrapper to call an nba_api endpoint with retries and basic logging.

    endpoint_cls: the nba_api endpoint class (e.g. leaguedashteamstats.LeagueDashTeamStats)
    logical_params: the same list you pass to build_kwargs
    label: short description for logging
    max_retries: how many times to retry on timeouts / transient errors
    """
    for attempt in range(1, max_retries + 1):
        kwargs = build_kwargs(endpoint_cls, logical_params)
        try:
            print(f"🔌 Calling {label} (attempt {attempt}/{max_retries})")
            ep = endpoint_cls(**kwargs)
            return ep
        except requests.exceptions.ReadTimeout as e:
            print(f"⏱️ ReadTimeout calling {label} (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                print(f"❌ Giving up on {label} after {max_retries} attempts due to repeated timeouts.")
                raise
            sleep_long()
        except Exception as e:
            print(f"⚠️ Error calling {label} (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                print(f"❌ Giving up on {label} after {max_retries} attempts due to errors.")
                raise
            sleep_long()

    # Should never reach here
    raise RuntimeError(f"Failed to call {label} after {max_retries} attempts.")


# =============================================================================
# STATIC TEAM METADATA
# =============================================================================

teams_list = nba_teams_static.get_teams()
teams_meta_df = pd.DataFrame(teams_list)

# Normalize columns
teams_meta_df["TEAM_ID"] = teams_meta_df["id"]
teams_meta_df["TEAM_NAME_STATIC"] = teams_meta_df["full_name"]
teams_meta_df["TEAM_ABBREVIATION_STATIC"] = teams_meta_df["abbreviation"]

teams_meta_short = teams_meta_df[["TEAM_ID", "TEAM_NAME_STATIC", "TEAM_ABBREVIATION_STATIC"]].copy()

print("✅ Loaded static team metadata:")
print(teams_meta_short.head(), "\n")


# =============================================================================
# TEAM-LEVEL STATS
# =============================================================================

def fetch_team_base(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    TEAM stats - base (counting box scores).
    """
    print(f"📆 Fetching TEAM Base stats for {season} (per_mode={per_mode})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
        ("Base", ["measure_type_detailed_def", "measure_type", "MeasureType"]),
    ]

    ep = call_endpoint(
        leaguedashteamstats.LeagueDashTeamStats,
        logical_params,
        label=f"LeagueDashTeamStats Base {season}",
    )
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season

    print("\nteam_base columns:", list(df.columns), "\n")
    return df


def fetch_team_advanced(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    TEAM stats - "Advanced" flavor (may mirror base columns; we suffix later).
    """
    print(f"📈 Fetching TEAM Advanced stats for {season} (per_mode={per_mode})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
        ("Advanced", ["measure_type_detailed_def", "measure_type", "MeasureType"]),
    ]

    ep = call_endpoint(
        leaguedashteamstats.LeagueDashTeamStats,
        logical_params,
        label=f"LeagueDashTeamStats Advanced {season}",
    )
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season

    print("\nteam_adv columns:", list(df.columns), "\n")
    return df


def fetch_team_opponent(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    TEAM opponent stats (depending on what the API exposes).
    """
    print(f"🛡 Fetching TEAM Opponent stats for {season} (per_mode={per_mode})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        (per_mode, ["per_mode_detailed", "per_mode_detailed_def", "per_mode"]),
        ("Opponent", ["measure_type_detailed_def", "measure_type", "MeasureType"]),
    ]

    ep = call_endpoint(
        leaguedashteamstats.LeagueDashTeamStats,
        logical_params,
        label=f"LeagueDashTeamStats Opponent {season}",
    )
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season

    print("\nteam_opp columns:", list(df.columns), "\n")
    return df


def build_team_frames(
    team_base: pd.DataFrame,
    team_adv: pd.DataFrame,
    team_opp: pd.DataFrame,
    teams_static: pd.DataFrame,
    season: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Attach team abbreviations and merge base + advanced + opponent into a single frame.
    Saves per-season CSVs:
      - team_base_adv_{season}.csv
      - team_opp_{season}.csv
      - team_full_{season}.csv
    """
    # Map TEAM_ID -> abbreviation
    id_abbr = teams_static[["TEAM_ID", "TEAM_ABBREVIATION_STATIC"]].rename(
        columns={"TEAM_ABBREVIATION_STATIC": "TEAM_ABBREVIATION"}
    )

    # Attach abbreviations
    tb = team_base.merge(id_abbr, on="TEAM_ID", how="left")
    ta = team_adv.merge(id_abbr, on="TEAM_ID", how="left")
    to = team_opp.merge(id_abbr, on="TEAM_ID", how="left")

    # Prepare ADV frame with suffixes
    adv_keep = [c for c in ta.columns if c not in ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "SEASON"]]
    ta_renamed = ta[["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"] + adv_keep].copy()
    ta_renamed = ta_renamed.rename(columns={c: f"{c}_ADV2" for c in adv_keep})

    # Prepare OPP frame with suffixes (TEAM_NAME becomes TEAM_NAME_OPP2)
    opp_keep = [c for c in to.columns if c not in ["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"]]
    rename_map_opp = {}
    for c in opp_keep:
        if c == "TEAM_NAME":
            rename_map_opp[c] = "TEAM_NAME_OPP2"
        else:
            rename_map_opp[c] = f"{c}_OPP2"

    to_renamed = to[["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"] + opp_keep].copy()
    to_renamed = to_renamed.rename(columns=rename_map_opp)

    # Merge base + advanced
    team_base_adv = tb.merge(
        ta_renamed,
        on=["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"],
        how="left"
    )

    # Merge in opponent
    team_full = team_base_adv.merge(
        to_renamed,
        on=["TEAM_ID", "TEAM_ABBREVIATION", "SEASON"],
        how="left"
    )

    print("\n🔗 Merging TEAM frames on keys: ['TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON']\n")
    print("==================== TEAM_FULL (Base+Adv+Opponent) ====================")
    print(f"Shape: {team_full.shape}\n")

    # Save per-season
    path_base_adv = os.path.join(OUTPUT_BASE_DIR, f"team_base_adv_{season}.csv")
    path_opp = os.path.join(OUTPUT_BASE_DIR, f"team_opp_{season}.csv")
    path_full = os.path.join(OUTPUT_BASE_DIR, f"team_full_{season}.csv")

    team_base_adv.to_csv(path_base_adv, index=False)
    to_renamed.to_csv(path_opp, index=False)
    team_full.to_csv(path_full, index=False)

    print("💾 Saved TEAM CSVs:")
    print(f" - {path_base_adv}")
    print(f" - {path_opp}")
    print(f" - {path_full}\n")

    return team_base_adv, to_renamed, team_full


# =============================================================================
# PLAYER-LEVEL SEASON STATS
# =============================================================================

def _fetch_player_variant(season: str,
                          per_mode: str,
                          measure_type_value: str | None = None) -> pd.DataFrame:
    """
    Internal helper to fetch a variant of LeagueDashPlayerStats.
    """
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

    label = f"LeagueDashPlayerStats {measure_type_value or 'Base'} {season} {per_mode}"
    ep = call_endpoint(
        leaguedashplayerstats.LeagueDashPlayerStats,
        logical_params,
        label=label,
    )
    df = ep.get_data_frames()[0].copy()
    df["SEASON"] = season
    return df


def fetch_player_all(season: str) -> pd.DataFrame:
    """
    PLAYER stats (season-wide):
      - Base PerGame
      - "Advanced" PerGame (suffix _ADV2)
      - Per36 (suffix _P36)

    Saves one combined per-season CSV: player_all_{season}.csv
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

    # Merge base + advanced
    key_cols = ["PLAYER_ID", "SEASON"]

    adv_keep = [c for c in adv_df.columns if c not in key_cols + ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    adv_renamed = adv_df[key_cols + adv_keep].copy()
    adv_renamed = adv_renamed.rename(columns={c: f"{c}_ADV2" for c in adv_keep})

    merged = base_df.merge(adv_renamed, on=key_cols, how="left")

    # Merge Per36
    p36_keep = [c for c in p36_df.columns if c not in key_cols + ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    p36_renamed = p36_df[key_cols + p36_keep].copy()
    p36_renamed = p36_renamed.rename(columns={c: f"{c}_P36" for c in p36_keep})

    merged = merged.merge(p36_renamed, on=key_cols, how="left")

    print("\n==================== PLAYER_ALL (PerGame + Advanced + Per36) ====================")
    print(f"Shape: {merged.shape} \n")

    out_path = os.path.join(OUTPUT_BASE_DIR, f"player_all_{season}.csv")
    merged.to_csv(out_path, index=False)
    print(f"💾 Saved PLAYER table to: {out_path}")
    print(f"🧮 PLAYER table shape: {merged.shape} \n")

    return merged


# =============================================================================
# SEASON SCHEDULE / RESULTS (TEAM VIEW)
# =============================================================================

def fetch_schedule_team(season: str) -> pd.DataFrame:
    """
    Use LeagueGameFinder to build a team-level schedule/results table.
    One row per TEAM per GAME.
    Always re-fetched because it's relatively light-weight.
    """
    print(f"📚 Fetching season schedule/results via LeagueGameFinder for {season}...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        ("Regular Season", ["season_type_all_star", "season_type_nullable"]),
        ("00", ["league_id_nullable", "league_id"]),
        ("T", ["player_or_team_abbreviation", "player_or_team"]),
    ]

    ep = call_endpoint(
        leaguegamefinder.LeagueGameFinder,
        logical_params,
        label=f"LeagueGameFinder schedule {season}",
    )
    df = ep.get_data_frames()[0].copy()

    # Keep a clean schedule subset
    cols = ["GAME_ID", "GAME_DATE", "MATCHUP", "WL", "TEAM_ID", "TEAM_NAME", "PTS", "PLUS_MINUS"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found in LeagueGameFinder output.")

    sched = df[cols].copy()
    sched["SEASON"] = season

    print("\n==================== SEASON SCHEDULE (Team-level) ====================")
    print(f"Shape: {sched.shape} \n")

    out_path = os.path.join(OUTPUT_BASE_DIR, f"schedule_season_team_{season}.csv")
    sched.to_csv(out_path, index=False)
    print(f"💾 Saved schedule to: {out_path}\n")

    return sched


# =============================================================================
# TODAY'S SCOREBOARD
# =============================================================================

def fetch_scoreboard_today() -> pd.DataFrame:
    """
    Fetch the current day's scoreboard and attach team abbreviations.
    """
    print("📺 Fetching scoreboard for today...")

    today_str = datetime.today().strftime("%Y-%m-%d")

    logical_params = [
        (today_str, ["game_date", "GameDate"]),
        ("00", ["league_id"]),
    ]

    ep = call_endpoint(
        scoreboardv2.ScoreboardV2,
        logical_params,
        label=f"ScoreboardV2 {today_str}",
    )
    games = ep.game_header.get_data_frame().copy()

    # Map team IDs to abbreviations
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

    out_path = os.path.join(OUTPUT_BASE_DIR, "scoreboard_today.csv")
    sb.to_csv(out_path, index=False)
    print(f"💾 Saved scoreboard to: {out_path}\n")

    return sb


# =============================================================================
# PLAYER GAME LOGS (LEAGUE-WIDE)
# =============================================================================

def fetch_player_gamelog(season: str,
                         season_type: str = "Regular Season") -> pd.DataFrame:
    """
    League-wide player game logs for a given season.
    One row per PLAYER per GAME.

    For simplicity, we re-scrape full-season logs each run.
    (Size is manageable compared to box scores.)
    """
    print(f"📓 Fetching PLAYER GAME LOGS for {season} ({season_type})...")
    sleep_short()

    logical_params = [
        (season, ["season", "season_nullable"]),
        (season_type, ["season_type_all_star", "season_type_nullable", "season_type"]),
        ("P", ["player_or_team_abbreviation", "player_or_team"]),
        ("DATE", ["sorter"]),
        ("DESC", ["direction"]),
    ]

    ep = call_endpoint(
        leaguegamelog.LeagueGameLog,
        logical_params,
        label=f"LeagueGameLog {season} {season_type}",
    )
    df = ep.get_data_frames()[0].copy()

    if "SEASON" not in df.columns:
        df["SEASON"] = season

    print("\n==================== PLAYER GAME LOGS ====================")
    print(f"Shape: {df.shape}\n")

    out_path = os.path.join(OUTPUT_BASE_DIR, f"player_gamelog_{season}.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved PLAYER GAME LOGS to: {out_path}\n")

    return df


# =============================================================================
# BOX SCORES (TRADITIONAL & ADVANCED) WITH INCREMENTAL UPDATE
# =============================================================================

def fetch_boxscores_traditional_for_season(
    schedule_df: pd.DataFrame,
    season: str,
    max_games: int | None = None,
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch BoxScoreTraditionalV3 for GAME_IDs in schedule_df.

    - First run (no existing CSV or force_refresh=True):
        Scrapes all games (or capped by max_games).
    - Later runs:
        Only scrapes new GAME_IDs not already in the CSV(s) and appends.

    Returns:
      - all_player_stats: row per PLAYER per GAME
      - all_team_stats: row per TEAM per GAME
    """
    all_game_ids = (
        schedule_df["GAME_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    path_p = os.path.join(OUTPUT_BASE_DIR, f"boxscore_traditional_players_{season}.csv")
    path_t = os.path.join(OUTPUT_BASE_DIR, f"boxscore_traditional_teams_{season}.csv")

    # Load existing if present and not forcing refresh
    existing_player_df = None
    existing_team_df = None
    existing_game_ids = set()

    if (not force_refresh) and os.path.exists(path_p) and os.path.exists(path_t):
        try:
            existing_player_df = pd.read_csv(path_p)
            existing_team_df = pd.read_csv(path_t)
            if "GAME_ID" in existing_player_df.columns:
                existing_game_ids = set(existing_player_df["GAME_ID"].astype(str).unique())
            print(f"📦 Found existing TRADITIONAL box score CSVs for {season}. Existing games: {len(existing_game_ids)}")
        except Exception as e:
            print(f"⚠️ Could not read existing traditional box score CSVs; will re-scrape. Error: {e}")
            existing_player_df = None
            existing_team_df = None
            existing_game_ids = set()

    # Identify new games to fetch
    new_game_ids = [gid for gid in all_game_ids if gid not in existing_game_ids]
    if max_games is not None:
        new_game_ids = new_game_ids[:max_games]

    if not new_game_ids and existing_player_df is not None and existing_team_df is not None:
        print(f"✅ No new traditional box score games to fetch for {season}. Returning cached data.")
        return existing_player_df, existing_team_df

    print(f"📦 Fetching TRADITIONAL box scores for {len(new_game_ids)} new games in {season}...")

    all_player_stats = []
    all_team_stats = []

    for i, gid in enumerate(new_game_ids, start=1):
        print(f"  [{i}/{len(new_game_ids)}] GAME_ID = {gid}")
        sleep_short()

        logical_params = [
            (gid, ["game_id", "game_id_nullable", "GameID"]),
        ]

        ep = call_endpoint(
            boxscoretraditionalv3.BoxScoreTraditionalV3,
            logical_params,
            label=f"BoxScoreTraditionalV3 {gid}",
        )

        ps = ep.player_stats.get_data_frame().copy()
        ps["GAME_ID"] = gid
        all_player_stats.append(ps)

        ts = ep.team_stats.get_data_frame().copy()
        ts["GAME_ID"] = gid
        all_team_stats.append(ts)

    new_player_df = pd.concat(all_player_stats, ignore_index=True) if all_player_stats else pd.DataFrame()
    new_team_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()

    # Combine with existing
    if existing_player_df is not None and not existing_player_df.empty:
        combined_player_df = pd.concat([existing_player_df, new_player_df], ignore_index=True)
        combined_player_df = combined_player_df.drop_duplicates(subset=["GAME_ID", "personId"], keep="last")
    else:
        combined_player_df = new_player_df

    if existing_team_df is not None and not existing_team_df.empty:
        combined_team_df = pd.concat([existing_team_df, new_team_df], ignore_index=True)
        combined_team_df = combined_team_df.drop_duplicates(subset=["GAME_ID", "teamId"], keep="last")
    else:
        combined_team_df = new_team_df

    print("\n==================== BOX SCORE TRADITIONAL - PLAYER ====================")
    print(f"Shape: {combined_player_df.shape}")
    print("==================== BOX SCORE TRADITIONAL - TEAM ====================")
    print(f"Shape: {combined_team_df.shape}")

    combined_player_df.to_csv(path_p, index=False)
    combined_team_df.to_csv(path_t, index=False)

    print(f"💾 Saved TRADITIONAL box score PLAYER stats to: {path_p}")
    print(f"💾 Saved TRADITIONAL box score TEAM stats to: {path_t}\n")

    return combined_player_df, combined_team_df


def fetch_boxscores_advanced_for_season(
    schedule_df: pd.DataFrame,
    season: str,
    max_games: int | None = None,
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch BoxScoreAdvancedV3 for GAME_IDs in schedule_df.

    Same incremental logic as traditional box scores.
    """
    all_game_ids = (
        schedule_df["GAME_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    path_p = os.path.join(OUTPUT_BASE_DIR, f"boxscore_advanced_players_{season}.csv")
    path_t = os.path.join(OUTPUT_BASE_DIR, f"boxscore_advanced_teams_{season}.csv")

    # Load existing if present and not forcing refresh
    existing_player_df = None
    existing_team_df = None
    existing_game_ids = set()

    if (not force_refresh) and os.path.exists(path_p) and os.path.exists(path_t):
        try:
            existing_player_df = pd.read_csv(path_p)
            existing_team_df = pd.read_csv(path_t)
            if "GAME_ID" in existing_player_df.columns:
                existing_game_ids = set(existing_player_df["GAME_ID"].astype(str).unique())
            print(f"📦 Found existing ADVANCED box score CSVs for {season}. Existing games: {len(existing_game_ids)}")
        except Exception as e:
            print(f"⚠️ Could not read existing advanced box score CSVs; will re-scrape. Error: {e}")
            existing_player_df = None
            existing_team_df = None
            existing_game_ids = set()

    # Identify new games to fetch
    new_game_ids = [gid for gid in all_game_ids if gid not in existing_game_ids]
    if max_games is not None:
        new_game_ids = new_game_ids[:max_games]

    if not new_game_ids and existing_player_df is not None and existing_team_df is not None:
        print(f"✅ No new advanced box score games to fetch for {season}. Returning cached data.")
        return existing_player_df, existing_team_df

    print(f"📦 Fetching ADVANCED box scores for {len(new_game_ids)} new games in {season}...")

    all_player_adv = []
    all_team_adv = []

    for i, gid in enumerate(new_game_ids, start=1):
        print(f"  [{i}/{len(new_game_ids)}] GAME_ID = {gid}")
        sleep_short()

        logical_params = [
            (gid, ["game_id", "game_id_nullable", "GameID"]),
        ]

        ep = call_endpoint(
            boxscoreadvancedv3.BoxScoreAdvancedV3,
            logical_params,
            label=f"BoxScoreAdvancedV3 {gid}",
        )

        ps = ep.player_stats.get_data_frame().copy()
        ps["GAME_ID"] = gid
        all_player_adv.append(ps)

        ts = ep.team_stats.get_data_frame().copy()
        ts["GAME_ID"] = gid
        all_team_adv.append(ts)

    new_player_df = pd.concat(all_player_adv, ignore_index=True) if all_player_adv else pd.DataFrame()
    new_team_df = pd.concat(all_team_adv, ignore_index=True) if all_team_adv else pd.DataFrame()

    # Combine with existing
    if existing_player_df is not None and not existing_player_df.empty:
        combined_player_df = pd.concat([existing_player_df, new_player_df], ignore_index=True)
        combined_player_df = combined_player_df.drop_duplicates(subset=["GAME_ID", "personId"], keep="last")
    else:
        combined_player_df = new_player_df

    if existing_team_df is not None and not existing_team_df.empty:
        combined_team_df = pd.concat([existing_team_df, new_team_df], ignore_index=True)
        combined_team_df = combined_team_df.drop_duplicates(subset=["GAME_ID", "teamId"], keep="last")
    else:
        combined_team_df = new_team_df

    print("\n==================== BOX SCORE ADVANCED - PLAYER ====================")
    print(f"Shape: {combined_player_df.shape}")
    print("==================== BOX SCORE ADVANCED - TEAM ====================")
    print(f"Shape: {combined_team_df.shape}")

    combined_player_df.to_csv(path_p, index=False)
    combined_team_df.to_csv(path_t, index=False)

    print(f"💾 Saved ADVANCED box score PLAYER stats to: {path_p}")
    print(f"💾 Saved ADVANCED box score TEAM stats to: {path_t}\n")

    return combined_player_df, combined_team_df


# =============================================================================
# DRIVER: RUN ALL SCRAPERS
# =============================================================================

def main():
    print("\n================================================================================")
    print("🚀 STARTING SCRAPE")
    print("================================================================================\n")
    print(f"Seasons: {SEASONS}")
    print(f"Output dir: {OUTPUT_BASE_DIR}")
    print(f"FORCE_REFRESH = {FORCE_REFRESH}\n")

    # Always fetch today's scoreboard (lightweight, no season dependency)
    try:
        scoreboard_today = fetch_scoreboard_today()
        print(f"✅ SCOREBOARD TODAY shape: {scoreboard_today.shape}")
    except Exception as e:
        print(f"⚠️ Failed to fetch today's scoreboard: {e}")

    # Loop over seasons and build all core data
    for season in SEASONS:
        print("\n================================================================================")
        print(f"🚀 STARTING SCRAPE FOR SEASON {season}")
        print("================================================================================\n")

        # TEAM tables
        team_base = fetch_team_base(season, PER_MODE)
        team_adv = fetch_team_advanced(season, PER_MODE)
        team_opp = fetch_team_opponent(season, PER_MODE)
        team_base_adv_all, team_opp_all, team_full_all = build_team_frames(
            team_base, team_adv, team_opp, teams_meta_short, season
        )

        # PLAYER table (season-wide)
        player_all = fetch_player_all(season)

        # Schedule (team-level view)
        schedule_season_team = fetch_schedule_team(season)

        # Player game logs (full season, overwrite each run)
        player_gamelog_all = fetch_player_gamelog(season)

        # Box scores (incremental)
        trad_player_bs, trad_team_bs = fetch_boxscores_traditional_for_season(
            schedule_season_team,
            season=season,
            max_games=None,              # set to an int for testing (e.g., 50)
            force_refresh=FORCE_REFRESH,
        )

        adv_player_bs, adv_team_bs = fetch_boxscores_advanced_for_season(
            schedule_season_team,
            season=season,
            max_games=None,
            force_refresh=FORCE_REFRESH,
        )

        print("\n✅ SEASON DATA INGEST COMPLETE.")
        print(f"Season: {season}")
        print(f" - team_base_adv_all: {team_base_adv_all.shape}")
        print(f" - team_opp_all: {team_opp_all.shape}")
        print(f" - team_full_all: {team_full_all.shape}")
        print(f" - player_all: {player_all.shape}")
        print(f" - schedule_season_team: {schedule_season_team.shape}")
        print(f" - player_gamelog_all: {player_gamelog_all.shape}")
        print(f" - trad_player_bs: {trad_player_bs.shape}")
        print(f" - trad_team_bs: {trad_team_bs.shape}")
        print(f" - adv_player_bs: {adv_player_bs.shape}")
        print(f" - adv_team_bs: {adv_team_bs.shape}")
        print("\n----------------------------------------\n")

    print("🎯 ALL SEASONS COMPLETE.")


if __name__ == "__main__":
    main()
