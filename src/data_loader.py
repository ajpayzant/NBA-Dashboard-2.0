# src/data_loader.py

import os
import pandas as pd

DEFAULT_DATA_DIR = os.environ.get("NBA_DATA_DIR", "./data")

def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DEFAULT_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")
    return pd.read_csv(path)

def load_team_full_all() -> pd.DataFrame:
    return _load_csv("team_full_all.csv")

def load_player_all() -> pd.DataFrame:
    return _load_csv("player_all_all.csv")

def load_schedule_all() -> pd.DataFrame:
    return _load_csv("schedule_season_team_all.csv")

def load_player_gamelog_all() -> pd.DataFrame:
    return _load_csv("player_gamelog_all.csv")

def load_boxscore_traditional_players_all() -> pd.DataFrame:
    return _load_csv("boxscore_traditional_players_all.csv")

def load_boxscore_traditional_teams_all() -> pd.DataFrame:
    return _load_csv("boxscore_traditional_teams_all.csv")

def load_boxscore_advanced_players_all() -> pd.DataFrame:
    return _load_csv("boxscore_advanced_players_all.csv")

def load_boxscore_advanced_teams_all() -> pd.DataFrame:
    return _load_csv("boxscore_advanced_teams_all.csv")

def load_scoreboard_today() -> pd.DataFrame:
    return _load_csv("scoreboard_today.csv")
