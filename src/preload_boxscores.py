# scripts/preload_boxscores.py
import time
import pandas as pd

from src.loaders import (
    get_teams_static,
    fetch_team_gamelog,
    fetch_game_team_boxscore,
)
from src.cache_store import write_parquet

SEASON = "2024-25"
SLEEP_BETWEEN_CALLS = 1.8  # critical to avoid silent throttling


def preload_team(team_id: int, season: str):
    print(f"▶ Loading team {team_id} ({season})")

    glog = fetch_team_gamelog(team_id, season)
    if glog.empty or "GAME_ID" not in glog.columns:
        print("  ⚠ No gamelog")
        return

    all_games = []

    for i, row in glog.iterrows():
        game_id = str(row["GAME_ID"])
        print(f"    → Game {i+1}/{len(glog)}: {game_id}")

        try:
            bx = fetch_game_team_boxscore(game_id, team_id)
            if bx.empty:
                print("      ⚠ empty boxscore")
                continue

            bx["GAME_ID"] = game_id
            bx["GAME_DATE"] = row.get("GAME_DATE")
            bx["TEAM_ID"] = team_id

            all_games.append(bx)
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"      ❌ failed: {e}")

    if not all_games:
        print("  ❌ No box scores collected")
        return

    full = pd.concat(all_games, ignore_index=True)

    key = f"team_boxscores__{season}__{team_id}"
    write_parquet(key, full)

    print(f"  ✅ Saved {len(full)} rows")


def main():
    teams = get_teams_static()
    for _, row in teams.iterrows():
        preload_team(int(row["TEAM_ID"]), SEASON)


if __name__ == "__main__":
    main()
