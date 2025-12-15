from __future__ import annotations

import argparse
import datetime as dt

from src.data_access import (
    get_teams_static,
    get_schedule_range,
    fetch_league_standings_cached,
    fetch_league_team_stats_cached,
    fetch_league_players_stats_cached,
    get_league_players_index,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", required=True, help='Season like "2025-26"')
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--days-ahead", type=int, default=10)
    args = p.parse_args()

    season = args.season
    start = dt.date.fromisoformat(args.start_date) if args.start_date else dt.date.today()

    get_teams_static(force_refresh=True)

    # League schedule and standings tables
    get_schedule_range(start, int(args.days_ahead), force_refresh=True)
    fetch_league_standings_cached(season, force_refresh=True)
    fetch_league_team_stats_cached(season, force_refresh=True)

    # Player dropdown index
    get_league_players_index(season, force_refresh=True)

    # Windows used throughout your app (leaders, roster snapshots, comparisons)
    per_modes = ["PerGame", "Per36"]
    last_windows = [0, 3, 5, 10, 15, 20]
    for pm in per_modes:
        for ln in last_windows:
            fetch_league_players_stats_cached(season, per_mode=pm, last_n_games=ln, force_refresh=True)

    print("✅ Full cache refresh complete.")

if __name__ == "__main__":
    main()
