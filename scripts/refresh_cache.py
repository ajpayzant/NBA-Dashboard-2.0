from __future__ import annotations

import argparse
import datetime as dt

from src.cache_store import write_parquet
from src.loaders import (
    fetch_daily_scoreboard,
    fetch_league_team_summary,
    fetch_league_player_stats,
)
from src.utils import now_utc

DEFAULT_SCOREBOARD_DAYS_BACK = 3
DEFAULT_SCOREBOARD_DAYS_FORWARD = 7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", required=True, help='Season string like "2025-26"')
    p.add_argument("--days-back", type=int, default=DEFAULT_SCOREBOARD_DAYS_BACK)
    p.add_argument("--days-forward", type=int, default=DEFAULT_SCOREBOARD_DAYS_FORWARD)
    args = p.parse_args()

    season = args.season
    today = now_utc().date()

    # 1) League team summary (standings + team stats + opponent table)
    team_table, opp_table = fetch_league_team_summary(season)
    write_parquet(f"league_team_summary__{season}", team_table)
    write_parquet(f"league_opp_summary__{season}", opp_table)

    # 2) League player stats: PerGame and Per36
    per_game = fetch_league_player_stats(season, per_mode="PerGame")
    per_36 = fetch_league_player_stats(season, per_mode="Per36")
    write_parquet(f"league_players__{season}__PerGame", per_game)
    write_parquet(f"league_players__{season}__Per36", per_36)

    # 3) Scoreboard cache for a date window (today - back ... today + forward)
    for offset in range(-args.days_back, args.days_forward + 1):
        d = today + dt.timedelta(days=offset)
        sb = fetch_daily_scoreboard(d)
        write_parquet(f"scoreboard__{d.isoformat()}", sb)

    print("✅ Refresh complete.")


if __name__ == "__main__":
    main()
