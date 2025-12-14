import datetime as dt

from src.cache_store import get_or_refresh
from src.loaders import (
    fetch_league_team_summary,
    fetch_league_player_stats,
    fetch_daily_scoreboard,
)
from src.utils import season_labels, today_la_date, coerce_team_keys

# Basic warm-cache script (run locally):
#   python scripts/refresh_cache.py

TTL = 60 * 60 * 24  # 24h


def main():
    seasons = season_labels(2020)
    season = seasons[-1]

    print("Warming league team summary...")
    get_or_refresh(
        key=f"league_team_summary__{season}",
        ttl_seconds=TTL,
        fetch_fn=lambda: fetch_league_team_summary(season)[0],
        normalize_fn=coerce_team_keys,
        force_refresh=True,
    )
    get_or_refresh(
        key=f"league_opp_summary__{season}",
        ttl_seconds=TTL,
        fetch_fn=lambda: fetch_league_team_summary(season)[1],
        normalize_fn=coerce_team_keys,
        force_refresh=True,
    )

    print("Warming league player stats (PerGame, Per36)...")
    get_or_refresh(
        key=f"league_players__{season}__PerGame",
        ttl_seconds=TTL,
        fetch_fn=lambda: fetch_league_player_stats(season, "PerGame"),
        force_refresh=True,
    )
    get_or_refresh(
        key=f"league_players__{season}__Per36",
        ttl_seconds=TTL,
        fetch_fn=lambda: fetch_league_player_stats(season, "Per36"),
        force_refresh=True,
    )

    d = today_la_date()
    print(f"Warming scoreboard for {d} ...")
    get_or_refresh(
        key=f"scoreboard__{d.isoformat()}",
        ttl_seconds=60 * 10,
        fetch_fn=lambda: fetch_daily_scoreboard(d),
        force_refresh=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
