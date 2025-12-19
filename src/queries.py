import pandas as pd

# -----------------------------
# Helper expressions (SQL)
# -----------------------------
# Possessions estimate (classic approximation)
POSS_EXPR = "(FGA + 0.44*FTA - OREB + TOV)"

def get_seasons_sql() -> str:
    return """
    SELECT DISTINCT SEASON
    FROM marts.team_games
    ORDER BY SEASON
    """

def get_team_list_sql() -> str:
    return """
    SELECT DISTINCT TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME
    FROM marts.team_games
    WHERE TEAM_NAME IS NOT NULL
    ORDER BY TEAM_ABBREVIATION
    """

def league_team_table_sql() -> str:
    # Aggregated team per-game + efficiency + standings-ready fields
    # Works entirely from marts.team_games
    return f"""
    WITH base AS (
      SELECT
        SEASON,
        TEAM_ID,
        TEAM_ABBREVIATION,
        TEAM_NAME,
        TEAM_CITY,
        TEAM_NICKNAME,
        WL,
        PTS, REB, AST, TOV,
        FGA, FGM, FG3A, FG3M, FTA, FTM,
        OREB, DREB,
        {POSS_EXPR} AS POSS
      FROM marts.team_games
      WHERE GAME_DATE IS NOT NULL
    ),
    agg AS (
      SELECT
        SEASON,
        TEAM_ID,
        ANY_VALUE(TEAM_ABBREVIATION) AS TEAM_ABBREVIATION,
        ANY_VALUE(TEAM_NAME) AS TEAM_NAME,
        SUM(CASE WHEN WL='W' THEN 1 ELSE 0 END) AS W,
        SUM(CASE WHEN WL='L' THEN 1 ELSE 0 END) AS L,
        COUNT(*) AS GP,
        AVG(PTS) AS PTS_PG,
        AVG(REB) AS REB_PG,
        AVG(AST) AS AST_PG,
        AVG(TOV) AS TOV_PG,
        AVG(FGA) AS FGA_PG,
        AVG(FGM) AS FGM_PG,
        AVG(FG3A) AS FG3A_PG,
        AVG(FG3M) AS FG3M_PG,
        AVG(FTA) AS FTA_PG,
        AVG(FTM) AS FTM_PG,
        AVG(OREB) AS OREB_PG,
        AVG(DREB) AS DREB_PG,
        AVG(POSS) AS POSS_PG,
        100.0 * AVG(PTS) / NULLIF(AVG(POSS), 0) AS OFF_RTG
      FROM base
      GROUP BY SEASON, TEAM_ID
    )
    SELECT
      *,
      CAST(W AS DOUBLE) / NULLIF(GP, 0) AS WIN_PCT,
      ROW_NUMBER() OVER (PARTITION BY SEASON ORDER BY (CAST(W AS DOUBLE)/NULLIF(GP,0)) DESC, W DESC) AS LEAGUE_RANK
    FROM agg
    """

def league_team_opponent_table_sql() -> str:
    # Opponent averages: join TEAM row to OPP row by GAME_ID + SEASON
    return """
    WITH g AS (
      SELECT
        SEASON,
        GAME_ID,
        TEAM_ID,
        TEAM_ABBREVIATION,
        TEAM_NAME,
        PTS, REB, AST, TOV,
        FGA, FGM, FG3A, FG3M, FTA, FTM, OREB, DREB
      FROM marts.team_games
      WHERE GAME_DATE IS NOT NULL
    ),
    paired AS (
      SELECT
        a.SEASON,
        a.TEAM_ID,
        ANY_VALUE(a.TEAM_ABBREVIATION) AS TEAM_ABBREVIATION,
        ANY_VALUE(a.TEAM_NAME) AS TEAM_NAME,
        b.PTS  AS OPP_PTS,
        b.REB  AS OPP_REB,
        b.AST  AS OPP_AST,
        b.TOV  AS OPP_TOV,
        b.FGA  AS OPP_FGA,
        b.FGM  AS OPP_FGM,
        b.FG3A AS OPP_FG3A,
        b.FG3M AS OPP_FG3M,
        b.FTA  AS OPP_FTA,
        b.FTM  AS OPP_FTM,
        b.OREB AS OPP_OREB,
        b.DREB AS OPP_DREB
      FROM g a
      JOIN g b
        ON a.SEASON = b.SEASON
       AND a.GAME_ID = b.GAME_ID
       AND a.TEAM_ID <> b.TEAM_ID
    )
    SELECT
      SEASON,
      TEAM_ID,
      TEAM_ABBREVIATION,
      TEAM_NAME,
      AVG(OPP_PTS)  AS OPP_PTS_PG,
      AVG(OPP_REB)  AS OPP_REB_PG,
      AVG(OPP_AST)  AS OPP_AST_PG,
      AVG(OPP_TOV)  AS OPP_TOV_PG,
      AVG(OPP_FGA)  AS OPP_FGA_PG,
      AVG(OPP_FGM)  AS OPP_FGM_PG,
      AVG(OPP_FG3A) AS OPP_FG3A_PG,
      AVG(OPP_FG3M) AS OPP_FG3M_PG,
      AVG(OPP_FTA)  AS OPP_FTA_PG,
      AVG(OPP_FTM)  AS OPP_FTM_PG,
      AVG(OPP_OREB) AS OPP_OREB_PG,
      AVG(OPP_DREB) AS OPP_DREB_PG
    FROM paired
    GROUP BY SEASON, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME
    ORDER BY SEASON, TEAM_ABBREVIATION
    """

def league_player_leaders_sql() -> str:
    # Player leader base; per-game stats aggregated from marts.player_games
    # Apply mins/games filters in python after query (fast enough) OR via SQL with params.
    return """
    WITH base AS (
      SELECT
        SEASON,
        PLAYER_ID,
        PLAYER_NAME,
        TEAM_ID,
        TEAM_ABBREVIATION,
        MIN AS MINUTES,
        PTS, REB, AST, STL, BLK, TOV,
        FGA, FGM, FG3A, FG3M, FTA, FTM
      FROM marts.player_games
      WHERE GAME_DATE IS NOT NULL
    ),
    agg AS (
      SELECT
        SEASON,
        PLAYER_ID,
        ANY_VALUE(PLAYER_NAME) AS PLAYER_NAME,
        ANY_VALUE(TEAM_ABBREVIATION) AS TEAM_ABBREVIATION,
        COUNT(*) AS GP,
        AVG(MINUTES) AS MIN_PG,
        AVG(PTS) AS PTS_PG,
        AVG(REB) AS REB_PG,
        AVG(AST) AS AST_PG,
        AVG(STL) AS STL_PG,
        AVG(BLK) AS BLK_PG,
        AVG(TOV) AS TOV_PG,
        AVG(FGA) AS FGA_PG,
        AVG(FGM) AS FGM_PG,
        AVG(FG3A) AS FG3A_PG,
        AVG(FG3M) AS FG3M_PG,
        AVG(FTA) AS FTA_PG,
        AVG(FTM) AS FTM_PG
      FROM base
      GROUP BY SEASON, PLAYER_ID
    )
    SELECT * FROM agg
    """

def team_roster_latest_sql() -> str:
    # Use player_latest, which is already 1 row per player (latest game)
    # We'll filter TEAM_ABBREVIATION + season in python (season not present in sample output; present in view though).
    return """
    SELECT
      SEASON,
      TEAM_ID,
      TEAM_ABBREVIATION,
      PLAYER_ID,
      PLAYER_NAME,
      GAME_DATE,
      MIN,
      MIN_L5,
      MIN_L10,
      PTS,
      PTS_L5,
      REB,
      REB_L5,
      AST,
      AST_L5,
      TOV,
      TOV_L5
    FROM marts.player_latest
    """

def player_game_logs_sql() -> str:
    return """
    SELECT
      SEASON,
      GAME_ID,
      GAME_DATE,
      TEAM_ID,
      TEAM_ABBREVIATION,
      PLAYER_ID,
      PLAYER_NAME,
      MIN,
      PTS, REB, AST, STL, BLK, TOV,
      FGA, FGM, FG3A, FG3M, FTA, FTM,
      PLUS_MINUS
    FROM marts.player_games
    WHERE GAME_DATE IS NOT NULL
    """

def team_game_logs_sql() -> str:
    return """
    SELECT
      SEASON,
      GAME_ID,
      GAME_DATE,
      TEAM_ID,
      TEAM_ABBREVIATION,
      TEAM_NAME,
      MATCHUP,
      WL,
      PTS, REB, AST, TOV,
      FGA, FGM, FG3A, FG3M, FTA, FTM,
      OREB, DREB,
      PLUS_MINUS
    FROM marts.team_games
    WHERE GAME_DATE IS NOT NULL
    """
