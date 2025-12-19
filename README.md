# NBA Offline Streamlit App (DuckDB)

This Streamlit app reads NBA data from a prebuilt DuckDB warehouse created in Google Colab.
It does NOT call nba_api at runtime.

## Expected DuckDB tables/views
- marts.team_latest
- marts.team_games
- marts.player_latest
- marts.player_games

## Setup
1) Create a virtual environment (optional)
2) Install requirements:
   pip install -r requirements.txt

3) Place your DuckDB file here:
   ./data/nba_warehouse.duckdb

   OR set an environment variable:
   DUCKDB_PATH=/full/path/to/nba_warehouse.duckdb

4) Run:
   streamlit run app.py
