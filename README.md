# NBA Dashboard (Streamlit)

This app pre-scrapes NBA stats into a local parquet cache (`data_cache/`) so users can switch tabs/filters without re-hitting the NBA API every time.

## Run locally
```bash
pip install -r requirements.txt
python scripts/refresh_cache.py --season "2025-26" --days-ahead 10
streamlit run app.py
