from __future__ import annotations

import os
import shutil

CACHE_DIR = os.environ.get("NBA_DASH_CACHE_DIR", "data_cache")

def main():
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"✅ Deleted cache folder: {CACHE_DIR}")
    else:
        print(f"ℹ️ Cache folder not found: {CACHE_DIR}")

if __name__ == "__main__":
    main()
