# src/cache_store.py
import os
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

DEFAULT_CACHE_DIR = os.environ.get("NBA_DASH_CACHE_DIR", ".cache_streamlit")


@dataclass
class CacheResult:
    df: pd.DataFrame
    meta: dict


def _safe_key(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return h


def get_or_refresh(
    key: str,
    ttl_seconds: int,
    fetch_fn: Callable[[], pd.DataFrame],
    force_refresh: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> CacheResult:
    os.makedirs(cache_dir, exist_ok=True)
    k = _safe_key(key)
    data_path = os.path.join(cache_dir, f"{k}.parquet")
    meta_path = os.path.join(cache_dir, f"{k}.meta.json")

    now = int(time.time())

    if (not force_refresh) and os.path.exists(data_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ts = int(meta.get("ts", 0))
            if now - ts <= int(ttl_seconds):
                df = pd.read_parquet(data_path)
                return CacheResult(df=df, meta=meta)
        except Exception:
            pass  # fall through to fetch

    df = fetch_fn()
    if df is None:
        df = pd.DataFrame()

    meta = {"key": key, "ts": now, "rows": int(len(df)), "cols": int(df.shape[1])}
    try:
        df.to_parquet(data_path, index=False)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    except Exception:
        # If disk write fails, still return data in-memory
        pass

    return CacheResult(df=df, meta=meta)
