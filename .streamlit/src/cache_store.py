from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

CACHE_DIR = os.environ.get("NBA_DASH_CACHE_DIR", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

META_EXT = ".meta.json"
DATA_EXT = ".parquet"


def _safe_key(key: str) -> str:
    # filesystem-safe key
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in key)


def _data_path(key: str) -> str:
    return os.path.join(CACHE_DIR, _safe_key(key) + DATA_EXT)


def _meta_path(key: str) -> str:
    return os.path.join(CACHE_DIR, _safe_key(key) + META_EXT)


def read_parquet(key: str) -> Optional[pd.DataFrame]:
    p = _data_path(key)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def write_parquet(key: str, df: pd.DataFrame) -> None:
    p = _data_path(key)
    df.to_parquet(p, index=False)
    meta = {"updated_at": int(time.time())}
    with open(_meta_path(key), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def last_updated_ts(key: str) -> Optional[int]:
    p = _meta_path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("updated_at"))
    except Exception:
        return None


def is_stale(key: str, ttl_seconds: int) -> bool:
    ts = last_updated_ts(key)
    if ts is None:
        return True
    return (time.time() - ts) > ttl_seconds


@dataclass
class CacheResult:
    df: pd.DataFrame
    from_cache: bool
    refreshed: bool


def get_or_refresh(
    key: str,
    ttl_seconds: int,
    fetch_fn: Callable[[], pd.DataFrame],
    normalize_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    force_refresh: bool = False,
) -> CacheResult:
    """
    Load from parquet if exists and not stale; otherwise fetch and store.
    normalize_fn is applied to the dataframe both when fetched and when read,
    to prevent dtype drift from breaking merges.
    """
    df = None if force_refresh else read_parquet(key)
    if df is not None and normalize_fn is not None:
        df = normalize_fn(df)

    if force_refresh or df is None or is_stale(key, ttl_seconds):
        fresh = fetch_fn()
        if normalize_fn is not None:
            fresh = normalize_fn(fresh)
        write_parquet(key, fresh)
        return CacheResult(df=fresh, from_cache=False, refreshed=True)

    return CacheResult(df=df, from_cache=True, refreshed=False)
