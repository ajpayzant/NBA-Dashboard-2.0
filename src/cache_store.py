from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Callable, Optional, List

import pandas as pd

# Persistent on-disk cache directory (created automatically at runtime).
# In Streamlit Cloud this will exist inside the container filesystem.
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


# -----------------------------------------------------------------------------
# NEW: helpers for persistent cache maintenance (safe additive; doesn't affect app flow)
# -----------------------------------------------------------------------------
def delete_key(key: str) -> bool:
    """Delete a single cached dataset + metadata. Returns True if anything was deleted."""
    deleted = False
    for p in (_data_path(key), _meta_path(key)):
        try:
            if os.path.exists(p):
                os.remove(p)
                deleted = True
        except Exception:
            pass
    return deleted


def clear_cache_dir() -> int:
    """Delete ALL parquet + meta files under CACHE_DIR. Returns number of files deleted."""
    n = 0
    try:
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith(DATA_EXT) or fname.endswith(META_EXT):
                try:
                    os.remove(os.path.join(CACHE_DIR, fname))
                    n += 1
                except Exception:
                    pass
    except Exception:
        return n
    return n


def list_cache_files() -> List[str]:
    """List filenames currently stored in CACHE_DIR (debug/diagnostics)."""
    try:
        return sorted(os.listdir(CACHE_DIR))
    except Exception:
        return []


# -----------------------------------------------------------------------------
# NEW: disk-only reader for preloaded team boxscores
# -----------------------------------------------------------------------------
def read_boxscores(team_id: int, season: str) -> pd.DataFrame:
    """
    Read preloaded team boxscores (ALL games x ALL players for a team/season).
    Produced by your preload script, stored as:
      key = team_boxscores__{season}__{team_id}.parquet
    """
    key = f"team_boxscores__{season}__{int(team_id)}"
    df = read_parquet(key)
    return df if df is not None else pd.DataFrame()
