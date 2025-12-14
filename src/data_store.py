from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class CacheSpec:
    key: str
    ttl_seconds: int


def get_cache_dir() -> Path:
    # Always relative to project root (where app.py is)
    # If app.py runs from repo root, this will be ./data_cache
    base = Path.cwd()
    d = base / "data_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(key: str) -> Path:
    safe = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in key])
    return get_cache_dir() / f"{safe}.parquet"


def is_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age <= ttl_seconds


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


@contextmanager
def file_lock(lock_name: str, wait_seconds: float = 30.0, poll: float = 0.2):
    """
    Simple cross-process lock using exclusive file creation.
    Prevents multiple Streamlit threads/processes from refreshing the same parquet simultaneously.
    """
    lock_path = get_cache_dir() / f"{lock_name}.lock"
    start = time.time()
    fd: Optional[int] = None

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if time.time() - start > wait_seconds:
                # Give up lock acquisition; proceed without lock (better than deadlock)
                fd = None
                break
            time.sleep(poll)

    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass


def read_or_refresh_parquet(
    spec: CacheSpec,
    fetch_fn: Callable[[], pd.DataFrame],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Primary disk-cache primitive.
    - If parquet is fresh: read and return
    - Else: acquire lock, refresh, atomic-write, return
    """
    path = cache_path(spec.key)

    if not force_refresh and is_fresh(path, spec.ttl_seconds):
        try:
            return read_parquet(path)
        except Exception:
            # Corrupt file: fall through to refresh
            pass

    # Refresh path (lock to avoid thundering herd)
    with file_lock(lock_name=spec.key):
        # Another process may have refreshed while we waited
        if not force_refresh and is_fresh(path, spec.ttl_seconds):
            try:
                return read_parquet(path)
            except Exception:
                pass

        df = fetch_fn()
        if df is None:
            df = pd.DataFrame()
        _atomic_write_parquet(df, path)
        return df
