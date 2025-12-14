from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Type

import pandas as pd


DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 4

# Small global pacing to avoid bursts (helps Streamlit Cloud a lot)
_MIN_GAP_SECONDS = 0.65
_last_call_ts = 0.0


def _pace():
    global _last_call_ts
    now = time.time()
    gap = now - _last_call_ts
    if gap < _MIN_GAP_SECONDS:
        time.sleep(_MIN_GAP_SECONDS - gap)
    _last_call_ts = time.time()


def retry_api(
    endpoint_cls: Type[Any],
    kwargs: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    base_sleep: float = 1.2,
    jitter: float = 0.35,
) -> List[pd.DataFrame]:
    """
    Robust wrapper around nba_api endpoints:
      - pacing to reduce bursts
      - retries with exponential backoff + jitter
      - consistent timeout
    """
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            _pace()
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            sleep_s = (base_sleep * (2 ** attempt)) * (1.0 + random.uniform(-jitter, jitter))
            time.sleep(max(0.5, sleep_s))

    raise last_err  # type: ignore[misc]
