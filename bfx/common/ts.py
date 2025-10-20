# bfx/common/ts.py
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
from ..utils import to_datetime_utc

def make_index(df: pd.DataFrame, start_ts: Optional[int], period_min: int) -> pd.DataFrame:
    """
    Return a copy of df indexed by UTC timestamps.
    """
    df = df.copy()
    start = to_datetime_utc(int(start_ts)) if start_ts is not None else pd.Timestamp.utcnow().tz_localize("UTC")
    period = int(max(1, period_min))
    df.index = start + pd.to_timedelta(np.arange(len(df)) * period, unit="m")
    return df

def window_masks(
    idx: pd.DatetimeIndex,
    start: Optional[int],
    end: Optional[int],
    a_start: Optional[int],
    a_end: Optional[int],
    fallback: str = "halves",
) -> Dict[str, np.ndarray]:
    """
    Build boolean masks for pre/during/post if anomaly bounds exist.
    Always returns 'full'. If bounds missing/invalid and fallback='halves',
    also returns 'first_half' and 'second_half'.
    """
    def ts(x): return to_datetime_utc(int(x)) if x is not None else None
    s, e, as_, ae = ts(start), ts(end), ts(a_start), ts(a_end)

    base = np.ones(len(idx), dtype=bool)
    if s is not None: base &= (idx >= s)
    if e is not None: base &= (idx <= e)

    out: Dict[str, np.ndarray] = {"full": base}

    if as_ is not None and ae is not None and as_ < ae:
        out["pre"]    = base & (idx >= s)   & (idx < as_)
        out["during"] = base & (idx >= as_) & (idx < ae)
        out["post"]   = base & (idx >= ae)  & (idx <= e)
        return out

    if fallback == "halves":
        order = np.where(base)[0]
        mid = order.size // 2
        first = np.zeros_like(base, dtype=bool); first[order[:mid]] = True
        second = np.zeros_like(base, dtype=bool); second[order[mid:]] = True
        out["first_half"] = first
        out["second_half"] = second
    return out

def numeric_feature_cols(df: pd.DataFrame) -> Sequence[str]:
    """All numeric columns (no need to skip any time column)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
