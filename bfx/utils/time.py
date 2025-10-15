from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

# --- Canonical: UTC-aware timestamp ---
def get_timestamp(y: int, m: int, d: int, h: int, mn: int, s: int) -> int:
    """Return Unix timestamp (seconds since epoch, UTC)."""
    dt = datetime(y, m, d, h, mn, s, tzinfo=timezone.utc)
    return int(dt.timestamp())

# --- Conversions to datetime (for charts) ---
def to_datetime_utc(ts: int) -> pd.Timestamp:
    """Convert Unix timestamp to UTC pandas.Timestamp."""
    return pd.to_datetime(ts, unit="s", utc=True)
