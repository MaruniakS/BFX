# file: evaluators/ks_evaluator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..utils import to_datetime_utc

# ---- helpers (kept local for self-containment) ----

def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    skip = {"timestamp", "time", "ts"}
    return [c for c in df.columns if c.lower() not in skip and pd.api.types.is_numeric_dtype(df[c])]

def _make_index(df: pd.DataFrame, start_ts: Optional[int], period_min: int) -> pd.DataFrame:
    time_col = next((c for c in df.columns if str(c).lower() in ("timestamp", "time", "ts")), None)
    if time_col is not None:
        idx = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
        df = df.drop(columns=[time_col])
        df.index = idx
        return df
    start = to_datetime_utc(int(start_ts)) if start_ts is not None else pd.Timestamp.utcnow().tz_localize("UTC")
    df.index = start + pd.to_timedelta(np.arange(len(df)) * int(max(period_min, 1)), unit="m")
    return df

def _window_masks(idx: pd.DatetimeIndex,
                  start: Optional[int], end: Optional[int],
                  a_start: Optional[int], a_end: Optional[int]) -> Dict[str, np.ndarray]:
    def ts(x: Optional[int]) -> Optional[pd.Timestamp]:
        return to_datetime_utc(int(x)) if x is not None else None
    s, e, as_, ae = ts(start), ts(end), ts(a_start), ts(a_end)

    base = np.ones(len(idx), dtype=bool)
    if s is not None: base &= (idx >= s)
    if e is not None: base &= (idx <= e)

    # If anomaly bounds invalid or missing, provide only "full"
    if as_ is None or ae is None or as_ >= ae:
        return {"full": base}

    pre  = base & (idx >= s)   & (idx < as_)
    dur  = base & (idx >= as_) & (idx < ae)
    post = base & (idx >= ae)  & (idx <= e)
    return {"pre": pre, "during": dur, "post": post}

def _ks_statistic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pure-numpy two-sample KS statistic D, plus the x-location where max gap occurs.
    Returns (D, x_at_max_gap).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = x.size, y.size
    if n1 == 0 or n2 == 0:
        return 0.0, np.nan

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    # Merge unique breakpoints from both samples
    points = np.concatenate([x_sorted, y_sorted])
    points.sort(kind="mergesort")

    # Walk both ECDFs simultaneously
    i = j = 0
    d_max = 0.0
    x_at = np.nan
    n1f = float(n1)
    n2f = float(n2)

    for v in points:
        while i < n1 and x_sorted[i] <= v:
            i += 1
        while j < n2 and y_sorted[j] <= v:
            j += 1
        fx = i / n1f
        gy = j / n2f
        d = abs(fx - gy)
        if d > d_max:
            d_max = d
            x_at = v

    return float(d_max), float(x_at)

# ---- Evaluator ----

class KSEvaluator(FeatureEvaluator):
    """
    Two-sample Kolmogorov–Smirnov statistic per feature.

    Parameters:
      window_pair: tuple[str, str] = ("pre", "during")
      min_samples_per_window: int = 5
      fallback: {"halves"|"none"}: if windows missing/invalid, split full into halves or return only "full".
    """

    name = "ks"

    def __init__(
        self,
        window_pair: Tuple[str, str] = ("pre", "during"),
        min_samples_per_window: int = 5,
        fallback: str = "halves",
    ) -> None:
        self.window_pair = tuple(window_pair)
        self.min_samples_per_window = int(max(1, min_samples_per_window))
        if fallback not in ("halves", "none"):
            raise ValueError("fallback must be 'halves' or 'none'")
        self.fallback = fallback

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = _make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = _numeric_feature_cols(df)
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for KS evaluation.")

        masks = _window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
        )

        # Resolve the pair of windows to compare
        win_a, win_b = self.window_pair
        if win_a not in masks or win_b not in masks:
            if self.fallback == "halves":
                # Build halves on the full slice
                mask_full = masks.get("full", np.ones(len(df), bool))
                idx = np.where(mask_full)[0]
                if idx.size < 2 * self.min_samples_per_window:
                    raise ValueError("Not enough samples in full window to build halves for KS.")
                mid = idx.size // 2
                masks = {
                    "first_half": mask_full.copy(),
                    "second_half": mask_full.copy(),
                }
                order = np.where(mask_full)[0]
                first = set(order[:mid])
                second = set(order[mid:])
                masks["first_half"][:] = False
                masks["second_half"][:] = False
                masks["first_half"][list(first)] = True
                masks["second_half"][list(second)] = True
                win_a, win_b = "first_half", "second_half"
            else:
                raise ValueError(f"Windows {self.window_pair} are not available and fallback is 'none'.")

        mask_a = masks[win_a]
        mask_b = masks[win_b]

        # Compute KS per feature
        scores: List[Dict[str, Any]] = []
        for f in feats:
            xa = df.loc[mask_a, f].to_numpy(dtype=float, copy=False)
            xb = df.loc[mask_b, f].to_numpy(dtype=float, copy=False)
            xa = xa[np.isfinite(xa)]
            xb = xb[np.isfinite(xb)]
            undefined = (xa.size < self.min_samples_per_window) or (xb.size < self.min_samples_per_window)

            if undefined:
                d, x_at = 0.0, np.nan
            else:
                d, x_at = _ks_statistic(xa, xb)

            scores.append({
                "feature": f,
                "score": float(d),                  # already in [0,1]
                "details": {
                    "D": float(d),
                    "x_at_max_gap": float(x_at) if np.isfinite(x_at) else None,
                    "n_a": int(xa.size),
                    "n_b": int(xb.size),
                    "window_a": win_a,
                    "window_b": win_b,
                    "undefined": bool(undefined),
                }
            })

        return {
            "method": self.name,
            "meta": {
                "window_pair": [win_a, win_b],
                "min_samples_per_window": self.min_samples_per_window,
                "fallback": self.fallback,
                "scaling": "identity",   # KS is already bounded [0,1]
            },
            "scores": scores,
        }
