# bfx/evaluators/auc_evaluator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..utils import to_datetime_utc


# ---------- helpers ----------

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

    if as_ is None or ae is None or as_ >= ae:
        order = np.where(base)[0]
        mid = order.size // 2
        first = np.zeros_like(base, dtype=bool); first[order[:mid]] = True
        second = np.zeros_like(base, dtype=bool); second[order[mid:]] = True
        return {"first_half": first, "second_half": second, "full": base}

    pre  = base & (idx >= s)   & (idx < as_)
    dur  = base & (idx >= as_) & (idx < ae)
    post = base & (idx >= ae)  & (idx <= e)
    return {"pre": pre, "during": dur, "post": post, "full": base}

def _mann_whitney_auc(pre: np.ndarray, dur: np.ndarray, ties: str = "average") -> float:
    """
    AUC = U / (m*n), U from Mann–Whitney (Wilcoxon rank-sum).
    ties: ranking method for ties (pandas rank): "average", "min", ...
    Returns AUC in [0,1]. With direction 'during larger' -> AUC > 0.5.
    """
    pre = pre[np.isfinite(pre)]
    dur = dur[np.isfinite(dur)]
    m, n = pre.size, dur.size
    if m == 0 or n == 0:
        return 0.5  # undefined → neutral
    z = np.concatenate([pre, dur])
    ranks = pd.Series(z).rank(method=ties).to_numpy(dtype=float)
    R_dur = float(np.sum(ranks[m:]))  # ranks of 'during'
    U = R_dur - n * (n + 1) / 2.0
    auc = U / float(m * n)
    # guard numeric
    if not np.isfinite(auc):
        return 0.5
    return float(max(0.0, min(1.0, auc)))


# ---------- main evaluator ----------

class AUCEvaluator(FeatureEvaluator):
    """
    AUC / Cliff's Δ separability between two windows (default: pre vs during).

    Score per feature:
      - delta = 2*AUC - 1  in [-1,1] (directional)
      - score = |delta|    in [0,1]  (for ranking)

    Parameters:
      window_pair: Tuple[str,str] = ("pre","during")   # e.g., ("first_half","second_half")
      min_samples_per_window: int = 5
      fallback: "halves"                                # if requested windows invalid
      ties: str = "average"                             # pandas rank() tie method
      scaling: "identity"                               # keep raw |Δ|
    """

    name = "auc"

    def __init__(self,
                 window_pair: Tuple[str, str] = ("pre", "during"),
                 min_samples_per_window: int = 5,
                 fallback: str = "halves",
                 ties: str = "average",
                 scaling: str = "identity") -> None:
        self.window_pair = tuple(window_pair)
        self.min_samples_per_window = int(max(1, min_samples_per_window))
        self.fallback = str(fallback)
        self.ties = str(ties)
        self.scaling = str(scaling)

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = _make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = _numeric_feature_cols(df)
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for AUC evaluation.")

        masks = _window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
        )

        a, b = self.window_pair
        if a not in masks or b not in masks or (masks[a].sum() < self.min_samples_per_window) or (masks[b].sum() < self.min_samples_per_window):
            if self.fallback == "halves" and "first_half" in masks and "second_half" in masks:
                a, b = "first_half", "second_half"
            else:
                raise ValueError("AUC: requested windows unavailable and no valid fallback.")

        scores: List[Dict[str, Any]] = []
        for f in feats:
            pre_vals = df.loc[masks[a], f].to_numpy(dtype=float, copy=False)
            dur_vals = df.loc[masks[b], f].to_numpy(dtype=float, copy=False)
            m, n = int(np.isfinite(pre_vals).sum()), int(np.isfinite(dur_vals).sum())
            if m < self.min_samples_per_window or n < self.min_samples_per_window:
                scores.append({
                    "feature": f,
                    "score": 0.0,
                    "details": {
                        "undefined": True,
                        "window_a": a, "window_b": b,
                        "n_a": m, "n_b": n
                    }
                })
                continue

            auc = _mann_whitney_auc(pre_vals, dur_vals, ties=self.ties)
            delta = 2.0 * auc - 1.0  # Cliff's Δ
            direction = "up" if delta > 0 else ("down" if delta < 0 else "none")
            score = abs(delta)

            scores.append({
                "feature": f,
                "score": float(score),
                "details": {
                    "undefined": False,
                    "auc": float(auc),
                    "delta": float(delta),
                    "direction": direction,
                    "window_a": a, "window_b": b,
                    "n_a": m, "n_b": n
                }
            })

        return {
            "method": self.name,
            "meta": {
                "window_pair": [a, b],
                "min_samples_per_window": self.min_samples_per_window,
                "fallback": self.fallback,
                "ties": self.ties,
                "scaling": self.scaling,
            },
            "scores": scores,
        }
