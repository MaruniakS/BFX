from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..utils import to_datetime_utc


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
        # no anomaly bounds → provide halves for baseline fallback
        order = np.where(base)[0]
        mid = order.size // 2
        first = np.zeros_like(base, dtype=bool); first[order[:mid]] = True
        second = np.zeros_like(base, dtype=bool); second[order[mid:]] = True
        return {"first_half": first, "second_half": second, "full": base}

    pre  = base & (idx >= s)   & (idx < as_)
    dur  = base & (idx >= as_) & (idx < ae)
    post = base & (idx >= ae)  & (idx <= e)
    return {"pre": pre, "during": dur, "post": post, "full": base}

def _standardize(x: np.ndarray, mu0: float, sd0: float, tiny: float) -> np.ndarray:
    if not np.isfinite(sd0) or abs(sd0) <= tiny:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu0) / sd0

def _cusum_one_sided(z: np.ndarray, k: float, sign: int) -> Tuple[np.ndarray, float, int]:
    """
    sign=+1 for upward S+, sign=-1 for downward S- (using -z).
    Returns (S curve, Smax, argmax_index)
    """
    if z.size == 0 or not np.isfinite(z).any():
        return np.zeros_like(z), 0.0, -1
    S = np.zeros_like(z, dtype=float)
    for t in range(1, z.size):
        incr = (z[t] if sign > 0 else -z[t]) - k
        S[t] = max(0.0, S[t-1] + incr)
    imax = int(np.nanargmax(S)) if S.size else -1
    Smax = float(S[imax]) if imax >= 0 else 0.0
    return S, Smax, imax

class CUSUMEvaluator(FeatureEvaluator):
    """
    CUSUM drift score per feature.

    Parameters:
      side: "up" | "down" | "both"          # which direction to score
      k: float = 0.0                         # reference value (in SD units)
      baseline_window: "auto"|"pre"|"first_half"
      min_samples_pre: int = 5
      tiny_sigma: float = 1e-12
    """

    name = "cusum"

    def __init__(self,
                 side: str = "both",
                 k: float = 0.0,
                 baseline_window: str = "auto",
                 min_samples_pre: int = 5,
                 tiny_sigma: float = 1e-12) -> None:
        self.side = side
        self.k = float(k)
        self.baseline_window = baseline_window
        self.min_samples_pre = int(max(1, min_samples_pre))
        self.tiny_sigma = float(tiny_sigma)

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = _make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = _numeric_feature_cols(df)
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for CUSUM evaluation.")

        masks = _window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
        )

        # choose baseline window
        if self.baseline_window == "pre" or (self.baseline_window == "auto" and "pre" in masks and masks["pre"].sum() >= self.min_samples_pre):
            base_name = "pre"; base_mask = masks["pre"]
        elif "first_half" in masks and masks["first_half"].sum() >= self.min_samples_pre:
            base_name = "first_half"; base_mask = masks["first_half"]
        else:
            raise ValueError("CUSUM: not enough samples for baseline window.")

        full_mask = masks.get("full", np.ones(len(df), bool))
        df_full = df.loc[full_mask, feats]
        df_base = df.loc[base_mask, feats]

        # compute per-feature
        scores: List[Dict[str, Any]] = []
        for f in feats:
            x_full = df_full[f].to_numpy(dtype=float, copy=False)
            x_base = df_base[f].to_numpy(dtype=float, copy=False)
            x_full = x_full[np.isfinite(x_full)]
            x_base = x_base[np.isfinite(x_base)]
            undefined = (x_base.size < self.min_samples_pre)

            if undefined:
                scores.append({
                    "feature": f,
                    "score": 0.0,
                    "details": {"undefined": True}
                })
                continue

            mu0 = float(np.mean(x_base))
            sd0 = float(np.std(x_base, ddof=1)) if x_base.size > 1 else 0.0
            if not np.isfinite(sd0) or sd0 <= self.tiny_sigma:
                scores.append({
                    "feature": f,
                    "score": 0.0,
                    "details": {
                        "undefined": True,
                        "mu0": mu0, "sigma0": sd0, "n_pre": int(x_base.size)
                    }
                })
                continue

            # standardize across the full window (respect original time order)
            z = _standardize(x_full, mu0, sd0, self.tiny_sigma)

            Splus, Smax_plus, arg_plus = _cusum_one_sided(z, self.k, sign=+1)
            Sminus, Smax_minus, arg_minus = _cusum_one_sided(z, self.k, sign=-1)

            if self.side == "up":
                score = Smax_plus; direction = "up"; arg_idx = arg_plus
            elif self.side == "down":
                score = Smax_minus; direction = "down"; arg_idx = arg_minus
            else:
                if Smax_plus >= Smax_minus:
                    score = Smax_plus; direction = "up"; arg_idx = arg_plus
                else:
                    score = Smax_minus; direction = "down"; arg_idx = arg_minus

            # map arg_idx back to timestamp (epoch seconds)
            if arg_idx >= 0:
                # arg_idx is within df_full order; find its absolute index in df
                full_indices = np.where(full_mask)[0]
                abs_i = int(full_indices[arg_idx]) if arg_idx < full_indices.size else None
                ts = df.index[abs_i].value // 10**9 if abs_i is not None else None
            else:
                ts = None

            scores.append({
                "feature": f,
                "score": float(score),
                "details": {
                    "undefined": False,
                    "direction": direction,
                    "Smax": float(score),
                    "t_at_Smax": int(ts) if ts is not None else None,
                    "mu0": float(mu0),
                    "sigma0": float(sd0),
                    "n_pre": int(x_base.size),
                    "baseline_window": base_name,
                }
            })

        return {
            "method": self.name,
            "meta": {
                "side": self.side,
                "k": self.k,
                "baseline_window": self.baseline_window,
                "resolved_baseline_window": base_name,
                "min_samples_pre": self.min_samples_pre,
                "tiny_sigma": self.tiny_sigma,
                "scaling": "identity",  # CUSUM is already an effect-size-like score
            },
            "scores": scores,
        }
