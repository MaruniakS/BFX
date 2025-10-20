from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..common.ts import make_index, window_masks, numeric_feature_cols
from ..common.numerics import TINY_SIGMA

def _standardize(x: np.ndarray, mu0: float, sd0: float) -> np.ndarray:
    if not np.isfinite(sd0) or abs(sd0) <= TINY_SIGMA:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu0) / sd0

def _cusum_one_sided(z: np.ndarray, k: float, sign: int) -> Tuple[np.ndarray, float, int]:
    """sign=+1 for S+, sign=-1 for S- (using -z). Returns (S curve, Smax, argmax)."""
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
    Score = max S+ or S- depending on 'side' ('both' takes the larger).
    """
    name = "cusum"

    def __init__(
        self,
        side: str = "both",
        k: float = 0.0,
        baseline_window: str = "auto",
        min_samples_pre: int = 5,
    ) -> None:
        self.side = side
        self.k = float(k)
        self.baseline_window = baseline_window
        self.min_samples_pre = int(max(1, min_samples_pre))

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = list(numeric_feature_cols(df))
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for CUSUM evaluation.")

        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback="halves",
        )

        # choose baseline: prefer pre; fallback to first_half
        if self.baseline_window == "pre" or (
            self.baseline_window == "auto" and "pre" in masks and masks["pre"].sum() >= self.min_samples_pre
        ):
            base_name, base_mask = "pre", masks["pre"]
        elif "first_half" in masks and masks["first_half"].sum() >= self.min_samples_pre:
            base_name, base_mask = "first_half", masks["first_half"]
        else:
            raise ValueError("CUSUM: not enough samples for baseline window.")

        full_mask = masks.get("full", np.ones(len(df), bool))
        df_full = df.loc[full_mask, feats]
        df_base = df.loc[base_mask, feats]
        full_indices = np.where(full_mask)[0]

        scores: List[Dict[str, any]] = []
        for f in feats:
            x_full = df_full[f].to_numpy(dtype=float, copy=False)
            x_base = df_base[f].to_numpy(dtype=float, copy=False)
            x_full = x_full[np.isfinite(x_full)]
            x_base = x_base[np.isfinite(x_base)]
            undefined = (x_base.size < self.min_samples_pre)

            if undefined:
                scores.append({"feature": f, "score": 0.0, "details": {"undefined": True}})
                continue

            mu0 = float(np.mean(x_base))
            sd0 = float(np.std(x_base, ddof=1)) if x_base.size > 1 else 0.0
            if not np.isfinite(sd0) or sd0 <= TINY_SIGMA:
                scores.append({
                    "feature": f, "score": 0.0,
                    "details": {"undefined": True, "mu0": mu0, "sigma0": sd0, "n_pre": int(x_base.size)}
                })
                continue

            z = _standardize(x_full, mu0, sd0)
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
            if arg_idx >= 0 and arg_idx < full_indices.size:
                abs_i = int(full_indices[arg_idx])
                ts = int(df.index[abs_i].value // 10**9)
            else:
                ts = None

            scores.append({
                "feature": f,
                "score": float(score),
                "details": {
                    "undefined": False,
                    "direction": direction,
                    "Smax": float(score),
                    "t_at_Smax": ts,
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
                "scaling": "identity",
            },
            "scores": scores,
        }
