from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..common.ts import make_index, window_masks, numeric_feature_cols

def _mann_whitney_auc(pre: np.ndarray, dur: np.ndarray, ties: str = "average") -> float:
    pre = np.asarray(pre, float); pre = pre[np.isfinite(pre)]
    dur = np.asarray(dur, float); dur = dur[np.isfinite(dur)]
    m, n = pre.size, dur.size
    if m == 0 or n == 0:
        return 0.5
    z = np.concatenate([pre, dur])
    ranks = pd.Series(z).rank(method=ties).to_numpy(dtype=float)
    R_dur = float(np.sum(ranks[m:]))
    U = R_dur - n * (n + 1) / 2.0
    auc = U / float(m * n)
    return float(min(1.0, max(0.0, auc)))

class AUCEvaluator(FeatureEvaluator):
    """
    AUC / Cliff's Δ separability between two windows (default: pre vs during).
    score = |Δ| where Δ = 2*AUC - 1.
    """
    name = "auc"

    def __init__(
        self,
        window_pair: Tuple[str, str] = ("pre", "during"),
        min_samples_per_window: int = 5,
        fallback: str = "halves",
        ties: str = "average",
        scaling: str = "identity",
    ) -> None:
        self.window_pair = tuple(window_pair)
        self.min_samples_per_window = int(max(1, min_samples_per_window))
        self.fallback = str(fallback)
        self.ties = str(ties)
        self.scaling = str(scaling)

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = list(numeric_feature_cols(df))
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for AUC evaluation.")

        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback=self.fallback,
        )

        a, b = self.window_pair
        if a not in masks or b not in masks or (masks[a].sum() < self.min_samples_per_window) or (masks[b].sum() < self.min_samples_per_window):
            if self.fallback == "halves" and "first_half" in masks and "second_half" in masks:
                a, b = "first_half", "second_half"
            else:
                raise ValueError("AUC: requested windows unavailable and no valid fallback.")

        scores: List[Dict[str, any]] = []
        for f in feats:
            x_a = df.loc[masks[a], f].to_numpy(dtype=float, copy=False)
            x_b = df.loc[masks[b], f].to_numpy(dtype=float, copy=False)
            n_a = int(np.isfinite(x_a).sum())
            n_b = int(np.isfinite(x_b).sum())

            if n_a < self.min_samples_per_window or n_b < self.min_samples_per_window:
                scores.append({
                    "feature": f,
                    "score": 0.0,
                    "details": {"undefined": True, "window_a": a, "window_b": b, "n_a": n_a, "n_b": n_b}
                })
                continue

            auc = _mann_whitney_auc(x_a, x_b, ties=self.ties)
            delta = 2.0 * auc - 1.0
            direction = "up" if delta > 0 else ("down" if delta < 0 else "none")
            scores.append({
                "feature": f,
                "score": float(abs(delta)),
                "details": {
                    "undefined": False,
                    "auc": float(auc),
                    "delta": float(delta),
                    "direction": direction,
                    "window_a": a, "window_b": b,
                    "n_a": n_a, "n_b": n_b,
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
