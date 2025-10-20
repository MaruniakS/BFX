from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..common.ts import make_index, window_masks, numeric_feature_cols

def _ks_stat_and_x(a: np.ndarray, b: np.ndarray) -> Tuple[float, Optional[float]]:
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    m, n = a.size, b.size
    if m == 0 or n == 0:
        return 0.0, None
    a = np.sort(a); b = np.sort(b)
    grid = np.sort(np.unique(np.concatenate([a, b])))
    # ECDF values at right side
    Fa = np.searchsorted(a, grid, side="right") / float(m)
    Fb = np.searchsorted(b, grid, side="right") / float(n)
    diffs = np.abs(Fa - Fb)
    i = int(np.argmax(diffs))
    return float(diffs[i]), (float(grid[i]) if grid.size else None)

class KSEvaluator(FeatureEvaluator):
    """
    Two-sample KS distance D between two windows (default: pre vs during).
    score = D in [0,1].
    """
    name = "ks"

    def __init__(
        self,
        window_pair: Tuple[str, str] = ("pre", "during"),
        min_samples_per_window: int = 5,
        fallback: str = "halves",
        scaling: str = "identity",
    ) -> None:
        self.window_pair = tuple(window_pair)
        self.min_samples_per_window = int(max(1, min_samples_per_window))
        self.fallback = str(fallback)
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
            raise ValueError("No numeric features available for KS evaluation.")

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
                raise ValueError("KS: requested windows unavailable and no valid fallback.")

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

            D, x_star = _ks_stat_and_x(x_a, x_b)
            scores.append({
                "feature": f,
                "score": float(D),
                "details": {
                    "D": float(D),
                    "x_at_max_gap": (float(x_star) if x_star is not None else None),
                    "n_a": n_a, "n_b": n_b,
                    "window_a": a, "window_b": b,
                    "undefined": False
                }
            })

        return {
            "method": self.name,
            "meta": {
                "window_pair": [a, b],
                "min_samples_per_window": self.min_samples_per_window,
                "fallback": self.fallback,
                "scaling": self.scaling,
            },
            "scores": scores,
        }
