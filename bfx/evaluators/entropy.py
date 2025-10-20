from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..common.ts import make_index, window_masks, numeric_feature_cols
from ..common.numerics import EPS

def _hist_entropy_bits(x: np.ndarray, edges: np.ndarray, base: float = 2.0, eps: float = EPS) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    counts, _ = np.histogram(x, bins=edges)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / float(n)
    if base == 2.0:
        return float(-np.sum(p * np.log2(p)))
    return float(-np.sum(p * (np.log(p + eps) / np.log(base))))

class EntropyEvaluator(FeatureEvaluator):
    """
    Shannon entropy over per-minute feature values (by window).
    Returns per-window min–max scaled scores (0..1) and raw deltas (during-pre, post-pre).
    """
    name = "entropy"

    def __init__(self, bins: Any = 20, base: float = 2.0, scaling: str = "minmax", epsilon: float = EPS) -> None:
        self.bins = bins
        self.base = float(base)
        self.scaling = scaling
        self.epsilon = float(epsilon)

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = list(numeric_feature_cols(df))
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for entropy evaluation.")

        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback="halves",
        )

        # compute per-feature histogram edges ONCE from the full investigation slice
        base_mask = np.logical_or.reduce(list(masks.values())) if masks else np.ones(len(df), bool)
        edges_by_feat: Dict[str, np.ndarray] = {}
        for f in feats:
            x_full = df.loc[base_mask, f].to_numpy(dtype=float, copy=False)
            x_full = x_full[np.isfinite(x_full)]
            if x_full.size > 0 and (np.nanmax(x_full) != np.nanmin(x_full)):
                edges_by_feat[f] = np.histogram_bin_edges(x_full, bins=self.bins)
            else:
                edges_by_feat[f] = np.array([0.0, 1.0], dtype=float)  # degenerate → entropy 0

        # raw entropies per window
        raw_by_win: Dict[str, Dict[str, float]] = {}
        for wname, mask in masks.items():
            sub = df.loc[mask, feats]
            vals: Dict[str, float] = {}
            for f in feats:
                vals[f] = _hist_entropy_bits(sub[f].to_numpy(dtype=float, copy=False),
                                             edges_by_feat[f], base=self.base, eps=self.epsilon)
            raw_by_win[wname] = vals

        # per-window min–max scaling across features
        def scale_minmax(d: Dict[str, float]) -> Dict[str, float]:
            xs = np.array(list(d.values()), dtype=float)
            if xs.size == 0:
                return {}
            lo, hi = np.nanmin(xs), np.nanmax(xs)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                return {k: 0.0 for k in d.keys()}
            return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}

        scores_by_win = {w: scale_minmax(raw) for w, raw in raw_by_win.items()}

        def pack_scores(d: Dict[str, float]) -> List[Dict[str, Any]]:
            return [{"feature": f, "score": float(d[f])} for f in sorted(d.keys())]

        windows: Dict[str, Any] = {w: {"scores": pack_scores(scores_by_win[w])} for w in scores_by_win.keys()}

        deltas: Dict[str, List[Dict[str, Any]]] = {}
        if "pre" in scores_by_win and "during" in scores_by_win:
            deltas["during_minus_pre"] = [
                {"feature": f, "delta": float(scores_by_win["during"].get(f, 0.0) - scores_by_win["pre"].get(f, 0.0))}
                for f in feats
            ]
        if "pre" in scores_by_win and "post" in scores_by_win:
            deltas["post_minus_pre"] = [
                {"feature": f, "delta": float(scores_by_win["post"].get(f, 0.0) - scores_by_win["pre"].get(f, 0.0))}
                for f in feats
            ]

        full_scores = pack_scores(scores_by_win["full"]) if "full" in scores_by_win else None

        return {
            "method": self.name,
            "meta": {
                "variant": "shannon_histogram",
                "bins": self.bins,
                "base": self.base,
                "scaling": self.scaling,
                "epsilon": self.epsilon,
                "edges": "global_per_feature",
            },
            "windows": windows,
            "deltas": deltas,
            **({"scores_full": full_scores} if full_scores is not None else {}),
        }
