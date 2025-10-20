# bfx/explainers/entropy_explainer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

from bfx.core.dataset import Dataset
from bfx.charts import plot_delta_bar, plot_triplet_bars, plot_scatter_xy
from .base import Explainer
from ..common.ts import make_index, window_masks


# ---------- local helpers (match evaluator math) ----------

def _hist_entropy_bits(x: np.ndarray, edges: np.ndarray, base: float = 2.0, eps: float = 1e-12) -> float:
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
    # generic base using change of base
    return float(-np.sum(p * (np.log(p + eps) / math.log(base))))

def _minmax_scale_map(d: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([v for v in d.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {k: 0.0 for k in d.keys()}
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return {k: 0.0 for k in d.keys()}
    return {k: float((d[k] - lo) / (hi - lo)) if np.isfinite(d[k]) else 0.0 for k in d.keys()}


class EntropyExplainer(Explainer):
    name = "entropy"

    def default_params(self) -> Dict[str, Any]:
        return {
            **super().default_params(),
            "delta_key": "during_minus_pre",
            "artifacts": ["delta_bar", "triplets", "scatter_pd"],
            "filenames": {
                "delta_bar": "entropy_delta_bar.png",
                "triplets": "entropy_triplets.png",
                "scatter_pd": "entropy_scatter_pre_vs_during.png",
            },
        }

    # ---- Fallback: compute windowed entropies if evaluator didn't provide them ----
    def _fallback_windows_and_deltas(
        self,
        dataset: Dataset,
        method_block: Dict[str, Any],
        features: Optional[List[str]],
        delta_key: str,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        """
        Returns (win_scores_map, deltas_list) compatible with what the explainer needs:
         - win_scores_map: {"pre": {feat: scaled}, "during": {...}, "post": {...}} (some keys may be missing)
         - deltas_list: [{"feature": f, "delta": during - pre}, ...] for the requested delta_key.
        """
        # Rebuild DF and masks
        df = make_index(
            pd.DataFrame(dataset.data),
            dataset.params.get("start_time"),
            int(dataset.params.get("period") or 1),
        )
        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback="halves",
        )

        # Feature universe
        all_feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        feats = features or all_feats
        if not feats:
            return {}, []

        # Mirror evaluator meta if available
        meta = dict(method_block.get("meta") or {})
        bins = meta.get("bins", 20)
        base = float(meta.get("base", 2.0))
        eps = float(meta.get("epsilon", 1e-12))

        # Compute global per-feature edges from the "full" slice of the investigation (masks union)
        base_mask = np.zeros(len(df), dtype=bool)
        for m in (masks or {}).values():
            base_mask |= m
        if not base_mask.any():  # fallback to entire df
            base_mask[:] = True

        edges_by_feat: Dict[str, np.ndarray] = {}
        for f in feats:
            x_full = df.loc[base_mask, f].to_numpy(dtype=float, copy=False)
            x_full = x_full[np.isfinite(x_full)]
            if x_full.size > 0 and (np.nanmax(x_full) != np.nanmin(x_full)):
                edges_by_feat[f] = np.histogram_bin_edges(x_full, bins=bins)
            else:
                edges_by_feat[f] = np.array([0.0, 1.0], dtype=float)  # degenerate → entropy 0

        # Raw entropies per available window
        raw_by_win: Dict[str, Dict[str, float]] = {}
        for wname in ("pre", "during", "post"):
            if wname in masks and masks[wname].any():
                sub = df.loc[masks[wname], feats]
                vals: Dict[str, float] = {}
                for f in feats:
                    vals[f] = _hist_entropy_bits(sub[f].to_numpy(dtype=float, copy=False),
                                                 edges_by_feat[f], base=base, eps=eps)
                raw_by_win[wname] = vals

        # Min–max scale across features within each window (match evaluator)
        win_scores = {w: _minmax_scale_map(raw_by_win[w]) for w in raw_by_win.keys()}

        # Deltas for requested key
        deltas: List[Dict[str, Any]] = []
        if delta_key == "during_minus_pre" and "pre" in win_scores and "during" in win_scores:
            for f in feats:
                deltas.append({"feature": f, "delta": float(win_scores["during"].get(f, 0.0) - win_scores["pre"].get(f, 0.0))})
        elif delta_key == "post_minus_pre" and "pre" in win_scores and "post" in win_scores:
            for f in feats:
                deltas.append({"feature": f, "delta": float(win_scores["post"].get(f, 0.0) - win_scores["pre"].get(f, 0.0))})

        return win_scores, deltas

    def explain(
        self,
        dataset: Dataset,
        method_block: Dict[str, Any],
        *,
        features: List[str],
        top_k: Optional[int],
    ) -> List[str]:
        outdir = self.base_dir(dataset)
        cfg = self.params
        delta_key = str(cfg["delta_key"])
        artifacts = set(cfg.get("artifacts") or ["delta_bar", "triplets", "scatter_pd"])

        # Prefer evaluator-provided windows & deltas
        windows = method_block.get("windows") or {}
        deltas_all = (method_block.get("deltas") or {}).get(delta_key, [])

        # If missing, compute a local fallback directly from the dataset
        if not windows or not deltas_all:
            win_map, deltas_all = self._fallback_windows_and_deltas(dataset, method_block, features, delta_key)
            if not win_map or not deltas_all:
                return []
            # Pack like the evaluator for plotting
            def _pack_scores(d: Dict[str, float]) -> Dict[str, Any]:
                return {"scores": [{"feature": f, "score": float(d[f])} for f in sorted(d.keys())]}
            windows = {w: _pack_scores(win_map[w]) for w in win_map.keys()}

        # Filter to explicit feature subset if provided
        feat_set = set(features) if features is not None else None
        deltas_f = [r for r in deltas_all if (feat_set is None or r.get("feature") in feat_set)]

        # Rank by |delta| desc (entropy deltas can be positive or negative)
        if top_k is not None:
            deltas_f = sorted(deltas_f, key=lambda r: abs(float(r.get("delta", 0.0))), reverse=True)[: int(top_k)]
        feats_ranked = [r["feature"] for r in deltas_f]
        if not feats_ranked:
            return []

        paths: List[str] = []

        # ---- Δ bar (during − pre) ----
        if "delta_bar" in artifacts:
            vals = [float(r["delta"]) for r in deltas_f]
            p = outdir / cfg["filenames"]["delta_bar"]
            plot_delta_bar(
                feats_ranked,
                vals,
                title=f"Entropy — Δ ({delta_key.replace('_', ' ')})",
                xlabel=f"Δ score ({delta_key.replace('_', ' ')})",
                save_path=str(p),
            )
            paths.append(str(p))

        # ---- triplets: pre / during / post ----
        if "triplets" in artifacts:
            def _win_map(name: str) -> Dict[str, float]:
                return {d["feature"]: float(d["score"])
                        for d in (windows.get(name, {}).get("scores", []))}
            pre_map, dur_map, post_map = _win_map("pre"), _win_map("during"), _win_map("post")
            rows = [(f, pre_map.get(f, np.nan), dur_map.get(f, np.nan), post_map.get(f, np.nan)) for f in feats_ranked]
            p = outdir / cfg["filenames"]["triplets"]
            plot_triplet_bars(
                feats_ranked,
                [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                title="Entropy — Pre / During / Post",
                xlabel="scaled entropy [0,1]",
                save_path=str(p),
            )
            paths.append(str(p))

        # ---- scatter: pre vs during (use all selected features present in both windows) ----
        if "scatter_pd" in artifacts and "pre" in windows and "during" in windows:
            pre_map = {d["feature"]: float(d["score"]) for d in (windows.get("pre", {}).get("scores", []))}
            dur_map = {d["feature"]: float(d["score"]) for d in (windows.get("during", {}).get("scores", []))}
            if feat_set is None:
                feats_scatter = sorted(set(pre_map.keys()) & set(dur_map.keys()))
            else:
                feats_scatter = sorted(list(set(feats_ranked) & set(pre_map.keys()) & set(dur_map.keys())))
            if feats_scatter:
                xvals = [pre_map.get(f, np.nan) for f in feats_scatter]
                yvals = [dur_map.get(f, np.nan) for f in feats_scatter]
                p = outdir / cfg["filenames"]["scatter_pd"]
                plot_scatter_xy(
                    xvals, yvals, feats_scatter,
                    title="Entropy — Pre vs During (per feature)",
                    xlabel="pre (scaled entropy)",
                    ylabel="during (scaled entropy)",
                    save_path=str(p),
                )
                paths.append(str(p))

        return paths
