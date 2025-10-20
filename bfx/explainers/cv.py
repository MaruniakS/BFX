from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bfx.core.dataset import Dataset
from bfx.charts import plot_delta_bar, plot_triplet_bars, plot_scatter_xy
from bfx.common.ts import make_index, window_masks

from .base import Explainer


def _cv_classic(x: np.ndarray, ddof: int, eps: float) -> float:
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    mu = float(np.mean(x))
    denom = abs(mu) + float(eps)
    if denom == 0.0:
        return np.nan
    sd = float(np.std(x, ddof=ddof))
    return sd / denom

def _cv_robust_mad(x: np.ndarray, eps: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = float(np.median(x))
    denom = abs(med) + float(eps)
    if denom == 0.0:
        return np.nan
    mad = float(np.median(np.abs(x - med)))
    sigma_hat = 1.4826 * mad
    return sigma_hat / denom

def _minmax_scale_map(d: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([v for v in d.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {k: 0.0 for k in d.keys()}
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return {k: 0.0 for k in d.keys()}
    return {k: float((d[k] - lo) / (hi - lo)) if np.isfinite(d[k]) else 0.0 for k in d.keys()}


class CVExplainer(Explainer):
    name = "cv"

    def default_params(self) -> Dict[str, Any]:
        return {
            **super().default_params(),
            "delta_key": "during_minus_pre",
            "artifacts": ["delta_bar", "triplets", "scatter_pd"],
            "filenames": {
                "delta_bar": "cv_delta_bar.png",
                "triplets": "cv_triplets.png",
                "scatter_pd": "cv_scatter_pre_vs_during.png",
            },
        }

    # ---- Fallback: compute windowed CVs if evaluator didn't provide them ----
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
        # Build DF with same time index
        df = make_index(
            pd.DataFrame(dataset.data),
            dataset.params.get("start_time"),
            int(dataset.params.get("period") or 1),
        )

        # Window masks (fallback halves if anomaly missing)
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

        # Match evaluator variant if present; else default to robust MAD
        meta = dict(method_block.get("meta") or {})
        variant = str(meta.get("variant") or "robust_mad")
        ddof = int(meta.get("ddof") or 1)
        eps = float(meta.get("epsilon") or 0.0)

        def _cv_for_mask(mask: np.ndarray) -> Dict[str, float]:
            raw: Dict[str, float] = {}
            for f in feats:
                x = df.loc[mask, f].to_numpy(dtype=float, copy=False)
                if variant == "classic":
                    v = _cv_classic(x, ddof, eps)
                else:
                    v = _cv_robust_mad(x, eps)
                raw[f] = v
            return _minmax_scale_map(raw)  # scale per-window across features

        win_scores: Dict[str, Dict[str, float]] = {}
        if "pre" in masks and masks["pre"].any():
            win_scores["pre"] = _cv_for_mask(masks["pre"])
        if "during" in masks and masks["during"].any():
            win_scores["during"] = _cv_for_mask(masks["during"])
        if "post" in masks and masks["post"].any():
            win_scores["post"] = _cv_for_mask(masks["post"])

        # Deltas for requested key
        deltas: List[Dict[str, Any]] = []
        if delta_key == "during_minus_pre" and "pre" in win_scores and "during" in win_scores:
            for f in feats:
                deltas.append({"feature": f, "delta": float(win_scores["during"].get(f, 0.0) - win_scores["pre"].get(f, 0.0))})
        elif delta_key == "post_minus_pre" and "pre" in win_scores and "post" in win_scores:
            for f in feats:
                deltas.append({"feature": f, "delta": float(win_scores["post"].get(f, 0.0) - win_scores["pre"].get(f, 0.0))})
        # else: leave empty (caller will handle and likely skip plotting)

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

        # Try the evaluator-provided windows/deltas first
        windows = method_block.get("windows") or {}
        deltas_all = (method_block.get("deltas") or {}).get(delta_key, [])

        # If missing, compute a local fallback set directly from the dataset
        if not windows or not deltas_all:
            win_map, deltas_all = self._fallback_windows_and_deltas(dataset, method_block, features, delta_key)
            if not win_map or not deltas_all:
                # nothing to plot
                return []
            # emulate the "windows" structure expected by plotting code
            def _pack_scores(d: Dict[str, float]) -> Dict[str, Any]:
                return {"scores": [{"feature": f, "score": float(d[f])} for f in sorted(d.keys())]}
            windows = {w: _pack_scores(win_map[w]) for w in win_map.keys()}

        # Filter deltas to the selected feature subset (if provided)
        feat_set = set(features) if features is not None else None
        deltas_f = [r for r in deltas_all if (feat_set is None or r.get("feature") in feat_set)]

        # Rank by delta (desc) and apply top_k if provided
        if top_k is not None:
            deltas_f = sorted(deltas_f, key=lambda r: float(r.get("delta", 0.0)), reverse=True)[: int(top_k)]
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
                title=f"CV — Δ ({delta_key.replace('_', ' ')})",
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
                title="CV — Pre / During / Post",
                xlabel="scaled CV [0,1]",
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
                    title="CV — Pre vs During (per feature)",
                    xlabel="pre (scaled CV)",
                    ylabel="during (scaled CV)",
                    save_path=str(p),
                )
                paths.append(str(p))

        return paths
