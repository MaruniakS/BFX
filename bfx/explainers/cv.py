from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np

from bfx.core.dataset import Dataset
from bfx.charts import plot_delta_bar, plot_triplet_bars
from .base import Explainer

class CVExplainer(Explainer):
    name = "cv"

    def default_params(self) -> Dict[str, Any]:
        return {
            **super().default_params(),
            "delta_key": "during_minus_pre",
            "artifacts": ["delta_bar", "triplets"],  # subset if you like
            "filenames": {
                "delta_bar": "cv_delta_bar.png",
                "triplets": "cv_triplets.png",
            },
        }

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
        arts = set(cfg.get("artifacts") or ["delta_bar", "triplets"])

        # Need windows/deltas for CV visual explanation
        windows = method_block.get("windows") or {}
        deltas_all = (method_block.get("deltas") or {}).get(delta_key, [])
        if not windows or not deltas_all:
            return []

        # Filter deltas to the selected feature subset
        feat_set = set(features) if features is not None else None
        deltas_f = [r for r in deltas_all if (feat_set is None or r.get("feature") in feat_set)]

        # Apply top_k if provided (rank by delta desc), else keep all filtered
        if top_k is not None:
            deltas_f = sorted(deltas_f, key=lambda r: float(r.get("delta", 0.0)), reverse=True)[: int(top_k)]
        feats = [r["feature"] for r in deltas_f]
        if not feats:
            return []

        paths: List[str] = []

        if "delta_bar" in arts:
            vals = [float(r["delta"]) for r in deltas_f]
            p = outdir / cfg["filenames"]["delta_bar"]
            plot_delta_bar(
                feats,
                vals,
                title=f"CV — Δ ({delta_key.replace('_', ' ')})",
                xlabel=f"Δ score ({delta_key.replace('_', ' ')})",
                save_path=str(p),
            )
            paths.append(str(p))

        if "triplets" in arts:
            def _win_map(name: str) -> Dict[str, float]:
                return {d["feature"]: float(d["score"])
                        for d in (windows.get(name, {}).get("scores", []))}
            pre, dur, post = _win_map("pre"), _win_map("during"), _win_map("post")
            rows = [(f, pre.get(f, np.nan), dur.get(f, np.nan), post.get(f, np.nan)) for f in feats]
            p = outdir / cfg["filenames"]["triplets"]
            plot_triplet_bars(
                feats,
                [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                title="CV — Pre / During / Post",
                xlabel="scaled CV [0,1]",
                save_path=str(p),
            )
            paths.append(str(p))

        return paths
