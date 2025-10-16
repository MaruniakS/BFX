from __future__ import annotations
from typing import Any, Dict, List, Optional

from bfx.core.dataset import Dataset
from bfx.charts import plot_delta_bar, plot_triplet_bars, plot_scatter_xy
from .base import Explainer


class EntropyExplainer(Explainer):
    """Entropy: Δ(during−pre) bar (if available) + pre/during/post triplets + pre-vs-during scatter."""
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
        arts = set(cfg.get("artifacts") or ["delta_bar", "triplets", "scatter_pd"])
        dkey = str(cfg.get("delta_key", "during_minus_pre"))

        # --- window scores (same contract as CV) ---
        wins = method_block.get("windows") or {}
        def wmap(name: str) -> Dict[str, float]:
            rows = (wins.get(name) or {}).get("scores") or []
            return {r["feature"]: float(r.get("score", 0.0))
                    for r in rows if isinstance(r, dict) and "feature" in r}

        pre  = wmap("pre")
        dur  = wmap("during")
        post = wmap("post")
        if not (pre or dur or post):
            return []  # nothing to plot

        # Available features in block; apply user selection; then rank
        avail = sorted(set(pre.keys()) | set(dur.keys()) | set(post.keys()))
        sel = [f for f in (features or avail) if f in avail] or avail

        # --- deltas (same contract as CV: list-of-dicts) ---
        deltas_rows = (method_block.get("deltas") or {}).get(dkey, [])
        delta_map = {r["feature"]: float(r.get("delta", 0.0))
                     for r in deltas_rows if isinstance(r, dict) and "feature" in r}

        # rank: by deltas if present; else by (during - pre)
        if delta_map:
            sel.sort(key=lambda f: delta_map.get(f, 0.0), reverse=True)
        else:
            sel.sort(key=lambda f: float(dur.get(f, 0.0) - pre.get(f, 0.0)), reverse=True)

        if top_k is not None:
            sel = sel[: int(top_k)]
        if not sel:
            return []

        # vectors for plotting
        preV  = [pre.get(f, 0.0) for f in sel]
        durV  = [dur.get(f, 0.0) for f in sel]
        postV = [post.get(f, 0.0) for f in sel]

        paths: List[str] = []

        # 1) Δ bar (only if deltas exist)
        if "delta_bar" in arts and delta_map:
            p = outdir / cfg["filenames"]["delta_bar"]
            plot_delta_bar(
                sel,
                [delta_map.get(f, 0.0) for f in sel],
                title="Entropy — Δ (during − pre)",
                xlabel="Δ score (during − pre)",
                save_path=str(p),
            )
            paths.append(str(p))

        # 2) Triplets (always)
        if "triplets" in arts:
            p = outdir / cfg["filenames"]["triplets"]
            plot_triplet_bars(
                sel, preV, durV, postV,
                title="Entropy — Pre / During / Post",
                xlabel="scaled entropy [0,1]",
                save_path=str(p),
            )
            paths.append(str(p))

        # 3) Scatter pre vs during (always helpful)
        if "scatter_pd" in arts:
            p = outdir / cfg["filenames"]["scatter_pd"]
            plot_scatter_xy(
                preV, durV, sel,
                title="Entropy — pre vs during",
                xlabel="pre (scaled)",
                ylabel="during (scaled)",
                save_path=str(p),
            )
            paths.append(str(p))

        return paths
