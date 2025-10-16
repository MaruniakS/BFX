from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from bfx.core.dataset import Dataset
from .base import Explainer

class GenericScoresExplainer(Explainer):
    """Fallback: bar chart of top-K scores for any method."""
    name = "generic"

    def default_params(self) -> Dict[str, Any]:
        return {**super().default_params(), "top_k": 10}

    def explain(self, dataset: Dataset, method_block: Dict[str, Any]) -> List[str]:
        top_k = int(self.params["top_k"])
        scores = list(method_block.get("scores") or [])
        scores.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        scores = scores[:top_k]

        feats = [r["feature"] for r in scores]
        vals  = [float(r["score"]) for r in scores]
        y = np.arange(len(feats))

        outdir = self.base_dir(dataset)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.45*len(feats))))
        ax.barh(y, vals)
        ax.set_yticks(y, feats); ax.invert_yaxis()
        ax.set_xlabel("score [0,1]")
        ax.set_title(f"{method_block.get('method','method')} — Top-{top_k} by score")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out = outdir / f"{method_block.get('method','method')}_scores_bar.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        return [str(out)]
