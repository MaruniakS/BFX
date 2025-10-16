from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional

from bfx.core.dataset import Dataset
from bfx.core.registry import get_explainer_class


class FeaturesExplainer:
    """
    Dispatch explainers by method name (mirrors FeaturesExaminer).

    Params:
      - methods: list[str]                  # which method blocks to explain (e.g., ["cv"])
      - explainer_params: dict[str, dict]   # per-method config (e.g., {"cv": {"delta_key": "...", ...}})
      - features: None | list[str]          # None -> all features; list -> restrict to these labels
      - top_k: None | int                   # None -> no cut; int -> cut within the selected set
    """
    def __init__(self, dataset: Dataset, result: Dict[str, Any]) -> None:
        self.dataset = dataset
        self.result = result
        self.params: Dict[str, Any] = {
            "methods": [],
            "explainer_params": {},
            "features": None,   # None -> all; or a list of feature labels
            "top_k": None,      # None -> use all; or int to pick top-N inside selection
        }

    def setParams(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if k in self.params:
                self.params[k] = v
            else:
                sys.exit("Unrecognized parameter:" + k)

    def run(self) -> Dict[str, List[str]]:
        outputs: Dict[str, List[str]] = {}
        res_by_method = self.result.get("results") or {}

        # Determine the global feature selection once (per method we’ll intersect with what exists)
        user_feats: Optional[List[str]] = self.params.get("features")
        global_top_k: Optional[int] = self.params.get("top_k")

        for name in (self.params.get("methods") or []):
            block = res_by_method.get(name)
            if not block:
                continue

            # Available features for this method (from its scores)
            method_feats = [row.get("feature") for row in (block.get("scores") or []) if "feature" in row]
            if not method_feats:
                continue

            # Apply user selection if provided; else use all method features
            if user_feats is None:
                selected_feats = method_feats
            else:
                want = set(user_feats)
                selected_feats = [f for f in method_feats if f in want]
                if not selected_feats:
                    # nothing to explain for this method; skip silently
                    continue

            # Resolve explainer class
            expl_cls = get_explainer_class(name)
            if expl_cls is None:
                continue

            # Per-method params override; top_k can be overridden per method, else inherit global
            per_cfg = (self.params.get("explainer_params") or {}).get(name, {}) or {}
            top_k = per_cfg.get("top_k", global_top_k)

            expl = expl_cls(**per_cfg)
            paths = expl.explain(self.dataset, block, features=selected_feats, top_k=top_k)
            outputs[name] = paths

        return outputs
