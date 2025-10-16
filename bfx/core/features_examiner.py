from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bfx.core.dataset import Dataset
from bfx.core.registry import get_evaluator_class
from bfx.utils import save_json


class FeaturesExaminer:
    """
    Runs one or more evaluators over the Dataset and returns a combined result.
    Params:
      - methods: List[str], e.g., ["entropy", "cv"]
      - features: None or List[str] -> which columns to use
      - evaluator_params: Dict[str, Dict] -> per-method params, e.g. {"entropy": {"nbins": 20}}
      - output_name: optional filename (without .json) to save under out/<anomaly_name>/
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.params: Dict[str, Any] = {
            "methods": [],           # list of evaluator names
            "features": None,        # None -> all; or List[str]
            "evaluator_params": {},  # per-method param dict
            "output_name": None,     # e.g., "google-leak-evals"
        }

    def setParams(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if k in self.params:
                self.params[k] = v
            else:
                sys.exit("Unrecognized parameter:" + k)

    def run(self) -> Tuple[Dict[str, Any], Optional[Path]]:
        methods: List[str] = list(self.params["methods"])
        features = self.params.get("features")
        per_params: Dict[str, Dict[str, Any]] = self.params.get("evaluator_params") or {}

        if not methods:
            raise ValueError("No methods specified. Set params: {'methods': ['entropy', ...]}")

        all_results: List[Dict[str, Any]] = []

        for mname in methods:
            cls = get_evaluator_class(mname)
            ev_params = per_params.get(mname, {})
            evaluator = cls(**ev_params)
            res = evaluator.evaluate(self.dataset, features=features)
            all_results.append(res)

        # meta
        data = self.dataset.data or []
        n_rows = len(data) if isinstance(data, list) else 0
        feature_count = len(data[0]) if n_rows and isinstance(data[0], dict) else 0

        combined: Dict[str, Any] = {
            "methods": methods,
            "input_meta": {
                "n_rows": n_rows,
                "feature_count_inferred": feature_count,
                "source": {
                    "type": "file",
                    "path": str(self.dataset.input_path) if self.dataset.input_path else str(self.dataset.params.get("input")),
                },
            },
            "results": {r["method"]: r for r in all_results},
        }

        # optional save
        saved_path: Optional[Path] = None
        out_name = self.params.get("output_name")
        if out_name:
            saved_path = save_json(
                combined,
                group=str(self.dataset.params.get("anomaly_name")),
                filename=str(out_name),
                root=self.dataset.params.get("outdir"),
            )
            print(f"Saved examiner results to: {saved_path or '(not saved)'}")

        return combined, saved_path
