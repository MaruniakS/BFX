from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from bfx.core.dataset import Dataset

class Explainer:
    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = self.default_params()
        # strict but shallow
        for k, v in params.items():
            if k in self.params:
                self.params[k] = v
            else:
                import sys
                sys.exit("Unrecognized parameter:" + k)

    def default_params(self) -> Dict[str, Any]:
        return {"subdir": "explain"}  # <out>/<anomaly>/<subdir>/<method>/

    def base_dir(self, dataset: Dataset) -> Path:
        outdir = str(dataset.params.get("outdir") or "out")
        group  = str(dataset.params.get("anomaly_name") or "Anomaly")
        p = Path(outdir) / group / str(self.params.get("subdir") or "explain") / self.name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def explain(
        self,
        dataset: Dataset,
        method_block: Dict[str, Any],
        *,
        features: List[str],
        top_k: Optional[int],
    ) -> List[str]:
        raise NotImplementedError
