from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from bfx.utils import load_json


class Dataset:
    """
    Minimal container for input data + shared parameters.
    - Initialize with defaults.
    - Update via setParams(params).
    - When 'input' is set, JSON is loaded and stored as self.data.
    """

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {
            "input": None,               # str path to input JSON
            "outdir": "out",             # output root directory
            "anomaly_name": "Anomaly",   # logical group/folder
            "start_time": None,          # investigation window start (unix ts)
            "end_time": None,            # investigation window end (unix ts)
            "anomaly_start_time": None,  # optional actual anomaly start (unix ts)
            "anomaly_end_time": None,    # optional actual anomaly end (unix ts)
            "period": None,              # integer minutes between snapshots
        }
        self.data: Optional[Any] = None
        self._input_path: Optional[Path] = None

    def setParams(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if k in self.params:
                self.params[k] = v
            else:
                sys.exit("Unrecognized parameter:" + k)

        # If input path provided/changed, (re)load JSON
        if "input" in params:
            inp = params.get("input")
            if inp is None:
                self.data = None
                self._input_path = None
            else:
                p = Path(str(inp))
                if not p.exists():
                    sys.exit(f"Input file not found: {p}")
                self.data = load_json(p)
                self._input_path = p

    # Optional simple accessors if needed later (do nothing extra)
    @property
    def input_path(self) -> Optional[Path]:
        return self._input_path
