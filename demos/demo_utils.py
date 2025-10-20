from __future__ import annotations
from typing import Tuple, Optional, List
from bfx.core import Dataset
from demos.scenarios import SCENARIOS

def build_dataset(scenario: str) -> Tuple[Dataset, Optional[List[str]]]:
    if scenario not in SCENARIOS:
        raise KeyError(f"Scenario '{scenario}' not found")
    s = SCENARIOS[scenario]

    ds = Dataset()
    ds.setParams({
        "input": s["input"],
        "outdir": s.get("outdir", "out"),
        "anomaly_name": s.get("anomaly_name", scenario),
        "start_time": s["start_time"],
        "end_time": s["end_time"],
        "anomaly_start_time": s.get("anomaly_start_time"),
        "anomaly_end_time": s.get("anomaly_end_time"),
        "period": s.get("period", 1),
    })

    return ds, s.get("features")
