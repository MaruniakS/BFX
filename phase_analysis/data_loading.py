from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import EventConfig


@dataclass
class EventData:
    frame: pd.DataFrame
    rows: List[Dict[str, Any]]
    config: EventConfig


def load_event_data(config: EventConfig) -> EventData:
    rows = json.loads(Path(config.input_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty event list in {config.input_path}")

    frame = pd.DataFrame(rows)
    for column in list(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="ignore")

    if config.features is not None:
        keep = [name for name in config.features if name in frame.columns]
        if not keep:
            raise ValueError("Requested features are not available in the event.")
        extra = [name for name in ["timestamp", "label"] if name in frame.columns]
        frame = frame[keep + extra]

    return EventData(frame=frame, rows=rows, config=config)
