from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from .config import EventConfig


@dataclass
class Segment:
    name: str
    frame: pd.DataFrame
    start_index: int
    end_index: int


def _parse_timestamp(value: str) -> int:
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _numeric_feature_columns(frame: pd.DataFrame) -> List[str]:
    return [
        column for column in frame.columns
        if column not in {"timestamp", "label"} and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _segments_from_labels(frame: pd.DataFrame) -> Dict[str, Segment]:
    if "label" not in frame.columns:
        raise ValueError("Label-based segmentation requested, but 'label' column is missing.")
    labels = frame["label"].tolist()
    anomaly_idx = [idx for idx, value in enumerate(labels) if value != -1]
    if not anomaly_idx:
        raise ValueError("No anomaly segment found in label column.")
    start = anomaly_idx[0]
    end = anomaly_idx[-1]
    n = len(frame)
    if start == 0 or end == n - 1:
        raise ValueError("Need non-empty pre and post segments for phase coefficients.")
    return {
        "pre": Segment("pre", frame.iloc[:start].copy(), 0, start - 1),
        "during": Segment("during", frame.iloc[start:end + 1].copy(), start, end),
        "post": Segment("post", frame.iloc[end + 1:].copy(), end + 1, n - 1),
    }


def _segments_from_timestamps(frame: pd.DataFrame, config: EventConfig) -> Dict[str, Segment]:
    required = [
        config.start_time,
        config.end_time,
        config.anomaly_start_time,
        config.anomaly_end_time,
    ]
    if any(value is None for value in required):
        raise ValueError("Timestamp-based segmentation requires full window and anomaly bounds.")

    if "timestamp" in frame.columns:
        timestamps = frame["timestamp"].astype(str).map(_parse_timestamp).tolist()
    else:
        base = int(config.start_time)
        step = int(config.period_minutes) * 60
        timestamps = [base + i * step for i in range(len(frame))]

    pre_idx = [i for i, ts in enumerate(timestamps) if int(config.start_time) <= ts < int(config.anomaly_start_time)]
    during_idx = [i for i, ts in enumerate(timestamps) if int(config.anomaly_start_time) <= ts <= int(config.anomaly_end_time)]
    post_idx = [i for i, ts in enumerate(timestamps) if int(config.anomaly_end_time) < ts <= int(config.end_time)]

    if not pre_idx or not during_idx or not post_idx:
        raise ValueError("Could not form non-empty pre/during/post segments from timestamps.")

    return {
        "pre": Segment("pre", frame.iloc[pre_idx].copy(), pre_idx[0], pre_idx[-1]),
        "during": Segment("during", frame.iloc[during_idx].copy(), during_idx[0], during_idx[-1]),
        "post": Segment("post", frame.iloc[post_idx].copy(), post_idx[0], post_idx[-1]),
    }


def segment_event(frame: pd.DataFrame, config: EventConfig) -> Dict[str, Segment]:
    if config.segmentation_mode == "labels":
        segments = _segments_from_labels(frame)
    else:
        segments = _segments_from_timestamps(frame, config)

    numeric_cols = _numeric_feature_columns(frame)
    for name, segment in segments.items():
        segments[name] = Segment(
            name=name,
            frame=segment.frame[numeric_cols].copy(),
            start_index=segment.start_index,
            end_index=segment.end_index,
        )
    return segments
