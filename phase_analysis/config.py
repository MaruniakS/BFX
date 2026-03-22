from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


DEFAULT_METRICS = ["mean", "std", "cv", "median", "mad", "entropy", "ks", "auc", "cusum"]


@dataclass
class EventConfig:
    input_path: str
    event_id: str
    anomaly_type: str
    label_value: int
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    anomaly_start_time: Optional[int] = None
    anomaly_end_time: Optional[int] = None
    period_minutes: int = 1
    features: Optional[List[str]] = None
    segmentation_mode: str = "timestamps"


@dataclass
class MetricConfig:
    metrics: List[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    entropy_bins: int = 12
    epsilon: float = 1e-12
    min_samples: int = 5
    auc_ties: str = "average"
    cusum_k: float = 0.0


@dataclass
class OutputConfig:
    root_dir: str = "out"
    subdir: str = "phase_analysis"
    export_json: bool = True
    export_csv: bool = True
    export_plots: bool = True

    def event_dir(self, event_id: str) -> Path:
        path = Path(self.root_dir) / event_id / self.subdir
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class PhaseAnalysisConfig:
    event: EventConfig
    metric: MetricConfig = field(default_factory=MetricConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class BatchConfig:
    metadata_path: str
    output_root: str = "out/phase_analysis_batch"
    metric: MetricConfig = field(default_factory=MetricConfig)
    output_subdir: str = "phase_analysis"
