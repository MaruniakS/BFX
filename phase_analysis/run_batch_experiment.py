from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .batch_summary import build_batch_outputs
from .config import BatchConfig, EventConfig, MetricConfig, OutputConfig, PhaseAnalysisConfig
from .pipeline import run_event_experiment


def _parse_timestamp(value: str) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _infer_event_config(item: Dict[str, object]) -> EventConfig:
    event_path = Path(str(item["output_file"]))
    rows = json.loads(event_path.read_text(encoding="utf-8"))
    labels = [row.get("label") for row in rows]
    anomaly_idx = [idx for idx, label in enumerate(labels) if label != -1]
    if not anomaly_idx:
        raise ValueError(f"No anomaly rows found for {item.get('event')}")
    return EventConfig(
        input_path=str(event_path),
        event_id=_slugify(str(item["event"])),
        anomaly_type=str(item.get("str_class", item.get("class", "unknown"))),
        label_value=int(labels[anomaly_idx[0]]),
        start_time=_parse_timestamp(str(rows[0]["timestamp"])),
        end_time=_parse_timestamp(str(rows[-1]["timestamp"])),
        anomaly_start_time=_parse_timestamp(str(rows[anomaly_idx[0]]["timestamp"])),
        anomaly_end_time=_parse_timestamp(str(rows[anomaly_idx[-1]]["timestamp"])),
        period_minutes=1,
        features=None,
        segmentation_mode="labels",
    )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="combined_events/anomalies_metadata.json")
    parser.add_argument("--metrics", default="mean,std,cv,median,mad,entropy,ks,auc,cusum")
    parser.add_argument("--entropy-bins", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--outdir", default="out/phase_analysis_batch")
    parser.add_argument("--subdir", default="phase_analysis")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    items = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    batch_rows: List[Dict[str, object]] = []
    metric_cfg = MetricConfig(
        metrics=[item.strip() for item in args.metrics.split(",") if item.strip()],
        entropy_bins=int(args.entropy_bins),
        epsilon=float(args.epsilon),
        min_samples=int(args.min_samples),
    )

    for item in items:
        event = _infer_event_config(item)
        config = PhaseAnalysisConfig(
            event=event,
            metric=metric_cfg,
            output=OutputConfig(
                root_dir=args.outdir,
                subdir=args.subdir,
                export_json=not args.no_json,
                export_csv=not args.no_csv,
                export_plots=not args.no_plots,
            ),
        )
        result = run_event_experiment(config)
        for row in result["rows"]:
            batch_rows.append(row)

    batch_dir = Path(args.outdir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    if batch_rows:
        summary_path = batch_dir / "phase_analysis_batch.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0].keys()))
            writer.writeheader()
            for row in batch_rows:
                writer.writerow(row)
        summary_exports = build_batch_outputs(batch_rows, batch_dir)
        print("[PhaseAnalysisBatch] rows:", len(batch_rows))
        print("[PhaseAnalysisBatch] batch_csv:", summary_path)
        for key, value in summary_exports.items():
            print(f"[PhaseAnalysisBatch] {key}:", value)


if __name__ == "__main__":
    main()
