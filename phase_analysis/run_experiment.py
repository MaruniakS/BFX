from __future__ import annotations

import argparse
from typing import Optional

from .config import EventConfig, MetricConfig, OutputConfig, PhaseAnalysisConfig
from .pipeline import run_event_experiment


def _scenario_to_event_config(name: str) -> EventConfig:
    from demos.scenarios import SCENARIOS

    if name not in SCENARIOS:
        raise KeyError(f"Scenario '{name}' not found")
    scenario = SCENARIOS[name]
    return EventConfig(
        input_path=scenario["input"],
        event_id=str(scenario.get("anomaly_name", name)),
        anomaly_type="scenario",
        label_value=-999,
        start_time=int(scenario["start_time"]),
        end_time=int(scenario["end_time"]),
        anomaly_start_time=int(scenario["anomaly_start_time"]),
        anomaly_end_time=int(scenario["anomaly_end_time"]),
        period_minutes=int(scenario.get("period", 1)),
        features=scenario.get("features"),
        segmentation_mode="timestamps",
    )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--input")
    parser.add_argument("--event-id")
    parser.add_argument("--anomaly-type", default="custom")
    parser.add_argument("--label-value", type=int, default=-999)
    parser.add_argument("--start-time", type=int)
    parser.add_argument("--end-time", type=int)
    parser.add_argument("--anomaly-start-time", type=int)
    parser.add_argument("--anomaly-end-time", type=int)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--segmentation-mode", default="timestamps")
    parser.add_argument("--metrics", default="mean,std,cv,median,mad,entropy,ks,auc,cusum")
    parser.add_argument("--entropy-bins", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--subdir", default="phase_analysis")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.scenario:
        event = _scenario_to_event_config(args.scenario)
    else:
        event = EventConfig(
            input_path=args.input,
            event_id=args.event_id,
            anomaly_type=args.anomaly_type,
            label_value=int(args.label_value),
            start_time=args.start_time,
            end_time=args.end_time,
            anomaly_start_time=args.anomaly_start_time,
            anomaly_end_time=args.anomaly_end_time,
            period_minutes=int(args.period),
            features=None,
            segmentation_mode=str(args.segmentation_mode),
        )

    config = PhaseAnalysisConfig(
        event=event,
        metric=MetricConfig(
            metrics=[item.strip() for item in args.metrics.split(",") if item.strip()],
            entropy_bins=int(args.entropy_bins),
            epsilon=float(args.epsilon),
            min_samples=int(args.min_samples),
        ),
        output=OutputConfig(
            root_dir=args.outdir,
            subdir=args.subdir,
            export_json=not args.no_json,
            export_csv=not args.no_csv,
            export_plots=not args.no_plots,
        ),
    )

    result = run_event_experiment(config)
    print("[PhaseAnalysis] event:", result["event_id"])
    print("[PhaseAnalysis] anomaly_type:", result["anomaly_type"])
    print("[PhaseAnalysis] outputs:", result["exports"])


if __name__ == "__main__":
    main()
