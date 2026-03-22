from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .base_metrics import compute_phase_metrics
from .config import PhaseAnalysisConfig
from .data_loading import load_event_data
from .export_results import write_csv, write_json
from .phase_coefficients import add_phase_coefficients
from .segmentation import segment_event
from .visualization import plot_scatter, plot_top_features


def _annotate_event_rankings(table: pd.DataFrame) -> tuple[pd.DataFrame, List[Dict[str, object]]]:
    annotated = table.copy()
    summary_rows: List[Dict[str, object]] = []

    annotated["contrast_threshold"] = pd.NA
    annotated["recovery_threshold"] = pd.NA
    annotated["phase_informative"] = False
    annotated["contrast_rank"] = pd.NA
    annotated["recovery_rank"] = pd.NA
    annotated["joint_score"] = pd.NA
    annotated["joint_rank"] = pd.NA

    for metric_name, subset in annotated.groupby("metric"):
        valid_mask = subset["metric_undefined"].astype(str).str.lower() != "true"
        valid = subset.loc[valid_mask].copy()
        if valid.empty:
            continue

        tau_c = float(valid["contrast_coefficient"].quantile(0.75))
        tau_r = float(valid["recovery_coefficient"].median())

        valid["phase_informative"] = (
            (valid["contrast_coefficient"] > tau_c)
            & (valid["recovery_coefficient"] < tau_r)
        )
        valid["contrast_rank"] = (
            valid["contrast_coefficient"].rank(method="min", ascending=False).astype(int)
        )
        valid["recovery_rank"] = (
            valid["recovery_coefficient"].rank(method="min", ascending=True).astype(int)
        )
        # Joint ranking favors strong anomaly contrast and small post-event displacement.
        valid["joint_score"] = (
            valid["contrast_coefficient"].clip(lower=0.0) / (1.0 + valid["recovery_coefficient"].clip(lower=0.0))
        )
        valid["joint_rank"] = valid["joint_score"].rank(method="min", ascending=False).astype(int)
        valid["contrast_threshold"] = tau_c
        valid["recovery_threshold"] = tau_r

        for row_index, row in valid.iterrows():
            annotated.loc[row_index, "contrast_threshold"] = row["contrast_threshold"]
            annotated.loc[row_index, "recovery_threshold"] = row["recovery_threshold"]
            annotated.loc[row_index, "phase_informative"] = bool(row["phase_informative"])
            annotated.loc[row_index, "contrast_rank"] = int(row["contrast_rank"])
            annotated.loc[row_index, "recovery_rank"] = int(row["recovery_rank"])
            annotated.loc[row_index, "joint_score"] = float(row["joint_score"])
            annotated.loc[row_index, "joint_rank"] = int(row["joint_rank"])

        summary_rows.append(
            {
                "metric": metric_name,
                "n_features": int(len(valid)),
                "contrast_threshold": tau_c,
                "recovery_threshold": tau_r,
                "selected_count": int(valid["phase_informative"].sum()),
                "selected_rate": float(valid["phase_informative"].mean()),
                "top_contrast_feature": str(
                    valid.sort_values(["contrast_rank", "recovery_rank", "feature"]).iloc[0]["feature"]
                ),
                "top_recovery_feature": str(
                    valid.sort_values(["recovery_rank", "contrast_rank", "feature"]).iloc[0]["feature"]
                ),
                "top_joint_feature": str(
                    valid.sort_values(["joint_rank", "contrast_rank", "feature"]).iloc[0]["feature"]
                ),
            }
        )

    return annotated, summary_rows


def run_event_experiment(config: PhaseAnalysisConfig) -> Dict[str, object]:
    event_data = load_event_data(config.event)
    segments = segment_event(event_data.frame, config.event)
    metric_rows = compute_phase_metrics(segments, config.metric)

    result_rows: List[Dict[str, object]] = []
    for row in metric_rows:
        enriched = add_phase_coefficients(row, config.metric.epsilon)
        enriched["event_id"] = config.event.event_id
        enriched["anomaly_type"] = config.event.anomaly_type
        enriched["label_value"] = int(config.event.label_value)
        result_rows.append(enriched)

    table = pd.DataFrame(result_rows)
    table, selection_rows = _annotate_event_rankings(table)
    result_rows = table.to_dict(orient="records")
    event_dir = config.output.event_dir(config.event.event_id)

    exports: Dict[str, object] = {"event_dir": str(event_dir)}
    if config.output.export_csv:
        summary_rows = []
        for metric_name, subset in table.groupby("metric"):
            summary_rows.append(
                {
                    "metric": metric_name,
                    "contrast_mean": float(subset["contrast_coefficient"].mean()),
                    "contrast_median": float(subset["contrast_coefficient"].median()),
                    "contrast_std": float(subset["contrast_coefficient"].std(ddof=1) if len(subset) > 1 else 0.0),
                    "recovery_mean": float(subset["recovery_coefficient"].mean()),
                    "recovery_median": float(subset["recovery_coefficient"].median()),
                    "recovery_std": float(subset["recovery_coefficient"].std(ddof=1) if len(subset) > 1 else 0.0),
                }
            )
        exports["csv"] = {
            "phase_coefficients": write_csv(event_dir / "phase_coefficients.csv", result_rows),
            "summary_by_metric": write_csv(event_dir / "phase_coefficients_summary_by_metric.csv", summary_rows),
            "selection_summary_by_metric": write_csv(
                event_dir / "phase_coefficients_selection_summary_by_metric.csv",
                selection_rows,
            ),
            "ranked_by_metric": write_csv(
                event_dir / "phase_coefficients_ranked_by_metric.csv",
                table.sort_values(["metric", "joint_rank", "contrast_rank", "feature"], na_position="last").to_dict(
                    orient="records"
                ),
            ),
        }

    if config.output.export_json:
        exports["json"] = write_json(
            event_dir / "phase_coefficients.json",
            {
                "event": {
                    "event_id": config.event.event_id,
                    "anomaly_type": config.event.anomaly_type,
                    "label_value": int(config.event.label_value),
                    "input_path": config.event.input_path,
                    "segments": {
                        name: {
                            "start_index": int(segment.start_index),
                            "end_index": int(segment.end_index),
                            "n_rows": int(len(segment.frame)),
                        }
                        for name, segment in segments.items()
                    },
                },
                "rows": result_rows,
                "selection_summary_by_metric": selection_rows,
            },
        )

    if config.output.export_plots:
        plots: Dict[str, str] = {}
        for metric_name in table["metric"].drop_duplicates().tolist():
            plots[f"{metric_name}_top_contrast"] = plot_top_features(
                table,
                metric_name=metric_name,
                coefficient_name="contrast_coefficient",
                title=f"{config.event.event_id} - {metric_name} contrast",
                output_path=event_dir / f"{metric_name}_top_contrast.png",
            )
            plots[f"{metric_name}_top_recovery"] = plot_top_features(
                table,
                metric_name=metric_name,
                coefficient_name="recovery_coefficient",
                title=f"{config.event.event_id} - {metric_name} recovery",
                output_path=event_dir / f"{metric_name}_top_recovery.png",
            )
            plots[f"{metric_name}_scatter"] = plot_scatter(
                table,
                metric_name=metric_name,
                output_path=event_dir / f"{metric_name}_contrast_vs_recovery.png",
            )
        exports["plots"] = plots

    return {
        "event_id": config.event.event_id,
        "anomaly_type": config.event.anomaly_type,
        "rows": result_rows,
        "exports": exports,
    }
