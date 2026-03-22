from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .export_results import write_csv
from .visualization import (
    plot_batch_metric_boxplots,
    plot_class_metric_heatmap,
    plot_feature_class_heatmap,
)


def build_batch_outputs(batch_rows: List[Dict[str, object]], batch_dir: Path) -> Dict[str, object]:
    if not batch_rows:
        return {}

    table = pd.DataFrame(batch_rows)
    table = table[table["metric_undefined"].astype(str).str.lower() != "true"].copy()
    exports: Dict[str, object] = {}

    class_metric_rows: List[Dict[str, object]] = []
    grouped = table.groupby(["anomaly_type", "metric"], as_index=False)
    for (anomaly_type, metric), subset in grouped:
        class_metric_rows.append(
            {
                "anomaly_type": anomaly_type,
                "metric": metric,
                "n_rows": int(len(subset)),
                "n_events": int(subset["event_id"].nunique()),
                "contrast_mean": float(subset["contrast_coefficient"].mean()),
                "contrast_median": float(subset["contrast_coefficient"].median()),
                "contrast_std": float(subset["contrast_coefficient"].std(ddof=1) if len(subset) > 1 else 0.0),
                "recovery_mean": float(subset["recovery_coefficient"].mean()),
                "recovery_median": float(subset["recovery_coefficient"].median()),
                "recovery_std": float(subset["recovery_coefficient"].std(ddof=1) if len(subset) > 1 else 0.0),
            }
        )

    class_metric_path = batch_dir / "class_metric_summary.csv"
    exports["class_metric_summary"] = write_csv(class_metric_path, class_metric_rows)

    selection_class_metric_rows: List[Dict[str, object]] = []
    for (anomaly_type, metric), subset in grouped:
        selected = subset["phase_informative"].astype(str).str.lower() == "true"
        joint_scores = pd.to_numeric(subset["joint_score"], errors="coerce")
        selection_class_metric_rows.append(
            {
                "anomaly_type": anomaly_type,
                "metric": metric,
                "n_rows": int(len(subset)),
                "n_events": int(subset["event_id"].nunique()),
                "selected_count": int(selected.sum()),
                "selected_rate": float(selected.mean()),
                "joint_score_mean": float(joint_scores.mean()),
                "joint_score_median": float(joint_scores.median()),
            }
        )

    selection_class_metric_path = batch_dir / "class_metric_selection_summary.csv"
    exports["class_metric_selection_summary"] = write_csv(
        selection_class_metric_path,
        selection_class_metric_rows,
    )

    feature_class_rows: List[Dict[str, object]] = []
    feature_grouped = table.groupby(["anomaly_type", "metric", "feature"], as_index=False)
    for (anomaly_type, metric, feature), subset in feature_grouped:
        feature_class_rows.append(
            {
                "anomaly_type": anomaly_type,
                "metric": metric,
                "feature": feature,
                "n_events": int(subset["event_id"].nunique()),
                "contrast_mean": float(subset["contrast_coefficient"].mean()),
                "contrast_median": float(subset["contrast_coefficient"].median()),
                "recovery_mean": float(subset["recovery_coefficient"].mean()),
                "recovery_median": float(subset["recovery_coefficient"].median()),
            }
        )

    feature_class_path = batch_dir / "feature_class_summary.csv"
    exports["feature_class_summary"] = write_csv(feature_class_path, feature_class_rows)

    feature_selection_rows: List[Dict[str, object]] = []
    for (anomaly_type, metric, feature), subset in feature_grouped:
        selected = subset["phase_informative"].astype(str).str.lower() == "true"
        joint_scores = pd.to_numeric(subset["joint_score"], errors="coerce")
        feature_selection_rows.append(
            {
                "anomaly_type": anomaly_type,
                "metric": metric,
                "feature": feature,
                "n_events": int(subset["event_id"].nunique()),
                "selected_events": int(selected.sum()),
                "selected_event_rate": float(selected.mean()),
                "joint_score_mean": float(joint_scores.mean()),
                "joint_score_median": float(joint_scores.median()),
            }
        )

    feature_selection_path = batch_dir / "feature_class_selection_summary.csv"
    exports["feature_class_selection_summary"] = write_csv(
        feature_selection_path,
        feature_selection_rows,
    )

    top_rows: List[Dict[str, object]] = []
    for anomaly_type in sorted(table["anomaly_type"].drop_duplicates()):
        for metric in sorted(table["metric"].drop_duplicates()):
            subset = pd.DataFrame(
                [
                    row for row in feature_class_rows
                    if row["anomaly_type"] == anomaly_type and row["metric"] == metric
                ]
            )
            if subset.empty:
                continue
            contrast_top = subset.sort_values("contrast_mean", ascending=False).head(10)
            recovery_top = subset.sort_values("recovery_mean", ascending=False).head(10)
            for rank, (_, row) in enumerate(contrast_top.iterrows(), start=1):
                top_rows.append(
                    {
                        "kind": "contrast",
                        "anomaly_type": anomaly_type,
                        "metric": metric,
                        "rank": rank,
                        "feature": row["feature"],
                        "value": float(row["contrast_mean"]),
                    }
                )
            for rank, (_, row) in enumerate(recovery_top.iterrows(), start=1):
                top_rows.append(
                    {
                        "kind": "recovery",
                        "anomaly_type": anomaly_type,
                        "metric": metric,
                        "rank": rank,
                        "feature": row["feature"],
                        "value": float(row["recovery_mean"]),
                    }
                )

    top_path = batch_dir / "top_features_by_class_metric.csv"
    exports["top_features_by_class_metric"] = write_csv(top_path, top_rows)

    plots: Dict[str, str] = {}
    plots["contrast_boxplot"] = plot_batch_metric_boxplots(
        table,
        coefficient_name="contrast_coefficient",
        title="Contrast Coefficient by Metric and Anomaly Type",
        output_path=batch_dir / "contrast_by_metric_and_class.png",
    )
    plots["recovery_boxplot"] = plot_batch_metric_boxplots(
        table,
        coefficient_name="recovery_coefficient",
        title="Recovery Coefficient by Metric and Anomaly Type",
        output_path=batch_dir / "recovery_by_metric_and_class.png",
    )
    plots["class_metric_contrast_heatmap"] = plot_class_metric_heatmap(
        pd.DataFrame(class_metric_rows),
        value_column="contrast_mean",
        title="Mean Contrast by Anomaly Type and Metric",
        output_path=batch_dir / "class_metric_contrast_heatmap.png",
    )
    plots["class_metric_recovery_heatmap"] = plot_class_metric_heatmap(
        pd.DataFrame(class_metric_rows),
        value_column="recovery_mean",
        title="Mean Recovery by Anomaly Type and Metric",
        output_path=batch_dir / "class_metric_recovery_heatmap.png",
    )

    preferred_metrics = ["mean", "std", "cv", "entropy", "ks", "auc", "cusum"]
    for metric in preferred_metrics:
        subset = pd.DataFrame([row for row in feature_class_rows if row["metric"] == metric])
        if subset.empty:
            continue
        plots[f"{metric}_feature_contrast_heatmap"] = plot_feature_class_heatmap(
            subset,
            value_column="contrast_mean",
            title=f"{metric} contrast by feature and anomaly type",
            output_path=batch_dir / f"{metric}_feature_contrast_heatmap.png",
            top_n=15,
        )
        plots[f"{metric}_feature_recovery_heatmap"] = plot_feature_class_heatmap(
            subset,
            value_column="recovery_mean",
            title=f"{metric} recovery by feature and anomaly type",
            output_path=batch_dir / f"{metric}_feature_recovery_heatmap.png",
            top_n=15,
        )

    exports["plots"] = plots
    return exports
