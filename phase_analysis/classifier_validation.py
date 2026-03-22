from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "combined_events" / "anomalies_metadata.json"
CONSENSUS_PATH = ROOT / "out" / "article_tables" / "class_level_consensus_top_bottom5_filtered.csv"
OUTDIR = ROOT / "out" / "article_tables"


def _load_metadata() -> List[Dict[str, object]]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _slugify(name: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _feature_sets_from_consensus() -> Dict[str, Dict[str, List[str]]]:
    with CONSENSUS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    out: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"top": [], "bottom": []})
    for row in rows:
        out[row["anomaly_type"]][row["group"]].append(row["feature"])
    return out


def _all_interpretable_features(example_row: Dict[str, object]) -> List[str]:
    features = []
    for key, value in example_row.items():
        if key in {"label", "timestamp"}:
            continue
        if key.startswith("editdist_"):
            continue
        if isinstance(value, (int, float)):
            features.append(key)
    return sorted(features)


def _load_event_rows(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _xy_from_rows(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row.get(feat, 0.0)) for feat in features] for row in rows], dtype=float)
    y = np.asarray([1 if int(row.get("label", -1)) != -1 else 0 for row in rows], dtype=int)
    return x, y


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def _fit_gaussian_nb(train_x: np.ndarray, train_y: np.ndarray) -> Dict[str, np.ndarray]:
    classes = np.array([0, 1], dtype=int)
    means = []
    variances = []
    priors = []
    for cls in classes:
        subset = train_x[train_y == cls]
        means.append(subset.mean(axis=0))
        var = subset.var(axis=0)
        var[var < 1e-9] = 1e-9
        variances.append(var)
        priors.append(float(len(subset)) / float(len(train_x)))
    return {
        "classes": classes,
        "means": np.asarray(means),
        "variances": np.asarray(variances),
        "log_priors": np.log(np.asarray(priors)),
    }


def _predict_proba(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    means = model["means"]
    variances = model["variances"]
    log_priors = model["log_priors"]

    log_probs = []
    for idx in range(len(model["classes"])):
        mean = means[idx]
        var = variances[idx]
        log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var, axis=1)
        log_probs.append(log_priors[idx] + log_likelihood)
    scores = np.vstack(log_probs).T
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum = ranks[pos].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def run_validation() -> List[Dict[str, object]]:
    metadata = _load_metadata()
    feature_sets = _feature_sets_from_consensus()

    # Build event lists by class and cache rows.
    by_class: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    event_rows: Dict[str, List[Dict[str, object]]] = {}
    for item in metadata:
        event_id = _slugify(str(item["event"]))
        rows = _load_event_rows(ROOT / str(item["output_file"]))
        event_rows[event_id] = rows
        by_class[str(item["str_class"])].append({"event_id": event_id, "rows": rows})

    example_row = next(iter(event_rows.values()))[0]
    all_features = _all_interpretable_features(example_row)

    results: List[Dict[str, object]] = []
    for anomaly_type, events in sorted(by_class.items()):
        top_features = feature_sets[anomaly_type]["top"]
        bottom_features = feature_sets[anomaly_type]["bottom"]
        configs = [
            ("all_interpretable", all_features),
            ("top_ranked", top_features),
            ("bottom_ranked", bottom_features),
        ]

        for holdout in events:
            train_rows: List[Dict[str, object]] = []
            for item in events:
                if item["event_id"] != holdout["event_id"]:
                    train_rows.extend(item["rows"])
            test_rows = list(holdout["rows"])

            for feature_set_name, features in configs:
                train_x, train_y = _xy_from_rows(train_rows, features)
                test_x, test_y = _xy_from_rows(test_rows, features)
                train_x, test_x = _standardize(train_x, test_x)
                model = _fit_gaussian_nb(train_x, train_y)
                y_score = _predict_proba(model, test_x)
                y_pred = (y_score >= 0.5).astype(int)

                results.append(
                    {
                        "anomaly_type": anomaly_type,
                        "holdout_event": holdout["event_id"],
                        "feature_set": feature_set_name,
                        "n_features": int(len(features)),
                        "roc_auc": _roc_auc_score(test_y, y_score),
                        "f1": _f1_score(test_y, y_pred),
                        "n_test_rows": int(len(test_y)),
                        "n_positive": int(test_y.sum()),
                    }
                )
    return results


def write_outputs(results: List[Dict[str, object]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTDIR / "classifier_validation_leave_one_event_out.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_rows: List[Dict[str, object]] = []
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[(str(row["anomaly_type"]), str(row["feature_set"]))].append(row)
    for (anomaly_type, feature_set), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "anomaly_type": anomaly_type,
                "feature_set": feature_set,
                "n_folds": int(len(rows)),
                "n_features": int(rows[0]["n_features"]),
                "roc_auc_mean": float(np.nanmean([float(r["roc_auc"]) for r in rows])),
                "roc_auc_std": float(np.nanstd([float(r["roc_auc"]) for r in rows])),
                "f1_mean": float(np.nanmean([float(r["f1"]) for r in rows])),
                "f1_std": float(np.nanstd([float(r["f1"]) for r in rows])),
            }
        )

    summary_path = OUTDIR / "classifier_validation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    write_outputs(run_validation())
