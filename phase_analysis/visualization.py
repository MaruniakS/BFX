from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_top_features(
    table: pd.DataFrame,
    *,
    metric_name: str,
    coefficient_name: str,
    title: str,
    output_path: Path,
    top_n: int = 12,
) -> str:
    subset = table[table["metric"] == metric_name].copy()
    subset = subset.sort_values(coefficient_name, ascending=False).head(top_n)
    labels = list(subset["feature"])[::-1]
    values = [float(v) for v in subset[coefficient_name]][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
    ax.barh(labels, values, color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel(coefficient_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path)


def plot_scatter(
    table: pd.DataFrame,
    *,
    metric_name: str,
    output_path: Path,
) -> str:
    subset = table[table["metric"] == metric_name].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    selected = None
    if "phase_informative" in subset.columns:
        selected = subset["phase_informative"].astype(str).str.lower() == "true"
        ax.scatter(
            subset.loc[~selected, "contrast_coefficient"].astype(float),
            subset.loc[~selected, "recovery_coefficient"].astype(float),
            alpha=0.65,
            color="#b7791f",
            label="other features",
        )
        ax.scatter(
            subset.loc[selected, "contrast_coefficient"].astype(float),
            subset.loc[selected, "recovery_coefficient"].astype(float),
            alpha=0.9,
            color="#2b6cb0",
            label="selected features",
        )
    else:
        ax.scatter(
            subset["contrast_coefficient"].astype(float),
            subset["recovery_coefficient"].astype(float),
            alpha=0.8,
            color="#b7791f",
        )

    if "contrast_threshold" in subset.columns and subset["contrast_threshold"].notna().any():
        ax.axvline(float(subset["contrast_threshold"].dropna().iloc[0]), color="#2f855a", linestyle="--", linewidth=1)
    if "recovery_threshold" in subset.columns and subset["recovery_threshold"].notna().any():
        ax.axhline(float(subset["recovery_threshold"].dropna().iloc[0]), color="#c53030", linestyle="--", linewidth=1)

    ax.set_xlabel("contrast_coefficient")
    ax.set_ylabel("recovery_coefficient")
    ax.set_title(f"{metric_name} phase coefficients")
    if selected is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path)


def plot_batch_metric_boxplots(
    table: pd.DataFrame,
    *,
    coefficient_name: str,
    title: str,
    output_path: Path,
) -> str:
    metrics = list(table["metric"].drop_duplicates())
    classes = list(table["anomaly_type"].drop_duplicates())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, max(4, 2.6 * len(metrics))), squeeze=False)
    for idx, metric in enumerate(metrics):
        ax = axes[idx][0]
        data = []
        labels = []
        for anomaly_type in classes:
            subset = table[(table["metric"] == metric) & (table["anomaly_type"] == anomaly_type)]
            vals = subset[coefficient_name].astype(float).tolist()
            if vals:
                data.append(vals)
                labels.append(anomaly_type)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(metric)
        ax.set_ylabel(coefficient_name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path)


def plot_class_metric_heatmap(
    table: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
) -> str:
    pivot = table.pivot(index="anomaly_type", columns="metric", values=value_column).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 3 + 0.5 * len(pivot.index)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path)


def plot_feature_class_heatmap(
    table: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> str:
    ranked = (
        table.groupby("feature", as_index=False)[value_column]
        .mean()
        .sort_values(value_column, ascending=False)
        .head(top_n)
    )
    keep = set(ranked["feature"])
    subset = table[table["feature"].isin(keep)].copy()
    pivot = subset.pivot(index="feature", columns="anomaly_type", values=value_column).fillna(0.0)
    pivot = pivot.loc[ranked["feature"]]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(pivot.index) + 1)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path)
