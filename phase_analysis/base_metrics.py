from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import MetricConfig
from .segmentation import Segment


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _safe_std(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _mean(x: np.ndarray) -> Tuple[float, bool]:
    x = _finite(x)
    if x.size == 0:
        return 0.0, True
    return float(np.mean(x)), False


def _std(x: np.ndarray) -> Tuple[float, bool]:
    x = _finite(x)
    if x.size <= 1:
        return 0.0, True
    return _safe_std(x), False


def _median(x: np.ndarray) -> Tuple[float, bool]:
    x = _finite(x)
    if x.size == 0:
        return 0.0, True
    return float(np.median(x)), False


def _mad(x: np.ndarray) -> Tuple[float, bool]:
    x = _finite(x)
    if x.size == 0:
        return 0.0, True
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad), False


def _cv(x: np.ndarray, eps: float) -> Tuple[float, bool]:
    x = _finite(x)
    if x.size <= 1:
        return 0.0, True
    mu = float(np.mean(x))
    denom = abs(mu) + eps
    if denom <= eps:
        return 0.0, True
    return float(_safe_std(x) / denom), False


def _entropy(x: np.ndarray, bins: int, eps: float) -> Tuple[float, bool]:
    x = _finite(x)
    if x.size == 0:
        return 0.0, True
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, True
    counts, _ = np.histogram(x, bins=max(2, int(bins)))
    total = int(counts.sum())
    if total == 0:
        return 0.0, True
    probs = counts[counts > 0] / float(total)
    return float(-np.sum(probs * np.log2(probs + eps))), False


def _ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, bool]:
    a = _finite(a)
    b = _finite(b)
    if a.size == 0 or b.size == 0:
        return 0.0, True
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.unique(np.concatenate([a, b])))
    if grid.size == 0:
        return 0.0, True
    fa = np.searchsorted(a, grid, side="right") / float(a.size)
    fb = np.searchsorted(b, grid, side="right") / float(b.size)
    return float(np.max(np.abs(fa - fb))), False


def _auc_delta(a: np.ndarray, b: np.ndarray, ties: str) -> Tuple[float, bool]:
    a = _finite(a)
    b = _finite(b)
    if a.size == 0 or b.size == 0:
        return 0.0, True
    values = np.concatenate([a, b])
    ranks = pd.Series(values).rank(method=ties).to_numpy(dtype=float)
    rank_sum_b = float(np.sum(ranks[a.size:]))
    u_value = rank_sum_b - b.size * (b.size + 1) / 2.0
    auc = u_value / float(a.size * b.size)
    delta = 2.0 * auc - 1.0
    return float(abs(delta)), False


def _cusum_peak(segment_values: np.ndarray, baseline_values: np.ndarray, k: float, eps: float) -> Tuple[float, bool]:
    x = _finite(segment_values)
    b = _finite(baseline_values)
    if x.size == 0 or b.size <= 1:
        return 0.0, True
    mu0 = float(np.mean(b))
    sd0 = _safe_std(b)
    if sd0 <= eps:
        return 0.0, True
    z = (x - mu0) / sd0
    pos = np.zeros_like(z, dtype=float)
    neg = np.zeros_like(z, dtype=float)
    for idx in range(1, z.size):
        pos[idx] = max(0.0, pos[idx - 1] + z[idx] - k)
        neg[idx] = max(0.0, neg[idx - 1] - z[idx] - k)
    return float(max(np.max(pos), np.max(neg))), False


def compute_phase_metrics(
    segments: Dict[str, Segment],
    metric_config: MetricConfig,
) -> List[Dict[str, object]]:
    pre = segments["pre"].frame
    during = segments["during"].frame
    post = segments["post"].frame
    feature_names = list(pre.columns)

    rows: List[Dict[str, object]] = []
    for feature in feature_names:
        x_pre = pre[feature].to_numpy(dtype=float, copy=False)
        x_dur = during[feature].to_numpy(dtype=float, copy=False)
        x_post = post[feature].to_numpy(dtype=float, copy=False)

        other_pre = np.concatenate([_finite(x_dur), _finite(x_post)])
        other_dur = np.concatenate([_finite(x_pre), _finite(x_post)])
        other_post = np.concatenate([_finite(x_pre), _finite(x_dur)])

        for metric_name in metric_config.metrics:
            notes = ""
            undefined = False

            if metric_name == "mean":
                m_pre, u1 = _mean(x_pre)
                m_dur, u2 = _mean(x_dur)
                m_post, u3 = _mean(x_post)
                undefined = u1 or u2 or u3
            elif metric_name == "std":
                m_pre, u1 = _std(x_pre)
                m_dur, u2 = _std(x_dur)
                m_post, u3 = _std(x_post)
                undefined = u1 or u2 or u3
            elif metric_name == "median":
                m_pre, u1 = _median(x_pre)
                m_dur, u2 = _median(x_dur)
                m_post, u3 = _median(x_post)
                undefined = u1 or u2 or u3
            elif metric_name == "mad":
                m_pre, u1 = _mad(x_pre)
                m_dur, u2 = _mad(x_dur)
                m_post, u3 = _mad(x_post)
                undefined = u1 or u2 or u3
            elif metric_name == "cv":
                m_pre, u1 = _cv(x_pre, metric_config.epsilon)
                m_dur, u2 = _cv(x_dur, metric_config.epsilon)
                m_post, u3 = _cv(x_post, metric_config.epsilon)
                undefined = u1 or u2 or u3
            elif metric_name == "entropy":
                m_pre, u1 = _entropy(x_pre, metric_config.entropy_bins, metric_config.epsilon)
                m_dur, u2 = _entropy(x_dur, metric_config.entropy_bins, metric_config.epsilon)
                m_post, u3 = _entropy(x_post, metric_config.entropy_bins, metric_config.epsilon)
                undefined = u1 or u2 or u3
            elif metric_name == "ks":
                m_pre, u1 = _ks(x_pre, other_pre)
                m_dur, u2 = _ks(x_dur, other_dur)
                m_post, u3 = _ks(x_post, other_post)
                undefined = u1 or u2 or u3
                notes = "KS is defined here as each phase versus the pooled complementary phases."
            elif metric_name == "auc":
                m_pre, u1 = _auc_delta(x_pre, other_pre, metric_config.auc_ties)
                m_dur, u2 = _auc_delta(x_dur, other_dur, metric_config.auc_ties)
                m_post, u3 = _auc_delta(x_post, other_post, metric_config.auc_ties)
                undefined = u1 or u2 or u3
                notes = "AUC is defined here as |2*AUC-1| for each phase versus the pooled complementary phases."
            elif metric_name == "cusum":
                m_pre, u1 = _cusum_peak(x_pre, x_pre, metric_config.cusum_k, metric_config.epsilon)
                m_dur, u2 = _cusum_peak(x_dur, x_pre, metric_config.cusum_k, metric_config.epsilon)
                m_post, u3 = _cusum_peak(x_post, x_pre, metric_config.cusum_k, metric_config.epsilon)
                undefined = u1 or u2 or u3
                notes = "CUSUM uses the pre phase as the baseline window for all three segment summaries."
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")

            rows.append(
                {
                    "feature": feature,
                    "metric": metric_name,
                    "M_pre": float(m_pre),
                    "M_during": float(m_dur),
                    "M_post": float(m_post),
                    "metric_undefined": bool(undefined),
                    "metric_notes": notes,
                }
            )

    return rows
