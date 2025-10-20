from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DateLike = Union[int, float, np.datetime64, pd.Timestamp]


def plot_cusum_top_k_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    save_path: str,
    title: str = "Top features by CUSUM drift score",
    xlabel: str = "CUSUM max (standardized units)",
    dpi: int = 160,
) -> Optional[str]:
    if not labels:
        return None
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    vals = [float(v) for v in values]
    y = np.arange(len(labels))
    plt.figure(figsize=(8, 0.35 * len(labels) + 1.5))
    plt.barh(y, vals)
    plt.yticks(y, list(labels))
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()
    return str(p)


def _to_ts(x: Optional[DateLike]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (int, float)):
        return pd.to_datetime(x, unit="s", utc=True)
    return pd.to_datetime(x)


def plot_cusum_curves(
    t, s_plus, s_minus,
    *,
    side: Optional[str] = None,          # "up" | "down" | None
    a0: Optional[DateLike] = None,       # anomaly start
    a1: Optional[DateLike] = None,       # anomaly end
    threshold: Optional[float] = None,   # horizontal h
    title: str,
    save_path: str,
    dpi: int = 160,
) -> Optional[str]:
    if t is None or len(t) == 0:
        return None
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9.2, 4.8))
    lw_sel, lw_alt = 2.2, 1.2
    if side == "up":
        plt.plot(t, s_plus,  label="S+ (up)",   linewidth=lw_sel)
        plt.plot(t, s_minus, label="S- (down)", linewidth=lw_alt, alpha=0.9)
    elif side == "down":
        plt.plot(t, s_plus,  label="S+ (up)",   linewidth=lw_alt, alpha=0.9)
        plt.plot(t, s_minus, label="S- (down)", linewidth=lw_sel)
    else:
        plt.plot(t, s_plus,  label="S+ (up)",   linewidth=1.6)
        plt.plot(t, s_minus, label="S- (down)", linewidth=1.6)

    A0 = _to_ts(a0)
    A1 = _to_ts(a1)
    if A0 is not None and A1 is not None and A0 < A1:
        plt.axvline(A0, linestyle="--", linewidth=1)
        plt.axvline(A1, linestyle="--", linewidth=1)

    if threshold is not None:
        plt.axhline(float(threshold), linestyle="--", linewidth=1)

    plt.title(title)
    plt.ylabel("CUSUM (standardized)")
    plt.xlabel("time (UTC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()
    return str(p)
