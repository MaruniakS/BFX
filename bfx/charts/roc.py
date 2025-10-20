from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _roc_points(pre: np.ndarray, dur: np.ndarray, sign: int = +1) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROC points (FPR, TPR) treating 'during' as positives."""
    pre = pre[np.isfinite(pre)]
    dur = dur[np.isfinite(dur)]
    m, n = pre.size, dur.size
    if m == 0 or n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    s_pre = sign * pre
    s_dur = sign * dur
    thresh = np.unique(np.concatenate([s_pre, s_dur]))
    thresh = np.concatenate(([np.inf], thresh[::-1], [-np.inf]))

    TPR, FPR = [], []
    for t in thresh:
        tp = float(np.sum(s_dur >= t)); fp = float(np.sum(s_pre >= t))
        TPR.append(tp / n if n else 0.0)
        FPR.append(fp / m if m else 0.0)
    return np.array(FPR), np.array(TPR)


def plot_roc_curve(pre: np.ndarray, dur: np.ndarray, *,
                   sign: int, label: str, title: str, outpath: Path,
                   dpi: int = 160) -> Optional[str]:
    """Compute ROC and save the plot to outpath."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    FPR, TPR = _roc_points(pre, dur, sign=sign)
    plt.figure(figsize=(6.4, 5.4))
    plt.plot(FPR, TPR, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    return str(outpath)
