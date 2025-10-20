from __future__ import annotations
import numpy as np

EPS = 1e-12
TINY_SIGMA = 1e-12

def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else 0.0

def safe_std(x: np.ndarray, ddof: int = 1) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= ddof:
        return 0.0
    return float(np.std(x, ddof=ddof))
