from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _ecdf_sorted(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys

def _ecdf_at(xs_sorted: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    if xs_sorted.size == 0:
        return np.zeros_like(xgrid, dtype=float)
    ranks = np.searchsorted(xs_sorted, xgrid, side="right")
    return ranks.astype(float) / float(xs_sorted.size)


def plot_ecdf_overlay(
    a: np.ndarray,
    b: np.ndarray,
    *,
    a_label: str,
    b_label: str,
    x_label: str,
    title: str,
    x_star: Optional[float] = None,
    d_value: Optional[float] = None,   # e.g., KS D
    save_path: str,
    dpi: int = 160,
) -> Optional[str]:
    """Empirical CDF overlay for two samples with optional x* and metric annotation."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None

    xs_a, F_a = _ecdf_sorted(a)
    xs_b, F_b = _ecdf_sorted(b)
    if xs_a.size == 0 or xs_b.size == 0:
        return None

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.8, 5.2))
    plt.step(xs_a, F_a, where="post", label=f"{a_label} (n={a.size})")
    plt.step(xs_b, F_b, where="post", label=f"{b_label} (n={b.size})")

    if isinstance(x_star, (int, float)) and np.isfinite(x_star):
        plt.axvline(float(x_star), linestyle=":", linewidth=1.0, alpha=0.7, label="x*")
    if isinstance(d_value, (int, float)) and np.isfinite(d_value):
        plt.text(0.02, 0.02, f"metric = {float(d_value):.3f}", transform=plt.gca().transAxes,
                 fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()
    return str(p)


def plot_cdf_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    a_label: str,
    b_label: str,
    x_label: str,
    title: str,
    x_star: Optional[float] = None,
    d_value: Optional[float] = None,   # e.g., max |Δ|
    save_path: str,
    dpi: int = 160,
) -> Optional[str]:
    """Plot (F_b - F_a)(x) over union support with optional x* marker."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None

    xs_a = np.sort(a)
    xs_b = np.sort(b)
    grid = np.unique(np.concatenate([xs_a, xs_b]))
    Fa = _ecdf_at(xs_a, grid)
    Fb = _ecdf_at(xs_b, grid)
    diff = Fb - Fa

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.8, 5.0))
    plt.plot(grid, diff, label=f"{b_label} - {a_label}")
    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)

    if isinstance(x_star, (int, float)) and np.isfinite(x_star):
        plt.axvline(float(x_star), linestyle=":", linewidth=1.0, alpha=0.7, label="x*")
    if isinstance(d_value, (int, float)) and np.isfinite(d_value):
        plt.text(0.02, 0.02, f"metric = {float(d_value):.3f}", transform=plt.gca().transAxes,
                 fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Δ CDF")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()
    return str(p)
