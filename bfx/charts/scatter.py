from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

def plot_scatter_xy(
    xvals: Sequence[float],
    yvals: Sequence[float],
    labels: Sequence[str],
    *,
    title: str = "pre vs during",
    xlabel: str = "pre",
    ylabel: str = "during",
    diagonal: bool = True,
    figsize: Tuple[float, float] = (6, 6),
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xvals, yvals, s=36, alpha=0.85)
    for lx, ly, lab in zip(xvals, yvals, labels):
        ax.annotate(lab, (lx, ly), xytext=(3, 3), textcoords="offset points", fontsize=8, alpha=0.75)
    if diagonal:
        lo = float(min(min(xvals), min(yvals)))
        hi = float(max(max(xvals), max(yvals)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    return fig, ax


def plot_scatter_deltas(
    xvals: Sequence[float],
    yvals: Sequence[float],
    labels: Sequence[str],
    *,
    title: str = "ΔEntropy vs ΔCV",
    xlabel: str = "ΔCV (during − pre)",
    ylabel: str = "ΔEntropy (during − pre)",
    figsize: Tuple[float, float] = (6.5, 6.5),
    zero_axes: bool = True,
    diagonal: bool = False,   # usually False for deltas
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xvals, yvals, s=40, alpha=0.9)
    for x, y, lab in zip(xvals, yvals, labels):
        ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8, alpha=0.75)

    if zero_axes:
        ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.5)
    if diagonal:
        lo = float(min(min(xvals), min(yvals)))
        hi = float(max(max(xvals), max(yvals)))
        ax.plot([lo, hi], [lo, hi], linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    return fig, ax
