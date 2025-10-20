from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_delta_bar(
    features: Sequence[str],
    deltas: Sequence[float],
    *,
    title: str = "Top Δ",
    xlabel: str = "Δ",
    ylabel: str = "",
    orientation: str = "h",  # "h" (horizontal) or "v" (vertical)
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    n = len(features)
    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.45 * max(1, n))))
    y = np.arange(n)

    if orientation == "v":
        ax.bar(np.arange(n), deltas)
        ax.set_xticks(np.arange(n), list(features), rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    else:
        ax.barh(y, deltas)
        ax.set_yticks(y, list(features))
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(True, axis=("x" if orientation != "v" else "y"), linestyle=":", alpha=0.5)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    return fig, ax


def plot_triplet_bars(
    features: Sequence[str],
    pre: Sequence[float],
    during: Sequence[float],
    post: Sequence[float],
    *,
    title: str = "Pre / During / Post",
    xlabel: str = "score",
    ylabel: str = "",
    orientation: str = "h",  # "h" or "v"
    figsize: Tuple[float, float] = (9, 5),
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    n = len(features)
    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.55 * max(1, n))))
    idx = np.arange(n)
    h = 0.25

    if orientation == "v":
        ax.bar(idx - h, pre,    width=h, label="pre")
        ax.bar(idx,      during, width=h, label="during")
        ax.bar(idx + h,  post,   width=h, label="post")
        ax.set_xticks(idx, list(features), rotation=45, ha="right")
        ax.set_ylabel(ylabel or xlabel)
        ax.set_xlabel("")
    else:
        ax.barh(idx - h, pre,    height=h, label="pre")
        ax.barh(idx,      during, height=h, label="during")
        ax.barh(idx + h,  post,   height=h, label="post")
        ax.set_yticks(idx, list(features))
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.legend(frameon=False, ncols=3, loc="lower right")
    ax.grid(True, axis=("x" if orientation != "v" else "y"), linestyle=":", alpha=0.5)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    return fig, ax

def top_k_barh(labels: Sequence[str], values: Sequence[float],
               xlabel: str, title: str, outpath: Path,
               dpi: int = 160) -> Optional[str]:
    labels = list(labels); values = list(values)
    if not labels:
        return None
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 0.35 * len(labels) + 1.5))
    y = np.arange(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    return str(outpath)
