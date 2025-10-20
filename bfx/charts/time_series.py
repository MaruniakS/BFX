from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from bfx.utils import to_datetime_utc
from bfx.core.dataset import Dataset


def _coerce_frame(dataset: Dataset) -> pd.DataFrame:
    """
    Expect dataset.data as a list of dict rows: [{feature: value, ...}, ...]
    No timestamps in the data. We synthesize a UTC datetime index using:
        t[i] = start_time + i * period_minutes * 60
    """
    data = dataset.data
    if not isinstance(data, list):
        raise ValueError("Expected dataset.data to be a list of dict rows.")
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Require start_time and period
    start = dataset.params.get("start_time")
    period_min = dataset.params.get("period")
    if start is None or period_min is None:
        raise ValueError("Missing 'start_time' or 'period' in Dataset.params.")

    start = int(start)
    step = int(period_min) * 60  # seconds per step
    idx = pd.to_datetime([start + i * step for i in range(len(df))], unit="s", utc=True)

    # Keep only numeric columns (drop columns that are entirely non-numeric)
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(axis=1, how="all", inplace=True)

    df.index = idx
    df.sort_index(inplace=True)
    return df


def _clip_window(df: pd.DataFrame, t0: Optional[int], t1: Optional[int]) -> pd.DataFrame:
    if t0 is None and t1 is None:
        return df
    left = to_datetime_utc(int(t0)) if t0 is not None else df.index.min()
    right = to_datetime_utc(int(t1)) if t1 is not None else df.index.max()
    return df.loc[(df.index >= left) & (df.index <= right)]


def _minmax(df: pd.DataFrame) -> pd.DataFrame:
    minv = df.min(axis=0)
    maxv = df.max(axis=0)
    denom = (maxv - minv).replace(0, np.nan)
    return ((df - minv) / denom).fillna(0.0)


def plot_feature_timeseries(
    dataset: Dataset,
    *,
    features: Optional[List[str]] = None,
    scaling: str = "raw",                      # 'raw' | 'minmax'
    tick_every_min: int = 10,                  # minutes between major ticks
    time_fmt: str = "%H.%M",                   # tick label format
    figsize: Tuple[float, float] = (6, 6),     # square by default
    linewidth: float = 1.2,
    legend: bool = True,
    # Y-axis controls are OPTIONAL; if None -> auto-scale
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    y_ticks: Optional[Sequence[float]] = None,
    anomaly_bg_color: str = "0.92",
    show_anomaly_label: bool = True,
    grid: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """
    One chart:
      x: time (forced to Dataset start_time..end_time if both provided)
      y: feature values (raw or min–max)
      anomaly window shaded if provided

    Returns:
        (fig, ax)
    """
    if dataset.data is None:
        raise ValueError("Dataset has no data loaded. Set 'input' and call setParams().")

    df = _coerce_frame(dataset)
    if df.empty:
        raise ValueError("Dataset is empty after coercion.")

    # clip to investigation window
    t0 = dataset.params.get("start_time")
    t1 = dataset.params.get("end_time")
    df = _clip_window(df, t0, t1)

    # feature selection
    cols = list(df.columns) if features is None else [c for c in features if c in df.columns]
    if not cols:
        raise ValueError("None of the requested features were found in data.")
    df = df[cols]

    # scaling
    if scaling == "minmax":
        df = _minmax(df)
    elif scaling not in (None, "raw"):
        raise ValueError(f"Unknown scaling: {scaling}")

    # If minmax scaling and Y not explicitly set -> apply BML-like defaults
    if scaling == "minmax":
        if y_min is None:
            y_min = -0.10
        if y_max is None:
            y_max = 1.75
        if y_ticks is None:
            y_ticks = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75)

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    for c in df.columns:
        ax.plot(df.index, df[c].values, label=c, linewidth=linewidth)

    # axis labels
    ax.set_xlabel("UTC time")
    ax.set_ylabel("min–max" if scaling == "minmax" else "value")

    # X-axis window & ticks: only force if both start & end are provided
    if t0 is not None and t1 is not None:
        start_dt = to_datetime_utc(int(t0))
        end_dt = to_datetime_utc(int(t1))
        ax.set_xlim(start_dt, end_dt)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, tick_every_min)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))

    # Y-axis limits/ticks: only apply if provided (None => auto)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    if y_ticks is not None:
        ax.set_yticks(list(y_ticks))

    # grid and spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    if grid:
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

       # anomaly shading + optional label
    a0 = dataset.params.get("anomaly_start_time")
    a1 = dataset.params.get("anomaly_end_time")
    if a0 is not None and a1 is not None:
        A0 = to_datetime_utc(int(a0))
        A1 = to_datetime_utc(int(a1))
        ax.axvspan(A0, A1, color=anomaly_bg_color, zorder=0)
        if show_anomaly_label:
            mid = A0 + (A1 - A0) / 2
            ax.text(
                mid, -0.025, "Anomaly",
                ha="center", va="top",
                fontsize=8, color="0.1", clip_on=False
            )

    if legend:
        ncol = 2 if len(df.columns) > 2 else 1
        ax.legend(loc="upper right", ncol=ncol, fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig, ax
