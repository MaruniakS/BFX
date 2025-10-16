from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bfx.core.dataset import Dataset


def _frame_from_dataset(dataset: Dataset) -> pd.DataFrame:
    """Expect dataset.data as List[dict] of feature values; numeric-only DataFrame."""
    data = dataset.data
    if not isinstance(data, list):
        raise ValueError("Expected dataset.data to be a list[dict].")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(axis=1, how="all", inplace=True)
    return df


# ---- time-slicing helpers (regular sampling assumed) ----
def _slice_indices_by_time(ds: Dataset, t0: int, t1: int) -> Tuple[int, int]:
    n = len(ds.data or [])
    if n == 0:
        return 0, -1
    base = int(ds.params.get("start_time"))
    step = int(ds.params.get("period")) * 60  # seconds
    if step <= 0:
        raise ValueError("Dataset 'period' must be > 0 minutes.")
    import math
    i0 = max(0, math.ceil((int(t0) - base) / step))
    i1 = min(n - 1, math.floor((int(t1) - base) / step))
    if i1 < i0:
        return 0, -1
    return i0, i1


def _dataset_slice(ds: Dataset, t0: int, t1: int) -> Dataset:
    i0, i1 = _slice_indices_by_time(ds, t0, t1)
    out = Dataset()
    out.setParams({k: v for k, v in ds.params.items() if k in out.params})
    out.params["start_time"] = int(t0)
    out.params["end_time"] = int(t1)
    if i1 >= i0 and isinstance(ds.data, list):
        out.data = ds.data[i0 : i1 + 1]
    else:
        out.data = []
    out._input_path = ds._input_path
    return out


def _window_specs(ds: Dataset) -> Dict[str, Tuple[int, int]]:
    st = ds.params.get("start_time")
    et = ds.params.get("end_time")
    a0 = ds.params.get("anomaly_start_time")
    a1 = ds.params.get("anomaly_end_time")
    wins: Dict[str, Tuple[int, int]] = {}
    if st is not None and a0 is not None and int(a0) > int(st):
        wins["pre"] = (int(st), int(a0) - 1)
    if a0 is not None and a1 is not None and int(a1) >= int(a0):
        wins["during"] = (int(a0), int(a1))
    if et is not None and a1 is not None and int(et) > int(a1):
        wins["post"] = (int(a1) + 1, int(et))
    return wins


class FeatureEvaluator:
    """
    Base evaluator.
    - setParams: strict (subclasses extend defaults via super().default_params()).
    - evaluate(dataset, features=None) -> result dict with:
        * method-wide scores on the full window
        * always-attached per-window scores (pre/during/post when available)
        * always-attached deltas: during_minus_pre, post_minus_during, post_minus_pre
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = self.default_params()
        if params:
            self.setParams(params)

    def default_params(self) -> Dict[str, Any]:
        """Subclasses may extend/override their own algo params."""
        return {}

    def setParams(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if k in self.params:
                self.params[k] = v
            else:
                sys.exit("Unrecognized parameter:" + k)

    # ---- public entrypoint ----
    def evaluate(self, dataset: Dataset, features: Optional[List[str]] = None) -> Dict[str, Any]:
        # full-window DF
        df_full = _frame_from_dataset(dataset)
        if df_full.empty:
            raise ValueError("Dataset is empty for evaluation.")

        cols = list(df_full.columns) if features is None else [c for c in df_full.columns if c in features]
        if not cols:
            raise ValueError("None of the requested features were found in data.")
        df_full = df_full[cols]

        # main result on the full window
        result = self._evaluate_df(df_full)

        # always compute pre/during/post if we can
        wins = _window_specs(dataset)
        if wins:
            windows_payload: Dict[str, Any] = {}
            for label, (w0, w1) in wins.items():
                ds_win = _dataset_slice(dataset, w0, w1)
                df_win = _frame_from_dataset(ds_win)
                df_win = df_win[cols] if not df_win.empty else df_win
                if df_win.empty:
                    windows_payload[label] = {"scores": []}
                else:
                    sub = self._evaluate_df(df_win)
                    compact = [{"feature": s["feature"], "score": float(s["score"])} for s in sub.get("scores", [])]
                    windows_payload[label] = {"scores": compact}
            result["windows"] = windows_payload

            # fixed set of deltas
            def _index(win: str) -> Dict[str, float]:
                arr = (windows_payload.get(win) or {}).get("scores") or []
                return {d["feature"]: float(d["score"]) for d in arr}

            deltas_out: Dict[str, List[Dict[str, Any]]] = {}
            for key, (Lw, Rw) in {
                "during_minus_pre":  ("pre", "during"),
                "post_minus_during": ("during", "post"),
                "post_minus_pre":    ("pre", "post"),
            }.items():
                L, R = _index(Lw), _index(Rw)
                common = set(L) & set(R)
                rows = [{"feature": f, "delta": R[f] - L[f], "left": L[f], "right": R[f]} for f in common]
                rows.sort(key=lambda x: x["delta"], reverse=True)
                for i, r in enumerate(rows, start=1):
                    r["rank"] = i
                deltas_out[key] = rows
            result["deltas"] = deltas_out

        return result

    # ---- subclass hook ----
    def _evaluate_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError
