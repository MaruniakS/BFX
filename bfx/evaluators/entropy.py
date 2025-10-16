from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureEvaluator


class EntropyEvaluator(FeatureEvaluator):
    """
    Shannon entropy per feature, with quantile binning.
    - score = normalized entropy in [0,1] (entropy / ln(k_effective))
    - details.entropy reported in nats
    """

    name = "entropy"

    def default_params(self) -> Dict[str, Any]:
        return {
            "nbins": 20,          # target number of bins
            "binning": "quantile" # 'quantile' | 'uniform'
        }

    def _bin_series(self, s: pd.Series, nbins: int, mode: str) -> np.ndarray:
        x = s.dropna().to_numpy()
        if x.size == 0:
            return np.array([], dtype=int)

        if mode == "quantile":
            # unique quantiles to avoid duplicate edges for constant/low-var series
            qs = np.linspace(0, 1, nbins + 1)
            edges = np.unique(np.quantile(x, qs, method="linear"))
            if edges.size < 2:
                # constant — everything in one bin
                return np.zeros_like(x, dtype=int)
            bins = np.digitize(x, edges[1:-1], right=False)
            return bins.astype(int)

        elif mode == "uniform":
            lo, hi = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return np.zeros_like(x, dtype=int)
            edges = np.linspace(lo, hi, nbins + 1)
            bins = np.digitize(x, edges[1:-1], right=False)
            return bins.astype(int)

        else:
            raise ValueError(f"Unknown binning mode: {mode}")

    def _entropy_nats(self, bins: np.ndarray) -> float:
        if bins.size == 0:
            return 0.0
        counts = np.bincount(bins)
        p = counts[counts > 0].astype(float)
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    def _evaluate_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        nbins = int(self.params["nbins"])
        mode = str(self.params["binning"])

        results: List[Dict[str, Any]] = []
        entropies: List[float] = []

        for col in df.columns:
            bins = self._bin_series(df[col], nbins, mode)
            H = self._entropy_nats(bins)

            # effective number of non-empty bins for normalization
            k_eff = max(1, np.count_nonzero(np.bincount(bins)))
            H_max = math.log(k_eff) if k_eff > 0 else 1.0
            score = 0.0 if H_max == 0 else float(H / H_max)

            results.append({
                "feature": col,
                "score": score,
                "details": {
                    "entropy": H,
                    "nbins": nbins,
                    "binning": mode,
                    "effective_bins": int(k_eff),
                    "n": int(df[col].dropna().shape[0]),
                }
            })
            entropies.append(H)

        return {
            "method": self.name,
            "meta": {"nbins": nbins, "binning": mode},
            "scores": results,
        }
