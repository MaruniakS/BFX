from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import FeatureEvaluator


class CVEvaluator(FeatureEvaluator):
    """
    Coefficient of variation per feature.

    Variants:
      - 'classic'     : CV = std / |mean|
      - 'robust_mad'  : CV = (1.4826 * MAD) / |median|

    Scaling across features:
      - min–max to [0,1] for comparability (reported in meta.scaling).
    """

    name = "cv"

    def default_params(self) -> Dict[str, Any]:
        params = super().default_params()
        params.update({
            "ddof": 1,                # only for 'classic'
            "variant": "robust_mad",  # 'classic' | 'robust_mad'
            "epsilon": 0.0,           # stabilizer
        })
        return params

    # ---- helpers ----

    def _classic_cv(self, s: pd.Series, ddof: int, eps: float) -> Tuple[float, bool]:
        x = s.dropna().to_numpy(dtype=float)
        if x.size <= 1:
            return 0.0, True
        mu = float(np.mean(x))
        denom = abs(mu) + float(eps)
        if denom == 0.0:
            return 0.0, True
        sd = float(np.std(x, ddof=ddof))
        return sd / denom, False

    def _robust_mad_cv(self, s: pd.Series, eps: float) -> Tuple[float, bool]:
        x = s.dropna().to_numpy(dtype=float)
        if x.size == 0:
            return 0.0, True
        med = float(np.median(x))
        denom = abs(med) + float(eps)
        if denom == 0.0:
            return 0.0, True
        mad = float(np.median(np.abs(x - med)))
        sigma_hat = 1.4826 * mad
        return sigma_hat / denom, False

    # ---- main hook ----

    def _evaluate_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        ddof = int(self.params["ddof"])
        variant = str(self.params["variant"])
        eps = float(self.params["epsilon"])

        raw_values: List[Tuple[str, float, bool]] = []
        for col in df.columns:
            if variant == "classic":
                val, undefined = self._classic_cv(df[col], ddof, eps)
            elif variant == "robust_mad":
                val, undefined = self._robust_mad_cv(df[col], eps)
            else:
                raise ValueError(f"Unknown CV variant: {variant}")
            raw_values.append((col, val, undefined))

        # min–max scaling across defined CVs
        vals = np.array([v for (_, v, u) in raw_values if not u], dtype=float)
        vmin = float(vals.min()) if vals.size else 0.0
        vmax = float(vals.max()) if vals.size else 1.0
        span = (vmax - vmin) if vmax > vmin else 1.0

        scores: List[Dict[str, Any]] = []
        for col, val, undefined in raw_values:
            score = 0.0 if undefined else (val - vmin) / span
            scores.append({
                "feature": col,
                "score": float(score),
                "details": {
                    "cv": float(val),
                    "undefined": bool(undefined),
                }
            })

        return {
            "method": self.name,
            "meta": {
                "variant": variant,
                "scaling": "minmax",
                "ddof": ddof if variant == "classic" else None,
                "epsilon": eps if eps > 0 else None,
            },
            "scores": scores,
        }
