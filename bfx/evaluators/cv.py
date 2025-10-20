from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .base import FeatureEvaluator
from ..common.ts import make_index, numeric_feature_cols
from ..common.numerics import TINY_SIGMA

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

    def __init__(self, variant: str = "robust_mad", ddof: int = 1, epsilon: float = 0.0) -> None:
        self.variant = str(variant)
        self.ddof = int(ddof)
        self.epsilon = float(epsilon)

    def _classic_cv(self, x: np.ndarray) -> Tuple[float, bool]:
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return 0.0, True
        mu = float(np.mean(x))
        denom = abs(mu) + self.epsilon
        if denom <= TINY_SIGMA:
            return 0.0, True
        sd = float(np.std(x, ddof=self.ddof))
        return sd / denom, False

    def _robust_mad_cv(self, x: np.ndarray) -> Tuple[float, bool]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0, True
        med = float(np.median(x))
        denom = abs(med) + self.epsilon
        if denom <= TINY_SIGMA:
            return 0.0, True
        mad = float(np.median(np.abs(x - med)))
        sigma_hat = 1.4826 * mad
        return sigma_hat / denom, False

    def evaluate(self, dataset, features: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        data = dataset.data
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset has no data loaded.")

        period_min = int(dataset.params.get("period") or 1)
        df = make_index(pd.DataFrame(data), dataset.params.get("start_time"), period_min)

        all_feats = list(numeric_feature_cols(df))
        feats = [f for f in (features or all_feats) if f in all_feats]
        if not feats:
            raise ValueError("No numeric features available for CV evaluation.")

        raw_values: List[Tuple[str, float, bool]] = []
        for col in feats:
            x = df[col].to_numpy(dtype=float, copy=False)
            if self.variant == "classic":
                val, undefined = self._classic_cv(x)
            elif self.variant == "robust_mad":
                val, undefined = self._robust_mad_cv(x)
            else:
                raise ValueError(f"Unknown CV variant: {self.variant}")
            raw_values.append((col, float(val), bool(undefined)))

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
                "details": {"cv": float(val), "undefined": bool(undefined)}
            })

        return {
            "method": self.name,
            "meta": {
                "variant": self.variant,
                "scaling": "minmax",
                "ddof": self.ddof if self.variant == "classic" else None,
                "epsilon": self.epsilon if self.epsilon > 0 else None,
            },
            "scores": scores,
        }
