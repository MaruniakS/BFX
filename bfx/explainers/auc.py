# bfx/explainers/auc_explainer.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from bfx.core.dataset import Dataset
from .base import Explainer
from ..common.ts import make_index, window_masks
from ..charts import top_k_barh, plot_roc_curve


class AUCExplainer(Explainer):
    """
    Artifacts:
      - "top_k_bar"       : horizontal bar of |Cliff's Δ| (larger = stronger separation)
      - "roc_curves"      : ROC plots (ALL selected features by default; limit with 'roc_top_k')
      - "threshold_table" : CSV of Youden-J* operating points for selected features

    Selection semantics:
      - If 'features' is provided by the caller => use exactly those features.
      - Else => use ALL defined features (ordered by score desc).
      - If 'top_k' is provided => limit the *selected features* for table/ROC to that many.
      - Additionally, 'roc_top_k' (explainer param) can *further* cap how many ROC plots are drawn.
      - The Top-K bar uses the global ranking and K = top_k or 12 if top_k is None.
    """
    name = "auc"

    def default_params(self) -> Dict[str, object]:
        p = super().default_params()
        p.update({
            "artifacts": [],   # e.g. ["top_k_bar","roc_curves","threshold_table"]
            "roc_top_k": None, # None => draw ROC for all selected features
        })
        return p

    # ---------- internals ----------

    @staticmethod
    def _sign_for_direction(pre: np.ndarray, dur: np.ndarray, delta: Optional[float]) -> int:
        """Positive sign if 'during' tends to be larger; negative otherwise."""
        if isinstance(delta, (int, float)):
            return +1 if delta >= 0 else -1
        return +1 if (np.nanmedian(dur) >= np.nanmedian(pre)) else -1

    # ---------- plots / tables ----------

    def _plot_top_k_bar(self, scores: List[Dict], k: int, outdir: Path) -> Optional[str]:
        rows = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:k]
        if not rows:
            return None
        labels = [r["feature"] for r in rows]
        values = [float(r["score"]) for r in rows]
        return top_k_barh(
            labels, values,
            xlabel="|Cliff's Δ| (0–1)",
            title="Top features by AUC/Cliff's Δ separability",
            outpath=outdir / "auc_top_k_bar.png",
        )

    def _plot_roc_curve(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        feature: str,
        meta: dict,
        details: dict,
        outdir: Path,
    ) -> Optional[str]:
        a, b = (meta.get("window_pair") or ["pre", "during"])
        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback="halves",
        )
        if a not in masks or b not in masks:
            return None

        pre = df.loc[masks[a], feature].to_numpy(dtype=float, copy=False)
        dur = df.loc[masks[b], feature].to_numpy(dtype=float, copy=False)
        sign = self._sign_for_direction(pre, dur, details.get("delta"))

        return plot_roc_curve(
            pre, dur,
            sign=sign,
            label=feature,
            title=f"ROC: {feature} ({a} vs {b})",
            outpath=outdir / f"auc_roc_{feature}.png",
        )

    def _write_threshold_table(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        features: Sequence[str],
        meta: dict,
        details_by_feat: Dict[str, Dict],
        outdir: Path,
    ) -> Optional[str]:
        a, b = (meta.get("window_pair") or ["pre", "during"])
        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback="halves",
        )
        if a not in masks or b not in masks:
            return None

        rows = []
        for f in features:
            pre = df.loc[masks[a], f].to_numpy(dtype=float, copy=False)
            dur = df.loc[masks[b], f].to_numpy(dtype=float, copy=False)
            pre = pre[np.isfinite(pre)]; dur = dur[np.isfinite(dur)]
            m, n = pre.size, dur.size
            if m == 0 or n == 0:
                continue

            d = details_by_feat.get(f, {})
            sign = self._sign_for_direction(pre, dur, d.get("delta"))

            # Re-compute Youden-J* over all unique thresholds in the signed space
            s_pre = sign * pre
            s_dur = sign * dur
            all_s = np.unique(np.concatenate([s_pre, s_dur]))
            thresh = np.concatenate(([np.inf], all_s[::-1], [-np.inf]))

            TPR, FPR = [], []
            for t in thresh:
                tp = float(np.sum(s_dur >= t)); fp = float(np.sum(s_pre >= t))
                TPR.append(tp / n if n else 0.0)
                FPR.append(fp / m if m else 0.0)
            TPR = np.array(TPR); FPR = np.array(FPR)
            J = TPR - FPR
            i = int(np.argmax(J))
            t_score = float(thresh[i])
            x_star = t_score / float(sign) if sign != 0 else np.nan

            tp = float(np.sum(s_dur >= t_score)); fp = float(np.sum(s_pre >= t_score))
            fn = float(n - tp); tn = float(m - fp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else None
            bal_acc = 0.5 * ((tp / n if n else 0.0) + (tn / m if m else 0.0))

            rows.append({
                "feature": f,
                "auc": float(d.get("auc", np.nan)),
                "delta": float(d.get("delta", np.nan)) if isinstance(d.get("delta"), (int, float)) else np.nan,
                "direction": ("up" if sign > 0 else "down"),
                "score_abs_delta": float(abs(d.get("delta", 0.0))) if isinstance(d.get("delta"), (int, float)) else np.nan,
                "x_at_max_J": float(x_star) if np.isfinite(x_star) else None,
                "TPR": float(TPR[i]),
                "FPR": float(FPR[i]),
                "precision": float(prec) if prec is not None else None,
                "balanced_accuracy": float(bal_acc),
                "n_pre": int(m),
                "n_during": int(n),
            })

        if not rows:
            return None
        out = outdir / "auc_thresholds.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        return str(out)

    # ---------- contract ----------

    def explain(
        self,
        dataset: Dataset,
        method_block: Dict[str, object],
        *,
        features: List[str],
        top_k: Optional[int],
    ) -> List[str]:
        outdir = self.base_dir(dataset)

        scores = list(method_block.get("scores", []) or [])
        meta   = dict(method_block.get("meta", {}) or {})

        # Global ordering by score desc
        rows_def = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows_def.sort(key=lambda r: r["score"], reverse=True)
        all_defined = [r["feature"] for r in rows_def]

        # Selected feature set for table/ROC:
        chosen = (features or all_defined)
        if top_k is not None:
            chosen = chosen[: int(top_k)]

        details_by_feat = {r["feature"]: r.get("details", {}) for r in scores}

        # Build DF with the same time index the evaluator used
        df = make_index(
            pd.DataFrame(dataset.data),
            dataset.params.get("start_time"),
            int(dataset.params.get("period") or 1),
        )

        outputs: List[str] = []
        for art in self.params["artifacts"]:
            if art == "top_k_bar":
                # For the bar only: default to K=12 if top_k is None
                k_bar = int(top_k or len(rows_def))
                p = self._plot_top_k_bar(scores, k=k_bar, outdir=outdir)
                if p: outputs.append(p)

            elif art == "roc_curves":
                # Respect an optional ROC-only cap
                roc_cap = self.params.get("roc_top_k")
                feats_for_roc = chosen[: int(roc_cap)] if roc_cap is not None else chosen
                for f in feats_for_roc:
                    p = self._plot_roc_curve(df, dataset, f, meta, details_by_feat.get(f, {}), outdir)
                    if p: outputs.append(p)

            elif art == "threshold_table":
                p = self._write_threshold_table(df, dataset, chosen, meta, details_by_feat, outdir)
                if p: outputs.append(p)

        return outputs
