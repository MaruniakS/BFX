# bfx/explainers/ks_explainer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from bfx.core.dataset import Dataset
from bfx.common.ts import make_index, window_masks
from bfx.charts import plot_delta_bar, plot_ecdf_overlay, plot_cdf_diff

from .base import Explainer


class KSExplainer(Explainer):
    """
    Artifacts:
      - "top_k_bar"      : horizontal bar of D (KS distance)
      - "ecdf_overlays"  : ECDF(pre) vs ECDF(during) with D and x*
      - "diff_curves"    : (F_during - F_pre)(x) with x*
      - "threshold_table": CSV of top rows (feature, D, x*, n_a, n_b, windows)
    """
    name = "ks"

    def default_params(self) -> Dict[str, Any]:
        return {
            **super().default_params(),
            "artifacts": ["top_k_bar", "ecdf_overlays"],
            "overlay_top_k": None,      # None => all selected
            "diff_top_k": None,         # None => overlay_top_k or all selected
            "threshold_top_k": None,    # None => global top_k
            "filenames": {
                "top_k_bar": "ks_top_k_bar.png",
                "threshold_table": "ks_thresholds.csv",
            },
        }

    def _write_threshold_table(self, rows_def: List[Dict[str, Any]], outdir: Path, k_rows: int) -> Optional[str]:
        if not rows_def:
            return None
        import pandas as pd
        rows = []
        for r in rows_def[:k_rows]:
            d = r.get("details", {}) or {}
            rows.append({
                "feature": r.get("feature"),
                "D": float(r.get("score", np.nan)),
                "x_at_max_gap": float(d.get("x_at_max_gap")) if isinstance(d.get("x_at_max_gap"), (int, float)) else None,
                "window_a": d.get("window_a"),
                "window_b": d.get("window_b"),
                "n_a": int(d.get("n_a")) if isinstance(d.get("n_a"), (int, float)) else None,
                "n_b": int(d.get("n_b")) if isinstance(d.get("n_b"), (int, float)) else None,
            })
        out = outdir / self.params["filenames"]["threshold_table"]
        pd.DataFrame(rows).to_csv(out, index=False)
        return str(out)

    def explain(
        self,
        dataset: Dataset,
        method_block: Dict[str, Any],
        *,
        features: List[str],
        top_k: Optional[int],
    ) -> List[str]:
        outdir = self.base_dir(dataset)

        scores = list((method_block or {}).get("scores", []) or [])
        meta   = dict((method_block or {}).get("meta", {}) or {})
        if not scores:
            return []

        # Order by D desc, skip undefined
        rows_def = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows_def.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        # Feature selection (global)
        all_defined = [r["feature"] for r in rows_def]
        chosen = (features or all_defined)
        if top_k is not None:
            chosen = chosen[: int(top_k)]

        details_by_feat = {r["feature"]: r.get("details", {}) for r in scores}

        # Build DF & masks once
        df = make_index(
            pd.DataFrame(dataset.data),
            dataset.params.get("start_time"),
            int(dataset.params.get("period") or 1),
        )
        masks = window_masks(
            df.index,
            dataset.params.get("start_time"),
            dataset.params.get("end_time"),
            dataset.params.get("anomaly_start_time"),
            dataset.params.get("anomaly_end_time"),
            fallback="halves",
        )
        a, b = (meta.get("window_pair") or ["pre", "during"])
        if a not in masks or b not in masks:
            return []

        outputs: List[str] = []

        arts: Sequence[str] = self.params.get("artifacts") or ["top_k_bar", "ecdf_overlays"]

        # Pull knobs
        overlay_cap = self.params.get("overlay_top_k")
        diff_cap    = self.params.get("diff_top_k")
        thresh_cap  = self.params.get("threshold_top_k")

        # ---- top_k_bar (generic bar helper) ----
        if "top_k_bar" in arts:
            k_bar = int(top_k or 12)
            labels = [r["feature"] for r in rows_def[:k_bar]]
            values = [float(r["score"]) for r in rows_def[:k_bar]]
            p = outdir / self.params["filenames"]["top_k_bar"]
            # plot_delta_bar returns (fig, ax); we save and return the path
            plot_delta_bar(labels, values, title="KS — Top features by D", xlabel="D (0–1)", save_path=str(p))
            outputs.append(str(p))

        # Feature caps for per-feature plots
        overlay_feats = chosen[: int(overlay_cap)] if overlay_cap is not None else chosen
        diff_feats = overlay_feats if diff_cap is None else chosen[: int(diff_cap)]

        # ---- ecdf_overlays (generic ECDF overlay) ----
        if "ecdf_overlays" in arts:
            for f in overlay_feats:
                xa = df.loc[masks[a], f].to_numpy(dtype=float, copy=False)
                xb = df.loc[masks[b], f].to_numpy(dtype=float, copy=False)
                det = details_by_feat.get(f, {}) or {}
                p = plot_ecdf_overlay(
                    xa, xb,
                    a_label=a, b_label=b,
                    x_label=f,
                    title=f"ECDF overlay — {f} ({a} vs {b})",
                    x_star=det.get("x_at_max_gap"),
                    d_value=det.get("D"),
                    save_path=str(outdir / f"ks_ecdf_{f}.png"),
                )
                if p:
                    outputs.append(p)

        # ---- diff_curves (generic CDF-difference) ----
        if "diff_curves" in arts:
            for f in diff_feats:
                xa = df.loc[masks[a], f].to_numpy(dtype=float, copy=False)
                xb = df.loc[masks[b], f].to_numpy(dtype=float, copy=False)
                det = details_by_feat.get(f, {}) or {}
                p = plot_cdf_diff(
                    xa, xb,
                    a_label=a, b_label=b,
                    x_label=f,
                    title=f"CDF difference — {f} (F_{b} - F_{a})",
                    x_star=det.get("x_at_max_gap"),
                    d_value=det.get("D"),
                    save_path=str(outdir / f"ks_diff_{f}.png"),
                )
                if p:
                    outputs.append(p)

        # ---- threshold_table (unchanged) ----
        if "threshold_table" in arts:
            k_rows = int(thresh_cap or (top_k or len(rows_def)))
            p = self._write_threshold_table(rows_def, outdir, k_rows=k_rows)
            if p:
                outputs.append(p)

        return outputs
