# bfx/explainers/auc_explainer.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils import to_datetime_utc


def _make_index(df: pd.DataFrame, start_ts: Optional[int], period_min: int) -> pd.DataFrame:
    time_col = next((c for c in df.columns if str(c).lower() in ("timestamp", "time", "ts")), None)
    if time_col is not None:
        idx = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
        df = df.drop(columns=[time_col])
        df.index = idx
        return df
    start = to_datetime_utc(int(start_ts)) if start_ts is not None else pd.Timestamp.utcnow().tz_localize("UTC")
    df.index = start + pd.to_timedelta(np.arange(len(df)) * int(max(period_min, 1)), unit="m")
    return df

def _extract_auc_block(result: dict) -> dict:
    r = result or {}
    if r.get("method") == "auc" and "scores" in r:
        return r
    results = r.get("results", {})
    if isinstance(results, dict) and isinstance(results.get("auc"), dict):
        return results["auc"]
    if isinstance(r.get("auc"), dict) and "scores" in r["auc"]:
        return r["auc"]
    return {}

def _roc_points(pre: np.ndarray, dur: np.ndarray, sign: int = +1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ROC by thresholding score = sign * x with rule score >= t => predict 'during'.
    Returns (FPR, TPR) arrays including (0,0) and (1,1).
    """
    pre = pre[np.isfinite(pre)]
    dur = dur[np.isfinite(dur)]
    m, n = pre.size, dur.size
    if m == 0 or n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    s_pre = sign * pre
    s_dur = sign * dur
    thresh = np.unique(np.concatenate([s_pre, s_dur]))
    # add +/- inf guards to close curve
    thresh = np.concatenate(([np.inf], thresh[::-1], [-np.inf]))

    TPR, FPR = [], []
    for t in thresh:
        tp = float(np.sum(s_dur >= t))
        fp = float(np.sum(s_pre >= t))
        fn = float(n - tp)
        tn = float(m - fp)
        TPR.append(tp / n if n else 0.0)
        FPR.append(fp / m if m else 0.0)
    return np.array(FPR), np.array(TPR)

def _youden_from_roc(thresh: np.ndarray, FPR: np.ndarray, TPR: np.ndarray) -> Tuple[int, float]:
    """
    Given ROC arrays in the same order as thresholds, return index of max Youden J and its value.
    """
    J = TPR - FPR
    if J.size == 0:
        return 0, 0.0
    i = int(np.argmax(J))
    return i, float(J[i])


class AUCExplainer:
    """
    Explainer for AUC / Cliff's Δ results.

    Params via constructor (from registry):
      artifacts: ["top_k_bar","roc_curves","threshold_table"]
      features: Optional[List[str]]   # if None, uses top-K by |Δ|
      top_k: int = 12                 # table / bar
      roc_top_k: int = 3              # #features for ROC overlays
      subdir: str = "explain"
    """

    name = "auc"

    def __init__(self, **params) -> None:
        self.params = {
            "artifacts": [],
            "features": None,
            "top_k": 12,
            "roc_top_k": 3,
            "subdir": "explain",
        }
        self.params.update(params or {})

    # ---------- plots & tables ----------

    def _plot_topk_bar(self, scores: List[Dict], k: int, outdir: Path) -> str:
        rows = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        # sort by |Δ| (score)
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:k]
        if not rows:
            return ""
        feats = [r["feature"] for r in rows]
        vals  = [r["score"] for r in rows]
        outpath = outdir / "auc_top_k_bar.png"
        plt.figure(figsize=(8, 0.35*len(rows) + 1.5))
        y = np.arange(len(rows))
        plt.barh(y, vals)
        plt.yticks(y, feats)
        plt.xlabel("|Cliff's Δ| (0–1)")
        plt.title("Top features by AUC/Cliff's Δ separability")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _plot_roc_curve(self, df: pd.DataFrame, dataset, feature: str, meta: dict, outdir: Path) -> str:
        # windows
        a, b = (meta.get("window_pair") or ["pre","during"])
        start = dataset.params.get("start_time"); end = dataset.params.get("end_time")
        as_   = dataset.params.get("anomaly_start_time"); ae = dataset.params.get("anomaly_end_time")

        # build masks (reuse evaluator logic succinctly)
        idx = df.index
        base = np.ones(len(idx), dtype=bool)
        if start is not None: base &= (idx >= to_datetime_utc(int(start)))
        if end   is not None: base &= (idx <= to_datetime_utc(int(end)))

        have_trip = (as_ is not None and ae is not None and as_ < ae)
        if have_trip:
            pre_mask  = base & (idx <  to_datetime_utc(int(as_)))
            dur_mask  = base & (idx >= to_datetime_utc(int(as_))) & (idx < to_datetime_utc(int(ae)))
            post_mask = base & (idx >= to_datetime_utc(int(ae)))
            masks = {"pre": pre_mask, "during": dur_mask, "post": post_mask}
        else:
            order = np.where(base)[0]; mid = order.size // 2
            m1 = np.zeros_like(base, dtype=bool); m1[order[:mid]] = True
            m2 = np.zeros_like(base, dtype=bool); m2[order[mid:]] = True
            masks = {"first_half": m1, "second_half": m2}

        if a not in masks or b not in masks:
            return ""

        pre = df.loc[masks[a], feature].to_numpy(dtype=float, copy=False)
        dur = df.loc[masks[b], feature].to_numpy(dtype=float, copy=False)

        # direction: use evaluator delta sign if available by recomputing a tiny AUC here
        from math import copysign
        # quick directional hint: if during tends larger, set sign=+1, else -1
        # (re-rank here is unnecessary; use medians as a heuristic fallback)
        med_pre = np.nanmedian(pre) if pre.size else 0.0
        med_dur = np.nanmedian(dur) if dur.size else 0.0
        sign = +1 if med_dur >= med_pre else -1

        FPR, TPR = _roc_points(pre, dur, sign=sign)

        outpath = outdir / f"auc_roc_{feature}.png"
        plt.figure(figsize=(6.4, 5.4))
        plt.plot(FPR, TPR, label=f"{feature}")
        plt.plot([0,1], [0,1], linestyle="--", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC: {feature} ({a} vs {b})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _write_threshold_table(
        self,
        df: pd.DataFrame,
        dataset,
        features: Sequence[str],
        meta: dict,
        scores_by_feat: Dict[str, Dict[str, float]],
        outdir: Path,
    ) -> str:
        """
        Compute Youden's J best threshold per feature (direction-aware),
        along with TPR/FPR/Precision/BalancedAccuracy.
        """
        a, b = (meta.get("window_pair") or ["pre","during"])
        start = dataset.params.get("start_time"); end = dataset.params.get("end_time")
        as_   = dataset.params.get("anomaly_start_time"); ae = dataset.params.get("anomaly_end_time")

        # masks again
        idx = df.index
        base = np.ones(len(idx), dtype=bool)
        if start is not None: base &= (idx >= to_datetime_utc(int(start)))
        if end   is not None: base &= (idx <= to_datetime_utc(int(end)))

        have_trip = (as_ is not None and ae is not None and as_ < ae)
        if have_trip:
            pre_mask  = base & (idx <  to_datetime_utc(int(as_)))
            dur_mask  = base & (idx >= to_datetime_utc(int(as_))) & (idx < to_datetime_utc(int(ae)))
            masks = {"pre": pre_mask, "during": dur_mask}
        else:
            order = np.where(base)[0]; mid = order.size // 2
            m1 = np.zeros_like(base, dtype=bool); m1[order[:mid]] = True
            m2 = np.zeros_like(base, dtype=bool); m2[order[mid:]] = True
            # interpret first_half→"pre", second_half→"during"
            masks = {"pre": m1, "during": m2}

        rows = []
        for f in features:
            pre = df.loc[masks[a], f].to_numpy(dtype=float, copy=False)
            dur = df.loc[masks[b], f].to_numpy(dtype=float, copy=False)
            pre = pre[np.isfinite(pre)]; dur = dur[np.isfinite(dur)]
            m, n = pre.size, dur.size
            if m == 0 or n == 0:
                continue

            # pick sign based on delta if available, else using medians
            d = scores_by_feat.get(f, {})
            delta = d.get("delta", None)
            if isinstance(delta, (int, float)):
                sign = +1 if delta >= 0 else -1
            else:
                sign = +1 if (np.nanmedian(dur) >= np.nanmedian(pre)) else -1

            s_pre = sign * pre
            s_dur = sign * dur
            all_s = np.unique(np.concatenate([s_pre, s_dur]))
            # close curve with guards
            thresh = np.concatenate(([np.inf], all_s[::-1], [-np.inf]))

            TPR, FPR = [], []
            for t in thresh:
                tp = float(np.sum(s_dur >= t)); fp = float(np.sum(s_pre >= t))
                fn = float(n - tp); tn = float(m - fp)
                TPR.append(tp / n if n else 0.0)
                FPR.append(fp / m if m else 0.0)
            TPR = np.array(TPR); FPR = np.array(FPR)
            J = TPR - FPR
            i = int(np.argmax(J))
            # map threshold back to original x scale
            t_score = float(thresh[i])
            x_star = t_score / float(sign) if sign != 0 else np.nan

            tp = float(np.sum(s_dur >= t_score)); fp = float(np.sum(s_pre >= t_score))
            fn = float(n - tp); tn = float(m - fp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else None
            bal_acc = 0.5 * ((tp / n if n else 0.0) + (tn / m if m else 0.0))

            rows.append({
                "feature": f,
                "auc": float(d.get("auc", np.nan)),
                "delta": float(d.get("delta", np.nan)),
                "direction": ("up" if sign > 0 else "down"),
                "score_abs_delta": float(abs(d.get("delta", 0.0))),
                "x_at_max_J": float(x_star) if np.isfinite(x_star) else None,
                "TPR": float(TPR[i]),
                "FPR": float(FPR[i]),
                "precision": float(prec) if prec is not None else None,
                "balanced_accuracy": float(bal_acc),
                "n_pre": int(m),
                "n_during": int(n),
            })

        out = outdir / "auc_thresholds.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        return str(out)

    # ---------- public API ----------

    def explain(
        self,
        dataset,
        result,
        artifacts: Sequence[str] | None = None,
        features: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        roc_top_k: Optional[int] = None,
        subdir: Optional[str] = None,
    ) -> List[str]:
        # merge params
        p = self.params.copy()
        if artifacts is not None: p["artifacts"] = list(artifacts)
        if features  is not None: p["features"]  = list(features)
        if top_k    is not None: p["top_k"]     = int(top_k)
        if roc_top_k is not None: p["roc_top_k"] = int(roc_top_k)
        if subdir   is not None: p["subdir"]    = str(subdir)

        # output dir
        root = Path(dataset.params.get("outdir") or "out")
        anomaly = dataset.params.get("anomaly_name") or "unknown"
        outdir = root / anomaly / p["subdir"] / "auc"
        outdir.mkdir(parents=True, exist_ok=True)

        # block & df
        block = _extract_auc_block(result)
        scores = block.get("scores", []) or []
        meta   = block.get("meta", {}) or {}

        df = _make_index(pd.DataFrame(dataset.data),
                         dataset.params.get("start_time"),
                         int(dataset.params.get("period") or 1))

        outputs: List[str] = []
        rows_def = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows_def.sort(key=lambda r: r["score"], reverse=True)
        top_feats = [r["feature"] for r in rows_def[:max(p["top_k"], p["roc_top_k"])]]
        use_feats = list(p["features"]) if p["features"] else top_feats

        # cache details by feature (auc, delta)
        details_by_feat = {r["feature"]: r.get("details", {}) for r in scores}

        for art in p["artifacts"]:
            if art in ("topk_bar", "top_k_bar"):
                path = self._plot_topk_bar(scores, k=p["top_k"], outdir=outdir)
                if path: outputs.append(path)

            elif art == "roc_curves":
                feats = (list(p["features"]) if p["features"] else top_feats[:p["roc_top_k"]])
                for f in feats:
                    path = self._plot_roc_curve(df, dataset, f, meta, outdir=outdir)
                    if path: outputs.append(path)

            elif art == "threshold_table":
                feats_tbl = (list(p["features"]) if p["features"] else top_feats[:p["top_k"]])
                path = self._write_threshold_table(df, dataset, feats_tbl, meta, details_by_feat, outdir)
                if path: outputs.append(path)

        return outputs

    def run(self, *args, **kwargs) -> List[str]:
        dataset = kwargs.get("dataset") if kwargs else None
        result  = kwargs.get("result")  if kwargs else None
        if (dataset is None or result is None) and len(args) >= 2:
            dataset, result = args[0], args[1]
        if dataset is None or result is None:
            raise ValueError("AUCExplainer.run expected (dataset, result).")
        return self.explain(
            dataset=dataset,
            result=result,
            artifacts=self.params["artifacts"],
            features=self.params["features"],
            top_k=self.params["top_k"],
            roc_top_k=self.params["roc_top_k"],
            subdir=self.params["subdir"],
        )
