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

def _window_mask(idx: pd.DatetimeIndex, start: Optional[int], end: Optional[int]) -> np.ndarray:
    s = to_datetime_utc(int(start)) if start is not None else None
    e = to_datetime_utc(int(end)) if end is not None else None
    m = np.ones(len(idx), dtype=bool)
    if s is not None: m &= (idx >= s)
    if e is not None: m &= (idx <= e)
    return m

def _standardize(x: np.ndarray, mu0: float, sd0: float) -> np.ndarray:
    return (x - mu0) / sd0 if sd0 > 0 else np.full_like(x, np.nan, dtype=float)

def _cusum_curve(z: np.ndarray, k: float, sign: int) -> np.ndarray:
    S = np.zeros_like(z, dtype=float)
    for t in range(1, z.size):
        incr = (z[t] if sign > 0 else -z[t]) - k
        S[t] = max(0.0, S[t-1] + incr)
    return S

class CUSUMExplainer:
    """
    Explainer for CUSUM results.

    Params via constructor (from registry):
      artifacts: ["top_k_bar","cusum_curves","cusum_table"]
      features: Optional[List[str]]
      top_k: int = 12
      curves_top_k: int = 3
      subdir: str = "explain"
      draw_threshold: Optional[float] = None  # if provided, draw a horizontal h
      draw_anomaly_bounds: bool = True
    """

    name = "cusum"

    def __init__(self, **params) -> None:
        self.params = {
            "artifacts": [],
            "features": None,
            "top_k": 12,
            "curves_top_k": 3,
            "subdir": "explain",
            "draw_threshold": None,
            "draw_anomaly_bounds": True,
        }
        self.params.update(params or {})

    # ---- helpers ----

    def _extract_cusum_block(self, result: dict) -> dict:
        r = result or {}
        if r.get("method") == "cusum" and "scores" in r:
            return r
        ks = r.get("results", {}).get("cusum")
        if isinstance(ks, dict) and "scores" in ks:
            return ks
        ks2 = r.get("cusum")
        return ks2 if isinstance(ks2, dict) and "scores" in ks2 else {}

    def _plot_topk_bar(self, scores: List[Dict], k: int, outdir: Path) -> str:
        rows = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:k]
        if not rows:
            return ""
        feats = [r["feature"] for r in rows]
        vals  = [r["score"] for r in rows]
        outpath = outdir / "cusum_top_k_bar.png"
        plt.figure(figsize=(8, 0.35*len(rows) + 1.5))
        y = np.arange(len(rows))
        plt.barh(y, vals)
        plt.yticks(y, feats)
        plt.xlabel("CUSUM max (standardized units)")
        plt.title("Top features by CUSUM drift score")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _plot_cusum_curve(self, df: pd.DataFrame, dataset, feature: str, meta: dict, outdir: Path) -> str:
        # params
        side = (meta.get("side") or "both")
        k = float(meta.get("k") or 0.0)
        draw_h = self.params.get("draw_threshold", None)

        # build baseline and standardized z
        start = dataset.params.get("start_time"); end = dataset.params.get("end_time")
        as_   = dataset.params.get("anomaly_start_time"); ae = dataset.params.get("anomaly_end_time")
        full_mask = _window_mask(df.index, start, end)
        pre_mask = (full_mask & (df.index < to_datetime_utc(as_))) if (as_ is not None and ae is not None and as_ < ae) else None
        if pre_mask is None or pre_mask.sum() < 2:
            # fallback to first half for baseline
            order = np.where(full_mask)[0]; mid = order.size // 2
            base_idx = order[:max(2, mid)]
        else:
            base_idx = np.where(pre_mask)[0]

        x_full = df.loc[full_mask, feature].to_numpy(dtype=float, copy=False)
        x_base = df.iloc[base_idx][feature].to_numpy(dtype=float, copy=False)
        x_full = x_full[np.isfinite(x_full)]
        x_base = x_base[np.isfinite(x_base)]
        if x_base.size < 2:
            return ""

        mu0 = float(np.mean(x_base))
        sd0 = float(np.std(x_base, ddof=1))
        if not np.isfinite(sd0) or sd0 <= 0:
            return ""

        z = _standardize(x_full, mu0, sd0)

        Splus = _cusum_curve(z, k, sign=+1)
        Sminus = _cusum_curve(z, k, sign=-1)

        # pick which to draw
        draw_both = (side == "both")
        S = Splus if side == "up" else (Sminus if side == "down" else None)

        # timestamps for x-axis
        t = df.index[full_mask]

        outpath = outdir / f"cusum_curve_{feature}.png"
        plt.figure(figsize=(9, 4.8))
        if draw_both:
            plt.plot(t, Splus, label="S+ (up)")
            plt.plot(t, Sminus, label="S- (down)")
        else:
            lab = "S+ (up)" if side == "up" else "S- (down)"
            plt.plot(t, S, label=lab)

        # mark anomaly bounds
        if self.params.get("draw_anomaly_bounds", True) and (as_ is not None and ae is not None and as_ < ae):
            plt.axvline(to_datetime_utc(as_), linestyle="--", linewidth=1)
            plt.axvline(to_datetime_utc(ae), linestyle="--", linewidth=1)
        # threshold
        if draw_h is not None:
            plt.axhline(float(draw_h), linestyle="--", linewidth=1)

        plt.title(f"CUSUM curve: {feature}  (k={k}, baseline=pre or first_half)")
        plt.ylabel("CUSUM (standardized)")
        plt.xlabel("time (UTC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _write_table(self, scores: List[Dict], outdir: Path, top_feats: List[str]) -> str:
        rows = []
        by_feat = {r["feature"]: r for r in scores}
        for f in top_feats:
            r = by_feat.get(f)
            if not r: continue
            d = r.get("details", {})
            rows.append({
                "feature": f,
                "score": float(r.get("score", 0.0)),
                "direction": d.get("direction"),
                "t_at_Smax": d.get("t_at_Smax"),
                "mu0": d.get("mu0"),
                "sigma0": d.get("sigma0"),
                "n_pre": d.get("n_pre"),
                "baseline_window": d.get("baseline_window"),
            })
        df = pd.DataFrame(rows)
        out = outdir / "cusum_summary.csv"
        df.to_csv(out, index=False)
        return str(out)

    # ---- public API ----

    def explain(self, dataset, result, **kwargs) -> List[str]:
        # merge params
        p = self.params.copy(); p.update(kwargs or {})

        # output dir
        root = Path(dataset.params.get("outdir") or "out")
        anomaly = dataset.params.get("anomaly_name") or "unknown"
        outdir = root / anomaly / p["subdir"] / "cusum"
        outdir.mkdir(parents=True, exist_ok=True)

        # get block & df
        block = self._extract_cusum_block(result)
        scores = block.get("scores", []) or []
        meta = block.get("meta", {}) or {}

        # build df
        df = _make_index(pd.DataFrame(dataset.data),
                         dataset.params.get("start_time"),
                         int(dataset.params.get("period") or 1))

        outputs: List[str] = []
        rows_defined = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows_defined.sort(key=lambda r: r["score"], reverse=True)
        top_feats = [r["feature"] for r in rows_defined[:max(p["top_k"], p["curves_top_k"])]]
        feats_for_use = list(p["features"]) if p["features"] else top_feats

        for art in p["artifacts"]:
            if art in ("topk_bar","top_k_bar"):
                path = self._plot_topk_bar(scores, k=p["top_k"], outdir=outdir)
                if path: outputs.append(path)
            elif art == "cusum_curves":
                feats = (list(p["features"]) if p["features"] else top_feats[:p["curves_top_k"]])
                for f in feats:
                    path = self._plot_cusum_curve(df, dataset, f, meta, outdir)
                    if path: outputs.append(path)
            elif art == "cusum_table":
                feats_tbl = (list(p["features"]) if p["features"] else top_feats[:p["top_k"]])
                path = self._write_table(scores, outdir, feats_tbl)
                if path: outputs.append(path)

        return outputs

    def run(self, *args, **kwargs) -> List[str]:
        dataset = kwargs.get("dataset") if kwargs else None
        result = kwargs.get("result") if kwargs else None
        if (dataset is None or result is None) and len(args) >= 2:
            dataset, result = args[0], args[1]
        if dataset is None or result is None:
            raise ValueError("CUSUMExplainer.run expected (dataset, result).")
        return self.explain(dataset, result,
                            artifacts=self.params["artifacts"],
                            features=self.params["features"],
                            top_k=self.params["top_k"],
                            curves_top_k=self.params["curves_top_k"],
                            subdir=self.params["subdir"],
                            draw_threshold=self.params["draw_threshold"],
                            draw_anomaly_bounds=self.params["draw_anomaly_bounds"])
