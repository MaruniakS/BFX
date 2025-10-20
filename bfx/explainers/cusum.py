from __future__ import annotations
from typing import Dict, List, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import to_datetime_utc
from bfx.charts import plot_cusum_top_k_bar, plot_cusum_curves


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
    if s is not None:
        m &= (idx >= s)
    if e is not None:
        m &= (idx <= e)
    return m


def _standardize(x: np.ndarray, mu0: float, sd0: float) -> np.ndarray:
    return (x - mu0) / sd0 if (np.isfinite(sd0) and sd0 > 0.0) else np.full_like(x, np.nan, dtype=float)


def _cusum_curve(z: np.ndarray, k: float, sign: int) -> np.ndarray:
    S = np.zeros_like(z, dtype=float)
    for t in range(1, z.size):
        incr = (z[t] if sign > 0 else -z[t]) - k
        S[t] = max(0.0, S[t - 1] + incr)
    return S


class CUSUMExplainer:
    name = "cusum"

    def __init__(self, **params) -> None:
        self.params = {
            "artifacts": [],
            "features": None,
            "top_k": None,
            "curves_top_k": None,
            "subdir": "explain",
            "draw_threshold": None,
            "draw_anomaly_bounds": True,
        }
        self.params.update(params or {})

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
        rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        rows = rows[:k]
        if not rows:
            return ""
        feats = [r["feature"] for r in rows]
        vals = [float(r["score"]) for r in rows]
        outpath = outdir / "cusum_top_k_bar.png"
        path = plot_cusum_top_k_bar(feats, vals, save_path=str(outpath))
        return path or ""

    def _plot_cusum_curve(self, df: pd.DataFrame, dataset, feature: str, meta: dict, outdir: Path) -> str:
        side = (meta.get("side") or "both")
        k = float(meta.get("k") or 0.0)
        draw_h = self.params.get("draw_threshold", None)

        start = dataset.params.get("start_time")
        end = dataset.params.get("end_time")
        as_ = dataset.params.get("anomaly_start_time")
        ae = dataset.params.get("anomaly_end_time")

        win_mask = _window_mask(df.index, start, end)

        x_series = df[feature].astype(float)
        finite_mask = np.isfinite(x_series.to_numpy(dtype=float, copy=False))
        full_mask = (win_mask & finite_mask)
        if full_mask.sum() < 2:
            return ""

        if (as_ is not None and ae is not None and as_ < ae):
            pre_mask = (df.index < to_datetime_utc(as_)) & full_mask
            base_idx = np.where(pre_mask)[0]
        else:
            base_idx = np.where(full_mask)[0]
            half = max(2, base_idx.size // 2)
            base_idx = base_idx[:half]

        x_full = x_series.to_numpy(dtype=float, copy=False)[full_mask]
        t_full = df.index[full_mask]
        x_base = x_series.to_numpy(dtype=float, copy=False)[base_idx]

        x_base = x_base[np.isfinite(x_base)]
        if x_base.size < 2:
            return ""

        mu0 = float(np.mean(x_base))
        sd0 = float(np.std(x_base, ddof=1))
        if not (np.isfinite(sd0) and sd0 > 0.0):
            return ""

        z = _standardize(x_full, mu0, sd0)
        if not np.isfinite(z).any():
            return ""

        s_plus = _cusum_curve(z, k, sign=+1)
        s_minus = _cusum_curve(z, k, sign=-1)

        outpath = outdir / f"cusum_curve_{feature}.png"
        a0_dt = to_datetime_utc(as_) if (self.params.get("draw_anomaly_bounds", True) and isinstance(as_, (int, float))) else None
        a1_dt = to_datetime_utc(ae)  if (self.params.get("draw_anomaly_bounds", True) and isinstance(ae, (int, float))) else None

        path = plot_cusum_curves(
            t_full, s_plus, s_minus,
            side=side,
            a0=a0_dt,
            a1=a1_dt,
            threshold=draw_h,
            title=f"CUSUM curves: {feature} (k={k}, baseline=pre or first_half)",
            save_path=str(outpath),
        )
        return path or ""

    def _write_table(self, scores: List[Dict], outdir: Path, top_feats: List[str]) -> str:
        rows = []
        by_feat = {r["feature"]: r for r in scores}
        for f in top_feats:
            r = by_feat.get(f)
            if not r:
                continue
            d = r.get("details", {}) or {}
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

    def explain(self, dataset, result, **kwargs) -> List[str]:
        p = self.params.copy()
        p.update(kwargs or {})

        root = Path(dataset.params.get("outdir") or "out")
        anomaly = dataset.params.get("anomaly_name") or "unknown"
        outdir = root / anomaly / p["subdir"] / "cusum"
        outdir.mkdir(parents=True, exist_ok=True)

        block = self._extract_cusum_block(result)
        scores = block.get("scores", []) or []
        meta = block.get("meta", {}) or {}

        df = _make_index(
            pd.DataFrame(dataset.data),
            dataset.params.get("start_time"),
            int(dataset.params.get("period") or 1),
        )

        outputs: List[str] = []

        rows_defined = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows_defined.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        ordered_feats = [r["feature"] for r in rows_defined]
        base_feats = [f for f in (p.get("features") or ordered_feats) if f in ordered_feats]

        def _to_int_or(val, default_len: int) -> int:
            return int(val) if isinstance(val, (int, float)) else int(default_len)

        top_k_safe = _to_int_or(p.get("top_k"), len(base_feats))
        curves_top_k_safe = _to_int_or(p.get("curves_top_k"), top_k_safe)

        top_feats = base_feats[:top_k_safe]
        curve_feats = base_feats[:curves_top_k_safe]

        arts: Sequence[str] = p.get("artifacts") or []

        for art in arts:
            if art == "top_k_bar":
                path = self._plot_topk_bar(scores, k=top_k_safe, outdir=outdir)
                if path:
                    outputs.append(path)
            elif art == "cusum_curves":
                for f in curve_feats:
                    path = self._plot_cusum_curve(df, dataset, f, meta, outdir)
                    if path:
                        outputs.append(path)
            elif art == "summary_table":
                path = self._write_table(scores, outdir, top_feats)
                if path:
                    outputs.append(path)

        return outputs

    def run(self, *args, **kwargs) -> List[str]:
        dataset = kwargs.get("dataset") if kwargs else None
        result = kwargs.get("result") if kwargs else None
        if (dataset is None or result is None) and len(args) >= 2:
            dataset, result = args[0], args[1]
        if dataset is None or result is None:
            raise ValueError("CUSUMExplainer.run expected (dataset, result).")
        return self.explain(
            dataset,
            result,
            artifacts=self.params["artifacts"],
            features=self.params["features"],
            top_k=self.params["top_k"],
            curves_top_k=self.params["curves_top_k"],
            subdir=self.params["subdir"],
            draw_threshold=self.params["draw_threshold"],
            draw_anomaly_bounds=self.params["draw_anomaly_bounds"],
        )
