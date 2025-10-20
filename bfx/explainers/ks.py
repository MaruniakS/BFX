# bfx/explainers/ks_explainer.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils import to_datetime_utc


def _make_index(df: pd.DataFrame, start_ts: Optional[int], period_min: int) -> pd.DataFrame:
    """
    Attach a UTC datetime index to the dataframe, preferring an explicit epoch-seconds time column.
    Falls back to synthetic minute steps from start_ts and period_min.
    """
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
    mask = np.ones(len(idx), dtype=bool)
    if s is not None:
        mask &= (idx >= s)
    if e is not None:
        mask &= (idx <= e)
    return mask


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return sorted samples and ECDF values (post-step).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    fs = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, fs


class KSExplainer:
    """
    Explainer for KS results. Expects a KS evaluator output under result["results"]["ks"].

    Params accepted via constructor (from registry):
      artifacts: ["topk_bar","ecdf_overlays","diff_curve","threshold_table"]
      features: Optional[List[str]]         # if None, uses top-K by KS
      top_k: int = 12                       # top-N for ranking/table
      overlay_top_k: int = 3                # top-N for ECDF overlays
      diff_top_k: int = None                # if None -> overlay_top_k
      threshold_top_k: int = None           # if None -> top_k
      subdir: str = "explain"

    Created with:
      expl = KSExplainer(**explainer_params["ks"])
    Then the framework calls:
      outputs = expl.run(dataset, result)
    """

    name = "ks"

    def __init__(self, **params) -> None:
        # default params consumed by run()/explain()
        self.params = {
            "artifacts": [],
            "features": None,
            "top_k": 12,
            "overlay_top_k": 3,
            "diff_top_k": None,         # defaults to overlay_top_k
            "threshold_top_k": None,    # defaults to top_k
            "subdir": "explain",
        }
        self.params.update(params or {})

    # ---------- window resolution ----------

    def _resolve_windows(self, df: pd.DataFrame, dataset, win_a: str, win_b: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map logical window names (pre/during/post or halves) to row indices.
        Falls back to halves if anomaly bounds are missing/invalid.
        """
        start = dataset.params.get("start_time")
        end = dataset.params.get("end_time")
        as_ = dataset.params.get("anomaly_start_time")
        ae = dataset.params.get("anomaly_end_time")

        full_mask = _window_mask(df.index, start, end)
        order = np.where(full_mask)[0]

        if win_a in ("pre", "during", "post") or win_b in ("pre", "during", "post"):
            if as_ is None or ae is None or as_ >= ae:
                # fall back to halves
                mid = order.size // 2
                first, second = order[:mid], order[mid:]
                m = {"first_half": first, "second_half": second}
                return m.get(win_a, first), m.get(win_b, second)

            pre_mask = full_mask & (df.index < to_datetime_utc(as_))
            dur_mask = full_mask & (df.index >= to_datetime_utc(as_)) & (df.index < to_datetime_utc(ae))
            post_mask = full_mask & (df.index >= to_datetime_utc(ae))
            m = {
                "pre": np.where(pre_mask)[0],
                "during": np.where(dur_mask)[0],
                "post": np.where(post_mask)[0],
            }
            return m.get(win_a, m["pre"]), m.get(win_b, m["during"])

        # halves
        mid = order.size // 2
        return order[:mid], order[mid:]

    # ---------- computation helpers ----------

    def _ecdf_components(self, df: pd.DataFrame, dataset, feature: str, win_a: str, win_b: str):
        """
        Compute ECDF grid, Fa (pre), Fb (during), D, x*, Fa(x*), Fb(x*).
        """
        idx_a, idx_b = self._resolve_windows(df, dataset, win_a, win_b)
        xa = df.iloc[idx_a][feature].to_numpy(dtype=float)
        xb = df.iloc[idx_b][feature].to_numpy(dtype=float)
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]

        xs_a, _ = _ecdf(xa)
        xs_b, _ = _ecdf(xb)

        grid = np.sort(np.unique(np.concatenate([xs_a, xs_b]))) if (xs_a.size or xs_b.size) else np.array([])
        def _F(xs: np.ndarray, grid: np.ndarray) -> np.ndarray:
            if xs.size == 0:
                return np.zeros_like(grid, dtype=float)
            idx = np.searchsorted(xs, grid, side="right")
            return idx / float(xs.size)

        Fa = _F(xs_a, grid)
        Fb = _F(xs_b, grid)
        diff = np.abs(Fa - Fb)
        dmax_idx = int(np.argmax(diff)) if diff.size else 0
        xstar = grid[dmax_idx] if grid.size else np.nan
        D = float(diff[dmax_idx]) if diff.size else 0.0
        Fa_at = float(Fa[dmax_idx]) if diff.size else 0.0
        Fb_at = float(Fb[dmax_idx]) if diff.size else 0.0
        return xa, xb, grid, Fa, Fb, D, xstar, Fa_at, Fb_at

    # ---------- plotting primitives ----------

    def _plot_topk_bar(self, ks_scores: List[Dict], k: int, outdir: Path) -> str:
        print("[KSExplainer]  - top-K bar: selecting top", k, "features")
        print("[KSExplainer]    from", ks_scores, "total features")
        rows = [r for r in ks_scores if not r.get("details", {}).get("undefined", False)]
        print("[KSExplainer]  - top-K bar: preparing top", k, "features from", len(rows), "defined features")
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:k]
        if not rows:
            return ""

        feats = [r["feature"] for r in rows]
        vals = [r["score"] for r in rows]

        outpath = outdir / "ks_topk_bar.png"
        plt.figure(figsize=(8, 0.35 * len(rows) + 1.5))
        y = np.arange(len(rows))
        plt.barh(y, vals)
        plt.yticks(y, feats)
        plt.xlabel("KS distance D (0–1)")
        plt.title("Top features by KS distribution shift")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _plot_ecdf_overlay(self, df: pd.DataFrame, dataset, feature: str, win_a: str, win_b: str, outdir: Path) -> str:
        xa, xb, grid, Fa, Fb, D, xstar, Fa_at, Fb_at = self._ecdf_components(df, dataset, feature, win_a, win_b)

        # Recompute ECDF x/steps for each sample set for nice step plots
        xs_a, fa = _ecdf(xa)
        xs_b, fb = _ecdf(xb)

        outpath = outdir / f"ks_ecdf_{feature}.png"
        plt.figure(figsize=(7, 5))
        if xs_a.size:
            plt.step(xs_a, fa, where="post", label=f"{win_a}")
        if xs_b.size:
            plt.step(xs_b, fb, where="post", label=f"{win_b}")
        if grid.size:
            plt.vlines(xstar, min(Fa_at, Fb_at), max(Fa_at, Fb_at), linestyles="dashed")
            plt.annotate(f"D = {D:.3f}", xy=(xstar, (Fa_at + Fb_at) / 2), xytext=(5, 5), textcoords="offset points")

        plt.xlabel(feature)
        plt.ylabel("ECDF")
        plt.title(f"ECDF overlay: {feature} ({win_a} vs {win_b})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _plot_diff_curve(self, df: pd.DataFrame, dataset, feature: str, win_a: str, win_b: str, outdir: Path) -> str:
        """
        Plot difference curve: F_during(x) - F_pre(x) vs x, mark x* where |diff| is max (= KS D).
        """
        xa, xb, grid, Fa, Fb, D, xstar, Fa_at, Fb_at = self._ecdf_components(df, dataset, feature, win_a, win_b)
        if grid.size == 0:
            return ""

        diff = Fb - Fa   # during - pre
        outpath = outdir / f"ks_diff_{feature}.png"
        plt.figure(figsize=(7, 4.5))
        plt.plot(grid, diff, linewidth=1.5)
        plt.axhline(0.0, linewidth=1.0)
        # mark max gap (sign included)
        sign = 1.0 if (Fb_at - Fa_at) >= 0 else -1.0
        plt.vlines(xstar, 0.0, sign * D, linestyles="dashed")
        plt.annotate(f"D = {D:.3f}", xy=(xstar, sign * D / 2.0), xytext=(6, 6), textcoords="offset points")
        plt.xlabel(feature)
        plt.ylabel("F_during(x) - F_pre(x)")
        plt.title(f"Difference curve: {feature} ({win_a} vs {win_b})")
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
        return str(outpath)

    def _write_threshold_table(
        self,
        df: pd.DataFrame,
        dataset,
        features: Sequence[str],
        ks_scores: Dict[str, float],
        win_a: str,
        win_b: str,
        outdir: Path,
    ) -> str:
        """
        Build a CSV with KS-optimal threshold x* per feature and counts/fractions on each side.
        """
        rows = []
        for f in features:
            xa, xb, grid, Fa, Fb, D, xstar, Fa_at, Fb_at = self._ecdf_components(df, dataset, f, win_a, win_b)
            n_pre = int(xa.size)
            n_dur = int(xb.size)

            # counts <= x* and > x*
            pre_le = int(np.sum(xa <= xstar)) if n_pre > 0 and np.isfinite(xstar) else 0
            pre_gt = n_pre - pre_le
            dur_le = int(np.sum(xb <= xstar)) if n_dur > 0 and np.isfinite(xstar) else 0
            dur_gt = n_dur - dur_le

            rows.append({
                "feature": f,
                "D": float(ks_scores.get(f, D)),
                "x_at_max_gap": float(xstar) if np.isfinite(xstar) else None,
                "pre_n": n_pre,
                "pre_le_count": pre_le,
                "pre_le_frac": (pre_le / n_pre) if n_pre else None,
                "pre_gt_count": pre_gt,
                "pre_gt_frac": (pre_gt / n_pre) if n_pre else None,
                "during_n": n_dur,
                "during_le_count": dur_le,
                "during_le_frac": (dur_le / n_dur) if n_dur else None,
                "during_gt_count": dur_gt,
                "during_gt_frac": (dur_gt / n_dur) if n_dur else None,
            })

        df_out = pd.DataFrame(rows)
        outpath = outdir / "ks_thresholds.csv"
        df_out.to_csv(outpath, index=False)
        return str(outpath)

    # ---------- public API ----------

    def explain(
        self,
        dataset,
        result,
        artifacts: Sequence[str] | None = None,
        features: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        overlay_top_k: Optional[int] = None,
        subdir: Optional[str] = None,
        diff_top_k: Optional[int] = None,
        threshold_top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Build requested artifacts and return list of generated file paths.
        Artifacts supported:
          - "topk_bar"
          - "ecdf_overlays"
          - "diff_curve"
          - "threshold_table"

        Saves under:
          <outdir>/<anomaly_name>/<subdir>/ks/
        """
        # merge call-time args with constructor params
        p = self.params.copy()
        if artifacts is not None:        p["artifacts"] = list(artifacts)
        if features is not None:         p["features"] = list(features)
        if top_k is not None:            p["top_k"] = int(top_k)
        if overlay_top_k is not None:    p["overlay_top_k"] = int(overlay_top_k)
        if subdir is not None:           p["subdir"] = str(subdir)
        if diff_top_k is not None:       p["diff_top_k"] = int(diff_top_k)
        if threshold_top_k is not None:  p["threshold_top_k"] = int(threshold_top_k)

        # Build working dataframe
        df = _make_index(
            pd.DataFrame(dataset.data),
            dataset.params.get("start_time"),
            int(dataset.params.get("period") or 1),
        )

        ks_block = result or {}
        scores = ks_block.get("scores", [])
        print("[KSExplainer] KS scores available for", len(scores), "features")
        meta = ks_block.get("meta", {})
        win_a, win_b = (meta.get("window_pair") or ["pre", "during"])

        # ---- OUTPUT DIRECTORY (fixed per your convention) ----
        root = Path(dataset.params.get("outdir") or "out")
        anomaly = dataset.params.get("anomaly_name") or "unknown"
        outdir = root / anomaly / p["subdir"] / "ks"
        outdir.mkdir(parents=True, exist_ok=True)

        outputs: List[str] = []

        # Resolve candidates for plots/tables
        rows_defined = [r for r in scores if not r.get("details", {}).get("undefined", False)]
        rows_defined.sort(key=lambda r: r["score"], reverse=True)
        top_feats = [r["feature"] for r in rows_defined[: max(p["top_k"], p["overlay_top_k"], p.get("diff_top_k") or 0)]]
        use_feats = list(p["features"]) if p["features"] else top_feats

        # Map feature -> D (for the table)
        ks_by_feat = {r["feature"]: float(r.get("details", {}).get("D", r["score"])) for r in scores}

        print("[KSExplainer] Generating artifacts:", p["artifacts"])
        for art in p["artifacts"]:
            if art == "topk_bar":
                path = self._plot_topk_bar(scores, k=p["top_k"], outdir=outdir)
                print("[KSExplainer]  - top-K bar:", path)
                if path: outputs.append(path)

            elif art == "ecdf_overlays":
                feats_for_overlay = (list(p["features"]) if p["features"] else top_feats[: p["overlay_top_k"]])
                for f in feats_for_overlay:
                    path = self._plot_ecdf_overlay(df, dataset, f, win_a, win_b, outdir=outdir)
                    if path: outputs.append(path)

            elif art == "diff_curve":
                k = p["diff_top_k"] if p["diff_top_k"] is not None else p["overlay_top_k"]
                feats_for_diff = (list(p["features"]) if p["features"] else top_feats[: k])
                for f in feats_for_diff:
                    path = self._plot_diff_curve(df, dataset, f, win_a, win_b, outdir=outdir)
                    if path: outputs.append(path)

            elif art == "threshold_table":
                k = p["threshold_top_k"] if p["threshold_top_k"] is not None else p["top_k"]
                feats_for_tbl = (list(p["features"]) if p["features"] else top_feats[: k])
                path = self._write_threshold_table(df, dataset, feats_for_tbl, ks_by_feat, win_a, win_b, outdir=outdir)
                if path: outputs.append(path)

        return outputs

    def run(self, *args, **kwargs) -> List[str]:
        """
        Compatibility with registries that call .run(dataset, result)
        or .run(dataset=..., result=...).
        """
        dataset = kwargs.get("dataset") if kwargs else None
        result = kwargs.get("result") if kwargs else None
        if (dataset is None or result is None) and len(args) >= 2:
            dataset, result = args[0], args[1]
        if dataset is None or result is None:
            raise ValueError("KSExplainer.run expected (dataset, result).")
        return self.explain(
            dataset=dataset,
            result=result,
            artifacts=self.params["artifacts"],
            features=self.params["features"],
            top_k=self.params["top_k"],
            overlay_top_k=self.params["overlay_top_k"],
            subdir=self.params["subdir"],
            diff_top_k=self.params["diff_top_k"],
            threshold_top_k=self.params["threshold_top_k"],
        )
