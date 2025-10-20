from __future__ import annotations
import argparse

from bfx.core import Dataset, FeaturesExaminer, FeaturesExplainer
from demos.scenarios import SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="google_leak")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--roc-top-k", type=int, default=None)
    ap.add_argument("--overlay-top-k", type=int, default=None)
    ap.add_argument("--diff-top-k", type=int, default=None)
    ap.add_argument("--cusum-curves-top-k", type=int, default=None)
    args = ap.parse_args()

    if args.scenario not in SCENARIOS:
        raise KeyError(f"Scenario '{args.scenario}' not found")
    s = SCENARIOS[args.scenario]

    ds = Dataset()
    ds.setParams({
        "input": s["input"],
        "outdir": s.get("outdir", "out"),
        "anomaly_name": s.get("anomaly_name", args.scenario),
        "start_time": s["start_time"],
        "end_time": s["end_time"],
        "anomaly_start_time": s.get("anomaly_start_time"),
        "anomaly_end_time": s.get("anomaly_end_time"),
        "period": s.get("period", 1),
    })
    scenario_features = s.get("features")

    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["cv", "entropy", "ks", "auc", "cusum"],
        "evaluator_params": {
            "cv": {
                "variant": "robust_mad",
                "epsilon": 0.0
            },
            "entropy": {
                "bins": 20,
                "base": 2.0,
                "scaling": "minmax",
                "epsilon": 1e-12
            },
            "ks": {
                "window_pair": ["pre", "during"],
                "min_samples_per_window": 5,
                "fallback": "halves"
            },
            "auc": {
                "window_pair": ["pre", "during"],
                "min_samples_per_window": 5,
                "fallback": "halves",
                "ties": "average"
            },
            "cusum": {
                "side": "both",
                "k": 0.0,
                "baseline_window": "auto",
                "min_samples_pre": 5
            }
        },
        "output_name": f"{args.scenario}-all-methods",
    })
    result, saved = fx.run()
    print("[BFX] Examiner JSON:", saved or "(not saved)")

    ex = FeaturesExplainer(ds, result)
    ex.setParams({
        "methods": ["cv", "entropy", "ks", "auc", "cusum"],
        "features": scenario_features,
        "top_k": args.top_k,
        "explainer_params": {
            "cv": {
                "artifacts": ["delta_bar", "triplets", "scatter_pd"],
                "subdir": "explain"
            },
            "entropy": {
                "artifacts": ["delta_bar", "triplets", "scatter_pd"],
                "subdir": "explain"
            },
            "ks": {
                "artifacts": ["topk_bar", "ecdf_overlays", "diff_curves", "threshold_table"],
                "overlay_top_k": args.overlay_top_k,
                "diff_top_k": args.diff_top_k,
                "threshold_top_k": args.top_k,
                "subdir": "explain"
            },
            "auc": {
                "artifacts": ["top_k_bar", "roc_curves", "threshold_table"],
                "roc_top_k": args.roc_top_k,
                "subdir": "explain"
            },
            "cusum": {
                "artifacts": ["top_k_bar", "cusum_curves", "summary_table"],
                "curves_top_k": args.cusum_curves_top_k,
                "draw_threshold": None,
                "draw_anomaly_bounds": True,
                "subdir": "explain"
            }
        }
    })
    outputs = ex.run()
    for m, files in outputs.items():
        print(f"[BFX] {m} outputs:", files)


if __name__ == "__main__":
    main()
