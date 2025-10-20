# Run: python -m demos.ks_explainer_demo
from bfx.core import Dataset, FeaturesExaminer, FeaturesExplainer
from bfx import get_timestamp

def main():
    y, m, d = 2017, 8, 25
    ds = Dataset()
    ds.setParams({
        "input": "data/Features_1.json",
        "outdir": "out",
        "anomaly_name": "Google_Leak",
        "start_time": get_timestamp(y, m, d, 3, 0, 0),
        "end_time":   get_timestamp(y, m, d, 4, 0, 0),
        "anomaly_start_time": get_timestamp(y, m, d, 3, 22, 0),
        "anomaly_end_time":   get_timestamp(y, m, d, 3, 36, 0),
        "period": 1,
    })

    # Evaluate KS (feature-agnostic distribution shift, no binning)
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["ks"],
        "evaluator_params": {
            "ks": {
                "window_pair": ["pre", "during"],  # fallback to halves if anomaly missing
                "min_samples_per_window": 5,
                "fallback": "halves"
            }
        },
        "output_name": "google-leak-ks",
    })
    result, saved = fx.run()
    print("[BFX] KS JSON:", saved or "(not saved)")

    # Explain KS:
    # - top-K bar (features ranked by KS D)
    # - ECDF overlays for the top 3 features, max gap annotated
    ex = FeaturesExplainer(ds, result)
    ex.setParams({
        "methods": ["ks"],
        "features": None,   # None -> consider all numeric features
        "top_k": 12,
        "explainer_params": {
            "ks": {
                "artifacts": ["topk_bar"],
                "overlay_top_k": 3,     # ECDF overlays
                "diff_top_k": 3,        # difference-curve plots (defaults to overlay_top_k if omitted)
                "threshold_top_k": 12,  # rows in ks_thresholds.csv (defaults to top_k)
                "subdir": "explain"
            }
        }
    })
    outputs = ex.run()
    print("[BFX] KS explainer outputs:", outputs.get("ks", []))

if __name__ == "__main__":
    main()
