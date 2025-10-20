# demos/cusum_explainer_demo.py
from bfx.core import Dataset, FeaturesExaminer, FeaturesExplainer
from bfx import get_timestamp

def main():
    y,m,d = 2017,8,25
    ds = Dataset()
    ds.setParams({
        "input": "data/Features_1.json",
        "outdir": "out",
        "anomaly_name": "Google_Leak",
        "start_time": get_timestamp(y,m,d,3,0,0),
        "end_time":   get_timestamp(y,m,d,4,0,0),
        "anomaly_start_time": get_timestamp(y,m,d,3,22,0),
        "anomaly_end_time":   get_timestamp(y,m,d,3,36,0),
        "period": 1,
    })

    # Evaluate CUSUM
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["cusum"],
        "evaluator_params": {
            "cusum": {
                "side": "both",           # "up"|"down"|"both"
                "k": 0.0,                 # reference in SD units
                "baseline_window": "auto",# prefer "pre", fallback to "first_half"
                "min_samples_pre": 5,
            }
        },
        "output_name": "google-leak-cusum",
    })
    result, saved = fx.run()
    print("[BFX] CUSUM JSON:", saved or "(not saved)")

    # Explain CUSUM: top-K bar, CUSUM curves, and a CSV summary
    ex = FeaturesExplainer(ds, result)
    ex.setParams({
        "methods": ["cusum"],
        "features": None,
        "top_k": 12,
        "explainer_params": {
            "cusum": {
                "artifacts": ["top_k_bar","cusum_curves","cusum_table"],
                "curves_top_k": 3,
                "subdir": "explain",
                "draw_threshold": None,           # e.g., 5.0 to draw a horizontal h
                "draw_anomaly_bounds": True
            }
        }
    })
    outputs = ex.run()
    print("[BFX] CUSUM explainer outputs:", outputs.get("cusum", []))

if __name__ == "__main__":
    main()
