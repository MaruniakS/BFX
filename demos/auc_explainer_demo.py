# demos/auc_explainer_demo.py
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

    # Evaluate AUC/Cliff's Δ
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["auc"],
        "evaluator_params": {
            "auc": {
                "window_pair": ["pre","during"],
                "min_samples_per_window": 5,
                "fallback": "halves",
                "ties": "average",
            }
        },
        "output_name": "google-leak-auc",
    })
    result, saved = fx.run()
    print("[BFX] AUC JSON:", saved or "(not saved)")

    # Explain: bar + ROC + threshold table
    ex = FeaturesExplainer(ds, result)
    ex.setParams({
        "methods": ["auc"],
        "features": None,
        "top_k": 12,
        "explainer_params": {
            "auc": {
                "artifacts": ["top_k_bar","roc_curves","threshold_table"],
                "roc_top_k": 4,
                "subdir": "explain"
            }
        }
    })
    outputs = ex.run()
    print("[BFX] AUC explainer outputs:", outputs.get("auc", []))

if __name__ == "__main__":
    main()
