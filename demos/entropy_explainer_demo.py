# Run: python -m demos.entropy_explainer_demo
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

    # Evaluate entropy (same contract as CV)
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["entropy"],
        "evaluator_params": {"entropy": {"bins": 20, "base": 2.0, "scaling": "minmax", "epsilon": 1e-12}},
        "output_name": "google-leak-entropy",
    })
    result, saved = fx.run()
    print("[BFX] Entropy JSON:", saved or "(not saved)")

    # Explain entropy (delta bar + triplets + scatter)
    ex = FeaturesExplainer(ds, result)
    ex.setParams({
        "methods": ["entropy"],
        "features": None,   # None -> consider all features
        "top_k": 12,        # sensible for a paper figure
        "explainer_params": {
            "entropy": {"artifacts": ["delta_bar","triplets","scatter_pd"], "subdir": "explain"}
        }
    })
    outputs = ex.run()
    print("[BFX] Entropy explainer outputs:", outputs.get("entropy", []))

if __name__ == "__main__":
    main()
