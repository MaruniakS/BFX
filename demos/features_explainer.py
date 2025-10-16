# Demo: CV evaluation + explanation for "Google Leak"
# Input:  data/Features_1.json
# Output: out/Google_Leak/google-leak-evals.json
#         out/Google_Leak/explain/cv/cv_delta_bar.png
#         out/Google_Leak/explain/cv/cv_triplets.png

from bfx.core import Dataset, FeaturesExaminer, FeaturesExplainer
from bfx import utils

def main() -> None:
    # ---- Dataset window (UTC) ----
    y, m, d = 2017, 8, 25
    ds = Dataset()
    ds.setParams({
        "input": "data/Features_1.json",  # root-level data/
        "outdir": "out",                  # root-level out/
        "anomaly_name": "Google_Leak",
        "start_time": utils.get_timestamp(y, m, d, 3, 0, 0),
        "end_time":   utils.get_timestamp(y, m, d, 4, 0, 0),
        "anomaly_start_time": utils.get_timestamp(y, m, d, 3, 22, 0),
        "anomaly_end_time":   utils.get_timestamp(y, m, d, 3, 36, 0),
        "period": 1,  # 1-minute sampling
    })

    # ---- Evaluate CV (robust, article-ready defaults) ----
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["cv"],
        "evaluator_params": {
            # robust to spikes; stable when mean ~ 0
            "cv": {"variant": "robust_mad", "epsilon": 1e-9}
        },
        "output_name": "google-leak-evals",
    })
    result, saved = fx.run()
    print("[BFX] Examiner JSON:", saved or "(not saved)")

    # ---- Explain CV (top-K Δ(during−pre) + triplets) ----
    expl = FeaturesExplainer(ds, result)
    expl.setParams({
        "methods": ["cv"],     # explain the 'cv' block
        "features": None,      # None -> consider all features (good for fairness)
        "top_k": 12,           # pick the top 12 for visuals (readable in a figure)
        "explainer_params": {
            "cv": {
                "delta_key": "during_minus_pre",   # “what popped at anomaly time”
                "artifacts": ["delta_bar", "triplets"],
                "subdir": "explain",               # saves under out/<anomaly>/explain/cv/
                # filenames default to: cv_delta_bar.png, cv_triplets.png
            }
        }
    })
    outputs = expl.run()
    print("[BFX] Explainer outputs:", outputs.get("cv", []))

if __name__ == "__main__":
    main()
