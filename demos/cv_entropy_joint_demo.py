# Joint ΔCV vs ΔEntropy scatter for Google Leak
# Input:  data/Features_1.json
# Output: out/Google_Leak/explain/joint/cv_vs_entropy_delta.png

from bfx.core import Dataset, FeaturesExaminer
from bfx.charts import plot_scatter_deltas
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

    # Evaluate both methods in one pass
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["cv", "entropy"],
        "evaluator_params": {
            "cv": {"variant": "robust_mad", "epsilon": 1e-9},
            "entropy": {"bins": 20, "base": 2.0, "scaling": "minmax", "epsilon": 1e-12},
        },
        "output_name": "google-leak-cv-entropy",
    })
    result, saved = fx.run()
    print("[BFX] Saved:", saved or "(not saved)")

    # ---- build per-feature deltas for both methods ----
    res = result["results"]
    cv_d = {r["feature"]: float(r["delta"])
            for r in res["cv"]["deltas"]["during_minus_pre"]}
    ent_d = {r["feature"]: float(r["delta"])
             for r in res["entropy"]["deltas"]["during_minus_pre"]}

    # Choose the features to compare (intersection, in your chosen order)
    # You can also hardcode a list if you prefer strict control.
    feats = [f for f in cv_d.keys() if f in ent_d]  # intersection

    x = [cv_d[f] for f in feats]    # ΔCV
    y = [ent_d[f] for f in feats]   # ΔEntropy

    out_png = ds.params["outdir"] + f"/{ds.params['anomaly_name']}/explain/joint/cv_vs_entropy_delta.png"
    plot_scatter_deltas(
        x, y, feats,
        title="ΔEntropy vs ΔCV (during − pre)",
        xlabel="ΔCV (during − pre)",
        ylabel="ΔEntropy (during − pre)",
        save_path=out_png,
    )
    print("[BFX] Joint scatter:", out_png)

if __name__ == "__main__":
    main()
