# Run with: python -m demos.entropy_examiner_demo
from bfx.core import Dataset, FeaturesExaminer
from bfx import get_timestamp

def main() -> None:
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

    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["entropy"],
        "evaluator_params": {
            "entropy": {"bins": 20, "base": 2.0, "scaling": "minmax", "epsilon": 1e-12}
        },
        "output_name": "google-leak-entropy",
    })
    _, saved = fx.run()
    print("[BFX] Saved:", saved or "(not saved)")

if __name__ == "__main__":
    main()
