from bfx.core import Dataset, FeaturesExaminer
from bfx import utils

def main():
    y,m,d = 2017,8,25
    ds = Dataset()
    ds.setParams({
        "input": "data/Features_1.json",
        "outdir": "out",
        "anomaly_name": "Google_Leak",
        "start_time": utils.get_timestamp(y,m,d,3,0,0),
        "end_time":   utils.get_timestamp(y,m,d,4,0,0),
        "anomaly_start_time": utils.get_timestamp(y,m,d,3,22,0),
        "anomaly_end_time":   utils.get_timestamp(y,m,d,3,36,0),
        "period": 1,
    })
    fx = FeaturesExaminer(ds)
    fx.setParams({
        "methods": ["cv"],
        "evaluator_params": {"cv": {"variant": "robust_mad", "epsilon": 1e-9}},
        "output_name": "google-leak-evals",
    })
    _, saved = fx.run()
    print("Saved:", saved or "(not saved)")

if __name__ == "__main__":
    main()
