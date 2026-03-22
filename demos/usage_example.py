from bfx.core import Dataset, FeaturesExaminer, FeaturesExplainer
from bfx.utils import get_timestamp

# Set up dataset for India Leak scenario
ds = Dataset()
ds.setParams({
  "input": "data/India_Leak.json",
  "anomaly_name": "India_Leak",
  # Window: April 16, 2021 13:20 - 14:20 UTC
  "start_time": get_timestamp(2021, 4, 16, 13, 20, 0),
  "end_time": get_timestamp(2021, 4, 16, 14, 20, 0),
  # Expected Anomaly: April 16, 2021 13:48 - 14:00 UTC
  "anomaly_start_time": get_timestamp(2021, 4, 16, 13, 48, 0),
  "anomaly_end_time": get_timestamp(2021, 4, 16, 14, 0, 0),
  "interval": 1,
})

# Examine features
fx = FeaturesExaminer(ds)
fx.setParams({
  "methods": ["cv", "entropy", "ks", "auc", "cusum"],
  "evaluator_params": {  # Optional parameters for each method
    "cv": {"variant": "robust_mad", "epsilon": 0.0},
  }
})
result = fx.run()

# Explain top features
ex = FeaturesExplainer(ds, result)
ex.setParams({
  "methods": ["cv", "entropy", "ks", "auc", "cusum"],
  "top_k": 8,
  "explanation_params": {
    "cv": {"delta_key": "during_minus_pre"},
  }
})
ex.run()



