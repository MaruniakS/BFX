# BFX: BGP Feature eXaminer <!-- omit in toc -->

A lightweight toolkit for **investigating BGP feature time-series** around suspected anomalies. It evaluates features with multiple statistical lenses and produces clear, publication-ready plots and tables.

---

## Table of Contents <!-- omit in toc -->
- [What’s inside](#whats-inside)
- [Install \& run](#install--run)
- [Data format](#data-format)
- [Scenarios](#scenarios)
- [Quick starts](#quick-starts)
  - [Run all explainers](#run-all-explainers)
  - [Run the chart examiner](#run-the-chart-examiner)
- [Outputs](#outputs)
- [Methods](#methods)
- [Usage \& Parameters (per method)](#usage--parameters-per-method)
  - [Common (Explainers)](#common-explainers)
  - [CV](#cv)
  - [Entropy](#entropy)
  - [KS (Kolmogorov–Smirnov)](#ks-kolmogorovsmirnov)
  - [AUC (Mann–Whitney / ROC)](#auc-mannwhitney--roc)
  - [CUSUM (Page CUSUM)](#cusum-page-cusum)
- [Explainers \& charts](#explainers--charts)
- [Extending](#extending)
  - [Add a new evaluator](#add-a-new-evaluator)
  - [Add a new explainer](#add-a-new-explainer)
  - [Add a new scenario](#add-a-new-scenario)
- [Project layout](#project-layout)
- [Conventions \& tips](#conventions--tips)
- [Roadmap ideas](#roadmap-ideas)
- [License](#license)

---

## What’s inside

- **Evaluators** (feature-agnostic): CV, Entropy, KS, AUC, CUSUM  
- **Explainers**: consistent, high-signal plots/tables per method  
- **Chart examiner**: min–max or raw time-series overlays with shaded anomaly window  
- **Demos**: unified runners using one source of truth for dataset parameters

---

## Install & run

```bash
# 1) ensure Pipenv is available
python -m pip install --user pipenv    # or: pip install --user pipenv

# 2) install dependencies from Pipfile / Pipfile.lock
pipenv install                         # adds dev deps if listed under [dev-packages]

# 3) activate the virtual environment
pipenv shell

# 4) run demos
python -m demos.run_explainers --scenario google_leak
python -m demos.run_chart_examiner --scenario google_leak

# (Alternative: without entering the shell)
pipenv run python -m demos.run_explainers --scenario google_leak
pipenv run python -m demos.run_chart_examiner --scenario google_leak
```

> Python 3.8+ is supported. The examples below assume `data/Google_Leak.json` exists as provided.

---

## Data format

Input is a **list of rows**, each row a dict of feature values for a fixed interval (e.g., one minute).  
No timestamps are required in the file; demos synthesize them from `start_time` and `period` (minutes).

```json
[
  {"nb_A": 612, "nb_W": 44, "avg_path_len": 5.7, "...": ...},
  {"nb_A": 630, "nb_W": 53, "avg_path_len": 5.6, "...": ...},
  ...
]
```

All non-numeric columns are dropped automatically by plotting utilities.

---

## Scenarios

All demo runners read the investigation setup from **one place**:

`demos/scenarios.py`
```python
from bfx import get_timestamp

SCENARIOS = {
    "google_leak": {
        "input": "data/Google_Leak.json",
        "outdir": "out",
        "anomaly_name": "Google_Leak",
        "start_time": get_timestamp(2017, 8, 25, 3, 0, 0),
        "end_time":   get_timestamp(2017, 8, 25, 4, 0, 0),
        "anomaly_start_time": get_timestamp(2017, 8, 25, 3, 22, 0),
        "anomaly_end_time":   get_timestamp(2017, 8, 25, 3, 36, 0),
        "period": 1,
        "features": None  # or a list to restrict which features are considered
    },
}
```

---

## Quick starts

### Run all explainers

Evaluates **CV, Entropy, KS, AUC, CUSUM** and renders all artifacts.

```bash
python -m demos.run_explainers \
  --scenario google_leak \
  --top-k 12 \
  --roc-top-k 4 \
  --overlay-top-k 6 \
  --diff-top-k 6 \
  --cusum-curves-top-k 6
```

### Run the chart examiner

Saves grouped time-series plots, with anomaly window shading.

```bash
python -m demos.run_chart_examiner \
  --scenario google_leak \
  --group-size 8 \
  --scaling minmax
```

---

## Outputs

Under `out/<Anomaly_Name>/`:

```
out/
└── Google_Leak/
    ├── examine/                    # chart examiner
    │   └── timeseries_minmax_01.png
    └── explain/                    # explainers by method
        ├── cv/
        │   ├── cv_delta_bar.png
        │   └── cv_triplets.png
        ├── entropy/
        │   ├── entropy_delta_bar.png
        │   ├── entropy_triplets.png
        │   └── entropy_scatter_pd.png
        ├── ks/
        │   ├── ks_top_k_bar.png
        │   ├── ks_ecdf_<feat>.png
        │   ├── ks_diff_<feat>.png
        │   └── ks_thresholds.csv
        ├── auc/
        │   ├── auc_top_k_bar.png
        │   ├── auc_roc_<feat>.png
        │   └── auc_thresholds.csv
        └── cusum/
            ├── cusum_top_k_bar.png
            ├── cusum_curve_<feat>.png
            └── cusum_summary.csv
```

---

## Methods

**All methods are per-feature, feature-agnostic.** They differ in the signal they emphasize:

- **CV** — scale-free variability  
  - Classic: `CV = σ / |μ|` ; Robust: `CV = (1.4826·MAD) / |median|`
- **Entropy** — dispersion/informativeness over a window  
  - Shannon entropy over histogram bins; global per-feature bin edges; per-window min–max scaling
- **KS** — two-sample Kolmogorov–Smirnov distance (`D`) between windows  
  - Detects distribution shape shifts, not just mean changes
- **AUC (Mann–Whitney)** — probability that a random during-value exceeds a random pre-value  
  - Directional separation (up/down); robust to scale
- **CUSUM** — cumulative sum of standardized residuals (Page CUSUM)  
  - Baseline from pre-window (or first half); plots S⁺/S⁻ curves over time

Each evaluator produces a compact JSON block with `method`, `meta`, and `scores`, optionally `windows` and `deltas` for windowed methods.

---

## Usage & Parameters (per method)

### Common (Explainers)
- `subdir`: `str` — default: `"explain"`
- `artifacts`: subset of the method’s allowed artifact names (see each method)

---

### CV

**Evaluator — `CVEvaluator`**

| Param | Type | Default | Notes |
|---|---|---|---|
| `variant` | `classic` \| `robust_mad` | `robust_mad` | Classic: `σ/\|μ\|`; Robust: `(1.4826·MAD)/\|median\|` |
| `ddof` | `int` | `1` | Used only when `variant = classic` |
| `epsilon` | `float` | `0.0` | Denominator stabilizer |

**Explainer — `CVExplainer`**

| Param | Type | Default | Allowed |
|---|---|---|---|
| `artifacts` | subset of enum | `["delta_bar","triplets"]` | `{ "delta_bar", "triplets", "scatter_pd" }` |
| `delta_key` | str | `during_minus_pre` | `during_minus_pre \| post_minus_pre` |
| `subdir` | `str` | `"explain"` | — | — |

---

### Entropy

**Evaluator — `EntropyEvaluator`**

| Param | Type | Default | Notes |
|---|---|---|---|
| `bins` | `int` \| `str` | `20` | Histogram bins (per-feature edges computed once) |
| `base` | `float` | `2.0` | Log base (bits when 2.0) |
| `scaling` | `str` | `"minmax"` | Per-window min–max across features |
| `epsilon` | `float` | `1e-12` | Log stability |

**Explainer — `EntropyExplainer`**

| Param | Type | Default | Allowed `artifacts` |
|---|---|---|---|
| `artifacts` | subset of enum | `["delta_bar","triplets","scatter_pd"]` | `{ "delta_bar", "triplets", "scatter_pd" }` |
| `delta_key` | str | `during_minus_pre` | `during_minus_pre \| post_minus_pre` |
| `subdir` | `str` | `"explain"` | — | — |

---

### KS (Kolmogorov–Smirnov)

**Evaluator — `KSEvaluator`**

| Param | Type | Default | Notes |
|---|---|---|---|
| `window_pair` | pair of enums | `["pre","during"]` | Each element ∈ `{ pre, during, post, first_half, second_half }`; fallback applies if invalid |
| `min_samples_per_window` | `int` | `5` | Minimum samples per window |
| `fallback` | `halves` | `halves` | Use first/second halves if needed |

**Explainer — `KSExplainer`**

| Param | Type | Default | Allowed `artifacts` | Notes |
|---|---|---|---|---|
| `artifacts` | subset of enum | `["top_k_bar","ecdf_overlays","diff_curves","threshold_table"]` | `{ "top_k_bar", "ecdf_overlays", "diff_curves", "threshold_table" }` | — |
| `overlay_top_k` | `int` \| `None` | inherits global cap | — | How many ECDF overlay plots |
| `diff_top_k` | `int` \| `None` | inherits overlay cap → else global cap | — | How many ECDF-difference plots |
| `threshold_top_k` | `int` \| `None` | inherits global cap | — | Rows in `ks_thresholds.csv` |
| `subdir` | `str` | `"explain"` | — | — |

---

### AUC (Mann–Whitney / ROC)

**Evaluator — `AUCEvaluator`**

| Param | Type | Default | Notes |
|---|---|---|---|
| `window_pair` | pair of enums | `["pre","during"]` | Each element ∈ `{ pre, during, post, first_half, second_half }`; fallback applies if invalid |
| `min_samples_per_window` | `int` | `5` | Minimum samples per window |
| `fallback` | `halves` | `halves` | Use first/second halves if needed |
| `ties` | `average` \| `max` \| `min` | `average` | Tie handling for U/AUC |

**Explainer — `AUCExplainer`**

| Param | Type | Default | Allowed `artifacts` | Notes |
|---|---|---|---|---|
| `artifacts` | subset of enum | `["top_k_bar","roc_curves","threshold_table"]` | `{ "top_k_bar", "roc_curves", "threshold_table" }` | — |
| `roc_top_k` | `int` \| `None` | inherits global cap | — | How many ROC curves |
| `subdir` | `str` | `"explain"` | — | — |

---

### CUSUM (Page CUSUM)

**Evaluator — `CUSUMEvaluator`**

| Param | Type | Default | Notes |
|---|---|---|---|
| `side` | `up` \| `down` \| `both` | `both` | Drift direction |
| `k` | `float` | `0.0` | Allowance (SD units) |
| `baseline_window` | `auto` \| `pre` \| `first_half` | `auto` | Prefers `pre`, else first half |
| `min_samples_pre` | `int` | `5` | Minimum for μ₀, σ₀ |

**Explainer — `CUSUMExplainer`**

| Param | Type | Default | Allowed `artifacts` | Notes |
|---|---|---|---|---|
| `artifacts` | subset of enum | `["top_k_bar","cusum_curves","summary_table"]` | `{ "top_k_bar", "cusum_curves", "summary_table" }` | — |
| `curves_top_k` | `int` \| `None` | inherits global cap | — | How many per-feature CUSUM plots |
| `draw_threshold` | `float` \| `None` | `None` | — | Horizontal reference line (y) |
| `draw_anomaly_bounds` | `bool` | `True` | — | Vertical lines at anomaly start/end |
| `subdir` | `str` | `"explain"` | — | — |


---

## Explainers & charts

Explainers turn evaluator outputs into plots and CSVs:

- **CV**: delta bar (during−pre), pre/during/post triplet bars
- **Entropy**: delta bar, triplets, ΔEntropy vs ΔCV scatter
- **KS**: top-K bar by `D`, ECDF overlays, ECDF-difference curves, threshold CSV
- **AUC**: top-K bar by |AUC−0.5|, ROC curves for top features, threshold CSV
- **CUSUM**: top-K bar by max CUSUM, S⁺/S⁻ curves, summary CSV

Reusable plotting lives in `bfx/charts/` (`time_series.py`, `scatter.py`, …).

---

## Extending

### Add a new evaluator

1) Create `bfx/evaluators/my_method.py`:

```python
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from .base import FeatureEvaluator

class MyMethodEvaluator(FeatureEvaluator):
    name = "my_method"

    def default_params(self) -> Dict[str, Any]:
        p = super().default_params()
        p.update({"param_a": 1.0})
        return p

    def _evaluate_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        scores: List[Dict[str, Any]] = []
        for col in df.columns:
            x = df[col].to_numpy(dtype=float)
            # compute per-feature score
            score = float(np.nanmean(x))  # placeholder
            scores.append({"feature": col, "score": score, "details": {}})
        return {"method": self.name, "meta": {"param_a": self.params["param_a"]}, "scores": scores}
```

2) Register it in your evaluator registry so `FeaturesExaminer` can discover `"my_method"`.

3) Run via demo by adding `"my_method"` to `methods` and optional params in `evaluator_params`.

### Add a new explainer

1) Create `bfx/explainers/my_method_explainer.py`:

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional
from bfx.core.dataset import Dataset
from .base import Explainer

class MyMethodExplainer(Explainer):
    name = "my_method"

    def default_params(self) -> Dict[str, Any]:
        return {**super().default_params(),
                "artifacts": ["top_k_bar"],
                "filenames": {"top_k_bar": "my_method_top_k_bar.png"}}

    def explain(self, dataset: Dataset, method_block: Dict[str, Any], *,
                features: List[str], top_k: Optional[int]) -> List[str]:
        outdir = self.base_dir(dataset)
        # parse method_block["scores"] and write plots under outdir
        # return list of saved file paths
        return []
```

2) Register the explainer in the explainer registry.

3) Use it from `demos/run_explainers.py` by adding it to `methods` and `explainer_params`.

### Add a new scenario

Edit `demos/scenarios.py` and add another key:

```python
SCENARIOS["my_case"] = {
    "input": "data/MyFeatures.json",
    "outdir": "out",
    "anomaly_name": "My_Case",
    "start_time": get_timestamp(YYYY, MM, DD, hh, mm, ss),
    "end_time":   get_timestamp(...),
    "anomaly_start_time": get_timestamp(...),
    "anomaly_end_time":   get_timestamp(...),
    "period": 1,
    "features": None
}
```

---

## Project layout

```
bfx/
├── core/
│   ├── dataset.py                # Dataset, FeaturesExaminer, FeaturesExplainer
│   └── ...                       # core wiring
├── evaluators/
│   ├── base.py
│   ├── cv.py
│   ├── ...
├── explainers/
│   ├── base.py
│   ├── cv.py
│   ├── ...
├── charts/
│   ├── time_series.py            # feature time-series plots (examiner)
│   ├── ...
└── utils.py                      # to_datetime_utc, helpers

demos/
├── scenarios.py                  # one source of truth for dataset params
├── run_explainers.py             # evaluate + explain all methods
└── run_chart_examiner.py         # time-series “examiner”

data/
└── Features.json               # sample 1-hour dataset (60 rows)

out/
└── ...                           # generated artifacts
```

---

## Conventions & tips

- **Windows**: many methods depend on `pre/during/post` derived from scenario times; if anomaly times are missing, fallbacks are applied (e.g., KS/AUC to halves; CUSUM baseline to first half).
- **Scaling**: chart examiner supports `raw` and `minmax`. For multipanel time-series, `minmax` with extended y-axis ticks improves readability.
- **Ties & constants**:
  - AUC uses a configurable tie-handling rule (`average` by default).
  - CUSUM charts are skipped for features with constant baselines (σ₀ ≤ 0) to avoid misleading standardized curves; this is intentional.
- **Artifacts naming** is stable and method-specific (see Outputs).

---

## Roadmap ideas

- Univariate change-point detectors (e.g., PELT/BOCPD) for richer drift timing
- Robust entropy variants (e.g., Kozachenko–Leonenko for continuous signals)
- Feature grouping (prefix-related, path-length-related) with aggregated explainers
- Packaging (`pyproject.toml`), simple CLI wrapper (`bfx-cli`), CI hooks, tests

---

## License

MIT License
