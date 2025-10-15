# Demo: FeatureChartExaminer for "Google Leak" anomaly
# Window: Aug 25, 2017, 03:00–04:00 UTC
# Anomaly shading: 03:22:00–03:36:00 UTC
# Input: data/Features_1.json
# Output: out/Google_Leak/google-leak-timeseries.png

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the demo without installing the package
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bfx.core import Dataset, FeaturesChartExaminer  # noqa: E402
from bfx.utils import get_timestamp  # noqa: E402


def main() -> None:
    # --- Config ---
    year, month, day = 2017, 8, 25

    # Investigation window
    start_time = get_timestamp(year, month, day, 3, 0, 0)   # 03:00:00 UTC
    end_time   = get_timestamp(year, month, day, 4, 0, 0)   # 04:00:00 UTC

    # Anomaly window (explicit HH:MM:SS)
    anomaly_start_time = get_timestamp(year, month, day, 3, 22, 0)  # 03:22:00
    anomaly_end_time   = get_timestamp(year, month, day, 3, 36, 0)  # 03:36:00

    # Input/output
    input_path = PROJECT_ROOT / "data" / "Features_1.json"
    anomaly_name = "Google_Leak"  # folder under out/

    # --- Dataset ---
    ds = Dataset()
    ds.setParams({
        "input": str(input_path),
        "outdir": "out",
        "anomaly_name": anomaly_name,
        "start_time": start_time,
        "end_time": end_time,
        "anomaly_start_time": anomaly_start_time,
        "anomaly_end_time": anomaly_end_time,
        "period": 1,
    })

    # --- Chart examiner ---
    chart = FeaturesChartExaminer(ds)
    chart.setParams({
        "chart_type": "feature_timeseries",       
        "scaling": "minmax", 
        "features": ['nb_A', 'nb_W', 'nb_implicit_W', 'nb_dup_A'],                       # None -> all; or provide a list of labels
        "filename": "google-leak-timeseries.png",
        "dpi": 140,
    })

    fig, ax, saved = chart.run()
    print(f"Saved chart to: {saved}" if saved else "Chart rendered (not saved).")


if __name__ == "__main__":
    main()
