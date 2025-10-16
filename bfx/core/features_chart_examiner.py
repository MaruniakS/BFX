from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bfx.core.dataset import Dataset
from bfx.charts import plot_feature_timeseries


class FeaturesChartExaminer:
    """
    Orchestrates chart rendering for a Dataset.
    Currently supports 'feature_timeseries' only.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.params: Dict[str, Any] = {
            "chart_type": "feature_timeseries",
            "scaling": "minmax",                    # 'raw' | 'minmax'
            "features": None,                       # None -> all, or List[str]
            # Chart cosmetics
            "figsize": (5, 5),
            "linewidth": 1.5,
            "grid": True,
            "legend": True,
            "anomaly_bg_color": "#eeeeee",
            # Output
            "filename": None,                       # e.g., 'timeseries.png'
            "dpi": 120,
        }

    def setParams(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if k in self.params:
                self.params[k] = v
            else:
                sys.exit("Unrecognized parameter:" + k)

    def run(self) -> Tuple[Any, Any, Optional[Path]]:
        """
        Render the selected chart and optionally save to disk.
        Saves to: <outdir>/<anomaly_name>/charts/<filename> if 'filename' is provided.
        Returns: (fig, ax, saved_path|None)
        """
        if self.dataset is None or self.dataset.data is None:
            raise ValueError("Dataset not ready: load input via Dataset.setParams({'input': ...}).")

        chart_type = self.params["chart_type"]
        if chart_type != "feature_timeseries":
            raise ValueError(f"Unsupported chart_type: {chart_type}")

        # Compute save path (optional)
        save_path: Optional[Path] = None
        filename: Optional[str] = self.params.get("filename")
        if filename:
            # ensure .png
            if not filename.lower().endswith(".png"):
                filename = f"{filename}.png"
            outdir = Path(self.dataset.params["outdir"])
            group = str(self.dataset.params["anomaly_name"] or "Anomaly")
            save_path = outdir / group / 'charts' / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plot_feature_timeseries(
            self.dataset,
            features=self.params.get("features"),
            scaling=self.params.get("scaling"),
            figsize=self.params.get("figsize"),
            linewidth=self.params.get("linewidth"),
            grid=self.params.get("grid"),
            legend=self.params.get("legend"),
            anomaly_bg_color=self.params.get("anomaly_bg_color"),
            save_path=str(save_path) if save_path else None,
            dpi=int(self.params.get("dpi")),
        )
        return fig, ax, save_path
