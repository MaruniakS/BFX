from __future__ import annotations
import argparse
from itertools import islice
from pathlib import Path
from typing import List

import pandas as pd
from bfx.charts import plot_feature_timeseries
from demos.demo_utils import build_dataset

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            cols.append(c)
    return cols

def _chunks(seq: List[str], size: int):
    it = iter(seq)
    while True:
        block = list(islice(it, size))
        if not block:
            return
        yield block

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="google_leak")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--scaling", choices=["raw", "minmax"], default="minmax")
    ap.add_argument("--tick-every-min", type=int, default=10)
    ap.add_argument("--time-fmt", default="%H.%M")
    ap.add_argument("--legend", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    args = ap.parse_args()

    ds, scenario_features = build_dataset(args.scenario)

    data = ds.data
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset has no data.")
    df = pd.DataFrame(data)
    feats = scenario_features or _numeric_cols(df)

    root = Path(ds.params.get("outdir") or "out")
    anomaly = ds.params.get("anomaly_name") or "unknown"
    outdir = root / anomaly / "examine"
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, group in enumerate(_chunks(feats, args.group_size), start=1):
        p = outdir / f"timeseries_{args.scaling}_{i:02d}.png"
        plot_feature_timeseries(
            ds,
            features=group,
            scaling=args.scaling,
            tick_every_min=args.tick_every_min,
            time_fmt=args.time_fmt,
            figsize=(10, 6),
            legend=bool(args.legend),
            save_path=str(p),
            dpi=150,
        )
        saved.append(str(p))

    print("[BFX] Chart examiner outputs:", saved)

if __name__ == "__main__":
    main()
