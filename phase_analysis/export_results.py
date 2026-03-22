from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def write_csv(path: Path, rows: List[Dict[str, object]]) -> str:
    if not rows:
        return str(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def write_json(path: Path, obj: object) -> str:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    return str(path)
