from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]

_OUT_DIR = "out"


def load_json(path: PathLike) -> Any:
    """
    Load a JSON file from an explicit path and return the parsed object.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    obj: Any,
    group: str,
    filename: str,
    *,
    root: PathLike | None = None,
    indent: int = 2,
) -> Path:
    """
    Save `obj` to out/<group>/<filename>.json under the current working directory
    (or a provided `root`).

    Args:
        obj: JSON-serializable object
        group: subfolder name under `out/` (e.g., 'anomaly_name')
        filename: file name without extension ('.json' is added)
        root: optional alternative root directory (defaults to cwd)
        indent: JSON indentation (default 2)

    Returns:
        Full Path to the written JSON file.
    """
    base = Path(root) if root is not None else Path.cwd()
    out_dir = base / _OUT_DIR / group
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{filename}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    return path
