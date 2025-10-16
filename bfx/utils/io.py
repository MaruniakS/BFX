from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def load_json(path: PathLike) -> Any:
    """
    Load a JSON file from an explicit path and return the parsed object.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, group: str, filename: str, root: str = ".") -> Path:
    """
    Save to <root>/<group>/<filename>.json.
    """
    base = Path(root) / Path(group)
    path = base / filename
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path