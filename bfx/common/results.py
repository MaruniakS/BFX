from __future__ import annotations
from typing import Dict, List, Any

def extract_method_block(result: Dict[str, Any], method: str) -> Dict[str, Any]:
    """
    Accepts either a whole FeaturesExaminer result (with 'results': {...})
    or a single-method block (with 'method': <method>, 'scores': ...).
    Returns {} if the method block is not found.
    """
    r = result or {}
    if r.get("method") == method and ("scores" in r or "windows" in r):
        return r
    res = r.get("results", {})
    if isinstance(res, dict):
        blk = res.get(method)
        if isinstance(blk, dict) and ("scores" in blk or "windows" in blk):
            return blk
    blk2 = r.get(method)
    if isinstance(blk2, dict) and ("scores" in blk2 or "windows" in blk2):
        return blk2
    return {}

def defined_scores(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return only rows whose details.undefined != True.
    If details are absent, consider the row defined.
    """
    scores = block.get("scores", []) or []
    out: List[Dict[str, Any]] = []
    for row in scores:
        det = row.get("details") or {}
        if not det.get("undefined", False):
            out.append(row)
    return out
