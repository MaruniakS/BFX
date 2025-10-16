from __future__ import annotations

from typing import Dict, Type

from bfx.evaluators import EntropyEvaluator, CVEvaluator, FeatureEvaluator

EVALUATOR_REGISTRY: Dict[str, Type[FeatureEvaluator]] = {
    "entropy": EntropyEvaluator,
    "cv": CVEvaluator,
}

def get_evaluator_class(name: str) -> Type[FeatureEvaluator]:
    try:
        return EVALUATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown evaluator: {name}")

def list_evaluators() -> Dict[str, str]:
    return {k: v.__name__ for k, v in EVALUATOR_REGISTRY.items()}
