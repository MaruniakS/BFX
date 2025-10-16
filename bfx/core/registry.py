from __future__ import annotations

from typing import Dict, Type

from bfx.evaluators import EntropyEvaluator, CVEvaluator, FeatureEvaluator
from bfx.explainers import Explainer, CVExplainer, GenericScoresExplainer


EVALUATOR_REGISTRY: Dict[str, Type[FeatureEvaluator]] = {
    "entropy": EntropyEvaluator,
    "cv": CVEvaluator,
}

EXPLAINER_REGISTRY: Dict[str, Type[Explainer]] = {
    "cv": CVExplainer,
    # add more here as you implement them (e.g., "entropy": EntropyExplainer)
}


def get_evaluator_class(name: str) -> Type[FeatureEvaluator]:
    try:
        return EVALUATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown evaluator: {name}")

def get_explainer_class(name: str) -> Type[Explainer]:
    try:
        return EXPLAINER_REGISTRY.get(name, GenericScoresExplainer)   
    except KeyError:
        raise ValueError(f"Unknown evaluator: {name}")  

def list_evaluators() -> Dict[str, str]:
    return {k: v.__name__ for k, v in EVALUATOR_REGISTRY.items()}
