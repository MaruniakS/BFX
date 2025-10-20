from __future__ import annotations

from typing import Dict, Type

from bfx.evaluators import FeatureEvaluator, CVEvaluator, EntropyEvaluator, KSEvaluator, CUSUMEvaluator, AUCEvaluator
from bfx.explainers import Explainer, CVExplainer, EntropyExplainer, KSExplainer, CUSUMExplainer,  AUCExplainer


EVALUATOR_REGISTRY: Dict[str, Type[FeatureEvaluator]] = {
    "cv": CVEvaluator,
    "entropy": EntropyEvaluator,
    "ks": KSEvaluator,
    "cusum": CUSUMEvaluator,
    "auc": AUCEvaluator,
}

EXPLAINER_REGISTRY: Dict[str, Type[Explainer]] = {
    "cv": CVExplainer,
    "entropy": EntropyExplainer,
    "ks": KSExplainer,
    "cusum": CUSUMExplainer,
    "auc": AUCExplainer,
}


def get_evaluator_class(name: str) -> Type[FeatureEvaluator]:
    try:
        return EVALUATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown evaluator: {name}")

def get_explainer_class(name: str) -> Type[Explainer]:
    try:
        return EXPLAINER_REGISTRY.get(name)   
    except KeyError:
        raise ValueError(f"Unknown explainer: {name}")  

def list_evaluators() -> Dict[str, str]:
    return {k: v.__name__ for k, v in EVALUATOR_REGISTRY.items()}
