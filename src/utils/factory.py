from __future__ import annotations

from importlib import import_module
from typing import Any

DATA_REGISTRY: dict[str, dict[str, str]] = {
    "default": {
        "path": "src.data.default",
        "class_name": "DefaultDataModule",
    },
    "shakespeare_char": {
        "path": "src.data.shakespeare_char",
        "class_name": "ShakespeareCharDataModule",
    },
}

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "default": {
        "path": "src.models.default",
        "class_name": "DefaultModel",
    },
    "gpt2_char": {
        "path": "src.models.gpt2",
        "class_name": "GPT2LanguageModel",
    },
    "gpt2_boosting_char": {
        "path": "src.models.gpt2",
        "class_name": "GPT2BoostingLanguageModel",
    },
}

SCORE_REGISTRY: dict[str, dict[str, str]] = {
    "default": {
        "path": "src.scores.default",
        "class_name": "DefaultScore",
    },
    "perplexity": {
        "path": "src.scores.perplexity",
        "class_name": "PerplexityScore",
    },
}


def _resolve_component_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    component_type = config.get("type") or "default"
    params = config.get("params")
    if params is None:
        params = {
            key: value
            for key, value in config.items()
            if key not in {"type", "name", "params"}
        }
    return component_type, params


def _create_component(config: dict[str, Any], registry: dict[str, dict[str, str]]):
    component_type, params = _resolve_component_config(config)
    if component_type not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown component type '{component_type}'. Available types: {available}")

    spec = registry[component_type]
    module = import_module(spec["path"])
    component_class = getattr(module, spec["class_name"])
    return component_class(params)


def create_data_module(config: dict[str, Any]):
    return _create_component(config, DATA_REGISTRY)


def create_model(config: dict[str, Any]):
    return _create_component(config, MODEL_REGISTRY)


def create_score(config: dict[str, Any]):
    return _create_component(config, SCORE_REGISTRY)
