from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Type


DATASET_REGISTRY: Dict[str, tuple[str, str]] = {
    # name: (module_path, class_name)
    "pure_esconv": ("utils.pure_esconv_dataset", "PureESConvDataset"),
    "strategy_esconv": ("utils.strategy_esconv_dataset", "StrategyESConvDataset"),
    "strategy_all_esconv": ("utils.strategy_all_esconv_dataset", "StrategyAllESConvDataset"),
    "situation_esconv": ("utils.situation_esconv_dataset", "SituationESConvDataset"),
    "emotion_type_esconv": ("utils.emotion_type_esconv_dataset", "EmotionTypeESConvDataset"),
    "problem_type_esconv": ("utils.problem_type_esconv_dataset", "ProblemTypeESConvDataset"),
}


def dataset_choices() -> List[str]:
    return sorted(list(DATASET_REGISTRY.keys()))


def get_dataset(name: str) -> Type[Any]:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset name: {name}. Available: {dataset_choices()}")
    module_path, class_name = DATASET_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


