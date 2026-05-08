from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.neural.nbeats.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    interval = int(getattr(combo, "interval", 60) or 60)
    input_length = 384 if interval <= 5 else 256 if interval <= 60 else 192
    return {**MODEL_PARAMS, "input_length": int(input_length)}


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {
        "num_stacks": trial.suggest_int("num_stacks", 2, 8),
        "num_blocks": trial.suggest_int("num_blocks", 1, 4),
        "layer_width": trial.suggest_int("layer_width", 64, 320, step=32),
        "generic_architecture": trial.suggest_categorical("generic_architecture", [True, False]),
        "input_length": trial.suggest_int("input_length", 128, 512, step=32),
    }
