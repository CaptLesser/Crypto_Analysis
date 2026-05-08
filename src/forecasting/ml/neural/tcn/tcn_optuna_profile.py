from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.neural.tcn.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    interval = int(getattr(combo, "interval", 60) or 60)
    sequence_length = 384 if interval <= 5 else 256 if interval <= 60 else 192
    return {**MODEL_PARAMS, "sequence_length": int(sequence_length)}


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {
        "channel_width": trial.suggest_int("channel_width", 16, 128, step=16),
        "dilation_depth": trial.suggest_int("dilation_depth", 3, 8),
        "kernel_size": trial.suggest_int("kernel_size", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "sequence_length": trial.suggest_int("sequence_length", 96, 512, step=32),
    }
