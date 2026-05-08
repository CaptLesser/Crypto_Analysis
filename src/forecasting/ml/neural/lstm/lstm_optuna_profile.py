from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.neural.lstm.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    interval = int(getattr(combo, "interval", 60) or 60)
    sequence_length = 384 if interval <= 5 else 256 if interval <= 60 else 192
    return {**MODEL_PARAMS, "sequence_length": int(sequence_length)}


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {
        "hidden_size": trial.suggest_int("hidden_size", 32, 160, step=32),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "sequence_length": trial.suggest_int("sequence_length", 96, 512, step=32),
    }


def finalize_params(params: Dict[str, Any], combo: Any) -> Dict[str, Any]:
    out = dict(params)
    if int(out.get("num_layers", 1)) <= 1:
        out["dropout"] = 0.0
    return out
