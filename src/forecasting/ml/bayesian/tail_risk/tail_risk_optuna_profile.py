from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.bayesian.tail_risk.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return dict(MODEL_PARAMS)


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {
        "threshold_quantile": trial.suggest_float("threshold_quantile", 0.03, 0.20),
        "min_tail_points": trial.suggest_int("min_tail_points", 6, 24),
        "shape_scale": trial.suggest_float("shape_scale", 0.5, 2.0),
        "tail_scale_multiplier": trial.suggest_float("tail_scale_multiplier", 0.5, 2.0),
        "model_two_tails": trial.suggest_categorical("model_two_tails", [False, True]),
        # Tail-risk stage 3 is scored on point RMSE, so expose a center shift
        # derived from tail imbalance instead of tuning only extreme quantiles.
        "center_tail_blend": trial.suggest_float("center_tail_blend", -1.5, 1.5),
    }
