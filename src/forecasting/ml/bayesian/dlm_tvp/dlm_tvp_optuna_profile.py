from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.bayesian.dlm_tvp.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return {
        "level_smoothing": float(MODEL_PARAMS["level_smoothing"]),
        "trend_smoothing": float(MODEL_PARAMS["trend_smoothing"]),
        "seasonal_strength": float(MODEL_PARAMS["seasonal_strength"]),
        "observation_scale": float(MODEL_PARAMS["observation_scale"]),
    }


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {"level_smoothing": trial.suggest_float("level_smoothing", 0.02, 0.20), "trend_smoothing": trial.suggest_float("trend_smoothing", 0.02, 0.20), "seasonal_strength": trial.suggest_float("seasonal_strength", 0.0, 1.5), "observation_scale": trial.suggest_float("observation_scale", 0.5, 2.0)}
