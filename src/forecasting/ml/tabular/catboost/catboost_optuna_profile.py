from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.tabular.catboost.numeric_profiles import resolve_regressor_params


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return resolve_regressor_params(
        task=str(task),
        model_threads=int(model_threads),
        interval_minutes=getattr(combo, "interval", None),
        horizon_minutes=getattr(combo, "horizon_minutes", None),
        training_window_months=getattr(combo, "training_window_months", None),
    )


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    params = {
        "loss_function": "RMSE",
        "iterations": trial.suggest_int("iterations", 120, 420, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.65, 1.0),
    }
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    params["bootstrap_type"] = bootstrap_type
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.65, 1.0)
    return params


def finalize_params(params: Dict[str, Any], combo: Any) -> Dict[str, Any]:
    cleaned = dict(params)
    bootstrap_type = str(cleaned.get("bootstrap_type", "")).strip()
    if bootstrap_type == "Bayesian":
        cleaned.pop("subsample", None)
    else:
        cleaned.pop("bagging_temperature", None)
    return cleaned
