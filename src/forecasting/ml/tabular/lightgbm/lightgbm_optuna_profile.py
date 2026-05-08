from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.tabular.lightgbm.numeric_profiles import resolve_regressor_params


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return resolve_regressor_params(
        task=str(task),
        model_threads=int(model_threads),
        interval_minutes=getattr(combo, "interval", None),
        horizon_minutes=getattr(combo, "horizon_minutes", None),
        training_window_months=getattr(combo, "training_window_months", None),
    )


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 3, 8)
    max_num_leaves = max(7, min(127, (2 ** int(max_depth)) - 1))
    return {
        "n_estimators": trial.suggest_int("n_estimators", 120, 420, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, max_num_leaves),
        "max_depth": int(max_depth),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
        "bagging_freq": 1,
        "reg_lambda": trial.suggest_float("reg_lambda", 0.25, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80, step=5),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.5),
        "feature_pre_filter": False,
    }
