from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.tabular.xgboost.numeric_profiles import resolve_regressor_params


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return resolve_regressor_params(
        task=str(task),
        model_threads=int(model_threads),
        interval_minutes=getattr(combo, "interval", None),
        horizon_minutes=getattr(combo, "horizon_minutes", None),
        training_window_months=getattr(combo, "training_window_months", None),
    )


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 120, 360, step=20),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }
