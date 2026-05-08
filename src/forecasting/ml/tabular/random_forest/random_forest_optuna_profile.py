from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.tabular.random_forest.numeric_profiles import resolve_regressor_params


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
        "n_estimators": trial.suggest_int("n_estimators", 120, 420, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0]),
        "bootstrap": True,
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    }
