from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.tabular.elasticnet.numeric_profiles import resolve_regressor_params


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
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
    }
