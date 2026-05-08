from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.bayesian.dynamic_regression_shrinkage.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return dict(MODEL_PARAMS)


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    return {"global_shrinkage": trial.suggest_float("global_shrinkage", 0.2, 2.5, log=True), "slab_scale": trial.suggest_float("slab_scale", 0.5, 3.0), "feature_corr_weight": trial.suggest_float("feature_corr_weight", 0.2, 2.0), "coefficient_drift_scale": trial.suggest_float("coefficient_drift_scale", 0.0, 0.75)}
