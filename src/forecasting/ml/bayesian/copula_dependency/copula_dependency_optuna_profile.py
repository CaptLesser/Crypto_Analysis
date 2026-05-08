from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.bayesian.copula_dependency.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return dict(MODEL_PARAMS)


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    use_tail = trial.suggest_categorical("use_tail", [False, True])
    params = {"dependence_regularization": trial.suggest_float("dependence_regularization", 1e-6, 1e-2, log=True), "factor_weight_scale": trial.suggest_float("factor_weight_scale", 0.25, 2.0), "marginal_vol_scale": trial.suggest_float("marginal_vol_scale", 0.5, 2.0), "tail_df": None}
    if use_tail:
        params["tail_df"] = trial.suggest_float("tail_df", 3.0, 12.0)
    return params
