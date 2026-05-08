from __future__ import annotations

from typing import Any, Dict

import optuna

from src.forecasting.ml.bayesian.stochastic_vol.numeric_profiles import MODEL_PARAMS


def resolve_baseline_params(*, task: str, model_threads: int, combo: Any = None) -> Dict[str, Any]:
    return dict(MODEL_PARAMS)


def suggest_trial_params(trial: optuna.Trial, combo: Any) -> Dict[str, Any]:
    use_heavy_tail = trial.suggest_categorical("use_heavy_tail", [False, True])
    params = {"persistence": trial.suggest_float("persistence", 0.75, 0.99), "horizon_vol_scale": trial.suggest_float("horizon_vol_scale", 0.05, 0.35), "innovation_scale": trial.suggest_float("innovation_scale", 0.5, 2.0), "heavy_tail_df": None}
    if use_heavy_tail:
        params["heavy_tail_df"] = trial.suggest_float("heavy_tail_df", 3.0, 12.0)
    return params
