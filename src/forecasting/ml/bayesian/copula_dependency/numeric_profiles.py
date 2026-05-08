from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import bayes_copula_dependency_predict
from src.forecasting.ml.bayesian.shared.bayesian_numeric_production_profiles import resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "bayes_copula_dependency"
MODEL_ID = "bayes_copula_dependency"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "Stats_Bayes_CopulaDependency"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_BAYES_COPULA_ROOT"
DEFAULT_INTERVALS = (30, 60, 240, 1440)
DEFAULT_HORIZONS = (240, 1440, 4320, 10080)
DEFAULT_TASKS = ("log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency")
USE_SEASONALITY = False
NEEDS_DYNAMIC_FEATURES = False
NEEDS_FACTOR_CACHE = True
DYNAMIC_FEATURE_CANDIDATES: Tuple[str, ...] = ()
MODEL_PARAMS: Dict[str, Any] = {
    "dependence_regularization": 1e-5,
    "factor_weight_scale": 1.0,
    "marginal_vol_scale": 1.0,
    "tail_df": None,
}


def predict_fn(*, y_hist: Sequence[float], horizon_bars: int, quantiles: Sequence[float], seasonal_period_bars: Optional[int], seed: int, model_params: Optional[Dict[str, Any]] = None, x_hist: Optional[np.ndarray] = None, x_last: Optional[np.ndarray] = None, factor_hist: Optional[np.ndarray] = None, factor_last: Optional[float] = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    if factor_hist is None or factor_last is None:
        raise RuntimeError("missing_factor")
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    return bayes_copula_dependency_predict(y_hist=y_hist, factor_hist=factor_hist, factor_last=float(factor_last), quantiles=quantiles, seed=seed, dependence_regularization=float(params["dependence_regularization"]), factor_weight_scale=float(params["factor_weight_scale"]), marginal_vol_scale=float(params["marginal_vol_scale"]), tail_df=(None if params.get("tail_df") is None else float(params["tail_df"])))


def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
    return [predict_fn(**item) for item in origin_batch]

STAGE1_MODE = "slim"
STAGE1_FEATURE_BLOCKS = {}
STAGE1_FORMULATION_OPTIONS = {
    "dependency_schema": ["market_factor_mean_link"],
    "marginal_transform": ["rank_uniform", "normal_score"],
    "factor_schema": ["peer_mean_factor", "peer_median_factor"],
}

RUNTIME_PARAMS: Dict[str, Any] = {"fit_days": 365, "refit_cadence": "auto"}


PRODUCTION_DEFAULT_COMBOS = tuple(
    _resolve_default_combo_specs(MODEL_ID, DEFAULT_INTERVALS, DEFAULT_HORIZONS, DEFAULT_TASKS)
)


def resolve_default_combo_specs() -> list[tuple[int, int, str]]:
    return list(PRODUCTION_DEFAULT_COMBOS)


def resolve_model_params(*, task: str, interval_minutes: int | None = None, horizon_minutes: int | None = None) -> Dict[str, Any]:
    return _resolve_model_params(
        MODEL_ID,
        MODEL_PARAMS,
        interval_minutes=interval_minutes,
        horizon_minutes=horizon_minutes,
        task=task,
    )

