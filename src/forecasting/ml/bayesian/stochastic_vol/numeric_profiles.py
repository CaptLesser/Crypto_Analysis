from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import bayes_stochastic_vol_predict
from src.forecasting.ml.bayesian.shared.bayesian_numeric_production_profiles import resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "bayes_stochastic_vol"
MODEL_ID = "bayes_stochastic_vol"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "Stats_Bayes_StochasticVol"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_BAYES_SV_ROOT"
DEFAULT_INTERVALS = (30, 60, 240, 1440)
DEFAULT_HORIZONS = (240, 1440, 4320, 10080)
DEFAULT_TASKS = ("log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency")
USE_SEASONALITY = False
NEEDS_DYNAMIC_FEATURES = False
NEEDS_FACTOR_CACHE = False
DYNAMIC_FEATURE_CANDIDATES: Tuple[str, ...] = ()
MODEL_PARAMS: Dict[str, Any] = {
    "persistence": 0.94,
    "horizon_vol_scale": 0.15,
    "innovation_scale": 1.0,
    "heavy_tail_df": None,
}


def predict_fn(*, y_hist: Sequence[float], horizon_bars: int, quantiles: Sequence[float], seasonal_period_bars: Optional[int], seed: int, model_params: Optional[Dict[str, Any]] = None, x_hist: Optional[np.ndarray] = None, x_last: Optional[np.ndarray] = None, factor_hist: Optional[np.ndarray] = None, factor_last: Optional[float] = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    return bayes_stochastic_vol_predict(y_hist=y_hist, horizon_bars=int(horizon_bars), quantiles=quantiles, seed=seed, persistence=float(params["persistence"]), horizon_vol_scale=float(params["horizon_vol_scale"]), innovation_scale=float(params["innovation_scale"]), heavy_tail_df=(None if params.get("heavy_tail_df") is None else float(params["heavy_tail_df"])))


def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
    return [predict_fn(**item) for item in origin_batch]

STAGE1_MODE = "slim"
STAGE1_FEATURE_BLOCKS = {}
STAGE1_FORMULATION_OPTIONS = {
    "target_schema": ["realized_volatility_path", "true_range_path", "log_return_scale"],
    "observation_transform": ["raw_scale", "log_abs_scale"],
    "asymmetry_mode": ["symmetric"],
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

