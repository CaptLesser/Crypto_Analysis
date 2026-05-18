from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import bayes_dlm_tvp_predict
from src.forecasting.ml.bayesian.shared.bayesian_numeric_production_profiles import resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "bayes_dlm_tvp"
MODEL_ID = "bayes_dlm_tvp"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "Stats_Bayes_DLM_TVP"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_BAYES_DLM_ROOT"
DEFAULT_INTERVALS = (30, 60, 240, 1440)
DEFAULT_HORIZONS = (240, 1440, 4320, 10080)
DEFAULT_TASKS = ("log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency")
USE_SEASONALITY = True
NEEDS_DYNAMIC_FEATURES = True
NEEDS_FACTOR_CACHE = False
DYNAMIC_FEATURE_CANDIDATES = (
    "log_return",
    "ret_std_20",
    "atr_14",
    "atr_pct_14",
    "rsi_14",
    "macd_12_26_9",
    "zscore_20",
    "range_efficiency_20",
    "range_efficiency_50",
    "volume_zscore_20",
)
MODEL_PARAMS: Dict[str, Any] = {
    "level_smoothing": 0.08,
    "trend_smoothing": 0.08,
    "seasonal_strength": 1.0,
    "observation_scale": 1.0,
    "exogenous_scale": 0.35,
}


def predict_fn(*, y_hist: Sequence[float], horizon_bars: int, quantiles: Sequence[float], seasonal_period_bars: Optional[int], seed: int, model_params: Optional[Dict[str, Any]] = None, x_hist: Optional[np.ndarray] = None, x_last: Optional[np.ndarray] = None, factor_hist: Optional[np.ndarray] = None, factor_last: Optional[float] = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    return bayes_dlm_tvp_predict(y_hist=y_hist, horizon_bars=int(horizon_bars), quantiles=quantiles, seasonal_period_bars=seasonal_period_bars, seed=seed, level_smoothing=float(params["level_smoothing"]), trend_smoothing=float(params["trend_smoothing"]), seasonal_strength=float(params["seasonal_strength"]), observation_scale=float(params["observation_scale"]), X_hist=x_hist, x_last=x_last, exogenous_scale=float(params["exogenous_scale"]))


def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
    return [predict_fn(**item) for item in origin_batch]

STAGE1_MODE = "full"
STAGE1_FEATURE_BLOCKS = {
    "state_history_core": ["target_history", "level_state", "trend_state"],
    "seasonal_state_block": ["seasonality_state", "seasonal_phase"],
    "recent_lag_block": ["lag_1", "lag_2", "lag_3", "lag_6"],
}
STAGE1_FORMULATION_OPTIONS = {
    "seasonality_mode": ["additive_seasonal_state"],
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

