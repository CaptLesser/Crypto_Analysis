from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import bayes_dynamic_regression_shrinkage_predict
from src.forecasting.ml.bayesian.shared.bayesian_numeric_production_profiles import resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "bayes_dynamic_regression_shrinkage"
MODEL_ID = "bayes_dynamic_regression_shrinkage"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "Stats_Bayes_DynamicRegression"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_BAYES_DYNREG_ROOT"
DEFAULT_INTERVALS = (30, 60, 240, 1440)
DEFAULT_HORIZONS = (240, 1440, 4320, 10080)
DEFAULT_TASKS = ("log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency")
USE_SEASONALITY = False
NEEDS_DYNAMIC_FEATURES = True
NEEDS_FACTOR_CACHE = False
DYNAMIC_FEATURE_CANDIDATES = (
    "log_return", "ret_std_14", "ret_std_30", "ret_std_60", "ret_std_120", "atr_14", "atr_30",
    "rsi_14", "macd", "macd_signal", "macd_hist", "zscore_30", "zscore_60", "ema_gap_12_26",
    "range_efficiency_30", "volume_zscore_30",
)
MODEL_PARAMS: Dict[str, Any] = {
    "global_shrinkage": 1.0,
    "slab_scale": 1.0,
    "feature_corr_weight": 1.0,
    "coefficient_drift_scale": 0.0,
}


def predict_fn(*, y_hist: Sequence[float], horizon_bars: int, quantiles: Sequence[float], seasonal_period_bars: Optional[int], seed: int, model_params: Optional[Dict[str, Any]] = None, x_hist: Optional[np.ndarray] = None, x_last: Optional[np.ndarray] = None, factor_hist: Optional[np.ndarray] = None, factor_last: Optional[float] = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    if x_hist is None or x_last is None:
        raise RuntimeError("missing_dynamic_features")
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    return bayes_dynamic_regression_shrinkage_predict(y_hist=y_hist, X_hist=x_hist, x_last=x_last, quantiles=quantiles, seed=seed, global_shrinkage=float(params["global_shrinkage"]), slab_scale=float(params["slab_scale"]), feature_corr_weight=float(params["feature_corr_weight"]), coefficient_drift_scale=float(params["coefficient_drift_scale"]))


def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
    return [predict_fn(**item) for item in origin_batch]

STAGE1_MODE = "full"
STAGE1_FEATURE_BLOCKS = {
    "return_context": ["log_return", "zscore_30", "zscore_60", "ema_gap_12_26"],
    "volatility_context": ["ret_std_14", "ret_std_30", "ret_std_60", "ret_std_120", "atr_14", "atr_30"],
    "momentum_context": ["rsi_14", "macd", "macd_signal", "macd_hist"],
    "efficiency_volume_context": ["range_efficiency_30", "volume_zscore_30"],
}
STAGE1_FORMULATION_OPTIONS = {
    "predictor_dynamics": ["dynamic_feature_block", "static_feature_block"],
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

