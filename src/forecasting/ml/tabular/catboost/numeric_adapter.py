from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, run_with_float_dtype_retry
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import ConstantRegressor, RegressionBundle

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None  # pragma: no cover

STAGE1_REGRESSOR_CLASS = CatBoostRegressor


def fit_regressor(x: np.ndarray, y: np.ndarray, regressor_params: Optional[Dict[str, Any]] = None) -> RegressionBundle:
    def _fit_for_dtype(dtype: Any) -> RegressionBundle:
        x_arr = np.asarray(x, dtype=dtype)
        y_arr = np.asarray(y, dtype=dtype)
        if x_arr.ndim != 2 or x_arr.shape[1] == 0 or len(y_arr) < 2:
            base = float(np.nanmean(y_arr)) if len(y_arr) else 0.0
            return RegressionBundle(mean_model=ConstantRegressor(base), residual_std=0.0)
        finite_mask = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
        if not np.any(finite_mask):
            return RegressionBundle(mean_model=ConstantRegressor(0.0), residual_std=0.0)
        x_fit = x_arr[finite_mask]
        y_fit = y_arr[finite_mask]
        if x_fit.ndim != 2 or x_fit.shape[1] == 0 or len(y_fit) < 2:
            base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
            return RegressionBundle(mean_model=ConstantRegressor(base), residual_std=0.0)
        if not np.isfinite(y_fit).all() or float(np.nanstd(y_fit)) <= 0.0:
            base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
            return RegressionBundle(mean_model=ConstantRegressor(base), residual_std=0.0)
        if STAGE1_REGRESSOR_CLASS is None:
            base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
            model = ConstantRegressor(base)
            resid = y_fit - model.predict(x_fit)
            return RegressionBundle(mean_model=model, residual_std=float(np.nanstd(resid)))
        mean_model = STAGE1_REGRESSOR_CLASS(**dict(regressor_params or {}))
        mean_model.fit(x_fit, y_fit)
        pred = np.asarray(mean_model.predict(x_fit), dtype=DEFAULT_FLOAT_DTYPE)
        resid_std = float(np.nanstd(y_fit - pred))
        return RegressionBundle(mean_model=mean_model, residual_std=max(0.0, resid_std))

    return run_with_float_dtype_retry(_fit_for_dtype)


def predict_block(bundle: RegressionBundle, x_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _predict_for_dtype(dtype: Any) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x_block, dtype=dtype)
        mean_v = np.asarray(bundle.mean_model.predict(arr), dtype=DEFAULT_FLOAT_DTYPE)
        std_v = np.full((len(arr),), float(max(0.0, bundle.residual_std)), dtype=DEFAULT_FLOAT_DTYPE)
        return mean_v, std_v

    return run_with_float_dtype_retry(_predict_for_dtype)
