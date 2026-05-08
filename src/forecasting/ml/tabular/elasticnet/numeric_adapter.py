from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, run_with_float_dtype_retry
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import ConstantRegressor, RegressionBundle

try:
    from sklearn.linear_model import ElasticNet as ElasticNetRegressor  # type: ignore
    from sklearn.exceptions import ConvergenceWarning  # type: ignore
except Exception:
    ElasticNetRegressor = None  # pragma: no cover
    ConvergenceWarning = Warning  # type: ignore

STAGE1_REGRESSOR_CLASS = ElasticNetRegressor


def _fit_elasticnet_with_warning_capture(x_fit: np.ndarray, y_fit: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, list[str]]:
    model = STAGE1_REGRESSOR_CLASS(**dict(params))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, y_fit)
    warning_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    return model, warning_messages


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
        params = dict(regressor_params or {})
        mean_model, convergence_warnings = _fit_elasticnet_with_warning_capture(x_fit, y_fit, params)
        retry_count = 0
        initial_warning_count = int(len(convergence_warnings))
        if convergence_warnings:
            retry_params = dict(params)
            current_max_iter = int(retry_params.get("max_iter", 1000) or 1000)
            retry_cap = int(os.getenv("EN_NUMERIC_CONVERGENCE_RETRY_MAX_ITER", "50000"))
            retry_params["max_iter"] = min(max(int(current_max_iter) * 4, int(current_max_iter) + 5000), int(retry_cap))
            if int(retry_params["max_iter"]) > int(current_max_iter):
                retry_count = 1
                retry_model, retry_warnings = _fit_elasticnet_with_warning_capture(x_fit, y_fit, retry_params)
                if len(retry_warnings) <= len(convergence_warnings):
                    mean_model = retry_model
                    params = retry_params
                    convergence_warnings = retry_warnings
        pred = np.asarray(mean_model.predict(x_fit), dtype=DEFAULT_FLOAT_DTYPE)
        resid_std = float(np.nanstd(y_fit - pred))
        return RegressionBundle(
            mean_model=mean_model,
            residual_std=max(0.0, resid_std),
            diagnostics={
                "convergence_warning_count": int(len(convergence_warnings)),
                "initial_convergence_warning_count": int(initial_warning_count),
                "convergence_retry_count": int(retry_count),
                "convergence_retry_resolved_count": int(initial_warning_count > 0 and len(convergence_warnings) == 0),
                "effective_max_iter": int(params.get("max_iter", 0) or 0),
            },
        )

    return run_with_float_dtype_retry(_fit_for_dtype)


def predict_block(bundle: RegressionBundle, x_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _predict_for_dtype(dtype: Any) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x_block, dtype=dtype)
        mean_v = np.asarray(bundle.mean_model.predict(arr), dtype=DEFAULT_FLOAT_DTYPE)
        std_v = np.full((len(arr),), float(max(0.0, bundle.residual_std)), dtype=DEFAULT_FLOAT_DTYPE)
        return mean_v, std_v

    return run_with_float_dtype_retry(_predict_for_dtype)
