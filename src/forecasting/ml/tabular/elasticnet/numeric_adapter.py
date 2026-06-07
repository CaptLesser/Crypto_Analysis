from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, run_with_float_dtype_retry
from src.forecasting.ml.tabular.shared.adapter_utils import (
    missing_regressor_bundle,
    prediction_block_from_bundle,
    prepare_regression_fit_inputs,
    regression_bundle_from_predictions,
)
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import RegressionBundle

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
        prepared = prepare_regression_fit_inputs(x, y, dtype=dtype)
        if prepared.fallback_bundle is not None:
            return prepared.fallback_bundle
        x_fit = prepared.x_fit
        y_fit = prepared.y_fit
        if STAGE1_REGRESSOR_CLASS is None:
            return missing_regressor_bundle(x_fit, y_fit)
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
        return regression_bundle_from_predictions(
            mean_model,
            y_fit,
            pred,
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
        return prediction_block_from_bundle(bundle, x_block, dtype=dtype)

    return run_with_float_dtype_retry(_predict_for_dtype)
