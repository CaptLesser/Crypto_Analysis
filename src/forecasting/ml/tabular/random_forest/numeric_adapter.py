from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import warnings

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
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
except Exception:
    RandomForestRegressor = None  # pragma: no cover

STAGE1_REGRESSOR_CLASS = RandomForestRegressor


_SKLEARN_PARALLEL_DELAYED_WARNING = "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"


def fit_regressor(x: np.ndarray, y: np.ndarray, regressor_params: Optional[Dict[str, Any]] = None) -> RegressionBundle:
    def _fit_for_dtype(dtype: Any) -> RegressionBundle:
        prepared = prepare_regression_fit_inputs(x, y, dtype=dtype)
        if prepared.fallback_bundle is not None:
            return prepared.fallback_bundle
        x_fit = prepared.x_fit
        y_fit = prepared.y_fit
        if STAGE1_REGRESSOR_CLASS is None:
            return missing_regressor_bundle(x_fit, y_fit)
        mean_model = STAGE1_REGRESSOR_CLASS(**dict(regressor_params or {}))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_SKLEARN_PARALLEL_DELAYED_WARNING, category=UserWarning)
            mean_model.fit(x_fit, y_fit)
            pred = np.asarray(mean_model.predict(x_fit), dtype=DEFAULT_FLOAT_DTYPE)
        return regression_bundle_from_predictions(mean_model, y_fit, pred)

    return run_with_float_dtype_retry(_fit_for_dtype)


def predict_block(bundle: RegressionBundle, x_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _predict_for_dtype(dtype: Any) -> Tuple[np.ndarray, np.ndarray]:
        def _predict(mean_model: Any, arr: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_SKLEARN_PARALLEL_DELAYED_WARNING, category=UserWarning)
                return mean_model.predict(arr)

        return prediction_block_from_bundle(bundle, x_block, dtype=dtype, predict_fn=_predict)

    return run_with_float_dtype_retry(_predict_for_dtype)
