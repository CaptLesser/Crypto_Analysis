from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, run_with_float_dtype_retry
from src.forecasting.ml.tabular.shared.adapter_utils import (
    missing_regressor_bundle,
    prediction_block_from_bundle,
    prepare_regression_fit_inputs,
    regression_bundle_from_predictions,
)
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import RegressionBundle

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None  # pragma: no cover


def fit_regressor(x: np.ndarray, y: np.ndarray, regressor_params: Optional[Dict[str, Any]] = None) -> RegressionBundle:
    def _fit_for_dtype(dtype: Any) -> RegressionBundle:
        prepared = prepare_regression_fit_inputs(x, y, dtype=dtype)
        if prepared.fallback_bundle is not None:
            return prepared.fallback_bundle
        x_fit = prepared.x_fit
        y_fit = prepared.y_fit
        if LGBMRegressor is None:
            return missing_regressor_bundle(x_fit, y_fit)
        mean_model = LGBMRegressor(**dict(regressor_params or {}))
        mean_model.fit(x_fit, y_fit)
        pred_input: Any = x_fit
        feature_names = getattr(mean_model, 'feature_names_in_', None)
        if feature_names is not None:
            try:
                names = [str(name) for name in list(feature_names)]
                if x_fit.ndim == 2 and x_fit.shape[1] == len(names):
                    pred_input = pd.DataFrame(x_fit, columns=names)
            except Exception:
                pred_input = x_fit
        pred = np.asarray(mean_model.predict(pred_input), dtype=DEFAULT_FLOAT_DTYPE)
        return regression_bundle_from_predictions(mean_model, y_fit, pred)

    return run_with_float_dtype_retry(_fit_for_dtype)


def predict_block(bundle: RegressionBundle, x_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _predict_for_dtype(dtype: Any) -> Tuple[np.ndarray, np.ndarray]:
        def _predict(mean_model: Any, arr: np.ndarray) -> np.ndarray:
            x_input: Any = arr
            feature_names = getattr(mean_model, 'feature_names_in_', None)
            if feature_names is not None:
                try:
                    names = [str(name) for name in list(feature_names)]
                    if arr.ndim == 2 and arr.shape[1] == len(names):
                        x_input = pd.DataFrame(arr, columns=names)
                except Exception:
                    x_input = arr
            return mean_model.predict(x_input)

        return prediction_block_from_bundle(bundle, x_block, dtype=dtype, predict_fn=_predict)

    return run_with_float_dtype_retry(_predict_for_dtype)
