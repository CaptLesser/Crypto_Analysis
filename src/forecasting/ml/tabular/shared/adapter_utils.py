from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import ConstantRegressor, RegressionBundle


@dataclass(frozen=True)
class PreparedRegressionFit:
    x_fit: np.ndarray
    y_fit: np.ndarray
    fallback_bundle: Optional[RegressionBundle] = None


def constant_regression_bundle(value: float, *, residual_std: float = 0.0) -> RegressionBundle:
    return RegressionBundle(mean_model=ConstantRegressor(float(value)), residual_std=float(max(0.0, residual_std)))


def prepare_regression_fit_inputs(x: np.ndarray, y: np.ndarray, *, dtype: Any) -> PreparedRegressionFit:
    x_arr = np.asarray(x, dtype=dtype)
    y_arr = np.asarray(y, dtype=dtype)
    if x_arr.ndim != 2 or x_arr.shape[1] == 0 or len(y_arr) < 2:
        base = float(np.nanmean(y_arr)) if len(y_arr) else 0.0
        return PreparedRegressionFit(x_fit=x_arr, y_fit=y_arr, fallback_bundle=constant_regression_bundle(base))

    finite_mask = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
    if not np.any(finite_mask):
        return PreparedRegressionFit(x_fit=x_arr[:0], y_fit=y_arr[:0], fallback_bundle=constant_regression_bundle(0.0))

    x_fit = x_arr[finite_mask]
    y_fit = y_arr[finite_mask]
    if x_fit.ndim != 2 or x_fit.shape[1] == 0 or len(y_fit) < 2:
        base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
        return PreparedRegressionFit(x_fit=x_fit, y_fit=y_fit, fallback_bundle=constant_regression_bundle(base))
    if not np.isfinite(y_fit).all() or float(np.nanstd(y_fit)) <= 0.0:
        base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
        return PreparedRegressionFit(x_fit=x_fit, y_fit=y_fit, fallback_bundle=constant_regression_bundle(base))

    return PreparedRegressionFit(x_fit=x_fit, y_fit=y_fit)


def missing_regressor_bundle(x_fit: np.ndarray, y_fit: np.ndarray) -> RegressionBundle:
    base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
    model = ConstantRegressor(base)
    resid = y_fit - model.predict(x_fit)
    return constant_regression_bundle(base, residual_std=float(np.nanstd(resid)))


def regression_bundle_from_predictions(
    mean_model: Any,
    y_fit: np.ndarray,
    pred: np.ndarray,
    *,
    diagnostics: Optional[dict[str, Any]] = None,
) -> RegressionBundle:
    pred_arr = np.asarray(pred, dtype=DEFAULT_FLOAT_DTYPE)
    resid_std = float(np.nanstd(y_fit - pred_arr))
    return RegressionBundle(mean_model=mean_model, residual_std=max(0.0, resid_std), diagnostics=diagnostics or {})


def prediction_block_from_bundle(
    bundle: RegressionBundle,
    x_block: np.ndarray,
    *,
    dtype: Any,
    predict_fn: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x_block, dtype=dtype)
    if predict_fn is None:
        mean_v = np.asarray(bundle.mean_model.predict(arr), dtype=DEFAULT_FLOAT_DTYPE)
    else:
        mean_v = np.asarray(predict_fn(bundle.mean_model, arr), dtype=DEFAULT_FLOAT_DTYPE)
    std_v = np.full((len(arr),), float(max(0.0, bundle.residual_std)), dtype=DEFAULT_FLOAT_DTYPE)
    return mean_v, std_v
