from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import bayes_tail_risk_predict
from src.forecasting.common.forecast_family_core import monotonic_quantiles
from src.forecasting.ml.bayesian.shared.bayesian_numeric_production_profiles import resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "bayes_tail_risk"
MODEL_ID = "bayes_tail_risk"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "Stats_Bayes_TailRisk"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_BAYES_TAIL_ROOT"
DEFAULT_INTERVALS = (30, 60, 240, 1440)
DEFAULT_HORIZONS = (240, 1440, 4320, 10080)
DEFAULT_TASKS = ("log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency")
USE_SEASONALITY = False
NEEDS_DYNAMIC_FEATURES = False
NEEDS_FACTOR_CACHE = False
DYNAMIC_FEATURE_CANDIDATES: Tuple[str, ...] = ()
MODEL_PARAMS: Dict[str, Any] = {
    "threshold_quantile": 0.1,
    "min_tail_points": 8,
    "shape_scale": 1.0,
    "tail_scale_multiplier": 1.0,
    "model_two_tails": True,
    "center_tail_blend": 0.0,
}


def _tail_adjusted_center_quantile(
    y_hist: Sequence[float],
    *,
    threshold_quantile: float,
    min_tail_points: int,
    center_tail_blend: float,
) -> Optional[float]:
    y = np.asarray(list(y_hist), dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 32:
        return None
    blend = float(center_tail_blend)
    if abs(blend) < 1e-9:
        return float(np.quantile(y, 0.5))

    qcut = min(0.25, max(0.01, float(threshold_quantile)))
    lo_thr = float(np.quantile(y, qcut))
    hi_thr = float(np.quantile(y, 1.0 - qcut))
    lo_excess = lo_thr - y[y < lo_thr]
    hi_excess = y[y > hi_thr] - hi_thr
    min_points = max(4, int(min_tail_points))

    lo_weight = min(1.0, float(lo_excess.size) / float(min_points)) if min_points > 0 else 1.0
    hi_weight = min(1.0, float(hi_excess.size) / float(min_points)) if min_points > 0 else 1.0
    lo_pressure = (float(np.mean(lo_excess)) if lo_excess.size else 0.0) * float(lo_weight)
    hi_pressure = (float(np.mean(hi_excess)) if hi_excess.size else 0.0) * float(hi_weight)
    tail_balance = hi_pressure - lo_pressure

    center = float(np.quantile(y, 0.5))
    iqr_scale = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))
    std_scale = float(np.std(y))
    scale = max(1e-8, iqr_scale, std_scale)
    capped_shift = float(np.clip(blend * tail_balance, -0.35 * scale, 0.35 * scale))
    return center + capped_shift


def predict_fn(*, y_hist: Sequence[float], horizon_bars: int, quantiles: Sequence[float], seasonal_period_bars: Optional[int], seed: int, model_params: Optional[Dict[str, Any]] = None, x_hist: Optional[np.ndarray] = None, x_last: Optional[np.ndarray] = None, factor_hist: Optional[np.ndarray] = None, factor_last: Optional[float] = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    qvals, meta = bayes_tail_risk_predict(
        y_hist=y_hist,
        quantiles=quantiles,
        threshold_quantile=float(params["threshold_quantile"]),
        min_tail_points=int(params["min_tail_points"]),
        shape_scale=float(params["shape_scale"]),
        tail_scale_multiplier=float(params["tail_scale_multiplier"]),
        model_two_tails=bool(params["model_two_tails"]),
    )
    center_q = _tail_adjusted_center_quantile(
        y_hist,
        threshold_quantile=float(params["threshold_quantile"]),
        min_tail_points=int(params["min_tail_points"]),
        center_tail_blend=float(params.get("center_tail_blend", 0.0)),
    )
    if center_q is not None and any(abs(float(q) - 0.5) < 1e-9 for q in quantiles):
        qvals[0.5] = float(center_q)
        qvals = monotonic_quantiles(qvals, quantiles)
    if isinstance(meta, dict):
        meta = {
            **meta,
            "center_tail_blend": float(params.get("center_tail_blend", 0.0)),
            "center_quantile": (float(qvals[0.5]) if 0.5 in qvals else None),
        }
    return qvals, meta


def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
    return [predict_fn(**item) for item in origin_batch]

STAGE1_MODE = "slim"
STAGE1_FEATURE_BLOCKS = {}
STAGE1_FORMULATION_OPTIONS = {
    "tail_schema": ["two_tail_evt", "lower_tail_evt"],
    "threshold_family": ["fixed_quantile_threshold", "adaptive_quantile_threshold"],
    "exceedance_rule": ["rolling_tail_window"],
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

