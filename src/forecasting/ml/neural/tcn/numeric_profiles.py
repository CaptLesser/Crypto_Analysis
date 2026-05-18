from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import neural_tcn_predict
from src.forecasting.ml.neural.shared.neural_numeric_production_profiles import explicit_combo_specs, resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "neural_tcn"
MODEL_ID = "neural_tcn"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "NeuralTS_TCN"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_NEURALTS_TCN_ROOT"
DEFAULT_INTERVALS = (5, 30, 60, 240, 720, 1440)
DEFAULT_HORIZON_MATRIX = {
    5: (30, 240, 720),
    30: (30, 240, 720),
    60: (240, 720, 1440),
    240: (720, 1440, 4320),
    720: (1440, 4320, 10080),
    1440: (4320, 10080, 20160),
}
DEFAULT_HORIZONS = (30, 240, 720, 1440, 4320, 10080, 20160)
DEFAULT_TASKS = ("log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency")
MODEL_PARAMS = {"seq_len_floor": 96, "channel_width": 32, "dilation_depth": 6, "kernel_size": 3, "dropout": 0.10}
NEEDS_DYNAMIC_FEATURES = True
DYNAMIC_FEATURE_CANDIDATES = (
    "log_return",
    "ret_std_20",
    "atr_14",
    "rsi_14",
    "macd_12_26_9",
    "zscore_20",
    "range_efficiency_20",
    "range_efficiency_50",
    "range_efficiency_100",
    "volume_zscore_20",
)
STAGE1_MODE = "full"
STAGE1_FEATURE_BLOCKS = {
    "target_history_core": ["target_history", "lag_1", "lag_2", "lag_3", "lag_6"],
    "range_vol_block": ["realized_vol", "true_range", "range_efficiency"],
    "path_extrema_block": ["max_drawdown", "max_runup"],
    "regime_context_block": ["short_term_trend", "volatility_regime"],
}
STAGE1_FORMULATION_OPTIONS = {
    "input_schema": ["causal_multivariate_sequence"],
    "lookback_design": ["dilated_receptive_field"],
}
RUNTIME_PARAMS = {"fit_days": 180, "stage0_fit_days_default": 180, "batch_size": 64, "stage0_batch_size_default": 64, "epochs": 20, "stage0_epochs_default": 20, "lr": 0.001, "seq_len_default": 256, "seq_len_1m": 512, "seq_len_5m": 256, "stage0_seq_len_default": 256, "stage0_seq_len_1m": 512, "stage0_seq_len_5m": 256, "refit_cadence": "auto"}


def predict_fn(*, y_hist: np.ndarray, horizon_bars: int, quantiles: Sequence[float], seq_len: int, seed: int, model_params: Dict[str, Any] | None = None, x_hist: np.ndarray | None = None, x_last: np.ndarray | None = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    seq_len = max(int(seq_len), int(params["seq_len_floor"]))
    return neural_tcn_predict(
        y_hist=y_hist,
        horizon_bars=int(horizon_bars),
        quantiles=quantiles,
        seq_len=seq_len,
        seed=seed,
        channel_width=int(params["channel_width"]),
        dilation_depth=int(params["dilation_depth"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
        X_hist=x_hist,
        x_last=x_last,
    )


def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
    return [predict_fn(**item) for item in origin_batch]


FALLBACK_COMBOS = tuple(explicit_combo_specs(DEFAULT_HORIZON_MATRIX, DEFAULT_TASKS))
PRODUCTION_DEFAULT_COMBOS = tuple(
    _resolve_default_combo_specs(MODEL_ID, DEFAULT_INTERVALS, DEFAULT_HORIZONS, DEFAULT_TASKS, fallback_combos=FALLBACK_COMBOS)
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
