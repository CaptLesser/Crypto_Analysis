from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.forecasting.common.model_zoo_forecast import neural_nbeats_predict
from src.forecasting.ml.neural.shared.neural_numeric_production_profiles import explicit_combo_specs, resolve_default_combo_specs as _resolve_default_combo_specs, resolve_model_params as _resolve_model_params

MODULE_TAG = "neural_nbeats"
MODEL_ID = "neural_nbeats"
MODEL_VERSION = "1.0.0"
FAMILY_ROOT_NAME = "NeuralTS_NBEATS"
FAMILY_ROOT_ENV = "PIPELINE_PARQUET_NEURALTS_NBEATS_ROOT"
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
MODEL_PARAMS = {"seq_len_floor": 128, "num_stacks": 4, "num_blocks": 2, "layer_width": 128, "generic_architecture": True}
NEEDS_DYNAMIC_FEATURES = False
DYNAMIC_FEATURE_CANDIDATES: tuple[str, ...] = ()
STAGE1_MODE = "slim"
STAGE1_FEATURE_BLOCKS = {
    "target_history_only": ["target_history"],
}
STAGE1_FORMULATION_OPTIONS = {
    "input_schema": ["univariate_backcast_forecast"],
    "lookback_design": ["basis_aligned_lookback"],
}
RUNTIME_PARAMS = {"fit_days": 180, "stage0_fit_days_default": 180, "batch_size": 64, "stage0_batch_size_default": 64, "epochs": 20, "stage0_epochs_default": 20, "lr": 0.001, "seq_len_default": 256, "seq_len_1m": 512, "seq_len_5m": 256, "stage0_seq_len_default": 256, "stage0_seq_len_1m": 512, "stage0_seq_len_5m": 256, "refit_cadence": "auto"}


def predict_fn(*, y_hist: np.ndarray, horizon_bars: int, quantiles: Sequence[float], seq_len: int, seed: int, model_params: Dict[str, Any] | None = None, x_hist: np.ndarray | None = None, x_last: np.ndarray | None = None) -> Tuple[Dict[float, float], Dict[str, Any]]:
    params = dict(MODEL_PARAMS)
    params.update(model_params or {})
    seq_len = max(int(seq_len), int(params["seq_len_floor"]))
    return neural_nbeats_predict(
        y_hist=y_hist,
        horizon_bars=int(horizon_bars),
        quantiles=quantiles,
        seq_len=seq_len,
        seed=seed,
        num_stacks=int(params["num_stacks"]),
        num_blocks=int(params["num_blocks"]),
        layer_width=int(params["layer_width"]),
        generic_architecture=bool(params["generic_architecture"]),
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
