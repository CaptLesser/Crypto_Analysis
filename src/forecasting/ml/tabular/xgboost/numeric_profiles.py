from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from src.features.numeric_forecast_profiles import (
    TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS,
    TABULAR_NUMERIC_TASK_HORIZON_MATRIX,
    normalize_refit_cadence,
)
from src.forecasting.common.runtime_config import get_model_threads
from src.forecasting.ml.tabular.shared.tabular_numeric_production_defaults import (
    XGBOOST_PRODUCTION_COMBO_REGRESSOR_PROFILES,
    XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS,
    XGBOOST_TUNED_TASK_REGRESSOR_PROFILES,
)

DEFAULT_INTERVALS = sorted({int(interval) for interval, _, _, _ in XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS})
DEFAULT_HORIZON_MINUTES = sorted({int(horizon) for _, horizon, _, _ in XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS})
PRODUCTION_DEFAULT_COMBO_WINDOWS = [
    (int(interval), int(horizon), str(task), int(months))
    for interval, horizon, task, months in XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS
]
PRODUCTION_DEFAULT_COMBOS = [(int(interval), int(horizon), str(task)) for interval, horizon, task, _ in PRODUCTION_DEFAULT_COMBO_WINDOWS]
PRODUCTION_DEFAULT_REFIT_POLICY: Dict[Tuple[int, int, str], str] = {
    (int(interval), int(horizon), str(task)): ("weekly" if int(interval) <= 60 else "biweekly" if int(interval) <= 240 else "monthly")
    for interval, horizon, task, _ in PRODUCTION_DEFAULT_COMBO_WINDOWS
}
NUMERIC_TASKS = ["log_return", "realized_vol", "true_range", "max_drawdown", "max_runup", "range_efficiency"]
TASK_SHORT = {
    "log_return": "logret",
    "realized_vol": "rv",
    "true_range": "tr",
    "max_drawdown": "mdd",
    "max_runup": "mru",
    "range_efficiency": "reff",
}
TASK_LABEL = {
    "log_return": "future_log_return",
    "realized_vol": "future_realized_vol",
    "true_range": "future_true_range",
    "max_drawdown": "future_max_drawdown",
    "max_runup": "future_max_runup",
    "range_efficiency": "future_range_efficiency",
}
FUTURE_LABEL_COLUMNS = list(TASK_LABEL.values()) + ["future_direction"]
ACTIVE_TASK_HORIZON_MATRIX: Dict[str, List[int]] = {str(task): list(horizons) for task, horizons in TABULAR_NUMERIC_TASK_HORIZON_MATRIX.items()}
PAIR_TRAINING_WINDOW_MONTHS: Dict[Tuple[str, int], int] = {(str(task), int(hm)): int(months) for (task, hm), months in TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS.items()}

BASE_REG_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": int(os.getenv("XGB_NUMERIC_RANDOM_STATE", "17")),
    "tree_method": os.getenv("XGB_NUMERIC_TREE_METHOD", "hist"),
}
LEGACY_TASK_REGRESSOR_PROFILE = {
    "n_estimators": int(os.getenv("XGB_NUMERIC_N_ESTIMATORS", "220")),
    "max_depth": int(os.getenv("XGB_NUMERIC_MAX_DEPTH", "4")),
    "learning_rate": float(os.getenv("XGB_NUMERIC_LEARNING_RATE", "0.05")),
    "subsample": float(os.getenv("XGB_NUMERIC_SUBSAMPLE", "0.9")),
    "colsample_bytree": float(os.getenv("XGB_NUMERIC_COLSAMPLE_BYTREE", "0.9")),
    "reg_lambda": float(os.getenv("XGB_NUMERIC_REG_LAMBDA", "1.0")),
}
TUNED_TASK_REGRESSOR_PROFILES: Dict[str, Dict[str, Any]] = dict(XGBOOST_TUNED_TASK_REGRESSOR_PROFILES)
COMBO_TUNED_REGRESSOR_PROFILES: Dict[Tuple[int, int, str, int], Dict[str, Any]] = {
    (int(interval), int(horizon), str(task), int(months)): dict(params)
    for (interval, horizon, task, months), params in XGBOOST_PRODUCTION_COMBO_REGRESSOR_PROFILES.items()
}
REG_PARAMS = {**BASE_REG_PARAMS, **LEGACY_TASK_REGRESSOR_PROFILE, "n_jobs": int(os.getenv("XGB_NUMERIC_N_JOBS", str(get_model_threads("xgboost_numerics", 6))))}


def _combo_key(interval_minutes: Optional[int], horizon_minutes: Optional[int], task: str, training_window_months: Optional[int]) -> Optional[Tuple[int, int, str, int]]:
    if interval_minutes is None or horizon_minutes is None or training_window_months is None:
        return None
    return (int(interval_minutes), int(horizon_minutes), str(task), int(training_window_months))


def resolve_regressor_params(task: str, model_threads: int, overrides: Optional[Dict[str, Any]] = None, *, interval_minutes: Optional[int] = None, horizon_minutes: Optional[int] = None, training_window_months: Optional[int] = None) -> Dict[str, Any]:
    params = dict(BASE_REG_PARAMS)
    combo_key = _combo_key(interval_minutes, horizon_minutes, task, training_window_months)
    if combo_key in COMBO_TUNED_REGRESSOR_PROFILES:
        params.update(dict(COMBO_TUNED_REGRESSOR_PROFILES[combo_key]))
    else:
        params.update(dict(TUNED_TASK_REGRESSOR_PROFILES.get(str(task), LEGACY_TASK_REGRESSOR_PROFILE)))
    params["n_jobs"] = int(model_threads)
    if overrides:
        params.update(dict(overrides))
    return params


def regressor_profile_label(task: str, *, interval_minutes: Optional[int] = None, horizon_minutes: Optional[int] = None, training_window_months: Optional[int] = None) -> str:
    combo_key = _combo_key(interval_minutes, horizon_minutes, task, training_window_months)
    if combo_key in COMBO_TUNED_REGRESSOR_PROFILES:
        return "tuned_combo_default"
    return "tuned_task_default" if str(task) in TUNED_TASK_REGRESSOR_PROFILES else "legacy_default"


def training_window_override_bars() -> Optional[int]:
    raw = os.getenv("XGB_NUMERIC_TRAIN_WINDOWS", "").strip()
    if not raw:
        return None
    vals: List[int] = []
    for tok in raw.split(","):
        s = str(tok).strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if v > 0:
            vals.append(int(v))
    return vals[0] if vals else None


def training_window_bars_for_pair(task: str, horizon_minutes: int, interval_minutes: int) -> int:
    override_bars = training_window_override_bars()
    if override_bars is not None:
        return max(1, int(override_bars))
    pair = (str(task), int(horizon_minutes))
    if pair not in PAIR_TRAINING_WINDOW_MONTHS:
        raise ValueError(f"Inactive task/horizon pair: {pair[0]}:{pair[1]}m")
    months = int(PAIR_TRAINING_WINDOW_MONTHS[pair])
    return max(1, int((months * 30 * 24 * 60) // int(interval_minutes)))


def training_window_bars_from_months(training_window_months: int, interval_minutes: int) -> int:
    return max(1, int((int(training_window_months) * 30 * 24 * 60) // int(interval_minutes)))


def default_training_window_months_for_combo(interval: int, horizon_minutes: int, task: str) -> int:
    combo = (int(interval), int(horizon_minutes), str(task))
    for default_interval, default_horizon, default_task, months in PRODUCTION_DEFAULT_COMBO_WINDOWS:
        if combo == (int(default_interval), int(default_horizon), str(default_task)):
            return int(months)
    pair = (str(task), int(horizon_minutes))
    if pair not in PAIR_TRAINING_WINDOW_MONTHS:
        raise ValueError(f"Inactive task/horizon pair: {pair[0]}:{pair[1]}m")
    return int(PAIR_TRAINING_WINDOW_MONTHS[pair])


def is_active_task_horizon(task: str, horizon_minutes: int) -> bool:
    return int(horizon_minutes) in ACTIVE_TASK_HORIZON_MATRIX.get(str(task), [])


def resolve_xgb_default_combo_profile() -> List[Tuple[int, int, str, int]]:
    return list(PRODUCTION_DEFAULT_COMBO_WINDOWS)


def resolve_xgb_default_refit_policy(combo: Tuple[int, int, str]) -> Optional[str]:
    return PRODUCTION_DEFAULT_REFIT_POLICY.get((int(combo[0]), int(combo[1]), str(combo[2])))
