from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.features.numeric_forecast_profiles import (
    SUPPORTED_REFIT_CADENCES,
    TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS,
    TABULAR_NUMERIC_TASK_HORIZON_MATRIX,
)
from src.forecasting.common.runtime_config import get_model_threads
from src.forecasting.ml.tabular.shared.tabular_numeric_production_defaults import (
    CATBOOST_PRODUCTION_COMBO_REGRESSOR_PROFILES,
    CATBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS,
    CATBOOST_TUNED_TASK_REGRESSOR_PROFILES,
)

DEFAULT_INTERVALS = sorted({int(interval) for interval, _, _, _ in CATBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS})
DEFAULT_HORIZON_MINUTES = sorted({int(horizon) for _, horizon, _, _ in CATBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS})
PRODUCTION_DEFAULT_COMBO_WINDOWS = [
    (int(interval), int(horizon), str(task), int(months))
    for interval, horizon, task, months in CATBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS
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
PAIR_TRAINING_WINDOW_MONTHS: Dict[Tuple[str, int], int] = {
    (str(task), int(hm)): int(months) for (task, hm), months in TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS.items()
}

BASE_REG_PARAMS = {
    "loss_function": "RMSE",
    "random_seed": int(os.getenv("CB_NUMERIC_RANDOM_STATE", "17")),
    "verbose": False,
}
LEGACY_TASK_REGRESSOR_PROFILE = {
    "iterations": int(os.getenv("CB_NUMERIC_ITERATIONS", "220")),
    "depth": int(os.getenv("CB_NUMERIC_DEPTH", "6")),
    "learning_rate": float(os.getenv("CB_NUMERIC_LEARNING_RATE", "0.05")),
    "l2_leaf_reg": float(os.getenv("CB_NUMERIC_L2_LEAF_REG", "3.0")),
    "random_strength": float(os.getenv("CB_NUMERIC_RANDOM_STRENGTH", "1.0")),
    "rsm": float(os.getenv("CB_NUMERIC_RSM", "0.9")),
    "bootstrap_type": os.getenv("CB_NUMERIC_BOOTSTRAP_TYPE", "Bayesian"),
    "bagging_temperature": float(os.getenv("CB_NUMERIC_BAGGING_TEMPERATURE", "1.0")),
}
TUNED_TASK_REGRESSOR_PROFILES: Dict[str, Dict[str, Any]] = dict(CATBOOST_TUNED_TASK_REGRESSOR_PROFILES)
COMBO_TUNED_REGRESSOR_PROFILES: Dict[Tuple[int, int, str, int], Dict[str, Any]] = {
    (int(interval), int(horizon), str(task), int(months)): dict(params)
    for (interval, horizon, task, months), params in CATBOOST_PRODUCTION_COMBO_REGRESSOR_PROFILES.items()
}
REG_PARAMS = {
    **BASE_REG_PARAMS,
    **LEGACY_TASK_REGRESSOR_PROFILE,
    "thread_count": int(os.getenv("CB_NUMERIC_N_JOBS", str(get_model_threads("catboost_numerics", 6)))),
}


def _combo_key(
    interval_minutes: Optional[int],
    horizon_minutes: Optional[int],
    task: str,
    training_window_months: Optional[int],
) -> Optional[Tuple[int, int, str, int]]:
    if interval_minutes is None or horizon_minutes is None or training_window_months is None:
        return None
    return (int(interval_minutes), int(horizon_minutes), str(task), int(training_window_months))


def _resolve_train_dir() -> str:
    candidates: List[Path] = []
    env_train_dir = os.getenv("CATBOOST_TRAIN_DIR", "").strip()
    if env_train_dir:
        candidates.append(Path(env_train_dir))
    profile = selected_profile()
    tmp_root = Path(resolve_path("tmp_root", profile=profile, required=False) or Path("tmp"))
    candidates.append(tmp_root / "catboost_numerics_stage" / "catboost_train")
    candidates.append(Path.cwd() / "catboost_info")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        except Exception:
            continue
    return str(Path.cwd() / "catboost_info")


def resolve_regressor_params(
    task: str,
    model_threads: int,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    interval_minutes: Optional[int] = None,
    horizon_minutes: Optional[int] = None,
    training_window_months: Optional[int] = None,
) -> Dict[str, Any]:
    params = dict(BASE_REG_PARAMS)
    combo_key = _combo_key(interval_minutes, horizon_minutes, task, training_window_months)
    if combo_key in COMBO_TUNED_REGRESSOR_PROFILES:
        params.update(dict(COMBO_TUNED_REGRESSOR_PROFILES[combo_key]))
    else:
        params.update(dict(TUNED_TASK_REGRESSOR_PROFILES.get(str(task), LEGACY_TASK_REGRESSOR_PROFILE)))
    params["thread_count"] = int(model_threads)
    params["train_dir"] = _resolve_train_dir()
    if overrides:
        params.update(dict(overrides))
        if "train_dir" not in dict(overrides):
            params["train_dir"] = _resolve_train_dir()
    return params


def regressor_profile_label(
    task: str,
    *,
    interval_minutes: Optional[int] = None,
    horizon_minutes: Optional[int] = None,
    training_window_months: Optional[int] = None,
) -> str:
    combo_key = _combo_key(interval_minutes, horizon_minutes, task, training_window_months)
    if combo_key in COMBO_TUNED_REGRESSOR_PROFILES:
        return "tuned_combo_default"
    return "tuned_task_default" if str(task) in TUNED_TASK_REGRESSOR_PROFILES else "legacy_default"


def normalize_refit_cadence(value: str) -> str:
    cadence = str(value).strip().lower()
    if cadence not in SUPPORTED_REFIT_CADENCES:
        raise ValueError(f"unsupported refit cadence: {value}")
    return cadence


def resolve_catboost_default_combo_profile() -> List[Tuple[int, int, str, int]]:
    return list(PRODUCTION_DEFAULT_COMBO_WINDOWS)


def resolve_catboost_default_refit_policy(combo: Tuple[int, int, str]) -> Optional[str]:
    return PRODUCTION_DEFAULT_REFIT_POLICY.get((int(combo[0]), int(combo[1]), str(combo[2])))


def training_window_override_bars() -> Optional[int]:
    raw = os.getenv("CB_NUMERIC_TRAIN_WINDOWS", "").strip()
    if not raw:
        return None
    values: List[int] = []
    for token in raw.split(","):
        token = str(token).strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            values.append(int(value))
    return values[0] if values else None


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


resolve_cb_default_combo_profile = resolve_catboost_default_combo_profile
resolve_cb_default_refit_policy = resolve_catboost_default_refit_policy
