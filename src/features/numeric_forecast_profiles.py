from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

from src.forecasting.ml.tabular.shared.tabular_numeric_production_defaults import (
    XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS as SHARED_XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS,
)

Combo = Tuple[int, int, str]
ComboWindow = Tuple[int, int, str, int]
ComboWindowRefit = Tuple[int, int, str, int, str]

XGB_PRODUCTION_DEFAULT_COMBO_WINDOWS: Tuple[ComboWindow, ...] = tuple(
    (int(interval), int(horizon), str(task), int(months))
    for interval, horizon, task, months in SHARED_XGBOOST_PRODUCTION_DEFAULT_COMBO_WINDOWS
)

XGB_PRODUCTION_DEFAULT_COMBOS: Tuple[Combo, ...] = tuple(
    (int(interval), int(horizon), str(task))
    for interval, horizon, task, _ in XGB_PRODUCTION_DEFAULT_COMBO_WINDOWS
)

SUPPORTED_REFIT_CADENCES: Tuple[str, ...] = ("weekly", "biweekly", "monthly")

XGB_PRODUCTION_DEFAULT_REFIT_POLICY: Dict[Combo, str] = {
    (int(interval), int(horizon), str(task)): (
        "weekly" if int(interval) <= 60 else "biweekly" if int(interval) <= 240 else "monthly"
    )
    for interval, horizon, task, _ in XGB_PRODUCTION_DEFAULT_COMBO_WINDOWS
}

# Shared tabular numeric candidate universe used across XGBoost, LightGBM,
# CatBoost, Random Forest, ElasticNet, and related tabular ML branches during
# the staged evaluation workflow.
TABULAR_NUMERIC_TASK_HORIZON_MATRIX: Dict[str, List[int]] = {
    "log_return": [60, 240, 720, 1440],
    "realized_vol": [60, 1440],
    "true_range": [60, 240, 720, 1440],
    "max_drawdown": [60, 1440],
    "max_runup": [60, 1440],
    "range_efficiency": [60, 240, 720, 1440],
}

TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS: Dict[Tuple[str, int], int] = {
    ("log_return", 60): 12,
    ("log_return", 240): 9,
    ("log_return", 720): 12,
    ("log_return", 1440): 12,
    ("realized_vol", 60): 9,
    ("realized_vol", 1440): 12,
    ("true_range", 60): 9,
    ("true_range", 240): 12,
    ("true_range", 720): 12,
    ("true_range", 1440): 12,
    ("max_drawdown", 60): 9,
    ("max_drawdown", 1440): 12,
    ("max_runup", 60): 12,
    ("max_runup", 1440): 12,
    ("range_efficiency", 60): 9,
    ("range_efficiency", 240): 12,
    ("range_efficiency", 720): 12,
    ("range_efficiency", 1440): 12,
}


def _unique_in_order(values: Iterable[int | str]) -> Tuple[int | str, ...]:
    seen: set[int | str] = set()
    out: List[int | str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


XGB_PRODUCTION_DEFAULT_INTERVALS: Tuple[int, ...] = tuple(
    int(value) for value in _unique_in_order(interval for interval, _, _ in XGB_PRODUCTION_DEFAULT_COMBOS)
)
XGB_PRODUCTION_DEFAULT_HORIZONS: Tuple[int, ...] = tuple(
    int(value) for value in _unique_in_order(horizon for _, horizon, _ in XGB_PRODUCTION_DEFAULT_COMBOS)
)
XGB_PRODUCTION_DEFAULT_TASKS: Tuple[str, ...] = tuple(
    str(value) for value in _unique_in_order(task for _, _, task in XGB_PRODUCTION_DEFAULT_COMBOS)
)
TABULAR_NUMERIC_TASKS: Tuple[str, ...] = tuple(str(task) for task in TABULAR_NUMERIC_TASK_HORIZON_MATRIX.keys())
TABULAR_NUMERIC_HORIZONS: Tuple[int, ...] = tuple(
    int(value) for value in _unique_in_order(horizon for horizons in TABULAR_NUMERIC_TASK_HORIZON_MATRIX.values() for horizon in horizons)
)


def format_combo(combo: Combo) -> str:
    interval, horizon, task = combo
    return f"{int(interval)}m/{int(horizon)}m/{str(task)}"


def format_combo_list(combos: Sequence[Combo]) -> str:
    return ",".join(format_combo(combo) for combo in combos)


def format_combo_window(combo_window: ComboWindow) -> str:
    interval, horizon, task, months = combo_window
    return f"{int(interval)}m/{int(horizon)}m/{str(task)}@{int(months)}m"


def format_combo_window_list(combo_windows: Sequence[ComboWindow]) -> str:
    return ",".join(format_combo_window(combo_window) for combo_window in combo_windows)


def format_combo_window_refit(combo_profile: ComboWindowRefit) -> str:
    interval, horizon, task, months, cadence = combo_profile
    return f"{int(interval)}m/{int(horizon)}m/{str(task)}@{int(months)}m@{str(cadence)}"


def format_combo_window_refit_list(combo_profiles: Sequence[ComboWindowRefit]) -> str:
    return ",".join(format_combo_window_refit(combo_profile) for combo_profile in combo_profiles)


def resolve_xgb_default_combo_profile() -> List[ComboWindow]:
    return [(int(interval), int(horizon), str(task), int(months)) for interval, horizon, task, months in XGB_PRODUCTION_DEFAULT_COMBO_WINDOWS]


def resolve_xgb_default_refit_policy(combo: Combo) -> str | None:
    interval, horizon, task = combo
    return XGB_PRODUCTION_DEFAULT_REFIT_POLICY.get((int(interval), int(horizon), str(task)))


def normalize_refit_cadence(value: str) -> str:
    cadence = str(value).strip().lower()
    if cadence not in SUPPORTED_REFIT_CADENCES:
        raise ValueError(f"unsupported refit cadence: {value}")
    return cadence


def _add_months_utc(ts: int, months: int) -> int:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    year = int(dt.year)
    month = int(dt.month) - 1 + int(months)
    year += month // 12
    month = month % 12 + 1
    day = min(
        int(dt.day),
        [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1],
    )
    next_dt = dt.replace(year=year, month=month, day=day)
    return int(next_dt.timestamp())


def next_refit_due_ts(cadence: str, last_refit_ts: int) -> int:
    normalized = normalize_refit_cadence(cadence)
    if normalized == "weekly":
        return int(last_refit_ts) + (7 * 24 * 60 * 60)
    if normalized == "biweekly":
        return int(last_refit_ts) + (14 * 24 * 60 * 60)
    return _add_months_utc(int(last_refit_ts), 1)


def should_refit(cadence: str, last_refit_ts: int | None, now_ts: int) -> bool:
    if last_refit_ts is None:
        return False
    return int(now_ts) >= int(next_refit_due_ts(cadence, int(last_refit_ts)))


def filter_xgb_production_default_combos(
    *,
    intervals: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    tasks: Sequence[str] | None = None,
) -> List[Combo]:
    interval_set = {int(value) for value in intervals} if intervals else None
    horizon_set = {int(value) for value in horizons} if horizons else None
    task_set = {str(value) for value in tasks} if tasks else None
    out: List[Combo] = []
    for interval, horizon, task in XGB_PRODUCTION_DEFAULT_COMBOS:
        if interval_set is not None and int(interval) not in interval_set:
            continue
        if horizon_set is not None and int(horizon) not in horizon_set:
            continue
        if task_set is not None and str(task) not in task_set:
            continue
        out.append((int(interval), int(horizon), str(task)))
    return out


def approved_xgb_pairs_by_interval(
    *,
    intervals: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    tasks: Sequence[str] | None = None,
) -> Dict[int, List[Tuple[str, int]]]:
    out: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for interval, horizon, task in filter_xgb_production_default_combos(
        intervals=intervals, horizons=horizons, tasks=tasks
    ):
        out[int(interval)].append((str(task), int(horizon)))
    return {int(interval): pairs for interval, pairs in out.items()}


# Backward-compatible aliases for existing consumers. XGBoost production policy
# remains branch-specific, while the full staged-workflow candidate universe is
# now exposed with tabular-family naming.
APPROVED_XGB_NUMERIC_COMBOS = XGB_PRODUCTION_DEFAULT_COMBOS
APPROVED_XGB_NUMERIC_COMBO_WINDOWS = XGB_PRODUCTION_DEFAULT_COMBO_WINDOWS
APPROVED_XGB_DEFAULT_INTERVALS = XGB_PRODUCTION_DEFAULT_INTERVALS
APPROVED_XGB_DEFAULT_HORIZONS = XGB_PRODUCTION_DEFAULT_HORIZONS
APPROVED_XGB_DEFAULT_TASKS = XGB_PRODUCTION_DEFAULT_TASKS
TABULAR_ACTIVE_TASK_HORIZON_MATRIX = TABULAR_NUMERIC_TASK_HORIZON_MATRIX
TABULAR_ACTIVE_PAIR_TRAINING_WINDOW_MONTHS = TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS
SUPPORTED_XGB_TASK_HORIZON_MATRIX = TABULAR_NUMERIC_TASK_HORIZON_MATRIX
SUPPORTED_XGB_PAIR_TRAINING_WINDOW_MONTHS = TABULAR_NUMERIC_PAIR_TRAINING_WINDOW_MONTHS
SUPPORTED_XGB_TASKS = TABULAR_NUMERIC_TASKS
SUPPORTED_XGB_HORIZONS = TABULAR_NUMERIC_HORIZONS
APPROVED_XGB_TASK_HORIZON_MATRIX = SUPPORTED_XGB_TASK_HORIZON_MATRIX
APPROVED_XGB_PAIR_TRAINING_WINDOW_MONTHS = SUPPORTED_XGB_PAIR_TRAINING_WINDOW_MONTHS
APPROVED_XGB_DEFAULT_REFIT_POLICY = XGB_PRODUCTION_DEFAULT_REFIT_POLICY
filter_approved_xgb_combos = filter_xgb_production_default_combos
