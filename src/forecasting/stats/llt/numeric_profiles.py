from __future__ import annotations

from src.forecasting.stats.llt_state_space import (
    DEFAULT_HORIZON_MINUTES,
    DEFAULT_INTERVALS,
    MIN_TRAIN_BARS,
    MODULE_SPEC,
    TRAIN_WINDOWS_BARS,
)

DEFAULT_TASKS = tuple(MODULE_SPEC.default_tasks)
PRODUCTION_DEFAULT_COMBOS = tuple(
    (int(interval), int(horizon), str(task))
    for interval in DEFAULT_INTERVALS
    for horizon in DEFAULT_HORIZON_MINUTES
    for task in DEFAULT_TASKS
)

__all__ = [
    "DEFAULT_HORIZON_MINUTES",
    "DEFAULT_INTERVALS",
    "DEFAULT_TASKS",
    "MIN_TRAIN_BARS",
    "MODULE_SPEC",
    "PRODUCTION_DEFAULT_COMBOS",
    "TRAIN_WINDOWS_BARS",
]
