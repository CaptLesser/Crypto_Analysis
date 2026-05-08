from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.forecasting.ml.shared.production_profiles_common import (
    fallback_combo_specs,
    load_latest_stage3_profiles as _load_latest_stage3_profiles,
    resolve_default_combo_specs as _resolve_default_combo_specs,
    resolve_model_params as _resolve_model_params,
)

Combo = Tuple[int, int, str]
DIAGNOSTICS_ROOT_NAME = "neural_numeric_family_test_orchestrator"


def explicit_combo_specs(interval_to_horizons: Dict[int, Sequence[int]], default_tasks: Sequence[str]) -> List[Combo]:
    combos: List[Combo] = []
    for interval, horizons in interval_to_horizons.items():
        for horizon in horizons:
            for task in default_tasks:
                if int(interval) > 0 and int(horizon) > 0 and int(horizon) % int(interval) == 0:
                    combos.append((int(interval), int(horizon), str(task)))
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))
load_latest_stage3_profiles = lambda model_key: _load_latest_stage3_profiles(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key))
resolve_default_combo_specs = lambda model_key, default_intervals, default_horizons, default_tasks, fallback_combos=None: _resolve_default_combo_specs(
    diagnostics_root_name=DIAGNOSTICS_ROOT_NAME,
    model_key=str(model_key),
    default_intervals=default_intervals,
    default_horizons=default_horizons,
    default_tasks=default_tasks,
    fallback_combos=fallback_combos,
)
resolve_model_params = lambda model_key, baseline_params, *, interval_minutes, horizon_minutes, task: _resolve_model_params(
    diagnostics_root_name=DIAGNOSTICS_ROOT_NAME,
    model_key=str(model_key),
    baseline_params=baseline_params,
    interval_minutes=interval_minutes,
    horizon_minutes=horizon_minutes,
    task=task,
)
