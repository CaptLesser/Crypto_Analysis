from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.forecasting.ml.shared.production_profiles_common import (
    fallback_combo_specs,
    load_latest_stage3_profiles as _load_latest_stage3_profiles,
    resolve_default_combo_specs as _resolve_default_combo_specs,
    resolve_model_params as _resolve_model_params,
)

Combo = Tuple[int, int, str]
DIAGNOSTICS_ROOT_NAME = "bayesian_numeric_family_test_orchestrator"
load_latest_stage3_profiles = lambda model_key: _load_latest_stage3_profiles(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key))
resolve_default_combo_specs = lambda model_key, default_intervals, default_horizons, default_tasks: _resolve_default_combo_specs(
    diagnostics_root_name=DIAGNOSTICS_ROOT_NAME,
    model_key=str(model_key),
    default_intervals=default_intervals,
    default_horizons=default_horizons,
    default_tasks=default_tasks,
)
resolve_model_params = lambda model_key, baseline_params, *, interval_minutes, horizon_minutes, task: _resolve_model_params(
    diagnostics_root_name=DIAGNOSTICS_ROOT_NAME,
    model_key=str(model_key),
    baseline_params=baseline_params,
    interval_minutes=interval_minutes,
    horizon_minutes=horizon_minutes,
    task=task,
)
