from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.forecasting.ml.shared.production_profiles_common import (
    load_latest_stage3_profiles as _load_latest_stage3_profiles,
    resolve_default_combo_specs as _resolve_default_combo_specs,
    resolve_model_params as _resolve_model_params,
)

Combo = Tuple[int, int, str]
DIAGNOSTICS_ROOT_NAME = "stats_numeric_family_test_orchestrator"


def load_latest_stage3_profiles(model_key: str) -> Dict[str, Any]:
    return _load_latest_stage3_profiles(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key))


def resolve_default_combo_specs(
    model_key: str,
    default_intervals: Sequence[int],
    default_horizons: Sequence[int],
    default_tasks: Sequence[str],
    fallback_combos: Sequence[Combo] | None = None,
) -> List[Combo]:
    return _resolve_default_combo_specs(
        diagnostics_root_name=DIAGNOSTICS_ROOT_NAME,
        model_key=str(model_key),
        default_intervals=default_intervals,
        default_horizons=default_horizons,
        default_tasks=default_tasks,
        fallback_combos=fallback_combos,
    )


def resolve_model_params(
    model_key: str,
    baseline_params: Dict[str, Any],
    *,
    interval_minutes: int | None,
    horizon_minutes: int | None,
    task: str | None,
) -> Dict[str, Any]:
    return _resolve_model_params(
        diagnostics_root_name=DIAGNOSTICS_ROOT_NAME,
        model_key=str(model_key),
        baseline_params=baseline_params,
        interval_minutes=interval_minutes,
        horizon_minutes=horizon_minutes,
        task=task,
    )
