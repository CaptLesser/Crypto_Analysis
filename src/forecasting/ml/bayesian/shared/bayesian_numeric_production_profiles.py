from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.forecasting.ml.shared.numeric_runner_common import load_stage3_combo_results
from src.forecasting.ml.shared.production_profiles_common import (
    diagnostics_root as _common_diagnostics_root,
    fallback_combo_specs,
    load_latest_stage3_profiles as _load_latest_stage3_profiles,
    latest_combo_results_path as _common_latest_combo_results_path,
)

Combo = Tuple[int, int, str]
DIAGNOSTICS_ROOT_NAME = "bayesian_numeric_family_test_orchestrator"


def _diagnostics_root(model_key: str) -> Path:
    return _common_diagnostics_root(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key))


def _latest_combo_results_path(model_key: str) -> Optional[Path]:
    root = _diagnostics_root(str(model_key))
    default_root = _common_diagnostics_root(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key)).resolve()
    if root.resolve() == default_root:
        return _common_latest_combo_results_path(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key))
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("combo_results.csv") if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


@lru_cache(maxsize=64)
def _load_latest_stage3_profiles_cached(model_key: str) -> Dict[str, Any]:
    root = _diagnostics_root(str(model_key))
    default_root = _common_diagnostics_root(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key)).resolve()
    if root.resolve() == default_root:
        return _load_latest_stage3_profiles(diagnostics_root_name=DIAGNOSTICS_ROOT_NAME, model_key=str(model_key))
    combo_results = _latest_combo_results_path(str(model_key))
    if combo_results is None:
        return {"source": None, "combos": [], "params": {}}
    combos, params = load_stage3_combo_results(combo_results)
    return {"source": str(combo_results), "combos": list(combos), "params": params}


def load_latest_stage3_profiles(model_key: str) -> Dict[str, Any]:
    return _load_latest_stage3_profiles_cached(str(model_key))


def _cache_clear() -> None:
    _load_latest_stage3_profiles_cached.cache_clear()
    _load_latest_stage3_profiles.cache_clear()


load_latest_stage3_profiles.cache_clear = _cache_clear  # type: ignore[attr-defined]


def resolve_default_combo_specs(
    model_key: str,
    default_intervals: Sequence[int],
    default_horizons: Sequence[int],
    default_tasks: Sequence[str],
) -> List[Combo]:
    payload = load_latest_stage3_profiles(str(model_key))
    combos = list(payload.get("combos") or [])
    if combos:
        return combos
    return fallback_combo_specs(default_intervals, default_horizons, default_tasks)


def resolve_model_params(
    model_key: str,
    baseline_params: Dict[str, Any],
    *,
    interval_minutes: Optional[int],
    horizon_minutes: Optional[int],
    task: Optional[str],
) -> Dict[str, Any]:
    params = dict(baseline_params)
    if interval_minutes is None or horizon_minutes is None or not task:
        return params
    payload = load_latest_stage3_profiles(str(model_key))
    discovered = payload.get("params") or {}
    combo = (int(interval_minutes), int(horizon_minutes), str(task))
    params.update(dict(discovered.get(combo) or {}))
    return params
