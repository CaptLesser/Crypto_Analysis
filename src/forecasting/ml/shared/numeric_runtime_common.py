from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.forecasting.common.ml_module_utils import get_module_logger
from src.forecasting.ml.shared.numeric_runner_common import (
    ProductionStreamScopeContract,
    discover_existing_combo_specs_from_canonical_physical_output as _discover_existing_combo_specs_from_canonical_physical_output,
    discover_existing_combo_windows_from_state_tree as _discover_existing_combo_windows_from_state_tree,
    json_load_dict as _load_json_dict,
    project_root as _project_root,
    require_production_stream_scope_contract,
    resolve_production_stream_scope_contract,
)


@dataclass(frozen=True)
class ModuleLogFn:
    module_name: str
    log_file: Path

    def __call__(self, msg: str) -> None:
        get_module_logger(str(self.module_name), Path(self.log_file))(msg)


@dataclass(frozen=True)
class TabularTestedProductionArtifactScope:
    handoff_path: Path
    feature_profile_json: Path
    cohort_assets: Tuple[str, ...]
    combo_windows: Tuple[Tuple[int, int, str, int], ...]


def resolve_planning_workers(unit_workers: int, total_units: int, *, minimum_workers: int) -> int:
    if int(total_units) <= 0:
        return 1
    return max(1, min(int(total_units), max(int(unit_workers), int(minimum_workers))))


def create_queue_with_retry(
    *,
    mp_ctx: Any,
    log: Callable[[str], None],
    log_prefix: str,
    retries: int,
    retry_seconds: float,
) -> Tuple[Any, bool]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, int(retries) + 1):
        try:
            return mp_ctx.Queue(), True
        except Exception as exc:
            last_exc = exc
            if attempt >= int(retries):
                break
            log(
                f"{log_prefix}[runtime-retry] stage queue init attempt={attempt}/{int(retries)} "
                f"failed; retrying in {float(retry_seconds):.2f}s: {exc}"
            )
            time.sleep(float(retry_seconds))
    log(f"{log_prefix}[runtime-fallback] stage queue/process pool unavailable; forcing serial execution: {last_exc}")
    return queue.Queue(), False


def env_int(name: str) -> Optional[int]:
    raw = os.getenv(str(name), "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def deadzone_by_task(prefix: str) -> Dict[str, float]:
    return {
        "log_return": float(os.getenv(f"{prefix}_DEADZONE_LOGRET", "0.0001")),
        "realized_vol": float(os.getenv(f"{prefix}_DEADZONE_RV", "0.0001")),
        "true_range": float(os.getenv(f"{prefix}_DEADZONE_TR", "0.0001")),
        "max_drawdown": float(os.getenv(f"{prefix}_DEADZONE_MDD", "0.0001")),
        "max_runup": float(os.getenv(f"{prefix}_DEADZONE_MRU", "0.0001")),
        "range_efficiency": float(os.getenv(f"{prefix}_DEADZONE_REFF", "0.0001")),
    }


def module_model_key(module_slug: str, *, suffix: str = "_numerics") -> str:
    slug = str(module_slug).strip()
    return slug[: -len(suffix)] if suffix and slug.endswith(str(suffix)) else slug


def parse_survivor_combo_windows(payload: Dict[str, Any], *, handoff_path: Path) -> Tuple[Tuple[int, int, str, int], ...]:
    survivors = payload.get("survivors")
    if not isinstance(survivors, list) or not survivors:
        raise RuntimeError(f"Tested production handoff has no survivors: {handoff_path}")
    parsed: List[Tuple[int, int, str, int]] = []
    for item in survivors:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid survivor entry in handoff: {handoff_path}")
        parsed.append(
            (
                int(item["interval_minutes"]),
                int(item["horizon_minutes"]),
                str(item["task"]),
                int(item["training_window_months"]),
            )
        )
    return tuple(sorted(parsed))


def discover_tabular_tested_production_artifact_scope(
    *,
    module_slug: str,
    log_prefix: str,
    project_root: Optional[Path] = None,
) -> Optional[TabularTestedProductionArtifactScope]:
    root = (project_root or _project_root()).resolve()
    model_key = module_model_key(module_slug)
    raw_profile_root = str(os.getenv("PIPELINE_TEST_BRANCH_PROFILE_ROOT") or os.getenv("PIPELINE_SANDBOX_DIAGNOSTICS_ROOT") or "").strip()
    diagnostics_root = (Path(raw_profile_root) if raw_profile_root else root / "logs" / "diagnostics") / "tabular_numeric_family_test_orchestrator"
    canonical_handoff = diagnostics_root / model_key / "stage2" / "stage3_survivor_handoff.json"
    handoff_paths = [canonical_handoff] if canonical_handoff.is_file() else []
    if not handoff_paths:
        handoff_paths = sorted((root / "logs" / "diagnostics" / "tabular_numeric_family_test_orchestrator").glob(f"run=*/{model_key}/stage2/run=*/stage3_survivor_handoff.json"))
    if not handoff_paths:
        return None
    handoff_path = handoff_paths[-1].resolve()
    payload = _load_json_dict(handoff_path)
    combo_windows = parse_survivor_combo_windows(payload, handoff_path=handoff_path)
    cohort_assets_raw = payload.get("cohort_assets")
    cohort_assets = tuple(str(asset) for asset in cohort_assets_raw if str(asset).strip()) if isinstance(cohort_assets_raw, list) else ()
    feature_profile_raw = str(payload.get("feature_profile_json") or "").strip()
    if not feature_profile_raw:
        raise SystemExit(
            f"{log_prefix}[error] tested production handoff exists but does not declare a Stage 1 feature profile artifact: {handoff_path}"
        )
    feature_profile_json = Path(feature_profile_raw)
    feature_profile_json = (root / feature_profile_json).resolve() if not feature_profile_json.is_absolute() else feature_profile_json.resolve()
    if not feature_profile_json.exists():
        raise SystemExit(
            f"{log_prefix}[error] tested production handoff exists but referenced Stage 1 feature profile artifact is missing: {feature_profile_json}"
        )
    return TabularTestedProductionArtifactScope(
        handoff_path=handoff_path,
        feature_profile_json=feature_profile_json,
        cohort_assets=tuple(sorted(set(cohort_assets))),
        combo_windows=combo_windows,
    )


def discover_tabular_existing_production_scope(
    *,
    state_root: Path,
    infer_training_window_months_fn: Callable[[int, int, str, Optional[int]], int],
    load_state_fn: Callable[[str, int, int, str], Dict[str, Any]],
    io_config: Any = None,
) -> Optional[Tuple[Tuple[int, int, str, int], ...]]:
    state_combos = _discover_existing_combo_windows_from_state_tree(
        state_root=Path(state_root),
        infer_training_window_months_fn=lambda interval, horizon_minutes, task, selected_window_bars: infer_training_window_months_fn(
            int(interval),
            int(horizon_minutes),
            str(task),
            (int(selected_window_bars) if selected_window_bars is not None else None),
        ),
        load_state_fn=lambda asset, interval, horizon_minutes, task: load_state_fn(
            str(asset),
            int(interval),
            int(horizon_minutes),
            str(task),
        )
        or {},
    )
    if io_config is not None:
        parquet_combos = _discover_existing_combo_specs_from_canonical_physical_output(io_config=io_config)
        if parquet_combos:
            state_months = {
                (int(interval), int(horizon), str(task)): int(months)
                for interval, horizon, task, months in state_combos
            }
            return tuple(
                sorted(
                    (
                        int(interval),
                        int(horizon),
                        str(task),
                        int(
                            state_months.get(
                                (int(interval), int(horizon), str(task)),
                                infer_training_window_months_fn(int(interval), int(horizon), str(task), None),
                            )
                        ),
                    )
                    for interval, horizon, task in parquet_combos
                )
            )
    return state_combos or None
