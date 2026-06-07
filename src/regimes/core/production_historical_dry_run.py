from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.regimes.asset_state.production_planner import (
    plan_asset_state_production_no_write,
    plan_default_asset_state_production_no_write,
)
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
)
from src.regimes.core.production_clamp_contract import (
    checkpoint_count_for_cadence,
    clamp_policy_window_summary,
)
from src.regimes.core.production_planner import RegimeProductionNoWritePlan
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.production_planner import (
    plan_cross_asset_state_production_no_write,
    plan_default_cross_asset_state_production_no_write,
)
from src.regimes.market_state.production_planner import (
    plan_default_market_state_production_no_write,
    plan_market_state_production_no_write,
)


REGIME_PRODUCTION_HISTORICAL_DRY_RUN_SCHEMA_VERSION = 1
REGIME_PRODUCTION_HISTORICAL_DRY_RUN_ARTIFACT_KIND = "regime_production_historical_walkthrough_dry_run"


@dataclass(frozen=True)
class RegimeProductionHistoricalDryRunConfig:
    asset_state_artifact_path: str | Path | None = None
    market_state_artifact_path: str | Path | None = None
    cross_asset_state_artifact_path: str | Path | None = None
    env: Mapping[str, str] | None = None
    include_branch_units: bool = False
    run_id: str = "regime_production_historical_dry_run"
    range_mode: str = "full_clamp_window_cadence_only"
    max_summary_units_per_branch: int = 3

    def __post_init__(self) -> None:
        if int(self.max_summary_units_per_branch) < 0:
            raise ValueError("Regime Production historical dry-run max_summary_units_per_branch cannot be negative")
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "range_mode", _text(self.range_mode, field_name="range_mode"))
        object.__setattr__(self, "max_summary_units_per_branch", int(self.max_summary_units_per_branch))


def run_regime_production_historical_walkthrough_dry_run(
    config: RegimeProductionHistoricalDryRunConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionHistoricalDryRunConfig) else RegimeProductionHistoricalDryRunConfig(**dict(config or {}))
    rss_start = _rss_bytes()
    child_start = _child_process_count()
    started = time.perf_counter()

    branch_plans = _build_branch_plans(cfg)
    branch_summaries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for branch, plan in branch_plans.items():
        summary = _branch_walkthrough_summary(branch, plan, max_units=cfg.max_summary_units_per_branch)
        branch_summaries[branch] = summary
        warnings.extend(str(warning) for warning in summary.get("warnings", ()) if warning)

    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes()
    child_end = _child_process_count()
    runtime = {
        "elapsed_seconds": round(float(elapsed), 6),
        "rss_start_bytes": rss_start,
        "rss_end_bytes": rss_end,
        "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
        "child_process_count_start": child_start,
        "child_process_count_end": child_end,
        "child_process_count_delta": None if child_start is None or child_end is None else int(child_end) - int(child_start),
        "subprocess_invocations_by_planner": 0,
        "frame_construction_performed": False,
        "repeated_frame_construction_detected": False,
        "long_pole_serial_loop_detected": False,
        "planner_execution_mode": "single_process_in_memory_metadata_planning",
    }
    payload = {
        "schema_version": REGIME_PRODUCTION_HISTORICAL_DRY_RUN_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_HISTORICAL_DRY_RUN_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "range_mode": cfg.range_mode,
        "range_used": _range_used(branch_plans),
        "branch_summaries": branch_summaries,
        "planned_unit_count": sum(int(item["planned_unit_count"]) for item in branch_summaries.values()),
        "selected_unit_count": sum(int(item["selected_unit_count"]) for item in branch_summaries.values()),
        "masked_unavailable_unit_count": sum(int(item["masked_unavailable_unit_count"]) for item in branch_summaries.values()),
        "skipped_or_filtered_unit_count": sum(int(item.get("skipped_or_filtered_unit_count", 0)) for item in branch_summaries.values()),
        "model_state_summary": {
            branch: summary["model_state_summary"]
            for branch, summary in branch_summaries.items()
        },
        "cross_relationship_freshness": branch_summaries[REGIME_BRANCH_CROSS_ASSET_STATE]["relationship_freshness"],
        "runtime_telemetry": runtime,
        "warnings": tuple(dict.fromkeys(warnings)),
        "branch_ready_for_sandbox_output_schema_sprint": {
            branch: _ready_for_sandbox_output_schema(summary)
            for branch, summary in branch_summaries.items()
        },
        "writer_finalizer": {
            "mode": "single_no_write_historical_walkthrough_finalizer",
            "production_write_allowed": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
        },
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "test_branch_rerun_performed": False,
        "optuna_or_campaign_run_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
        "cleanup_quarantine_delete_performed": False,
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def write_regime_production_historical_walkthrough_summary(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_branch_plans(cfg: RegimeProductionHistoricalDryRunConfig) -> dict[str, RegimeProductionNoWritePlan]:
    return {
        REGIME_BRANCH_ASSET_STATE: (
            plan_asset_state_production_no_write(cfg.asset_state_artifact_path)
            if cfg.asset_state_artifact_path is not None
            else plan_default_asset_state_production_no_write(env=cfg.env)
        ),
        REGIME_BRANCH_MARKET_STATE: (
            plan_market_state_production_no_write(cfg.market_state_artifact_path)
            if cfg.market_state_artifact_path is not None
            else plan_default_market_state_production_no_write(env=cfg.env)
        ),
        REGIME_BRANCH_CROSS_ASSET_STATE: (
            plan_cross_asset_state_production_no_write(cfg.cross_asset_state_artifact_path)
            if cfg.cross_asset_state_artifact_path is not None
            else plan_default_cross_asset_state_production_no_write(env=cfg.env)
        ),
    }


def _branch_walkthrough_summary(branch: str, plan: RegimeProductionNoWritePlan, *, max_units: int) -> dict[str, Any]:
    payload = plan.as_dict(include_units=False)
    telemetry = dict(payload.get("telemetry") or {})
    contract = dict(payload.get("planner_contract") or {})
    clamp = dict(contract.get("clamp_policy") or {})
    refit = dict(contract.get("refit_cadence") or {})
    status_counts: dict[str, int] = {}
    for unit in plan.planning_units:
        status_counts[unit.planning_status] = int(status_counts.get(unit.planning_status, 0)) + 1
    model_state = _model_state_summary(plan)
    cadence = _cadence_walkthrough(refit, clamp, model_state)
    relationship_freshness = _relationship_freshness_summary(plan) if branch == REGIME_BRANCH_CROSS_ASSET_STATE else {}
    return {
        "branch": branch,
        "status": payload.get("status"),
        "artifact_path": payload.get("artifact_path"),
        "planned_unit_count": int(payload.get("planning_unit_count") or 0),
        "selected_unit_count": int(status_counts.get("selected", 0)),
        "masked_unavailable_unit_count": int(status_counts.get("masked_unavailable", 0)),
        "skipped_or_filtered_unit_count": int(status_counts.get("skipped_or_filtered", 0)),
        "diagnostic_only_unit_count": int(status_counts.get("diagnostic_only", 0)),
        "expected_cell_count": int(telemetry.get("expected_cell_count") or 0),
        "status_counts": status_counts,
        "output_grain": dict(contract.get("output_grain") or {}),
        "clamp_policy": clamp,
        "refit_cadence": refit,
        "cadence_walkthrough": cadence,
        "model_state_summary": model_state,
        "relationship_freshness": relationship_freshness,
        "sample_units": [unit.as_dict() for unit in tuple(plan.planning_units)[:max_units]],
        "sample_units_omitted": max(0, len(plan.planning_units) - int(max_units)),
        "writer_enabled": False,
        "production_labels_written": False,
        "canonical_production_state_outputs_written": False,
        "frame_construction_performed": bool(telemetry.get("frame_construction_performed", False)),
        "test_branch_logic_executed": bool(telemetry.get("test_branch_logic_executed", False)),
        "warnings": tuple(payload.get("warnings") or ()),
    }


def _range_used(branch_plans: Mapping[str, RegimeProductionNoWritePlan]) -> dict[str, Any]:
    policies = [
        dict(plan.as_dict(include_units=False).get("planner_contract", {}).get("clamp_policy", {}))
        for plan in branch_plans.values()
    ]
    months = [int(policy.get("historical_output_months") or 0) for policy in policies]
    lookbacks = [int(policy.get("required_lookback_months") or 0) for policy in policies]
    sources = tuple(dict.fromkeys(str(policy.get("source_module")) for policy in policies if policy.get("source_module")))
    window = clamp_policy_window_summary(policies[0]) if policies else {}
    return {
        "mode": "full_clamp_window_cadence_only_no_timestamp_label_rows",
        "historical_output_months": max(months) if months else None,
        "required_lookback_months": max(lookbacks) if lookbacks else None,
        "output_start_ts": window.get("output_start_ts"),
        "output_start": window.get("output_start"),
        "output_end_ts": window.get("output_end_ts"),
        "output_end": window.get("output_end"),
        "required_lookback_start_ts": window.get("required_lookback_start_ts"),
        "required_lookback_start": window.get("required_lookback_start"),
        "numeric_forecaster_policy_reused": bool(window.get("numeric_forecaster_policy_reused", True)),
        "numeric_forecast_clamp_source_modules": sources,
        "timestamp_rows_materialized": False,
        "label_rows_materialized": False,
        "runtime_boundaries_required_before_writes": True,
    }


def _model_state_summary(plan: RegimeProductionNoWritePlan) -> dict[str, Any]:
    complete = 0
    missing = 0
    by_cadence: dict[str, int] = {}
    missing_reasons: dict[str, int] = {}
    for unit in plan.planning_units:
        model_state = dict(unit.model_state_definition or {})
        cadence_id = str(model_state.get("refit_cadence_id") or "missing_refit_cadence_id")
        by_cadence[cadence_id] = int(by_cadence.get(cadence_id, 0)) + 1
        missing_fields = tuple(str(item) for item in (model_state.get("missing_required_fields") or ()))
        if missing_fields:
            missing += 1
            for field in missing_fields:
                missing_reasons[field] = int(missing_reasons.get(field, 0)) + 1
        else:
            complete += 1
    return {
        "model_state_unit_count": len(plan.planning_units),
        "complete_model_state_unit_count": complete,
        "missing_required_field_unit_count": missing,
        "refit_cadence_id_counts": by_cadence,
        "missing_required_field_counts": missing_reasons,
    }


def _cadence_walkthrough(refit: Mapping[str, Any], clamp: Mapping[str, Any], model_state: Mapping[str, Any]) -> dict[str, Any]:
    months = max(1, int(clamp.get("historical_output_months") or 1))
    branch_cadence = str(refit.get("cadence") or "monthly")
    branch_count = checkpoint_count_for_cadence(branch_cadence, months)
    cadence_counts: dict[str, int] = {}
    for cadence_id, unit_count in dict(model_state.get("refit_cadence_id_counts") or {}).items():
        cadence = _cadence_from_id(cadence_id, fallback=branch_cadence)
        cadence_counts[cadence] = int(cadence_counts.get(cadence, 0)) + int(unit_count)
    checkpoint_counts = {
        cadence: checkpoint_count_for_cadence(cadence, months)
        for cadence in sorted(cadence_counts or {branch_cadence: 0})
    }
    return {
        "calendar_based": True,
        "branch_default_cadence": branch_cadence,
        "branch_default_refit_checkpoint_count": branch_count,
        "historical_output_months": months,
        "required_lookback_months": int(clamp.get("required_lookback_months") or 0),
        "refit_checkpoint_count_by_cadence": checkpoint_counts,
        "unit_count_by_cadence": cadence_counts,
        "timestamp_rows_materialized": False,
        "model_state_records_written": False,
    }


def _relationship_freshness_summary(plan: RegimeProductionNoWritePlan) -> dict[str, Any]:
    checks = [dict(item) for item in plan.relationship_input_checks]
    freshness_summaries = [
        item
        for item in checks
        if item.get("artifact_kind") == "regime_production_relationship_input_freshness_summary"
    ]
    freshness_summary = dict(freshness_summaries[-1]) if freshness_summaries else {}
    input_root_checks = [
        item
        for item in checks
        if item.get("artifact_kind") != "regime_production_relationship_input_freshness_summary"
    ]
    warnings = [
        str(item.get("reason_code"))
        for item in input_root_checks
        if item.get("reason_code") not in (None, "")
    ]
    warning_codes = dict(freshness_summary.get("reason_code_counts") or {})
    freshness_warning_codes = [
        str(code)
        for code, count in warning_codes.items()
        if str(code).strip()
        and str(code) != "relationship_input_available_fresh"
        and int(count or 0) > 0
    ]
    stale_units = sum(
        1
        for unit in plan.planning_units
        if any("stale_relationship" in str(reason) or "stale_snapshot" in str(reason) for reason in unit.reason_codes)
    )
    status_counts = dict(freshness_summary.get("status_counts") or {})
    stale_relationship_unit_count = int(status_counts.get("stale", stale_units) or 0)
    cadence_days = sorted(
        {
            int(value)
            for unit in plan.planning_units
            for value in (unit.method_metadata.get("snapshot_cadence_days"),)
            if value not in (None, "") and str(value).isdigit()
        }
    )
    clamp_history_checks = [
        dict(dict(unit.timestamp_plan or {}).get("relationship_input_history_check") or {})
        for unit in plan.planning_units
        if dict(dict(unit.timestamp_plan or {}).get("relationship_input_history_check") or {})
    ]
    clamp_history_reasons = tuple(
        dict.fromkeys(
            str(reason)
            for item in clamp_history_checks
            for reason in item.get("reason_codes", ())
            if str(reason or "").strip()
        )
    )
    return {
        "relationship_input_checks": input_root_checks,
        "relationship_input_freshness_summary": freshness_summary,
        "relationship_input_freshness_recorded": bool(freshness_summary),
        "relationship_inputs_available_fresh": freshness_summary.get("relationship_inputs_available_fresh"),
        "relationship_freshness_status_counts": status_counts,
        "relationship_freshness_status_counts_by_band": dict(freshness_summary.get("status_counts_by_band") or {}),
        "relationship_freshness_reason_code_counts": warning_codes,
        "relationship_freshness_policy": dict(freshness_summary.get("policy") or {}),
        "relationship_input_warning_count": len(tuple(dict.fromkeys(warnings + freshness_warning_codes))),
        "relationship_input_warnings": tuple(dict.fromkeys(warnings + freshness_warning_codes)),
        "stale_relationship_unit_count": stale_relationship_unit_count,
        "snapshot_cadence_days_observed": cadence_days,
        "clamp_history_check_count": len(clamp_history_checks),
        "clamp_history_check_warning_count": len(clamp_history_reasons),
        "clamp_history_check_warnings": clamp_history_reasons,
        "clamp_history_checks_passed": all(bool(item.get("passed")) for item in clamp_history_checks) if clamp_history_checks else None,
        "freshness_thresholds_source": "active_selected_profile_relationship_context_cadence_policy",
        "freshness_thresholds_provisional": False,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "relationship_sidecars_selected_profile_artifacts": False,
    }


def _ready_for_sandbox_output_schema(summary: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if str(summary.get("status")) != "ready_for_dry_consumption":
        blockers.append("planner_not_ready_for_dry_consumption")
    if int(summary.get("planned_unit_count") or 0) <= 0:
        blockers.append("no_planned_units")
    if summary.get("writer_enabled") is not False or summary.get("production_labels_written") is not False:
        blockers.append("write_or_label_gate_open")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "requires_writer_sprint_before_production_outputs": True,
    }


def _cadence_from_id(cadence_id: str, *, fallback: str) -> str:
    text = str(cadence_id).lower()
    for cadence in ("biweekly", "weekly", "monthly"):
        if cadence in text:
            return cadence
    return fallback


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _child_process_count() -> int | None:
    try:
        import psutil  # type: ignore

        return len(psutil.Process(os.getpid()).children(recursive=True))
    except Exception:
        return None


def _sanitize_workspace_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize_workspace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_workspace_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_workspace_paths(item) for item in value)
    if isinstance(value, str):
        return _workspace_relative_path_text(value)
    return value


def _workspace_relative_path_text(value: str) -> str:
    try:
        path = Path(value)
    except Exception:
        return value
    if not path.is_absolute():
        return value
    try:
        return str(path.resolve(strict=False).relative_to(Path.cwd().resolve(strict=False)))
    except Exception:
        return value


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production historical dry-run {field_name} must be non-empty")
    return text


__all__ = [
    "REGIME_PRODUCTION_HISTORICAL_DRY_RUN_ARTIFACT_KIND",
    "REGIME_PRODUCTION_HISTORICAL_DRY_RUN_SCHEMA_VERSION",
    "RegimeProductionHistoricalDryRunConfig",
    "run_regime_production_historical_walkthrough_dry_run",
    "write_regime_production_historical_walkthrough_summary",
]
