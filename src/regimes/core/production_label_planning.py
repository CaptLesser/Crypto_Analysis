from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.asset_state.production_planner import (
    plan_asset_state_production_no_write,
    plan_default_asset_state_production_no_write,
)
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_output_contracts import (
    BRANCH_PARTITION_FIELDS,
    resolve_regime_production_sandbox_output_root_contract,
)
from src.regimes.core.production_planner import (
    REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED,
    RegimeProductionNoWritePlan,
    validate_regime_production_planner_gates,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.production_planner import (
    plan_cross_asset_state_production_no_write,
    plan_default_cross_asset_state_production_no_write,
)
from src.regimes.market_state.production_planner import (
    plan_default_market_state_production_no_write,
    plan_market_state_production_no_write,
)


REGIME_PRODUCTION_LABEL_PLANNING_SCHEMA_VERSION = 1
REGIME_PRODUCTION_LABEL_PLANNING_RECORD_ARTIFACT_KIND = "regime_production_sandbox_label_planning_record"
REGIME_PRODUCTION_LABEL_PLANNING_SUMMARY_ARTIFACT_KIND = "regime_production_sandbox_label_planning_summary"

LABEL_PLANNING_STATUS_SELECTED = "selected"
LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE = "masked_unavailable"
LABEL_PLANNING_STATUS_SKIPPED_FILTERED = "skipped_filtered"
LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
LABEL_PLANNING_STATUS_MISSING_INPUT = "missing_input"
LABEL_PLANNING_STATUS_INVALID_PROFILE = "invalid_profile"

LABEL_ASSIGNMENT_STATUS_NOT_ASSIGNED = "not_assigned_planning_only"


@dataclass(frozen=True)
class RegimeProductionLabelPlanningConfig:
    asset_state_artifact_path: str | Path | None = None
    market_state_artifact_path: str | Path | None = None
    cross_asset_state_artifact_path: str | Path | None = None
    sandbox_output_root: str | Path | None = None
    env: Mapping[str, str] | None = None
    run_id: str = "regime_production_label_planning_records"
    include_records: bool = False
    max_sample_records_per_branch: int = 5

    def __post_init__(self) -> None:
        if int(self.max_sample_records_per_branch) < 0:
            raise ValueError("Regime Production label planning sample count cannot be negative")
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "max_sample_records_per_branch", int(self.max_sample_records_per_branch))


@dataclass(frozen=True)
class RegimeProductionLabelPlanningRecord:
    branch: str
    output_grain_key: Mapping[str, Any]
    timestamp_range_chunk: Mapping[str, Any]
    profile_id: str | None
    profile_version: str | None
    definition_id: str
    definition_version: str
    planning_status: str
    mask_reason: str | None
    source_tail_ts: Any
    known_at_ts: Any
    planned_output_partition_key: Mapping[str, Any]
    target_key: Mapping[str, Any]
    relationship_input_tail_ts: Any = None
    relationship_known_at_ts: Any = None
    relationship_input_metadata: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)
    writer_enabled: bool = False
    production_labels_written: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        status = _text(self.planning_status, field_name="planning_status")
        if status not in {
            LABEL_PLANNING_STATUS_SELECTED,
            LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
            LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
            LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
            LABEL_PLANNING_STATUS_MISSING_INPUT,
            LABEL_PLANNING_STATUS_INVALID_PROFILE,
        }:
            raise ValueError(f"Unsupported Regime Production label planning status: {status!r}")
        if self.writer_enabled or self.production_labels_written or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Regime Production label planning records cannot enable writes or labels")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "output_grain_key", to_jsonable(dict(self.output_grain_key)))
        object.__setattr__(self, "timestamp_range_chunk", to_jsonable(dict(self.timestamp_range_chunk)))
        object.__setattr__(self, "definition_id", _text(self.definition_id, field_name="definition_id"))
        object.__setattr__(self, "definition_version", _text(self.definition_version, field_name="definition_version"))
        object.__setattr__(self, "planning_status", status)
        object.__setattr__(self, "planned_output_partition_key", to_jsonable(dict(self.planned_output_partition_key)))
        object.__setattr__(self, "target_key", to_jsonable(dict(self.target_key)))
        object.__setattr__(self, "relationship_input_metadata", to_jsonable(dict(self.relationship_input_metadata)))
        object.__setattr__(self, "lineage", to_jsonable(dict(self.lineage)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_LABEL_PLANNING_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_LABEL_PLANNING_RECORD_ARTIFACT_KIND,
            "branch": self.branch,
            "target_key": to_jsonable(dict(self.target_key)),
            "output_grain_key": to_jsonable(dict(self.output_grain_key)),
            "timestamp_range_chunk": to_jsonable(dict(self.timestamp_range_chunk)),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "definition_id": self.definition_id,
            "definition_version": self.definition_version,
            "planning_status": self.planning_status,
            "availability_status": self.planning_status,
            "mask_reason": self.mask_reason,
            "source_tail_ts": self.source_tail_ts,
            "known_at_ts": self.known_at_ts,
            "relationship_input_tail_ts": self.relationship_input_tail_ts,
            "relationship_known_at_ts": self.relationship_known_at_ts,
            "relationship_input_metadata": to_jsonable(dict(self.relationship_input_metadata)),
            "planned_output_partition_key": to_jsonable(dict(self.planned_output_partition_key)),
            "lineage": to_jsonable(dict(self.lineage)),
            "state_id": None,
            "state_label": None,
            "label_value": None,
            "confidence": None,
            "label_assignment_status": LABEL_ASSIGNMENT_STATUS_NOT_ASSIGNED,
            "labels_generated": False,
            "writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def run_regime_production_label_planning_records(
    config: RegimeProductionLabelPlanningConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionLabelPlanningConfig) else RegimeProductionLabelPlanningConfig(**dict(config or {}))
    rss_start = _rss_bytes()
    child_start = _child_process_count()
    started = time.perf_counter()
    plans = _build_no_write_plans(cfg)
    branch_payloads: dict[str, dict[str, Any]] = {}
    for branch, plan in plans.items():
        branch_payloads[branch] = build_label_planning_payload_from_no_write_plan(
            plan,
            run_id=cfg.run_id,
            sandbox_output_root=cfg.sandbox_output_root,
            env=cfg.env,
            include_records=cfg.include_records,
            max_sample_records=cfg.max_sample_records_per_branch,
        )
    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes()
    child_end = _child_process_count()
    payload = {
        "schema_version": REGIME_PRODUCTION_LABEL_PLANNING_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_LABEL_PLANNING_SUMMARY_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "branch_plans": branch_payloads,
        "planning_record_count_by_branch": {branch: int(item["planning_record_count"]) for branch, item in branch_payloads.items()},
        "selected_record_count_by_branch": {branch: int(item["selected_record_count"]) for branch, item in branch_payloads.items()},
        "masked_unavailable_record_count_by_branch": {branch: int(item["masked_unavailable_record_count"]) for branch, item in branch_payloads.items()},
        "skipped_filtered_record_count_by_branch": {branch: int(item["skipped_filtered_record_count"]) for branch, item in branch_payloads.items()},
        "total_planning_record_count": sum(int(item["planning_record_count"]) for item in branch_payloads.values()),
        "total_selected_record_count": sum(int(item["selected_record_count"]) for item in branch_payloads.values()),
        "total_masked_unavailable_record_count": sum(int(item["masked_unavailable_record_count"]) for item in branch_payloads.values()),
        "total_skipped_filtered_record_count": sum(int(item["skipped_filtered_record_count"]) for item in branch_payloads.values()),
        "runtime_telemetry": {
            "elapsed_seconds": round(float(elapsed), 6),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
            "child_process_count_start": child_start,
            "child_process_count_end": child_end,
            "child_process_count_delta": None if child_start is None or child_end is None else int(child_end) - int(child_start),
            "subprocess_invocations_by_planner": 0,
            "frame_construction_performed": False,
            "label_assignment_performed": False,
        },
        "parent_finalizer": {
            "mode": "single_sandbox_label_planning_summary_finalizer",
            "planning_summary_artifact_allowed": True,
            "planning_records_artifact_written": bool(cfg.include_records),
            "production_write_allowed": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
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
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def build_label_planning_payload_from_no_write_plan(
    plan: RegimeProductionNoWritePlan,
    *,
    run_id: str,
    sandbox_output_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    include_records: bool = False,
    max_sample_records: int = 5,
) -> dict[str, Any]:
    if plan.as_dict(include_units=False).get("artifact_kind") != REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND:
        raise ValueError("Regime Production label planning requires a no-write branch plan")
    branch = _branch_name(plan.branch)
    output_root = resolve_regime_production_sandbox_output_root_contract(
        branch,
        sandbox_root=sandbox_output_root,
        env=env,
        project_root=project_root,
    )
    records: list[RegimeProductionLabelPlanningRecord] = []
    samples: list[RegimeProductionLabelPlanningRecord] = []
    status_counts: dict[str, int] = {}
    mask_counts: dict[str, int] = {}
    selected = 0
    masked = 0
    skipped = 0
    diagnostic = 0
    missing = 0
    relationship_warning_count = 0
    for unit in plan.planning_units:
        record = _label_planning_record_for_unit(
            unit,
            branch=branch,
            run_id=run_id,
            sandbox_root=str(output_root.sandbox_root),
        )
        status_counts[record.planning_status] = int(status_counts.get(record.planning_status, 0)) + 1
        if record.mask_reason:
            mask_counts[record.mask_reason] = int(mask_counts.get(record.mask_reason, 0)) + 1
        if record.planning_status == LABEL_PLANNING_STATUS_SELECTED:
            selected += 1
        elif record.planning_status == LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE:
            masked += 1
        elif record.planning_status == LABEL_PLANNING_STATUS_SKIPPED_FILTERED:
            skipped += 1
        elif record.planning_status == LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY:
            diagnostic += 1
        elif record.planning_status == LABEL_PLANNING_STATUS_MISSING_INPUT:
            missing += 1
        if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
            checks = record.relationship_input_metadata.get("relationship_input_checks", ())
            relationship_warning_count += sum(1 for check in checks if dict(check).get("reason_code") not in (None, ""))
        if include_records:
            records.append(record)
        if len(samples) < int(max_sample_records):
            samples.append(record)
    gate = validate_regime_production_planner_gates(branch)
    payload = {
        "schema_version": REGIME_PRODUCTION_LABEL_PLANNING_SCHEMA_VERSION,
        "artifact_kind": "regime_production_branch_label_planning_payload",
        "branch": branch,
        "source_no_write_plan": plan.as_dict(include_units=False),
        "sandbox_output_root_contract": output_root.as_dict(),
        "planning_record_count": len(plan.planning_units),
        "selected_record_count": selected,
        "masked_unavailable_record_count": masked,
        "skipped_filtered_record_count": skipped,
        "diagnostic_only_record_count": diagnostic,
        "missing_input_record_count": missing,
        "status_counts": status_counts,
        "mask_reason_counts": mask_counts,
        "relationship_input_warning_count": relationship_warning_count,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "planning_record_samples": [record.as_dict() for record in samples],
        "planning_records": [record.as_dict() for record in records],
        "planning_records_omitted": not include_records,
        "label_assignment_status": LABEL_ASSIGNMENT_STATUS_NOT_ASSIGNED,
        "labels_generated": False,
        "state_values_generated": False,
        "confidence_values_generated": False,
        "writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_gate_validation": gate.as_dict(),
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def write_regime_production_label_planning_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_no_write_plans(cfg: RegimeProductionLabelPlanningConfig) -> dict[str, RegimeProductionNoWritePlan]:
    return {
        REGIME_BRANCH_ASSET_STATE: (
            plan_asset_state_production_no_write(cfg.asset_state_artifact_path, expected_cell_count=3204)
            if cfg.asset_state_artifact_path is not None
            else plan_default_asset_state_production_no_write(expected_cell_count=3204, env=cfg.env)
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


def _label_planning_record_for_unit(
    unit,
    *,
    branch: str,
    run_id: str,
    sandbox_root: str,
) -> RegimeProductionLabelPlanningRecord:
    model_state = dict(unit.model_state_definition or {})
    normalized_lineage = dict(unit.normalized_lineage or {})
    status = _planning_status(unit.planning_status)
    cadence_id = str(model_state.get("refit_cadence_id") or "missing_refit_cadence_id")
    definition_id = f"{branch}:{_target_hash(unit.target_key)}:{cadence_id}:label_planning"
    timestamp_range = _timestamp_range_chunk(unit.timestamp_plan)
    mask_reason = _mask_reason(unit, status)
    relationship_tail = _relationship_tail(unit, model_state)
    return RegimeProductionLabelPlanningRecord(
        branch=branch,
        target_key=unit.target_key,
        output_grain_key=unit.output_grain_key,
        timestamp_range_chunk=timestamp_range,
        profile_id=unit.profile_id,
        profile_version=unit.profile_version,
        definition_id=definition_id,
        definition_version="definition_schedule_v1",
        planning_status=status,
        mask_reason=mask_reason,
        source_tail_ts=normalized_lineage.get("source_tail_ts") or model_state.get("source_tail_ts"),
        known_at_ts=normalized_lineage.get("known_at_ts") or model_state.get("definition_known_at_ts"),
        relationship_input_tail_ts=relationship_tail["relationship_input_tail_ts"],
        relationship_known_at_ts=relationship_tail["relationship_known_at_ts"],
        relationship_input_metadata=relationship_tail["relationship_input_metadata"],
        planned_output_partition_key=_planned_output_partition_key(
            branch,
            unit.target_key,
            run_id=run_id,
            timestamp_range_chunk=timestamp_range,
            sandbox_root=sandbox_root,
        ),
        lineage={
            "normalized_lineage": normalized_lineage,
            "profile_artifact_path": unit.profile_artifact_path,
            "profile_artifact_hash": unit.profile_artifact_hash,
            "source_planning_unit_id": unit.unit_id,
            "label_planning_only": True,
            "label_assignment_performed": False,
        },
    )


def _planning_status(value: str) -> str:
    return {
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED: LABEL_PLANNING_STATUS_SELECTED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE: LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED: LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY: LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT: LABEL_PLANNING_STATUS_MISSING_INPUT,
    }.get(str(value), LABEL_PLANNING_STATUS_INVALID_PROFILE)


def _mask_reason(unit, status: str) -> str | None:
    if status == LABEL_PLANNING_STATUS_SELECTED:
        return None
    for reason in unit.reason_codes:
        text = str(reason or "").strip()
        if text and text != "model_state_missing_required_fields":
            return text
    if status == LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE:
        return "masked_unavailable"
    if status == LABEL_PLANNING_STATUS_SKIPPED_FILTERED:
        return "skipped_filtered"
    if status == LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY:
        return "diagnostic_only"
    return status


def _timestamp_range_chunk(timestamp_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": "configured_clamp_full_range",
        "timestamp_field": timestamp_plan.get("timestamp_field", "timestamp"),
        "timestamp_rows_materialized": False,
        "timestamp_range_start": None,
        "timestamp_range_end": None,
        "historical_output_months": int(timestamp_plan.get("historical_output_months") or 0),
        "required_lookback_months": int(timestamp_plan.get("required_lookback_months") or 0),
        "clamp_policy_id": timestamp_plan.get("clamp_policy_id"),
        "runtime_boundaries_required": bool(timestamp_plan.get("runtime_boundaries_required", True)),
    }


def _planned_output_partition_key(
    branch: str,
    target_key: Mapping[str, Any],
    *,
    run_id: str,
    timestamp_range_chunk: Mapping[str, Any],
    sandbox_root: str,
) -> dict[str, Any]:
    partition = {"sandbox_output_root": sandbox_root, "run_id": run_id}
    for field in BRANCH_PARTITION_FIELDS[branch]:
        if field == "run_id":
            continue
        partition[field] = target_key.get(field)
    partition["timestamp_chunk_id"] = timestamp_range_chunk.get("chunk_id")
    partition["canonical_output_partition"] = False
    return partition


def _relationship_tail(unit, model_state: Mapping[str, Any]) -> dict[str, Any]:
    if unit.branch != REGIME_BRANCH_CROSS_ASSET_STATE:
        return {
            "relationship_input_tail_ts": None,
            "relationship_known_at_ts": None,
            "relationship_input_metadata": {},
        }
    checks = [to_jsonable(dict(item)) for item in unit.relationship_input_checks]
    return {
        "relationship_input_tail_ts": model_state.get("source_tail_ts"),
        "relationship_known_at_ts": model_state.get("definition_known_at_ts"),
        "relationship_input_metadata": {
            "relationship_input_checks": checks,
            "relationship_input_history_separate_from_selected_profile_artifact": True,
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "snapshot_cadence_days": unit.method_metadata.get("snapshot_cadence_days"),
            "stale_snapshot_policy": to_jsonable(unit.method_metadata.get("stale_snapshot_policy")),
        },
    }


def _target_hash(target_key: Mapping[str, Any]) -> str:
    raw = json.dumps(to_jsonable(dict(target_key)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


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
        return _portable_path_text(value) if _looks_like_absolute_path(value) else value
    return value


def _portable_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<external_configured_root>/{resolved.name}"


def _looks_like_absolute_path(value: str) -> bool:
    try:
        return Path(value).is_absolute()
    except Exception:
        return False


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    if text not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {text!r}")
    return text


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production label planning {field_name} must be non-empty")
    return text


__all__ = [
    "LABEL_ASSIGNMENT_STATUS_NOT_ASSIGNED",
    "LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY",
    "LABEL_PLANNING_STATUS_INVALID_PROFILE",
    "LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE",
    "LABEL_PLANNING_STATUS_MISSING_INPUT",
    "LABEL_PLANNING_STATUS_SELECTED",
    "LABEL_PLANNING_STATUS_SKIPPED_FILTERED",
    "REGIME_PRODUCTION_LABEL_PLANNING_RECORD_ARTIFACT_KIND",
    "REGIME_PRODUCTION_LABEL_PLANNING_SCHEMA_VERSION",
    "REGIME_PRODUCTION_LABEL_PLANNING_SUMMARY_ARTIFACT_KIND",
    "RegimeProductionLabelPlanningConfig",
    "RegimeProductionLabelPlanningRecord",
    "build_label_planning_payload_from_no_write_plan",
    "run_regime_production_label_planning_records",
    "write_regime_production_label_planning_summary",
]
