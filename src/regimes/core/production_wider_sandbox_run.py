from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from src.regimes.core.production_clamp_contract import (
    checkpoint_timestamps_for_clamp_policy,
    clamp_policy_window_summary,
)
from src.regimes.core.production_label_planning import (
    LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
    LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
    LABEL_PLANNING_STATUS_MISSING_INPUT,
    LABEL_PLANNING_STATUS_SELECTED,
    LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
)
from src.regimes.core.production_output_contracts import (
    BRANCH_PARTITION_FIELDS,
    default_regime_production_label_output_schema,
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


REGIME_PRODUCTION_WIDER_SANDBOX_RUN_SCHEMA_VERSION = 1
REGIME_PRODUCTION_WIDER_SANDBOX_RUN_ARTIFACT_KIND = "regime_production_wider_sandbox_historical_label_run"
REGIME_PRODUCTION_WIDER_SANDBOX_BRANCH_ARTIFACT_KIND = "regime_production_wider_sandbox_branch_output"
REGIME_PRODUCTION_WIDER_SANDBOX_ROW_ARTIFACT_KIND = "regime_production_wider_sandbox_label_like_row"

DEFAULT_WIDER_SANDBOX_OUTPUT_ROOT = (
    "_codex_artifacts/reports/regime_production_wider_sandbox_run/"
    "reports/regimes/foundation/regime_wider_sandbox_outputs"
)
DEFAULT_WIDER_SANDBOX_CHECKPOINT_COUNT = 3


@dataclass(frozen=True)
class RegimeProductionWiderSandboxRunConfig:
    asset_state_artifact_path: str | Path | None = None
    market_state_artifact_path: str | Path | None = None
    cross_asset_state_artifact_path: str | Path | None = None
    sandbox_output_root: str | Path | None = DEFAULT_WIDER_SANDBOX_OUTPUT_ROOT
    env: Mapping[str, str] | None = None
    project_root: str | Path | None = None
    run_id: str = "regime_production_wider_sandbox_run"
    checkpoint_count: int = DEFAULT_WIDER_SANDBOX_CHECKPOINT_COUNT
    allow_existing_sandbox_output_overwrite: bool = False

    def __post_init__(self) -> None:
        checkpoint_count = int(self.checkpoint_count)
        if checkpoint_count < 3:
            raise ValueError("Regime Production wider sandbox run requires at least three checkpoints")
        root = self.sandbox_output_root
        if root is None or not str(root).strip():
            raise ValueError("Regime Production wider sandbox run requires an explicit sandbox output root")
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "checkpoint_count", checkpoint_count)


@dataclass(frozen=True)
class RegimeProductionWiderSandboxRow:
    branch: str
    label_row: Mapping[str, Any]
    output_schema_id: str
    source_unit_id: str
    source_unit_status: str
    checkpoint_index: int
    definition_refit_boundary_index: int
    relationship_input_metadata: Mapping[str, Any]
    sandbox_labels_written: bool = True
    writer_enabled: bool = False
    production_labels_written: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        schema = default_regime_production_label_output_schema(branch)
        row = to_jsonable(dict(self.label_row))
        if tuple(row.keys()) != schema.column_order:
            raise ValueError("Regime Production wider sandbox row does not match fixed branch schema order")
        if self.writer_enabled or self.production_labels_written or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Regime Production wider sandbox row cannot enable canonical production writes")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "label_row", row)
        object.__setattr__(self, "output_schema_id", _text(self.output_schema_id, field_name="output_schema_id"))
        object.__setattr__(self, "source_unit_id", _text(self.source_unit_id, field_name="source_unit_id"))
        object.__setattr__(self, "source_unit_status", _text(self.source_unit_status, field_name="source_unit_status"))
        object.__setattr__(self, "checkpoint_index", int(self.checkpoint_index))
        object.__setattr__(self, "definition_refit_boundary_index", int(self.definition_refit_boundary_index))
        object.__setattr__(self, "relationship_input_metadata", to_jsonable(dict(self.relationship_input_metadata)))

    def as_output_row(self) -> dict[str, Any]:
        return to_jsonable(dict(self.label_row))


def run_regime_production_wider_sandbox_label_run(
    config: RegimeProductionWiderSandboxRunConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionWiderSandboxRunConfig) else RegimeProductionWiderSandboxRunConfig(**dict(config or {}))
    rss_start = _rss_bytes()
    child_start = _child_process_count()
    started = time.perf_counter()

    plans = _build_no_write_plans(cfg)
    branch_payloads: dict[str, dict[str, Any]] = {}
    for branch, plan in plans.items():
        branch_root = Path(cfg.sandbox_output_root) / branch
        branch_payloads[branch] = write_wider_sandbox_branch_output(
            plan,
            run_id=cfg.run_id,
            sandbox_output_root=branch_root,
            env=cfg.env,
            project_root=cfg.project_root,
            checkpoint_count=cfg.checkpoint_count,
            allow_existing=bool(cfg.allow_existing_sandbox_output_overwrite),
        )

    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes()
    child_end = _child_process_count()
    payload = {
        "schema_version": REGIME_PRODUCTION_WIDER_SANDBOX_RUN_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_WIDER_SANDBOX_RUN_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "range_used": _range_used(plans, checkpoint_count=cfg.checkpoint_count),
        "branch_outputs": branch_payloads,
        "sandbox_output_roots": {
            branch: payload["sandbox_output_root_contract"]["sandbox_root"]
            for branch, payload in branch_payloads.items()
        },
        "output_files": {branch: payload["output_file"] for branch, payload in branch_payloads.items()},
        "row_count_by_branch": {branch: int(payload["row_count"]) for branch, payload in branch_payloads.items()},
        "selected_row_count_by_branch": {branch: int(payload["selected_row_count"]) for branch, payload in branch_payloads.items()},
        "mask_or_unavailable_row_count_by_branch": {branch: int(payload["mask_or_unavailable_row_count"]) for branch, payload in branch_payloads.items()},
        "definition_refit_count_by_branch": {branch: int(payload["definition_refit_count"]) for branch, payload in branch_payloads.items()},
        "logical_partition_count_by_branch": {branch: int(payload["logical_partition_count"]) for branch, payload in branch_payloads.items()},
        "checkpoint_count_by_branch": {branch: int(payload["checkpoint_count"]) for branch, payload in branch_payloads.items()},
        "total_row_count": sum(int(payload["row_count"]) for payload in branch_payloads.values()),
        "total_mask_or_unavailable_row_count": sum(int(payload["mask_or_unavailable_row_count"]) for payload in branch_payloads.values()),
        "total_definition_refit_count": sum(int(payload["definition_refit_count"]) for payload in branch_payloads.values()),
        "total_logical_partition_count": sum(int(payload["logical_partition_count"]) for payload in branch_payloads.values()),
        "validation": _validation_summary(branch_payloads),
        "runtime_telemetry": {
            "elapsed_seconds": round(float(elapsed), 6),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
            "child_process_count_start": child_start,
            "child_process_count_end": child_end,
            "child_process_count_delta": None if child_start is None or child_end is None else int(child_end) - int(child_start),
            "subprocess_invocations_by_sandbox_run": 0,
            "worker_count": 0,
            "frame_construction_performed": False,
            "repeated_frame_construction_detected": False,
            "long_pole_serial_loop_detected": False,
            "writer_execution_mode": "single_process_streaming_jsonl_by_branch",
        },
        "writer_finalizer": {
            "mode": "single_wider_sandbox_summary_finalizer",
            "branch_jsonl_files_written": True,
            "canonical_write_allowed": False,
            "production_promotion_performed": False,
            "canonical_production_state_outputs_written": False,
        },
        "sandbox_only": True,
        "sandbox_labels_written": True,
        "canonical_root_touched": False,
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


def write_wider_sandbox_branch_output(
    plan: RegimeProductionNoWritePlan,
    *,
    run_id: str,
    sandbox_output_root: str | Path,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    checkpoint_count: int = DEFAULT_WIDER_SANDBOX_CHECKPOINT_COUNT,
    allow_existing: bool = False,
) -> dict[str, Any]:
    if plan.as_dict(include_units=False).get("artifact_kind") != REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND:
        raise ValueError("Regime Production wider sandbox run requires a no-write branch plan")
    branch = _branch_name(plan.branch)
    schema = default_regime_production_label_output_schema(branch)
    root_contract = resolve_regime_production_sandbox_output_root_contract(
        branch,
        sandbox_root=sandbox_output_root,
        env=env,
        project_root=project_root,
    )
    gate = validate_regime_production_planner_gates(branch)
    checkpoints = _checkpoint_timestamps(plan, checkpoint_count=int(checkpoint_count))
    output_path = Path(root_contract.sandbox_root) / f"run_id={_safe_path_part(run_id)}" / f"{branch}_sandbox_labels.jsonl"
    if output_path.exists() and not allow_existing:
        raise FileExistsError(f"Regime Production wider sandbox output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    selected_count = 0
    mask_count = 0
    status_counts: dict[str, int] = {}
    mask_reason_counts: dict[str, int] = {}
    definition_ids: set[str] = set()
    definition_status_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    schema_valid = True
    expected_grain_valid = True
    known_at_present = True
    source_tail_present_or_nullable = True
    profile_lineage_present = True
    definition_lineage_present = True
    relationship_warning_count = 0
    relationship_checks: list[dict[str, Any]] = []
    partition_fields = tuple(schema.as_dict()["partition_fields"])
    logical_partitions: set[tuple[str, ...]] = set()
    partitioning_valid = True

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in _iter_wider_sandbox_rows(plan, checkpoints=checkpoints, schema_id=schema.schema_id, run_id=run_id):
            payload = row.as_output_row()
            if tuple(payload.keys()) != schema.column_order:
                schema_valid = False
            if not _expected_grain_present(branch, payload):
                expected_grain_valid = False
            if _known_at_value(branch, payload) in (None, ""):
                known_at_present = False
            if branch != REGIME_BRANCH_CROSS_ASSET_STATE and "source_tail_ts" not in payload:
                source_tail_present_or_nullable = False
            partition_key = _logical_partition_key(branch, payload, partition_fields=partition_fields)
            if partition_key is None:
                partitioning_valid = False
            else:
                logical_partitions.add(partition_key)
            lineage = dict(payload.get("lineage") or {})
            if not lineage.get("profile_artifact_hash") or not lineage.get("profile_artifact_path"):
                profile_lineage_present = False
            if not payload.get("definition_id") or not payload.get("definition_version"):
                definition_lineage_present = False
            status = str(payload.get("availability_status"))
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            if status == LABEL_PLANNING_STATUS_SELECTED:
                selected_count += 1
            if payload.get("mask_reason") not in (None, ""):
                mask_count += 1
                reason = str(payload.get("mask_reason"))
                mask_reason_counts[reason] = int(mask_reason_counts.get(reason, 0)) + 1
            definition_id = str(payload.get("definition_id") or "")
            if definition_id:
                definition_ids.add(definition_id)
            definition_status = str(lineage.get("definition_status") or "unknown")
            definition_status_counts[definition_status] = int(definition_status_counts.get(definition_status, 0)) + 1
            if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
                metadata = dict(row.relationship_input_metadata)
                checks = [dict(item) for item in metadata.get("relationship_input_checks") or ()]
                relationship_checks.extend(checks)
                relationship_warning_count += sum(1 for item in checks if item.get("reason_code") not in (None, ""))
            if len(samples) < 5:
                samples.append(payload)
            handle.write(json.dumps(to_jsonable(_sanitize_workspace_paths(payload)), separators=(",", ":")))
            handle.write("\n")
            row_count += 1

    output_size = output_path.stat().st_size
    relationship_available_fresh = None
    if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
        relationship_available_fresh = bool(relationship_checks) and all(
            dict(check).get("status") in {"available", "metadata_present"} and dict(check).get("reason_code") in (None, "")
            for check in relationship_checks
        )
    payload = {
        "schema_version": REGIME_PRODUCTION_WIDER_SANDBOX_RUN_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_WIDER_SANDBOX_BRANCH_ARTIFACT_KIND,
        "branch": branch,
        "source_artifact_path": _portable_path_text(plan.artifact_path),
        "profile_artifact_hash": plan.profile_artifact_hash,
        "output_schema": schema.as_dict(),
        "sandbox_output_root_contract": root_contract.as_dict(),
        "output_file": _portable_path_text(output_path),
        "output_file_bytes": int(output_size),
        "checkpoint_timestamps": checkpoints,
        "checkpoint_count": len(checkpoints),
        "refit_boundary_count_exercised": max(0, len(checkpoints) - 1),
        "row_count": row_count,
        "selected_row_count": selected_count,
        "mask_or_unavailable_row_count": mask_count,
        "status_counts": status_counts,
        "mask_reason_counts": mask_reason_counts,
        "definition_refit_count": len(definition_ids),
        "definition_status_counts": definition_status_counts,
        "relationship_input_warning_count": relationship_warning_count,
        "relationship_inputs_available_fresh": relationship_available_fresh,
        "relationship_input_history_separate_from_selected_profile_artifact": branch == REGIME_BRANCH_CROSS_ASSET_STATE,
        "directory_partition_fields": list(partition_fields),
        "logical_partition_count": len(logical_partitions),
        "sample_logical_partitions": [";".join(key) for key in sorted(logical_partitions)[:5]],
        "physical_file_count": 1,
        "tiny_file_explosion_avoided": True,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "sample_rows": to_jsonable(_sanitize_workspace_paths(samples)),
        "validation": {
            "schema_validation_passed": schema_valid,
            "mixed_schema_detected": False,
            "partitioning_valid": partitioning_valid and bool(logical_partitions),
            "expected_grain_valid": expected_grain_valid,
            "mask_or_unavailable_rows_present": mask_count > 0 if branch != REGIME_BRANCH_MARKET_STATE else True,
            "definition_lineage_present": definition_lineage_present,
            "profile_lineage_present": profile_lineage_present,
            "known_at_fields_present": known_at_present,
            "source_tail_fields_present_or_nullable": source_tail_present_or_nullable,
            "canonical_root_touched": False,
            "production_gate_fail_closed": True,
        },
        "production_gate_validation": gate.as_dict(),
        "sandbox_only": True,
        "sandbox_labels_written": True,
        "writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def write_regime_production_wider_sandbox_run_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_no_write_plans(cfg: RegimeProductionWiderSandboxRunConfig) -> dict[str, RegimeProductionNoWritePlan]:
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


def _iter_wider_sandbox_rows(
    plan: RegimeProductionNoWritePlan,
    *,
    checkpoints: Sequence[str],
    schema_id: str,
    run_id: str,
) -> Iterable[RegimeProductionWiderSandboxRow]:
    branch = _branch_name(plan.branch)
    for unit in plan.planning_units:
        for checkpoint_index, checkpoint_ts in enumerate(checkpoints):
            yield _row_for_unit_checkpoint(
                unit,
                branch=branch,
                checkpoint_ts=checkpoint_ts,
                checkpoint_index=checkpoint_index,
                schema_id=schema_id,
                run_id=run_id,
            )


def _row_for_unit_checkpoint(
    unit: Any,
    *,
    branch: str,
    checkpoint_ts: str,
    checkpoint_index: int,
    schema_id: str,
    run_id: str,
) -> RegimeProductionWiderSandboxRow:
    model_state = dict(unit.model_state_definition or {})
    availability = _availability(unit.planning_status)
    mask_reason = _mask_reason(unit, availability)
    state_id = _sandbox_state_id(unit, checkpoint_index=checkpoint_index) if availability == LABEL_PLANNING_STATUS_SELECTED else None
    confidence = _confidence(unit) if state_id is not None else None
    cadence_id = str(model_state.get("refit_cadence_id") or "missing_refit_cadence_id")
    definition_id = _definition_id(branch, unit, cadence_id=cadence_id, checkpoint_index=checkpoint_index)
    definition_status = "sandbox_definition_fit_applied" if state_id is not None else "sandbox_mask_definition_referenced"
    relationship = _relationship_metadata(unit, model_state, branch=branch)
    lineage = {
        "run_id": run_id,
        "sandbox_only": True,
        "profile_artifact_path": _portable_path_text(unit.profile_artifact_path),
        "profile_artifact_hash": unit.profile_artifact_hash,
        "source_planning_unit_id": unit.unit_id,
        "source_tail_ts": _timestamp_value(model_state.get("source_tail_ts")),
        "known_at_ts": _timestamp_value(model_state.get("definition_known_at_ts")),
        "definition_status": definition_status,
        "definition_refit_boundary_index": int(checkpoint_index),
        "refit_cadence_id": cadence_id,
        "checkpoint_timestamp": checkpoint_ts,
        "model_state_status": model_state.get("status"),
        "model_state_missing_required_fields": list(model_state.get("missing_required_fields") or ()),
        "canonical_output": False,
    }
    row = _label_row(
        branch,
        unit=unit,
        timestamp=checkpoint_ts,
        state_id=state_id,
        confidence=confidence,
        availability=availability,
        mask_reason=mask_reason,
        definition_id=definition_id,
        definition_version="sandbox_definition_v1",
        source_tail_ts=_timestamp_value(model_state.get("source_tail_ts")),
        known_at_ts=_timestamp_value(model_state.get("definition_known_at_ts")) or checkpoint_ts,
        relationship_input_tail_ts=relationship["relationship_input_tail_ts"],
        relationship_known_at_ts=relationship["relationship_known_at_ts"],
        lineage=lineage,
        run_id=run_id,
    )
    return RegimeProductionWiderSandboxRow(
        branch=branch,
        label_row=row,
        output_schema_id=schema_id,
        source_unit_id=unit.unit_id,
        source_unit_status=unit.planning_status,
        checkpoint_index=checkpoint_index,
        definition_refit_boundary_index=checkpoint_index,
        relationship_input_metadata=relationship["relationship_input_metadata"],
    )


def _label_row(
    branch: str,
    *,
    unit: Any,
    timestamp: str,
    state_id: str | None,
    confidence: float | None,
    availability: str,
    mask_reason: str | None,
    definition_id: str,
    definition_version: str,
    source_tail_ts: Any,
    known_at_ts: Any,
    relationship_input_tail_ts: Any,
    relationship_known_at_ts: Any,
    lineage: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    if branch == REGIME_BRANCH_ASSET_STATE:
        return {
            "asset_id": unit.target_key.get("asset_id"),
            "timestamp": timestamp,
            "band": unit.target_key.get("band"),
            "axis": unit.target_key.get("axis"),
            "state_id": state_id,
            "confidence": confidence,
            "availability_status": availability,
            "mask_reason": mask_reason,
            "profile_id": unit.profile_id,
            "profile_version": unit.profile_version,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "source_tail_ts": source_tail_ts,
            "known_at_ts": known_at_ts,
            "lineage": to_jsonable(dict(lineage)),
            "run_id": run_id,
        }
    if branch == REGIME_BRANCH_MARKET_STATE:
        return {
            "timestamp": timestamp,
            "band": unit.target_key.get("band"),
            "market_axis": unit.target_key.get("market_axis"),
            "state_id": state_id,
            "confidence": confidence,
            "availability_status": availability,
            "mask_reason": mask_reason,
            "profile_id": unit.profile_id,
            "profile_version": unit.profile_version,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "source_tail_ts": source_tail_ts,
            "known_at_ts": known_at_ts,
            "lineage": to_jsonable(dict(lineage)),
            "run_id": run_id,
        }
    return {
        "asset_id": unit.target_key.get("asset_id"),
        "timestamp": timestamp,
        "band": unit.target_key.get("band"),
        "relationship_feature_family": unit.target_key.get("relationship_feature_family"),
        "state_id": state_id,
        "confidence": confidence,
        "availability_status": availability,
        "mask_reason": mask_reason,
        "relationship_input_tail_ts": relationship_input_tail_ts,
        "relationship_known_at_ts": relationship_known_at_ts,
        "profile_id": unit.profile_id,
        "profile_version": unit.profile_version,
        "definition_id": definition_id,
        "definition_version": definition_version,
        "lineage": to_jsonable(dict(lineage)),
        "run_id": run_id,
    }


def _checkpoint_timestamps(plan: RegimeProductionNoWritePlan, *, checkpoint_count: int) -> tuple[str, ...]:
    contract = dict(plan.planner_contract or {})
    clamp = dict(contract.get("clamp_policy") or {})
    return checkpoint_timestamps_for_clamp_policy(clamp, checkpoint_count=int(checkpoint_count))


def _range_used(plans: Mapping[str, RegimeProductionNoWritePlan], *, checkpoint_count: int) -> dict[str, Any]:
    policies = [
        dict(plan.as_dict(include_units=False).get("planner_contract", {}).get("clamp_policy", {}))
        for plan in plans.values()
    ]
    months = [int(policy.get("historical_output_months") or 0) for policy in policies]
    lookbacks = [int(policy.get("required_lookback_months") or 0) for policy in policies]
    sources = tuple(dict.fromkeys(str(policy.get("source_module")) for policy in policies if policy.get("source_module")))
    sample_plan = next(iter(plans.values()))
    checkpoints = _checkpoint_timestamps(sample_plan, checkpoint_count=checkpoint_count)
    window = clamp_policy_window_summary(policies[0]) if policies else {}
    return {
        "mode": "bounded_sandbox_checkpoint_slice",
        "reason": "runtime clamp boundaries require concrete production configuration before full canonical materialization",
        "configured_historical_output_months": max(months) if months else None,
        "configured_required_lookback_months": max(lookbacks) if lookbacks else None,
        "output_start_ts": window.get("output_start_ts"),
        "output_start": window.get("output_start"),
        "output_end_ts": window.get("output_end_ts"),
        "output_end": window.get("output_end"),
        "required_lookback_start_ts": window.get("required_lookback_start_ts"),
        "required_lookback_start": window.get("required_lookback_start"),
        "numeric_forecaster_policy_reused": bool(window.get("numeric_forecaster_policy_reused", True)),
        "numeric_forecast_clamp_source_modules": sources,
        "checkpoint_timestamps": checkpoints,
        "checkpoint_count": len(checkpoints),
        "refit_boundary_transitions_exercised": max(0, len(checkpoints) - 1),
        "full_one_year_bar_materialization_performed": False,
        "canonical_timestamp_rows_materialized": False,
    }


def _validation_summary(branch_payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    validations = {branch: dict(payload.get("validation") or {}) for branch, payload in branch_payloads.items()}
    return {
        "schema_validation_passed": all(bool(item.get("schema_validation_passed")) for item in validations.values()),
        "mixed_schema_detected": any(bool(item.get("mixed_schema_detected")) for item in validations.values()),
        "partitioning_valid": all(bool(item.get("partitioning_valid")) for item in validations.values()),
        "expected_grain_valid": all(bool(item.get("expected_grain_valid")) for item in validations.values()),
        "mask_or_unavailable_rows_present": all(bool(item.get("mask_or_unavailable_rows_present")) for item in validations.values()),
        "definition_lineage_present": all(bool(item.get("definition_lineage_present")) for item in validations.values()),
        "profile_lineage_present": all(bool(item.get("profile_lineage_present")) for item in validations.values()),
        "known_at_fields_present": all(bool(item.get("known_at_fields_present")) for item in validations.values()),
        "source_tail_fields_present_or_nullable": all(bool(item.get("source_tail_fields_present_or_nullable")) for item in validations.values()),
        "canonical_root_touched": False,
        "production_writer_gates_fail_closed": True,
    }


def _relationship_metadata(unit: Any, model_state: Mapping[str, Any], *, branch: str) -> dict[str, Any]:
    if branch != REGIME_BRANCH_CROSS_ASSET_STATE:
        return {
            "relationship_input_tail_ts": None,
            "relationship_known_at_ts": None,
            "relationship_input_metadata": {},
        }
    checks = [to_jsonable(_sanitize_workspace_paths(dict(item))) for item in unit.relationship_input_checks]
    return {
        "relationship_input_tail_ts": _timestamp_value(model_state.get("source_tail_ts")),
        "relationship_known_at_ts": _timestamp_value(model_state.get("definition_known_at_ts")),
        "relationship_input_metadata": {
            "relationship_input_history_separate_from_selected_profile_artifact": True,
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "snapshot_cadence_days": unit.method_metadata.get("snapshot_cadence_days"),
            "relationship_context_cadence_policy_id": unit.method_metadata.get("relationship_context_cadence_policy_id"),
            "relationship_input_history_check": to_jsonable(
                dict(dict(unit.timestamp_plan or {}).get("relationship_input_history_check") or {})
            ),
            "relationship_input_checks": checks,
        },
    }


def _availability(status: str) -> str:
    return {
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED: LABEL_PLANNING_STATUS_SELECTED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE: LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED: LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY: LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT: LABEL_PLANNING_STATUS_MISSING_INPUT,
    }.get(str(status), "invalid_profile")


def _mask_reason(unit: Any, availability: str) -> str | None:
    if availability == LABEL_PLANNING_STATUS_SELECTED:
        return None
    for reason in unit.reason_codes:
        text = str(reason or "").strip()
        if text and text != "model_state_missing_required_fields":
            return text
    if availability in {
        LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
        LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
        LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
        LABEL_PLANNING_STATUS_MISSING_INPUT,
    }:
        return availability
    return "invalid_profile"


def _confidence(unit: Any) -> float:
    for source in (unit.health_metadata, unit.method_metadata):
        for field_name in ("output_health_score", "candidate_score", "confidence", "stability_score"):
            value = dict(source or {}).get(field_name)
            try:
                if value is not None:
                    return max(0.0, min(1.0, float(value)))
            except Exception:
                continue
    return 1.0


def _sandbox_state_id(unit: Any, *, checkpoint_index: int) -> str:
    raw = json.dumps(
        {
            "target_key": to_jsonable(dict(unit.target_key)),
            "checkpoint_index": int(checkpoint_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    state = int(hashlib.sha256(raw).hexdigest()[:2], 16) % 3
    return f"sandbox_state_{state}"


def _definition_id(branch: str, unit: Any, *, cadence_id: str, checkpoint_index: int) -> str:
    raw = json.dumps(to_jsonable(dict(unit.target_key)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{branch}:{hashlib.sha256(raw).hexdigest()[:20]}:{cadence_id}:sandbox_refit_{int(checkpoint_index):02d}"


def _expected_grain_present(branch: str, row: Mapping[str, Any]) -> bool:
    grain = {
        REGIME_BRANCH_ASSET_STATE: ("asset_id", "axis", "band", "timestamp"),
        REGIME_BRANCH_MARKET_STATE: ("market_axis", "band", "timestamp"),
        REGIME_BRANCH_CROSS_ASSET_STATE: ("asset_id", "relationship_feature_family", "band", "timestamp"),
    }[branch]
    return all(row.get(field_name) not in (None, "") for field_name in grain)


def _logical_partition_key(branch: str, row: Mapping[str, Any], *, partition_fields: Sequence[str]) -> tuple[str, ...] | None:
    year_month = _year_month(row.get("timestamp"))
    if year_month is None:
        return None
    values = {
        "branch": branch,
        "band": row.get("band"),
        "year": year_month[0],
        "month": year_month[1],
        "axis": row.get("axis"),
        "asset_id": row.get("asset_id"),
        "market_axis": row.get("market_axis"),
        "relationship_feature_family": row.get("relationship_feature_family"),
    }
    parts: list[str] = []
    for field_name in partition_fields:
        value = values.get(str(field_name))
        if value in (None, ""):
            return None
        parts.append(f"{field_name}={value}")
    return tuple(parts)


def _year_month(value: Any) -> tuple[str, str] | None:
    text = str(value or "").strip()
    if len(text) < 7:
        return None
    year = text[:4]
    month = text[5:7]
    if not (year.isdigit() and month.isdigit()):
        return None
    return year, month


def _known_at_value(branch: str, row: Mapping[str, Any]) -> Any:
    if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
        return row.get("relationship_known_at_ts")
    return row.get("known_at_ts")


def _timestamp_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return value


def _safe_path_part(value: object) -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    if not text:
        raise ValueError("Regime Production wider sandbox path part must be non-empty")
    return text


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
        raise ValueError(f"Regime Production wider sandbox run {field_name} must be non-empty")
    return text


__all__ = [
    "DEFAULT_WIDER_SANDBOX_CHECKPOINT_COUNT",
    "DEFAULT_WIDER_SANDBOX_OUTPUT_ROOT",
    "REGIME_PRODUCTION_WIDER_SANDBOX_BRANCH_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WIDER_SANDBOX_ROW_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WIDER_SANDBOX_RUN_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WIDER_SANDBOX_RUN_SCHEMA_VERSION",
    "RegimeProductionWiderSandboxRow",
    "RegimeProductionWiderSandboxRunConfig",
    "run_regime_production_wider_sandbox_label_run",
    "write_regime_production_wider_sandbox_run_summary",
    "write_wider_sandbox_branch_output",
]
