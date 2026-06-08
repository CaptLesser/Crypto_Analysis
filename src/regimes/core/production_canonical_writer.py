from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.scalar_features import PARQUET_COMPRESSION, PARQUET_ROW_GROUP
from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path
from src.regimes.asset_state.production_planner import plan_default_asset_state_production_no_write
from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_canonical_readiness_gate import (
    DEFAULT_APPROVAL_SEARCH_ROOTS,
    DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH,
    discover_regime_production_approval_artifacts,
)
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
    resolve_active_selected_profile_artifact,
)
from src.regimes.core.production_idempotency import (
    OUTPUT_COMPLETION_STATUS_COMPLETED,
    RegimeProductionIdempotencyError,
    RegimeProductionIdempotencyKey,
    assert_regime_production_completed_output_reusable,
    evaluate_regime_production_existing_output_manifest,
    stable_payload_hash,
)
from src.regimes.core.production_incremental_planner import (
    plan_regime_production_incremental_update,
)
from src.regimes.core.production_label_materializer import (
    LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE,
    materialize_regime_production_label_rows,
)
from src.regimes.core.production_operator_approval import (
    OPERATOR_CHECKLIST_STATUS_OPERATOR_APPROVED,
    OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT,
    RegimeProductionOperatorChecklistContext,
    validate_regime_production_operator_approval_checklist,
)
from src.regimes.core.production_output_contracts import (
    BRANCH_LABEL_GRAIN_FIELDS,
    CANONICAL_LABEL_OUTPUT_NAMESPACE,
    CANONICAL_MODEL_STATE_NAMESPACE,
    default_regime_production_label_output_schema,
)
from src.regimes.core.production_planner import REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND
from src.regimes.core.production_precanonical_rehearsal import DEFAULT_CANONICAL_RUN_ORDER
from src.regimes.core.production_promotion_gate import (
    DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH,
    RegimeProductionPromotionGateContext,
    evaluate_regime_production_branch_promotion_gate,
)
from src.regimes.core.production_run_lock import (
    REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
    REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE,
    REGIME_PRODUCTION_LOCK_STATUS_RELEASED,
    RegimeProductionRunLockHandle,
    RegimeProductionRunLockTarget,
    acquire_regime_production_run_lock,
    release_regime_production_run_lock,
)
from src.regimes.core.production_wiring_polish import (
    DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH,
    load_regime_production_wiring_config,
    wiring_env,
)
from src.regimes.core.production_write_capable_sandbox import (
    _portable_path_text,
    _safe_path_part,
    _sanitize_workspace_paths,
    _staging_run_root,
    _superseded_run_root,
)
from src.regimes.core.production_write_capable_sandbox import _sha256_file
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.production_planner import plan_default_cross_asset_state_production_no_write
from src.regimes.market_state.production_planner import plan_default_market_state_production_no_write


REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION = 1
REGIME_PRODUCTION_CANONICAL_WRITER_ARTIFACT_KIND = "regime_production_canonical_writer_run"
REGIME_PRODUCTION_CANONICAL_BRANCH_RESULT_ARTIFACT_KIND = "regime_production_canonical_branch_writer_result"
REGIME_PRODUCTION_CANONICAL_MANIFEST_ARTIFACT_KIND = "regime_production_canonical_branch_manifest"
REGIME_PRODUCTION_CANONICAL_ACTIVE_POINTER_ARTIFACT_KIND = "regime_production_canonical_active_pointer"
REGIME_PRODUCTION_CANONICAL_STAGING_STATUS_ARTIFACT_KIND = "regime_production_canonical_staging_status"

CANONICAL_WRITER_STATUS_WRITTEN = "WRITTEN"
CANONICAL_WRITER_STATUS_BLOCKED = "BLOCKED"

DEFAULT_CANONICAL_WRITER_SUMMARY_PATH = (
    "_codex_artifacts/reports/regime_production_canonical_writer/"
    "regime_production_canonical_writer_summary.json"
)


@dataclass(frozen=True)
class RegimeProductionCanonicalWriterConfig:
    branch: str | Sequence[str] = "all"
    wiring_config_path: str | Path = DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH
    sandbox_validation_summary_path: str | Path = DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH
    wider_sandbox_summary_path: str | Path | None = None
    write_capable_sandbox_summary_path: str | Path = DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH
    approval_search_roots: Sequence[str | Path] = DEFAULT_APPROVAL_SEARCH_ROOTS
    branch_approval_paths: Mapping[str, str | Path] | None = None
    operator_checklist_path: str | Path | None = None
    operator_checklist_search_roots: Sequence[str | Path] = DEFAULT_APPROVAL_SEARCH_ROOTS
    require_operator_approval: bool = False
    run_id: str | None = None
    production_writer_enabled: bool = True
    max_logical_partitions_per_branch: int | None = None
    resume: bool = True
    allow_existing_overwrite: bool = False
    summary_output_path: str | Path | None = None
    env: Mapping[str, str] | None = None
    project_root: str | Path | None = None
    approval_scan_max_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.run_id is None:
            stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            object.__setattr__(self, "run_id", f"regime_production_canonical_{stamped}")
        if self.max_logical_partitions_per_branch is not None and int(self.max_logical_partitions_per_branch) <= 0:
            raise ValueError("Regime Production canonical writer max partitions must be positive or None")
        if int(self.approval_scan_max_bytes) <= 0:
            raise ValueError("Regime Production canonical writer approval scan size must be positive")
        object.__setattr__(self, "approval_search_roots", tuple(self.approval_search_roots))
        object.__setattr__(self, "operator_checklist_search_roots", tuple(self.operator_checklist_search_roots))
        object.__setattr__(self, "branch_approval_paths", dict(self.branch_approval_paths or {}))
        object.__setattr__(self, "env", None if self.env is None else dict(self.env))
        if self.max_logical_partitions_per_branch is not None:
            object.__setattr__(self, "max_logical_partitions_per_branch", int(self.max_logical_partitions_per_branch))


def run_regime_production_canonical_writer(
    config: RegimeProductionCanonicalWriterConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (
        config
        if isinstance(config, RegimeProductionCanonicalWriterConfig)
        else RegimeProductionCanonicalWriterConfig(**dict(config or {}))
    )
    started = time.perf_counter()
    wiring = load_regime_production_wiring_config(cfg.wiring_config_path, project_root=cfg.project_root)
    env = _merged_env(wiring, cfg)
    context = _promotion_context_for_run(cfg)
    operator_checklist_path: Path | None = None
    operator_validation: dict[str, Any] = _approval_not_required_validation()
    approval_inventory: dict[str, Any] = _approval_not_required_inventory()
    if cfg.require_operator_approval:
        write_capable = _load_json(cfg.write_capable_sandbox_summary_path, project_root=cfg.project_root)
        checklist_context = RegimeProductionOperatorChecklistContext(
            promotion_context=context,
            write_capable_sandbox_summary=write_capable,
        )
        operator_checklist_path = _resolve_operator_checklist_path(
            explicit_path=cfg.operator_checklist_path,
            search_roots=cfg.operator_checklist_search_roots,
            project_root=cfg.project_root,
            max_bytes=cfg.approval_scan_max_bytes,
        )
        operator_validation = validate_regime_production_operator_approval_checklist(
            operator_checklist_path,
            checklist_context,
            allow_sample_dry_fixture=False,
            env=env,
            project_root=cfg.project_root,
        )
        approval_inventory = discover_regime_production_approval_artifacts(
            cfg.approval_search_roots,
            branch_approval_paths=cfg.branch_approval_paths,
            project_root=cfg.project_root,
            max_bytes=cfg.approval_scan_max_bytes,
        )
        approval_inventory = _canonical_writer_approval_inventory(approval_inventory)
    branches = _requested_branches(cfg.branch)
    preflight = {
        branch: _preflight_branch(
            branch,
            cfg=cfg,
            context=context,
            approval_inventory=approval_inventory,
            operator_validation=operator_validation,
            operator_checklist_path=operator_checklist_path,
            env=env,
            require_operator_approval=bool(cfg.require_operator_approval),
        )
        for branch in branches
    }
    blocked = {branch: result["blockers"] for branch, result in preflight.items() if result["status"] == CANONICAL_WRITER_STATUS_BLOCKED}
    branch_results: dict[str, dict[str, Any]] = {}
    if not blocked:
        for branch in branches:
            branch_results[branch] = write_regime_production_canonical_branch_partitions(
                branch,
                context,
                run_id=str(cfg.run_id),
                approval=dict(approval_inventory.get("branch_approvals", {}).get(branch) or {}),
                canonical_output_root=_canonical_output_root(branch, env=env, project_root=cfg.project_root),
                lock_root=_canonical_lock_root(branch, env=env, project_root=cfg.project_root),
                production_writer_enabled=bool(cfg.production_writer_enabled),
                max_logical_partitions=cfg.max_logical_partitions_per_branch,
                resume=bool(cfg.resume),
                allow_existing_overwrite=bool(cfg.allow_existing_overwrite),
                env=env,
                require_operator_approval=bool(cfg.require_operator_approval),
                project_root=cfg.project_root,
            )
    else:
        branch_results = preflight

    elapsed = time.perf_counter() - started
    any_written = any(result.get("status") == CANONICAL_WRITER_STATUS_WRITTEN for result in branch_results.values())
    payload = {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_CANONICAL_WRITER_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "status": CANONICAL_WRITER_STATUS_BLOCKED if blocked else CANONICAL_WRITER_STATUS_WRITTEN,
        "requested_branches": branches,
        "branch_results": branch_results,
        "preflight": preflight,
        "exact_blockers": blocked,
        "operator_checklist_path": None if operator_checklist_path is None else _portable_path_text(operator_checklist_path),
        "operator_checklist_validation": operator_validation,
        "approval_paths": {
            branch: dict(approval_inventory.get("branch_approval_paths") or {}).get(branch)
            for branch in branches
        },
        "canonical_roots": {
            branch: {
                "output_root": _portable_path_text(_canonical_output_root(branch, env=env, project_root=cfg.project_root)),
                "lock_root": _portable_path_text(_canonical_lock_root(branch, env=env, project_root=cfg.project_root)),
                "model_state_root": _portable_path_text(_model_state_root(branch, env=env, project_root=cfg.project_root)),
            }
            for branch in branches
        },
        "rows_written_by_branch": {
            branch: int(result.get("rows_written") or 0)
            for branch, result in branch_results.items()
        },
        "partition_count_by_branch": {
            branch: int(result.get("logical_partitions_written") or 0)
            for branch, result in branch_results.items()
        },
        "compute_execution_by_branch": {
            branch: _branch_compute_execution(result)
            for branch, result in branch_results.items()
        },
        "runtime_telemetry": {
            "elapsed_seconds": round(float(elapsed), 6),
            "subprocess_invocations_by_writer": 0,
            "writer_execution_mode": "single_process_partitioned_atomic_parquet_canonical_writer",
            "compute_execution_by_branch": {
                branch: _branch_compute_execution(result)
                for branch, result in branch_results.items()
            },
            "full_branch_default": cfg.max_logical_partitions_per_branch is None,
            "operator_approval_required": bool(cfg.require_operator_approval),
            "parent_finalizer_writes_only": True,
            "writer_workers": 1,
        },
        "operator_approval_required": bool(cfg.require_operator_approval),
        "production_writer_requested": bool(cfg.production_writer_enabled),
        "production_writer_enabled": bool(any_written),
        "production_labels_written": bool(any_written),
        "production_outputs_written": bool(any_written),
        "canonical_label_outputs_written": bool(any_written),
        "canonical_root_touched": bool(any_written),
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "test_branch_rerun_performed": False,
        "optuna_or_campaign_run_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
        "cleanup_delete_actions_performed": False,
    }
    payload = to_jsonable(_sanitize_workspace_paths(payload))
    if cfg.summary_output_path is not None:
        write_regime_production_canonical_writer_summary(payload, cfg.summary_output_path, project_root=cfg.project_root)
    return payload


def write_regime_production_canonical_branch_partitions(
    branch: str,
    context: RegimeProductionPromotionGateContext | Mapping[str, Any],
    *,
    run_id: str,
    approval: Mapping[str, Any],
    canonical_output_root: str | Path,
    lock_root: str | Path,
    production_writer_enabled: bool = True,
    max_logical_partitions: int | None = None,
    resume: bool = False,
    allow_existing_overwrite: bool = False,
    env: Mapping[str, str] | None = None,
    require_operator_approval: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    if not production_writer_enabled:
        raise ValueError("Regime Production canonical writer requires production_writer_enabled=True")
    ctx = _context(context)
    gate = (
        evaluate_regime_production_branch_promotion_gate(
            approval,
            ctx,
            branch=branch_name,
            write_sprint_enable_requested=True,
        )
        if require_operator_approval
        else _default_production_branch_gate(branch_name, production_writer_enabled=production_writer_enabled)
    )
    if gate["production_write_preconditions_satisfied"] is not True:
        raise ValueError(f"Regime Production canonical writer gate blocked for {branch_name}: {gate['blockers']}")

    output_root = resolve_project_path(canonical_output_root, project_root=project_root)
    lock_root_path = resolve_project_path(lock_root, project_root=project_root)
    schema = default_regime_production_label_output_schema(branch_name)
    active_pointer_path = output_root / "active_writer_manifest.json"
    plan = _build_canonical_branch_plan(branch_name, env=env)
    materialized = materialize_regime_production_label_rows(
        plan,
        run_id=run_id,
        materialization_mode=LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE,
    )
    all_partition_rows = _canonical_partition_rows(branch_name, {"direct_materialized": materialized.rows})
    selected_canonical_partitions = _select_canonical_partitions(
        all_partition_rows,
        max_logical_partitions=max_logical_partitions,
    )
    canonical_partition_rows = {
        partition_key: rows
        for partition_key, rows in all_partition_rows.items()
        if partition_key in selected_canonical_partitions
    }
    selected_canonical_partitions = set(canonical_partition_rows)
    selected_materialized_rows = [
        dict(row)
        for row in materialized.rows
        if _canonical_partition_key(branch_name, row) in selected_canonical_partitions
    ]
    range_start, range_end = _materialized_lock_range(selected_materialized_rows, selected_canonical_partitions, branch=branch_name)
    source_tail_summary = _source_tail_summary_for_rows(
        materialized.rows,
        branch_name,
        selected_canonical_partitions,
    )
    source_artifact_path = resolve_project_path(plan.artifact_path, project_root=project_root)
    approval_hash = stable_payload_hash(
        approval if require_operator_approval else _default_production_branch_approval_payload(branch_name)
    )
    idempotency_key = RegimeProductionIdempotencyKey(
        branch=branch_name,
        mode=REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
        run_id=run_id,
        output_root=output_root,
        clamp_range=_materialized_clamp_range_payload(materialized.materialization_summary, range_start=range_start, range_end=range_end),
        output_schema_version=schema.schema_version,
        output_schema_hash=schema.schema_hash,
        source_artifact_path=source_artifact_path,
        source_artifact_hash=plan.profile_artifact_hash,
        selected_profile_artifact_hash=plan.profile_artifact_hash,
        approval_artifact_hash=approval_hash,
        source_tail_fingerprint=stable_payload_hash(source_tail_summary),
        selected_partitions=sorted(selected_canonical_partitions),
    )
    idempotency_payload = idempotency_key.as_dict()
    fingerprint = str(idempotency_payload["idempotency_key_hash"])
    run_root = output_root / f"run_id={_safe_path_part(run_id)}"
    manifest_path = run_root / "canonical_writer_manifest.json"
    active_output = _load_active_canonical_output(
        output_root,
        branch=branch_name,
        schema_hash=schema.schema_hash,
        project_root=project_root,
    )
    active_resume: dict[str, Any] = _canonical_resume_not_used("active_output_pointer_absent")
    if manifest_path.exists() and resume:
        existing = _load_json(manifest_path, project_root=project_root)
        try:
            rerun_evaluation = assert_regime_production_completed_output_reusable(
                existing,
                expected_idempotency_key=idempotency_payload,
                expected_schema_hash=schema.schema_hash,
                expected_partition_paths=_expected_canonical_partition_paths(run_root, branch_name, selected_canonical_partitions),
            )
        except RegimeProductionIdempotencyError as exc:
            raise ValueError("Regime Production canonical writer existing output requires recompute or repair") from exc
        return to_jsonable(
            _sanitize_workspace_paths(
                {
                    **existing,
                    "status": CANONICAL_WRITER_STATUS_WRITTEN,
                    "resume_skip_existing": True,
                    "rerun_action": rerun_evaluation["rerun_action"],
                    "rerun_evaluation": rerun_evaluation,
                    "rows_written": 0,
                    "partitions_written_this_run": 0,
                    "atomic_replace_count": 0,
                    "staged_temp_files_created": 0,
                }
            )
        )
    if active_output is not None and resume:
        active_manifest = dict(active_output["manifest"])
        active_manifest_path = Path(active_output["manifest_path"])
        active_expected_paths = _expected_canonical_partition_paths(
            active_manifest_path.parent,
            branch_name,
            selected_canonical_partitions,
        )
        rerun_evaluation = evaluate_regime_production_existing_output_manifest(
            active_manifest,
            expected_idempotency_key=idempotency_payload,
            expected_schema_hash=schema.schema_hash,
            expected_partition_paths=active_expected_paths,
        )
        active_resume = _plan_active_canonical_resume(
            branch_name,
            active_manifest=active_manifest,
            active_manifest_path=active_manifest_path,
            current_rows=selected_materialized_rows,
            current_partitions=selected_canonical_partitions,
            current_source_artifact_hash=plan.profile_artifact_hash,
            current_approval_artifact_hash=approval_hash,
            current_schema_hash=schema.schema_hash,
        )
        if active_resume["active_output_tail_beyond_source_edge"]:
            raise ValueError("Regime Production active output tail exceeds current source edge")
        if bool(active_resume["can_skip_active_completed_output"]):
            return to_jsonable(
                _sanitize_workspace_paths(
                    {
                        **active_manifest,
                        "status": CANONICAL_WRITER_STATUS_WRITTEN,
                        "resume_skip_existing": True,
                        "resume_source": "active_canonical_output_pointer",
                        "resume_source_run_id": active_manifest.get("run_id"),
                        "requested_run_id": run_id,
                        "rerun_action": "skip_completed",
                        "rerun_evaluation": {
                            **rerun_evaluation,
                            "rerun_action": "skip_completed",
                            "can_skip_existing_completed_output": True,
                            "requires_recompute": False,
                            "reject_existing_output": False,
                            "reason_codes": list(
                                dict.fromkeys(
                                    list(rerun_evaluation.get("reason_codes") or ())
                                    + ["active_output_tail_covers_current_range"]
                                )
                            ),
                        },
                        "incremental_resume": _resume_manifest_payload(active_resume),
                        "rows_written": 0,
                        "partitions_written_this_run": 0,
                        "atomic_replace_count": 0,
                        "staged_temp_files_created": 0,
                    }
                )
            )
        if bool(active_resume["can_carry_forward_completed_rows"]):
            selected_materialized_rows = list(active_resume["final_rows"])
            canonical_partition_rows = _canonical_partition_rows(
                branch_name,
                {"active_resume_final_rows": selected_materialized_rows},
            )
            selected_canonical_partitions = set(canonical_partition_rows)
            range_start, range_end = _materialized_lock_range(
                selected_materialized_rows,
                selected_canonical_partitions,
                branch=branch_name,
            )
            source_tail_summary = _source_tail_summary_for_rows(
                selected_materialized_rows,
                branch_name,
                selected_canonical_partitions,
            )
            idempotency_key = RegimeProductionIdempotencyKey(
                branch=branch_name,
                mode=REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
                run_id=run_id,
                output_root=output_root,
                clamp_range=_materialized_clamp_range_payload(
                    materialized.materialization_summary,
                    range_start=range_start,
                    range_end=range_end,
                ),
                output_schema_version=schema.schema_version,
                output_schema_hash=schema.schema_hash,
                source_artifact_path=source_artifact_path,
                source_artifact_hash=plan.profile_artifact_hash,
                selected_profile_artifact_hash=plan.profile_artifact_hash,
                approval_artifact_hash=approval_hash,
                source_tail_fingerprint=stable_payload_hash(source_tail_summary),
                selected_partitions=sorted(selected_canonical_partitions),
            )
            idempotency_payload = idempotency_key.as_dict()
            fingerprint = str(idempotency_payload["idempotency_key_hash"])
        else:
            active_resume = {
                **active_resume,
                "final_rows": [],
            }
    if manifest_path.exists() and not allow_existing_overwrite:
        raise FileExistsError(f"Regime Production canonical writer manifest already exists: {manifest_path}")

    staging_run_root = _staging_run_root(run_root, fingerprint)
    if staging_run_root.exists():
        raise FileExistsError(f"Regime Production canonical writer staging path already exists: {staging_run_root}")
    lock_handle = acquire_regime_production_run_lock(
        RegimeProductionRunLockTarget(
            branch=branch_name,
            mode=REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
            output_root=output_root,
            range_start=range_start,
            range_end=range_end,
        ),
        lock_root=lock_root_path,
        run_id=run_id,
        owner="regime_production_canonical_finalizer",
        project_root=project_root,
        production_writer_enabled=True,
        canonical_write_execution_allowed=True,
    )
    try:
        staging_run_root.mkdir(parents=True, exist_ok=False)
        _write_canonical_staging_status(
            staging_run_root,
            status="staging",
            branch=branch_name,
            run_id=run_id,
            final_run_root=run_root,
            active_pointer_path=active_pointer_path,
            reason=None,
        )
        rows_written = 0
        atomic_count = 0
        temp_count = 0
        schema_valid = True
        partition_files: list[dict[str, Any]] = []
        for partition_key, rows in sorted(canonical_partition_rows.items()):
            relative_partition_path = _canonical_partition_path(branch_name, partition_key)
            staging_partition_path = staging_run_root / relative_partition_path
            final_partition_path = run_root / relative_partition_path
            staging_partition_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = sibling_temp_path(staging_partition_path, suffix=".parquet.tmp")
            temp_count += 1
            for row in rows:
                if tuple(row.keys()) != schema.column_order:
                    schema_valid = False
            _write_parquet_partition(tmp, rows, schema=schema)
            atomic_replace(tmp, staging_partition_path)
            atomic_count += 1
            rows_written += len(rows)
            partition_files.append(
                {
                    "partition_key": partition_key,
                    "path": _portable_path_text(final_partition_path),
                    "staging_path": _portable_path_text(staging_partition_path),
                    "row_count": len(rows),
                    "file_bytes": staging_partition_path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_CANONICAL_MANIFEST_ARTIFACT_KIND,
            "status": CANONICAL_WRITER_STATUS_WRITTEN,
            "branch": branch_name,
            "run_id": run_id,
            "write_fingerprint": fingerprint,
            "source_file": None,
            "source_profile_artifact_path": _portable_path_text(source_artifact_path),
            "source_file_hash": idempotency_payload["source_artifact_hash"],
            "source_profile_artifact_hash": plan.profile_artifact_hash,
            "source_plan_artifact_kind": REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND,
            "source_plan_unit_count": len(plan.planning_units),
            "label_materialization": materialized.materialization_summary,
            "compute_execution": dict(materialized.materialization_summary.get("execution") or {}),
            "worker_profile_honored_by_materializer": bool(
                materialized.materialization_summary.get("worker_profile_honored")
            ),
            "parent_finalizer_writes_only": True,
            "workers_compute_only": True,
            "workers_write_outputs": False,
            "source_tail_summary": source_tail_summary,
            "incremental_resume": _resume_manifest_payload(active_resume),
            "canonical_output_root_contract": _canonical_output_root_contract(branch_name, output_root),
            "output_schema": schema.as_dict(),
            "partition_fields": list(schema.as_dict()["partition_fields"]),
            "source_logical_partition_fields": list(schema.as_dict()["partition_fields"]),
            "lock_range": {
                "range_start": range_start,
                "range_end": range_end,
            },
            "run_lock": lock_handle.as_dict(),
            "idempotency": {
                **idempotency_payload,
                "resume_supported": True,
                "resume_fingerprint": fingerprint,
                "allow_existing_overwrite": bool(allow_existing_overwrite),
                "rerun_policy": "skip_completed_when_idempotency_key_schema_and_partitions_match",
                "atomic_replace_policy": "allow_existing_overwrite_required_for_recompute",
            },
            "logical_partitions_written": len(partition_files),
            "physical_partition_files_written": len(partition_files),
            "rows_written": rows_written,
            "partition_files": partition_files,
            "storage_format": "parquet",
            "json_object_storage_policy": "canonical_json_string_for_parquet_dtype_stability",
            "schema_validation_passed": schema_valid,
            "mixed_schema_detected": False,
            "writer_gating": {
                "production_writer_enabled": True,
                "branch_promotion_gate_status": gate["gate_status"],
                "dry_write_planning_allowed": gate["dry_write_planning_allowed"],
                "production_write_preconditions_satisfied": gate["production_write_preconditions_satisfied"],
                "canonical_write_execution_allowed": True,
                "operator_approval_required": bool(require_operator_approval),
            },
            "atomic_staged_write": {
                "atomic_replace": True,
                "staging_root": _portable_path_text(staging_run_root),
                "final_run_root": _portable_path_text(run_root),
                "active_pointer_path": _portable_path_text(active_pointer_path),
                "staged_temp_files_created": temp_count,
                "atomic_replace_count": atomic_count,
                "finalizer_manifest_written_last": True,
                "schema_validation_before_commit": True,
                "row_count_validation_before_commit": True,
                "commit_policy": "staged_directory_rename_then_active_pointer_update",
                "failed_staging_remains_non_active": True,
                "active_pointer_updated_after_validation_only": True,
            },
            "completion_status": OUTPUT_COMPLETION_STATUS_COMPLETED,
            "manifest_status": OUTPUT_COMPLETION_STATUS_COMPLETED,
            "writer_finalization_status": OUTPUT_COMPLETION_STATUS_COMPLETED,
            "partial_output_marker": False,
            "resume_skip_existing": False,
            "canonical_root_touched": True,
            "production_writer_enabled": True,
            "production_labels_written": True,
            "production_outputs_written": True,
            "canonical_label_outputs_written": True,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "test_branch_rerun_performed": False,
            "optuna_or_campaign_run_performed": False,
            "relationship_discovery_or_pairwise_run_performed": False,
        }
        _validate_canonical_staged_branch_write(
            staging_run_root,
            manifest,
            partition_rows=canonical_partition_rows,
            selected_partitions=selected_canonical_partitions,
            schema_valid=schema_valid,
        )
        _write_canonical_staging_status(
            staging_run_root,
            status="validated",
            branch=branch_name,
            run_id=run_id,
            final_run_root=run_root,
            active_pointer_path=active_pointer_path,
            reason=None,
        )
        _write_json_atomic(staging_run_root / "canonical_writer_manifest.json", manifest)
        commit_info = _commit_canonical_staged_run_root(
            staging_run_root,
            run_root,
            allow_existing_overwrite=allow_existing_overwrite,
            fingerprint=fingerprint,
        )
        active_pointer = _write_canonical_active_pointer(
            active_pointer_path,
            manifest,
            manifest_path=manifest_path,
            commit_info=commit_info,
        )
        lock_release = release_regime_production_run_lock(
            lock_handle,
            status=REGIME_PRODUCTION_LOCK_STATUS_RELEASED,
            reason="success",
            production_writer_enabled=True,
            canonical_write_execution_allowed=True,
            canonical_root_touched=True,
            production_outputs_written=True,
            production_labels_written=True,
            canonical_label_outputs_written=True,
        )
        return to_jsonable(
            _sanitize_workspace_paths(
                {
                    **manifest,
                    "manifest_path": _portable_path_text(manifest_path),
                    "staged_commit": commit_info,
                    "active_pointer": active_pointer,
                    "run_lock_release": lock_release,
                }
            )
        )
    except Exception as exc:
        if staging_run_root.exists():
            _write_canonical_staging_status(
                staging_run_root,
                status="failed_non_active",
                branch=branch_name,
                run_id=run_id,
                final_run_root=run_root,
                active_pointer_path=active_pointer_path,
                reason=str(exc),
            )
        _mark_lock_failed_recoverable(lock_handle)
        raise


def write_regime_production_canonical_writer_summary(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_CANONICAL_WRITER_SUMMARY_PATH,
    *,
    project_root: str | Path | None = None,
) -> Path:
    path = resolve_project_path(output_path, project_root=project_root)
    _write_json_atomic(path, payload)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_regime_production_canonical_writer(_config_from_args(args))
    except Exception as exc:
        print(json.dumps({"status": CANONICAL_WRITER_STATUS_BLOCKED, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == CANONICAL_WRITER_STATUS_WRITTEN else 2


def main_for_branch(branch: str, argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser(default_branch=branch)
    args = parser.parse_args(argv)
    args.branch = _branch_name(branch)
    try:
        payload = run_regime_production_canonical_writer(_config_from_args(args))
    except Exception as exc:
        print(json.dumps({"status": CANONICAL_WRITER_STATUS_BLOCKED, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == CANONICAL_WRITER_STATUS_WRITTEN else 2


def _preflight_branch(
    branch: str,
    *,
    cfg: RegimeProductionCanonicalWriterConfig,
    context: RegimeProductionPromotionGateContext,
    approval_inventory: Mapping[str, Any],
    operator_validation: Mapping[str, Any],
    operator_checklist_path: Path | None,
    env: Mapping[str, str],
    require_operator_approval: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not cfg.production_writer_enabled:
        blockers.append("production_writer_disabled")
    if (
        require_operator_approval
        and operator_validation.get("validation_status") != OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT
    ):
        blockers.append(f"operator_checklist:{operator_validation.get('validation_status') or 'missing'}")
    if require_operator_approval and operator_checklist_path is None:
        blockers.append("missing_operator_approval_checklist")
    approval = dict(dict(approval_inventory.get("branch_approvals") or {}).get(branch) or {})
    approval_path = dict(approval_inventory.get("branch_approval_paths") or {}).get(branch)
    gate = (
        evaluate_regime_production_branch_promotion_gate(
            approval or None,
            context,
            branch=branch,
            write_sprint_enable_requested=bool(cfg.production_writer_enabled),
        )
        if require_operator_approval
        else _default_production_branch_gate(branch, production_writer_enabled=cfg.production_writer_enabled)
    )
    if require_operator_approval:
        blockers.extend(f"approval:{reason}" for reason in gate.get("blockers", ()) or ())
    try:
        branch_plan = _build_canonical_branch_plan(branch, env=env)
    except Exception as exc:
        branch_plan = None
        blockers.append(f"branch_plan:{type(exc).__name__}:{exc}")
    active = _active_artifact_check(
        branch,
        context,
        env=env,
        project_root=cfg.project_root,
        expected_artifact_path=None if branch_plan is None else branch_plan.artifact_path,
        expected_artifact_hash=None if branch_plan is None else branch_plan.profile_artifact_hash,
        require_context_expected=bool(require_operator_approval),
    )
    blockers.extend(active["blockers"])
    try:
        output_root = _canonical_output_root(branch, env=env, project_root=cfg.project_root)
    except Exception as exc:
        output_root = None
        blockers.append(f"canonical_output_root:{type(exc).__name__}:{exc}")
    try:
        lock_root = _canonical_lock_root(branch, env=env, project_root=cfg.project_root)
    except Exception as exc:
        lock_root = None
        blockers.append(f"write_lock_root:{type(exc).__name__}:{exc}")
    blockers = _stable_codes(blockers)
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_CANONICAL_BRANCH_RESULT_ARTIFACT_KIND,
        "status": CANONICAL_WRITER_STATUS_BLOCKED if blockers else "PREFLIGHT_PASS",
        "branch": branch,
        "blockers": blockers,
        "approval_path": approval_path,
        "operator_checklist_path": None if operator_checklist_path is None else _portable_path_text(operator_checklist_path),
        "approval_gate": gate,
        "active_selected_profile_artifact": active,
        "canonical_output_root": None if output_root is None else _portable_path_text(output_root),
        "write_lock_root": None if lock_root is None else _portable_path_text(lock_root),
        "source_profile_artifact_path": None if branch_plan is None else _portable_path_text(branch_plan.artifact_path),
        "source_profile_artifact_hash": None if branch_plan is None else branch_plan.profile_artifact_hash,
        "source_plan_status": None if branch_plan is None else branch_plan.safety_status,
        "source_plan_unit_count": 0 if branch_plan is None else len(branch_plan.planning_units),
        "rows_written": 0,
        "logical_partitions_written": 0,
        "production_writer_requested": bool(cfg.production_writer_enabled),
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_root_touched": False,
        "operator_approval_required": bool(require_operator_approval),
    }


def _build_canonical_branch_plan(branch: str, *, env: Mapping[str, str] | None):
    branch_name = _branch_name(branch)
    if branch_name == REGIME_BRANCH_ASSET_STATE:
        return plan_default_asset_state_production_no_write(expected_cell_count=3204, env=env)
    if branch_name == REGIME_BRANCH_MARKET_STATE:
        return plan_default_market_state_production_no_write(env=env)
    if branch_name == REGIME_BRANCH_CROSS_ASSET_STATE:
        return plan_default_cross_asset_state_production_no_write(env=env)
    raise ValueError(f"Unsupported Regime Production branch: {branch!r}")


def _promotion_context_for_run(cfg: RegimeProductionCanonicalWriterConfig) -> RegimeProductionPromotionGateContext:
    if not cfg.require_operator_approval:
        return RegimeProductionPromotionGateContext(sandbox_validation_summary={}, wider_sandbox_summary={})
    return RegimeProductionPromotionGateContext.from_paths(
        sandbox_validation_summary_path=cfg.sandbox_validation_summary_path,
        wider_sandbox_summary_path=cfg.wider_sandbox_summary_path,
    )


def _approval_not_required_validation() -> dict[str, Any]:
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_operator_approval_checklist_validation",
        "validation_status": "not_required_for_default_production_branch_run",
        "blockers": [],
        "warnings": [],
        "checklist_id": None,
        "checklist_status": None,
        "operator_approval_executed": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_write_execution_allowed": True,
    }


def _approval_not_required_inventory() -> dict[str, Any]:
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_default_branch_run_approval_inventory",
        "branch_approvals": {},
        "branch_approval_paths": {},
        "branch_approval_artifacts_present": {branch: False for branch in REGIME_PRODUCTION_BRANCHES},
        "approval_required": False,
    }


def _default_production_branch_gate(branch: str, *, production_writer_enabled: bool = True) -> dict[str, Any]:
    writer_enabled = bool(production_writer_enabled)
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_promotion_gate_result",
        "branch": _branch_name(branch),
        "gate_status": "accepted_for_default_production_branch_write",
        "blockers": [],
        "warnings": [],
        "per_branch_approval_required": False,
        "operator_approval_required": False,
        "dry_write_planning_allowed": True,
        "production_write_preconditions_satisfied": writer_enabled,
        "canonical_write_execution_allowed": writer_enabled,
        "write_sprint_enable_requested": writer_enabled,
        "write_sprint_enable_required_before_canonical_writes": False,
        "production_writer_enabled": writer_enabled,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "production_writer_gates_fail_closed": True,
    }


def _default_production_branch_approval_payload(branch: str) -> dict[str, Any]:
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_default_branch_command_write_authorization",
        "branch": _branch_name(branch),
        "authorization_source": "default_production_branch_command",
        "operator_approval_required": False,
    }


def _active_artifact_check(
    branch: str,
    context: RegimeProductionPromotionGateContext,
    *,
    env: Mapping[str, str],
    project_root: str | Path | None,
    expected_artifact_path: str | Path | None = None,
    expected_artifact_hash: str | None = None,
    require_context_expected: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    active = resolve_active_selected_profile_artifact(branch, env=env, project_root=project_root)
    active_path = active.artifact_path
    active_hash = _sha256_file(active_path) if active.passed and active_path is not None else None
    branch_output = dict(context.wider_sandbox_summary.get("branch_outputs", {}).get(branch) or {})
    expected_path = expected_artifact_path or branch_output.get("source_artifact_path")
    expected_hash = expected_artifact_hash or branch_output.get("profile_artifact_hash")
    path_match = None
    if active_path is not None and expected_path:
        path_match = resolve_project_path(expected_path, project_root=project_root).resolve() == Path(active_path).resolve()
    if not active.passed:
        blockers.extend(f"active_artifact:{reason}" for reason in active.reason_codes)
    if expected_path and path_match is not True:
        blockers.append("active_selected_profile_artifact_path_mismatch")
    if expected_hash and active_hash != expected_hash:
        blockers.append("active_selected_profile_artifact_hash_mismatch")
    if require_context_expected and not branch_output:
        blockers.append("supervised_context_branch_output_missing")
    if branch == REGIME_BRANCH_ASSET_STATE and (active_path is None or "lineage_repaired" not in active_path.name):
        blockers.append("repaired_asset_artifact_not_active")
    return {
        "passed": not blockers,
        "status": active.status,
        "path": None if active_path is None else _portable_path_text(active_path),
        "hash": active_hash,
        "expected_path": None if expected_path is None else _portable_path_text(expected_path),
        "expected_hash": expected_hash,
        "path_matches_expected": path_match,
        "hash_matches_expected": None if not expected_hash else active_hash == expected_hash,
        "repaired_asset_artifact_active": bool(branch == REGIME_BRANCH_ASSET_STATE and active_path is not None and "lineage_repaired" in active_path.name),
        "blockers": _stable_codes(blockers),
    }


def _select_canonical_partitions(
    partition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_logical_partitions: int | None,
) -> set[str]:
    if max_logical_partitions is None or len(partition_rows) <= int(max_logical_partitions):
        return set(partition_rows)
    status_by_partition = {
        partition_key: {str(row.get("availability_status")) for row in rows}
        for partition_key, rows in partition_rows.items()
    }
    selected = sorted(key for key, statuses in status_by_partition.items() if "selected" in statuses)
    masked = sorted(key for key, statuses in status_by_partition.items() if statuses - {"selected"})
    out: list[str] = []

    def _append(keys: Sequence[str]) -> bool:
        for key in keys:
            if key not in out:
                out.append(key)
            if len(out) >= int(max_logical_partitions):
                return True
        return False

    if masked:
        _append(masked[:1])
    if selected:
        _append(selected[:1])
    if len(out) >= int(max_logical_partitions):
        return set(out)
    if _append(masked):
        return set(out)
    if _append(selected):
        return set(out)
    for key in sorted(partition_rows):
        if key not in out:
            out.append(key)
        if len(out) >= int(max_logical_partitions):
            return set(out)
    return set(out)


def _canonical_resume_not_used(reason: str) -> dict[str, Any]:
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_canonical_incremental_resume_plan",
        "resume_considered": False,
        "resume_source": None,
        "status": "resume_not_used",
        "reason_codes": [str(reason)],
        "active_pointer_consulted": False,
        "can_skip_active_completed_output": False,
        "can_carry_forward_completed_rows": False,
        "active_output_tail_beyond_source_edge": False,
        "rows_reused_from_active_output": 0,
        "rows_materialized_for_new_or_recompute_range": 0,
        "final_cumulative_row_count": 0,
    }


def _load_active_canonical_output(
    output_root: Path,
    *,
    branch: str,
    schema_hash: str,
    project_root: str | Path | None,
) -> dict[str, Any] | None:
    pointer_path = output_root / "active_writer_manifest.json"
    if not pointer_path.exists():
        return None
    pointer = _load_json(pointer_path, project_root=project_root)
    if pointer.get("branch") != _branch_name(branch):
        raise ValueError("Regime Production active pointer branch mismatch")
    if bool(pointer.get("partial_output_marker")):
        raise ValueError("Regime Production active pointer is partial")
    if pointer.get("completion_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        raise ValueError("Regime Production active pointer is not completed")
    if pointer.get("manifest_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        raise ValueError("Regime Production active pointer manifest status is not completed")
    if pointer.get("output_schema_hash") != schema_hash:
        raise ValueError("Regime Production active pointer schema hash mismatch")
    active_run_id = _text(pointer.get("run_id"), field_name="active pointer run_id")
    manifest_path = output_root / f"run_id={_safe_path_part(active_run_id)}" / "canonical_writer_manifest.json"
    if not manifest_path.exists():
        raw_manifest_path = pointer.get("manifest_path")
        if raw_manifest_path not in (None, ""):
            candidate = resolve_project_path(raw_manifest_path, project_root=project_root)
            if candidate.exists():
                manifest_path = candidate
    if not manifest_path.exists():
        raise FileNotFoundError("Regime Production active pointer manifest is missing")
    manifest = _load_json(manifest_path, project_root=project_root)
    _validate_completed_canonical_manifest(manifest, branch=branch, schema_hash=schema_hash)
    return {
        "pointer": pointer,
        "pointer_path": pointer_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def _validate_completed_canonical_manifest(
    manifest: Mapping[str, Any],
    *,
    branch: str,
    schema_hash: str,
) -> None:
    if manifest.get("branch") != _branch_name(branch):
        raise ValueError("Regime Production active manifest branch mismatch")
    if bool(manifest.get("partial_output_marker")):
        raise ValueError("Regime Production active manifest is partial")
    if manifest.get("completion_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        raise ValueError("Regime Production active manifest completion status is not completed")
    if manifest.get("manifest_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        raise ValueError("Regime Production active manifest status is not completed")
    if manifest.get("writer_finalization_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        raise ValueError("Regime Production active manifest finalization status is not completed")
    if dict(manifest.get("output_schema") or {}).get("schema_hash") != schema_hash:
        raise ValueError("Regime Production active manifest schema hash mismatch")


def _plan_active_canonical_resume(
    branch: str,
    *,
    active_manifest: Mapping[str, Any],
    active_manifest_path: Path,
    current_rows: Sequence[Mapping[str, Any]],
    current_partitions: set[str],
    current_source_artifact_hash: str,
    current_approval_artifact_hash: str,
    current_schema_hash: str,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    current = [_schema_ordered_row(branch_name, row) for row in current_rows]
    active_rows = _load_active_manifest_rows(active_manifest, active_manifest_path, branch=branch_name)
    active_completed_rows = [row for row in active_rows if not _row_needs_recompute(row)]
    current_min, current_max = _timestamp_range_epoch(current)
    active_physical_tail = _max_row_timestamp(active_rows)
    active_completed_tail = _max_row_timestamp(active_completed_rows)
    active_tail_beyond_source_edge = (
        active_physical_tail is not None
        and current_max is not None
        and int(active_physical_tail) > int(current_max)
    )
    source_compatible, source_reasons = _active_manifest_source_compatible(
        active_manifest,
        branch=branch_name,
        current_source_artifact_hash=current_source_artifact_hash,
        current_approval_artifact_hash=current_approval_artifact_hash,
        current_schema_hash=current_schema_hash,
    )
    contiguous_write_tail = (
        None
        if active_tail_beyond_source_edge or not source_compatible
        else _contiguous_completed_tail(branch_name, expected_rows=current, active_completed_rows=active_rows)
    )
    contiguous_completed_tail = (
        None
        if active_tail_beyond_source_edge or not source_compatible
        else _contiguous_completed_tail(branch_name, expected_rows=current, active_completed_rows=active_completed_rows)
    )
    current_covered = (
        contiguous_write_tail is not None
        and current_max is not None
        and int(contiguous_write_tail) >= int(current_max)
    )
    new_rows = [
        row
        for row in current
        if contiguous_write_tail is None or _timestamp_epoch(row.get("timestamp")) > int(contiguous_write_tail)
    ]
    current_keys = {_label_row_key(branch_name, row) for row in current}
    carry_rows = [
        _schema_ordered_row(branch_name, row)
        for row in active_rows
        if contiguous_write_tail is not None
        and _label_row_key(branch_name, row) in current_keys
        and _timestamp_epoch(row.get("timestamp")) <= int(contiguous_write_tail)
    ]
    final_rows = _merge_resume_rows(branch_name, carry_rows=carry_rows, new_rows=new_rows)
    incremental_payload = _incremental_resume_payload(
        branch_name,
        source_tail_ts=current_max,
        last_output_tail_ts=contiguous_write_tail,
        current_rows=current,
        active_rows=active_rows,
    )
    status = "skip_active_completed_output" if current_covered else "incremental_carry_forward"
    if not source_compatible:
        status = "full_recompute_source_or_schema_changed"
    if contiguous_write_tail is None and source_compatible:
        status = "full_recompute_no_contiguous_completed_tail"
    return to_jsonable(
        {
            "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_canonical_incremental_resume_plan",
            "resume_considered": True,
            "resume_source": "active_canonical_output_pointer",
            "status": status,
            "active_manifest_run_id": active_manifest.get("run_id"),
            "active_manifest_path": _portable_path_text(active_manifest_path),
            "active_pointer_consulted": True,
            "source_compatible": bool(source_compatible),
            "source_compatibility_reason_codes": list(source_reasons),
            "current_partition_count": len(current_partitions),
            "current_range_start_ts": current_min,
            "current_range_end_ts": current_max,
            "active_physical_output_tail_ts": active_physical_tail,
            "active_completed_physical_tail_ts": active_completed_tail,
            "active_completed_contiguous_tail_ts": contiguous_completed_tail,
            "active_write_contiguous_tail_ts": contiguous_write_tail,
            "active_output_tail_beyond_source_edge": bool(active_tail_beyond_source_edge),
            "can_skip_active_completed_output": bool(source_compatible and current_covered),
            "can_carry_forward_completed_rows": bool(source_compatible and contiguous_write_tail is not None and not current_covered),
            "rows_reused_from_active_output": len(carry_rows),
            "rows_materialized_for_new_or_recompute_range": len(new_rows),
            "final_cumulative_row_count": len(final_rows),
            "incremental_update_plan": incremental_payload,
            "final_rows": final_rows,
            "reason_codes": _resume_reason_codes(
                source_compatible=source_compatible,
                source_reasons=source_reasons,
                active_tail_beyond_source_edge=active_tail_beyond_source_edge,
                contiguous_tail=contiguous_write_tail,
                current_covered=current_covered,
            ),
            "numeric_resume_standard_alignment": {
                "active_output_tail_detected_before_write": True,
                "physical_write_tail_includes_recompute_rows": True,
                "completed_tail_excludes_recompute_rows": True,
                "source_edge_compared_to_output_tail": True,
                "same_inputs_skip_completed_output": bool(source_compatible and current_covered),
                "incremental_range_starts_after_completed_tail": bool(
                    source_compatible and contiguous_write_tail is not None and not current_covered
                ),
                "partial_output_cannot_masquerade_as_complete": True,
            },
        }
    )


def _resume_reason_codes(
    *,
    source_compatible: bool,
    source_reasons: Sequence[str],
    active_tail_beyond_source_edge: bool,
    contiguous_tail: int | None,
    current_covered: bool,
) -> list[str]:
    reasons: list[str] = []
    if not source_compatible:
        reasons.extend(source_reasons)
    if active_tail_beyond_source_edge:
        reasons.append("active_output_tail_beyond_current_source_edge")
    if contiguous_tail is None:
        reasons.append("no_active_contiguous_completed_tail")
    elif current_covered:
        reasons.append("active_output_tail_covers_current_range")
    else:
        reasons.append("new_rows_after_active_output_tail")
    return list(dict.fromkeys(reasons))


def _resume_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in dict(payload).items() if key != "final_rows"}
    return to_jsonable(out)


def _active_manifest_source_compatible(
    manifest: Mapping[str, Any],
    *,
    branch: str,
    current_source_artifact_hash: str,
    current_approval_artifact_hash: str,
    current_schema_hash: str,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if manifest.get("branch") != _branch_name(branch):
        reasons.append("branch_mismatch")
    schema_hash = dict(manifest.get("output_schema") or {}).get("schema_hash")
    if schema_hash != current_schema_hash:
        reasons.append("output_schema_hash_mismatch")
    existing_source_hash = (
        manifest.get("source_profile_artifact_hash")
        or manifest.get("source_file_hash")
        or dict(manifest.get("idempotency") or {}).get("source_artifact_hash")
    )
    if existing_source_hash != current_source_artifact_hash:
        reasons.append("selected_profile_artifact_hash_mismatch")
    existing_approval_hash = dict(manifest.get("idempotency") or {}).get("approval_artifact_hash")
    if existing_approval_hash != current_approval_artifact_hash:
        reasons.append("approval_artifact_hash_mismatch")
    return not reasons, tuple(dict.fromkeys(reasons))


def _load_active_manifest_rows(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    branch: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in tuple(dict(part) for part in manifest.get("partition_files") or ()):
        partition_key = _text(item.get("partition_key"), field_name="active partition key")
        path = manifest_path.parent / _canonical_partition_path(branch, partition_key)
        if not path.exists():
            raw_path = item.get("path")
            if raw_path not in (None, ""):
                candidate = Path(str(raw_path))
                if not candidate.is_absolute():
                    candidate = resolve_project_path(candidate)
                if candidate.exists():
                    path = candidate
        if not path.is_file():
            raise FileNotFoundError(f"Regime Production active partition missing: {path}")
        frame = pd.read_parquet(path)
        schema = default_regime_production_label_output_schema(branch)
        if list(frame.columns) != list(schema.column_order):
            raise ValueError("Regime Production active partition schema mismatch")
        rows.extend(_normalize_existing_output_row(branch, record) for record in frame.to_dict(orient="records"))
    return rows


def _normalize_existing_output_row(branch: str, record: Mapping[str, Any]) -> dict[str, Any]:
    schema = default_regime_production_label_output_schema(branch)
    dtype_by_column = schema.dtype_by_column
    out: dict[str, Any] = {}
    for column in schema.column_order:
        value = record.get(column)
        if _nullish(value):
            out[column] = None
            continue
        logical_dtype = str(dtype_by_column.get(column) or "string")
        if logical_dtype == "timestamp_utc":
            out[column] = _iso_timestamp_text(value)
        elif logical_dtype == "float64":
            out[column] = float(value)
        elif logical_dtype == "json_object":
            out[column] = _json_object_value(value, field_name=column)
        else:
            out[column] = str(value)
    return out


def _json_object_value(value: Any, *, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return to_jsonable(dict(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Regime Production active JSON column is not valid JSON: {field_name}") from exc
    return to_jsonable(value)


def _schema_ordered_row(branch: str, row: Mapping[str, Any]) -> dict[str, Any]:
    schema = default_regime_production_label_output_schema(branch)
    return {column: row.get(column) for column in schema.column_order}


def _row_needs_recompute(row: Mapping[str, Any]) -> bool:
    lineage = row.get("lineage")
    payload = _json_object_value(lineage, field_name="lineage") if lineage not in (None, "") else {}
    if not isinstance(payload, Mapping):
        return False
    return bool(payload.get("needs_recompute")) or bool(payload.get("is_forward_filled"))


def _contiguous_completed_tail(
    branch: str,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    active_completed_rows: Sequence[Mapping[str, Any]],
) -> int | None:
    expected_by_ts: dict[int, set[tuple[Any, ...]]] = {}
    for row in expected_rows:
        ts = _timestamp_epoch(row.get("timestamp"))
        expected_by_ts.setdefault(ts, set()).add(_label_row_key(branch, row))
    completed_keys = {_label_row_key(branch, row) for row in active_completed_rows}
    tail: int | None = None
    for ts in sorted(expected_by_ts):
        if expected_by_ts[ts].issubset(completed_keys):
            tail = int(ts)
            continue
        break
    return tail


def _label_row_key(branch: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in BRANCH_LABEL_GRAIN_FIELDS[_branch_name(branch)]:
        if field == "timestamp":
            values.append(_timestamp_epoch(row.get(field)))
        else:
            values.append(str(row.get(field) or ""))
    return tuple(values)


def _merge_resume_rows(
    branch: str,
    *,
    carry_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in carry_rows:
        rows_by_key[_label_row_key(branch, row)] = _schema_ordered_row(branch, row)
    for row in new_rows:
        rows_by_key[_label_row_key(branch, row)] = _schema_ordered_row(branch, row)
    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            _timestamp_epoch(row.get("timestamp")),
            _canonical_partition_key(branch, row),
            _label_row_key(branch, row),
        ),
    )


def _incremental_resume_payload(
    branch: str,
    *,
    source_tail_ts: int | None,
    last_output_tail_ts: int | None,
    current_rows: Sequence[Mapping[str, Any]],
    active_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if source_tail_ts is None or last_output_tail_ts is None:
        return None
    interval_seconds = _minimum_timestamp_step_seconds(current_rows)
    if interval_seconds is None:
        return None
    current_relationship_tail = _max_epoch_for_field(current_rows, "relationship_input_tail_ts")
    previous_relationship_tail = _max_epoch_for_field(active_rows, "relationship_input_tail_ts")
    plan = plan_regime_production_incremental_update(
        branch=branch,
        source_tail_ts=int(source_tail_ts),
        last_output_tail_ts=int(last_output_tail_ts),
        last_definition_refit_ts=int(last_output_tail_ts),
        interval_seconds=int(interval_seconds),
        relationship_input_tail_ts=current_relationship_tail,
        previous_relationship_input_tail_ts=previous_relationship_tail,
    )
    return plan.as_dict()


def _minimum_timestamp_step_seconds(rows: Sequence[Mapping[str, Any]]) -> int | None:
    timestamps = sorted({_timestamp_epoch(row.get("timestamp")) for row in rows if row.get("timestamp") not in (None, "")})
    diffs = [int(right) - int(left) for left, right in zip(timestamps, timestamps[1:]) if int(right) > int(left)]
    return min(diffs) if diffs else None


def _timestamp_range_epoch(rows: Sequence[Mapping[str, Any]]) -> tuple[int | None, int | None]:
    timestamps = [_timestamp_epoch(row.get("timestamp")) for row in rows if row.get("timestamp") not in (None, "")]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _max_row_timestamp(rows: Sequence[Mapping[str, Any]]) -> int | None:
    _, max_ts = _timestamp_range_epoch(rows)
    return max_ts


def _max_epoch_for_field(rows: Sequence[Mapping[str, Any]], field: str) -> int | None:
    values = [_timestamp_epoch(row.get(field)) for row in rows if row.get(field) not in (None, "")]
    return max(values) if values else None


def _timestamp_epoch(value: Any) -> int:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Regime Production timestamp is not orderable")
    return int(ts.timestamp())


def _iso_timestamp_text(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Regime Production timestamp is not orderable")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _materialized_lock_range(
    rows: Sequence[Mapping[str, Any]],
    selected_partitions: set[str],
    *,
    branch: str,
) -> tuple[str, str]:
    timestamps = sorted(
        str(row.get("timestamp"))
        for row in rows
        if row.get("timestamp") not in (None, "")
        and _canonical_partition_key(branch, row) in selected_partitions
    )
    if timestamps:
        return timestamps[0], timestamps[-1]
    return "unknown_range_start", "unknown_range_end"


def _materialized_clamp_range_payload(
    materialization_summary: Mapping[str, Any],
    *,
    range_start: str,
    range_end: str,
) -> dict[str, Any]:
    return {
        "mode": materialization_summary.get("materialization_mode") or LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE,
        "output_start": range_start,
        "output_end": range_end,
        "range_start": range_start,
        "range_end": range_end,
        "source": "regime_production_direct_plan_materializer",
        "full_clamp_cadence_materialized": bool(materialization_summary.get("full_clamp_cadence_materialized")),
        "planned_unit_count": materialization_summary.get("planned_unit_count"),
        "row_count": materialization_summary.get("row_count"),
        "cadence_unit_counts": to_jsonable(dict(materialization_summary.get("cadence_unit_counts") or {})),
    }


def _source_tail_summary_for_rows(
    rows: Sequence[Mapping[str, Any]],
    branch: str,
    selected_partitions: set[str],
) -> dict[str, Any]:
    fields = (
        "source_tail_ts",
        "known_at_ts",
        "relationship_input_tail_ts",
        "relationship_known_at_ts",
        "definition_known_at_ts",
    )
    values: dict[str, set[str]] = {field: set() for field in fields}
    row_count = 0
    for row in rows:
        if _canonical_partition_key(branch, row) not in selected_partitions:
            continue
        row_count += 1
        for field in fields:
            value = row.get(field)
            if value not in (None, ""):
                values[field].add(str(value))
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_writer_source_tail_summary",
        "source": "direct_materialized_branch_plan_rows",
        "branch": _branch_name(branch),
        "selected_partition_count": len(selected_partitions),
        "selected_row_count": row_count,
        "fields": {
            field: {
                "distinct_count": len(field_values),
                "min": min(field_values) if field_values else None,
                "max": max(field_values) if field_values else None,
                "values_hash": stable_payload_hash({"values": sorted(field_values)}),
            }
            for field, field_values in values.items()
        },
    }


def _canonical_output_root(branch: str, *, env: Mapping[str, str], project_root: str | Path | None) -> Path:
    raw = resolve_path("output_parquet_root", env=env, required=True)
    return resolve_project_path(raw, project_root=project_root) / CANONICAL_LABEL_OUTPUT_NAMESPACE / _branch_name(branch)


def _canonical_lock_root(branch: str, *, env: Mapping[str, str], project_root: str | Path | None) -> Path:
    raw = resolve_path("state_root", env=env, required=True)
    return resolve_project_path(raw, project_root=project_root) / "regime_production_write_locks" / _branch_name(branch)


def _model_state_root(branch: str, *, env: Mapping[str, str], project_root: str | Path | None) -> Path:
    raw = resolve_path("state_root", env=env, required=True)
    return resolve_project_path(raw, project_root=project_root) / CANONICAL_MODEL_STATE_NAMESPACE / _branch_name(branch)


def _canonical_output_root_contract(branch: str, output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_canonical_output_root_contract",
        "branch": _branch_name(branch),
        "root_kind": "canonical_production_output_root",
        "configured_root_key": "output_parquet_root",
        "canonical_root": _portable_path_text(output_root),
        "canonical_root_touched": True,
        "canonical_write_test_performed": False,
        "writer_scope": "branch_canonical_labels",
    }


def _canonical_partition_rows(
    branch: str,
    source_partition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for rows in source_partition_rows.values():
        for row in rows:
            payload = dict(row)
            key = _canonical_partition_key(branch, payload)
            out.setdefault(key, []).append(payload)
    return out


def _canonical_partition_key(branch: str, row: Mapping[str, Any]) -> str:
    branch_name = _branch_name(branch)
    schema = default_regime_production_label_output_schema(branch_name)
    partition_fields = tuple(str(item) for item in schema.as_dict()["partition_fields"])
    year, month = _timestamp_partition_parts(row.get("timestamp"))
    values: dict[str, str] = {}
    for field in partition_fields:
        if field == "branch":
            value = branch_name
        elif field == "year":
            value = f"{year:04d}"
        elif field == "month":
            value = f"{month:02d}"
        else:
            value = row.get(field)
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"Regime Production canonical partition field missing: {branch_name}.{field}")
        values[str(field)] = text
    return "|".join(f"{field}={values[field]}" for field in partition_fields)


def _canonical_partition_path(branch: str, partition_key: str) -> Path:
    _branch_name(branch)
    parts = []
    for item in partition_key.split("|"):
        field, value = item.split("=", 1)
        parts.append(f"{_safe_path_part(field)}={_safe_path_part(value)}")
    return Path(*parts) / "part-000.parquet"


def _expected_canonical_partition_paths(run_root: Path, branch: str, selected_partitions: set[str]) -> tuple[Path, ...]:
    return tuple(run_root / _canonical_partition_path(branch, partition_key) for partition_key in sorted(selected_partitions))


def _timestamp_partition_parts(value: Any) -> tuple[int, int]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Regime Production canonical partition timestamp is invalid")
    return int(ts.year), int(ts.month)


def _write_parquet_partition(path: Path, rows: Sequence[Mapping[str, Any]], *, schema: Any) -> None:
    frame = _canonical_label_parquet_frame(rows, schema=schema)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        row_group_size=PARQUET_ROW_GROUP,
    )


def _canonical_label_parquet_frame(rows: Sequence[Mapping[str, Any]], *, schema: Any) -> pd.DataFrame:
    schema_payload = dict(schema.as_dict())
    columns = list(schema_payload["column_order"])
    dtype_by_column = dict(schema_payload["dtype_by_column"])
    nullable = set(str(col) for col in schema_payload["nullable_columns"])
    frame = pd.DataFrame([dict(row) for row in rows], columns=columns)
    missing_required = [col for col in columns if col not in frame.columns]
    if missing_required:
        raise ValueError(f"Regime Production canonical parquet frame missing columns: {missing_required}")
    for column in columns:
        logical_dtype = str(dtype_by_column.get(column) or "string")
        if logical_dtype == "timestamp_utc":
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if column not in nullable and frame[column].isna().any():
                raise ValueError(f"Regime Production canonical parquet timestamp column contains nulls: {column}")
        elif logical_dtype == "float64":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        elif logical_dtype == "json_object":
            frame[column] = frame[column].map(_canonical_json_text).astype(object)
            if column not in nullable and frame[column].isna().any():
                raise ValueError(f"Regime Production canonical parquet JSON column contains nulls: {column}")
        else:
            frame[column] = frame[column].map(_string_or_none).astype(object)
            if column not in nullable and frame[column].isna().any():
                raise ValueError(f"Regime Production canonical parquet string column contains nulls: {column}")
    return frame[columns]


def _canonical_json_text(value: Any) -> str | None:
    if _nullish(value):
        return None
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))


def _string_or_none(value: Any) -> str | None:
    if _nullish(value):
        return None
    return str(value)


def _nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _validate_canonical_staged_branch_write(
    staging_run_root: Path,
    manifest: Mapping[str, Any],
    *,
    partition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_partitions: set[str],
    schema_valid: bool,
) -> None:
    if not schema_valid:
        raise ValueError("Regime Production canonical writer schema validation failed before commit")
    partition_files = tuple(dict(item) for item in manifest.get("partition_files") or ())
    if len(partition_files) != len(selected_partitions) or len(partition_rows) != len(selected_partitions):
        raise ValueError("Regime Production canonical writer row-count validation failed before commit")
    expected_rows = sum(len(rows) for rows in partition_rows.values())
    declared_rows = sum(int(item.get("row_count") or 0) for item in partition_files)
    if int(manifest.get("rows_written") or 0) != expected_rows or declared_rows != expected_rows:
        raise ValueError("Regime Production canonical writer row-count validation failed before commit")
    for partition_key in sorted(selected_partitions):
        staged_path = staging_run_root / _canonical_partition_path(str(manifest["branch"]), partition_key)
        if not staged_path.is_file():
            raise ValueError("Regime Production canonical writer staged partition missing before commit")
        try:
            frame = pd.read_parquet(staged_path)
        except Exception as exc:
            raise ValueError("Regime Production canonical writer staged parquet unreadable before commit") from exc
        expected_columns = list(dict(manifest.get("output_schema") or {}).get("column_order") or ())
        if list(frame.columns) != expected_columns:
            raise ValueError("Regime Production canonical writer staged parquet schema mismatch before commit")


def _commit_canonical_staged_run_root(
    staging_run_root: Path,
    run_root: Path,
    *,
    allow_existing_overwrite: bool,
    fingerprint: str,
) -> dict[str, Any]:
    if not staging_run_root.is_dir():
        raise FileNotFoundError(f"Regime Production canonical writer staging root missing: {staging_run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    superseded_root: Path | None = None
    if run_root.exists():
        if not allow_existing_overwrite:
            raise FileExistsError(f"Regime Production canonical writer final run root already exists: {run_root}")
        superseded_root = _superseded_run_root(run_root, fingerprint)
        run_root.rename(superseded_root)
    try:
        staging_run_root.rename(run_root)
    except Exception:
        if superseded_root is not None and superseded_root.exists() and not run_root.exists():
            superseded_root.rename(run_root)
        raise
    return {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_canonical_staged_commit",
        "committed": True,
        "staged_root": _portable_path_text(staging_run_root),
        "final_run_root": _portable_path_text(run_root),
        "superseded_previous_run_root": None if superseded_root is None else _portable_path_text(superseded_root),
        "commit_mode": "directory_rename",
        "broad_cleanup_or_delete_performed": False,
        "canonical_root_touched": True,
        "production_outputs_written": True,
    }


def _write_canonical_active_pointer(
    active_pointer_path: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    commit_info: Mapping[str, Any],
) -> dict[str, Any]:
    pointer = {
        "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_CANONICAL_ACTIVE_POINTER_ARTIFACT_KIND,
        "branch": manifest.get("branch"),
        "run_id": manifest.get("run_id"),
        "manifest_path": _portable_path_text(manifest_path),
        "active_output_root": _portable_path_text(manifest_path.parent),
        "idempotency_key_hash": dict(manifest.get("idempotency") or {}).get("idempotency_key_hash"),
        "output_schema_hash": dict(manifest.get("output_schema") or {}).get("schema_hash"),
        "rows_written": int(manifest.get("rows_written") or 0),
        "logical_partitions_written": int(manifest.get("logical_partitions_written") or 0),
        "commit_info": to_jsonable(dict(commit_info)),
        "active_pointer_updated_after_validation_only": True,
        "completion_status": OUTPUT_COMPLETION_STATUS_COMPLETED,
        "manifest_status": OUTPUT_COMPLETION_STATUS_COMPLETED,
        "partial_output_marker": False,
        "canonical_root_touched": True,
        "production_writer_enabled": True,
        "production_labels_written": True,
        "production_outputs_written": True,
        "canonical_label_outputs_written": True,
        "canonical_production_state_outputs_written": False,
    }
    _write_json_atomic(active_pointer_path, pointer)
    return to_jsonable(pointer)


def _write_canonical_staging_status(
    staging_run_root: Path,
    *,
    status: str,
    branch: str,
    run_id: str,
    final_run_root: Path,
    active_pointer_path: Path,
    reason: str | None,
) -> None:
    _write_json_atomic(
        staging_run_root / "staging_status.json",
        {
            "schema_version": REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_CANONICAL_STAGING_STATUS_ARTIFACT_KIND,
            "branch": branch,
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "staging_root": _portable_path_text(staging_run_root),
            "final_run_root": _portable_path_text(final_run_root),
            "active_pointer_path": _portable_path_text(active_pointer_path),
            "active_output": False,
            "active_pointer_updated": False,
            "canonical_root_touched": False if status == "failed_non_active" else True,
            "production_writer_enabled": True,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        },
    )


def _mark_lock_failed_recoverable(lock_handle: RegimeProductionRunLockHandle) -> None:
    try:
        release_regime_production_run_lock(
            lock_handle,
            status=REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE,
            reason="branch_write_exception",
        )
    except Exception:
        pass


def _resolve_source_file(value: Any, *, project_root: str | Path | None) -> Path:
    text = _text(value, field_name="source_file")
    path = resolve_project_path(text, project_root=project_root)
    if not path.exists():
        raise FileNotFoundError(f"Regime Production canonical source output missing: {path}")
    return path


def _load_json(path: str | Path, *, project_root: str | Path | None) -> dict[str, Any]:
    resolved = resolve_project_path(path, project_root=project_root)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production canonical writer expected a JSON object")
    return payload


def _branch_compute_execution(result: Mapping[str, Any]) -> dict[str, Any]:
    materialization = dict(result.get("label_materialization") or {})
    execution = dict(materialization.get("execution") or result.get("compute_execution") or {})
    return to_jsonable(
        {
            "execution_mode": execution.get("execution_mode"),
            "backend": execution.get("backend"),
            "worker_profile_honored": bool(execution.get("worker_profile_honored")),
            "configured_workers": execution.get("configured_workers"),
            "effective_workers": execution.get("effective_workers"),
            "model_threads": execution.get("model_threads"),
            "writer_workers": int(execution.get("writer_workers") or 1),
            "parallel_processes_used": int(execution.get("parallel_processes_used") or 0),
            "parallel_threads_used": int(execution.get("parallel_threads_used") or 0),
            "workers_compute_only": bool(execution.get("workers_compute_only")),
            "workers_write_outputs": bool(execution.get("workers_write_outputs")),
            "parent_finalizer_writes_only": bool(execution.get("parent_finalizer_writes_only")),
        }
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path, suffix=".json.tmp")
    tmp.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def _resolve_operator_checklist_path(
    *,
    explicit_path: str | Path | None,
    search_roots: Sequence[str | Path],
    project_root: str | Path | None,
    max_bytes: int,
) -> Path | None:
    if explicit_path is not None and str(explicit_path).strip():
        return resolve_project_path(explicit_path, project_root=project_root)
    for raw_root in search_roots:
        root = resolve_project_path(raw_root, project_root=project_root)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*.json")
        for path in candidates:
            try:
                if not path.is_file() or path.stat().st_size > int(max_bytes):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                isinstance(payload, Mapping)
                and payload.get("artifact_kind") == "regime_production_operator_approval_checklist"
                and payload.get("checklist_status") == OPERATOR_CHECKLIST_STATUS_OPERATOR_APPROVED
                and payload.get("operator_approval_executed") is True
            ):
                return path
    return None


def _canonical_writer_approval_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    approvals = dict(inventory.get("branch_approvals") or {})
    paths = dict(inventory.get("branch_approval_paths") or {})
    executable_approvals: dict[str, dict[str, Any]] = {}
    executable_paths: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for branch, approval in approvals.items():
        payload = dict(approval)
        path = str(paths.get(branch) or "")
        if _approval_is_prewrite_scaffold(payload, path):
            skipped[str(branch)] = "prewrite_scaffold_not_executable_for_canonical_writer"
            continue
        executable_approvals[str(branch)] = payload
        if path:
            executable_paths[str(branch)] = path
    present = {branch: branch in executable_approvals for branch in REGIME_PRODUCTION_BRANCHES}
    return {
        **dict(inventory),
        "branch_approvals": executable_approvals,
        "branch_approval_paths": executable_paths,
        "branch_approval_artifacts_present": present,
        "skipped_non_executable_branch_approvals": skipped,
    }


def _approval_is_prewrite_scaffold(approval: Mapping[str, Any], path: str) -> bool:
    tokens = (
        str(approval.get("approval_id") or ""),
        str(approval.get("approval_source") or ""),
        str(approval.get("approval_operator") or ""),
        str(path or ""),
    )
    return any("prewrite" in token.lower() or "approval_scaffolding" in token.lower() for token in tokens)


def _requested_branches(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return DEFAULT_CANONICAL_RUN_ORDER
        return (_branch_name(value),)
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text.lower() == "all":
            out.extend(DEFAULT_CANONICAL_RUN_ORDER)
        else:
            out.append(_branch_name(text))
    ordered = []
    for branch in DEFAULT_CANONICAL_RUN_ORDER:
        if branch in out and branch not in ordered:
            ordered.append(branch)
    return tuple(ordered)


def _merged_env(wiring: Mapping[str, Any], cfg: RegimeProductionCanonicalWriterConfig) -> dict[str, str]:
    env = wiring_env(wiring, project_root=cfg.project_root)
    env.update(dict(cfg.env or {}))
    return env


def _build_arg_parser(*, default_branch: str = "all") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Regime Production canonical branch writer.")
    parser.add_argument("--branch", default=default_branch, choices=("all", *REGIME_PRODUCTION_BRANCHES))
    parser.add_argument("--wiring-config", default=str(DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH))
    parser.add_argument("--sandbox-validation-summary", default=DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH)
    parser.add_argument("--wider-sandbox-summary", default=None)
    parser.add_argument("--write-capable-sandbox-summary", default=DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH)
    parser.add_argument("--approval-search-root", action="append", default=None)
    parser.add_argument("--branch-approval", action="append", default=())
    parser.add_argument("--operator-checklist", default=None)
    parser.add_argument("--operator-checklist-search-root", action="append", default=None)
    parser.add_argument("--require-operator-approval", "--require-approval", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-write", action="store_true", help="Preflight only: keep writer disabled.")
    parser.add_argument("--max-logical-partitions-per-branch", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--allow-existing-overwrite", action="store_true")
    parser.add_argument("--summary-output", default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> RegimeProductionCanonicalWriterConfig:
    branch_approvals: dict[str, str] = {}
    for item in getattr(args, "branch_approval", ()) or ():
        if "=" not in str(item):
            raise ValueError("--branch-approval must be formatted as branch=path")
        branch, path = str(item).split("=", 1)
        branch_approvals[_branch_name(branch)] = path
    return RegimeProductionCanonicalWriterConfig(
        branch=args.branch,
        wiring_config_path=args.wiring_config,
        sandbox_validation_summary_path=args.sandbox_validation_summary,
        wider_sandbox_summary_path=args.wider_sandbox_summary,
        write_capable_sandbox_summary_path=args.write_capable_sandbox_summary,
        approval_search_roots=tuple(args.approval_search_root or DEFAULT_APPROVAL_SEARCH_ROOTS),
        branch_approval_paths=branch_approvals,
        operator_checklist_path=args.operator_checklist,
        operator_checklist_search_roots=tuple(args.operator_checklist_search_root or DEFAULT_APPROVAL_SEARCH_ROOTS),
        require_operator_approval=bool(args.require_operator_approval),
        run_id=args.run_id,
        production_writer_enabled=not bool(args.no_write),
        max_logical_partitions_per_branch=args.max_logical_partitions_per_branch,
        resume=bool(args.resume),
        allow_existing_overwrite=bool(args.allow_existing_overwrite),
        summary_output_path=args.summary_output,
    )


def _context(value: RegimeProductionPromotionGateContext | Mapping[str, Any]) -> RegimeProductionPromotionGateContext:
    if isinstance(value, RegimeProductionPromotionGateContext):
        return value
    payload = dict(value)
    return RegimeProductionPromotionGateContext(
        sandbox_validation_summary=dict(payload.get("sandbox_validation_summary") or {}),
        wider_sandbox_summary=dict(payload.get("wider_sandbox_summary") or {}),
    )


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    if text not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {text!r}")
    return text


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production canonical writer {field_name} must be non-empty")
    return text


def _stable_codes(values: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(str(value) for value in values if str(value).strip()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANONICAL_WRITER_STATUS_BLOCKED",
    "CANONICAL_WRITER_STATUS_WRITTEN",
    "DEFAULT_CANONICAL_WRITER_SUMMARY_PATH",
    "REGIME_PRODUCTION_CANONICAL_WRITER_ARTIFACT_KIND",
    "REGIME_PRODUCTION_CANONICAL_WRITER_SCHEMA_VERSION",
    "RegimeProductionCanonicalWriterConfig",
    "main",
    "main_for_branch",
    "run_regime_production_canonical_writer",
    "write_regime_production_canonical_branch_partitions",
    "write_regime_production_canonical_writer_summary",
]
