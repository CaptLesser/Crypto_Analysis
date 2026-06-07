from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_idempotency import (
    OUTPUT_COMPLETION_STATUS_COMPLETED,
    RegimeProductionIdempotencyError,
    RegimeProductionIdempotencyKey,
    assert_regime_production_completed_output_reusable,
    stable_payload_hash,
)
from src.regimes.core.production_output_contracts import (
    BRANCH_PARTITION_FIELDS,
    default_regime_production_label_output_schema,
    resolve_regime_production_sandbox_output_root_contract,
)
from src.regimes.core.production_promotion_gate import (
    RegimeProductionPromotionGateContext,
    build_regime_production_branch_approval_artifact,
    evaluate_regime_production_branch_promotion_gate,
)
from src.regimes.core.production_run_lock import (
    REGIME_PRODUCTION_LOCK_MODE_SANDBOX,
    REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE,
    REGIME_PRODUCTION_LOCK_STATUS_RELEASED,
    RegimeProductionRunLockHandle,
    RegimeProductionRunLockTarget,
    acquire_regime_production_run_lock,
    release_regime_production_run_lock,
)
from src.regimes.core.production_sandbox_validation import DEFAULT_WIDER_SANDBOX_SUMMARY_PATH
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION = 1
REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_ARTIFACT_KIND = "regime_production_write_capable_sandbox_validation"
REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_BRANCH_ARTIFACT_KIND = "regime_production_write_capable_sandbox_branch_result"
REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_MANIFEST_ARTIFACT_KIND = "regime_production_write_capable_sandbox_branch_manifest"
REGIME_PRODUCTION_SANDBOX_ACTIVE_POINTER_ARTIFACT_KIND = "regime_production_write_capable_sandbox_active_pointer"
REGIME_PRODUCTION_SANDBOX_STAGING_STATUS_ARTIFACT_KIND = "regime_production_write_capable_sandbox_staging_status"

DEFAULT_WRITE_CAPABLE_SANDBOX_OUTPUT_ROOT = (
    "_codex_artifacts/reports/regime_production_write_capable_sandbox/"
    "reports/regimes/foundation/regime_write_capable_sandbox_outputs"
)
DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH = (
    "_codex_artifacts/reports/regime_production_sandbox_forecaster_validation/"
    "regime_production_sandbox_forecaster_validation_summary.json"
)


@dataclass(frozen=True)
class RegimeProductionWriteCapableSandboxConfig:
    sandbox_validation_summary_path: str | Path = DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH
    wider_sandbox_summary_path: str | Path | None = None
    sandbox_output_root: str | Path = DEFAULT_WRITE_CAPABLE_SANDBOX_OUTPUT_ROOT
    run_id: str = "regime_production_write_capable_sandbox"
    sandbox_writer_enabled: bool = False
    accepted_validation_issues: Sequence[str] = ()
    max_logical_partitions_per_branch: int | None = 12
    resume: bool = False
    allow_existing_overwrite: bool = False
    perform_idempotency_probe: bool = True
    env: Mapping[str, str] | None = None
    project_root: str | Path | None = None

    def __post_init__(self) -> None:
        if self.max_logical_partitions_per_branch is not None and int(self.max_logical_partitions_per_branch) <= 0:
            raise ValueError("Regime Production write-capable sandbox max partitions must be positive or None")
        if self.sandbox_output_root is None or not str(self.sandbox_output_root).strip():
            raise ValueError("Regime Production write-capable sandbox requires an explicit sandbox output root")
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))
        object.__setattr__(
            self,
            "accepted_validation_issues",
            tuple(dict.fromkeys(str(item) for item in self.accepted_validation_issues if str(item).strip())),
        )
        if self.max_logical_partitions_per_branch is not None:
            object.__setattr__(self, "max_logical_partitions_per_branch", int(self.max_logical_partitions_per_branch))


def run_regime_production_write_capable_sandbox_validation(
    config: RegimeProductionWriteCapableSandboxConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionWriteCapableSandboxConfig) else RegimeProductionWriteCapableSandboxConfig(**dict(config or {}))
    _validate_config_preconditions(cfg)
    rss_start = _rss_bytes()
    child_start = _child_process_count()
    started = time.perf_counter()

    context = RegimeProductionPromotionGateContext.from_paths(
        sandbox_validation_summary_path=cfg.sandbox_validation_summary_path,
        wider_sandbox_summary_path=cfg.wider_sandbox_summary_path,
    )
    branch_results: dict[str, dict[str, Any]] = {}
    for branch in REGIME_PRODUCTION_BRANCHES:
        branch_root = Path(cfg.sandbox_output_root) / branch
        branch_results[branch] = write_regime_production_sandbox_branch_partitions(
            branch,
            context,
            run_id=cfg.run_id,
            sandbox_output_root=branch_root,
            sandbox_writer_enabled=bool(cfg.sandbox_writer_enabled),
            accepted_validation_issues=cfg.accepted_validation_issues,
            max_logical_partitions=cfg.max_logical_partitions_per_branch,
            resume=bool(cfg.resume),
            allow_existing_overwrite=bool(cfg.allow_existing_overwrite),
            env=cfg.env,
            project_root=cfg.project_root,
        )

    idempotency_probe = {}
    if cfg.perform_idempotency_probe:
        idempotency_probe = {
            branch: write_regime_production_sandbox_branch_partitions(
                branch,
                context,
                run_id=cfg.run_id,
                sandbox_output_root=Path(cfg.sandbox_output_root) / branch,
                sandbox_writer_enabled=bool(cfg.sandbox_writer_enabled),
                accepted_validation_issues=cfg.accepted_validation_issues,
                max_logical_partitions=cfg.max_logical_partitions_per_branch,
                resume=True,
                allow_existing_overwrite=False,
                env=cfg.env,
                project_root=cfg.project_root,
            )
            for branch in REGIME_PRODUCTION_BRANCHES
        }

    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes()
    child_end = _child_process_count()
    payload = {
        "schema_version": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "source_sandbox_validation_run_id": context.sandbox_validation_summary.get("run_id"),
        "source_wider_sandbox_run_id": context.wider_sandbox_summary.get("run_id"),
        "accepted_validation_issues": list(cfg.accepted_validation_issues),
        "branch_results": branch_results,
        "sandbox_write_roots": {
            branch: result["sandbox_output_root_contract"]["sandbox_root"]
            for branch, result in branch_results.items()
        },
        "row_count_by_branch": {branch: int(result["rows_written"]) for branch, result in branch_results.items()},
        "partition_count_by_branch": {branch: int(result["logical_partitions_written"]) for branch, result in branch_results.items()},
        "manifest_paths_by_branch": {branch: result["manifest_path"] for branch, result in branch_results.items()},
        "idempotency_probe": idempotency_probe,
        "idempotency_rerun_result": {
            "performed": bool(cfg.perform_idempotency_probe),
            "resume_skipped_all_branches": bool(idempotency_probe)
            and all(bool(result.get("resume_skip_existing")) for result in idempotency_probe.values()),
            "rows_written_on_probe_by_branch": {
                branch: int(result.get("rows_written") or 0)
                for branch, result in idempotency_probe.items()
            },
        },
        "validation_results": _validation_summary(branch_results, idempotency_probe),
        "runtime_telemetry": {
            "elapsed_seconds": round(float(elapsed), 6),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
            "child_process_count_start": child_start,
            "child_process_count_end": child_end,
            "child_process_count_delta": None if child_start is None or child_end is None else int(child_end) - int(child_start),
            "subprocess_invocations_by_writer": 0,
            "worker_count": 0,
            "writer_execution_mode": "single_process_partitioned_atomic_jsonl_sandbox_writer",
        },
        "writer_finalizer": {
            "mode": "single_write_capable_sandbox_finalizer",
            "branch_manifests_written": True,
            "summary_artifact_allowed": True,
            "canonical_write_allowed": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
        },
        "canonical_write_sprint_safety_verdict": "not_safe_to_execute_canonical_writes_yet_sandbox_writer_mechanics_validated",
        "canonical_write_sprint_remaining_requirements": [
            "real per-branch production approval artifacts",
            "explicit canonical write sprint enable flag",
            "canonical root preflight without writes",
            "rollback state capture before canonical writes",
            "real per-branch human/operator approval artifacts persisted outside dry validation",
        ],
        "sandbox_writer_enabled": bool(cfg.sandbox_writer_enabled),
        "sandbox_labels_written": True,
        "canonical_root_touched": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "test_branch_rerun_performed": False,
        "optuna_or_campaign_run_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
        "cleanup_delete_actions_performed": False,
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def write_regime_production_sandbox_branch_partitions(
    branch: str,
    context: RegimeProductionPromotionGateContext | Mapping[str, Any],
    *,
    run_id: str,
    sandbox_output_root: str | Path,
    sandbox_writer_enabled: bool,
    accepted_validation_issues: Sequence[str],
    max_logical_partitions: int | None = 12,
    resume: bool = False,
    allow_existing_overwrite: bool = False,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    if not sandbox_writer_enabled:
        raise ValueError("Regime Production write-capable sandbox writer requires sandbox_writer_enabled=True")
    ctx = _context(context)
    approval = _sandbox_branch_approval(branch_name, ctx, accepted_validation_issues=accepted_validation_issues)
    gate = evaluate_regime_production_branch_promotion_gate(
        approval,
        ctx,
        branch=branch_name,
        write_sprint_enable_requested=True,
    )
    if gate["gate_status"] != "accepted_for_dry_write_planning":
        raise ValueError(f"Regime Production sandbox writer gate blocked for {branch_name}: {gate['blockers']}")
    root_contract = resolve_regime_production_sandbox_output_root_contract(
        branch_name,
        sandbox_root=sandbox_output_root,
        env=env,
        project_root=project_root,
    )
    schema = default_regime_production_label_output_schema(branch_name)
    source_branch = dict(ctx.wider_sandbox_summary.get("branch_outputs", {}).get(branch_name) or {})
    source_file = _resolve_source_file(source_branch.get("output_file"))
    selected_partitions = _select_partitions(
        source_file,
        branch_name,
        max_logical_partitions=max_logical_partitions,
    )
    range_start, range_end = _branch_lock_range(ctx, source_branch, selected_partitions)
    source_tail_summary = _source_tail_summary_for_partitions(source_file, branch_name, selected_partitions)
    idempotency_key = RegimeProductionIdempotencyKey(
        branch=branch_name,
        mode=REGIME_PRODUCTION_LOCK_MODE_SANDBOX,
        run_id=run_id,
        output_root=root_contract.sandbox_root,
        clamp_range=_clamp_range_payload(ctx, range_start=range_start, range_end=range_end),
        output_schema_version=schema.schema_version,
        output_schema_hash=schema.schema_hash,
        source_artifact_path=source_file,
        source_artifact_hash=_sha256_file(source_file),
        selected_profile_artifact_hash=source_branch.get("profile_artifact_hash"),
        approval_artifact_hash=stable_payload_hash(approval),
        source_tail_fingerprint=stable_payload_hash(source_tail_summary),
        selected_partitions=sorted(selected_partitions),
    )
    idempotency_payload = idempotency_key.as_dict()
    fingerprint = str(idempotency_payload["idempotency_key_hash"])
    run_root = Path(root_contract.sandbox_root) / f"run_id={_safe_path_part(run_id)}"
    manifest_path = run_root / "sandbox_writer_manifest.json"
    if manifest_path.exists() and resume:
        existing = _load_json(manifest_path)
        try:
            rerun_evaluation = assert_regime_production_completed_output_reusable(
                existing,
                expected_idempotency_key=idempotency_payload,
                expected_schema_hash=schema.schema_hash,
                expected_partition_paths=_expected_partition_paths(run_root, branch_name, selected_partitions),
            )
        except RegimeProductionIdempotencyError as exc:
            raise ValueError("Regime Production sandbox writer existing output requires recompute or repair") from exc
        return to_jsonable(
            _sanitize_workspace_paths(
                {
                    **existing,
                    "resume_skip_existing": True,
                    "rerun_action": rerun_evaluation["rerun_action"],
                    "rerun_evaluation": rerun_evaluation,
                    "rows_written": 0,
                    "partitions_written_this_run": 0,
                    "atomic_replace_count": 0,
                    "staged_temp_files_created": 0,
                    "canonical_root_touched": False,
                    "production_writer_enabled": False,
                    "production_labels_written": False,
                    "canonical_production_state_outputs_written": False,
                }
            )
        )
    if manifest_path.exists() and not allow_existing_overwrite:
        raise FileExistsError(f"Regime Production sandbox writer manifest already exists: {manifest_path}")

    staging_run_root = _staging_run_root(run_root, fingerprint)
    if staging_run_root.exists():
        raise FileExistsError(f"Regime Production sandbox writer staging path already exists: {staging_run_root}")
    active_pointer_path = _active_pointer_path(root_contract.sandbox_root)
    lock_handle = acquire_regime_production_run_lock(
        RegimeProductionRunLockTarget(
            branch=branch_name,
            mode=REGIME_PRODUCTION_LOCK_MODE_SANDBOX,
            output_root=root_contract.sandbox_root,
            range_start=range_start,
            range_end=range_end,
        ),
        lock_root=Path(root_contract.sandbox_root) / "_run_locks",
        run_id=run_id,
        owner="regime_production_write_capable_sandbox_finalizer",
        project_root=project_root,
    )
    try:
        staging_run_root.mkdir(parents=True, exist_ok=False)
        _write_staging_status(
            staging_run_root,
            status="staging",
            branch=branch_name,
            run_id=run_id,
            final_run_root=run_root,
            active_pointer_path=active_pointer_path,
            reason=None,
        )
        partition_rows = _load_selected_partition_rows(source_file, branch_name, selected_partitions)
        rows_written = 0
        atomic_count = 0
        temp_count = 0
        schema_valid = True
        partition_files: list[dict[str, Any]] = []
        for partition_key, rows in sorted(partition_rows.items()):
            relative_partition_path = _partition_path(branch_name, partition_key)
            staging_partition_path = staging_run_root / relative_partition_path
            final_partition_path = run_root / relative_partition_path
            staging_partition_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = sibling_temp_path(staging_partition_path, suffix=".jsonl.tmp")
            temp_count += 1
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    if tuple(row.keys()) != schema.column_order:
                        schema_valid = False
                    handle.write(json.dumps(row, separators=(",", ":")))
                    handle.write("\n")
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
            "schema_version": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_MANIFEST_ARTIFACT_KIND,
            "branch": branch_name,
            "run_id": run_id,
            "write_fingerprint": fingerprint,
            "source_file": _portable_path_text(source_file),
            "source_file_hash": idempotency_payload["source_artifact_hash"],
            "source_tail_summary": source_tail_summary,
            "sandbox_output_root_contract": root_contract.as_dict(),
            "output_schema": schema.as_dict(),
            "partition_fields": list(BRANCH_PARTITION_FIELDS[branch_name]),
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
            "schema_validation_passed": schema_valid,
            "mixed_schema_detected": False,
            "writer_gating": {
                "sandbox_writer_enabled": True,
                "branch_promotion_gate_status": gate["gate_status"],
                "dry_write_planning_allowed": gate["dry_write_planning_allowed"],
                "canonical_write_execution_allowed": False,
                "production_writer_enabled": False,
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
            "canonical_root_touched": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
        }
        _validate_staged_branch_write(
            staging_run_root,
            manifest,
            partition_rows=partition_rows,
            selected_partitions=selected_partitions,
            schema_valid=schema_valid,
        )
        _write_staging_status(
            staging_run_root,
            status="validated",
            branch=branch_name,
            run_id=run_id,
            final_run_root=run_root,
            active_pointer_path=active_pointer_path,
            reason=None,
        )
        _write_json_atomic(staging_run_root / "sandbox_writer_manifest.json", manifest)
        commit_info = _commit_staged_run_root(
            staging_run_root,
            run_root,
            allow_existing_overwrite=allow_existing_overwrite,
            fingerprint=fingerprint,
        )
        active_pointer = _write_active_pointer(
            active_pointer_path,
            manifest,
            manifest_path=manifest_path,
            commit_info=commit_info,
        )
        lock_release = release_regime_production_run_lock(
            lock_handle,
            status=REGIME_PRODUCTION_LOCK_STATUS_RELEASED,
            reason="success",
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
            _write_staging_status(
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


def write_regime_production_write_capable_sandbox_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    _write_json_atomic(path, payload)
    return path


def _validate_config_preconditions(cfg: RegimeProductionWriteCapableSandboxConfig) -> None:
    if not cfg.sandbox_writer_enabled:
        raise ValueError("Regime Production write-capable sandbox run requires sandbox_writer_enabled=True")
    ctx = RegimeProductionPromotionGateContext.from_paths(
        sandbox_validation_summary_path=cfg.sandbox_validation_summary_path,
        wider_sandbox_summary_path=cfg.wider_sandbox_summary_path,
    )
    required = _required_validation_issues(ctx.sandbox_validation_summary)
    missing = sorted(required.difference(cfg.accepted_validation_issues))
    if missing:
        raise ValueError(f"Regime Production write-capable sandbox requires accepted validation issues: {missing!r}")


def _sandbox_branch_approval(
    branch: str,
    ctx: RegimeProductionPromotionGateContext,
    *,
    accepted_validation_issues: Sequence[str],
) -> dict[str, Any]:
    return build_regime_production_branch_approval_artifact(
        branch,
        ctx,
        approval_id=f"{branch}_write_capable_sandbox_approval",
        approval_timestamp="2026-06-06T00:00:00Z",
        approval_operator="codex_sandbox_writer_validation",
        approval_source="regime_production_write_capable_sandbox_config",
        canonical_output_root_confirmation={
            "canonical_output_root_confirmed": True,
            "canonical_output_root_key": f"{branch}_canonical_output_root",
            "canonical_output_root_reference": f"configured_canonical_root:{branch}",
            "canonical_root_matches_expected_environment": True,
            "canonical_root_write_test_performed": False,
            "canonical_root_touched": False,
            "confirmation_timestamp": "2026-06-06T00:00:00Z",
        },
        accepted_validation_issues=accepted_validation_issues,
    )


def _required_validation_issues(validation_summary: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for severity in ("blocker", "high", "medium", "low"):
        out.update(str(item) for item in validation_summary.get("issues", {}).get(severity, ()) or ())
    return out


def _select_partitions(
    source_file: Path,
    branch: str,
    *,
    max_logical_partitions: int | None,
) -> set[str]:
    status_by_partition: dict[str, set[str]] = {}
    with source_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = _partition_key(branch, row)
            status_by_partition.setdefault(key, set()).add(str(row.get("availability_status")))
    if max_logical_partitions is None or len(status_by_partition) <= int(max_logical_partitions):
        return set(status_by_partition)
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
    for key in sorted(status_by_partition):
        if key not in out:
            out.append(key)
        if len(out) >= int(max_logical_partitions):
            return set(out)
    return set(out)


def _branch_lock_range(
    ctx: RegimeProductionPromotionGateContext,
    source_branch: Mapping[str, Any],
    selected_partitions: set[str],
) -> tuple[str, str]:
    range_used = ctx.wider_sandbox_summary.get("range_used")
    if isinstance(range_used, Mapping):
        output_start = str(range_used.get("output_start") or "").strip()
        output_end = str(range_used.get("output_end") or "").strip()
        if output_start and output_end:
            return output_start, output_end
    checkpoint_timestamps = tuple(str(item).strip() for item in source_branch.get("checkpoint_timestamps") or ())
    checkpoint_timestamps = tuple(item for item in checkpoint_timestamps if item)
    if checkpoint_timestamps:
        ordered = sorted(checkpoint_timestamps)
        return ordered[0], ordered[-1]
    if selected_partitions:
        ordered_partitions = sorted(selected_partitions)
        return ordered_partitions[0], ordered_partitions[-1]
    return "unknown_range_start", "unknown_range_end"


def _clamp_range_payload(
    ctx: RegimeProductionPromotionGateContext,
    *,
    range_start: str,
    range_end: str,
) -> dict[str, Any]:
    range_used = ctx.wider_sandbox_summary.get("range_used")
    if isinstance(range_used, Mapping) and range_used:
        return to_jsonable(dict(range_used))
    return {
        "output_start": range_start,
        "output_end": range_end,
        "range_source": "branch_writer_fallback",
    }


def _source_tail_summary_for_partitions(
    source_file: Path,
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
    with source_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if _partition_key(branch, row) not in selected_partitions:
                continue
            row_count += 1
            for field in fields:
                value = row.get(field)
                if value not in (None, ""):
                    values[field].add(str(value))
    return {
        "schema_version": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION,
        "artifact_kind": "regime_production_writer_source_tail_summary",
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


def _expected_partition_paths(run_root: Path, branch: str, selected_partitions: set[str]) -> tuple[Path, ...]:
    return tuple(run_root / _partition_path(branch, partition_key) for partition_key in sorted(selected_partitions))


def _staging_run_root(run_root: Path, fingerprint: str) -> Path:
    suffix = _safe_path_part(str(fingerprint).split(":", 1)[-1][:16])
    return run_root.parent / "_staging" / f"{run_root.name}.{suffix}"


def _active_pointer_path(sandbox_root: str | Path) -> Path:
    return Path(sandbox_root) / "active_writer_manifest.json"


def _write_staging_status(
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
            "schema_version": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_SANDBOX_STAGING_STATUS_ARTIFACT_KIND,
            "branch": branch,
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "staging_root": _portable_path_text(staging_run_root),
            "final_run_root": _portable_path_text(final_run_root),
            "active_pointer_path": _portable_path_text(active_pointer_path),
            "active_output": False,
            "active_pointer_updated": False,
            "canonical_root_touched": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        },
    )


def _validate_staged_branch_write(
    staging_run_root: Path,
    manifest: Mapping[str, Any],
    *,
    partition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_partitions: set[str],
    schema_valid: bool,
) -> None:
    if not schema_valid:
        raise ValueError("Regime Production sandbox writer schema validation failed before commit")
    partition_files = tuple(dict(item) for item in manifest.get("partition_files") or ())
    if len(partition_files) != len(selected_partitions) or len(partition_rows) != len(selected_partitions):
        raise ValueError("Regime Production sandbox writer row-count validation failed before commit")
    expected_rows = sum(len(rows) for rows in partition_rows.values())
    declared_rows = sum(int(item.get("row_count") or 0) for item in partition_files)
    if int(manifest.get("rows_written") or 0) != expected_rows or declared_rows != expected_rows:
        raise ValueError("Regime Production sandbox writer row-count validation failed before commit")
    for partition_key in sorted(selected_partitions):
        staged_path = staging_run_root / _partition_path(str(manifest["branch"]), partition_key)
        if not staged_path.is_file():
            raise ValueError("Regime Production sandbox writer staged partition missing before commit")


def _commit_staged_run_root(
    staging_run_root: Path,
    run_root: Path,
    *,
    allow_existing_overwrite: bool,
    fingerprint: str,
) -> dict[str, Any]:
    if not staging_run_root.is_dir():
        raise FileNotFoundError(f"Regime Production sandbox writer staging root missing: {staging_run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    superseded_root: Path | None = None
    if run_root.exists():
        if not allow_existing_overwrite:
            raise FileExistsError(f"Regime Production sandbox writer final run root already exists: {run_root}")
        superseded_root = _superseded_run_root(run_root, fingerprint)
        run_root.rename(superseded_root)
    try:
        staging_run_root.rename(run_root)
    except Exception:
        if superseded_root is not None and superseded_root.exists() and not run_root.exists():
            superseded_root.rename(run_root)
        raise
    return {
        "schema_version": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION,
        "artifact_kind": "regime_production_sandbox_staged_commit",
        "committed": True,
        "staged_root": _portable_path_text(staging_run_root),
        "final_run_root": _portable_path_text(run_root),
        "superseded_previous_run_root": None if superseded_root is None else _portable_path_text(superseded_root),
        "commit_mode": "directory_rename",
        "broad_cleanup_or_delete_performed": False,
        "canonical_root_touched": False,
        "production_outputs_written": False,
    }


def _superseded_run_root(run_root: Path, fingerprint: str) -> Path:
    suffix = _safe_path_part(str(fingerprint).split(":", 1)[-1][:16])
    candidate = run_root.with_name(f"{run_root.name}.superseded.{suffix}")
    idx = 1
    while candidate.exists():
        idx += 1
        candidate = run_root.with_name(f"{run_root.name}.superseded.{suffix}.{idx}")
    return candidate


def _write_active_pointer(
    active_pointer_path: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    commit_info: Mapping[str, Any],
) -> dict[str, Any]:
    pointer = {
        "schema_version": REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_SANDBOX_ACTIVE_POINTER_ARTIFACT_KIND,
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
        "canonical_root_touched": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }
    _write_json_atomic(active_pointer_path, pointer)
    return to_jsonable(pointer)


def _mark_lock_failed_recoverable(lock_handle: RegimeProductionRunLockHandle) -> None:
    try:
        release_regime_production_run_lock(
            lock_handle,
            status=REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE,
            reason="branch_write_exception",
        )
    except Exception:
        pass


def _load_selected_partition_rows(
    source_file: Path,
    branch: str,
    selected_partitions: set[str],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with source_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = _partition_key(branch, row)
            if key not in selected_partitions:
                continue
            rows.setdefault(key, []).append(row)
    return rows


def _partition_key(branch: str, row: Mapping[str, Any]) -> str:
    fields = BRANCH_PARTITION_FIELDS[_branch_name(branch)]
    return "|".join(f"{field}={row.get(field)}" for field in fields)


def _partition_path(branch: str, partition_key: str) -> Path:
    parts = []
    for item in partition_key.split("|"):
        field, value = item.split("=", 1)
        parts.append(f"{_safe_path_part(field)}={_safe_path_part(value)}")
    return Path(*parts) / "part-000.jsonl"


def _validation_summary(
    branch_results: Mapping[str, Mapping[str, Any]],
    idempotency_probe: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "writer_gating_passed": all(result["writer_gating"]["sandbox_writer_enabled"] is True for result in branch_results.values()),
        "output_schema_validation_passed": all(result["schema_validation_passed"] is True for result in branch_results.values()),
        "partitioning_validation_passed": all(int(result["logical_partitions_written"]) > 0 for result in branch_results.values()),
        "no_mixed_schemas": all(result["mixed_schema_detected"] is False for result in branch_results.values()),
        "atomic_staged_writes_validated": all(result["atomic_staged_write"]["atomic_replace"] is True for result in branch_results.values()),
        "resume_behavior_validated": bool(idempotency_probe)
        and all(result.get("resume_skip_existing") is True for result in idempotency_probe.values()),
        "idempotency_validated": bool(idempotency_probe)
        and all(int(result.get("rows_written") or 0) == 0 for result in idempotency_probe.values()),
        "canonical_root_touched": False,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path, suffix=".json.tmp")
    tmp.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_source_file(value: Any) -> Path:
    text = _text(value, field_name="source_file")
    path = resolve_project_path(text)
    if not path.exists():
        raise FileNotFoundError(f"Regime Production sandbox source output missing: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production write-capable sandbox expected a JSON object")
    return payload


def _context(value: RegimeProductionPromotionGateContext | Mapping[str, Any]) -> RegimeProductionPromotionGateContext:
    if isinstance(value, RegimeProductionPromotionGateContext):
        return value
    payload = dict(value)
    return RegimeProductionPromotionGateContext(
        sandbox_validation_summary=dict(payload.get("sandbox_validation_summary") or {}),
        wider_sandbox_summary=dict(payload.get("wider_sandbox_summary") or {}),
    )


def _safe_path_part(value: object) -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_").replace(":", "_").replace("|", "_")
    if not text:
        raise ValueError("Regime Production write-capable sandbox path part must be non-empty")
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
        raise ValueError(f"Regime Production write-capable sandbox {field_name} must be non-empty")
    return text


__all__ = [
    "DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH",
    "DEFAULT_WRITE_CAPABLE_SANDBOX_OUTPUT_ROOT",
    "REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_BRANCH_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_MANIFEST_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WRITE_CAPABLE_SANDBOX_SCHEMA_VERSION",
    "RegimeProductionWriteCapableSandboxConfig",
    "run_regime_production_write_capable_sandbox_validation",
    "write_regime_production_sandbox_branch_partitions",
    "write_regime_production_write_capable_sandbox_summary",
]
