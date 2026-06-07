from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.path_config import PathConfigError, resolve_path
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
from src.regimes.core.production_operator_approval import (
    OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT,
    OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_PREWRITE_REHEARSAL,
    RegimeProductionOperatorChecklistContext,
    validate_regime_production_operator_approval_checklist,
)
from src.regimes.core.production_output_contracts import (
    CANONICAL_DEFINITION_NAMESPACE,
    CANONICAL_LABEL_OUTPUT_NAMESPACE,
    CANONICAL_MODEL_STATE_NAMESPACE,
    LOG_TELEMETRY_NAMESPACE,
    default_regime_production_label_output_schema,
)
from src.regimes.core.production_promotion_gate import (
    DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH,
    RegimeProductionPromotionGateContext,
    evaluate_regime_production_branch_promotion_gate,
)
from src.regimes.core.production_run_lock import (
    DEFAULT_REGIME_PRODUCTION_STALE_LOCK_SECONDS,
    REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
    REGIME_PRODUCTION_LOCK_STATUS_ACTIVE,
    RegimeProductionRunLockTarget,
    regime_production_run_lock_is_stale,
    regime_production_run_lock_path,
)
from src.regimes.core.root_resolution import (
    SOURCE_KIND_OHLCVT,
    SOURCE_KIND_REGIME_FEATURES,
    SOURCE_KIND_RELATIONSHIP_DISCOVERY,
    SOURCE_KIND_SCALAR_FEATURES,
    resolve_required_regime_production_source_root,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_SCHEMA_VERSION = 1
REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_ARTIFACT_KIND = "regime_production_precanonical_go_no_go_rehearsal"
REGIME_PRODUCTION_PRECANONICAL_BRANCH_ARTIFACT_KIND = "regime_production_precanonical_branch_rehearsal"

PRECANONICAL_PASS = "PASS"
PRECANONICAL_BLOCKED = "BLOCKED"

DEFAULT_OPERATOR_CHECKLIST_SEARCH_ROOTS: tuple[str, ...] = ("_codex_artifacts/reports",)
DEFAULT_CANONICAL_RUN_ORDER: tuple[str, ...] = (
    REGIME_BRANCH_MARKET_STATE,
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
)

SOURCE_ROOT_KINDS_BY_BRANCH: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: (SOURCE_KIND_OHLCVT, SOURCE_KIND_SCALAR_FEATURES, SOURCE_KIND_REGIME_FEATURES),
    REGIME_BRANCH_MARKET_STATE: (SOURCE_KIND_OHLCVT, SOURCE_KIND_SCALAR_FEATURES, SOURCE_KIND_REGIME_FEATURES),
    REGIME_BRANCH_CROSS_ASSET_STATE: (
        SOURCE_KIND_OHLCVT,
        SOURCE_KIND_SCALAR_FEATURES,
        SOURCE_KIND_REGIME_FEATURES,
        SOURCE_KIND_RELATIONSHIP_DISCOVERY,
    ),
}


@dataclass(frozen=True)
class RegimeProductionPreCanonicalRehearsalConfig:
    sandbox_validation_summary_path: str | Path = DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH
    wider_sandbox_summary_path: str | Path | None = None
    write_capable_sandbox_summary_path: str | Path = DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH
    approval_search_roots: Sequence[str | Path] = DEFAULT_APPROVAL_SEARCH_ROOTS
    branch_approval_paths: Mapping[str, str | Path] | None = None
    operator_checklist_path: str | Path | None = None
    operator_checklist_search_roots: Sequence[str | Path] = DEFAULT_OPERATOR_CHECKLIST_SEARCH_ROOTS
    run_id: str | None = None
    env: Mapping[str, str] | None = None
    project_root: str | Path | None = None
    approval_scan_max_bytes: int = 2_000_000
    stale_lock_seconds: int = DEFAULT_REGIME_PRODUCTION_STALE_LOCK_SECONDS

    def __post_init__(self) -> None:
        if int(self.approval_scan_max_bytes) <= 0:
            raise ValueError("Regime Production pre-canonical approval scan size must be positive")
        if int(self.stale_lock_seconds) <= 0:
            raise ValueError("Regime Production pre-canonical stale lock seconds must be positive")
        if self.run_id is None:
            stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            object.__setattr__(self, "run_id", f"regime_production_precanonical_rehearsal_{stamped}")
        object.__setattr__(self, "approval_search_roots", tuple(self.approval_search_roots))
        object.__setattr__(self, "operator_checklist_search_roots", tuple(self.operator_checklist_search_roots))
        object.__setattr__(self, "branch_approval_paths", dict(self.branch_approval_paths or {}))
        object.__setattr__(self, "env", None if self.env is None else dict(self.env))
        object.__setattr__(self, "approval_scan_max_bytes", int(self.approval_scan_max_bytes))
        object.__setattr__(self, "stale_lock_seconds", int(self.stale_lock_seconds))


def run_regime_production_precanonical_rehearsal(
    config: RegimeProductionPreCanonicalRehearsalConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (
        config
        if isinstance(config, RegimeProductionPreCanonicalRehearsalConfig)
        else RegimeProductionPreCanonicalRehearsalConfig(**dict(config or {}))
    )
    promotion_context = RegimeProductionPromotionGateContext.from_paths(
        sandbox_validation_summary_path=cfg.sandbox_validation_summary_path,
        wider_sandbox_summary_path=cfg.wider_sandbox_summary_path,
    )
    write_capable = _load_json(cfg.write_capable_sandbox_summary_path, project_root=cfg.project_root)
    checklist_context = RegimeProductionOperatorChecklistContext(
        promotion_context=promotion_context,
        write_capable_sandbox_summary=write_capable,
    )
    approval_inventory = discover_regime_production_approval_artifacts(
        cfg.approval_search_roots,
        branch_approval_paths=cfg.branch_approval_paths,
        project_root=cfg.project_root,
        max_bytes=cfg.approval_scan_max_bytes,
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
        env=cfg.env,
        project_root=cfg.project_root,
    )
    canonical_roots = _canonical_roots(env=cfg.env, project_root=cfg.project_root)
    branch_rehearsals = {
        branch: _branch_rehearsal(
            branch,
            cfg=cfg,
            promotion_context=promotion_context,
            write_capable=write_capable,
            approval_inventory=approval_inventory,
            operator_validation=operator_validation,
            canonical_roots=canonical_roots,
        )
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    blocked = {
        branch: payload["blockers"]
        for branch, payload in branch_rehearsals.items()
        if payload["status"] != PRECANONICAL_PASS
    }
    payload = {
        "schema_version": REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "overall_status": PRECANONICAL_BLOCKED if blocked else PRECANONICAL_PASS,
        "branch_rehearsals": branch_rehearsals,
        "branch_status": {branch: payload["status"] for branch, payload in branch_rehearsals.items()},
        "exact_blockers": blocked,
        "approval_inventory": {
            "branch_approval_artifacts_present": approval_inventory.get("branch_approval_artifacts_present"),
            "branch_approval_paths": approval_inventory.get("branch_approval_paths"),
            "unified_approval_artifact_present": approval_inventory.get("unified_approval_artifact_present"),
            "operator_checklist_path": None if operator_checklist_path is None else _portable_path_text(operator_checklist_path),
            "operator_checklist_validation_status": operator_validation.get("validation_status"),
            "operator_checklist_blockers": operator_validation.get("blockers"),
        },
        "expected_runtime_order": list(DEFAULT_CANONICAL_RUN_ORDER),
        "expected_runtime_order_reason": (
            "Market-State first, Asset-State second, Cross-Asset-State last because Cross has relationship-input freshness dependencies."
        ),
        "next_pass_branch": _next_pass_branch(branch_rehearsals),
        "safety_confirmation": {
            "canonical_write_attempted": False,
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
            "write_lock_acquired": False,
            "production_writer_gates_fail_closed": True,
        },
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def write_regime_production_precanonical_rehearsal_summary(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _branch_rehearsal(
    branch: str,
    *,
    cfg: RegimeProductionPreCanonicalRehearsalConfig,
    promotion_context: RegimeProductionPromotionGateContext,
    write_capable: Mapping[str, Any],
    approval_inventory: Mapping[str, Any],
    operator_validation: Mapping[str, Any],
    canonical_roots: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    branch_output = dict(promotion_context.wider_sandbox_summary.get("branch_outputs", {}).get(branch) or {})
    approval_path = dict(approval_inventory.get("branch_approval_paths") or {}).get(branch)
    approval = dict(dict(approval_inventory.get("branch_approvals") or {}).get(branch) or {})
    approval_gate = evaluate_regime_production_branch_promotion_gate(
        approval or None,
        promotion_context,
        branch=branch,
        write_sprint_enable_requested=False,
    )
    blockers.extend(f"approval:{reason}" for reason in approval_gate.get("blockers", ()) or ())

    active = _active_artifact_check(branch, branch_output, env=cfg.env, project_root=cfg.project_root)
    blockers.extend(active["blockers"])
    source_roots = _source_root_checks(branch, env=cfg.env, project_root=cfg.project_root)
    blockers.extend(source_roots["blockers"])
    output_roots = _output_root_checks(branch, canonical_roots)
    blockers.extend(output_roots["blockers"])
    schema_check = _schema_check(branch, branch_output)
    blockers.extend(schema_check["blockers"])
    clamp_check = _clamp_check(promotion_context.wider_sandbox_summary)
    blockers.extend(clamp_check["blockers"])
    relationship_check = (
        _cross_relationship_check(promotion_context.sandbox_validation_summary)
        if branch == REGIME_BRANCH_CROSS_ASSET_STATE
        else {"applicable": False, "passed": True, "blockers": []}
    )
    blockers.extend(relationship_check["blockers"])
    lock_check = _write_lock_check(
        branch,
        output_roots,
        clamp_check,
        stale_after_seconds=cfg.stale_lock_seconds,
        project_root=cfg.project_root,
    )
    blockers.extend(lock_check["blockers"])
    writer_gate = {
        "writer_gate_disabled": approval_gate.get("production_writer_enabled") is False,
        "canonical_write_execution_allowed": False,
        "production_writer_enabled": False,
        "production_outputs_written": False,
        "production_labels_written": False,
        "canonical_production_state_outputs_written": False,
    }
    if approval_gate.get("production_writer_enabled") is not False:
        blockers.append("writer_gate_not_disabled")
    accepted_checklist_statuses = {
        OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT,
        OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_PREWRITE_REHEARSAL,
    }
    if operator_validation.get("validation_status") not in accepted_checklist_statuses:
        blockers.append("operator_checklist:" + ",".join(operator_validation.get("blockers") or ("missing_operator_approval_checklist",)))
    expected_counts = _expected_counts(branch, promotion_context.sandbox_validation_summary, promotion_context.wider_sandbox_summary)
    status = PRECANONICAL_BLOCKED if blockers else PRECANONICAL_PASS
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_PRECANONICAL_BRANCH_ARTIFACT_KIND,
                "branch": branch,
                "status": status,
                "blockers": _stable_codes(blockers),
                "approval_artifact": {
                    "resolved": bool(approval),
                    "path": approval_path,
                    "approval_id": approval.get("approval_id"),
                    "gate": approval_gate,
                },
                "active_selected_profile_artifact": active,
                "input_roots": source_roots,
                "output_roots": output_roots,
                "model_state_root": output_roots.get("model_state_root"),
                "schemas": schema_check,
                "clamp_range": clamp_check,
                "relationship_inputs": relationship_check,
                "write_lock": lock_check,
                "writer_gate": writer_gate,
                "expected_counts": expected_counts,
                "expected_runtime_order_index": list(DEFAULT_CANONICAL_RUN_ORDER).index(branch),
                "canonical_command_config_if_pass": (
                    _canonical_command_config(branch, approval_path, output_roots, clamp_check)
                    if status == PRECANONICAL_PASS
                    else None
                ),
                "canonical_command_blocked_reason": None if status == PRECANONICAL_PASS else "branch_rehearsal_blocked",
                "canonical_write_attempted": False,
                "production_writer_enabled": False,
                "production_outputs_written": False,
                "canonical_production_state_outputs_written": False,
            }
        )
    )


def _active_artifact_check(
    branch: str,
    branch_output: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    active = resolve_active_selected_profile_artifact(branch, env=env, project_root=project_root)
    active_path = active.artifact_path
    active_hash = _sha256_file(active_path) if active.passed and active_path is not None else None
    expected_path = branch_output.get("source_artifact_path")
    expected_hash = branch_output.get("profile_artifact_hash")
    path_match = False
    if active_path is not None and expected_path:
        path_match = resolve_project_path(expected_path, project_root=project_root).resolve() == Path(active_path).resolve()
    if not active.passed:
        blockers.extend(f"active_artifact:{reason}" for reason in active.reason_codes)
    if not path_match:
        blockers.append("active_selected_profile_artifact_path_mismatch")
    if active_hash != expected_hash:
        blockers.append("active_selected_profile_artifact_hash_mismatch")
    repaired = bool(branch == REGIME_BRANCH_ASSET_STATE and active_path is not None and "lineage_repaired" in active_path.name)
    if branch == REGIME_BRANCH_ASSET_STATE and not repaired:
        blockers.append("repaired_asset_artifact_not_active")
    return {
        "passed": not blockers,
        "status": active.status,
        "path": None if active_path is None else _portable_path_text(active_path),
        "hash": active_hash,
        "expected_path": expected_path,
        "expected_hash": expected_hash,
        "path_matches_expected": path_match,
        "hash_matches_expected": active_hash == expected_hash,
        "repaired_asset_artifact_active": repaired,
        "blockers": _stable_codes(blockers),
    }


def _source_root_checks(
    branch: str,
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> dict[str, Any]:
    roots: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for kind in SOURCE_ROOT_KINDS_BY_BRANCH[branch]:
        try:
            root, source = resolve_required_regime_production_source_root(kind, env=env, project_root=project_root)
            roots[kind] = {
                "configured": True,
                "root": _portable_path_text(root),
                "root_source": source,
                "configured_roots_only": True,
            }
        except PathConfigError as exc:
            roots[kind] = {
                "configured": False,
                "error": str(exc),
                "configured_roots_only": True,
            }
            blockers.append(f"input_root_missing:{kind}")
    return {"passed": not blockers, "roots": roots, "blockers": _stable_codes(blockers)}


def _output_root_checks(branch: str, canonical_roots: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(canonical_roots.get("configured_roots") or {})
    missing = tuple(canonical_roots.get("missing_root_keys") or ())
    blockers = [f"canonical_root_missing:{key}" for key in missing]
    output_root = None
    model_state_root = None
    definition_root = None
    telemetry_root = None
    if "output_parquet_root" in configured:
        output_root = str(Path(configured["output_parquet_root"]) / CANONICAL_LABEL_OUTPUT_NAMESPACE / branch)
    if "state_root" in configured:
        model_state_root = str(Path(configured["state_root"]) / CANONICAL_MODEL_STATE_NAMESPACE / branch)
    if "regime_definition_root" in configured:
        definition_root = str(Path(configured["regime_definition_root"]) / CANONICAL_DEFINITION_NAMESPACE / branch)
    if "log_root" in configured:
        telemetry_root = str(Path(configured["log_root"]) / LOG_TELEMETRY_NAMESPACE / branch)
    return {
        "passed": not blockers,
        "configured_roots": configured,
        "missing_root_keys": list(missing),
        "canonical_output_root": output_root,
        "model_state_root": model_state_root,
        "definition_root": definition_root,
        "telemetry_root": telemetry_root,
        "canonical_root_touched": False,
        "canonical_write_test_performed": False,
        "blockers": _stable_codes(blockers),
    }


def _schema_check(branch: str, branch_output: Mapping[str, Any]) -> dict[str, Any]:
    schema = default_regime_production_label_output_schema(branch).as_dict()
    approved = dict(branch_output.get("output_schema") or {})
    blockers = []
    for field_name in ("schema_id", "schema_version", "schema_hash"):
        if schema.get(field_name) != approved.get(field_name):
            blockers.append(f"schema_{field_name}_mismatch")
    return {
        "passed": not blockers,
        "expected_schema": schema,
        "validated_sandbox_schema": approved,
        "fixed_schema": True,
        "blockers": _stable_codes(blockers),
    }


def _clamp_check(wider_summary: Mapping[str, Any]) -> dict[str, Any]:
    range_used = dict(wider_summary.get("range_used") or {})
    blockers: list[str] = []
    if not range_used:
        blockers.append("clamp_range_missing")
    if not range_used.get("checkpoint_timestamps"):
        blockers.append("clamp_checkpoint_timestamps_missing")
    if range_used.get("full_one_year_bar_materialization_performed") is not False:
        blockers.append("unexpected_full_materialization_in_rehearsal")
    return {
        "passed": not blockers,
        "accepted": not blockers,
        "range": to_jsonable(range_used),
        "range_hash": _mapping_hash(range_used),
        "range_start": range_used.get("output_start") or _min_value(range_used.get("checkpoint_timestamps") or ()),
        "range_end": range_used.get("output_end") or _max_value(range_used.get("checkpoint_timestamps") or ()),
        "blockers": _stable_codes(blockers),
    }


def _cross_relationship_check(sandbox_validation: Mapping[str, Any]) -> dict[str, Any]:
    findings = dict(sandbox_validation.get("cross_asset_findings") or {})
    blockers: list[str] = []
    if findings.get("relationship_input_freshness_recorded") is not True:
        blockers.append("cross_relationship_freshness_not_recorded")
    if findings.get("relationship_inputs_available_fresh") is not True:
        blockers.append("cross_relationship_inputs_not_fresh")
    if int(findings.get("relationship_input_warning_count") or 0) != 0:
        blockers.append("cross_relationship_input_warnings_present")
    if findings.get("relationship_discovery_executed") is not False:
        blockers.append("relationship_discovery_executed")
    if findings.get("broad_pairwise_run_executed") is not False:
        blockers.append("broad_pairwise_run_executed")
    return {
        "applicable": True,
        "passed": not blockers,
        "findings": to_jsonable(findings),
        "relationship_discovery_invoked": False,
        "broad_pairwise_invoked": False,
        "blockers": _stable_codes(blockers),
    }


def _write_lock_check(
    branch: str,
    output_roots: Mapping[str, Any],
    clamp_check: Mapping[str, Any],
    *,
    stale_after_seconds: int,
    project_root: str | Path | None,
) -> dict[str, Any]:
    canonical_output_root = output_roots.get("canonical_output_root")
    if not canonical_output_root:
        return {
            "passed": False,
            "mode": REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
            "lock_acquired": False,
            "lock_path": None,
            "existing_lock_status": None,
            "blockers": ["write_lock_output_root_missing"],
        }
    target = RegimeProductionRunLockTarget(
        branch=branch,
        output_root=canonical_output_root,
        range_start=str(clamp_check.get("range_start") or "unknown_range_start"),
        range_end=str(clamp_check.get("range_end") or "unknown_range_end"),
        mode=REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
    )
    lock_root = Path(str(canonical_output_root)) / "_run_locks"
    lock_path = regime_production_run_lock_path(target, lock_root=lock_root, project_root=project_root)
    blockers: list[str] = []
    existing_status = None
    stale = False
    if lock_path.exists():
        existing = _load_json(lock_path, project_root=project_root)
        existing_status = existing.get("status")
        stale = regime_production_run_lock_is_stale(existing)
        if existing_status == REGIME_PRODUCTION_LOCK_STATUS_ACTIVE and not stale:
            blockers.append("active_write_lock_exists")
    return {
        "passed": not blockers,
        "mode": REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
        "lock_acquired": False,
        "lock_path": _portable_path_text(lock_path),
        "existing_lock_status": existing_status,
        "existing_lock_stale": stale,
        "stale_after_seconds": int(stale_after_seconds),
        "lock_check_read_only": True,
        "blockers": _stable_codes(blockers),
    }


def _expected_counts(
    branch: str,
    sandbox_validation: Mapping[str, Any],
    wider_summary: Mapping[str, Any],
) -> dict[str, Any]:
    row_counts = dict(sandbox_validation.get("row_count_by_branch") or wider_summary.get("row_count_by_branch") or {})
    partition_counts = dict(
        sandbox_validation.get("directory_partition_count_by_branch")
        or sandbox_validation.get("logical_grain_count_by_branch")
        or wider_summary.get("logical_partition_count_by_branch")
        or {}
    )
    mask_counts = dict(sandbox_validation.get("mask_or_unavailable_row_count_by_branch") or wider_summary.get("mask_or_unavailable_row_count_by_branch") or {})
    return {
        "expected_row_count": int(row_counts.get(branch) or 0),
        "expected_partition_count": int(partition_counts.get(branch) or 0),
        "expected_mask_or_unavailable_count": int(mask_counts.get(branch) or 0),
        "count_source": "sandbox_forecaster_validation_or_wider_sandbox_summary",
    }


def _canonical_command_config(
    branch: str,
    approval_path: str | None,
    output_roots: Mapping[str, Any],
    clamp_check: Mapping[str, Any],
) -> dict[str, Any]:
    module_by_branch = {
        REGIME_BRANCH_MARKET_STATE: "src.regimes.market_state.production",
        REGIME_BRANCH_ASSET_STATE: "src.regimes.asset_state.production",
        REGIME_BRANCH_CROSS_ASSET_STATE: "src.regimes.cross_asset_state.production",
    }
    command = f"python -m {module_by_branch[branch]}"
    if approval_path:
        command = f"{command} --branch-approval {branch}={approval_path}"
    return {
        "command": command,
        "reason": "Default branch Production command runs the full canonical writer and remains gated by approval, root, lock, and operator checklist checks.",
        "config": {
            "branch": branch,
            "approval_artifact_path": approval_path,
            "canonical_output_root": output_roots.get("canonical_output_root"),
            "model_state_root": output_roots.get("model_state_root"),
            "definition_root": output_roots.get("definition_root"),
            "telemetry_root": output_roots.get("telemetry_root"),
            "clamp_range": clamp_check.get("range"),
            "production_writer_enabled": True,
            "canonical_write_execution_allowed": True,
            "default_run_scope": "full_branch",
        },
    }


def _canonical_roots(*, env: Mapping[str, str] | None, project_root: str | Path | None) -> dict[str, Any]:
    keys = ("output_parquet_root", "state_root", "regime_definition_root", "log_root")
    configured: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        raw = resolve_path(key, env=env, required=False)
        if raw is None:
            missing.append(key)
        else:
            configured[key] = str(resolve_project_path(raw, project_root=project_root))
    return {
        "configured_roots": configured,
        "missing_root_keys": missing,
        "all_required_roots_configured": not missing,
        "canonical_root_touched": False,
        "canonical_write_test_performed": False,
    }


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
            if isinstance(payload, Mapping) and payload.get("artifact_kind") == "regime_production_operator_approval_checklist":
                return path
    return None


def _next_pass_branch(branch_rehearsals: Mapping[str, Mapping[str, Any]]) -> str | None:
    for branch in DEFAULT_CANONICAL_RUN_ORDER:
        if branch_rehearsals.get(branch, {}).get("status") == PRECANONICAL_PASS:
            return branch
    return None


def _load_json(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_project_path(path, project_root=project_root)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production pre-canonical rehearsal expected a JSON object")
    return payload


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(to_jsonable(dict(value)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _min_value(values: Sequence[Any]) -> Any:
    cleaned = [value for value in values if value not in (None, "")]
    return min(cleaned) if cleaned else None


def _max_value(values: Sequence[Any]) -> Any:
    cleaned = [value for value in values if value not in (None, "")]
    return max(cleaned) if cleaned else None


def _stable_codes(values: Sequence[str]) -> list[str]:
    return list(sorted(dict.fromkeys(str(value).strip() for value in values if str(value).strip())))


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


def _looks_like_absolute_path(value: str) -> bool:
    try:
        return Path(value).is_absolute()
    except Exception:
        return False


__all__ = [
    "DEFAULT_CANONICAL_RUN_ORDER",
    "PRECANONICAL_BLOCKED",
    "PRECANONICAL_PASS",
    "REGIME_PRODUCTION_PRECANONICAL_BRANCH_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PRECANONICAL_REHEARSAL_SCHEMA_VERSION",
    "RegimeProductionPreCanonicalRehearsalConfig",
    "run_regime_production_precanonical_rehearsal",
    "write_regime_production_precanonical_rehearsal_summary",
]
