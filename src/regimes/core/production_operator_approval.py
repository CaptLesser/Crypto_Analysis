from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_canonical_readiness_gate import DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
    resolve_active_selected_profile_artifact,
)
from src.regimes.core.production_promotion_gate import (
    DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH,
    RegimeProductionPromotionGateContext,
)
from src.regimes.core.production_sandbox_validation import DEFAULT_WIDER_SANDBOX_SUMMARY_PATH
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_OPERATOR_APPROVAL_SCHEMA_VERSION = 1
REGIME_PRODUCTION_OPERATOR_APPROVAL_CHECKLIST_ARTIFACT_KIND = "regime_production_operator_approval_checklist"
REGIME_PRODUCTION_OPERATOR_APPROVAL_VALIDATION_ARTIFACT_KIND = "regime_production_operator_approval_checklist_validation"

OPERATOR_CHECKLIST_STATUS_SAMPLE_DRY_FIXTURE = "sample_dry_fixture_not_executable"
OPERATOR_CHECKLIST_STATUS_PREWRITE_REHEARSAL = "prewrite_rehearsal_scaffold"
OPERATOR_CHECKLIST_STATUS_OPERATOR_APPROVED = "operator_approved_for_canonical_write_preflight"

OPERATOR_CHECKLIST_VALIDATION_STATUS_BLOCKED = "blocked"
OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_DRY_CHECKLIST_VALIDATION = "accepted_for_dry_checklist_validation"
OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_PREWRITE_REHEARSAL = "accepted_for_prewrite_rehearsal"
OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT = "accepted_for_operator_preflight"

DEFAULT_CANONICAL_RUN_ORDER: tuple[str, ...] = (
    REGIME_BRANCH_MARKET_STATE,
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
)


@dataclass(frozen=True)
class RegimeProductionOperatorChecklistContext:
    promotion_context: RegimeProductionPromotionGateContext
    write_capable_sandbox_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "write_capable_sandbox_summary",
            to_jsonable(dict(self.write_capable_sandbox_summary)),
        )

    @classmethod
    def from_paths(
        cls,
        *,
        sandbox_validation_summary_path: str | Path = DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH,
        wider_sandbox_summary_path: str | Path | None = None,
        write_capable_sandbox_summary_path: str | Path = DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH,
        project_root: str | Path | None = None,
    ) -> "RegimeProductionOperatorChecklistContext":
        promotion_context = RegimeProductionPromotionGateContext.from_paths(
            sandbox_validation_summary_path=sandbox_validation_summary_path,
            wider_sandbox_summary_path=wider_sandbox_summary_path,
        )
        return cls(
            promotion_context=promotion_context,
            write_capable_sandbox_summary=_load_json(
                write_capable_sandbox_summary_path,
                project_root=project_root,
            ),
        )


def build_sample_dry_regime_production_operator_approval_fixture(
    context: RegimeProductionOperatorChecklistContext | Mapping[str, Any],
    *,
    checklist_id: str = "regime_production_sample_dry_operator_checklist",
    operator_timestamp: str = "2026-06-07T00:00:00Z",
    operator_source: str = "codex_sample_dry_fixture",
    operator_id: str = "codex_sample_fixture_not_human_approval",
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    return build_regime_production_operator_approval_checklist(
        context,
        checklist_id=checklist_id,
        checklist_status=OPERATOR_CHECKLIST_STATUS_SAMPLE_DRY_FIXTURE,
        operator_timestamp=operator_timestamp,
        operator_source=operator_source,
        operator_id=operator_id,
        operator_approval_executed=False,
        dry_fixture_only=True,
        accepted_validation_issues=_required_validation_notes(_ctx(context).promotion_context.sandbox_validation_summary),
        canonical_run_order=DEFAULT_CANONICAL_RUN_ORDER,
        output_root_confirmations=_default_output_root_confirmations(),
        rollback_plan=_default_rollback_plan(),
        clamp_range_accepted=True,
        cross_relationship_freshness_policy_accepted=True,
        env=env,
        project_root=project_root,
    )


def build_prewrite_regime_production_operator_approval_scaffold(
    context: RegimeProductionOperatorChecklistContext | Mapping[str, Any],
    *,
    checklist_id: str,
    operator_timestamp: str,
    operator_source: str,
    operator_id: str,
    output_root_confirmations: Mapping[str, Mapping[str, Any]],
    rollback_plan: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a non-executable checklist scaffold for no-write pre-canonical rehearsal."""
    return build_regime_production_operator_approval_checklist(
        context,
        checklist_id=checklist_id,
        checklist_status=OPERATOR_CHECKLIST_STATUS_PREWRITE_REHEARSAL,
        operator_timestamp=operator_timestamp,
        operator_source=operator_source,
        operator_id=operator_id,
        operator_approval_executed=False,
        dry_fixture_only=False,
        accepted_validation_issues=_required_validation_notes(_ctx(context).promotion_context.sandbox_validation_summary),
        canonical_run_order=DEFAULT_CANONICAL_RUN_ORDER,
        output_root_confirmations=output_root_confirmations,
        rollback_plan=rollback_plan or _default_rollback_plan(),
        clamp_range_accepted=True,
        cross_relationship_freshness_policy_accepted=True,
        env=env,
        project_root=project_root,
    )


def build_regime_production_operator_approval_checklist(
    context: RegimeProductionOperatorChecklistContext | Mapping[str, Any],
    *,
    checklist_id: str,
    checklist_status: str,
    operator_timestamp: str,
    operator_source: str,
    operator_id: str,
    operator_approval_executed: bool,
    dry_fixture_only: bool,
    accepted_validation_issues: Sequence[str],
    canonical_run_order: Sequence[str],
    output_root_confirmations: Mapping[str, Mapping[str, Any]],
    rollback_plan: Mapping[str, Any],
    clamp_range_accepted: bool,
    cross_relationship_freshness_policy_accepted: bool,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    promotion_context = ctx.promotion_context
    sandbox_validation = promotion_context.sandbox_validation_summary
    wider = promotion_context.wider_sandbox_summary
    write_capable = ctx.write_capable_sandbox_summary
    branch_evidence = {
        branch: _branch_evidence(branch, promotion_context, env=env, project_root=project_root)
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    payload = {
        "schema_version": REGIME_PRODUCTION_OPERATOR_APPROVAL_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_OPERATOR_APPROVAL_CHECKLIST_ARTIFACT_KIND,
        "checklist_id": _text(checklist_id, field_name="checklist_id"),
        "checklist_status": _text(checklist_status, field_name="checklist_status"),
        "dry_fixture_only": bool(dry_fixture_only),
        "operator_approval_executed": bool(operator_approval_executed),
        "operator": {
            "operator_id": _text(operator_id, field_name="operator_id"),
            "operator_timestamp": _text(operator_timestamp, field_name="operator_timestamp"),
            "operator_source": _text(operator_source, field_name="operator_source"),
        },
        "active_selected_profile_artifacts": branch_evidence,
        "repaired_asset_artifact_active": branch_evidence[REGIME_BRANCH_ASSET_STATE]["repaired_asset_artifact_active"],
        "market_artifact_active": branch_evidence[REGIME_BRANCH_MARKET_STATE]["active_resolution_passed"],
        "cross_artifact_active": branch_evidence[REGIME_BRANCH_CROSS_ASSET_STATE]["active_resolution_passed"],
        "sandbox_validation": {
            "run_id": sandbox_validation.get("run_id"),
            "validation_status": sandbox_validation.get("validation_status"),
            "summary_hash": _mapping_hash(sandbox_validation),
            "issues": to_jsonable(dict(sandbox_validation.get("issues") or {})),
            "accepted_validation_issues": list(dict.fromkeys(str(item) for item in accepted_validation_issues if str(item).strip())),
            "blocker_count": len(sandbox_validation.get("issues", {}).get("blocker", ()) or ()),
            "high_count": len(sandbox_validation.get("issues", {}).get("high", ()) or ()),
        },
        "write_capable_sandbox_validation": {
            "run_id": write_capable.get("run_id"),
            "summary_hash": _mapping_hash(write_capable),
            "validation_results": to_jsonable(dict(write_capable.get("validation_results") or {})),
            "canonical_root_touched": bool(write_capable.get("canonical_root_touched")),
            "accepted_validation_issues": list(write_capable.get("accepted_validation_issues") or ()),
        },
        "clamp_range": {
            "accepted": bool(clamp_range_accepted),
            "range": to_jsonable(dict(wider.get("range_used") or {})),
            "range_hash": _mapping_hash(dict(wider.get("range_used") or {})),
        },
        "cross_relationship_freshness_policy": {
            "accepted": bool(cross_relationship_freshness_policy_accepted),
            "current_findings": to_jsonable(dict(sandbox_validation.get("cross_asset_findings") or {})),
        },
        "output_root_confirmations": to_jsonable({branch: dict(output_root_confirmations.get(branch) or {}) for branch in REGIME_PRODUCTION_BRANCHES}),
        "schema_versions_confirmed": {
            branch: {
                "schema_id": dict(dict(wider.get("branch_outputs") or {}).get(branch, {}).get("output_schema") or {}).get("schema_id"),
                "schema_version": dict(dict(wider.get("branch_outputs") or {}).get(branch, {}).get("output_schema") or {}).get("schema_version"),
                "schema_hash": dict(dict(wider.get("branch_outputs") or {}).get(branch, {}).get("output_schema") or {}).get("schema_hash"),
                "confirmed": True,
            }
            for branch in REGIME_PRODUCTION_BRANCHES
        },
        "no_blocker_high_open_items": not (
            sandbox_validation.get("issues", {}).get("blocker")
            or sandbox_validation.get("issues", {}).get("high")
        ),
        "canonical_run_order": list(canonical_run_order),
        "canonical_run_order_selected": bool(canonical_run_order),
        "rollback_plan": to_jsonable(dict(rollback_plan)),
        "operator_checklist_schema_finalized": True,
        "sample_dry_approval_fixture_only": bool(dry_fixture_only),
        "canonical_write_execution_allowed": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "write_sprint_enable_required": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def validate_regime_production_operator_approval_checklist(
    checklist: Mapping[str, Any] | str | Path | None,
    context: RegimeProductionOperatorChecklistContext | Mapping[str, Any],
    *,
    allow_sample_dry_fixture: bool = False,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    ctx = _ctx(context)
    payload = _checklist_payload(checklist, project_root=project_root)
    blockers: list[str] = []
    warnings: list[str] = []
    payload_present = payload is not None
    if payload is None:
        blockers.append("missing_operator_approval_checklist")
        payload = {}
    else:
        blockers.extend(_validate_checklist_payload(payload, ctx, env=env, project_root=project_root))

    dry_fixture = bool(payload.get("dry_fixture_only"))
    if dry_fixture and not allow_sample_dry_fixture:
        blockers.append("sample_dry_fixture_not_executable")
    if dry_fixture:
        warnings.append("sample_dry_fixture_validates_schema_only")
    if payload_present and payload.get("canonical_write_execution_allowed") is not False:
        blockers.append("checklist_attempts_to_allow_canonical_write_execution")
    if payload_present and payload.get("production_writer_enabled") is not False:
        blockers.append("checklist_attempts_to_enable_writer")

    blockers = _stable_codes(blockers)
    warnings = _stable_codes(warnings)
    if blockers:
        status = OPERATOR_CHECKLIST_VALIDATION_STATUS_BLOCKED
    elif dry_fixture:
        status = OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_DRY_CHECKLIST_VALIDATION
    elif payload.get("checklist_status") == OPERATOR_CHECKLIST_STATUS_PREWRITE_REHEARSAL:
        status = OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_PREWRITE_REHEARSAL
    else:
        status = OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_OPERATOR_APPROVAL_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_OPERATOR_APPROVAL_VALIDATION_ARTIFACT_KIND,
                "validation_status": status,
                "checklist_id": payload.get("checklist_id"),
                "checklist_status": payload.get("checklist_status"),
                "dry_fixture_only": dry_fixture,
                "operator_approval_executed": bool(payload.get("operator_approval_executed")),
                "blockers": blockers,
                "warnings": warnings,
                "checklist_fields_verified": [
                    "active selected-profile artifact path/hash",
                    "repaired Asset artifact active",
                    "Market artifact active",
                    "Cross artifact active",
                    "sandbox validation clean or accepted notes",
                    "write-capable sandbox validation clean",
                    "clamp/range accepted",
                    "Cross relationship freshness policy accepted",
                    "output roots confirmed",
                    "schema versions confirmed",
                    "no blocker/high open items",
                    "canonical run order selected",
                    "rollback plan present",
                    "operator timestamp/source",
                ],
                "canonical_write_execution_allowed": False,
                "production_writer_enabled": False,
                "production_labels_written": False,
                "production_outputs_written": False,
                "canonical_production_state_outputs_written": False,
                "production_promotion_performed": False,
                "production_writer_gates_fail_closed": True,
            }
        )
    )


def _validate_checklist_payload(
    payload: Mapping[str, Any],
    ctx: RegimeProductionOperatorChecklistContext,
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> list[str]:
    blockers: list[str] = []
    sandbox_validation = ctx.promotion_context.sandbox_validation_summary
    wider = ctx.promotion_context.wider_sandbox_summary
    write_capable = ctx.write_capable_sandbox_summary
    if payload.get("artifact_kind") != REGIME_PRODUCTION_OPERATOR_APPROVAL_CHECKLIST_ARTIFACT_KIND:
        blockers.append("unsupported_operator_checklist_artifact_kind")
    status = str(payload.get("checklist_status") or "")
    if status not in {
        OPERATOR_CHECKLIST_STATUS_SAMPLE_DRY_FIXTURE,
        OPERATOR_CHECKLIST_STATUS_PREWRITE_REHEARSAL,
        OPERATOR_CHECKLIST_STATUS_OPERATOR_APPROVED,
    }:
        blockers.append("unsupported_operator_checklist_status")
    if status == OPERATOR_CHECKLIST_STATUS_OPERATOR_APPROVED and payload.get("operator_approval_executed") is not True:
        blockers.append("operator_approval_execution_missing")
    if status == OPERATOR_CHECKLIST_STATUS_PREWRITE_REHEARSAL and payload.get("operator_approval_executed") is not False:
        blockers.append("prewrite_scaffold_cannot_execute_operator_approval")
    if status == OPERATOR_CHECKLIST_STATUS_SAMPLE_DRY_FIXTURE and payload.get("operator_approval_executed") is not False:
        blockers.append("sample_fixture_cannot_execute_operator_approval")
    blockers.extend(_validate_operator(payload.get("operator") or {}))
    blockers.extend(_validate_active_artifacts(payload, ctx.promotion_context, env=env, project_root=project_root))
    sandbox = dict(payload.get("sandbox_validation") or {})
    if sandbox.get("run_id") != sandbox_validation.get("run_id"):
        blockers.append("sandbox_validation_run_id_mismatch")
    if sandbox.get("summary_hash") != _mapping_hash(sandbox_validation):
        blockers.append("sandbox_validation_summary_hash_mismatch")
    if sandbox_validation.get("issues", {}).get("blocker") or sandbox_validation.get("issues", {}).get("high"):
        blockers.append("sandbox_validation_has_blocker_or_high_items")
    accepted = set(str(item) for item in sandbox.get("accepted_validation_issues") or ())
    for note in _required_validation_notes(sandbox_validation):
        if note not in accepted:
            blockers.append(f"unaccepted_validation_issue:{note}")
    write_capable_payload = dict(payload.get("write_capable_sandbox_validation") or {})
    if write_capable_payload.get("run_id") != write_capable.get("run_id"):
        blockers.append("write_capable_sandbox_run_id_mismatch")
    if write_capable_payload.get("summary_hash") != _mapping_hash(write_capable):
        blockers.append("write_capable_sandbox_summary_hash_mismatch")
    blockers.extend(_validate_write_capable_summary(write_capable))
    clamp = dict(payload.get("clamp_range") or {})
    if clamp.get("accepted") is not True:
        blockers.append("clamp_range_not_accepted")
    if clamp.get("range_hash") != _mapping_hash(dict(wider.get("range_used") or {})):
        blockers.append("clamp_range_hash_mismatch")
    cross = dict(payload.get("cross_relationship_freshness_policy") or {})
    cross_findings = dict(sandbox_validation.get("cross_asset_findings") or {})
    if cross.get("accepted") is not True:
        blockers.append("cross_relationship_freshness_policy_not_accepted")
    if cross_findings.get("relationship_discovery_executed") is not False:
        blockers.append("cross_relationship_discovery_was_executed")
    if cross_findings.get("broad_pairwise_run_executed") is not False:
        blockers.append("cross_broad_pairwise_was_executed")
    if cross_findings.get("relationship_inputs_available_fresh") is not True:
        blockers.append("cross_relationship_inputs_not_fresh")
    blockers.extend(_validate_output_roots(payload.get("output_root_confirmations") or {}))
    blockers.extend(_validate_schema_confirmations(payload.get("schema_versions_confirmed") or {}, wider))
    if payload.get("no_blocker_high_open_items") is not True:
        blockers.append("no_blocker_high_open_items_not_confirmed")
    if tuple(payload.get("canonical_run_order") or ()) != DEFAULT_CANONICAL_RUN_ORDER:
        blockers.append("canonical_run_order_mismatch")
    rollback = dict(payload.get("rollback_plan") or {})
    if rollback.get("rollback_plan_present") is not True:
        blockers.append("rollback_plan_missing")
    if rollback.get("rollback_state_capture_required_before_write") is not True:
        blockers.append("rollback_state_capture_not_required")
    if rollback.get("rollback_execution_performed") is not False:
        blockers.append("rollback_execution_must_not_be_performed_in_approval")
    for flag_name in (
        "production_labels_written",
        "production_outputs_written",
        "canonical_production_state_outputs_written",
        "production_promotion_performed",
    ):
        if payload.get(flag_name) is not False:
            blockers.append(f"{flag_name}_must_remain_false")
    return blockers


def _branch_evidence(
    branch: str,
    promotion_context: RegimeProductionPromotionGateContext,
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> dict[str, Any]:
    active = resolve_active_selected_profile_artifact(branch, env=env, project_root=project_root)
    branch_output = dict(promotion_context.wider_sandbox_summary.get("branch_outputs", {}).get(branch) or {})
    active_path = active.artifact_path
    expected_path = branch_output.get("source_artifact_path")
    expected_hash = branch_output.get("profile_artifact_hash")
    active_hash = _sha256_file(active_path) if active.passed and active_path is not None else None
    path_matches = False
    if active_path is not None and expected_path:
        path_matches = resolve_project_path(expected_path, project_root=project_root).resolve() == Path(active_path).resolve()
    return {
        "branch": branch,
        "active_resolution_status": active.status,
        "active_resolution_passed": active.passed,
        "active_artifact_path": None if active_path is None else _portable_path_text(active_path),
        "active_artifact_hash": active_hash,
        "expected_selected_profile_artifact_path": expected_path,
        "expected_selected_profile_artifact_hash": expected_hash,
        "active_matches_expected_path": path_matches,
        "active_matches_expected_hash": active_hash == expected_hash,
        "repaired_asset_artifact_active": bool(
            branch == REGIME_BRANCH_ASSET_STATE
            and active_path is not None
            and "lineage_repaired" in active_path.name
        ),
    }


def _validate_active_artifacts(
    payload: Mapping[str, Any],
    promotion_context: RegimeProductionPromotionGateContext,
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> list[str]:
    blockers: list[str] = []
    current = {
        branch: _branch_evidence(branch, promotion_context, env=env, project_root=project_root)
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    approved = dict(payload.get("active_selected_profile_artifacts") or {})
    for branch, current_payload in current.items():
        approved_payload = dict(approved.get(branch) or {})
        if not approved_payload:
            blockers.append(f"{branch}:active_artifact_check_missing")
            continue
        if current_payload["active_resolution_passed"] is not True:
            blockers.append(f"{branch}:active_artifact_resolution_failed")
        for field_name in (
            "active_artifact_path",
            "active_artifact_hash",
            "expected_selected_profile_artifact_path",
            "expected_selected_profile_artifact_hash",
        ):
            if approved_payload.get(field_name) != current_payload.get(field_name):
                blockers.append(f"{branch}:{field_name}_mismatch")
        if current_payload["active_matches_expected_path"] is not True:
            blockers.append(f"{branch}:active_selected_profile_artifact_path_mismatch")
        if current_payload["active_matches_expected_hash"] is not True:
            blockers.append(f"{branch}:active_selected_profile_artifact_hash_mismatch")
    if current[REGIME_BRANCH_ASSET_STATE]["repaired_asset_artifact_active"] is not True:
        blockers.append("repaired_asset_artifact_not_active")
    if payload.get("repaired_asset_artifact_active") is not True:
        blockers.append("repaired_asset_artifact_not_confirmed")
    if payload.get("market_artifact_active") is not True:
        blockers.append("market_artifact_not_confirmed_active")
    if payload.get("cross_artifact_active") is not True:
        blockers.append("cross_artifact_not_confirmed_active")
    return blockers


def _validate_write_capable_summary(write_capable: Mapping[str, Any]) -> list[str]:
    results = dict(write_capable.get("validation_results") or {})
    required = {
        "writer_gating_passed": True,
        "output_schema_validation_passed": True,
        "partitioning_validation_passed": True,
        "no_mixed_schemas": True,
        "atomic_staged_writes_validated": True,
        "resume_behavior_validated": True,
        "idempotency_validated": True,
        "canonical_root_touched": False,
    }
    blockers = [
        f"write_capable_sandbox_validation:{key}_mismatch"
        for key, expected in required.items()
        if results.get(key) is not expected
    ]
    if bool(write_capable.get("canonical_root_touched")):
        blockers.append("write_capable_sandbox_canonical_root_touched")
    return blockers


def _validate_output_roots(value: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    roots = {branch: dict(dict(value).get(branch) or {}) for branch in REGIME_PRODUCTION_BRANCHES}
    for branch, root in roots.items():
        if root.get("canonical_output_root_confirmed") is not True:
            blockers.append(f"{branch}:canonical_output_root_not_confirmed")
        if not str(root.get("canonical_output_root_key") or "").strip():
            blockers.append(f"{branch}:canonical_output_root_key_missing")
        if not str(root.get("canonical_output_root_reference") or "").strip():
            blockers.append(f"{branch}:canonical_output_root_reference_missing")
        if root.get("canonical_root_touched") is not False:
            blockers.append(f"{branch}:canonical_root_touched_during_checklist")
        if root.get("canonical_root_write_test_performed") is not False:
            blockers.append(f"{branch}:canonical_root_write_test_performed")
    return blockers


def _validate_schema_confirmations(value: Mapping[str, Any], wider: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    confirmations = {branch: dict(dict(value).get(branch) or {}) for branch in REGIME_PRODUCTION_BRANCHES}
    outputs = dict(wider.get("branch_outputs") or {})
    for branch in REGIME_PRODUCTION_BRANCHES:
        expected = dict(dict(outputs.get(branch) or {}).get("output_schema") or {})
        actual = confirmations[branch]
        if actual.get("confirmed") is not True:
            blockers.append(f"{branch}:schema_not_confirmed")
        for field_name in ("schema_id", "schema_version", "schema_hash"):
            if actual.get(field_name) != expected.get(field_name):
                blockers.append(f"{branch}:{field_name}_mismatch")
    return blockers


def _validate_operator(value: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not str(value.get("operator_id") or "").strip():
        blockers.append("operator_id_missing")
    if not str(value.get("operator_source") or "").strip():
        blockers.append("operator_source_missing")
    timestamp = str(value.get("operator_timestamp") or "")
    if not _iso_timestamp(timestamp):
        blockers.append("operator_timestamp_invalid")
    return blockers


def _required_validation_notes(sandbox_validation: Mapping[str, Any]) -> tuple[str, ...]:
    notes = []
    for severity in ("medium", "low"):
        notes.extend(str(item) for item in sandbox_validation.get("issues", {}).get(severity, ()) or ())
    return tuple(dict.fromkeys(note for note in notes if note))


def _default_output_root_confirmations() -> dict[str, dict[str, Any]]:
    return {
        branch: {
            "canonical_output_root_confirmed": True,
            "canonical_output_root_key": f"{branch}_canonical_output_root",
            "canonical_output_root_reference": f"configured_canonical_root:{branch}",
            "canonical_root_matches_expected_environment": True,
            "canonical_root_write_test_performed": False,
            "canonical_root_touched": False,
        }
        for branch in REGIME_PRODUCTION_BRANCHES
    }


def _default_rollback_plan() -> dict[str, Any]:
    return {
        "rollback_plan_present": True,
        "rollback_plan_id": "regime_production_operator_manual_rollback_plan_v1",
        "previous_active_state_capture_required": True,
        "rollback_state_capture_required_before_write": True,
        "manual_rollback_confirmation_required": True,
        "rollback_execution_performed": False,
    }


def _ctx(value: RegimeProductionOperatorChecklistContext | Mapping[str, Any]) -> RegimeProductionOperatorChecklistContext:
    if isinstance(value, RegimeProductionOperatorChecklistContext):
        return value
    payload = dict(value)
    promotion_context = payload.get("promotion_context")
    if isinstance(promotion_context, RegimeProductionPromotionGateContext):
        context = promotion_context
    else:
        context_payload = dict(promotion_context or {})
        context = RegimeProductionPromotionGateContext(
            sandbox_validation_summary=dict(context_payload.get("sandbox_validation_summary") or {}),
            wider_sandbox_summary=dict(context_payload.get("wider_sandbox_summary") or {}),
        )
    return RegimeProductionOperatorChecklistContext(
        promotion_context=context,
        write_capable_sandbox_summary=dict(payload.get("write_capable_sandbox_summary") or {}),
    )


def _checklist_payload(value: Mapping[str, Any] | str | Path | None, *, project_root: str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return to_jsonable(dict(value))
    return _load_json(value, project_root=project_root)


def _load_json(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_project_path(path, project_root=project_root)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production operator approval expected a JSON object")
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


def _iso_timestamp(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


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


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production operator approval {field_name} must be non-empty")
    return text


__all__ = [
    "DEFAULT_CANONICAL_RUN_ORDER",
    "DEFAULT_WIDER_SANDBOX_SUMMARY_PATH",
    "OPERATOR_CHECKLIST_STATUS_OPERATOR_APPROVED",
    "OPERATOR_CHECKLIST_STATUS_PREWRITE_REHEARSAL",
    "OPERATOR_CHECKLIST_STATUS_SAMPLE_DRY_FIXTURE",
    "OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_DRY_CHECKLIST_VALIDATION",
    "OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_OPERATOR_PREFLIGHT",
    "OPERATOR_CHECKLIST_VALIDATION_STATUS_ACCEPTED_FOR_PREWRITE_REHEARSAL",
    "OPERATOR_CHECKLIST_VALIDATION_STATUS_BLOCKED",
    "REGIME_PRODUCTION_OPERATOR_APPROVAL_CHECKLIST_ARTIFACT_KIND",
    "REGIME_PRODUCTION_OPERATOR_APPROVAL_SCHEMA_VERSION",
    "REGIME_PRODUCTION_OPERATOR_APPROVAL_VALIDATION_ARTIFACT_KIND",
    "RegimeProductionOperatorChecklistContext",
    "build_prewrite_regime_production_operator_approval_scaffold",
    "build_regime_production_operator_approval_checklist",
    "build_sample_dry_regime_production_operator_approval_fixture",
    "validate_regime_production_operator_approval_checklist",
]
