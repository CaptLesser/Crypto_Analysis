from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_sandbox_validation import DEFAULT_WIDER_SANDBOX_SUMMARY_PATH
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION = 1
REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND = "regime_production_branch_promotion_approval"
REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND = "regime_production_unified_promotion_approval"
REGIME_PRODUCTION_PROMOTION_GATE_RESULT_ARTIFACT_KIND = "regime_production_promotion_gate_result"
REGIME_PRODUCTION_PROMOTION_GATE_SUMMARY_ARTIFACT_KIND = "regime_production_promotion_gate_design_summary"

APPROVAL_STATUS_APPROVED_FOR_DRY_WRITE_PLANNING = "approved_for_dry_write_planning"
APPROVAL_STATUS_TEMPLATE_NOT_EXECUTED = "approval_template_not_executed"

GATE_STATUS_BLOCKED = "blocked"
GATE_STATUS_ACCEPTED_FOR_DRY_WRITE_PLANNING = "accepted_for_dry_write_planning"

DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH = (
    "_codex_artifacts/reports/regime_production_sandbox_forecaster_validation/"
    "regime_production_sandbox_forecaster_validation_summary.json"
)


@dataclass(frozen=True)
class RegimeProductionPromotionGateContext:
    sandbox_validation_summary: Mapping[str, Any]
    wider_sandbox_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        validation = to_jsonable(dict(self.sandbox_validation_summary))
        wider = to_jsonable(dict(self.wider_sandbox_summary))
        object.__setattr__(self, "sandbox_validation_summary", validation)
        object.__setattr__(self, "wider_sandbox_summary", wider)

    @classmethod
    def from_paths(
        cls,
        *,
        sandbox_validation_summary_path: str | Path = DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH,
        wider_sandbox_summary_path: str | Path | None = None,
    ) -> "RegimeProductionPromotionGateContext":
        validation = _load_json(Path(sandbox_validation_summary_path))
        wider_path = wider_sandbox_summary_path or validation.get("source_wider_sandbox_summary_path") or DEFAULT_WIDER_SANDBOX_SUMMARY_PATH
        return cls(
            sandbox_validation_summary=validation,
            wider_sandbox_summary=_load_json(Path(wider_path)),
        )


def build_regime_production_branch_approval_artifact(
    branch: str,
    context: RegimeProductionPromotionGateContext | Mapping[str, Any],
    *,
    approval_id: str,
    approval_timestamp: str,
    approval_operator: str,
    approval_source: str,
    canonical_output_root_confirmation: Mapping[str, Any],
    accepted_validation_issues: Sequence[str] = (),
    rollback_safety: Mapping[str, Any] | None = None,
    approval_status: str = APPROVAL_STATUS_APPROVED_FOR_DRY_WRITE_PLANNING,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    ctx = _context(context)
    branch_output = _branch_output(ctx, branch_name)
    branch_validation = _branch_validation(ctx, branch_name)
    schema = dict(branch_output.get("output_schema") or {})
    range_used = dict(ctx.wider_sandbox_summary.get("range_used") or {})
    payload = {
        "schema_version": REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND,
        "approval_id": _text(approval_id, field_name="approval_id"),
        "approval_status": _text(approval_status, field_name="approval_status"),
        "branch": branch_name,
        "approved_selected_profile_artifact_path": branch_output.get("source_artifact_path"),
        "approved_selected_profile_artifact_hash": branch_output.get("profile_artifact_hash"),
        "approved_sandbox_validation_run_id": ctx.sandbox_validation_summary.get("run_id"),
        "approved_sandbox_validation_status": ctx.sandbox_validation_summary.get("validation_status"),
        "approved_sandbox_validation_summary_hash": _mapping_hash(ctx.sandbox_validation_summary),
        "approved_output_schema_version": int(schema.get("schema_version") or 0),
        "approved_output_schema_id": schema.get("schema_id"),
        "approved_output_schema_hash": schema.get("schema_hash"),
        "approved_clamp_range": range_used,
        "approved_clamp_range_hash": _mapping_hash(range_used),
        "accepted_validation_issues": list(dict.fromkeys(str(item) for item in accepted_validation_issues if str(item).strip())),
        "approval_timestamp": _text(approval_timestamp, field_name="approval_timestamp"),
        "approval_operator": _text(approval_operator, field_name="approval_operator"),
        "approval_source": _text(approval_source, field_name="approval_source"),
        "canonical_output_root_confirmation": to_jsonable(dict(canonical_output_root_confirmation)),
        "rollback_safety": to_jsonable(dict(rollback_safety or _default_rollback_safety(branch_name))),
        "branch_sandbox_validation_fingerprint": {
            "row_count": branch_validation.get("row_count"),
            "logical_partition_count": branch_validation.get("logical_partition_count"),
            "mask_or_unavailable_row_count": branch_validation.get("mask_or_unavailable_row_count"),
            "validation_status": branch_validation.get("validation_status"),
        },
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "write_sprint_enable_required": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def evaluate_regime_production_branch_promotion_gate(
    approval: Mapping[str, Any] | str | Path | None,
    context: RegimeProductionPromotionGateContext | Mapping[str, Any],
    *,
    branch: str,
    write_sprint_enable_requested: bool = False,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    ctx = _context(context)
    blockers: list[str] = []
    warnings: list[str] = []
    approval_payload = _approval_payload(approval)
    if approval_payload is None:
        blockers.append("missing_branch_approval")
    else:
        blockers.extend(_validate_branch_approval(approval_payload, ctx, branch_name))
    accepted = not blockers
    if accepted and not bool(write_sprint_enable_requested):
        warnings.append("write_sprint_enable_not_requested")
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_PROMOTION_GATE_RESULT_ARTIFACT_KIND,
                "branch": branch_name,
                "gate_status": GATE_STATUS_ACCEPTED_FOR_DRY_WRITE_PLANNING if accepted else GATE_STATUS_BLOCKED,
                "dry_write_planning_allowed": accepted,
                "production_write_preconditions_satisfied": bool(accepted and write_sprint_enable_requested),
                "canonical_write_execution_allowed": False,
                "blockers": blockers,
                "warnings": warnings,
                "approval_id": None if approval_payload is None else approval_payload.get("approval_id"),
                "approved_sandbox_validation_run_id": None if approval_payload is None else approval_payload.get("approved_sandbox_validation_run_id"),
                "approved_selected_profile_artifact_hash": None if approval_payload is None else approval_payload.get("approved_selected_profile_artifact_hash"),
                "per_branch_approval_required": True,
                "unified_approval_can_reference_branch_approvals": True,
                "production_writer_enabled": False,
                "production_labels_written": False,
                "production_outputs_written": False,
                "canonical_production_state_outputs_written": False,
                "production_promotion_performed": False,
                "write_sprint_enable_requested": bool(write_sprint_enable_requested),
                "write_sprint_enable_required_before_canonical_writes": True,
                "production_writer_gates_fail_closed": True,
            }
        )
    )


def build_regime_production_unified_approval_artifact(
    branch_approvals: Mapping[str, Mapping[str, Any]],
    *,
    unified_approval_id: str,
    approval_timestamp: str,
    approval_operator: str,
    approval_source: str,
) -> dict[str, Any]:
    approvals = {
        _branch_name(branch): to_jsonable(dict(payload))
        for branch, payload in dict(branch_approvals).items()
    }
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND,
                "unified_approval_id": _text(unified_approval_id, field_name="unified_approval_id"),
                "approval_timestamp": _text(approval_timestamp, field_name="approval_timestamp"),
                "approval_operator": _text(approval_operator, field_name="approval_operator"),
                "approval_source": _text(approval_source, field_name="approval_source"),
                "approval_model": "per_branch_approvals_referenced_by_unified_gate",
                "branch_approval_ids": {
                    branch: payload.get("approval_id")
                    for branch, payload in approvals.items()
                },
                "branch_approval_hashes": {
                    branch: _mapping_hash(payload)
                    for branch, payload in approvals.items()
                },
                "branch_approvals": approvals,
                "production_writer_enabled": False,
                "production_labels_written": False,
                "production_outputs_written": False,
                "canonical_production_state_outputs_written": False,
                "production_promotion_performed": False,
                "write_sprint_enable_required": True,
            }
        )
    )


def evaluate_regime_production_unified_promotion_gate(
    unified_approval: Mapping[str, Any] | str | Path | None,
    context: RegimeProductionPromotionGateContext | Mapping[str, Any],
    *,
    write_sprint_enable_requested: bool = False,
) -> dict[str, Any]:
    ctx = _context(context)
    payload = _approval_payload(unified_approval)
    blockers: list[str] = []
    if payload is None:
        blockers.append("missing_unified_approval")
        branch_results = {
            branch: evaluate_regime_production_branch_promotion_gate(None, ctx, branch=branch)
            for branch in REGIME_PRODUCTION_BRANCHES
        }
    elif payload.get("artifact_kind") != REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND:
        blockers.append("unsupported_unified_approval_artifact_kind")
        branch_results = {}
    else:
        approvals = dict(payload.get("branch_approvals") or {})
        branch_results = {
            branch: evaluate_regime_production_branch_promotion_gate(
                approvals.get(branch),
                ctx,
                branch=branch,
                write_sprint_enable_requested=write_sprint_enable_requested,
            )
            for branch in REGIME_PRODUCTION_BRANCHES
        }
        missing = [branch for branch in REGIME_PRODUCTION_BRANCHES if branch not in approvals]
        blockers.extend(f"{branch}:missing_branch_approval" for branch in missing)
    for branch, result in branch_results.items():
        for blocker in result.get("blockers", ()):
            blockers.append(f"{branch}:{blocker}")
    accepted = not blockers
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_PROMOTION_GATE_RESULT_ARTIFACT_KIND,
                "gate_scope": "unified_three_branch_gate",
                "gate_status": GATE_STATUS_ACCEPTED_FOR_DRY_WRITE_PLANNING if accepted else GATE_STATUS_BLOCKED,
                "approval_model_decision": {
                    "per_branch_approval_is_primary": True,
                    "unified_approval_role": "references_three_branch_approvals",
                    "reason": "per-branch approval isolates artifact hash, schema, validation, root, and rollback evidence by branch",
                },
                "branch_results": branch_results,
                "dry_write_planning_allowed": accepted,
                "production_write_preconditions_satisfied": bool(accepted and write_sprint_enable_requested),
                "canonical_write_execution_allowed": False,
                "blockers": list(dict.fromkeys(blockers)),
                "production_writer_enabled": False,
                "production_labels_written": False,
                "production_outputs_written": False,
                "canonical_production_state_outputs_written": False,
                "production_promotion_performed": False,
                "write_sprint_enable_requested": bool(write_sprint_enable_requested),
                "write_sprint_enable_required_before_canonical_writes": True,
                "production_writer_gates_fail_closed": True,
            }
        )
    )


def build_regime_production_promotion_gate_design_summary(
    context: RegimeProductionPromotionGateContext | Mapping[str, Any],
) -> dict[str, Any]:
    ctx = _context(context)
    no_approval_results = {
        branch: evaluate_regime_production_branch_promotion_gate(None, ctx, branch=branch)
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_PROMOTION_GATE_SUMMARY_ARTIFACT_KIND,
                "approval_model_decision": {
                    "per_branch_approval_is_primary": True,
                    "unified_approval_role": "optional_wrapper_referencing_three_branch_approvals",
                    "reason": "per-branch approval is safer because selected-profile artifact hashes, validation notes, schemas, roots, and rollback evidence can differ by branch",
                },
                "branch_approval_artifact_kind": REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND,
                "unified_approval_artifact_kind": REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND,
                "required_branch_approval_fields": [
                    "branch",
                    "approved_selected_profile_artifact_path",
                    "approved_selected_profile_artifact_hash",
                    "approved_sandbox_validation_run_id",
                    "approved_sandbox_validation_status",
                    "approved_sandbox_validation_summary_hash",
                    "approved_output_schema_version",
                    "approved_output_schema_id",
                    "approved_output_schema_hash",
                    "approved_clamp_range",
                    "approved_clamp_range_hash",
                    "accepted_validation_issues",
                    "approval_timestamp",
                    "approval_operator",
                    "approval_source",
                    "canonical_output_root_confirmation",
                    "rollback_safety",
                    "production_writer_enabled",
                ],
                "source_sandbox_validation_run_id": ctx.sandbox_validation_summary.get("run_id"),
                "source_wider_sandbox_run_id": ctx.wider_sandbox_summary.get("run_id"),
                "validation_issues_requiring_acceptance": ctx.sandbox_validation_summary.get("issues"),
                "branch_gate_results_without_approvals": no_approval_results,
                "production_writer_enabled": False,
                "production_labels_written": False,
                "production_outputs_written": False,
                "canonical_production_state_outputs_written": False,
                "production_promotion_performed": False,
                "write_sprint_enable_required_before_canonical_writes": True,
                "production_writer_gates_fail_closed": True,
            }
        )
    )


def write_regime_production_promotion_gate_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _validate_branch_approval(
    approval: Mapping[str, Any],
    ctx: RegimeProductionPromotionGateContext,
    branch: str,
) -> list[str]:
    blockers: list[str] = []
    branch_output = _branch_output(ctx, branch)
    schema = dict(branch_output.get("output_schema") or {})
    validation = ctx.sandbox_validation_summary
    approval_branch = str(approval.get("branch") or "")
    if approval.get("artifact_kind") != REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND:
        blockers.append("unsupported_approval_artifact_kind")
    if approval_branch != branch:
        blockers.append("approval_branch_mismatch")
    if approval.get("approval_status") != APPROVAL_STATUS_APPROVED_FOR_DRY_WRITE_PLANNING:
        blockers.append("branch_not_approved")
    if approval.get("approved_selected_profile_artifact_path") != branch_output.get("source_artifact_path"):
        blockers.append("selected_profile_artifact_path_mismatch")
    if approval.get("approved_selected_profile_artifact_hash") != branch_output.get("profile_artifact_hash"):
        blockers.append("selected_profile_artifact_hash_mismatch")
    if approval.get("approved_sandbox_validation_run_id") != validation.get("run_id"):
        blockers.append("sandbox_validation_run_id_mismatch_or_stale")
    if approval.get("approved_sandbox_validation_summary_hash") != _mapping_hash(validation):
        blockers.append("sandbox_validation_summary_hash_mismatch_or_stale")
    if int(approval.get("approved_output_schema_version") or 0) != int(schema.get("schema_version") or 0):
        blockers.append("output_schema_version_mismatch")
    if approval.get("approved_output_schema_id") != schema.get("schema_id"):
        blockers.append("output_schema_id_mismatch")
    if approval.get("approved_output_schema_hash") != schema.get("schema_hash"):
        blockers.append("output_schema_hash_mismatch")
    if approval.get("approved_clamp_range_hash") != _mapping_hash(dict(ctx.wider_sandbox_summary.get("range_used") or {})):
        blockers.append("clamp_range_hash_mismatch")
    if approval.get("approved_sandbox_validation_status") not in {"passed", "passed_with_issues"}:
        blockers.append("sandbox_validation_status_not_acceptable")
    blockers.extend(_unaccepted_validation_issues(approval, validation, branch))
    blockers.extend(_validate_approval_metadata(approval))
    if approval.get("production_writer_enabled") is not False:
        blockers.append("approval_attempts_to_enable_writer")
    for flag_name in (
        "production_labels_written",
        "production_outputs_written",
        "canonical_production_state_outputs_written",
        "production_promotion_performed",
    ):
        if approval.get(flag_name) is not False:
            blockers.append(f"{flag_name}_must_remain_false")
    return list(dict.fromkeys(blockers))


def _validate_approval_metadata(approval: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _iso_timestamp(str(approval.get("approval_timestamp") or "")):
        blockers.append("approval_timestamp_invalid")
    if not str(approval.get("approval_operator") or "").strip():
        blockers.append("approval_operator_missing")
    if not str(approval.get("approval_source") or "").strip():
        blockers.append("approval_source_missing")
    root = dict(approval.get("canonical_output_root_confirmation") or {})
    if root.get("canonical_output_root_confirmed") is not True:
        blockers.append("canonical_output_root_not_confirmed")
    if not str(root.get("canonical_output_root_key") or "").strip():
        blockers.append("canonical_output_root_key_missing")
    if not str(root.get("canonical_output_root_reference") or "").strip():
        blockers.append("canonical_output_root_reference_missing")
    if root.get("canonical_root_touched") is not False:
        blockers.append("canonical_root_touched_during_approval")
    if root.get("canonical_root_write_test_performed") is not False:
        blockers.append("canonical_root_write_test_performed")
    rollback = dict(approval.get("rollback_safety") or {})
    if rollback.get("rollback_supported") is not True:
        blockers.append("rollback_not_supported")
    if rollback.get("manual_rollback_confirmation_required") is not True:
        blockers.append("manual_rollback_confirmation_not_required")
    if not str(rollback.get("rollback_plan_id") or "").strip():
        blockers.append("rollback_plan_id_missing")
    return blockers


def _unaccepted_validation_issues(
    approval: Mapping[str, Any],
    validation: Mapping[str, Any],
    branch: str,
) -> list[str]:
    accepted = set(str(item) for item in approval.get("accepted_validation_issues") or ())
    required: list[str] = []
    for severity in ("blocker", "high", "medium", "low"):
        for issue in validation.get("issues", {}).get(severity, ()) or ():
            text = str(issue)
            if text.startswith(f"{branch}:") or ":" not in text:
                required.append(text)
    return [f"unaccepted_validation_issue:{issue}" for issue in required if issue not in accepted]


def _default_rollback_safety(branch: str) -> dict[str, Any]:
    return {
        "rollback_supported": True,
        "rollback_plan_id": f"{branch}_manual_revert_to_previous_canonical_state",
        "manual_rollback_confirmation_required": True,
        "previous_active_artifact_reference_required_before_write": True,
        "rollback_test_performed": False,
        "rollback_execution_performed": False,
    }


def _branch_output(ctx: RegimeProductionPromotionGateContext, branch: str) -> dict[str, Any]:
    output = dict(ctx.wider_sandbox_summary.get("branch_outputs", {}).get(branch) or {})
    if not output:
        raise ValueError(f"Missing wider sandbox branch output for {branch}")
    return output


def _branch_validation(ctx: RegimeProductionPromotionGateContext, branch: str) -> dict[str, Any]:
    validation = dict(ctx.sandbox_validation_summary.get("branch_validations", {}).get(branch) or {})
    if not validation:
        validation = _compact_branch_validation(ctx.sandbox_validation_summary, branch)
    if not validation:
        raise ValueError(f"Missing sandbox validation branch output for {branch}")
    return validation


def _compact_branch_validation(validation_summary: Mapping[str, Any], branch: str) -> dict[str, Any]:
    branch_status = dict(validation_summary.get("validation_status_per_branch") or {})
    if branch not in branch_status:
        return {}
    row_counts = dict(validation_summary.get("row_count_by_branch") or {})
    logical_counts = dict(
        validation_summary.get("logical_grain_count_by_branch")
        or validation_summary.get("logical_partition_count_by_branch")
        or {}
    )
    mask_counts = dict(validation_summary.get("mask_or_unavailable_row_count_by_branch") or {})
    return {
        "row_count": row_counts.get(branch),
        "logical_partition_count": logical_counts.get(branch),
        "mask_or_unavailable_row_count": mask_counts.get(branch),
        "validation_status": branch_status.get(branch),
    }


def _context(value: RegimeProductionPromotionGateContext | Mapping[str, Any]) -> RegimeProductionPromotionGateContext:
    if isinstance(value, RegimeProductionPromotionGateContext):
        return value
    payload = dict(value)
    return RegimeProductionPromotionGateContext(
        sandbox_validation_summary=dict(payload.get("sandbox_validation_summary") or {}),
        wider_sandbox_summary=dict(payload.get("wider_sandbox_summary") or {}),
    )


def _approval_payload(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return to_jsonable(dict(value))
    return _load_json(Path(value))


def _load_json(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production promotion gate expected a JSON object")
    return payload


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
        raise ValueError(f"Regime Production promotion gate {field_name} must be non-empty")
    return text


__all__ = [
    "APPROVAL_STATUS_APPROVED_FOR_DRY_WRITE_PLANNING",
    "APPROVAL_STATUS_TEMPLATE_NOT_EXECUTED",
    "DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH",
    "GATE_STATUS_ACCEPTED_FOR_DRY_WRITE_PLANNING",
    "GATE_STATUS_BLOCKED",
    "REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PROMOTION_GATE_RESULT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PROMOTION_GATE_SCHEMA_VERSION",
    "REGIME_PRODUCTION_PROMOTION_GATE_SUMMARY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND",
    "RegimeProductionPromotionGateContext",
    "build_regime_production_branch_approval_artifact",
    "build_regime_production_promotion_gate_design_summary",
    "build_regime_production_unified_approval_artifact",
    "evaluate_regime_production_branch_promotion_gate",
    "evaluate_regime_production_unified_promotion_gate",
    "write_regime_production_promotion_gate_summary",
]
