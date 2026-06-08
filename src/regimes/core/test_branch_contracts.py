from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable


PRODUCTION_GATE_FIELDS: tuple[str, ...] = (
    "production_approved",
    "production_writer_enabled",
    "production_labels_written",
    "production_outputs_written",
    "canonical_production_state_outputs_written",
    "requires_human_approval_before_production",
)

MASK_LOW_VARIANCE_NEAR_FLAT = "low_variance_near_flat"
MASK_INSUFFICIENT_HISTORY = "insufficient_history"
MASK_INSUFFICIENT_COVERAGE = "insufficient_coverage"
MASK_NON_CLUSTERABLE = "non_clusterable"
MASK_FAILED_HEALTH_GATE = "failed_health_gate"
MASK_MISSING_REQUIRED_FEATURES = "missing_required_features"
MASK_STALE_OR_INVALID_SOURCE = "stale_or_invalid_source"
MASK_UNAPPROVED_MANIFEST = "unapproved_manifest"
MASK_PRODUCTION_GATE_CLOSED = "production_gate_closed"

MASK_REASON_CODES: tuple[str, ...] = (
    MASK_LOW_VARIANCE_NEAR_FLAT,
    MASK_INSUFFICIENT_HISTORY,
    MASK_INSUFFICIENT_COVERAGE,
    MASK_NON_CLUSTERABLE,
    MASK_FAILED_HEALTH_GATE,
    MASK_MISSING_REQUIRED_FEATURES,
    MASK_STALE_OR_INVALID_SOURCE,
    MASK_UNAPPROVED_MANIFEST,
    MASK_PRODUCTION_GATE_CLOSED,
)


def nonproduction_gate_flags() -> dict[str, bool]:
    return {
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
    }


def test_branch_consumable_status_fields(
    *,
    validation_status: str = "passed",
    active_handoff_artifact: bool = True,
) -> dict[str, Any]:
    passed = str(validation_status) == "passed" and bool(active_handoff_artifact)
    return {
        "test_branch_validation_status": str(validation_status),
        "active_handoff_artifact": bool(active_handoff_artifact),
        "profile_selection_status": "approved_by_test_branch" if passed else "not_approved_by_test_branch",
        "production_consumable": bool(passed),
        "requires_human_approval_before_production": False,
        "requires_human_approval_before_production_consumption": False,
        "requires_human_approval_before_production_writes": True,
    }


def apply_nonproduction_gate_flags(payload: Mapping[str, Any], *, include_canonical: bool = True) -> dict[str, Any]:
    flags = nonproduction_gate_flags()
    if not include_canonical:
        flags.pop("canonical_production_state_outputs_written", None)
    return {**dict(payload), **flags}


def fail_closed_reason_flags(reason_code: str = MASK_PRODUCTION_GATE_CLOSED) -> dict[str, Any]:
    return {
        **nonproduction_gate_flags(),
        "production_write_allowed": False,
        "canonical_outputs_written": False,
        "production_promotion_performed": False,
        "reason_code": str(reason_code),
    }


def validate_nonproduction_gate_flags(
    payload: Mapping[str, Any],
    *,
    require_canonical: bool = False,
    expected_requires_human_approval: bool | None = True,
    prefix: str = "",
) -> tuple[str, ...]:
    reasons: list[str] = []
    fields = PRODUCTION_GATE_FIELDS if require_canonical else tuple(
        field for field in PRODUCTION_GATE_FIELDS if field != "canonical_production_state_outputs_written"
    )
    label = f"{prefix}_" if prefix else ""
    for field_name in fields:
        if field_name not in payload:
            reasons.append(f"{label}{field_name}_missing")
            continue
        if field_name == "requires_human_approval_before_production":
            if expected_requires_human_approval is None:
                continue
            expected = bool(expected_requires_human_approval)
        else:
            expected = False
        if payload.get(field_name) is not expected:
            reasons.append(f"{label}{field_name}_invalid")
    return tuple(reasons)


def normalize_mask_reason_code(reason_code: object) -> str:
    text = str(reason_code or "").strip().lower()
    aliases = {
        "required_features_missing": MASK_MISSING_REQUIRED_FEATURES,
        "missing_features": MASK_MISSING_REQUIRED_FEATURES,
        "insufficient_rows": MASK_INSUFFICIENT_HISTORY,
        "insufficient_finite_share": MASK_INSUFFICIENT_COVERAGE,
        "clusterability_filtered": MASK_NON_CLUSTERABLE,
        "source_invalid": MASK_STALE_OR_INVALID_SOURCE,
    }
    return aliases.get(text, text or MASK_NON_CLUSTERABLE)


def unavailable_record(
    *,
    profile_id: str,
    reason_code: str,
    reason: str,
    schema_version: int,
    run_id: str | None = None,
    selection_scope: str | None = None,
    profile_grain: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": int(schema_version),
        "profile_id": str(profile_id),
        "availability_status": "masked_unavailable",
        "mask_reason_code": normalize_mask_reason_code(reason_code),
        "reason": str(reason),
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    if selection_scope is not None:
        payload["selection_scope"] = str(selection_scope)
    if profile_grain is not None:
        payload["profile_grain"] = str(profile_grain)
    payload.update(dict(extra or {}))
    return apply_nonproduction_gate_flags(payload, include_canonical=False)


def score_contract_fields(
    *,
    semantic_candidate_score: object,
    runtime_penalty: object = 0.0,
    compatibility_alias: bool = True,
) -> dict[str, Any]:
    semantic = _finite_float(semantic_candidate_score, default=0.0)
    penalty = max(0.0, _finite_float(runtime_penalty, default=0.0))
    out = {
        "semantic_candidate_score": semantic,
        "runtime_penalty": penalty,
        "runtime_adjusted_score": semantic - penalty,
        "score_policy": "candidate_score_equals_semantic_candidate_score; runtime_adjusted_score_only_tiebreak",
        "runtime_leakage_into_semantic_score": False,
    }
    if compatibility_alias:
        out["candidate_score"] = semantic
    return out


def scoring_contract_manifest_section() -> dict[str, str]:
    return {
        "semantic_candidate_score": "primary profile-selection score",
        "candidate_score": "alias of semantic_candidate_score for compatibility",
        "runtime_penalty": "recorded separately",
        "runtime_adjusted_score": "tie-break only; not the semantic score",
        "runtime_leakage_into_semantic_score": "forbidden",
    }


def validate_score_contract(row: Mapping[str, Any], *, require_alias: bool = True) -> tuple[str, ...]:
    reasons: list[str] = []
    semantic = _maybe_float(row.get("semantic_candidate_score"))
    adjusted = _maybe_float(row.get("runtime_adjusted_score"))
    penalty = _maybe_float(row.get("runtime_penalty"))
    alias = _maybe_float(row.get("candidate_score"))
    if semantic is None:
        reasons.append("semantic_candidate_score_missing_or_invalid")
    if adjusted is None:
        reasons.append("runtime_adjusted_score_missing_or_invalid")
    if penalty is None:
        reasons.append("runtime_penalty_missing_or_invalid")
    if require_alias and alias is None:
        reasons.append("candidate_score_missing_or_invalid")
    if semantic is not None and adjusted is not None and adjusted > semantic + 1e-12:
        reasons.append("runtime_adjusted_score_exceeds_semantic_candidate_score")
    if semantic is not None and alias is not None and abs(alias - semantic) > 1e-12:
        reasons.append("candidate_score_not_semantic_alias")
    if str(row.get("score_policy") or "") and "runtime_adjusted_score_only_tiebreak" not in str(row.get("score_policy")):
        reasons.append("score_policy_does_not_document_runtime_tiebreak")
    return tuple(reasons)


@dataclass(frozen=True)
class SelectedProfileManifestValidation:
    status: str
    reason_codes: tuple[str, ...] = ()
    selected_count: int = 0
    masked_count: int = 0
    expected_cell_count: int = 0
    covered_cell_count: int = 0
    missing_cells: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "regime_selected_profile_manifest_validation",
            "status": self.status,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "selected_count": int(self.selected_count),
            "masked_count": int(self.masked_count),
            "expected_cell_count": int(self.expected_cell_count),
            "covered_cell_count": int(self.covered_cell_count),
            "missing_cells": [dict(item) for item in self.missing_cells],
            "metadata": to_jsonable(dict(self.metadata)),
            "production_write_allowed": False,
        }


def validate_selected_profile_manifest(
    manifest: Mapping[str, Any],
    *,
    active_filename: str,
    expected_cells: Sequence[Mapping[str, Any]] = (),
    selected_records_key: str = "selected_profiles",
    masked_records_key: str = "masked_or_skipped_cells",
    selected_cell_key_fields: Sequence[str],
    masked_cell_key_fields: Sequence[str] | None = None,
    single_active_field: str = "single_active_nonproduction_handoff_artifact",
    require_canonical_gate_field: bool = False,
    expected_requires_human_approval: bool | None = True,
) -> SelectedProfileManifestValidation:
    reasons: list[str] = []
    selected = [dict(item) for item in manifest.get(selected_records_key) or () if isinstance(item, Mapping)]
    masked = [dict(item) for item in manifest.get(masked_records_key) or () if isinstance(item, Mapping)]
    reasons.extend(
        validate_nonproduction_gate_flags(
            manifest,
            require_canonical=require_canonical_gate_field,
            expected_requires_human_approval=expected_requires_human_approval,
            prefix="manifest",
        )
    )
    if manifest.get(single_active_field) != active_filename:
        reasons.append("single_active_nonproduction_handoff_artifact_invalid")
    if bool(manifest.get("stale_sandbox_manifest_used", False)):
        reasons.append("stale_sandbox_manifest_marked_active")
    source_lineage = manifest.get("source_lineage")
    if isinstance(source_lineage, Mapping) and bool(source_lineage.get("stale_sandbox_manifest_used", False)):
        reasons.append("stale_sandbox_source_lineage_marked_active")
    for idx, profile in enumerate(selected):
        reasons.extend(
            validate_nonproduction_gate_flags(
                profile,
                require_canonical=False,
                expected_requires_human_approval=expected_requires_human_approval,
                prefix=f"selected_{idx}",
            )
        )
    for idx, mask in enumerate(masked):
        reasons.extend(
            validate_nonproduction_gate_flags(
                mask,
                require_canonical=False,
                expected_requires_human_approval=expected_requires_human_approval,
                prefix=f"masked_{idx}",
            )
        )

    masked_fields = tuple(masked_cell_key_fields or selected_cell_key_fields)
    expected_keys = {_cell_key(item, selected_cell_key_fields) for item in expected_cells}
    covered_keys = {
        _cell_key(item, selected_cell_key_fields)
        for item in selected
    } | {
        _cell_key(item, masked_fields)
        for item in masked
    }
    missing = tuple(dict(item) for item in expected_cells if _cell_key(item, selected_cell_key_fields) not in covered_keys)
    if missing:
        reasons.append("expected_cells_missing")
    return SelectedProfileManifestValidation(
        status="blocked" if reasons else "passed",
        reason_codes=tuple(dict.fromkeys(reasons)),
        selected_count=len(selected),
        masked_count=len(masked),
        expected_cell_count=len(expected_keys),
        covered_cell_count=len(covered_keys),
        missing_cells=missing,
        metadata={"active_filename": active_filename, "selected_records_key": selected_records_key, "masked_records_key": masked_records_key},
    )


def _cell_key(payload: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(payload.get(field)) for field in fields)


def _maybe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _finite_float(value: object, *, default: float) -> float:
    out = _maybe_float(value)
    return float(default) if out is None else float(out)


__all__ = [
    "MASK_FAILED_HEALTH_GATE",
    "MASK_INSUFFICIENT_COVERAGE",
    "MASK_INSUFFICIENT_HISTORY",
    "MASK_LOW_VARIANCE_NEAR_FLAT",
    "MASK_MISSING_REQUIRED_FEATURES",
    "MASK_NON_CLUSTERABLE",
    "MASK_PRODUCTION_GATE_CLOSED",
    "MASK_REASON_CODES",
    "MASK_STALE_OR_INVALID_SOURCE",
    "MASK_UNAPPROVED_MANIFEST",
    "PRODUCTION_GATE_FIELDS",
    "SelectedProfileManifestValidation",
    "apply_nonproduction_gate_flags",
    "fail_closed_reason_flags",
    "nonproduction_gate_flags",
    "normalize_mask_reason_code",
    "score_contract_fields",
    "scoring_contract_manifest_section",
    "test_branch_consumable_status_fields",
    "unavailable_record",
    "validate_nonproduction_gate_flags",
    "validate_score_contract",
    "validate_selected_profile_manifest",
]
