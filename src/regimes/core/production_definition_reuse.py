from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_definition_planner import REGIME_PRODUCTION_DEFINITION_RECORD_ARTIFACT_KIND
from src.regimes.core.production_planner import (
    BRANCH_TARGET_KEY_FIELDS,
    MODEL_STATE_REQUIRED_FIELDS,
    REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_ACTIVE,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_BLOCKED,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_FAILED_PROFILE_HEALTH,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_INVALID_PROFILE,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_STALE,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_DEFINITION_REUSE_SCHEMA_VERSION = 1
REGIME_PRODUCTION_DEFINITION_REUSE_VALIDATION_ARTIFACT_KIND = "regime_production_definition_reuse_validation"

REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REUSE = "reuse_definition"
REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REJECT = "reject_definition"

MISSING_HEALTH_METADATA_POLICY_REJECT = "reject"
MISSING_HEALTH_METADATA_POLICY_WARN = "warn"

_ALLOWED_ARTIFACT_KINDS = {
    REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND,
    REGIME_PRODUCTION_DEFINITION_RECORD_ARTIFACT_KIND,
}
_REUSABLE_MODEL_STATE_STATUSES = {
    REGIME_PRODUCTION_MODEL_STATE_STATUS_ACTIVE,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
}
_REJECTED_MODEL_STATE_STATUSES = {
    REGIME_PRODUCTION_MODEL_STATE_STATUS_BLOCKED,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_FAILED_PROFILE_HEALTH,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_INVALID_PROFILE,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_STALE,
}
_SELF_HASH_FIELDS = {
    "artifact_hash",
    "computed_definition_artifact_hash",
    "definition_artifact_hash",
}
_SAFETY_FLAG_FIELDS = (
    "production_writer_enabled",
    "definition_file_written",
    "model_state_record_written",
    "production_labels_written",
    "production_outputs_written",
    "canonical_production_state_outputs_written",
)


@dataclass(frozen=True)
class RegimeProductionDefinitionReuseContext:
    branch: str
    target_key: Mapping[str, Any]
    profile_id: str
    profile_version: str
    selected_profile_artifact_hash: str
    source_tail_ts: Any
    refit_window_start: Any
    refit_window_end: Any
    reuse_as_of_ts: Any | None = None
    expected_definition_artifact_hash: str | None = None
    expected_refit_cadence_id: str | None = None

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        target_key = to_jsonable(dict(self.target_key))
        missing = [field_name for field_name in BRANCH_TARGET_KEY_FIELDS[branch] if target_key.get(field_name) in (None, "")]
        if missing:
            raise ValueError(f"Regime Production definition reuse context target_key missing required fields: {missing!r}")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "profile_id", _text(self.profile_id, field_name="profile_id"))
        object.__setattr__(self, "profile_version", _text(self.profile_version, field_name="profile_version"))
        object.__setattr__(
            self,
            "selected_profile_artifact_hash",
            _text(self.selected_profile_artifact_hash, field_name="selected_profile_artifact_hash"),
        )


@dataclass(frozen=True)
class RegimeProductionDefinitionReuseValidation:
    decision: str
    branch: str | None
    definition_id: str | None
    definition_version: str | None
    computed_definition_artifact_hash: str
    expected_definition_artifact_hash: str | None
    embedded_definition_artifact_hash: str | None
    reason_codes: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        reasons = tuple(dict.fromkeys(str(item) for item in self.reason_codes if str(item or "").strip()))
        warnings = tuple(dict.fromkeys(str(item) for item in self.warnings if str(item or "").strip()))
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "warnings", warnings)

    @property
    def reuse_allowed(self) -> bool:
        return self.decision == REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REUSE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_DEFINITION_REUSE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DEFINITION_REUSE_VALIDATION_ARTIFACT_KIND,
            "decision": self.decision,
            "reuse_allowed": self.reuse_allowed,
            "branch": self.branch,
            "definition_id": self.definition_id,
            "definition_version": self.definition_version,
            "computed_definition_artifact_hash": self.computed_definition_artifact_hash,
            "expected_definition_artifact_hash": self.expected_definition_artifact_hash,
            "embedded_definition_artifact_hash": self.embedded_definition_artifact_hash,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "definition_artifact_reused": self.reuse_allowed,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def stable_regime_production_definition_artifact_hash(payload: Mapping[str, Any]) -> str:
    normalized = _hash_payload(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_regime_production_definition_reuse(
    definition_artifact: Mapping[str, Any],
    *,
    expected: RegimeProductionDefinitionReuseContext | Mapping[str, Any],
    missing_health_metadata_policy: str = MISSING_HEALTH_METADATA_POLICY_REJECT,
) -> RegimeProductionDefinitionReuseValidation:
    payload = to_jsonable(dict(definition_artifact))
    context = (
        expected
        if isinstance(expected, RegimeProductionDefinitionReuseContext)
        else RegimeProductionDefinitionReuseContext(**dict(expected))
    )
    health_policy = _text(missing_health_metadata_policy, field_name="missing_health_metadata_policy").lower()
    if health_policy not in {MISSING_HEALTH_METADATA_POLICY_REJECT, MISSING_HEALTH_METADATA_POLICY_WARN}:
        raise ValueError(f"Unsupported Regime Production missing health metadata policy: {health_policy!r}")

    reason_codes: list[str] = []
    warnings: list[str] = []
    branch = None
    try:
        branch = _branch_name(payload.get("branch"))
    except ValueError:
        reason_codes.append("branch_invalid_or_missing")

    artifact_kind = str(payload.get("artifact_kind") or "")
    if artifact_kind not in _ALLOWED_ARTIFACT_KINDS:
        reason_codes.append("definition_artifact_kind_invalid")
    if int(payload.get("schema_version") or -1) != REGIME_PRODUCTION_DEFINITION_REUSE_SCHEMA_VERSION:
        reason_codes.append("definition_schema_version_invalid")

    required = _required_fields_for_artifact_kind(artifact_kind)
    missing_required = [field_name for field_name in required if field_name not in payload or payload.get(field_name) in (None, "")]
    if missing_required:
        reason_codes.extend(f"definition_required_field_missing:{field_name}" for field_name in missing_required)

    if branch is not None and branch != context.branch:
        reason_codes.append("definition_branch_mismatch")
    target_key = dict(payload.get("target_key") or {})
    if branch is not None:
        missing_target = [field_name for field_name in BRANCH_TARGET_KEY_FIELDS[branch] if target_key.get(field_name) in (None, "")]
        if missing_target:
            reason_codes.extend(f"definition_target_key_missing:{field_name}" for field_name in missing_target)
    if to_jsonable(target_key) != dict(context.target_key):
        reason_codes.append("definition_target_key_mismatch")
    grain_key = payload.get("grain_key")
    if grain_key not in (None, "") and to_jsonable(dict(grain_key)) != target_key:
        reason_codes.append("definition_grain_key_target_key_mismatch")

    _compare_text(payload, "profile_id", context.profile_id, reason_codes)
    _compare_text(payload, "profile_version", context.profile_version, reason_codes)
    _compare_text(payload, "profile_artifact_hash", context.selected_profile_artifact_hash, reason_codes, reason="selected_profile_artifact_hash_mismatch")
    if context.expected_refit_cadence_id not in (None, ""):
        _compare_text(payload, "refit_cadence_id", str(context.expected_refit_cadence_id), reason_codes)

    _compare_ts(payload, "source_tail_ts", context.source_tail_ts, reason_codes)
    _compare_ts(payload, "refit_window_start", context.refit_window_start, reason_codes)
    _compare_ts(payload, "refit_window_end", context.refit_window_end, reason_codes)
    _validate_order(payload, "refit_window_start", "refit_window_end", reason_codes, reason="refit_window_invalid")
    _validate_order(payload, "source_tail_ts", "definition_known_at_ts", reason_codes, reason="source_tail_after_definition_known_at")
    if context.reuse_as_of_ts not in (None, ""):
        try:
            if _to_orderable(context.reuse_as_of_ts, field_name="reuse_as_of_ts") > _to_orderable(
                payload.get("refit_window_end"),
                field_name="refit_window_end",
            ):
                reason_codes.append("refit_window_expired")
        except ValueError:
            reason_codes.append("reuse_as_of_ts_or_refit_window_end_not_orderable")

    status = str(payload.get("status") or "").strip()
    if status in _REJECTED_MODEL_STATE_STATUSES:
        reason_codes.append("definition_status_not_reusable")
    elif status and status not in _REUSABLE_MODEL_STATE_STATUSES:
        warnings.append("definition_status_unrecognized_for_reuse")

    health_metadata = payload.get("health_metadata")
    if not isinstance(health_metadata, Mapping) or not dict(health_metadata):
        if health_policy == MISSING_HEALTH_METADATA_POLICY_REJECT:
            reason_codes.append("definition_health_metadata_missing")
        else:
            warnings.append("definition_health_metadata_missing")
    else:
        health = dict(health_metadata)
        if health.get("definition_health_pass") is False or str(health.get("status") or "").strip().lower() in {
            "failed",
            "fail",
            "blocked",
            "unhealthy",
        }:
            reason_codes.append("definition_health_failed")

    computed_hash = stable_regime_production_definition_artifact_hash(payload)
    embedded_hash = _first_present(payload, ("definition_artifact_hash", "artifact_hash"))
    expected_hash = context.expected_definition_artifact_hash or embedded_hash
    if not expected_hash:
        reason_codes.append("definition_artifact_hash_missing")
    if embedded_hash and embedded_hash != computed_hash:
        reason_codes.append("definition_artifact_hash_mismatch")
    if context.expected_definition_artifact_hash and context.expected_definition_artifact_hash != computed_hash:
        reason_codes.append("expected_definition_artifact_hash_mismatch")

    for field_name in _SAFETY_FLAG_FIELDS:
        if bool(payload.get(field_name)):
            reason_codes.append(f"definition_safety_flag_enabled:{field_name}")

    decision = (
        REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REJECT
        if reason_codes
        else REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REUSE
    )
    return RegimeProductionDefinitionReuseValidation(
        decision=decision,
        branch=branch,
        definition_id=None if payload.get("definition_id") in (None, "") else str(payload.get("definition_id")),
        definition_version=None if payload.get("definition_version") in (None, "") else str(payload.get("definition_version")),
        computed_definition_artifact_hash=computed_hash,
        expected_definition_artifact_hash=expected_hash,
        embedded_definition_artifact_hash=embedded_hash,
        reason_codes=tuple(reason_codes),
        warnings=tuple(warnings),
    )


def _hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = to_jsonable(dict(payload))
    return {key: value for key, value in sorted(normalized.items()) if key not in _SELF_HASH_FIELDS}


def _required_fields_for_artifact_kind(artifact_kind: str) -> tuple[str, ...]:
    common = tuple(field_name for field_name in MODEL_STATE_REQUIRED_FIELDS if field_name != "grain_key")
    if artifact_kind == REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND:
        return MODEL_STATE_REQUIRED_FIELDS
    if artifact_kind == REGIME_PRODUCTION_DEFINITION_RECORD_ARTIFACT_KIND:
        return ("definition_id", "definition_version", *common)
    return common


def _branch_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "asset": REGIME_BRANCH_ASSET_STATE,
        "asset_state_production": REGIME_BRANCH_ASSET_STATE,
        "market": REGIME_BRANCH_MARKET_STATE,
        "market_state_production": REGIME_BRANCH_MARKET_STATE,
        "cross_asset": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross_asset_state_production": REGIME_BRANCH_CROSS_ASSET_STATE,
    }
    branch = aliases.get(text, text)
    if branch not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {value!r}")
    return branch


def _compare_text(payload: Mapping[str, Any], field_name: str, expected: str, reason_codes: list[str], *, reason: str | None = None) -> None:
    if str(payload.get(field_name) or "").strip() != str(expected or "").strip():
        reason_codes.append(reason or f"{field_name}_mismatch")


def _compare_ts(payload: Mapping[str, Any], field_name: str, expected: Any, reason_codes: list[str]) -> None:
    try:
        actual_value = _to_orderable(payload.get(field_name), field_name=field_name)
        expected_value = _to_orderable(expected, field_name=f"expected_{field_name}")
    except ValueError:
        reason_codes.append(f"{field_name}_not_orderable")
        return
    if actual_value != expected_value:
        reason_codes.append(f"{field_name}_mismatch")


def _validate_order(payload: Mapping[str, Any], start_field: str, end_field: str, reason_codes: list[str], *, reason: str) -> None:
    try:
        if _to_orderable(payload.get(start_field), field_name=start_field) > _to_orderable(payload.get(end_field), field_name=end_field):
            reason_codes.append(reason)
    except ValueError:
        reason_codes.append(reason)


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime Production definition reuse {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    from datetime import datetime

    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime Production definition reuse {field_name} must be numeric or ISO datetime") from exc


def _first_present(payload: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field_name in fields:
        value = payload.get(field_name)
        if value not in (None, ""):
            return str(value)
    return None


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production definition reuse {field_name} must be non-empty")
    return text


__all__ = [
    "MISSING_HEALTH_METADATA_POLICY_REJECT",
    "MISSING_HEALTH_METADATA_POLICY_WARN",
    "REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REJECT",
    "REGIME_PRODUCTION_DEFINITION_REUSE_DECISION_REUSE",
    "REGIME_PRODUCTION_DEFINITION_REUSE_SCHEMA_VERSION",
    "REGIME_PRODUCTION_DEFINITION_REUSE_VALIDATION_ARTIFACT_KIND",
    "RegimeProductionDefinitionReuseContext",
    "RegimeProductionDefinitionReuseValidation",
    "stable_regime_production_definition_artifact_hash",
    "validate_regime_production_definition_reuse",
]
