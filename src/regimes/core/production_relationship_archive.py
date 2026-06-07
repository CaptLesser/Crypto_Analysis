from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.path_config import PathConfigError
from src.regimes.core.root_resolution import (
    SOURCE_KIND_RELATIONSHIP_DISCOVERY,
    resolve_regime_production_sidecar_input_path,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.production_consumer import REGIME_BRANCH_CROSS_ASSET_STATE
from src.regimes.core.production_relationship_freshness import (
    RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE,
    RELATIONSHIP_FRESHNESS_STATUS_MISSING,
    RELATIONSHIP_FRESHNESS_STATUS_STALE,
    RegimeProductionRelationshipFreshnessPolicy,
    relationship_freshness_policy_from_manifest,
)
from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason


REGIME_PRODUCTION_RELATIONSHIP_ARCHIVE_SCHEMA_VERSION = 1
REGIME_PRODUCTION_RELATIONSHIP_ARCHIVE_VALIDATION_ARTIFACT_KIND = (
    "regime_production_cross_asset_relationship_archive_validation"
)

RELATIONSHIP_ARCHIVE_ACTION_ALLOW = "allow"
RELATIONSHIP_ARCHIVE_ACTION_MASK_UNAVAILABLE = "mask_unavailable"

RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE = RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE
RELATIONSHIP_ARCHIVE_STATUS_STALE = RELATIONSHIP_FRESHNESS_STATUS_STALE
RELATIONSHIP_ARCHIVE_STATUS_MISSING = RELATIONSHIP_FRESHNESS_STATUS_MISSING

DEFAULT_RELATIONSHIP_INPUT_TYPE = "relationship_context_handoff"
DIAGNOSTIC_SIDECAR_ROLES: tuple[str, ...] = (
    "diagnostic",
    "diagnostic_sidecar",
    "not_model_facing",
    "sidecar",
    "sidecar_only",
)

_ARCHIVE_PATH_FIELDS: tuple[str, ...] = (
    "relationship_input_archive_path",
    "relationship_archive_path",
    "relationship_context_archive_path",
    "relationship_archive_root",
    "relationship_context_handoff_path",
)


@dataclass(frozen=True)
class CrossAssetRelationshipArchiveValidation:
    status: str
    action: str
    clamp_range: Mapping[str, Any]
    archive_root_resolution: Mapping[str, Any]
    policy: RegimeProductionRelationshipFreshnessPolicy
    expected_cell_count: int
    available_cell_count: int
    stale_cell_count: int
    missing_cell_count: int
    archive_record_count: int
    reason_codes: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    cell_checks: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        reasons = tuple(dict.fromkeys(str(item) for item in self.reason_codes if str(item or "").strip()))
        warnings = tuple(dict.fromkeys(str(item) for item in self.warnings if str(item or "").strip()))
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "clamp_range", to_jsonable(dict(self.clamp_range)))
        object.__setattr__(self, "archive_root_resolution", to_jsonable(dict(self.archive_root_resolution)))
        object.__setattr__(self, "cell_checks", tuple(to_jsonable(dict(item)) for item in self.cell_checks))

    @property
    def passed(self) -> bool:
        return self.status == RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_RELATIONSHIP_ARCHIVE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_RELATIONSHIP_ARCHIVE_VALIDATION_ARTIFACT_KIND,
            "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
            "status": self.status,
            "passed": self.passed,
            "action": self.action,
            "mask_reason": None if self.passed else _mask_reason(self.status, self.policy),
            "clamp_range": to_jsonable(dict(self.clamp_range)),
            "archive_root_resolution": to_jsonable(dict(self.archive_root_resolution)),
            "relationship_freshness_policy": self.policy.as_dict(),
            "expected_cell_count": int(self.expected_cell_count),
            "available_cell_count": int(self.available_cell_count),
            "stale_cell_count": int(self.stale_cell_count),
            "missing_cell_count": int(self.missing_cell_count),
            "archive_record_count": int(self.archive_record_count),
            "cell_checks": [to_jsonable(dict(item)) for item in self.cell_checks],
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "relationship_archive_partitioning_compatible_with_historical_walkthrough": self.status
            != RELATIONSHIP_ARCHIVE_STATUS_MISSING,
            "relationship_input_history_separate_from_selected_profile_artifact": True,
            "relationship_inputs_are_selected_profile_artifacts": False,
            "selected_profile_artifact": False,
            "peer_groups_role": "diagnostic_sidecar_unless_explicitly_model_facing",
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def validate_cross_asset_relationship_input_archive(
    *,
    manifest: Mapping[str, Any],
    archive_records: Sequence[Mapping[str, Any]],
    clamp_range: Mapping[str, Any],
    expected_cells: Sequence[Mapping[str, Any]],
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> CrossAssetRelationshipArchiveValidation:
    policy = relationship_freshness_policy_from_manifest(manifest)
    root_resolution, root_reasons = _archive_root_resolution(manifest, env=env, project_root=project_root)
    start_ts = _timestamp(_first_present(clamp_range, ("start_ts", "output_start_ts", "clamp_start_ts")), field_name="clamp_start_ts")
    end_ts = _timestamp(_first_present(clamp_range, ("end_ts", "output_end_ts", "clamp_end_ts")), field_name="clamp_end_ts")
    if start_ts > end_ts:
        raise ValueError("Cross-Asset relationship archive clamp start must be <= end")
    records = tuple(to_jsonable(dict(record)) for record in archive_records)
    checks: list[dict[str, Any]] = []
    reason_codes: list[str] = list(root_reasons)
    warnings: list[str] = []
    status_counts = {
        RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE: 0,
        RELATIONSHIP_ARCHIVE_STATUS_STALE: 0,
        RELATIONSHIP_ARCHIVE_STATUS_MISSING: 0,
    }

    for cell in expected_cells:
        check = _evaluate_cell_archive(
            cell,
            records=records,
            clamp_start_ts=start_ts,
            clamp_end_ts=end_ts,
            policy=policy,
        )
        checks.append(check)
        status_counts[check["status"]] += 1
        reason_codes.extend(check.get("reason_codes") or ())
        warnings.extend(check.get("warnings") or ())

    if not expected_cells:
        reason_codes.append("relationship_archive_expected_cells_missing")
        status_counts[RELATIONSHIP_ARCHIVE_STATUS_MISSING] += 1
    if root_reasons:
        status_counts[RELATIONSHIP_ARCHIVE_STATUS_MISSING] = max(status_counts[RELATIONSHIP_ARCHIVE_STATUS_MISSING], 1)

    if status_counts[RELATIONSHIP_ARCHIVE_STATUS_MISSING] > 0:
        status = RELATIONSHIP_ARCHIVE_STATUS_MISSING
    elif status_counts[RELATIONSHIP_ARCHIVE_STATUS_STALE] > 0:
        status = RELATIONSHIP_ARCHIVE_STATUS_STALE
    else:
        status = RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE
    action = RELATIONSHIP_ARCHIVE_ACTION_ALLOW if status == RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE else RELATIONSHIP_ARCHIVE_ACTION_MASK_UNAVAILABLE
    if action == RELATIONSHIP_ARCHIVE_ACTION_MASK_UNAVAILABLE:
        warnings.append("cross_asset_relationship_archive_masks_unavailable_cells")

    return CrossAssetRelationshipArchiveValidation(
        status=status,
        action=action,
        clamp_range={"start_ts": start_ts, "end_ts": end_ts},
        archive_root_resolution=root_resolution,
        policy=policy,
        expected_cell_count=len(expected_cells),
        available_cell_count=status_counts[RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE],
        stale_cell_count=status_counts[RELATIONSHIP_ARCHIVE_STATUS_STALE],
        missing_cell_count=status_counts[RELATIONSHIP_ARCHIVE_STATUS_MISSING],
        archive_record_count=len(records),
        reason_codes=tuple(reason_codes),
        warnings=tuple(warnings),
        cell_checks=tuple(checks),
    )


def _evaluate_cell_archive(
    cell: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    clamp_start_ts: int,
    clamp_end_ts: int,
    policy: RegimeProductionRelationshipFreshnessPolicy,
) -> dict[str, Any]:
    band = _band(cell.get("band"))
    relationship_input_type = str(cell.get("relationship_input_type") or DEFAULT_RELATIONSHIP_INPUT_TYPE)
    cadence_days = policy.cadence_days_for_band(band)
    stale_after_days = policy.stale_after_days_for_band(band)
    reason_codes: list[str] = []
    warnings: list[str] = []
    if cadence_days is None:
        reason_codes.append("relationship_archive_cadence_missing_for_band")
        cadence_days = 1
    if stale_after_days is None:
        reason_codes.append("relationship_archive_stale_policy_missing_for_band")
        stale_after_days = cadence_days

    matching = [
        dict(record)
        for record in records
        if _band(record.get("band")) == band
        and str(record.get("relationship_input_type") or DEFAULT_RELATIONSHIP_INPUT_TYPE) == relationship_input_type
    ]
    checkpoints = _checkpoints(clamp_start_ts, clamp_end_ts, cadence_days)
    missing_checkpoints = [ts for ts in checkpoints if not any(_record_covers(record, ts) for record in matching)]
    stale_records = []
    selected_records = [record for record in matching if any(_record_covers(record, ts) for ts in checkpoints)]
    for record in selected_records:
        record_reasons = _record_reason_codes(record, stale_after_days=stale_after_days)
        if record_reasons:
            stale_records.append({"relationship_snapshot_id": record.get("relationship_snapshot_id"), "reason_codes": record_reasons})
            reason_codes.extend(record_reasons)
        peer_role = str(record.get("peer_group_role") or record.get("peer_metadata_role") or "diagnostic_sidecar").strip().lower()
        if peer_role not in DIAGNOSTIC_SIDECAR_ROLES and record.get("model_facing_peer_groups_approved") is not True:
            reason_codes.append("peer_groups_model_facing_without_explicit_approval")
    if not matching:
        reason_codes.append("relationship_archive_records_missing")
    if missing_checkpoints:
        reason_codes.append("relationship_archive_clamp_coverage_gap")
    if stale_records:
        warnings.append("relationship_archive_stale_after_policy")

    peer_group_unsafe = "peer_groups_model_facing_without_explicit_approval" in reason_codes
    if peer_group_unsafe:
        warnings.append("relationship_archive_peer_groups_not_diagnostic_sidecar")

    if not matching or missing_checkpoints:
        status = RELATIONSHIP_ARCHIVE_STATUS_MISSING
    elif stale_records or peer_group_unsafe:
        status = RELATIONSHIP_ARCHIVE_STATUS_STALE
    else:
        status = RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE
    return {
        "target_key": {
            "asset_id": cell.get("asset_id"),
            "relationship_feature_family": cell.get("relationship_feature_family"),
            "band": band,
            "relationship_input_type": relationship_input_type,
        },
        "status": status,
        "passed": status == RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE,
        "action": RELATIONSHIP_ARCHIVE_ACTION_ALLOW
        if status == RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE
        else RELATIONSHIP_ARCHIVE_ACTION_MASK_UNAVAILABLE,
        "mask_reason": None if status == RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE else _mask_reason(status, policy),
        "snapshot_cadence_days": cadence_days,
        "stale_after_days": stale_after_days,
        "checkpoint_count": len(checkpoints),
        "missing_checkpoint_count": len(missing_checkpoints),
        "missing_checkpoints": missing_checkpoints[:10],
        "matching_archive_record_count": len(matching),
        "selected_archive_record_count": len(selected_records),
        "stale_archive_records": stale_records,
        "relationship_input_history_separate_from_selected_profile_artifact": True,
        "selected_profile_artifact": False,
        "peer_groups_role": "diagnostic_sidecar_unless_explicitly_model_facing",
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _archive_root_resolution(
    manifest: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    raw_path = _first_present(manifest, _ARCHIVE_PATH_FIELDS)
    if raw_path in (None, ""):
        return (
            {
                "status": RELATIONSHIP_ARCHIVE_STATUS_MISSING,
                "configured_root_policy": "missing",
                "selected_profile_artifact": False,
            },
            ("relationship_archive_path_not_declared",),
        )
    try:
        resolved = resolve_regime_production_sidecar_input_path(
            SOURCE_KIND_RELATIONSHIP_DISCOVERY,
            raw_path,
            field_name="relationship_archive_path",
            manifest=manifest,
            env=env,
            project_root=project_root,
        )
    except PathConfigError:
        return (
            {
                "status": RELATIONSHIP_ARCHIVE_STATUS_MISSING,
                "path": str(raw_path),
                "configured_root_policy": "configured_root_missing_or_mismatch",
                "selected_profile_artifact": False,
            },
            ("relationship_archive_root_not_configured",),
        )
    payload = resolved.as_dict()
    payload.update(
        {
            "status": RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE,
            "selected_profile_artifact": False,
            "relationship_inputs_are_selected_profile_artifacts": False,
        }
    )
    return payload, ()


def _record_reason_codes(record: Mapping[str, Any], *, stale_after_days: int) -> tuple[str, ...]:
    reasons: list[str] = []
    source_tail, source_reason = _optional_ts(record.get("source_tail_ts"), field_name="source_tail_ts")
    known_at, known_reason = _optional_ts(record.get("known_at_ts"), field_name="known_at_ts")
    valid_from, from_reason = _optional_ts(record.get("snapshot_valid_from_ts"), field_name="snapshot_valid_from_ts")
    valid_until, until_reason = _optional_ts(record.get("snapshot_valid_until_ts"), field_name="snapshot_valid_until_ts")
    for reason in (source_reason, known_reason, from_reason, until_reason):
        if reason:
            reasons.append(reason)
    if source_tail is not None and known_at is not None:
        if source_tail > known_at:
            reasons.append("relationship_archive_source_tail_after_known_at")
        if known_at - source_tail > int(stale_after_days) * 86_400:
            reasons.append("relationship_archive_stale_after_policy")
    if valid_from is not None and valid_until is not None and valid_from > valid_until:
        reasons.append("relationship_archive_valid_window_invalid")
    return tuple(dict.fromkeys(reasons))


def _record_covers(record: Mapping[str, Any], timestamp: int) -> bool:
    try:
        valid_from = _timestamp(record.get("snapshot_valid_from_ts"), field_name="snapshot_valid_from_ts")
        valid_until = _timestamp(record.get("snapshot_valid_until_ts"), field_name="snapshot_valid_until_ts")
    except ValueError:
        return False
    return valid_from <= int(timestamp) <= valid_until


def _checkpoints(start_ts: int, end_ts: int, cadence_days: int) -> list[int]:
    step = max(1, int(cadence_days)) * 86_400
    out = [int(start_ts)]
    current = int(start_ts)
    while current + step < int(end_ts):
        current += step
        out.append(current)
    if out[-1] != int(end_ts):
        out.append(int(end_ts))
    return out


def _mask_reason(status: str, policy: RegimeProductionRelationshipFreshnessPolicy) -> str:
    if status == RELATIONSHIP_ARCHIVE_STATUS_STALE:
        return policy.stale_mask_reason
    return policy.missing_mask_reason or CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT


def _first_present(payload: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field_name in fields:
        value = payload.get(field_name)
        if value not in (None, ""):
            return value
    return None


def _timestamp(value: Any, *, field_name: str) -> int:
    parsed, reason = _optional_ts(value, field_name=field_name)
    if reason or parsed is None:
        raise ValueError(f"Cross-Asset relationship archive {field_name} is invalid or missing")
    return int(parsed)


def _optional_ts(value: Any, *, field_name: str) -> tuple[int | None, str | None]:
    if value in (None, ""):
        return None, f"{field_name}_missing"
    if isinstance(value, bool):
        return None, f"{field_name}_invalid"
    try:
        return int(float(value)), None
    except Exception:
        pass
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp()), None
    except Exception:
        return None, f"{field_name}_invalid"


def _band(value: object) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


__all__ = [
    "DEFAULT_RELATIONSHIP_INPUT_TYPE",
    "REGIME_PRODUCTION_RELATIONSHIP_ARCHIVE_SCHEMA_VERSION",
    "REGIME_PRODUCTION_RELATIONSHIP_ARCHIVE_VALIDATION_ARTIFACT_KIND",
    "RELATIONSHIP_ARCHIVE_ACTION_ALLOW",
    "RELATIONSHIP_ARCHIVE_ACTION_MASK_UNAVAILABLE",
    "RELATIONSHIP_ARCHIVE_STATUS_AVAILABLE",
    "RELATIONSHIP_ARCHIVE_STATUS_MISSING",
    "RELATIONSHIP_ARCHIVE_STATUS_STALE",
    "CrossAssetRelationshipArchiveValidation",
    "validate_cross_asset_relationship_input_archive",
]
