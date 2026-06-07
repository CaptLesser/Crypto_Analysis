from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import REGIME_BRANCH_CROSS_ASSET_STATE
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason
from src.regimes.cross_asset_state.relationship_context import (
    DEFAULT_RELATIONSHIP_CONTEXT_CADENCE_POLICY_ID,
    DEFAULT_SNAPSHOT_CADENCE_DAYS,
    DEFAULT_STALE_AFTER_DAYS,
    DEFAULT_STALE_SNAPSHOT_POLICY_ID,
)


REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SCHEMA_VERSION = 1
REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_POLICY_ARTIFACT_KIND = "regime_production_relationship_input_freshness_policy"
REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_CHECK_ARTIFACT_KIND = "regime_production_relationship_input_freshness_check"
REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SUMMARY_ARTIFACT_KIND = "regime_production_relationship_input_freshness_summary"

RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE = "available"
RELATIONSHIP_FRESHNESS_STATUS_STALE = "stale"
RELATIONSHIP_FRESHNESS_STATUS_MISSING = "missing"
RELATIONSHIP_FRESHNESS_STATUSES: tuple[str, ...] = (
    RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE,
    RELATIONSHIP_FRESHNESS_STATUS_STALE,
    RELATIONSHIP_FRESHNESS_STATUS_MISSING,
)

RELATIONSHIP_INPUT_ROOT_NAMES: tuple[str, ...] = (
    "relationship_context_handoff",
    "relationship_eligibility_manifest",
)


@dataclass(frozen=True)
class RegimeProductionRelationshipFreshnessPolicy:
    policy_id: str = DEFAULT_RELATIONSHIP_CONTEXT_CADENCE_POLICY_ID
    snapshot_cadence_days_by_band: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_SNAPSHOT_CADENCE_DAYS))
    stale_after_days_by_band: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_STALE_AFTER_DAYS))
    stale_snapshot_policy_id: str = DEFAULT_STALE_SNAPSHOT_POLICY_ID
    stale_action: str = "mask_unavailable"
    stale_mask_reason: str = CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT
    missing_mask_reason: str = CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT
    source_contract: str = "src.regimes.cross_asset_state.relationship_context.CrossAssetRelationshipContextHandoff"

    def __post_init__(self) -> None:
        cadence = _positive_int_mapping(self.snapshot_cadence_days_by_band, field_name="snapshot_cadence_days_by_band")
        stale_after = _positive_int_mapping(self.stale_after_days_by_band, field_name="stale_after_days_by_band")
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "snapshot_cadence_days_by_band", cadence)
        object.__setattr__(self, "stale_after_days_by_band", stale_after)
        object.__setattr__(self, "stale_snapshot_policy_id", _text(self.stale_snapshot_policy_id, field_name="stale_snapshot_policy_id"))
        object.__setattr__(self, "stale_action", _text(self.stale_action, field_name="stale_action"))
        object.__setattr__(self, "stale_mask_reason", _text(self.stale_mask_reason, field_name="stale_mask_reason"))
        object.__setattr__(self, "missing_mask_reason", _text(self.missing_mask_reason, field_name="missing_mask_reason"))
        object.__setattr__(self, "source_contract", _text(self.source_contract, field_name="source_contract"))

    def cadence_days_for_band(self, band: object) -> int | None:
        return self.snapshot_cadence_days_by_band.get(_band(band))

    def stale_after_days_for_band(self, band: object) -> int | None:
        return self.stale_after_days_by_band.get(_band(band))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_POLICY_ARTIFACT_KIND,
            "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
            "policy_id": self.policy_id,
            "source_contract": self.source_contract,
            "snapshot_cadence_days_by_band": dict(self.snapshot_cadence_days_by_band),
            "stale_after_days_by_band": dict(self.stale_after_days_by_band),
            "stale_snapshot_policy_id": self.stale_snapshot_policy_id,
            "stale_action": self.stale_action,
            "stale_mask_reason": self.stale_mask_reason,
            "missing_mask_reason": self.missing_mask_reason,
            "relationship_inputs_are_selected_profile_artifacts": False,
            "relationship_discovery_allowed": False,
            "broad_pairwise_allowed": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionRelationshipFreshnessCheck:
    target_key: Mapping[str, Any]
    status: str
    policy: RegimeProductionRelationshipFreshnessPolicy
    relationship_input_tail_ts: Any = None
    relationship_known_at_ts: Any = None
    source_tail_ts: Any = None
    known_at_ts: Any = None
    snapshot_cadence_days: int | None = None
    stale_after_days: int | None = None
    reason_codes: Sequence[str] = ()
    relationship_input_roots: Sequence[Mapping[str, Any]] = ()
    sidecar_run_timestamps: Sequence[Mapping[str, Any]] = ()
    age_seconds: int | None = None
    stale_after_seconds: int | None = None

    def __post_init__(self) -> None:
        status = _text(self.status, field_name="status")
        if status not in RELATIONSHIP_FRESHNESS_STATUSES:
            raise ValueError(f"Unsupported relationship freshness status: {status!r}")
        target_key = to_jsonable(dict(self.target_key))
        for field_name in ("asset_id", "relationship_feature_family", "band"):
            if target_key.get(field_name) in (None, ""):
                raise ValueError(f"Relationship freshness target_key missing {field_name!r}")
        reasons = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes if str(reason or "").strip()))
        if status != RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE and not reasons:
            raise ValueError("Unavailable relationship freshness checks require reason codes")
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "relationship_input_roots", tuple(to_jsonable(dict(item)) for item in self.relationship_input_roots))
        object.__setattr__(self, "sidecar_run_timestamps", tuple(to_jsonable(dict(item)) for item in self.sidecar_run_timestamps))

    @property
    def passed(self) -> bool:
        return self.status == RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE

    @property
    def action(self) -> str:
        return "allow" if self.passed else "mask_unavailable"

    @property
    def mask_reason(self) -> str | None:
        if self.passed:
            return None
        if self.status == RELATIONSHIP_FRESHNESS_STATUS_STALE:
            return self.policy.stale_mask_reason
        return self.policy.missing_mask_reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_CHECK_ARTIFACT_KIND,
            "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
            "target_key": to_jsonable(dict(self.target_key)),
            "status": self.status,
            "passed": self.passed,
            "action": self.action,
            "mask_reason": self.mask_reason,
            "reason_codes": list(self.reason_codes),
            "policy": self.policy.as_dict(),
            "relationship_input_tail_ts": to_jsonable(self.relationship_input_tail_ts),
            "relationship_known_at_ts": to_jsonable(self.relationship_known_at_ts),
            "source_tail_ts": to_jsonable(self.source_tail_ts),
            "known_at_ts": to_jsonable(self.known_at_ts),
            "snapshot_cadence_days": self.snapshot_cadence_days,
            "stale_after_days": self.stale_after_days,
            "age_seconds": self.age_seconds,
            "stale_after_seconds": self.stale_after_seconds,
            "relationship_input_roots": [to_jsonable(dict(item)) for item in self.relationship_input_roots],
            "sidecar_run_timestamps": [to_jsonable(dict(item)) for item in self.sidecar_run_timestamps],
            "relationship_input_history_separate_from_selected_profile_artifact": True,
            "selected_profile_artifact": False,
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def relationship_freshness_policy_from_manifest(
    manifest: Mapping[str, Any],
) -> RegimeProductionRelationshipFreshnessPolicy:
    cadence_policy = dict(manifest.get("relationship_context_cadence_policy") or {})
    stale_policy = _mapping(cadence_policy.get("stale_snapshot_policy")) or {}
    policy_id = (
        cadence_policy.get("relationship_context_cadence_policy_id")
        or manifest.get("relationship_context_cadence_policy_id")
        or DEFAULT_RELATIONSHIP_CONTEXT_CADENCE_POLICY_ID
    )
    cadence_days = _mapping(cadence_policy.get("snapshot_cadence_days")) or DEFAULT_SNAPSHOT_CADENCE_DAYS
    stale_after = _mapping(stale_policy.get("stale_after_days_by_band")) or DEFAULT_STALE_AFTER_DAYS
    return RegimeProductionRelationshipFreshnessPolicy(
        policy_id=str(policy_id),
        snapshot_cadence_days_by_band=cadence_days,
        stale_after_days_by_band=stale_after,
        stale_snapshot_policy_id=str(stale_policy.get("policy_id") or DEFAULT_STALE_SNAPSHOT_POLICY_ID),
        stale_action=str(stale_policy.get("action") or "mask_unavailable"),
        stale_mask_reason=str(stale_policy.get("mask_reason") or CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT),
        missing_mask_reason=str(
            cadence_policy.get("missing_snapshot_mask_reason") or CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT
        ),
    )


def evaluate_cross_asset_relationship_input_freshness(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    relationship_input_checks: Sequence[Mapping[str, Any]],
) -> RegimeProductionRelationshipFreshnessCheck:
    policy = relationship_freshness_policy_from_manifest(manifest)
    target_key = {
        "asset_id": row.get("asset_id"),
        "relationship_feature_family": row.get("relationship_feature_family"),
        "band": _band(row.get("band")),
    }
    band = str(target_key["band"])
    relationship_roots = _relationship_input_roots(relationship_input_checks)
    sidecar_timestamps = _sidecar_run_timestamps(manifest=manifest, band=band)
    reasons: list[str] = []
    missing_inputs = _missing_input_reasons(relationship_input_checks)
    reasons.extend(missing_inputs)

    source_tail_ts = _first_present(row, ("source_tail_ts", "relationship_input_tail_ts"))
    known_at_ts = _first_present(row, ("known_at_ts", "relationship_known_at_ts"))
    schedule = _schedule_for_band(manifest, band)
    relationship_tail_ts = _first_present(row, ("relationship_input_tail_ts", "source_tail_ts"))
    relationship_known_at_ts = _first_present(row, ("relationship_known_at_ts", "known_at_ts"))
    if relationship_tail_ts in (None, ""):
        relationship_tail_ts = schedule.get("source_tail_ts")
    if relationship_known_at_ts in (None, ""):
        relationship_known_at_ts = schedule.get("known_at_ts")

    snapshot_cadence_days, cadence_reason = _positive_int(
        _first_present(row, ("snapshot_cadence_days",)) or schedule.get("snapshot_cadence_days") or policy.cadence_days_for_band(band),
        field_name="snapshot_cadence_days",
    )
    stale_after_days, stale_after_reason = _positive_int(
        _stale_after_days(row, policy=policy, band=band),
        field_name="stale_after_days",
    )
    tail_value, tail_reason = _optional_ts(relationship_tail_ts, field_name="relationship_input_tail_ts")
    known_value, known_reason = _optional_ts(relationship_known_at_ts, field_name="relationship_known_at_ts")
    source_value, source_reason = _optional_ts(source_tail_ts, field_name="source_tail_ts")
    selected_known_value, selected_known_reason = _optional_ts(known_at_ts, field_name="known_at_ts")
    for reason in (
        tail_reason,
        known_reason,
        source_reason,
        selected_known_reason,
        cadence_reason,
        stale_after_reason,
    ):
        if reason:
            reasons.append(reason)
    if tail_value is None:
        reasons.append("relationship_input_tail_ts_missing")
    if known_value is None:
        reasons.append("relationship_known_at_ts_missing")
    if source_value is not None and selected_known_value is not None and source_value > selected_known_value:
        reasons.append("source_tail_after_known_at")
    if tail_value is not None and known_value is not None and tail_value > known_value:
        reasons.append("relationship_input_tail_after_known_at")

    age_seconds = None
    stale_after_seconds = None
    stale_reasons: list[str] = []
    if tail_value is not None and known_value is not None and stale_after_days is not None:
        age_seconds = int(known_value) - int(tail_value)
        stale_after_seconds = int(stale_after_days) * 86400
        if age_seconds > stale_after_seconds:
            stale_reasons.append("relationship_input_stale_after_policy")
    reasons.extend(stale_reasons)
    reasons = list(dict.fromkeys(reasons))
    status = RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE
    if missing_inputs or any(_is_missing_reason(reason) for reason in reasons):
        status = RELATIONSHIP_FRESHNESS_STATUS_MISSING
    elif stale_reasons:
        status = RELATIONSHIP_FRESHNESS_STATUS_STALE

    return RegimeProductionRelationshipFreshnessCheck(
        target_key=target_key,
        status=status,
        policy=policy,
        relationship_input_tail_ts=relationship_tail_ts,
        relationship_known_at_ts=relationship_known_at_ts,
        source_tail_ts=source_tail_ts,
        known_at_ts=known_at_ts,
        snapshot_cadence_days=snapshot_cadence_days,
        stale_after_days=stale_after_days,
        reason_codes=tuple(reasons),
        relationship_input_roots=relationship_roots,
        sidecar_run_timestamps=sidecar_timestamps,
        age_seconds=age_seconds,
        stale_after_seconds=stale_after_seconds,
    )


def summarize_relationship_input_freshness(
    checks: Sequence[Mapping[str, Any] | RegimeProductionRelationshipFreshnessCheck],
) -> dict[str, Any]:
    payloads = [item.as_dict() if isinstance(item, RegimeProductionRelationshipFreshnessCheck) else dict(item) for item in checks]
    status_counts = {status: 0 for status in RELATIONSHIP_FRESHNESS_STATUSES}
    reason_counts: dict[str, int] = {}
    by_band: dict[str, dict[str, int]] = {}
    policy_payload = {}
    for item in payloads:
        status = str(item.get("status") or RELATIONSHIP_FRESHNESS_STATUS_MISSING)
        if status not in status_counts:
            status = RELATIONSHIP_FRESHNESS_STATUS_MISSING
        status_counts[status] += 1
        target = dict(item.get("target_key") or {})
        band = str(target.get("band") or "unknown")
        by_band.setdefault(band, {status_name: 0 for status_name in RELATIONSHIP_FRESHNESS_STATUSES})
        by_band[band][status] += 1
        for reason in item.get("reason_codes") or ():
            text = str(reason)
            reason_counts[text] = reason_counts.get(text, 0) + 1
        if not policy_payload and isinstance(item.get("policy"), Mapping):
            policy_payload = dict(item["policy"])
    return {
        "schema_version": REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SUMMARY_ARTIFACT_KIND,
        "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
        "policy": to_jsonable(dict(policy_payload)),
        "unit_count": len(payloads),
        "status_counts": status_counts,
        "status_counts_by_band": by_band,
        "reason_code_counts": reason_counts,
        "relationship_inputs_available_fresh": status_counts[RELATIONSHIP_FRESHNESS_STATUS_STALE] == 0
        and status_counts[RELATIONSHIP_FRESHNESS_STATUS_MISSING] == 0,
        "relationship_input_freshness_recorded": True,
        "selected_profile_artifact": False,
        "relationship_inputs_are_selected_profile_artifacts": False,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }


def _relationship_input_roots(checks: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    roots = []
    for check in checks:
        name = str(check.get("name") or "")
        if name in RELATIONSHIP_INPUT_ROOT_NAMES:
            roots.append(
                {
                    "name": name,
                    "status": check.get("status"),
                    "path": check.get("path"),
                    "root": check.get("root"),
                    "path_source": check.get("path_source"),
                    "root_source": check.get("root_source"),
                    "configured_root_policy": check.get("configured_root_policy"),
                    "selected_profile_artifact": False,
                }
            )
    return tuple(roots)


def _missing_input_reasons(checks: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    by_name = {str(check.get("name") or ""): dict(check) for check in checks}
    reasons: list[str] = []
    for name in RELATIONSHIP_INPUT_ROOT_NAMES:
        check = by_name.get(name)
        if check is None:
            reasons.append(f"{name}_not_declared")
            continue
        status = str(check.get("status") or "")
        reason = check.get("reason_code")
        if status != "available":
            base_reason = str(reason or f"{status or 'missing'}")
            if base_reason in {"relationship_input_missing", "relationship_input_not_declared"}:
                reasons.append(f"{name}_{base_reason}")
            else:
                reasons.append(base_reason)
    return tuple(dict.fromkeys(reasons))


def _sidecar_run_timestamps(*, manifest: Mapping[str, Any], band: str) -> tuple[dict[str, Any], ...]:
    schedule = _schedule_for_band(manifest, band)
    if not schedule:
        return ()
    return (
        {
            "source": "relationship_context_cadence_policy.backfill_snapshot_schedule",
            "band": schedule.get("band"),
            "relationship_snapshot_id": schedule.get("relationship_snapshot_id"),
            "source_tail_ts": schedule.get("source_tail_ts"),
            "known_at_ts": schedule.get("known_at_ts"),
            "snapshot_valid_from_ts": schedule.get("snapshot_valid_from_ts"),
            "snapshot_valid_until_ts": schedule.get("snapshot_valid_until_ts"),
            "snapshot_cadence_days": schedule.get("snapshot_cadence_days"),
            "production_enabled": schedule.get("production_enabled"),
        },
    )


def _schedule_for_band(manifest: Mapping[str, Any], band: str) -> dict[str, Any]:
    cadence_policy = dict(manifest.get("relationship_context_cadence_policy") or {})
    for item in cadence_policy.get("backfill_snapshot_schedule") or ():
        if isinstance(item, Mapping) and _band(item.get("band")) == _band(band):
            return dict(item)
    return {}


def _stale_after_days(row: Mapping[str, Any], *, policy: RegimeProductionRelationshipFreshnessPolicy, band: str) -> Any:
    stale_policy = _mapping(row.get("stale_snapshot_policy"))
    stale_by_band = _mapping(stale_policy.get("stale_after_days_by_band")) if stale_policy else {}
    if stale_by_band and stale_by_band.get(_band(band)) not in (None, ""):
        return stale_by_band.get(_band(band))
    return policy.stale_after_days_for_band(band)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def _first_present(payload: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = payload.get(field)
        if value not in (None, ""):
            return value
    return None


def _positive_int_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    out: dict[str, int] = {}
    for key, raw in value.items():
        parsed, reason = _positive_int(raw, field_name=f"{field_name}.{key}")
        if reason or parsed is None:
            raise ValueError(f"{field_name} values must be positive integers")
        out[_band(key)] = parsed
    if not out:
        raise ValueError(f"{field_name} must not be empty")
    return out


def _positive_int(value: Any, *, field_name: str) -> tuple[int | None, str | None]:
    if value in (None, ""):
        return None, f"{field_name}_missing"
    try:
        parsed = int(value)
    except Exception:
        return None, f"{field_name}_invalid"
    if parsed <= 0:
        return None, f"{field_name}_invalid"
    return parsed, None


def _optional_ts(value: Any, *, field_name: str) -> tuple[int | None, str | None]:
    if value in (None, ""):
        return None, f"{field_name}_missing"
    if isinstance(value, bool):
        return None, f"{field_name}_invalid"
    try:
        return int(float(value)), None
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp()), None
    except Exception:
        return None, f"{field_name}_invalid"


def _is_missing_reason(reason: str) -> bool:
    text = str(reason)
    return (
        "_missing" in text
        or text.endswith("_not_declared")
        or text.endswith("_missing_input")
        or text == "relationship_input_missing"
        or text == "relationship_input_not_declared"
    )


def _band(value: object) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production relationship freshness {field_name} must be non-empty")
    return text


__all__ = [
    "REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_CHECK_ARTIFACT_KIND",
    "REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_POLICY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SCHEMA_VERSION",
    "REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_SUMMARY_ARTIFACT_KIND",
    "RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE",
    "RELATIONSHIP_FRESHNESS_STATUS_MISSING",
    "RELATIONSHIP_FRESHNESS_STATUS_STALE",
    "RegimeProductionRelationshipFreshnessCheck",
    "RegimeProductionRelationshipFreshnessPolicy",
    "evaluate_cross_asset_relationship_input_freshness",
    "relationship_freshness_policy_from_manifest",
    "summarize_relationship_input_freshness",
]
