from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
)
from src.regimes.core.production_reuse_cache import RegimeProductionPlannerRunCache
from src.regimes.core.production_clamp_contract import (
    normalized_regime_production_clamp_range,
    relationship_input_history_check,
)
from src.regimes.core.production_planner import (
    BRANCH_OUTPUT_GRAIN_FIELDS,
    BRANCH_TARGET_KEY_FIELDS,
    MODEL_STATE_REQUIRED_FIELDS,
    REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    RegimeProductionModelStateDefinition,
    RegimeProductionNormalizedLineage,
    RegimeProductionPlannerContract,
)
from src.regimes.core.serialization import to_jsonable


def profile_artifact_sha256(path: str | Path, *, run_cache: RegimeProductionPlannerRunCache | None = None) -> str:
    if run_cache is not None:
        return run_cache.artifact_hash(path, _profile_artifact_sha256_uncached)
    return _profile_artifact_sha256_uncached(path)


def _profile_artifact_sha256_uncached(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return "unavailable:artifact_missing"
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def timestamp_plan_for_contract(
    contract: RegimeProductionPlannerContract,
    *,
    source_tail_ts: Any = None,
    known_at_ts: Any = None,
    production_input_edge_ts: Any = None,
    production_input_edge: Mapping[str, Any] | None = None,
    row_status: str | None = None,
    relationship_input_tail_ts: Any = None,
    relationship_known_at_ts: Any = None,
    snapshot_cadence_days: Any = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> dict[str, Any]:
    if run_cache is not None:
        return dict(
            run_cache.timestamp_plan(
                key_payload={
                    "branch": contract.branch,
                    "clamp_policy_id": contract.clamp_policy.policy_id,
                    "output_timestamp_field": contract.output_grain.timestamp_field,
                    "source_tail_ts": source_tail_ts,
                    "known_at_ts": known_at_ts,
                    "production_input_edge_ts": production_input_edge_ts,
                    "row_status": row_status,
                    "relationship_input_tail_ts": relationship_input_tail_ts,
                    "relationship_known_at_ts": relationship_known_at_ts,
                    "snapshot_cadence_days": snapshot_cadence_days,
                },
                builder=lambda: timestamp_plan_for_contract(
                    contract,
                    source_tail_ts=source_tail_ts,
                    known_at_ts=known_at_ts,
                    production_input_edge_ts=production_input_edge_ts,
                    production_input_edge=production_input_edge,
                    row_status=row_status,
                    relationship_input_tail_ts=relationship_input_tail_ts,
                    relationship_known_at_ts=relationship_known_at_ts,
                    snapshot_cadence_days=snapshot_cadence_days,
                    run_cache=None,
                ),
            )
        )
    clamp = contract.clamp_policy
    source_tail_required = row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
    normalized_range = None
    if (
        row_status is not None
        or source_tail_ts not in (None, "")
        or known_at_ts not in (None, "")
        or production_input_edge_ts not in (None, "")
    ):
        normalized_range = normalized_regime_production_clamp_range(
            branch=contract.branch,
            clamp_policy=clamp,
            source_tail_ts=source_tail_ts,
            known_at_ts=known_at_ts,
            production_input_edge_ts=production_input_edge_ts,
            row_status=row_status,
            source_tail_required=source_tail_required,
        )
    payload: dict[str, Any] = {
        "source": "shared_regime_production_clamp_policy",
        "timestamp_field": contract.output_grain.timestamp_field,
        "timestamps_materialized": False,
        "timestamp_rows_enumerated": False,
        "historical_output_months": int(clamp.historical_output_months),
        "required_lookback_months": int(clamp.required_lookback_months),
        "runtime_boundaries_required": bool(clamp.runtime_boundaries_required),
        "clamp_policy_id": clamp.policy_id,
    }
    if normalized_range is not None:
        normalized_payload = normalized_range.as_dict()
        payload.update(
            {
                "source": "shared_regime_production_clamp_contract",
                "normalized_clamp_range": normalized_payload,
                "clamp_range_status": normalized_payload["status"],
                "clamp_range_passed": bool(normalized_payload["passed"]),
                "source_tail_ts": to_jsonable(source_tail_ts),
                "known_at_ts": to_jsonable(known_at_ts),
                "production_input_edge": to_jsonable(dict(production_input_edge or {})),
                "production_input_edge_ts": normalized_payload["production_input_edge_ts"],
                "output_start_ts": normalized_payload["output_start_ts"],
                "output_end_ts": normalized_payload["output_end_ts"],
                "required_lookback_start_ts": normalized_payload["required_lookback_start_ts"],
                "clamp_reason_codes": list(normalized_payload["reason_codes"]),
            }
        )
        if contract.branch == REGIME_BRANCH_CROSS_ASSET_STATE and (
            relationship_input_tail_ts not in (None, "")
            or relationship_known_at_ts not in (None, "")
            or snapshot_cadence_days not in (None, "")
        ):
            relationship_check = relationship_input_history_check(
                branch=contract.branch,
                clamp_range=normalized_payload,
                relationship_input_tail_ts=relationship_input_tail_ts,
                relationship_known_at_ts=relationship_known_at_ts,
                snapshot_cadence_days=snapshot_cadence_days,
            )
            payload["relationship_input_history_check"] = relationship_check
            payload["relationship_input_history_passed"] = bool(relationship_check["passed"])
            payload["relationship_input_history_reason_codes"] = list(relationship_check["reason_codes"])
    return payload


def clamp_reason_codes_for_timestamp_plan(timestamp_plan: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    normalized = dict(timestamp_plan.get("normalized_clamp_range") or {})
    reasons.extend(str(reason) for reason in normalized.get("reason_codes") or () if str(reason or "").strip())
    relationship = dict(timestamp_plan.get("relationship_input_history_check") or {})
    reasons.extend(str(reason) for reason in relationship.get("reason_codes") or () if str(reason or "").strip())
    return tuple(dict.fromkeys(reasons))


def output_grain_key_for_target(branch: str, target_key: Mapping[str, Any], timestamp_plan: Mapping[str, Any]) -> dict[str, Any]:
    out = {field: target_key.get(field) for field in BRANCH_TARGET_KEY_FIELDS[str(branch)]}
    out["timestamp"] = {
        "source": timestamp_plan.get("source"),
        "field": timestamp_plan.get("timestamp_field", "timestamp"),
        "materialized": False,
    }
    missing = [field for field in BRANCH_OUTPUT_GRAIN_FIELDS[str(branch)] if field not in out]
    if missing:
        raise ValueError(f"Regime Production no-write output grain key missing fields: {missing!r}")
    return out


def normalized_lineage_for_row(
    *,
    branch: str,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_version: Mapping[str, Any],
    target_key: Mapping[str, Any],
    profile_id: str | None,
    profile_version: str | None,
    artifact_path: str | Path,
    artifact_hash: str,
    row_status: str,
    availability_reason_codes: Sequence[str] = (),
) -> RegimeProductionNormalizedLineage:
    branch_name = str(branch)
    raw_lineage_id = first_present(
        row,
        (
            "lineage_id",
            "row_lineage_id",
            "profile_lineage_id",
            "profile_artifact_lineage_id",
        ),
    )
    source_tail_ts = first_present(row, ("source_tail_ts", "relationship_input_tail_ts"))
    known_at_ts = first_present(row, ("known_at_ts", "definition_known_at_ts", "relationship_known_at_ts"))
    raw_lineage_fields = _raw_lineage_fields(row)
    run_id = first_present(row, ("run_id",)) or first_present(manifest, ("run_id",))
    source_run_reference = _source_run_reference(row, manifest)
    normalized_id, equivalent_missing = _normalized_row_lineage_id(
        branch=branch_name,
        row=row,
        manifest=manifest,
        target_key=target_key,
        profile_id=profile_id,
        profile_version=profile_version,
        artifact_hash=artifact_hash,
        raw_lineage_id=raw_lineage_id,
        row_status=row_status,
        source_tail_ts=source_tail_ts,
        known_at_ts=known_at_ts,
    )
    lineage_reasons = []
    if raw_lineage_id in (None, "") and branch_name == REGIME_BRANCH_ASSET_STATE and row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED:
        lineage_reasons.append("asset_state_selected_profile_lineage_id_missing")
    lineage_reasons.extend(equivalent_missing)
    return RegimeProductionNormalizedLineage(
        branch=branch_name,
        profile_id=profile_id,
        profile_version=profile_version,
        lineage_id=None if raw_lineage_id in (None, "") else str(raw_lineage_id),
        normalized_row_lineage_id=normalized_id,
        source_tail_ts=source_tail_ts,
        known_at_ts=known_at_ts,
        selected_profile_artifact_path=str(artifact_path),
        selected_profile_artifact_hash=artifact_hash,
        run_id=None if run_id in (None, "") else str(run_id),
        source_run_reference=None if source_run_reference in (None, "") else str(source_run_reference),
        manifest_schema_version=_manifest_schema_version(manifest, manifest_version),
        branch_schema_policy=_manifest_branch_schema_policy(manifest_version),
        raw_version_field=_manifest_raw_version_field(manifest_version),
        raw_version_value=_manifest_raw_version_value(manifest, manifest_version),
        raw_lineage_id=raw_lineage_id,
        raw_lineage_fields=raw_lineage_fields,
        row_status=row_status,
        availability_reason_codes=tuple(str(reason) for reason in availability_reason_codes if str(reason or "").strip()),
        lineage_reason_codes=tuple(lineage_reasons),
    )


def model_state_definition_or_stub(
    *,
    branch: str,
    target_key: Mapping[str, Any],
    profile_id: str | None,
    profile_version: str | None,
    profile_artifact_path: str,
    profile_artifact_hash: str,
    refit_window_start: Any,
    refit_window_end: Any,
    definition_known_at_ts: Any,
    source_tail_ts: Any,
    refit_cadence_id: str,
    status: str,
    health_metadata: Mapping[str, Any],
    lineage: Mapping[str, Any],
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> Mapping[str, Any]:
    if run_cache is not None:
        return run_cache.definition_state_plan(
            key_payload={
                "branch": branch,
                "target_key": to_jsonable(dict(target_key)),
                "profile_id": profile_id,
                "profile_version": profile_version,
                "profile_artifact_path": profile_artifact_path,
                "profile_artifact_hash": profile_artifact_hash,
                "refit_window_start": refit_window_start,
                "refit_window_end": refit_window_end,
                "definition_known_at_ts": definition_known_at_ts,
                "source_tail_ts": source_tail_ts,
                "refit_cadence_id": refit_cadence_id,
                "status": status,
            },
            builder=lambda: model_state_definition_or_stub(
                branch=branch,
                target_key=target_key,
                profile_id=profile_id,
                profile_version=profile_version,
                profile_artifact_path=profile_artifact_path,
                profile_artifact_hash=profile_artifact_hash,
                refit_window_start=refit_window_start,
                refit_window_end=refit_window_end,
                definition_known_at_ts=definition_known_at_ts,
                source_tail_ts=source_tail_ts,
                refit_cadence_id=refit_cadence_id,
                status=status,
                health_metadata=health_metadata,
                lineage=lineage,
                run_cache=None,
            ),
        )
    missing = []
    required_values = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "profile_artifact_path": profile_artifact_path,
        "profile_artifact_hash": profile_artifact_hash,
        "refit_window_start": refit_window_start,
        "refit_window_end": refit_window_end,
        "definition_known_at_ts": definition_known_at_ts,
        "source_tail_ts": source_tail_ts,
        "refit_cadence_id": refit_cadence_id,
    }
    for field_name, value in required_values.items():
        if value in (None, ""):
            missing.append(field_name)
    if not missing:
        try:
            return RegimeProductionModelStateDefinition(
                branch=branch,
                target_key=target_key,
                profile_id=str(profile_id),
                profile_version=str(profile_version),
                profile_artifact_path=str(profile_artifact_path),
                profile_artifact_hash=str(profile_artifact_hash),
                refit_window_start=refit_window_start,
                refit_window_end=refit_window_end,
                definition_known_at_ts=definition_known_at_ts,
                source_tail_ts=source_tail_ts,
                refit_cadence_id=str(refit_cadence_id),
                status=str(status),
                health_metadata=dict(health_metadata),
                lineage=dict(lineage),
            ).as_dict()
        except Exception as exc:
            missing.append(f"contract_validation:{type(exc).__name__}")
    return {
        "schema_version": 1,
        "artifact_kind": REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND,
        "branch": str(branch),
        "target_key": to_jsonable(dict(target_key)),
        "grain_key": to_jsonable(dict(target_key)),
        "profile_id": profile_id,
        "profile_version": profile_version,
        "profile_artifact_path": str(profile_artifact_path),
        "profile_artifact_hash": str(profile_artifact_hash),
        "refit_window_start": refit_window_start,
        "refit_window_end": refit_window_end,
        "definition_known_at_ts": definition_known_at_ts,
        "source_tail_ts": source_tail_ts,
        "refit_cadence_id": str(refit_cadence_id),
        "status": REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
        "health_metadata": to_jsonable(dict(health_metadata)),
        "lineage": to_jsonable(dict(lineage)),
        "required_fields": list(MODEL_STATE_REQUIRED_FIELDS),
        "missing_required_fields": tuple(dict.fromkeys(missing)),
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }


def first_present(payload: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = payload.get(field)
        if value not in (None, ""):
            return value
    return None


def first_mapping(payload: Mapping[str, Any], fields: Sequence[str]) -> Mapping[str, Any]:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, Mapping):
            return value
    return {}


def _raw_lineage_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "lineage_id",
        "profile_artifact_lineage_source",
        "profile_selection_lineage",
        "trial_study_lineage",
        "relationship_context_id",
        "relationship_snapshot_id",
        "feature_set_version",
        "selection_engine_version",
        "source_tail_ts",
        "known_at_ts",
    )
    return {key: to_jsonable(row.get(key)) for key in keys if key in row}


def _source_run_reference(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> Any:
    return first_present(
        row,
        (
            "source_run_reference",
            "trial_id",
            "selected_candidate_id",
            "relationship_context_id",
            "relationship_snapshot_id",
            "profile_candidate_set_id",
        ),
    ) or first_present(
        manifest,
        (
            "run_id",
            "artifact_label",
            "artifact_scope",
            "created_at",
            "created_at_utc",
        ),
    )


def _normalized_row_lineage_id(
    *,
    branch: str,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_key: Mapping[str, Any],
    profile_id: str | None,
    profile_version: str | None,
    artifact_hash: str,
    raw_lineage_id: Any,
    row_status: str,
    source_tail_ts: Any,
    known_at_ts: Any,
) -> tuple[str | None, tuple[str, ...]]:
    if raw_lineage_id not in (None, ""):
        return str(raw_lineage_id), ()
    missing: list[str] = []
    equivalent = {
        "branch": branch,
        "target_key": to_jsonable(dict(target_key)),
        "profile_id": profile_id,
        "profile_version": profile_version,
        "artifact_hash": artifact_hash,
        "source_tail_ts": source_tail_ts,
        "known_at_ts": known_at_ts,
    }
    if row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED:
        if source_tail_ts in (None, ""):
            missing.append("source_tail_ts_missing_for_lineage_equivalent")
        if known_at_ts in (None, ""):
            missing.append("known_at_ts_missing_for_lineage_equivalent")
    if branch == REGIME_BRANCH_ASSET_STATE and row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED:
        missing.append("asset_state_raw_lineage_id_required")
    elif branch == REGIME_BRANCH_MARKET_STATE:
        run_id = first_present(row, ("run_id",)) or first_present(manifest, ("run_id",))
        interval = first_present(row, ("source_interval", "interval"))
        window_profile_id = first_present(row, ("window_profile_id",))
        trial_study_lineage = row.get("trial_study_lineage")
        if row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED and run_id in (None, ""):
            missing.append("market_state_run_id_missing_for_lineage_equivalent")
        if row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED and interval in (None, "") and window_profile_id in (None, "") and not isinstance(trial_study_lineage, Mapping):
            missing.append("market_state_profile_source_reference_missing_for_lineage_equivalent")
        equivalent.update(
            {
                "run_id": run_id,
                "source_interval": interval,
                "window_profile_id": window_profile_id,
                "trial_study_lineage": to_jsonable(dict(trial_study_lineage)) if isinstance(trial_study_lineage, Mapping) else None,
            }
        )
    elif branch == REGIME_BRANCH_CROSS_ASSET_STATE:
        context_id = first_present(row, ("relationship_context_id",))
        snapshot_id = first_present(row, ("relationship_snapshot_id",))
        feature_set_version = first_present(row, ("feature_set_version",))
        selection_engine_version = first_present(row, ("selection_engine_version",)) or first_present(manifest, ("selection_engine_version",))
        if row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED and context_id in (None, "") and snapshot_id in (None, ""):
            missing.append("cross_asset_relationship_context_missing_for_lineage_equivalent")
        if row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED and feature_set_version in (None, "") and selection_engine_version in (None, ""):
            missing.append("cross_asset_profile_source_reference_missing_for_lineage_equivalent")
        equivalent.update(
            {
                "relationship_context_id": context_id,
                "relationship_snapshot_id": snapshot_id,
                "feature_set_version": feature_set_version,
                "selection_engine_version": selection_engine_version,
            }
        )
    if missing:
        return None, tuple(dict.fromkeys(missing))
    raw = json.dumps(to_jsonable(equivalent), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{branch}_normalized_row_lineage:{hashlib.sha256(raw).hexdigest()}", ()


def _manifest_schema_version(manifest: Mapping[str, Any], manifest_version: Mapping[str, Any]) -> int | None:
    value = (
        manifest_version.get("manifest_schema_version")
        if "manifest_schema_version" in manifest_version
        else manifest_version.get("schema_version")
    )
    if value is None:
        value = manifest.get("schema_version")
    if value is None and manifest_version.get("raw_version_field") == "selection_engine_version":
        raw_value = manifest_version.get("raw_version_value")
        if raw_value == manifest.get("selection_engine_version"):
            value = manifest_version.get("manifest_schema_version")
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _manifest_branch_schema_policy(manifest_version: Mapping[str, Any]) -> str | None:
    value = manifest_version.get("branch_schema_policy")
    if value in (None, "") and isinstance(manifest_version.get("manifest_version"), Mapping):
        value = dict(manifest_version["manifest_version"]).get("branch_schema_policy")
    return None if value in (None, "") else str(value)


def _manifest_raw_version_field(manifest_version: Mapping[str, Any]) -> str | None:
    value = manifest_version.get("raw_version_field")
    if value in (None, "") and isinstance(manifest_version.get("manifest_version"), Mapping):
        value = dict(manifest_version["manifest_version"]).get("raw_version_field")
    return None if value in (None, "") else str(value)


def _manifest_raw_version_value(manifest: Mapping[str, Any], manifest_version: Mapping[str, Any]) -> Any:
    if "raw_version_value" in manifest_version:
        return manifest_version.get("raw_version_value")
    if isinstance(manifest_version.get("manifest_version"), Mapping):
        nested = dict(manifest_version["manifest_version"])
        if "raw_version_value" in nested:
            return nested.get("raw_version_value")
    field = _manifest_raw_version_field(manifest_version)
    return manifest.get(field) if field else None


__all__ = [
    "clamp_reason_codes_for_timestamp_plan",
    "first_mapping",
    "first_present",
    "model_state_definition_or_stub",
    "normalized_lineage_for_row",
    "output_grain_key_for_target",
    "profile_artifact_sha256",
    "timestamp_plan_for_contract",
]
