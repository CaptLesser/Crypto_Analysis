from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.production_consumer import REGIME_BRANCH_CROSS_ASSET_STATE
from src.regimes.core.production_no_write_planner_utils import (
    clamp_reason_codes_for_timestamp_plan,
    first_mapping,
    first_present,
    model_state_definition_or_stub,
    normalized_lineage_for_row,
    output_grain_key_for_target,
    profile_artifact_sha256,
    timestamp_plan_for_contract,
)
from src.regimes.core.production_planner import (
    REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_STATUS_BLOCKED,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    RegimeProductionNoWritePlan,
    RegimeProductionPlanningUnit,
    RegimeProductionRelationshipInputContract,
    build_regime_production_planner_contract,
)
from src.regimes.core.production_input_edge import resolve_regime_production_input_edge
from src.regimes.core.production_relationship_freshness import (
    REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_CHECK_ARTIFACT_KIND,
    RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE,
    evaluate_cross_asset_relationship_input_freshness,
    summarize_relationship_input_freshness,
)
from src.regimes.core.production_reuse_cache import (
    RegimeProductionPlannerRunCache,
    build_profile_lookup_index,
    source_tail_fingerprint,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.production_worker_contract import build_regime_production_job_matrix
from src.regimes.cross_asset_state.production_consumer import (
    CROSS_ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION,
    CrossAssetStateProductionConsumerValidation,
    validate_cross_asset_state_selected_profiles_for_consumption,
    validate_default_cross_asset_state_selected_profiles_for_consumption,
)


CROSS_ASSET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION = 1


def plan_cross_asset_state_production_no_write(
    manifest_path: str | Path,
    *,
    active_filename: str | None = None,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="cross_asset_state_production_planner_local")
    validation = validate_cross_asset_state_selected_profiles_for_consumption(
        manifest_path,
        sandbox_nonproduction_mode=True,
        active_filename=active_filename,
        run_cache=cache,
    )
    return build_cross_asset_state_production_no_write_plan_from_validation(validation, run_cache=cache, env=env)


def plan_default_cross_asset_state_production_no_write(
    *,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="cross_asset_state_production_planner_local")
    validation = validate_default_cross_asset_state_selected_profiles_for_consumption(
        sandbox_nonproduction_mode=True,
        env=env,
        run_cache=cache,
    )
    return build_cross_asset_state_production_no_write_plan_from_validation(validation, run_cache=cache, env=env)


def build_cross_asset_state_production_no_write_plan_from_validation(
    validation: CrossAssetStateProductionConsumerValidation,
    *,
    run_cache: RegimeProductionPlannerRunCache | None = None,
    env: Mapping[str, str] | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="cross_asset_state_production_planner_local")
    consumer_payload = validation.as_dict()
    relationship_inputs = tuple(_relationship_contract_from_check(check) for check in validation.relationship_input_checks)
    contract = build_regime_production_planner_contract(
        REGIME_BRANCH_CROSS_ASSET_STATE,
        relationship_inputs=relationship_inputs,
    )
    input_edge = resolve_regime_production_input_edge(REGIME_BRANCH_CROSS_ASSET_STATE, env=env)
    input_edge_payload = input_edge.as_dict()
    timestamp_plan = timestamp_plan_for_contract(
        contract,
        production_input_edge_ts=input_edge.edge_ts,
        production_input_edge=input_edge_payload,
        run_cache=cache,
    )
    artifact_hash = profile_artifact_sha256(validation.manifest_path, run_cache=cache)
    manifest = dict(validation.manifest)
    manifest_version = consumer_payload.get("shared_validation") or {}
    ready = validation.status == CROSS_ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION
    selected = [dict(item) for item in manifest.get("selected_profiles") or () if isinstance(item, Mapping)] if ready else []
    diagnostic = [dict(item) for item in manifest.get("diagnostic_only_profiles") or () if isinstance(item, Mapping)] if ready else []
    masked = [dict(item) for item in manifest.get("masked_or_skipped_cells") or () if isinstance(item, Mapping)] if ready else []
    base_relationship_checks = tuple(dict(item) for item in validation.relationship_input_checks)
    lookup_index = cache.profile_lookup_index(
        branch=REGIME_BRANCH_CROSS_ASSET_STATE,
        artifact_hash=artifact_hash,
        source_tail_fingerprint=source_tail_fingerprint((*selected, *diagnostic, *masked)),
        config_fingerprint={
            "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
            "target_fields": ("asset_id", "relationship_feature_family", "band"),
        },
        builder=lambda: build_profile_lookup_index(
            branch=REGIME_BRANCH_CROSS_ASSET_STATE,
            artifact_hash=artifact_hash,
            target_fields=("asset_id", "relationship_feature_family", "band"),
            selected_records=selected,
            diagnostic_records=diagnostic,
            unavailable_records=masked,
        ),
    )
    units = tuple(
        [
            _cross_asset_state_planning_unit(
                row,
                manifest=manifest,
                manifest_version=manifest_version,
                artifact_path=validation.manifest_path,
                artifact_hash=artifact_hash,
                contract=contract,
                timestamp_plan=timestamp_plan,
                planning_status=REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
                relationship_checks=base_relationship_checks,
                run_cache=cache,
            )
            for row in selected
        ]
        + [
            _cross_asset_state_planning_unit(
                row,
                manifest=manifest,
                manifest_version=manifest_version,
                artifact_path=validation.manifest_path,
                artifact_hash=artifact_hash,
                contract=contract,
                timestamp_plan=timestamp_plan,
                planning_status=REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY,
                relationship_checks=base_relationship_checks,
                run_cache=cache,
            )
            for row in diagnostic
        ]
        + [
            _cross_asset_state_planning_unit(
                row,
                manifest=manifest,
                manifest_version=manifest_version,
                artifact_path=validation.manifest_path,
                artifact_hash=artifact_hash,
                contract=contract,
                timestamp_plan=timestamp_plan,
                planning_status=REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
                relationship_checks=base_relationship_checks,
                run_cache=cache,
            )
            for row in masked
        ]
    )
    relationship_freshness_checks = tuple(_relationship_freshness_check_from_unit(unit) for unit in units)
    relationship_freshness_summary = summarize_relationship_input_freshness(relationship_freshness_checks)
    relationship_checks = (*base_relationship_checks, relationship_freshness_summary)
    selected_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED)
    diagnostic_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY)
    masked_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE)
    normalized_lineage_blocked_count = sum(1 for unit in units if not dict(unit.normalized_lineage or {}).get("passed"))
    selected_lineage_blocked_count = sum(
        1
        for unit in units
        if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
        and not dict(unit.normalized_lineage or {}).get("passed")
    )
    relationship_warnings = tuple(
        dict.fromkeys(
            (
                *(
                    str(check.get("reason_code"))
                    for check in base_relationship_checks
                    if check.get("reason_code") not in (None, "")
                ),
                *(relationship_freshness_summary.get("reason_code_counts") or {}).keys(),
            )
        )
    )
    job_matrix = build_regime_production_job_matrix(REGIME_BRANCH_CROSS_ASSET_STATE, units)
    job_matrix_payload = job_matrix.as_dict()
    telemetry = {
        "schema_version": CROSS_ASSET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION,
        "artifact_kind": "cross_asset_state_production_no_write_planner_telemetry",
        "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
        "planned_unit_count": len(units),
        "selected_unit_count": selected_count,
        "diagnostic_only_unit_count": diagnostic_count,
        "masked_unavailable_unit_count": masked_count,
        "expected_cell_count": int(validation.expected_cell_count),
        "selected_model_facing_profile_count": int(validation.selected_model_facing_profile_count),
        "diagnostic_only_profile_count": int(validation.diagnostic_only_profile_count),
        "masked_or_skipped_cell_count": int(validation.masked_or_skipped_cell_count),
        "missing_cell_count": int(validation.missing_cell_count),
        "normalized_lineage_blocked_unit_count": normalized_lineage_blocked_count,
        "selected_normalized_lineage_blocked_unit_count": selected_lineage_blocked_count,
        "relationship_input_checks": [to_jsonable(dict(item)) for item in relationship_checks],
        "relationship_input_freshness": relationship_freshness_summary,
        "relationship_input_freshness_recorded": True,
        "relationship_inputs_available_fresh": bool(relationship_freshness_summary["relationship_inputs_available_fresh"]),
        "relationship_input_warning_count": len(relationship_warnings),
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "relationship_sidecars_selected_profile_artifacts": False,
        "clamp_policy": contract.clamp_policy.as_dict(),
        "production_input_edge": input_edge_payload,
        "refit_cadence": contract.refit_cadence.as_dict(),
        "job_matrix_summary": _job_matrix_summary(job_matrix_payload),
        "profile_lookup_index": lookup_index.as_dict(include_records=False),
        "reuse_cache_telemetry": cache.as_dict(),
        "frame_construction_performed": False,
        "test_branch_logic_executed": False,
        "production_labels_emitted": False,
        "production_writes_performed": False,
    }
    safety_status = REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION if ready else REGIME_PRODUCTION_STATUS_BLOCKED
    return RegimeProductionNoWritePlan(
        branch=REGIME_BRANCH_CROSS_ASSET_STATE,
        artifact_path=validation.manifest_path,
        profile_artifact_hash=artifact_hash,
        consumer_validation=consumer_payload,
        planner_contract=contract,
        shared_dry_run_plan=consumer_payload.get("dry_run_plan") or {},
        planning_units=units,
        telemetry=telemetry,
        job_matrix=job_matrix_payload,
        relationship_input_checks=relationship_checks,
        warnings=relationship_warnings,
        safety_status=safety_status,
    )


def _cross_asset_state_planning_unit(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_version: Mapping[str, Any],
    artifact_path: Path,
    artifact_hash: str,
    contract,
    timestamp_plan: Mapping[str, Any],
    planning_status: str,
    relationship_checks,
    run_cache: RegimeProductionPlannerRunCache | None,
) -> RegimeProductionPlanningUnit:
    target_key = {
        "asset_id": str(row.get("asset_id") or ""),
        "relationship_feature_family": str(row.get("relationship_feature_family") or ""),
        "band": str(row.get("band") or ""),
    }
    profile_id = str(row.get("profile_id") or "|".join(target_key.values()))
    profile_version = _profile_version(row, manifest)
    freshness_check = evaluate_cross_asset_relationship_input_freshness(
        row,
        manifest=manifest,
        relationship_input_checks=relationship_checks,
    ).as_dict()
    effective_status = planning_status
    if (
        planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
        and freshness_check["status"] != RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE
    ):
        effective_status = REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE
    availability_reasons = tuple(
        reason
        for reason in (
            row.get("mask_reason"),
            row.get("filter_reason_code"),
            row.get("diagnostic_only_reason"),
            row.get("selection_exclusion_reason"),
            freshness_check.get("mask_reason") if effective_status != planning_status else None,
            *(freshness_check.get("reason_codes") or ()),
        )
        if reason
    )
    normalized_lineage = normalized_lineage_for_row(
        branch=REGIME_BRANCH_CROSS_ASSET_STATE,
        row=row,
        manifest=manifest,
        manifest_version=manifest_version,
        target_key=target_key,
        profile_id=profile_id,
        profile_version=profile_version,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        row_status=effective_status,
        availability_reason_codes=availability_reasons,
    )
    normalized_lineage_payload = normalized_lineage.as_dict()
    model_state = model_state_definition_or_stub(
        branch=REGIME_BRANCH_CROSS_ASSET_STATE,
        target_key=target_key,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_artifact_path=str(artifact_path),
        profile_artifact_hash=artifact_hash,
        refit_window_start=first_present(row, ("snapshot_valid_from_ts", "refit_window_start", "window_start_ts")),
        refit_window_end=first_present(row, ("snapshot_valid_until_ts", "refit_window_end", "window_end_ts")),
        definition_known_at_ts=row.get("known_at_ts"),
        source_tail_ts=row.get("source_tail_ts"),
        refit_cadence_id=contract.refit_cadence.refit_cadence_id,
        status=(
            REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED
            if effective_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
            else REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT
        ),
        health_metadata={
            "output_health_status": row.get("output_health_status"),
            "output_health_score": row.get("output_health_score"),
            "candidate_readiness_status": row.get("candidate_readiness_status"),
            "selection_eligible": row.get("selection_eligible"),
        },
        lineage={
            "selection_engine_version": row.get("selection_engine_version") or manifest.get("selection_engine_version"),
            "relationship_context_id": row.get("relationship_context_id"),
            "relationship_snapshot_id": row.get("relationship_snapshot_id"),
            "feature_set_version": row.get("feature_set_version"),
            "split_policy_id": row.get("split_policy_id"),
            "normalized_lineage": normalized_lineage_payload,
        },
        run_cache=run_cache,
    )
    unit_timestamp_plan = timestamp_plan_for_contract(
        contract,
        source_tail_ts=model_state.get("source_tail_ts"),
        known_at_ts=model_state.get("definition_known_at_ts"),
        production_input_edge_ts=timestamp_plan.get("production_input_edge_ts"),
        production_input_edge=dict(timestamp_plan.get("production_input_edge") or {}),
        row_status=effective_status,
        relationship_input_tail_ts=freshness_check.get("relationship_input_tail_ts"),
        relationship_known_at_ts=freshness_check.get("relationship_known_at_ts"),
        snapshot_cadence_days=freshness_check.get("snapshot_cadence_days"),
        run_cache=run_cache,
    )
    clamp_reasons = clamp_reason_codes_for_timestamp_plan(unit_timestamp_plan)
    return RegimeProductionPlanningUnit(
        branch=REGIME_BRANCH_CROSS_ASSET_STATE,
        target_key=target_key,
        output_grain_key=output_grain_key_for_target(REGIME_BRANCH_CROSS_ASSET_STATE, target_key, unit_timestamp_plan),
        timestamp_plan=unit_timestamp_plan,
        planning_status=effective_status,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_artifact_path=str(artifact_path),
        profile_artifact_hash=artifact_hash,
        model_state_definition=model_state,
        method_metadata={
            "selected_method_family": row.get("selected_method_family") or row.get("method_family"),
            "clusterer_family": row.get("clusterer_family"),
            "embedding": row.get("embedding"),
            "adapter_name": row.get("adapter_name"),
            "shared_adapter_used": row.get("shared_adapter_used"),
            "selected_parameter_grid_id": row.get("selected_parameter_grid_id"),
            "candidate_params": to_jsonable(dict(row.get("candidate_params") or {})),
            "window_profile_id": row.get("window_profile_id"),
            "relationship_context_cadence_policy_id": row.get("relationship_context_cadence_policy_id"),
            "snapshot_cadence_days": row.get("snapshot_cadence_days"),
            "stale_snapshot_policy": row.get("stale_snapshot_policy"),
            "relationship_input_freshness_check": to_jsonable(dict(freshness_check)),
            "normalized_clamp_range": to_jsonable(dict(unit_timestamp_plan.get("normalized_clamp_range") or {})),
            "relationship_input_history_check": to_jsonable(dict(unit_timestamp_plan.get("relationship_input_history_check") or {})),
        },
        health_metadata={
            "output_health_status": row.get("output_health_status"),
            "output_health_score": row.get("output_health_score"),
            "semantic_score": row.get("semantic_score"),
            "temporal_score": row.get("temporal_score"),
            "coverage_score": row.get("coverage_score"),
            "dominant_state_share": row.get("dominant_state_share"),
        },
        normalized_lineage=normalized_lineage,
        relationship_input_checks=(*relationship_checks, freshness_check),
        reason_codes=tuple(
            reason
            for reason in (
                *availability_reasons,
                *clamp_reasons,
                "normalized_lineage_not_traceable" if not normalized_lineage.passed else None,
                "model_state_missing_required_fields" if model_state.get("missing_required_fields") else None,
            )
            if reason
        ),
    )


def _relationship_freshness_check_from_unit(unit: RegimeProductionPlanningUnit) -> Mapping[str, Any]:
    for check in unit.relationship_input_checks:
        if dict(check).get("artifact_kind") == REGIME_PRODUCTION_RELATIONSHIP_FRESHNESS_CHECK_ARTIFACT_KIND:
            return dict(check)
    return {}


def _relationship_contract_from_check(check: Mapping[str, Any]) -> RegimeProductionRelationshipInputContract:
    return RegimeProductionRelationshipInputContract(
        input_id=str(check.get("name") or check.get("field") or "relationship_input"),
        input_kind=str(check.get("field") or check.get("input_role") or "relationship_input"),
        input_role=str(check.get("input_role") or "time_indexed_relationship_data_input"),
        path=str(check.get("path")) if check.get("path") not in (None, "") else None,
        metadata={
            "status": check.get("status"),
            "reason_code": check.get("reason_code"),
            "execution_performed": False,
            "selected_profile_artifact": False,
        },
    )


def _profile_version(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    for value in (
        row.get("profile_version"),
        row.get("selection_engine_version"),
        manifest.get("selection_engine_version"),
        row.get("scoring_schema_version"),
        row.get("window_profile_id"),
    ):
        if value not in (None, ""):
            text = str(value)
            return text if text.startswith("profile_") or text.startswith("schema_") else f"profile_version_{text}"
    return "profile_version_unversioned"


def _job_matrix_summary(job_matrix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_kind": job_matrix.get("artifact_kind"),
        "work_unit_count": job_matrix.get("work_unit_count"),
        "job_batch_count": job_matrix.get("job_batch_count"),
        "relationship_input_check_count": job_matrix.get("relationship_input_check_count"),
        "relationship_input_check_batch_count": job_matrix.get("relationship_input_check_batch_count"),
        "relationship_input_checks_batched": job_matrix.get("relationship_input_checks_batched"),
        "relationship_discovery_or_pairwise_run_performed": job_matrix.get(
            "relationship_discovery_or_pairwise_run_performed"
        ),
        "workers": job_matrix.get("workers"),
        "effective_workers": job_matrix.get("effective_workers"),
        "model_threads": job_matrix.get("model_threads"),
        "writer_workers": job_matrix.get("writer_workers"),
        "backend": job_matrix.get("backend"),
        "batch_size": job_matrix.get("batch_size"),
        "grouping_fields": list(job_matrix.get("grouping_fields") or ()),
        "parent_single_finalizer": dict(job_matrix.get("parent_finalizer") or {}).get("parent_single_finalizer"),
        "workers_write_outputs": job_matrix.get("workers_write_outputs"),
    }


__all__ = [
    "CROSS_ASSET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION",
    "build_cross_asset_state_production_no_write_plan_from_validation",
    "plan_cross_asset_state_production_no_write",
    "plan_default_cross_asset_state_production_no_write",
]
