from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.regimes.asset_state.production_consumer import (
    ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION,
    AssetStateProductionConsumerValidation,
    validate_asset_state_selected_profiles_for_consumption,
    validate_default_asset_state_selected_profiles_for_consumption,
)
from src.regimes.core.production_consumer import REGIME_BRANCH_ASSET_STATE
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
    REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED,
    REGIME_PRODUCTION_STATUS_BLOCKED,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    RegimeProductionNoWritePlan,
    RegimeProductionPlanningUnit,
    build_regime_production_planner_contract,
)
from src.regimes.core.production_input_edge import resolve_regime_production_input_edge
from src.regimes.core.production_reuse_cache import (
    RegimeProductionPlannerRunCache,
    build_profile_lookup_index,
    source_tail_fingerprint,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.production_worker_contract import build_regime_production_job_matrix


ASSET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION = 1


def plan_asset_state_production_no_write(
    manifest_path: str | Path,
    *,
    expected_cell_count: int | None = None,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="asset_state_production_planner_local")
    validation = validate_asset_state_selected_profiles_for_consumption(
        manifest_path,
        sandbox_nonproduction_mode=True,
        expected_cell_count=expected_cell_count,
        run_cache=cache,
    )
    return build_asset_state_production_no_write_plan_from_validation(validation, run_cache=cache, env=env)


def plan_default_asset_state_production_no_write(
    *,
    expected_cell_count: int | None = None,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="asset_state_production_planner_local")
    validation = validate_default_asset_state_selected_profiles_for_consumption(
        sandbox_nonproduction_mode=True,
        expected_cell_count=expected_cell_count,
        env=env,
        run_cache=cache,
    )
    return build_asset_state_production_no_write_plan_from_validation(validation, run_cache=cache, env=env)


def build_asset_state_production_no_write_plan_from_validation(
    validation: AssetStateProductionConsumerValidation,
    *,
    run_cache: RegimeProductionPlannerRunCache | None = None,
    env: Mapping[str, str] | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="asset_state_production_planner_local")
    consumer_payload = validation.as_dict()
    contract = build_regime_production_planner_contract(REGIME_BRANCH_ASSET_STATE)
    input_edge = resolve_regime_production_input_edge(REGIME_BRANCH_ASSET_STATE, env=env)
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
    ready = validation.status == ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION
    profiles = [dict(item) for item in manifest.get("profiles") or () if isinstance(item, Mapping)] if ready else []
    selected_profiles = [profile for profile in profiles if str(profile.get("selection_status") or "").startswith("selected")]
    skipped_profiles = [profile for profile in profiles if not str(profile.get("selection_status") or "").startswith("selected")]
    lookup_index = cache.profile_lookup_index(
        branch=REGIME_BRANCH_ASSET_STATE,
        artifact_hash=artifact_hash,
        source_tail_fingerprint=source_tail_fingerprint(profiles),
        config_fingerprint={"branch": REGIME_BRANCH_ASSET_STATE, "target_fields": ("asset_id", "axis", "band")},
        builder=lambda: build_profile_lookup_index(
            branch=REGIME_BRANCH_ASSET_STATE,
            artifact_hash=artifact_hash,
            target_fields=("asset_id", "axis", "band"),
            selected_records=selected_profiles,
            unavailable_records=skipped_profiles,
        ),
    )
    units = tuple(
        _asset_state_planning_unit(
            profile,
            manifest=manifest,
            manifest_version=manifest_version,
            artifact_path=validation.manifest_path,
            artifact_hash=artifact_hash,
            contract=contract,
                timestamp_plan=timestamp_plan,
            run_cache=cache,
        )
        for profile in profiles
    )
    selected_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED)
    skipped_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED)
    model_state_missing_count = sum(
        1
        for unit in units
        if isinstance(unit.model_state_definition, Mapping) and unit.model_state_definition.get("missing_required_fields")
    )
    normalized_lineage_blocked_count = sum(1 for unit in units if not dict(unit.normalized_lineage or {}).get("passed"))
    selected_lineage_blocked_count = sum(
        1
        for unit in units
        if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
        and not dict(unit.normalized_lineage or {}).get("passed")
    )
    warnings = []
    if model_state_missing_count:
        warnings.append("asset_state_model_state_lineage_or_refit_window_fields_missing_in_active_artifact")
    if selected_lineage_blocked_count:
        warnings.append("asset_state_selected_normalized_lineage_not_traceable")
    job_matrix = build_regime_production_job_matrix(REGIME_BRANCH_ASSET_STATE, units)
    job_matrix_payload = job_matrix.as_dict()
    telemetry = {
        "schema_version": ASSET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION,
        "artifact_kind": "asset_state_production_no_write_planner_telemetry",
        "branch": REGIME_BRANCH_ASSET_STATE,
        "planned_unit_count": len(units),
        "selected_unit_count": selected_count,
        "skipped_or_filtered_unit_count": skipped_count,
        "masked_unavailable_unit_count": 0,
        "expected_cell_count": int(validation.expected_cell_count),
        "selected_profile_count": int(validation.selected_profile_count),
        "skipped_or_filtered_count": int(validation.skipped_or_filtered_count),
        "model_state_missing_required_field_unit_count": model_state_missing_count,
        "normalized_lineage_blocked_unit_count": normalized_lineage_blocked_count,
        "selected_normalized_lineage_blocked_unit_count": selected_lineage_blocked_count,
        "clamp_policy": contract.clamp_policy.as_dict(),
        "production_input_edge": input_edge_payload,
        "refit_cadence": contract.refit_cadence.as_dict(),
        "job_matrix_summary": _job_matrix_summary(job_matrix_payload),
        "profile_lookup_index": lookup_index.as_dict(include_records=False),
        "reuse_cache_telemetry": cache.as_dict(),
        "local_reselection_policy_metadata_only": True,
        "local_reselection_execution_performed": False,
        "frame_construction_performed": False,
        "test_branch_logic_executed": False,
        "production_labels_emitted": False,
        "production_writes_performed": False,
    }
    safety_status = REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION if ready else REGIME_PRODUCTION_STATUS_BLOCKED
    return RegimeProductionNoWritePlan(
        branch=REGIME_BRANCH_ASSET_STATE,
        artifact_path=validation.manifest_path,
        profile_artifact_hash=artifact_hash,
        consumer_validation=consumer_payload,
        planner_contract=contract,
        shared_dry_run_plan=consumer_payload.get("dry_run_plan") or {},
        planning_units=units,
        telemetry=telemetry,
        job_matrix=job_matrix_payload,
        warnings=tuple(warnings),
        safety_status=safety_status,
    )


def _asset_state_planning_unit(
    profile: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_version: Mapping[str, Any],
    artifact_path: Path,
    artifact_hash: str,
    contract,
    timestamp_plan: Mapping[str, Any],
    run_cache: RegimeProductionPlannerRunCache | None,
) -> RegimeProductionPlanningUnit:
    target_key = {
        "asset_id": str(profile.get("asset_id") or ""),
        "axis": str(profile.get("axis") or ""),
        "band": str(profile.get("band") or ""),
    }
    selection_status = str(profile.get("selection_status") or "")
    planning_status = (
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
        if selection_status.startswith("selected")
        else REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED
    )
    profile_id = str(profile.get("profile_id") or "::".join(target_key.values()))
    profile_version = _profile_version(profile, manifest)
    evidence = first_mapping(profile, ("evidence", "score_evidence_summary"))
    refit_safety = first_mapping(manifest, ("refit_label_safety",))
    local_reselection = first_mapping(refit_safety, ("local_reselection_policy",))
    availability_reasons = tuple(
        reason
        for reason in (
            profile.get("skipped_or_filtered_reason"),
        )
        if reason
    )
    normalized_lineage = normalized_lineage_for_row(
        branch=REGIME_BRANCH_ASSET_STATE,
        row=profile,
        manifest=manifest,
        manifest_version=manifest_version,
        target_key=target_key,
        profile_id=profile_id,
        profile_version=profile_version,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        row_status=planning_status,
        availability_reason_codes=availability_reasons,
    )
    normalized_lineage_payload = normalized_lineage.as_dict()
    model_state = model_state_definition_or_stub(
        branch=REGIME_BRANCH_ASSET_STATE,
        target_key=target_key,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_artifact_path=str(artifact_path),
        profile_artifact_hash=artifact_hash,
        refit_window_start=first_present(profile, ("refit_window_start", "selected_window_start_ts", "window_start_ts")),
        refit_window_end=first_present(profile, ("refit_window_end", "selected_window_end_ts", "window_end_ts")),
        definition_known_at_ts=first_present(profile, ("known_at_ts",)) or manifest.get("created_at"),
        source_tail_ts=first_present(profile, ("source_tail_ts",)),
        refit_cadence_id=contract.refit_cadence.refit_cadence_id,
        status=REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
        health_metadata=evidence,
        lineage={
            "run_id": profile.get("run_id") or manifest.get("run_id"),
            "trial_id": profile.get("trial_id"),
            "selection_scope": profile.get("selection_scope") or manifest.get("selection_scope"),
            "production_handoff_artifact": manifest.get("production_handoff_artifact"),
            "lineage_id": profile.get("lineage_id"),
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
        row_status=planning_status,
        run_cache=run_cache,
    )
    clamp_reasons = clamp_reason_codes_for_timestamp_plan(unit_timestamp_plan)
    return RegimeProductionPlanningUnit(
        branch=REGIME_BRANCH_ASSET_STATE,
        target_key=target_key,
        output_grain_key=output_grain_key_for_target(REGIME_BRANCH_ASSET_STATE, target_key, unit_timestamp_plan),
        timestamp_plan=unit_timestamp_plan,
        planning_status=planning_status,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_artifact_path=str(artifact_path),
        profile_artifact_hash=artifact_hash,
        model_state_definition=model_state,
        method_metadata={
            "selected_clusterer_family": profile.get("selected_clusterer_family"),
            "selected_clustering_library_method": profile.get("selected_clustering_library_method"),
            "selected_embedding": profile.get("selected_embedding"),
            "selected_preprocessing_profile": profile.get("selected_preprocessing_profile"),
            "selected_feature_pool_id": profile.get("selected_feature_pool_id"),
            "selected_assignment_policy": profile.get("selected_assignment_policy"),
            "selected_core_parameters": to_jsonable(dict(profile.get("selected_core_parameters") or {})),
            "selected_assignment_policy_metadata": to_jsonable(dict(profile.get("selected_assignment_policy_metadata") or {})),
            "window_profile_id": profile.get("window_profile_id") or profile.get("selected_window_profile_id"),
            "window_profile_lookback_days": profile.get("selected_window_profile_lookback_days"),
            "normalized_clamp_range": to_jsonable(dict(unit_timestamp_plan.get("normalized_clamp_range") or {})),
        },
        health_metadata=evidence,
        local_reselection_metadata={
            "metadata_only": True,
            "execution_performed": False,
            "local_reselection_policy": to_jsonable(dict(local_reselection)),
        },
        normalized_lineage=normalized_lineage,
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


def _profile_version(profile: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    for value in (
        profile.get("profile_version"),
        profile.get("window_profile_id"),
        profile.get("selected_window_profile_id"),
        profile.get("schema_version"),
        manifest.get("optuna_profile_schema_version"),
        manifest.get("schema_version"),
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
    "ASSET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION",
    "build_asset_state_production_no_write_plan_from_validation",
    "plan_asset_state_production_no_write",
    "plan_default_asset_state_production_no_write",
]
