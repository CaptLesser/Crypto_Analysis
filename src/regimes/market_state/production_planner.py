from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.production_consumer import REGIME_BRANCH_MARKET_STATE
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
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_STATUS_BLOCKED,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    RegimeProductionNoWritePlan,
    RegimeProductionPlanningUnit,
    build_regime_production_planner_contract,
    resolve_regime_production_refit_cadence,
)
from src.regimes.core.production_input_edge import resolve_regime_production_input_edge
from src.regimes.core.production_reuse_cache import (
    RegimeProductionPlannerRunCache,
    build_profile_lookup_index,
    source_tail_fingerprint,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.production_worker_contract import build_regime_production_job_matrix
from src.regimes.market_state.production_consumer import (
    MARKET_STATE_CONSUMER_STATUS_READY_FOR_SANDBOX_DRY_RUN,
    MarketStateProductionConsumerValidation,
    validate_default_market_state_selected_profiles_for_consumption,
    validate_market_state_selected_profiles_for_consumption,
)


MARKET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION = 1


def plan_market_state_production_no_write(
    manifest_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="market_state_production_planner_local")
    validation = validate_market_state_selected_profiles_for_consumption(
        manifest_path,
        sandbox_nonproduction_mode=True,
        run_cache=cache,
    )
    return build_market_state_production_no_write_plan_from_validation(validation, run_cache=cache, env=env)


def plan_default_market_state_production_no_write(
    *,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="market_state_production_planner_local")
    validation = validate_default_market_state_selected_profiles_for_consumption(
        sandbox_nonproduction_mode=True,
        env=env,
        run_cache=cache,
    )
    return build_market_state_production_no_write_plan_from_validation(validation, run_cache=cache, env=env)


def build_market_state_production_no_write_plan_from_validation(
    validation: MarketStateProductionConsumerValidation,
    *,
    run_cache: RegimeProductionPlannerRunCache | None = None,
    env: Mapping[str, str] | None = None,
) -> RegimeProductionNoWritePlan:
    cache = run_cache or RegimeProductionPlannerRunCache(cache_id="market_state_production_planner_local")
    consumer_payload = validation.as_dict()
    contract = build_regime_production_planner_contract(REGIME_BRANCH_MARKET_STATE)
    input_edge = resolve_regime_production_input_edge(REGIME_BRANCH_MARKET_STATE, env=env)
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
    ready = validation.status == MARKET_STATE_CONSUMER_STATUS_READY_FOR_SANDBOX_DRY_RUN
    selected = [dict(item) for item in manifest.get("selected_profiles") or () if isinstance(item, Mapping)] if ready else []
    masked = [dict(item) for item in manifest.get("masked_or_skipped_cells") or () if isinstance(item, Mapping)] if ready else []
    lookup_index = cache.profile_lookup_index(
        branch=REGIME_BRANCH_MARKET_STATE,
        artifact_hash=artifact_hash,
        source_tail_fingerprint=source_tail_fingerprint((*selected, *masked)),
        config_fingerprint={"branch": REGIME_BRANCH_MARKET_STATE, "target_fields": ("market_axis", "band")},
        builder=lambda: build_profile_lookup_index(
            branch=REGIME_BRANCH_MARKET_STATE,
            artifact_hash=artifact_hash,
            target_fields=("market_axis", "band"),
            selected_records=selected,
            unavailable_records=tuple({**item, "market_axis": item.get("axis")} for item in masked),
        ),
    )
    units = tuple(
        [
            _market_state_planning_unit(
                profile,
                manifest=manifest,
                manifest_version=manifest_version,
                artifact_path=validation.manifest_path,
                artifact_hash=artifact_hash,
                contract=contract,
                timestamp_plan=timestamp_plan,
                selected=True,
                run_cache=cache,
            )
            for profile in selected
        ]
        + [
            _market_state_planning_unit(
                mask,
                manifest=manifest,
                manifest_version=manifest_version,
                artifact_path=validation.manifest_path,
                artifact_hash=artifact_hash,
                contract=contract,
                timestamp_plan=timestamp_plan,
                selected=False,
                run_cache=cache,
            )
            for mask in masked
        ]
    )
    selected_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED)
    masked_count = sum(1 for unit in units if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE)
    normalized_lineage_blocked_count = sum(1 for unit in units if not dict(unit.normalized_lineage or {}).get("passed"))
    selected_lineage_blocked_count = sum(
        1
        for unit in units
        if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
        and not dict(unit.normalized_lineage or {}).get("passed")
    )
    job_matrix = build_regime_production_job_matrix(REGIME_BRANCH_MARKET_STATE, units)
    job_matrix_payload = job_matrix.as_dict()
    telemetry = {
        "schema_version": MARKET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION,
        "artifact_kind": "market_state_production_no_write_planner_telemetry",
        "branch": REGIME_BRANCH_MARKET_STATE,
        "planned_unit_count": len(units),
        "selected_unit_count": selected_count,
        "masked_unavailable_unit_count": masked_count,
        "expected_cell_count": int(validation.expected_cell_count),
        "selected_profile_count": int(validation.selected_profile_count),
        "masked_or_skipped_count": int(validation.masked_or_skipped_count),
        "normalized_lineage_blocked_unit_count": normalized_lineage_blocked_count,
        "selected_normalized_lineage_blocked_unit_count": selected_lineage_blocked_count,
        "per_asset_logic_planned": False,
        "composite_market_state_v1_label_planned": False,
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
        branch=REGIME_BRANCH_MARKET_STATE,
        artifact_path=validation.manifest_path,
        profile_artifact_hash=artifact_hash,
        consumer_validation=consumer_payload,
        planner_contract=contract,
        shared_dry_run_plan=consumer_payload.get("dry_run_plan") or {},
        planning_units=units,
        telemetry=telemetry,
        job_matrix=job_matrix_payload,
        safety_status=safety_status,
    )


def _market_state_planning_unit(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_version: Mapping[str, Any],
    artifact_path: Path,
    artifact_hash: str,
    contract,
    timestamp_plan: Mapping[str, Any],
    selected: bool,
    run_cache: RegimeProductionPlannerRunCache | None,
) -> RegimeProductionPlanningUnit:
    target_key = {
        "market_axis": str(row.get("market_axis") or row.get("axis") or ""),
        "band": str(row.get("band") or ""),
    }
    interval = first_present(row, ("source_interval", "interval"))
    unit_cadence = resolve_regime_production_refit_cadence(
        REGIME_BRANCH_MARKET_STATE,
        interval_minutes=int(interval) if interval not in (None, "") else None,
    )
    profile_id = str(row.get("profile_id") or f"{target_key['market_axis']}::{target_key['band']}")
    profile_version = _profile_version(row, manifest)
    coverage = first_mapping(row, ("coverage_summary", "coverage_gate"))
    window = first_mapping(row, ("window_profile",))
    planning_status = (
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
        if selected
        else REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE
    )
    availability_reasons = tuple(
        reason
        for reason in (
            row.get("mask_reason_code"),
            row.get("reason"),
        )
        if reason
    )
    normalized_lineage = normalized_lineage_for_row(
        branch=REGIME_BRANCH_MARKET_STATE,
        row=row,
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
        branch=REGIME_BRANCH_MARKET_STATE,
        target_key=target_key,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_artifact_path=str(artifact_path),
        profile_artifact_hash=artifact_hash,
        refit_window_start=coverage.get("start_ts") or window.get("start_ts"),
        refit_window_end=coverage.get("end_ts") or window.get("end_ts"),
        definition_known_at_ts=row.get("known_at_ts"),
        source_tail_ts=row.get("source_tail_ts") or coverage.get("source_tail_ts"),
        refit_cadence_id=unit_cadence.refit_cadence_id,
        status=REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
        health_metadata=first_mapping(row, ("label_output_health_gate_summary", "coverage_gate")),
        lineage={
            "run_id": row.get("run_id") or manifest.get("run_id"),
            "trial_study_lineage": to_jsonable(dict(row.get("trial_study_lineage") or {})),
            "selection_scope": row.get("selection_scope") or manifest.get("selection_scope"),
            "source_interval": interval,
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
        branch=REGIME_BRANCH_MARKET_STATE,
        target_key=target_key,
        output_grain_key=output_grain_key_for_target(REGIME_BRANCH_MARKET_STATE, target_key, unit_timestamp_plan),
        timestamp_plan=unit_timestamp_plan,
        planning_status=planning_status,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_artifact_path=str(artifact_path),
        profile_artifact_hash=artifact_hash,
        model_state_definition=model_state,
        method_metadata={
            "selected_method_profile": row.get("selected_method_profile"),
            "selected_method_family": row.get("selected_method_family"),
            "selected_feature_pool": row.get("selected_feature_pool"),
            "selected_feature_set": to_jsonable(row.get("selected_feature_set")),
            "preprocessing": row.get("preprocessing"),
            "reducer": row.get("reducer"),
            "tuned_core_parameters": to_jsonable(dict(row.get("tuned_core_parameters") or {})),
            "window_profile_id": row.get("window_profile_id"),
            "window_profile": to_jsonable(dict(window)),
            "refit_cadence": unit_cadence.as_dict(),
            "normalized_clamp_range": to_jsonable(dict(unit_timestamp_plan.get("normalized_clamp_range") or {})),
        },
        health_metadata=first_mapping(row, ("label_output_health_gate_summary", "coverage_gate")),
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


def _profile_version(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    for value in (
        row.get("profile_version"),
        row.get("window_profile_id"),
        row.get("schema_version"),
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
    "MARKET_STATE_PRODUCTION_PLANNER_SCHEMA_VERSION",
    "build_market_state_production_no_write_plan_from_validation",
    "plan_default_market_state_production_no_write",
    "plan_market_state_production_no_write",
]
