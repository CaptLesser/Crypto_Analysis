from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.test_branch_maturity import FILTER_LOW_FEATURE_SPREAD, normalize_test_branch_filter_reason
from src.regimes.cross_asset_state.dataset_builder import (
    CrossAssetFeaturePanelMatrixCache,
    build_relationship_value_availability_index,
    load_relationship_value_availability,
)
from src.regimes.cross_asset_state.economic_diagnostics import build_cross_asset_state_economic_diagnostic_contract
from src.regimes.cross_asset_state.execution_maturity import (
    CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENTS_FILENAME,
    aggregate_runtime_telemetry_events,
    family_task_worker_id,
    write_progress_heartbeat,
    write_runtime_telemetry_event,
)
from src.regimes.cross_asset_state.feature_families import (
    CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
    CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
    CrossAssetStateFeatureFamilySpec,
    resolve_feature_families,
)
from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason
from src.regimes.cross_asset_state.method_universe import (
    CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
    CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
    cross_asset_state_method_universe_manifest,
)
from src.regimes.cross_asset_state.production_consumer import (
    CrossAssetStateProductionGateError,
    validate_cross_asset_state_manifest_for_production,
)
from src.regimes.cross_asset_state.profile_candidates import (
    CrossAssetStateCandidateDefinitionCache,
    candidate_definition_shape,
    choose_best_candidate,
    diagnostic_only_result,
    run_profile_candidates,
)
from src.regimes.cross_asset_state.profile_manifest import validate_cross_asset_state_selected_profile_manifest
from src.regimes.cross_asset_state.relationship_context import CrossAssetRelationshipContextResolver
from src.regimes.cross_asset_state.scoring_contract import ECONOMIC_DIAGNOSTIC_PENDING, SCORING_SCHEMA_ID, SCORING_SCHEMA_VERSION
from src.regimes.cross_asset_state.search_spaces import CROSS_ASSET_STATE_SEARCH_SPACE_ID
from src.regimes.cross_asset_state.test_prototype import (
    PROTOTYPE_PROFILE_TYPES,
    build_cross_asset_state_valid_target_universe,
    select_cross_asset_state_sample,
    _assert_no_peer_identity_leak,
    _candidate_profile_universe,
    _counts,
    _dominant_or_tiny_issue_count,
    _economic_diagnostic_readiness,
    _health_failure_diagnosis,
    _health_failure_rows,
    _human_inspection_summary,
    _inspection_row,
    _load_feature_frames,
    _load_handoff,
    _manual_sample_plan_rows,
    _masked_cell,
    _maturity_contracts_implemented,
    _output_health_gate_calibration,
    _profile_record_from_masked_cell,
    _profile_record_from_selected_cell,
    _resolution_ts,
    _rss_bytes,
    _sample_selection_rows,
    _scoring_schema_summary,
    _selected_cell,
    _target_universe_policy,
    _valid_assets_across_bands,
    _valid_target_counts_by_band,
    _warning_reason_counts,
    _write_csv,
)
from src.regimes.cross_asset_state.window_profiles import (
    apply_cross_asset_state_window_policy,
    cross_asset_state_window_ladder_policies,
    cross_asset_state_window_policy_manifest,
    family_window_sensitivity_summary,
)


DEFAULT_MASKED_CONTROL_ASSETS: tuple[str, ...] = ("MISSING_ASSET_USD",)
_SETUP_CACHE_LOCK = threading.Lock()
_SETUP_CACHE: dict[tuple[str, int, int, int], tuple[Any, CrossAssetRelationshipContextResolver]] = {}


@dataclass(frozen=True)
class CrossAssetStateMiniTestConfig:
    handoff_path: str | Path
    output_root: str | Path
    artifact_label: str = "v1_mini_test"
    summary_filename: str = "cross_asset_state_v1_mini_test_summary.json"
    selected_profiles_filename: str = "cross_asset_state_selected_profiles.mini_test.nonprod.json"
    cells_filename: str = "cross_asset_state_mini_test_cells.csv"
    family_summary_filename: str = "cross_asset_state_mini_test_family_summary.csv"
    inspection_examples_filename: str = "cross_asset_state_mini_test_inspection_examples.csv"
    window_comparison_filename: str = "cross_asset_state_mini_test_window_comparison.csv"
    profile_type_comparison_filename: str = "cross_asset_state_mini_test_profile_type_comparison.csv"
    health_failures_filename: str = "cross_asset_state_mini_test_health_failures.csv"
    runtime_filename: str = "cross_asset_state_mini_test_runtime.csv"
    bands: tuple[str, ...] = ("meso", "macro")
    max_valid_assets: int = 40
    min_valid_assets_for_meaningful_sample: int = 40
    max_masked_controls_from_handoff: int = 0
    masked_control_assets: tuple[str, ...] = DEFAULT_MASKED_CONTROL_ASSETS
    include_masked_probe_asset: bool = True
    min_rows_per_cell: int = 8
    max_parquet_files_per_asset: int = 4
    max_inspection_examples_per_family: int = 10
    allow_stale_sandbox_artifacts: bool = False
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL
    family_names: tuple[str, ...] = ()
    sample_assets: tuple[str, ...] = ()
    family_column_overrides: Mapping[str, tuple[str, ...]] | None = None
    progress_root: str | Path | None = None
    progress_run_id: str | None = None
    progress_shard_id: str | None = None
    progress_family: str | None = None
    progress_worker_id: str | None = None
    progress_flush_cell_interval: int = 1


REPAIRED_VARIABLE_PEER_DEFAULT_SUPPORT_DEFINITION_ID = "dynamic_variable_peer_fallback_v1"
SUPPORT_METADATA_COLUMNS: tuple[str, ...] = (
    "support_definition_id",
    "support_size",
    "support_quality",
    "support_rank_max",
    "support_threshold",
    "support_fallback_path",
    "repaired_feature_manifest_id",
)


def _with_feature_set(cell: Mapping[str, Any], feature_set_version: str, frame: Any | None = None) -> dict[str, Any]:
    out = dict(cell)
    out["feature_set_version"] = str(feature_set_version)
    metadata = _feature_support_metadata(frame)
    default_support_definition_id = "original_fixed_top3_v1"
    if feature_set_version == CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX:
        default_support_definition_id = REPAIRED_VARIABLE_PEER_DEFAULT_SUPPORT_DEFINITION_ID
    out["support_definition_id"] = (
        metadata.get("support_definition_id") or out.get("support_definition_id") or default_support_definition_id
    )
    out["repaired_feature_manifest_id"] = metadata.get("repaired_feature_manifest_id") or out.get("repaired_feature_manifest_id")
    for column in ("support_size", "support_quality", "support_rank_max", "support_threshold", "support_fallback_path"):
        if column in metadata or column in out:
            out[column] = metadata.get(column, out.get(column))
    return out


def _feature_support_metadata(frame: Any | None) -> dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        return {}
    out: dict[str, Any] = {}
    latest = frame
    if "ts" in latest.columns:
        latest = latest.sort_values("ts")
    row = latest.iloc[-1] if len(latest) else None
    if row is None:
        return out
    for column in SUPPORT_METADATA_COLUMNS:
        if column not in frame.columns:
            continue
        value = row.get(column)
        if value is None:
            continue
        try:
            if value != value:
                continue
        except Exception:
            pass
        if column in {"support_size", "support_rank_max"}:
            try:
                out[column] = int(value)
            except Exception:
                continue
        elif column == "support_threshold":
            try:
                out[column] = float(value)
            except Exception:
                continue
        else:
            text = str(value).strip()
            if text:
                out[column] = text
    return out


def _apply_family_column_overrides(
    families: Sequence[CrossAssetStateFeatureFamilySpec],
    overrides: Mapping[str, tuple[str, ...]] | None,
) -> tuple[CrossAssetStateFeatureFamilySpec, ...]:
    if not overrides:
        return tuple(families)
    normalized = {
        str(family): tuple(str(column).strip() for column in columns if str(column).strip())
        for family, columns in dict(overrides).items()
    }
    known = {family.name for family in families}
    missing = sorted(set(normalized) - known)
    if missing:
        raise ValueError(f"Unsupported Cross-Asset-State family_column_overrides: {missing}")
    out: list[CrossAssetStateFeatureFamilySpec] = []
    for family in families:
        columns = normalized.get(family.name)
        if columns is None:
            out.append(family)
            continue
        out.append(
            CrossAssetStateFeatureFamilySpec(
                name=family.name,
                required_columns=columns,
                method_family=family.method_family,
                model_facing_v1=family.model_facing_v1,
                feature_set_version=family.feature_set_version,
            )
        )
    return tuple(out)


def _load_handoff_and_resolver_cached(
    handoff_path: str | Path,
    *,
    max_parquet_files_per_root: int,
) -> tuple[Any, CrossAssetRelationshipContextResolver, dict[str, Any]]:
    path = Path(handoff_path)
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size), int(max_parquet_files_per_root))
    with _SETUP_CACHE_LOCK:
        cached = _SETUP_CACHE.get(key)
        if cached is not None:
            return (
                cached[0],
                cached[1],
                {
                    "artifact_kind": "cross_asset_state_setup_cache_telemetry",
                    "cache_scope": "per_process_in_memory",
                    "cache_hit": True,
                    "cache_miss": False,
                    "cache_key_path": str(resolved),
                    "cache_key_mtime_ns": int(stat.st_mtime_ns),
                    "cache_key_size": int(stat.st_size),
                    "cache_size": len(_SETUP_CACHE),
                    "production_write_allowed": False,
                },
            )
        handoff = _load_handoff(path)
        resolver = CrossAssetRelationshipContextResolver.from_handoff(handoff, max_parquet_files_per_root=max_parquet_files_per_root)
        _SETUP_CACHE[key] = (handoff, resolver)
        return (
            handoff,
            resolver,
            {
                "artifact_kind": "cross_asset_state_setup_cache_telemetry",
                "cache_scope": "per_process_in_memory",
                "cache_hit": False,
                "cache_miss": True,
                "cache_key_path": str(resolved),
                "cache_key_mtime_ns": int(stat.st_mtime_ns),
                "cache_key_size": int(stat.st_size),
                "cache_size": len(_SETUP_CACHE),
                "production_write_allowed": False,
            },
        )


def run_cross_asset_state_v1_mini_test(config: CrossAssetStateMiniTestConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase_started = started
    setup_phase_timings: list[dict[str, Any]] = []

    def mark_setup_phase(name: str, **extra: Any) -> None:
        nonlocal phase_started
        now = time.perf_counter()
        row = {
            "phase": str(name),
            "elapsed_s": round(max(0.0, now - phase_started), 6),
            "since_run_start_s": round(max(0.0, now - started), 6),
        }
        row.update(extra)
        setup_phase_timings.append(row)
        phase_started = now

    handoff, resolver, setup_cache_telemetry = _load_handoff_and_resolver_cached(
        config.handoff_path,
        max_parquet_files_per_root=250,
    )
    mark_setup_phase(
        "load_handoff_and_relationship_context_resolver",
        setup_cache_hit=bool(setup_cache_telemetry["cache_hit"]),
        setup_cache_miss=bool(setup_cache_telemetry["cache_miss"]),
        availability_rows=int(len(resolver.availability_frame)) if resolver.availability_frame is not None else 0,
        availability_index_built=resolver.availability_index is not None,
    )
    families = resolve_feature_families(feature_set_version=config.feature_set_version)
    if config.family_names:
        wanted = {str(name) for name in config.family_names}
        families = tuple(family for family in families if family.name in wanted)
        missing = sorted(wanted - {family.name for family in families})
        if missing:
            raise ValueError(f"Unsupported Cross-Asset-State mini-test family_names: {missing}")
    families = _apply_family_column_overrides(families, config.family_column_overrides)
    mark_setup_phase("resolve_feature_families", family_count=len(families))
    availability = resolver.availability_frame
    availability_index = resolver.availability_index or build_relationship_value_availability_index(availability)
    mark_setup_phase(
        "reuse_or_build_availability_index",
        reused_resolver_availability_index=resolver.availability_index is availability_index,
        availability_index_built=availability_index is not None,
    )
    target_universe = build_cross_asset_state_valid_target_universe(
        handoff,
        resolver,
        availability,
        bands=config.bands,
        families=families,
        target_assets=config.sample_assets if config.sample_assets else None,
    )
    target_universe_scope = "explicit_sample_assets" if config.sample_assets else "full_handoff_feature_universe"
    mark_setup_phase(
        "build_target_universe",
        target_universe_rows=len(target_universe),
        target_universe_scope=target_universe_scope,
        scoped_target_asset_count=len(tuple(config.sample_assets or ())),
    )
    valid_target_assets_across_bands = _valid_assets_across_bands(target_universe, bands=config.bands)
    active_surface = _active_run_surface(
        config=config,
        handoff=handoff,
        availability=availability,
        target_universe=target_universe,
        valid_target_assets_across_bands=valid_target_assets_across_bands,
    )
    if not active_surface["safe_to_run"]:
        raise ValueError("Cross-Asset-State mini-test active run surface is not safe: " + "|".join(active_surface["blockers"]))
    mark_setup_phase("validate_active_surface", safe_to_run=bool(active_surface["safe_to_run"]))

    sample_plan = (
        _manual_sample_plan_rows(
            config.sample_assets,
            target_universe=target_universe,
            include_masked_probe_asset=config.include_masked_probe_asset,
        )
        if config.sample_assets
        else select_cross_asset_state_sample(
            target_universe,
            bands=config.bands,
            max_valid_assets=config.max_valid_assets,
            max_masked_controls=config.max_masked_controls_from_handoff,
            masked_control_assets=config.masked_control_assets,
            include_masked_probe_asset=config.include_masked_probe_asset,
        )
    )
    mark_setup_phase("build_sample_plan", sample_plan_rows=len(sample_plan))
    sample_assets = tuple(dict.fromkeys(str(row["asset_id"]) for row in sample_plan if str(row.get("asset_id", "")).strip()))
    valid_sample_assets = tuple(row["asset_id"] for row in sample_plan if row.get("sample_kind") == "valid_target")
    if len(valid_sample_assets) != int(config.max_valid_assets):
        raise ValueError(
            "Cross-Asset-State mini-test expected "
            f"{config.max_valid_assets} valid targets but selected {len(valid_sample_assets)}"
        )

    feature_frames = _load_feature_frames(handoff.feature_roots, sample_assets, config.bands, max_files_per_asset=config.max_parquet_files_per_asset)
    mark_setup_phase("load_feature_frames", feature_frame_count=len(feature_frames), sample_asset_count=len(sample_assets))
    _assert_no_peer_identity_leak(feature_frames)
    feature_panel_matrix_cache = CrossAssetFeaturePanelMatrixCache()
    candidate_definition_cache = CrossAssetStateCandidateDefinitionCache()
    mark_setup_phase("initialize_execution_caches")

    cells: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    window_comparisons: list[dict[str, Any]] = []
    selected_profiles: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    inspection_counts: dict[str, int] = {}
    label_rows: list[dict[str, Any]] = []
    expected_cells = len(sample_assets) * len(config.bands) * len(families)
    progress_root = Path(config.progress_root) if config.progress_root is not None else None
    progress_family = str(config.progress_family or (families[0].name if len(families) == 1 else "multi_family"))
    progress_worker_id = config.progress_worker_id or family_task_worker_id()
    runtime_telemetry_root = (progress_root if progress_root is not None else output_root) / "runtime_telemetry"
    runtime_telemetry_run_id = str(config.progress_run_id or config.artifact_label)
    runtime_telemetry_shard_id = str(config.progress_shard_id or config.progress_run_id or config.artifact_label)
    runtime_telemetry_event_paths: list[str] = []

    def emit_runtime_telemetry(
        *,
        telemetry_level: str,
        status: str,
        family_name: str | None = None,
        band_name: str | None = None,
        window_policy_id: str | None = None,
        asset_id: str | None = None,
        cell_id: str | None = None,
        candidate_eval_count: int = 0,
        method_family: str | None = None,
        profile_type: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        cache_counts = _runtime_cache_counts(feature_panel_matrix_cache.stats(), candidate_definition_cache.stats())
        result = write_runtime_telemetry_event(
            runtime_telemetry_root,
            run_id=runtime_telemetry_run_id,
            shard_id=runtime_telemetry_shard_id,
            worker_id=progress_worker_id,
            telemetry_level=telemetry_level,
            status=status,
            family=family_name or progress_family,
            band=band_name,
            window_policy_id=window_policy_id,
            asset_id=asset_id,
            cell_id=cell_id,
            candidate_eval_count=candidate_eval_count,
            method_family=method_family,
            profile_type=profile_type,
            elapsed_s=time.perf_counter() - started,
            cache_hit_count=cache_counts["cache_hit_count"],
            cache_miss_count=cache_counts["cache_miss_count"],
            cache_stats=cache_counts,
            extra=extra,
        )
        runtime_telemetry_event_paths.append(str(result["telemetry_events_path"]))

    emit_runtime_telemetry(
        telemetry_level="run",
        status="running",
        candidate_eval_count=0,
        extra={"stage": "mini_test_started", "expected_cells": expected_cells},
    )
    if progress_root is not None:
        write_progress_heartbeat(
            progress_root,
            run_id=config.progress_run_id,
            family=progress_family,
            shard_status="running",
            expected_cells=expected_cells,
            completed_cells=0,
            candidate_evaluations=0,
            worker_id=progress_worker_id,
            extra={"stage": "mini_test_started"},
        )

    def flush_cell_progress(*, asset_id: str, band_name: str, family_name: str, window_policy_id: str | None) -> None:
        if progress_root is None or not _should_flush_progress(len(cells), config.progress_flush_cell_interval):
            pass
        else:
            write_progress_heartbeat(
                progress_root,
                run_id=config.progress_run_id,
                family=progress_family,
                shard_status="running",
                band=band_name,
                window_policy_id=window_policy_id,
                asset_id=asset_id,
                expected_cells=expected_cells,
                completed_cells=len(cells),
                candidate_evaluations=len(comparisons),
                worker_id=progress_worker_id,
                extra={"stage": "cell_completed", "relationship_feature_family": family_name},
            )
        cell = cells[-1] if cells else {}
        emit_runtime_telemetry(
            telemetry_level="cell",
            status=str(cell.get("cell_status") or cell.get("profile_selection_status") or "cell_completed"),
            family_name=family_name,
            band_name=band_name,
            window_policy_id=window_policy_id,
            asset_id=asset_id,
            candidate_eval_count=len(comparisons),
            method_family=cell.get("method_family"),
            profile_type=cell.get("selected_profile_type") or cell.get("profile_type"),
            extra={"completed_cells": len(cells), "expected_cells": expected_cells},
        )

    def emit_evaluation_telemetry(*, asset_id: str, band_name: str, family_name: str, evaluated: Mapping[str, Any]) -> None:
        for row in evaluated.get("window_comparisons") or ():
            if not isinstance(row, Mapping):
                continue
            emit_runtime_telemetry(
                telemetry_level="window",
                status=str(row.get("candidate_status") or row.get("window_status") or "window_evaluated"),
                family_name=family_name,
                band_name=band_name,
                window_policy_id=row.get("window_policy_id"),
                asset_id=asset_id,
                candidate_eval_count=int(row.get("evaluated_profile_count") or 0),
                profile_type=row.get("best_profile_type"),
                extra={
                    "window_candidate_name": row.get("window_candidate_name"),
                    "window_passed": row.get("window_passed"),
                },
            )
        for row in evaluated.get("profile_comparisons") or ():
            if not isinstance(row, Mapping):
                continue
            emit_runtime_telemetry(
                telemetry_level="method_profile_type",
                status=str(row.get("candidate_status") or row.get("output_health_status") or "candidate_evaluated"),
                family_name=family_name,
                band_name=band_name,
                window_policy_id=row.get("window_policy_id"),
                asset_id=asset_id,
                candidate_eval_count=1,
                method_family=row.get("method_family"),
                profile_type=row.get("profile_type"),
                extra={
                    "candidate_id": row.get("candidate_id"),
                    "parameter_grid_id": row.get("parameter_grid_id"),
                },
            )

    for asset in sample_assets:
        for band in config.bands:
            ts = _resolution_ts(feature_frames.get((asset, band)))
            for family in families:
                context = resolver.resolve(
                    asset_id=asset,
                    band=band,
                    ts=ts,
                    feature_families=(family.name,),
                    feature_family_columns={family.name: family.required_columns},
                ).as_dict()
                current_policy = cross_asset_state_window_ladder_policies(
                    relationship_feature_family=family.name,
                    band=band,
                )[0]
                frame = feature_frames.get((asset, band))
                if context["context_status"] != "available" or frame is None:
                    reason = context["mask_reason"] or CrossAssetStateMaskReason.MISSING_REQUIRED_FAMILY_FIELDS
                    cell = _masked_cell(asset, band, family.name, context, reason, window_policy=current_policy)
                    cell = _with_feature_set(cell, config.feature_set_version, frame)
                    cells.append(cell)
                    selected_profiles.append(_mini_profile_record(_profile_record_from_masked_cell(cell, handoff)))
                    flush_cell_progress(
                        asset_id=asset,
                        band_name=band,
                        family_name=family.name,
                        window_policy_id=current_policy.window_policy_id,
                    )
                    continue
                frame, availability_reason = feature_panel_matrix_cache.filtered_feature_panel(
                    frame,
                    availability_index or availability,
                    asset=asset,
                    band=band,
                    family=family.name,
                    required_columns=family.required_columns,
                    window_policy_id=current_policy.window_policy_id,
                    feature_set_version=config.feature_set_version,
                    relationship_context_id=context.get("relationship_context_id") or handoff.relationship_context_id,
                    relationship_snapshot_id=context.get("relationship_snapshot_id"),
                    known_at_ts=context.get("known_at_ts"),
                    source_tail_ts=context.get("source_tail_ts"),
                )
                if frame is None or len(frame) < config.min_rows_per_cell:
                    cell = _masked_cell(
                        asset,
                        band,
                        family.name,
                        context,
                        availability_reason or CrossAssetStateMaskReason.INSUFFICIENT_ROWS,
                        window_policy=current_policy,
                    )
                    cell = _with_feature_set(cell, config.feature_set_version, frame)
                    cells.append(cell)
                    selected_profiles.append(_mini_profile_record(_profile_record_from_masked_cell(cell, handoff)))
                    flush_cell_progress(
                        asset_id=asset,
                        band_name=band,
                        family_name=family.name,
                        window_policy_id=current_policy.window_policy_id,
                    )
                    continue

                evaluated = _evaluate_cell_windows(
                    asset=asset,
                    band=band,
                    family_name=family.name,
                    feature_columns=family.required_columns,
                    context=context,
                    frame=frame,
                    min_rows=config.min_rows_per_cell,
                    feature_set_version=config.feature_set_version,
                    feature_panel_matrix_cache=feature_panel_matrix_cache,
                    candidate_definition_cache=candidate_definition_cache,
                )
                emit_evaluation_telemetry(asset_id=asset, band_name=band, family_name=family.name, evaluated=evaluated)
                comparisons.extend(evaluated["profile_comparisons"])
                window_comparisons.extend(evaluated["window_comparisons"])
                best_candidate = evaluated["best_candidate"]
                best_frame = evaluated["best_frame"]
                best_policy = evaluated["best_policy"]
                best_coverage = evaluated["best_coverage"]
                if best_candidate is None or best_frame is None or best_policy is None or best_coverage is None:
                    mask_reason = _mask_reason_from_profile_comparisons(evaluated["profile_comparisons"])
                    cell = _masked_cell(
                        asset,
                        band,
                        family.name,
                        context,
                        mask_reason,
                        window_policy=current_policy,
                    )
                    cell = _with_feature_set(cell, config.feature_set_version, frame)
                    cells.append(cell)
                    selected_profiles.append(_mini_profile_record(_profile_record_from_masked_cell(cell, handoff)))
                    flush_cell_progress(
                        asset_id=asset,
                        band_name=band,
                        family_name=family.name,
                        window_policy_id=current_policy.window_policy_id,
                    )
                    continue
                if not best_candidate.output_health.get("passed") or not getattr(best_candidate, "selection_eligible", True):
                    best_candidate = diagnostic_only_result(
                        best_frame,
                        family=family.name,
                        feature_columns=family.required_columns,
                        reason="no_candidate_passed_output_health",
                    )
                for label, count in best_candidate.label_counts.items():
                    label_rows.append(
                        {
                            "asset_id": asset,
                            "band": band,
                            "relationship_feature_family": family.name,
                            "window_policy_id": best_policy.window_policy_id,
                            "window_candidate_name": best_policy.window_candidate_name,
                            "profile_type": best_candidate.profile_type,
                            "label": label,
                            "count": count,
                        }
                    )
                cell = _selected_cell(
                    asset,
                    band,
                    family.name,
                    context,
                    best_candidate,
                    len(best_frame),
                    window_policy=best_policy,
                    window_coverage=best_coverage.as_dict(),
                )
                cell = _with_feature_set(cell, config.feature_set_version, best_frame)
                cells.append(cell)
                selected_profiles.append(_mini_profile_record(_profile_record_from_selected_cell(cell, family.required_columns, handoff)))
                if inspection_counts.get(family.name, 0) < int(config.max_inspection_examples_per_family) and cell["cell_status"] == "selected_coherent":
                    inspection_rows.append(
                        _inspection_row(
                            asset,
                            band,
                            family.name,
                            context,
                            best_candidate,
                            best_frame,
                            window_policy=best_policy,
                            window_coverage=best_coverage.as_dict(),
                        )
                    )
                    inspection_counts[family.name] = inspection_counts.get(family.name, 0) + 1
                flush_cell_progress(
                    asset_id=asset,
                    band_name=band,
                    family_name=family.name,
                    window_policy_id=str(best_policy.window_policy_id) if best_policy is not None else current_policy.window_policy_id,
                )

    family_summary = _mini_family_summary(cells, comparisons, window_comparisons, inspection_rows)
    health_failures = _health_failure_rows(comparisons)
    runtime_rows = _mini_runtime_rows(cells, comparisons, window_comparisons, started)
    economic_contract = build_cross_asset_state_economic_diagnostic_contract(handoff)
    sample_selection = _sample_selection_rows(sample_assets, valid_sample_assets, cells, feature_frames, sample_plan)

    cells_path = output_root / config.cells_filename
    family_path = output_root / config.family_summary_filename
    inspection_path = output_root / config.inspection_examples_filename
    window_comparison_path = output_root / config.window_comparison_filename
    profile_comparison_path = output_root / config.profile_type_comparison_filename
    health_failures_path = output_root / config.health_failures_filename
    runtime_path = output_root / config.runtime_filename
    selected_profiles_path = output_root / config.selected_profiles_filename
    summary_path = output_root / config.summary_filename
    labels_path = output_root / "cross_asset_state_mini_test_label_distributions.csv"
    sample_selection_path = output_root / "cross_asset_state_mini_test_sample_selection.csv"
    candidate_pool_path = output_root / "cross_asset_state_mini_test_candidate_pool.csv"

    _write_csv(cells_path, cells)
    _write_csv(family_path, family_summary)
    _write_csv(inspection_path, inspection_rows)
    _write_csv(window_comparison_path, window_comparisons)
    _write_csv(profile_comparison_path, comparisons)
    _write_csv(health_failures_path, health_failures)
    _write_csv(runtime_path, runtime_rows)
    _write_csv(labels_path, label_rows)
    _write_csv(sample_selection_path, sample_selection)
    _write_csv(candidate_pool_path, target_universe)

    selected_records = [
        profile
        for profile in selected_profiles
        if profile.get("selected_profile_type") is not None and profile.get("output_health_status") != "masked_unavailable"
    ]
    masked_records = [
        profile
        for profile in selected_profiles
        if profile.get("selected_profile_type") is None or profile.get("output_health_status") == "masked_unavailable"
    ]
    expected_cell_records = [
        {
            "asset_id": cell["asset_id"],
            "relationship_feature_family": cell["relationship_feature_family"],
            "band": cell["band"],
        }
        for cell in cells
    ]
    selected_payload = {
        "artifact_kind": "cross_asset_state_selected_profiles_mature_nonprod",
        "artifact_label": config.artifact_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set_version": config.feature_set_version,
        "support_definition_ids": sorted(
            {
                str(profile.get("support_definition_id"))
                for profile in selected_profiles
                if profile.get("support_definition_id")
            }
        ),
        "support_qualities": sorted(
            {
                str(profile.get("support_quality"))
                for profile in selected_profiles
                if profile.get("support_quality")
            }
        ),
        "support_sizes": sorted(
            {
                int(profile.get("support_size"))
                for profile in selected_profiles
                if profile.get("support_size") is not None
            }
        ),
        "repaired_feature_manifest_ids": sorted(
            {
                str(profile.get("repaired_feature_manifest_id"))
                for profile in selected_profiles
                if profile.get("repaired_feature_manifest_id")
            }
        ),
        "grain": "asset_id x relationship_feature_family x band",
        "expected_cell_count": len(cells),
        "selected_profile_count": len(selected_records),
        "masked_or_skipped_cell_count": len(masked_records),
        "single_active_nonproduction_handoff_artifact": config.selected_profiles_filename,
        "active_relationship_context_handoff_artifact": str(config.handoff_path),
        "stale_sandbox_manifest_used": False,
        "relationship_context_id": handoff.relationship_context_id,
        "relationship_context_handoff_path": str(config.handoff_path),
        "relationship_context_cadence_policy": handoff.cadence_policy_as_dict(),
        "scoring_schema_id": SCORING_SCHEMA_ID,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "window_policy_manifest": cross_asset_state_window_policy_manifest(),
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
        "candidate_definition_cache_telemetry": candidate_definition_cache.stats(),
        "economic_diagnostic_contract": economic_contract.as_dict(),
        "source_lineage": {
            "relationship_context_id": handoff.relationship_context_id,
            "regime_feature_manifest_id": handoff.regime_feature_manifest_id,
            "relationship_snapshot_roots": list(handoff.relationship_snapshot_roots),
            "feature_roots": list(handoff.feature_roots),
            "availability_sidecar_refs": list(handoff.availability_sidecar_refs),
            "relationship_context_cadence_policy": handoff.cadence_policy_as_dict(),
            "stale_sandbox_manifest_used": False,
        },
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
        "peer_groups_model_facing": False,
        "selected_profiles": selected_records,
        "masked_or_skipped_cells": masked_records,
        "profiles": selected_profiles,
    }
    selected_payload["selected_profile_manifest_validation"] = validate_cross_asset_state_selected_profile_manifest(
        selected_payload,
        active_filename=config.selected_profiles_filename,
        expected_cells=expected_cell_records,
    )
    selected_profiles_path.write_text(json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8")

    selected_count = sum(1 for cell in cells if cell["cell_status"] == "selected_coherent")
    masked_count = sum(1 for cell in cells if cell["cell_status"] == "masked_unavailable")
    diagnostic_only_count = sum(1 for cell in cells if cell["cell_status"] == "diagnostic_only")
    missing_cells = len(sample_assets) * len(config.bands) * len(families) - len(cells)
    best_window_patterns = _best_window_policy_patterns(cells, window_comparisons)
    family_readiness = {str(row["relationship_feature_family"]): str(row["family_readiness"]) for row in family_summary}
    emit_runtime_telemetry(
        telemetry_level="run",
        status="complete",
        candidate_eval_count=len(comparisons),
        extra={"stage": "mini_test_complete", "completed_cells": len(cells), "expected_cells": expected_cells},
    )
    runtime_telemetry_events_path = str(runtime_telemetry_root / CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENTS_FILENAME)
    runtime_telemetry_aggregation = aggregate_runtime_telemetry_events(
        runtime_telemetry_event_paths or (runtime_telemetry_events_path,),
        final_summary=True,
    )
    summary = {
        "artifact_kind": "cross_asset_state_v1_mini_test_summary",
        "artifact_label": config.artifact_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set_version": config.feature_set_version,
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "support_definition_ids": selected_payload["support_definition_ids"],
        "support_qualities": selected_payload["support_qualities"],
        "support_sizes": selected_payload["support_sizes"],
        "repaired_feature_manifest_ids": selected_payload["repaired_feature_manifest_ids"],
        "relationship_context_handoff_path": str(config.handoff_path),
        "relationship_context_id": handoff.relationship_context_id,
        "relationship_context_cadence_policy": handoff.cadence_policy_as_dict(),
        "regime_feature_manifest_id": handoff.regime_feature_manifest_id,
        "active_run_surface": active_surface,
        "sample_size": {
            "valid_real_target_assets": len(valid_sample_assets),
            "masked_control_assets": len(sample_assets) - len(valid_sample_assets),
            "sample_assets_including_controls": len(sample_assets),
            "bands": list(config.bands),
            "families": [family.name for family in families],
        },
        "sample_assets": list(sample_assets),
        "valid_target_asset_counts_by_band": _valid_target_counts_by_band(target_universe),
        "valid_target_assets_across_bands_count": len(valid_target_assets_across_bands),
        "target_universe_scope": target_universe_scope,
        "target_universe_scoped_to_explicit_sample_assets": bool(config.sample_assets),
        "grain": "asset_id x relationship_feature_family x band",
        "expected_cells": len(sample_assets) * len(config.bands) * len(families),
        "selected_cells": selected_count,
        "diagnostic_only_cells": diagnostic_only_count,
        "masked_unavailable_cells": masked_count,
        "missing_cells": missing_cells,
        "total_candidate_evaluations": len(comparisons),
        "window_candidate_evaluations": len(window_comparisons),
        "profile_types_tested": list(PROTOTYPE_PROFILE_TYPES),
        "family_required_columns": {family.name: list(family.required_columns) for family in families},
        "family_column_overrides_applied": bool(config.family_column_overrides),
        "window_policies_tested": _window_policy_rows_for_summary(families, config.bands),
        "best_window_policy_patterns": best_window_patterns,
        "best_profile_type_by_family": {row["relationship_feature_family"]: row["best_profile_type"] for row in family_summary},
        "family_readiness": family_readiness,
        "specific_family_answers": _specific_family_answers(cells, comparisons, window_comparisons, inspection_rows, family_summary),
        "economic_diagnostic_status": economic_contract.economic_diagnostic_status,
        "economic_diagnostic_contract": economic_contract.as_dict(),
        "economic_diagnostic_readiness": _economic_diagnostic_readiness(handoff),
        "major_failure_modes": _major_failure_modes(comparisons, window_comparisons),
        "human_inspection": _human_inspection_summary(inspection_rows),
        "runtime_telemetry": {
            **_mini_runtime_summary(runtime_rows),
            "runtime_telemetry_events_path": runtime_telemetry_events_path,
            "runtime_telemetry_aggregation": runtime_telemetry_aggregation,
            "setup_phase_timings": list(setup_phase_timings),
            "setup_elapsed_s": round(sum(float(row.get("elapsed_s") or 0.0) for row in setup_phase_timings), 6),
        },
        "runtime_telemetry_aggregation": runtime_telemetry_aggregation,
        "execution_cache_telemetry": {
            "setup_phase_timings": list(setup_phase_timings),
            "setup_elapsed_s": round(sum(float(row.get("elapsed_s") or 0.0) for row in setup_phase_timings), 6),
            "setup_cache": dict(setup_cache_telemetry),
            "feature_panel_matrix_cache": feature_panel_matrix_cache.stats(),
            "candidate_definition_cache": candidate_definition_cache.stats(),
            "availability_index": availability_index.stats()
            if availability_index is not None
            else {
                "artifact_kind": "cross_asset_state_availability_index_telemetry",
                "index_built": False,
                "build_count": 0,
                "build_seconds": 0.0,
                "rows_indexed": 0,
                "lookup_count": 0,
                "miss_count": 0,
                "logical_key_fields": [],
                "physical_key_columns": [],
                "optional_scope_fields": [],
            },
            "relationship_context_availability_index": resolver.availability_index_stats(),
        },
        "scale_estimate": _mini_scale_estimate(
            cells=cells,
            comparisons=comparisons,
            window_comparisons=window_comparisons,
            valid_target_assets=len(valid_target_assets_across_bands),
            elapsed_seconds=time.perf_counter() - started,
        ),
        "paths": {
            "summary_path": str(summary_path),
            "selected_profiles_path": str(selected_profiles_path),
            "cells_path": str(cells_path),
            "family_summary_path": str(family_path),
            "inspection_examples_path": str(inspection_path),
            "window_comparison_path": str(window_comparison_path),
            "profile_type_comparison_path": str(profile_comparison_path),
            "health_failures_path": str(health_failures_path),
            "runtime_path": str(runtime_path),
            "runtime_telemetry_events_path": runtime_telemetry_events_path,
            "label_distributions_path": str(labels_path),
            "sample_selection_path": str(sample_selection_path),
            "candidate_pool_path": str(candidate_pool_path),
        },
        "mask_reason_counts": _counts([str(cell.get("mask_reason")) for cell in cells if cell["cell_status"] == "masked_unavailable"]),
        "health_failure_diagnosis": _health_failure_diagnosis(comparisons),
        "output_health_gate_calibration": _output_health_gate_calibration(),
        "scoring_schema": _scoring_schema_summary(),
        "maturity_contracts_implemented": _maturity_contracts_implemented(),
        "selected_profile_manifest_validation": selected_payload["selected_profile_manifest_validation"],
        "candidate_profile_universe": _candidate_profile_universe(),
        "window_policy_manifest": cross_asset_state_window_policy_manifest(),
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
        "window_sensitivity_summary": family_window_sensitivity_summary(),
        "target_universe_policy": _target_universe_policy(),
        "final_verdict": _mini_final_verdict(family_readiness, best_window_patterns, economic_contract.as_dict()),
        "exact_next_recommended_sprint": _mini_next_recommended_sprint(family_readiness, best_window_patterns, economic_contract.as_dict()),
        "parent_finalizer_artifact_writing": "single_summary_json_under_configured_output_root",
        "stale_sandbox_artifact_resolution_default": "blocked" if not config.allow_stale_sandbox_artifacts else "allowed_by_explicit_config",
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "safety_confirmation": {
            "production_writes": False,
            "production_labels": False,
            "canonical_production_state_outputs": False,
            "final_production_promotion": False,
            "full_universe_cross_asset_state_campaign": False,
            "broad_pairwise_all_to_all_relationship_discovery": False,
            "canonical_parquet_root_rewrite": False,
            "cleanup_quarantine_delete_actions": False,
            "hardcoded_local_paths_introduced": False,
            "production_writer_gates_remained_fail_closed": True,
            "peer_edge_group_metadata_model_facing": False,
        },
    }
    if progress_root is not None:
        write_progress_heartbeat(
            progress_root,
            run_id=config.progress_run_id,
            family=progress_family,
            shard_status="complete",
            expected_cells=expected_cells,
            completed_cells=len(cells),
            candidate_evaluations=len(comparisons),
            worker_id=progress_worker_id,
            extra={"stage": "mini_test_complete"},
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _evaluate_cell_windows(
    *,
    asset: str,
    band: str,
    family_name: str,
    feature_columns: Sequence[str],
    context: Mapping[str, Any],
    frame: Any,
    min_rows: int,
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
    feature_panel_matrix_cache: CrossAssetFeaturePanelMatrixCache | None = None,
    candidate_definition_cache: CrossAssetStateCandidateDefinitionCache | None = None,
) -> dict[str, Any]:
    best_candidate = None
    best_frame = None
    best_policy = None
    best_coverage = None
    profile_comparisons: list[dict[str, Any]] = []
    window_comparisons: list[dict[str, Any]] = []
    for policy in cross_asset_state_window_ladder_policies(relationship_feature_family=family_name, band=band):
        if feature_panel_matrix_cache is not None:
            window_frame, coverage = feature_panel_matrix_cache.dataset_matrix(
                frame,
                asset=asset,
                band=band,
                family=family_name,
                required_columns=feature_columns,
                window_policy_id=policy.window_policy_id,
                feature_set_version=feature_set_version,
                relationship_context_id=context.get("relationship_context_id"),
                relationship_snapshot_id=context.get("relationship_snapshot_id"),
                known_at_ts=context.get("known_at_ts"),
                source_tail_ts=context.get("source_tail_ts"),
                matrix_role="window_feature_panel",
                min_rows=min_rows,
                build_fn=lambda policy=policy: apply_cross_asset_state_window_policy(
                    frame,
                    policy,
                    source_tail_ts=context.get("source_tail_ts"),
                    min_rows=min_rows,
                ),
            )
        else:
            window_frame, coverage = apply_cross_asset_state_window_policy(
                frame,
                policy,
                source_tail_ts=context.get("source_tail_ts"),
                min_rows=min_rows,
            )
        if not coverage.passed:
            window_comparisons.append(
                _window_comparison_row(
                    asset=asset,
                    band=band,
                    family=family_name,
                    policy=policy,
                    coverage=coverage.as_dict(),
                    candidate_status="window_masked_unavailable",
                    evaluated_profile_count=0,
                    passed_profile_count=0,
                    failed_profile_count=0,
                    best_candidate=None,
                )
            )
            continue
        candidate_definitions = None
        if candidate_definition_cache is not None:
            definition_shape = candidate_definition_shape(window_frame, family=family_name, feature_columns=feature_columns)
            candidate_definitions = candidate_definition_cache.definitions(
                family=family_name,
                band=band,
                window_policy_id=policy.window_policy_id,
                feature_set_version=feature_set_version,
                method_universe_version=CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
                search_space_id=CROSS_ASSET_STATE_SEARCH_SPACE_ID,
                feature_columns=feature_columns,
                row_count=definition_shape["row_count"],
                feature_count=definition_shape["feature_count"],
            )
        candidates = run_profile_candidates(
            window_frame,
            family=family_name,
            feature_columns=feature_columns,
            candidate_definitions=candidate_definitions,
        )
        if candidate_definition_cache is not None:
            candidate_definition_cache.record_candidate_evaluations(len(candidates))
        best_for_window = choose_best_candidate(candidates)
        for candidate in candidates:
            row = _profile_comparison_row(asset, band, family_name, candidate, window_policy=policy, window_coverage=coverage.as_dict())
            profile_comparisons.append(row)
        passed_count = sum(1 for candidate in candidates if candidate.output_health.get("passed"))
        failed_count = len(candidates) - passed_count
        window_comparisons.append(
            _window_comparison_row(
                asset=asset,
                band=band,
                family=family_name,
                policy=policy,
                coverage=coverage.as_dict(),
                candidate_status="window_evaluated",
                evaluated_profile_count=len(candidates),
                passed_profile_count=passed_count,
                failed_profile_count=failed_count,
                best_candidate=best_for_window,
            )
        )
        if not best_for_window.output_health.get("passed") or not getattr(best_for_window, "selection_eligible", True):
            continue
        if best_candidate is None or _candidate_rank(best_for_window) > _candidate_rank(best_candidate):
            best_candidate = best_for_window
            best_frame = window_frame
            best_policy = policy
            best_coverage = coverage
    return {
        "best_candidate": best_candidate,
        "best_frame": best_frame,
        "best_policy": best_policy,
        "best_coverage": best_coverage,
        "profile_comparisons": profile_comparisons,
        "window_comparisons": window_comparisons,
    }


def _profile_comparison_row(
    asset: str,
    band: str,
    family: str,
    candidate: Any,
    *,
    window_policy: Any,
    window_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    from src.regimes.cross_asset_state.test_prototype import _comparison_row

    return _comparison_row(asset, band, family, candidate, window_policy=window_policy, window_coverage=window_coverage)


def _window_comparison_row(
    *,
    asset: str,
    band: str,
    family: str,
    policy: Any,
    coverage: Mapping[str, Any],
    candidate_status: str,
    evaluated_profile_count: int,
    passed_profile_count: int,
    failed_profile_count: int,
    best_candidate: Any | None,
) -> dict[str, Any]:
    score = best_candidate.diagnostic_score.as_dict() if best_candidate is not None else {}
    health = best_candidate.output_health if best_candidate is not None else {}
    return {
        "asset_id": asset,
        "band": band,
        "relationship_feature_family": family,
        "window_policy_id": policy.window_policy_id,
        "window_candidate_name": policy.window_candidate_name,
        "lookback_days": policy.lookback_days,
        "source_tail_anchor": policy.source_tail_anchor,
        "window_status": coverage.get("status"),
        "window_passed": coverage.get("passed"),
        "window_reason_code": coverage.get("reason_code"),
        "window_observed_rows": coverage.get("observed_rows"),
        "window_required_min_rows": coverage.get("min_rows"),
        "window_start_ts": coverage.get("start_ts"),
        "window_end_ts": coverage.get("end_ts"),
        "candidate_status": candidate_status,
        "evaluated_profile_count": int(evaluated_profile_count),
        "passed_profile_count": int(passed_profile_count),
        "failed_profile_count": int(failed_profile_count),
        "best_profile_type": getattr(best_candidate, "profile_type", None),
        "best_candidate_id": getattr(best_candidate, "candidate_id", None),
        "best_parameter_grid_id": getattr(best_candidate, "parameter_grid_id", None),
        "best_readiness_status": getattr(best_candidate, "readiness_status", None),
        "best_selection_eligible": getattr(best_candidate, "selection_eligible", None),
        "best_diagnostic_only": getattr(best_candidate, "diagnostic_only", None),
        "best_total_candidate_score": score.get("total_candidate_score"),
        "best_semantic_separation_score": score.get("semantic_separation_score"),
        "best_temporal_stability_score": score.get("temporal_stability_score"),
        "best_coverage_score": score.get("coverage_score"),
        "best_output_health_status": "passed" if health.get("passed") else ("failed" if best_candidate is not None else None),
        "dominant_state_share": health.get("dominant_state_share"),
        "tiny_state_count": health.get("tiny_state_count"),
        "warning_reasons": "|".join(str(reason) for reason in health.get("warning_reasons") or ()),
        "failure_reasons": "|".join(str(reason) for reason in health.get("failure_reasons") or ()),
    }


def _candidate_rank(candidate: Any) -> tuple[bool, float, float]:
    score = candidate.diagnostic_score.as_dict()
    return (
        bool(candidate.output_health.get("passed", False)),
        float(score.get("total_candidate_score") or 0.0),
        float(score.get("runtime_tiebreak_score") or 0.0),
    )


def _should_flush_progress(completed_cells: int, interval: int) -> bool:
    step = max(1, int(interval))
    return completed_cells <= 1 or completed_cells % step == 0


def _mask_reason_from_profile_comparisons(comparisons: Sequence[Mapping[str, Any]]) -> str:
    reasons = [
        normalize_test_branch_filter_reason(str(row.get("selection_exclusion_reason") or ""))
        for row in comparisons
        if row.get("selection_exclusion_reason") not in (None, "", "None")
    ]
    if reasons and all(reason == FILTER_LOW_FEATURE_SPREAD for reason in reasons):
        return FILTER_LOW_FEATURE_SPREAD
    if FILTER_LOW_FEATURE_SPREAD in set(reasons):
        return FILTER_LOW_FEATURE_SPREAD
    return "no_candidate_passed_output_health"


def _mini_profile_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["profile_id"] = f"{out['asset_id']}|{out['relationship_feature_family']}|{out['band']}|mini_test_v1"
    out.setdefault("economic_diagnostic_status", ECONOMIC_DIAGNOSTIC_PENDING)
    out.setdefault("production_approved", False)
    out.setdefault("production_writer_enabled", False)
    out.setdefault("production_labels_written", False)
    out.setdefault("production_outputs_written", False)
    out.setdefault("canonical_production_state_outputs_written", False)
    out.setdefault("requires_human_approval_before_production", True)
    return out


def _active_run_surface(
    *,
    config: CrossAssetStateMiniTestConfig,
    handoff: Any,
    availability: Any | None,
    target_universe: Sequence[Mapping[str, Any]],
    valid_target_assets_across_bands: Sequence[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    valid_target_count = len(valid_target_assets_across_bands)
    if valid_target_count < int(config.max_valid_assets):
        blockers.append(f"valid_target_count_below_{config.max_valid_assets}")
    availability_active = availability is not None and not getattr(availability, "empty", True)
    if not availability_active:
        blockers.append("value_availability_sidecar_missing_or_empty")
    production_consumer_fail_closed = False
    try:
        validate_cross_asset_state_manifest_for_production(
            {
                "artifact_kind": "cross_asset_state_v1_shape_probe_summary",
                "production_approved": False,
                "production_writer_enabled": False,
                "canonical_production_state_outputs_written": False,
            }
        )
    except CrossAssetStateProductionGateError:
        production_consumer_fail_closed = True
    if not production_consumer_fail_closed:
        blockers.append("production_consumer_not_fail_closed")
    return {
        "safe_to_run": not blockers,
        "blockers": blockers,
        "active_nonproduction_handoff_resolved": True,
        "relationship_context_id": handoff.relationship_context_id,
        "relationship_snapshot_root_count": len(handoff.relationship_snapshot_roots),
        "feature_root_count": len(handoff.feature_roots),
        "availability_sidecar_active": availability_active,
        "availability_row_count": 0 if availability is None else int(len(availability)),
        "availability_status_counts": _availability_status_counts(availability),
        "unavailable_zero_treated_as_neutral": False,
        "unavailable_zero_policy": "masked_unavailable availability rows are dropped before scoring; they are never filled as neutral 0.0",
        "model_facing_peer_identity_columns_blocked": True,
        "peer_metadata_status": "sidecar_only_not_model_facing",
        "production_consumer_fail_closed": production_consumer_fail_closed,
        "valid_target_assets_across_bands_count": valid_target_count,
        "valid_target_assets_across_bands_sample": list(valid_target_assets_across_bands[:50]),
        "valid_target_asset_counts_by_band": _valid_target_counts_by_band(target_universe),
        "stale_sandbox_artifact_resolution_default": "blocked",
        "active_handoff_path": str(config.handoff_path),
    }


def _availability_status_counts(availability: Any | None) -> dict[str, int]:
    if availability is None or getattr(availability, "empty", True) or "value_status" not in availability.columns:
        return {}
    return _counts([str(value) for value in availability["value_status"].fillna("").tolist()])


def _mini_family_summary(
    cells: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    window_comparisons: Sequence[Mapping[str, Any]],
    inspection_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = sorted({str(cell["relationship_feature_family"]) for cell in cells})
    questionable_counts = _counts(
        [
            str(row.get("relationship_feature_family"))
            for row in inspection_rows
            if str(row.get("coherence_status")) == "questionable"
        ]
    )
    for family in families:
        fam_cells = [cell for cell in cells if cell["relationship_feature_family"] == family]
        selected = [cell for cell in fam_cells if cell["cell_status"] == "selected_coherent"]
        masked = [cell for cell in fam_cells if cell["cell_status"] == "masked_unavailable"]
        fam_comp = [row for row in comparisons if row["relationship_feature_family"] == family]
        fam_windows = [row for row in window_comparisons if row["relationship_feature_family"] == family]
        selected_profile_counts = _counts([str(cell.get("selected_profile_type")) for cell in selected if cell.get("selected_profile_type")])
        selected_window_counts = _counts([str(cell.get("window_candidate_name")) for cell in selected if cell.get("window_candidate_name")])
        best_profile_type = _most_common_from_counts(selected_profile_counts)
        best_window = _most_common_from_counts(selected_window_counts)
        health_failures = sum(1 for row in fam_comp if str(row.get("candidate_status")) == "candidate_failed_health")
        birch_warnings = sum(1 for row in fam_comp if str(row.get("profile_type")) == "birch" and "birch_too_few_subclusters" in str(row.get("warning_reasons") or ""))
        dominant_tiny = _dominant_or_tiny_issue_count(fam_comp)
        avg_score = _avg(selected, "total_candidate_score")
        avg_semantic = _avg(selected, "semantic_separation_score")
        avg_temporal = _avg(selected, "temporal_persistence_score")
        non_default_share = _non_default_window_share(selected_window_counts)
        readiness = _family_readiness(
            family=family,
            selected_count=len(selected),
            masked_count=len(masked),
            health_failures=health_failures,
            dominant_tiny=dominant_tiny,
            questionable_examples=questionable_counts.get(family, 0),
            non_default_window_share=non_default_share,
            selected_profile_counts=selected_profile_counts,
        )
        rows.append(
            {
                "relationship_feature_family": family,
                "expected_cells": len(fam_cells),
                "selected_cells": len(selected),
                "masked_unavailable_cells": len(masked),
                "missing_cells": 0,
                "best_profile_type": best_profile_type,
                "best_window_candidate_name": best_window,
                "selected_profile_type_counts": json.dumps(selected_profile_counts, sort_keys=True),
                "selected_window_policy_counts": json.dumps(selected_window_counts, sort_keys=True),
                "avg_total_candidate_score": round(avg_score, 6),
                "avg_semantic_separation_score": round(avg_semantic, 6),
                "avg_temporal_persistence_score": round(avg_temporal, 6),
                "candidate_health_failure_count": health_failures,
                "dominant_or_tiny_state_failure_count": dominant_tiny,
                "birch_warning_count": birch_warnings,
                "window_masked_unavailable_count": sum(1 for row in fam_windows if row.get("window_passed") is False),
                "questionable_inspection_example_count": questionable_counts.get(family, 0),
                "economic_diagnostic_status": ECONOMIC_DIAGNOSTIC_PENDING,
                "family_readiness": readiness,
            }
        )
    return rows


def _family_readiness(
    *,
    family: str,
    selected_count: int,
    masked_count: int,
    health_failures: int,
    dominant_tiny: int,
    questionable_examples: int,
    non_default_window_share: float,
    selected_profile_counts: Mapping[str, int],
) -> str:
    if selected_count <= 0:
        return "blocked"
    if family == "residual_peer_signal" and questionable_examples == 0:
        return "ready_for_larger_bounded_campaign"
    if non_default_window_share >= 0.35:
        return "needs_window_policy_repair"
    if family == "relationship_concentration_entropy" and (questionable_examples or dominant_tiny > selected_count):
        return "diagnostic_only_for_now"
    if family == "peer_strength_stability" and (questionable_examples or dominant_tiny > selected_count):
        return "needs_profile_type_repair"
    if family == "anchor_core_exposure" and selected_profile_counts.get("rule_threshold", 0) == 0:
        return "needs_scoring_repair"
    if health_failures > selected_count * 2:
        return "needs_profile_type_repair"
    if masked_count > selected_count:
        return "blocked"
    return "ready_for_larger_bounded_campaign"


def _best_window_policy_patterns(cells: Sequence[Mapping[str, Any]], window_comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["cell_status"] == "selected_coherent"]
    by_family: dict[str, Any] = {}
    for family in sorted({str(cell["relationship_feature_family"]) for cell in selected}):
        fam_selected = [cell for cell in selected if str(cell["relationship_feature_family"]) == family]
        selected_counts = _counts([str(cell.get("window_candidate_name")) for cell in fam_selected if cell.get("window_candidate_name")])
        fam_windows = [row for row in window_comparisons if str(row.get("relationship_feature_family")) == family]
        evaluated_counts = _counts([str(row.get("window_candidate_name")) for row in fam_windows if str(row.get("candidate_status")) == "window_evaluated"])
        by_family[family] = {
            "selected_window_counts": selected_counts,
            "evaluated_window_counts": evaluated_counts,
            "non_default_selected_share": round(_non_default_window_share(selected_counts), 6),
            "best_window_candidate_name": _most_common_from_counts(selected_counts),
        }
    return by_family


def _specific_family_answers(
    cells: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    window_comparisons: Sequence[Mapping[str, Any]],
    inspection_rows: Sequence[Mapping[str, Any]],
    family_summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    best_profiles = {str(row["relationship_feature_family"]): str(row.get("best_profile_type")) for row in family_summary}
    readiness = {str(row["relationship_feature_family"]): str(row.get("family_readiness")) for row in family_summary}
    selected_counts = _counts([str(cell.get("selected_profile_type")) for cell in cells if cell["cell_status"] == "selected_coherent"])
    cluster_profiles = {
        "kmeans",
        "minibatch_kmeans",
        "pca_kmeans",
        "factor_analysis_kmeans",
        "gaussian_mixture",
        "factor_analysis_gaussian_mixture",
        "bayesian_gaussian_mixture",
        "birch",
        "hdbscan",
        "optics",
        "agglomerative",
    }
    cluster_selected = sum(count for profile, count in selected_counts.items() if profile in cluster_profiles)
    rule_ordinal_selected = sum(count for profile, count in selected_counts.items() if profile in {"rule_threshold", "ordinal_quantile"})
    residual_inspection_counts = _counts(
        [
            str(row.get("coherence_status"))
            for row in inspection_rows
            if row.get("relationship_feature_family") == "residual_peer_signal"
        ]
    )
    return {
        "residual_peer_signal_positive_control": residual_inspection_counts.get("inspectable", 0)
        >= max(1, residual_inspection_counts.get("questionable", 0)),
        "residual_peer_signal_positive_control_note": (
            "Residual states remain human-inspectable enough to act as the scoring positive control, "
            "but selected windows still need formalization before scale."
        ),
        "peer_strength_stability_improved": _family_non_default_window_selected("peer_strength_stability", cells)
        or best_profiles.get("peer_strength_stability") in {"rule_threshold", "ordinal_quantile"},
        "relationship_concentration_entropy_improved": _family_non_default_window_selected("relationship_concentration_entropy", cells)
        or best_profiles.get("relationship_concentration_entropy") in {"rule_threshold", "ordinal_quantile"},
        "anchor_core_exposure_still_saturated": _anchor_saturation_signal(comparisons, inspection_rows),
        "clustering_profiles_still_dominating": cluster_selected > rule_ordinal_selected,
        "rule_or_ordinal_profiles_useful": rule_ordinal_selected > 0,
        "profile_selection_counts": dict(selected_counts),
        "window_materiality_by_family": _best_window_policy_patterns(cells, window_comparisons),
    }


def _anchor_saturation_signal(comparisons: Sequence[Mapping[str, Any]], inspection_rows: Sequence[Mapping[str, Any]]) -> bool:
    anchor_failures = [
        row
        for row in comparisons
        if row.get("relationship_feature_family") == "anchor_core_exposure"
        and (
            "all_one_state_collapse" in str(row.get("failure_reasons") or "")
            or float(row.get("dominant_state_share") or 0.0) > 0.90
        )
    ]
    questionable = [
        row
        for row in inspection_rows
        if row.get("relationship_feature_family") == "anchor_core_exposure" and row.get("coherence_status") == "questionable"
    ]
    return bool(anchor_failures or questionable)


def _family_non_default_window_selected(family: str, cells: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        cell["cell_status"] == "selected_coherent"
        and cell.get("relationship_feature_family") == family
        and cell.get("window_candidate_name") not in (None, "", "current_default")
        for cell in cells
    )


def _major_failure_modes(comparisons: Sequence[Mapping[str, Any]], window_comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [row for row in comparisons if str(row.get("candidate_status")) == "candidate_failed_health"]
    warnings = [row for row in comparisons if str(row.get("warning_reasons") or "").strip()]
    window_failed = [row for row in window_comparisons if row.get("window_passed") is False]
    return {
        "candidate_health_failure_count": len(failed),
        "candidate_failure_counts_by_family": _counts([str(row.get("relationship_feature_family")) for row in failed]),
        "candidate_failure_counts_by_profile_type": _counts([str(row.get("profile_type")) for row in failed]),
        "candidate_failure_reason_counts": _counts(
            [
                reason
                for row in failed
                for reason in str(row.get("failure_reasons") or row.get("failure_mode") or "output_health_gate_failed").split("|")
                if reason
            ]
        ),
        "warning_reason_counts": _warning_reason_counts(warnings),
        "window_masked_unavailable_count": len(window_failed),
        "window_failure_counts_by_candidate": _counts([str(row.get("window_candidate_name")) for row in window_failed]),
        "window_failure_counts_by_family": _counts([str(row.get("relationship_feature_family")) for row in window_failed]),
        "dominant_or_tiny_issue_count": _dominant_or_tiny_issue_count(comparisons),
    }


def _mini_runtime_rows(
    cells: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    window_comparisons: Sequence[Mapping[str, Any]],
    started: float,
) -> list[dict[str, Any]]:
    rows = [
        {
            "aggregate_scope": "overall",
            "key": "all",
            "cell_count": len(cells),
            "candidate_count": len(comparisons),
            "window_candidate_count": len(window_comparisons),
            "selected_count": sum(1 for cell in cells if cell["cell_status"] == "selected_coherent"),
            "masked_count": sum(1 for cell in cells if cell["cell_status"] == "masked_unavailable"),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "memory_rss_bytes": _rss_bytes(),
        }
    ]
    for scope, key_name in (
        ("family", "relationship_feature_family"),
        ("profile_type", "profile_type"),
        ("window_policy", "window_policy_id"),
        ("window_candidate", "window_candidate_name"),
        ("band", "band"),
    ):
        keys = sorted({str(row.get(key_name)) for row in comparisons if row.get(key_name) is not None})
        for key in keys:
            scoped = [row for row in comparisons if str(row.get(key_name)) == key]
            rows.append(
                {
                    "aggregate_scope": scope,
                    "key": key,
                    "cell_count": "",
                    "candidate_count": len(scoped),
                    "window_candidate_count": "",
                    "selected_count": "",
                    "masked_count": "",
                    "runtime_seconds": round(sum(float(row.get("runtime_seconds") or 0.0) for row in scoped), 6),
                    "memory_rss_bytes": "",
                }
            )
    return rows


def _mini_runtime_summary(runtime_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = next((row for row in runtime_rows if row.get("aggregate_scope") == "overall"), {})
    return {
        "total_cells": overall.get("cell_count", 0),
        "candidate_profile_evaluations": overall.get("candidate_count", 0),
        "window_candidate_evaluations": overall.get("window_candidate_count", 0),
        "selected_count": overall.get("selected_count", 0),
        "masked_count": overall.get("masked_count", 0),
        "wall_time_seconds": overall.get("runtime_seconds", 0.0),
        "memory_rss_bytes": overall.get("memory_rss_bytes"),
    }


def _runtime_cache_counts(feature_panel_matrix_cache: Mapping[str, Any], candidate_definition_cache: Mapping[str, Any]) -> dict[str, Any]:
    panel = feature_panel_matrix_cache.get("feature_panel_cache") if isinstance(feature_panel_matrix_cache.get("feature_panel_cache"), Mapping) else {}
    matrix = feature_panel_matrix_cache.get("dataset_matrix_cache") if isinstance(feature_panel_matrix_cache.get("dataset_matrix_cache"), Mapping) else {}
    candidate = candidate_definition_cache if isinstance(candidate_definition_cache, Mapping) else {}
    hit_count = int(panel.get("hit_count") or 0) + int(matrix.get("hit_count") or 0) + int(candidate.get("hit_count") or 0)
    miss_count = int(panel.get("miss_count") or 0) + int(matrix.get("miss_count") or 0) + int(candidate.get("miss_count") or 0)
    return {
        "cache_hit_count": hit_count,
        "cache_miss_count": miss_count,
        "feature_panel_hit_count": int(panel.get("hit_count") or 0),
        "feature_panel_miss_count": int(panel.get("miss_count") or 0),
        "dataset_matrix_hit_count": int(matrix.get("hit_count") or 0),
        "dataset_matrix_miss_count": int(matrix.get("miss_count") or 0),
        "candidate_definition_hit_count": int(candidate.get("hit_count") or 0),
        "candidate_definition_miss_count": int(candidate.get("miss_count") or 0),
    }


def _mini_scale_estimate(
    *,
    cells: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    window_comparisons: Sequence[Mapping[str, Any]],
    valid_target_assets: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    selected_assets = {str(cell["asset_id"]) for cell in cells if cell["cell_status"] == "selected_coherent"}
    selected_asset_count = max(1, len(selected_assets))
    seconds_per_selected_asset = elapsed_seconds / selected_asset_count
    estimated_seconds = seconds_per_selected_asset * max(valid_target_assets, selected_asset_count)
    profiles_per_asset = len(comparisons) / selected_asset_count
    windows_per_asset = len(window_comparisons) / selected_asset_count
    return {
        "observed_selected_asset_count": len(selected_assets),
        "active_handoff_valid_target_assets": int(valid_target_assets),
        "seconds_per_selected_asset_observed": round(seconds_per_selected_asset, 6),
        "profiles_evaluated_per_selected_asset": round(profiles_per_asset, 6),
        "windows_evaluated_per_selected_asset": round(windows_per_asset, 6),
        "estimated_seconds_for_active_handoff_valid_targets": round(estimated_seconds, 6),
        "larger_bounded_campaign_estimate_note": "Multiply seconds_per_selected_asset by the planned bounded target count; estimate excludes full-universe discovery.",
        "full_universe_campaign_manageability_signal": (
            "not_assessed_no_full_universe_campaign"
            if valid_target_assets <= 50
            else "requires_worker_orchestration_before_full_universe_consideration"
        ),
    }


def _window_policy_rows_for_summary(families: Sequence[Any], bands: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for family in families:
        for band in bands:
            for policy in cross_asset_state_window_ladder_policies(relationship_feature_family=family.name, band=band):
                key = (policy.relationship_feature_family, policy.band, policy.window_candidate_name)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(policy.as_dict())
    return rows


def _mini_final_verdict(
    family_readiness: Mapping[str, str],
    best_window_patterns: Mapping[str, Any],
    economic_contract: Mapping[str, Any],
) -> str:
    if any(status == "blocked" for status in family_readiness.values()):
        return "F. V1 shape/profile approach needs redesign."
    materially_window_sensitive = any(
        float(data.get("non_default_selected_share") or 0.0) >= 0.35
        for data in best_window_patterns.values()
        if isinstance(data, Mapping)
    )
    if materially_window_sensitive:
        return "D. Window policy materially changes outputs and needs formalization before larger campaign."
    weak = {
        family: status
        for family, status in family_readiness.items()
        if status not in {"ready_for_larger_bounded_campaign"}
    }
    if weak:
        return "B. Mini Test Branch shape works, but specific families need repair."
    if economic_contract.get("economic_diagnostic_status") != "computed":
        return "C. Economic diagnostics block larger campaign confidence."
    return "A. Mini Test Branch is coherent; proceed to larger bounded campaign."


def _mini_next_recommended_sprint(
    family_readiness: Mapping[str, str],
    best_window_patterns: Mapping[str, Any],
    economic_contract: Mapping[str, Any],
) -> str:
    window_families = sorted(
        family
        for family, data in best_window_patterns.items()
        if isinstance(data, Mapping) and float(data.get("non_default_selected_share") or 0.0) >= 0.35
    )
    repair_families = sorted(
        family
        for family, status in family_readiness.items()
        if status not in {"ready_for_larger_bounded_campaign"} and family not in window_families
    )
    if window_families:
        return "Formalize Cross-Asset-State window policy for " + ", ".join(window_families) + " and rerun the same 40-target mini-test."
    if repair_families:
        return "Run a focused family repair sprint for " + ", ".join(repair_families) + " before larger bounded campaign design."
    if economic_contract.get("economic_diagnostic_status") != "computed":
        return "Implement leakage-safe economic outcome panels, then rerun this mini-test with economic scores enabled."
    return "Design the next larger bounded non-production campaign using this mini-test contract."


def _avg(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field) or 0.0))
        except Exception:
            continue
    return sum(values) / len(values) if values else 0.0


def _non_default_window_share(counts: Mapping[str, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    current = int(counts.get("current_default", 0))
    return (total - current) / total


def _most_common_from_counts(counts: Mapping[str, int]) -> str | None:
    if not counts:
        return None
    return max(sorted(counts), key=lambda key: counts[key])


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Cross-Asset-State v1 mini Test Branch diagnostic.")
    parser.add_argument("--handoff-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-valid-assets", type=int, default=40)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)
    summary = run_cross_asset_state_v1_mini_test(
        CrossAssetStateMiniTestConfig(
            handoff_path=args.handoff_path,
            output_root=args.output_root,
            max_valid_assets=args.max_valid_assets,
            min_valid_assets_for_meaningful_sample=args.max_valid_assets,
        )
    )
    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CrossAssetStateMiniTestConfig",
    "run_cross_asset_state_v1_mini_test",
]
