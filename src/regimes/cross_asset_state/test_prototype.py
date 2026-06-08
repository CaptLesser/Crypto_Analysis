from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.test_branch_maturity import normalize_test_branch_filter_reason
from src.regimes.cross_asset_state.dataset_builder import filter_frame_to_available_family_rows, load_relationship_value_availability
from src.regimes.cross_asset_state.economic_diagnostics import build_cross_asset_state_economic_diagnostic_contract
from src.regimes.cross_asset_state.feature_families import (
    CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
    CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
    resolve_feature_families,
)
from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason
from src.regimes.cross_asset_state.method_universe import (
    CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
    CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
    cross_asset_state_method_universe_manifest,
    validate_cross_asset_state_method_universe_manifest,
)
from src.regimes.cross_asset_state.profile_manifest import validate_cross_asset_state_selected_profile_manifest
from src.regimes.cross_asset_state.profile_candidates import choose_best_candidate, diagnostic_only_result, run_profile_candidates
from src.regimes.cross_asset_state.relationship_context import (
    CrossAssetRelationshipContextHandoff,
    CrossAssetRelationshipContextResolver,
)
from src.regimes.cross_asset_state.scoring_contract import ECONOMIC_DIAGNOSTIC_PENDING, SCORING_SCHEMA_ID, SCORING_SCHEMA_VERSION
from src.regimes.cross_asset_state.window_profiles import (
    apply_cross_asset_state_window_policy,
    cross_asset_state_window_policy_manifest,
    family_window_sensitivity_summary,
    resolve_cross_asset_state_window_policy,
)


DEFAULT_MASKED_CONTROL_ASSETS: tuple[str, ...] = ("MISSING_ASSET_USD",)
DEFAULT_TARGET_SAMPLE_PRIORITY: tuple[str, ...] = (
    "XBTUSD",
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "XRPUSD",
    "ADAUSD",
    "AVAXUSD",
    "LINKUSD",
    "AAVEUSD",
    "DOGEUSD",
)

PROTOTYPE_PROFILE_TYPES: tuple[str, ...] = (
    "rule_threshold",
    "ordinal_quantile",
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
    "diagnostic_only",
)

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


@dataclass(frozen=True)
class CrossAssetStateTestPrototypeConfig:
    handoff_path: str | Path
    output_root: str | Path
    artifact_label: str = "v1_test_prototype"
    csv_prefix: str = "cross_asset_state_v1"
    summary_filename: str = "cross_asset_state_v1_test_prototype_summary.json"
    selected_profiles_filename: str = "cross_asset_state_selected_profiles.prototype.nonprod.json"
    inspection_examples_filename: str | None = None
    health_failures_filename: str | None = None
    sample_assets: tuple[str, ...] = ()
    bands: tuple[str, ...] = ("meso", "macro")
    max_valid_assets: int = 25
    min_valid_assets_for_meaningful_sample: int = 20
    max_masked_controls: int = 4
    masked_control_assets: tuple[str, ...] = DEFAULT_MASKED_CONTROL_ASSETS
    include_masked_probe_asset: bool = True
    min_rows_per_cell: int = 8
    max_parquet_files_per_asset: int = 4
    window_policy_id: str | None = None
    allow_stale_sandbox_artifacts: bool = False
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL


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
    latest = frame
    if "ts" in latest.columns:
        latest = latest.sort_values("ts")
    row = latest.iloc[-1] if len(latest) else None
    if row is None:
        return {}
    out: dict[str, Any] = {}
    for column in SUPPORT_METADATA_COLUMNS:
        if column not in latest.columns:
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


def run_cross_asset_state_test_prototype(config: CrossAssetStateTestPrototypeConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    handoff = _load_handoff(config.handoff_path)
    resolver = CrossAssetRelationshipContextResolver.from_handoff(handoff, max_parquet_files_per_root=250)
    families = resolve_feature_families(feature_set_version=config.feature_set_version)
    availability = load_relationship_value_availability(handoff.availability_sidecar_refs)
    target_universe = build_cross_asset_state_valid_target_universe(
        handoff,
        resolver,
        availability,
        bands=config.bands,
        families=families,
    )
    sample_plan = _manual_sample_plan_rows(
        config.sample_assets,
        target_universe=target_universe,
        include_masked_probe_asset=config.include_masked_probe_asset,
    )
    if not sample_plan:
        sample_plan = select_cross_asset_state_sample(
            target_universe,
            bands=config.bands,
            max_valid_assets=config.max_valid_assets,
            max_masked_controls=config.max_masked_controls,
            masked_control_assets=config.masked_control_assets,
            include_masked_probe_asset=config.include_masked_probe_asset,
        )
    sample_assets = tuple(dict.fromkeys(str(row["asset_id"]) for row in sample_plan if str(row.get("asset_id", "")).strip()))
    valid_sample_assets = tuple(row["asset_id"] for row in sample_plan if row.get("sample_kind") == "valid_target")
    if not sample_assets:
        raise ValueError("Cross-Asset-State v1 test prototype found no sample assets")

    feature_frames = _load_feature_frames(handoff.feature_roots, sample_assets, config.bands, max_files_per_asset=config.max_parquet_files_per_asset)
    _assert_no_peer_identity_leak(feature_frames)

    cells: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    selected_profiles: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    inspection_counts: dict[str, int] = {}

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
                window_policy = resolve_cross_asset_state_window_policy(
                    relationship_feature_family=family.name,
                    band=band,
                    window_policy_id=config.window_policy_id,
                )
                frame = feature_frames.get((asset, band))
                if context["context_status"] != "available" or frame is None:
                    reason = context["mask_reason"] or "missing_required_field"
                    cell = _masked_cell(asset, band, family.name, context, reason, window_policy=window_policy)
                    cell = _with_feature_set(cell, config.feature_set_version, frame)
                    cells.append(cell)
                    selected_profiles.append(_profile_record_from_masked_cell(cell, handoff))
                    continue
                frame, availability_reason = filter_frame_to_available_family_rows(
                    frame,
                    availability,
                    asset=asset,
                    band=band,
                    family=family.name,
                    required_columns=family.required_columns,
                )
                if frame is None or len(frame) < config.min_rows_per_cell:
                    cell = _masked_cell(asset, band, family.name, context, availability_reason or "insufficient_rows", window_policy=window_policy)
                    cell = _with_feature_set(cell, config.feature_set_version, frame)
                    cells.append(cell)
                    selected_profiles.append(_profile_record_from_masked_cell(cell, handoff))
                    continue
                frame, window_coverage = apply_cross_asset_state_window_policy(
                    frame,
                    window_policy,
                    source_tail_ts=context.get("source_tail_ts"),
                    min_rows=config.min_rows_per_cell,
                )
                if not window_coverage.passed:
                    cell = _masked_cell(
                        asset,
                        band,
                        family.name,
                        context,
                        CrossAssetStateMaskReason.INSUFFICIENT_WINDOW_HISTORY,
                        window_policy=window_policy,
                        window_coverage=window_coverage.as_dict(),
                    )
                    cell = _with_feature_set(cell, config.feature_set_version, frame)
                    cells.append(cell)
                    selected_profiles.append(_profile_record_from_masked_cell(cell, handoff))
                    continue
                candidates = run_profile_candidates(frame, family=family.name, feature_columns=family.required_columns)
                best = choose_best_candidate(candidates)
                if not best.output_health.get("passed") or not getattr(best, "selection_eligible", True):
                    best = diagnostic_only_result(frame, family=family.name, feature_columns=family.required_columns, reason="no_candidate_passed_output_health")
                for candidate in candidates:
                    comparisons.append(_comparison_row(asset, band, family.name, candidate, window_policy=window_policy, window_coverage=window_coverage.as_dict()))
                for label, count in best.label_counts.items():
                    label_rows.append(
                        {
                            "asset_id": asset,
                            "band": band,
                            "relationship_feature_family": family.name,
                            "profile_type": best.profile_type,
                            "label": label,
                            "count": count,
                        }
                    )
                cell = _selected_cell(asset, band, family.name, context, best, len(frame), window_policy=window_policy, window_coverage=window_coverage.as_dict())
                cell = _with_feature_set(cell, config.feature_set_version, frame)
                cells.append(cell)
                selected_profiles.append(_profile_record_from_selected_cell(cell, family.required_columns, handoff))
                if inspection_counts.get(family.name, 0) < 5 and cell["cell_status"] == "selected_coherent":
                    inspection_rows.append(_inspection_row(asset, band, family.name, context, best, frame, window_policy=window_policy, window_coverage=window_coverage.as_dict()))
                    inspection_counts[family.name] = inspection_counts.get(family.name, 0) + 1

    family_summary = _family_summary(cells, comparisons)
    open_items = _open_items(family_summary, comparisons)
    health_failures = _health_failure_rows(comparisons)

    cells_path = output_root / f"{config.csv_prefix}_cells.csv"
    family_path = output_root / f"{config.csv_prefix}_family_summary.csv"
    comparison_path = output_root / f"{config.csv_prefix}_profile_type_comparison.csv"
    inspection_path = output_root / (config.inspection_examples_filename or f"{config.csv_prefix}_inspection_examples.csv")
    health_failures_path = output_root / (config.health_failures_filename or f"{config.csv_prefix}_health_failures.csv")
    labels_path = output_root / f"{config.csv_prefix}_label_distributions.csv"
    open_items_path = output_root / f"{config.csv_prefix}_open_items.csv"
    runtime_path = output_root / f"{config.csv_prefix}_runtime.csv"
    sample_selection_path = output_root / f"{config.csv_prefix}_sample_selection.csv"
    candidate_pool_path = output_root / f"{config.csv_prefix}_candidate_pool.csv"
    selected_profiles_path = output_root / config.selected_profiles_filename
    summary_path = output_root / config.summary_filename
    runtime_rows = _runtime_rows(cells, comparisons, started)
    sample_selection = _sample_selection_rows(sample_assets, valid_sample_assets, cells, feature_frames, sample_plan)
    economic_contract = build_cross_asset_state_economic_diagnostic_contract(handoff)

    _write_csv(cells_path, cells)
    _write_csv(family_path, family_summary)
    _write_csv(comparison_path, comparisons)
    _write_csv(inspection_path, inspection_rows)
    _write_csv(health_failures_path, health_failures)
    _write_csv(labels_path, label_rows)
    _write_csv(open_items_path, open_items)
    _write_csv(runtime_path, runtime_rows)
    _write_csv(sample_selection_path, sample_selection)
    _write_csv(candidate_pool_path, target_universe)
    selected_records = [profile for profile in selected_profiles if profile.get("selected_profile_type") is not None and profile.get("output_health_status") != "masked_unavailable"]
    masked_records = [profile for profile in selected_profiles if profile.get("selected_profile_type") is None or profile.get("output_health_status") == "masked_unavailable"]
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
        "grain": "asset_id x relationship_feature_family x band",
        "expected_cell_count": len(cells),
        "selected_profile_count": len(selected_records),
        "masked_or_skipped_cell_count": len(masked_records),
        "single_active_nonproduction_handoff_artifact": config.selected_profiles_filename,
        "stale_sandbox_manifest_used": False,
        "relationship_context_id": handoff.relationship_context_id,
        "relationship_context_handoff_path": str(config.handoff_path),
        "relationship_context_cadence_policy": handoff.cadence_policy_as_dict(),
        "scoring_schema_id": SCORING_SCHEMA_ID,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "window_policy_manifest": cross_asset_state_window_policy_manifest(),
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
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
    selected_asset_band_pairs = sorted({f"{cell['asset_id']}|{cell['band']}" for cell in cells if cell["cell_status"] == "selected_coherent"})
    masked_asset_band_pairs = sorted({f"{cell['asset_id']}|{cell['band']}" for cell in cells if cell["cell_status"] == "masked_unavailable"})
    selected_asset_count = len({str(cell["asset_id"]) for cell in cells if cell["cell_status"] == "selected_coherent"})
    valid_target_counts_by_band = _valid_target_counts_by_band(target_universe)
    valid_target_assets_across_bands = _valid_assets_across_bands(target_universe, bands=config.bands)
    coverage_limited = selected_asset_count < int(config.min_valid_assets_for_meaningful_sample)
    summary = {
        "artifact_kind": f"cross_asset_state_{config.artifact_label}_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "relationship_context_handoff_path": str(config.handoff_path),
        "feature_set_version": config.feature_set_version,
        "support_definition_ids": selected_payload["support_definition_ids"],
        "support_qualities": selected_payload["support_qualities"],
        "support_sizes": selected_payload["support_sizes"],
        "relationship_context_id": handoff.relationship_context_id,
        "relationship_context_cadence_policy": handoff.cadence_policy_as_dict(),
        "regime_feature_manifest_id": handoff.regime_feature_manifest_id,
        "sample_assets": list(sample_assets),
        "feature_sample_asset_count": len(valid_sample_assets),
        "valid_sample_asset_count": len(valid_sample_assets),
        "sample_asset_count_including_mask_probe": len(sample_assets),
        "valid_target_asset_counts_by_band": valid_target_counts_by_band,
        "valid_target_assets_across_bands_count": len(valid_target_assets_across_bands),
        "valid_target_assets_across_bands_sample": valid_target_assets_across_bands[:50],
        "meaningful_sample_min_valid_assets": int(config.min_valid_assets_for_meaningful_sample),
        "meaningful_sample_available": len(valid_target_assets_across_bands) >= int(config.min_valid_assets_for_meaningful_sample),
        "selected_asset_band_pair_count": len(selected_asset_band_pairs),
        "selected_asset_count": selected_asset_count,
        "coverage_limited_below_target": coverage_limited,
        "target_valid_asset_range": "20-50",
        "selected_asset_band_pairs": selected_asset_band_pairs,
        "masked_asset_band_pair_count": len(masked_asset_band_pairs),
        "masked_asset_band_pairs": masked_asset_band_pairs,
        "bands": list(config.bands),
        "relationship_feature_families": [family.name for family in families],
        "grain": "asset_id x relationship_feature_family x band",
        "profile_types_tested": list(PROTOTYPE_PROFILE_TYPES),
        "expected_cells": len(sample_assets) * len(config.bands) * len(families),
        "selected_cells": selected_count,
        "diagnostic_only_cells": diagnostic_only_count,
        "masked_unavailable_cells": masked_count,
        "missing_cells": 0,
        "cells_path": str(cells_path),
        "family_summary_path": str(family_path),
        "profile_type_comparison_path": str(comparison_path),
        "inspection_examples_path": str(inspection_path),
        "health_failures_path": str(health_failures_path),
        "label_distributions_path": str(labels_path),
        "open_items_path": str(open_items_path),
        "runtime_path": str(runtime_path),
        "sample_selection_path": str(sample_selection_path),
        "candidate_pool_path": str(candidate_pool_path),
        "selected_profiles_path": str(selected_profiles_path),
        "target_universe_policy": _target_universe_policy(),
        "profile_type_counts_by_family": _profile_type_counts_by_family(cells),
        "best_profile_type_by_family": {row["relationship_feature_family"]: row["best_profile_type"] for row in family_summary},
        "family_readiness": {row["relationship_feature_family"]: row["family_verdict"] for row in family_summary},
        "mask_reason_counts": _counts([str(cell.get("mask_reason")) for cell in cells if cell["cell_status"] == "masked_unavailable"]),
        "label_health_failure_count": sum(1 for row in comparisons if str(row.get("candidate_status")) == "candidate_failed_health"),
        "dominant_or_tiny_cluster_issue_count": _dominant_or_tiny_issue_count(comparisons),
        "health_failure_diagnosis": _health_failure_diagnosis(comparisons),
        "output_health_gate_calibration": _output_health_gate_calibration(),
        "scoring_schema": _scoring_schema_summary(),
        "maturity_contracts_implemented": _maturity_contracts_implemented(),
        "selected_profile_manifest_validation": selected_payload["selected_profile_manifest_validation"],
        "candidate_profile_universe": _candidate_profile_universe(),
        "window_lookback_policy_status": _window_lookback_policy_status(),
        "window_policy_manifest": cross_asset_state_window_policy_manifest(),
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
        "window_sensitivity_summary": family_window_sensitivity_summary(),
        "human_inspection": _human_inspection_summary(inspection_rows),
        "temporal_flicker_issue_count": sum(1 for cell in cells if float(cell.get("temporal_persistence_score") or 0.0) < 0.20 and cell["cell_status"] == "selected_coherent"),
        "economic_diagnostic_status": economic_contract.economic_diagnostic_status,
        "economic_diagnostic_contract": economic_contract.as_dict(),
        "economic_diagnostic_readiness": _economic_diagnostic_readiness(handoff),
        "runtime_telemetry": _runtime_summary(runtime_rows),
        "scale_estimate": _scale_estimate(
            cells=cells,
            comparisons=comparisons,
            discovered_asset_count=len(valid_target_assets_across_bands),
            elapsed_seconds=time.perf_counter() - started,
        ),
        "shape_only_verdict": _overall_verdict(family_summary),
        "major_family_issues": _major_family_issues(family_summary, comparisons, inspection_rows),
        "overall_verdict": (
            "B. Coverage still too narrow; Relationship Context handoff must be widened before method-quality inspection."
            if coverage_limited
            else _overall_verdict(family_summary)
        ),
        "final_verdict": _final_verdict(family_summary, inspection_rows, economic_contract.as_dict()),
        "exact_next_recommended_sprint": _next_recommended_sprint(family_summary, inspection_rows, economic_contract.as_dict()),
        "finalizer_artifact_path": str(summary_path),
        "parent_finalizer_artifact_writing": "single_summary_json_under_configured_output_root",
        "stale_sandbox_artifact_resolution_default": "blocked" if not config.allow_stale_sandbox_artifacts else "allowed_by_explicit_config",
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "safety_confirmation": {
            "production_writes": False,
            "production_labels": False,
            "canonical_production_state_outputs": False,
            "final_production_promotion": False,
            "broad_full_universe_cross_asset_state_campaign": False,
            "broad_pairwise_all_to_all_relationship_discovery": False,
            "canonical_parquet_root_rewrite": False,
            "cleanup_quarantine_delete_actions": False,
            "hardcoded_local_paths_introduced": False,
            "production_writer_gates_remained_fail_closed": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _load_handoff(path: str | Path) -> CrossAssetRelationshipContextHandoff:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_kind") != "cross_asset_relationship_context_handoff":
        raise ValueError("Cross-Asset-State prototype requires a relationship-context handoff")
    if payload.get("active_nonproduction_handoff") is not True or payload.get("production_enabled") is not False:
        raise ValueError("Cross-Asset relationship context handoff is not active non-production")
    return CrossAssetRelationshipContextHandoff(
        relationship_context_id=payload["relationship_context_id"],
        relationship_snapshot_roots=payload["relationship_snapshot_roots"],
        feature_roots=payload["feature_roots"],
        availability_sidecar_refs=payload["availability_sidecar_refs"],
        peer_metadata_refs=payload.get("peer_metadata_refs") or (),
        future_outcome_panel_refs=payload.get("future_outcome_panel_refs") or (),
        feature_set_version=payload.get("feature_set_version"),
        relationship_context_cadence_policy_id=payload.get("relationship_context_cadence_policy_id"),
        snapshot_cadence_days=payload.get("snapshot_cadence_days") or {},
        backfill_snapshot_schedule=payload.get("backfill_snapshot_schedule") or (),
        stale_snapshot_policy=payload.get("stale_snapshot_policy") or {},
        missing_snapshot_mask_reason=payload.get("missing_snapshot_mask_reason", "missing_relationship_snapshot"),
        no_future_graph_backfill=payload.get("no_future_graph_backfill", True),
        regime_feature_manifest_id=payload.get("regime_feature_manifest_id"),
        created_at_utc=payload.get("created_at_utc"),
    )


def build_cross_asset_state_valid_target_universe(
    handoff: CrossAssetRelationshipContextHandoff,
    resolver: CrossAssetRelationshipContextResolver,
    availability: Any | None,
    *,
    bands: Sequence[str],
    families: Sequence[Any],
    target_assets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    feature_assets_by_band = _discover_assets_by_band(handoff.feature_roots, bands=bands)
    if target_assets:
        assets = sorted(dict.fromkeys(str(asset) for asset in target_assets if str(asset).strip()))
    else:
        assets = sorted({asset for assets_for_band in feature_assets_by_band.values() for asset in assets_for_band})
    rows: list[dict[str, Any]] = []
    for asset in assets:
        for band in bands:
            feature_rows_present = asset in feature_assets_by_band.get(str(band), set())
            family_contexts = []
            for family in families:
                context = resolver.resolve(
                    asset_id=asset,
                    band=band,
                    ts=_relationship_resolution_ts(resolver, band),
                    feature_families=(family.name,),
                    feature_family_columns={family.name: family.required_columns},
                ).as_dict()
                family_contexts.append((family.name, context))
            available_families = [family for family, context in family_contexts if context.get("context_status") == "available"]
            mask_reasons = sorted(
                {
                    str(context.get("mask_reason"))
                    for _, context in family_contexts
                    if context.get("mask_reason") not in (None, "", "None")
                }
            )
            snapshot_context = next((context for _, context in family_contexts if context.get("relationship_snapshot_id")), family_contexts[0][1] if family_contexts else {})
            target_status = "valid_target" if feature_rows_present and len(available_families) == len(families) else "masked_unavailable"
            if not feature_rows_present:
                mask_reason = "missing_feature_rows"
            elif mask_reasons:
                mask_reason = "|".join(mask_reasons)
            elif target_status != "valid_target":
                mask_reason = "missing_required_field"
            else:
                mask_reason = ""
            rows.append(
                {
                    "asset_id": asset,
                    "band": band,
                    "target_status": target_status,
                    "mask_reason": mask_reason,
                    "feature_rows_present": feature_rows_present,
                    "relationship_snapshot_id": snapshot_context.get("relationship_snapshot_id"),
                    "known_at_ts": snapshot_context.get("known_at_ts"),
                    "source_tail_ts": snapshot_context.get("source_tail_ts"),
                    "available_family_count": len(available_families),
                    "required_family_count": len(families),
                    "relationship_feature_family_availability": json.dumps(
                        {family: context.get("context_status") for family, context in family_contexts},
                        sort_keys=True,
                    ),
                    "availability_sidecar_loaded": availability is not None and not getattr(availability, "empty", True),
                    "production_enabled": False,
                }
            )
    return rows


def select_cross_asset_state_sample(
    target_universe: Sequence[Mapping[str, Any]],
    *,
    bands: Sequence[str],
    max_valid_assets: int,
    max_masked_controls: int,
    masked_control_assets: Sequence[str],
    include_masked_probe_asset: bool,
) -> list[dict[str, Any]]:
    valid_assets = _valid_assets_across_bands(target_universe, bands=bands)
    ordered_valid = _ordered_target_assets(valid_assets)[: max(0, int(max_valid_assets))]
    rows: list[dict[str, Any]] = [
        _sample_plan_row(asset, sample_kind="valid_target", inclusion_reason=_sample_role(asset, idx))
        for idx, asset in enumerate(ordered_valid)
    ]
    selected = {row["asset_id"] for row in rows}
    masked_candidates = _masked_control_candidates(target_universe, selected_assets=selected)
    for asset in masked_candidates[: max(0, int(max_masked_controls))]:
        rows.append(_sample_plan_row(asset, sample_kind="masked_control", inclusion_reason="masked_control_from_candidate_pool"))
        selected.add(asset)
    if include_masked_probe_asset:
        for asset in masked_control_assets:
            cleaned = str(asset).strip()
            if cleaned and cleaned not in selected:
                rows.append(_sample_plan_row(cleaned, sample_kind="masked_control", inclusion_reason="deliberate_mask_control"))
                selected.add(cleaned)
    return rows


def _discover_assets(roots: Sequence[str | Path], *, bands: Sequence[str], max_assets: int) -> tuple[str, ...]:
    discovered: list[str] = []
    for root in roots:
        root_path = Path(root)
        for band in bands:
            for path in sorted((root_path / f"band={band}").glob("asset=*")):
                if path.is_dir() and "=" in path.name:
                    asset = path.name.split("=", 1)[1]
                    if asset not in discovered:
                        discovered.append(asset)
    priority = ("XBTUSD", "BTCUSD", "ETHUSD", "AAVEUSD", "ADAUSD", "SOLUSD", "XRPUSD", "DOGEUSD")
    ordered = [asset for asset in priority if asset in discovered]
    ordered.extend(asset for asset in discovered if asset not in ordered)
    return tuple(ordered[:max_assets])


def _discover_assets_by_band(roots: Sequence[str | Path], *, bands: Sequence[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {str(band): set() for band in bands}
    for root in roots:
        root_path = Path(root)
        for band in bands:
            band_key = str(band)
            for path in sorted((root_path / f"band={band_key}").glob("asset=*")):
                if path.is_dir() and "=" in path.name:
                    out.setdefault(band_key, set()).add(path.name.split("=", 1)[1])
    return out


def _relationship_resolution_ts(resolver: CrossAssetRelationshipContextResolver, band: str) -> float:
    snapshots = [
        snapshot
        for snapshot in resolver.snapshot_resolver.index.list_available_snapshots(band=band)
        if snapshot.has_causal_lineage
    ]
    if not snapshots:
        return 0.0
    return max(snapshot.known_at_order() for snapshot in snapshots) + 1.0


def _load_feature_frames(roots: Sequence[str | Path], assets: Sequence[str], bands: Sequence[str], *, max_files_per_asset: int) -> dict[tuple[str, str], Any]:
    pd = _pandas()
    out: dict[tuple[str, str], Any] = {}
    for root in roots:
        root_path = Path(root)
        for asset in assets:
            for band in bands:
                paths = sorted((root_path / f"band={band}" / f"asset={asset}").rglob("*.parquet"))[-max_files_per_asset:]
                if not paths:
                    continue
                frames = []
                for path in paths:
                    try:
                        frames.append(pd.read_parquet(path))
                    except Exception:
                        continue
                if frames:
                    out[(asset, band)] = pd.concat(frames, ignore_index=True).sort_values("ts")
    return out


def _assert_no_peer_identity_leak(feature_frames: Mapping[tuple[str, str], Any]) -> None:
    blocked = {"peer_group_id", "peer_id", "peer_asset", "peer_asset_id"}
    for frame in feature_frames.values():
        leaked = blocked.intersection({str(column) for column in frame.columns})
        if leaked:
            raise ValueError(f"Cross-Asset-State model-facing prototype input contains peer identity columns: {sorted(leaked)}")


def _resolution_ts(frame: Any | None) -> float:
    if frame is None or frame.empty or "known_at_ts" not in frame.columns:
        return 1780257601.0
    return float(frame["known_at_ts"].max()) + 1.0


def _masked_cell(
    asset: str,
    band: str,
    family: str,
    context: Mapping[str, Any],
    reason: str,
    *,
    window_policy: Any | None = None,
    window_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    window_fields = _window_fields(window_policy=window_policy, window_coverage=window_coverage)
    return {
        "asset_id": asset,
        "band": band,
        "relationship_feature_family": family,
        "cell_status": "masked_unavailable",
        "mask_reason": reason,
        "filter_reason_code": normalize_test_branch_filter_reason(reason),
        "selected_profile_type": None,
        "selected_method_family": None,
        "selected_candidate_id": None,
        "selected_parameter_grid_id": None,
        "profile_type": None,
        "method_family": None,
        "clusterer_family": None,
        "embedding": None,
        "readiness_status": None,
        "candidate_readiness_status": None,
        "selection_eligible": False,
        "selection_eligibility": False,
        "diagnostic_only": False,
        "selection_exclusion_reason": reason,
        "diagnostic_only_reason": reason,
        "shared_adapter_used": False,
        "adapter_name": None,
        "search_space_id": None,
        "split_policy_id": None,
        "validation_assignment_policy": None,
        "validation_assignment_status": None,
        "validation_assignment_scope": None,
        "validation_health": "{}",
        "train_row_count": None,
        "validation_row_count": None,
        "holdout_row_count": None,
        "method_universe_version": CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
        "profile_candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "candidate_params": "{}",
        "family_transform_id": None,
        "feature_diagnostics": "{}",
        "state_count": 0,
        "diagnostic_score": 0.0,
        "total_candidate_score": 0.0,
        "output_health_score": 0.0,
        "semantic_separation_score": 0.0,
        "coverage_score": 0.0,
        "temporal_stability_score": 0.0,
        "temporal_persistence_score": 0.0,
        "runtime_tiebreak": 0.0,
        "runtime_tiebreak_score": 0.0,
        "economic_diagnostic_score": None,
        "label_counts": "{}",
        "output_health_status": "masked_unavailable",
        "relationship_context_id": context.get("relationship_context_id"),
        "relationship_snapshot_id": context.get("relationship_snapshot_id"),
        "known_at_ts": context.get("known_at_ts"),
        "source_tail_ts": context.get("source_tail_ts"),
        "relationship_context_cadence_policy_id": context.get("relationship_context_cadence_policy_id"),
        "snapshot_cadence_days": context.get("snapshot_cadence_days"),
        "stale_snapshot_policy": json.dumps(context.get("stale_snapshot_policy") or {}, sort_keys=True),
        "no_future_graph_backfill": context.get("no_future_graph_backfill"),
        "snapshot_valid_from_ts": context.get("snapshot_valid_from_ts"),
        "snapshot_valid_until_ts": context.get("snapshot_valid_until_ts"),
        "economic_diagnostic_status": ECONOMIC_DIAGNOSTIC_PENDING,
        "shape_preserving": True,
        **window_fields,
    }


def _selected_cell(
    asset: str,
    band: str,
    family: str,
    context: Mapping[str, Any],
    candidate: Any,
    row_count: int,
    *,
    window_policy: Any | None = None,
    window_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    score = candidate.diagnostic_score.as_dict()
    health = candidate.output_health
    window_fields = _window_fields(window_policy=window_policy, window_coverage=window_coverage)
    return {
        "asset_id": asset,
        "band": band,
        "relationship_feature_family": family,
        "cell_status": "selected_coherent" if candidate.selected_status != "diagnostic_only" else "diagnostic_only",
        "mask_reason": candidate.failure_mode,
        "filter_reason_code": normalize_test_branch_filter_reason(
            getattr(candidate, "selection_exclusion_reason", None) or candidate.failure_mode
        ),
        "selected_profile_type": candidate.profile_type,
        "selected_method_family": getattr(candidate, "method_family", None) or _method_family(candidate.profile_type),
        "selected_candidate_id": getattr(candidate, "candidate_id", None),
        "selected_parameter_grid_id": getattr(candidate, "parameter_grid_id", None),
        "profile_type": candidate.profile_type,
        "method_family": getattr(candidate, "method_family", None),
        "clusterer_family": getattr(candidate, "clusterer_family", None),
        "embedding": getattr(candidate, "embedding", None),
        "shared_adapter_used": getattr(candidate, "shared_adapter_used", False),
        "adapter_name": getattr(candidate, "adapter_name", None),
        "search_space_id": getattr(candidate, "search_space_id", None) or getattr(candidate, "parameter_grid_id", None),
        "split_policy_id": getattr(candidate, "split_policy_id", None),
        "validation_assignment_policy": getattr(candidate, "validation_assignment_policy", None),
        "validation_assignment_status": getattr(candidate, "validation_assignment_status", None),
        "validation_assignment_scope": getattr(candidate, "validation_assignment_scope", None),
        "validation_health": json.dumps(getattr(candidate, "validation_health", {}) or {}, sort_keys=True),
        "train_row_count": getattr(candidate, "train_row_count", None),
        "validation_row_count": getattr(candidate, "validation_row_count", None),
        "holdout_row_count": getattr(candidate, "holdout_row_count", None),
        "method_universe_version": getattr(candidate, "method_universe_version", None),
        "profile_candidate_set_id": getattr(candidate, "profile_candidate_set_id", None),
        "readiness_status": getattr(candidate, "readiness_status", None),
        "candidate_readiness_status": getattr(candidate, "readiness_status", None),
        "selection_eligible": getattr(candidate, "selection_eligible", True),
        "selection_eligibility": getattr(candidate, "selection_eligible", True),
        "diagnostic_only": getattr(candidate, "diagnostic_only", False),
        "selection_exclusion_reason": getattr(candidate, "selection_exclusion_reason", None),
        "diagnostic_only_reason": getattr(candidate, "selection_exclusion_reason", None)
        if getattr(candidate, "diagnostic_only", False) or not getattr(candidate, "selection_eligible", True)
        else None,
        "filter_reason_code": normalize_test_branch_filter_reason(getattr(candidate, "selection_exclusion_reason", None)),
        "candidate_params": json.dumps(getattr(candidate, "candidate_params", {}) or {}, sort_keys=True),
        "family_transform_id": getattr(candidate, "family_transform_id", None),
        "feature_diagnostics": json.dumps(getattr(candidate, "feature_diagnostics", {}) or {}, sort_keys=True),
        "state_count": health.get("state_count", 0),
        "diagnostic_score": score["total_candidate_score"],
        "total_candidate_score": score["total_candidate_score"],
        "output_health_score": score["output_health_score"],
        "semantic_separation_score": score["semantic_separation_score"],
        "coverage_score": score["coverage_score"],
        "temporal_stability_score": score["temporal_stability_score"],
        "temporal_persistence_score": score["temporal_persistence_score"],
        "economic_diagnostic_score": score["economic_diagnostic_score"],
        "runtime_tiebreak": score["runtime_tiebreak_score"],
        "runtime_tiebreak_score": score["runtime_tiebreak_score"],
        "row_count": row_count,
        "label_counts": json.dumps(candidate.label_counts, sort_keys=True),
        "output_health_status": "passed" if health.get("passed") else "failed",
        "dominant_state_share": health.get("dominant_state_share"),
        "tiny_state_count": health.get("tiny_state_count"),
        "health_failure_reasons": "|".join(str(reason) for reason in health.get("failure_reasons") or ()),
        "health_warning_reasons": "|".join(str(reason) for reason in health.get("warning_reasons") or ()),
        "relationship_context_id": context.get("relationship_context_id"),
        "relationship_snapshot_id": context.get("relationship_snapshot_id"),
        "known_at_ts": context.get("known_at_ts"),
        "source_tail_ts": context.get("source_tail_ts"),
        "relationship_context_cadence_policy_id": context.get("relationship_context_cadence_policy_id"),
        "snapshot_cadence_days": context.get("snapshot_cadence_days"),
        "stale_snapshot_policy": json.dumps(context.get("stale_snapshot_policy") or {}, sort_keys=True),
        "no_future_graph_backfill": context.get("no_future_graph_backfill"),
        "snapshot_valid_from_ts": context.get("snapshot_valid_from_ts"),
        "snapshot_valid_until_ts": context.get("snapshot_valid_until_ts"),
        "economic_diagnostic_status": score["economic_diagnostic_status"],
        "shape_preserving": True,
        **window_fields,
    }


def _comparison_row(
    asset: str,
    band: str,
    family: str,
    candidate: Any,
    *,
    window_policy: Any | None = None,
    window_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    score = candidate.diagnostic_score.as_dict()
    window_fields = _window_fields(window_policy=window_policy, window_coverage=window_coverage)
    return {
        "asset_id": asset,
        "band": band,
        "relationship_feature_family": family,
        "profile_type": candidate.profile_type,
        "candidate_id": getattr(candidate, "candidate_id", None),
        "parameter_grid_id": getattr(candidate, "parameter_grid_id", None),
        "method_family": getattr(candidate, "method_family", None),
        "clusterer_family": getattr(candidate, "clusterer_family", None),
        "embedding": getattr(candidate, "embedding", None),
        "shared_adapter_used": getattr(candidate, "shared_adapter_used", False),
        "adapter_name": getattr(candidate, "adapter_name", None),
        "search_space_id": getattr(candidate, "search_space_id", None) or getattr(candidate, "parameter_grid_id", None),
        "split_policy_id": getattr(candidate, "split_policy_id", None),
        "validation_assignment_policy": getattr(candidate, "validation_assignment_policy", None),
        "validation_assignment_status": getattr(candidate, "validation_assignment_status", None),
        "validation_assignment_scope": getattr(candidate, "validation_assignment_scope", None),
        "validation_health": json.dumps(getattr(candidate, "validation_health", {}) or {}, sort_keys=True),
        "train_row_count": getattr(candidate, "train_row_count", None),
        "validation_row_count": getattr(candidate, "validation_row_count", None),
        "holdout_row_count": getattr(candidate, "holdout_row_count", None),
        "method_universe_version": getattr(candidate, "method_universe_version", None),
        "profile_candidate_set_id": getattr(candidate, "profile_candidate_set_id", None),
        "readiness_status": getattr(candidate, "readiness_status", None),
        "selection_eligible": getattr(candidate, "selection_eligible", True),
        "selection_eligibility": getattr(candidate, "selection_eligible", True),
        "diagnostic_only": getattr(candidate, "diagnostic_only", False),
        "selection_exclusion_reason": getattr(candidate, "selection_exclusion_reason", None),
        "diagnostic_only_reason": getattr(candidate, "selection_exclusion_reason", None)
        if getattr(candidate, "diagnostic_only", False) or not getattr(candidate, "selection_eligible", True)
        else None,
        "filter_reason_code": normalize_test_branch_filter_reason(getattr(candidate, "selection_exclusion_reason", None)),
        "candidate_params": json.dumps(getattr(candidate, "candidate_params", {}) or {}, sort_keys=True),
        "family_transform_id": getattr(candidate, "family_transform_id", None),
        "feature_diagnostics": json.dumps(getattr(candidate, "feature_diagnostics", {}) or {}, sort_keys=True),
        "candidate_status": candidate.selected_status,
        "failure_mode": candidate.failure_mode,
        "diagnostic_score": score["total_candidate_score"],
        "total_candidate_score": score["total_candidate_score"],
        "output_health_score": score["output_health_score"],
        "semantic_separation_score": score["semantic_separation_score"],
        "coverage_score": score["coverage_score"],
        "temporal_stability_score": score["temporal_stability_score"],
        "temporal_persistence_score": score["temporal_persistence_score"],
        "economic_diagnostic_status": score["economic_diagnostic_status"],
        "economic_diagnostic_score": score["economic_diagnostic_score"],
        "runtime_seconds": score["runtime_seconds"],
        "runtime_tiebreak": score["runtime_tiebreak_score"],
        "runtime_tiebreak_score": score["runtime_tiebreak_score"],
        "output_health_passed": candidate.output_health.get("passed"),
        "dominant_state_share": candidate.output_health.get("dominant_state_share"),
        "tiny_state_count": candidate.output_health.get("tiny_state_count"),
        "state_count": candidate.output_health.get("state_count"),
        "valid_row_count": candidate.output_health.get("valid_row_count"),
        "nonfinite_count": candidate.output_health.get("nonfinite_count"),
        "failure_reasons": "|".join(str(reason) for reason in candidate.output_health.get("failure_reasons") or ()),
        "warning_reasons": "|".join(str(reason) for reason in candidate.output_health.get("warning_reasons") or ()),
        "health_failure_type": candidate.output_health.get("failure_type"),
        "label_counts": json.dumps(candidate.label_counts, sort_keys=True),
        **window_fields,
    }


def _window_fields(*, window_policy: Any | None, window_coverage: Mapping[str, Any] | None) -> dict[str, Any]:
    policy = window_policy.as_dict() if hasattr(window_policy, "as_dict") else dict(window_policy or {})
    coverage = dict(window_coverage or {})
    return {
        "window_policy_id": policy.get("window_policy_id"),
        "window_profile_id": policy.get("window_profile_id") or policy.get("window_policy_id"),
        "window_candidate_name": policy.get("window_candidate_name"),
        "window_policy": json.dumps(policy, sort_keys=True) if policy else "{}",
        "window_coverage_status": coverage.get("status"),
        "window_coverage_passed": coverage.get("passed"),
        "window_observed_rows": coverage.get("observed_rows"),
        "window_required_min_rows": coverage.get("min_rows") or policy.get("min_rows"),
        "window_lookback_days": policy.get("lookback_days"),
        "window_start_ts": coverage.get("start_ts"),
        "window_end_ts": coverage.get("end_ts"),
    }


def _profile_record_from_selected_cell(cell: Mapping[str, Any], feature_columns: Sequence[str], handoff: CrossAssetRelationshipContextHandoff) -> dict[str, Any]:
    label_distribution = _loads_json_dict(cell.get("label_counts"))
    return {
        "asset_id": cell["asset_id"],
        "relationship_feature_family": cell["relationship_feature_family"],
        "band": cell["band"],
        "feature_set_version": cell.get("feature_set_version") or CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
        "support_definition_id": cell.get("support_definition_id") or "original_fixed_top3_v1",
        "support_size": cell.get("support_size"),
        "support_quality": cell.get("support_quality"),
        "support_rank_max": cell.get("support_rank_max"),
        "support_threshold": cell.get("support_threshold"),
        "support_fallback_path": cell.get("support_fallback_path"),
        "repaired_feature_manifest_id": cell.get("repaired_feature_manifest_id"),
        "profile_id": f"{cell['asset_id']}|{cell['relationship_feature_family']}|{cell['band']}|prototype_v1",
        "selected_profile_type": cell.get("selected_profile_type"),
        "selected_method_family": cell.get("selected_method_family"),
        "selected_candidate_id": cell.get("selected_candidate_id"),
        "selected_parameter_grid_id": cell.get("selected_parameter_grid_id"),
        "profile_type": cell.get("profile_type") or cell.get("selected_profile_type"),
        "method_family": cell.get("method_family") or cell.get("selected_method_family"),
        "clusterer_family": cell.get("clusterer_family"),
        "embedding": cell.get("embedding"),
        "shared_adapter_used": cell.get("shared_adapter_used"),
        "adapter_name": cell.get("adapter_name"),
        "search_space_id": cell.get("search_space_id") or cell.get("selected_parameter_grid_id"),
        "split_policy_id": cell.get("split_policy_id"),
        "validation_assignment_policy": cell.get("validation_assignment_policy"),
        "validation_assignment_status": cell.get("validation_assignment_status"),
        "validation_assignment_scope": cell.get("validation_assignment_scope"),
        "validation_health": _loads_json_mapping(cell.get("validation_health")),
        "train_row_count": cell.get("train_row_count"),
        "validation_row_count": cell.get("validation_row_count"),
        "holdout_row_count": cell.get("holdout_row_count"),
        "method_universe_version": cell.get("method_universe_version"),
        "profile_candidate_set_id": cell.get("profile_candidate_set_id"),
        "readiness_status": cell.get("readiness_status") or cell.get("candidate_readiness_status"),
        "candidate_readiness_status": cell.get("candidate_readiness_status"),
        "selection_eligible": cell.get("selection_eligible"),
        "selection_eligibility": cell.get("selection_eligibility", cell.get("selection_eligible")),
        "diagnostic_only": cell.get("diagnostic_only"),
        "selection_exclusion_reason": cell.get("selection_exclusion_reason"),
        "diagnostic_only_reason": cell.get("diagnostic_only_reason") or cell.get("selection_exclusion_reason"),
        "filter_reason_code": cell.get("filter_reason_code"),
        "candidate_params": _loads_json_mapping(cell.get("candidate_params")),
        "family_transform_id": cell.get("family_transform_id"),
        "feature_diagnostics": _loads_json_mapping(cell.get("feature_diagnostics")),
        "feature_columns_used": list(feature_columns),
        "window_policy_id": cell.get("window_policy_id"),
        "window_profile_id": cell.get("window_profile_id"),
        "window_candidate_name": cell.get("window_candidate_name"),
        "window_policy": _loads_json_mapping(cell.get("window_policy")),
        "window_coverage_status": cell.get("window_coverage_status"),
        "window_observed_rows": cell.get("window_observed_rows"),
        "window_required_min_rows": cell.get("window_required_min_rows"),
        "state_count": int(cell.get("state_count") or len(label_distribution)),
        "label_distribution": label_distribution,
        "output_health_status": cell.get("output_health_status"),
        "output_health_score": cell.get("output_health_score"),
        "dominant_state_share": cell.get("dominant_state_share"),
        "tiny_state_count": cell.get("tiny_state_count"),
        "semantic_score": cell.get("semantic_separation_score"),
        "semantic_separation_score": cell.get("semantic_separation_score"),
        "temporal_stability_score": cell.get("temporal_stability_score"),
        "temporal_score": cell.get("temporal_persistence_score"),
        "stability_temporal_score": cell.get("temporal_persistence_score"),
        "coverage_score": cell.get("coverage_score"),
        "economic_diagnostic_score": cell.get("economic_diagnostic_score"),
        "runtime_tiebreak": cell.get("runtime_tiebreak", cell.get("runtime_tiebreak_score")),
        "total_candidate_score": cell.get("diagnostic_score"),
        "scoring_schema_id": SCORING_SCHEMA_ID,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "economic_diagnostic_status": cell.get("economic_diagnostic_status"),
        "mask_reason": cell.get("mask_reason"),
        "relationship_context_id": handoff.relationship_context_id,
        "relationship_snapshot_id": cell.get("relationship_snapshot_id"),
        "source_tail_ts": cell.get("source_tail_ts"),
        "known_at_ts": cell.get("known_at_ts"),
        "relationship_context_cadence_policy_id": cell.get("relationship_context_cadence_policy_id")
        or handoff.relationship_context_cadence_policy_id,
        "snapshot_cadence_days": cell.get("snapshot_cadence_days") or handoff.cadence_days_for_band(str(cell["band"])),
        "stale_snapshot_policy": _loads_json_mapping(cell.get("stale_snapshot_policy")) or dict(handoff.stale_snapshot_policy),
        "no_future_graph_backfill": cell.get("no_future_graph_backfill")
        if cell.get("no_future_graph_backfill") is not None
        else handoff.no_future_graph_backfill,
        "snapshot_valid_from_ts": cell.get("snapshot_valid_from_ts"),
        "snapshot_valid_until_ts": cell.get("snapshot_valid_until_ts"),
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
    }


def _profile_record_from_masked_cell(cell: Mapping[str, Any], handoff: CrossAssetRelationshipContextHandoff) -> dict[str, Any]:
    return _profile_record_from_selected_cell(cell, (), handoff)


def _inspection_row(
    asset: str,
    band: str,
    family: str,
    context: Mapping[str, Any],
    candidate: Any,
    frame: Any,
    *,
    window_policy: Any | None = None,
    window_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = frame.loc[:, list(candidate.feature_columns)].copy()
    labels = list(candidate.labels)
    feature_summary = _feature_summary(data)
    centroids = _state_centroids(data, labels)
    examples = _example_rows(data, labels, limit=6)
    interpretation = _state_interpretation(family, centroids)
    between_state_separation = _between_state_separation(data, labels)
    score = candidate.diagnostic_score.as_dict()
    health = candidate.output_health
    agglomerative_interpretation = _agglomerative_interpretation(candidate, interpretation, between_state_separation, health=health)
    window_fields = _window_fields(window_policy=window_policy, window_coverage=window_coverage)
    return {
        "asset_id": asset,
        "relationship_feature_family": family,
        "band": band,
        "selected_profile_type": candidate.profile_type,
        "selected_method_family": getattr(candidate, "method_family", None) or _method_family(candidate.profile_type),
        "selected_candidate_id": getattr(candidate, "candidate_id", None),
        "selected_parameter_grid_id": getattr(candidate, "parameter_grid_id", None),
        "profile_type": candidate.profile_type,
        "method_family": getattr(candidate, "method_family", None),
        "clusterer_family": getattr(candidate, "clusterer_family", None),
        "embedding": getattr(candidate, "embedding", None),
        "shared_adapter_used": getattr(candidate, "shared_adapter_used", False),
        "adapter_name": getattr(candidate, "adapter_name", None),
        "search_space_id": getattr(candidate, "search_space_id", None) or getattr(candidate, "parameter_grid_id", None),
        "split_policy_id": getattr(candidate, "split_policy_id", None),
        "validation_assignment_policy": getattr(candidate, "validation_assignment_policy", None),
        "validation_assignment_status": getattr(candidate, "validation_assignment_status", None),
        "validation_assignment_scope": getattr(candidate, "validation_assignment_scope", None),
        "validation_health": json.dumps(getattr(candidate, "validation_health", {}) or {}, sort_keys=True),
        "train_row_count": getattr(candidate, "train_row_count", None),
        "validation_row_count": getattr(candidate, "validation_row_count", None),
        "holdout_row_count": getattr(candidate, "holdout_row_count", None),
        "readiness_status": getattr(candidate, "readiness_status", None),
        "candidate_readiness_status": getattr(candidate, "readiness_status", None),
        "selection_eligible": getattr(candidate, "selection_eligible", True),
        "selection_eligibility": getattr(candidate, "selection_eligible", True),
        "diagnostic_only": getattr(candidate, "diagnostic_only", False),
        "selection_exclusion_reason": getattr(candidate, "selection_exclusion_reason", None),
        "diagnostic_only_reason": getattr(candidate, "selection_exclusion_reason", None)
        if getattr(candidate, "diagnostic_only", False) or not getattr(candidate, "selection_eligible", True)
        else None,
        "candidate_params": json.dumps(getattr(candidate, "candidate_params", {}) or {}, sort_keys=True),
        "family_transform_id": getattr(candidate, "family_transform_id", None),
        "feature_diagnostics": json.dumps(getattr(candidate, "feature_diagnostics", {}) or {}, sort_keys=True),
        "state_count": health.get("state_count", 0),
        "label_distribution": json.dumps(candidate.label_counts, sort_keys=True),
        "dominant_state_share": health.get("dominant_state_share"),
        "tiny_state_count": health.get("tiny_state_count"),
        "output_health_status": "passed" if health.get("passed") else "failed",
        "output_health_score": score["output_health_score"],
        "semantic_separation_score": score["semantic_separation_score"],
        "temporal_stability_score": score["temporal_stability_score"],
        "coverage_score": score["coverage_score"],
        "total_candidate_score": score["total_candidate_score"],
        "runtime_tiebreak": score["runtime_tiebreak_score"],
        "runtime_tiebreak_score": score["runtime_tiebreak_score"],
        "economic_diagnostic_status": score["economic_diagnostic_status"],
        "economic_diagnostic_score": score["economic_diagnostic_score"],
        "semantic_score_components": json.dumps(
            {
                "semantic_separation_score": score["semantic_separation_score"],
                "temporal_stability_score": score["temporal_stability_score"],
                "coverage_score": score["coverage_score"],
                "output_health_score": score["output_health_score"],
            },
            sort_keys=True,
        ),
        "output_health_gate_values": json.dumps(
            {
                "state_count": health.get("state_count"),
                "dominant_state_share": health.get("dominant_state_share"),
                "tiny_state_count": health.get("tiny_state_count"),
                "valid_row_count": health.get("valid_row_count"),
                "nonfinite_count": health.get("nonfinite_count"),
                "failure_reasons": health.get("failure_reasons"),
                "warning_reasons": health.get("warning_reasons"),
            },
            sort_keys=True,
        ),
        "raw_feature_min_median_max": json.dumps(feature_summary, sort_keys=True),
        "state_centroids": json.dumps(centroids, sort_keys=True),
        "between_state_separation": between_state_separation,
        "example_input_rows_to_state": json.dumps(examples, sort_keys=True),
        "state_interpretation": interpretation["state_interpretation"],
        "coherence_status": interpretation["coherence_status"],
        "coherence_reason": interpretation["coherence_reason"],
        "agglomerative_interpretation_status": agglomerative_interpretation["status"],
        "agglomerative_interpretation_reason": agglomerative_interpretation["reason"],
        "mask_reason": None,
        "relationship_context_id": context.get("relationship_context_id"),
        "relationship_snapshot_id": context.get("relationship_snapshot_id"),
        "known_at_ts": context.get("known_at_ts"),
        "source_tail_ts": context.get("source_tail_ts"),
        **window_fields,
    }


def _feature_summary(data: Any) -> dict[str, dict[str, float | None]]:
    pd = _pandas()
    out: dict[str, dict[str, float | None]] = {}
    for column in data.columns:
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        if series.empty:
            out[str(column)] = {"min": None, "median": None, "max": None}
            continue
        out[str(column)] = {
            "min": round(float(series.min()), 6),
            "median": round(float(series.median()), 6),
            "max": round(float(series.max()), 6),
        }
    return out


def _state_centroids(data: Any, labels: Sequence[str]) -> dict[str, dict[str, float]]:
    pd = _pandas()
    if data.empty or not labels:
        return {}
    frame = data.copy()
    frame["__state_label"] = list(labels)
    out: dict[str, dict[str, float]] = {}
    for label, group in frame.groupby("__state_label"):
        out[str(label)] = {
            str(column): round(float(pd.to_numeric(group[column], errors="coerce").mean()), 6)
            for column in data.columns
        }
    return out


def _between_state_separation(data: Any, labels: Sequence[str]) -> float:
    pd = _pandas()
    if data.empty or len(set(labels)) < 2:
        return 0.0
    frame = data.copy()
    frame["__state_label"] = list(labels)
    spreads: list[float] = []
    for column in data.columns:
        grouped = frame.groupby("__state_label")[column].mean()
        if len(grouped) < 2:
            continue
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        denom = float(series.max() - series.min()) if not series.empty else 0.0
        if denom <= 0.0:
            continue
        spreads.append(float((grouped.max() - grouped.min()) / denom))
    if not spreads:
        return 0.0
    return round(max(0.0, min(1.0, sum(spreads) / len(spreads))), 6)


def _agglomerative_interpretation(candidate: Any, interpretation: Mapping[str, str], separation: float, health: Mapping[str, Any]) -> dict[str, str]:
    if getattr(candidate, "profile_type", None) != "agglomerative":
        return {"status": "not_agglomerative", "reason": "Selected profile is not agglomerative."}
    if not health.get("passed"):
        return {"status": "failed", "reason": "Agglomerative output did not pass output-health gates."}
    dominant = float(health.get("dominant_state_share") or 1.0)
    tiny = int(health.get("tiny_state_count") or 0)
    if tiny > 0 or dominant > 0.90:
        return {"status": "questionable", "reason": "Agglomerative state distribution has tiny or dominant-state artifact risk."}
    if str(interpretation.get("coherence_status")) == "inspectable" and float(separation) >= 0.15:
        return {"status": "coherent", "reason": "Agglomerative states separate along family axes and pass distribution gates."}
    if float(separation) < 0.08:
        return {"status": "failed", "reason": "Agglomerative state centroids have too little between-state separation."}
    return {"status": "questionable", "reason": "Agglomerative states separate weakly or need human inspection."}


def _example_rows(data: Any, labels: Sequence[str], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(data.head(limit).iterrows()):
        values = {str(column): _round_or_none(row[column]) for column in data.columns}
        rows.append({"row_offset": idx, "input_relationship_values": values, "assigned_state": str(labels[idx]) if idx < len(labels) else None})
    return rows


def _state_interpretation(family: str, centroids: Mapping[str, Mapping[str, float]]) -> dict[str, str]:
    if not centroids:
        return {
            "state_interpretation": "not_inspectable_no_state_centroids",
            "coherence_status": "questionable",
            "coherence_reason": "No state centroid values were available.",
        }
    scored = _state_semantic_scores(family, centroids)
    if not scored:
        return {
            "state_interpretation": "questionable_family_axes_unavailable",
            "coherence_status": "questionable",
            "coherence_reason": "The selected feature columns do not support the requested family interpretation.",
        }
    ordered = sorted(scored.items(), key=lambda item: item[1])
    spread = ordered[-1][1] - ordered[0][1]
    if spread < 0.05:
        return {
            "state_interpretation": json.dumps({label: "questionable_low_separation" for label, _ in ordered}, sort_keys=True),
            "coherence_status": "questionable",
            "coherence_reason": "State centroids are too close to support a clear semantic label.",
        }
    labels = _semantic_labels_for_family(family, ordered)
    return {
        "state_interpretation": json.dumps(labels, sort_keys=True),
        "coherence_status": "inspectable" if spread >= 0.15 else "questionable",
        "coherence_reason": "State centroids separate along the expected family axis." if spread >= 0.15 else "State centroids separate weakly; inspect before relying on labels.",
    }


def _state_semantic_scores(family: str, centroids: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, values in centroids.items():
        if family == "anchor_core_exposure":
            cols = ("corr_to_anchor_primary", "corr_to_anchor_secondary", "corr_to_core_basket", "beta_to_core_basket")
            available = [abs(float(values[col])) for col in cols if col in values]
            if available:
                out[label] = sum(available) / len(available)
        elif family == "peer_strength_stability":
            strength = float(values.get("strongest_peer_slot_1_strength", 0.0))
            stability = float(values.get("top_peer_stability_mean", 0.0))
            count = float(values.get("top_peer_count", 0.0))
            out[label] = (strength + stability + min(count / 8.0, 1.0)) / 3.0
        elif family == "relationship_concentration_entropy":
            concentration = float(values.get("relationship_concentration", 0.0))
            entropy = float(values.get("relationship_entropy", 0.0))
            out[label] = concentration - entropy
        elif family == "residual_peer_signal" and "residual_peer_signal_score" in values:
            out[label] = float(values["residual_peer_signal_score"])
    return out


def _semantic_labels_for_family(family: str, ordered_scores: Sequence[tuple[str, float]]) -> dict[str, str]:
    if family == "anchor_core_exposure":
        names = ("low_core_coupling", "moderate_core_coupling", "high_core_coupling")
    elif family == "peer_strength_stability":
        names = ("weak_or_noisy_peer_structure", "moderate_peer_structure", "strong_stable_peer_basket")
    elif family == "relationship_concentration_entropy":
        names = ("diffuse_relationship_exposure", "mixed_dependency", "concentrated_dependency")
    elif family == "residual_peer_signal":
        names = ("negative_residual_peer_signal", "neutral_residual_peer_signal", "positive_residual_peer_signal")
    else:
        names = ("low_state", "middle_state", "high_state")
    if len(ordered_scores) == 1:
        return {ordered_scores[0][0]: names[1]}
    labels: dict[str, str] = {}
    for idx, (label, _) in enumerate(ordered_scores):
        if idx == 0:
            labels[label] = names[0]
        elif idx == len(ordered_scores) - 1:
            labels[label] = names[2]
        else:
            labels[label] = names[1]
    return labels


def _manual_sample_plan_rows(
    sample_assets: Sequence[str],
    *,
    target_universe: Sequence[Mapping[str, Any]],
    include_masked_probe_asset: bool,
) -> list[dict[str, Any]]:
    if not sample_assets:
        return []
    valid_assets = set(_valid_assets_across_bands(target_universe, bands=sorted({str(row.get("band")) for row in target_universe})))
    rows: list[dict[str, Any]] = []
    selected: set[str] = set()
    for asset in sample_assets:
        cleaned = str(asset).strip()
        if not cleaned or cleaned in selected:
            continue
        selected.add(cleaned)
        kind = "valid_target" if cleaned in valid_assets else "manual_or_masked_target"
        rows.append(_sample_plan_row(cleaned, sample_kind=kind, inclusion_reason="manual_sample_asset"))
    if include_masked_probe_asset and "MISSING_ASSET_USD" not in selected:
        rows.append(_sample_plan_row("MISSING_ASSET_USD", sample_kind="masked_control", inclusion_reason="deliberate_mask_control"))
    return rows


def _valid_target_counts_by_band(target_universe: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for band in sorted({str(row.get("band")) for row in target_universe}):
        out[band] = len({str(row.get("asset_id")) for row in target_universe if str(row.get("band")) == band and row.get("target_status") == "valid_target"})
    return out


def _valid_assets_across_bands(target_universe: Sequence[Mapping[str, Any]], *, bands: Sequence[str]) -> list[str]:
    required_bands = {str(band) for band in bands}
    by_asset: dict[str, set[str]] = {}
    for row in target_universe:
        if row.get("target_status") != "valid_target":
            continue
        by_asset.setdefault(str(row.get("asset_id")), set()).add(str(row.get("band")))
    return sorted(asset for asset, asset_bands in by_asset.items() if required_bands.issubset(asset_bands))


def _ordered_target_assets(assets: Sequence[str]) -> list[str]:
    available = list(dict.fromkeys(str(asset) for asset in assets if str(asset).strip()))
    priority = [asset for asset in DEFAULT_TARGET_SAMPLE_PRIORITY if asset in available]
    remaining = [asset for asset in sorted(available) if asset not in priority]
    return [*priority, *remaining]


def _masked_control_candidates(target_universe: Sequence[Mapping[str, Any]], *, selected_assets: set[str]) -> list[str]:
    candidates = sorted(
        {
            str(row.get("asset_id"))
            for row in target_universe
            if row.get("target_status") != "valid_target" and str(row.get("asset_id")) not in selected_assets
        }
    )
    return candidates


def _sample_plan_row(asset: str, *, sample_kind: str, inclusion_reason: str) -> dict[str, Any]:
    return {
        "asset_id": asset,
        "sample_kind": sample_kind,
        "planned_inclusion_reason": inclusion_reason,
        "production_enabled": False,
    }


def _sample_role(asset: str, index: int) -> str:
    if asset in DEFAULT_TARGET_SAMPLE_PRIORITY:
        return "liquid_or_major_like_valid_target"
    if index % 3 == 0:
        return "broad_valid_target"
    if index % 3 == 1:
        return "mid_universe_valid_target"
    return "volatile_or_tail_valid_target"


def _target_universe_policy() -> dict[str, Any]:
    return {
        "downstream_output_shape": "asset_id x relationship_feature_family x band",
        "all_expected_cells_represented": "selected_or_masked",
        "assets_not_in_relationship_snapshot": "masked_unavailable_with_explicit_reason",
        "target_eligibility": "per_band_per_snapshot",
        "anchors_or_core_assets": "valid_targets_when_relationship_context_and_model_facing_rows_are_available; otherwise masked",
        "peer_only_assets": "valid_targets_when present in relationship snapshot and required family values are available",
        "transient_unavailable_assets": "shape_preserving_masked_rows_not_dropped",
        "sample_source": "valid targets resolved from active relationship context, not guessed from major assets",
    }


def _sample_selection_rows(
    sample_assets: Sequence[str],
    valid_sample_assets: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    feature_frames: Mapping[tuple[str, str], Any],
    sample_plan_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_assets = {str(cell["asset_id"]) for cell in cells if cell["cell_status"] == "selected_coherent"}
    valid_sample_set = {str(asset) for asset in valid_sample_assets}
    plan_by_asset = {str(row.get("asset_id")): dict(row) for row in sample_plan_rows}
    for asset in sample_assets:
        asset_cells = [cell for cell in cells if str(cell["asset_id"]) == str(asset)]
        mask_reasons = _counts([str(cell.get("mask_reason")) for cell in asset_cells if cell["cell_status"] == "masked_unavailable"])
        bands_available = sorted({band for (frame_asset, band), frame in feature_frames.items() if frame_asset == asset and frame is not None})
        plan = plan_by_asset.get(str(asset), {})
        if asset in selected_assets:
            reason = "selected_valid_relationship_context"
        elif asset in valid_sample_set:
            reason = "feature_rows_present_but_relationship_context_masked"
        else:
            reason = "deliberate_mask_control"
        rows.append(
            {
                "asset_id": asset,
                "inclusion_reason": reason,
                "planned_sample_kind": plan.get("sample_kind"),
                "planned_inclusion_reason": plan.get("planned_inclusion_reason"),
                "bands_available": "|".join(bands_available),
                "relationship_snapshot_coverage": "selected" if asset in selected_assets else "masked_unavailable",
                "mask_reason_counts": json.dumps(mask_reasons, sort_keys=True),
            }
        )
    return rows


def _runtime_rows(cells: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]], started: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "aggregate_scope": "overall",
            "key": "all",
            "cell_count": len(cells),
            "candidate_count": len(comparisons),
            "selected_count": sum(1 for cell in cells if cell["cell_status"] == "selected_coherent"),
            "masked_count": sum(1 for cell in cells if cell["cell_status"] == "masked_unavailable"),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "memory_rss_bytes": _rss_bytes(),
        }
    ]
    for scope, key_name in (("family", "relationship_feature_family"), ("candidate_type", "profile_type"), ("band", "band")):
        keys = sorted({str(row.get(key_name)) for row in comparisons if row.get(key_name) is not None})
        for key in keys:
            scoped = [row for row in comparisons if str(row.get(key_name)) == key]
            rows.append(
                {
                    "aggregate_scope": scope,
                    "key": key,
                    "cell_count": "",
                    "candidate_count": len(scoped),
                    "selected_count": "",
                    "masked_count": "",
                    "runtime_seconds": round(sum(float(row.get("runtime_seconds") or 0.0) for row in scoped), 6),
                    "memory_rss_bytes": "",
                }
            )
    return rows


def _runtime_summary(runtime_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = next((row for row in runtime_rows if row.get("aggregate_scope") == "overall"), {})
    return {
        "total_cells": overall.get("cell_count", 0),
        "candidate_profile_count": overall.get("candidate_count", 0),
        "selected_count": overall.get("selected_count", 0),
        "masked_count": overall.get("masked_count", 0),
        "wall_time_seconds": overall.get("runtime_seconds", 0.0),
        "memory_rss_bytes": overall.get("memory_rss_bytes"),
    }


def _health_failure_rows(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in comparisons:
        if str(row.get("candidate_status")) != "candidate_failed_health":
            continue
        reasons = str(row.get("failure_reasons") or row.get("failure_mode") or "output_health_gate_failed")
        split_reasons = [reason for reason in reasons.split("|") if reason]
        for reason in split_reasons or ["output_health_gate_failed"]:
            rows.append(
                {
                    "asset_id": row.get("asset_id"),
                    "band": row.get("band"),
                    "relationship_feature_family": row.get("relationship_feature_family"),
                    "profile_type": row.get("profile_type"),
                    "failure_reason": reason,
                    "failure_mode": row.get("failure_mode"),
                    "warning_reasons": row.get("warning_reasons"),
                    "dominant_state_share": row.get("dominant_state_share"),
                    "tiny_state_count": row.get("tiny_state_count"),
                    "state_count": row.get("state_count"),
                    "valid_row_count": row.get("valid_row_count"),
                    "nonfinite_count": row.get("nonfinite_count"),
                    "label_counts": row.get("label_counts"),
                }
            )
    return rows


def _health_failure_diagnosis(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [row for row in comparisons if str(row.get("candidate_status")) == "candidate_failed_health"]
    warning_rows = [row for row in comparisons if str(row.get("warning_reasons") or "").strip()]
    reason_rows = _health_failure_rows(failed)
    return {
        "candidate_health_failure_count": len(failed),
        "failure_counts_by_family": _counts([str(row.get("relationship_feature_family")) for row in failed]),
        "failure_counts_by_profile_type": _counts([str(row.get("profile_type")) for row in failed]),
        "failure_counts_by_band": _counts([str(row.get("band")) for row in failed]),
        "failure_counts_by_asset_top20": _top_counts([str(row.get("asset_id")) for row in failed], limit=20),
        "failure_reason_counts": _counts([str(row.get("failure_reason")) for row in reason_rows]),
        "warning_reason_counts": _warning_reason_counts(warning_rows),
        "classification_policy": {
            "dominant_state_failure": "valid_hard_failure_when_share_exceeds_0.95",
            "tiny_state_failure": "valid_hard_failure_when_any_state_has_share_below_0.03",
            "all_one_state_collapse": "valid_hard_failure",
            "all_mask_output": "valid_hard_failure",
            "insufficient_valid_rows": "valid_hard_failure",
            "nonfinite_values": "valid_hard_failure",
            "birch_too_few_subclusters": "warning_if_health_passes; profile_type_specific_hard_failure_when_it_causes_single_state_or_insufficient_states",
        },
    }


def _output_health_gate_calibration() -> dict[str, Any]:
    return {
        "max_dominant_state_share": 0.95,
        "minimum_state_count": 2,
        "minimum_valid_rows": 8,
        "tiny_state_share_threshold": 0.03,
        "rare_state_exception_policy": "none_for_v1_diagnostic; rare states stay hard failures until economic diagnostics justify an exception",
        "birch_warning_handling": "captured as warning; selected only if normal output-health gates still pass",
        "family_specific_gate_differences": "none_yet; use common gate for comparability in bounded v1 diagnostics",
        "runtime_policy": "runtime_tiebreak_only_not_part_of_total_candidate_score",
    }


def _scoring_schema_summary() -> dict[str, Any]:
    return {
        "scoring_schema_id": SCORING_SCHEMA_ID,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "components": [
            "output_health_score",
            "semantic_separation_score",
            "temporal_stability_score",
            "coverage_score",
            "economic_diagnostic_status",
            "economic_diagnostic_score",
            "runtime_tiebreak_score",
            "total_candidate_score",
        ],
        "total_score_policy": "hard_health_failures_score_zero; economic_score_excluded_until_computed; runtime_is_tiebreak_only",
        "production_enabled": False,
    }


def _maturity_contracts_implemented() -> dict[str, Any]:
    return {
        "explicit_profile_grain_validator": "asset_id x relationship_feature_family x band",
        "selected_profile_manifest_schema": "cross_asset_state_selected_profiles_mature_nonprod",
        "single_active_nonproduction_handoff_artifact": True,
        "mask_unavailable_cell_artifact": True,
        "output_health_gates": _output_health_gate_calibration(),
        "scoring_schema": _scoring_schema_summary(),
        "window_lookback_policy": "cross_asset_state_window_policy_v1",
        "economic_diagnostic_contract": "cross_asset_state_economic_diagnostic_contract",
        "source_tail_known_at_lineage": True,
        "relationship_context_snapshot_lineage": True,
        "production_approval_flags_false": True,
        "production_consumer_fail_closed": True,
        "parent_single_finalizer_artifact": True,
        "stale_sandbox_artifact_resolution_by_default": "blocked",
    }


def _candidate_profile_universe() -> dict[str, Any]:
    method_manifest = cross_asset_state_method_universe_manifest()
    specs = {
        str(spec["profile_type"]): spec
        for spec in method_manifest.get("method_specs", [])
        if isinstance(spec, dict)
    }
    return {
        "method_universe_version": CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
        "profile_candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "method_universe_validation": validate_cross_asset_state_method_universe_manifest(method_manifest),
        "selectable_profile_types": list(method_manifest["selectable_profile_types"]),
        "fallback_profile_types": list(method_manifest["fallback_profile_types"]),
        "density_profile_types": {
            "hdbscan": specs.get("hdbscan", {}).get("readiness_status", "diagnostic_only_recommended"),
            "optics": specs.get("optics", {}).get("readiness_status", "diagnostic_only_recommended"),
        },
        "readiness_status_by_profile_type": {
            profile_type: str(spec.get("readiness_status")) for profile_type, spec in sorted(specs.items())
        },
        "parameterization_status_by_profile_type": {
            profile_type: str(spec.get("parameterization_status")) for profile_type, spec in sorted(specs.items())
        },
        "early_pruning": False,
        "birch_warning_policy": "captured; warning-only unless normal output health gates fail",
        "birch_readiness_status": specs.get("birch", {}).get("readiness_status"),
    }


def _window_lookback_policy_status() -> dict[str, Any]:
    return {
        "status": "implemented_and_visible_in_profile_artifacts",
        "policy_set_id": "cross_asset_state_window_policy_v1",
        "dataset_builder_uses_policy": True,
        "insufficient_window_history_masks_enabled": True,
        "bounded_diagnostic_comparison_scope": "active widened 40-valid-target handoff sample",
        "current_default_documented": True,
        "full_window_grid_testing_status": "deferred_to_next_bounded_campaign_design",
    }


def _economic_diagnostic_readiness(handoff: CrossAssetRelationshipContextHandoff) -> dict[str, Any]:
    return {
        "economic_diagnostic_status": ECONOMIC_DIAGNOSTIC_PENDING,
        "economic_diagnostic_score": None,
        "safe_to_compute_now": False,
        "reason": "No leakage-safe future outcome panel refs or horizon/alignment contract were provided in the relationship context handoff.",
        "required_inputs": [
            "future_relative_return_vs_anchor_or_core_basket_panel",
            "future_realized_volatility_panel",
            "future_drawdown_panel",
            "future_beta_or_correlation_shift_panel",
            "known_at_ts_and_source_tail_ts_alignment_rules",
            "horizon_and_window_contract",
        ],
        "handoff_has_future_outcome_refs": bool(getattr(handoff, "future_outcome_panel_refs", ())),
    }


def _major_family_issues(
    family_summary: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    inspection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    questionable_by_family = _counts(
        [
            str(row.get("relationship_feature_family"))
            for row in inspection_rows
            if str(row.get("coherence_status")) == "questionable"
        ]
    )
    failure_by_family = _counts(
        [
            str(row.get("relationship_feature_family"))
            for row in comparisons
            if str(row.get("candidate_status")) == "candidate_failed_health"
        ]
    )
    selected_best = {str(row["relationship_feature_family"]): str(row.get("best_profile_type")) for row in family_summary}
    return {
        "anchor_core_exposure": {
            "status": "usable_but_watch_primary_correlation_saturation",
            "best_profile_type": selected_best.get("anchor_core_exposure"),
            "candidate_health_failures": failure_by_family.get("anchor_core_exposure", 0),
            "questionable_inspection_examples": questionable_by_family.get("anchor_core_exposure", 0),
        },
        "peer_strength_stability": {
            "status": "questionable_requires_family_repair",
            "best_profile_type": selected_best.get("peer_strength_stability"),
            "candidate_health_failures": failure_by_family.get("peer_strength_stability", 0),
            "questionable_inspection_examples": questionable_by_family.get("peer_strength_stability", 0),
            "repair_direction": "rank/ordinal stability and window sensitivity need further bounded calibration",
        },
        "relationship_concentration_entropy": {
            "status": "questionable_diagnostic_only_until_spread_validates",
            "best_profile_type": selected_best.get("relationship_concentration_entropy"),
            "candidate_health_failures": failure_by_family.get("relationship_concentration_entropy", 0),
            "questionable_inspection_examples": questionable_by_family.get("relationship_concentration_entropy", 0),
            "repair_direction": "spread/rank scaling works as scaffold but needs economic and window confirmation",
        },
        "residual_peer_signal": {
            "status": "positive_control_usable_for_scoring_diagnostics",
            "best_profile_type": selected_best.get("residual_peer_signal"),
            "candidate_health_failures": failure_by_family.get("residual_peer_signal", 0),
            "questionable_inspection_examples": questionable_by_family.get("residual_peer_signal", 0),
        },
    }


def _final_verdict(
    family_summary: Sequence[Mapping[str, Any]],
    inspection_rows: Sequence[Mapping[str, Any]],
    economic_contract: Mapping[str, Any],
) -> str:
    if economic_contract.get("economic_diagnostic_status") != "computed":
        questionable = {
            str(row.get("relationship_feature_family"))
            for row in inspection_rows
            if str(row.get("coherence_status")) == "questionable"
        }
        if {"peer_strength_stability", "relationship_concentration_entropy"}.intersection(questionable):
            return "B. Mature scaffold works, but specific families need repair before larger campaign."
        return "C. Economic diagnostics block further confidence."
    blocked = [row for row in family_summary if row.get("family_verdict") != "ready_for_v1"]
    if blocked:
        return "B. Mature scaffold works, but specific families need repair before larger campaign."
    return "A. Cross-Asset-State Test Branch is now mature enough for larger bounded campaign design."


def _next_recommended_sprint(
    family_summary: Sequence[Mapping[str, Any]],
    inspection_rows: Sequence[Mapping[str, Any]],
    economic_contract: Mapping[str, Any],
) -> str:
    questionable = sorted(
        {
            str(row.get("relationship_feature_family"))
            for row in inspection_rows
            if str(row.get("coherence_status")) == "questionable"
        }
    )
    if questionable:
        return (
            "Run a bounded family-repair sprint for "
            + ", ".join(questionable)
            + " with window-policy A/B checks, then wire a leakage-safe future outcome panel contract."
        )
    if economic_contract.get("economic_diagnostic_status") != "computed":
        return "Implement leakage-safe Cross-Asset-State economic outcome panels and rerun the same 40-target mature sandbox."
    return "Design a larger bounded campaign using the mature non-production manifest and explicit window policies."


def _human_inspection_summary(inspection_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "inspection_example_count": len(inspection_rows),
        "examples_by_family": _counts([str(row.get("relationship_feature_family")) for row in inspection_rows]),
        "coherence_status_counts": _counts([str(row.get("coherence_status")) for row in inspection_rows]),
    }


def _scale_estimate(
    *,
    cells: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    discovered_asset_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    selected_assets = {str(cell["asset_id"]) for cell in cells if cell["cell_status"] == "selected_coherent"}
    selected_asset_count = max(1, len(selected_assets))
    seconds_per_selected_asset = elapsed_seconds / selected_asset_count
    estimated_seconds = seconds_per_selected_asset * max(discovered_asset_count, selected_asset_count)
    return {
        "observed_selected_asset_count": len(selected_assets),
        "active_handoff_feature_asset_count": discovered_asset_count,
        "seconds_per_selected_asset_observed": round(seconds_per_selected_asset, 6),
        "estimated_seconds_for_active_handoff_feature_assets": round(estimated_seconds, 6),
        "runtime_appears_manageable_for_active_handoff_surface": bool(estimated_seconds < 600),
        "candidate_profile_types_need_narrowing_before_scale": bool(_dominant_or_tiny_issue_count(comparisons) > 0),
        "worker_orchestration_needed_before_full_universe": bool(estimated_seconds >= 600 or discovered_asset_count > 50),
        "estimate_note": "Estimate is limited to active non-production handoff feature assets, not a full production universe.",
    }


def _family_summary(cells: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = sorted({str(cell["relationship_feature_family"]) for cell in cells})
    for family in families:
        fam_cells = [cell for cell in cells if cell["relationship_feature_family"] == family]
        selected = [cell for cell in fam_cells if cell["cell_status"] == "selected_coherent"]
        diagnostic_only = [cell for cell in fam_cells if cell["cell_status"] == "diagnostic_only"]
        masked = [cell for cell in fam_cells if cell["cell_status"] == "masked_unavailable"]
        fam_comp = [row for row in comparisons if row["relationship_feature_family"] == family]
        best_type = _most_common([str(cell["selected_profile_type"]) for cell in selected if cell.get("selected_profile_type")])
        avg_score = sum(float(cell["diagnostic_score"]) for cell in selected) / len(selected) if selected else 0.0
        avg_semantic = sum(float(cell["semantic_separation_score"]) for cell in selected) / len(selected) if selected else 0.0
        avg_temporal = sum(float(cell["temporal_persistence_score"]) for cell in selected) / len(selected) if selected else 0.0
        verdict = "ready_for_v1" if selected and len(selected) >= len(fam_cells) / 3 else "diagnostic_only_or_blocked"
        out.append(
            {
                "relationship_feature_family": family,
                "expected_cells": len(fam_cells),
                "selected_cells": len(selected),
                "diagnostic_only_cells": len(diagnostic_only),
                "masked_unavailable_cells": len(masked),
                "missing_cells": 0,
                "best_profile_type": best_type,
                "avg_diagnostic_score": round(avg_score, 6),
                "avg_semantic_separation_score": round(avg_semantic, 6),
                "avg_temporal_persistence_score": round(avg_temporal, 6),
                "profile_types_tested": "|".join(sorted({str(row["profile_type"]) for row in fam_comp} | {"diagnostic_only"})),
                "profile_type_counts": json.dumps(_counts([str(cell.get("selected_profile_type")) for cell in selected]), sort_keys=True),
                "mask_reason_counts": json.dumps(_counts([str(cell.get("mask_reason")) for cell in masked]), sort_keys=True),
                "economic_diagnostic_status": "pending_not_computed",
                "family_verdict": verdict,
            }
        )
    return out


def _open_items(family_summary: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "severity": "medium",
            "item": "economic downstream diagnostics not computed",
            "evidence": "No leakage-safe forward outcome panel was provided to the prototype.",
            "recommendation": "Define leakage-safe economic diagnostics before final scoring contract promotion.",
        }
    ]
    failures = sum(1 for row in comparisons if str(row.get("candidate_status")) == "candidate_failed_health")
    if failures:
        rows.append(
            {
                "severity": "medium",
                "item": "candidate profile health failures observed",
                "evidence": str(failures),
                "recommendation": "Inspect dominant-state and tiny-state diagnostics before widening the sample.",
            }
        )
    blocked = [row for row in family_summary if row["family_verdict"] != "ready_for_v1"]
    if blocked:
        rows.append(
            {
                "severity": "high",
                "item": "one or more families are diagnostic-only or blocked",
                "evidence": "|".join(str(row["relationship_feature_family"]) for row in blocked),
                "recommendation": "Keep blocked families out of model-facing labels until scoring/profile repair passes.",
            }
        )
    return rows


def _overall_verdict(family_summary: Sequence[Mapping[str, Any]]) -> str:
    ready = [row for row in family_summary if row["family_verdict"] == "ready_for_v1"]
    if len(ready) == len(family_summary):
        return "A. Cross-Asset-State v1 prototype is coherent; proceed to formal scoring contract hardening."
    if ready:
        return "B. V1 shape is coherent, but specific families need scoring/profile repair."
    return "E. Input data is insufficient for meaningful v1."


def _method_family(profile_type: object) -> str | None:
    if profile_type is None:
        return None
    text = str(profile_type)
    if text in {"rule_threshold", "ordinal_quantile", "diagnostic_only"}:
        return text
    return "sklearn_cluster"


def _profile_type_counts_by_family(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cell in cells:
        if cell["cell_status"] != "selected_coherent":
            continue
        family = str(cell["relationship_feature_family"])
        profile_type = str(cell.get("selected_profile_type"))
        out.setdefault(family, {})
        out[family][profile_type] = out[family].get(profile_type, 0) + 1
    return out


def _dominant_or_tiny_issue_count(comparisons: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in comparisons
        if float(row.get("dominant_state_share") or 0.0) > 0.95 or int(row.get("tiny_state_count") or 0) > 0
    )


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _loads_json_dict(value: object) -> dict[str, int]:
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, Mapping):
        return {}
    return {str(key): int(val) for key, val in loaded.items()}


def _loads_json_mapping(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, Mapping):
        return {}
    return dict(loaded)


def _round_or_none(value: object) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if val != val:
        return None
    return round(val, 6)


def _counts(values: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = value if value and value != "None" else "none"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _top_counts(values: Sequence[str], *, limit: int) -> dict[str, int]:
    counts = _counts(values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _warning_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    reasons: list[str] = []
    for row in rows:
        reasons.extend(reason for reason in str(row.get("warning_reasons") or "").split("|") if reason)
    return _counts(reasons)


def _most_common(values: Sequence[str]) -> str | None:
    if not values:
        return None
    counts = _counts(values)
    return max(counts, key=counts.get)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Cross-Asset-State test prototype requires pandas") from exc
    return pd


__all__ = [
    "CrossAssetStateTestPrototypeConfig",
    "PROTOTYPE_PROFILE_TYPES",
    "run_cross_asset_state_test_prototype",
]
