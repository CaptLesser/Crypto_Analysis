from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.test_branch_maturity import FILTER_FAMILY_DIAGNOSTIC_ONLY, normalize_test_branch_filter_reason
from src.regimes.cross_asset_state.economic_diagnostics import build_cross_asset_state_economic_diagnostic_contract
from src.regimes.cross_asset_state.feature_families import CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL
from src.regimes.cross_asset_state.family_policies import (
    CrossAssetStateFamilyPolicy,
    classify_cross_asset_state_family,
    cross_asset_state_family_policy_manifest,
    validate_cross_asset_state_family_policy_manifest,
)
from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason
from src.regimes.cross_asset_state.method_universe import cross_asset_state_method_universe_manifest
from src.regimes.cross_asset_state.mini_test import CrossAssetStateMiniTestConfig, run_cross_asset_state_v1_mini_test
from src.regimes.cross_asset_state.profile_manifest import validate_cross_asset_state_profile_grain
from src.regimes.cross_asset_state.search_spaces import (
    CROSS_ASSET_STATE_SEARCH_SPACE_ID,
    CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
    CROSS_ASSET_STATE_TUNING_MODE,
    cross_asset_state_search_space_manifest,
    validate_cross_asset_state_search_space_manifest,
)
from src.regimes.cross_asset_state.window_profiles import cross_asset_state_window_policy_manifest


@dataclass(frozen=True)
class CrossAssetStateSelectionEngineConfig:
    handoff_path: str | Path
    output_root: str | Path
    artifact_label: str = "selection_engine"
    summary_filename: str = "cross_asset_state_selection_engine_summary.json"
    selected_profiles_filename: str = "cross_asset_state_selected_profiles.selection_engine.nonprod.json"
    cells_filename: str = "cross_asset_state_selection_engine_cells.csv"
    family_summary_filename: str = "cross_asset_state_selection_engine_family_summary.csv"
    inspection_examples_filename: str = "cross_asset_state_selection_engine_inspection_examples.csv"
    candidate_diagnostics_filename: str = "cross_asset_state_selection_engine_candidate_diagnostics.csv"
    window_summary_filename: str = "cross_asset_state_selection_engine_window_summary.csv"
    runtime_filename: str = "cross_asset_state_selection_engine_runtime.csv"
    masked_unavailable_filename: str = "cross_asset_state_masked_unavailable.selection_engine.nonprod.json"
    eligibility_manifest_path: str | Path | None = None
    bands: tuple[str, ...] = ("meso", "macro")
    max_valid_assets: int = 40
    min_valid_assets_for_meaningful_sample: int = 40
    max_inspection_examples_per_family: int = 10
    max_parquet_files_per_asset: int = 4
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL
    family_names: tuple[str, ...] = ()
    sample_assets: tuple[str, ...] = ()
    include_masked_probe_asset: bool = True
    progress_root: str | Path | None = None
    progress_run_id: str | None = None
    progress_shard_id: str | None = None
    progress_family: str | None = None
    progress_worker_id: str | None = None
    progress_flush_cell_interval: int = 1


def run_cross_asset_state_selection_engine(config: CrossAssetStateSelectionEngineConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_summary = run_cross_asset_state_v1_mini_test(
        CrossAssetStateMiniTestConfig(
            handoff_path=config.handoff_path,
            output_root=output_root,
            artifact_label=f"{config.artifact_label}_raw_mini",
            summary_filename="_cross_asset_state_selection_engine_raw_mini_summary.json",
            selected_profiles_filename="_cross_asset_state_selected_profiles.selection_engine_raw.nonprod.json",
            cells_filename="_cross_asset_state_selection_engine_raw_cells.csv",
            family_summary_filename="_cross_asset_state_selection_engine_raw_family_summary.csv",
            inspection_examples_filename="_cross_asset_state_selection_engine_raw_inspection_examples.csv",
            window_comparison_filename="_cross_asset_state_selection_engine_raw_window_comparison.csv",
            profile_type_comparison_filename="_cross_asset_state_selection_engine_raw_profile_type_comparison.csv",
            health_failures_filename="_cross_asset_state_selection_engine_raw_health_failures.csv",
            runtime_filename="_cross_asset_state_selection_engine_raw_runtime.csv",
            bands=config.bands,
            max_valid_assets=config.max_valid_assets,
            min_valid_assets_for_meaningful_sample=config.min_valid_assets_for_meaningful_sample,
            max_inspection_examples_per_family=config.max_inspection_examples_per_family,
            max_parquet_files_per_asset=config.max_parquet_files_per_asset,
            feature_set_version=config.feature_set_version,
            family_names=config.family_names,
            sample_assets=config.sample_assets,
            include_masked_probe_asset=config.include_masked_probe_asset,
            progress_root=config.progress_root,
            progress_run_id=config.progress_run_id,
            progress_shard_id=config.progress_shard_id,
            progress_family=config.progress_family,
            progress_worker_id=config.progress_worker_id,
            progress_flush_cell_interval=config.progress_flush_cell_interval,
        )
    )

    raw_cells = _read_csv(raw_summary["paths"]["cells_path"])
    raw_family_summary = _read_csv(raw_summary["paths"]["family_summary_path"])
    raw_inspection = _read_csv(raw_summary["paths"]["inspection_examples_path"])
    raw_comparisons = _read_csv(raw_summary["paths"]["profile_type_comparison_path"])
    raw_windows = _read_csv(raw_summary["paths"]["window_comparison_path"])
    raw_runtime = _read_csv(raw_summary["paths"]["runtime_path"])
    raw_selected_payload = json.loads(Path(raw_summary["paths"]["selected_profiles_path"]).read_text(encoding="utf-8"))
    eligibility_rows = _read_csv(config.eligibility_manifest_path) if config.eligibility_manifest_path else []

    policies = _family_policies(raw_family_summary, raw_cells, raw_summary)
    policy_by_family = {policy.relationship_feature_family: policy for policy in policies}
    cells = [_selection_cell(row, policy_by_family[str(row["relationship_feature_family"])]) for row in raw_cells]
    family_summary = _selection_family_summary(raw_family_summary, cells, policies)
    inspection_rows = [_selection_inspection_row(row, policy_by_family[str(row["relationship_feature_family"])]) for row in raw_inspection]
    candidate_diagnostics = _candidate_diagnostics(raw_comparisons, cells)
    window_summary = _window_summary(cells, raw_windows)
    runtime_rows = _runtime_rows(raw_runtime, raw_summary, cells, started)
    selected_payload = _selected_profile_payload(
        raw_selected_payload,
        raw_summary=raw_summary,
        policies=policies,
        active_filename=config.selected_profiles_filename,
        eligibility_rows=eligibility_rows,
        feature_set_version=config.feature_set_version,
    )

    cells_path = output_root / config.cells_filename
    family_summary_path = output_root / config.family_summary_filename
    inspection_path = output_root / config.inspection_examples_filename
    candidate_diagnostics_path = output_root / config.candidate_diagnostics_filename
    window_summary_path = output_root / config.window_summary_filename
    runtime_path = output_root / config.runtime_filename
    masked_unavailable_path = output_root / config.masked_unavailable_filename
    selected_profiles_path = output_root / config.selected_profiles_filename
    summary_path = output_root / config.summary_filename

    _write_csv(cells_path, cells)
    _write_csv(family_summary_path, family_summary)
    _write_csv(inspection_path, inspection_rows)
    _write_csv(candidate_diagnostics_path, candidate_diagnostics)
    _write_csv(window_summary_path, window_summary)
    _write_csv(runtime_path, runtime_rows)
    selected_profiles_path.write_text(json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8")
    masked_payload = _masked_unavailable_payload(
        selected_payload,
        active_filename=config.masked_unavailable_filename,
        selected_profiles_path=selected_profiles_path,
    )
    masked_unavailable_path.write_text(json.dumps(masked_payload, indent=2, sort_keys=True), encoding="utf-8")

    selected_model_facing = [row for row in cells if row["cell_status"] == "selected_model_facing"]
    diagnostic_only = [row for row in cells if row["cell_status"] == "diagnostic_only"]
    masked = [row for row in cells if row["cell_status"] == "masked_unavailable"]
    manifest_selected = [dict(row) for row in selected_payload.get("selected_profiles") or ()]
    manifest_diagnostic = [dict(row) for row in selected_payload.get("diagnostic_only_profiles") or ()]
    manifest_masked = [dict(row) for row in selected_payload.get("masked_or_skipped_cells") or ()]
    selected_profile_counts = _counts([str(row.get("selected_profile_type")) for row in selected_model_facing if row.get("selected_profile_type")])
    diagnostic_profile_counts = _counts([str(row.get("selected_profile_type")) for row in diagnostic_only if row.get("selected_profile_type")])
    scale_estimate = _scale_estimate(raw_summary, elapsed_seconds=time.perf_counter() - started)
    family_policy_manifest = cross_asset_state_family_policy_manifest(policies)
    search_space_manifest = cross_asset_state_search_space_manifest()
    economic_contract = raw_summary.get("economic_diagnostic_contract") or build_cross_asset_state_economic_diagnostic_contract({}).as_dict()

    summary = {
        "artifact_kind": "cross_asset_state_selection_engine_summary",
        "artifact_label": config.artifact_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_engine_version": CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
        "feature_set_version": config.feature_set_version,
        "selection_unit": "asset_id x relationship_feature_family x band",
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "tuning_mode": CROSS_ASSET_STATE_TUNING_MODE,
        "optuna_style_selection_status": "bounded_grid_tuning_no_external_optuna_dependency",
        "relationship_context_id": raw_summary["relationship_context_id"],
        "active_baseline_confirmation": raw_summary["active_run_surface"],
        "sample_size": raw_summary["sample_size"],
        "valid_target_assets_across_bands_count": raw_summary["valid_target_assets_across_bands_count"],
        "raw_selection_engine_expected_cells": raw_summary["expected_cells"],
        "raw_selected_model_facing_cells": len(selected_model_facing),
        "raw_diagnostic_only_cells": len(diagnostic_only),
        "raw_masked_unavailable_cells": len(masked),
        "expected_cells": int(selected_payload.get("expected_cell_count") or raw_summary["expected_cells"]),
        "selected_model_facing_cells": len(manifest_selected),
        "diagnostic_only_cells": len(manifest_diagnostic),
        "masked_unavailable_cells": len(manifest_masked),
        "missing_cells": int(selected_payload.get("missing_cell_count") or 0),
        "eligibility_manifest_path": str(config.eligibility_manifest_path) if config.eligibility_manifest_path else None,
        "eligibility_overlay_applied": bool(eligibility_rows),
        "eligibility_overlay_row_count": len(eligibility_rows),
        "total_candidate_evaluations": raw_summary["total_candidate_evaluations"],
        "total_tuning_trials_or_evals": raw_summary["total_candidate_evaluations"],
        "window_candidate_evaluations": raw_summary["window_candidate_evaluations"],
        "selected_relationship_families": family_policy_manifest["selected_model_facing_families"],
        "diagnostic_only_families": family_policy_manifest["diagnostic_only_families"],
        "blocked_families": family_policy_manifest["blocked_families"],
        "family_policy_manifest": family_policy_manifest,
        "family_policy_manifest_validation": validate_cross_asset_state_family_policy_manifest(family_policy_manifest),
        "window_policy_manifest": cross_asset_state_window_policy_manifest(),
        "selected_window_patterns": _selected_window_patterns(cells),
        "selected_window_patterns_by_band": _selected_window_patterns_by_band(cells),
        "selected_profile_type_counts": selected_profile_counts,
        "diagnostic_only_profile_type_counts": diagnostic_profile_counts,
        "selected_profile_type_by_family": {
            row["relationship_feature_family"]: row["best_profile_type"]
            for row in family_summary
        },
        "search_space_manifest": search_space_manifest,
        "search_space_manifest_validation": validate_cross_asset_state_search_space_manifest(search_space_manifest),
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
        "candidate_fairness": _candidate_fairness(candidate_diagnostics),
        "agglomerative_dominance": _agglomerative_dominance(cells, candidate_diagnostics),
        "economic_diagnostic_status": economic_contract.get("economic_diagnostic_status"),
        "economic_diagnostic_contract": economic_contract,
        "runtime_telemetry": {
            "wall_time_seconds": round(time.perf_counter() - started, 6),
            "raw_mini_runtime_seconds": raw_summary["runtime_seconds"],
            "runtime_rows_path": str(runtime_path),
            "runtime_telemetry_events_path": (raw_summary.get("paths") or {}).get("runtime_telemetry_events_path"),
            "runtime_telemetry_aggregation": raw_summary.get("runtime_telemetry_aggregation"),
        },
        "runtime_telemetry_aggregation": raw_summary.get("runtime_telemetry_aggregation"),
        "execution_cache_telemetry": raw_summary.get("execution_cache_telemetry") or {},
        "scale_estimate": scale_estimate,
        "paths": {
            "summary_path": str(summary_path),
            "selected_profiles_path": str(selected_profiles_path),
            "cells_path": str(cells_path),
            "family_summary_path": str(family_summary_path),
            "inspection_examples_path": str(inspection_path),
            "candidate_diagnostics_path": str(candidate_diagnostics_path),
            "window_summary_path": str(window_summary_path),
            "runtime_path": str(runtime_path),
            "runtime_telemetry_events_path": (raw_summary.get("paths") or {}).get("runtime_telemetry_events_path"),
            "masked_unavailable_path": str(masked_unavailable_path),
            "raw_mini_summary_path": raw_summary["paths"]["summary_path"],
        },
        "selected_profile_manifest_validation": selected_payload["selection_engine_manifest_validation"],
        "final_verdict": _final_verdict(family_policy_manifest, economic_contract, cells),
        "exact_next_recommended_sprint": _next_sprint(family_policy_manifest, economic_contract),
        "safety_confirmation": raw_summary["safety_confirmation"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _family_policies(
    raw_family_summary: Sequence[Mapping[str, Any]],
    raw_cells: Sequence[Mapping[str, Any]],
    raw_summary: Mapping[str, Any],
) -> tuple[CrossAssetStateFamilyPolicy, ...]:
    policies: list[CrossAssetStateFamilyPolicy] = []
    window_patterns = raw_summary.get("best_window_policy_patterns") or {}
    for row in raw_family_summary:
        family = str(row["relationship_feature_family"])
        selected_counts = _loads_mapping(row.get("selected_profile_type_counts"))
        fam_masks = [
            cell for cell in raw_cells
            if cell.get("relationship_feature_family") == family and cell.get("cell_status") == "masked_unavailable"
        ]
        low_spread_masks = sum(1 for cell in fam_masks if str(cell.get("filter_reason_code") or cell.get("mask_reason")) == "low_feature_spread")
        patterns = dict(window_patterns.get(family) or {})
        policies.append(
            classify_cross_asset_state_family(
                family,
                selected_count=int(float(row.get("selected_cells") or 0)),
                masked_count=int(float(row.get("masked_unavailable_cells") or 0)),
                selected_profile_counts={str(key): int(value) for key, value in selected_counts.items()},
                low_spread_mask_count=low_spread_masks,
                questionable_count=int(float(row.get("questionable_inspection_example_count") or 0)),
                non_default_window_share=float(patterns.get("non_default_selected_share") or 0.0),
            )
        )
    return tuple(policies)


def _selection_cell(row: Mapping[str, Any], policy: CrossAssetStateFamilyPolicy) -> dict[str, Any]:
    out = dict(row)
    out["selection_engine_version"] = CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION
    out["selection_unit"] = "asset_id x relationship_feature_family x band"
    out["selection_search_space_id"] = CROSS_ASSET_STATE_SEARCH_SPACE_ID
    out["search_space_id"] = out.get("search_space_id") or CROSS_ASSET_STATE_SEARCH_SPACE_ID
    out["tuning_mode"] = CROSS_ASSET_STATE_TUNING_MODE
    out["family_selection_status"] = policy.family_selection_status
    out["family_model_facing_eligible"] = bool(policy.model_facing_eligible)
    out["family_selection_reason"] = policy.reason
    if row.get("cell_status") == "selected_coherent":
        if policy.model_facing_eligible:
            out["cell_status"] = "selected_model_facing"
            out["profile_selection_status"] = "selected_model_facing"
        else:
            out["cell_status"] = "diagnostic_only"
            out["profile_selection_status"] = "diagnostic_only"
            out["diagnostic_only_reason"] = policy.reason
            out["filter_reason_code"] = FILTER_FAMILY_DIAGNOSTIC_ONLY
    elif row.get("cell_status") == "diagnostic_only" and policy.model_facing_eligible:
        out = _masked_record_from_unselectable_candidate(out, default_reason=CrossAssetStateMaskReason.NO_VIABLE_PROFILE)
        out["cell_status"] = "masked_unavailable"
        out["profile_selection_status"] = "masked_unavailable"
        out["family_selection_status"] = policy.family_selection_status
        out["family_model_facing_eligible"] = bool(policy.model_facing_eligible)
        out["family_selection_reason"] = policy.reason
    elif row.get("cell_status") == "masked_unavailable":
        out["profile_selection_status"] = "masked_unavailable"
    else:
        out["profile_selection_status"] = row.get("cell_status")
    out["production_approved"] = False
    out["production_writer_enabled"] = False
    out["production_labels_written"] = False
    out["production_outputs_written"] = False
    return out


def _selection_family_summary(
    raw_family_summary: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    policies: Sequence[CrossAssetStateFamilyPolicy],
) -> list[dict[str, Any]]:
    policy_by_family = {policy.relationship_feature_family: policy for policy in policies}
    rows: list[dict[str, Any]] = []
    for raw in raw_family_summary:
        family = str(raw["relationship_feature_family"])
        fam_cells = [row for row in cells if row.get("relationship_feature_family") == family]
        selected_model = [row for row in fam_cells if row.get("cell_status") == "selected_model_facing"]
        diagnostic = [row for row in fam_cells if row.get("cell_status") == "diagnostic_only"]
        masked = [row for row in fam_cells if row.get("cell_status") == "masked_unavailable"]
        selected_counts = _counts([str(row.get("selected_profile_type")) for row in fam_cells if row.get("selected_profile_type")])
        window_counts = _counts([str(row.get("window_candidate_name")) for row in fam_cells if row.get("window_candidate_name") and row.get("cell_status") != "masked_unavailable"])
        policy = policy_by_family[family]
        rows.append(
            {
                "relationship_feature_family": family,
                "expected_cells": len(fam_cells),
                "selected_model_facing_cells": len(selected_model),
                "diagnostic_only_cells": len(diagnostic),
                "masked_unavailable_cells": len(masked),
                "missing_cells": 0,
                "family_selection_status": policy.family_selection_status,
                "family_model_facing_eligible": bool(policy.model_facing_eligible),
                "family_selection_reason": policy.reason,
                "best_profile_type": _most_common(selected_counts),
                "best_window_candidate_name": _most_common(window_counts),
                "selected_profile_type_counts": json.dumps(selected_counts, sort_keys=True),
                "selected_window_policy_counts": json.dumps(window_counts, sort_keys=True),
                "raw_family_readiness": raw.get("family_readiness"),
                "candidate_health_failure_count": raw.get("candidate_health_failure_count"),
                "dominant_or_tiny_state_failure_count": raw.get("dominant_or_tiny_state_failure_count"),
                "birch_warning_count": raw.get("birch_warning_count"),
                "economic_diagnostic_status": raw.get("economic_diagnostic_status"),
                "recommended_next_action": policy.recommended_next_action,
            }
        )
    return rows


def _selection_inspection_row(row: Mapping[str, Any], policy: CrossAssetStateFamilyPolicy) -> dict[str, Any]:
    out = dict(row)
    out["selection_engine_version"] = CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION
    out["selection_search_space_id"] = CROSS_ASSET_STATE_SEARCH_SPACE_ID
    out["search_space_id"] = out.get("search_space_id") or CROSS_ASSET_STATE_SEARCH_SPACE_ID
    out["family_selection_status"] = policy.family_selection_status
    out["family_model_facing_eligible"] = bool(policy.model_facing_eligible)
    out["family_selection_reason"] = policy.reason
    out["interpretation_status"] = out.get("coherence_status") or out.get("agglomerative_interpretation_status")
    if not policy.model_facing_eligible:
        out["profile_selection_status"] = "diagnostic_only"
        out["diagnostic_only_reason"] = policy.reason
    else:
        out["profile_selection_status"] = "selected_model_facing"
    return out


def _selected_profile_payload(
    raw_payload: Mapping[str, Any],
    *,
    raw_summary: Mapping[str, Any],
    policies: Sequence[CrossAssetStateFamilyPolicy],
    active_filename: str,
    eligibility_rows: Sequence[Mapping[str, Any]] = (),
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
) -> dict[str, Any]:
    policy_by_family = {policy.relationship_feature_family: policy for policy in policies}
    profiles: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    masked: list[dict[str, Any]] = []
    for item in raw_payload.get("profiles") or ():
        record = dict(item)
        family = str(record.get("relationship_feature_family"))
        policy = policy_by_family[family]
        record["selection_engine_version"] = CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION
        record["selection_unit"] = "asset_id x relationship_feature_family x band"
        record["selection_search_space_id"] = CROSS_ASSET_STATE_SEARCH_SPACE_ID
        record["search_space_id"] = record.get("search_space_id") or CROSS_ASSET_STATE_SEARCH_SPACE_ID
        record["tuning_mode"] = CROSS_ASSET_STATE_TUNING_MODE
        record["family_selection_status"] = policy.family_selection_status
        record["family_model_facing_eligible"] = bool(policy.model_facing_eligible)
        record["family_selection_reason"] = policy.reason
        record["production_approved"] = False
        record["production_writer_enabled"] = False
        record["production_labels_written"] = False
        record["production_outputs_written"] = False
        record["canonical_production_state_outputs_written"] = False
        mask_reason = _unselectable_candidate_reason(record) if policy.model_facing_eligible else None
        if record.get("output_health_status") == "masked_unavailable" or record.get("selected_profile_type") is None:
            record["profile_selection_status"] = "masked_unavailable"
            masked.append(record)
        elif mask_reason:
            record = _masked_record_from_unselectable_candidate(record, default_reason=mask_reason)
            record["family_selection_status"] = policy.family_selection_status
            record["family_model_facing_eligible"] = bool(policy.model_facing_eligible)
            record["family_selection_reason"] = policy.reason
            masked.append(record)
        elif policy.model_facing_eligible:
            record["profile_selection_status"] = "selected_model_facing"
            selected.append(record)
        else:
            record["profile_selection_status"] = "diagnostic_only"
            record["diagnostic_only_reason"] = policy.reason
            record["filter_reason_code"] = FILTER_FAMILY_DIAGNOSTIC_ONLY
            diagnostic.append(record)
        profiles.append(record)
    if eligibility_rows:
        profiles, selected, diagnostic, masked = _apply_eligibility_overlay(
            profiles,
            selected,
            diagnostic,
            masked,
            eligibility_rows=eligibility_rows,
            policy_by_family=policy_by_family,
            raw_summary=raw_summary,
            feature_set_version=feature_set_version,
        )
    payload = {
        "artifact_kind": "cross_asset_state_selected_profiles_selection_engine_nonprod",
        "artifact_label": "selection_engine",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grain": "asset_id x relationship_feature_family x band",
        "selection_engine_version": CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
        "selection_unit": "asset_id x relationship_feature_family x band",
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "tuning_mode": CROSS_ASSET_STATE_TUNING_MODE,
        "expected_cell_count": len(profiles),
        "selected_model_facing_profile_count": len(selected),
        "diagnostic_only_profile_count": len(diagnostic),
        "masked_or_skipped_cell_count": len(masked),
        "missing_cell_count": 0 if eligibility_rows else raw_summary["expected_cells"] - len(profiles),
        "eligibility_overlay_applied": bool(eligibility_rows),
        "eligibility_overlay_row_count": len(eligibility_rows),
        "single_active_nonproduction_handoff_artifact": active_filename,
        "stale_sandbox_manifest_used": False,
        "relationship_context_id": raw_summary["relationship_context_id"],
        "relationship_context_handoff_path": raw_summary["relationship_context_handoff_path"],
        "relationship_context_cadence_policy": raw_summary.get("relationship_context_cadence_policy"),
        "family_policy_manifest": cross_asset_state_family_policy_manifest(policies),
        "search_space_manifest": cross_asset_state_search_space_manifest(),
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
        "window_policy_manifest": cross_asset_state_window_policy_manifest(),
        "economic_diagnostic_contract": raw_summary.get("economic_diagnostic_contract"),
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
        "peer_groups_model_facing": False,
        "selected_profiles": selected,
        "diagnostic_only_profiles": diagnostic,
        "masked_or_skipped_cells": masked,
        "profiles": profiles,
    }
    payload["selection_engine_manifest_validation"] = validate_cross_asset_state_selection_engine_manifest(payload)
    return payload


def _unselectable_candidate_reason(record: Mapping[str, Any]) -> str | None:
    selected_profile_type = str(record.get("selected_profile_type") or record.get("profile_type") or "")
    if selected_profile_type == "diagnostic_only" or _truthy(record.get("diagnostic_only")):
        return str(
            record.get("selection_exclusion_reason")
            or record.get("diagnostic_only_reason")
            or record.get("filter_reason_code")
            or record.get("mask_reason")
            or CrossAssetStateMaskReason.NO_VIABLE_PROFILE
        )
    for field in ("selection_eligible", "selection_eligibility"):
        if field in record and record.get(field) not in (None, "") and not _truthy(record.get(field)):
            return str(
                record.get("selection_exclusion_reason")
                or record.get("filter_reason_code")
                or CrossAssetStateMaskReason.PROFILE_TYPE_NOT_SELECTION_ELIGIBLE
            )
    output_status = str(record.get("output_health_status") or "")
    if output_status and output_status not in {"passed", "masked_unavailable"}:
        return str(
            record.get("mask_reason")
            or record.get("filter_reason_code")
            or record.get("selection_exclusion_reason")
            or CrossAssetStateMaskReason.NO_VIABLE_PROFILE
        )
    return None


def _masked_record_from_unselectable_candidate(
    record: Mapping[str, Any],
    *,
    default_reason: str,
) -> dict[str, Any]:
    out = dict(record)
    reason = str(default_reason or _unselectable_candidate_reason(record) or CrossAssetStateMaskReason.NO_VIABLE_PROFILE)
    normalized_reason = normalize_test_branch_filter_reason(reason) or reason
    out["profile_selection_status"] = "masked_unavailable"
    out["cell_status"] = "masked_unavailable"
    out["mask_reason"] = normalized_reason
    out["filter_reason_code"] = normalized_reason
    out["selection_exclusion_reason"] = normalized_reason
    out["diagnostic_only"] = False
    out["diagnostic_only_reason"] = normalized_reason
    out["selection_eligible"] = False
    out["selection_eligibility"] = False
    out["selected_profile_type"] = None
    out["selected_method_family"] = None
    out["selected_candidate_id"] = None
    out["selected_parameter_grid_id"] = None
    out["profile_type"] = None
    out["method_family"] = None
    out["clusterer_family"] = None
    out["embedding"] = None
    out["readiness_status"] = "masked_unavailable"
    out["candidate_readiness_status"] = "masked_unavailable"
    out["validation_assignment_policy"] = out.get("validation_assignment_policy") or "not_applicable_masked"
    out["split_policy_id"] = out.get("split_policy_id") or "not_applicable_masked"
    out["output_health_status"] = "masked_unavailable"
    out["output_health_score"] = None
    out["semantic_score"] = None
    out["semantic_separation_score"] = None
    out["temporal_score"] = None
    out["temporal_stability_score"] = None
    out["coverage_score"] = None
    out["runtime_tiebreak"] = None
    out["total_candidate_score"] = None
    out["state_count"] = 0
    out["label_distribution"] = {}
    out["production_approved"] = False
    out["production_writer_enabled"] = False
    out["production_labels_written"] = False
    out["production_outputs_written"] = False
    out["canonical_production_state_outputs_written"] = False
    return out


def _apply_eligibility_overlay(
    profiles: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    diagnostic: Sequence[Mapping[str, Any]],
    masked: Sequence[Mapping[str, Any]],
    *,
    eligibility_rows: Sequence[Mapping[str, Any]],
    policy_by_family: Mapping[str, CrossAssetStateFamilyPolicy],
    raw_summary: Mapping[str, Any],
    feature_set_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_by_key = {_profile_key(row): dict(row) for row in profiles}
    selected_keys = {_profile_key(row) for row in selected}
    diagnostic_keys = {_profile_key(row) for row in diagnostic}
    masked_keys = {_profile_key(row) for row in masked}
    overlaid_profiles: list[dict[str, Any]] = []
    overlaid_selected: list[dict[str, Any]] = []
    overlaid_diagnostic: list[dict[str, Any]] = []
    overlaid_masked: list[dict[str, Any]] = []
    for row in eligibility_rows:
        key = _profile_key(row)
        record = raw_by_key.get(key)
        if _eligibility_row_is_selected(row) and record is not None and key not in masked_keys:
            record = dict(record)
            if key in selected_keys:
                overlaid_selected.append(record)
            elif key in diagnostic_keys:
                overlaid_diagnostic.append(record)
            else:
                record["profile_selection_status"] = "masked_unavailable"
                record["mask_reason"] = record.get("mask_reason") or CrossAssetStateMaskReason.NO_VIABLE_PROFILE
                overlaid_masked.append(record)
            overlaid_profiles.append(record)
            continue
        if record is not None and key in masked_keys:
            masked_record = dict(record)
        else:
            policy = policy_by_family.get(str(row.get("relationship_feature_family")))
            masked_record = _masked_profile_from_eligibility_row(
                row,
                raw_summary=raw_summary,
                policy=policy,
                feature_set_version=feature_set_version,
            )
        overlaid_masked.append(masked_record)
        overlaid_profiles.append(masked_record)
    return overlaid_profiles, overlaid_selected, overlaid_diagnostic, overlaid_masked


def _masked_profile_from_eligibility_row(
    row: Mapping[str, Any],
    *,
    raw_summary: Mapping[str, Any],
    policy: CrossAssetStateFamilyPolicy | None,
    feature_set_version: str,
) -> dict[str, Any]:
    asset = str(row.get("asset_id") or row.get("asset") or "")
    family = str(row.get("relationship_feature_family") or "")
    band = str(row.get("band") or "")
    reason = str(row.get("filter_reason_code") or row.get("mask_reason") or CrossAssetStateMaskReason.NO_VIABLE_PROFILE)
    if not reason.strip():
        reason = CrossAssetStateMaskReason.NO_VIABLE_PROFILE
    support_definition_id = "dynamic_variable_peer_fallback_v1" if "variable_peer" in str(feature_set_version) else "original_fixed_top3_v1"
    return {
        "asset_id": asset,
        "relationship_feature_family": family,
        "band": band,
        "feature_set_version": str(row.get("feature_set_version") or feature_set_version),
        "support_definition_id": row.get("support_definition_id") or support_definition_id,
        "support_size": _optional_int(row.get("support_size")),
        "support_quality": row.get("support_quality"),
        "support_rank_max": _optional_int(row.get("support_rank_max")),
        "support_threshold": _optional_float(row.get("support_threshold")),
        "support_fallback_path": row.get("support_fallback_path"),
        "repaired_feature_manifest_id": row.get("repaired_feature_manifest_id"),
        "profile_id": f"{asset}|{family}|{band}|selection_engine_masked_v1",
        "selected_profile_type": None,
        "selected_method_family": None,
        "selected_candidate_id": None,
        "selected_parameter_grid_id": None,
        "profile_type": None,
        "method_family": None,
        "clusterer_family": None,
        "embedding": None,
        "readiness_status": "masked_unavailable",
        "candidate_readiness_status": "masked_unavailable",
        "selection_eligible": False,
        "selection_eligibility": False,
        "selection_exclusion_reason": reason,
        "diagnostic_only": False,
        "diagnostic_only_reason": reason,
        "shared_adapter_used": False,
        "adapter_name": None,
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "validation_assignment_policy": "not_applicable_masked",
        "split_policy_id": "not_applicable_masked",
        "feature_columns_used": [],
        "window_policy_id": row.get("window_policy_id") or "current_default",
        "window_profile_id": row.get("window_policy_id") or "current_default",
        "window_candidate_name": None,
        "window_policy": {},
        "window_coverage_status": "masked_unavailable",
        "window_observed_rows": _optional_int(row.get("valid_row_count")),
        "window_required_min_rows": None,
        "state_count": 0,
        "label_distribution": {},
        "output_health_status": "masked_unavailable",
        "output_health_score": None,
        "dominant_state_share": None,
        "tiny_state_count": None,
        "semantic_score": None,
        "semantic_separation_score": None,
        "temporal_score": None,
        "temporal_stability_score": None,
        "coverage_score": None,
        "economic_diagnostic_status": "pending_not_computed",
        "runtime_tiebreak": None,
        "total_candidate_score": None,
        "mask_reason": reason,
        "filter_reason_code": reason,
        "relationship_context_id": row.get("relationship_context_id") or raw_summary.get("relationship_context_id"),
        "relationship_snapshot_id": row.get("relationship_snapshot_id"),
        "source_tail_ts": _optional_int(row.get("source_tail_ts")),
        "known_at_ts": _optional_int(row.get("known_at_ts")),
        "relationship_context_cadence_policy_id": row.get("relationship_context_cadence_policy_id")
        or (raw_summary.get("relationship_context_cadence_policy") or {}).get("relationship_context_cadence_policy_id"),
        "snapshot_cadence_days": _optional_int(row.get("snapshot_cadence_days")),
        "stale_snapshot_policy": row.get("stale_snapshot_policy")
        or (raw_summary.get("relationship_context_cadence_policy") or {}).get("stale_snapshot_policy")
        or {},
        "no_future_graph_backfill": _optional_bool(
            row.get("no_future_graph_backfill"),
            default=(raw_summary.get("relationship_context_cadence_policy") or {}).get("no_future_graph_backfill", True),
        ),
        "snapshot_valid_from_ts": _optional_int(row.get("snapshot_valid_from_ts")),
        "snapshot_valid_until_ts": _optional_int(row.get("snapshot_valid_until_ts")),
        "profile_selection_status": "masked_unavailable",
        "selection_engine_version": CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
        "selection_unit": "asset_id x relationship_feature_family x band",
        "selection_search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "tuning_mode": CROSS_ASSET_STATE_TUNING_MODE,
        "family_selection_status": policy.family_selection_status if policy is not None else "masked_unavailable",
        "family_model_facing_eligible": bool(policy.model_facing_eligible) if policy is not None else False,
        "family_selection_reason": policy.reason if policy is not None else "masked by eligibility manifest",
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
    }


def _masked_unavailable_payload(
    selected_payload: Mapping[str, Any],
    *,
    active_filename: str,
    selected_profiles_path: Path,
) -> dict[str, Any]:
    masked = [dict(row) for row in selected_payload.get("masked_or_skipped_cells") or () if isinstance(row, Mapping)]
    return {
        "artifact_kind": "cross_asset_state_masked_unavailable_selection_engine_nonprod",
        "artifact_label": "selection_engine_masked_unavailable",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grain": "asset_id x relationship_feature_family x band",
        "selection_engine_version": CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
        "single_active_nonproduction_handoff_artifact": active_filename,
        "selected_profiles_manifest_path": str(selected_profiles_path),
        "masked_or_skipped_cell_count": len(masked),
        "masked_or_skipped_cells": masked,
        "mask_reason_counts": _counts([str(row.get("mask_reason") or row.get("filter_reason_code") or "unknown") for row in masked]),
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
    }


def _profile_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("asset_id") or row.get("asset") or ""),
        str(row.get("relationship_feature_family") or ""),
        str(row.get("band") or ""),
    )


def _eligibility_row_is_selected(row: Mapping[str, Any]) -> bool:
    return str(row.get("eligibility_status") or "") == "eligible_for_profile_selection" or _truthy(
        row.get("profile_selection_eligible_cell")
    )


def _optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except Exception:
        return None


def _optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _optional_bool(value: Any, *, default: Any = None) -> bool:
    raw = default if value in (None, "") else value
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def validate_cross_asset_state_selection_engine_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    profiles = [dict(item) for item in manifest.get("profiles") or () if isinstance(item, Mapping)]
    selected = [dict(item) for item in manifest.get("selected_profiles") or () if isinstance(item, Mapping)]
    diagnostic = [dict(item) for item in manifest.get("diagnostic_only_profiles") or () if isinstance(item, Mapping)]
    masked = [dict(item) for item in manifest.get("masked_or_skipped_cells") or () if isinstance(item, Mapping)]
    reason_codes: list[str] = []
    expected = int(manifest.get("expected_cell_count") or 0)
    if manifest.get("artifact_kind") != "cross_asset_state_selected_profiles_selection_engine_nonprod":
        reason_codes.append("artifact_kind_invalid")
    if manifest.get("selection_engine_version") != CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION:
        reason_codes.append("selection_engine_version_invalid")
    if manifest.get("production_approved") is not False or manifest.get("production_writer_enabled") is not False:
        reason_codes.append("production_flags_not_fail_closed")
    if expected != len(profiles):
        reason_codes.append("expected_cell_count_mismatch")
    if expected != len(selected) + len(diagnostic) + len(masked):
        reason_codes.append("selected_diagnostic_masked_count_mismatch")
    grain = validate_cross_asset_state_profile_grain(profiles)
    if not grain["passed"]:
        reason_codes.extend(grain["errors"])
    method_parity_fields = (
        "method_family",
        "profile_type",
        "readiness_status",
        "selection_eligibility",
        "diagnostic_only_reason",
        "shared_adapter_used",
        "adapter_name",
        "search_space_id",
        "validation_assignment_policy",
        "split_policy_id",
        "window_policy_id",
        "output_health_status",
        "semantic_score",
        "temporal_score",
        "coverage_score",
        "economic_diagnostic_status",
        "runtime_tiebreak",
    )
    for index, record in enumerate(profiles):
        if record.get("window_policy_id") in (None, ""):
            reason_codes.append(f"profile_{index}_window_policy_id_missing")
        if record.get("profile_selection_status") not in {"selected_model_facing", "diagnostic_only", "masked_unavailable"}:
            reason_codes.append(f"profile_{index}_selection_status_invalid")
        for field in method_parity_fields:
            if field not in record:
                reason_codes.append(f"profile_{index}_{field}_missing")
        for field in (
            "production_approved",
            "production_writer_enabled",
            "production_labels_written",
            "production_outputs_written",
            "canonical_production_state_outputs_written",
        ):
            if record.get(field) is not False:
                reason_codes.append(f"profile_{index}_{field}_not_false")
    return {
        "artifact_kind": "cross_asset_state_selection_engine_manifest_validation",
        "status": "passed" if not reason_codes else "blocked",
        "passed": not reason_codes,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "expected_cell_count": expected,
        "selected_model_facing_profile_count": len(selected),
        "diagnostic_only_profile_count": len(diagnostic),
        "masked_cell_count": len(masked),
        "profile_grain_validation": grain,
        "production_write_allowed": False,
    }


def _candidate_diagnostics(comparisons: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected_counts = Counter(
        (str(row.get("relationship_feature_family")), str(row.get("selected_profile_type")))
        for row in cells
        if row.get("selected_profile_type") and row.get("cell_status") in {"selected_model_facing", "diagnostic_only"}
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparisons:
        groups[(str(row.get("relationship_feature_family")), str(row.get("profile_type")))].append(row)
    rows: list[dict[str, Any]] = []
    for (family, profile_type), group in sorted(groups.items()):
        rows.append(
            {
                "relationship_feature_family": family,
                "profile_type": profile_type,
                "candidate_count": len(group),
                "pass_count": sum(1 for row in group if _truthy(row.get("output_health_passed"))),
                "fail_count": sum(1 for row in group if not _truthy(row.get("output_health_passed"))),
                "selection_eligible_count": sum(1 for row in group if _truthy(row.get("selection_eligible"))),
                "diagnostic_only_count": sum(1 for row in group if _truthy(row.get("diagnostic_only"))),
                "warning_count": sum(1 for row in group if str(row.get("warning_reasons") or "").strip()),
                "dominant_state_failure_count": sum(1 for row in group if "dominant_state_failure" in str(row.get("failure_reasons") or "")),
                "tiny_state_failure_count": sum(1 for row in group if "tiny_state_failure" in str(row.get("failure_reasons") or "")),
                "all_one_state_collapse_count": sum(1 for row in group if "all_one_state_collapse" in str(row.get("failure_reasons") or "")),
                "median_semantic_score": round(_median([_float(row.get("semantic_separation_score")) for row in group]), 6),
                "median_temporal_score": round(_median([_float(row.get("temporal_persistence_score")) for row in group]), 6),
                "median_coverage_score": round(_median([_float(row.get("coverage_score")) for row in group]), 6),
                "selected_count": selected_counts[(family, profile_type)],
            }
        )
    return rows


def _window_summary(cells: Sequence[Mapping[str, Any]], raw_windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cells:
        if row.get("cell_status") == "masked_unavailable":
            continue
        groups[(str(row.get("relationship_feature_family")), str(row.get("band")), str(row.get("window_candidate_name")))].append(row)
    eval_counts = Counter(
        (str(row.get("relationship_feature_family")), str(row.get("band")), str(row.get("window_candidate_name")))
        for row in raw_windows
        if str(row.get("candidate_status")) == "window_evaluated"
    )
    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        family, band, window = key
        rows.append(
            {
                "relationship_feature_family": family,
                "band": band,
                "window_candidate_name": window,
                "selected_or_diagnostic_cell_count": len(group),
                "model_facing_cell_count": sum(1 for row in group if row.get("cell_status") == "selected_model_facing"),
                "diagnostic_only_cell_count": sum(1 for row in group if row.get("cell_status") == "diagnostic_only"),
                "evaluated_window_count": eval_counts[key],
            }
        )
    return rows


def _runtime_rows(
    raw_runtime: Sequence[Mapping[str, Any]],
    raw_summary: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    started: float,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in raw_runtime]
    rows.append(
        {
            "aggregate_scope": "selection_engine",
            "aggregate_key": CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "expected_cells": raw_summary["expected_cells"],
            "selected_model_facing_cells": sum(1 for row in cells if row.get("cell_status") == "selected_model_facing"),
            "diagnostic_only_cells": sum(1 for row in cells if row.get("cell_status") == "diagnostic_only"),
            "masked_unavailable_cells": sum(1 for row in cells if row.get("cell_status") == "masked_unavailable"),
            "candidate_evaluations": raw_summary["total_candidate_evaluations"],
            "window_evaluations": raw_summary["window_candidate_evaluations"],
        }
    )
    return rows


def _candidate_fairness(candidate_diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_profile = _counts([str(row.get("profile_type")) for row in candidate_diagnostics])
    selected_by_profile = Counter()
    warnings_by_profile = Counter()
    for row in candidate_diagnostics:
        selected_by_profile[str(row.get("profile_type"))] += int(row.get("selected_count") or 0)
        warnings_by_profile[str(row.get("profile_type"))] += int(row.get("warning_count") or 0)
    return {
        "search_space_recorded": True,
        "diagnostic_only_candidates_can_win_model_facing": False,
        "candidate_type_rows": by_profile,
        "selected_counts_by_profile_type": dict(selected_by_profile),
        "warning_counts_by_profile_type": dict(warnings_by_profile),
        "birch_selected_count": int(selected_by_profile.get("birch", 0)),
        "hdbscan_selected_count": int(selected_by_profile.get("hdbscan", 0)),
        "optics_selected_count": int(selected_by_profile.get("optics", 0)),
        "runtime_tiebreak_only": True,
    }


def _agglomerative_dominance(cells: Sequence[Mapping[str, Any]], candidate_diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in cells if row.get("cell_status") in {"selected_model_facing", "diagnostic_only"}]
    profile_counts = _counts([str(row.get("selected_profile_type")) for row in selected if row.get("selected_profile_type")])
    agglomerative_count = int(profile_counts.get("agglomerative", 0))
    share = agglomerative_count / max(1, len(selected))
    return {
        "agglomerative_selected_count": agglomerative_count,
        "agglomerative_selected_share": round(share, 6),
        "profile_selection_counts": profile_counts,
        "verdict": "persists_but_not_credible_for_all_families" if share >= 0.5 else "reduced_or_mixed",
        "reason": "Family demotion reduces model-facing trust, but agglomerative still dominates selected/diagnostic cells.",
    }


def _selected_window_patterns(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted({str(row.get("relationship_feature_family")) for row in cells}):
        fam = [row for row in cells if str(row.get("relationship_feature_family")) == family and row.get("cell_status") != "masked_unavailable"]
        counts = _counts([str(row.get("window_candidate_name")) for row in fam if row.get("window_candidate_name")])
        out[family] = {
            "selected_window_counts": counts,
            "best_window_candidate_name": _most_common(counts),
            "non_default_selected_share": _non_default_share(counts),
        }
    return out


def _selected_window_patterns_by_band(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for band in sorted({str(row.get("band")) for row in cells}):
        band_rows = [row for row in cells if str(row.get("band")) == band and row.get("cell_status") != "masked_unavailable"]
        counts = _counts([str(row.get("window_candidate_name")) for row in band_rows if row.get("window_candidate_name")])
        out[band] = {"selected_window_counts": counts, "best_window_candidate_name": _most_common(counts)}
    return out


def _scale_estimate(raw_summary: Mapping[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    selected_assets = int(raw_summary.get("sample_size", {}).get("valid_real_target_assets") or 1)
    per_asset = elapsed_seconds / max(1, selected_assets)
    return {
        "runtime_seconds_per_valid_asset": round(per_asset, 6),
        "estimated_100_valid_asset_seconds": round(per_asset * 100, 3),
        "estimated_larger_bounded_campaign_note": "Manageable for larger bounded design if run remains parent-finalized and avoids full-universe pairwise discovery.",
        "full_universe_feasibility_later": "not_assessed_in_this_sprint",
    }


def _final_verdict(family_manifest: Mapping[str, Any], economic_contract: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> str:
    if economic_contract.get("economic_diagnostic_status") == "computed":
        economic_blocking = False
    else:
        economic_blocking = True
    diagnostic_families = set(family_manifest.get("diagnostic_only_families") or ())
    if diagnostic_families:
        return "B. Selection engine works but specific families need repair."
    if economic_blocking:
        return "C. Economic diagnostics block further confidence."
    if not any(row.get("cell_status") == "selected_model_facing" for row in cells):
        return "E. V1 family set should be reduced."
    return "A. Bounded selection engine is coherent; proceed to larger bounded campaign design."


def _next_sprint(family_manifest: Mapping[str, Any], economic_contract: Mapping[str, Any]) -> str:
    if economic_contract.get("economic_diagnostic_status") != "computed":
        return (
            "Selection-engine family repair plus economic panel design: keep residual/anchor model-facing candidates, "
            "repair peer_strength_stability scoring, keep concentration_entropy diagnostic-only, and wire leakage-safe future outcome refs."
        )
    return "Run larger bounded campaign design with family policy gates preserved."


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _loads_mapping(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(key): int(val) for key, val in value.items()}
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, Mapping):
        return {}
    return {str(key): int(val) for key, val in data.items()}


def _counts(values: Sequence[str]) -> dict[str, int]:
    return dict(Counter(value for value in values if value not in ("", "None", "none")))


def _most_common(counts: Mapping[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _non_default_share(counts: Mapping[str, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if not total:
        return 0.0
    default = int(counts.get("current_default", 0))
    return round((total - default) / total, 6)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _median(values: Sequence[float]) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


__all__ = [
    "CrossAssetStateSelectionEngineConfig",
    "run_cross_asset_state_selection_engine",
    "validate_cross_asset_state_selection_engine_manifest",
]
