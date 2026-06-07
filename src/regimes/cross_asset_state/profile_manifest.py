from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.cross_asset_state.feature_families import CROSS_ASSET_STATE_SCHEMA_VERSION
from src.regimes.cross_asset_state.schema import CrossAssetStateProfileGrain
from src.regimes.core.test_branch_contracts import validate_selected_profile_manifest


CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_KIND = "cross_asset_state_selected_profiles_mature_nonprod"
CROSS_ASSET_STATE_PROFILE_GRAIN = "asset_id x relationship_feature_family x band"

REQUIRED_SELECTED_PROFILE_FIELDS: tuple[str, ...] = (
    "asset_id",
    "relationship_feature_family",
    "band",
    "feature_set_version",
    "support_definition_id",
    "support_size",
    "support_quality",
    "support_rank_max",
    "support_threshold",
    "support_fallback_path",
    "repaired_feature_manifest_id",
    "profile_id",
    "selected_profile_type",
    "profile_type",
    "method_family",
    "readiness_status",
    "selection_eligibility",
    "diagnostic_only_reason",
    "shared_adapter_used",
    "adapter_name",
    "search_space_id",
    "validation_assignment_policy",
    "split_policy_id",
    "feature_columns_used",
    "window_policy_id",
    "state_count",
    "label_distribution",
    "output_health_status",
    "semantic_score",
    "temporal_score",
    "coverage_score",
    "economic_diagnostic_status",
    "runtime_tiebreak",
    "total_candidate_score",
    "relationship_context_id",
    "relationship_snapshot_id",
    "source_tail_ts",
    "known_at_ts",
    "relationship_context_cadence_policy_id",
    "snapshot_cadence_days",
    "stale_snapshot_policy",
    "no_future_graph_backfill",
    "snapshot_valid_from_ts",
    "snapshot_valid_until_ts",
    "production_approved",
    "production_writer_enabled",
    "production_labels_written",
    "production_outputs_written",
    "requires_human_approval_before_production",
)

REQUIRED_MASKED_PROFILE_FIELDS: tuple[str, ...] = (
    "asset_id",
    "relationship_feature_family",
    "band",
    "feature_set_version",
    "support_definition_id",
    "support_size",
    "support_quality",
    "support_rank_max",
    "support_threshold",
    "support_fallback_path",
    "repaired_feature_manifest_id",
    "profile_id",
    "selected_profile_type",
    "profile_type",
    "method_family",
    "readiness_status",
    "selection_eligibility",
    "diagnostic_only_reason",
    "shared_adapter_used",
    "adapter_name",
    "search_space_id",
    "validation_assignment_policy",
    "split_policy_id",
    "feature_columns_used",
    "window_policy_id",
    "state_count",
    "label_distribution",
    "output_health_status",
    "semantic_score",
    "temporal_score",
    "coverage_score",
    "economic_diagnostic_status",
    "runtime_tiebreak",
    "total_candidate_score",
    "mask_reason",
    "relationship_context_id",
    "source_tail_ts",
    "known_at_ts",
    "relationship_context_cadence_policy_id",
    "snapshot_cadence_days",
    "stale_snapshot_policy",
    "no_future_graph_backfill",
    "snapshot_valid_from_ts",
    "snapshot_valid_until_ts",
    "production_approved",
    "production_writer_enabled",
    "production_labels_written",
    "production_outputs_written",
    "requires_human_approval_before_production",
)


@dataclass(frozen=True)
class CrossAssetStateSelectedProfileManifest:
    run_id: str
    expected_cell_count: int
    selected_cell_count: int
    masked_cell_count: int
    sample_assets: tuple[str, ...]
    bands: tuple[str, ...]
    relationship_feature_families: tuple[str, ...]
    source_roots: Sequence[str] = ()
    registry_summary: Mapping[str, Any] | None = None
    created_at_utc: str | None = None
    selected_profiles_path: str | None = None
    masked_cells_path: str | None = None
    label_distribution_path: str | None = None
    input_schema_probe_path: str | None = None
    production_approved: bool = False
    production_writer_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False
    single_active_nonproduction_handoff_artifact: bool = True
    schema_version: int = CROSS_ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.production_approved or self.production_writer_enabled or self.production_outputs_written:
            raise ValueError("Cross-Asset-State v1 shape probe manifests are sandbox/non-production only")
        if self.canonical_production_state_outputs_written:
            raise ValueError("Cross-Asset-State v1 shape probe cannot write canonical production state outputs")
        expected = int(self.expected_cell_count)
        selected = int(self.selected_cell_count)
        masked = int(self.masked_cell_count)
        if expected != selected + masked:
            raise ValueError("Cross-Asset-State shape probe must emit exactly one selected-or-masked record per expected cell")
        object.__setattr__(self, "expected_cell_count", expected)
        object.__setattr__(self, "selected_cell_count", selected)
        object.__setattr__(self, "masked_cell_count", masked)
        object.__setattr__(self, "sample_assets", tuple(str(asset) for asset in self.sample_assets))
        object.__setattr__(self, "bands", tuple(str(band).lower() for band in self.bands))
        object.__setattr__(self, "relationship_feature_families", tuple(str(family) for family in self.relationship_feature_families))
        object.__setattr__(self, "production_approved", False)
        object.__setattr__(self, "production_writer_enabled", False)
        object.__setattr__(self, "production_outputs_written", False)
        object.__setattr__(self, "canonical_production_state_outputs_written", False)

    @property
    def missing_cell_count(self) -> int:
        return self.expected_cell_count - self.selected_cell_count - self.masked_cell_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_v1_shape_probe_summary",
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "grain": "asset_id x relationship_feature_family x band",
            "expected_cell_count": int(self.expected_cell_count),
            "selected_cell_count": int(self.selected_cell_count),
            "masked_cell_count": int(self.masked_cell_count),
            "missing_cell_count": int(self.missing_cell_count),
            "sample_assets": list(self.sample_assets),
            "bands": list(self.bands),
            "relationship_feature_families": list(self.relationship_feature_families),
            "source_roots": list(self.source_roots),
            "registry_summary": dict(self.registry_summary or {}),
            "selected_profiles_path": self.selected_profiles_path,
            "masked_cells_path": self.masked_cells_path,
            "label_distribution_path": self.label_distribution_path,
            "input_schema_probe_path": self.input_schema_probe_path,
            "peer_group_sidecar_status": "diagnostic_support_metadata_only_not_model_facing",
            "scoring_recommendation": "do not use placeholder scores for production selection; use them only to design the next bounded campaign",
            "real_enough_for_next_campaign_design_sprint": bool(self.selected_cell_count > 0 and self.missing_cell_count == 0),
            "production_approved": False,
            "production_writer_enabled": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "single_active_nonproduction_handoff_artifact": bool(self.single_active_nonproduction_handoff_artifact),
        }


def validate_cross_asset_state_profile_grain(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seen: set[tuple[str, str, str]] = set()
    duplicate_keys: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            key = CrossAssetStateProfileGrain(
                asset_id=str(record.get("asset_id", "")),
                relationship_feature_family=str(record.get("relationship_feature_family", "")),
                band=str(record.get("band", "")),
            ).key()
        except Exception as exc:
            errors.append(f"record_{index}_invalid_grain:{exc}")
            continue
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
    if duplicate_keys:
        errors.append("duplicate_profile_grain_keys")
    return {
        "artifact_kind": "cross_asset_state_profile_grain_validation",
        "grain": CROSS_ASSET_STATE_PROFILE_GRAIN,
        "status": "passed" if not errors else "blocked",
        "passed": not errors,
        "record_count": len(records),
        "unique_grain_count": len(seen),
        "duplicate_grain_keys": ["|".join(key) for key in duplicate_keys],
        "errors": errors,
        "production_write_allowed": False,
    }


def validate_cross_asset_state_selected_profile_manifest(
    manifest: Mapping[str, Any],
    *,
    active_filename: str,
    expected_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [dict(item) for item in manifest.get("selected_profiles") or () if isinstance(item, Mapping)]
    masked = [dict(item) for item in manifest.get("masked_or_skipped_cells") or () if isinstance(item, Mapping)]
    all_records = [*selected, *masked]
    reason_codes: list[str] = []
    if manifest.get("artifact_kind") != CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_KIND:
        reason_codes.append("artifact_kind_invalid")
    if manifest.get("grain") != CROSS_ASSET_STATE_PROFILE_GRAIN:
        reason_codes.append("profile_grain_invalid")
    if manifest.get("single_active_nonproduction_handoff_artifact") != active_filename:
        reason_codes.append("single_active_nonproduction_handoff_artifact_invalid")
    if manifest.get("stale_sandbox_manifest_used") is not False:
        reason_codes.append("stale_sandbox_manifest_used_invalid")
    for index, record in enumerate(selected):
        missing = _missing_fields(record, REQUIRED_SELECTED_PROFILE_FIELDS)
        reason_codes.extend(f"selected_{index}_{field}_missing" for field in missing)
        if record.get("relationship_snapshot_id") in (None, ""):
            reason_codes.append(f"selected_{index}_relationship_snapshot_id_missing")
    for index, record in enumerate(masked):
        missing = _missing_fields(record, REQUIRED_MASKED_PROFILE_FIELDS)
        reason_codes.extend(f"masked_{index}_{field}_missing" for field in missing)
        if record.get("mask_reason") in (None, ""):
            reason_codes.append(f"masked_{index}_mask_reason_missing")
    grain_validation = validate_cross_asset_state_profile_grain(all_records)
    if not grain_validation["passed"]:
        reason_codes.extend(grain_validation["errors"])
    shared = validate_selected_profile_manifest(
        manifest,
        active_filename=active_filename,
        expected_cells=expected_cells,
        selected_records_key="selected_profiles",
        masked_records_key="masked_or_skipped_cells",
        selected_cell_key_fields=("asset_id", "relationship_feature_family", "band"),
        require_canonical_gate_field=True,
        expected_requires_human_approval=True,
    )
    reason_codes.extend(shared.reason_codes)
    reason_codes = list(dict.fromkeys(reason_codes))
    return {
        "artifact_kind": "cross_asset_state_selected_profile_manifest_validation",
        "status": "passed" if not reason_codes else "blocked",
        "passed": not reason_codes,
        "reason_codes": reason_codes,
        "profile_grain_validation": grain_validation,
        "shared_manifest_validation": shared.as_dict(),
        "selected_profile_count": len(selected),
        "masked_cell_count": len(masked),
        "expected_cell_count": len(expected_cells),
        "production_write_allowed": False,
    }


def _missing_fields(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(field for field in fields if field not in record)
