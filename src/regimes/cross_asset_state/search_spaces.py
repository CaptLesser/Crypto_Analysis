from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.cross_asset_state.method_universe import (
    CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
    CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
    CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS,
    CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS,
    CROSS_ASSET_STATE_SPLIT_POLICY_ID,
    cross_asset_state_method_universe_manifest,
)


CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION = "cross_asset_state_selection_engine_v1"
CROSS_ASSET_STATE_SEARCH_SPACE_ID = "cross_asset_state_shared_adapter_bounded_grid_search_v3_full_method_parity"
CROSS_ASSET_STATE_TUNING_MODE = "bounded_grid_tuning_v1"


@dataclass(frozen=True)
class CrossAssetStateSearchSpace:
    profile_type: str
    method_family: str
    clusterer_family: str | None
    embedding: str
    search_space_id: str
    selection_eligible: bool
    diagnostic_only: bool
    shared_adapter_backed: bool
    split_policy_id: str | None
    validation_assignment_required: bool
    parameter_grid: Mapping[str, Any]
    relationship_feature_families: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_type": self.profile_type,
            "method_family": self.method_family,
            "clusterer_family": self.clusterer_family,
            "embedding": self.embedding,
            "search_space_id": self.search_space_id,
            "selection_eligible": bool(self.selection_eligible),
            "diagnostic_only": bool(self.diagnostic_only),
            "shared_adapter_backed": bool(self.shared_adapter_backed),
            "split_policy_id": self.split_policy_id,
            "validation_assignment_required": bool(self.validation_assignment_required),
            "parameter_grid": dict(self.parameter_grid),
            "relationship_feature_families": list(self.relationship_feature_families),
            "notes": list(self.notes),
        }


def cross_asset_state_bounded_search_spaces() -> tuple[CrossAssetStateSearchSpace, ...]:
    families = (
        "anchor_core_exposure",
        "peer_strength_stability",
        "relationship_concentration_entropy",
        "residual_peer_signal",
    )
    return (
        CrossAssetStateSearchSpace(
            profile_type="rule_threshold",
            method_family="rule",
            clusterer_family=None,
            embedding="none",
            search_space_id=f"{CROSS_ASSET_STATE_SEARCH_SPACE_ID}:rule_threshold",
            selection_eligible=True,
            diagnostic_only=False,
            shared_adapter_backed=False,
            split_policy_id=None,
            validation_assignment_required=False,
            relationship_feature_families=families,
            parameter_grid={
                "anchor_core_exposure": {
                    "variants": ["balanced_coupling", "beta_core_joint", "core_secondary_joint"],
                    "low_max": [0.33, 0.35],
                    "high_min": [0.66, 0.75],
                },
                "peer_strength_stability": {
                    "variants": ["strength_stability_joint", "count_strength_joint", "ranked_peer_context"],
                    "low_max": [0.33],
                    "high_min": [0.66],
                },
                "relationship_concentration_entropy": {
                    "variants": ["concentration_entropy_joint", "spread_guarded_rank", "concentration_only"],
                    "spread_guard_required": True,
                },
                "residual_peer_signal": {
                    "variants": ["signed_standard", "signed_narrow", "signed_magnitude"],
                    "negative_max": [-0.35, -0.20],
                    "positive_min": [0.20, 0.35],
                },
            },
            notes=("family-specific rule grids are eligible but remain bounded",),
        ),
        CrossAssetStateSearchSpace(
            profile_type="ordinal_quantile",
            method_family="ordinal",
            clusterer_family=None,
            embedding="none",
            search_space_id=f"{CROSS_ASSET_STATE_SEARCH_SPACE_ID}:ordinal_quantile",
            selection_eligible=True,
            diagnostic_only=False,
            shared_adapter_backed=False,
            split_policy_id=None,
            validation_assignment_required=False,
            relationship_feature_families=families,
            parameter_grid={"rank_bins": [2, 3, 4], "tie_method": ["first"], "spread_guarded_bins": True},
            notes=("ordinal bins are eligible only when spread guards pass",),
        ),
        *_shared_adapter_search_spaces(families),
        CrossAssetStateSearchSpace(
            profile_type="diagnostic_only",
            method_family="fallback",
            clusterer_family=None,
            embedding="none",
            search_space_id=f"{CROSS_ASSET_STATE_SEARCH_SPACE_ID}:fallback",
            selection_eligible=False,
            diagnostic_only=True,
            shared_adapter_backed=False,
            split_policy_id=None,
            validation_assignment_required=False,
            relationship_feature_families=families,
            parameter_grid={"fallback": ["masked_unavailable_or_family_diagnostic_only"]},
            notes=("not a model-facing competing candidate",),
        ),
    )


def _shared_adapter_search_spaces(families: tuple[str, ...]) -> tuple[CrossAssetStateSearchSpace, ...]:
    manifest_specs = {
        str(spec.get("profile_type")): spec
        for spec in cross_asset_state_method_universe_manifest().get("method_specs", ())
        if isinstance(spec, Mapping)
    }
    rows: list[CrossAssetStateSearchSpace] = []
    for method in CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS:
        spec = manifest_specs[method]
        rows.append(
            CrossAssetStateSearchSpace(
                profile_type=method,
                method_family=method,
                clusterer_family=spec.get("clusterer_family"),
                embedding=str(spec.get("embedding") or "none"),
                search_space_id=f"{CROSS_ASSET_STATE_SEARCH_SPACE_ID}:{method}",
                selection_eligible=method not in CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS,
                diagnostic_only=method in CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS,
                shared_adapter_backed=True,
                split_policy_id=CROSS_ASSET_STATE_SPLIT_POLICY_ID,
                validation_assignment_required=True,
                relationship_feature_families=families,
                parameter_grid=dict(spec.get("current_parameterization") or {}),
                notes=(
                    "shared adapter fit-train and assign-score semantics",
                    "diagnostic-only by Cross-Asset policy" if method in CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS else "selection eligible bounded adapter grid",
                ),
            )
        )
    return tuple(rows)


def cross_asset_state_search_space_manifest() -> dict[str, Any]:
    spaces = cross_asset_state_bounded_search_spaces()
    return {
        "artifact_kind": "cross_asset_state_selection_search_space_manifest",
        "selection_engine_version": CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION,
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "tuning_mode": CROSS_ASSET_STATE_TUNING_MODE,
        "method_universe_version": CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
        "profile_candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "optuna_dependency_required": False,
        "bounded_grid_search": True,
        "full_asset_state_scale_optuna_run": False,
        "profile_type_count": len(spaces),
        "search_spaces": [space.as_dict() for space in spaces],
        "method_universe_manifest": cross_asset_state_method_universe_manifest(),
        "production_approved": False,
        "production_writer_enabled": False,
    }


def validate_cross_asset_state_search_space_manifest(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(manifest or cross_asset_state_search_space_manifest())
    spaces = [dict(item) for item in payload.get("search_spaces") or () if isinstance(item, Mapping)]
    reason_codes: list[str] = []
    if payload.get("artifact_kind") != "cross_asset_state_selection_search_space_manifest":
        reason_codes.append("artifact_kind_invalid")
    if payload.get("selection_engine_version") != CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION:
        reason_codes.append("selection_engine_version_invalid")
    if payload.get("search_space_id") != CROSS_ASSET_STATE_SEARCH_SPACE_ID:
        reason_codes.append("search_space_id_invalid")
    if payload.get("production_approved") is not False or payload.get("production_writer_enabled") is not False:
        reason_codes.append("production_flags_not_fail_closed")
    by_profile = {str(item.get("profile_type")): item for item in spaces}
    required = {"rule_threshold", "ordinal_quantile", *CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS, "diagnostic_only"}
    missing = sorted(required - set(by_profile))
    if missing:
        reason_codes.append("required_profile_types_missing")
    if by_profile.get("birch", {}).get("selection_eligible") is not False:
        reason_codes.append("birch_selection_eligible")
    if by_profile.get("birch", {}).get("diagnostic_only") is not True:
        reason_codes.append("birch_not_diagnostic_only")
    for profile_type in ("hdbscan", "optics"):
        if by_profile.get(profile_type, {}).get("selection_eligible") is not True:
            reason_codes.append(f"{profile_type}_not_selection_eligible")
        if by_profile.get(profile_type, {}).get("diagnostic_only") is not False:
            reason_codes.append(f"{profile_type}_diagnostic_only_unexpected")
    for profile_type in ("rule_threshold", "ordinal_quantile"):
        if by_profile.get(profile_type, {}).get("selection_eligible") is not True:
            reason_codes.append(f"{profile_type}_not_selection_eligible")
    for profile_type in CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS:
        row = by_profile.get(profile_type, {})
        if row.get("shared_adapter_backed") is not True:
            reason_codes.append(f"{profile_type}_not_shared_adapter_backed")
        if row.get("split_policy_id") != CROSS_ASSET_STATE_SPLIT_POLICY_ID:
            reason_codes.append(f"{profile_type}_split_policy_invalid")
        if row.get("validation_assignment_required") is not True:
            reason_codes.append(f"{profile_type}_validation_assignment_not_required")
        if profile_type in CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS:
            if row.get("selection_eligible") is not False or row.get("diagnostic_only") is not True:
                reason_codes.append(f"{profile_type}_diagnostic_policy_invalid")
        elif row.get("selection_eligible") is not True:
            reason_codes.append(f"{profile_type}_not_selection_eligible")
    return {
        "artifact_kind": "cross_asset_state_selection_search_space_manifest_validation",
        "status": "passed" if not reason_codes else "blocked",
        "passed": not reason_codes,
        "reason_codes": reason_codes,
        "missing_profile_types": missing,
        "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        "production_write_allowed": False,
    }


__all__ = [
    "CROSS_ASSET_STATE_SEARCH_SPACE_ID",
    "CROSS_ASSET_STATE_SELECTION_ENGINE_VERSION",
    "CROSS_ASSET_STATE_TUNING_MODE",
    "CrossAssetStateSearchSpace",
    "cross_asset_state_bounded_search_spaces",
    "cross_asset_state_search_space_manifest",
    "validate_cross_asset_state_search_space_manifest",
]
