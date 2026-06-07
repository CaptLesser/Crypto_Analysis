from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.core.clustering_candidates import candidate_spec_for_method
from src.regimes.core.test_branch_maturity import (
    METHOD_STATUS_DIAGNOSTIC_ONLY_NOT_RUN,
    METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED,
    METHOD_STATUS_FALLBACK_ONLY,
    METHOD_STATUS_SELECTABLE_PARTIAL,
)


CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION = "cross_asset_state_method_universe_v3_full_method_parity"
CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID = "cross_asset_state_profile_candidate_set_v3_full_method_parity"
CROSS_ASSET_STATE_SPLIT_POLICY_ID = "cross_asset_state_time_ordered_train60_score40_v1"

CROSS_ASSET_STATE_METHOD_FAMILIES: tuple[str, ...] = (
    "anchor_core_exposure",
    "peer_strength_stability",
    "relationship_concentration_entropy",
    "residual_peer_signal",
)

CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS: tuple[str, ...] = (
    "kmeans",
    "minibatch_kmeans",
    "pca_kmeans",
    "factor_analysis_kmeans",
    "gaussian_mixture",
    "factor_analysis_gaussian_mixture",
    "bayesian_gaussian_mixture",
    "hdbscan",
    "birch",
    "optics",
    "agglomerative",
)

CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS: tuple[str, ...] = (
    "birch",
)


@dataclass(frozen=True)
class CrossAssetStateMethodSpec:
    profile_type: str
    readiness_status: str
    method_family: str
    clusterer_family: str | None
    embedding: str
    shared_adapter_backed: bool
    split_policy_id: str | None
    validation_assignment_required: bool
    parameterization_status: str
    selectable: bool
    selection_eligible: bool
    diagnostic_only: bool
    fallback_only: bool
    relationship_feature_families: tuple[str, ...]
    current_parameterization: Mapping[str, Any]
    fairness_risks: tuple[str, ...]
    recommended_next_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_type": self.profile_type,
            "readiness_status": self.readiness_status,
            "method_family": self.method_family,
            "clusterer_family": self.clusterer_family,
            "embedding": self.embedding,
            "shared_adapter_backed": bool(self.shared_adapter_backed),
            "split_policy_id": self.split_policy_id,
            "validation_assignment_required": bool(self.validation_assignment_required),
            "parameterization_status": self.parameterization_status,
            "selectable": bool(self.selectable),
            "selection_eligible": bool(self.selection_eligible),
            "diagnostic_only": bool(self.diagnostic_only),
            "fallback_only": bool(self.fallback_only),
            "relationship_feature_families": list(self.relationship_feature_families),
            "current_parameterization": dict(self.current_parameterization),
            "fairness_risks": list(self.fairness_risks),
            "recommended_next_action": self.recommended_next_action,
            "production_approved": False,
            "production_writer_enabled": False,
        }


def default_cross_asset_state_method_universe() -> tuple[CrossAssetStateMethodSpec, ...]:
    families = CROSS_ASSET_STATE_METHOD_FAMILIES
    return (
        CrossAssetStateMethodSpec(
            profile_type="rule_threshold",
            readiness_status=METHOD_STATUS_SELECTABLE_PARTIAL,
            method_family="rule",
            clusterer_family=None,
            embedding="none",
            shared_adapter_backed=False,
            split_policy_id=None,
            validation_assignment_required=False,
            parameterization_status="family_specific_bounded_rule_grid",
            selectable=True,
            selection_eligible=True,
            diagnostic_only=False,
            fallback_only=False,
            relationship_feature_families=families,
            current_parameterization={
                "grid_id": "rule_threshold_grid_v2",
                "anchor_core_exposure": ["balanced_coupling", "beta_core_joint", "core_secondary_joint"],
                "peer_strength_stability": ["strength_stability_joint", "count_strength_joint", "ranked_peer_context"],
                "relationship_concentration_entropy": ["concentration_entropy_joint", "spread_guarded_rank", "concentration_only"],
                "residual_peer_signal": ["signed_standard", "signed_narrow", "signed_magnitude"],
            },
            fairness_risks=(
                "single fixed thresholds are simpler than clustering candidates",
                "dominant-state failures can be threshold-design failures rather than true family failures",
            ),
            recommended_next_action="add family-specific threshold grids and signed residual threshold variants",
        ),
        CrossAssetStateMethodSpec(
            profile_type="ordinal_quantile",
            readiness_status=METHOD_STATUS_SELECTABLE_PARTIAL,
            method_family="ordinal",
            clusterer_family=None,
            embedding="none",
            shared_adapter_backed=False,
            split_policy_id=None,
            validation_assignment_required=False,
            parameterization_status="bounded_rank_bin_grid",
            selectable=True,
            selection_eligible=True,
            diagnostic_only=False,
            fallback_only=False,
            relationship_feature_families=families,
            current_parameterization={"grid_id": "ordinal_quantile_grid_v2", "rank_bins": [2, 3, 4], "tie_method": "first", "spread_guarded_families": ["relationship_concentration_entropy"]},
            fairness_risks=(
                "terciles can create tidy labels even when absolute feature spread is tiny",
                "no two-bin or four-bin alternatives are currently compared",
            ),
            recommended_next_action="add two-bin, three-bin, four-bin, and family-specific ordinal variants with spread guards",
        ),
        *_shared_adapter_method_specs(families),
        CrossAssetStateMethodSpec(
            profile_type="diagnostic_only",
            readiness_status=METHOD_STATUS_FALLBACK_ONLY,
            method_family="fallback",
            clusterer_family=None,
            embedding="none",
            shared_adapter_backed=False,
            split_policy_id=None,
            validation_assignment_required=False,
            parameterization_status="single_label_masked_fallback",
            selectable=False,
            selection_eligible=False,
            diagnostic_only=True,
            fallback_only=True,
            relationship_feature_families=families,
            current_parameterization={"selection_policy": "used when no candidate passes output health"},
            fairness_risks=("not a competing profile type",),
            recommended_next_action="keep as unavailable/fallback record only",
        ),
    )


def _shared_adapter_method_specs(families: tuple[str, ...]) -> tuple[CrossAssetStateMethodSpec, ...]:
    specs: list[CrossAssetStateMethodSpec] = []
    for method in CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS:
        shared = candidate_spec_for_method(method)
        diagnostic_only = method in CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS
        specs.append(
            CrossAssetStateMethodSpec(
                profile_type=method,
                readiness_status=METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED if diagnostic_only else METHOD_STATUS_SELECTABLE_PARTIAL,
                method_family=method,
                clusterer_family=shared.clusterer_family,
                embedding=shared.embedding,
                shared_adapter_backed=True,
                split_policy_id=CROSS_ASSET_STATE_SPLIT_POLICY_ID,
                validation_assignment_required=True,
                parameterization_status=_shared_parameterization_status(method),
                selectable=True,
                selection_eligible=not diagnostic_only,
                diagnostic_only=diagnostic_only,
                fallback_only=False,
                relationship_feature_families=families,
                current_parameterization=_shared_current_parameterization(method),
                fairness_risks=_shared_fairness_risks(method, diagnostic_only=diagnostic_only),
                recommended_next_action=_shared_recommended_next_action(method, diagnostic_only=diagnostic_only),
            )
        )
    return tuple(specs)


def _shared_parameterization_status(method: str) -> str:
    if method in {"pca_kmeans", "factor_analysis_kmeans", "factor_analysis_gaussian_mixture"}:
        return "shared_adapter_bounded_clusterer_plus_reducer_grid"
    if method in {"hdbscan", "optics"}:
        return "shared_adapter_bounded_density_grid"
    if method == "birch":
        return "shared_adapter_diagnostic_bounded_grid"
    return "shared_adapter_bounded_grid"


def _shared_current_parameterization(method: str) -> dict[str, Any]:
    grids: dict[str, dict[str, Any]] = {
        "kmeans": {"grid_id": "kmeans_adapter_bounded_grid_v2", "n_clusters": [2, 3], "n_init": [10], "init": ["k-means++"], "random_state": [17]},
        "minibatch_kmeans": {"grid_id": "minibatch_kmeans_adapter_bounded_grid_v2", "n_clusters": [2], "batch_size": [64], "n_init": [3], "init": ["k-means++"], "random_state": [17]},
        "pca_kmeans": {"grid_id": "pca_kmeans_adapter_bounded_grid_v2", "n_clusters": [2], "embedding": "pca", "embedding_n_components": ["min(2, feature_count)"], "n_init": [10], "random_state": [17]},
        "factor_analysis_kmeans": {"grid_id": "factor_analysis_kmeans_adapter_bounded_grid_v2", "n_clusters": [2], "embedding": "factor_analysis", "embedding_n_components": ["min(2, feature_count)"], "n_init": [10], "random_state": [17]},
        "gaussian_mixture": {"grid_id": "gaussian_mixture_adapter_bounded_grid_v2", "n_components": [2], "covariance_type": ["full"], "reg_covar": [1e-6], "n_init": [2], "random_state": [17]},
        "factor_analysis_gaussian_mixture": {"grid_id": "factor_analysis_gaussian_mixture_adapter_bounded_grid_v2", "n_components": [2], "embedding": "factor_analysis", "embedding_n_components": ["min(2, feature_count)"], "covariance_type": ["full"], "reg_covar": [1e-6], "n_init": [2], "random_state": [17]},
        "bayesian_gaussian_mixture": {"grid_id": "bayesian_gaussian_mixture_adapter_bounded_grid_v2", "n_components": [2], "covariance_type": ["full"], "reg_covar": [1e-6], "n_init": [1], "random_state": [17]},
        "birch": {
            "grid_id": "birch_adapter_diagnostic_bounded_grid_v3",
            "n_clusters": [2],
            "threshold": [0.35],
            "branching_factor": [25],
            "selection_eligible": False,
            "diagnostic_only_reason": "bounded_health_warning_evidence_too_few_subclusters",
        },
        "hdbscan": {
            "grid_id": "hdbscan_adapter_bounded_density_grid_v3",
            "min_cluster_size": ["bounded_by_rows"],
            "min_samples": [1],
            "allow_single_cluster": [True],
            "prediction_data": [True],
            "cluster_selection_method": ["eom"],
            "selection_eligible": True,
            "adapter_execution_default": "adapter_fit_train_assign_validation_selection_eligible",
        },
        "optics": {
            "grid_id": "optics_adapter_bounded_density_grid_v3",
            "min_samples": [2],
            "xi": [0.05],
            "cluster_method": ["xi"],
            "assignment_k": [1],
            "assignment_policy": "nearest_labeled_fit_neighbor",
            "selection_eligible": True,
            "adapter_execution_default": "adapter_fit_train_assign_validation_selection_eligible",
        },
        "agglomerative": {"grid_id": "agglomerative_adapter_bounded_grid_v2", "n_clusters": [2, 3], "linkage": ["ward"], "metric": ["euclidean"], "assignment_policy": "prototype_or_medoid"},
    }
    return {**grids[method], "split_policy_id": CROSS_ASSET_STATE_SPLIT_POLICY_ID, "shared_adapter_backed": True}


def _shared_fairness_risks(method: str, *, diagnostic_only: bool) -> tuple[str, ...]:
    if diagnostic_only:
        risks = [
            "BIRCH remains diagnostic-only until bounded warning/health evidence supports promotion",
            "selection eligibility is intentionally false while too-few-subclusters warnings remain common",
        ]
        return tuple(risks)
    if method in {"hdbscan", "optics"}:
        return (
            "density methods can emit noise or adapter-assigned validation labels and must lose through output-health gates when unstable",
            "bounded Cross-Asset grids are intentionally lighter than Asset-State Optuna search but are no longer policy-excluded",
        )
    if method == "agglomerative":
        return (
            "hierarchical partitions can still dominate semantic spread unless validation assignment and interpretability evidence agree",
            "economic diagnostics remain pending",
        )
    return (
        "bounded Cross-Asset grids remain lighter than Asset-State Optuna search",
        "relationship feature dimensionality may justify lighter grids but must be documented by method diagnostics",
    )


def _shared_recommended_next_action(method: str, *, diagnostic_only: bool) -> str:
    if diagnostic_only:
        return "keep diagnostic-only until adapter-backed warning, assignment, and output-health evidence clears promotion gates"
    if method in {"hdbscan", "optics"}:
        return "keep selection-eligible and let bounded output-health, warning, assignment, runtime, and interpretability evidence decide wins or losses"
    if method == "agglomerative":
        return "treat wins as credible only if adapter-backed assignment and interpretability diagnostics beat comparable peers"
    return "keep selectable and compare through shared adapter train/score assignment semantics in bounded diagnostics"


def cross_asset_state_method_universe_manifest() -> dict[str, Any]:
    specs = default_cross_asset_state_method_universe()
    return {
        "artifact_kind": "cross_asset_state_method_universe_manifest",
        "method_universe_version": CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
        "profile_candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "profile_type_count": len(specs),
        "selectable_profile_types": [spec.profile_type for spec in specs if spec.selection_eligible and not spec.fallback_only],
        "active_diagnostic_profile_types": [spec.profile_type for spec in specs if spec.selectable and spec.diagnostic_only],
        "diagnostic_only_profile_types": [spec.profile_type for spec in specs if spec.diagnostic_only],
        "fallback_profile_types": [spec.profile_type for spec in specs if spec.fallback_only],
        "method_specs": [spec.as_dict() for spec in specs],
        "fair_method_comparison_contract": {
            "requires_declared_readiness_status": True,
            "requires_declared_parameterization": True,
            "requires_warning_telemetry": True,
            "requires_output_health_gate_telemetry": True,
            "requires_runtime_as_tiebreak_only": True,
            "requires_family_specific_repair_before_final_test_branch": True,
        },
        "production_approved": False,
        "production_writer_enabled": False,
    }


def validate_cross_asset_state_method_universe_manifest(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(manifest or cross_asset_state_method_universe_manifest())
    specs = [dict(item) for item in payload.get("method_specs") or () if isinstance(item, Mapping)]
    reason_codes: list[str] = []
    if payload.get("artifact_kind") != "cross_asset_state_method_universe_manifest":
        reason_codes.append("artifact_kind_invalid")
    if payload.get("method_universe_version") != CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION:
        reason_codes.append("method_universe_version_invalid")
    if payload.get("profile_candidate_set_id") != CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID:
        reason_codes.append("profile_candidate_set_id_invalid")
    if payload.get("production_approved") is not False or payload.get("production_writer_enabled") is not False:
        reason_codes.append("production_flags_not_fail_closed")
    observed = {str(spec.get("profile_type")) for spec in specs}
    required = {
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
    }
    missing = sorted(required - observed)
    if missing:
        reason_codes.append("required_profile_types_missing")
    for index, spec in enumerate(specs):
        for field in (
            "profile_type",
            "readiness_status",
            "method_family",
            "embedding",
            "parameterization_status",
            "selection_eligible",
            "current_parameterization",
            "fairness_risks",
            "recommended_next_action",
        ):
            if spec.get(field) in (None, "", [], {}):
                reason_codes.append(f"method_spec_{index}_{field}_missing")
        if spec.get("production_approved") is not False or spec.get("production_writer_enabled") is not False:
            reason_codes.append(f"method_spec_{index}_production_flags_not_fail_closed")
    by_profile = {str(spec.get("profile_type")): spec for spec in specs}
    birch = by_profile.get("birch", {})
    if (
        birch.get("readiness_status") != METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED
        or birch.get("diagnostic_only") is not True
        or birch.get("selection_eligible") is not False
    ):
        reason_codes.append("birch_readiness_not_diagnostic_only_recommended")
    for profile_type in ("hdbscan", "optics"):
        spec = by_profile.get(profile_type, {})
        if spec.get("readiness_status") != METHOD_STATUS_SELECTABLE_PARTIAL:
            reason_codes.append(f"{profile_type}_readiness_not_selectable_partial")
        if spec.get("diagnostic_only") is not False or spec.get("selection_eligible") is not True:
            reason_codes.append(f"{profile_type}_selection_policy_invalid")
    for profile_type in CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS:
        spec = by_profile.get(profile_type, {})
        if spec.get("shared_adapter_backed") is not True:
            reason_codes.append(f"{profile_type}_not_shared_adapter_backed")
        if spec.get("split_policy_id") != CROSS_ASSET_STATE_SPLIT_POLICY_ID:
            reason_codes.append(f"{profile_type}_split_policy_invalid")
        if spec.get("validation_assignment_required") is not True:
            reason_codes.append(f"{profile_type}_validation_assignment_not_required")
    for profile_type in CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS:
        spec = by_profile.get(profile_type, {})
        if spec.get("diagnostic_only") is not True or spec.get("selection_eligible") is not False:
            reason_codes.append(f"{profile_type}_diagnostic_policy_invalid")
    if by_profile.get("diagnostic_only", {}).get("fallback_only") is not True:
        reason_codes.append("diagnostic_only_fallback_status_invalid")
    return {
        "artifact_kind": "cross_asset_state_method_universe_manifest_validation",
        "status": "passed" if not reason_codes else "blocked",
        "passed": not reason_codes,
        "method_universe_version": CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
        "profile_candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        "profile_type_count": len(specs),
        "missing_profile_types": missing,
        "reason_codes": reason_codes,
        "production_write_allowed": False,
    }


__all__ = [
    "CROSS_ASSET_STATE_METHOD_FAMILIES",
    "CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION",
    "CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID",
    "CROSS_ASSET_STATE_SHARED_ADAPTER_METHODS",
    "CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_SHARED_METHODS",
    "CROSS_ASSET_STATE_SPLIT_POLICY_ID",
    "METHOD_STATUS_DIAGNOSTIC_ONLY_NOT_RUN",
    "METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED",
    "METHOD_STATUS_FALLBACK_ONLY",
    "METHOD_STATUS_SELECTABLE_PARTIAL",
    "CrossAssetStateMethodSpec",
    "cross_asset_state_method_universe_manifest",
    "default_cross_asset_state_method_universe",
    "validate_cross_asset_state_method_universe_manifest",
]
