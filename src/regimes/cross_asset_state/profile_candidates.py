from __future__ import annotations

import time
import warnings
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.clusterer_base import ASSIGN_STATUS_ASSIGNED, FIT_STATUS_FITTED
from src.regimes.core.clusterer_registry import default_clusterer_registry
from src.regimes.core.clustering_candidates import candidate_spec_for_method
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.feature_preprocessing import fit_regime_preprocessor, transform_regime_preprocessor
from src.regimes.core.test_branch_maturity import (
    FILTER_LOW_FEATURE_SPREAD,
    FILTER_PROFILE_TYPE_NOT_SELECTION_ELIGIBLE,
    METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED,
    METHOD_STATUS_SELECTABLE_PARTIAL,
    candidate_selection_status,
    feature_spread_diagnostics,
)
from src.regimes.cross_asset_state.method_universe import (
    CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
    CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
)
from src.regimes.cross_asset_state.output_health import (
    HEALTH_WARNING_BIRCH_TOO_FEW_SUBCLUSTERS,
    evaluate_output_health,
    label_counts,
)
from src.regimes.cross_asset_state.scoring_contract import CrossAssetStateDiagnosticScore, bounded_score
from src.regimes.cross_asset_state.search_spaces import CROSS_ASSET_STATE_SEARCH_SPACE_ID


PROFILE_RULE = "rule_threshold"
PROFILE_ORDINAL = "ordinal_quantile"
PROFILE_KMEANS = "kmeans"
PROFILE_MINIBATCH_KMEANS = "minibatch_kmeans"
PROFILE_PCA_KMEANS = "pca_kmeans"
PROFILE_FACTOR_ANALYSIS_KMEANS = "factor_analysis_kmeans"
PROFILE_GAUSSIAN_MIXTURE = "gaussian_mixture"
PROFILE_FACTOR_ANALYSIS_GAUSSIAN_MIXTURE = "factor_analysis_gaussian_mixture"
PROFILE_BAYESIAN_GAUSSIAN_MIXTURE = "bayesian_gaussian_mixture"
PROFILE_BIRCH = "birch"
PROFILE_HDBSCAN = "hdbscan"
PROFILE_OPTICS = "optics"
PROFILE_AGGLOMERATIVE = "agglomerative"
PROFILE_DIAGNOSTIC_ONLY = "diagnostic_only"
MISSING_SCORING_COLUMN = "missing_scoring_column"
CROSS_ASSET_STATE_SPLIT_POLICY_ID = "cross_asset_state_time_ordered_train60_score40_v1"
CROSS_ASSET_STATE_VALIDATION_SCOPE = "validation_and_holdout_score_window"
CROSS_ASSET_STATE_PREPROCESSING_PROFILE = "noop"
CROSS_ASSET_STATE_ADAPTER_METHODS: tuple[str, ...] = (
    PROFILE_KMEANS,
    PROFILE_MINIBATCH_KMEANS,
    PROFILE_PCA_KMEANS,
    PROFILE_FACTOR_ANALYSIS_KMEANS,
    PROFILE_GAUSSIAN_MIXTURE,
    PROFILE_FACTOR_ANALYSIS_GAUSSIAN_MIXTURE,
    PROFILE_BAYESIAN_GAUSSIAN_MIXTURE,
    PROFILE_BIRCH,
    PROFILE_HDBSCAN,
    PROFILE_OPTICS,
    PROFILE_AGGLOMERATIVE,
)
CROSS_ASSET_STATE_DIAGNOSTIC_ONLY_ADAPTER_METHODS: tuple[str, ...] = (
    PROFILE_BIRCH,
)
OLD_FIXED_TOP3_COLUMNS: tuple[str, ...] = (
    "strongest_peer_slot_1_strength",
    "top_peer_count",
    "top_peer_stability_mean",
    "relationship_concentration",
    "relationship_entropy",
)
PEER_STRENGTH_COLUMNS: tuple[str, ...] = (
    "strongest_peer_slot_1_strength",
    "total_peer_strength",
    "avg_peer_strength",
    "median_peer_strength",
    "top1_peer_strength",
    "top2_peer_strength",
    "top3_peer_strength",
    "top5_peer_strength",
    "top7_peer_strength",
    "persistence_weighted_peer_strength",
)
PEER_SUPPORT_COLUMNS: tuple[str, ...] = (
    "top_peer_count",
    "eligible_peer_count",
    "peer_count_above_threshold",
    "top1_to_total_ratio",
    "top3_to_total_ratio",
    "top1_to_top5_ratio",
    "top3_to_top5_ratio",
    "top1_to_top7_ratio",
    "top3_to_top7_ratio",
)
PEER_STABILITY_COLUMNS: tuple[str, ...] = (
    "top_peer_stability_mean",
    "peer_strength_dispersion",
    "peer_strength_iqr",
    "peer_strength_slope_by_rank",
    "peer_stability_dispersion",
    "peer_stability_iqr",
    "peer_strength_dispersion_top7",
    "peer_strength_iqr_top7",
    "peer_strength_slope_by_rank_top7",
    "peer_stability_dispersion_top7",
    "peer_stability_iqr_top7",
    "peer_membership_churn",
    "peer_set_jaccard_vs_prior_snapshot",
)
RELATIONSHIP_CONCENTRATION_COLUMNS: tuple[str, ...] = (
    "relationship_concentration",
    "max_peer_share",
    "top1_share",
    "top3_share",
    "top5_share",
    "top7_share",
    "hhi_top7_peers",
    "hhi_all_eligible_peers",
    "peer_weight_gini_top7",
    "peer_weight_gini",
)
RELATIONSHIP_ENTROPY_COLUMNS: tuple[str, ...] = (
    "relationship_entropy",
    "raw_entropy_top7",
    "normalized_entropy_top7",
    "raw_entropy_variable_support",
    "normalized_entropy_variable_support",
    "effective_peer_count_top7",
    "effective_peer_count",
)
RELATIONSHIP_SPREAD_COLUMNS: tuple[str, ...] = (
    "edge_weight_spread_top7",
    "edge_weight_iqr_top7",
    "edge_weight_spread",
    "edge_weight_iqr",
    "peer_count_to_50pct_mass",
    "peer_count_to_80pct_mass",
)


class MissingScoringColumnError(ValueError):
    def __init__(self, missing_columns: Sequence[str], *, family: str | None = None, available_columns: Sequence[str] = ()) -> None:
        self.missing_columns = tuple(str(column) for column in missing_columns)
        self.family = family
        self.available_columns = tuple(str(column) for column in available_columns)
        super().__init__(
            f"{MISSING_SCORING_COLUMN}: missing={list(self.missing_columns)}"
            + (f" family={family}" if family else "")
        )


@dataclass(frozen=True)
class CrossAssetStateProfileCandidateResult:
    relationship_feature_family: str
    profile_type: str
    candidate_id: str
    parameter_grid_id: str
    labels: tuple[str, ...]
    feature_columns: tuple[str, ...]
    diagnostic_score: CrossAssetStateDiagnosticScore
    output_health: Mapping[str, Any]
    label_counts: Mapping[str, int]
    selected_status: str
    method_family: str | None = None
    clusterer_family: str | None = None
    embedding: str = "none"
    shared_adapter_used: bool = False
    adapter_name: str | None = None
    split_policy_id: str | None = None
    validation_assignment_policy: str | None = None
    validation_assignment_status: str | None = None
    validation_assignment_scope: str | None = None
    validation_health: Mapping[str, Any] | None = None
    train_row_count: int | None = None
    validation_row_count: int | None = None
    holdout_row_count: int | None = None
    readiness_status: str = METHOD_STATUS_SELECTABLE_PARTIAL
    selection_eligible: bool = True
    diagnostic_only: bool = False
    selection_exclusion_reason: str | None = None
    search_space_id: str = CROSS_ASSET_STATE_SEARCH_SPACE_ID
    method_universe_version: str = CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION
    profile_candidate_set_id: str = CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID
    candidate_params: Mapping[str, Any] | None = None
    family_transform_id: str | None = None
    feature_diagnostics: Mapping[str, Any] | None = None
    failure_mode: str | None = None
    warning_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "relationship_feature_family": self.relationship_feature_family,
            "profile_type": self.profile_type,
            "candidate_id": self.candidate_id,
            "parameter_grid_id": self.parameter_grid_id,
            "method_family": self.method_family,
            "clusterer_family": self.clusterer_family,
            "embedding": self.embedding,
            "shared_adapter_used": bool(self.shared_adapter_used),
            "adapter_name": self.adapter_name,
            "split_policy_id": self.split_policy_id,
            "validation_assignment_policy": self.validation_assignment_policy,
            "validation_assignment_status": self.validation_assignment_status,
            "validation_assignment_scope": self.validation_assignment_scope,
            "validation_health": dict(self.validation_health or {}),
            "train_row_count": self.train_row_count,
            "validation_row_count": self.validation_row_count,
            "holdout_row_count": self.holdout_row_count,
            "method_universe_version": self.method_universe_version,
            "profile_candidate_set_id": self.profile_candidate_set_id,
            "readiness_status": self.readiness_status,
            "selection_eligible": bool(self.selection_eligible),
            "diagnostic_only": bool(self.diagnostic_only),
            "selection_exclusion_reason": self.selection_exclusion_reason,
            "search_space_id": self.search_space_id,
            "candidate_params": dict(self.candidate_params or {}),
            "family_transform_id": self.family_transform_id,
            "feature_diagnostics": dict(self.feature_diagnostics or {}),
            "feature_columns": list(self.feature_columns),
            "label_counts": dict(self.label_counts),
            "selected_status": self.selected_status,
            "failure_mode": self.failure_mode,
            "warning_reasons": list(self.warning_reasons),
            "output_health": dict(self.output_health),
            "diagnostic_score": self.diagnostic_score.as_dict(),
        }


@dataclass(frozen=True)
class CrossAssetStateCandidateDefinitionBatch:
    fingerprint: Mapping[str, Any]
    candidate_set_id: str
    method_universe_version: str
    search_space_id: str
    definitions: tuple[Mapping[str, Any], ...]

    def by_profile_type(self, profile_type: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(definition for definition in self.definitions if str(definition.get("profile_type")) == str(profile_type))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_candidate_definition_batch",
            "fingerprint": dict(self.fingerprint),
            "candidate_set_id": self.candidate_set_id,
            "method_universe_version": self.method_universe_version,
            "search_space_id": self.search_space_id,
            "definition_count": len(self.definitions),
            "definitions": [dict(definition) for definition in self.definitions],
            "fitted_models_reused": False,
        }


@dataclass
class CrossAssetStateCandidateDefinitionCache:
    _definitions: dict[tuple[Any, ...], CrossAssetStateCandidateDefinitionBatch] = field(default_factory=dict)
    lookup_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    build_count: int = 0
    candidate_definition_reused_count: int = 0
    candidate_eval_count: int = 0
    build_seconds: float = 0.0

    key_fields: tuple[str, ...] = (
        "family",
        "band",
        "window_policy_id",
        "feature_set_version",
        "method_universe_version",
        "search_space_id",
        "feature_columns",
        "row_count_bucket",
        "feature_count",
    )

    def definitions(
        self,
        *,
        family: str,
        band: str,
        window_policy_id: str,
        feature_set_version: str,
        method_universe_version: str = CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
        search_space_id: str = CROSS_ASSET_STATE_SEARCH_SPACE_ID,
        feature_columns: Sequence[str],
        row_count: int,
        feature_count: int,
    ) -> CrossAssetStateCandidateDefinitionBatch:
        key = self._key(
            family=family,
            band=band,
            window_policy_id=window_policy_id,
            feature_set_version=feature_set_version,
            method_universe_version=method_universe_version,
            search_space_id=search_space_id,
            feature_columns=feature_columns,
            row_count=row_count,
            feature_count=feature_count,
        )
        self.lookup_count += 1
        cached = self._definitions.get(key)
        if cached is not None:
            self.hit_count += 1
            self.candidate_definition_reused_count += len(cached.definitions)
            return cached
        self.miss_count += 1
        started = time.perf_counter()
        batch = _build_candidate_definition_batch(
            family=family,
            band=band,
            window_policy_id=window_policy_id,
            feature_set_version=feature_set_version,
            method_universe_version=method_universe_version,
            search_space_id=search_space_id,
            feature_columns=feature_columns,
            row_count=row_count,
            feature_count=feature_count,
        )
        self.build_seconds += time.perf_counter() - started
        self.build_count += 1
        self._definitions[key] = batch
        return batch

    def record_candidate_evaluations(self, count: int) -> None:
        self.candidate_eval_count += int(count)

    def stats(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_candidate_definition_cache_telemetry",
            "cache_scope": "per_run_in_memory",
            "disk_cache_enabled": False,
            "key_fields": list(self.key_fields),
            "lookup_count": int(self.lookup_count),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
            "candidate_definitions_built": int(self.build_count),
            "candidate_definition_reused_count": int(self.candidate_definition_reused_count),
            "candidate_eval_count": int(self.candidate_eval_count),
            "entry_count": int(len(self._definitions)),
            "build_seconds": round(float(self.build_seconds), 6),
            "candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
            "method_universe_version": CROSS_ASSET_STATE_METHOD_UNIVERSE_VERSION,
            "search_space_id": CROSS_ASSET_STATE_SEARCH_SPACE_ID,
            "reuse_policy": "definition_metadata_only_no_fitted_models_reused",
            "fitted_model_reuse_enabled": False,
            "stale_reuse_guard": {
                "feature_set_version_in_key": True,
                "method_universe_version_in_key": True,
                "search_space_id_in_key": True,
                "window_policy_id_in_key": True,
                "family_in_key": True,
                "band_in_key": True,
            },
        }

    def _key(
        self,
        *,
        family: str,
        band: str,
        window_policy_id: str,
        feature_set_version: str,
        method_universe_version: str,
        search_space_id: str,
        feature_columns: Sequence[str],
        row_count: int,
        feature_count: int,
    ) -> tuple[Any, ...]:
        return (
            str(family),
            str(band),
            str(window_policy_id),
            str(feature_set_version),
            str(method_universe_version),
            str(search_space_id),
            tuple(str(column) for column in feature_columns),
            _row_count_bucket(row_count),
            int(feature_count),
        )


def run_profile_candidates(
    frame: Any,
    *,
    family: str,
    feature_columns: Sequence[str],
    candidate_definitions: CrossAssetStateCandidateDefinitionBatch | None = None,
) -> tuple[CrossAssetStateProfileCandidateResult, ...]:
    candidates: list[CrossAssetStateProfileCandidateResult] = []
    try:
        candidates.extend(_run_rule_profiles(frame, family=family, feature_columns=feature_columns, candidate_definitions=candidate_definitions))
        candidates.extend(_run_ordinal_profiles(frame, family=family, feature_columns=feature_columns, candidate_definitions=candidate_definitions))
        candidates.extend(_run_shared_adapter_profiles(frame, family=family, feature_columns=feature_columns, candidate_definitions=candidate_definitions))
    except MissingScoringColumnError as exc:
        return (
            diagnostic_only_result(
                frame,
                family=family,
                feature_columns=feature_columns,
                reason=MISSING_SCORING_COLUMN,
                missing_columns=exc.missing_columns,
            ),
        )
    return tuple(candidates)


def candidate_definition_shape(frame: Any, *, family: str, feature_columns: Sequence[str]) -> dict[str, int]:
    data = _numeric_frame(frame, feature_columns)
    cluster_data, _ = _cluster_frame(data, family=family)
    return {
        "row_count": int(len(cluster_data)),
        "feature_count": int(cluster_data.shape[1] if hasattr(cluster_data, "shape") else len(tuple(feature_columns))),
    }


def choose_best_candidate(candidates: Sequence[CrossAssetStateProfileCandidateResult]) -> CrossAssetStateProfileCandidateResult:
    viable = [
        candidate
        for candidate in candidates
        if candidate.output_health.get("passed") and candidate.selection_eligible and not candidate.diagnostic_only
    ]
    pool = viable or list(candidates)
    return max(
        pool,
        key=lambda candidate: (
            bool(candidate.output_health.get("passed", False)),
            bool(candidate.selection_eligible and not candidate.diagnostic_only),
            candidate.diagnostic_score.total_candidate_score,
            candidate.diagnostic_score.runtime_tiebreak_score,
        ),
    )


def _build_candidate_definition_batch(
    *,
    family: str,
    band: str,
    window_policy_id: str,
    feature_set_version: str,
    method_universe_version: str,
    search_space_id: str,
    feature_columns: Sequence[str],
    row_count: int,
    feature_count: int,
) -> CrossAssetStateCandidateDefinitionBatch:
    fingerprint = {
        "family": str(family),
        "band": str(band),
        "window_policy_id": str(window_policy_id),
        "feature_set_version": str(feature_set_version),
        "method_universe_version": str(method_universe_version),
        "search_space_id": str(search_space_id),
        "feature_columns": [str(column) for column in feature_columns],
        "row_count_bucket": _row_count_bucket(row_count),
        "feature_count": int(feature_count),
        "candidate_set_id": CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
    }
    definitions = (
        *_rule_profile_definitions(family),
        *_ordinal_profile_definitions(family),
        *_adapter_profile_definitions(row_count=row_count, feature_count=feature_count),
    )
    return CrossAssetStateCandidateDefinitionBatch(
        fingerprint=fingerprint,
        candidate_set_id=CROSS_ASSET_STATE_PROFILE_CANDIDATE_SET_ID,
        method_universe_version=method_universe_version,
        search_space_id=search_space_id,
        definitions=definitions,
    )


def _row_count_bucket(row_count: int) -> str:
    return f"rows={int(row_count)}"


def _rule_profile_definitions(family: str) -> tuple[dict[str, Any], ...]:
    if family == "anchor_core_exposure":
        variants = (
            ("balanced_coupling", {"low_max": 0.35, "high_min": 0.75}),
            ("beta_core_joint", {"low_max": 0.33, "high_min": 0.66}),
            ("core_secondary_joint", {"low_max": 0.33, "high_min": 0.66}),
        )
    elif family == "peer_strength_stability":
        variants = (
            ("strength_stability_joint", {"low_max": 0.33, "high_min": 0.66}),
            ("count_strength_joint", {"low_max": 0.33, "high_min": 0.66}),
            ("ranked_peer_context", {"low_max": 0.33, "high_min": 0.66}),
        )
    elif family == "relationship_concentration_entropy":
        variants = (
            ("concentration_entropy_joint", {"low_max": 0.33, "high_min": 0.66}),
            ("spread_guarded_rank", {"low_max": 0.33, "high_min": 0.66}),
            ("concentration_only", {"low_max": 0.33, "high_min": 0.66}),
        )
    elif family == "residual_peer_signal":
        variants = (
            ("signed_standard", {"negative_max": -0.35, "positive_min": 0.35}),
            ("signed_narrow", {"negative_max": -0.20, "positive_min": 0.20}),
            ("signed_magnitude", {"neutral_abs_max": 0.20, "large_abs_min": 0.60}),
        )
    else:
        variants = (("single_axis_rank", {"low_max": 0.33, "high_min": 0.66}),)
    return tuple(
        {
            "definition_kind": "rule_variant",
            "profile_type": PROFILE_RULE,
            "variant_id": variant_id,
            "candidate_id": f"{PROFILE_RULE}:{family}:{variant_id}",
            "parameter_grid_id": f"{PROFILE_RULE}_grid_v2:{variant_id}",
            "candidate_params": dict(params),
            "fitted_model_definition": False,
        }
        for variant_id, params in variants
    )


def _ordinal_profile_definitions(family: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "definition_kind": "ordinal_variant",
            "profile_type": PROFILE_ORDINAL,
            "variant_id": f"q{bins}",
            "bins": bins,
            "candidate_id": f"{PROFILE_ORDINAL}:{family}:q{bins}",
            "parameter_grid_id": f"{PROFILE_ORDINAL}_grid_v2:q{bins}",
            "candidate_params": {"bins": bins, "tie_method": "first"},
            "fitted_model_definition": False,
        }
        for bins in (2, 3, 4)
    )


def _adapter_profile_definitions(*, row_count: int, feature_count: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "definition_kind": "adapter_arm",
            "profile_type": str(arm["profile_type"]),
            "candidate_arm": dict(arm),
            "grid_id": str(arm.get("grid_id")),
            "grid_suffix": str(arm.get("grid_suffix")),
            "fitted_model_definition": False,
        }
        for arm in _adapter_method_arms(row_count=row_count, feature_count=feature_count)
    )


def _definitions_by_kind(
    candidate_definitions: CrossAssetStateCandidateDefinitionBatch | None,
    kind: str,
) -> tuple[Mapping[str, Any], ...]:
    if candidate_definitions is None:
        return ()
    return tuple(definition for definition in candidate_definitions.definitions if str(definition.get("definition_kind")) == str(kind))


def _run_rule_profiles(
    frame: Any,
    *,
    family: str,
    feature_columns: Sequence[str],
    candidate_definitions: CrossAssetStateCandidateDefinitionBatch | None = None,
) -> tuple[CrossAssetStateProfileCandidateResult, ...]:
    data = _numeric_frame(frame, feature_columns)
    diagnostics = _feature_diagnostics(data, family=family)
    candidates: list[CrossAssetStateProfileCandidateResult] = []
    definition_rows = _definitions_by_kind(candidate_definitions, "rule_variant")
    rule_variants = (
        tuple(_rule_labels_from_definition(data, family=family, definition=definition) for definition in definition_rows)
        if definition_rows
        else _rule_label_variants(data, family=family)
    )
    for variant_id, params, labels in rule_variants:
        started = time.perf_counter()
        candidates.append(
            _result(
                frame,
                family,
                PROFILE_RULE,
                feature_columns,
                labels,
                started,
                candidate_id=f"{PROFILE_RULE}:{family}:{variant_id}",
                parameter_grid_id=f"{PROFILE_RULE}_grid_v2:{variant_id}",
                candidate_params=params,
                family_transform_id=params.get("family_transform_id"),
                feature_diagnostics=diagnostics,
                warning_reasons=diagnostics.get("warning_reasons", ()),
            )
        )
    return tuple(candidates)


def _run_ordinal_profiles(
    frame: Any,
    *,
    family: str,
    feature_columns: Sequence[str],
    candidate_definitions: CrossAssetStateCandidateDefinitionBatch | None = None,
) -> tuple[CrossAssetStateProfileCandidateResult, ...]:
    data = _numeric_frame(frame, feature_columns)
    diagnostics = _feature_diagnostics(data, family=family)
    score = _family_score(data, family=family)
    candidates: list[CrossAssetStateProfileCandidateResult] = []
    definition_rows = _definitions_by_kind(candidate_definitions, "ordinal_variant")
    ordinal_defs = definition_rows or _ordinal_profile_definitions(family)
    for definition in ordinal_defs:
        bins = int(definition.get("bins") or str(definition.get("variant_id") or "q2").lstrip("q") or 2)
        started = time.perf_counter()
        guarded = family == "relationship_concentration_entropy" and diagnostics.get("low_spread_warning")
        labels = [f"ordinal_q{bins}_low_spread_single"] * len(score) if guarded else _quantile_labels(score, bins=bins, prefix=f"ordinal_q{bins}")
        params = {**dict(definition.get("candidate_params") or {}), "spread_guarded": guarded}
        candidates.append(
            _result(
                frame,
                family,
                PROFILE_ORDINAL,
                feature_columns,
                labels,
                started,
                candidate_id=str(definition.get("candidate_id") or f"{PROFILE_ORDINAL}:{family}:q{bins}"),
                parameter_grid_id=str(definition.get("parameter_grid_id") or f"{PROFILE_ORDINAL}_grid_v2:q{bins}"),
                candidate_params=params,
                family_transform_id=f"{family}_rank_score_v1",
                feature_diagnostics=diagnostics,
                warning_reasons=diagnostics.get("warning_reasons", ()),
            )
        )
    return tuple(candidates)


def _run_shared_adapter_profiles(
    frame: Any,
    *,
    family: str,
    feature_columns: Sequence[str],
    candidate_definitions: CrossAssetStateCandidateDefinitionBatch | None = None,
) -> tuple[CrossAssetStateProfileCandidateResult, ...]:
    data = _numeric_frame(frame, feature_columns)
    cluster_data, transform_id = _cluster_frame(data, family=family)
    diagnostics = _feature_diagnostics(data, family=family)
    candidates: list[CrossAssetStateProfileCandidateResult] = []
    split = _time_ordered_split(cluster_data)
    adapter_definitions = _definitions_by_kind(candidate_definitions, "adapter_arm")
    arms = (
        tuple(dict(definition.get("candidate_arm") or {}) for definition in adapter_definitions)
        if adapter_definitions
        else _adapter_method_arms(row_count=len(cluster_data), feature_count=int(cluster_data.shape[1] if hasattr(cluster_data, "shape") else 0))
    )
    for arm in arms:
        started = time.perf_counter()
        params = dict(arm["hyperparameters"])
        warning_reasons: tuple[str, ...] = tuple(diagnostics.get("warning_reasons", ()))
        if diagnostics.get("low_spread_warning"):
            labels, metadata = _skipped_adapter_labels(cluster_data, split=split, arm=arm, reason=FILTER_LOW_FEATURE_SPREAD)
        else:
            labels, metadata = _fit_assign_adapter_labels(cluster_data, split=split, arm=arm)
        warning_reasons = tuple([*warning_reasons, *metadata.get("warning_reasons", ())])
        candidates.append(
            _result(
                frame,
                family,
                str(arm["profile_type"]),
                feature_columns,
                labels,
                started,
                candidate_id=f"{arm['method_family']}:{family}:{arm['grid_suffix']}",
                parameter_grid_id=f"{arm['grid_id']}:{arm['grid_suffix']}",
                candidate_params={**params, **metadata},
                family_transform_id=transform_id,
                method_family=str(arm["method_family"]),
                clusterer_family=str(arm["clusterer_family"]),
                embedding=str(arm["embedding"]),
                shared_adapter_used=True,
                adapter_name=metadata.get("adapter_name"),
                split_policy_id=CROSS_ASSET_STATE_SPLIT_POLICY_ID,
                validation_assignment_policy=metadata.get("validation_assignment_policy"),
                validation_assignment_status=metadata.get("validation_assignment_status"),
                validation_assignment_scope=CROSS_ASSET_STATE_VALIDATION_SCOPE,
                validation_health=metadata.get("validation_health"),
                train_row_count=split["train_row_count"],
                validation_row_count=split["validation_row_count"],
                holdout_row_count=split["holdout_row_count"],
                readiness_status=arm["readiness_status"],
                selection_eligible=bool(arm["selection_eligible"]),
                diagnostic_only=bool(arm["diagnostic_only"]),
                selection_exclusion_reason=arm["selection_exclusion_reason"],
                feature_diagnostics=diagnostics,
                warning_reasons=warning_reasons,
            )
        )
    return tuple(candidates)


def _time_ordered_split(data: Any) -> dict[str, Any]:
    row_count = int(len(data))
    if row_count <= 1:
        train_end = row_count
        validation_end = row_count
    elif row_count < 8:
        train_end = max(1, row_count - 1)
        validation_end = row_count
    else:
        train_end = max(4, int(row_count * 0.60))
        train_end = min(train_end, max(1, row_count - 2))
        validation_rows = max(1, int(row_count * 0.20))
        validation_end = min(row_count, train_end + validation_rows)
        if validation_end <= train_end:
            validation_end = min(row_count, train_end + 1)
    return {
        "split_policy_id": CROSS_ASSET_STATE_SPLIT_POLICY_ID,
        "train_frame": data.iloc[:train_end].copy(),
        "validation_frame": data.iloc[train_end:validation_end].copy(),
        "holdout_frame": data.iloc[validation_end:].copy(),
        "score_frame": data.iloc[train_end:].copy(),
        "train_row_count": int(train_end),
        "validation_row_count": int(max(0, validation_end - train_end)),
        "holdout_row_count": int(max(0, row_count - validation_end)),
        "score_row_count": int(max(0, row_count - train_end)),
    }


def _adapter_method_arms(*, row_count: int, feature_count: int) -> tuple[dict[str, Any], ...]:
    counts = tuple(count for count in _cluster_count_grid(row_count) if count <= 3) or (2,)
    small_counts = counts[:1]
    component_counts = small_counts
    embedding_components = (min(2, max(1, feature_count)),)
    arms: list[dict[str, Any]] = []

    def add(
        method_family: str,
        *,
        hyperparameters: Mapping[str, Any],
        grid_suffix: str,
        selection_eligible: bool = True,
        diagnostic_only: bool = False,
        readiness_status: str = METHOD_STATUS_SELECTABLE_PARTIAL,
        selection_exclusion_reason: str | None = None,
        grid_id: str | None = None,
    ) -> None:
        spec = candidate_spec_for_method(method_family)
        arms.append(
            {
                "method_family": method_family,
                "profile_type": method_family,
                "clusterer_family": spec.clusterer_family,
                "embedding": spec.embedding,
                "hyperparameters": dict(hyperparameters),
                "grid_suffix": grid_suffix,
                "grid_id": grid_id or f"{method_family}_adapter_bounded_grid_v2",
                "selection_eligible": bool(selection_eligible),
                "diagnostic_only": bool(diagnostic_only),
                "readiness_status": readiness_status,
                "selection_exclusion_reason": selection_exclusion_reason,
            }
        )

    for n_clusters in counts:
        add(PROFILE_KMEANS, hyperparameters={"n_clusters": n_clusters, "n_init": 10, "random_state": 17, "init": "k-means++"}, grid_suffix=f"k{n_clusters}")
    for n_clusters in small_counts:
        add(
            PROFILE_MINIBATCH_KMEANS,
            hyperparameters={"n_clusters": n_clusters, "batch_size": 64, "n_init": 3, "random_state": 17, "init": "k-means++"},
            grid_suffix=f"k{n_clusters}:batch64",
        )
    for n_clusters in small_counts:
        for n_components in embedding_components:
            add(
                PROFILE_PCA_KMEANS,
                hyperparameters={"n_clusters": n_clusters, "n_init": 10, "random_state": 17, "init": "k-means++", "embedding__n_components": n_components},
                grid_suffix=f"k{n_clusters}:pca{n_components}",
            )
            add(
                PROFILE_FACTOR_ANALYSIS_KMEANS,
                hyperparameters={"n_clusters": n_clusters, "n_init": 10, "random_state": 17, "init": "k-means++", "embedding__n_components": n_components},
                grid_suffix=f"k{n_clusters}:fa{n_components}",
            )
    for n_components in component_counts:
        covariance_type, reg_covar = ("full", 1e-6)
        add(
            PROFILE_GAUSSIAN_MIXTURE,
            hyperparameters={"n_components": n_components, "covariance_type": covariance_type, "reg_covar": reg_covar, "random_state": 17, "n_init": 2},
            grid_suffix=f"k{n_components}:{covariance_type}:reg{reg_covar}",
        )
        add(
            PROFILE_BAYESIAN_GAUSSIAN_MIXTURE,
            hyperparameters={"n_components": n_components, "covariance_type": covariance_type, "reg_covar": reg_covar, "random_state": 17, "n_init": 1},
            grid_suffix=f"k{n_components}:{covariance_type}:reg{reg_covar}",
        )
        for embedding_n in embedding_components[:1]:
            add(
                PROFILE_FACTOR_ANALYSIS_GAUSSIAN_MIXTURE,
                hyperparameters={
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "reg_covar": reg_covar,
                    "random_state": 17,
                    "n_init": 2,
                    "embedding__n_components": embedding_n,
                },
                grid_suffix=f"k{n_components}:{covariance_type}:fa{embedding_n}:reg{reg_covar}",
            )
    for n_clusters, threshold, branching_factor in _birch_grid(row_count)[:1]:
        add(
            PROFILE_BIRCH,
            hyperparameters={"n_clusters": n_clusters, "threshold": threshold, "branching_factor": branching_factor},
            grid_suffix=f"k{n_clusters}:thr{threshold}:bf{branching_factor}",
            grid_id="birch_adapter_diagnostic_bounded_grid_v3",
            selection_eligible=False,
            diagnostic_only=True,
            readiness_status=METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED,
            selection_exclusion_reason=FILTER_PROFILE_TYPE_NOT_SELECTION_ELIGIBLE,
        )
    for min_cluster_size in (max(2, min(5, row_count // 2)),):
        add(
            PROFILE_HDBSCAN,
            hyperparameters={
                "min_cluster_size": min_cluster_size,
                "min_samples": 1,
                "allow_single_cluster": True,
                "prediction_data": True,
                "cluster_selection_method": "eom",
            },
            grid_suffix=f"mcs{min_cluster_size}:ms1",
            grid_id="hdbscan_adapter_bounded_density_grid_v3",
        )
    for min_samples in (2,):
        add(
            PROFILE_OPTICS,
            hyperparameters={
                "min_samples": min_samples,
                "xi": 0.05,
                "cluster_method": "xi",
                "max_eps": float("inf"),
                "metric": "euclidean",
                "assignment_k": 1,
                "assignment_threshold_quantile": 0.95,
                "assignment_threshold_multiplier": 1.25,
                "unknown_label": -1,
                "exclude_noise_from_assignment": True,
            },
            grid_suffix=f"ms{min_samples}:xi005",
            grid_id="optics_adapter_bounded_density_grid_v3",
        )
    for n_clusters, linkage, metric in _agglomerative_grid(row_count):
        add(
            PROFILE_AGGLOMERATIVE,
            hyperparameters={
                "n_clusters": n_clusters,
                "linkage": linkage,
                "metric": metric,
                "assignment_threshold_quantile": 0.95,
                "assignment_threshold_multiplier": 1.25,
                "unknown_label": -1,
                "exclude_noise_from_assignment": True,
            },
            grid_suffix=f"k{n_clusters}:{linkage}:{metric}",
        )
    return tuple(arms)


def _skipped_adapter_labels(data: Any, *, split: Mapping[str, Any], arm: Mapping[str, Any], reason: str) -> tuple[list[str], dict[str, Any]]:
    method_family = str(arm["method_family"])
    clusterer_family = str(arm["clusterer_family"])
    embedding = str(arm.get("embedding") or "none")
    hyperparameters = dict(arm.get("hyperparameters") or {})
    warning_reasons = [reason]
    if method_family == PROFILE_BIRCH and reason == FILTER_LOW_FEATURE_SPREAD:
        warning_reasons.append(HEALTH_WARNING_BIRCH_TOO_FEW_SUBCLUSTERS)
    metadata: dict[str, Any] = {
        "shared_adapter_used": True,
        "adapter_name": f"{clusterer_family}_shared_adapter_not_run",
        "adapter_execution_status": "skipped",
        "adapter_execution_skip_reason": reason,
        "split_policy_id": CROSS_ASSET_STATE_SPLIT_POLICY_ID,
        "validation_assignment_policy": f"not_run_{reason}",
        "validation_assignment_status": "not_run",
        "validation_assignment_scope": CROSS_ASSET_STATE_VALIDATION_SCOPE,
        "clusterer_family": clusterer_family,
        "method_family": method_family,
        "embedding": embedding,
        "preprocessing_profile": CROSS_ASSET_STATE_PREPROCESSING_PROFILE,
        "train_row_count": int(split.get("train_row_count") or 0),
        "validation_row_count": int(split.get("validation_row_count") or 0),
        "holdout_row_count": int(split.get("holdout_row_count") or 0),
        "score_row_count": int(split.get("score_row_count") or 0),
        "warning_reasons": tuple(dict.fromkeys(warning_reasons)),
        "hyperparameters": hyperparameters,
    }
    labels = [f"{method_family}_{reason}"] * int(len(data))
    return labels, metadata


def _fit_assign_adapter_labels(data: Any, *, split: Mapping[str, Any], arm: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    method_family = str(arm["method_family"])
    clusterer_family = str(arm["clusterer_family"])
    embedding = str(arm.get("embedding") or "none")
    hyperparameters = dict(arm.get("hyperparameters") or {})
    embedding_params = {str(key)[len("embedding__") :]: value for key, value in hyperparameters.items() if str(key).startswith("embedding__")}
    clusterer_params = {key: value for key, value in hyperparameters.items() if not str(key).startswith("embedding__")}
    columns = tuple(str(column) for column in getattr(data, "columns", ()))
    metadata: dict[str, Any] = {
        "shared_adapter_used": True,
        "adapter_name": f"{clusterer_family}_shared_adapter",
        "split_policy_id": CROSS_ASSET_STATE_SPLIT_POLICY_ID,
        "validation_assignment_scope": CROSS_ASSET_STATE_VALIDATION_SCOPE,
        "clusterer_family": clusterer_family,
        "method_family": method_family,
        "embedding": embedding,
        "preprocessing_profile": CROSS_ASSET_STATE_PREPROCESSING_PROFILE,
        "train_row_count": int(split.get("train_row_count") or 0),
        "validation_row_count": int(split.get("validation_row_count") or 0),
        "holdout_row_count": int(split.get("holdout_row_count") or 0),
        "score_row_count": int(split.get("score_row_count") or 0),
        "warning_reasons": (),
    }
    if not columns or int(split.get("train_row_count") or 0) < 2 or int(split.get("score_row_count") or 0) < 1:
        metadata.update({"fit_status": "skipped", "validation_assignment_status": "skipped", "failure_reason": "insufficient_split_rows"})
        return [f"{method_family}_insufficient_split"] * int(len(data)), metadata
    try:
        preprocess_name = embedding if embedding != "none" else CROSS_ASSET_STATE_PREPROCESSING_PROFILE
        fitted = fit_regime_preprocessor(
            split["train_frame"],
            columns,
            preprocess=preprocess_name,
            preprocess_params=embedding_params,
            fit_window_role="train",
        )
        score = transform_regime_preprocessor(split["score_frame"], fitted, window_role=CROSS_ASSET_STATE_VALIDATION_SCOPE)
        metadata["preprocessor"] = fitted.to_metadata()
        metadata["score_transform"] = score.to_metadata()
        if int(fitted.x.shape[0]) < 2 or int(score.x.shape[0]) < 1:
            warning_reasons = []
            if method_family == PROFILE_BIRCH:
                warning_reasons.append(HEALTH_WARNING_BIRCH_TOO_FEW_SUBCLUSTERS)
                metadata["birch_expected_subclusters"] = int(clusterer_params.get("n_clusters") or 0)
                metadata["birch_observed_subclusters"] = 0
            metadata.update(
                {
                    "fit_status": "skipped",
                    "validation_assignment_status": "skipped",
                    "failure_reason": "insufficient_clean_split_rows",
                    "warning_reasons": tuple(dict.fromkeys(warning_reasons)),
                }
            )
            return [f"{method_family}_insufficient_clean_split"] * int(len(data)), metadata
        adapter = default_clusterer_registry().build(clusterer_family, **clusterer_params)
        metadata["adapter_name"] = type(adapter).__name__
        metadata["adapter_capabilities"] = adapter.report_capabilities().as_dict()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_result = adapter.fit(fitted.x)
        warning_reasons = list(_warning_reasons(caught, profile_type=method_family))
        metadata["fit_status"] = fit_result.status
        metadata["fit_metadata"] = to_jsonable(dict(fit_result.metadata or {}))
        if fit_result.status != FIT_STATUS_FITTED:
            failure = fit_result.failure_metadata.as_dict() if fit_result.failure_metadata is not None else {}
            metadata.update({"fit_failure_metadata": failure, "validation_assignment_status": "not_applicable", "warning_reasons": tuple(warning_reasons)})
            return [f"{method_family}_fit_failed"] * int(len(data)), metadata
        with warnings.catch_warnings(record=True) as caught_assign:
            warnings.simplefilter("always")
            assignment = adapter.assign(score.x)
        warning_reasons.extend(_warning_reasons(caught_assign, profile_type=method_family))
        metadata["validation_assignment_policy"] = assignment.assignment_policy
        metadata["validation_assignment_status"] = assignment.status
        metadata["assignment_metadata"] = to_jsonable(dict(assignment.metadata or {}))
        metadata["warning_reasons"] = tuple(dict.fromkeys(warning_reasons))
        if assignment.status != ASSIGN_STATUS_ASSIGNED:
            failure = assignment.failure_metadata.as_dict() if assignment.failure_metadata is not None else {}
            metadata["assignment_failure_metadata"] = failure
            return [f"{method_family}_assignment_failed"] * int(len(data)), metadata
        train_labels = _labels_to_ints(fit_result.labels)
        score_labels = _labels_to_ints(assignment.labels)
        score_label_strings = [f"{method_family}_{value}" for value in score_labels]
        metadata["validation_health"] = evaluate_output_health(score_label_strings, row_count=len(score_label_strings)).as_dict()
        metadata["train_label_count"] = len(train_labels)
        metadata["assigned_score_label_count"] = len(score_labels)
        labels = [f"{method_family}_{value}" for value in train_labels + score_labels]
        if method_family == PROFILE_BIRCH:
            expected_clusters = int(clusterer_params.get("n_clusters") or 0)
            observed_clusters = len(set(train_labels + score_labels))
            metadata["birch_expected_subclusters"] = expected_clusters
            metadata["birch_observed_subclusters"] = observed_clusters
            if expected_clusters > 0 and observed_clusters < expected_clusters:
                warning_reasons.append(HEALTH_WARNING_BIRCH_TOO_FEW_SUBCLUSTERS)
                metadata["warning_reasons"] = tuple(dict.fromkeys(warning_reasons))
        expected = int(len(data))
        if len(labels) < expected:
            labels.extend([f"{method_family}_dropped_nonfinite"] * (expected - len(labels)))
        return labels[:expected], metadata
    except Exception as exc:
        metadata.update(
            {
                "fit_status": "failed",
                "validation_assignment_status": "failed",
                "failure_reason": type(exc).__name__,
                "failure_message": str(exc),
            }
        )
        return [f"{method_family}_adapter_failed"] * int(len(data)), metadata


def _labels_to_ints(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        raw = values.tolist()
    else:
        raw = list(values or ())
    return [int(value) for value in raw]


def diagnostic_only_result(
    frame: Any,
    *,
    family: str,
    feature_columns: Sequence[str],
    reason: str,
    missing_columns: Sequence[str] = (),
) -> CrossAssetStateProfileCandidateResult:
    labels = ["diagnostic_only"] * int(len(frame))
    health = evaluate_output_health(labels, row_count=len(frame)).as_dict()
    missing = tuple(str(column) for column in missing_columns)
    score = CrossAssetStateDiagnosticScore(
        coverage_score=1.0 if len(frame) else 0.0,
        output_health_score=0.0,
        semantic_separation_score=0.0,
        temporal_persistence_score=0.0,
    )
    return CrossAssetStateProfileCandidateResult(
        relationship_feature_family=family,
        profile_type=PROFILE_DIAGNOSTIC_ONLY,
        candidate_id=f"{PROFILE_DIAGNOSTIC_ONLY}:{family}:fallback",
        parameter_grid_id=f"{PROFILE_DIAGNOSTIC_ONLY}_fallback_v1",
        labels=tuple(labels),
        feature_columns=tuple(feature_columns),
        diagnostic_score=score,
        output_health=health,
        label_counts=label_counts(labels),
        selected_status="diagnostic_only",
        readiness_status="fallback_only",
        selection_eligible=False,
        diagnostic_only=True,
        selection_exclusion_reason=reason,
        candidate_params={"reason": reason, "missing_scoring_columns": list(missing)},
        feature_diagnostics={
            "diagnostic_status": reason,
            "missing_scoring_columns": list(missing),
            "requested_scoring_columns": [str(column) for column in feature_columns],
        },
        failure_mode=reason,
    )


def _result(
    frame: Any,
    family: str,
    profile_type: str,
    feature_columns: Sequence[str],
    labels: list[str],
    started: float,
    *,
    candidate_id: str,
    parameter_grid_id: str,
    candidate_params: Mapping[str, Any] | None = None,
    method_family: str | None = None,
    clusterer_family: str | None = None,
    embedding: str = "none",
    shared_adapter_used: bool = False,
    adapter_name: str | None = None,
    split_policy_id: str | None = None,
    validation_assignment_policy: str | None = None,
    validation_assignment_status: str | None = None,
    validation_assignment_scope: str | None = None,
    validation_health: Mapping[str, Any] | None = None,
    train_row_count: int | None = None,
    validation_row_count: int | None = None,
    holdout_row_count: int | None = None,
    readiness_status: str = METHOD_STATUS_SELECTABLE_PARTIAL,
    selection_eligible: bool = True,
    diagnostic_only: bool = False,
    selection_exclusion_reason: str | None = None,
    family_transform_id: str | None = None,
    feature_diagnostics: Mapping[str, Any] | None = None,
    warning_reasons: Sequence[str] = (),
) -> CrossAssetStateProfileCandidateResult:
    data = _numeric_frame(frame, feature_columns)
    nonfinite_count = int(data.isna().sum().sum())
    health_obj = evaluate_output_health(labels, row_count=len(frame), nonfinite_count=nonfinite_count, warning_reasons=warning_reasons)
    health = health_obj.as_dict()
    guard_reason = _selection_guard_reason(family=family, profile_type=profile_type, feature_diagnostics=feature_diagnostics or {})
    effective_selection_eligible = bool(selection_eligible) and guard_reason is None
    effective_exclusion_reason = selection_exclusion_reason or guard_reason
    score = CrossAssetStateDiagnosticScore(
        coverage_score=bounded_score(len(labels) / max(1, len(frame))),
        output_health_score=1.0 if health_obj.passed else 0.0,
        semantic_separation_score=_semantic_separation(data, labels, family=family),
        temporal_persistence_score=_persistence_score(labels),
        runtime_seconds=time.perf_counter() - started,
        hard_health_failure=not health_obj.passed,
    )
    return CrossAssetStateProfileCandidateResult(
        relationship_feature_family=family,
        profile_type=profile_type,
        candidate_id=candidate_id,
        parameter_grid_id=parameter_grid_id,
        labels=tuple(labels),
        feature_columns=tuple(str(column) for column in feature_columns),
        diagnostic_score=score,
        output_health=health,
        label_counts=label_counts(labels),
        selected_status=_candidate_status(
            health_passed=health_obj.passed,
            selection_eligible=effective_selection_eligible,
            diagnostic_only=diagnostic_only,
        ),
        method_family=method_family or profile_type,
        clusterer_family=clusterer_family,
        embedding=embedding,
        shared_adapter_used=shared_adapter_used,
        adapter_name=adapter_name,
        split_policy_id=split_policy_id,
        validation_assignment_policy=validation_assignment_policy,
        validation_assignment_status=validation_assignment_status,
        validation_assignment_scope=validation_assignment_scope,
        validation_health=dict(validation_health or {}),
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
        holdout_row_count=holdout_row_count,
        readiness_status=readiness_status,
        selection_eligible=effective_selection_eligible,
        diagnostic_only=bool(diagnostic_only),
        selection_exclusion_reason=effective_exclusion_reason,
        candidate_params=dict(candidate_params or {}),
        family_transform_id=family_transform_id,
        feature_diagnostics=dict(feature_diagnostics or {}),
        failure_mode=None if health_obj.passed else (health.get("failure_type") or "output_health_gate_failed"),
        warning_reasons=tuple(warning_reasons),
    )


def _candidate_status(*, health_passed: bool, selection_eligible: bool, diagnostic_only: bool) -> str:
    return candidate_selection_status(
        health_passed=health_passed,
        selection_eligible=selection_eligible,
        diagnostic_only=diagnostic_only,
    )


def _selection_guard_reason(*, family: str, profile_type: str, feature_diagnostics: Mapping[str, Any]) -> str | None:
    if family == "relationship_concentration_entropy" and feature_diagnostics.get("low_spread_warning"):
        return FILTER_LOW_FEATURE_SPREAD
    return None


def _numeric_frame(frame: Any, feature_columns: Sequence[str]) -> Any:
    pd = _pandas()
    columns = tuple(str(column) for column in feature_columns)
    present = {str(column) for column in getattr(frame, "columns", ())}
    missing = tuple(column for column in columns if column not in present)
    if missing:
        raise MissingScoringColumnError(missing, available_columns=tuple(sorted(present)))
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()


def _family_score(data: Any, *, family: str) -> Any:
    pd = _pandas()
    if data.empty:
        return pd.Series([], dtype=float)
    if family == "anchor_core_exposure":
        primary = _scaled_abs(data, "corr_to_anchor_primary")
        secondary = _scaled_abs(data, "corr_to_anchor_secondary")
        core = _scaled_abs(data, "corr_to_core_basket")
        beta = _rank_scaled(_abs_series(data, "beta_to_core_basket"))
        return 0.20 * primary + 0.25 * secondary + 0.30 * core + 0.25 * beta
    if family == "peer_strength_stability":
        if {"strongest_peer_slot_1_strength", "top_peer_stability_mean", "top_peer_count"}.issubset(set(data.columns)):
            strength = _rank_scaled(_numeric_series(data, "strongest_peer_slot_1_strength"))
            stability = _rank_scaled(_numeric_series(data, "top_peer_stability_mean"))
            count = _rank_scaled(_numeric_series(data, "top_peer_count").apply(lambda value: math.log1p(max(0.0, float(value)))))
            joint = _rank_scaled(_numeric_series(data, "strongest_peer_slot_1_strength").combine(_numeric_series(data, "top_peer_stability_mean"), min))
            return 0.30 * strength + 0.30 * stability + 0.20 * count + 0.20 * joint
        return _semantic_axis_mean(data, family=family)
    if family == "relationship_concentration_entropy":
        if {"relationship_concentration", "relationship_entropy"}.issubset(set(data.columns)):
            concentration = _rank_scaled(_numeric_series(data, "relationship_concentration"))
            inverse_entropy = 1.0 - _rank_scaled(_numeric_series(data, "relationship_entropy"))
            spread = _rank_scaled(_numeric_series(data, "relationship_concentration") - _numeric_series(data, "relationship_entropy"))
            return 0.45 * concentration + 0.35 * inverse_entropy + 0.20 * spread
        return _semantic_axis_mean(data, family=family)
    return data.iloc[:, 0]


def _rule_label_variants(data: Any, *, family: str) -> tuple[tuple[str, dict[str, Any], list[str]], ...]:
    if family == "anchor_core_exposure":
        primary = _scaled_abs(data, "corr_to_anchor_primary")
        secondary = _scaled_abs(data, "corr_to_anchor_secondary")
        core = _scaled_abs(data, "corr_to_core_basket")
        beta = _rank_scaled(_abs_series(data, "beta_to_core_basket"))
        coupled = (0.30 * primary + 0.20 * secondary + 0.30 * core + 0.20 * beta)
        variants = [
            ("balanced_coupling", {"low_max": 0.35, "high_min": 0.75}, _rule_labels_from_score(coupled, low="low_coupling", mid="moderate_coupling", high="high_coupling", low_max=0.35, high_min=0.75)),
            ("beta_core_joint", {"low_max": 0.33, "high_min": 0.66}, _joint_rule_labels(beta, core, low="low_beta_core", mid="mixed_beta_core", high="high_beta_core", low_max=0.33, high_min=0.66)),
            ("core_secondary_joint", {"low_max": 0.33, "high_min": 0.66}, _joint_rule_labels(core, secondary, low="decoupled_core", mid="mixed_core_exposure", high="coupled_core", low_max=0.33, high_min=0.66)),
        ]
    elif family == "peer_strength_stability":
        axes = _semantic_axes(data, family=family)
        strength = axes.iloc[:, 0] if not axes.empty else _family_score(data, family=family)
        stability = axes.iloc[:, 1] if len(axes.columns) >= 2 else strength
        count = axes.iloc[:, 2] if len(axes.columns) >= 3 else _family_score(data, family=family)
        joint = _family_score(data, family=family)
        variants = [
            ("strength_stability_joint", {"low_max": 0.33, "high_min": 0.66}, _joint_rule_labels(strength, stability, low="weak_unstable_peers", mid="mixed_peer_context", high="strong_stable_peers", low_max=0.33, high_min=0.66)),
            ("count_strength_joint", {"low_max": 0.33, "high_min": 0.66}, _joint_rule_labels(count, strength, low="thin_peer_support", mid="moderate_peer_support", high="broad_strong_peer_support", low_max=0.33, high_min=0.66)),
            ("ranked_peer_context", {"low_max": 0.33, "high_min": 0.66}, _rule_labels_from_score(joint, low="weak_peer_context", mid="moderate_peer_context", high="strong_peer_context", low_max=0.33, high_min=0.66)),
        ]
    elif family == "relationship_concentration_entropy":
        axes = _semantic_axes(data, family=family)
        concentration = axes.iloc[:, 0] if not axes.empty else _family_score(data, family=family)
        inverse_entropy = axes.iloc[:, 1] if len(axes.columns) >= 2 else concentration
        spread_score = _family_score(data, family=family)
        guarded = _low_spread(data, family=family)
        guarded_labels = [f"{family}_low_spread_single"] * len(data) if guarded else _rule_labels_from_score(spread_score, low="diffuse_dependency", mid="mixed_dependency", high="concentrated_dependency", low_max=0.33, high_min=0.66)
        variants = [
            ("concentration_entropy_joint", {"low_max": 0.33, "high_min": 0.66}, _joint_rule_labels(concentration, inverse_entropy, low="diffuse_high_entropy", mid="mixed_dependency", high="concentrated_low_entropy", low_max=0.33, high_min=0.66)),
            ("spread_guarded_rank", {"low_max": 0.33, "high_min": 0.66, "spread_guarded": guarded}, guarded_labels),
            ("concentration_only", {"low_max": 0.33, "high_min": 0.66}, _rule_labels_from_score(concentration, low="low_concentration", mid="medium_concentration", high="high_concentration", low_max=0.33, high_min=0.66)),
        ]
    elif family == "residual_peer_signal":
        values = _numeric_series(data, "residual_peer_signal_score") if "residual_peer_signal_score" in data.columns else _family_score(data, family=family)
        variants = [
            ("signed_standard", {"negative_max": -0.35, "positive_min": 0.35}, _signed_residual_labels(values, negative_max=-0.35, positive_min=0.35)),
            ("signed_narrow", {"negative_max": -0.20, "positive_min": 0.20}, _signed_residual_labels(values, negative_max=-0.20, positive_min=0.20)),
            ("signed_magnitude", {"neutral_abs_max": 0.20, "large_abs_min": 0.60}, _residual_magnitude_labels(values, neutral_abs_max=0.20, large_abs_min=0.60)),
        ]
    else:
        score = _family_score(data, family=family)
        variants = [("default_family_score", {"low_max": 0.35, "high_min": 0.70}, _rule_labels_from_score(score, low="low_state", mid="middle_state", high="high_state", low_max=0.35, high_min=0.70))]
    return tuple(
        (
            variant_id,
            {"variant_id": variant_id, "family_transform_id": f"{family}_rule_grid_v2", **params},
            labels,
        )
        for variant_id, params, labels in variants
    )


def _rule_labels_from_definition(
    data: Any,
    *,
    family: str,
    definition: Mapping[str, Any],
) -> tuple[str, dict[str, Any], list[str]]:
    variant = str(definition.get("variant_id") or "")
    for variant_id, params, labels in _rule_label_variants(data, family=family):
        if str(variant_id) == variant:
            return variant_id, {**dict(definition.get("candidate_params") or {}), **dict(params)}, labels
    score = _family_score(data, family=family)
    labels = _rule_labels_from_score(score, low="low_state", mid="middle_state", high="high_state", low_max=0.35, high_min=0.70)
    return variant or "default_family_score", dict(definition.get("candidate_params") or {}), labels


def _rule_labels_from_score(score: Any, *, low: str, mid: str, high: str, low_max: float, high_min: float) -> list[str]:
    labels: list[str] = []
    for value in score:
        val = float(value)
        if val < low_max:
            labels.append(low)
        elif val >= high_min:
            labels.append(high)
        else:
            labels.append(mid)
    return labels


def _joint_rule_labels(left: Any, right: Any, *, low: str, mid: str, high: str, low_max: float, high_min: float) -> list[str]:
    labels: list[str] = []
    for left_value, right_value in zip(left, right):
        l_val = float(left_value)
        r_val = float(right_value)
        if l_val >= high_min and r_val >= high_min:
            labels.append(high)
        elif l_val < low_max or r_val < low_max:
            labels.append(low)
        else:
            labels.append(mid)
    return labels


def _signed_residual_labels(values: Any, *, negative_max: float, positive_min: float) -> list[str]:
    labels = []
    for value in values:
        val = float(value)
        if val <= negative_max:
            labels.append("negative_residual_peer_signal")
        elif val >= positive_min:
            labels.append("positive_residual_peer_signal")
        else:
            labels.append("neutral_residual_peer_signal")
    return labels


def _residual_magnitude_labels(values: Any, *, neutral_abs_max: float, large_abs_min: float) -> list[str]:
    labels = []
    for value in values:
        val = float(value)
        mag = abs(val)
        if mag <= neutral_abs_max:
            labels.append("neutral_low_magnitude_residual")
        elif mag >= large_abs_min and val < 0:
            labels.append("large_negative_residual")
        elif mag >= large_abs_min and val > 0:
            labels.append("large_positive_residual")
        elif val < 0:
            labels.append("moderate_negative_residual")
        else:
            labels.append("moderate_positive_residual")
    return labels


def _rule_labels(data: Any, *, family: str) -> list[str]:
    if family == "residual_peer_signal":
        values = _numeric_series(data, "residual_peer_signal_score")
        labels = []
        for value in values:
            val = float(value)
            if val <= -0.35:
                labels.append("negative_residual_peer_signal")
            elif val >= 0.35:
                labels.append("positive_residual_peer_signal")
            else:
                labels.append("neutral_residual_peer_signal")
        return labels
    score = _family_score(data, family=family)
    labels = []
    for value in score:
        val = float(value)
        if family == "anchor_core_exposure":
            labels.append("high_coupling" if val >= 0.75 else ("low_coupling" if val < 0.35 else "moderate_coupling"))
        elif family == "peer_strength_stability":
            labels.append("strong_stable_peers" if val >= 0.75 else ("weak_peer_context" if val < 0.35 else "moderate_peer_context"))
        elif family == "relationship_concentration_entropy":
            labels.append("concentrated" if val >= 0.70 else ("diffuse" if val < 0.35 else "mixed_dependency"))
        else:
            labels.append("positive_residual_peer_signal" if val >= 0.35 else ("negative_residual_peer_signal" if val <= -0.35 else "neutral_residual_peer_signal"))
    return labels


def _quantile_labels(score: Any, *, bins: int = 3, prefix: str = "ordinal") -> list[str]:
    pd = _pandas()
    if len(score) < 3 or float(score.max()) == float(score.min()):
        return [f"{prefix}_single"] * len(score)
    ranks = pd.Series(score).rank(method="first", pct=True)
    labels = []
    for value in ranks:
        bucket = min(int(math.ceil(float(value) * int(bins))), int(bins))
        if bins == 2:
            labels.append(f"{prefix}_{'low' if bucket == 1 else 'high'}")
        elif bins == 3:
            labels.append(f"{prefix}_{('low', 'mid', 'high')[bucket - 1]}")
        else:
            labels.append(f"{prefix}_bin{bucket}")
    return labels


def _semantic_separation(data: Any, labels: list[str], *, family: str) -> float:
    if not labels or len(set(labels)) < 2:
        return 0.0
    if family == "relationship_concentration_entropy" and _low_spread(data, family=family):
        return 0.0
    pd = _pandas()
    axes = _semantic_axes(data, family=family)
    if axes.empty:
        return 0.0
    scores: list[float] = []
    for column in axes.columns:
        grouped = pd.DataFrame({"score": axes[column].to_numpy(), "label": labels}).groupby("label")["score"].mean()
        if len(grouped) < 2:
            continue
        scores.append(float(grouped.max() - grouped.min()))
    if not scores:
        return 0.0
    return bounded_score(sum(scores) / len(scores))


def _semantic_axes(data: Any, *, family: str) -> Any:
    pd = _pandas()
    if family == "anchor_core_exposure":
        source = pd.DataFrame(
            {
                "anchor_primary_abs": _abs_series(data, "corr_to_anchor_primary"),
                "anchor_secondary_abs": _abs_series(data, "corr_to_anchor_secondary"),
                "core_basket_abs": _abs_series(data, "corr_to_core_basket"),
                "beta_to_core_rank": _numeric_series(data, "beta_to_core_basket"),
            }
        )
    elif family == "peer_strength_stability":
        source = _feature_set_aware_source(
            data,
            groups=(
                ("peer_strength", PEER_STRENGTH_COLUMNS),
                ("peer_support", PEER_SUPPORT_COLUMNS),
                ("peer_stability", PEER_STABILITY_COLUMNS),
            ),
            family=family,
        )
    elif family == "relationship_concentration_entropy" and {"relationship_concentration", "relationship_entropy"}.issubset(set(data.columns)):
        source = pd.DataFrame(
            {
                "relationship_concentration": data["relationship_concentration"],
                "inverse_relationship_entropy": -data["relationship_entropy"],
                "concentration_minus_entropy": data["relationship_concentration"] - data["relationship_entropy"],
            }
        )
    elif family == "relationship_concentration_entropy":
        source = _feature_set_aware_source(
            data,
            groups=(
                ("relationship_concentration", RELATIONSHIP_CONCENTRATION_COLUMNS),
                ("inverse_relationship_entropy", RELATIONSHIP_ENTROPY_COLUMNS),
                ("relationship_weight_spread", RELATIONSHIP_SPREAD_COLUMNS),
            ),
            family=family,
            invert_groups=("inverse_relationship_entropy",),
        )
    elif family == "residual_peer_signal" and "residual_peer_signal_score" in data.columns:
        source = pd.DataFrame(
            {
                "signed_residual_peer_signal": data["residual_peer_signal_score"],
                "absolute_residual_peer_signal": data["residual_peer_signal_score"].abs(),
            }
        )
    else:
        source = data.copy()
    axes = pd.DataFrame(index=source.index)
    for column in source.columns:
        series = pd.to_numeric(source[column], errors="coerce")
        span = float(series.max() - series.min()) if len(series) else 0.0
        if not span:
            axes[column] = 0.0
        else:
            axes[column] = (series - float(series.min())) / span
    return axes


def _semantic_axis_mean(data: Any, *, family: str) -> Any:
    pd = _pandas()
    axes = _semantic_axes(data, family=family)
    if axes.empty:
        return pd.Series([0.0] * len(data), index=data.index, dtype=float)
    return axes.mean(axis=1)


def _feature_set_aware_source(
    data: Any,
    *,
    groups: Sequence[tuple[str, Sequence[str]]],
    family: str,
    invert_groups: Sequence[str] = (),
) -> Any:
    pd = _pandas()
    out = pd.DataFrame(index=data.index)
    for group_name, columns in groups:
        available = _available_columns(data, columns)
        if not available:
            continue
        out[group_name] = _composite_feature_axis(data, available, invert=group_name in set(invert_groups))
    if out.empty and len(getattr(data, "columns", ())) > 0:
        for column in data.columns:
            out[str(column)] = _numeric_series(data, str(column))
    if out.empty:
        raise MissingScoringColumnError(tuple(column for _, columns in groups for column in columns), family=family)
    return out


def _available_columns(data: Any, candidates: Sequence[str]) -> tuple[str, ...]:
    present = {str(column) for column in getattr(data, "columns", ())}
    return tuple(str(column) for column in candidates if str(column) in present)


def _composite_feature_axis(data: Any, columns: Sequence[str], *, invert: bool = False) -> Any:
    pd = _pandas()
    scaled = []
    for column in columns:
        values = _rank_scaled(_numeric_series(data, str(column)))
        if invert:
            values = 1.0 - values
        scaled.append(values.rename(str(column)))
    if not scaled:
        return pd.Series([0.0] * len(data), index=data.index, dtype=float)
    return pd.concat(scaled, axis=1).mean(axis=1)


def _cluster_frame(data: Any, *, family: str) -> tuple[Any, str]:
    axes = _semantic_axes(data, family=family)
    if axes.empty:
        return data, f"{family}_raw_numeric_v1"
    return axes.fillna(0.0), f"{family}_semantic_axis_scaled_v1"


def _cluster_count_grid(row_count: int) -> tuple[int, ...]:
    if row_count < 8:
        return (2,)
    if row_count < 16:
        return (2, 3)
    return (2, 3, 4)


def _gmm_grid(row_count: int) -> tuple[tuple[int, str, float], ...]:
    counts = (2, 3) if row_count < 24 else (2, 3, 4)
    grid: list[tuple[int, str, float]] = []
    for count in counts:
        grid.append((count, "full", 1e-6))
        grid.append((count, "diag", 1e-4))
    return tuple(grid)


def _birch_grid(row_count: int) -> tuple[tuple[int, float, int], ...]:
    counts = (2, 3) if row_count >= 12 else (2,)
    grid: list[tuple[int, float, int]] = []
    for count in counts:
        grid.append((count, 0.35, 25))
        grid.append((count, 0.70, 50))
    return tuple(grid[:4])


def _agglomerative_grid(row_count: int) -> tuple[tuple[int, str, str], ...]:
    if row_count < 12:
        return ((2, "ward", "euclidean"), (2, "average", "euclidean"))
    if row_count < 24:
        return ((2, "ward", "euclidean"), (3, "ward", "euclidean"), (3, "average", "euclidean"))
    return (
        (2, "ward", "euclidean"),
        (3, "ward", "euclidean"),
    )


def _feature_diagnostics(data: Any, *, family: str) -> dict[str, Any]:
    raw_low_spread_columns = (
        tuple(str(column) for column in data.columns)
        if family in {"peer_strength_stability", "relationship_concentration_entropy"}
        else ()
    )
    saturated_columns = ("corr_to_anchor_primary",) if family == "anchor_core_exposure" else ()
    diagnostics = feature_spread_diagnostics(
        data,
        score_values=_family_score(data, family=family),
        family_transform_id=f"{family}_diagnostic_spread_v1",
        raw_low_spread_columns=raw_low_spread_columns,
        raw_span_epsilon=0.005,
        raw_iqr_epsilon=0.001,
        saturated_columns=saturated_columns,
        saturation_abs_median_min=0.85,
        saturation_iqr_max=0.05,
    ).as_dict()
    saturated_primary = bool(diagnostics.get("saturated_corr_to_anchor_primary_warning"))
    diagnostics["saturated_primary_correlation_warning"] = saturated_primary
    diagnostics["scoring_columns"] = tuple(str(column) for column in data.columns)
    diagnostics["low_spread_gate_columns"] = tuple(str(column) for column in raw_low_spread_columns)
    diagnostics["old_fixed_top3_columns_used"] = tuple(
        column for column in OLD_FIXED_TOP3_COLUMNS if column in {str(col) for col in data.columns}
    )
    warnings_out = list(diagnostics.get("warning_reasons") or ())
    if saturated_primary:
        warnings_out = [
            "saturated_primary_correlation" if reason == "saturated_corr_to_anchor_primary" else reason
            for reason in warnings_out
        ]
        if "saturated_primary_correlation" not in warnings_out:
            warnings_out.append("saturated_primary_correlation")
    diagnostics["warning_reasons"] = tuple(dict.fromkeys(warnings_out))
    return diagnostics


def _low_spread(data: Any, *, family: str) -> bool:
    diagnostics = _feature_diagnostics(data, family=family)
    return bool(diagnostics.get("low_spread_warning"))


def _numeric_series(data: Any, column: str) -> Any:
    pd = _pandas()
    if column not in data.columns:
        raise MissingScoringColumnError((column,), available_columns=tuple(str(col) for col in getattr(data, "columns", ())))
    return pd.to_numeric(data[column], errors="coerce").fillna(0.0).astype(float)


def _abs_series(data: Any, column: str) -> Any:
    return _numeric_series(data, column).abs()


def _scaled_abs(data: Any, column: str) -> Any:
    return _rank_scaled(_abs_series(data, column))


def _rank_scaled(series: Any) -> Any:
    pd = _pandas()
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if clean.empty:
        return clean
    if float(clean.max()) == float(clean.min()):
        return pd.Series([0.5] * len(clean), index=clean.index, dtype=float)
    return clean.rank(method="average", pct=True).astype(float)


def _warning_reasons(caught: Sequence[Any], *, profile_type: str) -> tuple[str, ...]:
    reasons: list[str] = []
    profile = str(profile_type)
    for warning in caught:
        text = str(getattr(warning, "message", ""))
        if profile == PROFILE_BIRCH and "subclusters found" in text:
            reasons.append(HEALTH_WARNING_BIRCH_TOO_FEW_SUBCLUSTERS)
        if "gaussian_mixture" in profile and text:
            reasons.append("gaussian_mixture_warning")
        if profile in {PROFILE_HDBSCAN, PROFILE_OPTICS, PROFILE_AGGLOMERATIVE} and text:
            reasons.append(f"{profile}_warning")
    return tuple(dict.fromkeys(reasons))


def _persistence_score(labels: list[str]) -> float:
    if len(labels) <= 1:
        return 0.0
    same = sum(1 for left, right in zip(labels, labels[1:]) if left == right)
    return bounded_score(same / (len(labels) - 1))


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Cross-Asset-State profile diagnostics require pandas") from exc
    return pd


__all__ = [
    "PROFILE_DIAGNOSTIC_ONLY",
    "PROFILE_AGGLOMERATIVE",
    "PROFILE_BAYESIAN_GAUSSIAN_MIXTURE",
    "PROFILE_BIRCH",
    "PROFILE_FACTOR_ANALYSIS_GAUSSIAN_MIXTURE",
    "PROFILE_FACTOR_ANALYSIS_KMEANS",
    "PROFILE_GAUSSIAN_MIXTURE",
    "PROFILE_HDBSCAN",
    "PROFILE_KMEANS",
    "PROFILE_MINIBATCH_KMEANS",
    "PROFILE_ORDINAL",
    "PROFILE_OPTICS",
    "PROFILE_PCA_KMEANS",
    "PROFILE_RULE",
    "CROSS_ASSET_STATE_ADAPTER_METHODS",
    "CROSS_ASSET_STATE_SPLIT_POLICY_ID",
    "CrossAssetStateCandidateDefinitionBatch",
    "CrossAssetStateCandidateDefinitionCache",
    "CrossAssetStateProfileCandidateResult",
    "candidate_definition_shape",
    "choose_best_candidate",
    "diagnostic_only_result",
    "run_profile_candidates",
]
