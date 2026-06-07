from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.core.clusterer_adapters import clusterer_adapter_registry
from src.regimes.core.clusterer_registry import clusterer_capabilities_registry
from src.regimes.core.serialization import to_jsonable


REGIME_CLUSTERING_CANDIDATE_SCHEMA_VERSION = 1

CANDIDATE_STATUS_PRODUCTION_CANDIDATE = "production_candidate"
CANDIDATE_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
CANDIDATE_STATUS_EXCLUDED = "excluded"

REGIME_PRODUCTION_CANDIDATE_METHODS: tuple[str, ...] = (
    "kmeans",
    "minibatch_kmeans",
    "pca_kmeans",
    "factor_analysis_kmeans",
    "gaussian_mixture",
    "factor_analysis_gaussian_mixture",
    "bayesian_gaussian_mixture",
    "birch",
)

REGIME_DIAGNOSTIC_CONDITIONAL_METHODS: tuple[str, ...] = (
    "hdbscan",
    "optics",
    "agglomerative",
)

REGIME_EXCLUDED_METHODS: tuple[str, ...] = ("spectral_clustering",)

SHARED_REGIME_CLUSTERING_METHODS: tuple[str, ...] = (
    *REGIME_PRODUCTION_CANDIDATE_METHODS,
    *REGIME_DIAGNOSTIC_CONDITIONAL_METHODS,
)


@dataclass(frozen=True)
class RegimeClusteringCandidateSpec:
    method_family: str
    clusterer_family: str
    embedding: str
    default_status: str
    adapter_family: str
    assignment_policy: str | None
    inductive_classification: str | None
    dependency_name: str | None
    dependency_available: bool
    supports_fit: bool
    supports_assign: bool
    supports_refit_or_recluster: bool
    supports_noise_labels: bool
    supports_probabilities: bool
    supports_soft_membership: bool
    production_candidate_default: bool
    diagnostic_only_default: bool
    excluded_default: bool
    reason: str
    schema_version: int = REGIME_CLUSTERING_CANDIDATE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "method_family": self.method_family,
            "clusterer_family": self.clusterer_family,
            "embedding": self.embedding,
            "default_status": self.default_status,
            "adapter_family": self.adapter_family,
            "assignment_policy": self.assignment_policy,
            "inductive_classification": self.inductive_classification,
            "dependency_name": self.dependency_name,
            "dependency_available": bool(self.dependency_available),
            "supports_fit": bool(self.supports_fit),
            "supports_assign": bool(self.supports_assign),
            "supports_refit_or_recluster": bool(self.supports_refit_or_recluster),
            "supports_noise_labels": bool(self.supports_noise_labels),
            "supports_probabilities": bool(self.supports_probabilities),
            "supports_soft_membership": bool(self.supports_soft_membership),
            "production_candidate_default": bool(self.production_candidate_default),
            "diagnostic_only_default": bool(self.diagnostic_only_default),
            "excluded_default": bool(self.excluded_default),
            "reason": self.reason,
        }


def clusterer_and_embedding_for_method(method_family: str) -> tuple[str, str]:
    method = _normalize_method(method_family)
    if method == "pca_kmeans":
        return "kmeans", "pca"
    if method == "factor_analysis_kmeans":
        return "kmeans", "factor_analysis"
    if method == "factor_analysis_gaussian_mixture":
        return "gaussian_mixture", "factor_analysis"
    if method in {
        "kmeans",
        "minibatch_kmeans",
        "gaussian_mixture",
        "bayesian_gaussian_mixture",
        "birch",
        "hdbscan",
        "optics",
        "agglomerative",
        "spectral_clustering",
    }:
        return method, "none"
    raise ValueError(f"Unsupported Regime clustering method family {method_family!r}")


def default_regime_clustering_candidate_registry(
    *,
    include_excluded: bool = True,
) -> dict[str, RegimeClusteringCandidateSpec]:
    methods = list(SHARED_REGIME_CLUSTERING_METHODS)
    if include_excluded:
        methods.extend(REGIME_EXCLUDED_METHODS)
    return {method: _build_candidate_spec(method) for method in methods}


def regime_clustering_candidate_specs(
    methods: Sequence[str] | None = None,
    *,
    include_excluded: bool = False,
) -> tuple[RegimeClusteringCandidateSpec, ...]:
    registry = default_regime_clustering_candidate_registry(include_excluded=include_excluded)
    requested = tuple(methods or registry)
    return tuple(candidate_spec_for_method(method, registry=registry) for method in requested)


def candidate_spec_for_method(
    method_family: str,
    *,
    registry: Mapping[str, RegimeClusteringCandidateSpec] | None = None,
) -> RegimeClusteringCandidateSpec:
    method = _normalize_method(method_family)
    candidates = registry or default_regime_clustering_candidate_registry(include_excluded=True)
    try:
        return candidates[method]
    except KeyError as exc:
        valid = ", ".join(sorted(candidates))
        raise ValueError(f"Unsupported Regime clustering method family {method!r}; expected one of: {valid}") from exc


def pathway_candidate_policy_rows(
    *,
    pathway: str,
    production_candidate_methods: Sequence[str],
    diagnostic_methods: Sequence[str] = (),
    excluded_methods: Sequence[str] = REGIME_EXCLUDED_METHODS,
    methods: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    prod = {_normalize_method(item) for item in production_candidate_methods}
    diagnostic = {_normalize_method(item) for item in diagnostic_methods}
    excluded = {_normalize_method(item) for item in excluded_methods}
    selected_methods = tuple(methods or (*prod, *diagnostic, *excluded))
    rows: list[dict[str, Any]] = []
    for method in selected_methods:
        spec = candidate_spec_for_method(method)
        if spec.method_family in excluded:
            status = CANDIDATE_STATUS_EXCLUDED
        elif spec.method_family in diagnostic:
            status = CANDIDATE_STATUS_DIAGNOSTIC_ONLY
        elif spec.method_family in prod:
            status = CANDIDATE_STATUS_PRODUCTION_CANDIDATE
        else:
            status = spec.default_status
        row = spec.as_dict()
        row.update(
            {
                "pathway": str(pathway),
                "pathway_status": status,
                "pathway_production_candidate": status == CANDIDATE_STATUS_PRODUCTION_CANDIDATE,
                "pathway_diagnostic_only": status == CANDIDATE_STATUS_DIAGNOSTIC_ONLY,
                "pathway_excluded": status == CANDIDATE_STATUS_EXCLUDED,
            }
        )
        rows.append(to_jsonable(row))
    return tuple(rows)


def _build_candidate_spec(method_family: str) -> RegimeClusteringCandidateSpec:
    method = _normalize_method(method_family)
    clusterer_family, embedding = clusterer_and_embedding_for_method(method)
    legacy = clusterer_adapter_registry(include_placeholders=True).get(clusterer_family)
    caps = clusterer_capabilities_registry().get(clusterer_family)
    production_default = method in REGIME_PRODUCTION_CANDIDATE_METHODS
    diagnostic_default = method in REGIME_DIAGNOSTIC_CONDITIONAL_METHODS
    excluded_default = method in REGIME_EXCLUDED_METHODS
    if excluded_default:
        status = CANDIDATE_STATUS_EXCLUDED
    elif diagnostic_default:
        status = CANDIDATE_STATUS_DIAGNOSTIC_ONLY
    else:
        status = CANDIDATE_STATUS_PRODUCTION_CANDIDATE
    return RegimeClusteringCandidateSpec(
        method_family=method,
        clusterer_family=clusterer_family,
        embedding=embedding,
        default_status=status,
        adapter_family=clusterer_family,
        assignment_policy=None if legacy is None else legacy.assignment_policy,
        inductive_classification=None if legacy is None else legacy.inductive_classification,
        dependency_name=None if legacy is None else legacy.dependency_name,
        dependency_available=False if legacy is None else bool(legacy.dependency_available),
        supports_fit=False if caps is None else bool(caps.supports_fit),
        supports_assign=False if caps is None else bool(caps.supports_assign),
        supports_refit_or_recluster=False if caps is None else bool(caps.supports_refit_or_recluster),
        supports_noise_labels=False if caps is None else bool(caps.supports_noise_labels),
        supports_probabilities=False if legacy is None else bool(legacy.supports_probabilities),
        supports_soft_membership=False if legacy is None else bool(legacy.supports_soft_membership),
        production_candidate_default=production_default,
        diagnostic_only_default=diagnostic_default,
        excluded_default=excluded_default,
        reason=_default_reason(method),
    )


def _default_reason(method_family: str) -> str:
    if method_family == "spectral_clustering":
        return "excluded from production-candidate tuning; diagnostic graph method unless a pathway explicitly proves otherwise"
    if method_family in REGIME_DIAGNOSTIC_CONDITIONAL_METHODS:
        return "causal assignment adapter is available, but pathway policy must prove suitability before production-candidate promotion"
    return "shared production-candidate method metadata; pathway policy owns enablement and search space"


def _normalize_method(method_family: str) -> str:
    method = str(method_family).strip().lower()
    if not method:
        raise ValueError("Regime clustering method family must be non-empty")
    return method


__all__ = [
    "CANDIDATE_STATUS_DIAGNOSTIC_ONLY",
    "CANDIDATE_STATUS_EXCLUDED",
    "CANDIDATE_STATUS_PRODUCTION_CANDIDATE",
    "REGIME_CLUSTERING_CANDIDATE_SCHEMA_VERSION",
    "REGIME_DIAGNOSTIC_CONDITIONAL_METHODS",
    "REGIME_EXCLUDED_METHODS",
    "REGIME_PRODUCTION_CANDIDATE_METHODS",
    "SHARED_REGIME_CLUSTERING_METHODS",
    "RegimeClusteringCandidateSpec",
    "candidate_spec_for_method",
    "clusterer_and_embedding_for_method",
    "default_regime_clustering_candidate_registry",
    "pathway_candidate_policy_rows",
    "regime_clustering_candidate_specs",
]
