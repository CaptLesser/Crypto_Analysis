from __future__ import annotations

from typing import Any

from src.regimes.core.clustering_candidates import (
    CANDIDATE_STATUS_DIAGNOSTIC_ONLY,
    CANDIDATE_STATUS_EXCLUDED,
    CANDIDATE_STATUS_PRODUCTION_CANDIDATE,
    pathway_candidate_policy_rows,
)


MARKET_STATE_PRODUCTION_CANDIDATE_METHODS: tuple[str, ...] = (
    "kmeans",
    "minibatch_kmeans",
    "gaussian_mixture",
    "pca_kmeans",
    "factor_analysis_kmeans",
    "factor_analysis_gaussian_mixture",
    "bayesian_gaussian_mixture",
    "birch",
)

MARKET_STATE_DIAGNOSTIC_CONDITIONAL_METHODS: tuple[str, ...] = (
    "hdbscan",
    "optics",
    "agglomerative",
)

MARKET_STATE_EXCLUDED_METHODS: tuple[str, ...] = ("spectral_clustering",)


def market_state_clustering_candidate_policy() -> tuple[dict[str, Any], ...]:
    return pathway_candidate_policy_rows(
        pathway="market_state",
        production_candidate_methods=MARKET_STATE_PRODUCTION_CANDIDATE_METHODS,
        diagnostic_methods=MARKET_STATE_DIAGNOSTIC_CONDITIONAL_METHODS,
        excluded_methods=MARKET_STATE_EXCLUDED_METHODS,
    )


def market_state_candidate_methods_by_status() -> dict[str, tuple[str, ...]]:
    rows = market_state_clustering_candidate_policy()
    statuses = {
        CANDIDATE_STATUS_PRODUCTION_CANDIDATE: [],
        CANDIDATE_STATUS_DIAGNOSTIC_ONLY: [],
        CANDIDATE_STATUS_EXCLUDED: [],
    }
    for row in rows:
        status = str(row["pathway_status"])
        statuses.setdefault(status, []).append(str(row["method_family"]))
    return {status: tuple(methods) for status, methods in statuses.items()}


__all__ = [
    "MARKET_STATE_DIAGNOSTIC_CONDITIONAL_METHODS",
    "MARKET_STATE_EXCLUDED_METHODS",
    "MARKET_STATE_PRODUCTION_CANDIDATE_METHODS",
    "market_state_candidate_methods_by_status",
    "market_state_clustering_candidate_policy",
]
