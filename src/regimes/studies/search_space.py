from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_json_mapping, require_non_empty_string, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.studies.manifest import StudyManifest


REGIME_SEARCH_SPACE_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION


def default_clusterer_hyperparameters(clusterer_family: str, *, random_seed: int = 17) -> dict[str, Any]:
    family = require_non_empty_string(clusterer_family, field_name="clusterer family").lower()
    if family in {"kmeans", "minibatch_kmeans"}:
        params: dict[str, Any] = {"n_clusters": 2, "n_init": 10, "random_state": int(random_seed)}
        if family == "minibatch_kmeans":
            params["batch_size"] = 8
        return params
    if family in {"gaussian_mixture", "bayesian_gaussian_mixture"}:
        return {"n_components": 2, "covariance_type": "full", "random_state": int(random_seed)}
    if family == "agglomerative":
        return {"n_clusters": 2}
    if family == "optics":
        return {"min_samples": 2}
    return {}


@dataclass(frozen=True)
class StudySearchSpace:
    study_id: str
    dimensions: Mapping[str, Sequence[str]]
    candidate_trials: Sequence[Mapping[str, Any]]
    budget: Mapping[str, Any]
    selected_single_trial: Mapping[str, Any]
    schema_version: int = REGIME_SEARCH_SPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_id = require_non_empty_string(self.study_id, field_name="search space study_id")
        dimensions = require_json_mapping(self.dimensions, field_name="search space dimensions")
        candidate_trials = tuple(require_json_object(row, context="Regime search space candidate") for row in self.candidate_trials)
        if not candidate_trials:
            raise ValueError("Regime search space must include at least one candidate trial")
        selected = require_json_object(self.selected_single_trial, context="Regime selected single trial")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "dimensions", to_jsonable(dimensions))
        object.__setattr__(self, "candidate_trials", candidate_trials)
        object.__setattr__(self, "budget", to_jsonable(require_json_mapping(self.budget, field_name="search space budget")))
        object.__setattr__(self, "selected_single_trial", selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_id": self.study_id,
            "dimensions": to_jsonable(self.dimensions),
            "candidate_count": int(len(self.candidate_trials)),
            "candidate_trials": to_jsonable(list(self.candidate_trials)),
            "budget": to_jsonable(self.budget),
            "selected_single_trial": to_jsonable(self.selected_single_trial),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudySearchSpace":
        obj = require_json_object(payload, context="Regime StudySearchSpace")
        return cls(
            schema_version=obj.get("schema_version", REGIME_SEARCH_SPACE_SCHEMA_VERSION),
            study_id=obj["study_id"],
            dimensions=obj["dimensions"],
            candidate_trials=obj["candidate_trials"],
            budget=obj["budget"],
            selected_single_trial=obj["selected_single_trial"],
        )

    @classmethod
    def from_json(cls, text: str) -> "StudySearchSpace":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime StudySearchSpace JSON"))


def build_search_space(manifest: StudyManifest | Mapping[str, Any]) -> StudySearchSpace:
    study = manifest if isinstance(manifest, StudyManifest) else StudyManifest.from_dict(manifest)
    random_seed = int(study.budget.get("random_seed", 17))
    candidates: list[dict[str, Any]] = []
    for idx, (feature_family, preprocess, clusterer) in enumerate(
        product(study.feature_families, study.preprocessing_options, study.candidate_clusterer_families),
        start=1,
    ):
        candidates.append(
            {
                "trial_index": idx,
                "trial_id": f"{study.study_id}_trial_{idx:03d}",
                "feature_family": feature_family,
                "preprocessing": preprocess,
                "clusterer_family": clusterer,
                "clusterer_hyperparameters": default_clusterer_hyperparameters(clusterer, random_seed=random_seed),
                "execution_status": "candidate_only",
            }
        )
    return StudySearchSpace(
        study_id=study.study_id,
        dimensions={
            "feature_families": list(study.feature_families),
            "preprocessing_options": list(study.preprocessing_options),
            "candidate_clusterer_families": list(study.candidate_clusterer_families),
        },
        candidate_trials=candidates,
        budget=study.budget,
        selected_single_trial=candidates[0],
    )


__all__ = [
    "REGIME_SEARCH_SPACE_SCHEMA_VERSION",
    "StudySearchSpace",
    "build_search_space",
    "default_clusterer_hyperparameters",
]
