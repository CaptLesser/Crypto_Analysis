from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    RegimeAxis,
    RegimeBand,
    RegimeClassification,
    RegimeLayer,
    StudyKey,
    normalize_enum_value,
    normalize_string_tuple,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
    validate_layer_axis_band,
)
from src.regimes.core.paths import default_foundation_report_root
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


REGIME_STUDY_MANIFEST_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
DEFAULT_STUDY_REPORT_ROOT = default_foundation_report_root("studies")
DEFAULT_STUDY_SPLIT_POLICY: dict[str, Any] = {
    "name": "deterministic_head_tail",
    "train_fraction": 2.0 / 3.0,
}
DEFAULT_STUDY_BUDGET: dict[str, Any] = {
    "max_trials": 1,
    "timeout_seconds": 60,
    "random_seed": 17,
    "tiny_cluster_threshold": 1,
}


def _normalize_token(value: object, *, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    return require_non_empty_string(value, field_name=field_name).lower()


def _normalize_unique_tokens(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_normalize_token(value, field_name=field_name) for value in values))


def _normalize_split_policy(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    split = {**DEFAULT_STUDY_SPLIT_POLICY, **dict(payload or {})}
    name = _normalize_token(split.get("name"), field_name="split_policy.name")
    if name != "deterministic_head_tail":
        raise ValueError("Regime study split_policy.name must be deterministic_head_tail")
    has_fraction = split.get("train_fraction") is not None
    has_rows = split.get("train_rows") is not None
    if has_fraction and has_rows:
        raise ValueError("Regime study split_policy must not set both train_fraction and train_rows")
    if has_rows:
        train_rows = int(split["train_rows"])
        if train_rows <= 0:
            raise ValueError("Regime study split_policy.train_rows must be positive")
        return {"name": name, "train_rows": train_rows}
    train_fraction = float(split.get("train_fraction", DEFAULT_STUDY_SPLIT_POLICY["train_fraction"]))
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("Regime study split_policy.train_fraction must be in (0, 1)")
    return {"name": name, "train_fraction": train_fraction}


def _normalize_budget(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    budget = {**DEFAULT_STUDY_BUDGET, **dict(payload or {})}
    max_trials = int(budget.get("max_trials", 1))
    timeout_seconds = float(budget.get("timeout_seconds", 60))
    random_seed = int(budget.get("random_seed", 17))
    tiny_cluster_threshold = int(budget.get("tiny_cluster_threshold", 1))
    if max_trials <= 0:
        raise ValueError("Regime study budget.max_trials must be positive")
    if timeout_seconds <= 0.0:
        raise ValueError("Regime study budget.timeout_seconds must be positive")
    if tiny_cluster_threshold < 1:
        raise ValueError("Regime study budget.tiny_cluster_threshold must be positive")
    return {
        "max_trials": max_trials,
        "timeout_seconds": timeout_seconds,
        "random_seed": random_seed,
        "tiny_cluster_threshold": tiny_cluster_threshold,
    }


@dataclass(frozen=True)
class StudyManifest:
    study_id: str
    layer: str | RegimeLayer
    axis: str | RegimeAxis
    band: str | RegimeBand
    feature_families: Sequence[str]
    preprocessing_options: Sequence[str]
    candidate_clusterer_families: Sequence[str]
    split_policy: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_STUDY_SPLIT_POLICY))
    budget: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_STUDY_BUDGET))
    report_root: str | Path = field(default_factory=lambda: default_foundation_report_root("studies"))
    classification: str | RegimeClassification = RegimeClassification.STAGED.value
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_STUDY_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_id = require_non_empty_string(self.study_id, field_name="study_id")
        layer = normalize_enum_value(self.layer, RegimeLayer, field_name="layer")
        axis = normalize_enum_value(self.axis, RegimeAxis, field_name="axis")
        band = normalize_enum_value(self.band, RegimeBand, field_name="band")
        classification = normalize_enum_value(self.classification, RegimeClassification, field_name="classification")
        validate_layer_axis_band(layer=layer, axis=axis, band=band)
        feature_families = _normalize_unique_tokens(self.feature_families, field_name="feature_families")
        preprocessing_options = _normalize_unique_tokens(self.preprocessing_options, field_name="preprocessing_options")
        clusterers = _normalize_unique_tokens(self.candidate_clusterer_families, field_name="candidate_clusterer_families")
        if not feature_families:
            raise ValueError("Regime study feature_families must include at least one value")
        if not preprocessing_options:
            raise ValueError("Regime study preprocessing_options must include at least one value")
        if not clusterers:
            raise ValueError("Regime study candidate_clusterer_families must include at least one value")
        report_root = require_non_empty_string(self.report_root, field_name="report_root")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "feature_families", feature_families)
        object.__setattr__(self, "preprocessing_options", preprocessing_options)
        object.__setattr__(self, "candidate_clusterer_families", clusterers)
        object.__setattr__(self, "split_policy", _normalize_split_policy(self.split_policy))
        object.__setattr__(self, "budget", _normalize_budget(self.budget))
        object.__setattr__(self, "report_root", report_root)
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    @property
    def study_key(self) -> StudyKey:
        return StudyKey(
            study_id=self.study_id,
            layer=self.layer,
            axis=self.axis,
            band=self.band,
            classification=self.classification,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_id": self.study_id,
            "layer": self.layer,
            "axis": self.axis,
            "band": self.band,
            "classification": self.classification,
            "feature_families": list(self.feature_families),
            "preprocessing_options": list(self.preprocessing_options),
            "candidate_clusterer_families": list(self.candidate_clusterer_families),
            "split_policy": to_jsonable(self.split_policy),
            "budget": to_jsonable(self.budget),
            "report_root": str(self.report_root),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudyManifest":
        obj = require_json_object(payload, context="Regime StudyManifest")
        return cls(
            schema_version=obj.get("schema_version", REGIME_STUDY_MANIFEST_SCHEMA_VERSION),
            study_id=obj["study_id"],
            layer=obj["layer"],
            axis=obj["axis"],
            band=obj["band"],
            classification=obj.get("classification", RegimeClassification.STAGED.value),
            feature_families=obj["feature_families"],
            preprocessing_options=obj["preprocessing_options"],
            candidate_clusterer_families=obj["candidate_clusterer_families"],
            split_policy=obj.get("split_policy", DEFAULT_STUDY_SPLIT_POLICY),
            budget=obj.get("budget", DEFAULT_STUDY_BUDGET),
            report_root=obj.get("report_root", DEFAULT_STUDY_REPORT_ROOT),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "StudyManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime StudyManifest JSON"))


def default_asset_trend_manifest(*, report_root: str | Path = DEFAULT_STUDY_REPORT_ROOT) -> StudyManifest:
    return StudyManifest(
        study_id="foundation_asset_trend_micro_single_trial",
        layer=RegimeLayer.ASSET_STATE,
        axis=RegimeAxis.TREND,
        band=RegimeBand.MICRO,
        classification=RegimeClassification.STAGED,
        feature_families=("asset_state_trend_metadata_only",),
        preprocessing_options=("robust_scale",),
        candidate_clusterer_families=("kmeans",),
        report_root=report_root,
        metadata={
            "purpose": "foundation_single_trial_stub",
            "production_outputs_written": False,
        },
    )


__all__ = [
    "DEFAULT_STUDY_BUDGET",
    "DEFAULT_STUDY_REPORT_ROOT",
    "DEFAULT_STUDY_SPLIT_POLICY",
    "REGIME_STUDY_MANIFEST_SCHEMA_VERSION",
    "StudyManifest",
    "default_asset_trend_manifest",
]
