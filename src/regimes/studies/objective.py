from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_json_mapping, require_non_empty_string, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.studies.fixtures import synthetic_asset_state_fixture
from src.regimes.studies.manifest import StudyManifest
from src.regimes.studies.search_space import StudySearchSpace, build_search_space
from src.regimes.studies.single_trial import SingleTrialResult, run_single_trial


OPTUNA_OBJECTIVE_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
OPTUNA_TRIAL_SUMMARY_ARTIFACT_KIND = "regime_optuna_trial_summary"
OPTUNA_OBJECTIVE_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "internal_validity_silhouette",
        "direction": "maximize",
        "scoreboard_path": "sections.internal_validity.metrics.silhouette.value",
    },
    {
        "name": "effective_state_count",
        "direction": "maximize",
        "scoreboard_path": "sections.coverage_degeneracy.metrics.effective_state_count",
    },
    {
        "name": "degeneracy_penalty",
        "direction": "minimize",
        "scoreboard_path": "sections.coverage_degeneracy.metrics",
    },
)
OPTUNA_OBJECTIVE_METRIC_NAMES: tuple[str, ...] = tuple(spec["name"] for spec in OPTUNA_OBJECTIVE_SPECS)
OPTUNA_OBJECTIVE_DIRECTIONS: tuple[str, ...] = tuple(spec["direction"] for spec in OPTUNA_OBJECTIVE_SPECS)
FAILED_TRIAL_OBJECTIVE_VALUES: tuple[float, ...] = (-1.0, 0.0, 999.0)


def _resolve_root(report_root: str | Path) -> Path:
    root = Path(report_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _safe_float(value: object, *, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _metric_value(metric: Mapping[str, Any] | None, *, default: float) -> float:
    if not isinstance(metric, Mapping):
        return float(default)
    if metric.get("status") != "computed":
        return float(default)
    return _safe_float(metric.get("value"), default=default)


def _copy_split_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    split = dict(policy)
    if split.get("train_rows") is not None and "train_fraction" not in split:
        split["train_fraction"] = None
    return split


def scoreboard_objective_values(scoreboard: Mapping[str, Any]) -> tuple[float, ...]:
    payload = require_json_mapping(scoreboard, field_name="scoreboard")
    sections = require_json_mapping(payload.get("sections"), field_name="scoreboard.sections")
    internal = require_json_mapping(sections.get("internal_validity"), field_name="scoreboard.sections.internal_validity")
    coverage = require_json_mapping(sections.get("coverage_degeneracy"), field_name="scoreboard.sections.coverage_degeneracy")
    internal_metrics = require_json_mapping(internal.get("metrics"), field_name="scoreboard internal metrics")
    coverage_metrics = require_json_mapping(coverage.get("metrics"), field_name="scoreboard coverage metrics")

    silhouette = _metric_value(internal_metrics.get("silhouette"), default=-1.0)
    effective_state_count = _safe_float(coverage_metrics.get("effective_state_count"), default=0.0)
    tiny_share = _safe_float(coverage_metrics.get("singleton_or_tiny_cluster_share"), default=0.0)
    noise_share = _safe_float(coverage_metrics.get("noise_share"), default=0.0)
    unknown_or_null_share = _safe_float(coverage_metrics.get("unknown_or_null_share"), default=0.0)
    degeneracy_penalty = tiny_share + noise_share + unknown_or_null_share
    if bool(coverage_metrics.get("one_cluster_flag")):
        degeneracy_penalty += 1.0
    if bool(coverage_metrics.get("all_noise_flag")):
        degeneracy_penalty += 1.0
    if bool(coverage_metrics.get("all_unknown_or_null_flag")):
        degeneracy_penalty += 1.0
    return (
        float(silhouette),
        float(effective_state_count),
        float(degeneracy_penalty),
    )


def scoreboard_runtime_diagnostics(scoreboard: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_json_mapping(scoreboard, field_name="scoreboard")
    sections = require_json_mapping(payload.get("sections"), field_name="scoreboard.sections")
    runtime = require_json_mapping(sections.get("runtime"), field_name="scoreboard.sections.runtime")
    runtime_metrics = require_json_mapping(runtime.get("metrics"), field_name="scoreboard runtime metrics")
    fit_seconds = _safe_float(runtime_metrics.get("fit_seconds"), default=0.0)
    assign_seconds = _safe_float(runtime_metrics.get("assign_seconds"), default=0.0)
    return {
        "status": runtime.get("status"),
        "fit_seconds": fit_seconds,
        "assign_seconds": assign_seconds,
        "runtime_seconds": fit_seconds + assign_seconds,
        "objective_role": "diagnostic_metadata_only",
    }


def single_candidate_manifest(
    *,
    base_manifest: StudyManifest | Mapping[str, Any],
    candidate: Mapping[str, Any],
    report_root: str | Path,
    trial_number: int,
) -> StudyManifest:
    base = base_manifest if isinstance(base_manifest, StudyManifest) else StudyManifest.from_dict(base_manifest)
    row = require_json_object(candidate, context="Regime Optuna search candidate")
    return StudyManifest(
        study_id=f"{base.study_id}_optuna_{int(trial_number):03d}",
        layer=base.layer,
        axis=base.axis,
        band=base.band,
        classification=base.classification,
        feature_families=(row["feature_family"],),
        preprocessing_options=(row["preprocessing"],),
        candidate_clusterer_families=(row["clusterer_family"],),
        split_policy=_copy_split_policy(base.split_policy),
        budget={**dict(base.budget), "max_trials": 1},
        report_root=report_root,
        metadata={
            **dict(base.metadata),
            "parent_study_id": base.study_id,
            "optuna_trial_number": int(trial_number),
            "optuna_candidate": to_jsonable(row),
            "production_outputs_written": False,
            "study_runner": "src.regimes.studies.optuna_runner",
        },
    )


@dataclass(frozen=True)
class OptunaTrialSummary:
    trial_number: int
    trial_id: str
    params: Mapping[str, Any]
    values: Sequence[float]
    status: str
    metric_names: Sequence[str] = OPTUNA_OBJECTIVE_METRIC_NAMES
    directions: Sequence[str] = OPTUNA_OBJECTIVE_DIRECTIONS
    single_trial: Mapping[str, Any] | None = None
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    failure_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = OPTUNA_OBJECTIVE_SCHEMA_VERSION
    artifact_kind: str = OPTUNA_TRIAL_SUMMARY_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        trial_id = require_non_empty_string(self.trial_id, field_name="optuna trial_id")
        status = require_non_empty_string(self.status, field_name="optuna trial status").lower()
        values = tuple(_safe_float(value, default=999.0) for value in self.values)
        if len(values) != len(OPTUNA_OBJECTIVE_METRIC_NAMES):
            raise ValueError("Regime Optuna trial values must match objective metric count")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "trial_number", int(self.trial_number))
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "params", require_json_mapping(self.params, field_name="optuna trial params"))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metric_names", tuple(str(value) for value in self.metric_names))
        object.__setattr__(self, "directions", tuple(str(value) for value in self.directions))
        object.__setattr__(
            self,
            "single_trial",
            None if self.single_trial is None else require_json_mapping(self.single_trial, field_name="single_trial"),
        )
        object.__setattr__(self, "artifact_paths", require_json_mapping(self.artifact_paths, field_name="artifact_paths"))
        object.__setattr__(
            self,
            "failure_metadata",
            require_json_mapping(self.failure_metadata, field_name="failure_metadata"),
        )
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "trial_number": int(self.trial_number),
            "trial_id": self.trial_id,
            "params": to_jsonable(self.params),
            "metric_names": list(self.metric_names),
            "directions": list(self.directions),
            "values": [float(value) for value in self.values],
            "status": self.status,
            "single_trial": None if self.single_trial is None else to_jsonable(self.single_trial),
            "artifact_paths": dict(self.artifact_paths),
            "failure_metadata": to_jsonable(self.failure_metadata),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptunaTrialSummary":
        obj = require_json_object(payload, context="Regime OptunaTrialSummary")
        return cls(
            schema_version=obj.get("schema_version", OPTUNA_OBJECTIVE_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", OPTUNA_TRIAL_SUMMARY_ARTIFACT_KIND),
            trial_number=obj["trial_number"],
            trial_id=obj["trial_id"],
            params=obj["params"],
            metric_names=obj.get("metric_names", OPTUNA_OBJECTIVE_METRIC_NAMES),
            directions=obj.get("directions", OPTUNA_OBJECTIVE_DIRECTIONS),
            values=obj["values"],
            status=obj["status"],
            single_trial=obj.get("single_trial"),
            artifact_paths=obj.get("artifact_paths", {}),
            failure_metadata=obj.get("failure_metadata", {}),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "OptunaTrialSummary":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime OptunaTrialSummary JSON"))


class RegimeOptunaObjective:
    def __init__(
        self,
        manifest: StudyManifest | Mapping[str, Any],
        *,
        report_root: str | Path,
        dataset: pd.DataFrame | None = None,
        seed: int = 17,
        write_outputs: bool = True,
    ) -> None:
        self.manifest = manifest if isinstance(manifest, StudyManifest) else StudyManifest.from_dict(manifest)
        self.report_root = _resolve_root(report_root)
        self.dataset = dataset.copy() if dataset is not None else synthetic_asset_state_fixture(periods=12)
        self.seed = int(seed)
        self.write_outputs = bool(write_outputs)
        self.search_space: StudySearchSpace = build_search_space(self.manifest)
        self.trial_summaries: list[OptunaTrialSummary] = []

    @property
    def candidate_count(self) -> int:
        return int(len(self.search_space.candidate_trials))

    def __call__(self, trial: Any) -> tuple[float, ...]:
        if self.candidate_count <= 0:
            raise ValueError("Regime Optuna objective requires at least one candidate trial")
        trial_number = int(getattr(trial, "number", len(self.trial_summaries)))
        candidate_index = int(trial.suggest_categorical("candidate_index", list(range(self.candidate_count))))
        candidate = self.search_space.candidate_trials[candidate_index]
        trial_id = f"optuna_trial_{trial_number:03d}_candidate_{candidate_index:03d}"
        params = {
            "candidate_index": candidate_index,
            "feature_family": candidate["feature_family"],
            "preprocessing": candidate["preprocessing"],
            "clusterer_family": candidate["clusterer_family"],
        }
        try:
            manifest = single_candidate_manifest(
                base_manifest=self.manifest,
                candidate=candidate,
                report_root=self.report_root,
                trial_number=trial_number,
            )
            result: SingleTrialResult = run_single_trial(
                manifest,
                dataset=self.dataset.copy(),
                trial_id=trial_id,
                write_outputs=self.write_outputs,
            )
            payload = result.as_dict()
            values = scoreboard_objective_values(payload["scoreboard"])
            runtime_diagnostics = scoreboard_runtime_diagnostics(payload["scoreboard"])
            summary = OptunaTrialSummary(
                trial_number=trial_number,
                trial_id=trial_id,
                params=params,
                values=values,
                status="completed",
                single_trial=payload,
                artifact_paths=payload["artifact_paths"],
                metadata={
                    "candidate": to_jsonable(candidate),
                    "seed": self.seed,
                    "promotion_gate_status": payload["promotion_gate"]["status"],
                    "runtime_diagnostics": runtime_diagnostics,
                    "runtime_objective_policy": "runtime is diagnostic metadata only in the deterministic foundation stub",
                },
            )
        except Exception as exc:
            values = FAILED_TRIAL_OBJECTIVE_VALUES
            summary = OptunaTrialSummary(
                trial_number=trial_number,
                trial_id=trial_id,
                params=params,
                values=values,
                status="failed",
                failure_metadata={
                    "reason_code": "single_trial_execution_failed",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                    "candidate": to_jsonable(candidate),
                },
                metadata={"seed": self.seed},
            )
        self.trial_summaries.append(summary)
        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("regime_trial_summary", summary.as_dict())
            trial.set_user_attr("regime_trial_id", trial_id)
            trial.set_user_attr("regime_clusterer_family", params["clusterer_family"])
            trial.set_user_attr("regime_trial_status", summary.status)
            if summary.failure_metadata:
                trial.set_user_attr("regime_failure_metadata", summary.failure_metadata)
        return tuple(float(value) for value in values)


__all__ = [
    "FAILED_TRIAL_OBJECTIVE_VALUES",
    "OPTUNA_OBJECTIVE_DIRECTIONS",
    "OPTUNA_OBJECTIVE_METRIC_NAMES",
    "OPTUNA_OBJECTIVE_SCHEMA_VERSION",
    "OPTUNA_OBJECTIVE_SPECS",
    "OPTUNA_TRIAL_SUMMARY_ARTIFACT_KIND",
    "OptunaTrialSummary",
    "RegimeOptunaObjective",
    "scoreboard_runtime_diagnostics",
    "scoreboard_objective_values",
    "single_candidate_manifest",
]
