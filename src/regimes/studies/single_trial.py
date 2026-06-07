from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.clusterer_base import ASSIGN_STATUS_ASSIGNED, FIT_STATUS_FITTED
from src.regimes.core.clusterer_registry import default_clusterer_registry
from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, RunStatus, require_json_mapping, require_non_empty_string, require_schema_version
from src.regimes.core.feature_cache import (
    build_feature_cache_manifest,
    can_reuse_cache,
    sandbox_feature_cache_noop_writer,
)
from src.regimes.core.feature_registry import default_feature_family_registry
from src.regimes.core.flat_asset_policy import evaluate_flat_asset_policy
from src.regimes.core.paths import (
    has_path_parts,
    is_production_adjacent_path,
    is_relative_to,
    resolve_project_path,
)
from src.regimes.core.preprocessing import (
    fit_preprocessing_pipeline,
    transform_score_window_preprocessor,
)
from src.regimes.core.promotion_gate import PromotionGateInput, evaluate_promotion_gate
from src.regimes.core.scoreboard import build_regime_scoreboard
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.core.splits import split_train_score_by_rows
from src.regimes.studies.fixtures import synthetic_asset_state_fixture
from src.regimes.studies.manifest import StudyManifest, default_asset_trend_manifest
from src.regimes.studies.search_space import StudySearchSpace, build_search_space


REGIME_SINGLE_TRIAL_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
REGIME_SINGLE_TRIAL_ARTIFACT_KIND = "regime_single_trial_result"
_FOUNDATION_ROOT_PARTS = ("reports", "regimes", "foundation")


def _has_foundation_report_root(path: Path) -> bool:
    return has_path_parts(path, _FOUNDATION_ROOT_PARTS)


def validate_single_trial_report_root(
    report_root: str | Path,
    *,
    project_root: str | Path | None = None,
    allow_test_only_non_foundation_report_root: bool = False,
) -> Path:
    root = resolve_project_path(report_root, project_root=project_root)
    if is_production_adjacent_path(root, project_root=project_root):
        raise ValueError("Regime single-trial report root is production-adjacent and is not allowed")
    if not allow_test_only_non_foundation_report_root and not _has_foundation_report_root(root):
        raise ValueError("Regime single-trial report root must be under reports/regimes/foundation")
    return root


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve()
    if not is_relative_to(candidate, root):
        raise ValueError("Regime single-trial artifact path must stay under report_root")
    return candidate


def _safe_path_token(value: str, *, field_name: str) -> str:
    text = require_non_empty_string(value, field_name=field_name)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Regime single-trial artifact path token {field_name} must not contain path separators")
    return text


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _split_dataset(frame: pd.DataFrame, split_policy: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_train_score_by_rows(
        frame,
        split_policy,
        min_total_rows=3,
        min_total_rows_error="Regime single trial requires at least three dataset rows",
    )


def _stability_perturbations(labels: Sequence[int]) -> tuple[dict[str, Any], ...]:
    label_list = [int(label) for label in labels]
    return ({"name": "identity_precomputed_stub", "labels": label_list},)


def _artifact_paths(root: Path, *, study_id: str, trial_id: str) -> dict[str, Path]:
    study_part = _safe_path_token(study_id, field_name="study_id")
    trial_part = _safe_path_token(trial_id, field_name="trial_id")
    trial_root = _safe_child(root, study_part, trial_part)
    return {
        "trial_root": trial_root,
        "manifest_json": _safe_child(root, study_part, trial_part, "study_manifest.json"),
        "search_space_json": _safe_child(root, study_part, trial_part, "search_space.json"),
        "preprocessing_json": _safe_child(root, study_part, trial_part, "preprocessing_diagnostics.json"),
        "fit_result_json": _safe_child(root, study_part, trial_part, "clusterer_fit_result.json"),
        "assignment_result_json": _safe_child(root, study_part, trial_part, "clusterer_assignment_result.json"),
        "feature_cache_manifest_json": _safe_child(root, study_part, trial_part, "feature_cache_manifest.json"),
        "feature_cache_decision_json": _safe_child(root, study_part, trial_part, "feature_cache_decision.json"),
        "feature_cache_noop_writer_json": _safe_child(root, study_part, trial_part, "feature_cache_noop_writer.json"),
        "flat_asset_policy_json": _safe_child(root, study_part, trial_part, "flat_asset_policy.json"),
        "scoreboard_json": _safe_child(root, study_part, trial_part, "scoreboard.json"),
        "promotion_gate_json": _safe_child(root, study_part, trial_part, "promotion_gate.json"),
        "trial_result_json": _safe_child(root, study_part, trial_part, "trial_result.json"),
        "trial_report_md": _safe_child(root, study_part, trial_part, "trial_report.md"),
    }


def _timestamp_value(frame: pd.DataFrame, *, first: bool, fallback: int) -> str | int:
    if "timestamp" not in frame.columns or frame.empty:
        return int(fallback)
    value = frame["timestamp"].iloc[0 if first else -1]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _train_window_identity(train_frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "start_ts": _timestamp_value(train_frame, first=True, fallback=0),
        "end_ts": _timestamp_value(train_frame, first=False, fallback=max(int(len(train_frame)) - 1, 0)),
    }


def _dataframe_content_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    scoped = frame.loc[:, available].copy() if available else frame.copy()
    payload = pd.util.hash_pandas_object(scoped, index=True).values.tobytes()
    digest = hashlib.sha256(payload)
    digest.update("|".join(str(column) for column in scoped.columns).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _cache_source_lineage(
    *,
    frame: pd.DataFrame,
    dataset_kind: str,
    feature_columns: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "artifact_kind": "regime_single_trial_dataframe_fixture",
            "artifact_path": f"memory://regimes/single_trial/{dataset_kind}",
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "content_hash": _dataframe_content_hash(frame, ("timestamp", "asset", *feature_columns)),
            "produced_by": "src.regimes.studies.single_trial",
            "metadata": {
                "dataset_kind": dataset_kind,
                "row_count": int(len(frame)),
                "columns": list(str(column) for column in frame.columns),
                "production_input": False,
            },
        },
    )


@dataclass(frozen=True)
class SingleTrialResult:
    manifest: StudyManifest | Mapping[str, Any]
    search_space: StudySearchSpace | Mapping[str, Any]
    trial_id: str
    artifact_paths: Mapping[str, str]
    feature_family: str
    preprocessing: str
    clusterer_family: str
    train_row_count: int
    score_row_count: int
    selected_columns: Sequence[str]
    fit_result: Mapping[str, Any]
    assignment_result: Mapping[str, Any]
    scoreboard: Mapping[str, Any]
    promotion_gate: Mapping[str, Any]
    feature_cache_manifest: Mapping[str, Any] = field(default_factory=dict)
    feature_cache_decision: Mapping[str, Any] = field(default_factory=dict)
    feature_cache_noop_writer: Mapping[str, Any] = field(default_factory=dict)
    flat_asset_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_SINGLE_TRIAL_SCHEMA_VERSION
    artifact_kind: str = REGIME_SINGLE_TRIAL_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        manifest = self.manifest if isinstance(self.manifest, StudyManifest) else StudyManifest.from_dict(self.manifest)
        search_space = (
            self.search_space
            if isinstance(self.search_space, StudySearchSpace)
            else StudySearchSpace.from_dict(self.search_space)
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "search_space", search_space)
        object.__setattr__(self, "trial_id", require_non_empty_string(self.trial_id, field_name="trial_id"))
        object.__setattr__(self, "artifact_paths", require_json_mapping(self.artifact_paths, field_name="artifact_paths"))
        object.__setattr__(self, "feature_family", require_non_empty_string(self.feature_family, field_name="feature_family"))
        object.__setattr__(self, "preprocessing", require_non_empty_string(self.preprocessing, field_name="preprocessing"))
        object.__setattr__(self, "clusterer_family", require_non_empty_string(self.clusterer_family, field_name="clusterer_family"))
        object.__setattr__(self, "train_row_count", int(self.train_row_count))
        object.__setattr__(self, "score_row_count", int(self.score_row_count))
        object.__setattr__(self, "selected_columns", tuple(str(column) for column in self.selected_columns))
        object.__setattr__(self, "fit_result", require_json_mapping(self.fit_result, field_name="fit_result"))
        object.__setattr__(self, "assignment_result", require_json_mapping(self.assignment_result, field_name="assignment_result"))
        object.__setattr__(self, "scoreboard", require_json_mapping(self.scoreboard, field_name="scoreboard"))
        object.__setattr__(self, "promotion_gate", require_json_mapping(self.promotion_gate, field_name="promotion_gate"))
        object.__setattr__(
            self,
            "feature_cache_manifest",
            require_json_mapping(self.feature_cache_manifest, field_name="feature_cache_manifest"),
        )
        object.__setattr__(
            self,
            "feature_cache_decision",
            require_json_mapping(self.feature_cache_decision, field_name="feature_cache_decision"),
        )
        object.__setattr__(
            self,
            "feature_cache_noop_writer",
            require_json_mapping(self.feature_cache_noop_writer, field_name="feature_cache_noop_writer"),
        )
        object.__setattr__(self, "flat_asset_policy", require_json_mapping(self.flat_asset_policy, field_name="flat_asset_policy"))
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "manifest": self.manifest.as_dict(),
            "search_space": self.search_space.as_dict(),
            "trial_id": self.trial_id,
            "artifact_paths": dict(self.artifact_paths),
            "feature_family": self.feature_family,
            "preprocessing": self.preprocessing,
            "clusterer_family": self.clusterer_family,
            "train_row_count": int(self.train_row_count),
            "score_row_count": int(self.score_row_count),
            "selected_columns": list(self.selected_columns),
            "fit_result": to_jsonable(self.fit_result),
            "assignment_result": to_jsonable(self.assignment_result),
            "scoreboard": to_jsonable(self.scoreboard),
            "promotion_gate": to_jsonable(self.promotion_gate),
            "feature_cache_manifest": to_jsonable(self.feature_cache_manifest),
            "feature_cache_decision": to_jsonable(self.feature_cache_decision),
            "feature_cache_noop_writer": to_jsonable(self.feature_cache_noop_writer),
            "flat_asset_policy": to_jsonable(self.flat_asset_policy),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SingleTrialResult":
        obj = require_json_object(payload, context="Regime SingleTrialResult")
        return cls(
            schema_version=obj.get("schema_version", REGIME_SINGLE_TRIAL_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_SINGLE_TRIAL_ARTIFACT_KIND),
            manifest=obj["manifest"],
            search_space=obj["search_space"],
            trial_id=obj["trial_id"],
            artifact_paths=obj["artifact_paths"],
            feature_family=obj["feature_family"],
            preprocessing=obj["preprocessing"],
            clusterer_family=obj["clusterer_family"],
            train_row_count=obj["train_row_count"],
            score_row_count=obj["score_row_count"],
            selected_columns=obj["selected_columns"],
            fit_result=obj["fit_result"],
            assignment_result=obj["assignment_result"],
            scoreboard=obj["scoreboard"],
            promotion_gate=obj["promotion_gate"],
            feature_cache_manifest=obj.get("feature_cache_manifest", {}),
            feature_cache_decision=obj.get("feature_cache_decision", {}),
            feature_cache_noop_writer=obj.get("feature_cache_noop_writer", {}),
            flat_asset_policy=obj.get("flat_asset_policy", {}),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "SingleTrialResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime SingleTrialResult JSON"))


def run_single_trial(
    manifest: StudyManifest | Mapping[str, Any] | None = None,
    *,
    dataset: pd.DataFrame | None = None,
    trial_id: str | None = None,
    write_outputs: bool = True,
    project_root: str | Path | None = None,
    allow_test_only_non_foundation_report_root: bool = False,
) -> SingleTrialResult:
    study = manifest if isinstance(manifest, StudyManifest) else StudyManifest.from_dict(manifest) if manifest else default_asset_trend_manifest()
    report_root = validate_single_trial_report_root(
        study.report_root,
        project_root=project_root,
        allow_test_only_non_foundation_report_root=allow_test_only_non_foundation_report_root,
    )
    dataset_kind = "synthetic_fixture" if dataset is None else "provided_fixture"
    frame = dataset.copy() if dataset is not None else synthetic_asset_state_fixture()
    train_frame, score_frame = _split_dataset(frame, study.split_policy)
    search_space = build_search_space(study)
    selected = search_space.selected_single_trial
    selected_trial_id = require_non_empty_string(trial_id or selected["trial_id"], field_name="trial_id")

    feature_spec = default_feature_family_registry().get(str(selected["feature_family"]))
    if not feature_spec.is_compatible(layer=study.layer, axis=study.axis, band=study.band):
        raise ValueError("Regime feature family is not compatible with the study manifest")
    feature_columns = feature_spec.required_source_columns

    preprocessing = fit_preprocessing_pipeline(
        train_frame,
        feature_columns,
        preprocess=str(selected["preprocessing"]),
        fit_window={
            "split_policy": study.split_policy,
            "train_row_count": int(len(train_frame)),
            "score_row_count": int(len(score_frame)),
        },
        fit_window_role="train",
    )
    if preprocessing.fitted.x.shape[0] == 0 or preprocessing.fitted.x.shape[1] == 0:
        raise ValueError("Regime single trial preprocessing retained no finite train features")
    score_matrix = transform_score_window_preprocessor(score_frame, preprocessing.fitted, window_role="score")

    clusterer = default_clusterer_registry().build(
        str(selected["clusterer_family"]),
        **dict(selected.get("clusterer_hyperparameters", {})),
    )
    fit_result = clusterer.fit(preprocessing.fitted.x)
    if fit_result.status != FIT_STATUS_FITTED:
        raise ValueError(f"Regime single trial clusterer fit failed: {fit_result.failure_metadata}")
    assignment_result = clusterer.assign(score_matrix.x)
    if assignment_result.status != ASSIGN_STATUS_ASSIGNED:
        raise ValueError(f"Regime single trial clusterer assignment failed: {assignment_result.failure_metadata}")

    flat_policy = evaluate_flat_asset_policy(
        preprocessing.fitted.clean_frame,
        preprocessing.fitted.selected_columns,
        labels=fit_result.labels,
        layer=study.layer,
        axis=study.axis,
        band=study.band,
        source_metadata={
            "study_id": study.study_id,
            "trial_id": selected_trial_id,
            "feature_family": feature_spec.family_name,
            "preprocessing": preprocessing.fitted.preprocess_name,
        },
    )

    scoreboard = build_regime_scoreboard(
        trial_id=selected_trial_id,
        clusterer_family=str(selected["clusterer_family"]),
        labels=fit_result.labels,
        features=preprocessing.fitted.x,
        fit_result=fit_result,
        assignment_result=assignment_result,
        stability_perturbations=_stability_perturbations(fit_result.labels),
        forward_frame=preprocessing.fitted.clean_frame,
        tiny_cluster_threshold=int(study.budget.get("tiny_cluster_threshold", 1)),
        metadata={
            "study_id": study.study_id,
            "feature_family": feature_spec.family_name,
            "preprocessing": preprocessing.fitted.preprocess_name,
        },
    )

    paths = _artifact_paths(report_root, study_id=study.study_id, trial_id=selected_trial_id)
    artifact_paths = {name: str(path) for name, path in paths.items()}
    cache_source_lineage = _cache_source_lineage(
        frame=frame,
        dataset_kind=dataset_kind,
        feature_columns=feature_columns,
    )
    cache_train_window = _train_window_identity(train_frame)
    feature_cache_manifest = build_feature_cache_manifest(
        cache_id=f"{selected_trial_id}_feature_cache",
        source_lineage=cache_source_lineage,
        feature_family=feature_spec.family_name,
        preprocessing_family=preprocessing.fitted.preprocess_name,
        train_window=cache_train_window,
        source_columns=feature_columns,
        selected_columns=preprocessing.fitted.selected_columns,
        shape_metadata={
            "train_row_count": int(len(train_frame)),
            "score_row_count": int(len(score_frame)),
            "retained_feature_count": int(len(preprocessing.fitted.selected_columns)),
            "fit_matrix_shape": [int(preprocessing.fitted.x.shape[0]), int(preprocessing.fitted.x.shape[1])],
            "score_matrix_shape": [int(score_matrix.x.shape[0]), int(score_matrix.x.shape[1])],
        },
        preprocessing_metadata=preprocessing.fitted.to_metadata(),
        cache_artifact_path=str(_safe_child(paths["trial_root"], "feature_cache_matrix.not_materialized")),
        diagnostics={
            "materialization": "noop",
            "cache_reuse_optional": True,
            "production_cache_write_enabled": False,
        },
    )
    feature_cache_decision = can_reuse_cache(
        feature_cache_manifest,
        source_lineage=cache_source_lineage,
        feature_family=feature_spec.family_name,
        preprocessing_family=preprocessing.fitted.preprocess_name,
        train_window=cache_train_window,
    )
    feature_cache_noop_writer = sandbox_feature_cache_noop_writer(feature_cache_manifest)
    gate = evaluate_promotion_gate(
        PromotionGateInput(
            gate_id=f"{selected_trial_id}_promotion_gate",
            study_key=study.study_key,
            scoreboard=scoreboard.as_dict(),
            artifact_kind=REGIME_SINGLE_TRIAL_ARTIFACT_KIND,
            artifact_classification=study.classification,
            artifact_metadata={
                "status": "foundation_single_trial",
                "production_outputs_written": False,
                "study_runner": "src.regimes.studies.single_trial",
            },
            artifact_paths=artifact_paths,
            run_status=RunStatus.SUCCEEDED,
        )
    )

    result = SingleTrialResult(
        manifest=study,
        search_space=search_space,
        trial_id=selected_trial_id,
        artifact_paths=artifact_paths,
        feature_family=feature_spec.family_name,
        preprocessing=preprocessing.fitted.preprocess_name,
        clusterer_family=str(selected["clusterer_family"]),
        train_row_count=int(len(train_frame)),
        score_row_count=int(len(score_frame)),
        selected_columns=preprocessing.fitted.selected_columns,
        fit_result=fit_result.as_dict(),
        assignment_result=assignment_result.as_dict(),
        scoreboard=scoreboard.as_dict(),
        promotion_gate=gate.as_dict(),
        feature_cache_manifest=feature_cache_manifest.as_dict(),
        feature_cache_decision=feature_cache_decision.as_dict(),
        feature_cache_noop_writer=feature_cache_noop_writer,
        flat_asset_policy=flat_policy.as_dict(),
        metadata={
            "dataset_rows": int(len(frame)),
            "dataset_kind": dataset_kind,
            "report_root": str(report_root),
            "write_outputs": bool(write_outputs),
            "report_root_policy": {
                "must_be_under_reports_regimes_foundation": not bool(allow_test_only_non_foundation_report_root),
                "test_only_non_foundation_bypass": bool(allow_test_only_non_foundation_report_root),
                "production_adjacent_roots_rejected": True,
            },
        },
    )

    if write_outputs:
        _write_json(paths["manifest_json"], study.as_dict())
        _write_json(paths["search_space_json"], search_space.as_dict())
        _write_json(paths["preprocessing_json"], preprocessing.as_dict())
        _write_json(paths["fit_result_json"], fit_result.as_dict())
        _write_json(paths["assignment_result_json"], assignment_result.as_dict())
        _write_json(paths["feature_cache_manifest_json"], feature_cache_manifest.as_dict())
        _write_json(paths["feature_cache_decision_json"], feature_cache_decision.as_dict())
        _write_json(paths["feature_cache_noop_writer_json"], feature_cache_noop_writer)
        _write_json(paths["flat_asset_policy_json"], flat_policy.as_dict())
        _write_json(paths["scoreboard_json"], scoreboard.as_dict())
        _write_json(paths["promotion_gate_json"], gate.as_dict())
        _write_json(paths["trial_result_json"], result.as_dict())
        _write_markdown(paths["trial_report_md"], _trial_markdown(result))

    return result


def _trial_markdown(result: SingleTrialResult) -> str:
    payload = result.as_dict()
    scoreboard = payload["scoreboard"]["sections"]
    gate = payload["promotion_gate"]
    coverage = scoreboard["coverage_degeneracy"]["metrics"]
    return "\n".join(
        [
            "# Regime Foundation Single Trial",
            "",
            f"- Trial: `{payload['trial_id']}`",
            f"- Study: `{payload['manifest']['study_id']}`",
            f"- Feature family: `{payload['feature_family']}`",
            f"- Preprocessing: `{payload['preprocessing']}`",
            f"- Clusterer: `{payload['clusterer_family']}`",
            f"- Train rows: `{payload['train_row_count']}`",
            f"- Score rows: `{payload['score_row_count']}`",
            f"- Effective states: `{coverage['effective_state_count']}`",
            f"- Stability status: `{scoreboard['stability']['status']}`",
            f"- Economic status: `{scoreboard['economic_separability']['status']}`",
            f"- Feature-cache decision: `{payload['feature_cache_decision'].get('decision', 'not_reported')}`",
            f"- Flat asset policy status: `{payload['flat_asset_policy']['status']}`",
            f"- Promotion gate status: `{gate['status']}`",
            "",
            "This artifact is a bounded foundation runner output. It is not a production promotion or release artifact.",
        ]
    )


__all__ = [
    "REGIME_SINGLE_TRIAL_ARTIFACT_KIND",
    "REGIME_SINGLE_TRIAL_SCHEMA_VERSION",
    "SingleTrialResult",
    "run_single_trial",
    "validate_single_trial_report_root",
]
