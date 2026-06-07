from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    RegimeAxis,
    RegimeBand,
    RegimeClassification,
    RegimeLayer,
    RunStatus,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
)
from src.regimes.core.feature_cache import (
    FEATURE_CACHE_MANIFEST_ARTIFACT_KIND as FEATURE_CACHE_ARTIFACT_KIND,
    build_feature_cache_manifest,
    sandbox_feature_cache_noop_writer,
)
from src.regimes.core.feature_registry import default_feature_family_registry
from src.regimes.core.foundation_contracts import SourceArtifactLineage
from src.regimes.core.paths import default_foundation_report_root, is_relative_to, require_foundation_report_root
from src.regimes.core.preprocessing import fit_preprocessing_pipeline, transform_score_window_preprocessor
from src.regimes.core.promotion_gate import PROMOTION_STATUS_BLOCKED, PromotionGateInput, evaluate_promotion_gate
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.core.splits import split_train_score_by_rows
from src.regimes.studies.fixtures import synthetic_asset_state_fixture
from src.regimes.studies.manifest import StudyManifest
from src.regimes.studies.search_space import build_search_space
from src.regimes.studies.single_trial import REGIME_SINGLE_TRIAL_ARTIFACT_KIND, SingleTrialResult, run_single_trial


FOUNDATION_SMOKE_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
FOUNDATION_SMOKE_ARTIFACT_KIND = "regime_foundation_smoke_summary"
DEFAULT_FOUNDATION_SMOKE_REPORT_ROOT = default_foundation_report_root("smoke")
FOUNDATION_SMOKE_SUMMARY_JSON = "foundation_smoke_summary.json"
FOUNDATION_SMOKE_SUMMARY_MD = "foundation_smoke_summary.md"


def _validate_smoke_report_root(report_root: str | Path, *, project_root: str | Path | None = None) -> Path:
    return require_foundation_report_root(
        report_root,
        project_root=project_root,
        required_suffix=("reports", "regimes", "foundation", "smoke"),
        error_prefix="Regime foundation smoke report root",
    )


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve()
    if not is_relative_to(candidate, root):
        raise ValueError("Regime foundation smoke artifact path must stay under the report root")
    return candidate


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _timestamp_bound(frame: pd.DataFrame, *, fallback: int) -> str | int:
    if "timestamp" not in frame.columns or frame.empty:
        return int(fallback)
    value = frame["timestamp"].iloc[0 if fallback == 0 else -1]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def foundation_smoke_study_manifest(
    *,
    report_root: str | Path = DEFAULT_FOUNDATION_SMOKE_REPORT_ROOT,
    seed: int = 17,
) -> StudyManifest:
    root = _validate_smoke_report_root(report_root)
    return StudyManifest(
        study_id="foundation_smoke_asset_trend_micro",
        layer=RegimeLayer.ASSET_STATE,
        axis=RegimeAxis.TREND,
        band=RegimeBand.MICRO,
        classification=RegimeClassification.SANDBOX,
        feature_families=("asset_state_trend_metadata_only",),
        preprocessing_options=("robust_scale",),
        candidate_clusterer_families=("kmeans",),
        split_policy={"name": "deterministic_head_tail", "train_fraction": None, "train_rows": 8},
        budget={
            "max_trials": 1,
            "timeout_seconds": 60,
            "random_seed": int(seed),
            "tiny_cluster_threshold": 1,
        },
        report_root=_safe_child(root, "single_trial"),
        metadata={
            "purpose": "foundation_smoke",
            "synthetic_fixture": True,
            "production_outputs_written": False,
            "market_state_execution_enabled": False,
            "relative_state_execution_enabled": False,
        },
    )


def _split_dataset(frame: pd.DataFrame, manifest: StudyManifest) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_train_score_by_rows(frame, manifest.split_policy)


def _source_lineage(source_path: Path) -> tuple[SourceArtifactLineage, ...]:
    return (
        SourceArtifactLineage(
            artifact_kind="synthetic_regime_foundation_smoke_fixture",
            artifact_path=str(source_path),
            schema_version=FOUNDATION_SMOKE_SCHEMA_VERSION,
            content_hash="sha256:foundation-smoke-synthetic-fixture-v1",
            produced_by="src.regimes.studies.foundation_smoke.synthetic_asset_state_fixture",
            metadata={"synthetic": True, "production_input": False},
        ),
    )


def _write_source_fixture(path: Path, frame: pd.DataFrame) -> None:
    payload = {
        "schema_version": FOUNDATION_SMOKE_SCHEMA_VERSION,
        "artifact_kind": "regime_foundation_smoke_source_fixture",
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "synthetic": True,
        "production_input": False,
    }
    _write_json(path, payload)


def _build_feature_cache_manifest(
    *,
    manifest: StudyManifest,
    dataset: pd.DataFrame,
    report_root: Path,
    source_fixture_path: Path,
) -> dict[str, Any]:
    train_frame, score_frame = _split_dataset(dataset, manifest)
    selected = build_search_space(manifest).selected_single_trial
    feature_spec = default_feature_family_registry().get(str(selected["feature_family"]))
    preprocessing = fit_preprocessing_pipeline(
        train_frame,
        feature_spec.required_source_columns,
        preprocess=str(selected["preprocessing"]),
        fit_window={
            "start_ts": _timestamp_bound(train_frame, fallback=0),
            "end_ts": _timestamp_bound(train_frame, fallback=1),
            "role": "train",
        },
        fit_window_role="train",
    )
    transformed = transform_score_window_preprocessor(score_frame, preprocessing.fitted, window_role="score")
    cache_manifest = build_feature_cache_manifest(
        cache_id="foundation_smoke_feature_cache",
        source_lineage=tuple(item.as_dict() for item in _source_lineage(source_fixture_path)),
        feature_family=feature_spec.family_name,
        preprocessing_family=preprocessing.fitted.preprocess_name,
        train_window={
            "start_ts": _timestamp_bound(train_frame, fallback=0),
            "end_ts": _timestamp_bound(train_frame, fallback=1),
        },
        source_columns=feature_spec.required_source_columns,
        selected_columns=preprocessing.fitted.selected_columns,
        shape_metadata={
            "train_row_count": int(len(train_frame)),
            "score_row_count": int(len(score_frame)),
            "retained_feature_count": int(len(preprocessing.fitted.selected_columns)),
            "fit_matrix_shape": [int(preprocessing.fitted.x.shape[0]), int(preprocessing.fitted.x.shape[1])],
            "score_matrix_shape": [int(transformed.x.shape[0]), int(transformed.x.shape[1])],
        },
        preprocessing_metadata=preprocessing.fitted.to_metadata(),
        cache_artifact_path=_safe_child(report_root, "feature_cache", "foundation_smoke_matrix.not_materialized"),
        diagnostics={
            "asset_scope": "single_asset",
            "asset": str(dataset["asset"].iloc[0]) if "asset" in dataset.columns and not dataset.empty else "XBTUSD",
            "dataset_row_count": int(len(dataset)),
            "synthetic_fixture": True,
            "materialization": "noop",
            "production_cache_write_enabled": False,
        },
    )
    return {
        "manifest": cache_manifest.as_dict(),
        "no_op_writer": sandbox_feature_cache_noop_writer(cache_manifest),
    }


def _smoke_promotion_gate(
    *,
    manifest: StudyManifest,
    trial_result: SingleTrialResult,
    artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    return evaluate_promotion_gate(
        PromotionGateInput(
            gate_id="foundation_smoke_promotion_gate",
            study_key=manifest.study_key,
            scoreboard=trial_result.as_dict()["scoreboard"],
            artifact_kind=FOUNDATION_SMOKE_ARTIFACT_KIND,
            artifact_classification=RegimeClassification.SANDBOX,
            artifact_metadata={
                "status": "sandbox_only",
                "sandbox": True,
                "diagnostics_only": True,
                "production_outputs_written": False,
                "study_runner": "src.regimes.studies.foundation_smoke",
            },
            artifact_paths=artifact_paths,
            run_status=RunStatus.SUCCEEDED,
        )
    ).as_dict()


def _markdown_summary(payload: Mapping[str, Any]) -> str:
    scoreboard = payload["scoreboard"]["sections"]
    coverage = scoreboard["coverage_degeneracy"]["metrics"]
    gate = payload["promotion_gate"]
    flat = payload["flat_asset_policy"]
    cache = payload["feature_cache_manifest"]
    return "\n".join(
        [
            "# Regime Foundation Smoke",
            "",
            f"- Run: `{payload['run_id']}`",
            f"- Study: `{payload['study_manifest']['study_id']}`",
            f"- Feature family: `{payload['single_trial']['feature_family']}`",
            f"- Preprocessing: `{payload['single_trial']['preprocessing']}`",
            f"- Clusterer: `{payload['single_trial']['clusterer_family']}`",
            f"- Effective states: `{coverage['effective_state_count']}`",
            f"- Feature-cache manifest status: `{cache['status']}`",
            f"- Flat-asset policy status: `{flat['status']}`",
            f"- Promotion gate status: `{gate['status']}`",
            "",
            "This smoke is deterministic, synthetic, sandbox-only, and blocked from production advancement.",
        ]
    )


@dataclass(frozen=True)
class FoundationSmokeResult:
    summary: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    schema_version: int = FOUNDATION_SMOKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "summary", require_json_mapping(self.summary, field_name="foundation_smoke summary"))
        object.__setattr__(self, "artifact_paths", require_json_mapping(self.artifact_paths, field_name="foundation_smoke artifact_paths"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "summary": to_jsonable(self.summary),
            "artifact_paths": dict(self.artifact_paths),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoundationSmokeResult":
        obj = require_json_object(payload, context="Regime FoundationSmokeResult")
        return cls(
            schema_version=obj.get("schema_version", FOUNDATION_SMOKE_SCHEMA_VERSION),
            summary=obj["summary"],
            artifact_paths=obj["artifact_paths"],
        )

    @classmethod
    def from_json(cls, text: str) -> "FoundationSmokeResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FoundationSmokeResult JSON"))


def run_foundation_smoke(
    *,
    report_root: str | Path = DEFAULT_FOUNDATION_SMOKE_REPORT_ROOT,
    run_id: str = "foundation_smoke",
    seed: int = 17,
    dataset: pd.DataFrame | None = None,
    write_outputs: bool = True,
    project_root: str | Path | None = None,
) -> FoundationSmokeResult:
    root = _validate_smoke_report_root(report_root, project_root=project_root)
    run_token = require_non_empty_string(run_id, field_name="foundation smoke run_id")
    np.random.seed(int(seed))
    frame = dataset.copy() if dataset is not None else synthetic_asset_state_fixture(periods=12)
    manifest = foundation_smoke_study_manifest(report_root=root, seed=int(seed))
    source_fixture_path = _safe_child(root, "foundation_smoke_source_fixture.json")
    summary_json_path = _safe_child(root, FOUNDATION_SMOKE_SUMMARY_JSON)
    summary_md_path = _safe_child(root, FOUNDATION_SMOKE_SUMMARY_MD)
    feature_cache_json_path = _safe_child(root, "feature_cache_manifest.json")

    if write_outputs:
        _write_source_fixture(source_fixture_path, frame)

    trial = run_single_trial(
        manifest,
        dataset=frame,
        trial_id=f"{run_token}_single_trial",
        write_outputs=write_outputs,
        project_root=project_root,
    )
    feature_cache = _build_feature_cache_manifest(
        manifest=manifest,
        dataset=frame,
        report_root=root,
        source_fixture_path=source_fixture_path,
    )
    artifact_paths = {
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
        "feature_cache_manifest_json": str(feature_cache_json_path),
        "source_fixture_json": str(source_fixture_path),
        **{f"single_trial_{key}": value for key, value in trial.as_dict()["artifact_paths"].items()},
    }
    gate = _smoke_promotion_gate(manifest=manifest, trial_result=trial, artifact_paths=artifact_paths)
    summary = {
        "schema_version": FOUNDATION_SMOKE_SCHEMA_VERSION,
        "artifact_kind": FOUNDATION_SMOKE_ARTIFACT_KIND,
        "status": "completed",
        "run_id": run_token,
        "seed": int(seed),
        "study_manifest": manifest.as_dict(),
        "search_space": build_search_space(manifest).as_dict(),
        "single_trial": trial.as_dict(),
        "scoreboard": trial.as_dict()["scoreboard"],
        "feature_cache_manifest": feature_cache["manifest"],
        "feature_cache_no_op_writer": feature_cache["no_op_writer"],
        "flat_asset_policy": trial.as_dict()["flat_asset_policy"],
        "promotion_gate": gate,
        "artifact_paths": artifact_paths,
        "artifact_boundary": {
            "synthetic_fixture": True,
            "sandbox_only": True,
            "metadata_safe_market_relative_surfaces_only": True,
            "production_outputs_written": False,
            "production_writes_enabled": False,
            "production_labels_written": False,
            "production_definitions_written": False,
            "parquet_writes_enabled": False,
            "market_state_execution_enabled": False,
            "relative_state_execution_enabled": False,
        },
    }
    if gate["status"] != PROMOTION_STATUS_BLOCKED:
        raise RuntimeError("Regime foundation smoke promotion gate must block production advancement")
    if write_outputs:
        _write_json(feature_cache_json_path, feature_cache["manifest"])
        _write_json(summary_json_path, summary)
        _write_markdown(summary_md_path, _markdown_summary(summary))
    return FoundationSmokeResult(summary=summary, artifact_paths=artifact_paths)


__all__ = [
    "DEFAULT_FOUNDATION_SMOKE_REPORT_ROOT",
    "FEATURE_CACHE_ARTIFACT_KIND",
    "FOUNDATION_SMOKE_ARTIFACT_KIND",
    "FOUNDATION_SMOKE_SCHEMA_VERSION",
    "FOUNDATION_SMOKE_SUMMARY_JSON",
    "FOUNDATION_SMOKE_SUMMARY_MD",
    "FoundationSmokeResult",
    "foundation_smoke_study_manifest",
    "run_foundation_smoke",
]
