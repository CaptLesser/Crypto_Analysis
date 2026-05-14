from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    RegimeAxis,
    RegimeBand,
    RegimeClassification,
    RegimeLayer,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
)
from src.regimes.core.paths import default_foundation_report_root, is_relative_to, require_foundation_report_root
from src.regimes.core.promotion_gate import PROMOTION_STATUS_BLOCKED
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.studies.fixtures import synthetic_asset_state_fixture
from src.regimes.studies.manifest import StudyManifest
from src.regimes.studies.objective import (
    OPTUNA_OBJECTIVE_DIRECTIONS,
    OPTUNA_OBJECTIVE_METRIC_NAMES,
    OPTUNA_OBJECTIVE_SPECS,
    RegimeOptunaObjective,
)
from src.regimes.studies.search_space import build_search_space


try:  # pragma: no cover - exercised only when optional dependency is missing
    import optuna as _OPTUNA

    OPTUNA_AVAILABLE = True
    OPTUNA_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only when optional dependency is missing
    _OPTUNA = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False
    OPTUNA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


OPTUNA_STUB_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
OPTUNA_STUB_ARTIFACT_KIND = "regime_optuna_stub_study_summary"
DEFAULT_OPTUNA_REPORT_ROOT = default_foundation_report_root("optuna")
OPTUNA_STUB_SUMMARY_JSON = "optuna_study_summary.json"
OPTUNA_STUB_SUMMARY_MD = "optuna_study_summary.md"
OPTUNA_STUB_TRIAL_SUMMARIES_JSON = "optuna_trial_summaries.json"


class OptunaUnavailableError(RuntimeError):
    pass


def require_optuna() -> Any:
    if not OPTUNA_AVAILABLE or _OPTUNA is None:
        details = f" Import error: {OPTUNA_IMPORT_ERROR}" if OPTUNA_IMPORT_ERROR else ""
        raise OptunaUnavailableError(
            "Optuna is required for the Regime Optuna harness; install optuna to run this bounded stub."
            + details
        )
    return _OPTUNA


def validate_optuna_report_root(report_root: str | Path, *, project_root: str | Path | None = None) -> Path:
    return require_foundation_report_root(
        report_root,
        project_root=project_root,
        required_suffix=("reports", "regimes", "foundation", "optuna"),
        error_prefix="Regime Optuna report root",
    )


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve()
    if not is_relative_to(candidate, root):
        raise ValueError("Regime Optuna artifact path must stay under report_root")
    return candidate


def _copy_split_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    split = dict(policy)
    if split.get("train_rows") is not None and "train_fraction" not in split:
        split["train_fraction"] = None
    return split


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def default_optuna_study_manifest(
    *,
    report_root: str | Path = DEFAULT_OPTUNA_REPORT_ROOT,
    seed: int = 17,
) -> StudyManifest:
    root = validate_optuna_report_root(report_root)
    return StudyManifest(
        study_id="foundation_asset_trend_micro_optuna_stub",
        layer=RegimeLayer.ASSET_STATE,
        axis=RegimeAxis.TREND,
        band=RegimeBand.MICRO,
        classification=RegimeClassification.SANDBOX,
        feature_families=("asset_state_trend_metadata_only",),
        preprocessing_options=("robust_scale",),
        candidate_clusterer_families=("kmeans", "minibatch_kmeans", "gaussian_mixture"),
        split_policy={"name": "deterministic_head_tail", "train_fraction": None, "train_rows": 8},
        budget={
            "max_trials": 3,
            "timeout_seconds": 60,
            "random_seed": int(seed),
            "tiny_cluster_threshold": 1,
        },
        report_root=_safe_child(root, "single_trials"),
        metadata={
            "purpose": "optuna_multi_trial_stub",
            "synthetic_fixture": True,
            "single_process": True,
            "production_outputs_written": False,
            "distributed_execution_enabled": False,
        },
    )


def _manifest_with_report_root(manifest: StudyManifest | Mapping[str, Any], *, report_root: Path) -> StudyManifest:
    study = manifest if isinstance(manifest, StudyManifest) else StudyManifest.from_dict(manifest)
    return StudyManifest(
        study_id=study.study_id,
        layer=study.layer,
        axis=study.axis,
        band=study.band,
        classification=study.classification,
        feature_families=study.feature_families,
        preprocessing_options=study.preprocessing_options,
        candidate_clusterer_families=study.candidate_clusterer_families,
        split_policy=_copy_split_policy(study.split_policy),
        budget=study.budget,
        report_root=_safe_child(report_root, "single_trials"),
        metadata={
            **dict(study.metadata),
            "production_outputs_written": False,
            "study_runner": "src.regimes.studies.optuna_runner",
        },
    )


def _storage(optuna: Any, root: Path, *, storage_mode: str) -> tuple[Any, dict[str, Any]]:
    mode = require_non_empty_string(storage_mode, field_name="Optuna storage_mode").lower()
    if mode == "memory":
        return None, {"storage_mode": "memory", "storage_path": None}
    root.mkdir(parents=True, exist_ok=True)
    if mode == "sqlite":
        path = _safe_child(root, "optuna_study.sqlite3")
        return "sqlite:///" + path.as_posix(), {"storage_mode": "sqlite", "storage_path": str(path)}
    if mode == "journal":
        journal_storage = getattr(optuna.storages, "JournalStorage", None)
        journal_file_storage = getattr(optuna.storages, "JournalFileStorage", None)
        if journal_storage is None or journal_file_storage is None:
            raise ValueError("Optuna journal storage is not available in this Optuna installation")
        path = _safe_child(root, "optuna_study.journal")
        return journal_storage(journal_file_storage(str(path))), {"storage_mode": "journal", "storage_path": str(path)}
    raise ValueError("Regime Optuna storage_mode must be one of: memory, sqlite, journal")


def _study_trials_payload(study: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in getattr(study, "trials", ()):
        rows.append(
            {
                "number": int(trial.number),
                "state": str(trial.state.name if hasattr(trial.state, "name") else trial.state),
                "params": to_jsonable(dict(trial.params)),
                "values": None if trial.values is None else [float(value) for value in trial.values],
                "user_attrs": to_jsonable(dict(trial.user_attrs)),
            }
        )
    return rows


def _markdown_summary(payload: Mapping[str, Any]) -> str:
    rows = [
        "# Regime Optuna Stub",
        "",
        f"- Study: `{payload['study_name']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Storage mode: `{payload['storage']['storage_mode']}`",
        f"- Trial count: `{payload['executed_trial_count']}`",
        f"- Promotion gates blocked: `{payload['all_promotion_gates_blocked']}`",
        f"- Runtime objective policy: {payload['runtime_objective_policy']}",
        "",
        "| Trial | Candidate | Clusterer | Values | Status | Gate |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    for trial in payload["trial_summaries"]:
        gate_status = "not_reported"
        if trial.get("single_trial"):
            gate_status = trial["single_trial"]["promotion_gate"]["status"]
        rows.append(
            f"| {trial['trial_number']} | {trial['params'].get('candidate_index')} | "
            f"`{trial['params'].get('clusterer_family')}` | `{trial['values']}` | "
            f"`{trial['status']}` | `{gate_status}` |"
        )
    rows.extend(
        [
            "",
            "This study is a bounded, single-process Optuna stub over synthetic foundation trials. "
            "It is sandbox-only and blocks production advancement.",
        ]
    )
    return "\n".join(rows)


@dataclass(frozen=True)
class OptunaStubStudyResult:
    summary: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    schema_version: int = OPTUNA_STUB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "summary", require_json_mapping(self.summary, field_name="optuna_stub summary"))
        object.__setattr__(self, "artifact_paths", require_json_mapping(self.artifact_paths, field_name="artifact_paths"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "summary": to_jsonable(self.summary),
            "artifact_paths": dict(self.artifact_paths),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptunaStubStudyResult":
        obj = require_json_object(payload, context="Regime OptunaStubStudyResult")
        return cls(
            schema_version=obj.get("schema_version", OPTUNA_STUB_SCHEMA_VERSION),
            summary=obj["summary"],
            artifact_paths=obj["artifact_paths"],
        )

    @classmethod
    def from_json(cls, text: str) -> "OptunaStubStudyResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime OptunaStubStudyResult JSON"))


def run_optuna_stub(
    manifest: StudyManifest | Mapping[str, Any] | None = None,
    *,
    dataset: pd.DataFrame | None = None,
    report_root: str | Path = DEFAULT_OPTUNA_REPORT_ROOT,
    study_name: str = "regime_foundation_optuna_stub",
    storage_mode: str = "memory",
    n_trials: int | None = None,
    seed: int = 17,
    write_outputs: bool = True,
    project_root: str | Path | None = None,
) -> OptunaStubStudyResult:
    root = validate_optuna_report_root(report_root, project_root=project_root)
    optuna = require_optuna()
    study = (
        default_optuna_study_manifest(report_root=root, seed=int(seed))
        if manifest is None
        else _manifest_with_report_root(manifest, report_root=root)
    )
    search_space = build_search_space(study)
    max_trials = int(n_trials if n_trials is not None else study.budget.get("max_trials", 1))
    max_trials = max(1, min(max_trials, int(len(search_space.candidate_trials))))
    storage, storage_metadata = _storage(optuna, root, storage_mode=storage_mode)
    sampler = optuna.samplers.GridSampler(
        {"candidate_index": list(range(int(len(search_space.candidate_trials))))},
        seed=int(seed),
    )
    frame = dataset.copy() if dataset is not None else synthetic_asset_state_fixture(periods=12)
    objective = RegimeOptunaObjective(
        study,
        report_root=_safe_child(root, "single_trials"),
        dataset=frame,
        seed=int(seed),
        write_outputs=write_outputs,
    )
    optuna_study = optuna.create_study(
        study_name=require_non_empty_string(study_name, field_name="Optuna study_name"),
        directions=list(OPTUNA_OBJECTIVE_DIRECTIONS),
        sampler=sampler,
        storage=storage,
        load_if_exists=storage is not None,
    )
    optuna_study.optimize(
        objective,
        n_trials=max_trials,
        n_jobs=1,
        timeout=float(study.budget.get("timeout_seconds", 60)),
        show_progress_bar=False,
    )
    trial_summaries = [summary.as_dict() for summary in objective.trial_summaries]
    summary_json_path = _safe_child(root, OPTUNA_STUB_SUMMARY_JSON)
    summary_md_path = _safe_child(root, OPTUNA_STUB_SUMMARY_MD)
    trial_summaries_json_path = _safe_child(root, OPTUNA_STUB_TRIAL_SUMMARIES_JSON)
    artifact_paths = {
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
        "trial_summaries_json": str(trial_summaries_json_path),
    }
    if storage_metadata.get("storage_path"):
        artifact_paths["storage_path"] = str(storage_metadata["storage_path"])
    artifact_paths.update(
        {
            f"trial_{trial['trial_number']:03d}_{name}": path
            for trial in trial_summaries
            for name, path in trial.get("artifact_paths", {}).items()
        }
    )
    all_gates_blocked = all(
        trial.get("single_trial")
        and trial["single_trial"]["promotion_gate"]["status"] == PROMOTION_STATUS_BLOCKED
        for trial in trial_summaries
    )
    try:
        pareto_trial_numbers = [int(trial.number) for trial in optuna_study.best_trials]
    except Exception:
        pareto_trial_numbers = []
    summary = {
        "schema_version": OPTUNA_STUB_SCHEMA_VERSION,
        "artifact_kind": OPTUNA_STUB_ARTIFACT_KIND,
        "status": "completed",
        "study_name": optuna_study.study_name,
        "seed": int(seed),
        "optuna_available": True,
        "storage": storage_metadata,
        "objective_specs": to_jsonable(OPTUNA_OBJECTIVE_SPECS),
        "metric_names": list(OPTUNA_OBJECTIVE_METRIC_NAMES),
        "directions": list(OPTUNA_OBJECTIVE_DIRECTIONS),
        "runtime_objective_policy": "runtime is diagnostic metadata only; raw wall-clock time is not an Optuna objective in the deterministic foundation stub",
        "study_manifest": study.as_dict(),
        "search_space": search_space.as_dict(),
        "requested_trial_count": int(max_trials),
        "executed_trial_count": int(len(trial_summaries)),
        "optuna_trials": _study_trials_payload(optuna_study),
        "trial_summaries": trial_summaries,
        "pareto_trial_numbers": pareto_trial_numbers,
        "all_promotion_gates_blocked": bool(all_gates_blocked),
        "artifact_paths": artifact_paths,
        "artifact_boundary": {
            "optuna_stub_only": True,
            "single_process": True,
            "synthetic_fixture": dataset is None,
            "production_outputs_written": False,
            "production_writes_enabled": False,
            "production_labels_written": False,
            "distributed_execution_enabled": False,
            "ray_integration_enabled": False,
            "heavy_optimization_budget": False,
            "runtime_objective_enabled": False,
        },
    }
    if write_outputs:
        _write_json(trial_summaries_json_path, {"trial_summaries": trial_summaries})
        _write_json(summary_json_path, summary)
        _write_markdown(summary_md_path, _markdown_summary(summary))
    return OptunaStubStudyResult(summary=summary, artifact_paths=artifact_paths)


__all__ = [
    "DEFAULT_OPTUNA_REPORT_ROOT",
    "OPTUNA_AVAILABLE",
    "OPTUNA_IMPORT_ERROR",
    "OPTUNA_STUB_ARTIFACT_KIND",
    "OPTUNA_STUB_SCHEMA_VERSION",
    "OPTUNA_STUB_SUMMARY_JSON",
    "OPTUNA_STUB_SUMMARY_MD",
    "OPTUNA_STUB_TRIAL_SUMMARIES_JSON",
    "OptunaStubStudyResult",
    "OptunaUnavailableError",
    "default_optuna_study_manifest",
    "require_optuna",
    "run_optuna_stub",
    "validate_optuna_report_root",
]
