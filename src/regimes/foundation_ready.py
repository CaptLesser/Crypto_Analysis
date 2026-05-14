from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    DatasetWindowSpec,
    RegimeAxis,
    RegimeBand,
    RegimeClassification,
    RegimeLayer,
    RunStatus,
    StudyKey,
    TrialArtifacts,
    TrialResultEnvelope,
    TrialSpec,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
)
from src.regimes.core.paths import (
    default_foundation_report_root,
    is_relative_to,
    require_foundation_report_root,
    resolve_project_root,
)
from src.regimes.core.promotion_gate import PROMOTION_STATUS_BLOCKED
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.studies.fixtures import synthetic_asset_state_fixture
from src.regimes.studies.foundation_smoke import run_foundation_smoke
from src.regimes.studies.manifest import StudyManifest
from src.regimes.studies.optuna_runner import OPTUNA_AVAILABLE, OptunaUnavailableError, run_optuna_stub
from src.regimes.studies.single_trial import run_single_trial
from src.regimes.studies.small_panel_benchmark import run_small_panel_benchmark


FOUNDATION_READY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
FOUNDATION_READY_ARTIFACT_KIND = "regime_foundation_ready_closeout"
DEFAULT_FOUNDATION_READY_REPORT_ROOT = default_foundation_report_root()
FOUNDATION_INDEX_JSON = "foundation_index.json"
FOUNDATION_READY_MD = "foundation_ready.md"


def validate_foundation_report_root(report_root: str | Path, *, project_root: str | Path | None = None) -> Path:
    return require_foundation_report_root(
        report_root,
        project_root=project_root,
        required_suffix=("reports", "regimes", "foundation"),
        error_prefix="Regime foundation readiness report root",
    )


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve()
    if not is_relative_to(candidate, root):
        raise ValueError("Regime foundation readiness artifact path must stay under report_root")
    return candidate


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _check_payload(
    name: str,
    *,
    status: str,
    required: bool = True,
    details: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "status": status,
        "required": bool(required),
        "details": to_jsonable(details or {}),
        "artifact_paths": dict(artifact_paths or {}),
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
    }
    return payload


def _assert_artifact_paths_under_root(paths: Mapping[str, str], root: Path) -> None:
    for name, raw_path in paths.items():
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = resolve_project_root() / path
        if not is_relative_to(path.resolve(), root):
            raise ValueError(f"Regime foundation readiness artifact path {name!r} escaped report_root")


def _core_contract_round_trip(root: Path, *, write_outputs: bool) -> dict[str, Any]:
    study_key = StudyKey(
        study_id="foundation_ready_contract_round_trip",
        layer=RegimeLayer.ASSET_STATE,
        axis=RegimeAxis.TREND,
        band=RegimeBand.MICRO,
        classification=RegimeClassification.SANDBOX,
    )
    dataset_window = DatasetWindowSpec(
        study_key=study_key,
        dataset_id="foundation_ready_synthetic_window",
        start_ts="2026-01-01T00:00:00Z",
        end_ts="2026-01-01T03:30:00Z",
        interval_minutes=30,
        source_artifacts=("memory://regimes/foundation-ready",),
        feature_columns=("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"),
        asset="XBTUSD",
        metadata={"validation": "foundation_ready"},
    )
    trial = TrialSpec(
        study_key=study_key,
        trial_id="foundation_ready_contract_trial",
        dataset_window=dataset_window,
        feature_family="asset_state_trend_metadata_only",
        preprocessing_family="robust_scale",
        clusterer_family="kmeans",
        assignment_policy="native_predict",
        hyperparameters={"n_clusters": 2, "random_state": 17},
        random_seed=17,
    )
    artifacts = TrialArtifacts(
        study_key=study_key,
        trial_id=trial.trial_id,
        artifact_paths={"contract_round_trip": str(_safe_child(root, "ready", "contracts_round_trip.json"))},
        production_outputs_written=False,
    )
    envelope = TrialResultEnvelope(
        study_key=study_key,
        trial=trial,
        artifacts=artifacts,
        status=RunStatus.SUCCEEDED,
        metrics={"round_trip": True},
    )
    models: Sequence[Any] = (study_key, dataset_window, trial, artifacts, envelope)
    decoded = [type(model).from_json(model.to_json()).as_dict() for model in models]
    if decoded != [model.as_dict() for model in models]:
        raise ValueError("Regime foundation core contract JSON round-trip mismatch")
    artifact_path = Path(artifacts.artifact_paths["contract_round_trip"])
    if write_outputs:
        _write_json(
            artifact_path,
            {
                "schema_version": FOUNDATION_READY_SCHEMA_VERSION,
                "artifact_kind": "regime_foundation_ready_contract_round_trip",
                "round_tripped_models": [type(model).__name__ for model in models],
                "envelope": envelope.as_dict(),
            },
        )
    return {
        "round_tripped_models": [type(model).__name__ for model in models],
        "study_key": study_key.as_dict(),
        "artifact_paths": artifacts.as_dict()["artifact_paths"],
    }


def _single_trial_manifest(root: Path, *, seed: int) -> StudyManifest:
    return StudyManifest(
        study_id="foundation_ready_single_trial",
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
        report_root=_safe_child(root, "ready", "single_trial"),
        metadata={
            "purpose": "foundation_ready_validator",
            "production_outputs_written": False,
        },
    )


def _run_core_contract_check(root: Path, *, write_outputs: bool) -> dict[str, Any]:
    try:
        details = _core_contract_round_trip(root, write_outputs=write_outputs)
        return _check_payload("core_contracts_round_trip", status="passed", details=details, artifact_paths=details["artifact_paths"])
    except Exception as exc:
        return _check_payload("core_contracts_round_trip", status="failed", details={}, error=exc)


def _run_single_trial_check(root: Path, *, seed: int, write_outputs: bool, project_root: Path) -> dict[str, Any]:
    try:
        result = run_single_trial(
            _single_trial_manifest(root, seed=seed),
            dataset=synthetic_asset_state_fixture(periods=12),
            trial_id="foundation_ready_single_trial",
            write_outputs=write_outputs,
            project_root=project_root,
        )
        payload = result.as_dict()
        paths = payload["artifact_paths"]
        _assert_artifact_paths_under_root(paths, root)
        passed = (
            payload["fit_result"]["status"] == "fitted"
            and payload["assignment_result"]["status"] == "assigned"
            and payload["promotion_gate"]["status"] == PROMOTION_STATUS_BLOCKED
        )
        return _check_payload(
            "single_trial_runner",
            status="passed" if passed else "failed",
            details={
                "trial_id": payload["trial_id"],
                "clusterer_family": payload["clusterer_family"],
                "promotion_gate_status": payload["promotion_gate"]["status"],
                "scoreboard_sections": sorted(payload["scoreboard"]["sections"].keys()),
            },
            artifact_paths=paths,
        )
    except Exception as exc:
        return _check_payload("single_trial_runner", status="failed", error=exc)


def _run_foundation_smoke_check(root: Path, *, seed: int, project_root: Path, write_outputs: bool) -> dict[str, Any]:
    try:
        result = run_foundation_smoke(
            report_root=_safe_child(root, "smoke"),
            run_id="foundation_ready_smoke",
            seed=int(seed),
            project_root=project_root,
            write_outputs=write_outputs,
        )
        payload = result.summary
        _assert_artifact_paths_under_root(result.artifact_paths, root)
        passed = payload["status"] == "completed" and payload["promotion_gate"]["status"] == PROMOTION_STATUS_BLOCKED
        return _check_payload(
            "foundation_smoke",
            status="passed" if passed else "failed",
            details={
                "run_id": payload["run_id"],
                "promotion_gate_status": payload["promotion_gate"]["status"],
                "feature_cache_status": payload["feature_cache_manifest"]["status"],
            },
            artifact_paths=result.artifact_paths,
        )
    except Exception as exc:
        return _check_payload("foundation_smoke", status="failed", error=exc)


def _run_optuna_check(root: Path, *, seed: int, optuna_trials: int, project_root: Path, write_outputs: bool) -> dict[str, Any]:
    if not OPTUNA_AVAILABLE:
        return _check_payload(
            "optuna_stub",
            status="skipped_dependency_missing",
            required=False,
            details={"dependency": "optuna", "message": "Optuna is not available in this environment"},
        )
    try:
        result = run_optuna_stub(
            report_root=_safe_child(root, "optuna"),
            n_trials=int(optuna_trials),
            seed=int(seed),
            storage_mode="memory",
            project_root=project_root,
            write_outputs=write_outputs,
        )
        payload = result.summary
        _assert_artifact_paths_under_root(result.artifact_paths, root)
        passed = payload["status"] == "completed" and payload["all_promotion_gates_blocked"] is True
        return _check_payload(
            "optuna_stub",
            status="passed" if passed else "failed",
            required=True,
            details={
                "executed_trial_count": payload["executed_trial_count"],
                "directions": payload["directions"],
                "all_promotion_gates_blocked": payload["all_promotion_gates_blocked"],
            },
            artifact_paths=result.artifact_paths,
        )
    except OptunaUnavailableError as exc:
        return _check_payload("optuna_stub", status="skipped_dependency_missing", required=False, error=exc)
    except Exception as exc:
        return _check_payload("optuna_stub", status="failed", required=True, error=exc)


def _run_small_panel_check(root: Path, *, seed: int, project_root: Path, write_outputs: bool) -> dict[str, Any]:
    try:
        result = run_small_panel_benchmark(
            report_root=_safe_child(root, "benchmarks"),
            run_id="foundation_ready_small_panel",
            seed=int(seed),
            project_root=project_root,
            write_outputs=write_outputs,
        )
        payload = result.summary
        _assert_artifact_paths_under_root(result.artifact_paths, root)
        passed = payload["status"] == "completed" and payload["all_promotion_gates_blocked"] is True
        return _check_payload(
            "small_panel_benchmark",
            status="passed" if passed else "failed",
            details={
                "candidate_families": payload["candidate_families"],
                "family_count": len(payload["family_results"]),
                "all_promotion_gates_blocked": payload["all_promotion_gates_blocked"],
            },
            artifact_paths=result.artifact_paths,
        )
    except Exception as exc:
        return _check_payload("small_panel_benchmark", status="failed", error=exc)


def _created_modules() -> list[dict[str, str]]:
    return [
        {"path": "src/regimes/core/contracts.py", "purpose": "Canonical typed contracts and JSON round-trip models."},
        {"path": "src/regimes/core/manifests.py", "purpose": "Report-root constrained manifest load/save helpers."},
        {"path": "src/regimes/core/feature_registry.py", "purpose": "Feature family metadata registry."},
        {"path": "src/regimes/core/preprocessing.py", "purpose": "Train-window-safe preprocessing and filtering."},
        {"path": "src/regimes/core/clusterer_base.py", "purpose": "Clusterer interface and normalized result payloads."},
        {"path": "src/regimes/core/clusterer_adapters.py", "purpose": "Tier A clusterer adapters."},
        {"path": "src/regimes/core/scoreboard.py", "purpose": "Validity, degeneracy, runtime, stability, and economic scoring envelope."},
        {"path": "src/regimes/core/promotion_gate.py", "purpose": "Fail-closed promotion gate."},
        {"path": "src/regimes/core/feature_cache.py", "purpose": "Feature-cache identity and reuse decisions."},
        {"path": "src/regimes/core/flat_asset_policy.py", "purpose": "Flat and pegged asset preflight diagnostics."},
        {"path": "src/regimes/studies/single_trial.py", "purpose": "Deterministic single-trial foundation runner."},
        {"path": "src/regimes/studies/foundation_smoke.py", "purpose": "Bounded foundation smoke entrypoint."},
        {"path": "src/regimes/studies/optuna_runner.py", "purpose": "Bounded Optuna multi-trial stub."},
        {"path": "src/regimes/studies/small_panel_benchmark.py", "purpose": "Deterministic small-panel benchmark."},
        {"path": "src/regimes/foundation_ready.py", "purpose": "Foundation readiness validator and closeout index writer."},
    ]


def _entrypoints() -> list[dict[str, str]]:
    return [
        {
            "callable": "src.regimes.foundation_ready.validate_regimes_foundation_ready",
            "command": "python -c \"from src.regimes.foundation_ready import validate_regimes_foundation_ready; validate_regimes_foundation_ready()\"",
        },
        {"callable": "src.regimes.studies.single_trial.run_single_trial", "command": "pytest tests/test_regime_single_trial_runner.py -q"},
        {"callable": "src.regimes.studies.foundation_smoke.run_foundation_smoke", "command": "pytest tests/test_regime_foundation_smoke.py -q"},
        {"callable": "src.regimes.studies.optuna_runner.run_optuna_stub", "command": "pytest tests/test_regime_optuna_stub.py -q"},
        {
            "callable": "src.regimes.studies.small_panel_benchmark.run_small_panel_benchmark",
            "command": "pytest tests/test_regime_small_panel_benchmark.py -q",
        },
    ]


def _output_locations(root: Path) -> dict[str, str]:
    return {
        "foundation_index_json": str(_safe_child(root, FOUNDATION_INDEX_JSON)),
        "foundation_ready_markdown": str(_safe_child(root, FOUNDATION_READY_MD)),
        "single_trial_root": str(_safe_child(root, "ready", "single_trial")),
        "foundation_smoke_root": str(_safe_child(root, "smoke")),
        "optuna_stub_root": str(_safe_child(root, "optuna")),
        "small_panel_benchmark_root": str(_safe_child(root, "benchmarks")),
    }


def _ready_status(checks: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    failed = [check for check in checks if check["required"] and check["status"] != "passed"]
    if failed:
        return "failed", False
    return "ready", True


def _markdown_summary(payload: Mapping[str, Any]) -> str:
    rows = [
        "# Regime Foundation Ready",
        "",
        f"- Status: `{payload['status']}`",
        f"- Ready: `{str(bool(payload['ready'])).lower()}`",
        f"- Seed: `{payload['seed']}`",
        f"- Report root: `{payload['report_root']}`",
        f"- Write outputs: `{str(bool(payload['artifact_boundary']['write_outputs'])).lower()}`",
        f"- Child artifacts written: `{str(bool(payload['artifact_boundary']['child_artifacts_written'])).lower()}`",
        "",
        "## Validation Checks",
        "",
        "| Check | Status | Required |",
        "| --- | --- | --- |",
    ]
    for check in payload["validation_checks"]:
        rows.append(f"| `{check['name']}` | `{check['status']}` | `{check['required']}` |")
    rows.extend(
        [
            "",
            "## Foundation Surfaces",
            "",
            "- Canonical contracts: `src/regimes/core/contracts.py`.",
            "- Canonical study manifest and runner: `src/regimes/studies/manifest.py` and `src/regimes/studies/single_trial.py`.",
            "- Canonical feature cache: `src/regimes/core/feature_cache.py`.",
            "- Canonical scoreboard: `src/regimes/core/scoreboard.py`.",
            "- Canonical smoke path: `src/regimes/studies/foundation_smoke.py`.",
            "- Legacy compatibility modules remain importable and explicitly documented as compatibility surfaces.",
            "",
            "## Intentionally Out Of Scope",
            "",
            "- Production promotion, production label generation changes, full benchmark campaigns, distributed execution, runtime-profile tuning, market-state execution, and relative-state production execution.",
            "",
            "## Later Safe Passes",
            "",
            "- Polish report formatting, broaden deterministic fixtures, add larger studies, tune adapter hyperparameters, profile runtime, and integrate production promotion only after separate approval gates.",
        ]
    )
    return "\n".join(rows)


@dataclass(frozen=True)
class FoundationReadyResult:
    summary: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    schema_version: int = FOUNDATION_READY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "summary", require_json_mapping(self.summary, field_name="foundation_ready summary"))
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoundationReadyResult":
        obj = require_json_object(payload, context="Regime FoundationReadyResult")
        return cls(
            schema_version=obj.get("schema_version", FOUNDATION_READY_SCHEMA_VERSION),
            summary=obj["summary"],
            artifact_paths=obj["artifact_paths"],
        )

    @classmethod
    def from_json(cls, text: str) -> "FoundationReadyResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FoundationReadyResult JSON"))


def validate_regimes_foundation_ready(
    *,
    report_root: str | Path = DEFAULT_FOUNDATION_READY_REPORT_ROOT,
    run_id: str = "foundation_ready",
    seed: int = 17,
    optuna_trials: int = 1,
    write_outputs: bool = True,
    project_root: str | Path | None = None,
) -> FoundationReadyResult:
    root = validate_foundation_report_root(report_root, project_root=project_root)
    project = resolve_project_root(project_root)
    run_token = require_non_empty_string(run_id, field_name="foundation readiness run_id")
    checks = [
        _run_core_contract_check(root, write_outputs=write_outputs),
        _run_single_trial_check(root, seed=int(seed), write_outputs=write_outputs, project_root=project),
        _run_foundation_smoke_check(root, seed=int(seed), project_root=project, write_outputs=write_outputs),
        _run_optuna_check(root, seed=int(seed), optuna_trials=int(optuna_trials), project_root=project, write_outputs=write_outputs),
        _run_small_panel_check(root, seed=int(seed), project_root=project, write_outputs=write_outputs),
    ]
    status, ready = _ready_status(checks)
    artifact_paths = _output_locations(root)
    summary = {
        "schema_version": FOUNDATION_READY_SCHEMA_VERSION,
        "artifact_kind": FOUNDATION_READY_ARTIFACT_KIND,
        "status": status,
        "ready": bool(ready),
        "run_id": run_token,
        "seed": int(seed),
        "report_root": str(root),
        "validation_checks": checks,
        "created_modules": _created_modules(),
        "entrypoints": _entrypoints(),
        "output_artifact_locations": artifact_paths,
        "reproduction": {
            "primary_callable": "src.regimes.foundation_ready.validate_regimes_foundation_ready",
            "primary_command": "python -c \"from src.regimes.foundation_ready import validate_regimes_foundation_ready; validate_regimes_foundation_ready()\"",
            "test_command": "pytest tests/test_regime_foundation_ready.py -q",
        },
        "artifact_boundary": {
            "foundation_closeout_only": True,
            "bounded_validation": True,
            "report_root_only": True,
            "write_outputs": bool(write_outputs),
            "child_artifacts_written": bool(write_outputs),
            "production_outputs_written": False,
            "production_writes_enabled": False,
            "production_labels_written": False,
            "full_benchmark_campaign": False,
            "release_or_publish_wiring": False,
            "market_state_production_execution": False,
            "relative_state_production_execution": False,
        },
        "intentionally_out_of_scope": [
            "production promotion",
            "full benchmark campaigns",
            "release or publish wiring",
            "runtime-profile tuning",
            "market-state production execution",
            "relative-state production execution",
        ],
        "safe_later_passes": [
            "report polish",
            "larger deterministic study fixtures",
            "adapter hyperparameter tuning",
            "runtime profiling",
            "expanded benchmark campaigns",
            "separate production promotion integration",
        ],
    }
    if write_outputs:
        _write_json(Path(artifact_paths["foundation_index_json"]), summary)
        _write_markdown(Path(artifact_paths["foundation_ready_markdown"]), _markdown_summary(summary))
    return FoundationReadyResult(summary=summary, artifact_paths=artifact_paths)


__all__ = [
    "DEFAULT_FOUNDATION_READY_REPORT_ROOT",
    "FOUNDATION_INDEX_JSON",
    "FOUNDATION_READY_ARTIFACT_KIND",
    "FOUNDATION_READY_MD",
    "FOUNDATION_READY_SCHEMA_VERSION",
    "FoundationReadyResult",
    "validate_foundation_report_root",
    "validate_regimes_foundation_ready",
]
