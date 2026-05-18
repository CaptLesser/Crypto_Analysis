from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.clusterability_preflight import (
    AXIS_ACTIVITY_COLUMNS,
    AXIS_FEATURES,
    AXIS_MOVEMENT_COLUMNS,
    BAND_ORDER,
    CLUSTERABILITY_ARTIFACT_KIND,
    COHORT_ORDER,
    PATHWAY_ASSET_STATE,
    PANEL_STRATIFIED,
    build_stratified_feature_frame,
    validate_clusterability_write_root,
)
from src.regimes.contracts import REGIME_AXIS_ORDER
from src.regimes.core.clusterer_adapters import FIT_STATUS_DEPENDENCY_UNAVAILABLE, FIT_STATUS_FITTED
from src.regimes.core.clusterer_registry import build_clusterer_adapter, clusterer_capabilities_registry
from src.regimes.core.feature_preprocessing import preprocessing_registry
from src.regimes.core.preprocessing import fit_preprocessing_pipeline

try:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    _HAS_SKLEARN_METRICS = True
except Exception:  # pragma: no cover - exercised only in minimal dependency environments
    calinski_harabasz_score = None  # type: ignore[assignment]
    davies_bouldin_score = None  # type: ignore[assignment]
    silhouette_score = None  # type: ignore[assignment]
    _HAS_SKLEARN_METRICS = False


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_MATRIX_ARTIFACT_KIND = "regime_asset_state_benchmark_matrix"
BENCHMARK_RESULTS_ARTIFACT_KIND = "regime_asset_state_benchmark_results"
BENCHMARK_SCOREBOARD_ARTIFACT_KIND = "regime_asset_state_benchmark_scoreboard"

MATRIX_FILENAME = "asset_state_benchmark_matrix.json"
RESULTS_FILENAME = "asset_state_benchmark_results.jsonl"
SCOREBOARD_JSON_FILENAME = "asset_state_scoreboard.json"
SCOREBOARD_MD_FILENAME = "asset_state_scoreboard.md"
PREFLIGHT_FILENAME = "clusterability_preflight.json"

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_SKIPPED_DEPENDENCY_UNAVAILABLE = "skipped_dependency_unavailable"

LABEL_BENCHMARK_CANDIDATE = "benchmark_candidate"
LABEL_DEGENERATE = "degenerate_single_state"
LABEL_NOISE_DOMINATED = "noise_dominated"
LABEL_SKIPPED = "skipped"
LABEL_FAILED = "failed"

OPTUNA_HELPER = "optuna_1trial_helper"
NO_OPTIMIZER = "none"
HDBSCAN_DEPENDENCY = "hdbscan"
OPTUNA_DEPENDENCY = "optuna"

_OPTIONAL_DEPENDENCY_OVERRIDES: dict[str, bool] = {}


@dataclass(frozen=True)
class BenchmarkMatrixResult:
    payload: Mapping[str, Any]
    artifact_paths: Mapping[str, str]


@dataclass(frozen=True)
class BenchmarkRunResult:
    scoreboard: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    artifact_paths: Mapping[str, str]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "\n".join(json.dumps(_jsonable(row), sort_keys=True) for row in rows)
    _write_text(path, text)


def _optional_dependency_available(name: str) -> bool:
    key = str(name).strip().lower()
    if key in _OPTIONAL_DEPENDENCY_OVERRIDES:
        return bool(_OPTIONAL_DEPENDENCY_OVERRIDES[key])
    return importlib.util.find_spec(key) is not None


def _ordered_unique(values: Sequence[object]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return tuple(out)


def _normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(sorted({int(seed) for seed in seeds})) or (11,)
    return selected


def _rank(value: object, order: Sequence[str]) -> tuple[int, str]:
    text = str(value)
    try:
        return (list(order).index(text), text)
    except ValueError:
        return (len(order), text)


def _feature_columns(axis: str, feature_family: str) -> tuple[str, ...]:
    axis_key = str(axis)
    family = str(feature_family)
    if family == "compact_axis":
        columns = AXIS_MOVEMENT_COLUMNS[axis_key] or AXIS_FEATURES[axis_key][:2]
    elif family == "full_axis":
        columns = AXIS_FEATURES[axis_key]
    elif family == "movement_activity":
        columns = (*AXIS_MOVEMENT_COLUMNS[axis_key], *AXIS_ACTIVITY_COLUMNS[axis_key])
        if not columns:
            columns = AXIS_FEATURES[axis_key][:2]
    else:
        raise ValueError(f"unsupported asset-state feature family: {feature_family!r}")
    return tuple(dict.fromkeys(str(column) for column in columns))


def _config_templates(axis: str) -> tuple[dict[str, Any], ...]:
    templates: tuple[dict[str, Any], ...] = (
        {
            "config_id": "cfg001_compact_noop_kmeans",
            "feature_family": "compact_axis",
            "preprocessing": "noop",
            "clusterer": "kmeans",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_clusters": 3, "n_init": 10},
        },
        {
            "config_id": "cfg002_compact_robust_kmeans",
            "feature_family": "compact_axis",
            "preprocessing": "robust_scale",
            "clusterer": "kmeans",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_clusters": 3, "n_init": 10},
        },
        {
            "config_id": "cfg003_full_standard_kmeans",
            "feature_family": "full_axis",
            "preprocessing": "standard_scale",
            "clusterer": "kmeans",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_clusters": 3, "n_init": 10},
        },
        {
            "config_id": "cfg004_full_robust_minibatch",
            "feature_family": "full_axis",
            "preprocessing": "robust_scale",
            "clusterer": "minibatch_kmeans",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_clusters": 3, "n_init": 5, "batch_size": 32},
        },
        {
            "config_id": "cfg005_full_standard_gmm",
            "feature_family": "full_axis",
            "preprocessing": "standard_scale",
            "clusterer": "gaussian_mixture",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_components": 3, "covariance_type": "full"},
        },
        {
            "config_id": "cfg006_compact_robust_gmm_diag",
            "feature_family": "compact_axis",
            "preprocessing": "robust_scale",
            "clusterer": "gaussian_mixture",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_components": 3, "covariance_type": "diag"},
        },
        {
            "config_id": "cfg007_movement_robust_agglomerative",
            "feature_family": "movement_activity",
            "preprocessing": "robust_scale",
            "clusterer": "agglomerative",
            "optimizer": NO_OPTIMIZER,
            "hyperparameters": {"n_clusters": 3},
        },
        {
            "config_id": "cfg008_movement_robust_hdbscan",
            "feature_family": "movement_activity",
            "preprocessing": "robust_scale",
            "clusterer": "hdbscan",
            "optimizer": NO_OPTIMIZER,
            "optional_dependency": HDBSCAN_DEPENDENCY,
            "hyperparameters": {"min_cluster_size": 5, "min_samples": 1, "allow_single_cluster": True},
        },
        {
            "config_id": "cfg009_full_robust_kmeans_optuna_1trial",
            "feature_family": "full_axis",
            "preprocessing": "robust_scale",
            "clusterer": "kmeans",
            "optimizer": OPTUNA_HELPER,
            "optional_dependency": OPTUNA_DEPENDENCY,
            "optimizer_budget": {"n_trials": 1, "n_jobs": 1},
            "hyperparameters": {"n_init": 10},
        },
    )
    rendered: list[dict[str, Any]] = []
    for template in templates:
        item = dict(template)
        item["feature_columns"] = list(_feature_columns(axis, str(item["feature_family"])))
        item["dependency_available"] = _optional_dependency_available(str(item["optional_dependency"])) if item.get("optional_dependency") else True
        rendered.append(item)
    return tuple(rendered)


def _load_preflight_payload(write_root: Path, preflight_path: str | Path | None) -> Mapping[str, Any]:
    path = Path(preflight_path) if preflight_path is not None else write_root / PREFLIGHT_FILENAME
    if not path.is_absolute():
        path = Path.cwd() / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("clusterability preflight payload must be a JSON object")
    return payload


def _eligible_preflight_rows(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("clusterability preflight payload must expose a rows array")
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("final_label")) != "clusterable":
            continue
        if row.get("clusterable_candidate") is not True:
            continue
        item = {
            "cohort": str(row.get("cohort")),
            "asset": str(row.get("asset")),
            "axis": str(row.get("axis")),
            "band": str(row.get("band")),
            "preflight_status": str(row.get("status", "unspecified")),
            "preflight_confidence_score_mean": row.get("confidence_score_mean"),
            "preflight_finite_row_count_min": row.get("finite_row_count_min"),
        }
        eligible.append(item)
    eligible.sort(
        key=lambda item: (
            _rank(item["cohort"], COHORT_ORDER),
            str(item["asset"]),
            _rank(item["axis"], REGIME_AXIS_ORDER),
            _rank(item["band"], BAND_ORDER),
        )
    )
    return tuple(eligible)


def _artifact_boundary() -> dict[str, Any]:
    return {
        "pathway": PATHWAY_ASSET_STATE,
        "panel": PANEL_STRATIFIED,
        "stratified_panel_only": True,
        "production_labels_written": False,
        "production_outputs_written": False,
        "promotion_allowed": False,
        "optuna_bounded_helper_only": True,
        "optuna_max_trials": 1,
        "n_jobs": 1,
    }


def build_asset_state_benchmark_matrix(
    *,
    write_root: str | Path,
    preflight_path: str | Path | None = None,
    preflight_payload: Mapping[str, Any] | None = None,
    no_write: bool = False,
    project_root: str | Path | None = None,
) -> BenchmarkMatrixResult:
    root = Path(write_root) if no_write else validate_clusterability_write_root(write_root, project_root=project_root)
    payload = preflight_payload if preflight_payload is not None else _load_preflight_payload(root, preflight_path)
    eligible_rows = _eligible_preflight_rows(payload)
    cells: list[dict[str, Any]] = []
    for row in eligible_rows:
        cell_id = f"{row['cohort']}__{row['asset']}__{row['axis']}__{row['band']}"
        cells.append(
            {
                "cell_id": cell_id,
                "pathway": PATHWAY_ASSET_STATE,
                "panel": PANEL_STRATIFIED,
                "cohort": row["cohort"],
                "asset": row["asset"],
                "axis": row["axis"],
                "band": row["band"],
                "preflight": {
                    "source_artifact_kind": CLUSTERABILITY_ARTIFACT_KIND,
                    "status": row["preflight_status"],
                    "confidence_score_mean": row["preflight_confidence_score_mean"],
                    "finite_row_count_min": row["preflight_finite_row_count_min"],
                    "final_label": "clusterable",
                },
                "configs": list(_config_templates(str(row["axis"]))),
            }
        )
    capability_registry = clusterer_capabilities_registry()
    preprocessor_registry = preprocessing_registry()
    matrix = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": BENCHMARK_MATRIX_ARTIFACT_KIND,
        "created_at_utc": _now_utc(),
        "source_preflight_artifact_kind": payload.get("artifact_kind", "unspecified"),
        "source_preflight_path": str(preflight_path or Path(root) / PREFLIGHT_FILENAME),
        "artifact_boundary": _artifact_boundary(),
        "pathway": PATHWAY_ASSET_STATE,
        "panel": PANEL_STRATIFIED,
        "row_order": ["cohort_order", "asset", "axis_order", "band_order"],
        "eligible_asset_axis_band_count": len(cells),
        "eligible_assets": [
            {"cohort": cohort, "asset": asset}
            for cohort, asset in sorted(
                {(str(row["cohort"]), str(row["asset"])) for row in eligible_rows},
                key=lambda item: (_rank(item[0], COHORT_ORDER), item[1]),
            )
        ],
        "bands": [band for band in BAND_ORDER if any(cell["band"] == band for cell in cells)],
        "feature_families": {
            "compact_axis": "Axis movement subset for bounded low-dimensional checks.",
            "full_axis": "Canonical full feature set for the requested asset-state axis.",
            "movement_activity": "Movement columns plus activity columns when available for the axis.",
        },
        "preprocessing": {
            name: preprocessor_registry[name].as_dict()
            for name in ("noop", "standard_scale", "robust_scale")
            if name in preprocessor_registry
        },
        "clusterers": {
            name: capability_registry[name].as_dict()
            for name in ("kmeans", "minibatch_kmeans", "gaussian_mixture", "agglomerative", "hdbscan")
            if name in capability_registry
        },
        "optional_dependencies": {
            HDBSCAN_DEPENDENCY: _optional_dependency_available(HDBSCAN_DEPENDENCY),
            OPTUNA_DEPENDENCY: _optional_dependency_available(OPTUNA_DEPENDENCY),
        },
        "max_configs_per_cell_required_by_sprint": 12,
        "configs_per_cell": max((len(cell["configs"]) for cell in cells), default=0),
        "cells": cells,
    }
    artifact_paths: dict[str, str] = {}
    if not no_write:
        matrix_path = root / MATRIX_FILENAME
        _write_json(matrix_path, matrix)
        artifact_paths["matrix_json"] = str(matrix_path)
    return BenchmarkMatrixResult(payload=matrix, artifact_paths=artifact_paths)


def _bounded_hyperparameters(clusterer: str, hyperparameters: Mapping[str, Any], *, seed: int, row_count: int) -> dict[str, Any]:
    params = dict(hyperparameters)
    max_states = max(2, min(3, max(int(row_count) - 1, 2)))
    if clusterer in {"kmeans", "minibatch_kmeans", "agglomerative"}:
        params["n_clusters"] = max(2, min(int(params.get("n_clusters", max_states)), max_states))
    if clusterer == "minibatch_kmeans":
        params["batch_size"] = max(8, min(int(params.get("batch_size", 32)), max(int(row_count), 8)))
    if clusterer == "gaussian_mixture":
        params["n_components"] = max(2, min(int(params.get("n_components", max_states)), max_states))
    if clusterer == "hdbscan":
        params["min_cluster_size"] = max(2, min(int(params.get("min_cluster_size", 5)), max(2, int(row_count))))
        params["min_samples"] = max(1, min(int(params.get("min_samples", 1)), int(params["min_cluster_size"])))
    if clusterer in {"kmeans", "minibatch_kmeans", "gaussian_mixture"}:
        params["random_state"] = int(seed)
    return params


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _cluster_metrics(x: np.ndarray, labels: Sequence[int], *, runtime_sec: float) -> dict[str, Any]:
    arr = np.asarray(labels, dtype=int)
    row_count = int(arr.size)
    label_counts = Counter(int(value) for value in arr.tolist())
    non_noise_mask = arr != -1
    non_noise_labels = arr[non_noise_mask]
    effective_state_count = int(len(set(int(value) for value in non_noise_labels.tolist())))
    noise_count = int(np.sum(arr == -1))
    noise_share = float(noise_count / row_count) if row_count else None
    silhouette = None
    calinski = None
    davies = None
    if (
        _HAS_SKLEARN_METRICS
        and row_count
        and effective_state_count >= 2
        and int(np.sum(non_noise_mask)) > effective_state_count
    ):
        try:
            x_score = np.asarray(x, dtype=float)[non_noise_mask]
            y_score = non_noise_labels
            silhouette = _finite_float(silhouette_score(x_score, y_score))  # type: ignore[misc]
            calinski = _finite_float(calinski_harabasz_score(x_score, y_score))  # type: ignore[misc]
            davies = _finite_float(davies_bouldin_score(x_score, y_score))  # type: ignore[misc]
        except Exception:
            silhouette = None
            calinski = None
            davies = None
    quality_base = silhouette if silhouette is not None else (-0.25 if effective_state_count < 2 else 0.0)
    quality_score = float(quality_base - (noise_share or 0.0) * 0.25)
    return {
        "row_count": row_count,
        "feature_count": int(np.asarray(x).shape[1]) if np.asarray(x).ndim == 2 else 0,
        "effective_state_count": effective_state_count,
        "noise_count": noise_count,
        "noise_share": noise_share,
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "quality_score": quality_score,
        "runtime_sec": runtime_sec,
    }


def _benchmark_label(status: str, metrics: Mapping[str, Any]) -> str:
    if status == STATUS_SKIPPED_DEPENDENCY_UNAVAILABLE:
        return LABEL_SKIPPED
    if status != STATUS_SUCCEEDED:
        return LABEL_FAILED
    if int(metrics.get("effective_state_count") or 0) < 2:
        return LABEL_DEGENERATE
    if float(metrics.get("noise_share") or 0.0) > 0.65:
        return LABEL_NOISE_DOMINATED
    return LABEL_BENCHMARK_CANDIDATE


def _skip_result(
    *,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    dependency: str,
    started: float,
) -> dict[str, Any]:
    metrics = {
        "row_count": 0,
        "feature_count": 0,
        "effective_state_count": 0,
        "noise_count": 0,
        "noise_share": None,
        "label_counts": {},
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "quality_score": None,
        "runtime_sec": float(time.monotonic() - started),
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": BENCHMARK_RESULTS_ARTIFACT_KIND,
        "status": STATUS_SKIPPED_DEPENDENCY_UNAVAILABLE,
        "final_label": LABEL_SKIPPED,
        "skip_reason": f"{dependency}_unavailable",
        "seed": int(seed),
        "cell": _cell_identity(cell),
        "config": _config_identity(config),
        "labels": [],
        "metrics": metrics,
        "failure_metadata": {"reason_code": "dependency_unavailable", "dependency": dependency},
    }


def _cell_identity(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": str(cell.get("cell_id")),
        "cohort": str(cell.get("cohort")),
        "asset": str(cell.get("asset")),
        "axis": str(cell.get("axis")),
        "band": str(cell.get("band")),
        "pathway": PATHWAY_ASSET_STATE,
        "panel": PANEL_STRATIFIED,
    }


def _config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_id": str(config.get("config_id")),
        "feature_family": str(config.get("feature_family")),
        "feature_columns": list(config.get("feature_columns") or ()),
        "preprocessing": str(config.get("preprocessing")),
        "clusterer": str(config.get("clusterer")),
        "optimizer": str(config.get("optimizer", NO_OPTIMIZER)),
        "optional_dependency": config.get("optional_dependency"),
        "hyperparameters": _jsonable(dict(config.get("hyperparameters") or {})),
    }


def _fit_clusterer_result(
    *,
    x: np.ndarray,
    clusterer: str,
    hyperparameters: Mapping[str, Any],
    max_runtime_per_fit_sec: float,
) -> tuple[str, np.ndarray, Mapping[str, Any], Mapping[str, Any], float]:
    started = time.monotonic()
    adapter = build_clusterer_adapter(clusterer, **hyperparameters)
    fit = adapter.fit(x)
    runtime_sec = float(time.monotonic() - started)
    status = str(fit.status)
    fit_metadata = getattr(fit, "fit_metadata", None)
    if fit_metadata is None:
        fit_metadata = getattr(fit, "metadata", {}) or {}
    failure_metadata = getattr(fit, "failure_metadata", None)
    if failure_metadata is None:
        failure_payload: Mapping[str, Any] = {}
    elif hasattr(failure_metadata, "as_dict"):
        failure_payload = failure_metadata.as_dict()
    else:
        failure_payload = dict(failure_metadata)
    if runtime_sec > float(max_runtime_per_fit_sec):
        return STATUS_FAILED, np.asarray(fit.labels, dtype=int), dict(fit_metadata), {
            "reason_code": "fit_runtime_exceeded",
            "error": f"fit runtime {runtime_sec:.3f}s exceeded {max_runtime_per_fit_sec:.3f}s",
            "fit_status": status,
        }, runtime_sec
    if status == FIT_STATUS_FITTED:
        return STATUS_SUCCEEDED, np.asarray(fit.labels, dtype=int), dict(fit_metadata), failure_payload, runtime_sec
    if status == FIT_STATUS_DEPENDENCY_UNAVAILABLE:
        return STATUS_SKIPPED_DEPENDENCY_UNAVAILABLE, np.asarray(fit.labels, dtype=int), dict(fit_metadata), failure_payload, runtime_sec
    return STATUS_FAILED, np.asarray(fit.labels, dtype=int), dict(fit_metadata), failure_payload, runtime_sec


def _fit_optuna_helper(
    *,
    x: np.ndarray,
    seed: int,
    base_hyperparameters: Mapping[str, Any],
    max_runtime_per_fit_sec: float,
) -> tuple[str, np.ndarray, Mapping[str, Any], Mapping[str, Any], float]:
    if not _optional_dependency_available(OPTUNA_DEPENDENCY):
        return STATUS_SKIPPED_DEPENDENCY_UNAVAILABLE, np.empty(0, dtype=int), {}, {
            "reason_code": "dependency_unavailable",
            "dependency": OPTUNA_DEPENDENCY,
        }, 0.0
    started = time.monotonic()
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        max_k = max(2, min(4, int(x.shape[0]) - 1))
        choices = [value for value in (2, 3, 4) if value <= max_k]
        if not choices:
            choices = [2]

        def objective(trial: Any) -> float:
            k = int(trial.suggest_categorical("n_clusters", choices))
            params = _bounded_hyperparameters(
                "kmeans",
                {**dict(base_hyperparameters), "n_clusters": k},
                seed=seed,
                row_count=int(x.shape[0]),
            )
            status, labels, _, _, _ = _fit_clusterer_result(
                x=x,
                clusterer="kmeans",
                hyperparameters=params,
                max_runtime_per_fit_sec=max_runtime_per_fit_sec,
            )
            if status != STATUS_SUCCEEDED:
                return -1.0
            metrics = _cluster_metrics(x, labels, runtime_sec=0.0)
            return float(metrics.get("quality_score") or -1.0)

        sampler = optuna.samplers.RandomSampler(seed=int(seed))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=1, n_jobs=1, timeout=float(max_runtime_per_fit_sec), show_progress_bar=False)
        best_params = dict(study.best_params or {})
        final_params = _bounded_hyperparameters(
            "kmeans",
            {**dict(base_hyperparameters), **best_params},
            seed=seed,
            row_count=int(x.shape[0]),
        )
        status, labels, fit_metadata, failure_metadata, fit_runtime = _fit_clusterer_result(
            x=x,
            clusterer="kmeans",
            hyperparameters=final_params,
            max_runtime_per_fit_sec=max_runtime_per_fit_sec,
        )
        runtime_sec = float(time.monotonic() - started)
        metadata = {
            **dict(fit_metadata),
            "optimizer": OPTUNA_HELPER,
            "optuna_n_trials": 1,
            "optuna_n_jobs": 1,
            "optuna_best_params": best_params,
            "optuna_best_value": _finite_float(getattr(study, "best_value", None)),
            "fit_runtime_sec": fit_runtime,
        }
        return status, labels, metadata, failure_metadata, runtime_sec
    except Exception as exc:
        return STATUS_FAILED, np.empty(0, dtype=int), {"optimizer": OPTUNA_HELPER}, {
            "reason_code": "optuna_helper_failed",
            "error": str(exc),
        }, float(time.monotonic() - started)


def _run_one_config(
    *,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    max_runtime_per_fit_sec: float,
) -> dict[str, Any]:
    started = time.monotonic()
    dependency = config.get("optional_dependency")
    if dependency and not _optional_dependency_available(str(dependency)):
        return _skip_result(cell=cell, config=config, seed=seed, dependency=str(dependency), started=started)
    frame = build_stratified_feature_frame(
        cohort=str(cell["cohort"]),
        asset=str(cell["asset"]),
        axis=str(cell["axis"]),
        band=str(cell["band"]),
        seed=int(seed),
    )
    try:
        preprocessing = fit_preprocessing_pipeline(
            frame,
            tuple(str(column) for column in config.get("feature_columns") or ()),
            preprocess=str(config["preprocessing"]),
            fit_window={
                "pathway": PATHWAY_ASSET_STATE,
                "panel": PANEL_STRATIFIED,
                "cohort": str(cell["cohort"]),
                "asset": str(cell["asset"]),
                "axis": str(cell["axis"]),
                "band": str(cell["band"]),
                "seed": int(seed),
            },
            fit_window_role="train",
        )
    except Exception as exc:
        metrics = {
            "row_count": 0,
            "feature_count": 0,
            "effective_state_count": 0,
            "noise_count": 0,
            "noise_share": None,
            "label_counts": {},
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "quality_score": None,
            "runtime_sec": float(time.monotonic() - started),
        }
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "artifact_kind": BENCHMARK_RESULTS_ARTIFACT_KIND,
            "status": STATUS_INVALID_INPUT,
            "final_label": LABEL_FAILED,
            "seed": int(seed),
            "cell": _cell_identity(cell),
            "config": _config_identity(config),
            "labels": [],
            "metrics": metrics,
            "preprocessing": {"error": str(exc)},
            "failure_metadata": {"reason_code": "preprocessing_failed", "error": str(exc)},
        }
    x = np.asarray(preprocessing.fitted.x, dtype=float)
    if x.ndim != 2 or int(x.shape[0]) < 3 or int(x.shape[1]) < 1:
        metrics = {
            "row_count": int(x.shape[0]) if x.ndim == 2 else 0,
            "feature_count": int(x.shape[1]) if x.ndim == 2 else 0,
            "effective_state_count": 0,
            "noise_count": 0,
            "noise_share": None,
            "label_counts": {},
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "quality_score": None,
            "runtime_sec": float(time.monotonic() - started),
        }
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "artifact_kind": BENCHMARK_RESULTS_ARTIFACT_KIND,
            "status": STATUS_INVALID_INPUT,
            "final_label": LABEL_FAILED,
            "seed": int(seed),
            "cell": _cell_identity(cell),
            "config": _config_identity(config),
            "labels": [],
            "metrics": metrics,
            "preprocessing": preprocessing.as_dict(),
            "failure_metadata": {"reason_code": "insufficient_preprocessed_matrix"},
        }
    clusterer = str(config["clusterer"])
    params = _bounded_hyperparameters(clusterer, dict(config.get("hyperparameters") or {}), seed=int(seed), row_count=int(x.shape[0]))
    if str(config.get("optimizer", NO_OPTIMIZER)) == OPTUNA_HELPER:
        status, labels, fit_metadata, failure_metadata, fit_runtime = _fit_optuna_helper(
            x=x,
            seed=int(seed),
            base_hyperparameters=params,
            max_runtime_per_fit_sec=max_runtime_per_fit_sec,
        )
    else:
        status, labels, fit_metadata, failure_metadata, fit_runtime = _fit_clusterer_result(
            x=x,
            clusterer=clusterer,
            hyperparameters=params,
            max_runtime_per_fit_sec=max_runtime_per_fit_sec,
        )
    metrics = _cluster_metrics(x, labels, runtime_sec=fit_runtime) if labels.size else {
        "row_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "effective_state_count": 0,
        "noise_count": 0,
        "noise_share": None,
        "label_counts": {},
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "quality_score": None,
        "runtime_sec": fit_runtime,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": BENCHMARK_RESULTS_ARTIFACT_KIND,
        "status": status,
        "final_label": _benchmark_label(status, metrics),
        "seed": int(seed),
        "cell": _cell_identity(cell),
        "config": {**_config_identity(config), "resolved_hyperparameters": _jsonable(params)},
        "labels": [int(label) for label in labels.tolist()],
        "metrics": metrics,
        "preprocessing": preprocessing.as_dict(),
        "fit_metadata": _jsonable(dict(fit_metadata)),
        "failure_metadata": _jsonable(dict(failure_metadata)),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def _scoreboard_group(rows: Sequence[Mapping[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        config = row["config"]
        cell = row["cell"]
        if group_key == "band":
            key = (cell["band"], config["config_id"])
        elif group_key == "clusterer":
            key = (config["clusterer"], config["config_id"])
        else:
            key = (config["config_id"],)
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        first = items[0]
        status_counts = Counter(str(item.get("status")) for item in items)
        label_counts = Counter(str(item.get("final_label")) for item in items)
        quality_values = [item.get("metrics", {}).get("quality_score") for item in items if item.get("status") == STATUS_SUCCEEDED]
        silhouette_values = [item.get("metrics", {}).get("silhouette") for item in items if item.get("status") == STATUS_SUCCEEDED]
        runtime_values = [item.get("metrics", {}).get("runtime_sec") for item in items if item.get("metrics")]
        group = {
            "group_key": group_key,
            "group_value": str(key[0]),
            "config_id": str(first["config"]["config_id"]),
            "feature_family": str(first["config"]["feature_family"]),
            "preprocessing": str(first["config"]["preprocessing"]),
            "clusterer": str(first["config"]["clusterer"]),
            "optimizer": str(first["config"].get("optimizer", NO_OPTIMIZER)),
            "fit_count": len(items),
            "status_counts": dict(sorted(status_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "mean_quality_score": _mean([_finite_float(value) for value in quality_values]),
            "mean_silhouette": _mean([_finite_float(value) for value in silhouette_values]),
            "mean_runtime_sec": _mean([_finite_float(value) for value in runtime_values]),
        }
        out.append(group)
    out.sort(
        key=lambda item: (
            -(item["mean_quality_score"] if item["mean_quality_score"] is not None else -999.0),
            int(item["status_counts"].get(STATUS_FAILED, 0)),
            str(item["group_value"]),
            str(item["config_id"]),
        )
    )
    return out


def _build_scoreboard(*, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> Mapping[str, Any]:
    status_counts = Counter(str(row.get("status")) for row in rows)
    label_counts = Counter(str(row.get("final_label")) for row in rows)
    top_configs = _scoreboard_group(rows, "config")[:12]
    by_band = _scoreboard_group(rows, "band")
    by_clusterer = _scoreboard_group(rows, "clusterer")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": BENCHMARK_SCOREBOARD_ARTIFACT_KIND,
        "created_at_utc": _now_utc(),
        "artifact_boundary": _artifact_boundary(),
        "source_manifest_artifact_kind": manifest.get("artifact_kind", "unspecified"),
        "source_manifest_cell_count": len(manifest.get("cells") or ()),
        "seeds": list(_normalize_seeds([int(seed) for seed in seeds])),
        "summary": {
            "result_count": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "succeeded_count": int(status_counts.get(STATUS_SUCCEEDED, 0)),
            "skipped_dependency_unavailable_count": int(status_counts.get(STATUS_SKIPPED_DEPENDENCY_UNAVAILABLE, 0)),
            "failed_count": int(status_counts.get(STATUS_FAILED, 0) + status_counts.get(STATUS_INVALID_INPUT, 0)),
        },
        "optional_dependencies": {
            HDBSCAN_DEPENDENCY: _optional_dependency_available(HDBSCAN_DEPENDENCY),
            OPTUNA_DEPENDENCY: _optional_dependency_available(OPTUNA_DEPENDENCY),
        },
        "top_configs": top_configs,
        "by_band": by_band,
        "by_clusterer": by_clusterer,
    }


def _scoreboard_markdown(scoreboard: Mapping[str, Any]) -> str:
    summary = scoreboard["summary"]
    optional = scoreboard["optional_dependencies"]
    lines = [
        "# Asset-State Benchmark Scoreboard",
        "",
        f"- Artifact kind: `{scoreboard['artifact_kind']}`",
        f"- Result rows: {summary['result_count']}",
        f"- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        f"- Final label counts: `{json.dumps(summary['label_counts'], sort_keys=True)}`",
        f"- Optional dependencies: `hdbscan={optional.get(HDBSCAN_DEPENDENCY)}`, `optuna={optional.get(OPTUNA_DEPENDENCY)}`",
        f"- Boundary: stratified panel only; production labels written = `{scoreboard['artifact_boundary']['production_labels_written']}`; promotion allowed = `{scoreboard['artifact_boundary']['promotion_allowed']}`",
        "",
        "## Top Configs",
        "",
        "| Rank | Config | Feature family | Preprocessing | Clusterer | Optimizer | Mean quality | Mean silhouette | Fits |",
        "|---:|---|---|---|---|---|---:|---:|---:|",
    ]
    for idx, item in enumerate(scoreboard.get("top_configs") or (), start=1):
        quality = item.get("mean_quality_score")
        silhouette = item.get("mean_silhouette")
        lines.append(
            "| {rank} | `{config}` | `{features}` | `{pre}` | `{clusterer}` | `{optimizer}` | {quality} | {silhouette} | {fits} |".format(
                rank=idx,
                config=item["config_id"],
                features=item["feature_family"],
                pre=item["preprocessing"],
                clusterer=item["clusterer"],
                optimizer=item["optimizer"],
                quality="n/a" if quality is None else f"{float(quality):.4f}",
                silhouette="n/a" if silhouette is None else f"{float(silhouette):.4f}",
                fits=item["fit_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Band Leaders",
            "",
            "| Band | Config | Clusterer | Mean quality | Fits |",
            "|---|---|---|---:|---:|",
        ]
    )
    seen_bands: set[str] = set()
    for item in scoreboard.get("by_band") or ():
        band = str(item["group_value"])
        if band in seen_bands:
            continue
        seen_bands.add(band)
        quality = item.get("mean_quality_score")
        lines.append(
            f"| `{band}` | `{item['config_id']}` | `{item['clusterer']}` | "
            f"{'n/a' if quality is None else f'{float(quality):.4f}'} | {item['fit_count']} |"
        )
    return "\n".join(lines)


def run_asset_state_benchmark(
    *,
    manifest: str | Path | Mapping[str, Any],
    seeds: Sequence[int],
    max_configs_per_cell: int,
    max_runtime_per_fit_sec: float,
    n_jobs: int,
    write_root: str | Path,
    no_write: bool = False,
    project_root: str | Path | None = None,
) -> BenchmarkRunResult:
    if int(n_jobs) != 1:
        raise ValueError("asset-state benchmark runner requires n_jobs=1")
    if int(max_configs_per_cell) < 1:
        raise ValueError("asset-state benchmark runner requires max_configs_per_cell >= 1")
    if float(max_runtime_per_fit_sec) <= 0:
        raise ValueError("asset-state benchmark runner requires max_runtime_per_fit_sec > 0")
    root = Path(write_root) if no_write else validate_clusterability_write_root(write_root, project_root=project_root)
    if isinstance(manifest, Mapping):
        matrix = manifest
    else:
        matrix_path = Path(manifest)
        if not matrix_path.is_absolute():
            matrix_path = Path.cwd() / matrix_path
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("artifact_kind") != BENCHMARK_MATRIX_ARTIFACT_KIND:
        raise ValueError("asset-state benchmark manifest must be an asset-state benchmark matrix")
    normalized_seeds = _normalize_seeds([int(seed) for seed in seeds])
    rows: list[Mapping[str, Any]] = []
    for cell in matrix.get("cells") or ():
        configs = list(cell.get("configs") or ())[: int(max_configs_per_cell)]
        for config in configs:
            for seed in normalized_seeds:
                rows.append(
                    _run_one_config(
                        cell=cell,
                        config=config,
                        seed=seed,
                        max_runtime_per_fit_sec=float(max_runtime_per_fit_sec),
                    )
                )
    scoreboard = _build_scoreboard(manifest=matrix, rows=rows, seeds=normalized_seeds)
    artifact_paths: dict[str, str] = {}
    if not no_write:
        results_path = root / RESULTS_FILENAME
        scoreboard_json_path = root / SCOREBOARD_JSON_FILENAME
        scoreboard_md_path = root / SCOREBOARD_MD_FILENAME
        _write_jsonl(results_path, rows)
        _write_json(scoreboard_json_path, scoreboard)
        _write_text(scoreboard_md_path, _scoreboard_markdown(scoreboard))
        artifact_paths = {
            "results_jsonl": str(results_path),
            "scoreboard_json": str(scoreboard_json_path),
            "scoreboard_markdown": str(scoreboard_md_path),
        }
    return BenchmarkRunResult(scoreboard=scoreboard, rows=tuple(rows), artifact_paths=artifact_paths)


def _build_matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build bounded asset-state benchmark matrix from clusterability preflight.")
    parser.add_argument("--write-root", required=True)
    parser.add_argument("--preflight", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def _build_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded asset-state benchmark matrix.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-configs-per-cell", type=int, required=True)
    parser.add_argument("--max-runtime-per-fit-sec", type=float, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--write-root", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    raw_args = list(sys.argv[1:] if argv is None else argv)
    runner_arg_names = {"--manifest", "--seeds", "--max-configs-per-cell", "--max-runtime-per-fit-sec", "--n-jobs"}
    if raw_args and raw_args[0] not in {"build-matrix", "run"}:
        if runner_arg_names.intersection(raw_args):
            runner_args = _build_runner_parser().parse_args(raw_args)
            result = run_asset_state_benchmark(
                manifest=runner_args.manifest,
                seeds=runner_args.seeds,
                max_configs_per_cell=runner_args.max_configs_per_cell,
                max_runtime_per_fit_sec=runner_args.max_runtime_per_fit_sec,
                n_jobs=runner_args.n_jobs,
                write_root=runner_args.write_root,
                no_write=runner_args.no_write,
                project_root=runner_args.project_root,
            )
            print(json.dumps({"artifact_paths": result.artifact_paths, "result_count": len(result.rows)}, sort_keys=True))
            return 0
        if "--write-root" in raw_args:
            matrix_args = _build_matrix_parser().parse_args(raw_args)
            result = build_asset_state_benchmark_matrix(
                write_root=matrix_args.write_root,
                preflight_path=matrix_args.preflight,
                no_write=matrix_args.no_write,
                project_root=matrix_args.project_root,
            )
            print(json.dumps({"artifact_paths": result.artifact_paths, "cell_count": len(result.payload.get("cells") or ())}, sort_keys=True))
            return 0

    parser = argparse.ArgumentParser(description="Asset-state benchmark matrix builder and runner.")
    subparsers = parser.add_subparsers(dest="command")
    matrix_parser = subparsers.add_parser("build-matrix", parents=[_build_matrix_parser()], add_help=False)
    matrix_parser.set_defaults(command="build-matrix")
    runner_parser = subparsers.add_parser("run", parents=[_build_runner_parser()], add_help=False)
    runner_parser.set_defaults(command="run")
    args, unknown = parser.parse_known_args(raw_args)

    if args.command == "build-matrix":
        result = build_asset_state_benchmark_matrix(
            write_root=args.write_root,
            preflight_path=args.preflight,
            no_write=args.no_write,
            project_root=args.project_root,
        )
        print(json.dumps({"artifact_paths": result.artifact_paths, "cell_count": len(result.payload.get("cells") or ())}, sort_keys=True))
        return 0
    if args.command == "run":
        result = run_asset_state_benchmark(
            manifest=args.manifest,
            seeds=args.seeds,
            max_configs_per_cell=args.max_configs_per_cell,
            max_runtime_per_fit_sec=args.max_runtime_per_fit_sec,
            n_jobs=args.n_jobs,
            write_root=args.write_root,
            no_write=args.no_write,
            project_root=args.project_root,
        )
        print(json.dumps({"artifact_paths": result.artifact_paths, "result_count": len(result.rows)}, sort_keys=True))
        return 0

    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
