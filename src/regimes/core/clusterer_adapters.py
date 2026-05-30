from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.foundation_contracts import REGIME_LAYER_AXES, REGIME_LAYERS, REGIME_STUDY_BANDS
from src.regimes.core.clusterer_base import (
    AssignmentPolicy,
    BaseClustererAdapter,
    ClustererAssignmentResult,
    ClustererCapabilities,
    ClustererFailureMetadata,
    ClustererFitResult,
    ClustererRuntimeMetadata,
)

try:
    from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, MiniBatchKMeans, OPTICS
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

    _HAS_SKLEARN = True
    _SKLEARN_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only in minimal dependency environments
    AgglomerativeClustering = None  # type: ignore[assignment]
    Birch = None  # type: ignore[assignment]
    KMeans = None  # type: ignore[assignment]
    MiniBatchKMeans = None  # type: ignore[assignment]
    OPTICS = None  # type: ignore[assignment]
    BayesianGaussianMixture = None  # type: ignore[assignment]
    GaussianMixture = None  # type: ignore[assignment]
    _HAS_SKLEARN = False
    _SKLEARN_IMPORT_ERROR = str(exc)

try:
    import hdbscan  # type: ignore

    _HAS_HDBSCAN = True
    _HDBSCAN_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - optional dependency guard
    hdbscan = None  # type: ignore[assignment]
    _HAS_HDBSCAN = False
    _HDBSCAN_IMPORT_ERROR = str(exc)


REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION = 1

ASSIGNMENT_NATIVE_PREDICT = "native_predict"
ASSIGNMENT_APPROXIMATE_PREDICT = "approximate_predict"
ASSIGNMENT_SCORE_SAMPLES_OR_PROBABILITIES = "score_samples_or_probabilities"
ASSIGNMENT_NEAREST_LABELED_NEIGHBOR = "nearest_labeled_neighbor"
ASSIGNMENT_PROTOTYPE_OR_MEDOID = "prototype_or_medoid"
ASSIGNMENT_SCHEDULED_REFIT = "scheduled_refit"
ASSIGNMENT_FULL_RECLUSTER = "full_recluster"
REGIME_ASSIGNMENT_POLICIES: tuple[str, ...] = (
    ASSIGNMENT_NATIVE_PREDICT,
    ASSIGNMENT_APPROXIMATE_PREDICT,
    ASSIGNMENT_SCORE_SAMPLES_OR_PROBABILITIES,
    ASSIGNMENT_NEAREST_LABELED_NEIGHBOR,
    ASSIGNMENT_PROTOTYPE_OR_MEDOID,
    ASSIGNMENT_SCHEDULED_REFIT,
    ASSIGNMENT_FULL_RECLUSTER,
)

INDUCTIVE = "inductive"
TRANSDUCTIVE = "transductive"

FIT_STATUS_FITTED = "fitted"
FIT_STATUS_FAILED = "failed"
FIT_STATUS_DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
FIT_STATUS_INVALID_INPUT = "invalid_input"
FIT_STATUS_PLACEHOLDER = "placeholder"

ASSIGN_STATUS_ASSIGNED = "assigned"
ASSIGN_STATUS_UNSUPPORTED = "unsupported"
ASSIGN_STATUS_FAILED = "failed"
ASSIGN_STATUS_DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
ASSIGN_STATUS_NOT_FITTED = "not_fitted"

TIER_A_CLUSTERERS: tuple[str, ...] = (
    "kmeans",
    "minibatch_kmeans",
    "gaussian_mixture",
    "bayesian_gaussian_mixture",
    "hdbscan",
    "optics",
    "agglomerative",
    "birch",
)
TIER_B_PLACEHOLDER_CLUSTERERS: tuple[str, ...] = ("spectral_clustering",)

_ALL_AXES = tuple(dict.fromkeys(axis for axes in REGIME_LAYER_AXES.values() for axis in axes))


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, BaseException):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Regime clusterer {field_name} must be non-empty")
    return text


def _require_members(values: Sequence[object], allowed: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(_normalize_token(value, field_name=field_name) for value in values)
    if not normalized:
        raise ValueError(f"Regime clusterer {field_name} must include at least one value")
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime clusterer {field_name} {invalid[0]!r}; expected one of: {valid}")
    return tuple(dict.fromkeys(normalized))


def _dependency_available(dependency_name: str | None) -> bool:
    if dependency_name is None:
        return True
    if dependency_name == "sklearn":
        return bool(_HAS_SKLEARN)
    if dependency_name == "hdbscan":
        return bool(_HAS_HDBSCAN)
    return False


def _dependency_error(dependency_name: str | None) -> str | None:
    if dependency_name == "sklearn":
        return _SKLEARN_IMPORT_ERROR
    if dependency_name == "hdbscan":
        return _HDBSCAN_IMPORT_ERROR
    return None


@dataclass(frozen=True)
class RegimeClustererSpec:
    family_name: str
    library_source: str
    tier: str
    inductive_classification: str
    assignment_policy: str
    supported_layers: tuple[str, ...] = REGIME_LAYERS
    supported_axes: tuple[str, ...] = _ALL_AXES
    supported_bands: tuple[str, ...] = REGIME_STUDY_BANDS
    dependency_name: str | None = "sklearn"
    supports_fit: bool = True
    supports_assign: bool = False
    supports_refit_recluster: bool = False
    supports_probabilities: bool = False
    supports_soft_membership: bool = False
    supports_noise: bool = False
    default_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    hyperparameter_schema: Mapping[str, Any] = field(default_factory=dict)
    search_space_hook: str = "static_hyperparameter_schema"
    production_caveats: tuple[str, ...] = ()
    schema_version: int = REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        family = _normalize_token(self.family_name, field_name="family name")
        policy = _require_members((self.assignment_policy,), REGIME_ASSIGNMENT_POLICIES, field_name="assignment policy")[0]
        inductive = _require_members(
            (self.inductive_classification,),
            (INDUCTIVE, TRANSDUCTIVE),
            field_name="inductive classification",
        )[0]
        layers = _require_members(self.supported_layers, REGIME_LAYERS, field_name="supported layers")
        axes = _require_members(self.supported_axes, _ALL_AXES, field_name="supported axes")
        bands = _require_members(self.supported_bands, REGIME_STUDY_BANDS, field_name="supported bands")
        if not str(self.library_source).strip():
            raise ValueError("Regime clusterer library_source must be non-empty")
        if not str(self.tier).strip():
            raise ValueError("Regime clusterer tier must be non-empty")
        object.__setattr__(self, "family_name", family)
        object.__setattr__(self, "assignment_policy", policy)
        object.__setattr__(self, "inductive_classification", inductive)
        object.__setattr__(self, "supported_layers", layers)
        object.__setattr__(self, "supported_axes", axes)
        object.__setattr__(self, "supported_bands", bands)
        object.__setattr__(self, "default_hyperparameters", dict(self.default_hyperparameters))
        object.__setattr__(self, "hyperparameter_schema", dict(self.hyperparameter_schema))
        object.__setattr__(self, "production_caveats", tuple(str(item) for item in self.production_caveats))

    @property
    def dependency_available(self) -> bool:
        return _dependency_available(self.dependency_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "library_source": self.library_source,
            "tier": self.tier,
            "inductive_classification": self.inductive_classification,
            "assignment_policy": self.assignment_policy,
            "supported_layers": list(self.supported_layers),
            "supported_axes": list(self.supported_axes),
            "supported_bands": list(self.supported_bands),
            "dependency_name": self.dependency_name,
            "dependency_available": bool(self.dependency_available),
            "dependency_error": _dependency_error(self.dependency_name),
            "supports_fit": bool(self.supports_fit),
            "supports_assign": bool(self.supports_assign),
            "supports_refit_recluster": bool(self.supports_refit_recluster),
            "supports_probabilities": bool(self.supports_probabilities),
            "supports_soft_membership": bool(self.supports_soft_membership),
            "supports_noise": bool(self.supports_noise),
            "default_hyperparameters": _jsonable(dict(self.default_hyperparameters)),
            "hyperparameter_schema": _jsonable(dict(self.hyperparameter_schema)),
            "search_space_hook": self.search_space_hook,
            "production_caveats": list(self.production_caveats),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def _confidence_from_probabilities(probabilities: np.ndarray | None) -> np.ndarray | None:
    if probabilities is None:
        return None
    arr = np.asarray(probabilities, dtype=float)
    if arr.ndim == 2:
        return np.nanmax(arr, axis=1)
    if arr.ndim == 1:
        return arr
    return None


def _noise_mask(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(labels, dtype=int) == -1


def _label_outcome_metadata(labels: Sequence[int] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(labels, dtype=int)
    row_count = int(arr.size)
    non_noise = [int(value) for value in arr.tolist() if int(value) != -1]
    cluster_count = int(len(set(non_noise)))
    noise_count = int(np.sum(arr == -1))
    noise_share = float(noise_count / row_count) if row_count else None
    return {
        "row_count": row_count,
        "effective_cluster_count": cluster_count,
        "noise_count": noise_count,
        "noise_share": noise_share,
        "single_cluster_outcome": bool(row_count > 0 and cluster_count == 1),
        "all_noise_outcome": bool(row_count > 0 and noise_count == row_count),
        "mostly_noise_outcome": bool(noise_share is not None and noise_share >= 0.8),
        "unique_labels": sorted(int(value) for value in set(arr.tolist())) if row_count else [],
    }


@dataclass(frozen=True)
class RegimeClusterFitResult:
    family_name: str
    status: str
    labels: np.ndarray
    probabilities: np.ndarray | None
    soft_membership: np.ndarray | None
    confidence: np.ndarray | None
    noise_mask: np.ndarray
    spec: RegimeClustererSpec
    hyperparameters: Mapping[str, Any]
    failure_metadata: Mapping[str, Any]
    runtime_metadata: Mapping[str, Any]
    fit_metadata: Mapping[str, Any]
    estimator: Any = field(default=None, repr=False, compare=False)
    schema_version: int = REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION

    @property
    def effective_cluster_count(self) -> int:
        return int(self.fit_metadata.get("effective_cluster_count", 0) or 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "status": self.status,
            "labels": _jsonable(self.labels),
            "probabilities": _jsonable(self.probabilities),
            "soft_membership": _jsonable(self.soft_membership),
            "confidence": _jsonable(self.confidence),
            "noise_mask": _jsonable(self.noise_mask),
            "spec": self.spec.as_dict(),
            "hyperparameters": _jsonable(dict(self.hyperparameters)),
            "failure_metadata": _jsonable(dict(self.failure_metadata)),
            "runtime_metadata": _jsonable(dict(self.runtime_metadata)),
            "fit_metadata": _jsonable(dict(self.fit_metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class RegimeClusterAssignmentResult:
    family_name: str
    assignment_policy: str
    status: str
    labels: np.ndarray
    probabilities: np.ndarray | None
    soft_membership: np.ndarray | None
    confidence: np.ndarray | None
    noise_mask: np.ndarray
    supported: bool
    failure_metadata: Mapping[str, Any]
    runtime_metadata: Mapping[str, Any]
    assignment_metadata: Mapping[str, Any]
    schema_version: int = REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION

    @property
    def assigned_count(self) -> int:
        return int(np.asarray(self.labels, dtype=int).size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "assignment_policy": self.assignment_policy,
            "status": self.status,
            "labels": _jsonable(self.labels),
            "probabilities": _jsonable(self.probabilities),
            "soft_membership": _jsonable(self.soft_membership),
            "confidence": _jsonable(self.confidence),
            "noise_mask": _jsonable(self.noise_mask),
            "supported": bool(self.supported),
            "failure_metadata": _jsonable(dict(self.failure_metadata)),
            "runtime_metadata": _jsonable(dict(self.runtime_metadata)),
            "assignment_metadata": _jsonable(dict(self.assignment_metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def _validate_feature_matrix(x: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as exc:
        return None, {"reason_code": "invalid_feature_matrix", "error": str(exc)}
    if arr.ndim != 2:
        return None, {"reason_code": "invalid_feature_matrix", "error": "feature matrix must be 2D"}
    if arr.shape[0] == 0:
        return None, {"reason_code": "empty_feature_matrix", "error": "feature matrix has zero rows"}
    if arr.shape[1] == 0:
        return None, {"reason_code": "empty_feature_matrix", "error": "feature matrix has zero columns"}
    if not np.isfinite(arr).all():
        return None, {"reason_code": "nonfinite_feature_matrix", "error": "feature matrix contains NaN or infinite values"}
    return arr, None


def _fit_result(
    *,
    spec: RegimeClustererSpec,
    labels: Sequence[int] | np.ndarray,
    hyperparameters: Mapping[str, Any],
    started: float,
    estimator: Any = None,
    probabilities: Sequence[float] | np.ndarray | None = None,
    soft_membership: Sequence[float] | np.ndarray | None = None,
    status: str = FIT_STATUS_FITTED,
    failure_metadata: Mapping[str, Any] | None = None,
    fit_metadata: Mapping[str, Any] | None = None,
) -> RegimeClusterFitResult:
    label_arr = np.asarray(labels, dtype=int)
    probability_arr = None if probabilities is None else np.asarray(probabilities, dtype=float)
    soft_arr = None if soft_membership is None else np.asarray(soft_membership, dtype=float)
    metadata = {
        **_label_outcome_metadata(label_arr),
        "assignment_policy": spec.assignment_policy,
        "inductive_classification": spec.inductive_classification,
    }
    metadata.update(dict(fit_metadata or {}))
    runtime = {"fit_time_s": float(time.monotonic() - started)}
    return RegimeClusterFitResult(
        family_name=spec.family_name,
        status=status,
        labels=label_arr,
        probabilities=probability_arr,
        soft_membership=soft_arr,
        confidence=_confidence_from_probabilities(probability_arr if probability_arr is not None else soft_arr),
        noise_mask=_noise_mask(label_arr),
        spec=spec,
        hyperparameters=dict(hyperparameters),
        failure_metadata=dict(failure_metadata or {}),
        runtime_metadata=runtime,
        fit_metadata=metadata,
        estimator=estimator,
    )


def _failed_fit_result(
    *,
    spec: RegimeClustererSpec,
    status: str,
    started: float,
    hyperparameters: Mapping[str, Any],
    reason_code: str,
    error: str,
) -> RegimeClusterFitResult:
    return _fit_result(
        spec=spec,
        labels=np.empty(0, dtype=int),
        probabilities=None,
        soft_membership=None,
        hyperparameters=hyperparameters,
        started=started,
        status=status,
        failure_metadata={"reason_code": reason_code, "error": error},
        fit_metadata={"assignment_policy": spec.assignment_policy},
    )


def _assignment_result(
    *,
    spec: RegimeClustererSpec,
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray | None,
    soft_membership: Sequence[float] | np.ndarray | None = None,
    status: str,
    supported: bool,
    started: float,
    assignment_policy: str | None = None,
    failure_metadata: Mapping[str, Any] | None = None,
    assignment_metadata: Mapping[str, Any] | None = None,
) -> RegimeClusterAssignmentResult:
    label_arr = np.asarray(labels, dtype=int)
    probability_arr = None if probabilities is None else np.asarray(probabilities, dtype=float)
    soft_arr = None if soft_membership is None else np.asarray(soft_membership, dtype=float)
    metadata = {
        **_label_outcome_metadata(label_arr),
        "inductive_classification": spec.inductive_classification,
    }
    metadata.update(dict(assignment_metadata or {}))
    return RegimeClusterAssignmentResult(
        family_name=spec.family_name,
        assignment_policy=str(assignment_policy or spec.assignment_policy),
        status=str(status),
        labels=label_arr,
        probabilities=probability_arr,
        soft_membership=soft_arr,
        confidence=_confidence_from_probabilities(probability_arr if probability_arr is not None else soft_arr),
        noise_mask=_noise_mask(label_arr),
        supported=bool(supported),
        failure_metadata=dict(failure_metadata or {}),
        runtime_metadata={"assign_time_s": float(time.monotonic() - started)},
        assignment_metadata=metadata,
    )


class RegimeClustererAdapter:
    spec: RegimeClustererSpec

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters = {**dict(self.spec.default_hyperparameters), **dict(hyperparameters)}
        self.estimator: Any = None
        self.last_fit_result: RegimeClusterFitResult | None = None
        self._train_x: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None

    @property
    def family_name(self) -> str:
        return self.spec.family_name

    @property
    def assignment_policy(self) -> str:
        return self.spec.assignment_policy

    def hyperparameter_schema(self) -> dict[str, Any]:
        return _jsonable(dict(self.spec.hyperparameter_schema))

    def search_space(self) -> dict[str, Any]:
        return {
            "hook": self.spec.search_space_hook,
            "family_name": self.family_name,
            "hyperparameter_schema": self.hyperparameter_schema(),
        }

    def _dependency_failure(self, started: float) -> RegimeClusterFitResult:
        return _failed_fit_result(
            spec=self.spec,
            status=FIT_STATUS_DEPENDENCY_UNAVAILABLE,
            started=started,
            hyperparameters=self.hyperparameters,
            reason_code="dependency_unavailable",
            error=f"{self.spec.dependency_name} is unavailable: {_dependency_error(self.spec.dependency_name)}",
        )

    def fit(self, x: Sequence[Sequence[float]] | np.ndarray) -> RegimeClusterFitResult:
        started = time.monotonic()
        if not self.spec.dependency_available:
            self.last_fit_result = self._dependency_failure(started)
            return self.last_fit_result
        arr, failure = _validate_feature_matrix(x)
        if failure is not None:
            self.last_fit_result = _failed_fit_result(
                spec=self.spec,
                status=FIT_STATUS_INVALID_INPUT,
                started=started,
                hyperparameters=self.hyperparameters,
                reason_code=str(failure["reason_code"]),
                error=str(failure["error"]),
            )
            return self.last_fit_result
        try:
            self.last_fit_result = self._fit(np.asarray(arr, dtype=float), started=started)
            return self.last_fit_result
        except Exception as exc:
            self.last_fit_result = _failed_fit_result(
                spec=self.spec,
                status=FIT_STATUS_FAILED,
                started=started,
                hyperparameters=self.hyperparameters,
                reason_code="fit_failed",
                error=str(exc),
            )
            return self.last_fit_result

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        raise NotImplementedError

    def assign(
        self,
        x: Sequence[Sequence[float]] | np.ndarray,
        *,
        assignment_policy: str | None = None,
    ) -> RegimeClusterAssignmentResult:
        started = time.monotonic()
        requested = self.spec.assignment_policy if assignment_policy is None else _normalize_token(
            assignment_policy,
            field_name="assignment policy",
        )
        if requested not in REGIME_ASSIGNMENT_POLICIES:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_UNSUPPORTED,
                supported=False,
                started=started,
                assignment_policy=requested,
                failure_metadata={
                    "reason_code": "unsupported_assignment_policy",
                    "error": f"Unsupported assignment policy {requested!r}",
                },
            )
        if requested != self.spec.assignment_policy:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_UNSUPPORTED,
                supported=False,
                started=started,
                assignment_policy=requested,
                failure_metadata={
                    "reason_code": "unsupported_assignment_policy",
                    "error": f"{self.family_name} supports {self.spec.assignment_policy!r}, not {requested!r}",
                },
            )
        if not self.spec.supports_assign:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_UNSUPPORTED,
                supported=False,
                started=started,
                assignment_policy=requested,
                failure_metadata={
                    "reason_code": "assignment_requires_refit_or_recluster",
                    "error": f"{self.family_name} is {self.spec.inductive_classification}; use refit_recluster",
                },
            )
        if not self.spec.dependency_available:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_DEPENDENCY_UNAVAILABLE,
                supported=False,
                started=started,
                assignment_policy=requested,
                failure_metadata={
                    "reason_code": "dependency_unavailable",
                    "error": f"{self.spec.dependency_name} is unavailable: {_dependency_error(self.spec.dependency_name)}",
                },
            )
        if self.estimator is None:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_NOT_FITTED,
                supported=True,
                started=started,
                assignment_policy=requested,
                failure_metadata={"reason_code": "not_fitted", "error": "adapter has not been fitted"},
            )
        arr, failure = _validate_feature_matrix(x)
        if failure is not None:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_FAILED,
                supported=True,
                started=started,
                assignment_policy=requested,
                failure_metadata={"reason_code": str(failure["reason_code"]), "error": str(failure["error"])},
            )
        try:
            return self._assign(np.asarray(arr, dtype=float), started=started, assignment_policy=requested)
        except Exception as exc:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_FAILED,
                supported=True,
                started=started,
                assignment_policy=requested,
                failure_metadata={"reason_code": "assignment_failed", "error": str(exc)},
            )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        raise NotImplementedError

    def refit_recluster(self, x: Sequence[Sequence[float]] | np.ndarray) -> RegimeClusterFitResult:
        return self.fit(x)


def _native_predict(adapter: RegimeClustererAdapter, x: np.ndarray, *, started: float, policy: str) -> RegimeClusterAssignmentResult:
    labels = np.asarray(adapter.estimator.predict(x), dtype=int)
    return _assignment_result(
        spec=adapter.spec,
        labels=labels,
        probabilities=None,
        status=ASSIGN_STATUS_ASSIGNED,
        supported=True,
        started=started,
        assignment_policy=policy,
        assignment_metadata={"assignment_method": "estimator.predict"},
    )


def _probability_predict(
    adapter: RegimeClustererAdapter,
    x: np.ndarray,
    *,
    started: float,
    policy: str,
) -> RegimeClusterAssignmentResult:
    probabilities = np.asarray(adapter.estimator.predict_proba(x), dtype=float)
    labels = np.asarray(np.argmax(probabilities, axis=1), dtype=int)
    return _assignment_result(
        spec=adapter.spec,
        labels=labels,
        probabilities=probabilities,
        soft_membership=probabilities,
        status=ASSIGN_STATUS_ASSIGNED,
        supported=True,
        started=started,
        assignment_policy=policy,
        assignment_metadata={"assignment_method": "estimator.predict_proba_argmax"},
    )


def _assignment_policy_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "assignment_k": max(1, int(params.get("assignment_k", 1))),
        "assignment_threshold_quantile": max(0.0, min(float(params.get("assignment_threshold_quantile", 0.95)), 1.0)),
        "assignment_threshold_multiplier": max(0.0, float(params.get("assignment_threshold_multiplier", 1.25))),
        "unknown_label": int(params.get("unknown_label", -1)),
        "exclude_noise_from_assignment": bool(params.get("exclude_noise_from_assignment", True)),
    }


def _strip_assignment_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in params.items()
        if str(key)
        not in {
            "assignment_k",
            "assignment_threshold_quantile",
            "assignment_threshold_multiplier",
            "unknown_label",
            "exclude_noise_from_assignment",
        }
    }


def _valid_train_mask(labels: np.ndarray, *, exclude_noise: bool) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    return labels != -1 if exclude_noise else np.ones(labels.shape, dtype=bool)


def _cluster_radius_threshold(
    train_x: np.ndarray,
    labels: np.ndarray,
    *,
    policy_params: Mapping[str, Any],
) -> float | None:
    valid = _valid_train_mask(labels, exclude_noise=bool(policy_params.get("exclude_noise_from_assignment", True)))
    if not bool(np.any(valid)):
        return None
    distances: list[float] = []
    for label in sorted(int(value) for value in set(labels[valid].tolist())):
        members = train_x[labels == label]
        if members.size == 0:
            continue
        prototype = np.mean(members, axis=0)
        distances.extend(np.linalg.norm(members - prototype, axis=1).astype(float).tolist())
    if not distances:
        return None
    q = float(policy_params.get("assignment_threshold_quantile", 0.95))
    multiplier = float(policy_params.get("assignment_threshold_multiplier", 1.25))
    return float(np.quantile(np.asarray(distances, dtype=float), q) * multiplier)


def _prototype_or_medoid_assignment(
    *,
    spec: RegimeClustererSpec,
    train_x: np.ndarray,
    train_labels: np.ndarray,
    score_x: np.ndarray,
    started: float,
    policy: str,
    policy_params: Mapping[str, Any],
) -> RegimeClusterAssignmentResult:
    valid = _valid_train_mask(train_labels, exclude_noise=bool(policy_params.get("exclude_noise_from_assignment", True)))
    labels = sorted(int(value) for value in set(train_labels[valid].tolist()))
    unknown_label = int(policy_params.get("unknown_label", -1))
    if not labels:
        return _assignment_result(
            spec=spec,
            labels=np.full(score_x.shape[0], unknown_label, dtype=int),
            probabilities=np.zeros(score_x.shape[0], dtype=float),
            status=ASSIGN_STATUS_ASSIGNED,
            supported=True,
            started=started,
            assignment_policy=policy,
            assignment_metadata={
                "assignment_method": "prototype_or_medoid",
                "confidence_semantics": "1_minus_distance_over_threshold_clipped",
                "unknown_reason": "no_valid_train_clusters",
            },
        )
    prototypes = np.asarray([np.mean(train_x[train_labels == label], axis=0) for label in labels], dtype=float)
    distances = np.linalg.norm(score_x[:, None, :] - prototypes[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(score_x.shape[0]), nearest]
    threshold = _cluster_radius_threshold(train_x, train_labels, policy_params=policy_params)
    assigned = np.asarray([labels[int(idx)] for idx in nearest], dtype=int)
    if threshold is not None and threshold > 0:
        unknown_mask = nearest_distances > threshold
        assigned[unknown_mask] = unknown_label
        confidence = np.clip(1.0 - nearest_distances / threshold, 0.0, 1.0)
    else:
        unknown_mask = np.zeros(score_x.shape[0], dtype=bool)
        confidence = 1.0 / (1.0 + nearest_distances)
    return _assignment_result(
        spec=spec,
        labels=assigned,
        probabilities=confidence,
        status=ASSIGN_STATUS_ASSIGNED,
        supported=True,
        started=started,
        assignment_policy=policy,
        assignment_metadata={
            "assignment_method": "prototype_or_medoid",
            "confidence_semantics": "1_minus_distance_over_threshold_clipped" if threshold else "inverse_distance",
            "assignment_distance_metric": "euclidean",
            "assignment_threshold": threshold,
            "unknown_label": unknown_label,
            "unknown_count": int(np.sum(unknown_mask)),
            "assignment_policy_parameters": dict(policy_params),
            "prototype_labels": labels,
            "assignment_distance_sample": [float(value) for value in nearest_distances[:10].tolist()],
        },
    )


def _nearest_labeled_neighbor_assignment(
    *,
    spec: RegimeClustererSpec,
    train_x: np.ndarray,
    train_labels: np.ndarray,
    score_x: np.ndarray,
    started: float,
    policy: str,
    policy_params: Mapping[str, Any],
) -> RegimeClusterAssignmentResult:
    valid = _valid_train_mask(train_labels, exclude_noise=bool(policy_params.get("exclude_noise_from_assignment", True)))
    unknown_label = int(policy_params.get("unknown_label", -1))
    if not bool(np.any(valid)):
        return _assignment_result(
            spec=spec,
            labels=np.full(score_x.shape[0], unknown_label, dtype=int),
            probabilities=np.zeros(score_x.shape[0], dtype=float),
            status=ASSIGN_STATUS_ASSIGNED,
            supported=True,
            started=started,
            assignment_policy=policy,
            assignment_metadata={
                "assignment_method": "nearest_labeled_neighbor",
                "confidence_semantics": "confidence_unavailable_no_valid_labeled_neighbors",
                "unknown_reason": "no_valid_labeled_train_rows",
            },
        )
    valid_x = np.asarray(train_x[valid], dtype=float)
    valid_labels = np.asarray(train_labels[valid], dtype=int)
    distances = np.linalg.norm(score_x[:, None, :] - valid_x[None, :, :], axis=2)
    k = min(int(policy_params.get("assignment_k", 1)), int(valid_x.shape[0]))
    neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, neighbor_idx, axis=1)
    neighbor_labels = valid_labels[neighbor_idx]
    assigned: list[int] = []
    for labels, dists in zip(neighbor_labels, nearest_distances):
        counts: dict[int, tuple[int, float]] = {}
        for label, dist in zip(labels.tolist(), dists.tolist()):
            current_count, current_dist = counts.get(int(label), (0, 0.0))
            counts[int(label)] = (current_count + 1, current_dist + float(dist))
        winner = sorted(counts.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))[0][0]
        assigned.append(int(winner))
    assigned_arr = np.asarray(assigned, dtype=int)
    min_distances = np.min(nearest_distances, axis=1)
    threshold = _cluster_radius_threshold(train_x, train_labels, policy_params=policy_params)
    if threshold is not None and threshold > 0:
        unknown_mask = min_distances > threshold
        assigned_arr[unknown_mask] = unknown_label
        confidence = np.clip(1.0 - min_distances / threshold, 0.0, 1.0)
    else:
        unknown_mask = np.zeros(score_x.shape[0], dtype=bool)
        confidence = 1.0 / (1.0 + min_distances)
    return _assignment_result(
        spec=spec,
        labels=assigned_arr,
        probabilities=confidence,
        status=ASSIGN_STATUS_ASSIGNED,
        supported=True,
        started=started,
        assignment_policy=policy,
        assignment_metadata={
            "assignment_method": "nearest_labeled_neighbor",
            "confidence_semantics": "1_minus_distance_over_threshold_clipped" if threshold else "inverse_distance",
            "assignment_distance_metric": "euclidean",
            "assignment_k": k,
            "assignment_threshold": threshold,
            "unknown_label": unknown_label,
            "unknown_count": int(np.sum(unknown_mask)),
            "assignment_policy_parameters": dict(policy_params),
            "assignment_distance_sample": [float(value) for value in min_distances[:10].tolist()],
        },
    )


def _patch_hdbscan_check_array_compat() -> None:
    if hdbscan is None:
        return
    try:
        import inspect
        import sklearn.utils.validation as sklearn_validation

        hdbscan_module = hdbscan.hdbscan_  # type: ignore[attr-defined]
        current = getattr(hdbscan_module, "check_array", None)
        if current is None:
            return
        if "force_all_finite" in inspect.signature(current).parameters:
            return

        def check_array_compat(*args: Any, force_all_finite: Any = None, **kwargs: Any) -> Any:
            if force_all_finite is not None and "ensure_all_finite" not in kwargs:
                kwargs["ensure_all_finite"] = force_all_finite
            return sklearn_validation.check_array(*args, **kwargs)

        hdbscan_module.check_array = check_array_compat
    except Exception:
        return


def _common_schema(*names: str) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for name in names:
        if name in {"n_clusters", "n_components"}:
            schema[name] = {"type": "integer", "min": 1, "max": 12, "search": [1, 2, 3, 4, 6, 8]}
        elif name == "random_state":
            schema[name] = {"type": "integer", "default": 17, "search": "fixed_or_seeded"}
        elif name == "min_cluster_size":
            schema[name] = {"type": "integer", "min": 2, "max": 200, "search": [2, 5, 10, 20, 50]}
        elif name == "min_samples":
            schema[name] = {"type": "integer", "min": 1, "max": 100, "search": [1, 2, 5, 10, 20]}
        elif name == "covariance_type":
            schema[name] = {"type": "categorical", "values": ["full", "tied", "diag", "spherical"]}
        elif name == "batch_size":
            schema[name] = {"type": "integer", "min": 16, "max": 4096, "search": [64, 256, 1024]}
        elif name == "xi":
            schema[name] = {"type": "float", "min": 0.01, "max": 0.2, "search": [0.03, 0.05, 0.1]}
    return schema


class KMeansAdapter(RegimeClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="kmeans",
        library_source="sklearn.cluster.KMeans",
        tier="tier_a",
        inductive_classification=INDUCTIVE,
        assignment_policy=ASSIGNMENT_NATIVE_PREDICT,
        supports_assign=True,
        default_hyperparameters={"n_clusters": 3, "n_init": 10, "random_state": 17},
        hyperparameter_schema=_common_schema("n_clusters", "random_state"),
        production_caveats=("spherical-distance bias", "requires explicit k selection"),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        estimator = KMeans(**self.hyperparameters)  # type: ignore[misc]
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=None,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={"inertia": _safe_float(getattr(estimator, "inertia_", None))},
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        return _native_predict(self, x, started=started, policy=assignment_policy)


class MiniBatchKMeansAdapter(KMeansAdapter):
    spec = RegimeClustererSpec(
        family_name="minibatch_kmeans",
        library_source="sklearn.cluster.MiniBatchKMeans",
        tier="tier_a",
        inductive_classification=INDUCTIVE,
        assignment_policy=ASSIGNMENT_NATIVE_PREDICT,
        supports_assign=True,
        default_hyperparameters={"n_clusters": 3, "n_init": 10, "random_state": 17, "batch_size": 256},
        hyperparameter_schema=_common_schema("n_clusters", "batch_size", "random_state"),
        production_caveats=("spherical-distance bias", "mini-batch stochasticity requires seed stability checks"),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        estimator = MiniBatchKMeans(**self.hyperparameters)  # type: ignore[misc]
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=None,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={"inertia": _safe_float(getattr(estimator, "inertia_", None))},
        )


class GaussianMixtureAdapter(RegimeClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="gaussian_mixture",
        library_source="sklearn.mixture.GaussianMixture",
        tier="tier_a",
        inductive_classification=INDUCTIVE,
        assignment_policy=ASSIGNMENT_SCORE_SAMPLES_OR_PROBABILITIES,
        supports_assign=True,
        supports_probabilities=True,
        supports_soft_membership=True,
        default_hyperparameters={"n_components": 3, "covariance_type": "full", "random_state": 17},
        hyperparameter_schema=_common_schema("n_components", "covariance_type", "random_state"),
        production_caveats=("Gaussian component assumptions", "requires explicit component-count search"),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        estimator = GaussianMixture(**self.hyperparameters)  # type: ignore[misc]
        estimator.fit(x)
        labels = estimator.predict(x)
        probabilities = estimator.predict_proba(x)
        self.estimator = estimator
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=probabilities,
            soft_membership=probabilities,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={"bic": _safe_float(estimator.bic(x)), "aic": _safe_float(estimator.aic(x))},
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        return _probability_predict(self, x, started=started, policy=assignment_policy)


class BayesianGaussianMixtureAdapter(GaussianMixtureAdapter):
    spec = RegimeClustererSpec(
        family_name="bayesian_gaussian_mixture",
        library_source="sklearn.mixture.BayesianGaussianMixture",
        tier="tier_a",
        inductive_classification=INDUCTIVE,
        assignment_policy=ASSIGNMENT_SCORE_SAMPLES_OR_PROBABILITIES,
        supports_assign=True,
        supports_probabilities=True,
        supports_soft_membership=True,
        default_hyperparameters={"n_components": 6, "covariance_type": "full", "random_state": 17},
        hyperparameter_schema=_common_schema("n_components", "covariance_type", "random_state"),
        production_caveats=("Bayesian truncation can leave unused components", "requires active-component diagnostics"),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        estimator = BayesianGaussianMixture(**self.hyperparameters)  # type: ignore[misc]
        estimator.fit(x)
        labels = estimator.predict(x)
        probabilities = estimator.predict_proba(x)
        self.estimator = estimator
        weights = getattr(estimator, "weights_", None)
        active_components = None if weights is None else int(np.sum(np.asarray(weights, dtype=float) > 1e-3))
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=probabilities,
            soft_membership=probabilities,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={"active_component_count": active_components},
        )


class HDBSCANAdapter(RegimeClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="hdbscan",
        library_source="hdbscan.HDBSCAN",
        tier="tier_a",
        inductive_classification=TRANSDUCTIVE,
        assignment_policy=ASSIGNMENT_APPROXIMATE_PREDICT,
        dependency_name="hdbscan",
        supports_assign=True,
        supports_refit_recluster=True,
        supports_probabilities=True,
        supports_soft_membership=True,
        supports_noise=True,
        default_hyperparameters={
            "min_cluster_size": 5,
            "min_samples": 1,
            "allow_single_cluster": True,
            "prediction_data": True,
            "cluster_selection_method": "eom",
        },
        hyperparameter_schema=_common_schema("min_cluster_size", "min_samples"),
        production_caveats=(
            "transductive density method",
            "out-of-sample assignment is approximate_predict, not native predict",
            "noise labels are valid model output",
        ),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        _patch_hdbscan_check_array_compat()
        params = {**self.hyperparameters, "prediction_data": True}
        estimator = hdbscan.HDBSCAN(**params)  # type: ignore[union-attr]
        labels = estimator.fit_predict(x)
        probabilities = getattr(estimator, "probabilities_", None)
        soft_membership = None
        all_points_membership_vectors = getattr(hdbscan, "all_points_membership_vectors", None)
        if all_points_membership_vectors is not None:
            try:
                soft_membership = all_points_membership_vectors(estimator)
            except Exception:
                soft_membership = None
        self.estimator = estimator
        persistence = getattr(estimator, "cluster_persistence_", None)
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=probabilities,
            soft_membership=soft_membership,
            hyperparameters=params,
            started=started,
            fit_metadata={
                "cluster_persistence": [] if persistence is None else [float(v) for v in persistence],
                "assignment_method": "hdbscan.approximate_predict",
            },
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        approximate_predict = getattr(hdbscan, "approximate_predict", None)
        if approximate_predict is None:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_UNSUPPORTED,
                supported=False,
                started=started,
                assignment_policy=assignment_policy,
                failure_metadata={
                    "reason_code": "approximate_predict_unavailable",
                    "error": "hdbscan.approximate_predict is unavailable",
                },
            )
        labels, strengths = approximate_predict(self.estimator, x)
        return _assignment_result(
            spec=self.spec,
            labels=np.asarray(labels, dtype=int),
            probabilities=np.asarray(strengths, dtype=float),
            status=ASSIGN_STATUS_ASSIGNED,
            supported=True,
            started=started,
            assignment_policy=assignment_policy,
            assignment_metadata={"assignment_method": "hdbscan.approximate_predict", "approximate_assignment": True},
        )


class OPTICSAdapter(RegimeClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="optics",
        library_source="sklearn.cluster.OPTICS",
        tier="tier_a",
        inductive_classification=TRANSDUCTIVE,
        assignment_policy=ASSIGNMENT_NEAREST_LABELED_NEIGHBOR,
        supports_assign=True,
        supports_refit_recluster=True,
        supports_noise=True,
        default_hyperparameters={
            "min_samples": 5,
            "xi": 0.05,
            "assignment_k": 1,
            "assignment_threshold_quantile": 0.95,
            "assignment_threshold_multiplier": 1.25,
            "unknown_label": -1,
            "exclude_noise_from_assignment": True,
        },
        hyperparameter_schema={
            **_common_schema("min_samples", "xi"),
            "assignment_k": {"type": "integer", "min": 1, "max": 15, "search": [1, 3, 5]},
            "assignment_threshold_quantile": {"type": "float", "min": 0.5, "max": 1.0, "search": [0.8, 0.9, 0.95]},
            "assignment_threshold_multiplier": {"type": "float", "min": 0.5, "max": 3.0, "search": [1.0, 1.25, 1.5]},
        },
        production_caveats=(
            "transductive density fit; score-window assignment uses explicit nearest_labeled_neighbor policy",
            "noise labels are valid output and unknown/out-of-domain rows are assigned unknown_label",
            "assignment distance uses fitted train embedding space without train+validation recluster",
        ),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        policy_params = _assignment_policy_params(self.hyperparameters)
        cache_hit = False
        try:
            from src.regimes.asset_state.optics_reuse import extract_optics_labels, fit_optics_base_cached

            base_fit, cache_hit = fit_optics_base_cached(x, _strip_assignment_params(self.hyperparameters))
            estimator = base_fit.estimator
            labels = extract_optics_labels(base_fit, _strip_assignment_params(self.hyperparameters))
        except Exception:
            estimator = OPTICS(**_strip_assignment_params(self.hyperparameters))  # type: ignore[misc]
            labels = estimator.fit_predict(x)
        self.estimator = estimator
        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_labels = np.asarray(labels, dtype=int).copy()
        reachability = getattr(estimator, "reachability_", None)
        finite = np.asarray([], dtype=float)
        if reachability is not None:
            finite = np.asarray(reachability, dtype=float)
            finite = finite[np.isfinite(finite)]
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=None,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={
                "reachability_mean": float(finite.mean()) if finite.size else None,
                "reachability_p90": float(np.quantile(finite, 0.9)) if finite.size else None,
                "assignment_method": ASSIGNMENT_NEAREST_LABELED_NEIGHBOR,
                "assignment_policy_parameters": policy_params,
                "optics_base_fit_reuse_cache_hit": bool(cache_hit),
                "optics_base_fit_reuse_active": True,
            },
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        if self._train_x is None or self._train_labels is None:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_NOT_FITTED,
                supported=True,
                started=started,
                assignment_policy=assignment_policy,
                failure_metadata={"reason_code": "not_fitted", "error": "train embeddings and labels are unavailable"},
            )
        return _nearest_labeled_neighbor_assignment(
            spec=self.spec,
            train_x=self._train_x,
            train_labels=self._train_labels,
            score_x=x,
            started=started,
            policy=assignment_policy,
            policy_params=_assignment_policy_params(self.hyperparameters),
        )


class AgglomerativeAdapter(RegimeClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="agglomerative",
        library_source="sklearn.cluster.AgglomerativeClustering",
        tier="tier_a",
        inductive_classification=TRANSDUCTIVE,
        assignment_policy=ASSIGNMENT_PROTOTYPE_OR_MEDOID,
        supports_assign=True,
        supports_refit_recluster=True,
        default_hyperparameters={
            "n_clusters": 3,
            "assignment_threshold_quantile": 0.95,
            "assignment_threshold_multiplier": 1.25,
            "unknown_label": -1,
            "exclude_noise_from_assignment": True,
        },
        hyperparameter_schema={
            **_common_schema("n_clusters"),
            "linkage": {"type": "categorical", "values": ["ward", "complete", "average", "single"]},
            "metric": {"type": "categorical", "values": ["euclidean", "manhattan", "cosine"]},
            "assignment_threshold_quantile": {"type": "float", "min": 0.5, "max": 1.0, "search": [0.8, 0.9, 0.95]},
            "assignment_threshold_multiplier": {"type": "float", "min": 0.5, "max": 3.0, "search": [1.0, 1.25, 1.5]},
        },
        production_caveats=(
            "transductive hierarchical fit; score-window assignment uses explicit prototype_or_medoid policy",
            "cluster labels can be fit-population-sensitive, so train-only fit lineage and assignment policy metadata are required",
            "assignment distance uses fitted train embedding space without train+validation recluster",
        ),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        policy_params = _assignment_policy_params(self.hyperparameters)
        params = _strip_assignment_params(self.hyperparameters)
        if params.get("linkage") == "ward":
            params["metric"] = "euclidean"
        estimator = AgglomerativeClustering(**params)  # type: ignore[misc]
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_labels = np.asarray(labels, dtype=int).copy()
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=None,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={
                "assignment_method": ASSIGNMENT_PROTOTYPE_OR_MEDOID,
                "assignment_policy_parameters": policy_params,
            },
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        if self._train_x is None or self._train_labels is None:
            return _assignment_result(
                spec=self.spec,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status=ASSIGN_STATUS_NOT_FITTED,
                supported=True,
                started=started,
                assignment_policy=assignment_policy,
                failure_metadata={"reason_code": "not_fitted", "error": "train embeddings and labels are unavailable"},
            )
        return _prototype_or_medoid_assignment(
            spec=self.spec,
            train_x=self._train_x,
            train_labels=self._train_labels,
            score_x=x,
            started=started,
            policy=assignment_policy,
            policy_params=_assignment_policy_params(self.hyperparameters),
        )


class BirchAdapter(RegimeClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="birch",
        library_source="sklearn.cluster.Birch",
        tier="tier_a",
        inductive_classification=INDUCTIVE,
        assignment_policy=ASSIGNMENT_NATIVE_PREDICT,
        supports_assign=True,
        default_hyperparameters={"n_clusters": 3, "threshold": 0.5, "branching_factor": 50},
        hyperparameter_schema={
            **_common_schema("n_clusters"),
            "threshold": {"type": "float", "min": 0.05, "max": 2.0, "search": [0.25, 0.5, 0.75]},
            "branching_factor": {"type": "integer", "min": 10, "max": 200, "search": [25, 50, 100]},
        },
        production_caveats=("threshold-sensitive CF tree", "requires explicit cluster-count and threshold validation"),
    )

    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        estimator = Birch(**self.hyperparameters)  # type: ignore[misc]
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        subclusters = getattr(estimator, "subcluster_centers_", None)
        return _fit_result(
            spec=self.spec,
            estimator=estimator,
            labels=labels,
            probabilities=None,
            hyperparameters=self.hyperparameters,
            started=started,
            fit_metadata={
                "subcluster_count": 0 if subclusters is None else int(len(subclusters)),
                "threshold": _safe_float(self.hyperparameters.get("threshold")),
            },
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        return _native_predict(self, x, started=started, policy=assignment_policy)


class PlaceholderClustererAdapter(RegimeClustererAdapter):
    def _fit(self, x: np.ndarray, *, started: float) -> RegimeClusterFitResult:
        return _failed_fit_result(
            spec=self.spec,
            status=FIT_STATUS_PLACEHOLDER,
            started=started,
            hyperparameters=self.hyperparameters,
            reason_code="tier_b_placeholder",
            error=f"{self.family_name} is declared as a Tier B placeholder only",
        )

    def _assign(self, x: np.ndarray, *, started: float, assignment_policy: str) -> RegimeClusterAssignmentResult:
        return _assignment_result(
            spec=self.spec,
            labels=np.empty(0, dtype=int),
            probabilities=None,
            status=ASSIGN_STATUS_UNSUPPORTED,
            supported=False,
            started=started,
            assignment_policy=assignment_policy,
            failure_metadata={"reason_code": "tier_b_placeholder", "error": "placeholder adapter has no assignment"},
        )


class SpectralClusteringPlaceholderAdapter(PlaceholderClustererAdapter):
    spec = RegimeClustererSpec(
        family_name="spectral_clustering",
        library_source="sklearn.cluster.SpectralClustering",
        tier="tier_b_placeholder",
        inductive_classification=TRANSDUCTIVE,
        assignment_policy=ASSIGNMENT_FULL_RECLUSTER,
        dependency_name=None,
        supports_fit=False,
        supports_assign=False,
        supports_refit_recluster=False,
        default_hyperparameters={"n_clusters": 3},
        hyperparameter_schema=_common_schema("n_clusters"),
        production_caveats=("placeholder only; transductive graph method has no stable native predict"),
    )


_ADAPTERS: dict[str, type[RegimeClustererAdapter]] = {
    "kmeans": KMeansAdapter,
    "minibatch_kmeans": MiniBatchKMeansAdapter,
    "gaussian_mixture": GaussianMixtureAdapter,
    "bayesian_gaussian_mixture": BayesianGaussianMixtureAdapter,
    "hdbscan": HDBSCANAdapter,
    "optics": OPTICSAdapter,
    "agglomerative": AgglomerativeAdapter,
    "birch": BirchAdapter,
    "spectral_clustering": SpectralClusteringPlaceholderAdapter,
}


def clusterer_adapter_registry(*, include_placeholders: bool = True) -> dict[str, RegimeClustererSpec]:
    families = tuple(_ADAPTERS) if include_placeholders else TIER_A_CLUSTERERS
    return {family: _ADAPTERS[family].spec for family in families}


def tier_a_clusterer_families() -> tuple[str, ...]:
    return TIER_A_CLUSTERERS


def tier_b_placeholder_families() -> tuple[str, ...]:
    return TIER_B_PLACEHOLDER_CLUSTERERS


def build_regime_clusterer_adapter(family_name: str, **hyperparameters: Any) -> RegimeClustererAdapter:
    key = _normalize_token(family_name, field_name="family name")
    try:
        cls = _ADAPTERS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported Regime clusterer family {family_name!r}; expected one of: {valid}") from exc
    return cls(**hyperparameters)


def _shared_policy_for_spec(spec: RegimeClustererSpec) -> str:
    if spec.assignment_policy == ASSIGNMENT_APPROXIMATE_PREDICT:
        return AssignmentPolicy.APPROXIMATE_PREDICT.value
    if spec.assignment_policy == ASSIGNMENT_FULL_RECLUSTER:
        return AssignmentPolicy.FULL_RECLUSTER.value
    if spec.assignment_policy == ASSIGNMENT_NEAREST_LABELED_NEIGHBOR:
        return AssignmentPolicy.NEAREST_LABELED_NEIGHBOR.value
    if spec.assignment_policy == ASSIGNMENT_PROTOTYPE_OR_MEDOID:
        return AssignmentPolicy.PROTOTYPE_OR_MEDOID.value
    return AssignmentPolicy.NATIVE_PREDICT.value


def _shared_capabilities_for_spec(
    spec: RegimeClustererSpec,
    *,
    assignment_policies: Sequence[str | AssignmentPolicy] | None = None,
) -> ClustererCapabilities:
    default_policy = _shared_policy_for_spec(spec)
    policies = tuple(assignment_policies or (default_policy,))
    return ClustererCapabilities(
        family_name=spec.family_name,
        assignment_policies=policies,
        default_assignment_policy=default_policy,
        inductive_behavior=spec.inductive_classification,
        supports_fit=bool(spec.supports_fit),
        supports_assign=bool(spec.supports_assign),
        supports_refit_or_recluster=bool(spec.supports_refit_recluster),
        supports_soft_membership=bool(spec.supports_soft_membership or spec.supports_probabilities),
        supports_noise_labels=bool(spec.supports_noise),
        deterministic=False if spec.family_name in {"minibatch_kmeans"} else True,
        dependency_name=spec.dependency_name,
        implementation_status="implemented",
    )


def _failure_from_legacy(payload: Mapping[str, Any] | None) -> ClustererFailureMetadata | None:
    data = dict(payload or {})
    if not data:
        return None
    return ClustererFailureMetadata(
        reason_code=str(data.get("reason_code") or "clusterer_failure"),
        message=str(data.get("error") or data.get("message") or data),
        recoverable=str(data.get("reason_code")) in {"dependency_unavailable", "unsupported_assignment_policy"},
        details={key: value for key, value in data.items() if key not in {"reason_code", "error", "message"}},
    )


def _runtime_from_legacy(
    *,
    family_name: str,
    operation: str,
    x: np.ndarray,
    assignment_policy: str | AssignmentPolicy | None,
    legacy_runtime: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> ClustererRuntimeMetadata:
    runtime = dict(legacy_runtime)
    elapsed = runtime.get("fit_time_s", runtime.get("assign_time_s", runtime.get("elapsed_s", 0.0)))
    return ClustererRuntimeMetadata(
        family_name=family_name,
        operation=operation,
        row_count=int(x.shape[0]),
        feature_count=int(x.shape[1]),
        elapsed_s=float(elapsed or 0.0),
        assignment_policy=assignment_policy,
        metadata={**runtime, **dict(metadata)},
    )


def _shared_soft_membership(
    soft_membership: np.ndarray | Sequence[float] | None,
    probabilities: np.ndarray | Sequence[float] | None,
) -> np.ndarray | None:
    if soft_membership is not None:
        soft = np.asarray(soft_membership, dtype=float)
        if soft.ndim == 2:
            return soft
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim == 2:
            return probs
    return None


class _SharedTierAClustererAdapter(BaseClustererAdapter):
    legacy_family_name: str

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.legacy_adapter = build_regime_clusterer_adapter(self.legacy_family_name, **hyperparameters)

    def _fit_matrix(self, x: np.ndarray, *, started: float) -> ClustererFitResult:
        legacy = self.legacy_adapter.fit(x)
        return self._fit_from_legacy(legacy, x=x, operation="fit", assignment_policy=None)

    def _assign_matrix(
        self,
        x: np.ndarray,
        *,
        assignment_policy: str,
        started: float,
    ) -> ClustererAssignmentResult:
        legacy = self.legacy_adapter.assign(x)
        return self._assignment_from_legacy(legacy, x=x, assignment_policy=assignment_policy)

    def _refit_or_recluster_matrix(
        self,
        x: np.ndarray,
        *,
        assignment_policy: str,
        started: float,
    ) -> ClustererFitResult:
        legacy = self.legacy_adapter.refit_recluster(x)
        return self._fit_from_legacy(
            legacy,
            x=x,
            operation="refit_or_recluster",
            assignment_policy=assignment_policy,
        )

    def _fit_from_legacy(
        self,
        legacy: RegimeClusterFitResult,
        *,
        x: np.ndarray,
        operation: str,
        assignment_policy: str | AssignmentPolicy | None,
    ) -> ClustererFitResult:
        failure = _failure_from_legacy(legacy.failure_metadata)
        return ClustererFitResult(
            family_name=legacy.family_name,
            status=legacy.status,
            labels=legacy.labels,
            soft_membership=_shared_soft_membership(legacy.soft_membership, legacy.probabilities),
            runtime_metadata=_runtime_from_legacy(
                family_name=legacy.family_name,
                operation=operation,
                x=x,
                assignment_policy=assignment_policy,
                legacy_runtime=legacy.runtime_metadata,
                metadata=legacy.fit_metadata,
            ),
            capabilities=self.capabilities,
            failure_metadata=failure,
            metadata={
                "legacy_status": legacy.status,
                "hyperparameters": legacy.hyperparameters,
                "probabilities": None if legacy.probabilities is None else legacy.probabilities.tolist(),
                "confidence": None if legacy.confidence is None else legacy.confidence.tolist(),
                "noise_mask": legacy.noise_mask.tolist(),
                "outcome": legacy.fit_metadata,
            },
        )

    def _assignment_from_legacy(
        self,
        legacy: RegimeClusterAssignmentResult,
        *,
        x: np.ndarray,
        assignment_policy: str | AssignmentPolicy,
    ) -> ClustererAssignmentResult:
        failure = _failure_from_legacy(legacy.failure_metadata)
        return ClustererAssignmentResult(
            family_name=legacy.family_name,
            assignment_policy=assignment_policy,
            status=legacy.status,
            labels=legacy.labels,
            soft_membership=_shared_soft_membership(legacy.soft_membership, legacy.probabilities),
            runtime_metadata=_runtime_from_legacy(
                family_name=legacy.family_name,
                operation="assign",
                x=x,
                assignment_policy=assignment_policy,
                legacy_runtime=legacy.runtime_metadata,
                metadata=legacy.assignment_metadata,
            ),
            capabilities=self.capabilities,
            failure_metadata=failure,
            metadata={
                "legacy_status": legacy.status,
                "supported": legacy.supported,
                "probabilities": None if legacy.probabilities is None else legacy.probabilities.tolist(),
                "confidence": None if legacy.confidence is None else legacy.confidence.tolist(),
                "noise_mask": legacy.noise_mask.tolist(),
                "outcome": legacy.assignment_metadata,
            },
        )


class SharedKMeansAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "kmeans"
    capabilities = _shared_capabilities_for_spec(KMeansAdapter.spec)


class SharedMiniBatchKMeansAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "minibatch_kmeans"
    capabilities = _shared_capabilities_for_spec(MiniBatchKMeansAdapter.spec)


class SharedGaussianMixtureAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "gaussian_mixture"
    capabilities = _shared_capabilities_for_spec(GaussianMixtureAdapter.spec)


class SharedBayesianGaussianMixtureAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "bayesian_gaussian_mixture"
    capabilities = _shared_capabilities_for_spec(BayesianGaussianMixtureAdapter.spec)


class SharedHDBSCANAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "hdbscan"
    capabilities = _shared_capabilities_for_spec(HDBSCANAdapter.spec)


class SharedOPTICSAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "optics"
    capabilities = _shared_capabilities_for_spec(OPTICSAdapter.spec)


class SharedAgglomerativeAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "agglomerative"
    capabilities = _shared_capabilities_for_spec(AgglomerativeAdapter.spec)


class SharedBirchAdapter(_SharedTierAClustererAdapter):
    legacy_family_name = "birch"
    capabilities = _shared_capabilities_for_spec(BirchAdapter.spec)


SHARED_TIER_A_CLUSTERER_ADAPTERS: tuple[type[BaseClustererAdapter], ...] = (
    SharedKMeansAdapter,
    SharedMiniBatchKMeansAdapter,
    SharedGaussianMixtureAdapter,
    SharedBayesianGaussianMixtureAdapter,
    SharedHDBSCANAdapter,
    SharedOPTICSAdapter,
    SharedAgglomerativeAdapter,
    SharedBirchAdapter,
)


def shared_tier_a_clusterer_adapter_types() -> tuple[type[BaseClustererAdapter], ...]:
    return SHARED_TIER_A_CLUSTERER_ADAPTERS


__all__ = [
    "ASSIGNMENT_APPROXIMATE_PREDICT",
    "ASSIGNMENT_FULL_RECLUSTER",
    "ASSIGNMENT_NATIVE_PREDICT",
    "ASSIGNMENT_SCHEDULED_REFIT",
    "ASSIGNMENT_SCORE_SAMPLES_OR_PROBABILITIES",
    "ASSIGN_STATUS_ASSIGNED",
    "ASSIGN_STATUS_DEPENDENCY_UNAVAILABLE",
    "ASSIGN_STATUS_FAILED",
    "ASSIGN_STATUS_NOT_FITTED",
    "ASSIGN_STATUS_UNSUPPORTED",
    "FIT_STATUS_DEPENDENCY_UNAVAILABLE",
    "FIT_STATUS_FAILED",
    "FIT_STATUS_FITTED",
    "FIT_STATUS_INVALID_INPUT",
    "FIT_STATUS_PLACEHOLDER",
    "INDUCTIVE",
    "REGIME_ASSIGNMENT_POLICIES",
    "REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION",
    "TIER_A_CLUSTERERS",
    "TIER_B_PLACEHOLDER_CLUSTERERS",
    "TRANSDUCTIVE",
    "AgglomerativeAdapter",
    "BayesianGaussianMixtureAdapter",
    "BirchAdapter",
    "GaussianMixtureAdapter",
    "HDBSCANAdapter",
    "KMeansAdapter",
    "MiniBatchKMeansAdapter",
    "OPTICSAdapter",
    "PlaceholderClustererAdapter",
    "RegimeClusterAssignmentResult",
    "RegimeClusterFitResult",
    "RegimeClustererAdapter",
    "RegimeClustererSpec",
    "SpectralClusteringPlaceholderAdapter",
    "SHARED_TIER_A_CLUSTERER_ADAPTERS",
    "SharedAgglomerativeAdapter",
    "SharedBirchAdapter",
    "SharedBayesianGaussianMixtureAdapter",
    "SharedGaussianMixtureAdapter",
    "SharedHDBSCANAdapter",
    "SharedKMeansAdapter",
    "SharedMiniBatchKMeansAdapter",
    "SharedOPTICSAdapter",
    "build_regime_clusterer_adapter",
    "clusterer_adapter_registry",
    "shared_tier_a_clusterer_adapter_types",
    "tier_a_clusterer_families",
    "tier_b_placeholder_families",
]
