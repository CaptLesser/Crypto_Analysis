from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_non_empty_string, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_known_fields, require_json_object, to_jsonable


class AssignmentPolicy(str, Enum):
    NATIVE_PREDICT = "native_predict"
    APPROXIMATE_PREDICT = "approximate_predict"
    NEAREST_LABELED_NEIGHBOR = "nearest_labeled_neighbor"
    PROTOTYPE_OR_MEDOID = "prototype_or_medoid"
    SCHEDULED_REFIT = "scheduled_refit"
    FULL_RECLUSTER = "full_recluster"


ASSIGNMENT_POLICY_VALUES: tuple[str, ...] = tuple(policy.value for policy in AssignmentPolicy)

FIT_STATUS_FITTED = "fitted"
FIT_STATUS_FAILED = "failed"
FIT_STATUS_UNSUPPORTED = "unsupported"
ASSIGN_STATUS_ASSIGNED = "assigned"
ASSIGN_STATUS_FAILED = "failed"
ASSIGN_STATUS_UNSUPPORTED = "unsupported"


def normalize_assignment_policy(value: str | AssignmentPolicy) -> str:
    text = require_non_empty_string(value.value if isinstance(value, AssignmentPolicy) else value, field_name="assignment policy")
    text = text.lower()
    if text not in ASSIGNMENT_POLICY_VALUES:
        valid = ", ".join(ASSIGNMENT_POLICY_VALUES)
        raise ValueError(f"Unsupported Regime assignment policy {text!r}; expected one of: {valid}")
    return text


def _string_tuple(values: Sequence[object], *, field_name: str, require_non_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Regime clusterer {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not out:
        raise ValueError(f"Regime clusterer {field_name} must include at least one value")
    return out


def _policy_tuple(values: Sequence[str | AssignmentPolicy], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Regime clusterer {field_name} must be a sequence of assignment policies")
    out = tuple(dict.fromkeys(normalize_assignment_policy(value) for value in values))
    if not out:
        raise ValueError(f"Regime clusterer {field_name} must include at least one policy")
    return out


def _empty_labels() -> tuple[int, ...]:
    return ()


def _coerce_labels(labels: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(labels, dtype=int)
    if arr.ndim != 1:
        raise ValueError("Regime clusterer labels must be one-dimensional")
    return tuple(int(value) for value in arr.tolist())


def _coerce_soft_membership(value: Sequence[Sequence[float]] | np.ndarray | None, *, row_count: int) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Regime clusterer soft_membership must be two-dimensional")
    if int(arr.shape[0]) != int(row_count):
        raise ValueError("Regime clusterer soft_membership row count must match labels")
    return tuple(tuple(float(item) for item in row) for row in arr.tolist())


def _coerce_feature_matrix(x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Regime clusterer feature matrix must be two-dimensional")
    if arr.shape[0] == 0:
        raise ValueError("Regime clusterer feature matrix must include at least one row")
    if arr.shape[1] == 0:
        raise ValueError("Regime clusterer feature matrix must include at least one feature")
    if not bool(np.isfinite(arr).all()):
        raise ValueError("Regime clusterer feature matrix must contain only finite values")
    return arr


@dataclass(frozen=True)
class ClustererCapabilities:
    family_name: str
    assignment_policies: Sequence[str | AssignmentPolicy]
    default_assignment_policy: str | AssignmentPolicy
    inductive_behavior: str = "inductive"
    supports_fit: bool = True
    supports_assign: bool = True
    supports_refit_or_recluster: bool = False
    supports_soft_membership: bool = False
    supports_noise_labels: bool = False
    deterministic: bool = True
    dependency_name: str | None = None
    implementation_status: str = "fixture"
    schema_version: int = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family_name = require_non_empty_string(self.family_name, field_name="clusterer family name").lower()
        policies = _policy_tuple(self.assignment_policies, field_name="assignment_policies")
        default_policy = normalize_assignment_policy(self.default_assignment_policy)
        if default_policy not in policies:
            raise ValueError("Regime clusterer default_assignment_policy must be included in assignment_policies")
        inductive_behavior = require_non_empty_string(self.inductive_behavior, field_name="inductive_behavior").lower()
        if inductive_behavior not in {"inductive", "transductive"}:
            raise ValueError("Regime clusterer inductive_behavior must be inductive or transductive")
        implementation_status = require_non_empty_string(self.implementation_status, field_name="implementation_status").lower()
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family_name)
        object.__setattr__(self, "assignment_policies", policies)
        object.__setattr__(self, "default_assignment_policy", default_policy)
        object.__setattr__(self, "inductive_behavior", inductive_behavior)
        object.__setattr__(self, "supports_fit", bool(self.supports_fit))
        object.__setattr__(self, "supports_assign", bool(self.supports_assign))
        object.__setattr__(self, "supports_refit_or_recluster", bool(self.supports_refit_or_recluster))
        object.__setattr__(self, "supports_soft_membership", bool(self.supports_soft_membership))
        object.__setattr__(self, "supports_noise_labels", bool(self.supports_noise_labels))
        object.__setattr__(self, "deterministic", bool(self.deterministic))
        object.__setattr__(self, "dependency_name", None if self.dependency_name is None else str(self.dependency_name).strip() or None)
        object.__setattr__(self, "implementation_status", implementation_status)

    def supports_policy(self, policy: str | AssignmentPolicy) -> bool:
        return normalize_assignment_policy(policy) in self.assignment_policies

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "assignment_policies": list(self.assignment_policies),
            "default_assignment_policy": self.default_assignment_policy,
            "inductive_behavior": self.inductive_behavior,
            "supports_fit": bool(self.supports_fit),
            "supports_assign": bool(self.supports_assign),
            "supports_refit_or_recluster": bool(self.supports_refit_or_recluster),
            "supports_soft_membership": bool(self.supports_soft_membership),
            "supports_noise_labels": bool(self.supports_noise_labels),
            "deterministic": bool(self.deterministic),
            "dependency_name": self.dependency_name,
            "implementation_status": self.implementation_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClustererCapabilities":
        obj = require_known_fields(
            payload,
            required={"schema_version", "family_name", "assignment_policies", "default_assignment_policy"},
            optional={
                "supports_fit",
                "inductive_behavior",
                "supports_assign",
                "supports_refit_or_recluster",
                "supports_soft_membership",
                "supports_noise_labels",
                "deterministic",
                "dependency_name",
                "implementation_status",
            },
            context="Regime ClustererCapabilities",
        )
        return cls(
            schema_version=obj["schema_version"],
            family_name=obj["family_name"],
            assignment_policies=obj["assignment_policies"],
            default_assignment_policy=obj["default_assignment_policy"],
            inductive_behavior=obj.get("inductive_behavior", "inductive"),
            supports_fit=bool(obj.get("supports_fit", True)),
            supports_assign=bool(obj.get("supports_assign", True)),
            supports_refit_or_recluster=bool(obj.get("supports_refit_or_recluster", False)),
            supports_soft_membership=bool(obj.get("supports_soft_membership", False)),
            supports_noise_labels=bool(obj.get("supports_noise_labels", False)),
            deterministic=bool(obj.get("deterministic", True)),
            dependency_name=obj.get("dependency_name"),
            implementation_status=obj.get("implementation_status", "fixture"),
        )

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "ClustererCapabilities":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime ClustererCapabilities JSON"))


@dataclass(frozen=True)
class ClustererRuntimeMetadata:
    family_name: str
    operation: str
    row_count: int
    feature_count: int
    elapsed_s: float
    assignment_policy: str | AssignmentPolicy | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family_name = require_non_empty_string(self.family_name, field_name="clusterer family name").lower()
        operation = require_non_empty_string(self.operation, field_name="clusterer operation").lower()
        row_count = int(self.row_count)
        feature_count = int(self.feature_count)
        elapsed_s = float(self.elapsed_s)
        if row_count < 0 or feature_count < 0:
            raise ValueError("Regime clusterer runtime row_count and feature_count must be non-negative")
        if elapsed_s < 0.0:
            raise ValueError("Regime clusterer runtime elapsed_s must be non-negative")
        policy = None if self.assignment_policy is None else normalize_assignment_policy(self.assignment_policy)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family_name)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "feature_count", feature_count)
        object.__setattr__(self, "elapsed_s", elapsed_s)
        object.__setattr__(self, "assignment_policy", policy)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "operation": self.operation,
            "row_count": int(self.row_count),
            "feature_count": int(self.feature_count),
            "elapsed_s": float(self.elapsed_s),
            "assignment_policy": self.assignment_policy,
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ClustererFailureMetadata:
    reason_code: str
    message: str
    recoverable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        reason_code = require_non_empty_string(self.reason_code, field_name="failure reason_code")
        message = require_non_empty_string(self.message, field_name="failure message")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "recoverable", bool(self.recoverable))
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "reason_code": self.reason_code,
            "message": self.message,
            "recoverable": bool(self.recoverable),
            "details": to_jsonable(self.details),
        }


@dataclass(frozen=True)
class ClustererFitResult:
    family_name: str
    status: str
    labels: Sequence[int]
    runtime_metadata: ClustererRuntimeMetadata | Mapping[str, Any]
    capabilities: ClustererCapabilities | Mapping[str, Any]
    soft_membership: Sequence[Sequence[float]] | np.ndarray | None = None
    failure_metadata: ClustererFailureMetadata | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family_name = require_non_empty_string(self.family_name, field_name="clusterer family name").lower()
        status = require_non_empty_string(self.status, field_name="fit status").lower()
        labels = _coerce_labels(self.labels)
        soft_membership = _coerce_soft_membership(self.soft_membership, row_count=len(labels))
        runtime = self.runtime_metadata if isinstance(self.runtime_metadata, ClustererRuntimeMetadata) else ClustererRuntimeMetadata(**self.runtime_metadata)
        capabilities = self.capabilities if isinstance(self.capabilities, ClustererCapabilities) else ClustererCapabilities.from_dict(self.capabilities)
        failure = None
        if self.failure_metadata is not None:
            failure = (
                self.failure_metadata
                if isinstance(self.failure_metadata, ClustererFailureMetadata)
                else ClustererFailureMetadata(**self.failure_metadata)
            )
        if status in {FIT_STATUS_FAILED, FIT_STATUS_UNSUPPORTED} and failure is None:
            raise ValueError("Regime failed or unsupported fit results require failure_metadata")
        if status == FIT_STATUS_FITTED and failure is not None:
            raise ValueError("Regime fitted results cannot carry failure_metadata")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family_name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "soft_membership", soft_membership)
        object.__setattr__(self, "runtime_metadata", runtime)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "failure_metadata", failure)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "status": self.status,
            "labels": list(self.labels),
            "soft_membership": None if self.soft_membership is None else [list(row) for row in self.soft_membership],
            "runtime_metadata": self.runtime_metadata.as_dict(),
            "capabilities": self.capabilities.as_dict(),
            "failure_metadata": None if self.failure_metadata is None else self.failure_metadata.as_dict(),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class ClustererAssignmentResult:
    family_name: str
    assignment_policy: str | AssignmentPolicy
    status: str
    labels: Sequence[int]
    runtime_metadata: ClustererRuntimeMetadata | Mapping[str, Any]
    capabilities: ClustererCapabilities | Mapping[str, Any]
    soft_membership: Sequence[Sequence[float]] | np.ndarray | None = None
    failure_metadata: ClustererFailureMetadata | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family_name = require_non_empty_string(self.family_name, field_name="clusterer family name").lower()
        policy = normalize_assignment_policy(self.assignment_policy)
        status = require_non_empty_string(self.status, field_name="assignment status").lower()
        labels = _coerce_labels(self.labels)
        soft_membership = _coerce_soft_membership(self.soft_membership, row_count=len(labels))
        runtime = self.runtime_metadata if isinstance(self.runtime_metadata, ClustererRuntimeMetadata) else ClustererRuntimeMetadata(**self.runtime_metadata)
        capabilities = self.capabilities if isinstance(self.capabilities, ClustererCapabilities) else ClustererCapabilities.from_dict(self.capabilities)
        failure = None
        if self.failure_metadata is not None:
            failure = (
                self.failure_metadata
                if isinstance(self.failure_metadata, ClustererFailureMetadata)
                else ClustererFailureMetadata(**self.failure_metadata)
            )
        if status in {ASSIGN_STATUS_FAILED, ASSIGN_STATUS_UNSUPPORTED} and failure is None:
            raise ValueError("Regime failed or unsupported assignment results require failure_metadata")
        if status == ASSIGN_STATUS_ASSIGNED and failure is not None:
            raise ValueError("Regime assigned results cannot carry failure_metadata")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family_name)
        object.__setattr__(self, "assignment_policy", policy)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "soft_membership", soft_membership)
        object.__setattr__(self, "runtime_metadata", runtime)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "failure_metadata", failure)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "assignment_policy": self.assignment_policy,
            "status": self.status,
            "labels": list(self.labels),
            "soft_membership": None if self.soft_membership is None else [list(row) for row in self.soft_membership],
            "runtime_metadata": self.runtime_metadata.as_dict(),
            "capabilities": self.capabilities.as_dict(),
            "failure_metadata": None if self.failure_metadata is None else self.failure_metadata.as_dict(),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


class BaseClustererAdapter(ABC):
    capabilities: ClustererCapabilities

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters = dict(hyperparameters)
        self._fitted = False

    @property
    def family_name(self) -> str:
        return self.capabilities.family_name

    def report_capabilities(self) -> ClustererCapabilities:
        return self.capabilities

    def fit(self, x: Sequence[Sequence[float]] | np.ndarray) -> ClustererFitResult:
        if not self.capabilities.supports_fit:
            return self._fit_failure(
                operation="fit",
                x=None,
                reason_code="fit_unsupported",
                message=f"Clusterer family {self.family_name!r} does not support fit",
            )
        try:
            matrix = _coerce_feature_matrix(x)
        except ValueError as exc:
            return self._fit_failure(
                operation="fit",
                x=None,
                reason_code="invalid_feature_matrix",
                message=str(exc),
            )
        started = time.perf_counter()
        try:
            result = self._fit_matrix(matrix, started=started)
            self._fitted = result.status == FIT_STATUS_FITTED
            return result
        except Exception as exc:
            return self._fit_failure(
                operation="fit",
                x=matrix,
                reason_code="fit_failed",
                message=str(exc),
                started=started,
            )

    def assign(
        self,
        x: Sequence[Sequence[float]] | np.ndarray,
        *,
        assignment_policy: str | AssignmentPolicy | None = None,
    ) -> ClustererAssignmentResult:
        policy = normalize_assignment_policy(assignment_policy or self.capabilities.default_assignment_policy)
        try:
            matrix = _coerce_feature_matrix(x)
        except ValueError as exc:
            return self._assignment_failure(
                policy=policy,
                operation="assign",
                x=None,
                reason_code="invalid_feature_matrix",
                message=str(exc),
                status=ASSIGN_STATUS_FAILED,
            )
        if not self.capabilities.supports_assign:
            return self._assignment_failure(
                policy=policy,
                operation="assign",
                x=matrix,
                reason_code="assign_unsupported",
                message=f"Clusterer family {self.family_name!r} does not support assign",
            )
        if not self.capabilities.supports_policy(policy):
            supported = ", ".join(self.capabilities.assignment_policies)
            return self._assignment_failure(
                policy=policy,
                operation="assign",
                x=matrix,
                reason_code="unsupported_assignment_policy",
                message=f"Clusterer family {self.family_name!r} does not support assignment policy {policy!r}; supported: {supported}",
            )
        if not self._fitted:
            return self._assignment_failure(
                policy=policy,
                operation="assign",
                x=matrix,
                reason_code="not_fitted",
                message=f"Clusterer family {self.family_name!r} must be fit before assign",
                status=ASSIGN_STATUS_FAILED,
            )
        started = time.perf_counter()
        try:
            return self._assign_matrix(matrix, assignment_policy=policy, started=started)
        except Exception as exc:
            return self._assignment_failure(
                policy=policy,
                operation="assign",
                x=matrix,
                reason_code="assign_failed",
                message=str(exc),
                started=started,
                status=ASSIGN_STATUS_FAILED,
            )

    def refit_or_recluster(
        self,
        x: Sequence[Sequence[float]] | np.ndarray,
        *,
        assignment_policy: str | AssignmentPolicy = AssignmentPolicy.FULL_RECLUSTER,
    ) -> ClustererFitResult:
        policy = normalize_assignment_policy(assignment_policy)
        if not self.capabilities.supports_refit_or_recluster:
            return self._fit_failure(
                operation="refit_or_recluster",
                x=None,
                reason_code="refit_or_recluster_unsupported",
                message=f"Clusterer family {self.family_name!r} does not support refit_or_recluster",
                assignment_policy=policy,
                status=FIT_STATUS_UNSUPPORTED,
            )
        try:
            matrix = _coerce_feature_matrix(x)
        except ValueError as exc:
            return self._fit_failure(
                operation="refit_or_recluster",
                x=None,
                reason_code="invalid_feature_matrix",
                message=str(exc),
                assignment_policy=policy,
            )
        started = time.perf_counter()
        try:
            result = self._refit_or_recluster_matrix(matrix, assignment_policy=policy, started=started)
            self._fitted = result.status == FIT_STATUS_FITTED
            return result
        except Exception as exc:
            return self._fit_failure(
                operation="refit_or_recluster",
                x=matrix,
                reason_code="refit_or_recluster_failed",
                message=str(exc),
                assignment_policy=policy,
                started=started,
            )

    @abstractmethod
    def _fit_matrix(self, x: np.ndarray, *, started: float) -> ClustererFitResult:
        raise NotImplementedError

    @abstractmethod
    def _assign_matrix(
        self,
        x: np.ndarray,
        *,
        assignment_policy: str,
        started: float,
    ) -> ClustererAssignmentResult:
        raise NotImplementedError

    def _refit_or_recluster_matrix(
        self,
        x: np.ndarray,
        *,
        assignment_policy: str,
        started: float,
    ) -> ClustererFitResult:
        return self._fit_matrix(x, started=started)

    def _runtime(
        self,
        *,
        operation: str,
        x: np.ndarray | None,
        started: float | None = None,
        assignment_policy: str | AssignmentPolicy | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClustererRuntimeMetadata:
        if x is None:
            row_count = 0
            feature_count = 0
        else:
            row_count = int(x.shape[0])
            feature_count = int(x.shape[1])
        elapsed_s = 0.0 if started is None else max(0.0, float(time.perf_counter() - started))
        return ClustererRuntimeMetadata(
            family_name=self.family_name,
            operation=operation,
            row_count=row_count,
            feature_count=feature_count,
            elapsed_s=elapsed_s,
            assignment_policy=assignment_policy,
            metadata=metadata or {},
        )

    def _fit_success(
        self,
        *,
        labels: Sequence[int] | np.ndarray,
        x: np.ndarray,
        started: float,
        operation: str = "fit",
        assignment_policy: str | AssignmentPolicy | None = None,
        soft_membership: Sequence[Sequence[float]] | np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClustererFitResult:
        return ClustererFitResult(
            family_name=self.family_name,
            status=FIT_STATUS_FITTED,
            labels=labels,
            soft_membership=soft_membership,
            runtime_metadata=self._runtime(
                operation=operation,
                x=x,
                started=started,
                assignment_policy=assignment_policy,
            ),
            capabilities=self.capabilities,
            metadata=metadata or {},
        )

    def _assignment_success(
        self,
        *,
        labels: Sequence[int] | np.ndarray,
        x: np.ndarray,
        started: float,
        assignment_policy: str | AssignmentPolicy,
        soft_membership: Sequence[Sequence[float]] | np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClustererAssignmentResult:
        return ClustererAssignmentResult(
            family_name=self.family_name,
            assignment_policy=assignment_policy,
            status=ASSIGN_STATUS_ASSIGNED,
            labels=labels,
            soft_membership=soft_membership,
            runtime_metadata=self._runtime(
                operation="assign",
                x=x,
                started=started,
                assignment_policy=assignment_policy,
            ),
            capabilities=self.capabilities,
            metadata=metadata or {},
        )

    def _fit_failure(
        self,
        *,
        operation: str,
        x: np.ndarray | None,
        reason_code: str,
        message: str,
        assignment_policy: str | AssignmentPolicy | None = None,
        started: float | None = None,
        status: str = FIT_STATUS_FAILED,
        details: Mapping[str, Any] | None = None,
    ) -> ClustererFitResult:
        return ClustererFitResult(
            family_name=self.family_name,
            status=status,
            labels=_empty_labels(),
            runtime_metadata=self._runtime(
                operation=operation,
                x=x,
                started=started,
                assignment_policy=assignment_policy,
            ),
            capabilities=self.capabilities,
            failure_metadata=ClustererFailureMetadata(
                reason_code=reason_code,
                message=message,
                recoverable=status == FIT_STATUS_UNSUPPORTED,
                details=details or {},
            ),
        )

    def _assignment_failure(
        self,
        *,
        policy: str | AssignmentPolicy,
        operation: str,
        x: np.ndarray | None,
        reason_code: str,
        message: str,
        started: float | None = None,
        status: str = ASSIGN_STATUS_UNSUPPORTED,
        details: Mapping[str, Any] | None = None,
    ) -> ClustererAssignmentResult:
        return ClustererAssignmentResult(
            family_name=self.family_name,
            assignment_policy=policy,
            status=status,
            labels=_empty_labels(),
            runtime_metadata=self._runtime(
                operation=operation,
                x=x,
                started=started,
                assignment_policy=policy,
            ),
            capabilities=self.capabilities,
            failure_metadata=ClustererFailureMetadata(
                reason_code=reason_code,
                message=message,
                recoverable=status == ASSIGN_STATUS_UNSUPPORTED,
                details=details or {},
            ),
        )


class DummyClustererAdapter(BaseClustererAdapter):
    capabilities = ClustererCapabilities(
        family_name="dummy_threshold",
        assignment_policies=(AssignmentPolicy.NATIVE_PREDICT, AssignmentPolicy.FULL_RECLUSTER),
        default_assignment_policy=AssignmentPolicy.NATIVE_PREDICT,
        inductive_behavior="inductive",
        supports_fit=True,
        supports_assign=True,
        supports_refit_or_recluster=True,
        supports_soft_membership=True,
        deterministic=True,
        dependency_name=None,
        implementation_status="fixture",
    )

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.threshold_: float | None = None

    def _labels_and_membership(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.threshold_ is None:
            raise ValueError("dummy adapter is not fit")
        labels = (x[:, 0] > float(self.threshold_)).astype(int)
        distance = np.abs(x[:, 0] - float(self.threshold_))
        confidence = np.clip(distance / (distance.max() if distance.max() > 0 else 1.0), 0.0, 1.0)
        soft = np.column_stack((1.0 - labels * confidence, labels * confidence))
        return labels, soft

    def _fit_matrix(self, x: np.ndarray, *, started: float) -> ClustererFitResult:
        self.threshold_ = float(np.median(x[:, 0]))
        labels, soft = self._labels_and_membership(x)
        return self._fit_success(
            labels=labels,
            soft_membership=soft,
            x=x,
            started=started,
            metadata={
                "threshold": self.threshold_,
                "cluster_count": int(len(set(labels.tolist()))),
                "hyperparameters": to_jsonable(self.hyperparameters),
            },
        )

    def _assign_matrix(
        self,
        x: np.ndarray,
        *,
        assignment_policy: str,
        started: float,
    ) -> ClustererAssignmentResult:
        labels, soft = self._labels_and_membership(x)
        return self._assignment_success(
            labels=labels,
            soft_membership=soft,
            x=x,
            started=started,
            assignment_policy=assignment_policy,
            metadata={"threshold": self.threshold_},
        )

    def _refit_or_recluster_matrix(
        self,
        x: np.ndarray,
        *,
        assignment_policy: str,
        started: float,
    ) -> ClustererFitResult:
        self.threshold_ = float(np.median(x[:, 0]))
        labels, soft = self._labels_and_membership(x)
        return self._fit_success(
            labels=labels,
            soft_membership=soft,
            x=x,
            started=started,
            operation="refit_or_recluster",
            assignment_policy=assignment_policy,
            metadata={"threshold": self.threshold_, "refit_or_recluster": True},
        )


__all__ = [
    "ASSIGNMENT_POLICY_VALUES",
    "ASSIGN_STATUS_ASSIGNED",
    "ASSIGN_STATUS_FAILED",
    "ASSIGN_STATUS_UNSUPPORTED",
    "FIT_STATUS_FAILED",
    "FIT_STATUS_FITTED",
    "FIT_STATUS_UNSUPPORTED",
    "AssignmentPolicy",
    "BaseClustererAdapter",
    "ClustererAssignmentResult",
    "ClustererCapabilities",
    "ClustererFailureMetadata",
    "ClustererFitResult",
    "ClustererRuntimeMetadata",
    "DummyClustererAdapter",
    "normalize_assignment_policy",
]
