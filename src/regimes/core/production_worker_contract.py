from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.forecasting.common.concurrency import resolve_concurrency_profile
from src.forecasting.common.runtime_config import resolve_worker_setting
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_planner import BRANCH_TARGET_KEY_FIELDS, RegimeProductionPlanningUnit
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_WORKER_CONTRACT_SCHEMA_VERSION = 1
REGIME_PRODUCTION_WORKER_PROFILE_ARTIFACT_KIND = "regime_production_worker_profile"
REGIME_PRODUCTION_BATCHABLE_WORK_UNIT_ARTIFACT_KIND = "regime_production_batchable_work_unit"
REGIME_PRODUCTION_JOB_BATCH_ARTIFACT_KIND = "regime_production_job_batch"
REGIME_PRODUCTION_JOB_MATRIX_ARTIFACT_KIND = "regime_production_job_matrix"

REGIME_PRODUCTION_BACKEND_PROCESS = "process"
REGIME_PRODUCTION_BACKEND_THREAD = "thread"
REGIME_PRODUCTION_BACKEND_HYBRID = "hybrid"
REGIME_PRODUCTION_BACKEND_SERIAL = "serial"
REGIME_PRODUCTION_BACKENDS: tuple[str, ...] = (
    REGIME_PRODUCTION_BACKEND_PROCESS,
    REGIME_PRODUCTION_BACKEND_THREAD,
    REGIME_PRODUCTION_BACKEND_HYBRID,
    REGIME_PRODUCTION_BACKEND_SERIAL,
)

DEFAULT_REGIME_PRODUCTION_JOB_BATCH_SIZE = 256
REGIME_PRODUCTION_PARENT_FINALIZER_ID = "regime_production_parent_single_finalizer_v1"

BRANCH_WORKER_MODULES: Mapping[str, str] = {
    REGIME_BRANCH_ASSET_STATE: "regime_production_asset_state",
    REGIME_BRANCH_MARKET_STATE: "regime_production_market_state",
    REGIME_BRANCH_CROSS_ASSET_STATE: "regime_production_cross_asset_state",
}

BRANCH_WORK_UNIT_GROUPING_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("branch", "band", "axis"),
    REGIME_BRANCH_MARKET_STATE: ("branch", "band", "market_axis"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("branch", "band", "relationship_feature_family"),
}

MATURE_WORKER_SOURCE_MODULES: tuple[str, ...] = (
    "src.forecasting.common.runtime_config",
    "src.forecasting.common.concurrency",
    "src.forecasting.ml.shared.numeric_runner_common",
    "src.regimes.asset_state.mature_orchestrator",
    "src.regimes.cross_asset_state.execution_profile",
    "src.regimes.cross_asset_state.default_test_branch",
)


@dataclass(frozen=True)
class RegimeProductionWorkerProfile:
    branch: str
    workers: int
    model_threads: int
    writer_workers: int
    backend: str
    batch_size: int
    grouping_fields: Sequence[str]
    worker_source: str
    worker_source_detail: str
    model_threads_source: str
    model_threads_source_detail: str
    batch_size_source: str
    batch_size_source_detail: str
    parent_single_finalizer: bool = True
    workers_compute_only: bool = True
    workers_write_outputs: bool = False
    nested_parallelism_disabled: bool = True
    relationship_discovery_allowed: bool = False
    broad_pairwise_allowed: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        grouping_fields = _string_tuple(self.grouping_fields, field_name="grouping_fields")
        if grouping_fields != BRANCH_WORK_UNIT_GROUPING_FIELDS[branch]:
            raise ValueError(
                f"Regime Production worker grouping for {branch!r} must be "
                f"{BRANCH_WORK_UNIT_GROUPING_FIELDS[branch]!r}"
            )
        backend = str(self.backend or "").strip().lower()
        if backend not in REGIME_PRODUCTION_BACKENDS:
            raise ValueError(f"Unsupported Regime Production worker backend: {self.backend!r}")
        if int(self.workers) <= 0 or int(self.model_threads) <= 0 or int(self.batch_size) <= 0:
            raise ValueError("Regime Production worker profile workers/model_threads/batch_size must be positive")
        if int(self.writer_workers) != 1:
            raise ValueError("Regime Production worker profile must enforce writer_workers=1")
        if not self.parent_single_finalizer:
            raise ValueError("Regime Production worker profile must use the parent/single finalizer pattern")
        if self.workers_write_outputs:
            raise ValueError("Regime Production workers may compute only; they cannot write outputs")
        if self.relationship_discovery_allowed or self.broad_pairwise_allowed:
            raise ValueError("Regime Production worker profile cannot allow discovery or broad pairwise work")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "workers", int(self.workers))
        object.__setattr__(self, "model_threads", int(self.model_threads))
        object.__setattr__(self, "writer_workers", 1)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "grouping_fields", grouping_fields)
        object.__setattr__(self, "worker_source", _text(self.worker_source, field_name="worker_source"))
        object.__setattr__(
            self,
            "worker_source_detail",
            _source_detail(self.worker_source_detail),
        )
        object.__setattr__(
            self,
            "model_threads_source",
            _text(self.model_threads_source, field_name="model_threads_source"),
        )
        object.__setattr__(
            self,
            "model_threads_source_detail",
            _source_detail(self.model_threads_source_detail),
        )
        object.__setattr__(self, "batch_size_source", _text(self.batch_size_source, field_name="batch_size_source"))
        object.__setattr__(
            self,
            "batch_size_source_detail",
            _source_detail(self.batch_size_source_detail),
        )

    def effective_workers(self, work_unit_count: int) -> int:
        units = max(0, int(work_unit_count))
        if units == 0:
            return 0
        if self.backend == REGIME_PRODUCTION_BACKEND_SERIAL:
            return 1
        return max(1, min(int(self.workers), units))

    def as_dict(self, *, work_unit_count: int | None = None) -> dict[str, Any]:
        effective_workers = (
            int(self.workers)
            if work_unit_count is None
            else int(self.effective_workers(int(work_unit_count)))
        )
        return {
            "schema_version": REGIME_PRODUCTION_WORKER_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_WORKER_PROFILE_ARTIFACT_KIND,
            "branch": self.branch,
            "workers": int(self.workers),
            "configured_workers": int(self.workers),
            "effective_workers": effective_workers,
            "model_threads": int(self.model_threads),
            "writer_workers": 1,
            "backend": self.backend,
            "process_thread_backend": self.backend,
            "batch_size": int(self.batch_size),
            "grouping_fields": list(self.grouping_fields),
            "worker_source": self.worker_source,
            "worker_source_detail": self.worker_source_detail,
            "model_threads_source": self.model_threads_source,
            "model_threads_source_detail": self.model_threads_source_detail,
            "batch_size_source": self.batch_size_source,
            "batch_size_source_detail": self.batch_size_source_detail,
            "parent_single_finalizer": True,
            "workers_compute_only": True,
            "workers_write_outputs": False,
            "nested_parallelism_disabled": True,
            "relationship_discovery_allowed": False,
            "broad_pairwise_allowed": False,
            "mature_worker_source_modules": list(MATURE_WORKER_SOURCE_MODULES),
            "production_outputs_written": False,
            "production_labels_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionBatchableWorkUnit:
    branch: str
    unit_id: str
    target_key: Mapping[str, Any]
    planning_status: str
    group_key: Mapping[str, Any]
    relationship_input_check_count: int = 0

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        unit_id = _text(self.unit_id, field_name="unit_id")
        target_key = to_jsonable(dict(self.target_key))
        group_key = to_jsonable(dict(self.group_key))
        for field_name in BRANCH_TARGET_KEY_FIELDS[branch]:
            if target_key.get(field_name) in (None, ""):
                raise ValueError(f"Regime Production batchable unit target_key missing {field_name!r}")
        for field_name in BRANCH_WORK_UNIT_GROUPING_FIELDS[branch]:
            if group_key.get(field_name) in (None, ""):
                raise ValueError(f"Regime Production batchable unit group_key missing {field_name!r}")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "group_key", group_key)
        object.__setattr__(self, "planning_status", _text(self.planning_status, field_name="planning_status"))
        object.__setattr__(self, "relationship_input_check_count", max(0, int(self.relationship_input_check_count)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_WORKER_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_BATCHABLE_WORK_UNIT_ARTIFACT_KIND,
            "branch": self.branch,
            "unit_id": self.unit_id,
            "target_key": to_jsonable(dict(self.target_key)),
            "planning_status": self.planning_status,
            "group_key": to_jsonable(dict(self.group_key)),
            "relationship_input_check_count": int(self.relationship_input_check_count),
            "label_rows_materialized": False,
            "worker_writes_output": False,
            "production_labels_written": False,
            "production_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionJobBatch:
    branch: str
    job_id: str
    batch_index: int
    batch_count: int
    group_key: Mapping[str, Any]
    work_unit_ids: Sequence[str]
    relationship_input_check_count: int
    backend: str
    parent_finalizer_id: str = REGIME_PRODUCTION_PARENT_FINALIZER_ID

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        unit_ids = _string_tuple(self.work_unit_ids, field_name="work_unit_ids")
        backend = str(self.backend or "").strip().lower()
        if backend not in REGIME_PRODUCTION_BACKENDS:
            raise ValueError(f"Unsupported Regime Production job backend: {self.backend!r}")
        if int(self.batch_index) < 0 or int(self.batch_count) <= 0 or int(self.batch_index) >= int(self.batch_count):
            raise ValueError("Regime Production job batch index/count are invalid")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "job_id", _text(self.job_id, field_name="job_id"))
        object.__setattr__(self, "batch_index", int(self.batch_index))
        object.__setattr__(self, "batch_count", int(self.batch_count))
        object.__setattr__(self, "group_key", to_jsonable(dict(self.group_key)))
        object.__setattr__(self, "work_unit_ids", unit_ids)
        object.__setattr__(self, "relationship_input_check_count", max(0, int(self.relationship_input_check_count)))
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "parent_finalizer_id", _text(self.parent_finalizer_id, field_name="parent_finalizer_id"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_WORKER_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_JOB_BATCH_ARTIFACT_KIND,
            "branch": self.branch,
            "job_id": self.job_id,
            "batch_index": int(self.batch_index),
            "batch_count": int(self.batch_count),
            "group_key": to_jsonable(dict(self.group_key)),
            "work_unit_ids": list(self.work_unit_ids),
            "work_unit_count": len(self.work_unit_ids),
            "relationship_input_check_count": int(self.relationship_input_check_count),
            "backend": self.backend,
            "parent_finalizer_id": self.parent_finalizer_id,
            "workers_compute_only": True,
            "workers_write_outputs": False,
            "production_labels_written": False,
            "production_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionJobMatrix:
    branch: str
    worker_profile: RegimeProductionWorkerProfile
    work_units: Sequence[RegimeProductionBatchableWorkUnit]
    job_batches: Sequence[RegimeProductionJobBatch]
    grouping_fields: Sequence[str]
    parent_finalizer_id: str = REGIME_PRODUCTION_PARENT_FINALIZER_ID

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        if self.worker_profile.branch != branch:
            raise ValueError("Regime Production job matrix worker profile branch mismatch")
        work_units = tuple(self.work_units)
        job_batches = tuple(self.job_batches)
        if any(unit.branch != branch for unit in work_units) or any(batch.branch != branch for batch in job_batches):
            raise ValueError("Regime Production job matrix entries must match branch")
        grouping_fields = _string_tuple(self.grouping_fields, field_name="grouping_fields")
        if grouping_fields != BRANCH_WORK_UNIT_GROUPING_FIELDS[branch]:
            raise ValueError("Regime Production job matrix grouping fields do not match branch policy")
        if work_units and not job_batches:
            raise ValueError("Regime Production job matrix has units but no batches")
        if self.worker_profile.writer_workers != 1:
            raise ValueError("Regime Production job matrix requires writer_workers=1")
        if self.worker_profile.backend == REGIME_PRODUCTION_BACKEND_SERIAL and len(work_units) > self.worker_profile.batch_size:
            raise ValueError("Regime Production job matrix cannot use serial backend for large branch work")
        batch_unit_ids = tuple(unit_id for batch in job_batches for unit_id in batch.work_unit_ids)
        unit_ids = tuple(unit.unit_id for unit in work_units)
        if sorted(batch_unit_ids) != sorted(unit_ids):
            raise ValueError("Regime Production job matrix batches must cover every work unit exactly once")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "work_units", work_units)
        object.__setattr__(self, "job_batches", job_batches)
        object.__setattr__(self, "grouping_fields", grouping_fields)
        object.__setattr__(self, "parent_finalizer_id", _text(self.parent_finalizer_id, field_name="parent_finalizer_id"))

    @property
    def relationship_input_check_count(self) -> int:
        return sum(int(unit.relationship_input_check_count) for unit in self.work_units)

    @property
    def relationship_input_check_batch_count(self) -> int:
        return sum(1 for batch in self.job_batches if int(batch.relationship_input_check_count) > 0)

    def as_dict(self, *, include_work_units: bool = False) -> dict[str, Any]:
        work_unit_count = len(self.work_units)
        return {
            "schema_version": REGIME_PRODUCTION_WORKER_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_JOB_MATRIX_ARTIFACT_KIND,
            "branch": self.branch,
            "worker_profile": self.worker_profile.as_dict(work_unit_count=work_unit_count),
            "workers": int(self.worker_profile.workers),
            "effective_workers": int(self.worker_profile.effective_workers(work_unit_count)),
            "model_threads": int(self.worker_profile.model_threads),
            "writer_workers": 1,
            "backend": self.worker_profile.backend,
            "process_thread_backend": self.worker_profile.backend,
            "batch_size": int(self.worker_profile.batch_size),
            "grouping_fields": list(self.grouping_fields),
            "work_unit_count": work_unit_count,
            "work_units": [unit.as_dict() for unit in self.work_units] if include_work_units else [],
            "work_units_omitted": not include_work_units,
            "job_batch_count": len(self.job_batches),
            "job_batches": [batch.as_dict() for batch in self.job_batches],
            "relationship_input_check_count": int(self.relationship_input_check_count),
            "relationship_input_check_batch_count": int(self.relationship_input_check_batch_count),
            "relationship_input_checks_batched": self.branch == REGIME_BRANCH_CROSS_ASSET_STATE,
            "relationship_discovery_or_pairwise_run_performed": False,
            "broad_pairwise_run_performed": False,
            "parent_finalizer": {
                "parent_finalizer_id": self.parent_finalizer_id,
                "mode": "parent_single_finalizer",
                "parent_single_finalizer": True,
                "writer_workers": 1,
                "workers_write_outputs": False,
                "future_label_writer_owner": "parent_finalizer_only",
                "dry_run_artifact_write_allowed": False,
                "production_write_allowed": False,
                "production_labels_written": False,
                "canonical_production_state_outputs_written": False,
            },
            "workers_compute_only": True,
            "workers_write_outputs": False,
            "label_rows_materialized": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def default_regime_production_worker_profile(
    branch: str,
    *,
    backend: str | None = None,
    batch_size: int | None = None,
) -> RegimeProductionWorkerProfile:
    branch_name = _branch_name(branch)
    module_name = BRANCH_WORKER_MODULES[branch_name]
    concurrency = resolve_concurrency_profile(
        module_name,
        profile="outer_parallel",
        worker_key="unit_workers",
        model_threads_key="model_threads",
    )
    writer_resolved = resolve_worker_setting(module_name, "writer_workers", fallback=1)
    if int(writer_resolved["value"]) != 1:
        raise ValueError("Regime Production resolved writer_workers must remain 1")
    batch_resolved = resolve_worker_setting(
        module_name,
        "batch_size",
        fallback=batch_size or DEFAULT_REGIME_PRODUCTION_JOB_BATCH_SIZE,
    )
    return RegimeProductionWorkerProfile(
        branch=branch_name,
        workers=int(concurrency.effective_workers),
        model_threads=int(concurrency.effective_model_threads),
        writer_workers=1,
        backend=_resolve_backend(branch_name, backend=backend),
        batch_size=int(batch_resolved["value"]),
        grouping_fields=BRANCH_WORK_UNIT_GROUPING_FIELDS[branch_name],
        worker_source=str(concurrency.worker_source or "derived default"),
        worker_source_detail=_source_detail(concurrency.worker_source_detail),
        model_threads_source=str(concurrency.model_threads_source or "derived default"),
        model_threads_source_detail=_source_detail(concurrency.model_threads_source_detail),
        batch_size_source=str(batch_resolved.get("source") or "derived default"),
        batch_size_source_detail=_source_detail(batch_resolved.get("source_detail")),
    )


def build_regime_production_job_matrix(
    branch: str,
    planning_units: Sequence[RegimeProductionPlanningUnit],
    *,
    worker_profile: RegimeProductionWorkerProfile | None = None,
) -> RegimeProductionJobMatrix:
    branch_name = _branch_name(branch)
    units = tuple(planning_units)
    if any(unit.branch != branch_name for unit in units):
        raise ValueError("Regime Production job matrix planning units must match branch")
    profile = worker_profile or default_regime_production_worker_profile(branch_name)
    batchable_units = tuple(_batchable_unit_from_planning_unit(unit) for unit in units)
    grouped: dict[tuple[tuple[str, Any], ...], list[RegimeProductionBatchableWorkUnit]] = {}
    for unit in batchable_units:
        key = tuple((field_name, unit.group_key[field_name]) for field_name in profile.grouping_fields)
        grouped.setdefault(key, []).append(unit)

    batches: list[RegimeProductionJobBatch] = []
    for group_index, (group_key_items, grouped_units) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        group_key = dict(group_key_items)
        chunks = list(_chunks(grouped_units, profile.batch_size))
        for batch_index, chunk in enumerate(chunks):
            batches.append(
                RegimeProductionJobBatch(
                    branch=branch_name,
                    job_id=_job_id(branch_name, group_index, batch_index, group_key),
                    batch_index=batch_index,
                    batch_count=len(chunks),
                    group_key=group_key,
                    work_unit_ids=tuple(unit.unit_id for unit in chunk),
                    relationship_input_check_count=sum(int(unit.relationship_input_check_count) for unit in chunk),
                    backend=profile.backend,
                )
            )
    return RegimeProductionJobMatrix(
        branch=branch_name,
        worker_profile=profile,
        work_units=batchable_units,
        job_batches=tuple(batches),
        grouping_fields=profile.grouping_fields,
    )


def _batchable_unit_from_planning_unit(unit: RegimeProductionPlanningUnit) -> RegimeProductionBatchableWorkUnit:
    group_key = {"branch": unit.branch}
    for field_name in BRANCH_WORK_UNIT_GROUPING_FIELDS[unit.branch]:
        if field_name == "branch":
            continue
        group_key[field_name] = unit.target_key.get(field_name)
    return RegimeProductionBatchableWorkUnit(
        branch=unit.branch,
        unit_id=unit.unit_id,
        target_key=unit.target_key,
        planning_status=unit.planning_status,
        group_key=group_key,
        relationship_input_check_count=len(tuple(unit.relationship_input_checks or ())),
    )


def _chunks(values: Sequence[RegimeProductionBatchableWorkUnit], size: int) -> tuple[tuple[RegimeProductionBatchableWorkUnit, ...], ...]:
    chunk_size = max(1, int(size))
    return tuple(tuple(values[index : index + chunk_size]) for index in range(0, len(values), chunk_size))


def _job_id(branch: str, group_index: int, batch_index: int, group_key: Mapping[str, Any]) -> str:
    key_text = "_".join(_safe_token(group_key[field_name]) for field_name in BRANCH_WORK_UNIT_GROUPING_FIELDS[branch])
    return f"{branch}_job_group{int(group_index):04d}_batch{int(batch_index):04d}_{key_text}"


def _resolve_backend(branch: str, *, backend: str | None = None) -> str:
    if backend is not None:
        value = str(backend).strip().lower()
    else:
        env_name = f"PIPELINE_{BRANCH_WORKER_MODULES[_branch_name(branch)].upper()}_BACKEND"
        value = str(os.getenv(env_name) or REGIME_PRODUCTION_BACKEND_PROCESS).strip().lower()
    if value not in REGIME_PRODUCTION_BACKENDS:
        raise ValueError(f"Unsupported Regime Production worker backend: {value!r}")
    return value


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    aliases = {
        "asset": REGIME_BRANCH_ASSET_STATE,
        "asset-state": REGIME_BRANCH_ASSET_STATE,
        "asset_state_production": REGIME_BRANCH_ASSET_STATE,
        "market": REGIME_BRANCH_MARKET_STATE,
        "market-state": REGIME_BRANCH_MARKET_STATE,
        "market_state_production": REGIME_BRANCH_MARKET_STATE,
        "cross_asset": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross-asset-state": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross_asset_state_production": REGIME_BRANCH_CROSS_ASSET_STATE,
    }
    resolved = aliases.get(text, text)
    if resolved not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {value!r}")
    return resolved


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production worker contract {field_name} must be non-empty")
    return text


def _string_tuple(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Production worker contract {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if not out:
        raise ValueError(f"Regime Production worker contract {field_name} must be non-empty")
    return out


def _source_detail(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "not_applicable"
    if "\\" in text or "/" in text:
        return text.replace("\\", "/").rsplit("/", 1)[-1]
    return text


def _safe_token(value: object) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().lower())
    return token.strip("_") or "unknown"


__all__ = [
    "BRANCH_WORK_UNIT_GROUPING_FIELDS",
    "DEFAULT_REGIME_PRODUCTION_JOB_BATCH_SIZE",
    "MATURE_WORKER_SOURCE_MODULES",
    "REGIME_PRODUCTION_BACKENDS",
    "REGIME_PRODUCTION_BACKEND_HYBRID",
    "REGIME_PRODUCTION_BACKEND_PROCESS",
    "REGIME_PRODUCTION_BACKEND_SERIAL",
    "REGIME_PRODUCTION_BACKEND_THREAD",
    "REGIME_PRODUCTION_BATCHABLE_WORK_UNIT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_JOB_BATCH_ARTIFACT_KIND",
    "REGIME_PRODUCTION_JOB_MATRIX_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PARENT_FINALIZER_ID",
    "REGIME_PRODUCTION_WORKER_CONTRACT_SCHEMA_VERSION",
    "REGIME_PRODUCTION_WORKER_PROFILE_ARTIFACT_KIND",
    "RegimeProductionBatchableWorkUnit",
    "RegimeProductionJobBatch",
    "RegimeProductionJobMatrix",
    "RegimeProductionWorkerProfile",
    "build_regime_production_job_matrix",
    "default_regime_production_worker_profile",
]
