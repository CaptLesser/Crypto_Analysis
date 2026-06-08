from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.path_config import PATH_KEYS, resolve_path
from src.regimes.core.path_safety import validate_non_production_write_root, validate_report_root
from src.regimes.core.paths import is_relative_to, resolve_project_path
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_planner import validate_regime_production_planner_gates
from src.regimes.core.root_resolution import resolve_regime_production_write_root
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION = 1
REGIME_PRODUCTION_LABEL_OUTPUT_SCHEMA_ARTIFACT_KIND = "regime_production_label_output_schema_contract"
REGIME_PRODUCTION_SANDBOX_ROOT_ARTIFACT_KIND = "regime_production_sandbox_output_root_contract"
REGIME_PRODUCTION_OUTPUT_CONTRACT_BUNDLE_ARTIFACT_KIND = "regime_production_output_contract_bundle"
REGIME_PRODUCTION_OUTPUT_CONTRACT_SUMMARY_ARTIFACT_KIND = "regime_production_output_contract_summary"
REGIME_PRODUCTION_PARTITION_CONTRACT_ARTIFACT_KIND = "regime_production_label_partition_contract"
REGIME_PRODUCTION_DIRECTORY_ROOT_CONTRACT_ARTIFACT_KIND = "regime_production_directory_root_contract"
REGIME_PRODUCTION_DIRECTORY_BUNDLE_ARTIFACT_KIND = "regime_production_output_directory_contract_bundle"
REGIME_PRODUCTION_DIRECTORY_SUMMARY_ARTIFACT_KIND = "regime_production_output_directory_contract_summary"

SANDBOX_OUTPUT_ROOT_NAMESPACE = "regime_output_sandbox_contracts"
VALIDATION_ARTIFACT_ROOT_NAMESPACE = "regime_validation_artifacts"
CANONICAL_LABEL_OUTPUT_NAMESPACE = "regime_production_labels"
CANONICAL_MODEL_STATE_NAMESPACE = "regime_production_model_state"
CANONICAL_DEFINITION_NAMESPACE = "regime_production_definitions"
LOG_TELEMETRY_NAMESPACE = "regime_production"
STATE_IDENTITY_FIELD = "state_id"
AVAILABILITY_FIELD = "availability_status"

LOGICAL_DTYPE_STRING = "string"
LOGICAL_DTYPE_TIMESTAMP = "timestamp_utc"
LOGICAL_DTYPE_FLOAT = "float64"
LOGICAL_DTYPE_JSON = "json_object"
LOGICAL_DTYPES: tuple[str, ...] = (
    LOGICAL_DTYPE_STRING,
    LOGICAL_DTYPE_TIMESTAMP,
    LOGICAL_DTYPE_FLOAT,
    LOGICAL_DTYPE_JSON,
)

CANONICAL_ROOT_KEYS: tuple[str, ...] = (
    "output_parquet_root",
    "state_root",
    "regime_definition_root",
)


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production output contract {field_name} must be non-empty")
    return text


@dataclass(frozen=True)
class RegimeProductionOutputField:
    name: str
    logical_dtype: str
    nullable: bool = False
    role: str = "value"

    def __post_init__(self) -> None:
        name = _text(self.name, field_name="field name")
        dtype = _text(self.logical_dtype, field_name="logical_dtype")
        if dtype not in LOGICAL_DTYPES:
            raise ValueError(f"Unsupported Regime Production label output dtype: {dtype!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "logical_dtype", dtype)
        object.__setattr__(self, "role", _text(self.role, field_name="field role"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "logical_dtype": self.logical_dtype,
            "nullable": bool(self.nullable),
            "role": self.role,
        }


BRANCH_OUTPUT_FIELDS: Mapping[str, tuple[RegimeProductionOutputField, ...]] = {
    REGIME_BRANCH_ASSET_STATE: (
        RegimeProductionOutputField("asset_id", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField("timestamp", LOGICAL_DTYPE_TIMESTAMP, role="time"),
        RegimeProductionOutputField("band", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField("axis", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField(STATE_IDENTITY_FIELD, LOGICAL_DTYPE_STRING, nullable=True, role="state_identity"),
        RegimeProductionOutputField("confidence", LOGICAL_DTYPE_FLOAT, nullable=True, role="state_confidence"),
        RegimeProductionOutputField(AVAILABILITY_FIELD, LOGICAL_DTYPE_STRING, role="availability"),
        RegimeProductionOutputField("mask_reason", LOGICAL_DTYPE_STRING, nullable=True, role="availability_reason"),
        RegimeProductionOutputField("profile_id", LOGICAL_DTYPE_STRING, role="profile"),
        RegimeProductionOutputField("profile_version", LOGICAL_DTYPE_STRING, role="profile"),
        RegimeProductionOutputField("definition_id", LOGICAL_DTYPE_STRING, role="definition"),
        RegimeProductionOutputField("definition_version", LOGICAL_DTYPE_STRING, role="definition"),
        RegimeProductionOutputField("source_tail_ts", LOGICAL_DTYPE_TIMESTAMP, nullable=True, role="lineage_time"),
        RegimeProductionOutputField("known_at_ts", LOGICAL_DTYPE_TIMESTAMP, role="lineage_time"),
        RegimeProductionOutputField("lineage", LOGICAL_DTYPE_JSON, role="lineage"),
        RegimeProductionOutputField("run_id", LOGICAL_DTYPE_STRING, role="lineage"),
    ),
    REGIME_BRANCH_MARKET_STATE: (
        RegimeProductionOutputField("timestamp", LOGICAL_DTYPE_TIMESTAMP, role="time"),
        RegimeProductionOutputField("band", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField("market_axis", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField(STATE_IDENTITY_FIELD, LOGICAL_DTYPE_STRING, nullable=True, role="state_identity"),
        RegimeProductionOutputField("confidence", LOGICAL_DTYPE_FLOAT, nullable=True, role="state_confidence"),
        RegimeProductionOutputField(AVAILABILITY_FIELD, LOGICAL_DTYPE_STRING, role="availability"),
        RegimeProductionOutputField("mask_reason", LOGICAL_DTYPE_STRING, nullable=True, role="availability_reason"),
        RegimeProductionOutputField("profile_id", LOGICAL_DTYPE_STRING, role="profile"),
        RegimeProductionOutputField("profile_version", LOGICAL_DTYPE_STRING, role="profile"),
        RegimeProductionOutputField("definition_id", LOGICAL_DTYPE_STRING, role="definition"),
        RegimeProductionOutputField("definition_version", LOGICAL_DTYPE_STRING, role="definition"),
        RegimeProductionOutputField("source_tail_ts", LOGICAL_DTYPE_TIMESTAMP, nullable=True, role="lineage_time"),
        RegimeProductionOutputField("known_at_ts", LOGICAL_DTYPE_TIMESTAMP, role="lineage_time"),
        RegimeProductionOutputField("lineage", LOGICAL_DTYPE_JSON, role="lineage"),
        RegimeProductionOutputField("run_id", LOGICAL_DTYPE_STRING, role="lineage"),
    ),
    REGIME_BRANCH_CROSS_ASSET_STATE: (
        RegimeProductionOutputField("asset_id", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField("timestamp", LOGICAL_DTYPE_TIMESTAMP, role="time"),
        RegimeProductionOutputField("band", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField("relationship_feature_family", LOGICAL_DTYPE_STRING, role="grain"),
        RegimeProductionOutputField(STATE_IDENTITY_FIELD, LOGICAL_DTYPE_STRING, nullable=True, role="state_identity"),
        RegimeProductionOutputField("confidence", LOGICAL_DTYPE_FLOAT, nullable=True, role="state_confidence"),
        RegimeProductionOutputField(AVAILABILITY_FIELD, LOGICAL_DTYPE_STRING, role="availability"),
        RegimeProductionOutputField("mask_reason", LOGICAL_DTYPE_STRING, nullable=True, role="availability_reason"),
        RegimeProductionOutputField("relationship_input_tail_ts", LOGICAL_DTYPE_TIMESTAMP, nullable=True, role="relationship_lineage_time"),
        RegimeProductionOutputField("relationship_known_at_ts", LOGICAL_DTYPE_TIMESTAMP, role="relationship_lineage_time"),
        RegimeProductionOutputField("profile_id", LOGICAL_DTYPE_STRING, role="profile"),
        RegimeProductionOutputField("profile_version", LOGICAL_DTYPE_STRING, role="profile"),
        RegimeProductionOutputField("definition_id", LOGICAL_DTYPE_STRING, role="definition"),
        RegimeProductionOutputField("definition_version", LOGICAL_DTYPE_STRING, role="definition"),
        RegimeProductionOutputField("lineage", LOGICAL_DTYPE_JSON, role="lineage"),
        RegimeProductionOutputField("run_id", LOGICAL_DTYPE_STRING, role="lineage"),
    ),
}

BRANCH_PARTITION_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("run_id", "axis", "band", "asset_id"),
    REGIME_BRANCH_MARKET_STATE: ("run_id", "market_axis", "band"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("run_id", "relationship_feature_family", "band", "asset_id"),
}

BRANCH_LABEL_GRAIN_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("asset_id", "axis", "band", "timestamp"),
    REGIME_BRANCH_MARKET_STATE: ("market_axis", "band", "timestamp"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("asset_id", "relationship_feature_family", "band", "timestamp"),
}

BRANCH_DIRECTORY_PARTITION_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("branch", "band", "year", "month", "axis", "asset_id"),
    REGIME_BRANCH_MARKET_STATE: ("branch", "band", "year", "month", "market_axis"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("branch", "band", "year", "month", "relationship_feature_family", "asset_id"),
}


@dataclass(frozen=True)
class RegimeProductionLabelOutputSchema:
    branch: str
    fields: Sequence[RegimeProductionOutputField] | None = None
    schema_version: int = REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        fields = tuple(self.fields or BRANCH_OUTPUT_FIELDS[branch])
        expected = BRANCH_OUTPUT_FIELDS[branch]
        if tuple(field.name for field in fields) != tuple(field.name for field in expected):
            raise ValueError("Regime Production label output schema fields are fixed per branch")
        if tuple(field.logical_dtype for field in fields) != tuple(field.logical_dtype for field in expected):
            raise ValueError("Regime Production label output schema dtypes are fixed per branch")
        if len({field.name for field in fields}) != len(fields):
            raise ValueError("Regime Production label output schema fields must be unique")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def column_order(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields or ())

    @property
    def dtype_by_column(self) -> dict[str, str]:
        return {field.name: field.logical_dtype for field in self.fields or ()}

    @property
    def nullable_columns(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields or () if field.nullable)

    @property
    def schema_id(self) -> str:
        return f"{self.branch}_production_label_output_schema_v{int(self.schema_version)}"

    @property
    def schema_hash(self) -> str:
        stable = {
            "schema_version": int(self.schema_version),
            "branch": self.branch,
            "fields": [field.as_dict() for field in self.fields or ()],
            "label_grain_fields": list(BRANCH_LABEL_GRAIN_FIELDS[self.branch]),
            "directory_partition_fields": list(BRANCH_DIRECTORY_PARTITION_FIELDS[self.branch]),
        }
        raw = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": REGIME_PRODUCTION_LABEL_OUTPUT_SCHEMA_ARTIFACT_KIND,
            "branch": self.branch,
            "schema_id": self.schema_id,
            "schema_hash": self.schema_hash,
            "fields": [field.as_dict() for field in self.fields or ()],
            "column_order": list(self.column_order),
            "dtype_by_column": self.dtype_by_column,
            "nullable_columns": list(self.nullable_columns),
            "required_columns": list(self.column_order),
            "label_grain_fields": list(BRANCH_LABEL_GRAIN_FIELDS[self.branch]),
            "partition_fields": list(BRANCH_DIRECTORY_PARTITION_FIELDS[self.branch]),
            "legacy_sandbox_writer_partition_fields": list(BRANCH_PARTITION_FIELDS[self.branch]),
            "state_identity_field": STATE_IDENTITY_FIELD,
            "availability_field": AVAILABILITY_FIELD,
            "mixed_branch_schema_allowed": False,
            "branch_specific_schema_required": True,
            "labels_generated": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionPartitionContract:
    branch: str
    label_grain_fields: Sequence[str]
    partition_fields: Sequence[str]
    file_format: str = "parquet"
    file_name: str = "part-000.parquet"
    partition_time_resolution: str = "calendar_month"
    mixed_schema_allowed: bool = False
    parent_finalizer_owns_writes: bool = True
    preserve_downstream_matrix_shape: bool = True
    tiny_file_risk: str = "bounded_by_monthly_partitions_and_branch_grain"

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        label_grain = tuple(str(item) for item in self.label_grain_fields)
        partition_fields = tuple(str(item) for item in self.partition_fields)
        if label_grain != BRANCH_LABEL_GRAIN_FIELDS[branch]:
            raise ValueError("Regime Production partition contract label grain mismatch")
        if partition_fields != BRANCH_DIRECTORY_PARTITION_FIELDS[branch]:
            raise ValueError("Regime Production partition contract partition fields mismatch")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "label_grain_fields", label_grain)
        object.__setattr__(self, "partition_fields", partition_fields)
        object.__setattr__(self, "file_format", _text(self.file_format, field_name="file_format"))
        object.__setattr__(self, "file_name", _text(self.file_name, field_name="file_name"))

    @property
    def path_template(self) -> str:
        parts = [f"{field}=<value>" for field in self.partition_fields]
        return "/".join([*parts, self.file_name])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_PARTITION_CONTRACT_ARTIFACT_KIND,
            "branch": self.branch,
            "label_grain_fields": list(self.label_grain_fields),
            "partition_fields": list(self.partition_fields),
            "partition_time_resolution": self.partition_time_resolution,
            "path_template": self.path_template,
            "file_format": self.file_format,
            "file_name": self.file_name,
            "mixed_schema_allowed": False,
            "parent_single_finalizer_owns_writes": bool(self.parent_finalizer_owns_writes),
            "preserve_downstream_matrix_shape": bool(self.preserve_downstream_matrix_shape),
            "tiny_file_risk": self.tiny_file_risk,
            "timestamp_level_partitioning_allowed": False,
            "labels_generated": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionDirectoryRootContract:
    branch: str
    root_kind: str
    planned_root: str | Path
    root_source: str
    root_role: str
    canonical: bool = False
    configured_root_required: bool = True
    must_be_nonproduction_report_root: bool = False
    writer_enabled: bool = False
    production_labels_written: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False
    production_promotion_performed: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        root = resolve_project_path(self.planned_root)
        if self.must_be_nonproduction_report_root:
            root = validate_report_root(root, allow_foundation_descendant=True)
            root = validate_non_production_write_root(root)
        if (
            self.writer_enabled
            or self.production_labels_written
            or self.production_outputs_written
            or self.canonical_production_state_outputs_written
            or self.production_promotion_performed
        ):
            raise ValueError("Regime Production directory root contract cannot enable writes or promotion")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "root_kind", _text(self.root_kind, field_name="root_kind"))
        object.__setattr__(self, "planned_root", root)
        object.__setattr__(self, "root_source", _text(self.root_source, field_name="root_source"))
        object.__setattr__(self, "root_role", _text(self.root_role, field_name="root_role"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DIRECTORY_ROOT_CONTRACT_ARTIFACT_KIND,
            "branch": self.branch,
            "root_kind": self.root_kind,
            "root_role": self.root_role,
            "planned_root": _portable_path_text(self.planned_root),
            "root_source": self.root_source,
            "canonical": bool(self.canonical),
            "configured_root_required": bool(self.configured_root_required),
            "directory_contract_only": True,
            "root_created": False,
            "writer_enabled": False,
            "canonical_write_allowed": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "canonical_root_touched": False,
        }


@dataclass(frozen=True)
class RegimeProductionSandboxOutputRootContract:
    branch: str
    sandbox_root: str | Path
    root_source: str
    schema_id: str
    root_kind: str = "noncanonical_sandbox_output_root"
    safe_to_delete_later_with_explicit_request: bool = True
    writer_enabled: bool = False
    production_labels: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False
    canonical_root_collision: bool = False
    canonical_root_checks: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        root = validate_report_root(self.sandbox_root, allow_foundation_descendant=True)
        root = validate_non_production_write_root(root)
        if self.writer_enabled or self.production_labels or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Regime Production sandbox output root contract cannot enable writes or labels")
        if self.canonical_root_collision:
            raise ValueError("Regime Production sandbox output root cannot collide with canonical roots")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "sandbox_root", root)
        object.__setattr__(self, "root_source", _text(self.root_source, field_name="root_source"))
        object.__setattr__(self, "schema_id", _text(self.schema_id, field_name="schema_id"))
        object.__setattr__(self, "canonical_root_checks", tuple(to_jsonable(dict(item)) for item in self.canonical_root_checks))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_SANDBOX_ROOT_ARTIFACT_KIND,
            "branch": self.branch,
            "schema_id": self.schema_id,
            "root_kind": self.root_kind,
            "sandbox_root": _portable_path_text(self.sandbox_root),
            "root_source": self.root_source,
            "noncanonical": True,
            "separate_from_model_facing_canonical_roots": True,
            "canonical_root_collision": False,
            "canonical_root_checks": [to_jsonable(dict(item)) for item in self.canonical_root_checks],
            "canonical_root_touched": False,
            "safe_to_delete_later_with_explicit_request": bool(self.safe_to_delete_later_with_explicit_request),
            "writer_enabled": False,
            "production_labels": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
        }


@dataclass(frozen=True)
class RegimeProductionOutputContractBundle:
    branch: str
    label_schema: RegimeProductionLabelOutputSchema
    sandbox_root_contract: RegimeProductionSandboxOutputRootContract
    production_gate_validation: Mapping[str, Any]

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        if self.label_schema.branch != branch or self.sandbox_root_contract.branch != branch:
            raise ValueError("Regime Production output contract bundle branch mismatch")
        gate = to_jsonable(dict(self.production_gate_validation))
        if gate.get("production_write_allowed") or gate.get("writer_enabled") or gate.get("production_labels_allowed"):
            raise ValueError("Regime Production output contract bundle received an open production gate")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "production_gate_validation", gate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_OUTPUT_CONTRACT_BUNDLE_ARTIFACT_KIND,
            "branch": self.branch,
            "label_schema": self.label_schema.as_dict(),
            "sandbox_root_contract": self.sandbox_root_contract.as_dict(),
            "production_gate_validation": to_jsonable(dict(self.production_gate_validation)),
            "labels_generated": False,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "production_writer_gates_fail_closed": True,
        }


@dataclass(frozen=True)
class RegimeProductionOutputDirectoryContractBundle:
    branch: str
    label_schema: RegimeProductionLabelOutputSchema
    partition_contract: RegimeProductionPartitionContract
    sandbox_output_root_contract: RegimeProductionSandboxOutputRootContract
    root_contracts: Mapping[str, RegimeProductionDirectoryRootContract]
    production_gate_validation: Mapping[str, Any]

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        if self.label_schema.branch != branch or self.partition_contract.branch != branch or self.sandbox_output_root_contract.branch != branch:
            raise ValueError("Regime Production directory contract bundle branch mismatch")
        roots = {str(key): value for key, value in dict(self.root_contracts).items()}
        for key, contract in roots.items():
            if contract.branch != branch:
                raise ValueError(f"Regime Production directory root branch mismatch: {key}")
        gate = to_jsonable(dict(self.production_gate_validation))
        if gate.get("production_write_allowed") or gate.get("writer_enabled") or gate.get("production_labels_allowed"):
            raise ValueError("Regime Production directory contract bundle received an open production gate")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "root_contracts", roots)
        object.__setattr__(self, "production_gate_validation", gate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DIRECTORY_BUNDLE_ARTIFACT_KIND,
            "branch": self.branch,
            "label_schema": self.label_schema.as_dict(),
            "partition_contract": self.partition_contract.as_dict(),
            "sandbox_output_root_contract": self.sandbox_output_root_contract.as_dict(),
            "root_contracts": {key: contract.as_dict() for key, contract in self.root_contracts.items()},
            "production_gate_validation": to_jsonable(dict(self.production_gate_validation)),
            "directory_contract_only": True,
            "labels_generated": False,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "production_writer_gates_fail_closed": True,
        }


def default_regime_production_label_output_schema(branch: str) -> RegimeProductionLabelOutputSchema:
    return RegimeProductionLabelOutputSchema(branch=_branch_name(branch))


def default_regime_production_partition_contract(branch: str) -> RegimeProductionPartitionContract:
    branch_name = _branch_name(branch)
    return RegimeProductionPartitionContract(
        branch=branch_name,
        label_grain_fields=BRANCH_LABEL_GRAIN_FIELDS[branch_name],
        partition_fields=BRANCH_DIRECTORY_PARTITION_FIELDS[branch_name],
    )


def validate_regime_production_label_output_schema(
    branch: str,
    columns: Sequence[str],
) -> RegimeProductionLabelOutputSchema:
    schema = default_regime_production_label_output_schema(branch)
    if tuple(str(column) for column in columns) != schema.column_order:
        raise ValueError("Regime Production label output columns do not match the fixed branch schema order")
    return schema


def validate_regime_production_branch_grain(branch: str, fields: Sequence[str]) -> tuple[str, ...]:
    branch_name = _branch_name(branch)
    available = {str(field) for field in fields}
    missing = tuple(field for field in BRANCH_LABEL_GRAIN_FIELDS[branch_name] if field not in available)
    if missing:
        raise ValueError(f"Regime Production branch grain fields missing for {branch_name}: {missing!r}")
    return BRANCH_LABEL_GRAIN_FIELDS[branch_name]


def build_regime_production_label_partition_path(
    branch: str,
    values: Mapping[str, Any],
    *,
    root: str | Path,
    file_name: str = "part-000.parquet",
) -> Path:
    branch_name = _branch_name(branch)
    validate_regime_production_branch_grain(branch_name, values.keys())
    year, month = _timestamp_year_month(values.get("timestamp"))
    partition_values = {
        **dict(values),
        "branch": branch_name,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
    }
    parts = [
        f"{field}={_safe_path_part(partition_values.get(field))}"
        for field in BRANCH_DIRECTORY_PARTITION_FIELDS[branch_name]
    ]
    return Path(root).joinpath(*parts, _safe_path_part(file_name))


def resolve_regime_production_sandbox_output_root_contract(
    branch: str,
    *,
    sandbox_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> RegimeProductionSandboxOutputRootContract:
    branch_name = _branch_name(branch)
    source_env = env if env is not None else os.environ
    if sandbox_root is None:
        root, source = resolve_regime_production_write_root(
            None,
            env=source_env,
            project_root=project_root,
            subdir=f"{SANDBOX_OUTPUT_ROOT_NAMESPACE}/{branch_name}",
        )
    else:
        root, source = resolve_regime_production_write_root(
            sandbox_root,
            env=source_env,
            project_root=project_root,
            allow_explicit_dry_test_override=True,
        )
    root = validate_non_production_write_root(root, project_root=project_root)
    checks = _canonical_root_checks(root, env=source_env, project_root=project_root)
    if any(bool(item.get("collision")) for item in checks):
        raise ValueError("Regime Production sandbox root collides with a configured canonical root")
    schema = default_regime_production_label_output_schema(branch_name)
    return RegimeProductionSandboxOutputRootContract(
        branch=branch_name,
        sandbox_root=root,
        root_source=source,
        schema_id=schema.schema_id,
        canonical_root_checks=checks,
    )


def build_regime_production_output_directory_contract_bundle(
    branch: str,
    *,
    sandbox_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> RegimeProductionOutputDirectoryContractBundle:
    branch_name = _branch_name(branch)
    source_env = env if env is not None else os.environ
    schema = default_regime_production_label_output_schema(branch_name)
    partition = default_regime_production_partition_contract(branch_name)
    sandbox = resolve_regime_production_sandbox_output_root_contract(
        branch_name,
        sandbox_root=sandbox_root,
        env=source_env,
        project_root=project_root,
    )
    validation_root, validation_source = resolve_regime_production_write_root(
        None,
        env=source_env,
        project_root=project_root,
        subdir=f"{VALIDATION_ARTIFACT_ROOT_NAMESPACE}/{branch_name}",
    )
    roots = {
        "canonical_label_output_root": _configured_directory_root_contract(
            branch_name,
            key="output_parquet_root",
            subdir=f"{CANONICAL_LABEL_OUTPUT_NAMESPACE}/branch={branch_name}",
            root_kind="canonical_production_label_output_root",
            root_role="model_facing_label_outputs",
            env=source_env,
            project_root=project_root,
            canonical=True,
        ),
        "canonical_model_state_root": _configured_directory_root_contract(
            branch_name,
            key="state_root",
            subdir=f"{CANONICAL_MODEL_STATE_NAMESPACE}/branch={branch_name}",
            root_kind="canonical_model_state_root",
            root_role="model_state",
            env=source_env,
            project_root=project_root,
            canonical=True,
        ),
        "canonical_definition_root": _configured_directory_root_contract(
            branch_name,
            key="regime_definition_root",
            subdir=f"{CANONICAL_DEFINITION_NAMESPACE}/branch={branch_name}",
            root_kind="canonical_definition_root",
            root_role="model_definition",
            env=source_env,
            project_root=project_root,
            canonical=True,
        ),
        "log_telemetry_root": _configured_directory_root_contract(
            branch_name,
            key="log_root",
            subdir=f"{LOG_TELEMETRY_NAMESPACE}/branch={branch_name}",
            root_kind="log_telemetry_root",
            root_role="logs_and_telemetry",
            env=source_env,
            project_root=project_root,
            canonical=False,
        ),
        "validation_artifact_root": RegimeProductionDirectoryRootContract(
            branch=branch_name,
            root_kind="validation_artifact_root",
            root_role="validation_artifacts",
            planned_root=validation_root,
            root_source=validation_source,
            canonical=False,
            must_be_nonproduction_report_root=True,
        ),
    }
    gate = validate_regime_production_planner_gates(branch_name)
    return RegimeProductionOutputDirectoryContractBundle(
        branch=branch_name,
        label_schema=schema,
        partition_contract=partition,
        sandbox_output_root_contract=sandbox,
        root_contracts=roots,
        production_gate_validation=gate.as_dict(),
    )


def build_regime_production_output_directory_contract_summary(
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    bundles = {
        branch: build_regime_production_output_directory_contract_bundle(branch, env=env, project_root=project_root).as_dict()
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    return to_jsonable(
        {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DIRECTORY_SUMMARY_ARTIFACT_KIND,
            "branches": bundles,
            "branch_count": len(bundles),
            "label_grains": {branch: list(BRANCH_LABEL_GRAIN_FIELDS[branch]) for branch in bundles},
            "partition_fields": {branch: list(BRANCH_DIRECTORY_PARTITION_FIELDS[branch]) for branch in bundles},
            "directory_contract_only": True,
            "sandbox_root_separate_from_canonical_root": True,
            "parent_single_finalizer_owns_writes": True,
            "mixed_schema_allowed": False,
            "labels_generated": False,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "production_writer_gates_fail_closed": True,
        }
    )


def build_regime_production_output_contract_bundle(
    branch: str,
    *,
    sandbox_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> RegimeProductionOutputContractBundle:
    branch_name = _branch_name(branch)
    schema = default_regime_production_label_output_schema(branch_name)
    root = resolve_regime_production_sandbox_output_root_contract(
        branch_name,
        sandbox_root=sandbox_root,
        env=env,
        project_root=project_root,
    )
    gate = validate_regime_production_planner_gates(branch_name)
    return RegimeProductionOutputContractBundle(
        branch=branch_name,
        label_schema=schema,
        sandbox_root_contract=root,
        production_gate_validation=gate.as_dict(),
    )


def build_regime_production_output_contract_summary(
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    bundles = {
        branch: build_regime_production_output_contract_bundle(branch, env=env, project_root=project_root).as_dict()
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    return to_jsonable(
        {
            "schema_version": REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_OUTPUT_CONTRACT_SUMMARY_ARTIFACT_KIND,
            "branches": bundles,
            "branch_count": len(bundles),
            "schema_ids": {branch: payload["label_schema"]["schema_id"] for branch, payload in bundles.items()},
            "schema_hashes": {branch: payload["label_schema"]["schema_hash"] for branch, payload in bundles.items()},
            "sandbox_roots": {branch: payload["sandbox_root_contract"]["sandbox_root"] for branch, payload in bundles.items()},
            "noncanonical_sandbox_roots": True,
            "canonical_root_touched": False,
            "labels_generated": False,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "production_writer_gates_fail_closed": True,
        }
    )


def write_regime_production_output_contract_summary(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _canonical_root_checks(
    sandbox_root: Path,
    *,
    env: Mapping[str, str],
    project_root: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    checks: list[dict[str, Any]] = []
    root = Path(sandbox_root).resolve()
    for key in CANONICAL_ROOT_KEYS:
        configured = key in PATH_KEYS
        raw = resolve_path(key, env=env, required=False) if configured else None
        if raw is None:
            checks.append({"root_key": key, "configured": False, "collision": False})
            continue
        candidate = resolve_project_path(raw, project_root=project_root)
        collision = is_relative_to(root, candidate) or is_relative_to(candidate, root)
        checks.append({"root_key": key, "configured": True, "collision": bool(collision)})
    return tuple(checks)


def _configured_directory_root_contract(
    branch: str,
    *,
    key: str,
    subdir: str,
    root_kind: str,
    root_role: str,
    env: Mapping[str, str],
    project_root: str | Path | None,
    canonical: bool,
) -> RegimeProductionDirectoryRootContract:
    raw = resolve_path(key, env=env, required=True)
    if raw is None:
        raise ValueError(f"Regime Production directory root missing for {key}")
    root = resolve_project_path(raw, project_root=project_root) / subdir
    return RegimeProductionDirectoryRootContract(
        branch=branch,
        root_kind=root_kind,
        root_role=root_role,
        planned_root=root,
        root_source=f"path_config.{key}/subdir",
        canonical=canonical,
    )


def _timestamp_year_month(value: Any) -> tuple[int, int]:
    if value in (None, ""):
        raise ValueError("Regime Production partition path requires timestamp")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return dt.year, dt.month
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"Regime Production partition timestamp is invalid: {value!r}") from exc
    return dt.year, dt.month


def _safe_path_part(value: object) -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_").replace(":", "_").replace("|", "_")
    if not text:
        raise ValueError("Regime Production output partition path part must be non-empty")
    return text


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    if text not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {text!r}")
    return text


def _portable_path_text(value: str | Path) -> str:
    path = Path(value).resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return f"<external_configured_root>/{path.name}"


__all__ = [
    "AVAILABILITY_FIELD",
    "BRANCH_DIRECTORY_PARTITION_FIELDS",
    "BRANCH_LABEL_GRAIN_FIELDS",
    "BRANCH_OUTPUT_FIELDS",
    "BRANCH_PARTITION_FIELDS",
    "CANONICAL_ROOT_KEYS",
    "CANONICAL_DEFINITION_NAMESPACE",
    "CANONICAL_LABEL_OUTPUT_NAMESPACE",
    "CANONICAL_MODEL_STATE_NAMESPACE",
    "LOG_TELEMETRY_NAMESPACE",
    "LOGICAL_DTYPE_FLOAT",
    "LOGICAL_DTYPE_JSON",
    "LOGICAL_DTYPE_STRING",
    "LOGICAL_DTYPE_TIMESTAMP",
    "REGIME_PRODUCTION_DIRECTORY_BUNDLE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DIRECTORY_ROOT_CONTRACT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DIRECTORY_SUMMARY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_LABEL_OUTPUT_SCHEMA_ARTIFACT_KIND",
    "REGIME_PRODUCTION_OUTPUT_CONTRACT_BUNDLE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_OUTPUT_CONTRACT_SCHEMA_VERSION",
    "REGIME_PRODUCTION_OUTPUT_CONTRACT_SUMMARY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PARTITION_CONTRACT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_SANDBOX_ROOT_ARTIFACT_KIND",
    "SANDBOX_OUTPUT_ROOT_NAMESPACE",
    "STATE_IDENTITY_FIELD",
    "VALIDATION_ARTIFACT_ROOT_NAMESPACE",
    "RegimeProductionDirectoryRootContract",
    "RegimeProductionLabelOutputSchema",
    "RegimeProductionOutputContractBundle",
    "RegimeProductionOutputDirectoryContractBundle",
    "RegimeProductionOutputField",
    "RegimeProductionPartitionContract",
    "RegimeProductionSandboxOutputRootContract",
    "build_regime_production_label_partition_path",
    "build_regime_production_output_directory_contract_bundle",
    "build_regime_production_output_directory_contract_summary",
    "build_regime_production_output_contract_bundle",
    "build_regime_production_output_contract_summary",
    "default_regime_production_label_output_schema",
    "default_regime_production_partition_contract",
    "resolve_regime_production_sandbox_output_root_contract",
    "validate_regime_production_branch_grain",
    "validate_regime_production_label_output_schema",
    "write_regime_production_output_contract_summary",
]
