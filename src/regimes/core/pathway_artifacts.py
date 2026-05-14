from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.forecasting.common.sandbox_paths import default_production_roots, is_relative_to
from src.regimes.contracts import REGIME_BANDS, RegimeBandContract, band_for_ceiling
from src.regimes.core.artifacts import read_json, safe_path_part, validate_partition_month, write_json


PATHWAY_MANIFEST_STATUS = "scaffold_only"
PATHWAY_DRY_RUN_DIAGNOSTIC_SCHEMA_VERSION = 1
PATHWAY_SOURCE_PROBE_SCHEMA_VERSION = 1
PATHWAY_DIAGNOSTICS_ROOT_POLICY_VERSION = 1
MARKET_MEMBERSHIP_SOURCE_POLICY_SCHEMA_VERSION = 1
MARKET_MEMBERSHIP_LIFECYCLE_STATUS = "scaffold_explicit_members_only"
MARKET_UNIVERSE_LIFECYCLE_POLICY_SCHEMA_VERSION = 1
MARKET_UNIVERSE_LIFECYCLE_POLICY_ARTIFACT_KIND = "market_universe_lifecycle_policy"
MARKET_UNIVERSE_LIFECYCLE_OWNER = "regimes.market_state.scaffold_explicit_membership"
MARKET_UNIVERSE_REFRESH_MODE = "manual_scaffold_handoff"
MARKET_UNIVERSE_REFRESH_CADENCE = "per_run_or_manual"
MARKET_UNIVERSE_STALENESS_WINDOW = "not_enforced_until_registry_contract_exists"
MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS = "registry_not_implemented"
MARKET_UNIVERSE_FAILURE_POLICY = "fail_closed_on_registry_or_inference_claims"
MARKET_MEMBERSHIP_SOURCE_CLI_EXPLICIT = "cli.explicit_members"
MARKET_MEMBERSHIP_SOURCE_CLI_UNIVERSE_FILE = "cli.universe_file"
MARKET_MEMBERSHIP_SOURCE_CONFIG_MEMBERS = "config.member_assets"
MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES = (
    MARKET_MEMBERSHIP_SOURCE_CLI_EXPLICIT,
    MARKET_MEMBERSHIP_SOURCE_CLI_UNIVERSE_FILE,
    MARKET_MEMBERSHIP_SOURCE_CONFIG_MEMBERS,
)
MARKET_MEMBERSHIP_UNSUPPORTED_SOURCES = (
    "disk.inferred_members",
    "production.registry",
    "aggregate_frame.inferred_members",
    "scalar_partitions.inferred_members",
)
MARKET_UNIVERSE_MEMBERSHIP_INPUT_SCHEMA_VERSION = 1
MARKET_UNIVERSE_MEMBERSHIP_INPUT_ARTIFACT_KIND = "market_universe_membership_input"
MARKET_UNIVERSE_MEMBERSHIP_INPUT_PATH_POLICY_SCHEMA_VERSION = 1
MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_SCHEMA_VERSION = 1
MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_ARTIFACT_KIND = "market_universe_membership_snapshot"
MARKET_SOURCE_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION = 1
MARKET_SOURCE_COVERAGE_DIAGNOSTIC_ARTIFACT_KIND = "market_source_coverage_diagnostic"
MARKET_SOURCE_AVAILABILITY_LINEAGE_SCHEMA_VERSION = 1
MARKET_SOURCE_AVAILABILITY_LINEAGE_ARTIFACT_KIND = "market_source_availability_lineage"
MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_SCHEMA_VERSION = 1
MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_ARTIFACT_KIND = "market_aggregation_source_read_precondition"
MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_FAILURE_POLICY = (
    "fail_closed_on_missing_source_root_or_scalar_coverage_or_membership_claims"
)
RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION = 1
RELATIVE_BENCHMARK_SOURCE_POLICY_ARTIFACT_KIND = "relative_benchmark_source_policy"
RELATIVE_PEER_BASKET_SOURCE_POLICY_ARTIFACT_KIND = "relative_peer_basket_source_policy"
RELATIVE_PEER_BASKET_LIFECYCLE_POLICY_ARTIFACT_KIND = "relative_peer_basket_lifecycle_policy"
RELATIVE_SOURCE_READ_PRECONDITION_SCHEMA_VERSION = 1
RELATIVE_SOURCE_READ_PRECONDITION_ARTIFACT_KIND = "relative_source_read_precondition"
RELATIVE_SOURCE_READ_PRECONDITION_FAILURE_POLICY = (
    "fail_closed_on_relative_source_read_alignment_or_readiness_claims"
)
RELATIVE_OWNERSHIP_LIFECYCLE_STATUS = "scaffold_explicit_relative_inputs_only"
RELATIVE_OWNERSHIP_OWNER = "regimes.relative_state.scaffold_explicit_inputs"
RELATIVE_BENCHMARK_SOURCE_CONFIG = "config.benchmark"
RELATIVE_BENCHMARK_SOURCE_CLI = "cli.benchmark"
RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES = (
    RELATIVE_BENCHMARK_SOURCE_CONFIG,
    RELATIVE_BENCHMARK_SOURCE_CLI,
)
RELATIVE_BENCHMARK_UNSUPPORTED_SOURCES = (
    "production.benchmark_registry",
    "disk.inferred_benchmark",
    "market_universe.proxy_benchmark",
    "source_probe.inferred_benchmark",
    "alignment_frame.inferred_benchmark",
)
RELATIVE_PEER_BASKET_SOURCE_CONFIG = "config.peer_assets"
RELATIVE_PEER_BASKET_SOURCE_CLI = "cli.peer_assets"
RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES = (
    RELATIVE_PEER_BASKET_SOURCE_CONFIG,
    RELATIVE_PEER_BASKET_SOURCE_CLI,
)
RELATIVE_PEER_BASKET_UNSUPPORTED_SOURCES = (
    "production.peer_registry",
    "disk.inferred_peer_basket",
    "market_universe.inferred_peers",
    "scalar_partitions.inferred_peers",
    "alignment_frame.inferred_peers",
)
RELATIVE_BENCHMARK_SUBSTITUTION_POLICY_NONE = "none"
RELATIVE_MISSING_BENCHMARK_POLICIES = (
    "require",
    "drop_timestamp",
    "carry_forward",
    "use_universe_proxy",
)
RELATIVE_PEER_BASKET_REFRESH_MODE = "manual_scaffold_handoff"
RELATIVE_PEER_BASKET_REFRESH_CADENCE = "per_run_or_manual"
RELATIVE_PEER_BASKET_STALENESS_WINDOW = "not_enforced_until_registry_contract_exists"
RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS = "registry_not_implemented"
RELATIVE_OWNERSHIP_FAILURE_POLICY = "fail_closed_on_registry_inference_proxy_or_readiness_claims"
SCALAR_FEATURE_SOURCE_PROBE_KIND = "scalar_feature_partitions"

PATHWAY_DIAGNOSTICS_ROOT_REPORT = "report_root"
PATHWAY_DIAGNOSTICS_ROOT_SANDBOX_TEMP = "sandbox_temp_root"
PATHWAY_DIAGNOSTICS_ROOT_EXPLICIT = "explicit_scaffold_root"
PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION = "unsafe_production_adjacent_root"


class PathwayContractLike(Protocol):
    name: str
    table_prefix: str
    key_columns: tuple[str, ...]
    partition_columns: tuple[str, ...]
    required_output_columns: tuple[str, ...]


@dataclass(frozen=True)
class PathwayDiagnosticsRootPolicy:
    root: str
    classification: str
    allowed_for_scaffold_json: bool
    allowed_for_source_probe: bool
    reason: str
    schema_version: int = PATHWAY_DIAGNOSTICS_ROOT_POLICY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "root": self.root,
            "classification": self.classification,
            "allowed_for_scaffold_json": bool(self.allowed_for_scaffold_json),
            "allowed_for_source_probe": bool(self.allowed_for_source_probe),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MarketUniverseSnapshotMetadata:
    universe: str
    member_assets: tuple[str, ...]
    membership_source: str
    snapshot_timestamp_utc: str
    snapshot_scope: str
    min_assets: int
    status: str = PATHWAY_MANIFEST_STATUS

    def as_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "member_assets": list(self.member_assets),
            "member_assets_count": int(len(self.member_assets)),
            "membership_source": self.membership_source,
            "snapshot_timestamp_utc": self.snapshot_timestamp_utc,
            "snapshot_scope": self.snapshot_scope,
            "min_assets": int(self.min_assets),
            "status": self.status,
        }


@dataclass(frozen=True)
class MarketUniverseMembershipValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MarketUniverseMembershipInputPathPolicy:
    schema_version: int
    path: str
    parent_root: str
    classification: str
    allowed_for_input_json: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "path": self.path,
            "parent_root": self.parent_root,
            "classification": self.classification,
            "allowed_for_input_json": bool(self.allowed_for_input_json),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MarketMembershipSourcePolicy:
    schema_version: int
    lifecycle_status: str
    source_kind: str
    source_detail: str
    implemented: bool
    support_status: str
    explicit_member_assets_required: bool
    explicit_member_assets_supplied: bool
    durable_registry_reference: str | None
    source_feature_root: str | None
    inference_policy: Mapping[str, Any]
    lifecycle_policy: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "lifecycle_status": self.lifecycle_status,
            "source_kind": self.source_kind,
            "source_detail": self.source_detail,
            "implemented": bool(self.implemented),
            "support_status": self.support_status,
            "explicit_member_assets_required": bool(self.explicit_member_assets_required),
            "explicit_member_assets_supplied": bool(self.explicit_member_assets_supplied),
            "durable_registry_reference": self.durable_registry_reference,
            "source_feature_root": self.source_feature_root,
            "inference_policy": dict(self.inference_policy),
            "lifecycle_policy": dict(self.lifecycle_policy),
        }


@dataclass(frozen=True)
class MarketUniverseLifecyclePolicy:
    schema_version: int
    artifact_kind: str
    owner: str
    lifecycle_status: str
    membership_source: str
    refresh_mode: str
    refresh_cadence: str
    staleness_window: str
    registry_support_status: str
    registry_lookup_allowed: bool
    explicit_member_handoff_allowed: bool
    aggregation_readiness: bool
    failure_policy: str
    supported_membership_sources: tuple[str, ...]
    unsupported_membership_sources: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "owner": self.owner,
            "lifecycle_status": self.lifecycle_status,
            "membership_source": self.membership_source,
            "refresh_mode": self.refresh_mode,
            "refresh_cadence": self.refresh_cadence,
            "staleness_window": self.staleness_window,
            "registry_support_status": self.registry_support_status,
            "registry_lookup_allowed": bool(self.registry_lookup_allowed),
            "explicit_member_handoff_allowed": bool(self.explicit_member_handoff_allowed),
            "aggregation_readiness": bool(self.aggregation_readiness),
            "failure_policy": self.failure_policy,
            "supported_membership_sources": list(self.supported_membership_sources),
            "unsupported_membership_sources": list(self.unsupported_membership_sources),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RelativeBenchmarkSourcePolicy:
    schema_version: int
    artifact_kind: str
    lifecycle_status: str
    benchmark: str
    benchmark_source_kind: str
    benchmark_source_detail: str
    substitution_policy: str
    missing_benchmark_policy: str
    implemented: bool
    support_status: str
    durable_registry_reference: str | None
    benchmark_proxy_reference: str | None
    supported_benchmark_sources: tuple[str, ...]
    unsupported_benchmark_sources: tuple[str, ...]
    readiness_flags: Mapping[str, Any]
    artifact_boundary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "lifecycle_status": self.lifecycle_status,
            "benchmark": self.benchmark,
            "benchmark_source_kind": self.benchmark_source_kind,
            "benchmark_source_detail": self.benchmark_source_detail,
            "substitution_policy": self.substitution_policy,
            "missing_benchmark_policy": self.missing_benchmark_policy,
            "implemented": bool(self.implemented),
            "support_status": self.support_status,
            "durable_registry_reference": self.durable_registry_reference,
            "benchmark_proxy_reference": self.benchmark_proxy_reference,
            "supported_benchmark_sources": list(self.supported_benchmark_sources),
            "unsupported_benchmark_sources": list(self.unsupported_benchmark_sources),
            "readiness_flags": dict(self.readiness_flags),
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class RelativePeerBasketLifecyclePolicy:
    schema_version: int
    artifact_kind: str
    owner: str
    lifecycle_status: str
    universe: str
    peer_source_kind: str
    refresh_mode: str
    refresh_cadence: str
    staleness_window: str
    registry_support_status: str
    registry_lookup_allowed: bool
    explicit_peer_handoff_allowed: bool
    min_peer_assets: int
    peer_assets_count: int
    failure_policy: str
    supported_peer_sources: tuple[str, ...]
    unsupported_peer_sources: tuple[str, ...]
    readiness_flags: Mapping[str, Any]
    artifact_boundary: Mapping[str, Any]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "owner": self.owner,
            "lifecycle_status": self.lifecycle_status,
            "universe": self.universe,
            "peer_source_kind": self.peer_source_kind,
            "refresh_mode": self.refresh_mode,
            "refresh_cadence": self.refresh_cadence,
            "staleness_window": self.staleness_window,
            "registry_support_status": self.registry_support_status,
            "registry_lookup_allowed": bool(self.registry_lookup_allowed),
            "explicit_peer_handoff_allowed": bool(self.explicit_peer_handoff_allowed),
            "min_peer_assets": int(self.min_peer_assets),
            "peer_assets_count": int(self.peer_assets_count),
            "failure_policy": self.failure_policy,
            "supported_peer_sources": list(self.supported_peer_sources),
            "unsupported_peer_sources": list(self.unsupported_peer_sources),
            "readiness_flags": dict(self.readiness_flags),
            "artifact_boundary": dict(self.artifact_boundary),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RelativePeerBasketSourcePolicy:
    schema_version: int
    artifact_kind: str
    lifecycle_status: str
    universe: str
    peer_assets: tuple[str, ...]
    min_peer_assets: int
    peer_source_kind: str
    peer_source_detail: str
    implemented: bool
    support_status: str
    explicit_peer_assets_required: bool
    explicit_peer_assets_supplied: bool
    durable_registry_reference: str | None
    inference_policy: Mapping[str, Any]
    lifecycle_policy: Mapping[str, Any]
    artifact_boundary: Mapping[str, Any]

    @property
    def peer_assets_count(self) -> int:
        return int(len(self.peer_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "lifecycle_status": self.lifecycle_status,
            "universe": self.universe,
            "peer_assets": list(self.peer_assets),
            "peer_assets_count": int(self.peer_assets_count),
            "min_peer_assets": int(self.min_peer_assets),
            "peer_source_kind": self.peer_source_kind,
            "peer_source_detail": self.peer_source_detail,
            "implemented": bool(self.implemented),
            "support_status": self.support_status,
            "explicit_peer_assets_required": bool(self.explicit_peer_assets_required),
            "explicit_peer_assets_supplied": bool(self.explicit_peer_assets_supplied),
            "durable_registry_reference": self.durable_registry_reference,
            "inference_policy": dict(self.inference_policy),
            "lifecycle_policy": dict(self.lifecycle_policy),
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class RelativeSourceReadPreconditionValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RelativeSourceReadPrecondition:
    schema_version: int
    artifact_kind: str
    status: str
    pathway: str
    run_id: str
    created_at_utc: str
    universe: str
    benchmark: str
    assets: tuple[str, ...]
    peer_assets: tuple[str, ...]
    min_peer_assets: int
    band: str
    ceiling_interval_min: int
    required_intervals: tuple[int, ...]
    missing_benchmark_policy: str
    benchmark_source_policy: Mapping[str, Any]
    peer_basket_lifecycle_policy: Mapping[str, Any]
    peer_basket_source_policy: Mapping[str, Any]
    source_requirements: Mapping[str, Any]
    ownership_references: Mapping[str, Any]
    failure_policy: str
    readiness_flags: Mapping[str, Any]
    validation_result: RelativeSourceReadPreconditionValidationResult
    artifact_boundary: Mapping[str, Any]

    @property
    def asset_count(self) -> int:
        return int(len(self.assets))

    @property
    def peer_assets_count(self) -> int:
        return int(len(self.peer_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "pathway": self.pathway,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "universe": self.universe,
            "benchmark": self.benchmark,
            "assets": list(self.assets),
            "asset_count": int(self.asset_count),
            "peer_assets": list(self.peer_assets),
            "peer_assets_count": int(self.peer_assets_count),
            "min_peer_assets": int(self.min_peer_assets),
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "required_intervals": list(self.required_intervals),
            "missing_benchmark_policy": self.missing_benchmark_policy,
            "benchmark_source_policy": dict(self.benchmark_source_policy),
            "peer_basket_lifecycle_policy": dict(self.peer_basket_lifecycle_policy),
            "peer_basket_source_policy": dict(self.peer_basket_source_policy),
            "source_requirements": dict(self.source_requirements),
            "ownership_references": dict(self.ownership_references),
            "failure_policy": self.failure_policy,
            "readiness_flags": dict(self.readiness_flags),
            "validation_result": self.validation_result.as_dict(),
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class MarketUniverseMembershipInput:
    schema_version: int
    artifact_kind: str
    universe: str
    member_assets: tuple[str, ...]
    membership_source: str
    source_detail: Mapping[str, Any] | str
    created_at_utc: str
    owner: str | None
    refresh_policy: str | None
    staleness_policy: str | None
    membership_source_policy: Mapping[str, Any]
    validation_result: MarketUniverseMembershipValidationResult
    source_path: str | None = None
    input_path_policy: Mapping[str, Any] | None = None

    @property
    def member_assets_count(self) -> int:
        return int(len(self.member_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "universe": self.universe,
            "member_assets": list(self.member_assets),
            "member_assets_count": int(self.member_assets_count),
            "membership_source": self.membership_source,
            "source_detail": dict(self.source_detail) if isinstance(self.source_detail, Mapping) else self.source_detail,
            "created_at_utc": self.created_at_utc,
            "owner": self.owner,
            "refresh_policy": self.refresh_policy,
            "staleness_policy": self.staleness_policy,
            "membership_source_policy": dict(self.membership_source_policy),
            "validation_result": self.validation_result.as_dict(),
            "source_path": self.source_path,
            "input_path_policy": dict(self.input_path_policy or {}),
        }


@dataclass(frozen=True)
class MarketUniverseMembershipSnapshot:
    schema_version: int
    artifact_kind: str
    status: str
    universe: str
    member_assets: tuple[str, ...]
    min_assets: int
    membership_source: str
    provenance: Mapping[str, Any]
    membership_source_policy: Mapping[str, Any]
    snapshot_timestamp_utc: str
    snapshot_scope: str
    validation_result: MarketUniverseMembershipValidationResult
    artifact_boundary: Mapping[str, Any]

    @property
    def member_assets_count(self) -> int:
        return int(len(self.member_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "universe": self.universe,
            "member_assets": list(self.member_assets),
            "member_assets_count": int(self.member_assets_count),
            "min_assets": int(self.min_assets),
            "membership_source": self.membership_source,
            "provenance": dict(self.provenance),
            "membership_source_policy": dict(self.membership_source_policy),
            "snapshot_timestamp_utc": self.snapshot_timestamp_utc,
            "snapshot_scope": self.snapshot_scope,
            "validation_result": self.validation_result.as_dict(),
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class MarketSourceCoverageDiagnosticRecord:
    schema_version: int
    artifact_kind: str
    status: str
    pathway: str
    run_id: str
    created_at_utc: str
    lifecycle_policy: Mapping[str, Any]
    universe: str
    band: str
    ceiling_interval_min: int
    required_intervals: tuple[int, ...]
    member_assets: tuple[str, ...]
    min_assets: int
    source_coverage_status: str
    membership_snapshot_reference: Mapping[str, Any]
    source_probe_reference: Mapping[str, Any] | None
    source_availability_lineage: Mapping[str, Any]
    source_coverage: Mapping[str, Any]
    validation_result: MarketUniverseMembershipValidationResult
    artifact_boundary: Mapping[str, Any]

    @property
    def member_assets_count(self) -> int:
        return int(len(self.member_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "pathway": self.pathway,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "lifecycle_policy": dict(self.lifecycle_policy),
            "universe": self.universe,
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "required_intervals": list(self.required_intervals),
            "member_assets": list(self.member_assets),
            "member_assets_count": int(self.member_assets_count),
            "min_assets": int(self.min_assets),
            "source_coverage_status": self.source_coverage_status,
            "membership_snapshot_reference": dict(self.membership_snapshot_reference),
            "source_probe_reference": dict(self.source_probe_reference) if self.source_probe_reference is not None else None,
            "source_availability_lineage": dict(self.source_availability_lineage),
            "source_coverage": dict(self.source_coverage),
            "validation_result": self.validation_result.as_dict(),
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class MarketSourceAvailabilityLineage:
    schema_version: int
    artifact_kind: str
    status: str
    pathway: str
    run_id: str
    universe: str
    band: str
    ceiling_interval_min: int
    membership_source: str
    source_coverage_status: str
    artifact_references: Mapping[str, Any]
    readiness_boundary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "pathway": self.pathway,
            "run_id": self.run_id,
            "universe": self.universe,
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "membership_source": self.membership_source,
            "source_coverage_status": self.source_coverage_status,
            "artifact_references": dict(self.artifact_references),
            "readiness_boundary": dict(self.readiness_boundary),
        }


@dataclass(frozen=True)
class MarketAggregationSourceReadPrecondition:
    schema_version: int
    artifact_kind: str
    status: str
    pathway: str
    run_id: str
    created_at_utc: str
    lifecycle_policy: Mapping[str, Any]
    universe: str
    band: str
    ceiling_interval_min: int
    required_intervals: tuple[int, ...]
    member_assets: tuple[str, ...]
    min_assets: int
    membership_source: str
    membership_source_policy: Mapping[str, Any]
    source_root_policy: Mapping[str, Any]
    source_coverage_status: str
    source_coverage: Mapping[str, Any]
    source_probe_reference: Mapping[str, Any]
    membership_snapshot_reference: Mapping[str, Any]
    source_availability_lineage: Mapping[str, Any]
    failure_policy: str
    readiness_flags: Mapping[str, Any]
    validation_result: MarketUniverseMembershipValidationResult
    artifact_boundary: Mapping[str, Any]

    @property
    def member_assets_count(self) -> int:
        return int(len(self.member_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "pathway": self.pathway,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "lifecycle_policy": dict(self.lifecycle_policy),
            "universe": self.universe,
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "required_intervals": list(self.required_intervals),
            "member_assets": list(self.member_assets),
            "member_assets_count": int(self.member_assets_count),
            "min_assets": int(self.min_assets),
            "membership_source": self.membership_source,
            "membership_source_policy": dict(self.membership_source_policy),
            "source_root_policy": dict(self.source_root_policy),
            "source_coverage_status": self.source_coverage_status,
            "source_coverage": dict(self.source_coverage),
            "source_probe_reference": dict(self.source_probe_reference),
            "membership_snapshot_reference": dict(self.membership_snapshot_reference),
            "source_availability_lineage": dict(self.source_availability_lineage),
            "failure_policy": self.failure_policy,
            "readiness_flags": dict(self.readiness_flags),
            "validation_result": self.validation_result.as_dict(),
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class ScalarFeatureIntervalPartitionSummary:
    asset: str
    interval_minutes: int
    asset_partition: str
    asset_partition_exists: bool
    month_partitions: int
    parquet_files: int
    row_count_estimate: int | None
    row_count_estimate_complete: bool
    schema_available: bool
    schema_columns_sample: tuple[str, ...]
    first_ts: Any | None
    last_ts: Any | None
    read_errors: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.asset_partition_exists and self.parquet_files > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "interval_minutes": int(self.interval_minutes),
            "asset_partition": self.asset_partition,
            "asset_partition_exists": bool(self.asset_partition_exists),
            "month_partitions": int(self.month_partitions),
            "parquet_files": int(self.parquet_files),
            "row_count_estimate": self.row_count_estimate,
            "row_count_estimate_complete": bool(self.row_count_estimate_complete),
            "schema_available": bool(self.schema_available),
            "schema_columns_sample": list(self.schema_columns_sample),
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "available": self.available,
            "read_errors": list(self.read_errors),
        }


@dataclass(frozen=True)
class ScalarFeatureAssetPartitionSummary:
    asset: str
    intervals: tuple[ScalarFeatureIntervalPartitionSummary, ...]

    @property
    def missing_intervals(self) -> tuple[int, ...]:
        return tuple(int(row.interval_minutes) for row in self.intervals if not row.available)

    @property
    def parquet_files(self) -> int:
        return sum(int(row.parquet_files) for row in self.intervals)

    @property
    def month_partitions(self) -> int:
        return sum(int(row.month_partitions) for row in self.intervals)

    @property
    def row_count_estimate(self) -> int | None:
        total = 0
        for row in self.intervals:
            if row.row_count_estimate is None:
                return None
            total += int(row.row_count_estimate)
        return int(total)

    @property
    def complete(self) -> bool:
        return not self.missing_intervals

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "complete": self.complete,
            "missing_intervals": list(self.missing_intervals),
            "month_partitions": int(self.month_partitions),
            "parquet_files": int(self.parquet_files),
            "row_count_estimate": self.row_count_estimate,
            "intervals": [row.as_dict() for row in self.intervals],
        }


@dataclass(frozen=True)
class ScalarFeatureSourcePartitionSummary:
    source_root: str
    source_root_exists: bool
    band: str
    ceiling_interval_min: int
    required_intervals: tuple[int, ...]
    member_assets: tuple[str, ...]
    timestamp_column: str
    assets: tuple[ScalarFeatureAssetPartitionSummary, ...]
    membership_source: str
    probe_kind: str = SCALAR_FEATURE_SOURCE_PROBE_KIND

    @property
    def missing_assets(self) -> tuple[str, ...]:
        return tuple(row.asset for row in self.assets if row.missing_intervals)

    @property
    def assets_with_all_intervals(self) -> int:
        return sum(1 for row in self.assets if row.complete)

    @property
    def parquet_files(self) -> int:
        return sum(int(row.parquet_files) for row in self.assets)

    @property
    def month_partitions(self) -> int:
        return sum(int(row.month_partitions) for row in self.assets)

    @property
    def row_count_estimate(self) -> int | None:
        total = 0
        for row in self.assets:
            if row.row_count_estimate is None:
                return None
            total += int(row.row_count_estimate)
        return int(total)

    @property
    def row_count_estimate_complete(self) -> bool:
        return self.row_count_estimate is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_kind": self.probe_kind,
            "source_root": self.source_root,
            "source_root_exists": bool(self.source_root_exists),
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "required_intervals": list(self.required_intervals),
            "member_assets": list(self.member_assets),
            "member_assets_count": int(len(self.member_assets)),
            "membership_source": self.membership_source,
            "timestamp_column": self.timestamp_column,
            "assets_with_all_intervals": int(self.assets_with_all_intervals),
            "missing_assets": list(self.missing_assets),
            "missing_intervals_by_asset": {
                row.asset: list(row.missing_intervals) for row in self.assets if row.missing_intervals
            },
            "month_partitions": int(self.month_partitions),
            "parquet_files": int(self.parquet_files),
            "row_count_estimate": self.row_count_estimate,
            "row_count_estimate_complete": bool(self.row_count_estimate_complete),
            "assets": [row.as_dict() for row in self.assets],
        }


@dataclass(frozen=True)
class PathwayArtifactMetadataSchema:
    pathway: str
    schema_version: int
    key_columns: tuple[str, ...]
    partition_columns: tuple[str, ...]
    required_output_columns: tuple[str, ...]
    feature_manifest_groups: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "schema_version": int(self.schema_version),
            "key_columns": list(self.key_columns),
            "partition_columns": list(self.partition_columns),
            "required_output_columns": list(self.required_output_columns),
            "feature_manifest_groups": list(self.feature_manifest_groups),
        }


@dataclass(frozen=True)
class PathwayDiagnosticReportSchema:
    pathway: str
    trial_key_columns: tuple[str, ...]
    summary_fields: tuple[str, ...]
    feature_group_fields: tuple[str, ...]
    output_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "trial_key_columns": list(self.trial_key_columns),
            "summary_fields": list(self.summary_fields),
            "feature_group_fields": list(self.feature_group_fields),
            "output_fields": list(self.output_fields),
        }


@dataclass(frozen=True)
class PathwayDryRunDiagnosticRecord:
    schema_version: int
    pathway: str
    run_id: str
    status: str
    created_at_utc: str
    config_summary: Mapping[str, Any]
    input_frame_contract: Mapping[str, Any]
    artifact_boundary: Mapping[str, Any]
    validation_results: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "pathway": str(self.pathway),
            "run_id": str(self.run_id),
            "status": str(self.status),
            "created_at_utc": str(self.created_at_utc),
            "config_summary": dict(self.config_summary),
            "input_frame_contract": dict(self.input_frame_contract),
            "artifact_boundary": dict(self.artifact_boundary),
            "validation_results": [dict(result) for result in self.validation_results],
        }


@dataclass(frozen=True)
class PathwaySourceProbeRecord:
    schema_version: int
    pathway: str
    run_id: str
    status: str
    created_at_utc: str
    source_summary: Mapping[str, Any]
    input_validation: Mapping[str, Any]
    artifact_boundary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "pathway": str(self.pathway),
            "run_id": str(self.run_id),
            "status": str(self.status),
            "created_at_utc": str(self.created_at_utc),
            "source_summary": dict(self.source_summary),
            "input_validation": dict(self.input_validation),
            "artifact_boundary": dict(self.artifact_boundary),
        }


def pathway_table_dir(table_prefix: str, ceiling_interval_min: int) -> str:
    band_for_ceiling(int(ceiling_interval_min))
    return f"{safe_path_part(table_prefix, context='Regime pathway table prefix')}_{int(ceiling_interval_min)}"


def pathway_month_dir(
    root: Path,
    *,
    table_prefix: str,
    ceiling_interval_min: int,
    partitions: Mapping[str, object],
    year: int,
    month: int,
) -> Path:
    path = Path(root) / pathway_table_dir(table_prefix, int(ceiling_interval_min))
    for key, value in partitions.items():
        key_part = safe_path_part(key, context="Regime pathway partition keys")
        value_part = safe_path_part(value, context=f"Regime pathway partition {key_part}")
        path = path / f"{key_part}={value_part}"
    return path / f"year={int(year)}" / f"month={validate_partition_month(month, context='Regime pathway month'):02d}"


def pathway_part_path(
    root: Path,
    *,
    table_prefix: str,
    ceiling_interval_min: int,
    partitions: Mapping[str, object],
    year: int,
    month: int,
    filename: str = "part-000.parquet",
) -> Path:
    return pathway_month_dir(
        root,
        table_prefix=table_prefix,
        ceiling_interval_min=int(ceiling_interval_min),
        partitions=partitions,
        year=int(year),
        month=int(month),
    ) / safe_path_part(filename, context="Regime pathway artifact filename")


def pathway_dry_run_diagnostic_path(
    diagnostics_root: Path,
    *,
    pathway: str,
    run_id: str,
    filename: str = "dry_run_diagnostic.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "pathway_dry_runs"
        / safe_path_part(pathway, context="Regime pathway diagnostic pathway")
        / safe_path_part(run_id, context="Regime pathway diagnostic run id")
        / safe_path_part(filename, context="Regime pathway diagnostic filename")
    )


def pathway_source_probe_path(
    diagnostics_root: Path,
    *,
    pathway: str,
    run_id: str,
    filename: str = "source_probe.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "pathway_source_probes"
        / safe_path_part(pathway, context="Regime pathway source-probe pathway")
        / safe_path_part(run_id, context="Regime pathway source-probe run id")
        / safe_path_part(filename, context="Regime pathway source-probe filename")
    )


def market_universe_membership_snapshot_path(
    diagnostics_root: Path,
    *,
    universe: str,
    run_id: str,
    filename: str = "membership_snapshot.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "market_universe_snapshots"
        / _safe_snapshot_path_part(universe, field_name="universe")
        / safe_path_part(run_id, context="Market universe membership snapshot run id")
        / safe_path_part(filename, context="Market universe membership snapshot filename")
    )


def market_source_coverage_diagnostic_path(
    diagnostics_root: Path,
    *,
    universe: str,
    run_id: str,
    filename: str = "coverage_diagnostic.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "market_source_coverage_diagnostics"
        / _safe_snapshot_path_part(universe, field_name="coverage universe")
        / safe_path_part(run_id, context="Market source coverage diagnostic run id")
        / safe_path_part(filename, context="Market source coverage diagnostic filename")
    )


def market_aggregation_source_read_precondition_path(
    diagnostics_root: Path,
    *,
    universe: str,
    run_id: str,
    filename: str = "source_read_precondition.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "market_aggregation_source_read_preconditions"
        / _safe_snapshot_path_part(universe, field_name="source-read precondition universe")
        / safe_path_part(run_id, context="Market aggregation source-read precondition run id")
        / safe_path_part(filename, context="Market aggregation source-read precondition filename")
    )


def relative_source_read_precondition_path(
    diagnostics_root: Path,
    *,
    universe: str,
    benchmark: str,
    run_id: str,
    filename: str = "source_read_precondition.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "relative_source_read_preconditions"
        / _safe_snapshot_path_part(universe, field_name="relative source-read precondition universe")
        / _safe_snapshot_path_part(benchmark, field_name="relative source-read precondition benchmark")
        / safe_path_part(run_id, context="Relative source-read precondition run id")
        / safe_path_part(filename, context="Relative source-read precondition filename")
    )


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _under_any(path: Path, roots: Sequence[Path]) -> bool:
    resolved = _resolved(path)
    for root in roots:
        root_resolved = _resolved(root)
        if resolved == root_resolved or is_relative_to(resolved, root_resolved):
            return True
    return False


def _project_report_roots(project_root: Path) -> tuple[Path, ...]:
    root = _resolved(project_root)
    return (
        root / "reports",
        root / "logs" / "diagnostics",
    )


def _project_production_adjacent_roots(project_root: Path) -> tuple[Path, ...]:
    root = _resolved(project_root)
    return (
        root / "parquet",
        root / "regime_definitions",
        root / "model_states",
    )


def _temp_like_roots() -> tuple[Path, ...]:
    roots = {Path(tempfile.gettempdir())}
    for raw in ("D:/pipeline_codex_temp", "D:/reports"):
        roots.add(Path(raw))
    return tuple(roots)


def classify_pathway_diagnostics_root(
    diagnostics_root: Path,
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> PathwayDiagnosticsRootPolicy:
    root = _resolved(Path(diagnostics_root))
    project = _resolved(project_root or Path.cwd())
    production_roots = (*default_production_roots(env), *_project_production_adjacent_roots(project))
    if _under_any(root, production_roots):
        return PathwayDiagnosticsRootPolicy(
            root=str(root),
            classification=PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION,
            allowed_for_scaffold_json=False,
            allowed_for_source_probe=False,
            reason="Diagnostics root is inside a production or production-adjacent artifact root.",
        )
    if _under_any(root, _project_report_roots(project)):
        return PathwayDiagnosticsRootPolicy(
            root=str(root),
            classification=PATHWAY_DIAGNOSTICS_ROOT_REPORT,
            allowed_for_scaffold_json=True,
            allowed_for_source_probe=True,
            reason="Diagnostics root is under the project report/diagnostics tree.",
        )
    if _under_any(root, _temp_like_roots()) or any(
        token in str(part).lower() for part in root.parts for token in ("tmp", "temp", "pytest")
    ):
        return PathwayDiagnosticsRootPolicy(
            root=str(root),
            classification=PATHWAY_DIAGNOSTICS_ROOT_SANDBOX_TEMP,
            allowed_for_scaffold_json=True,
            allowed_for_source_probe=True,
            reason="Diagnostics root is under a sandbox or temporary tree.",
        )
    return PathwayDiagnosticsRootPolicy(
        root=str(root),
        classification=PATHWAY_DIAGNOSTICS_ROOT_EXPLICIT,
        allowed_for_scaffold_json=True,
        allowed_for_source_probe=False,
        reason="Explicit non-production root is allowed for dry-run JSON; real source probes need report or sandbox/temp roots.",
    )


def require_pathway_diagnostics_root(
    diagnostics_root: Path,
    *,
    for_source_probe: bool = False,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> PathwayDiagnosticsRootPolicy:
    policy = classify_pathway_diagnostics_root(diagnostics_root, project_root=project_root, env=env)
    allowed = policy.allowed_for_source_probe if bool(for_source_probe) else policy.allowed_for_scaffold_json
    if not allowed:
        use_case = "source-probe" if bool(for_source_probe) else "scaffold diagnostic"
        raise ValueError(f"Regime pathway {use_case} diagnostics root is not allowed: {policy.reason}")
    return policy


def classify_market_universe_membership_input_path(
    input_path: Path,
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> MarketUniverseMembershipInputPathPolicy:
    path = _resolved(Path(input_path))
    parent = _resolved(path.parent)
    parent_policy = classify_pathway_diagnostics_root(parent, project_root=project_root, env=env)
    allowed_classifications = {PATHWAY_DIAGNOSTICS_ROOT_REPORT, PATHWAY_DIAGNOSTICS_ROOT_SANDBOX_TEMP}
    allowed = bool(parent_policy.classification in allowed_classifications and parent_policy.allowed_for_scaffold_json)
    reason = parent_policy.reason
    if path.suffix.lower() != ".json":
        allowed = False
        reason = "Market universe membership input files must be JSON."
    elif parent_policy.classification == PATHWAY_DIAGNOSTICS_ROOT_EXPLICIT:
        allowed = False
        reason = "Market universe membership input files must live under report or temporary scaffold roots."
    elif parent_policy.classification == PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION:
        allowed = False
        reason = "Market universe membership input files cannot live under production-adjacent artifact roots."
    return MarketUniverseMembershipInputPathPolicy(
        schema_version=MARKET_UNIVERSE_MEMBERSHIP_INPUT_PATH_POLICY_SCHEMA_VERSION,
        path=str(path),
        parent_root=str(parent),
        classification=parent_policy.classification,
        allowed_for_input_json=allowed,
        reason=reason,
    )


def require_market_universe_membership_input_path(
    input_path: Path,
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> MarketUniverseMembershipInputPathPolicy:
    policy = classify_market_universe_membership_input_path(input_path, project_root=project_root, env=env)
    if not policy.allowed_for_input_json:
        raise ValueError(f"Market universe membership input path is not allowed: {policy.reason}")
    return policy


def _safe_snapshot_path_part(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    cleaned = safe_path_part(text, context=f"Market universe membership snapshot {field_name}")
    if cleaned != text or text in {".", ".."}:
        raise ValueError(f"Market universe membership snapshot {field_name} has unsafe path characters")
    return cleaned


def _utc_timestamp_text(value: str | None) -> str:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Market universe membership snapshot timestamp must be valid ISO-8601 UTC text") from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset != timezone.utc.utcoffset(parsed):
        raise ValueError("Market universe membership snapshot timestamp must include UTC timezone")
    return raw


def _sorted_unique_members(member_assets: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(asset).strip() for asset in member_assets if str(asset).strip()}, key=str.lower))


def _source_detail_policy_text(source_detail: Any, *, source_path: Path | str | None = None) -> str:
    if source_path is not None:
        return str(Path(source_path))
    if isinstance(source_detail, Mapping):
        for key in ("source_file", "description", "name", "path"):
            raw = source_detail.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip()
        return "market_universe_membership_input"
    return str(source_detail or "").strip()


def _claim_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _append_unsupported_membership_claims(payload: Mapping[str, Any], errors: list[str], *, context: str) -> None:
    unsupported = (
        "durable_registry_reference",
        "registry_reference",
        "registry_lookup_requested",
        "production_registry_lookup_requested",
        "membership_inferred_from_disk",
        "disk_inference_requested",
        "scalar_partitions_inferred",
        "aggregate_frame_inferred",
    )
    for key in unsupported:
        if key in payload and _claim_present(payload.get(key)):
            errors.append(f"{context} cannot claim unsupported membership field {key!r}")


def validate_market_universe_membership_input_payload(
    payload: Mapping[str, Any],
    *,
    expected_universe: str | None = None,
    source_path: Path | str | None = None,
    input_path_policy: Mapping[str, Any] | None = None,
) -> MarketUniverseMembershipValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return MarketUniverseMembershipValidationResult(
            ok=False,
            errors=("membership input payload must be a JSON object",),
        )
    data = dict(payload)
    try:
        schema_version = int(data.get("schema_version"))
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != MARKET_UNIVERSE_MEMBERSHIP_INPUT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {MARKET_UNIVERSE_MEMBERSHIP_INPUT_SCHEMA_VERSION}"
        )
    if str(data.get("artifact_kind", "")).strip() != MARKET_UNIVERSE_MEMBERSHIP_INPUT_ARTIFACT_KIND:
        errors.append(f"artifact_kind must be {MARKET_UNIVERSE_MEMBERSHIP_INPUT_ARTIFACT_KIND!r}")

    universe_text = str(data.get("universe", "")).strip()
    if not universe_text:
        errors.append("universe must be non-empty")
    else:
        try:
            _safe_snapshot_path_part(universe_text, field_name="universe input")
        except ValueError as exc:
            errors.append(str(exc))
    expected = str(expected_universe or "").strip()
    if expected and universe_text and universe_text != expected:
        errors.append("universe does not match expected_universe")

    raw_member_value = data.get("member_assets")
    if not isinstance(raw_member_value, (list, tuple)):
        raw_members: tuple[str, ...] = ()
        errors.append("member_assets must be an explicit JSON array")
    else:
        raw_members = tuple(str(asset).strip() for asset in raw_member_value)
    blank_members = sum(1 for asset in raw_members if not asset)
    nonblank_members = tuple(asset for asset in raw_members if asset)
    unique_members = _sorted_unique_members(nonblank_members)
    if blank_members:
        errors.append("member_assets must not include blank assets")
    if not nonblank_members:
        errors.append("member_assets must be non-empty")
    if len(set(nonblank_members)) != len(nonblank_members):
        errors.append("member_assets must be unique")
    for asset in unique_members:
        try:
            _safe_snapshot_path_part(asset, field_name=f"member asset {asset!r}")
        except ValueError as exc:
            errors.append(str(exc))

    source = str(data.get("membership_source", "")).strip()
    if source != MARKET_MEMBERSHIP_SOURCE_CLI_UNIVERSE_FILE:
        errors.append(f"membership_source must be {MARKET_MEMBERSHIP_SOURCE_CLI_UNIVERSE_FILE!r}")

    source_detail = data.get("source_detail")
    if isinstance(source_detail, Mapping):
        if not source_detail:
            errors.append("source_detail must be non-empty")
        _append_unsupported_membership_claims(source_detail, errors, context="source_detail")
    elif source_detail is None or not str(source_detail).strip():
        errors.append("source_detail must be non-empty")

    _append_unsupported_membership_claims(data, errors, context="membership input")

    created_raw = str(data.get("created_at_utc", "")).strip()
    if not created_raw:
        errors.append("created_at_utc must be non-empty")
    else:
        try:
            _utc_timestamp_text(created_raw)
        except ValueError as exc:
            errors.append(str(exc).replace("snapshot", "input"))

    for optional_key in ("owner", "refresh_policy", "staleness_policy"):
        value = data.get(optional_key)
        if value is not None and not str(value).strip():
            errors.append(f"{optional_key} must be non-empty when supplied")

    path_policy = dict(input_path_policy or {})
    if path_policy and path_policy.get("allowed_for_input_json") is not True:
        errors.append("input_path_policy does not allow this membership input file")

    if not errors:
        try:
            market_membership_source_policy(
                membership_source=source,
                member_assets=unique_members,
                source_detail=_source_detail_policy_text(source_detail, source_path=source_path),
            )
        except ValueError as exc:
            errors.append(str(exc))

    return MarketUniverseMembershipValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def market_universe_membership_input_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_universe: str | None = None,
    source_path: Path | str | None = None,
    input_path_policy: Mapping[str, Any] | None = None,
) -> MarketUniverseMembershipInput:
    data = dict(payload)
    policy_payload = dict(input_path_policy or {})
    validation = validate_market_universe_membership_input_payload(
        data,
        expected_universe=expected_universe,
        source_path=source_path,
        input_path_policy=policy_payload,
    )
    if not validation.ok:
        raise ValueError("Market universe membership input invalid: " + "; ".join(validation.errors))
    members = _sorted_unique_members(tuple(str(asset).strip() for asset in data.get("member_assets", ())))
    source_detail = data.get("source_detail")
    membership_policy = market_membership_source_policy(
        membership_source=str(data.get("membership_source", "")).strip(),
        member_assets=members,
        source_detail=_source_detail_policy_text(source_detail, source_path=source_path),
    ).as_dict()
    return MarketUniverseMembershipInput(
        schema_version=MARKET_UNIVERSE_MEMBERSHIP_INPUT_SCHEMA_VERSION,
        artifact_kind=MARKET_UNIVERSE_MEMBERSHIP_INPUT_ARTIFACT_KIND,
        universe=str(data.get("universe", "")).strip(),
        member_assets=members,
        membership_source=MARKET_MEMBERSHIP_SOURCE_CLI_UNIVERSE_FILE,
        source_detail=dict(source_detail) if isinstance(source_detail, Mapping) else str(source_detail),
        created_at_utc=_utc_timestamp_text(str(data.get("created_at_utc", ""))),
        owner=str(data["owner"]).strip() if data.get("owner") is not None else None,
        refresh_policy=str(data["refresh_policy"]).strip() if data.get("refresh_policy") is not None else None,
        staleness_policy=str(data["staleness_policy"]).strip() if data.get("staleness_policy") is not None else None,
        membership_source_policy=membership_policy,
        validation_result=validation,
        source_path=str(Path(source_path)) if source_path is not None else None,
        input_path_policy=policy_payload,
    )


def load_market_universe_membership_input_file(
    input_path: Path,
    *,
    expected_universe: str | None = None,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> MarketUniverseMembershipInput:
    policy = require_market_universe_membership_input_path(input_path, project_root=project_root, env=env)
    payload = read_json(Path(input_path))
    if not payload:
        raise ValueError("Market universe membership input file must contain a JSON object")
    return market_universe_membership_input_from_payload(
        payload,
        expected_universe=expected_universe,
        source_path=Path(input_path),
        input_path_policy=policy.as_dict(),
    )


def _default_market_membership_snapshot_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "membership_snapshot_json_only",
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "membership_inferred_from_disk": False,
        "aggregation_frame_built": False,
        "aggregation_values_built": False,
    }


def _market_membership_inference_policy(
    *,
    membership_inferred_from_disk: bool = False,
    scalar_partitions_inferred: bool = False,
    aggregate_frame_inferred: bool = False,
    registry_lookup_requested: bool = False,
) -> dict[str, Any]:
    return {
        "membership_inference_allowed": False,
        "disk_inference_allowed": False,
        "scalar_partition_inference_allowed": False,
        "aggregate_frame_inference_allowed": False,
        "production_registry_lookup_allowed": False,
        "membership_inferred_from_disk": bool(membership_inferred_from_disk),
        "scalar_partitions_inferred": bool(scalar_partitions_inferred),
        "aggregate_frame_inferred": bool(aggregate_frame_inferred),
        "registry_lookup_requested": bool(registry_lookup_requested),
    }


def market_universe_lifecycle_policy(
    *,
    membership_source: str,
    owner: str = MARKET_UNIVERSE_LIFECYCLE_OWNER,
    lifecycle_status: str = MARKET_MEMBERSHIP_LIFECYCLE_STATUS,
    refresh_mode: str = MARKET_UNIVERSE_REFRESH_MODE,
    refresh_cadence: str = MARKET_UNIVERSE_REFRESH_CADENCE,
    staleness_window: str = MARKET_UNIVERSE_STALENESS_WINDOW,
    registry_support_status: str = MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS,
    registry_lookup_allowed: bool = False,
    explicit_member_handoff_allowed: bool = True,
    aggregation_readiness: bool = False,
    failure_policy: str = MARKET_UNIVERSE_FAILURE_POLICY,
    schema_version: int = MARKET_UNIVERSE_LIFECYCLE_POLICY_SCHEMA_VERSION,
) -> MarketUniverseLifecyclePolicy:
    source = str(membership_source or "").strip()
    if not source:
        raise ValueError("market universe lifecycle policy membership_source must be non-empty")
    if source in MARKET_MEMBERSHIP_UNSUPPORTED_SOURCES:
        raise ValueError(f"market universe lifecycle policy source {source!r} is unsupported")
    if source not in MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES:
        valid = ", ".join(MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown market universe lifecycle policy source {source!r}; expected one of: {valid}")
    if str(lifecycle_status) != MARKET_MEMBERSHIP_LIFECYCLE_STATUS:
        raise ValueError(
            "market universe lifecycle policy lifecycle_status must be "
            f"{MARKET_MEMBERSHIP_LIFECYCLE_STATUS!r}"
        )
    for field_name, raw_value in (
        ("owner", owner),
        ("refresh_mode", refresh_mode),
        ("refresh_cadence", refresh_cadence),
        ("staleness_window", staleness_window),
        ("registry_support_status", registry_support_status),
        ("failure_policy", failure_policy),
    ):
        if not str(raw_value or "").strip():
            raise ValueError(f"market universe lifecycle policy {field_name} must be non-empty")
    if bool(registry_lookup_allowed):
        raise ValueError("market universe lifecycle policy cannot allow registry lookup")
    if not bool(explicit_member_handoff_allowed):
        raise ValueError("market universe lifecycle policy must allow explicit-member handoff")
    if bool(aggregation_readiness):
        raise ValueError("market universe lifecycle policy cannot claim aggregation readiness")
    if str(registry_support_status).strip() != MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS:
        raise ValueError(
            "market universe lifecycle policy registry_support_status must remain "
            f"{MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS!r}"
        )
    return MarketUniverseLifecyclePolicy(
        schema_version=int(schema_version),
        artifact_kind=MARKET_UNIVERSE_LIFECYCLE_POLICY_ARTIFACT_KIND,
        owner=str(owner).strip(),
        lifecycle_status=MARKET_MEMBERSHIP_LIFECYCLE_STATUS,
        membership_source=source,
        refresh_mode=str(refresh_mode).strip(),
        refresh_cadence=str(refresh_cadence).strip(),
        staleness_window=str(staleness_window).strip(),
        registry_support_status=MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS,
        registry_lookup_allowed=False,
        explicit_member_handoff_allowed=True,
        aggregation_readiness=False,
        failure_policy=str(failure_policy).strip(),
        supported_membership_sources=tuple(MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES),
        unsupported_membership_sources=tuple(MARKET_MEMBERSHIP_UNSUPPORTED_SOURCES),
        notes=(
            "Universe files and CLI/config member lists are scaffold explicit-member handoffs.",
            "Durable registry lookup, inferred membership, and aggregate-frame readiness are disabled.",
        ),
    )


def market_membership_source_policy(
    *,
    membership_source: str,
    member_assets: Sequence[str] | None = None,
    source_detail: str | None = None,
    source_feature_root: Path | str | None = None,
    durable_registry_reference: str | None = None,
    lifecycle_status: str = MARKET_MEMBERSHIP_LIFECYCLE_STATUS,
    membership_inferred_from_disk: bool = False,
    scalar_partitions_inferred: bool = False,
    aggregate_frame_inferred: bool = False,
    registry_lookup_requested: bool = False,
    schema_version: int = MARKET_MEMBERSHIP_SOURCE_POLICY_SCHEMA_VERSION,
) -> MarketMembershipSourcePolicy:
    source = str(membership_source or "").strip()
    if not source:
        raise ValueError("market membership source policy membership_source must be non-empty")
    if str(lifecycle_status) != MARKET_MEMBERSHIP_LIFECYCLE_STATUS:
        raise ValueError(f"market membership source policy lifecycle_status must be {MARKET_MEMBERSHIP_LIFECYCLE_STATUS!r}")
    if source in MARKET_MEMBERSHIP_UNSUPPORTED_SOURCES:
        raise ValueError(f"market membership source {source!r} is unsupported and not implemented")
    if source not in MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES:
        valid = ", ".join(MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown market membership source {source!r}; expected one of: {valid}")

    members = tuple(str(asset).strip() for asset in (member_assets or ()) if str(asset).strip())
    if not members:
        raise ValueError("market membership source policy requires explicit member_assets")
    if len(set(members)) != len(members):
        raise ValueError("market membership source policy member_assets must be unique")

    if durable_registry_reference is not None and str(durable_registry_reference).strip():
        raise ValueError("durable market membership registry references are not implemented")
    if any(
        (
            bool(membership_inferred_from_disk),
            bool(scalar_partitions_inferred),
            bool(aggregate_frame_inferred),
            bool(registry_lookup_requested),
        )
    ):
        raise ValueError("market membership source policy cannot claim inferred or registry-backed membership")

    detail = str(source_detail or source).strip()
    if not detail:
        raise ValueError("market membership source policy source_detail must be non-empty")
    root_text = str(Path(source_feature_root)) if source_feature_root is not None else None
    inference_policy = _market_membership_inference_policy(
        membership_inferred_from_disk=membership_inferred_from_disk,
        scalar_partitions_inferred=scalar_partitions_inferred,
        aggregate_frame_inferred=aggregate_frame_inferred,
        registry_lookup_requested=registry_lookup_requested,
    )
    lifecycle_policy = market_universe_lifecycle_policy(
        membership_source=source,
        lifecycle_status=MARKET_MEMBERSHIP_LIFECYCLE_STATUS,
    ).as_dict()
    return MarketMembershipSourcePolicy(
        schema_version=int(schema_version),
        lifecycle_status=MARKET_MEMBERSHIP_LIFECYCLE_STATUS,
        source_kind=source,
        source_detail=detail,
        implemented=True,
        support_status="implemented_scaffold_explicit_members",
        explicit_member_assets_required=True,
        explicit_member_assets_supplied=True,
        durable_registry_reference=None,
        source_feature_root=root_text,
        inference_policy=inference_policy,
        lifecycle_policy=lifecycle_policy,
    )


def _default_relative_ownership_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "relative_ownership_metadata_only",
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "source_read_ready": False,
        "alignment_frame_ready": False,
        "production_output_ready": False,
        "downstream_reader_ready": False,
        "benchmark_registry_ready": False,
        "benchmark_proxy_ready": False,
        "peer_registry_ready": False,
        "benchmark_substitution_inferred": False,
        "peer_basket_inferred": False,
    }


def _default_relative_ownership_readiness_flags() -> dict[str, Any]:
    return {
        "source_read_ready": False,
        "alignment_frame_ready": False,
        "production_output_ready": False,
        "downstream_reader_ready": False,
        "benchmark_registry_ready": False,
        "benchmark_proxy_ready": False,
        "peer_registry_ready": False,
        "parquet_writer_ready": False,
        "definition_writer_ready": False,
    }


def _validate_relative_metadata_boundary(boundary: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    merged = _default_relative_ownership_artifact_boundary()
    merged.update(dict(boundary))
    checked = _validate_scaffold_artifact_boundary(merged, context=context)
    for field_name in (
        "source_read_ready",
        "alignment_frame_ready",
        "production_output_ready",
        "downstream_reader_ready",
        "benchmark_registry_ready",
        "benchmark_proxy_ready",
        "peer_registry_ready",
        "benchmark_substitution_inferred",
        "peer_basket_inferred",
    ):
        if checked.get(field_name) is not False:
            raise ValueError(f"{context} cannot claim {field_name}")
    return checked


def _validate_relative_readiness_flags(flags: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    checked = _default_relative_ownership_readiness_flags()
    checked.update(dict(flags))
    for field_name in (
        "source_read_ready",
        "alignment_frame_ready",
        "production_output_ready",
        "downstream_reader_ready",
        "benchmark_registry_ready",
        "benchmark_proxy_ready",
        "peer_registry_ready",
        "parquet_writer_ready",
        "definition_writer_ready",
    ):
        if checked.get(field_name) is not False:
            raise ValueError(f"{context} cannot claim {field_name}")
    return checked


def _default_relative_source_read_precondition_artifact_boundary() -> dict[str, Any]:
    boundary = _default_relative_ownership_artifact_boundary()
    boundary.update(
        {
            "write_mode": "relative_source_read_precondition_metadata_only",
            "source_read_precondition_only": True,
            "source_probe_enabled": False,
            "source_reader_enabled": False,
            "benchmark_reader_enabled": False,
            "peer_reader_enabled": False,
            "benchmark_substitution_enabled": False,
            "benchmark_proxy_enabled": False,
            "alignment_frame_built": False,
            "registry_lookup_allowed": False,
            "inference_allowed": False,
        }
    )
    return boundary


def _default_relative_source_read_precondition_readiness_flags() -> dict[str, Any]:
    flags = _default_relative_ownership_readiness_flags()
    flags.update(
        {
            "source_read_preconditions_met": False,
            "benchmark_source_read_ready": False,
            "peer_source_read_ready": False,
            "benchmark_read_ready": False,
            "peer_read_ready": False,
            "benchmark_substitution_ready": False,
        }
    )
    return flags


def _validate_relative_source_read_precondition_boundary(
    boundary: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    merged = _default_relative_source_read_precondition_artifact_boundary()
    merged.update(dict(boundary))
    checked = _validate_relative_metadata_boundary(merged, context=context)
    if checked.get("source_read_precondition_only") is not True:
        raise ValueError(f"{context} must remain source_read_precondition_only")
    for field_name in (
        "source_probe_enabled",
        "source_reader_enabled",
        "benchmark_reader_enabled",
        "peer_reader_enabled",
        "benchmark_substitution_enabled",
        "benchmark_proxy_enabled",
        "alignment_frame_built",
        "registry_lookup_allowed",
        "inference_allowed",
    ):
        if checked.get(field_name) is not False:
            raise ValueError(f"{context} cannot claim {field_name}")
    return checked


def _validate_relative_source_read_precondition_readiness_flags(
    flags: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    checked = _default_relative_source_read_precondition_readiness_flags()
    checked.update(dict(flags))
    checked = _validate_relative_readiness_flags(checked, context=context)
    for field_name in (
        "source_read_preconditions_met",
        "benchmark_source_read_ready",
        "peer_source_read_ready",
        "benchmark_read_ready",
        "peer_read_ready",
        "benchmark_substitution_ready",
    ):
        if checked.get(field_name) is not False:
            raise ValueError(f"{context} cannot claim {field_name}")
    return checked


def _relative_peer_inference_policy(
    *,
    peer_basket_inferred: bool = False,
    registry_lookup_requested: bool = False,
    scalar_partitions_inferred: bool = False,
    alignment_frame_inferred: bool = False,
) -> dict[str, Any]:
    return {
        "peer_basket_inference_allowed": False,
        "disk_inference_allowed": False,
        "scalar_partition_inference_allowed": False,
        "alignment_frame_inference_allowed": False,
        "production_registry_lookup_allowed": False,
        "peer_basket_inferred": bool(peer_basket_inferred),
        "registry_lookup_requested": bool(registry_lookup_requested),
        "scalar_partitions_inferred": bool(scalar_partitions_inferred),
        "alignment_frame_inferred": bool(alignment_frame_inferred),
    }


def relative_benchmark_source_policy(
    *,
    benchmark: str,
    benchmark_source_kind: str = RELATIVE_BENCHMARK_SOURCE_CONFIG,
    benchmark_source_detail: str | None = None,
    substitution_policy: str = RELATIVE_BENCHMARK_SUBSTITUTION_POLICY_NONE,
    missing_benchmark_policy: str = "require",
    durable_registry_reference: str | None = None,
    benchmark_proxy_reference: str | None = None,
    registry_lookup_requested: bool = False,
    benchmark_substitution_inferred: bool = False,
    lifecycle_status: str = RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
    artifact_boundary: Mapping[str, Any] | None = None,
    readiness_flags: Mapping[str, Any] | None = None,
    schema_version: int = RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION,
) -> RelativeBenchmarkSourcePolicy:
    benchmark_text = str(benchmark or "").strip()
    if not benchmark_text:
        raise ValueError("relative benchmark source policy benchmark must be non-empty")
    source = str(benchmark_source_kind or "").strip()
    if not source:
        raise ValueError("relative benchmark source policy benchmark_source_kind must be non-empty")
    if source in RELATIVE_BENCHMARK_UNSUPPORTED_SOURCES:
        raise ValueError(f"relative benchmark source {source!r} is unsupported and not implemented")
    if source not in RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES:
        valid = ", ".join(RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown relative benchmark source {source!r}; expected one of: {valid}")
    if str(lifecycle_status) != RELATIVE_OWNERSHIP_LIFECYCLE_STATUS:
        raise ValueError(
            "relative benchmark source policy lifecycle_status must be "
            f"{RELATIVE_OWNERSHIP_LIFECYCLE_STATUS!r}"
        )
    if str(substitution_policy or "").strip() != RELATIVE_BENCHMARK_SUBSTITUTION_POLICY_NONE:
        raise ValueError("relative benchmark source policy cannot claim benchmark substitution")
    missing_policy = str(missing_benchmark_policy or "").strip()
    if missing_policy not in RELATIVE_MISSING_BENCHMARK_POLICIES:
        valid = ", ".join(RELATIVE_MISSING_BENCHMARK_POLICIES)
        raise ValueError(f"unsupported missing benchmark policy {missing_policy!r}; expected one of: {valid}")
    if durable_registry_reference is not None and str(durable_registry_reference).strip():
        raise ValueError("relative benchmark registry references are not implemented")
    if benchmark_proxy_reference is not None and str(benchmark_proxy_reference).strip():
        raise ValueError("relative benchmark proxy references are not implemented")
    if bool(registry_lookup_requested):
        raise ValueError("relative benchmark source policy cannot request registry lookup")
    if bool(benchmark_substitution_inferred):
        raise ValueError("relative benchmark source policy cannot infer benchmark substitution")
    boundary = _validate_relative_metadata_boundary(
        artifact_boundary or _default_relative_ownership_artifact_boundary(),
        context="Relative benchmark source policy",
    )
    flags = _validate_relative_readiness_flags(
        readiness_flags or _default_relative_ownership_readiness_flags(),
        context="Relative benchmark source policy",
    )
    detail = str(benchmark_source_detail or source).strip()
    if not detail:
        raise ValueError("relative benchmark source policy benchmark_source_detail must be non-empty")
    return RelativeBenchmarkSourcePolicy(
        schema_version=int(schema_version),
        artifact_kind=RELATIVE_BENCHMARK_SOURCE_POLICY_ARTIFACT_KIND,
        lifecycle_status=RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
        benchmark=benchmark_text,
        benchmark_source_kind=source,
        benchmark_source_detail=detail,
        substitution_policy=RELATIVE_BENCHMARK_SUBSTITUTION_POLICY_NONE,
        missing_benchmark_policy=missing_policy,
        implemented=True,
        support_status="implemented_scaffold_explicit_benchmark",
        durable_registry_reference=None,
        benchmark_proxy_reference=None,
        supported_benchmark_sources=tuple(RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES),
        unsupported_benchmark_sources=tuple(RELATIVE_BENCHMARK_UNSUPPORTED_SOURCES),
        readiness_flags=flags,
        artifact_boundary=boundary,
    )


def relative_peer_basket_lifecycle_policy(
    *,
    universe: str,
    peer_assets: Sequence[str] | None = None,
    min_peer_assets: int,
    peer_source_kind: str = RELATIVE_PEER_BASKET_SOURCE_CONFIG,
    owner: str = RELATIVE_OWNERSHIP_OWNER,
    lifecycle_status: str = RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
    refresh_mode: str = RELATIVE_PEER_BASKET_REFRESH_MODE,
    refresh_cadence: str = RELATIVE_PEER_BASKET_REFRESH_CADENCE,
    staleness_window: str = RELATIVE_PEER_BASKET_STALENESS_WINDOW,
    registry_support_status: str = RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS,
    registry_lookup_allowed: bool = False,
    explicit_peer_handoff_allowed: bool = True,
    failure_policy: str = RELATIVE_OWNERSHIP_FAILURE_POLICY,
    artifact_boundary: Mapping[str, Any] | None = None,
    readiness_flags: Mapping[str, Any] | None = None,
    schema_version: int = RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION,
) -> RelativePeerBasketLifecyclePolicy:
    universe_text = str(universe or "").strip()
    if not universe_text:
        raise ValueError("relative peer basket lifecycle policy universe must be non-empty")
    source = str(peer_source_kind or "").strip()
    if not source:
        raise ValueError("relative peer basket lifecycle policy peer_source_kind must be non-empty")
    if source in RELATIVE_PEER_BASKET_UNSUPPORTED_SOURCES:
        raise ValueError(f"relative peer basket source {source!r} is unsupported and not implemented")
    if source not in RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES:
        valid = ", ".join(RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown relative peer basket source {source!r}; expected one of: {valid}")
    if str(lifecycle_status) != RELATIVE_OWNERSHIP_LIFECYCLE_STATUS:
        raise ValueError(
            "relative peer basket lifecycle policy lifecycle_status must be "
            f"{RELATIVE_OWNERSHIP_LIFECYCLE_STATUS!r}"
        )
    peers = tuple(str(asset).strip() for asset in (peer_assets or ()) if str(asset).strip())
    if len(set(peers)) != len(peers):
        raise ValueError("relative peer basket lifecycle policy peer_assets must be unique")
    min_peers = int(min_peer_assets)
    if min_peers < 1:
        raise ValueError("relative peer basket lifecycle policy min_peer_assets must be positive")
    if bool(registry_lookup_allowed):
        raise ValueError("relative peer basket lifecycle policy cannot allow registry lookup")
    if not bool(explicit_peer_handoff_allowed):
        raise ValueError("relative peer basket lifecycle policy must allow explicit-peer handoff")
    if str(registry_support_status).strip() != RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS:
        raise ValueError(
            "relative peer basket lifecycle policy registry_support_status must remain "
            f"{RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS!r}"
        )
    for field_name, raw_value in (
        ("owner", owner),
        ("refresh_mode", refresh_mode),
        ("refresh_cadence", refresh_cadence),
        ("staleness_window", staleness_window),
        ("failure_policy", failure_policy),
    ):
        if not str(raw_value or "").strip():
            raise ValueError(f"relative peer basket lifecycle policy {field_name} must be non-empty")
    boundary = _validate_relative_metadata_boundary(
        artifact_boundary or _default_relative_ownership_artifact_boundary(),
        context="Relative peer basket lifecycle policy",
    )
    flags = _validate_relative_readiness_flags(
        readiness_flags or _default_relative_ownership_readiness_flags(),
        context="Relative peer basket lifecycle policy",
    )
    return RelativePeerBasketLifecyclePolicy(
        schema_version=int(schema_version),
        artifact_kind=RELATIVE_PEER_BASKET_LIFECYCLE_POLICY_ARTIFACT_KIND,
        owner=str(owner).strip(),
        lifecycle_status=RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
        universe=universe_text,
        peer_source_kind=source,
        refresh_mode=str(refresh_mode).strip(),
        refresh_cadence=str(refresh_cadence).strip(),
        staleness_window=str(staleness_window).strip(),
        registry_support_status=RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS,
        registry_lookup_allowed=False,
        explicit_peer_handoff_allowed=True,
        min_peer_assets=min_peers,
        peer_assets_count=len(peers),
        failure_policy=str(failure_policy).strip(),
        supported_peer_sources=tuple(RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES),
        unsupported_peer_sources=tuple(RELATIVE_PEER_BASKET_UNSUPPORTED_SOURCES),
        readiness_flags=flags,
        artifact_boundary=boundary,
        notes=(
            "Peer assets remain explicit scaffold inputs when supplied.",
            "Registry lookup, inferred peer baskets, source reads, and alignment frames are disabled.",
        ),
    )


def relative_peer_basket_source_policy(
    *,
    universe: str,
    peer_assets: Sequence[str] | None = None,
    min_peer_assets: int,
    peer_source_kind: str = RELATIVE_PEER_BASKET_SOURCE_CONFIG,
    peer_source_detail: str | None = None,
    durable_registry_reference: str | None = None,
    peer_basket_inferred: bool = False,
    registry_lookup_requested: bool = False,
    scalar_partitions_inferred: bool = False,
    alignment_frame_inferred: bool = False,
    lifecycle_status: str = RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
    artifact_boundary: Mapping[str, Any] | None = None,
    readiness_flags: Mapping[str, Any] | None = None,
    schema_version: int = RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION,
) -> RelativePeerBasketSourcePolicy:
    if str(lifecycle_status) != RELATIVE_OWNERSHIP_LIFECYCLE_STATUS:
        raise ValueError(
            "relative peer basket source policy lifecycle_status must be "
            f"{RELATIVE_OWNERSHIP_LIFECYCLE_STATUS!r}"
        )
    peers = tuple(str(asset).strip() for asset in (peer_assets or ()) if str(asset).strip())
    if len(set(peers)) != len(peers):
        raise ValueError("relative peer basket source policy peer_assets must be unique")
    if durable_registry_reference is not None and str(durable_registry_reference).strip():
        raise ValueError("relative peer basket registry references are not implemented")
    if any(
        (
            bool(peer_basket_inferred),
            bool(registry_lookup_requested),
            bool(scalar_partitions_inferred),
            bool(alignment_frame_inferred),
        )
    ):
        raise ValueError("relative peer basket source policy cannot claim inferred or registry-backed peers")
    boundary = _validate_relative_metadata_boundary(
        artifact_boundary or _default_relative_ownership_artifact_boundary(),
        context="Relative peer basket source policy",
    )
    flags = _validate_relative_readiness_flags(
        readiness_flags or _default_relative_ownership_readiness_flags(),
        context="Relative peer basket source policy",
    )
    lifecycle = relative_peer_basket_lifecycle_policy(
        universe=universe,
        peer_assets=peers,
        min_peer_assets=int(min_peer_assets),
        peer_source_kind=peer_source_kind,
        lifecycle_status=RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
        artifact_boundary=boundary,
        readiness_flags=flags,
    ).as_dict()
    source = str(peer_source_kind or "").strip()
    detail = str(peer_source_detail or source).strip()
    if not detail:
        raise ValueError("relative peer basket source policy peer_source_detail must be non-empty")
    supplied = bool(peers)
    return RelativePeerBasketSourcePolicy(
        schema_version=int(schema_version),
        artifact_kind=RELATIVE_PEER_BASKET_SOURCE_POLICY_ARTIFACT_KIND,
        lifecycle_status=RELATIVE_OWNERSHIP_LIFECYCLE_STATUS,
        universe=str(universe).strip(),
        peer_assets=peers,
        min_peer_assets=int(min_peer_assets),
        peer_source_kind=source,
        peer_source_detail=detail,
        implemented=True,
        support_status="implemented_scaffold_explicit_peers" if supplied else "scaffold_peer_assets_not_supplied",
        explicit_peer_assets_required=False,
        explicit_peer_assets_supplied=supplied,
        durable_registry_reference=None,
        inference_policy=_relative_peer_inference_policy(
            peer_basket_inferred=peer_basket_inferred,
            registry_lookup_requested=registry_lookup_requested,
            scalar_partitions_inferred=scalar_partitions_inferred,
            alignment_frame_inferred=alignment_frame_inferred,
        ),
        lifecycle_policy=lifecycle,
        artifact_boundary=boundary,
    )


def _normalized_unique_texts(values: Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in (values or ()))
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} must not include blank values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique")
    return normalized


def _validate_relative_benchmark_source_policy_payload(
    payload: Mapping[str, Any],
    *,
    benchmark: str,
    missing_benchmark_policy: str,
) -> dict[str, Any]:
    data = dict(payload)
    if int(data.get("schema_version", 0) or 0) != RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION:
        raise ValueError("relative benchmark source policy schema_version is unsupported")
    if str(data.get("artifact_kind", "")).strip() != RELATIVE_BENCHMARK_SOURCE_POLICY_ARTIFACT_KIND:
        raise ValueError("relative benchmark source policy artifact_kind is unsupported")
    if str(data.get("lifecycle_status", "")).strip() != RELATIVE_OWNERSHIP_LIFECYCLE_STATUS:
        raise ValueError("relative benchmark source policy lifecycle_status is unsupported")
    if str(data.get("benchmark", "")).strip() != str(benchmark).strip():
        raise ValueError("relative benchmark source policy benchmark must match precondition benchmark")
    if str(data.get("missing_benchmark_policy", "")).strip() != str(missing_benchmark_policy).strip():
        raise ValueError(
            "relative benchmark source policy missing_benchmark_policy must match precondition policy"
        )
    source = str(data.get("benchmark_source_kind", "")).strip()
    if source in RELATIVE_BENCHMARK_UNSUPPORTED_SOURCES:
        raise ValueError(f"relative benchmark source {source!r} is unsupported and not implemented")
    if source not in RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES:
        valid = ", ".join(RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown relative benchmark source {source!r}; expected one of: {valid}")
    if str(data.get("substitution_policy", "")).strip() != RELATIVE_BENCHMARK_SUBSTITUTION_POLICY_NONE:
        raise ValueError("relative benchmark source policy cannot claim benchmark substitution")
    if data.get("durable_registry_reference") is not None:
        raise ValueError("relative benchmark source policy cannot claim registry references")
    if data.get("benchmark_proxy_reference") is not None:
        raise ValueError("relative benchmark source policy cannot claim benchmark proxy references")
    data["artifact_boundary"] = _validate_relative_metadata_boundary(
        data.get("artifact_boundary", {}),
        context="Relative benchmark source policy",
    )
    data["readiness_flags"] = _validate_relative_readiness_flags(
        data.get("readiness_flags", {}),
        context="Relative benchmark source policy",
    )
    return data


def _validate_relative_peer_basket_lifecycle_policy_payload(
    payload: Mapping[str, Any],
    *,
    universe: str,
    peer_assets: Sequence[str],
    min_peer_assets: int,
) -> dict[str, Any]:
    data = dict(payload)
    if int(data.get("schema_version", 0) or 0) != RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION:
        raise ValueError("relative peer basket lifecycle policy schema_version is unsupported")
    if str(data.get("artifact_kind", "")).strip() != RELATIVE_PEER_BASKET_LIFECYCLE_POLICY_ARTIFACT_KIND:
        raise ValueError("relative peer basket lifecycle policy artifact_kind is unsupported")
    if str(data.get("lifecycle_status", "")).strip() != RELATIVE_OWNERSHIP_LIFECYCLE_STATUS:
        raise ValueError("relative peer basket lifecycle policy lifecycle_status is unsupported")
    if str(data.get("universe", "")).strip() != str(universe).strip():
        raise ValueError("relative peer basket lifecycle policy universe must match precondition universe")
    if data.get("registry_lookup_allowed") is not False:
        raise ValueError("relative peer basket lifecycle policy cannot allow registry lookup")
    if data.get("explicit_peer_handoff_allowed") is not True:
        raise ValueError("relative peer basket lifecycle policy must allow explicit-peer handoff")
    if str(data.get("registry_support_status", "")).strip() != RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS:
        raise ValueError("relative peer basket lifecycle policy registry_support_status is unsupported")
    if int(data.get("min_peer_assets", 0)) != int(min_peer_assets):
        raise ValueError("relative peer basket lifecycle policy min_peer_assets must match precondition")
    if int(data.get("peer_assets_count", -1)) != len(tuple(peer_assets)):
        raise ValueError("relative peer basket lifecycle policy peer_assets_count must match precondition")
    source = str(data.get("peer_source_kind", "")).strip()
    if source in RELATIVE_PEER_BASKET_UNSUPPORTED_SOURCES:
        raise ValueError(f"relative peer basket source {source!r} is unsupported and not implemented")
    if source not in RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES:
        valid = ", ".join(RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown relative peer basket source {source!r}; expected one of: {valid}")
    data["artifact_boundary"] = _validate_relative_metadata_boundary(
        data.get("artifact_boundary", {}),
        context="Relative peer basket lifecycle policy",
    )
    data["readiness_flags"] = _validate_relative_readiness_flags(
        data.get("readiness_flags", {}),
        context="Relative peer basket lifecycle policy",
    )
    return data


def _validate_relative_peer_basket_source_policy_payload(
    payload: Mapping[str, Any],
    *,
    universe: str,
    peer_assets: Sequence[str],
    min_peer_assets: int,
) -> dict[str, Any]:
    data = dict(payload)
    if int(data.get("schema_version", 0) or 0) != RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION:
        raise ValueError("relative peer basket source policy schema_version is unsupported")
    if str(data.get("artifact_kind", "")).strip() != RELATIVE_PEER_BASKET_SOURCE_POLICY_ARTIFACT_KIND:
        raise ValueError("relative peer basket source policy artifact_kind is unsupported")
    if str(data.get("lifecycle_status", "")).strip() != RELATIVE_OWNERSHIP_LIFECYCLE_STATUS:
        raise ValueError("relative peer basket source policy lifecycle_status is unsupported")
    if str(data.get("universe", "")).strip() != str(universe).strip():
        raise ValueError("relative peer basket source policy universe must match precondition universe")
    payload_peers = tuple(str(asset).strip() for asset in data.get("peer_assets", ()))
    if payload_peers != tuple(peer_assets):
        raise ValueError("relative peer basket source policy peer_assets must match precondition peer_assets")
    if int(data.get("min_peer_assets", 0)) != int(min_peer_assets):
        raise ValueError("relative peer basket source policy min_peer_assets must match precondition")
    if data.get("durable_registry_reference") is not None:
        raise ValueError("relative peer basket source policy cannot claim registry references")
    inference_policy = dict(data.get("inference_policy", {}))
    for field_name in (
        "peer_basket_inferred",
        "registry_lookup_requested",
        "scalar_partitions_inferred",
        "alignment_frame_inferred",
    ):
        if inference_policy.get(field_name) is not False:
            raise ValueError(f"relative peer basket source policy cannot claim {field_name}")
    source = str(data.get("peer_source_kind", "")).strip()
    if source in RELATIVE_PEER_BASKET_UNSUPPORTED_SOURCES:
        raise ValueError(f"relative peer basket source {source!r} is unsupported and not implemented")
    if source not in RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES:
        valid = ", ".join(RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES)
        raise ValueError(f"unknown relative peer basket source {source!r}; expected one of: {valid}")
    data["artifact_boundary"] = _validate_relative_metadata_boundary(
        data.get("artifact_boundary", {}),
        context="Relative peer basket source policy",
    )
    lifecycle = _validate_relative_peer_basket_lifecycle_policy_payload(
        data.get("lifecycle_policy", {}),
        universe=universe,
        peer_assets=peer_assets,
        min_peer_assets=int(min_peer_assets),
    )
    data["lifecycle_policy"] = lifecycle
    return data


def _relative_source_requirements(
    *,
    required_intervals: Sequence[int],
    peer_assets_count: int,
    min_peer_assets: int,
    missing_benchmark_policy: str,
) -> dict[str, Any]:
    return {
        "benchmark_source_required": True,
        "peer_basket_source_required": True,
        "required_intervals": [int(interval) for interval in required_intervals],
        "min_peer_assets": int(min_peer_assets),
        "peer_assets_count": int(peer_assets_count),
        "min_peer_assets_met": bool(int(peer_assets_count) >= int(min_peer_assets)),
        "missing_benchmark_policy": str(missing_benchmark_policy),
        "source_root_required_for_this_artifact": False,
        "source_probe_allowed": False,
        "source_reader_allowed": False,
        "benchmark_reader_allowed": False,
        "peer_reader_allowed": False,
        "alignment_frame_allowed": False,
        "benchmark_substitution_allowed": False,
        "benchmark_proxy_allowed": False,
    }


def _relative_ownership_references(
    *,
    benchmark_source_policy: Mapping[str, Any],
    peer_basket_lifecycle_policy: Mapping[str, Any],
    peer_basket_source_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_source_policy": {
            "artifact_kind": benchmark_source_policy.get("artifact_kind"),
            "benchmark": benchmark_source_policy.get("benchmark"),
            "benchmark_source_kind": benchmark_source_policy.get("benchmark_source_kind"),
            "missing_benchmark_policy": benchmark_source_policy.get("missing_benchmark_policy"),
            "substitution_policy": benchmark_source_policy.get("substitution_policy"),
        },
        "peer_basket_lifecycle_policy": {
            "artifact_kind": peer_basket_lifecycle_policy.get("artifact_kind"),
            "universe": peer_basket_lifecycle_policy.get("universe"),
            "peer_source_kind": peer_basket_lifecycle_policy.get("peer_source_kind"),
            "peer_assets_count": peer_basket_lifecycle_policy.get("peer_assets_count"),
            "min_peer_assets": peer_basket_lifecycle_policy.get("min_peer_assets"),
            "registry_lookup_allowed": peer_basket_lifecycle_policy.get("registry_lookup_allowed"),
        },
        "peer_basket_source_policy": {
            "artifact_kind": peer_basket_source_policy.get("artifact_kind"),
            "universe": peer_basket_source_policy.get("universe"),
            "peer_source_kind": peer_basket_source_policy.get("peer_source_kind"),
            "peer_assets_count": peer_basket_source_policy.get("peer_assets_count"),
            "explicit_peer_assets_supplied": peer_basket_source_policy.get("explicit_peer_assets_supplied"),
        },
    }


def _validate_relative_source_read_precondition_inputs(
    *,
    universe: str,
    benchmark: str,
    assets: Sequence[str],
    peer_assets: Sequence[str],
    min_peer_assets: int,
    band: str,
    missing_benchmark_policy: str,
    status: str,
    failure_policy: str,
    artifact_boundary: Mapping[str, Any],
    readiness_flags: Mapping[str, Any],
) -> RelativeSourceReadPreconditionValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    universe_text = str(universe).strip()
    if not universe_text:
        errors.append("universe must be non-empty")
    else:
        try:
            _safe_snapshot_path_part(universe_text, field_name="relative source-read precondition universe")
        except ValueError as exc:
            errors.append(str(exc))

    benchmark_text = str(benchmark).strip()
    if not benchmark_text:
        errors.append("benchmark must be non-empty")
    else:
        try:
            _safe_snapshot_path_part(benchmark_text, field_name="relative source-read precondition benchmark")
        except ValueError as exc:
            errors.append(str(exc))

    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        errors.append(f"Unsupported Regime band {band!r}; expected one of: {valid}")

    if str(missing_benchmark_policy) not in RELATIVE_MISSING_BENCHMARK_POLICIES:
        valid = ", ".join(RELATIVE_MISSING_BENCHMARK_POLICIES)
        errors.append(f"unsupported missing benchmark policy {missing_benchmark_policy!r}; expected one of: {valid}")

    if str(status) != PATHWAY_MANIFEST_STATUS:
        errors.append(f"status must be {PATHWAY_MANIFEST_STATUS!r}")
    if str(failure_policy).strip() != RELATIVE_SOURCE_READ_PRECONDITION_FAILURE_POLICY:
        errors.append("relative source-read precondition failure_policy is unsupported")

    if int(min_peer_assets) < 1:
        errors.append("min_peer_assets must be positive")
    if not peer_assets:
        warnings.append("peer_assets are empty; source_read_ready remains false")
    elif len(peer_assets) < int(min_peer_assets):
        warnings.append("peer_assets_count is below min_peer_assets; source_read_ready remains false")
    if not assets:
        warnings.append("assets are empty; precondition applies to future explicit relative assets")

    if artifact_boundary.get("source_read_precondition_only") is not True:
        errors.append("relative source-read precondition must be precondition-only")
    for field_name in (
        "production_writes_enabled",
        "parquet_writes_enabled",
        "definition_writes_enabled",
        "source_read_ready",
        "alignment_frame_ready",
        "production_output_ready",
        "downstream_reader_ready",
        "benchmark_registry_ready",
        "benchmark_proxy_ready",
        "peer_registry_ready",
        "benchmark_substitution_inferred",
        "peer_basket_inferred",
        "source_probe_enabled",
        "source_reader_enabled",
        "benchmark_reader_enabled",
        "peer_reader_enabled",
        "benchmark_substitution_enabled",
        "benchmark_proxy_enabled",
        "alignment_frame_built",
        "registry_lookup_allowed",
        "inference_allowed",
    ):
        if artifact_boundary.get(field_name) is not False:
            errors.append(f"relative source-read precondition cannot claim {field_name}")

    for field_name in (
        "source_read_preconditions_met",
        "source_read_ready",
        "benchmark_source_read_ready",
        "peer_source_read_ready",
        "benchmark_read_ready",
        "peer_read_ready",
        "benchmark_substitution_ready",
        "alignment_frame_ready",
        "production_output_ready",
        "downstream_reader_ready",
        "benchmark_registry_ready",
        "benchmark_proxy_ready",
        "peer_registry_ready",
        "parquet_writer_ready",
        "definition_writer_ready",
    ):
        if readiness_flags.get(field_name) is not False:
            errors.append(f"readiness_flags cannot claim {field_name}")

    return RelativeSourceReadPreconditionValidationResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def relative_source_read_precondition(
    *,
    run_id: str,
    universe: str,
    benchmark: str,
    assets: Sequence[str] | None = None,
    peer_assets: Sequence[str] | None = None,
    min_peer_assets: int,
    band: str,
    missing_benchmark_policy: str = "require",
    benchmark_source_policy: Mapping[str, Any] | None = None,
    peer_basket_lifecycle_policy: Mapping[str, Any] | None = None,
    peer_basket_source_policy: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    status: str = PATHWAY_MANIFEST_STATUS,
    failure_policy: str = RELATIVE_SOURCE_READ_PRECONDITION_FAILURE_POLICY,
    artifact_boundary: Mapping[str, Any] | None = None,
    readiness_flags: Mapping[str, Any] | None = None,
    schema_version: int = RELATIVE_SOURCE_READ_PRECONDITION_SCHEMA_VERSION,
) -> RelativeSourceReadPrecondition:
    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        raise ValueError(f"Unsupported Regime band {band!r}; expected one of: {valid}")
    universe_text = str(universe).strip()
    benchmark_text = str(benchmark).strip()
    asset_values = _normalized_unique_texts(assets, field_name="relative source-read precondition assets")
    peer_values = _normalized_unique_texts(peer_assets, field_name="relative source-read precondition peer_assets")
    min_peers = int(min_peer_assets)
    missing_policy = str(missing_benchmark_policy).strip()
    band_contract = REGIME_BANDS[str(band)]
    required_intervals = tuple(int(interval) for interval in band_contract.member_intervals)
    boundary = _validate_relative_source_read_precondition_boundary(
        artifact_boundary or _default_relative_source_read_precondition_artifact_boundary(),
        context="Relative source-read precondition",
    )
    flags = _validate_relative_source_read_precondition_readiness_flags(
        readiness_flags or _default_relative_source_read_precondition_readiness_flags(),
        context="Relative source-read precondition",
    )
    benchmark_policy = _validate_relative_benchmark_source_policy_payload(
        benchmark_source_policy
        or relative_benchmark_source_policy(
            benchmark=benchmark_text,
            missing_benchmark_policy=missing_policy,
        ).as_dict(),
        benchmark=benchmark_text,
        missing_benchmark_policy=missing_policy,
    )
    peer_lifecycle = _validate_relative_peer_basket_lifecycle_policy_payload(
        peer_basket_lifecycle_policy
        or relative_peer_basket_lifecycle_policy(
            universe=universe_text,
            peer_assets=peer_values,
            min_peer_assets=min_peers,
        ).as_dict(),
        universe=universe_text,
        peer_assets=peer_values,
        min_peer_assets=min_peers,
    )
    peer_source = _validate_relative_peer_basket_source_policy_payload(
        peer_basket_source_policy
        or relative_peer_basket_source_policy(
            universe=universe_text,
            peer_assets=peer_values,
            min_peer_assets=min_peers,
        ).as_dict(),
        universe=universe_text,
        peer_assets=peer_values,
        min_peer_assets=min_peers,
    )
    validation = _validate_relative_source_read_precondition_inputs(
        universe=universe_text,
        benchmark=benchmark_text,
        assets=asset_values,
        peer_assets=peer_values,
        min_peer_assets=min_peers,
        band=str(band),
        missing_benchmark_policy=missing_policy,
        status=str(status),
        failure_policy=str(failure_policy),
        artifact_boundary=boundary,
        readiness_flags=flags,
    )
    if not validation.ok:
        raise ValueError("Relative source-read precondition invalid: " + "; ".join(validation.errors))
    requirements = _relative_source_requirements(
        required_intervals=required_intervals,
        peer_assets_count=len(peer_values),
        min_peer_assets=min_peers,
        missing_benchmark_policy=missing_policy,
    )
    references = _relative_ownership_references(
        benchmark_source_policy=benchmark_policy,
        peer_basket_lifecycle_policy=peer_lifecycle,
        peer_basket_source_policy=peer_source,
    )
    return RelativeSourceReadPrecondition(
        schema_version=int(schema_version),
        artifact_kind=RELATIVE_SOURCE_READ_PRECONDITION_ARTIFACT_KIND,
        status=str(status),
        pathway="relative_state",
        run_id=str(run_id),
        created_at_utc=_utc_timestamp_text(created_at_utc),
        universe=universe_text,
        benchmark=benchmark_text,
        assets=asset_values,
        peer_assets=peer_values,
        min_peer_assets=min_peers,
        band=str(band),
        ceiling_interval_min=int(band_contract.ceiling_interval_min),
        required_intervals=required_intervals,
        missing_benchmark_policy=missing_policy,
        benchmark_source_policy=benchmark_policy,
        peer_basket_lifecycle_policy=peer_lifecycle,
        peer_basket_source_policy=peer_source,
        source_requirements=requirements,
        ownership_references=references,
        failure_policy=str(failure_policy).strip(),
        readiness_flags=flags,
        validation_result=validation,
        artifact_boundary=boundary,
    )


def market_membership_snapshot_provenance(
    *,
    membership_source: str,
    member_assets: Sequence[str],
    run_id: str,
    source_detail: str | None = None,
    source_feature_root: Path | str | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = market_membership_source_policy(
        membership_source=membership_source,
        member_assets=member_assets,
        source_detail=source_detail,
        source_feature_root=source_feature_root,
    )
    provenance: dict[str, Any] = {
        "pathway": "market_state",
        "run_id": str(run_id),
        "lifecycle_status": policy.lifecycle_status,
        "source_kind": policy.source_kind,
        "source_detail": policy.source_detail,
        "member_assets_source": policy.source_kind,
    }
    if source_feature_root is not None:
        provenance["source_feature_root"] = str(Path(source_feature_root))
    for key, value in dict(extra_provenance or {}).items():
        provenance[str(key)] = value
    return provenance


def _validate_market_universe_lifecycle_policy_payload(
    payload: Mapping[str, Any],
    *,
    membership_source: str,
) -> dict[str, Any]:
    data = dict(payload)
    if not data:
        return market_universe_lifecycle_policy(membership_source=membership_source).as_dict()
    if int(data.get("schema_version", 0) or 0) != MARKET_UNIVERSE_LIFECYCLE_POLICY_SCHEMA_VERSION:
        raise ValueError("Market universe lifecycle policy schema_version is unsupported")
    if str(data.get("artifact_kind", "")).strip() != MARKET_UNIVERSE_LIFECYCLE_POLICY_ARTIFACT_KIND:
        raise ValueError("Market universe lifecycle policy artifact_kind is unsupported")
    if str(data.get("membership_source", "")).strip() != str(membership_source).strip():
        raise ValueError("Market universe lifecycle policy membership_source must match membership_source")
    if str(data.get("lifecycle_status", "")).strip() != MARKET_MEMBERSHIP_LIFECYCLE_STATUS:
        raise ValueError("Market universe lifecycle policy lifecycle_status is unsupported")
    if str(data.get("registry_support_status", "")).strip() != MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS:
        raise ValueError("Market universe lifecycle policy registry_support_status is unsupported")
    if data.get("registry_lookup_allowed") is not False:
        raise ValueError("Market universe lifecycle policy cannot allow registry lookup")
    if data.get("explicit_member_handoff_allowed") is not True:
        raise ValueError("Market universe lifecycle policy must allow explicit-member handoff")
    if data.get("aggregation_readiness") is not False:
        raise ValueError("Market universe lifecycle policy cannot claim aggregation readiness")
    return data


def validate_market_universe_membership_snapshot_inputs(
    *,
    universe: str,
    member_assets: Sequence[str],
    membership_source: str,
    provenance: Mapping[str, Any] | None,
    snapshot_timestamp_utc: str | None,
    snapshot_scope: str,
    min_assets: int,
    status: str = PATHWAY_MANIFEST_STATUS,
) -> MarketUniverseMembershipValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    universe_text = str(universe).strip()
    if not universe_text:
        errors.append("universe must be non-empty")
    else:
        try:
            _safe_snapshot_path_part(universe_text, field_name="universe")
        except ValueError as exc:
            errors.append(str(exc))

    raw_members = tuple(str(asset).strip() for asset in member_assets)
    blank_members = sum(1 for asset in raw_members if not asset)
    nonblank_members = tuple(asset for asset in raw_members if asset)
    unique_members = _sorted_unique_members(nonblank_members)
    if blank_members:
        errors.append("member_assets must not include blank assets")
    if not nonblank_members:
        errors.append("member_assets must be non-empty")
    if len(set(nonblank_members)) != len(nonblank_members):
        errors.append("member_assets must be unique")
    for asset in unique_members:
        try:
            _safe_snapshot_path_part(asset, field_name=f"member asset {asset!r}")
        except ValueError as exc:
            errors.append(str(exc))

    source = str(membership_source).strip()
    if not source:
        errors.append("membership_source must be non-empty")
    else:
        try:
            market_membership_source_policy(
                membership_source=source,
                member_assets=unique_members,
            )
        except ValueError as exc:
            errors.append(str(exc))
    scope = str(snapshot_scope).strip()
    if not scope:
        errors.append("snapshot_scope must be non-empty")

    try:
        min_assets_int = int(min_assets)
    except (TypeError, ValueError):
        min_assets_int = 0
        errors.append("min_assets must be an integer")
    if min_assets_int < 1:
        errors.append("min_assets must be positive")
    elif unique_members and min_assets_int > len(unique_members):
        errors.append("min_assets cannot exceed unique member_assets count")

    provenance_payload = dict(provenance or {})
    if not provenance_payload:
        errors.append("provenance must be non-empty")
    for key, value in provenance_payload.items():
        if not str(key).strip():
            errors.append("provenance keys must be non-empty")
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"provenance field {key!r} must be non-empty")

    try:
        _utc_timestamp_text(snapshot_timestamp_utc)
    except ValueError as exc:
        errors.append(str(exc))

    if str(status) != PATHWAY_MANIFEST_STATUS:
        errors.append(f"status must be {PATHWAY_MANIFEST_STATUS!r}")

    return MarketUniverseMembershipValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def market_universe_membership_snapshot(
    *,
    universe: str,
    member_assets: Sequence[str],
    membership_source: str,
    provenance: Mapping[str, Any],
    membership_source_policy: Mapping[str, Any] | None = None,
    snapshot_timestamp_utc: str | None = None,
    snapshot_scope: str = "market_state_universe_membership",
    min_assets: int,
    status: str = PATHWAY_MANIFEST_STATUS,
    schema_version: int = MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_SCHEMA_VERSION,
    artifact_boundary: Mapping[str, Any] | None = None,
) -> MarketUniverseMembershipSnapshot:
    timestamp = _utc_timestamp_text(snapshot_timestamp_utc)
    validation = validate_market_universe_membership_snapshot_inputs(
        universe=universe,
        member_assets=member_assets,
        membership_source=membership_source,
        provenance=provenance,
        snapshot_timestamp_utc=timestamp,
        snapshot_scope=snapshot_scope,
        min_assets=int(min_assets),
        status=status,
    )
    if not validation.ok:
        raise ValueError("Market universe membership snapshot invalid: " + "; ".join(validation.errors))
    boundary = _validate_scaffold_artifact_boundary(
        artifact_boundary or _default_market_membership_snapshot_artifact_boundary(),
        context="Market universe membership snapshot",
    )
    if boundary.get("membership_inferred_from_disk") is not False:
        raise ValueError("Market universe membership snapshot cannot infer membership from disk")
    if boundary.get("aggregation_frame_built") is not False:
        raise ValueError("Market universe membership snapshot cannot claim aggregation-frame construction")
    if boundary.get("aggregation_values_built") is not False:
        raise ValueError("Market universe membership snapshot cannot claim aggregation values")
    policy = dict(
        membership_source_policy
        or market_membership_source_policy(
            membership_source=str(membership_source).strip(),
            member_assets=_sorted_unique_members(member_assets),
        ).as_dict()
    )
    if policy.get("source_kind") != str(membership_source).strip():
        raise ValueError("Market universe membership snapshot policy source_kind must match membership_source")
    if policy.get("lifecycle_status") != MARKET_MEMBERSHIP_LIFECYCLE_STATUS:
        raise ValueError("Market universe membership snapshot policy lifecycle_status is unsupported")
    if policy.get("durable_registry_reference") is not None:
        raise ValueError("Market universe membership snapshot cannot claim a durable registry reference")
    inference_policy = dict(policy.get("inference_policy") or {})
    for field_name in (
        "membership_inferred_from_disk",
        "scalar_partitions_inferred",
        "aggregate_frame_inferred",
        "registry_lookup_requested",
    ):
        if inference_policy.get(field_name):
            raise ValueError(f"Market universe membership snapshot cannot claim {field_name}")
    policy["lifecycle_policy"] = _validate_market_universe_lifecycle_policy_payload(
        policy.get("lifecycle_policy") or {},
        membership_source=str(membership_source).strip(),
    )
    return MarketUniverseMembershipSnapshot(
        schema_version=int(schema_version),
        artifact_kind=MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_ARTIFACT_KIND,
        status=str(status),
        universe=str(universe).strip(),
        member_assets=_sorted_unique_members(member_assets),
        min_assets=int(min_assets),
        membership_source=str(membership_source).strip(),
        provenance=dict(provenance),
        membership_source_policy=policy,
        snapshot_timestamp_utc=timestamp,
        snapshot_scope=str(snapshot_scope).strip(),
        validation_result=validation,
        artifact_boundary=boundary,
    )


def market_universe_snapshot_metadata(
    *,
    universe: str,
    member_assets: Sequence[str],
    membership_source: str,
    snapshot_timestamp_utc: str | None = None,
    snapshot_scope: str = "market_state_aggregation_frame",
    min_assets: int,
    status: str = PATHWAY_MANIFEST_STATUS,
) -> MarketUniverseSnapshotMetadata:
    universe_text = str(universe).strip()
    members = tuple(str(asset).strip() for asset in member_assets if str(asset).strip())
    source = str(membership_source).strip()
    scope = str(snapshot_scope).strip()
    if not universe_text:
        raise ValueError("Market universe snapshot universe must be non-empty")
    if not members:
        raise ValueError("Market universe snapshot member_assets must be non-empty")
    if not source:
        raise ValueError("Market universe snapshot membership_source must be non-empty")
    if not scope:
        raise ValueError("Market universe snapshot snapshot_scope must be non-empty")
    if int(min_assets) < 1:
        raise ValueError("Market universe snapshot min_assets must be positive")
    if int(min_assets) > len(members):
        raise ValueError("Market universe snapshot min_assets cannot exceed member_assets count")
    if str(status) != PATHWAY_MANIFEST_STATUS:
        raise ValueError(f"Market universe snapshot status must be {PATHWAY_MANIFEST_STATUS!r}")
    timestamp = snapshot_timestamp_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return MarketUniverseSnapshotMetadata(
        universe=universe_text,
        member_assets=members,
        membership_source=source,
        snapshot_timestamp_utc=str(timestamp),
        snapshot_scope=scope,
        min_assets=int(min_assets),
        status=str(status),
    )


def _default_market_source_coverage_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "market_source_coverage_json_only",
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "aggregation_frame_built": False,
        "aggregation_values_built": False,
        "membership_inferred_from_disk": False,
    }


def _extract_source_summary(source_probe: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if source_probe is None:
        return None
    payload = dict(source_probe)
    if "source_summary" in payload and isinstance(payload.get("source_summary"), Mapping):
        return dict(payload["source_summary"])
    return payload


def _snapshot_reference(
    membership_snapshot: Mapping[str, Any] | None,
    *,
    membership_snapshot_path: Path | str | None,
    universe: str,
    member_assets: Sequence[str],
    membership_source: str,
    min_assets: int,
) -> dict[str, Any]:
    snapshot = dict(membership_snapshot or {})
    policy = dict(snapshot.get("membership_source_policy", {}))
    lifecycle_policy = _validate_market_universe_lifecycle_policy_payload(
        policy.get("lifecycle_policy") or {},
        membership_source=str(snapshot.get("membership_source", membership_source)).strip(),
    )
    return {
        "path": str(membership_snapshot_path) if membership_snapshot_path is not None else None,
        "schema_version": snapshot.get("schema_version"),
        "artifact_kind": snapshot.get("artifact_kind"),
        "universe": snapshot.get("universe", universe),
        "member_assets_count": int(snapshot.get("member_assets_count", len(member_assets)) or 0),
        "member_assets": list(snapshot.get("member_assets", list(member_assets))),
        "membership_source": snapshot.get("membership_source", membership_source),
        "membership_source_policy": policy,
        "lifecycle_policy": lifecycle_policy,
        "snapshot_scope": snapshot.get("snapshot_scope"),
        "min_assets": int(snapshot.get("min_assets", min_assets) or 0),
        "status": snapshot.get("status"),
    }


def _source_probe_reference(
    source_probe: Mapping[str, Any] | None,
    source_summary: Mapping[str, Any] | None,
    *,
    source_probe_path: Path | str | None,
) -> dict[str, Any] | None:
    if source_probe is None and source_summary is None and source_probe_path is None:
        return None
    probe = dict(source_probe or {})
    summary = dict(source_summary or {})
    lifecycle_payload = summary.get("lifecycle_policy")
    membership_source = str(summary.get("membership_source") or summary.get("member_assets_source") or "").strip()
    lifecycle_policy = (
        _validate_market_universe_lifecycle_policy_payload(
            dict(lifecycle_payload),
            membership_source=membership_source,
        )
        if isinstance(lifecycle_payload, Mapping) and membership_source
        else {}
    )
    return {
        "path": str(source_probe_path) if source_probe_path is not None else None,
        "schema_version": probe.get("schema_version"),
        "pathway": probe.get("pathway", summary.get("pathway")),
        "run_id": probe.get("run_id"),
        "created_at_utc": probe.get("created_at_utc"),
        "probe_kind": summary.get("probe_kind"),
        "lifecycle_policy": lifecycle_policy,
        "status": probe.get("status", summary.get("status")),
    }


def _default_source_availability_readiness_boundary() -> dict[str, Any]:
    return {
        "source_availability_metadata_only": True,
        "source_read_precondition_only": False,
        "scalar_partition_coverage_is_aggregate_feature_readiness": False,
        "aggregation_readiness": False,
        "aggregate_frame_construction_allowed": False,
        "aggregation_frame_built": False,
        "aggregation_values_built": False,
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "registry_lookup_allowed": False,
        "membership_inferred_from_disk": False,
        "durable_registry_reference": None,
        "downstream_reader_ready": False,
    }


def _validate_source_availability_reference(reference: Mapping[str, Any], *, context: str) -> None:
    if not reference:
        return
    status = reference.get("status")
    if status is not None and str(status) != PATHWAY_MANIFEST_STATUS:
        raise ValueError(f"{context} reference status must be {PATHWAY_MANIFEST_STATUS!r}")
    lifecycle_policy = reference.get("lifecycle_policy")
    if isinstance(lifecycle_policy, Mapping) and lifecycle_policy:
        _validate_market_universe_lifecycle_policy_payload(
            lifecycle_policy,
            membership_source=str(lifecycle_policy.get("membership_source") or reference.get("membership_source") or ""),
        )
    membership_source_policy = reference.get("membership_source_policy")
    if isinstance(membership_source_policy, Mapping):
        if membership_source_policy.get("durable_registry_reference") is not None:
            raise ValueError(f"{context} reference cannot claim a durable registry reference")
        nested_lifecycle = membership_source_policy.get("lifecycle_policy")
        if isinstance(nested_lifecycle, Mapping) and nested_lifecycle:
            _validate_market_universe_lifecycle_policy_payload(
                nested_lifecycle,
                membership_source=str(
                    nested_lifecycle.get("membership_source")
                    or membership_source_policy.get("source_kind")
                    or reference.get("membership_source")
                    or ""
                ),
            )
    artifact_boundary = reference.get("artifact_boundary")
    if isinstance(artifact_boundary, Mapping) and artifact_boundary:
        _validate_scaffold_artifact_boundary(artifact_boundary, context=f"{context} reference")
        for field_name in ("membership_inferred_from_disk", "aggregation_frame_built", "aggregation_values_built"):
            if artifact_boundary.get(field_name) is not False:
                raise ValueError(f"{context} reference cannot claim {field_name}")
        if artifact_boundary.get("aggregation_readiness") not in (None, False):
            raise ValueError(f"{context} reference cannot claim aggregation_readiness")
    readiness_flags = reference.get("readiness_flags")
    if isinstance(readiness_flags, Mapping) and readiness_flags:
        for field_name in (
            "aggregation_readiness",
            "aggregate_frame_construction_allowed",
            "aggregate_values_available",
            "parquet_writer_ready",
            "definition_writer_ready",
            "production_output_ready",
            "membership_registry_ready",
        ):
            if readiness_flags.get(field_name) is not False:
                raise ValueError(f"{context} reference cannot claim {field_name}")


def _coverage_diagnostic_reference(
    source_coverage_diagnostic: Mapping[str, Any] | None,
    *,
    source_coverage_diagnostic_path: Path | str | None,
    universe: str,
    band: str,
    run_id: str,
    membership_source: str,
    source_coverage_status: str,
) -> dict[str, Any] | None:
    if source_coverage_diagnostic is None and source_coverage_diagnostic_path is None:
        return None
    payload = dict(source_coverage_diagnostic or {})
    return {
        "path": str(source_coverage_diagnostic_path) if source_coverage_diagnostic_path is not None else None,
        "schema_version": payload.get("schema_version", MARKET_SOURCE_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION),
        "artifact_kind": payload.get("artifact_kind", MARKET_SOURCE_COVERAGE_DIAGNOSTIC_ARTIFACT_KIND),
        "pathway": payload.get("pathway", "market_state"),
        "run_id": payload.get("run_id", run_id),
        "universe": payload.get("universe", universe),
        "band": payload.get("band", band),
        "membership_source": (
            payload.get("lifecycle_policy", {}).get("membership_source")
            if isinstance(payload.get("lifecycle_policy"), Mapping)
            else membership_source
        ),
        "source_coverage_status": payload.get("source_coverage_status", source_coverage_status),
        "status": payload.get("status", PATHWAY_MANIFEST_STATUS),
        "lifecycle_policy": dict(payload.get("lifecycle_policy", {}))
        if isinstance(payload.get("lifecycle_policy"), Mapping)
        else {},
        "artifact_boundary": dict(payload.get("artifact_boundary", {}))
        if isinstance(payload.get("artifact_boundary"), Mapping)
        else {},
    }


def _source_read_precondition_reference(
    source_read_precondition: Mapping[str, Any] | None,
    *,
    source_read_precondition_path: Path | str | None,
    universe: str,
    band: str,
    run_id: str,
    membership_source: str,
    source_coverage_status: str,
) -> dict[str, Any] | None:
    if source_read_precondition is None and source_read_precondition_path is None:
        return None
    payload = dict(source_read_precondition or {})
    return {
        "path": str(source_read_precondition_path) if source_read_precondition_path is not None else None,
        "schema_version": payload.get("schema_version", MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_SCHEMA_VERSION),
        "artifact_kind": payload.get("artifact_kind", MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_ARTIFACT_KIND),
        "pathway": payload.get("pathway", "market_state"),
        "run_id": payload.get("run_id", run_id),
        "universe": payload.get("universe", universe),
        "band": payload.get("band", band),
        "membership_source": payload.get("membership_source", membership_source),
        "source_coverage_status": payload.get("source_coverage_status", source_coverage_status),
        "status": payload.get("status", PATHWAY_MANIFEST_STATUS),
        "readiness_flags": dict(payload.get("readiness_flags", {}))
        if isinstance(payload.get("readiness_flags"), Mapping)
        else {},
        "artifact_boundary": dict(payload.get("artifact_boundary", {}))
        if isinstance(payload.get("artifact_boundary"), Mapping)
        else {},
    }


def market_source_availability_lineage(
    *,
    run_id: str,
    universe: str,
    band: str,
    membership_source: str,
    source_coverage_status: str,
    membership_snapshot_reference: Mapping[str, Any] | None = None,
    source_probe_reference: Mapping[str, Any] | None = None,
    source_coverage_diagnostic: Mapping[str, Any] | None = None,
    source_coverage_diagnostic_path: Path | str | None = None,
    source_read_precondition: Mapping[str, Any] | None = None,
    source_read_precondition_path: Path | str | None = None,
    readiness_boundary: Mapping[str, Any] | None = None,
    status: str = PATHWAY_MANIFEST_STATUS,
    schema_version: int = MARKET_SOURCE_AVAILABILITY_LINEAGE_SCHEMA_VERSION,
) -> MarketSourceAvailabilityLineage:
    if str(status) != PATHWAY_MANIFEST_STATUS:
        raise ValueError(f"Market source availability lineage status must be {PATHWAY_MANIFEST_STATUS!r}")
    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        raise ValueError(f"Unsupported Regime band {band!r}; expected one of: {valid}")
    coverage_status = str(source_coverage_status).strip()
    if coverage_status not in {"probed", "not_probed"}:
        raise ValueError("source_coverage_status must be 'probed' or 'not_probed'")
    source = str(membership_source).strip()
    if not source:
        raise ValueError("membership_source must be non-empty")
    band_contract = REGIME_BANDS[str(band)]
    references = {
        "membership_snapshot": dict(membership_snapshot_reference or {}),
        "source_probe": dict(source_probe_reference) if source_probe_reference is not None else None,
        "source_coverage_diagnostic": _coverage_diagnostic_reference(
            source_coverage_diagnostic,
            source_coverage_diagnostic_path=source_coverage_diagnostic_path,
            universe=str(universe).strip(),
            band=str(band),
            run_id=str(run_id),
            membership_source=source,
            source_coverage_status=coverage_status,
        ),
        "source_read_precondition": _source_read_precondition_reference(
            source_read_precondition,
            source_read_precondition_path=source_read_precondition_path,
            universe=str(universe).strip(),
            band=str(band),
            run_id=str(run_id),
            membership_source=source,
            source_coverage_status=coverage_status,
        ),
    }
    boundary = _default_source_availability_readiness_boundary()
    boundary.update(dict(readiness_boundary or {}))
    if references["source_read_precondition"] is not None:
        boundary["source_read_precondition_only"] = True

    for field_name in (
        "source_availability_metadata_only",
    ):
        if boundary.get(field_name) is not True:
            raise ValueError(f"Market source availability lineage must mark {field_name} true")
    for field_name in (
        "scalar_partition_coverage_is_aggregate_feature_readiness",
        "aggregation_readiness",
        "aggregate_frame_construction_allowed",
        "aggregation_frame_built",
        "aggregation_values_built",
        "production_writes_enabled",
        "parquet_writes_enabled",
        "definition_writes_enabled",
        "registry_lookup_allowed",
        "membership_inferred_from_disk",
        "downstream_reader_ready",
    ):
        if boundary.get(field_name) is not False:
            raise ValueError(f"Market source availability lineage cannot claim {field_name}")
    if boundary.get("durable_registry_reference") is not None:
        raise ValueError("Market source availability lineage cannot claim a durable registry reference")
    for name, reference in references.items():
        if isinstance(reference, Mapping):
            _validate_source_availability_reference(reference, context=str(name))
    return MarketSourceAvailabilityLineage(
        schema_version=int(schema_version),
        artifact_kind=MARKET_SOURCE_AVAILABILITY_LINEAGE_ARTIFACT_KIND,
        status=str(status),
        pathway="market_state",
        run_id=str(run_id),
        universe=str(universe).strip(),
        band=str(band),
        ceiling_interval_min=int(band_contract.ceiling_interval_min),
        membership_source=source,
        source_coverage_status=coverage_status,
        artifact_references=references,
        readiness_boundary=boundary,
    )


def _market_source_coverage_from_summary(
    source_summary: Mapping[str, Any] | None,
    *,
    member_assets: Sequence[str],
    required_intervals: Sequence[int],
    min_assets: int,
) -> tuple[str, dict[str, Any]]:
    members = tuple(str(asset).strip() for asset in member_assets if str(asset).strip())
    required = tuple(int(interval) for interval in required_intervals)
    if source_summary is None:
        return "not_probed", {
            "source_root": None,
            "source_root_exists": None,
            "required_intervals": list(required),
            "member_assets": list(members),
            "member_assets_count": int(len(members)),
            "assets_with_all_intervals": 0,
            "missing_assets": [],
            "missing_intervals_by_asset": {},
            "month_partitions": 0,
            "parquet_files": 0,
            "row_count_estimate": None,
            "row_count_estimate_complete": None,
            "min_assets_met": False,
        }
    summary = dict(source_summary)
    assets_with_all_intervals = int(summary.get("assets_with_all_intervals", 0) or 0)
    return "probed", {
        "source_root": summary.get("source_root"),
        "source_root_exists": bool(summary.get("source_root_exists")),
        "required_intervals": [int(interval) for interval in summary.get("required_intervals", required)],
        "member_assets": list(summary.get("member_assets", list(members))),
        "member_assets_count": int(summary.get("member_assets_count", len(members)) or 0),
        "assets_with_all_intervals": int(assets_with_all_intervals),
        "missing_assets": list(summary.get("missing_assets", [])),
        "missing_intervals_by_asset": dict(summary.get("missing_intervals_by_asset", {})),
        "month_partitions": int(summary.get("month_partitions", 0) or 0),
        "parquet_files": int(summary.get("parquet_files", 0) or 0),
        "row_count_estimate": summary.get("row_count_estimate"),
        "row_count_estimate_complete": bool(summary.get("row_count_estimate_complete", False)),
        "min_assets_met": bool(assets_with_all_intervals >= int(min_assets)),
    }


def _validate_market_source_coverage_inputs(
    *,
    universe: str,
    band: str,
    member_assets: Sequence[str],
    membership_source: str,
    membership_snapshot: Mapping[str, Any] | None,
    source_summary: Mapping[str, Any] | None,
    source_coverage_status: str,
    source_coverage: Mapping[str, Any],
    required_intervals: Sequence[int],
    min_assets: int,
    status: str,
    artifact_boundary: Mapping[str, Any],
) -> MarketUniverseMembershipValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    universe_text = str(universe).strip()
    if not universe_text:
        errors.append("universe must be non-empty")
    else:
        try:
            _safe_snapshot_path_part(universe_text, field_name="coverage universe")
        except ValueError as exc:
            errors.append(str(exc))

    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        errors.append(f"Unsupported Regime band {band!r}; expected one of: {valid}")

    raw_members = tuple(str(asset).strip() for asset in member_assets)
    nonblank_members = tuple(asset for asset in raw_members if asset)
    sorted_members = _sorted_unique_members(nonblank_members)
    if not nonblank_members:
        errors.append("member_assets must be explicit and non-empty")
    if len(set(nonblank_members)) != len(nonblank_members):
        errors.append("member_assets must be unique")
    if len(nonblank_members) != len(raw_members):
        errors.append("member_assets must not include blank assets")

    membership_source_text = str(membership_source).strip()
    if not membership_source_text:
        errors.append("membership_source must be non-empty")

    try:
        min_assets_int = int(min_assets)
    except (TypeError, ValueError):
        min_assets_int = 0
        errors.append("min_assets must be an integer")
    if min_assets_int < 1:
        errors.append("min_assets must be positive")
    elif sorted_members and min_assets_int > len(sorted_members):
        errors.append("min_assets cannot exceed explicit member_assets count")

    if str(status) != PATHWAY_MANIFEST_STATUS:
        errors.append(f"status must be {PATHWAY_MANIFEST_STATUS!r}")
    if artifact_boundary.get("membership_inferred_from_disk") is not False:
        errors.append("market source coverage diagnostic cannot infer membership from disk")
    if artifact_boundary.get("aggregation_frame_built") is not False:
        errors.append("market source coverage diagnostic cannot claim aggregation-frame construction")
    if artifact_boundary.get("aggregation_values_built") is not False:
        errors.append("market source coverage diagnostic cannot claim aggregation values")

    snapshot = dict(membership_snapshot or {})
    if snapshot:
        snapshot_boundary = dict(snapshot.get("artifact_boundary", {}))
        if snapshot_boundary.get("membership_inferred_from_disk") is not False:
            errors.append("membership snapshot reference cannot infer membership from disk")
        if snapshot.get("artifact_kind") not in (None, MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_ARTIFACT_KIND):
            errors.append("membership snapshot artifact_kind is not market_universe_membership_snapshot")
        if str(snapshot.get("status", PATHWAY_MANIFEST_STATUS)) != PATHWAY_MANIFEST_STATUS:
            errors.append("membership snapshot status must be scaffold_only")
        if str(snapshot.get("universe", universe_text)) != universe_text:
            errors.append("membership snapshot universe does not match coverage universe")
        snapshot_members = _sorted_unique_members(tuple(snapshot.get("member_assets", ())))
        if snapshot_members and snapshot_members != sorted_members:
            errors.append("membership snapshot member_assets do not match coverage member_assets")
        if int(snapshot.get("min_assets", min_assets_int) or 0) != min_assets_int:
            errors.append("membership snapshot min_assets does not match coverage min_assets")

    summary = dict(source_summary or {})
    if summary:
        if summary.get("probe_kind") != SCALAR_FEATURE_SOURCE_PROBE_KIND:
            errors.append("source_probe must describe scalar_feature_partitions")
        if bool(summary.get("membership_inferred_from_disk")):
            errors.append("source_probe cannot infer membership from disk")
        if bool(summary.get("aggregation_frame_built")):
            errors.append("source_probe cannot claim aggregation-frame construction")
        if bool(summary.get("aggregation_values_built")):
            errors.append("source_probe cannot claim aggregation values")
        summary_members = _sorted_unique_members(tuple(summary.get("member_assets", ())))
        if summary_members and summary_members != sorted_members:
            errors.append("source_probe member_assets do not match coverage member_assets")
        summary_required = tuple(int(interval) for interval in summary.get("required_intervals", ()))
        expected_required = tuple(int(interval) for interval in required_intervals)
        if summary_required and summary_required != expected_required:
            errors.append("source_probe required_intervals do not match coverage required_intervals")
        if not bool(summary.get("source_root_exists")):
            errors.append("scalar feature source root does not exist")
        if summary.get("missing_assets"):
            errors.append("scalar feature partitions are missing for at least one explicit member asset")
        if int(summary.get("assets_with_all_intervals", 0) or 0) < min_assets_int:
            errors.append("scalar feature source partition coverage is below min_assets")
        if not bool(summary.get("row_count_estimate_complete", False)):
            warnings.append("scalar feature row-count estimate is incomplete")
    elif source_coverage_status == "not_probed":
        warnings.append("scalar feature source coverage was not probed")

    coverage_required = tuple(int(interval) for interval in source_coverage.get("required_intervals", ()))
    if coverage_required and coverage_required != tuple(int(interval) for interval in required_intervals):
        errors.append("source_coverage required_intervals do not match coverage required_intervals")

    return MarketUniverseMembershipValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def market_source_coverage_diagnostic_record(
    *,
    run_id: str,
    universe: str,
    band: str,
    member_assets: Sequence[str],
    min_assets: int,
    membership_source: str,
    membership_snapshot: Mapping[str, Any] | None = None,
    membership_snapshot_path: Path | str | None = None,
    source_probe: Mapping[str, Any] | None = None,
    source_probe_path: Path | str | None = None,
    created_at_utc: str | None = None,
    status: str = PATHWAY_MANIFEST_STATUS,
    schema_version: int = MARKET_SOURCE_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
    artifact_boundary: Mapping[str, Any] | None = None,
) -> MarketSourceCoverageDiagnosticRecord:
    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        raise ValueError(f"Unsupported Regime band {band!r}; expected one of: {valid}")
    members = _sorted_unique_members(member_assets)
    band_contract = REGIME_BANDS[str(band)]
    required_intervals = tuple(int(interval) for interval in band_contract.member_intervals)
    source_summary = _extract_source_summary(source_probe)
    source_coverage_status, source_coverage = _market_source_coverage_from_summary(
        source_summary,
        member_assets=members,
        required_intervals=required_intervals,
        min_assets=int(min_assets),
    )
    boundary = _validate_scaffold_artifact_boundary(
        artifact_boundary or _default_market_source_coverage_artifact_boundary(),
        context="Market source coverage diagnostic",
    )
    for field_name in ("membership_inferred_from_disk", "aggregation_frame_built", "aggregation_values_built"):
        if boundary.get(field_name) is not False:
            raise ValueError(f"Market source coverage diagnostic cannot claim {field_name}")
    validation = _validate_market_source_coverage_inputs(
        universe=universe,
        band=str(band),
        member_assets=members,
        membership_source=membership_source,
        membership_snapshot=membership_snapshot,
        source_summary=source_summary,
        source_coverage_status=source_coverage_status,
        source_coverage=source_coverage,
        required_intervals=required_intervals,
        min_assets=int(min_assets),
        status=status,
        artifact_boundary=boundary,
    )
    if not validation.ok:
        raise ValueError("Market source coverage diagnostic invalid: " + "; ".join(validation.errors))
    created = _utc_timestamp_text(created_at_utc)
    lifecycle_policy = market_universe_lifecycle_policy(
        membership_source=str(membership_source).strip(),
    ).as_dict()
    membership_snapshot_reference = _snapshot_reference(
        membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        universe=str(universe).strip(),
        member_assets=members,
        membership_source=str(membership_source).strip(),
        min_assets=int(min_assets),
    )
    source_probe_reference = _source_probe_reference(
        source_probe,
        source_summary,
        source_probe_path=source_probe_path,
    )
    source_availability_lineage = market_source_availability_lineage(
        run_id=str(run_id),
        universe=str(universe).strip(),
        band=str(band),
        membership_source=str(membership_source).strip(),
        source_coverage_status=source_coverage_status,
        membership_snapshot_reference=membership_snapshot_reference,
        source_probe_reference=source_probe_reference,
    ).as_dict()
    return MarketSourceCoverageDiagnosticRecord(
        schema_version=int(schema_version),
        artifact_kind=MARKET_SOURCE_COVERAGE_DIAGNOSTIC_ARTIFACT_KIND,
        status=str(status),
        pathway="market_state",
        run_id=str(run_id),
        created_at_utc=created,
        lifecycle_policy=lifecycle_policy,
        universe=str(universe).strip(),
        band=str(band),
        ceiling_interval_min=int(band_contract.ceiling_interval_min),
        required_intervals=required_intervals,
        member_assets=members,
        min_assets=int(min_assets),
        source_coverage_status=source_coverage_status,
        membership_snapshot_reference=membership_snapshot_reference,
        source_probe_reference=source_probe_reference,
        source_availability_lineage=source_availability_lineage,
        source_coverage=source_coverage,
        validation_result=validation,
        artifact_boundary=boundary,
    )


def _default_market_aggregation_source_read_precondition_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "market_aggregation_source_read_precondition_json_only",
        "source_read_precondition_only": True,
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "aggregation_readiness": False,
        "aggregation_frame_built": False,
        "aggregation_values_built": False,
        "membership_inferred_from_disk": False,
    }


def _default_market_aggregation_source_readiness_flags() -> dict[str, Any]:
    return {
        "source_read_preconditions_met": True,
        "scalar_source_metadata_available": True,
        "aggregation_readiness": False,
        "aggregate_frame_construction_allowed": False,
        "aggregate_values_available": False,
        "parquet_writer_ready": False,
        "definition_writer_ready": False,
        "production_output_ready": False,
        "membership_registry_ready": False,
    }


def _market_aggregation_source_root_policy(
    source_root: Path | str | None,
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    raw = str(source_root or "").strip()
    root = _resolved(Path(raw)) if raw else None
    project = _resolved(project_root or Path.cwd())
    exists = bool(root is not None and root.exists())
    is_dir = bool(root is not None and root.is_dir())
    unsafe_parts = {"regime_definitions", "model_states"}
    unsafe = False
    if root is not None:
        lower_parts = tuple(str(part).lower() for part in root.parts)
        unsafe = any(part in unsafe_parts for part in lower_parts) or any(
            part.startswith(("regimes_", "market_regimes_", "relative_regimes_")) for part in lower_parts
        )
        unsafe = unsafe or _under_any(root, (project / "regime_definitions", project / "model_states"))
    approved = bool(root is not None and exists and is_dir and not unsafe)
    if not raw:
        classification = "missing_source_root"
        reason = "Scalar feature source root is required for source-read preconditions."
    elif not exists:
        classification = "missing_source_root"
        reason = "Scalar feature source root does not exist."
    elif not is_dir:
        classification = "not_directory"
        reason = "Scalar feature source root must be a directory."
    elif unsafe:
        classification = PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION
        reason = "Scalar feature source root points at an output or production-adjacent artifact root."
    else:
        classification = "read_only_source_root"
        reason = "Scalar feature source root is approved for read-only precondition metadata."
    production_root_hits = []
    if root is not None:
        production_root_hits = [
            str(_resolved(prod_root))
            for prod_root in default_production_roots(env)
            if root == _resolved(prod_root) or is_relative_to(root, _resolved(prod_root))
        ]
    return {
        "source_root": str(root) if root is not None else None,
        "classification": classification,
        "source_root_exists": exists,
        "source_root_is_directory": is_dir,
        "read_only": True,
        "writes_allowed": False,
        "approved_for_source_read_precondition": approved,
        "production_root_overlap": production_root_hits,
        "reason": reason,
    }


def _validate_market_aggregation_source_read_precondition_inputs(
    *,
    universe: str,
    band: str,
    member_assets: Sequence[str],
    membership_source: str,
    source_summary: Mapping[str, Any] | None,
    source_coverage_status: str,
    source_coverage: Mapping[str, Any],
    source_root_policy: Mapping[str, Any],
    required_intervals: Sequence[int],
    min_assets: int,
    status: str,
    artifact_boundary: Mapping[str, Any],
    readiness_flags: Mapping[str, Any],
    failure_policy: str,
) -> MarketUniverseMembershipValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    universe_text = str(universe).strip()
    if not universe_text:
        errors.append("universe must be non-empty")
    else:
        try:
            _safe_snapshot_path_part(universe_text, field_name="source-read precondition universe")
        except ValueError as exc:
            errors.append(str(exc))

    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        errors.append(f"Unsupported Regime band {band!r}; expected one of: {valid}")

    raw_members = tuple(str(asset).strip() for asset in member_assets)
    nonblank_members = tuple(asset for asset in raw_members if asset)
    sorted_members = _sorted_unique_members(nonblank_members)
    if not nonblank_members:
        errors.append("member_assets must be explicit and non-empty")
    if len(set(nonblank_members)) != len(nonblank_members):
        errors.append("member_assets must be unique")
    if len(nonblank_members) != len(raw_members):
        errors.append("member_assets must not include blank assets")

    membership_source_text = str(membership_source).strip()
    if not membership_source_text:
        errors.append("membership_source must be non-empty")
    else:
        try:
            market_membership_source_policy(
                membership_source=membership_source_text,
                member_assets=sorted_members,
                source_feature_root=source_root_policy.get("source_root"),
            )
        except ValueError as exc:
            errors.append(str(exc))

    try:
        min_assets_int = int(min_assets)
    except (TypeError, ValueError):
        min_assets_int = 0
        errors.append("min_assets must be an integer")
    if min_assets_int < 1:
        errors.append("min_assets must be positive")
    elif sorted_members and min_assets_int > len(sorted_members):
        errors.append("min_assets cannot exceed explicit member_assets count")

    if str(status) != PATHWAY_MANIFEST_STATUS:
        errors.append(f"status must be {PATHWAY_MANIFEST_STATUS!r}")
    if not str(failure_policy or "").strip():
        errors.append("failure_policy must be non-empty")

    if artifact_boundary.get("source_read_precondition_only") is not True:
        errors.append("market aggregation source-read precondition must be precondition-only")
    for field_name in (
        "production_writes_enabled",
        "parquet_writes_enabled",
        "definition_writes_enabled",
        "aggregation_readiness",
        "aggregation_frame_built",
        "aggregation_values_built",
        "membership_inferred_from_disk",
    ):
        if artifact_boundary.get(field_name) is not False:
            errors.append(f"market aggregation source-read precondition cannot claim {field_name}")

    if readiness_flags.get("source_read_preconditions_met") is not True:
        errors.append("readiness_flags must mark source_read_preconditions_met true only after validation")
    for field_name in (
        "aggregation_readiness",
        "aggregate_frame_construction_allowed",
        "aggregate_values_available",
        "parquet_writer_ready",
        "definition_writer_ready",
        "production_output_ready",
        "membership_registry_ready",
    ):
        if readiness_flags.get(field_name) is not False:
            errors.append(f"readiness_flags cannot claim {field_name}")

    if source_root_policy.get("read_only") is not True or source_root_policy.get("writes_allowed") is not False:
        errors.append("source root policy must be read-only with writes disabled")
    if source_root_policy.get("approved_for_source_read_precondition") is not True:
        errors.append("source root is not approved for source-read preconditions")

    summary = dict(source_summary or {})
    if not summary:
        errors.append("source_probe is required for market aggregation source-read preconditions")
    else:
        if summary.get("probe_kind") != SCALAR_FEATURE_SOURCE_PROBE_KIND:
            errors.append("source_probe must describe scalar_feature_partitions")
        if bool(summary.get("membership_inferred_from_disk")):
            errors.append("source_probe cannot infer membership from disk")
        if bool(summary.get("aggregation_frame_built")):
            errors.append("source_probe cannot claim aggregation-frame construction")
        if bool(summary.get("aggregation_values_built")):
            errors.append("source_probe cannot claim aggregation values")
        summary_source = str(summary.get("membership_source") or summary.get("member_assets_source") or "").strip()
        if summary_source and summary_source != membership_source_text:
            errors.append("source_probe membership_source does not match precondition membership_source")
        summary_members = _sorted_unique_members(tuple(summary.get("member_assets", ())))
        if summary_members and summary_members != sorted_members:
            errors.append("source_probe member_assets do not match precondition member_assets")
        summary_required = tuple(int(interval) for interval in summary.get("required_intervals", ()))
        expected_required = tuple(int(interval) for interval in required_intervals)
        if not summary_required:
            errors.append("source_probe required_intervals must be explicit")
        elif summary_required != expected_required:
            errors.append("source_probe required_intervals do not match precondition required_intervals")
        if not bool(summary.get("source_root_exists")):
            errors.append("scalar feature source root does not exist")
        if summary.get("missing_assets"):
            errors.append("scalar feature partitions are missing for at least one explicit member asset")
        if int(summary.get("assets_with_all_intervals", 0) or 0) < min_assets_int:
            errors.append("scalar feature source partition coverage is below min_assets")
        if not bool(summary.get("row_count_estimate_complete", False)):
            warnings.append("scalar feature row-count estimate is incomplete")

    if str(source_coverage_status) != "probed":
        errors.append("source_coverage_status must be 'probed' for source-read preconditions")
    coverage_required = tuple(int(interval) for interval in source_coverage.get("required_intervals", ()))
    if coverage_required and coverage_required != tuple(int(interval) for interval in required_intervals):
        errors.append("source_coverage required_intervals do not match precondition required_intervals")
    if source_coverage.get("min_assets_met") is not True:
        errors.append("source_coverage must meet min_assets")

    return MarketUniverseMembershipValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def market_aggregation_source_read_precondition(
    *,
    run_id: str,
    universe: str,
    band: str,
    member_assets: Sequence[str],
    min_assets: int,
    membership_source: str,
    source_probe: Mapping[str, Any],
    source_probe_path: Path | str | None = None,
    membership_snapshot: Mapping[str, Any] | None = None,
    membership_snapshot_path: Path | str | None = None,
    source_coverage_diagnostic: Mapping[str, Any] | None = None,
    source_coverage_diagnostic_path: Path | str | None = None,
    created_at_utc: str | None = None,
    status: str = PATHWAY_MANIFEST_STATUS,
    failure_policy: str = MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_FAILURE_POLICY,
    artifact_boundary: Mapping[str, Any] | None = None,
    readiness_flags: Mapping[str, Any] | None = None,
    schema_version: int = MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_SCHEMA_VERSION,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> MarketAggregationSourceReadPrecondition:
    if str(band) not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        raise ValueError(f"Unsupported Regime band {band!r}; expected one of: {valid}")
    raw_members = tuple(str(asset).strip() for asset in member_assets)
    if len(tuple(asset for asset in raw_members if asset)) != len(raw_members):
        raise ValueError("Market aggregation source-read precondition member_assets must not include blank assets")
    if len(set(raw_members)) != len(raw_members):
        raise ValueError("Market aggregation source-read precondition member_assets must be unique")
    members = _sorted_unique_members(raw_members)
    band_contract = REGIME_BANDS[str(band)]
    required_intervals = tuple(int(interval) for interval in band_contract.member_intervals)
    source_summary = _extract_source_summary(source_probe)
    source_coverage_status, source_coverage = _market_source_coverage_from_summary(
        source_summary,
        member_assets=members,
        required_intervals=required_intervals,
        min_assets=int(min_assets),
    )
    source_root_policy = _market_aggregation_source_root_policy(
        (source_summary or {}).get("source_root") if source_summary is not None else None,
        project_root=project_root,
        env=env,
    )
    boundary = _validate_scaffold_artifact_boundary(
        artifact_boundary or _default_market_aggregation_source_read_precondition_artifact_boundary(),
        context="Market aggregation source-read precondition",
    )
    flags = dict(readiness_flags or _default_market_aggregation_source_readiness_flags())
    validation = _validate_market_aggregation_source_read_precondition_inputs(
        universe=universe,
        band=str(band),
        member_assets=members,
        membership_source=membership_source,
        source_summary=source_summary,
        source_coverage_status=source_coverage_status,
        source_coverage=source_coverage,
        source_root_policy=source_root_policy,
        required_intervals=required_intervals,
        min_assets=int(min_assets),
        status=str(status),
        artifact_boundary=boundary,
        readiness_flags=flags,
        failure_policy=failure_policy,
    )
    if not validation.ok:
        raise ValueError("Market aggregation source-read precondition invalid: " + "; ".join(validation.errors))
    created = _utc_timestamp_text(created_at_utc)
    source = str(membership_source).strip()
    lifecycle_policy = market_universe_lifecycle_policy(membership_source=source).as_dict()
    source_probe_reference = _source_probe_reference(
        source_probe,
        source_summary,
        source_probe_path=source_probe_path,
    )
    if source_probe_reference is None:
        raise ValueError("Market aggregation source-read precondition requires a source_probe reference")
    membership_snapshot_reference = _snapshot_reference(
        membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        universe=str(universe).strip(),
        member_assets=members,
        membership_source=source,
        min_assets=int(min_assets),
    )
    source_availability_lineage = market_source_availability_lineage(
        run_id=str(run_id),
        universe=str(universe).strip(),
        band=str(band),
        membership_source=source,
        source_coverage_status=source_coverage_status,
        membership_snapshot_reference=membership_snapshot_reference,
        source_probe_reference=source_probe_reference,
        source_coverage_diagnostic=source_coverage_diagnostic,
        source_coverage_diagnostic_path=source_coverage_diagnostic_path,
        readiness_boundary={"source_read_precondition_only": True},
    ).as_dict()
    return MarketAggregationSourceReadPrecondition(
        schema_version=int(schema_version),
        artifact_kind=MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_ARTIFACT_KIND,
        status=str(status),
        pathway="market_state",
        run_id=str(run_id),
        created_at_utc=created,
        lifecycle_policy=lifecycle_policy,
        universe=str(universe).strip(),
        band=str(band),
        ceiling_interval_min=int(band_contract.ceiling_interval_min),
        required_intervals=required_intervals,
        member_assets=members,
        min_assets=int(min_assets),
        membership_source=source,
        membership_source_policy=market_membership_source_policy(
            membership_source=source,
            member_assets=members,
            source_feature_root=source_root_policy.get("source_root"),
        ).as_dict(),
        source_root_policy=source_root_policy,
        source_coverage_status=source_coverage_status,
        source_coverage=source_coverage,
        source_probe_reference=source_probe_reference,
        membership_snapshot_reference=membership_snapshot_reference,
        source_availability_lineage=source_availability_lineage,
        failure_policy=str(failure_policy).strip(),
        readiness_flags=flags,
        validation_result=validation,
        artifact_boundary=boundary,
    )


def write_market_aggregation_source_read_precondition(
    diagnostics_root: Path,
    *,
    run_id: str,
    universe: str,
    band: str,
    member_assets: Sequence[str],
    min_assets: int,
    membership_source: str,
    source_probe: Mapping[str, Any],
    source_probe_path: Path | str | None = None,
    membership_snapshot: Mapping[str, Any] | None = None,
    membership_snapshot_path: Path | str | None = None,
    source_coverage_diagnostic: Mapping[str, Any] | None = None,
    source_coverage_diagnostic_path: Path | str | None = None,
    created_at_utc: str | None = None,
    artifact_boundary: Mapping[str, Any] | None = None,
    readiness_flags: Mapping[str, Any] | None = None,
    write_kind: str = "Regime market aggregation source-read precondition",
) -> Path:
    require_pathway_diagnostics_root(Path(diagnostics_root), for_source_probe=True)
    path = market_aggregation_source_read_precondition_path(diagnostics_root, universe=universe, run_id=run_id)
    record = market_aggregation_source_read_precondition(
        run_id=run_id,
        universe=universe,
        band=band,
        member_assets=member_assets,
        min_assets=int(min_assets),
        membership_source=membership_source,
        source_probe=source_probe,
        source_probe_path=source_probe_path,
        membership_snapshot=membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        source_coverage_diagnostic=source_coverage_diagnostic,
        source_coverage_diagnostic_path=source_coverage_diagnostic_path,
        created_at_utc=created_at_utc,
        artifact_boundary=artifact_boundary,
        readiness_flags=readiness_flags,
    )
    write_json(path, record.as_dict(), write_kind=write_kind)
    return path


def artifact_metadata_schema(
    contract: PathwayContractLike,
    *,
    schema_version: int,
    feature_manifest_groups: Sequence[str],
) -> PathwayArtifactMetadataSchema:
    return PathwayArtifactMetadataSchema(
        pathway=str(contract.name),
        schema_version=int(schema_version),
        key_columns=tuple(contract.key_columns),
        partition_columns=tuple(contract.partition_columns),
        required_output_columns=tuple(contract.required_output_columns),
        feature_manifest_groups=tuple(str(group) for group in feature_manifest_groups),
    )


def diagnostic_report_schema(contract: PathwayContractLike) -> PathwayDiagnosticReportSchema:
    return PathwayDiagnosticReportSchema(
        pathway=str(contract.name),
        trial_key_columns=tuple(contract.key_columns),
        summary_fields=(
            "rows_evaluated",
            "rows_emitted",
            "unknown_fraction",
            "feature_schema_hash",
            "candidate_method",
            "fallback_reason",
        ),
        feature_group_fields=(
            "candidate_columns",
            "columns_present",
            "columns_missing",
            "coverage_pct",
            "stability_score",
        ),
        output_fields=(
            "artifact_path",
            "definition_path",
            "diagnostic_path",
            "created_at_utc",
        ),
    )


def band_metadata(bands: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for band_name in bands:
        band: RegimeBandContract = REGIME_BANDS[str(band_name)]
        out.append(
            {
                "name": band.name,
                "ceiling_interval_min": int(band.ceiling_interval_min),
                "member_intervals": list(band.member_intervals),
                "train_days": int(band.train_days),
            }
        )
    return out


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _metadata_for_parquet(path: Path, *, timestamp_column: str) -> tuple[int | None, tuple[str, ...], Any | None, Any | None, str | None]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        rows = int(metadata.num_rows) if metadata is not None else None
        columns = tuple(str(name) for name in parquet_file.schema_arrow.names)
        first_ts = None
        last_ts = None
        if str(timestamp_column) in columns and metadata is not None:
            column_index = columns.index(str(timestamp_column))
            for row_group_index in range(metadata.num_row_groups):
                row_group = metadata.row_group(row_group_index)
                stats = row_group.column(column_index).statistics
                if stats is None or not getattr(stats, "has_min_max", False):
                    continue
                cur_min = _json_safe_scalar(stats.min)
                cur_max = _json_safe_scalar(stats.max)
                first_ts = cur_min if first_ts is None or cur_min < first_ts else first_ts
                last_ts = cur_max if last_ts is None or cur_max > last_ts else last_ts
        return rows, columns, first_ts, last_ts, None
    except Exception as exc:
        return None, (), None, None, f"{type(exc).__name__}: {exc}"


def _summarize_scalar_feature_interval(
    source_root: Path,
    *,
    asset: str,
    interval_minutes: int,
    timestamp_column: str,
) -> ScalarFeatureIntervalPartitionSummary:
    asset_dir = Path(source_root) / f"scalar_features_{int(interval_minutes)}" / f"asset={str(asset)}"
    month_partitions = 0
    parquet_files = 0
    row_count_estimate = 0
    row_count_estimate_complete = True
    schema_columns: list[str] = []
    first_ts = None
    last_ts = None
    read_errors: list[str] = []

    if asset_dir.exists():
        for year_dir in sorted(asset_dir.glob("year=*"), key=lambda path: str(path).lower()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.glob("month=*"), key=lambda path: str(path).lower()):
                if not month_dir.is_dir():
                    continue
                files = sorted(month_dir.glob("*.parquet"), key=lambda path: str(path).lower())
                if files:
                    month_partitions += 1
                    parquet_files += len(files)
                for part in files:
                    rows, columns, part_first_ts, part_last_ts, error = _metadata_for_parquet(
                        part,
                        timestamp_column=timestamp_column,
                    )
                    if error is not None:
                        row_count_estimate_complete = False
                        read_errors.append(f"{part.name}: {error}")
                    elif rows is None:
                        row_count_estimate_complete = False
                    else:
                        row_count_estimate += int(rows)
                    if columns and not schema_columns:
                        schema_columns = list(columns)
                    if part_first_ts is not None:
                        first_ts = part_first_ts if first_ts is None or part_first_ts < first_ts else first_ts
                    if part_last_ts is not None:
                        last_ts = part_last_ts if last_ts is None or part_last_ts > last_ts else last_ts

    return ScalarFeatureIntervalPartitionSummary(
        asset=str(asset),
        interval_minutes=int(interval_minutes),
        asset_partition=str(asset_dir),
        asset_partition_exists=bool(asset_dir.exists()),
        month_partitions=int(month_partitions),
        parquet_files=int(parquet_files),
        row_count_estimate=int(row_count_estimate) if row_count_estimate_complete else None,
        row_count_estimate_complete=bool(row_count_estimate_complete),
        schema_available=bool(schema_columns),
        schema_columns_sample=tuple(schema_columns[:50]),
        first_ts=_json_safe_scalar(first_ts),
        last_ts=_json_safe_scalar(last_ts),
        read_errors=tuple(read_errors),
    )


def summarize_scalar_feature_source_partitions(
    *,
    source_root: Path,
    band: str,
    member_assets: Sequence[str],
    timestamp_column: str = "ts",
    membership_source: str = "config.member_assets",
) -> ScalarFeatureSourcePartitionSummary:
    band_name = str(band).strip()
    if band_name not in REGIME_BANDS:
        valid = ", ".join(REGIME_BANDS)
        raise ValueError(f"Unsupported Regime band {band!r}; expected one of: {valid}")
    members = tuple(str(asset).strip() for asset in member_assets if str(asset).strip())
    if not members:
        raise ValueError("Scalar feature source partition probe requires explicit member_assets")
    timestamp = str(timestamp_column).strip()
    if not timestamp:
        raise ValueError("Scalar feature source partition probe timestamp_column must be non-empty")
    source = Path(source_root).expanduser()
    band_contract = REGIME_BANDS[band_name]
    required_intervals = tuple(int(interval) for interval in band_contract.member_intervals)
    assets = tuple(
        ScalarFeatureAssetPartitionSummary(
            asset=str(asset),
            intervals=tuple(
                _summarize_scalar_feature_interval(
                    source,
                    asset=str(asset),
                    interval_minutes=int(interval),
                    timestamp_column=timestamp,
                )
                for interval in required_intervals
            ),
        )
        for asset in members
    )
    return ScalarFeatureSourcePartitionSummary(
        source_root=str(source),
        source_root_exists=bool(source.exists()),
        band=band_name,
        ceiling_interval_min=int(band_contract.ceiling_interval_min),
        required_intervals=required_intervals,
        member_assets=members,
        timestamp_column=timestamp,
        assets=assets,
        membership_source=str(membership_source).strip() or "config.member_assets",
    )


def dry_run_manifest(
    *,
    run_id: str,
    config: Mapping[str, Any],
    contract: PathwayArtifactMetadataSchema,
    feature_manifest: Mapping[str, Any],
    diagnostics: PathwayDiagnosticReportSchema,
    bands: Sequence[str],
    status: str = PATHWAY_MANIFEST_STATUS,
) -> dict[str, Any]:
    schema_version = int(config.get("schema_version", contract.schema_version))
    return {
        "run_id": str(run_id),
        "status": str(status),
        "schema_version": schema_version,
        "config": dict(config),
        "contract": contract.as_dict(),
        "feature_manifest": dict(feature_manifest),
        "diagnostics": diagnostics.as_dict(),
        "bands": band_metadata(bands),
        "artifact_boundary": {
            "write_mode": "dry_run_manifest_only",
            "production_writes_enabled": False,
            "parquet_writes_enabled": False,
            "definition_writes_enabled": False,
        },
    }


def _default_dry_run_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "dry_run_diagnostic_only",
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
    }


def _default_source_probe_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "source_probe_json_only",
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
    }


def _validate_scaffold_artifact_boundary(boundary: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    checked = dict(boundary)
    if checked.get("production_writes_enabled") is not False:
        raise ValueError(f"{context} cannot claim production writes")
    if checked.get("parquet_writes_enabled") is not False:
        raise ValueError(f"{context} cannot claim parquet writes")
    if checked.get("definition_writes_enabled") is not False:
        raise ValueError(f"{context} cannot claim definition writes")
    return checked


def pathway_dry_run_diagnostic_record(
    *,
    pathway: str,
    run_id: str,
    config_summary: Mapping[str, Any],
    input_frame_contract: Mapping[str, Any],
    validation_results: Sequence[Mapping[str, Any]] = (),
    artifact_boundary: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    status: str = PATHWAY_MANIFEST_STATUS,
    schema_version: int = PATHWAY_DRY_RUN_DIAGNOSTIC_SCHEMA_VERSION,
) -> PathwayDryRunDiagnosticRecord:
    if str(status) != PATHWAY_MANIFEST_STATUS:
        raise ValueError(f"Pathway dry-run diagnostic status must be {PATHWAY_MANIFEST_STATUS!r}")
    boundary = _validate_scaffold_artifact_boundary(
        artifact_boundary or _default_dry_run_artifact_boundary(),
        context="Pathway dry-run diagnostic",
    )
    created = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return PathwayDryRunDiagnosticRecord(
        schema_version=int(schema_version),
        pathway=str(pathway),
        run_id=str(run_id),
        status=str(status),
        created_at_utc=str(created),
        config_summary=dict(config_summary),
        input_frame_contract=dict(input_frame_contract),
        artifact_boundary=boundary,
        validation_results=tuple(dict(result) for result in validation_results),
    )


def pathway_source_probe_record(
    *,
    pathway: str,
    run_id: str,
    source_summary: Mapping[str, Any],
    input_validation: Mapping[str, Any],
    artifact_boundary: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    status: str = PATHWAY_MANIFEST_STATUS,
    schema_version: int = PATHWAY_SOURCE_PROBE_SCHEMA_VERSION,
) -> PathwaySourceProbeRecord:
    if str(status) != PATHWAY_MANIFEST_STATUS:
        raise ValueError(f"Pathway source-probe status must be {PATHWAY_MANIFEST_STATUS!r}")
    boundary = _validate_scaffold_artifact_boundary(
        artifact_boundary or _default_source_probe_artifact_boundary(),
        context="Pathway source-probe",
    )
    created = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return PathwaySourceProbeRecord(
        schema_version=int(schema_version),
        pathway=str(pathway),
        run_id=str(run_id),
        status=str(status),
        created_at_utc=str(created),
        source_summary=dict(source_summary),
        input_validation=dict(input_validation),
        artifact_boundary=boundary,
    )


def write_pathway_dry_run_diagnostic(
    diagnostics_root: Path,
    *,
    pathway: str,
    run_id: str,
    config_summary: Mapping[str, Any],
    input_frame_contract: Mapping[str, Any],
    validation_results: Sequence[Mapping[str, Any]] = (),
    artifact_boundary: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    write_kind: str = "Regime pathway dry-run diagnostic",
) -> Path:
    require_pathway_diagnostics_root(Path(diagnostics_root))
    path = pathway_dry_run_diagnostic_path(diagnostics_root, pathway=pathway, run_id=run_id)
    record = pathway_dry_run_diagnostic_record(
        pathway=pathway,
        run_id=run_id,
        config_summary=config_summary,
        input_frame_contract=input_frame_contract,
        validation_results=validation_results,
        artifact_boundary=artifact_boundary,
        created_at_utc=created_at_utc,
    )
    write_json(path, record.as_dict(), write_kind=write_kind)
    return path


def write_pathway_source_probe(
    diagnostics_root: Path,
    *,
    pathway: str,
    run_id: str,
    source_summary: Mapping[str, Any],
    input_validation: Mapping[str, Any],
    artifact_boundary: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    write_kind: str = "Regime pathway source-probe diagnostic",
) -> Path:
    require_pathway_diagnostics_root(Path(diagnostics_root), for_source_probe=True)
    path = pathway_source_probe_path(diagnostics_root, pathway=pathway, run_id=run_id)
    record = pathway_source_probe_record(
        pathway=pathway,
        run_id=run_id,
        source_summary=source_summary,
        input_validation=input_validation,
        artifact_boundary=artifact_boundary,
        created_at_utc=created_at_utc,
    )
    write_json(path, record.as_dict(), write_kind=write_kind)
    return path


def write_market_universe_membership_snapshot(
    diagnostics_root: Path,
    *,
    universe: str,
    run_id: str,
    member_assets: Sequence[str],
    membership_source: str,
    provenance: Mapping[str, Any],
    snapshot_timestamp_utc: str | None = None,
    snapshot_scope: str = "market_state_universe_membership",
    min_assets: int,
    artifact_boundary: Mapping[str, Any] | None = None,
    write_kind: str = "Regime market universe membership snapshot",
) -> Path:
    require_pathway_diagnostics_root(Path(diagnostics_root), for_source_probe=True)
    path = market_universe_membership_snapshot_path(diagnostics_root, universe=universe, run_id=run_id)
    record = market_universe_membership_snapshot(
        universe=universe,
        member_assets=member_assets,
        membership_source=membership_source,
        provenance=provenance,
        snapshot_timestamp_utc=snapshot_timestamp_utc,
        snapshot_scope=snapshot_scope,
        min_assets=int(min_assets),
        artifact_boundary=artifact_boundary,
    )
    write_json(path, record.as_dict(), write_kind=write_kind)
    return path


def write_market_source_coverage_diagnostic(
    diagnostics_root: Path,
    *,
    run_id: str,
    universe: str,
    band: str,
    member_assets: Sequence[str],
    min_assets: int,
    membership_source: str,
    membership_snapshot: Mapping[str, Any] | None = None,
    membership_snapshot_path: Path | str | None = None,
    source_probe: Mapping[str, Any] | None = None,
    source_probe_path: Path | str | None = None,
    created_at_utc: str | None = None,
    artifact_boundary: Mapping[str, Any] | None = None,
    write_kind: str = "Regime market source coverage diagnostic",
) -> Path:
    require_pathway_diagnostics_root(Path(diagnostics_root), for_source_probe=True)
    path = market_source_coverage_diagnostic_path(diagnostics_root, universe=universe, run_id=run_id)
    record = market_source_coverage_diagnostic_record(
        run_id=run_id,
        universe=universe,
        band=band,
        member_assets=member_assets,
        min_assets=int(min_assets),
        membership_source=membership_source,
        membership_snapshot=membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        source_probe=source_probe,
        source_probe_path=source_probe_path,
        created_at_utc=created_at_utc,
        artifact_boundary=artifact_boundary,
    )
    write_json(path, record.as_dict(), write_kind=write_kind)
    return path


__all__ = [
    "MARKET_MEMBERSHIP_IMPLEMENTED_SOURCES",
    "MARKET_MEMBERSHIP_LIFECYCLE_STATUS",
    "MARKET_MEMBERSHIP_SOURCE_CLI_EXPLICIT",
    "MARKET_MEMBERSHIP_SOURCE_CLI_UNIVERSE_FILE",
    "MARKET_MEMBERSHIP_SOURCE_CONFIG_MEMBERS",
    "MARKET_MEMBERSHIP_SOURCE_POLICY_SCHEMA_VERSION",
    "MARKET_MEMBERSHIP_UNSUPPORTED_SOURCES",
    "MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_ARTIFACT_KIND",
    "MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_FAILURE_POLICY",
    "MARKET_AGGREGATION_SOURCE_READ_PRECONDITION_SCHEMA_VERSION",
    "MARKET_SOURCE_AVAILABILITY_LINEAGE_ARTIFACT_KIND",
    "MARKET_SOURCE_AVAILABILITY_LINEAGE_SCHEMA_VERSION",
    "MARKET_SOURCE_COVERAGE_DIAGNOSTIC_ARTIFACT_KIND",
    "MARKET_SOURCE_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION",
    "MARKET_UNIVERSE_MEMBERSHIP_INPUT_ARTIFACT_KIND",
    "MARKET_UNIVERSE_MEMBERSHIP_INPUT_PATH_POLICY_SCHEMA_VERSION",
    "MARKET_UNIVERSE_MEMBERSHIP_INPUT_SCHEMA_VERSION",
    "MARKET_UNIVERSE_LIFECYCLE_OWNER",
    "MARKET_UNIVERSE_LIFECYCLE_POLICY_ARTIFACT_KIND",
    "MARKET_UNIVERSE_LIFECYCLE_POLICY_SCHEMA_VERSION",
    "MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_ARTIFACT_KIND",
    "MARKET_UNIVERSE_MEMBERSHIP_SNAPSHOT_SCHEMA_VERSION",
    "MARKET_UNIVERSE_REFRESH_CADENCE",
    "MARKET_UNIVERSE_REFRESH_MODE",
    "MARKET_UNIVERSE_REGISTRY_SUPPORT_STATUS",
    "MARKET_UNIVERSE_STALENESS_WINDOW",
    "MARKET_UNIVERSE_FAILURE_POLICY",
    "PATHWAY_DIAGNOSTICS_ROOT_EXPLICIT",
    "PATHWAY_DIAGNOSTICS_ROOT_POLICY_VERSION",
    "PATHWAY_DIAGNOSTICS_ROOT_REPORT",
    "PATHWAY_DIAGNOSTICS_ROOT_SANDBOX_TEMP",
    "PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION",
    "PATHWAY_DRY_RUN_DIAGNOSTIC_SCHEMA_VERSION",
    "PATHWAY_MANIFEST_STATUS",
    "PATHWAY_SOURCE_PROBE_SCHEMA_VERSION",
    "RELATIVE_BENCHMARK_IMPLEMENTED_SOURCES",
    "RELATIVE_BENCHMARK_SOURCE_CLI",
    "RELATIVE_BENCHMARK_SOURCE_CONFIG",
    "RELATIVE_BENCHMARK_SOURCE_POLICY_ARTIFACT_KIND",
    "RELATIVE_BENCHMARK_SUBSTITUTION_POLICY_NONE",
    "RELATIVE_BENCHMARK_UNSUPPORTED_SOURCES",
    "RELATIVE_MISSING_BENCHMARK_POLICIES",
    "RELATIVE_OWNERSHIP_FAILURE_POLICY",
    "RELATIVE_OWNERSHIP_LIFECYCLE_STATUS",
    "RELATIVE_OWNERSHIP_OWNER",
    "RELATIVE_OWNERSHIP_POLICY_SCHEMA_VERSION",
    "RELATIVE_PEER_BASKET_IMPLEMENTED_SOURCES",
    "RELATIVE_PEER_BASKET_LIFECYCLE_POLICY_ARTIFACT_KIND",
    "RELATIVE_PEER_BASKET_REFRESH_CADENCE",
    "RELATIVE_PEER_BASKET_REFRESH_MODE",
    "RELATIVE_PEER_BASKET_REGISTRY_SUPPORT_STATUS",
    "RELATIVE_PEER_BASKET_SOURCE_CLI",
    "RELATIVE_PEER_BASKET_SOURCE_CONFIG",
    "RELATIVE_PEER_BASKET_SOURCE_POLICY_ARTIFACT_KIND",
    "RELATIVE_PEER_BASKET_STALENESS_WINDOW",
    "RELATIVE_PEER_BASKET_UNSUPPORTED_SOURCES",
    "RELATIVE_SOURCE_READ_PRECONDITION_ARTIFACT_KIND",
    "RELATIVE_SOURCE_READ_PRECONDITION_FAILURE_POLICY",
    "RELATIVE_SOURCE_READ_PRECONDITION_SCHEMA_VERSION",
    "SCALAR_FEATURE_SOURCE_PROBE_KIND",
    "MarketAggregationSourceReadPrecondition",
    "MarketMembershipSourcePolicy",
    "MarketSourceAvailabilityLineage",
    "MarketUniverseMembershipInput",
    "MarketUniverseMembershipInputPathPolicy",
    "MarketUniverseMembershipSnapshot",
    "MarketUniverseMembershipValidationResult",
    "MarketUniverseLifecyclePolicy",
    "MarketUniverseSnapshotMetadata",
    "MarketSourceCoverageDiagnosticRecord",
    "PathwayArtifactMetadataSchema",
    "PathwayContractLike",
    "PathwayDiagnosticReportSchema",
    "PathwayDiagnosticsRootPolicy",
    "PathwayDryRunDiagnosticRecord",
    "PathwaySourceProbeRecord",
    "RelativeBenchmarkSourcePolicy",
    "RelativePeerBasketLifecyclePolicy",
    "RelativePeerBasketSourcePolicy",
    "RelativeSourceReadPrecondition",
    "RelativeSourceReadPreconditionValidationResult",
    "ScalarFeatureAssetPartitionSummary",
    "ScalarFeatureIntervalPartitionSummary",
    "ScalarFeatureSourcePartitionSummary",
    "artifact_metadata_schema",
    "band_metadata",
    "classify_market_universe_membership_input_path",
    "classify_pathway_diagnostics_root",
    "diagnostic_report_schema",
    "dry_run_manifest",
    "load_market_universe_membership_input_file",
    "market_aggregation_source_read_precondition",
    "market_aggregation_source_read_precondition_path",
    "market_membership_snapshot_provenance",
    "market_membership_source_policy",
    "market_source_availability_lineage",
    "market_universe_lifecycle_policy",
    "market_universe_membership_input_from_payload",
    "market_universe_membership_snapshot",
    "market_universe_membership_snapshot_path",
    "market_universe_snapshot_metadata",
    "market_source_coverage_diagnostic_path",
    "market_source_coverage_diagnostic_record",
    "pathway_dry_run_diagnostic_path",
    "pathway_dry_run_diagnostic_record",
    "pathway_month_dir",
    "pathway_part_path",
    "relative_benchmark_source_policy",
    "relative_peer_basket_lifecycle_policy",
    "relative_peer_basket_source_policy",
    "relative_source_read_precondition",
    "relative_source_read_precondition_path",
    "pathway_source_probe_path",
    "pathway_source_probe_record",
    "pathway_table_dir",
    "require_market_universe_membership_input_path",
    "require_pathway_diagnostics_root",
    "summarize_scalar_feature_source_partitions",
    "validate_market_universe_membership_input_payload",
    "validate_market_universe_membership_snapshot_inputs",
    "write_market_aggregation_source_read_precondition",
    "write_market_universe_membership_snapshot",
    "write_market_source_coverage_diagnostic",
    "write_pathway_dry_run_diagnostic",
    "write_pathway_source_probe",
]
