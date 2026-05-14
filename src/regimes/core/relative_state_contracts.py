from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    RegimeAxis,
    RegimeBand,
    RegimeClassification,
    RegimeLayer,
    normalize_enum_value,
    normalize_string_tuple,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
    validate_layer_axis_band,
)
from src.regimes.core.market_state_contracts import validate_market_state_metadata_report_root
from src.regimes.core.paths import default_foundation_report_root
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


RELATIVE_STATE_METADATA_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
RELATIVE_STATE_METADATA_ARTIFACT_KIND = "regime_relative_state_metadata_manifest"
RELATIVE_STATE_FEATURE_FAMILY_ARTIFACT_KIND = "regime_relative_state_feature_family_metadata"
RELATIVE_STATE_LAYER = RegimeLayer.RELATIVE_STATE.value

RELATIVE_ALIGNMENT_POLICIES: tuple[str, ...] = (
    "exact_timestamp_intersection",
    "calendar_intersection",
    "asof_backward_with_tolerance",
    "metadata_only_unspecified",
)
RELATIVE_FEATURE_FAMILY_NAMES: tuple[str, ...] = (
    "relative_return",
    "rolling_beta",
    "rolling_correlation",
    "relative_strength_rank",
    "relative_volatility",
    "peer_dispersion_distance",
)


def _token(value: object, *, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    return require_non_empty_string(value, field_name=field_name).lower()


def _text(value: object, *, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    return require_non_empty_string(value, field_name=field_name)


def validate_relative_state_metadata_report_root(
    report_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    policy = validate_market_state_metadata_report_root(report_root, project_root=project_root)
    return {
        **policy,
        "validator": "relative_state_metadata_report_root",
    }


def _metadata_boundary(write_mode: str) -> dict[str, Any]:
    return {
        "write_mode": write_mode,
        "metadata_only": True,
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "state_writes_enabled": False,
        "production_outputs_written": False,
        "alignment_execution_enabled": False,
        "production_readers_enabled": False,
        "downstream_non_asset_readers_enabled": False,
    }


@dataclass(frozen=True)
class BenchmarkIdentity:
    benchmark_id: str
    benchmark_kind: str = "asset"
    source: str = "metadata_declaration"
    metadata_only: bool = True
    production_writes_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = RELATIVE_STATE_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        if self.metadata_only is not True:
            raise ValueError("Regime relative-state benchmark identity must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime relative-state benchmark identity cannot enable production writes")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "benchmark_id", _text(self.benchmark_id, field_name="benchmark_id"))
        object.__setattr__(self, "benchmark_kind", _token(self.benchmark_kind, field_name="benchmark_kind"))
        object.__setattr__(self, "source", _text(self.source, field_name="source"))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "benchmark_id": self.benchmark_id,
            "benchmark_kind": self.benchmark_kind,
            "source": self.source,
            "metadata_only": True,
            "production_writes_enabled": False,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkIdentity":
        obj = require_json_object(payload, context="Regime BenchmarkIdentity")
        return cls(
            schema_version=obj.get("schema_version", RELATIVE_STATE_METADATA_SCHEMA_VERSION),
            benchmark_id=obj["benchmark_id"],
            benchmark_kind=obj.get("benchmark_kind", "asset"),
            source=obj.get("source", "metadata_declaration"),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
            metadata=obj.get("metadata", {}),
        )


@dataclass(frozen=True)
class PeerGroupIdentity:
    peer_group_id: str
    member_assets: Sequence[str]
    universe: str = "global"
    membership_source: str = "metadata_declaration"
    metadata_only: bool = True
    production_writes_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = RELATIVE_STATE_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        members = normalize_string_tuple(self.member_assets, field_name="member_assets", require_non_empty=True)
        if self.metadata_only is not True:
            raise ValueError("Regime relative-state peer group must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime relative-state peer group cannot enable production writes")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "peer_group_id", _text(self.peer_group_id, field_name="peer_group_id"))
        object.__setattr__(self, "member_assets", members)
        object.__setattr__(self, "universe", _text(self.universe, field_name="universe"))
        object.__setattr__(self, "membership_source", _text(self.membership_source, field_name="membership_source"))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "peer_group_id": self.peer_group_id,
            "universe": self.universe,
            "member_assets": list(self.member_assets),
            "member_asset_count": int(len(self.member_assets)),
            "membership_source": self.membership_source,
            "metadata_only": True,
            "production_writes_enabled": False,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeerGroupIdentity":
        obj = require_json_object(payload, context="Regime PeerGroupIdentity")
        return cls(
            schema_version=obj.get("schema_version", RELATIVE_STATE_METADATA_SCHEMA_VERSION),
            peer_group_id=obj["peer_group_id"],
            member_assets=obj["member_assets"],
            universe=obj.get("universe", "global"),
            membership_source=obj.get("membership_source", "metadata_declaration"),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
            metadata=obj.get("metadata", {}),
        )


@dataclass(frozen=True)
class AlignmentFramePolicy:
    policy_name: str = "metadata_only_alignment_policy"
    timestamp_alignment_policy: str = "exact_timestamp_intersection"
    missing_benchmark_policy: str = "require"
    stale_data_policy: Mapping[str, Any] = field(default_factory=lambda: {"policy": "fail_closed", "max_stale_intervals": 1})
    lookback_windows: Sequence[int] = (20, 60)
    metadata_only: bool = True
    production_writes_enabled: bool = False
    schema_version: int = RELATIVE_STATE_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        alignment = _token(self.timestamp_alignment_policy, field_name="timestamp_alignment_policy")
        if alignment not in RELATIVE_ALIGNMENT_POLICIES:
            valid = ", ".join(RELATIVE_ALIGNMENT_POLICIES)
            raise ValueError(f"Unsupported Regime relative-state alignment policy {alignment!r}; expected one of: {valid}")
        lookbacks = tuple(int(window) for window in self.lookback_windows)
        if not lookbacks or any(window <= 0 for window in lookbacks):
            raise ValueError("Regime relative-state alignment lookback_windows must be positive")
        if self.metadata_only is not True:
            raise ValueError("Regime relative-state alignment policy must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime relative-state alignment policy cannot enable production writes")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "policy_name", _text(self.policy_name, field_name="policy_name"))
        object.__setattr__(self, "timestamp_alignment_policy", alignment)
        object.__setattr__(self, "missing_benchmark_policy", _token(self.missing_benchmark_policy, field_name="missing_benchmark_policy"))
        object.__setattr__(self, "stale_data_policy", require_json_mapping(self.stale_data_policy, field_name="stale_data_policy"))
        object.__setattr__(self, "lookback_windows", lookbacks)
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "policy_name": self.policy_name,
            "timestamp_alignment_policy": self.timestamp_alignment_policy,
            "missing_benchmark_policy": self.missing_benchmark_policy,
            "stale_data_policy": to_jsonable(self.stale_data_policy),
            "lookback_windows": list(self.lookback_windows),
            "alignment_frame_materialized": False,
            "metadata_only": True,
            "production_writes_enabled": False,
            "artifact_boundary": _metadata_boundary("relative_alignment_frame_policy_metadata_only"),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AlignmentFramePolicy":
        obj = require_json_object(payload, context="Regime AlignmentFramePolicy")
        return cls(
            schema_version=obj.get("schema_version", RELATIVE_STATE_METADATA_SCHEMA_VERSION),
            policy_name=obj.get("policy_name", "metadata_only_alignment_policy"),
            timestamp_alignment_policy=obj.get("timestamp_alignment_policy", "exact_timestamp_intersection"),
            missing_benchmark_policy=obj.get("missing_benchmark_policy", "require"),
            stale_data_policy=obj.get("stale_data_policy", {"policy": "fail_closed", "max_stale_intervals": 1}),
            lookback_windows=obj.get("lookback_windows", (20, 60)),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
        )


@dataclass(frozen=True)
class RelativeFeatureFamilyDeclaration:
    family_name: str
    axis: str | RegimeAxis
    required_source_columns: Sequence[str]
    derived_feature_columns: Sequence[str]
    requires_benchmark: bool = True
    requires_peer_group: bool = False
    lookback_windows: Sequence[int] = (20, 60)
    lineage_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_only: bool = True
    production_writes_enabled: bool = False
    schema_version: int = RELATIVE_STATE_METADATA_SCHEMA_VERSION
    artifact_kind: str = RELATIVE_STATE_FEATURE_FAMILY_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family = _token(self.family_name, field_name="feature family name")
        if family not in RELATIVE_FEATURE_FAMILY_NAMES:
            valid = ", ".join(RELATIVE_FEATURE_FAMILY_NAMES)
            raise ValueError(f"Unsupported Regime relative-state feature family {family!r}; expected one of: {valid}")
        axis = normalize_enum_value(self.axis, RegimeAxis, field_name="axis")
        validate_layer_axis_band(layer=RELATIVE_STATE_LAYER, axis=axis, band=RegimeBand.MICRO.value)
        source_columns = normalize_string_tuple(self.required_source_columns, field_name="required_source_columns", require_non_empty=True)
        derived_columns = normalize_string_tuple(self.derived_feature_columns, field_name="derived_feature_columns", require_non_empty=True)
        lookbacks = tuple(int(window) for window in self.lookback_windows)
        if not lookbacks or any(window <= 0 for window in lookbacks):
            raise ValueError("Regime relative-state feature lookback_windows must be positive")
        if self.metadata_only is not True:
            raise ValueError("Regime relative-state feature metadata must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime relative-state feature metadata cannot enable production writes")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "required_source_columns", source_columns)
        object.__setattr__(self, "derived_feature_columns", derived_columns)
        object.__setattr__(self, "lookback_windows", lookbacks)
        object.__setattr__(self, "requires_benchmark", bool(self.requires_benchmark))
        object.__setattr__(self, "requires_peer_group", bool(self.requires_peer_group))
        object.__setattr__(
            self,
            "lineage_metadata",
            {
                "artifact_kind": "regime_relative_state_metadata_contract",
                "produced_by": "src.regimes.core.relative_state_contracts",
                **to_jsonable(require_json_mapping(self.lineage_metadata, field_name="lineage_metadata")),
            },
        )
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        return _metadata_boundary("relative_state_feature_family_metadata_only")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIVE_STATE_LAYER,
            "family_name": self.family_name,
            "axis": self.axis,
            "required_source_columns": list(self.required_source_columns),
            "derived_feature_columns": list(self.derived_feature_columns),
            "lookback_windows": list(self.lookback_windows),
            "requires_benchmark": bool(self.requires_benchmark),
            "requires_peer_group": bool(self.requires_peer_group),
            "metadata_only": True,
            "production_writes_enabled": False,
            "lineage_metadata": to_jsonable(self.lineage_metadata),
            "artifact_boundary": self.artifact_boundary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelativeFeatureFamilyDeclaration":
        obj = require_json_object(payload, context="Regime RelativeFeatureFamilyDeclaration")
        return cls(
            schema_version=obj.get("schema_version", RELATIVE_STATE_METADATA_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", RELATIVE_STATE_FEATURE_FAMILY_ARTIFACT_KIND),
            family_name=obj["family_name"],
            axis=obj["axis"],
            required_source_columns=obj["required_source_columns"],
            derived_feature_columns=obj["derived_feature_columns"],
            requires_benchmark=obj.get("requires_benchmark", True),
            requires_peer_group=obj.get("requires_peer_group", False),
            lookback_windows=obj.get("lookback_windows", (20, 60)),
            lineage_metadata=obj.get("lineage_metadata", {}),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
        )


@dataclass(frozen=True)
class RelativeStateMetadataManifest:
    manifest_id: str
    primary_asset: str
    benchmark_identity: BenchmarkIdentity | Mapping[str, Any]
    peer_group_identity: PeerGroupIdentity | Mapping[str, Any]
    alignment_frame_policy: AlignmentFramePolicy | Mapping[str, Any]
    feature_families: Sequence[RelativeFeatureFamilyDeclaration | Mapping[str, Any]]
    report_root: str | Path
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_only: bool = True
    production_writes_enabled: bool = False
    classification: str | RegimeClassification = RegimeClassification.METADATA_ONLY.value
    schema_version: int = RELATIVE_STATE_METADATA_SCHEMA_VERSION
    artifact_kind: str = RELATIVE_STATE_METADATA_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        classification = normalize_enum_value(self.classification, RegimeClassification, field_name="classification")
        if classification != RegimeClassification.METADATA_ONLY.value:
            raise ValueError("Regime relative-state metadata manifest classification must be metadata_only")
        benchmark = (
            self.benchmark_identity
            if isinstance(self.benchmark_identity, BenchmarkIdentity)
            else BenchmarkIdentity.from_dict(self.benchmark_identity)
        )
        peers = (
            self.peer_group_identity
            if isinstance(self.peer_group_identity, PeerGroupIdentity)
            else PeerGroupIdentity.from_dict(self.peer_group_identity)
        )
        alignment = (
            self.alignment_frame_policy
            if isinstance(self.alignment_frame_policy, AlignmentFramePolicy)
            else AlignmentFramePolicy.from_dict(self.alignment_frame_policy)
        )
        families = tuple(
            family if isinstance(family, RelativeFeatureFamilyDeclaration) else RelativeFeatureFamilyDeclaration.from_dict(family)
            for family in self.feature_families
        )
        if not families:
            raise ValueError("Regime relative-state metadata manifest requires feature families")
        if self.metadata_only is not True:
            raise ValueError("Regime relative-state metadata manifest must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime relative-state metadata manifest cannot enable production writes")
        report_policy = validate_relative_state_metadata_report_root(self.report_root)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "manifest_id", _text(self.manifest_id, field_name="manifest_id"))
        object.__setattr__(self, "primary_asset", _text(self.primary_asset, field_name="primary_asset"))
        object.__setattr__(self, "benchmark_identity", benchmark)
        object.__setattr__(self, "peer_group_identity", peers)
        object.__setattr__(self, "alignment_frame_policy", alignment)
        object.__setattr__(self, "feature_families", families)
        object.__setattr__(self, "report_root", report_policy["root"])
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)
        object.__setattr__(self, "classification", classification)

    @property
    def report_root_policy(self) -> dict[str, Any]:
        return validate_relative_state_metadata_report_root(self.report_root)

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        return _metadata_boundary("relative_state_metadata_manifest_json_only")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "manifest_id": self.manifest_id,
            "layer": RELATIVE_STATE_LAYER,
            "classification": self.classification,
            "primary_asset": self.primary_asset,
            "metadata_only": True,
            "production_writes_enabled": False,
            "benchmark_identity": self.benchmark_identity.as_dict(),
            "peer_group_identity": self.peer_group_identity.as_dict(),
            "alignment_frame_policy": self.alignment_frame_policy.as_dict(),
            "feature_families": [family.as_dict() for family in self.feature_families],
            "report_root": str(self.report_root),
            "report_root_policy": self.report_root_policy,
            "metadata": to_jsonable(self.metadata),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelativeStateMetadataManifest":
        obj = require_json_object(payload, context="Regime RelativeStateMetadataManifest")
        return cls(
            schema_version=obj.get("schema_version", RELATIVE_STATE_METADATA_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", RELATIVE_STATE_METADATA_ARTIFACT_KIND),
            manifest_id=obj["manifest_id"],
            primary_asset=obj["primary_asset"],
            classification=obj.get("classification", RegimeClassification.METADATA_ONLY.value),
            benchmark_identity=obj["benchmark_identity"],
            peer_group_identity=obj["peer_group_identity"],
            alignment_frame_policy=obj["alignment_frame_policy"],
            feature_families=obj["feature_families"],
            report_root=obj["report_root"],
            metadata=obj.get("metadata", {}),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
        )

    @classmethod
    def from_json(cls, text: str) -> "RelativeStateMetadataManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime RelativeStateMetadataManifest JSON"))


def default_relative_feature_family_declarations() -> tuple[RelativeFeatureFamilyDeclaration, ...]:
    lineage = {"source": "default_relative_feature_family_declarations"}
    return (
        RelativeFeatureFamilyDeclaration(
            family_name="relative_return",
            axis=RegimeAxis.RELATIVE,
            required_source_columns=("timestamp", "asset_return", "benchmark_return"),
            derived_feature_columns=("relative_return", "excess_return_zscore"),
            lineage_metadata=lineage,
        ),
        RelativeFeatureFamilyDeclaration(
            family_name="rolling_beta",
            axis=RegimeAxis.BETA,
            required_source_columns=("timestamp", "asset_return", "benchmark_return"),
            derived_feature_columns=("rolling_beta_to_benchmark", "downside_beta"),
            lineage_metadata=lineage,
        ),
        RelativeFeatureFamilyDeclaration(
            family_name="rolling_correlation",
            axis=RegimeAxis.CORRELATION,
            required_source_columns=("timestamp", "asset_return", "benchmark_return", "peer_basket_return"),
            derived_feature_columns=("rolling_corr_to_benchmark", "rolling_corr_to_peer_group"),
            requires_peer_group=True,
            lineage_metadata=lineage,
        ),
        RelativeFeatureFamilyDeclaration(
            family_name="relative_strength_rank",
            axis=RegimeAxis.RELATIVE_STRENGTH,
            required_source_columns=("timestamp", "asset_return", "peer_group_returns"),
            derived_feature_columns=("relative_momentum_rank", "cross_sectional_return_percentile"),
            requires_peer_group=True,
            lineage_metadata=lineage,
        ),
        RelativeFeatureFamilyDeclaration(
            family_name="relative_volatility",
            axis=RegimeAxis.RELATIVE_DISPERSION,
            required_source_columns=("timestamp", "asset_return", "benchmark_return", "peer_group_returns"),
            derived_feature_columns=("relative_volatility_ratio", "rank_volatility"),
            requires_peer_group=True,
            lineage_metadata=lineage,
        ),
        RelativeFeatureFamilyDeclaration(
            family_name="peer_dispersion_distance",
            axis=RegimeAxis.RELATIVE_DISPERSION,
            required_source_columns=("timestamp", "asset_return", "peer_group_returns"),
            derived_feature_columns=("distance_from_peer_median", "peer_dispersion_zscore"),
            requires_peer_group=True,
            lineage_metadata=lineage,
        ),
    )


def build_relative_state_metadata_manifest(
    *,
    manifest_id: str = "relative_state_metadata_manifest",
    primary_asset: str = "ETHUSD",
    benchmark_identity: BenchmarkIdentity | Mapping[str, Any] | None = None,
    peer_group_identity: PeerGroupIdentity | Mapping[str, Any] | None = None,
    alignment_frame_policy: AlignmentFramePolicy | Mapping[str, Any] | None = None,
    feature_families: Sequence[RelativeFeatureFamilyDeclaration | Mapping[str, Any]] | None = None,
    report_root: str | Path = default_foundation_report_root("relative_state_metadata"),
    metadata: Mapping[str, Any] | None = None,
) -> RelativeStateMetadataManifest:
    return RelativeStateMetadataManifest(
        manifest_id=manifest_id,
        primary_asset=primary_asset,
        benchmark_identity=benchmark_identity
        or BenchmarkIdentity(benchmark_id="BTCUSD", source="default_relative_state_metadata"),
        peer_group_identity=peer_group_identity
        or PeerGroupIdentity(
            peer_group_id="crypto_large_cap_peers",
            universe="global",
            member_assets=("BTCUSD", "SOLUSD", "ADAUSD"),
            membership_source="default_relative_state_metadata",
        ),
        alignment_frame_policy=alignment_frame_policy or AlignmentFramePolicy(),
        feature_families=tuple(feature_families or default_relative_feature_family_declarations()),
        report_root=report_root,
        metadata=metadata or {},
    )


__all__ = [
    "RELATIVE_ALIGNMENT_POLICIES",
    "RELATIVE_FEATURE_FAMILY_NAMES",
    "RELATIVE_STATE_FEATURE_FAMILY_ARTIFACT_KIND",
    "RELATIVE_STATE_LAYER",
    "RELATIVE_STATE_METADATA_ARTIFACT_KIND",
    "RELATIVE_STATE_METADATA_SCHEMA_VERSION",
    "AlignmentFramePolicy",
    "BenchmarkIdentity",
    "PeerGroupIdentity",
    "RelativeFeatureFamilyDeclaration",
    "RelativeStateMetadataManifest",
    "build_relative_state_metadata_manifest",
    "default_relative_feature_family_declarations",
    "validate_relative_state_metadata_report_root",
]
