from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.regime_features.lineage import (
    RegimeFeatureKnownAtSpec,
    RegimeFeatureLineageSpec,
)


REGIME_FEATURES_LAYER = "regime_features"
REGIME_FEATURES_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION

MARKET_REGIME_FEATURES = "market_regime_features"
PAIRWISE_RELATIONSHIP_FEATURES = "pairwise_relationship_features"
CROSS_ASSET_SUMMARY_FEATURES = "cross_asset_summary_features"
ELIGIBILITY_SNAPSHOT = "eligibility_snapshot"
UNIVERSE_SNAPSHOT = "universe_snapshot"

REGIME_FEATURE_ARTIFACT_FAMILIES: tuple[str, ...] = (
    MARKET_REGIME_FEATURES,
    PAIRWISE_RELATIONSHIP_FEATURES,
    CROSS_ASSET_SUMMARY_FEATURES,
    ELIGIBILITY_SNAPSHOT,
    UNIVERSE_SNAPSHOT,
)

REGIME_FEATURE_OUTPUT_SCHEMA_ARTIFACT_KIND = "regime_feature_output_schema"
REGIME_FEATURE_ARTIFACT_BOUNDARY_KIND = "regime_feature_artifact_boundary"
REGIME_FEATURE_CONSUMER_CONTRACT_KIND = "regime_feature_consumer_contract"

MARKET_REGIME_FEATURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts",
    "interval",
    "band",
    "feature_set_id",
    "universe_policy_id",
    "universe_snapshot_id",
    "known_at_ts",
    "source_tail_ts",
    "lineage_id",
    "schema_version",
)

PAIRWISE_RELATIONSHIP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts",
    "interval",
    "band",
    "window",
    "asset",
    "related_asset_or_benchmark",
    "relationship_type",
    "value",
    "sample_count",
    "coverage",
    "known_at_ts",
    "lineage_id",
    "schema_version",
)

CROSS_ASSET_SUMMARY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts",
    "asset",
    "interval",
    "band",
    "universe_policy_id",
    "known_at_ts",
    "source_tail_ts",
    "lineage_id",
    "schema_version",
)

UNIVERSE_SNAPSHOT_REQUIRED_FIELDS: tuple[str, ...] = (
    "selection_policy_id",
    "refit_key",
    "interval",
    "band",
    "known_at",
    "source_tail_ts",
    "core_basket_assets",
    "broad_universe_assets",
    "excluded_assets_with_reasons",
    "eligibility_diagnostics",
    "lineage",
    "schema_version",
)


class RegimeFeatureSchemaVersion(IntEnum):
    V1 = REGIME_FEATURES_SCHEMA_VERSION


class RegimeFeatureArtifactKind(str, Enum):
    MARKET_FEATURES = "market_features"
    PAIRWISE_RELATIONSHIP_FEATURES = "pairwise_relationship_features"
    CROSS_ASSET_SUMMARY_FEATURES = "cross_asset_summary_features"
    UNIVERSE_SNAPSHOT = "universe_snapshot"
    BASKET_SNAPSHOT = "basket_snapshot"
    PEER_SNAPSHOT_PLACEHOLDER = "peer_snapshot_placeholder"


class RegimeFeatureFamilyKind(str, Enum):
    MARKET = "market"
    PAIRWISE_RELATIONSHIP = "pairwise_relationship"
    CROSS_ASSET_SUMMARY = "cross_asset_summary"
    UNIVERSE = "universe"
    BASKET = "basket"
    PEER_PLACEHOLDER = "peer_placeholder"


class RegimeFeatureScope(str, Enum):
    MARKET = "market"
    ASSET = "asset"
    PAIR = "pair"
    UNIVERSE = "universe"
    BASKET = "basket"
    PEER_PLACEHOLDER = "peer_placeholder"


class RegimeFeatureBuildStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    EMPTY = "empty"
    SCAFFOLD_ONLY = "scaffold_only"


ARTIFACT_KIND_TO_FAMILY: Mapping[RegimeFeatureArtifactKind, RegimeFeatureFamilyKind] = {
    RegimeFeatureArtifactKind.MARKET_FEATURES: RegimeFeatureFamilyKind.MARKET,
    RegimeFeatureArtifactKind.PAIRWISE_RELATIONSHIP_FEATURES: RegimeFeatureFamilyKind.PAIRWISE_RELATIONSHIP,
    RegimeFeatureArtifactKind.CROSS_ASSET_SUMMARY_FEATURES: RegimeFeatureFamilyKind.CROSS_ASSET_SUMMARY,
    RegimeFeatureArtifactKind.UNIVERSE_SNAPSHOT: RegimeFeatureFamilyKind.UNIVERSE,
    RegimeFeatureArtifactKind.BASKET_SNAPSHOT: RegimeFeatureFamilyKind.BASKET,
    RegimeFeatureArtifactKind.PEER_SNAPSHOT_PLACEHOLDER: RegimeFeatureFamilyKind.PEER_PLACEHOLDER,
}

FAMILY_TO_DEFAULT_SCOPE: Mapping[RegimeFeatureFamilyKind, RegimeFeatureScope] = {
    RegimeFeatureFamilyKind.MARKET: RegimeFeatureScope.MARKET,
    RegimeFeatureFamilyKind.PAIRWISE_RELATIONSHIP: RegimeFeatureScope.PAIR,
    RegimeFeatureFamilyKind.CROSS_ASSET_SUMMARY: RegimeFeatureScope.ASSET,
    RegimeFeatureFamilyKind.UNIVERSE: RegimeFeatureScope.UNIVERSE,
    RegimeFeatureFamilyKind.BASKET: RegimeFeatureScope.BASKET,
    RegimeFeatureFamilyKind.PEER_PLACEHOLDER: RegimeFeatureScope.PEER_PLACEHOLDER,
}


@dataclass(frozen=True)
class RegimeFeatureRowKey:
    artifact_kind: RegimeFeatureArtifactKind | str
    interval: int
    band: str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    lineage_id: str
    ts: int | float | str | None = None
    asset: str | None = None
    related_asset_or_benchmark: str | None = None
    universe_policy_id: str | None = None
    refit_key: str | None = None
    clamp_policy_id: str | None = None
    schema_version: int | RegimeFeatureSchemaVersion = RegimeFeatureSchemaVersion.V1

    def __post_init__(self) -> None:
        artifact = _artifact_kind(self.artifact_kind)
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature row key interval must be positive")
        if artifact in {
            RegimeFeatureArtifactKind.MARKET_FEATURES,
            RegimeFeatureArtifactKind.PAIRWISE_RELATIONSHIP_FEATURES,
            RegimeFeatureArtifactKind.CROSS_ASSET_SUMMARY_FEATURES,
        } and self.ts is None:
            raise ValueError("Regime Feature row key ts is required for materialized feature rows")
        if artifact == RegimeFeatureArtifactKind.CROSS_ASSET_SUMMARY_FEATURES and _optional_text(self.asset) is None:
            raise ValueError("Regime Feature cross-asset row key requires asset")
        if artifact == RegimeFeatureArtifactKind.PAIRWISE_RELATIONSHIP_FEATURES:
            if _optional_text(self.asset) is None:
                raise ValueError("Regime Feature pairwise row key requires asset")
            if _optional_text(self.related_asset_or_benchmark) is None:
                raise ValueError("Regime Feature pairwise row key requires related_asset_or_benchmark")
        _to_orderable(self.known_at_ts, field_name="known_at_ts")
        _to_orderable(self.source_tail_ts, field_name="source_tail_ts")
        if self.ts is not None:
            _to_orderable(self.ts, field_name="ts")
        object.__setattr__(self, "artifact_kind", artifact)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        object.__setattr__(self, "asset", _optional_text(self.asset))
        object.__setattr__(self, "related_asset_or_benchmark", _optional_text(self.related_asset_or_benchmark))
        object.__setattr__(self, "universe_policy_id", _optional_text(self.universe_policy_id))
        object.__setattr__(self, "refit_key", _optional_text(self.refit_key))
        object.__setattr__(self, "clamp_policy_id", _optional_text(self.clamp_policy_id))
        object.__setattr__(self, "schema_version", require_schema_version(int(self.schema_version)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind.value,
            "interval": int(self.interval),
            "band": self.band,
            "ts": self.ts,
            "asset": self.asset,
            "related_asset_or_benchmark": self.related_asset_or_benchmark,
            "universe_policy_id": self.universe_policy_id,
            "refit_key": self.refit_key,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "lineage_id": self.lineage_id,
            "clamp_policy_id": self.clamp_policy_id,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureRowKey":
        obj = require_json_object(payload, context="RegimeFeatureRowKey")
        required = {"artifact_kind", "interval", "band", "known_at_ts", "source_tail_ts", "lineage_id", "schema_version"}
        missing = sorted(required.difference(obj))
        if missing:
            raise ValueError(f"RegimeFeatureRowKey missing required fields: {', '.join(missing)}")
        return cls(
            artifact_kind=obj["artifact_kind"],
            interval=obj["interval"],
            band=obj["band"],
            ts=obj.get("ts"),
            asset=obj.get("asset"),
            related_asset_or_benchmark=obj.get("related_asset_or_benchmark"),
            universe_policy_id=obj.get("universe_policy_id"),
            refit_key=obj.get("refit_key"),
            known_at_ts=obj["known_at_ts"],
            source_tail_ts=obj["source_tail_ts"],
            lineage_id=obj["lineage_id"],
            clamp_policy_id=obj.get("clamp_policy_id"),
            schema_version=obj["schema_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureRowKey":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureRowKey JSON"))


@dataclass(frozen=True)
class RegimeFeatureArtifactBoundary:
    artifact_family: str | RegimeFeatureFamilyKind | None = None
    artifact_kind: str | RegimeFeatureArtifactKind | None = None
    write_scope: str = "foundation_or_sandbox_only"
    persistent: bool = True
    production_enabled: bool = False
    production_parquet_enabled: bool = False
    production_promotion_enabled: bool = False
    pairwise_materialization_enabled: bool = False
    broad_all_to_all_pairwise_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    boundary_kind: str = REGIME_FEATURE_ARTIFACT_BOUNDARY_KIND

    def __post_init__(self) -> None:
        artifact_kind = _coerce_boundary_artifact_kind(self.artifact_kind, self.artifact_family)
        family_kind = _coerce_boundary_family_kind(self.artifact_family, artifact_kind)
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "boundary_kind", _text(self.boundary_kind, field_name="boundary_kind"))
        object.__setattr__(self, "artifact_kind", artifact_kind)
        object.__setattr__(self, "artifact_family", family_kind)
        object.__setattr__(self, "write_scope", _text(self.write_scope, field_name="write_scope"))
        _require_disabled(self.production_enabled, field_name="production_enabled")
        _require_disabled(self.production_parquet_enabled, field_name="production_parquet_enabled")
        _require_disabled(self.production_promotion_enabled, field_name="production_promotion_enabled")
        _require_disabled(self.pairwise_materialization_enabled, field_name="pairwise_materialization_enabled")
        _require_disabled(self.broad_all_to_all_pairwise_enabled, field_name="broad_all_to_all_pairwise_enabled")
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "production_parquet_enabled", False)
        object.__setattr__(self, "production_promotion_enabled", False)
        object.__setattr__(self, "pairwise_materialization_enabled", False)
        object.__setattr__(self, "broad_all_to_all_pairwise_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind.value,
            "artifact_family": self.artifact_family.value,
            "boundary_kind": self.boundary_kind,
            "layer": REGIME_FEATURES_LAYER,
            "write_scope": self.write_scope,
            "persistent": bool(self.persistent),
            "production_enabled": False,
            "production_parquet_enabled": False,
            "production_promotion_enabled": False,
            "pairwise_materialization_enabled": False,
            "broad_all_to_all_pairwise_enabled": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureArtifactBoundary":
        obj = require_json_object(payload, context="RegimeFeatureArtifactBoundary")
        if "artifact_kind" not in obj and "artifact_family" not in obj:
            raise ValueError("RegimeFeatureArtifactBoundary missing required field: artifact_kind")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            boundary_kind=obj.get("boundary_kind", REGIME_FEATURE_ARTIFACT_BOUNDARY_KIND),
            artifact_kind=obj.get("artifact_kind"),
            artifact_family=obj.get("artifact_family"),
            write_scope=obj.get("write_scope", "foundation_or_sandbox_only"),
            persistent=bool(obj.get("persistent", True)),
            production_enabled=bool(obj.get("production_enabled", False)),
            production_parquet_enabled=bool(obj.get("production_parquet_enabled", False)),
            production_promotion_enabled=bool(obj.get("production_promotion_enabled", False)),
            pairwise_materialization_enabled=bool(obj.get("pairwise_materialization_enabled", False)),
            broad_all_to_all_pairwise_enabled=bool(obj.get("broad_all_to_all_pairwise_enabled", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureArtifactBoundary":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureArtifactBoundary JSON"))


@dataclass(frozen=True)
class RegimeFeatureManifest:
    manifest_id: str
    artifact_kind: RegimeFeatureArtifactKind | str
    family_kind: RegimeFeatureFamilyKind | str
    scope: RegimeFeatureScope | str
    row_key: RegimeFeatureRowKey | Mapping[str, Any]
    artifact_boundary: RegimeFeatureArtifactBoundary | Mapping[str, Any]
    build_status: RegimeFeatureBuildStatus | str = RegimeFeatureBuildStatus.SCAFFOLD_ONLY
    required_columns: Sequence[str] = ()
    feature_columns: Sequence[str] = ()
    source_roots: Mapping[str, Any] = field(default_factory=dict)
    output_root: str | None = None
    clamp_policy_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | RegimeFeatureSchemaVersion = RegimeFeatureSchemaVersion.V1

    def __post_init__(self) -> None:
        artifact_kind = _artifact_kind(self.artifact_kind)
        family_kind = _family_kind(self.family_kind)
        expected_family = ARTIFACT_KIND_TO_FAMILY[artifact_kind]
        if family_kind != expected_family:
            raise ValueError(
                f"Regime Feature manifest family_kind {family_kind.value!r} does not match artifact_kind {artifact_kind.value!r}"
            )
        scope = _scope(self.scope)
        row_key = self.row_key if isinstance(self.row_key, RegimeFeatureRowKey) else RegimeFeatureRowKey.from_dict(self.row_key)
        if row_key.artifact_kind != artifact_kind:
            raise ValueError("Regime Feature manifest row_key artifact_kind must match manifest artifact_kind")
        boundary = (
            self.artifact_boundary
            if isinstance(self.artifact_boundary, RegimeFeatureArtifactBoundary)
            else RegimeFeatureArtifactBoundary.from_dict(self.artifact_boundary)
        )
        if boundary.artifact_kind != artifact_kind:
            raise ValueError("Regime Feature manifest artifact_boundary artifact_kind must match manifest artifact_kind")
        status = _build_status(self.build_status)
        object.__setattr__(self, "schema_version", require_schema_version(int(self.schema_version)))
        object.__setattr__(self, "manifest_id", _text(self.manifest_id, field_name="manifest_id"))
        object.__setattr__(self, "artifact_kind", artifact_kind)
        object.__setattr__(self, "family_kind", family_kind)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "row_key", row_key)
        object.__setattr__(self, "artifact_boundary", boundary)
        object.__setattr__(self, "build_status", status)
        object.__setattr__(self, "required_columns", _string_tuple(self.required_columns, field_name="required_columns", require_non_empty=False))
        object.__setattr__(self, "feature_columns", _string_tuple(self.feature_columns, field_name="feature_columns", require_non_empty=False))
        object.__setattr__(self, "source_roots", to_jsonable(dict(self.source_roots)))
        object.__setattr__(self, "output_root", _optional_text(self.output_root))
        object.__setattr__(self, "clamp_policy_id", _optional_text(self.clamp_policy_id or row_key.clamp_policy_id))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind.value,
            "family_kind": self.family_kind.value,
            "scope": self.scope.value,
            "manifest_id": self.manifest_id,
            "row_key": self.row_key.as_dict(),
            "artifact_boundary": self.artifact_boundary.as_dict(),
            "build_status": self.build_status.value,
            "required_columns": list(self.required_columns),
            "feature_columns": list(self.feature_columns),
            "source_roots": to_jsonable(dict(self.source_roots)),
            "output_root": self.output_root,
            "known_at_ts": self.row_key.known_at_ts,
            "source_tail_ts": self.row_key.source_tail_ts,
            "lineage_id": self.row_key.lineage_id,
            "clamp_policy_id": self.clamp_policy_id,
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureManifest":
        obj = require_json_object(payload, context="RegimeFeatureManifest")
        required = {
            "schema_version",
            "artifact_kind",
            "family_kind",
            "scope",
            "manifest_id",
            "row_key",
            "artifact_boundary",
            "build_status",
        }
        missing = sorted(required.difference(obj))
        if missing:
            raise ValueError(f"RegimeFeatureManifest missing required fields: {', '.join(missing)}")
        return cls(
            schema_version=obj["schema_version"],
            artifact_kind=obj["artifact_kind"],
            family_kind=obj["family_kind"],
            scope=obj["scope"],
            manifest_id=obj["manifest_id"],
            row_key=obj["row_key"],
            artifact_boundary=obj["artifact_boundary"],
            build_status=obj["build_status"],
            required_columns=obj.get("required_columns", ()),
            feature_columns=obj.get("feature_columns", ()),
            source_roots=obj.get("source_roots", {}),
            output_root=obj.get("output_root"),
            clamp_policy_id=obj.get("clamp_policy_id"),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureManifest JSON"))


@dataclass(frozen=True)
class RegimeFeatureOutputSchema:
    artifact_family: str
    required_columns: Sequence[str]
    partition_columns: Sequence[str]
    optional_columns: Sequence[str] = ()
    row_grain: str = "unspecified"
    materialization_status: str = "scaffold_or_primitive"
    persistent: bool = True
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    artifact_kind: str = REGIME_FEATURE_OUTPUT_SCHEMA_ARTIFACT_KIND

    def __post_init__(self) -> None:
        required = _string_tuple(self.required_columns, field_name="required_columns", require_non_empty=True)
        partitions = _string_tuple(self.partition_columns, field_name="partition_columns", require_non_empty=True)
        optional = _string_tuple(self.optional_columns, field_name="optional_columns", require_non_empty=False)
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "artifact_family", _artifact_family(self.artifact_family))
        object.__setattr__(self, "required_columns", required)
        object.__setattr__(self, "partition_columns", partitions)
        object.__setattr__(self, "optional_columns", optional)
        object.__setattr__(self, "row_grain", _text(self.row_grain, field_name="row_grain"))
        object.__setattr__(self, "materialization_status", _text(self.materialization_status, field_name="materialization_status"))

    def validate_columns(self, columns: Sequence[str]) -> None:
        present = {str(column) for column in columns}
        missing = [column for column in self.required_columns if column not in present]
        if missing:
            raise ValueError(f"Regime Feature {self.artifact_family} missing required columns: {missing}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": REGIME_FEATURES_LAYER,
            "artifact_family": self.artifact_family,
            "row_grain": self.row_grain,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "partition_columns": list(self.partition_columns),
            "materialization_status": self.materialization_status,
            "persistent": bool(self.persistent),
            "production_enabled": False,
        }


@dataclass(frozen=True)
class RegimeFeatureConsumerContract:
    consumer_pathway: str
    artifact_family: str
    required_columns: Sequence[str]
    expected_grain: str
    optional_columns: Sequence[str] = ()
    requires_known_at: bool = True
    requires_lineage: bool = True
    requires_universe_snapshot: bool = True
    production_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    artifact_kind: str = REGIME_FEATURE_CONSUMER_CONTRACT_KIND
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pathway = _member(self.consumer_pathway, ("market_state", "relative_state"), field_name="consumer_pathway")
        _require_disabled(self.production_enabled, field_name="production_enabled")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "consumer_pathway", pathway)
        object.__setattr__(self, "artifact_family", _artifact_family(self.artifact_family))
        object.__setattr__(self, "required_columns", _string_tuple(self.required_columns, field_name="required_columns", require_non_empty=True))
        object.__setattr__(self, "optional_columns", _string_tuple(self.optional_columns, field_name="optional_columns", require_non_empty=False))
        object.__setattr__(self, "expected_grain", _text(self.expected_grain, field_name="expected_grain"))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": REGIME_FEATURES_LAYER,
            "consumer_pathway": self.consumer_pathway,
            "artifact_family": self.artifact_family,
            "expected_grain": self.expected_grain,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "requires_known_at": bool(self.requires_known_at),
            "requires_lineage": bool(self.requires_lineage),
            "requires_universe_snapshot": bool(self.requires_universe_snapshot),
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }


def market_regime_feature_output_schema() -> RegimeFeatureOutputSchema:
    return RegimeFeatureOutputSchema(
        artifact_family=MARKET_REGIME_FEATURES,
        required_columns=MARKET_REGIME_FEATURE_REQUIRED_COLUMNS,
        optional_columns=(
            "market_return_equal_weight",
            "market_return_core_equal_weight",
            "market_return_median",
            "market_return_q10",
            "market_return_q90",
            "market_realized_volatility",
            "market_volatility_median",
            "share_assets_up",
            "share_assets_down",
            "return_dispersion_std",
            "return_dispersion_iqr",
            "core_pairwise_corr_median",
            "core_pairwise_corr_status",
            "covariance_summary_status",
            "covariance_trace",
            "aggregate_volume",
            "aggregate_trades",
            "activity_breadth",
            "stress_down_participation",
        ),
        partition_columns=("interval", "band", "year", "month"),
        row_grain="one row per ts/interval/band/universe snapshot",
        materialization_status="primitive_market_features_implemented",
    )


def pairwise_relationship_output_schema() -> RegimeFeatureOutputSchema:
    return RegimeFeatureOutputSchema(
        artifact_family=PAIRWISE_RELATIONSHIP_FEATURES,
        required_columns=PAIRWISE_RELATIONSHIP_REQUIRED_COLUMNS,
        partition_columns=("interval", "relationship_type", "band", "window", "year", "month"),
        row_grain="one row per asset/related asset/timestamp/window/relationship type",
        materialization_status="scaffold_only_materialization_disabled",
    )


def cross_asset_summary_output_schema() -> RegimeFeatureOutputSchema:
    return RegimeFeatureOutputSchema(
        artifact_family=CROSS_ASSET_SUMMARY_FEATURES,
        required_columns=CROSS_ASSET_SUMMARY_REQUIRED_COLUMNS,
        optional_columns=(
            "corr_to_core_basket",
            "beta_to_core_basket",
            "residual_return_vs_market",
            "residual_volatility_vs_market",
            "top_peer_corr_mean",
            "top_peer_corr_max",
            "top_peer_count",
            "peer_cluster_id",
            "distance_to_peer_centroid",
            "relative_strength_percentile",
            "volume_rank_percentile",
            "volatility_rank_percentile",
            "lead_lag_score_vs_peer_group",
            "benchmark_anchor_id",
            "benchmark_anchor_role",
        ),
        partition_columns=("interval", "band", "asset", "year", "month"),
        row_grain="one row per asset/timestamp/interval/band",
        materialization_status="scaffold_only_materialization_disabled",
    )


def universe_snapshot_output_schema() -> dict[str, Any]:
    return {
        "schema_version": int(REGIME_FEATURES_SCHEMA_VERSION),
        "artifact_kind": REGIME_FEATURE_OUTPUT_SCHEMA_ARTIFACT_KIND,
        "layer": REGIME_FEATURES_LAYER,
        "artifact_family": UNIVERSE_SNAPSHOT,
        "required_fields": list(UNIVERSE_SNAPSHOT_REQUIRED_FIELDS),
        "partition_shape": "regime_feature_snapshots/policy_id=<policy_id>/refit_key=<refit_key>/interval=<interval>/band=<band>/snapshot.json",
        "materialization_status": "snapshot_scaffold_implemented",
        "production_enabled": False,
    }


def _artifact_family(value: object) -> str:
    text = _text(value, field_name="artifact_family").lower()
    if text not in REGIME_FEATURE_ARTIFACT_FAMILIES:
        valid = ", ".join(REGIME_FEATURE_ARTIFACT_FAMILIES)
        raise ValueError(f"Unsupported Regime Feature artifact_family {text!r}; expected one of: {valid}")
    return text


def _artifact_kind(value: object) -> RegimeFeatureArtifactKind:
    if isinstance(value, RegimeFeatureArtifactKind):
        return value
    text = _text(value, field_name="artifact_kind").lower()
    for item in RegimeFeatureArtifactKind:
        if text == item.value:
            return item
    legacy_map = {
        MARKET_REGIME_FEATURES: RegimeFeatureArtifactKind.MARKET_FEATURES,
        PAIRWISE_RELATIONSHIP_FEATURES: RegimeFeatureArtifactKind.PAIRWISE_RELATIONSHIP_FEATURES,
        CROSS_ASSET_SUMMARY_FEATURES: RegimeFeatureArtifactKind.CROSS_ASSET_SUMMARY_FEATURES,
        UNIVERSE_SNAPSHOT: RegimeFeatureArtifactKind.UNIVERSE_SNAPSHOT,
        ELIGIBILITY_SNAPSHOT: RegimeFeatureArtifactKind.UNIVERSE_SNAPSHOT,
    }
    if text in legacy_map:
        return legacy_map[text]
    valid = ", ".join(item.value for item in RegimeFeatureArtifactKind)
    raise ValueError(f"Unsupported Regime Feature artifact_kind {text!r}; expected one of: {valid}")


def _family_kind(value: object) -> RegimeFeatureFamilyKind:
    if isinstance(value, RegimeFeatureFamilyKind):
        return value
    text = _text(value, field_name="family_kind").lower()
    for item in RegimeFeatureFamilyKind:
        if text == item.value:
            return item
    legacy_map = {
        MARKET_REGIME_FEATURES: RegimeFeatureFamilyKind.MARKET,
        PAIRWISE_RELATIONSHIP_FEATURES: RegimeFeatureFamilyKind.PAIRWISE_RELATIONSHIP,
        CROSS_ASSET_SUMMARY_FEATURES: RegimeFeatureFamilyKind.CROSS_ASSET_SUMMARY,
        UNIVERSE_SNAPSHOT: RegimeFeatureFamilyKind.UNIVERSE,
        ELIGIBILITY_SNAPSHOT: RegimeFeatureFamilyKind.UNIVERSE,
    }
    if text in legacy_map:
        return legacy_map[text]
    valid = ", ".join(item.value for item in RegimeFeatureFamilyKind)
    raise ValueError(f"Unsupported Regime Feature family_kind {text!r}; expected one of: {valid}")


def _scope(value: object) -> RegimeFeatureScope:
    if isinstance(value, RegimeFeatureScope):
        return value
    text = _text(value, field_name="scope").lower()
    for item in RegimeFeatureScope:
        if text == item.value:
            return item
    valid = ", ".join(item.value for item in RegimeFeatureScope)
    raise ValueError(f"Unsupported Regime Feature scope {text!r}; expected one of: {valid}")


def _build_status(value: object) -> RegimeFeatureBuildStatus:
    if isinstance(value, RegimeFeatureBuildStatus):
        return value
    text = _text(value, field_name="build_status").lower()
    for item in RegimeFeatureBuildStatus:
        if text == item.value:
            return item
    valid = ", ".join(item.value for item in RegimeFeatureBuildStatus)
    raise ValueError(f"Unsupported Regime Feature build_status {text!r}; expected one of: {valid}")


def _coerce_boundary_artifact_kind(
    artifact_kind: str | RegimeFeatureArtifactKind | None,
    artifact_family: str | RegimeFeatureFamilyKind | None,
) -> RegimeFeatureArtifactKind:
    if artifact_kind is not None:
        return _artifact_kind(artifact_kind)
    if artifact_family is None:
        raise ValueError("Regime Feature artifact boundary requires artifact_kind")
    try:
        return _artifact_kind(artifact_family)
    except ValueError:
        pass
    family = _family_kind(artifact_family)
    defaults = {
        RegimeFeatureFamilyKind.MARKET: RegimeFeatureArtifactKind.MARKET_FEATURES,
        RegimeFeatureFamilyKind.PAIRWISE_RELATIONSHIP: RegimeFeatureArtifactKind.PAIRWISE_RELATIONSHIP_FEATURES,
        RegimeFeatureFamilyKind.CROSS_ASSET_SUMMARY: RegimeFeatureArtifactKind.CROSS_ASSET_SUMMARY_FEATURES,
        RegimeFeatureFamilyKind.UNIVERSE: RegimeFeatureArtifactKind.UNIVERSE_SNAPSHOT,
        RegimeFeatureFamilyKind.BASKET: RegimeFeatureArtifactKind.BASKET_SNAPSHOT,
        RegimeFeatureFamilyKind.PEER_PLACEHOLDER: RegimeFeatureArtifactKind.PEER_SNAPSHOT_PLACEHOLDER,
    }
    return defaults[family]


def _coerce_boundary_family_kind(
    artifact_family: str | RegimeFeatureFamilyKind | None,
    artifact_kind: RegimeFeatureArtifactKind,
) -> RegimeFeatureFamilyKind:
    if artifact_family is None:
        return ARTIFACT_KIND_TO_FAMILY[artifact_kind]
    try:
        family = _family_kind(artifact_family)
    except ValueError:
        if _artifact_kind(artifact_family) == artifact_kind:
            return ARTIFACT_KIND_TO_FAMILY[artifact_kind]
        raise
    expected = ARTIFACT_KIND_TO_FAMILY[artifact_kind]
    if family != expected:
        raise ValueError("Regime Feature artifact boundary family does not match artifact_kind")
    return family


def _member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    text = _text(value, field_name=field_name).lower()
    if text not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime Feature {field_name} {text!r}; expected one of: {valid}")
    return text


def _mapping_tuple(values: Sequence[Mapping[str, Any]], *, field_name: str) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature {field_name} must be a sequence of JSON objects")
    out: list[dict[str, Any]] = []
    for value in values:
        if hasattr(value, "as_dict"):
            value = value.as_dict()
        if not isinstance(value, Mapping):
            raise ValueError(f"Regime Feature {field_name} entries must be JSON objects")
        out.append(to_jsonable(dict(value)))
    return tuple(out)


def _string_tuple(values: Sequence[str], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature {field_name} must be a sequence of strings")
    out = tuple(_text(value, field_name=field_name) for value in values)
    if require_non_empty and not out:
        raise ValueError(f"Regime Feature {field_name} must be non-empty")
    return out


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_disabled(value: object, *, field_name: str) -> None:
    if value is not False:
        raise ValueError(f"Regime Feature {field_name} must remain disabled in foundation scaffolding")


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime Feature {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime Feature {field_name} must be numeric or ISO datetime") from exc


def _validate_order(start: object, end: object, *, context: str) -> None:
    if _to_orderable(start, field_name=f"{context} start") > _to_orderable(end, field_name=f"{context} end"):
        raise ValueError(f"Regime Feature {context} start must be <= end")


__all__ = [
    "CROSS_ASSET_SUMMARY_FEATURES",
    "CROSS_ASSET_SUMMARY_REQUIRED_COLUMNS",
    "ELIGIBILITY_SNAPSHOT",
    "MARKET_REGIME_FEATURES",
    "MARKET_REGIME_FEATURE_REQUIRED_COLUMNS",
    "PAIRWISE_RELATIONSHIP_FEATURES",
    "PAIRWISE_RELATIONSHIP_REQUIRED_COLUMNS",
    "REGIME_FEATURES_LAYER",
    "REGIME_FEATURES_SCHEMA_VERSION",
    "REGIME_FEATURE_ARTIFACT_FAMILIES",
    "RegimeFeatureArtifactKind",
    "RegimeFeatureArtifactBoundary",
    "RegimeFeatureBuildStatus",
    "RegimeFeatureConsumerContract",
    "RegimeFeatureFamilyKind",
    "RegimeFeatureKnownAtSpec",
    "RegimeFeatureLineageSpec",
    "RegimeFeatureManifest",
    "RegimeFeatureOutputSchema",
    "RegimeFeatureRowKey",
    "RegimeFeatureSchemaVersion",
    "RegimeFeatureScope",
    "UNIVERSE_SNAPSHOT",
    "UNIVERSE_SNAPSHOT_REQUIRED_FIELDS",
    "cross_asset_summary_output_schema",
    "market_regime_feature_output_schema",
    "pairwise_relationship_output_schema",
    "universe_snapshot_output_schema",
]
