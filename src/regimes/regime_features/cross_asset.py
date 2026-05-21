"""Legacy gated Cross-Asset scaffold.

The active v1 Cross-Asset Feature Handoff generator lives in
``src.regimes.regime_features.cross_asset_feature_generator`` and consumes
canonical Process 1 Relationship Discovery artifacts. This module remains a
compatibility surface for older summary placeholders and must stay disabled by
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.regime_features.contracts import (
    CROSS_ASSET_SUMMARY_FEATURES,
    CROSS_ASSET_SUMMARY_REQUIRED_COLUMNS,
    REGIME_FEATURES_SCHEMA_VERSION,
    RegimeFeatureArtifactBoundary,
    cross_asset_summary_output_schema,
)


STATUS_AVAILABLE = "available"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED = "not_available_peer_discovery_required"
STATUS_PLACEHOLDER_PEER_CLUSTERING_REQUIRED = "not_available_peer_clustering_required"
CROSS_ASSET_EXECUTION_DISABLED = "cross_asset_execution_disabled"
PEER_DISCOVERY_NOT_IMPLEMENTED = "peer_discovery_not_implemented"
CROSS_ASSET_CLUSTERING_DISABLED = "cross_asset_clustering_disabled"
CROSS_ASSET_LEGACY_SURFACE_STATUS = "legacy_scaffold_only"
CROSS_ASSET_CANONICAL_GENERATOR_MODULE = "src.regimes.regime_features.cross_asset_feature_generator"

CROSS_ASSET_SUMMARY_FEATURE_COLUMNS: tuple[str, ...] = (
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
)

CROSS_ASSET_SUMMARY_PLACEHOLDERS: tuple[str, ...] = CROSS_ASSET_SUMMARY_FEATURE_COLUMNS

CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS: tuple[str, ...] = (
    "top_peer_corr_mean",
    "top_peer_corr_max",
    "top_peer_count",
    "peer_cluster_id",
    "distance_to_peer_centroid",
    "lead_lag_score_vs_peer_group",
)

CROSS_ASSET_PRIMITIVE_SUMMARY_COLUMNS: tuple[str, ...] = tuple(
    column for column in CROSS_ASSET_SUMMARY_FEATURE_COLUMNS if column not in CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS
)

CROSS_ASSET_BENCHMARK_METADATA_COLUMNS: tuple[str, ...] = (
    "benchmark_anchor_id",
    "benchmark_anchor_role",
)


@dataclass(frozen=True)
class CrossAssetSummaryFeatureSpec:
    name: str
    dtype: str
    nullable: bool = True
    status: str = STATUS_NOT_AVAILABLE
    requires_peer_discovery: bool = False
    requires_peer_clustering: bool = False
    requires_benchmark_anchor: bool = False
    description: str = ""
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = _member(self.name, CROSS_ASSET_SUMMARY_FEATURE_COLUMNS, field_name="feature name")
        status = _member(
            self.status,
            (
                STATUS_AVAILABLE,
                STATUS_NOT_AVAILABLE,
                STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED,
                STATUS_PLACEHOLDER_PEER_CLUSTERING_REQUIRED,
            ),
            field_name="feature status",
        )
        requires_peer_discovery = bool(self.requires_peer_discovery)
        requires_peer_clustering = bool(self.requires_peer_clustering)
        if name in CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS:
            requires_peer_discovery = True
            if name in {"peer_cluster_id", "distance_to_peer_centroid"}:
                requires_peer_clustering = True
            if status == STATUS_AVAILABLE:
                raise ValueError("Regime Feature cross-asset peer placeholder fields cannot be marked available before peer discovery")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", _text(self.dtype, field_name="dtype"))
        object.__setattr__(self, "nullable", bool(self.nullable))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "requires_peer_discovery", requires_peer_discovery)
        object.__setattr__(self, "requires_peer_clustering", requires_peer_clustering)
        object.__setattr__(self, "requires_benchmark_anchor", bool(self.requires_benchmark_anchor))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "cross_asset_summary_feature_spec",
            "name": self.name,
            "dtype": self.dtype,
            "nullable": bool(self.nullable),
            "status": self.status,
            "requires_peer_discovery": bool(self.requires_peer_discovery),
            "requires_peer_clustering": bool(self.requires_peer_clustering),
            "requires_benchmark_anchor": bool(self.requires_benchmark_anchor),
            "description": self.description,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossAssetSummaryFeatureSpec":
        obj = require_json_object(payload, context="CrossAssetSummaryFeatureSpec")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            name=obj["name"],
            dtype=obj["dtype"],
            nullable=bool(obj.get("nullable", True)),
            status=obj.get("status", STATUS_NOT_AVAILABLE),
            requires_peer_discovery=bool(obj.get("requires_peer_discovery", False)),
            requires_peer_clustering=bool(obj.get("requires_peer_clustering", False)),
            requires_benchmark_anchor=bool(obj.get("requires_benchmark_anchor", False)),
            description=obj.get("description", ""),
        )

    @classmethod
    def from_json(cls, text: str) -> "CrossAssetSummaryFeatureSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="CrossAssetSummaryFeatureSpec JSON"))


@dataclass(frozen=True)
class CrossAssetSummarySchema:
    schema_id: str = "cross_asset_summary_schema_v1"
    feature_specs: Sequence[CrossAssetSummaryFeatureSpec | Mapping[str, Any]] = field(default_factory=tuple)
    benchmark_anchor_columns: Sequence[str] = CROSS_ASSET_BENCHMARK_METADATA_COLUMNS
    peer_discovery_enabled: bool = False
    cross_asset_clustering_enabled: bool = False
    production_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_disabled(self.peer_discovery_enabled, field_name="peer_discovery_enabled")
        _require_disabled(self.cross_asset_clustering_enabled, field_name="cross_asset_clustering_enabled")
        _require_disabled(self.production_enabled, field_name="production_enabled")
        specs = tuple(
            spec if isinstance(spec, CrossAssetSummaryFeatureSpec) else CrossAssetSummaryFeatureSpec.from_dict(spec)
            for spec in self.feature_specs
        )
        if not specs:
            specs = default_cross_asset_summary_feature_specs()
        names = tuple(spec.name for spec in specs)
        missing = [column for column in CROSS_ASSET_SUMMARY_FEATURE_COLUMNS if column not in names]
        if missing:
            raise ValueError(f"Regime Feature cross-asset summary schema missing feature columns: {missing}")
        if len(set(names)) != len(names):
            raise ValueError("Regime Feature cross-asset summary schema feature names must be unique")
        object.__setattr__(self, "schema_id", _text(self.schema_id, field_name="schema_id"))
        object.__setattr__(self, "feature_specs", tuple(sorted(specs, key=lambda spec: CROSS_ASSET_SUMMARY_FEATURE_COLUMNS.index(spec.name))))
        object.__setattr__(self, "benchmark_anchor_columns", _string_tuple(self.benchmark_anchor_columns, field_name="benchmark_anchor_columns", require_non_empty=False))
        object.__setattr__(self, "peer_discovery_enabled", False)
        object.__setattr__(self, "cross_asset_clustering_enabled", False)
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(CROSS_ASSET_SUMMARY_REQUIRED_COLUMNS)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.feature_specs)

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.required_columns, *self.feature_columns, *self.benchmark_anchor_columns)

    def validate_columns(self, columns: Sequence[str]) -> None:
        present = {str(column) for column in columns}
        missing = [column for column in self.required_columns if column not in present]
        if missing:
            raise ValueError(f"Regime Feature cross-asset summary missing required columns: {missing}")
        wide_peer_columns = [
            column
            for column in present
            if column.startswith(("corr_to_", "beta_to_", "distance_to_"))
            and column not in set(CROSS_ASSET_SUMMARY_FEATURE_COLUMNS)
        ]
        if wide_peer_columns:
            raise ValueError(f"Regime Feature cross-asset summary forbids one-column-per-related-asset output: {wide_peer_columns}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "cross_asset_summary_schema",
            "schema_id": self.schema_id,
            "artifact_family": CROSS_ASSET_SUMMARY_FEATURES,
            "required_columns": list(self.required_columns),
            "feature_columns": list(self.feature_columns),
            "columns": list(self.columns),
            "feature_specs": [spec.as_dict() for spec in self.feature_specs],
            "peer_placeholder_columns": list(CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS),
            "primitive_summary_columns": list(CROSS_ASSET_PRIMITIVE_SUMMARY_COLUMNS),
            "benchmark_anchor_columns": list(self.benchmark_anchor_columns),
            "blocked_statuses": [PEER_DISCOVERY_NOT_IMPLEMENTED],
            "benchmark_anchors_metadata_only": True,
            "compact_summary_only": True,
            "one_column_per_related_asset_allowed": False,
            "execution_enabled": False,
            "peer_discovery_enabled": False,
            "peer_discovery_performed": False,
            "clustering_enabled": False,
            "cross_asset_clustering_enabled": False,
            "cross_asset_clustering_performed": False,
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossAssetSummarySchema":
        obj = require_json_object(payload, context="CrossAssetSummarySchema")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            schema_id=obj.get("schema_id", "cross_asset_summary_schema_v1"),
            feature_specs=obj.get("feature_specs", ()),
            benchmark_anchor_columns=obj.get("benchmark_anchor_columns", CROSS_ASSET_BENCHMARK_METADATA_COLUMNS),
            peer_discovery_enabled=bool(obj.get("peer_discovery_enabled", False)),
            cross_asset_clustering_enabled=bool(obj.get("cross_asset_clustering_enabled", False)),
            production_enabled=bool(obj.get("production_enabled", False)),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "CrossAssetSummarySchema":
        return cls.from_dict(require_json_object(loads_json(text), context="CrossAssetSummarySchema JSON"))


@dataclass(frozen=True)
class CrossAssetSummaryRow:
    ts: int | float | str
    asset: str
    interval: int
    band: str
    universe_policy_id: str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    lineage_id: str
    feature_values: Mapping[str, Any] = field(default_factory=dict)
    benchmark_anchor_id: str | None = None
    benchmark_anchor_role: str | None = None
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature cross-asset summary row interval must be positive")
        values = _default_feature_values()
        values.update(dict(self.feature_values))
        unknown = sorted(set(values).difference(CROSS_ASSET_SUMMARY_FEATURE_COLUMNS))
        if unknown:
            raise ValueError(f"Regime Feature cross-asset summary row unknown feature values: {unknown}")
        for column in CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS:
            if values.get(column) is not None:
                raise ValueError(f"Regime Feature cross-asset peer placeholder {column!r} must remain null until peer discovery exists")
        object.__setattr__(self, "ts", _orderable(self.ts, field_name="ts"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "universe_policy_id", _text(self.universe_policy_id, field_name="universe_policy_id"))
        object.__setattr__(self, "known_at_ts", _orderable(self.known_at_ts, field_name="known_at_ts"))
        object.__setattr__(self, "source_tail_ts", _orderable(self.source_tail_ts, field_name="source_tail_ts"))
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        object.__setattr__(self, "feature_values", to_jsonable(values))
        object.__setattr__(self, "benchmark_anchor_id", None if self.benchmark_anchor_id is None else _text(self.benchmark_anchor_id, field_name="benchmark_anchor_id"))
        object.__setattr__(self, "benchmark_anchor_role", None if self.benchmark_anchor_role is None else _text(self.benchmark_anchor_role, field_name="benchmark_anchor_role"))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "cross_asset_summary_row",
            "ts": self.ts,
            "asset": self.asset,
            "interval": int(self.interval),
            "band": self.band,
            "universe_policy_id": self.universe_policy_id,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "lineage_id": self.lineage_id,
            **to_jsonable(dict(self.feature_values)),
            "benchmark_anchor_id": self.benchmark_anchor_id,
            "benchmark_anchor_role": self.benchmark_anchor_role,
            "peer_discovery_performed": False,
            "cross_asset_clustering_performed": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossAssetSummaryRow":
        obj = require_json_object(payload, context="CrossAssetSummaryRow")
        feature_values = {column: obj.get(column) for column in CROSS_ASSET_SUMMARY_FEATURE_COLUMNS}
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            ts=obj["ts"],
            asset=obj["asset"],
            interval=obj["interval"],
            band=obj["band"],
            universe_policy_id=obj["universe_policy_id"],
            known_at_ts=obj["known_at_ts"],
            source_tail_ts=obj["source_tail_ts"],
            lineage_id=obj["lineage_id"],
            feature_values=feature_values,
            benchmark_anchor_id=obj.get("benchmark_anchor_id"),
            benchmark_anchor_role=obj.get("benchmark_anchor_role"),
        )

    @classmethod
    def from_json(cls, text: str) -> "CrossAssetSummaryRow":
        return cls.from_dict(require_json_object(loads_json(text), context="CrossAssetSummaryRow JSON"))


@dataclass(frozen=True)
class CrossAssetSummaryScaffold:
    scaffold_id: str = "cross_asset_summary_scaffold_v1"
    placeholder_features: Sequence[str] = CROSS_ASSET_SUMMARY_PLACEHOLDERS
    schema: CrossAssetSummarySchema | Mapping[str, Any] | None = None
    fixed_benchmark_anchors: Sequence[str] = ()
    execution_enabled: bool = False
    peer_discovery_enabled: bool = False
    peer_cluster_enabled: bool = False
    broad_pairwise_dependency_enabled: bool = False
    production_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = _string_tuple(self.placeholder_features, field_name="placeholder_features", require_non_empty=True)
        missing = [column for column in CROSS_ASSET_SUMMARY_FEATURE_COLUMNS if column not in features]
        if missing:
            raise ValueError(f"Regime Feature cross-asset scaffold missing placeholder features: {missing}")
        _require_disabled(self.execution_enabled, field_name="execution_enabled", status=CROSS_ASSET_EXECUTION_DISABLED)
        _require_disabled(self.peer_discovery_enabled, field_name="peer_discovery_enabled", status=PEER_DISCOVERY_NOT_IMPLEMENTED)
        _require_disabled(self.peer_cluster_enabled, field_name="peer_cluster_enabled", status=PEER_DISCOVERY_NOT_IMPLEMENTED)
        _require_disabled(self.broad_pairwise_dependency_enabled, field_name="broad_pairwise_dependency_enabled", status=CROSS_ASSET_EXECUTION_DISABLED)
        _require_disabled(self.production_enabled, field_name="production_enabled")
        schema = self.schema
        if schema is None:
            schema = CrossAssetSummarySchema()
        elif not isinstance(schema, CrossAssetSummarySchema):
            schema = CrossAssetSummarySchema.from_dict(schema)
        object.__setattr__(self, "scaffold_id", _text(self.scaffold_id, field_name="scaffold_id"))
        object.__setattr__(self, "placeholder_features", features)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "fixed_benchmark_anchors", _string_tuple(self.fixed_benchmark_anchors, field_name="fixed_benchmark_anchors", require_non_empty=False))
        object.__setattr__(self, "execution_enabled", False)
        object.__setattr__(self, "peer_discovery_enabled", False)
        object.__setattr__(self, "peer_cluster_enabled", False)
        object.__setattr__(self, "broad_pairwise_dependency_enabled", False)
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def blocked_statuses(self) -> tuple[str, ...]:
        return (
            CROSS_ASSET_EXECUTION_DISABLED,
            PEER_DISCOVERY_NOT_IMPLEMENTED,
            CROSS_ASSET_CLUSTERING_DISABLED,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "cross_asset_summary_feature_scaffold",
            "scaffold_id": self.scaffold_id,
            "artifact_family": CROSS_ASSET_SUMMARY_FEATURES,
            "placeholder_features": list(self.placeholder_features),
            "peer_placeholder_features": list(CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS),
            "primitive_summary_features": list(CROSS_ASSET_PRIMITIVE_SUMMARY_COLUMNS),
            "output_schema": cross_asset_summary_output_schema().as_dict(),
            "summary_schema": self.schema.as_dict(),
            "artifact_boundary": RegimeFeatureArtifactBoundary(artifact_family=CROSS_ASSET_SUMMARY_FEATURES).as_dict(),
            "fixed_benchmark_anchors": list(self.fixed_benchmark_anchors),
            "fixed_benchmark_anchors_metadata_only": True,
            "permanent_peer_assumptions": [],
            "blocked_statuses": list(self.blocked_statuses),
            "blocked_reasons": list(self.blocked_statuses),
            "compact_summary_only": True,
            "one_column_per_related_asset_allowed": False,
            "execution_enabled": False,
            "peer_discovery_enabled": False,
            "peer_discovery_performed": False,
            "peer_cluster_enabled": False,
            "clustering_enabled": False,
            "cross_asset_clustering_enabled": False,
            "cross_asset_clustering_performed": False,
            "broad_pairwise_dependency_enabled": False,
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossAssetSummaryScaffold":
        obj = require_json_object(payload, context="CrossAssetSummaryScaffold")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            scaffold_id=obj["scaffold_id"],
            placeholder_features=obj.get("placeholder_features", CROSS_ASSET_SUMMARY_PLACEHOLDERS),
            schema=obj.get("summary_schema", obj.get("schema")),
            fixed_benchmark_anchors=obj.get("fixed_benchmark_anchors", ()),
            execution_enabled=bool(obj.get("execution_enabled", False)),
            peer_discovery_enabled=bool(obj.get("peer_discovery_enabled", False)),
            peer_cluster_enabled=bool(obj.get("peer_cluster_enabled", False)),
            broad_pairwise_dependency_enabled=bool(obj.get("broad_pairwise_dependency_enabled", False)),
            production_enabled=bool(obj.get("production_enabled", False)),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "CrossAssetSummaryScaffold":
        return cls.from_dict(require_json_object(loads_json(text), context="CrossAssetSummaryScaffold JSON"))


def default_cross_asset_summary_feature_specs() -> tuple[CrossAssetSummaryFeatureSpec, ...]:
    primitive_status = STATUS_NOT_AVAILABLE
    return (
        CrossAssetSummaryFeatureSpec("corr_to_core_basket", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("beta_to_core_basket", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("residual_return_vs_market", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("residual_volatility_vs_market", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("top_peer_corr_mean", "float64", status=STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED),
        CrossAssetSummaryFeatureSpec("top_peer_corr_max", "float64", status=STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED),
        CrossAssetSummaryFeatureSpec("top_peer_count", "int64", status=STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED),
        CrossAssetSummaryFeatureSpec("peer_cluster_id", "string", status=STATUS_PLACEHOLDER_PEER_CLUSTERING_REQUIRED),
        CrossAssetSummaryFeatureSpec("distance_to_peer_centroid", "float64", status=STATUS_PLACEHOLDER_PEER_CLUSTERING_REQUIRED),
        CrossAssetSummaryFeatureSpec("relative_strength_percentile", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("volume_rank_percentile", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("volatility_rank_percentile", "float64", status=primitive_status),
        CrossAssetSummaryFeatureSpec("lead_lag_score_vs_peer_group", "float64", status=STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED),
    )


def default_cross_asset_summary_schema() -> CrossAssetSummarySchema:
    return CrossAssetSummarySchema(feature_specs=default_cross_asset_summary_feature_specs())


def default_cross_asset_summary_scaffold() -> CrossAssetSummaryScaffold:
    return CrossAssetSummaryScaffold(schema=default_cross_asset_summary_schema())


def cross_asset_legacy_scaffold_status() -> dict[str, Any]:
    """Return the legacy Cross-Asset routing contract without enabling execution."""

    return {
        "artifact_kind": "cross_asset_legacy_scaffold_status",
        "surface_status": CROSS_ASSET_LEGACY_SURFACE_STATUS,
        "canonical_generator_module": CROSS_ASSET_CANONICAL_GENERATOR_MODULE,
        "scaffold_only": True,
        "wraps_canonical_generator": False,
        "execution_enabled": False,
        "peer_discovery_enabled": False,
        "cross_asset_clustering_enabled": False,
        "production_enabled": False,
        "production_writer_exposed": False,
        "duplicate_writer_path_exposed": False,
    }


def require_cross_asset_execution_enabled(scaffold: CrossAssetSummaryScaffold | Mapping[str, Any]) -> None:
    resolved = scaffold if isinstance(scaffold, CrossAssetSummaryScaffold) else CrossAssetSummaryScaffold.from_dict(scaffold)
    raise ValueError(
        f"{CROSS_ASSET_EXECUTION_DISABLED}: Cross-Asset execution is disabled; "
        f"blocked statuses: {', '.join(resolved.blocked_statuses)}"
    )


def _default_feature_values() -> dict[str, Any]:
    return {column: None for column in CROSS_ASSET_SUMMARY_FEATURE_COLUMNS}


def _string_tuple(values: Sequence[str], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature cross-asset {field_name} must be a sequence of strings")
    out = tuple(_text(value, field_name=field_name) for value in values)
    if require_non_empty and not out:
        raise ValueError(f"Regime Feature cross-asset {field_name} must be non-empty")
    return out


def _member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if text not in allowed:
        valid = ", ".join(allowed)
        raise ValueError(f"Unsupported Regime Feature cross-asset {field_name} {text!r}; expected one of: {valid}")
    return text


def _orderable(value: Any, *, field_name: str) -> int | float | str:
    if value is None:
        raise ValueError(f"Regime Feature cross-asset {field_name} must be non-empty")
    return value


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature cross-asset {field_name} must be non-empty")
    return text


def _require_disabled(value: object, *, field_name: str, status: str | None = None) -> None:
    if value is not False:
        prefix = f"{status}: " if status else ""
        raise ValueError(f"{prefix}Regime Feature cross-asset {field_name} must remain disabled in this sprint")


__all__ = [
    "CROSS_ASSET_BENCHMARK_METADATA_COLUMNS",
    "CROSS_ASSET_CLUSTERING_DISABLED",
    "CROSS_ASSET_CANONICAL_GENERATOR_MODULE",
    "CROSS_ASSET_EXECUTION_DISABLED",
    "CROSS_ASSET_LEGACY_SURFACE_STATUS",
    "CROSS_ASSET_PEER_PLACEHOLDER_COLUMNS",
    "CROSS_ASSET_PRIMITIVE_SUMMARY_COLUMNS",
    "CROSS_ASSET_SUMMARY_FEATURE_COLUMNS",
    "CROSS_ASSET_SUMMARY_PLACEHOLDERS",
    "STATUS_AVAILABLE",
    "STATUS_NOT_AVAILABLE",
    "STATUS_PLACEHOLDER_PEER_CLUSTERING_REQUIRED",
    "STATUS_PLACEHOLDER_PEER_DISCOVERY_REQUIRED",
    "CrossAssetSummaryFeatureSpec",
    "CrossAssetSummaryRow",
    "CrossAssetSummaryScaffold",
    "CrossAssetSummarySchema",
    "default_cross_asset_summary_feature_specs",
    "default_cross_asset_summary_scaffold",
    "default_cross_asset_summary_schema",
    "cross_asset_legacy_scaffold_status",
    "require_cross_asset_execution_enabled",
]
