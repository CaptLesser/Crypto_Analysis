from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


RELATIONSHIP_DISCOVERY_LAYER = "relationship_discovery"
RELATIONSHIP_DISCOVERY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION

RELATIONSHIP_METHOD_MANIFEST_ARTIFACT_KIND = "relationship_method_manifest"
RELATIONSHIP_REFIT_SNAPSHOT_ARTIFACT_KIND = "relationship_refit_snapshot"
RELATIONSHIP_EDGE_ARTIFACT_KIND = "relationship_edge"
ASSET_RELATIONSHIP_PROFILE_ARTIFACT_KIND = "asset_relationship_profile"
RELATIONSHIP_STABILITY_SCORE_ARTIFACT_KIND = "relationship_stability_score"
EDGE_ALIAS_MANIFEST_ARTIFACT_KIND = "edge_alias_manifest"
ISOLATED_ASSET_PROFILE_ARTIFACT_KIND = "isolated_asset_profile"
RELATIONSHIP_SCOREBOARD_ARTIFACT_KIND = "relationship_scoreboard"

RELATIONSHIP_EDGE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "interval",
    "window",
    "asset",
    "related_asset_or_benchmark",
    "relationship_type",
    "value",
    "abs_value",
    "direction",
    "sample_count",
    "coverage",
    "method_id",
    "known_at_ts",
    "lineage_id",
    "schema_version",
)
RELATIONSHIP_EDGE_OPTIONAL_COLUMNS: tuple[str, ...] = ("ts", "refit_key", "stability_score")

ASSET_RELATIONSHIP_PROFILE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "asset",
    "interval",
    "window",
    "method_id",
    "corr_to_anchor_primary",
    "corr_to_anchor_secondary",
    "beta_to_core_basket",
    "residual_return_vs_core",
    "residual_volatility_vs_core",
    "top_relationship_strength",
    "relationship_concentration",
    "relationship_entropy",
    "relationship_count_above_threshold",
    "stability_summary",
    "known_at_ts",
    "lineage_id",
    "schema_version",
)

RELATIONSHIP_STABILITY_SCORE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "asset",
    "related_asset_or_benchmark",
    "method_id",
    "interval",
    "window",
    "survival_count",
    "survival_share",
    "mean_strength",
    "strength_std",
    "sign_stability",
    "rank_stability",
    "activation_status",
    "schema_version",
)

EDGE_ALIAS_MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = (
    "refit_key",
    "asset",
    "slot_id",
    "related_asset",
    "method_id",
    "interval",
    "window",
    "selection_rank",
    "strength",
    "stability_status",
    "alias_status",
    "known_at_ts",
    "lineage_id",
    "schema_version",
)

ISOLATED_ASSET_PROFILE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "refit_key",
    "asset",
    "interval",
    "window",
    "isolated_asset_score",
    "peer_signal_availability_status",
    "reason_codes",
    "stable_relationship_count",
    "candidate_relationship_count",
    "known_at_ts",
    "lineage_id",
    "schema_version",
)

PARQUET_SCALAR_TYPES: frozenset[str] = frozenset({"string", "int64", "double", "bool"})


@dataclass(frozen=True)
class RelationshipColumnSpec:
    name: str
    logical_type: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, field_name="column name"))
        logical_type = _text(self.logical_type, field_name="logical_type").lower()
        if logical_type not in PARQUET_SCALAR_TYPES:
            valid = ", ".join(sorted(PARQUET_SCALAR_TYPES))
            raise ValueError(f"Relationship Discovery column logical_type must be one of: {valid}")
        object.__setattr__(self, "logical_type", logical_type)
        object.__setattr__(self, "required", bool(self.required))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "logical_type": self.logical_type, "required": bool(self.required)}


@dataclass(frozen=True)
class RelationshipOutputSchema:
    schema_id: str
    row_grain: str
    columns: Sequence[RelationshipColumnSpec | Mapping[str, Any]]
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = "relationship_output_schema"

    def __post_init__(self) -> None:
        cols = tuple(column if isinstance(column, RelationshipColumnSpec) else RelationshipColumnSpec(**column) for column in self.columns)
        if not cols:
            raise ValueError("Relationship Discovery output schema requires at least one column")
        names = [column.name for column in cols]
        if len(names) != len(set(names)):
            raise ValueError("Relationship Discovery output schema column names must be unique")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "schema_id", _text(self.schema_id, field_name="schema_id"))
        object.__setattr__(self, "row_grain", _text(self.row_grain, field_name="row_grain"))
        object.__setattr__(self, "columns", cols)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if not column.required)

    def validate_columns(self, columns: Sequence[str]) -> None:
        present = {str(column) for column in columns}
        missing = [column for column in self.required_columns if column not in present]
        if missing:
            raise ValueError(f"Relationship Discovery {self.schema_id} missing required columns: {missing}")

    def validate_row(self, row: Mapping[str, Any]) -> None:
        obj = require_json_object(row, context=f"Relationship Discovery {self.schema_id} row")
        self.validate_columns(tuple(obj))
        for column in self.columns:
            if column.name not in obj:
                continue
            if column.required:
                _validate_required_value(obj[column.name], logical_type=column.logical_type, column_name=column.name)
            _validate_scalar_type(obj[column.name], logical_type=column.logical_type, column_name=column.name)
        if "known_at_ts" in obj and "source_tail_ts" in obj:
            if _to_orderable(obj["source_tail_ts"], field_name="source_tail_ts") > _to_orderable(obj["known_at_ts"], field_name="known_at_ts"):
                raise ValueError(f"Relationship Discovery {self.schema_id} source_tail_ts must not exceed known_at_ts")

    def validate_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("Relationship Discovery rows must be a sequence of JSON objects")
        for row in rows:
            self.validate_row(row)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "schema_id": self.schema_id,
            "row_grain": self.row_grain,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "columns": [column.as_dict() for column in self.columns],
            "parquet_compatible": True,
            "production_enabled": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class RelationshipMethodManifest:
    method_id: str
    method_family: str
    source_data: str
    interval: int
    rolling_window: int
    universe_scope: str
    residualization_policy: str
    normalization_policy: str
    min_observations: int
    min_coverage: float
    known_at_policy: str
    production_enabled: bool = False
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = RELATIONSHIP_METHOD_MANIFEST_ARTIFACT_KIND

    def __post_init__(self) -> None:
        interval = _positive_int(self.interval, field_name="interval")
        rolling_window = _positive_int(self.rolling_window, field_name="rolling_window")
        min_observations = _positive_int(self.min_observations, field_name="min_observations")
        if min_observations > rolling_window:
            raise ValueError("Relationship Discovery min_observations must be <= rolling_window")
        _require_disabled(self.production_enabled, field_name="production_enabled")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "method_id", _text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_family", _text(self.method_family, field_name="method_family").lower())
        object.__setattr__(self, "source_data", _text(self.source_data, field_name="source_data").lower())
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "rolling_window", rolling_window)
        object.__setattr__(self, "universe_scope", _text(self.universe_scope, field_name="universe_scope").lower())
        object.__setattr__(self, "residualization_policy", _text(self.residualization_policy, field_name="residualization_policy"))
        object.__setattr__(self, "normalization_policy", _text(self.normalization_policy, field_name="normalization_policy"))
        object.__setattr__(self, "min_observations", min_observations)
        object.__setattr__(self, "min_coverage", _share(self.min_coverage, field_name="min_coverage"))
        object.__setattr__(self, "known_at_policy", _text(self.known_at_policy, field_name="known_at_policy"))
        object.__setattr__(self, "production_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "method_id": self.method_id,
            "method_family": self.method_family,
            "source_data": self.source_data,
            "interval": int(self.interval),
            "rolling_window": int(self.rolling_window),
            "universe_scope": self.universe_scope,
            "residualization_policy": self.residualization_policy,
            "normalization_policy": self.normalization_policy,
            "min_observations": int(self.min_observations),
            "min_coverage": float(self.min_coverage),
            "known_at_policy": self.known_at_policy,
            "production_enabled": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationshipMethodManifest":
        obj = require_json_object(payload, context="RelationshipMethodManifest")
        return cls(
            schema_version=obj.get("schema_version", RELATIONSHIP_DISCOVERY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", RELATIONSHIP_METHOD_MANIFEST_ARTIFACT_KIND),
            method_id=obj["method_id"],
            method_family=obj["method_family"],
            source_data=obj["source_data"],
            interval=obj["interval"],
            rolling_window=obj["rolling_window"],
            universe_scope=obj["universe_scope"],
            residualization_policy=obj["residualization_policy"],
            normalization_policy=obj["normalization_policy"],
            min_observations=obj["min_observations"],
            min_coverage=obj["min_coverage"],
            known_at_policy=obj["known_at_policy"],
            production_enabled=bool(obj.get("production_enabled", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "RelationshipMethodManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="RelationshipMethodManifest JSON"))


@dataclass(frozen=True)
class RelationshipRefitSnapshot:
    refit_key: str
    snapshot_start: int | float | str
    snapshot_end: int | float | str
    known_at_ts: int | float | str
    eligible_assets: Sequence[str]
    anchors: Sequence[str]
    core_assets: Sequence[str]
    broad_sample_assets: Sequence[str]
    excluded_assets_with_reasons: Mapping[str, Any] = field(default_factory=dict)
    universe_manifest_ref: str | None = None
    universe_manifest_hash: str | None = None
    source_tail_ts: int | float | str | None = None
    production_enabled: bool = False
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = RELATIONSHIP_REFIT_SNAPSHOT_ARTIFACT_KIND

    def __post_init__(self) -> None:
        _validate_order(self.snapshot_start, self.snapshot_end, context="snapshot")
        known_at = _to_orderable(self.known_at_ts, field_name="known_at_ts")
        if _to_orderable(self.snapshot_end, field_name="snapshot_end") > known_at:
            raise ValueError("Relationship Discovery snapshot_end must not exceed known_at_ts")
        source_tail = self.source_tail_ts if self.source_tail_ts is not None else self.snapshot_end
        if _to_orderable(source_tail, field_name="source_tail_ts") > known_at:
            raise ValueError("Relationship Discovery source_tail_ts must not exceed known_at_ts")
        eligible = _string_tuple(self.eligible_assets, field_name="eligible_assets", require_non_empty=True)
        anchors = _string_tuple(self.anchors, field_name="anchors", require_non_empty=True)
        core_assets = _string_tuple(self.core_assets, field_name="core_assets", require_non_empty=True)
        broad_sample_assets = _string_tuple(self.broad_sample_assets, field_name="broad_sample_assets", require_non_empty=False)
        _require_disabled(self.production_enabled, field_name="production_enabled")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "eligible_assets", eligible)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "core_assets", core_assets)
        object.__setattr__(self, "broad_sample_assets", broad_sample_assets)
        object.__setattr__(self, "excluded_assets_with_reasons", to_jsonable(dict(self.excluded_assets_with_reasons)))
        object.__setattr__(self, "universe_manifest_ref", _optional_text(self.universe_manifest_ref))
        object.__setattr__(self, "universe_manifest_hash", _optional_text(self.universe_manifest_hash))
        object.__setattr__(self, "source_tail_ts", source_tail)
        object.__setattr__(self, "production_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "refit_key": self.refit_key,
            "snapshot_start": self.snapshot_start,
            "snapshot_end": self.snapshot_end,
            "known_at_ts": self.known_at_ts,
            "eligible_assets": list(self.eligible_assets),
            "anchors": list(self.anchors),
            "core_assets": list(self.core_assets),
            "broad_sample_assets": list(self.broad_sample_assets),
            "excluded_assets_with_reasons": dict(self.excluded_assets_with_reasons),
            "universe_manifest_ref": self.universe_manifest_ref,
            "universe_manifest_hash": self.universe_manifest_hash,
            "source_tail_ts": self.source_tail_ts,
            "production_enabled": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationshipRefitSnapshot":
        obj = require_json_object(payload, context="RelationshipRefitSnapshot")
        return cls(
            schema_version=obj.get("schema_version", RELATIONSHIP_DISCOVERY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", RELATIONSHIP_REFIT_SNAPSHOT_ARTIFACT_KIND),
            refit_key=obj["refit_key"],
            snapshot_start=obj["snapshot_start"],
            snapshot_end=obj["snapshot_end"],
            known_at_ts=obj["known_at_ts"],
            eligible_assets=obj["eligible_assets"],
            anchors=obj["anchors"],
            core_assets=obj["core_assets"],
            broad_sample_assets=obj.get("broad_sample_assets", ()),
            excluded_assets_with_reasons=obj.get("excluded_assets_with_reasons", {}),
            universe_manifest_ref=obj.get("universe_manifest_ref"),
            universe_manifest_hash=obj.get("universe_manifest_hash"),
            source_tail_ts=obj.get("source_tail_ts"),
            production_enabled=bool(obj.get("production_enabled", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "RelationshipRefitSnapshot":
        return cls.from_dict(require_json_object(loads_json(text), context="RelationshipRefitSnapshot JSON"))


@dataclass(frozen=True)
class RelationshipEdge:
    interval: int
    window: int
    asset: str
    related_asset_or_benchmark: str
    relationship_type: str
    value: float
    abs_value: float | None
    direction: str
    sample_count: int
    coverage: float
    method_id: str
    known_at_ts: int | float | str
    lineage_id: str
    ts: int | float | str | None = None
    refit_key: str | None = None
    stability_score: float | None = None
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = RELATIONSHIP_EDGE_ARTIFACT_KIND

    def __post_init__(self) -> None:
        if self.ts is None and _optional_text(self.refit_key) is None:
            raise ValueError("Relationship Discovery edge requires ts or refit_key")
        value = _finite_float(self.value, field_name="value")
        abs_value = abs(value) if self.abs_value is None else _nonnegative_float(self.abs_value, field_name="abs_value")
        if abs(abs_value - abs(value)) > 1e-12:
            raise ValueError("Relationship Discovery edge abs_value must equal abs(value)")
        direction = _direction(self.direction, sign_value=value)
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "related_asset_or_benchmark", _text(self.related_asset_or_benchmark, field_name="related_asset_or_benchmark"))
        object.__setattr__(self, "relationship_type", _text(self.relationship_type, field_name="relationship_type").lower())
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "abs_value", abs_value)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "sample_count", _nonnegative_int(self.sample_count, field_name="sample_count"))
        object.__setattr__(self, "coverage", _share(self.coverage, field_name="coverage"))
        object.__setattr__(self, "method_id", _text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        object.__setattr__(self, "refit_key", _optional_text(self.refit_key))
        if self.ts is not None:
            _to_orderable(self.ts, field_name="ts")
        _to_orderable(self.known_at_ts, field_name="known_at_ts")
        object.__setattr__(self, "stability_score", None if self.stability_score is None else _share(self.stability_score, field_name="stability_score"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "ts": self.ts,
            "refit_key": self.refit_key,
            "interval": int(self.interval),
            "window": int(self.window),
            "asset": self.asset,
            "related_asset_or_benchmark": self.related_asset_or_benchmark,
            "relationship_type": self.relationship_type,
            "value": float(self.value),
            "abs_value": float(self.abs_value),
            "direction": self.direction,
            "sample_count": int(self.sample_count),
            "coverage": float(self.coverage),
            "stability_score": self.stability_score,
            "method_id": self.method_id,
            "known_at_ts": self.known_at_ts,
            "lineage_id": self.lineage_id,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationshipEdge":
        obj = require_json_object(payload, context="RelationshipEdge")
        return cls(
            schema_version=obj.get("schema_version", RELATIONSHIP_DISCOVERY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", RELATIONSHIP_EDGE_ARTIFACT_KIND),
            ts=obj.get("ts"),
            refit_key=obj.get("refit_key"),
            interval=obj["interval"],
            window=obj["window"],
            asset=obj["asset"],
            related_asset_or_benchmark=obj["related_asset_or_benchmark"],
            relationship_type=obj["relationship_type"],
            value=obj["value"],
            abs_value=obj.get("abs_value"),
            direction=obj["direction"],
            sample_count=obj["sample_count"],
            coverage=obj["coverage"],
            stability_score=obj.get("stability_score"),
            method_id=obj["method_id"],
            known_at_ts=obj["known_at_ts"],
            lineage_id=obj["lineage_id"],
        )

    @classmethod
    def from_json(cls, text: str) -> "RelationshipEdge":
        return cls.from_dict(require_json_object(loads_json(text), context="RelationshipEdge JSON"))


@dataclass(frozen=True)
class AssetRelationshipProfile:
    asset: str
    interval: int
    window: int
    method_id: str
    corr_to_anchor_primary: float
    corr_to_anchor_secondary: float
    beta_to_core_basket: float
    residual_return_vs_core: float
    residual_volatility_vs_core: float
    top_relationship_strength: float
    relationship_concentration: float
    relationship_entropy: float
    relationship_count_above_threshold: int
    stability_summary: Mapping[str, Any] | str
    known_at_ts: int | float | str
    lineage_id: str
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = ASSET_RELATIONSHIP_PROFILE_ARTIFACT_KIND

    def __post_init__(self) -> None:
        stability_summary = (
            self.stability_summary
            if isinstance(self.stability_summary, str)
            else dumps_json(to_jsonable(dict(self.stability_summary)), separators=(",", ":"))
        )
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "method_id", _text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "corr_to_anchor_primary", _correlation(self.corr_to_anchor_primary, field_name="corr_to_anchor_primary"))
        object.__setattr__(self, "corr_to_anchor_secondary", _correlation(self.corr_to_anchor_secondary, field_name="corr_to_anchor_secondary"))
        object.__setattr__(self, "beta_to_core_basket", _finite_float(self.beta_to_core_basket, field_name="beta_to_core_basket"))
        object.__setattr__(self, "residual_return_vs_core", _finite_float(self.residual_return_vs_core, field_name="residual_return_vs_core"))
        object.__setattr__(self, "residual_volatility_vs_core", _nonnegative_float(self.residual_volatility_vs_core, field_name="residual_volatility_vs_core"))
        object.__setattr__(self, "top_relationship_strength", _share(self.top_relationship_strength, field_name="top_relationship_strength"))
        object.__setattr__(self, "relationship_concentration", _share(self.relationship_concentration, field_name="relationship_concentration"))
        object.__setattr__(self, "relationship_entropy", _nonnegative_float(self.relationship_entropy, field_name="relationship_entropy"))
        object.__setattr__(
            self,
            "relationship_count_above_threshold",
            _nonnegative_int(self.relationship_count_above_threshold, field_name="relationship_count_above_threshold"),
        )
        object.__setattr__(self, "stability_summary", _text(stability_summary, field_name="stability_summary"))
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        _to_orderable(self.known_at_ts, field_name="known_at_ts")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "asset": self.asset,
            "interval": int(self.interval),
            "window": int(self.window),
            "method_id": self.method_id,
            "corr_to_anchor_primary": float(self.corr_to_anchor_primary),
            "corr_to_anchor_secondary": float(self.corr_to_anchor_secondary),
            "beta_to_core_basket": float(self.beta_to_core_basket),
            "residual_return_vs_core": float(self.residual_return_vs_core),
            "residual_volatility_vs_core": float(self.residual_volatility_vs_core),
            "top_relationship_strength": float(self.top_relationship_strength),
            "relationship_concentration": float(self.relationship_concentration),
            "relationship_entropy": float(self.relationship_entropy),
            "relationship_count_above_threshold": int(self.relationship_count_above_threshold),
            "stability_summary": self.stability_summary,
            "known_at_ts": self.known_at_ts,
            "lineage_id": self.lineage_id,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetRelationshipProfile":
        obj = require_json_object(payload, context="AssetRelationshipProfile")
        return cls(
            schema_version=obj.get("schema_version", RELATIONSHIP_DISCOVERY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", ASSET_RELATIONSHIP_PROFILE_ARTIFACT_KIND),
            asset=obj["asset"],
            interval=obj["interval"],
            window=obj["window"],
            method_id=obj["method_id"],
            corr_to_anchor_primary=obj["corr_to_anchor_primary"],
            corr_to_anchor_secondary=obj["corr_to_anchor_secondary"],
            beta_to_core_basket=obj["beta_to_core_basket"],
            residual_return_vs_core=obj["residual_return_vs_core"],
            residual_volatility_vs_core=obj["residual_volatility_vs_core"],
            top_relationship_strength=obj["top_relationship_strength"],
            relationship_concentration=obj["relationship_concentration"],
            relationship_entropy=obj["relationship_entropy"],
            relationship_count_above_threshold=obj["relationship_count_above_threshold"],
            stability_summary=obj["stability_summary"],
            known_at_ts=obj["known_at_ts"],
            lineage_id=obj["lineage_id"],
        )

    @classmethod
    def from_json(cls, text: str) -> "AssetRelationshipProfile":
        return cls.from_dict(require_json_object(loads_json(text), context="AssetRelationshipProfile JSON"))


@dataclass(frozen=True)
class RelationshipStabilityScore:
    asset: str
    related_asset_or_benchmark: str
    method_id: str
    interval: int
    window: int
    survival_count: int
    survival_share: float
    mean_strength: float
    strength_std: float
    sign_stability: float
    rank_stability: float
    activation_status: str
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = RELATIONSHIP_STABILITY_SCORE_ARTIFACT_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "related_asset_or_benchmark", _text(self.related_asset_or_benchmark, field_name="related_asset_or_benchmark"))
        object.__setattr__(self, "method_id", _text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "survival_count", _nonnegative_int(self.survival_count, field_name="survival_count"))
        object.__setattr__(self, "survival_share", _share(self.survival_share, field_name="survival_share"))
        object.__setattr__(self, "mean_strength", _finite_float(self.mean_strength, field_name="mean_strength"))
        object.__setattr__(self, "strength_std", _nonnegative_float(self.strength_std, field_name="strength_std"))
        object.__setattr__(self, "sign_stability", _share(self.sign_stability, field_name="sign_stability"))
        object.__setattr__(self, "rank_stability", _share(self.rank_stability, field_name="rank_stability"))
        object.__setattr__(self, "activation_status", _text(self.activation_status, field_name="activation_status").lower())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "asset": self.asset,
            "related_asset_or_benchmark": self.related_asset_or_benchmark,
            "method_id": self.method_id,
            "interval": int(self.interval),
            "window": int(self.window),
            "survival_count": int(self.survival_count),
            "survival_share": float(self.survival_share),
            "mean_strength": float(self.mean_strength),
            "strength_std": float(self.strength_std),
            "sign_stability": float(self.sign_stability),
            "rank_stability": float(self.rank_stability),
            "activation_status": self.activation_status,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationshipStabilityScore":
        obj = require_json_object(payload, context="RelationshipStabilityScore")
        return cls(
            schema_version=obj.get("schema_version", RELATIONSHIP_DISCOVERY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", RELATIONSHIP_STABILITY_SCORE_ARTIFACT_KIND),
            asset=obj["asset"],
            related_asset_or_benchmark=obj["related_asset_or_benchmark"],
            method_id=obj["method_id"],
            interval=obj["interval"],
            window=obj["window"],
            survival_count=obj["survival_count"],
            survival_share=obj["survival_share"],
            mean_strength=obj["mean_strength"],
            strength_std=obj["strength_std"],
            sign_stability=obj["sign_stability"],
            rank_stability=obj["rank_stability"],
            activation_status=obj["activation_status"],
        )

    @classmethod
    def from_json(cls, text: str) -> "RelationshipStabilityScore":
        return cls.from_dict(require_json_object(loads_json(text), context="RelationshipStabilityScore JSON"))


@dataclass(frozen=True)
class EdgeAliasManifestRow:
    refit_key: str
    asset: str
    slot_id: str
    related_asset: str
    method_id: str
    interval: int
    window: int
    selection_rank: int
    strength: float
    stability_status: str
    alias_status: str
    known_at_ts: int | float | str
    lineage_id: str
    selected_edge_ref: str | None = None
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = EDGE_ALIAS_MANIFEST_ARTIFACT_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "slot_id", _text(self.slot_id, field_name="slot_id"))
        object.__setattr__(self, "related_asset", _text(self.related_asset, field_name="related_asset"))
        object.__setattr__(self, "method_id", _text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "selection_rank", _positive_int(self.selection_rank, field_name="selection_rank"))
        object.__setattr__(self, "strength", _finite_float(self.strength, field_name="strength"))
        object.__setattr__(self, "stability_status", _text(self.stability_status, field_name="stability_status").lower())
        object.__setattr__(self, "alias_status", _text(self.alias_status, field_name="alias_status").lower())
        _to_orderable(self.known_at_ts, field_name="known_at_ts")
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        object.__setattr__(self, "selected_edge_ref", _optional_text(self.selected_edge_ref))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "refit_key": self.refit_key,
            "asset": self.asset,
            "slot_id": self.slot_id,
            "related_asset": self.related_asset,
            "method_id": self.method_id,
            "interval": int(self.interval),
            "window": int(self.window),
            "selection_rank": int(self.selection_rank),
            "strength": float(self.strength),
            "stability_status": self.stability_status,
            "alias_status": self.alias_status,
            "known_at_ts": self.known_at_ts,
            "lineage_id": self.lineage_id,
            "selected_edge_ref": self.selected_edge_ref,
            "production_enabled": False,
            "final_peer_membership_claimed": False,
        }


@dataclass(frozen=True)
class IsolatedAssetProfile:
    refit_key: str
    asset: str
    interval: int
    window: int
    isolated_asset_score: float
    peer_signal_availability_status: str
    reason_codes: Sequence[str] = ()
    stable_relationship_count: int = 0
    candidate_relationship_count: int = 0
    known_at_ts: int | float | str = 0
    lineage_id: str = "unknown_lineage"
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = ISOLATED_ASSET_PROFILE_ARTIFACT_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "isolated_asset_score", _share(self.isolated_asset_score, field_name="isolated_asset_score"))
        object.__setattr__(self, "peer_signal_availability_status", _text(self.peer_signal_availability_status, field_name="peer_signal_availability_status").lower())
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(str(reason) for reason in self.reason_codes if str(reason).strip())))
        object.__setattr__(self, "stable_relationship_count", _nonnegative_int(self.stable_relationship_count, field_name="stable_relationship_count"))
        object.__setattr__(self, "candidate_relationship_count", _nonnegative_int(self.candidate_relationship_count, field_name="candidate_relationship_count"))
        _to_orderable(self.known_at_ts, field_name="known_at_ts")
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "refit_key": self.refit_key,
            "asset": self.asset,
            "interval": int(self.interval),
            "window": int(self.window),
            "isolated_asset_score": float(self.isolated_asset_score),
            "peer_signal_availability_status": self.peer_signal_availability_status,
            "reason_codes": list(self.reason_codes),
            "stable_relationship_count": int(self.stable_relationship_count),
            "candidate_relationship_count": int(self.candidate_relationship_count),
            "known_at_ts": self.known_at_ts,
            "lineage_id": self.lineage_id,
            "production_enabled": False,
        }


@dataclass(frozen=True)
class RelationshipScoreboard:
    refit_key: str
    status: str
    selected_edge_count: int
    candidate_edge_count: int
    isolated_asset_count: int
    unstable_asset_count: int
    intervals: Sequence[int]
    windows: Sequence[int]
    artifact_paths: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = RELATIONSHIP_SCOREBOARD_ARTIFACT_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "status", _text(self.status, field_name="status").lower())
        object.__setattr__(self, "selected_edge_count", _nonnegative_int(self.selected_edge_count, field_name="selected_edge_count"))
        object.__setattr__(self, "candidate_edge_count", _nonnegative_int(self.candidate_edge_count, field_name="candidate_edge_count"))
        object.__setattr__(self, "isolated_asset_count", _nonnegative_int(self.isolated_asset_count, field_name="isolated_asset_count"))
        object.__setattr__(self, "unstable_asset_count", _nonnegative_int(self.unstable_asset_count, field_name="unstable_asset_count"))
        object.__setattr__(self, "intervals", tuple(_positive_int(value, field_name="interval") for value in self.intervals))
        object.__setattr__(self, "windows", tuple(_positive_int(value, field_name="window") for value in self.windows))
        object.__setattr__(self, "artifact_paths", to_jsonable(dict(self.artifact_paths)))
        object.__setattr__(self, "diagnostics", to_jsonable(dict(self.diagnostics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": RELATIONSHIP_DISCOVERY_LAYER,
            "refit_key": self.refit_key,
            "status": self.status,
            "selected_edge_count": int(self.selected_edge_count),
            "candidate_edge_count": int(self.candidate_edge_count),
            "isolated_asset_count": int(self.isolated_asset_count),
            "unstable_asset_count": int(self.unstable_asset_count),
            "intervals": [int(value) for value in self.intervals],
            "windows": [int(value) for value in self.windows],
            "artifact_paths": to_jsonable(dict(self.artifact_paths)),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "production_enabled": False,
        }


def relationship_edge_output_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id="relationship_edge",
        row_grain="one row per asset/related asset/refit or timestamp/window/method",
        columns=(
            RelationshipColumnSpec("ts", "string", required=False),
            RelationshipColumnSpec("refit_key", "string", required=False),
            RelationshipColumnSpec("interval", "int64"),
            RelationshipColumnSpec("window", "int64"),
            RelationshipColumnSpec("asset", "string"),
            RelationshipColumnSpec("related_asset_or_benchmark", "string"),
            RelationshipColumnSpec("relationship_type", "string"),
            RelationshipColumnSpec("value", "double"),
            RelationshipColumnSpec("abs_value", "double"),
            RelationshipColumnSpec("direction", "string"),
            RelationshipColumnSpec("sample_count", "int64"),
            RelationshipColumnSpec("coverage", "double"),
            RelationshipColumnSpec("stability_score", "double", required=False),
            RelationshipColumnSpec("method_id", "string"),
            RelationshipColumnSpec("known_at_ts", "string"),
            RelationshipColumnSpec("lineage_id", "string"),
            RelationshipColumnSpec("schema_version", "int64"),
        ),
    )


def asset_relationship_profile_output_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id="asset_relationship_profile",
        row_grain="one row per asset/refit or timestamp/window/method",
        columns=(
            RelationshipColumnSpec("asset", "string"),
            RelationshipColumnSpec("interval", "int64"),
            RelationshipColumnSpec("window", "int64"),
            RelationshipColumnSpec("method_id", "string"),
            RelationshipColumnSpec("corr_to_anchor_primary", "double"),
            RelationshipColumnSpec("corr_to_anchor_secondary", "double"),
            RelationshipColumnSpec("beta_to_core_basket", "double"),
            RelationshipColumnSpec("residual_return_vs_core", "double"),
            RelationshipColumnSpec("residual_volatility_vs_core", "double"),
            RelationshipColumnSpec("top_relationship_strength", "double"),
            RelationshipColumnSpec("relationship_concentration", "double"),
            RelationshipColumnSpec("relationship_entropy", "double"),
            RelationshipColumnSpec("relationship_count_above_threshold", "int64"),
            RelationshipColumnSpec("stability_summary", "string"),
            RelationshipColumnSpec("known_at_ts", "string"),
            RelationshipColumnSpec("lineage_id", "string"),
            RelationshipColumnSpec("schema_version", "int64"),
        ),
    )


def relationship_stability_score_output_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id="relationship_stability_score",
        row_grain="one row per asset/related asset/window/method stability summary",
        columns=(
            RelationshipColumnSpec("asset", "string"),
            RelationshipColumnSpec("related_asset_or_benchmark", "string"),
            RelationshipColumnSpec("method_id", "string"),
            RelationshipColumnSpec("interval", "int64"),
            RelationshipColumnSpec("window", "int64"),
            RelationshipColumnSpec("survival_count", "int64"),
            RelationshipColumnSpec("survival_share", "double"),
            RelationshipColumnSpec("mean_strength", "double"),
            RelationshipColumnSpec("strength_std", "double"),
            RelationshipColumnSpec("sign_stability", "double"),
            RelationshipColumnSpec("rank_stability", "double"),
            RelationshipColumnSpec("activation_status", "string"),
            RelationshipColumnSpec("schema_version", "int64"),
        ),
    )


def edge_alias_manifest_output_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id="edge_alias_manifest",
        row_grain="one row per refit/asset/slot selected residual peer alias",
        columns=tuple(RelationshipColumnSpec(column, "string") for column in EDGE_ALIAS_MANIFEST_REQUIRED_COLUMNS),
    )


def isolated_asset_profile_output_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id="isolated_asset_profile",
        row_grain="one row per refit/asset/interval/window isolation status",
        columns=(
            RelationshipColumnSpec("refit_key", "string"),
            RelationshipColumnSpec("asset", "string"),
            RelationshipColumnSpec("interval", "int64"),
            RelationshipColumnSpec("window", "int64"),
            RelationshipColumnSpec("isolated_asset_score", "double"),
            RelationshipColumnSpec("peer_signal_availability_status", "string"),
            RelationshipColumnSpec("reason_codes", "string"),
            RelationshipColumnSpec("stable_relationship_count", "int64"),
            RelationshipColumnSpec("candidate_relationship_count", "int64"),
            RelationshipColumnSpec("known_at_ts", "string"),
            RelationshipColumnSpec("lineage_id", "string"),
            RelationshipColumnSpec("schema_version", "int64"),
        ),
    )


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Relationship Discovery {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(values: Sequence[object], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Relationship Discovery {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not out:
        raise ValueError(f"Relationship Discovery {field_name} must include at least one value")
    return out


def _positive_int(value: object, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Relationship Discovery {field_name} must be positive")
    return out


def _nonnegative_int(value: object, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be an integer") from exc
    if out < 0:
        raise ValueError(f"Relationship Discovery {field_name} must be non-negative")
    return out


def _finite_float(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be numeric") from exc
    if out != out or out in {float("inf"), float("-inf")}:
        raise ValueError(f"Relationship Discovery {field_name} must be finite")
    return out


def _nonnegative_float(value: object, *, field_name: str) -> float:
    out = _finite_float(value, field_name=field_name)
    if out < 0:
        raise ValueError(f"Relationship Discovery {field_name} must be non-negative")
    return out


def _share(value: object, *, field_name: str) -> float:
    out = _finite_float(value, field_name=field_name)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"Relationship Discovery {field_name} must be between 0 and 1")
    return out


def _correlation(value: object, *, field_name: str) -> float:
    out = _finite_float(value, field_name=field_name)
    if out < -1.0 or out > 1.0:
        raise ValueError(f"Relationship Discovery {field_name} must be between -1 and 1")
    return out


def _direction(direction: object, *, sign_value: float) -> str:
    text = _text(direction, field_name="direction").lower()
    if text not in {"positive", "negative", "neutral"}:
        raise ValueError("Relationship Discovery direction must be positive, negative, or neutral")
    expected = "positive" if sign_value > 0 else "negative" if sign_value < 0 else "neutral"
    if text != expected:
        raise ValueError("Relationship Discovery edge direction must match value sign")
    return text


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Relationship Discovery {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be numeric or ISO datetime") from exc


def _validate_order(start: object, end: object, *, context: str) -> None:
    if _to_orderable(start, field_name=f"{context} start") > _to_orderable(end, field_name=f"{context} end"):
        raise ValueError(f"Relationship Discovery {context} start must be <= end")


def _require_disabled(value: object, *, field_name: str) -> None:
    if bool(value):
        raise ValueError(f"Relationship Discovery {field_name} must remain disabled in prototype contracts")


def _validate_scalar_type(value: Any, *, logical_type: str, column_name: str) -> None:
    if value is None:
        return
    if logical_type == "string":
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"Relationship Discovery column {column_name} must be string-compatible")
        return
    if logical_type == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"Relationship Discovery column {column_name} must be bool")
        return
    if logical_type == "int64":
        if isinstance(value, bool):
            raise ValueError(f"Relationship Discovery column {column_name} must be int64")
        int(value)
        return
    if logical_type == "double":
        _finite_float(value, field_name=column_name)
        return
    raise ValueError(f"Unsupported Relationship Discovery logical_type {logical_type!r}")


def _validate_required_value(value: Any, *, logical_type: str, column_name: str) -> None:
    if value is None:
        raise ValueError(f"Relationship Discovery required column {column_name} must be non-null")
    if column_name in {"known_at_ts", "source_tail_ts", "lineage_id"} and not str(value).strip():
        raise ValueError(f"Relationship Discovery required column {column_name} must be non-empty")


__all__ = [
    "ASSET_RELATIONSHIP_PROFILE_ARTIFACT_KIND",
    "ASSET_RELATIONSHIP_PROFILE_REQUIRED_COLUMNS",
    "EDGE_ALIAS_MANIFEST_ARTIFACT_KIND",
    "EDGE_ALIAS_MANIFEST_REQUIRED_COLUMNS",
    "ISOLATED_ASSET_PROFILE_ARTIFACT_KIND",
    "ISOLATED_ASSET_PROFILE_REQUIRED_COLUMNS",
    "RELATIONSHIP_DISCOVERY_LAYER",
    "RELATIONSHIP_DISCOVERY_SCHEMA_VERSION",
    "RELATIONSHIP_EDGE_ARTIFACT_KIND",
    "RELATIONSHIP_EDGE_OPTIONAL_COLUMNS",
    "RELATIONSHIP_EDGE_REQUIRED_COLUMNS",
    "RELATIONSHIP_METHOD_MANIFEST_ARTIFACT_KIND",
    "RELATIONSHIP_REFIT_SNAPSHOT_ARTIFACT_KIND",
    "RELATIONSHIP_SCOREBOARD_ARTIFACT_KIND",
    "RELATIONSHIP_STABILITY_SCORE_ARTIFACT_KIND",
    "RELATIONSHIP_STABILITY_SCORE_REQUIRED_COLUMNS",
    "AssetRelationshipProfile",
    "EdgeAliasManifestRow",
    "IsolatedAssetProfile",
    "RelationshipColumnSpec",
    "RelationshipEdge",
    "RelationshipMethodManifest",
    "RelationshipOutputSchema",
    "RelationshipRefitSnapshot",
    "RelationshipScoreboard",
    "RelationshipStabilityScore",
    "asset_relationship_profile_output_schema",
    "edge_alias_manifest_output_schema",
    "isolated_asset_profile_output_schema",
    "relationship_edge_output_schema",
    "relationship_stability_score_output_schema",
]
