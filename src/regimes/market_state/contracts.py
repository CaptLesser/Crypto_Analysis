from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION
from src.regimes.core.serialization import dumps_json, loads_json, require_known_fields, require_json_object, to_jsonable


class MarketStateSchemaVersion(IntEnum):
    V1 = CANONICAL_SCHEMA_VERSION


MARKET_STATE_SCHEMA_VERSION = int(MarketStateSchemaVersion.V1)


class MarketStatePathway(str, Enum):
    MARKET_STATE = "market_state"


class MarketStateAxis(str, Enum):
    MARKET_RETURN = "market_return"
    MARKET_VOLATILITY = "market_volatility"
    MARKET_BREADTH = "market_breadth"
    MARKET_DISPERSION = "market_dispersion"
    MARKET_CORRELATION = "market_correlation"
    MARKET_LIQUIDITY_ACTIVITY = "market_liquidity_activity"
    MARKET_STRESS = "market_stress"


class MarketStateBand(str, Enum):
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


MARKET_STATE_AXIS_VALUES: tuple[str, ...] = tuple(axis.value for axis in MarketStateAxis)
MARKET_STATE_BAND_VALUES: tuple[str, ...] = tuple(band.value for band in MarketStateBand)
MARKET_STATE_PATHWAY_VALUE = MarketStatePathway.MARKET_STATE.value
MARKET_STATE_LINEAGE_REQUIRED_FIELDS: tuple[str, ...] = (
    "pathway",
    "axis",
    "band",
    "interval",
    "profile_id",
    "feature_family_id",
    "clusterer_family",
    "schema_version",
    "source_data_kind",
    "source_partition_lineage",
    "source_tail_ts",
    "train_window_start",
    "train_window_end",
    "score_window_start",
    "score_window_end",
    "generated_at",
    "run_id",
)
MARKET_STATE_KNOWN_AT_REQUIRED_FIELDS: tuple[str, ...] = (
    "ts",
    "known_at_ts",
    "source_tail_ts",
    "label_available_at_ts",
    "alignment_policy",
    "latency_policy",
    "no_lookahead_verified",
)


def _schema_version(value: object) -> int:
    try:
        version = int(value)
    except Exception as exc:
        raise ValueError(f"Market-state schema_version must be {MARKET_STATE_SCHEMA_VERSION}") from exc
    if version != MARKET_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported market-state schema_version {version!r}; expected {MARKET_STATE_SCHEMA_VERSION}")
    return version


def _non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Market-state {field_name} must be non-empty")
    return text


def _enum_value(value: object, enum_type: type[Enum], *, field_name: str) -> str:
    raw = value.value if isinstance(value, Enum) else value
    text = _non_empty_text(raw, field_name=field_name).lower()
    valid = tuple(str(member.value) for member in enum_type)
    if text not in valid:
        raise ValueError(f"Unsupported market-state {field_name} {text!r}; expected one of: {', '.join(valid)}")
    return text


def _string_tuple(values: Sequence[object] | None, *, field_name: str, require_non_empty: bool = False) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Market-state {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not out:
        raise ValueError(f"Market-state {field_name} must include at least one value")
    return tuple(dict.fromkeys(out))


def _mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Market-state {field_name} must be a JSON object")
    return dict(value)


def default_market_state_lineage_requirements() -> dict[str, Any]:
    return {
        "contract": "RegimeLineageSpec",
        "required": True,
        "pathway": MARKET_STATE_PATHWAY_VALUE,
        "required_fields": list(MARKET_STATE_LINEAGE_REQUIRED_FIELDS),
        "source_partition_lineage_required": True,
        "source_tail_ts_required": True,
        "market_level_only": True,
    }


def default_market_state_known_at_requirements() -> dict[str, Any]:
    return {
        "contract": "KnownAtSpec",
        "required": True,
        "required_fields": list(MARKET_STATE_KNOWN_AT_REQUIRED_FIELDS),
        "known_at_ts_required": True,
        "source_tail_must_not_exceed_known_at": True,
        "no_lookahead_verified_required": True,
    }


def _requirements_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
    default: Mapping[str, Any],
    contract: str,
) -> dict[str, Any]:
    obj = _mapping(value if value is not None else default, field_name=field_name)
    if not obj:
        raise ValueError(f"Market-state {field_name} must declare requirements")
    if _non_empty_text(obj.get("contract", ""), field_name=f"{field_name}.contract") != contract:
        raise ValueError(f"Market-state {field_name} must reference {contract}")
    if obj.get("required") is not True:
        raise ValueError(f"Market-state {field_name} must be required")
    required_fields = _string_tuple(obj.get("required_fields"), field_name=f"{field_name}.required_fields", require_non_empty=True)
    normalized = dict(obj)
    normalized["contract"] = contract
    normalized["required"] = True
    normalized["required_fields"] = list(required_fields)
    return to_jsonable(normalized)


@dataclass(frozen=True)
class MarketStateArtifactBoundary:
    classification: str = "sandbox_non_production"
    artifact_kind: str = "market_state_artifact_boundary"
    metadata_only: bool = True
    production_output: bool = False
    production_parquet_allowed: bool = False
    production_regime_labels_allowed: bool = False
    production_profile_promotion_allowed: bool = False
    asset_level_labels_allowed: bool = False
    relative_cross_asset_execution_allowed: bool = False
    clustering_enabled: bool = False
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        classification = _non_empty_text(self.classification, field_name="classification").lower()
        if classification in {"production", "prod"}:
            raise ValueError("Market-state artifact boundary cannot use production classification")
        blocking_flags = {
            "production_output": self.production_output,
            "production_parquet_allowed": self.production_parquet_allowed,
            "production_regime_labels_allowed": self.production_regime_labels_allowed,
            "production_profile_promotion_allowed": self.production_profile_promotion_allowed,
            "asset_level_labels_allowed": self.asset_level_labels_allowed,
            "relative_cross_asset_execution_allowed": self.relative_cross_asset_execution_allowed,
            "clustering_enabled": self.clustering_enabled,
        }
        enabled = [name for name, flag in blocking_flags.items() if bool(flag)]
        if enabled:
            raise ValueError(f"Market-state artifact boundary must remain non-production; enabled={enabled}")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "metadata_only", bool(self.metadata_only))
        object.__setattr__(self, "production_output", False)
        object.__setattr__(self, "production_parquet_allowed", False)
        object.__setattr__(self, "production_regime_labels_allowed", False)
        object.__setattr__(self, "production_profile_promotion_allowed", False)
        object.__setattr__(self, "asset_level_labels_allowed", False)
        object.__setattr__(self, "relative_cross_asset_execution_allowed", False)
        object.__setattr__(self, "clustering_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": MARKET_STATE_PATHWAY_VALUE,
            "classification": self.classification,
            "metadata_only": bool(self.metadata_only),
            "market_level_labels_only": True,
            "asset_level_labels_allowed": False,
            "relative_cross_asset_execution_allowed": False,
            "clustering_enabled": False,
            "production_output": False,
            "production_parquet_allowed": False,
            "production_regime_labels_allowed": False,
            "production_profile_promotion_allowed": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateArtifactBoundary":
        obj = require_json_object(payload, context="MarketStateArtifactBoundary")
        obj.pop("pathway", None)
        obj.pop("market_level_labels_only", None)
        return cls(
            schema_version=obj.get("schema_version", MARKET_STATE_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", "market_state_artifact_boundary"),
            classification=obj.get("classification", "sandbox_non_production"),
            metadata_only=obj.get("metadata_only", True),
            production_output=obj.get("production_output", False),
            production_parquet_allowed=obj.get("production_parquet_allowed", False),
            production_regime_labels_allowed=obj.get("production_regime_labels_allowed", False),
            production_profile_promotion_allowed=obj.get("production_profile_promotion_allowed", False),
            asset_level_labels_allowed=obj.get("asset_level_labels_allowed", False),
            relative_cross_asset_execution_allowed=obj.get("relative_cross_asset_execution_allowed", False),
            clustering_enabled=obj.get("clustering_enabled", False),
        )


@dataclass(frozen=True)
class MarketStateBandSpec:
    band: str | MarketStateBand
    ceiling_interval_min: int
    member_intervals: Sequence[int]
    train_days: int
    validation_horizons_min: Sequence[int]
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        band = _enum_value(self.band, MarketStateBand, field_name="band")
        intervals = tuple(int(value) for value in self.member_intervals)
        horizons = tuple(int(value) for value in self.validation_horizons_min)
        if not intervals or any(value <= 0 for value in intervals):
            raise ValueError("Market-state band member_intervals must be positive and non-empty")
        if int(self.ceiling_interval_min) <= 0:
            raise ValueError("Market-state band ceiling_interval_min must be positive")
        if int(self.train_days) <= 0:
            raise ValueError("Market-state band train_days must be positive")
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("Market-state band validation_horizons_min must be positive and non-empty")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "ceiling_interval_min", int(self.ceiling_interval_min))
        object.__setattr__(self, "member_intervals", intervals)
        object.__setattr__(self, "train_days", int(self.train_days))
        object.__setattr__(self, "validation_horizons_min", horizons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "member_intervals": list(self.member_intervals),
            "train_days": int(self.train_days),
            "validation_horizons_min": list(self.validation_horizons_min),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateBandSpec":
        obj = require_known_fields(
            payload,
            required={"schema_version", "band", "ceiling_interval_min", "member_intervals", "train_days", "validation_horizons_min"},
            optional=set(),
            context="MarketStateBandSpec",
        )
        return cls(**obj)


@dataclass(frozen=True)
class MarketStateAxisSpec:
    axis_id: str | MarketStateAxis
    purpose: str
    compatible_bands: Sequence[str | MarketStateBand]
    expected_feature_families: Sequence[str]
    expected_validation_targets: Sequence[str]
    allow_single_state_output: bool
    requires_covariance_correlation_features: bool
    requires_broad_universe_aggregates: bool
    requires_core_basket_features: bool
    label_set: Sequence[str] = ()
    lineage_requirements: Mapping[str, Any] | None = None
    known_at_requirements: Mapping[str, Any] | None = None
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        axis_id = _enum_value(self.axis_id, MarketStateAxis, field_name="axis_id")
        purpose = _non_empty_text(self.purpose, field_name="axis purpose")
        bands = tuple(_enum_value(value, MarketStateBand, field_name="compatible_bands") for value in self.compatible_bands)
        families = _string_tuple(self.expected_feature_families, field_name="expected_feature_families", require_non_empty=True)
        targets = _string_tuple(self.expected_validation_targets, field_name="expected_validation_targets", require_non_empty=True)
        labels = _string_tuple(self.label_set, field_name="label_set")
        lineage_requirements = _requirements_mapping(
            self.lineage_requirements,
            field_name="lineage_requirements",
            default=default_market_state_lineage_requirements(),
            contract="RegimeLineageSpec",
        )
        known_at_requirements = _requirements_mapping(
            self.known_at_requirements,
            field_name="known_at_requirements",
            default=default_market_state_known_at_requirements(),
            contract="KnownAtSpec",
        )
        if not bands:
            raise ValueError("Market-state axis compatible_bands must include at least one band")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "axis_id", axis_id)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "compatible_bands", tuple(dict.fromkeys(bands)))
        object.__setattr__(self, "expected_feature_families", families)
        object.__setattr__(self, "expected_validation_targets", targets)
        object.__setattr__(self, "allow_single_state_output", bool(self.allow_single_state_output))
        object.__setattr__(
            self,
            "requires_covariance_correlation_features",
            bool(self.requires_covariance_correlation_features),
        )
        object.__setattr__(self, "requires_broad_universe_aggregates", bool(self.requires_broad_universe_aggregates))
        object.__setattr__(self, "requires_core_basket_features", bool(self.requires_core_basket_features))
        object.__setattr__(self, "label_set", labels)
        object.__setattr__(self, "lineage_requirements", lineage_requirements)
        object.__setattr__(self, "known_at_requirements", known_at_requirements)

    def supports_band(self, band: str | MarketStateBand) -> bool:
        return _enum_value(band, MarketStateBand, field_name="band") in self.compatible_bands

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "axis_id": self.axis_id,
            "purpose": self.purpose,
            "compatible_bands": list(self.compatible_bands),
            "expected_feature_families": list(self.expected_feature_families),
            "expected_validation_targets": list(self.expected_validation_targets),
            "allow_single_state_output": bool(self.allow_single_state_output),
            "requires_covariance_correlation_features": bool(self.requires_covariance_correlation_features),
            "requires_broad_universe_aggregates": bool(self.requires_broad_universe_aggregates),
            "requires_core_basket_features": bool(self.requires_core_basket_features),
            "label_set": list(self.label_set),
            "lineage_requirements": to_jsonable(self.lineage_requirements),
            "known_at_requirements": to_jsonable(self.known_at_requirements),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateAxisSpec":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "axis_id",
                "purpose",
                "compatible_bands",
                "expected_feature_families",
                "expected_validation_targets",
                "allow_single_state_output",
                "requires_covariance_correlation_features",
                "requires_broad_universe_aggregates",
                "requires_core_basket_features",
            },
            optional={"label_set", "lineage_requirements", "known_at_requirements"},
            context="MarketStateAxisSpec",
        )
        return cls(**obj)


@dataclass(frozen=True)
class MarketStateStudyKey:
    study_id: str
    axis: str | MarketStateAxis
    band: str | MarketStateBand
    pathway: str | MarketStatePathway = MarketStatePathway.MARKET_STATE
    classification: str = "sandbox"
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        pathway = _enum_value(self.pathway, MarketStatePathway, field_name="pathway")
        axis = _enum_value(self.axis, MarketStateAxis, field_name="axis")
        band = _enum_value(self.band, MarketStateBand, field_name="band")
        classification = _non_empty_text(self.classification, field_name="classification").lower()
        if classification in {"production", "prod"}:
            raise ValueError("Market-state study keys cannot use production classification")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "study_id", _non_empty_text(self.study_id, field_name="study_id"))
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "classification", classification)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_id": self.study_id,
            "pathway": self.pathway,
            "axis": self.axis,
            "band": self.band,
            "classification": self.classification,
            "market_level_scope": "whole_market",
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateStudyKey":
        obj = require_known_fields(
            payload,
            required={"schema_version", "study_id", "pathway", "axis", "band", "classification"},
            optional={"market_level_scope"},
            context="MarketStateStudyKey",
        )
        obj.pop("market_level_scope", None)
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "MarketStateStudyKey":
        return cls.from_dict(require_json_object(loads_json(text), context="MarketStateStudyKey JSON"))


@dataclass(frozen=True)
class MarketStateOutputSchema:
    key_columns: Sequence[str]
    partition_columns: Sequence[str]
    state_columns: Sequence[str]
    diagnostic_columns: Sequence[str]
    artifact_boundary: MarketStateArtifactBoundary | Mapping[str, Any] = field(default_factory=MarketStateArtifactBoundary)
    pathway: str | MarketStatePathway = MarketStatePathway.MARKET_STATE
    classification: str = "sandbox_non_production"
    artifact_kind: str = "market_state_output_schema"
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        boundary = self.artifact_boundary if isinstance(self.artifact_boundary, MarketStateArtifactBoundary) else MarketStateArtifactBoundary.from_dict(self.artifact_boundary)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "pathway", _enum_value(self.pathway, MarketStatePathway, field_name="pathway"))
        object.__setattr__(self, "classification", _non_empty_text(self.classification, field_name="classification"))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "key_columns", _string_tuple(self.key_columns, field_name="key_columns", require_non_empty=True))
        object.__setattr__(
            self,
            "partition_columns",
            _string_tuple(self.partition_columns, field_name="partition_columns", require_non_empty=True),
        )
        object.__setattr__(self, "state_columns", _string_tuple(self.state_columns, field_name="state_columns", require_non_empty=True))
        object.__setattr__(
            self,
            "diagnostic_columns",
            _string_tuple(self.diagnostic_columns, field_name="diagnostic_columns", require_non_empty=True),
        )
        forbidden = {"asset", "asset_id", "symbol"}
        present = sorted(forbidden.intersection(self.required_columns).union(forbidden.intersection(self.partition_columns)))
        if present:
            raise ValueError(f"Market-state output schema cannot include asset-level label columns: {present}")
        object.__setattr__(self, "artifact_boundary", boundary)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.key_columns, *self.state_columns, *self.diagnostic_columns)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "classification": self.classification,
            "scope": "whole_market_timestamp_window",
            "key_columns": list(self.key_columns),
            "partition_columns": list(self.partition_columns),
            "state_columns": list(self.state_columns),
            "diagnostic_columns": list(self.diagnostic_columns),
            "required_columns": list(self.required_columns),
            "artifact_boundary": self.artifact_boundary.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateOutputSchema":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "artifact_kind",
                "pathway",
                "classification",
                "key_columns",
                "partition_columns",
                "state_columns",
                "diagnostic_columns",
                "artifact_boundary",
            },
            optional={"required_columns", "scope"},
            context="MarketStateOutputSchema",
        )
        obj.pop("required_columns", None)
        obj.pop("scope", None)
        return cls(**obj)


@dataclass(frozen=True)
class MarketStateTaxonomyManifest:
    axes: Mapping[str, MarketStateAxisSpec | Mapping[str, Any]]
    bands: Mapping[str, MarketStateBandSpec | Mapping[str, Any]]
    output_schema: MarketStateOutputSchema | Mapping[str, Any]
    artifact_boundary: MarketStateArtifactBoundary | Mapping[str, Any] = field(default_factory=MarketStateArtifactBoundary)
    pathway: str | MarketStatePathway = MarketStatePathway.MARKET_STATE
    artifact_kind: str = "market_state_taxonomy_manifest"
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axes: dict[str, MarketStateAxisSpec] = {}
        for key, value in self.axes.items():
            spec = value if isinstance(value, MarketStateAxisSpec) else MarketStateAxisSpec.from_dict(value)
            if str(key) != spec.axis_id:
                raise ValueError("Market-state taxonomy axis keys must match axis_id")
            axes[spec.axis_id] = spec
        bands: dict[str, MarketStateBandSpec] = {}
        for key, value in self.bands.items():
            spec = value if isinstance(value, MarketStateBandSpec) else MarketStateBandSpec.from_dict(value)
            if str(key) != spec.band:
                raise ValueError("Market-state taxonomy band keys must match band")
            bands[spec.band] = spec
        if set(axes) != set(MARKET_STATE_AXIS_VALUES):
            missing = sorted(set(MARKET_STATE_AXIS_VALUES).difference(axes))
            extra = sorted(set(axes).difference(MARKET_STATE_AXIS_VALUES))
            raise ValueError(f"Market-state taxonomy axes mismatch; missing={missing}, extra={extra}")
        if set(bands) != set(MARKET_STATE_BAND_VALUES):
            missing = sorted(set(MARKET_STATE_BAND_VALUES).difference(bands))
            extra = sorted(set(bands).difference(MARKET_STATE_BAND_VALUES))
            raise ValueError(f"Market-state taxonomy bands mismatch; missing={missing}, extra={extra}")
        for axis in axes.values():
            unknown_bands = sorted(set(axis.compatible_bands).difference(bands))
            if unknown_bands:
                raise ValueError(f"Market-state axis {axis.axis_id!r} declares unknown bands: {unknown_bands}")
        schema = self.output_schema if isinstance(self.output_schema, MarketStateOutputSchema) else MarketStateOutputSchema.from_dict(self.output_schema)
        boundary = self.artifact_boundary if isinstance(self.artifact_boundary, MarketStateArtifactBoundary) else MarketStateArtifactBoundary.from_dict(self.artifact_boundary)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "pathway", _enum_value(self.pathway, MarketStatePathway, field_name="pathway"))
        object.__setattr__(self, "axes", dict(sorted(axes.items())))
        object.__setattr__(self, "bands", dict(sorted(bands.items())))
        object.__setattr__(self, "output_schema", schema)
        object.__setattr__(self, "artifact_boundary", boundary)
        object.__setattr__(self, "metadata", to_jsonable(_mapping(self.metadata, field_name="metadata")))

    def axis_spec(self, axis: str | MarketStateAxis) -> MarketStateAxisSpec:
        axis_value = _enum_value(axis, MarketStateAxis, field_name="axis")
        return self.axes[axis_value]

    def band_spec(self, band: str | MarketStateBand) -> MarketStateBandSpec:
        band_value = _enum_value(band, MarketStateBand, field_name="band")
        return self.bands[band_value]

    def validate_axis_band(self, axis: str | MarketStateAxis, band: str | MarketStateBand) -> None:
        axis_spec = self.axis_spec(axis)
        band_value = _enum_value(band, MarketStateBand, field_name="band")
        if band_value not in axis_spec.compatible_bands:
            raise ValueError(f"Unsupported market-state axis/band combination: {axis_spec.axis_id}/{band_value}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "scope": "whole_market_not_asset_level",
            "axes": {key: spec.as_dict() for key, spec in sorted(self.axes.items())},
            "bands": {key: spec.as_dict() for key, spec in sorted(self.bands.items())},
            "output_schema": self.output_schema.as_dict(),
            "artifact_boundary": self.artifact_boundary.as_dict(),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateTaxonomyManifest":
        obj = require_known_fields(
            payload,
            required={"schema_version", "artifact_kind", "pathway", "axes", "bands", "output_schema", "artifact_boundary"},
            optional={"metadata", "scope"},
            context="MarketStateTaxonomyManifest",
        )
        obj.pop("scope", None)
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "MarketStateTaxonomyManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="MarketStateTaxonomyManifest JSON"))


__all__ = [
    "MARKET_STATE_AXIS_VALUES",
    "MARKET_STATE_BAND_VALUES",
    "MARKET_STATE_KNOWN_AT_REQUIRED_FIELDS",
    "MARKET_STATE_LINEAGE_REQUIRED_FIELDS",
    "MARKET_STATE_PATHWAY_VALUE",
    "MARKET_STATE_SCHEMA_VERSION",
    "MarketStateArtifactBoundary",
    "MarketStateAxis",
    "MarketStateAxisSpec",
    "MarketStateBand",
    "MarketStateBandSpec",
    "MarketStateOutputSchema",
    "MarketStatePathway",
    "MarketStateSchemaVersion",
    "MarketStateStudyKey",
    "MarketStateTaxonomyManifest",
    "default_market_state_known_at_requirements",
    "default_market_state_lineage_requirements",
]
