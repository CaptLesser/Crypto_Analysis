from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION
from src.regimes.core.serialization import dumps_json, loads_json, require_known_fields, require_json_object, to_jsonable


class AssetStateSchemaVersion(IntEnum):
    V1 = CANONICAL_SCHEMA_VERSION


ASSET_STATE_SCHEMA_VERSION = int(AssetStateSchemaVersion.V1)


class AssetStatePathway(str, Enum):
    ASSET_STATE = "asset_state"


class AssetStateAxis(str, Enum):
    TREND = "trend"
    VOLATILITY = "volatility"
    ACTIVITY = "activity"
    MEAN_REVERSION = "mean_reversion"
    DRAWDOWN = "drawdown"
    RANGE_EFFICIENCY = "range_efficiency"


class AssetStateBand(str, Enum):
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


class AssetStateClusterabilityStatus(str, Enum):
    CLUSTERABLE_CANDIDATE = "clusterable_candidate"
    SINGLE_STATE_FALLBACK = "single_state_fallback"
    NOT_CLUSTERABLE = "not_clusterable"
    INSUFFICIENT_DATA = "insufficient_data"
    MISSING_FEATURES = "missing_features"
    LEAKAGE_BLOCKED = "leakage_blocked"
    BLOCKED = "blocked"


class AssetStateFallbackStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    ALLOWED_NOT_USED = "allowed_not_used"
    APPLIED_NEUTRAL_SINGLE_STATE = "applied_neutral_single_state"
    BLOCKED = "blocked"
    MISSING_DATA = "missing_data"


class AssetStateProfileDecisionStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    CANDIDATE = "candidate"
    WATCHLIST = "watchlist"
    REJECTED = "rejected"
    FALLBACK_ONLY = "fallback_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    SELECTED_NON_PRODUCTION = "selected_non_production"


ASSET_STATE_AXIS_VALUES: tuple[str, ...] = tuple(axis.value for axis in AssetStateAxis)
ASSET_STATE_BAND_VALUES: tuple[str, ...] = tuple(band.value for band in AssetStateBand)
ASSET_STATE_PATHWAY_VALUE = AssetStatePathway.ASSET_STATE.value


def _schema_version(value: object) -> int:
    try:
        version = int(value)
    except Exception as exc:
        raise ValueError(f"Asset-state schema_version must be {ASSET_STATE_SCHEMA_VERSION}") from exc
    if version != ASSET_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported asset-state schema_version {version!r}; expected {ASSET_STATE_SCHEMA_VERSION}")
    return version


def _non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Asset-state {field_name} must be non-empty")
    return text


def _enum_value(value: object, enum_type: type[Enum], *, field_name: str) -> str:
    raw = value.value if isinstance(value, Enum) else value
    text = _non_empty_text(raw, field_name=field_name).lower()
    valid = tuple(str(member.value) for member in enum_type)
    if text not in valid:
        raise ValueError(f"Unsupported asset-state {field_name} {text!r}; expected one of: {', '.join(valid)}")
    return text


def _string_tuple(values: Sequence[object] | None, *, field_name: str, require_non_empty: bool = False) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Asset-state {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not out:
        raise ValueError(f"Asset-state {field_name} must include at least one value")
    return tuple(dict.fromkeys(out))


def _mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Asset-state {field_name} must be a JSON object")
    return dict(value)


@dataclass(frozen=True)
class AssetStateBandSpec:
    band: str | AssetStateBand
    ceiling_interval_min: int
    member_intervals: Sequence[int]
    train_days: int
    validation_horizons_min: Sequence[int]
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        band = _enum_value(self.band, AssetStateBand, field_name="band")
        intervals = tuple(int(value) for value in self.member_intervals)
        horizons = tuple(int(value) for value in self.validation_horizons_min)
        if not intervals or any(value <= 0 for value in intervals):
            raise ValueError("Asset-state band member_intervals must be positive and non-empty")
        if int(self.ceiling_interval_min) <= 0:
            raise ValueError("Asset-state band ceiling_interval_min must be positive")
        if int(self.train_days) <= 0:
            raise ValueError("Asset-state band train_days must be positive")
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("Asset-state band validation_horizons_min must be positive and non-empty")
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetStateBandSpec":
        obj = require_known_fields(
            payload,
            required={"schema_version", "band", "ceiling_interval_min", "member_intervals", "train_days", "validation_horizons_min"},
            optional=set(),
            context="AssetStateBandSpec",
        )
        return cls(**obj)


@dataclass(frozen=True)
class AssetStateAxisSpec:
    axis_id: str | AssetStateAxis
    purpose: str
    compatible_bands: Sequence[str | AssetStateBand]
    expected_feature_families: Sequence[str]
    expected_forward_validation_targets: Sequence[str]
    fallback_policy_hints: Mapping[str, Any]
    allow_single_state_output: bool
    label_set: Sequence[str] = ()
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        axis_id = _enum_value(self.axis_id, AssetStateAxis, field_name="axis_id")
        purpose = _non_empty_text(self.purpose, field_name="axis purpose")
        bands = tuple(_enum_value(value, AssetStateBand, field_name="compatible_bands") for value in self.compatible_bands)
        families = _string_tuple(self.expected_feature_families, field_name="expected_feature_families", require_non_empty=True)
        targets = _string_tuple(
            self.expected_forward_validation_targets,
            field_name="expected_forward_validation_targets",
            require_non_empty=True,
        )
        labels = _string_tuple(self.label_set, field_name="label_set")
        hints = _mapping(self.fallback_policy_hints, field_name="fallback_policy_hints")
        if not bands:
            raise ValueError("Asset-state axis compatible_bands must include at least one band")
        if "policy" not in hints or not str(hints.get("policy", "")).strip():
            raise ValueError("Asset-state fallback_policy_hints requires policy")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "axis_id", axis_id)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "compatible_bands", tuple(dict.fromkeys(bands)))
        object.__setattr__(self, "expected_feature_families", families)
        object.__setattr__(self, "expected_forward_validation_targets", targets)
        object.__setattr__(self, "fallback_policy_hints", to_jsonable(hints))
        object.__setattr__(self, "allow_single_state_output", bool(self.allow_single_state_output))
        object.__setattr__(self, "label_set", labels)

    def supports_band(self, band: str | AssetStateBand) -> bool:
        return _enum_value(band, AssetStateBand, field_name="band") in self.compatible_bands

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "axis_id": self.axis_id,
            "purpose": self.purpose,
            "compatible_bands": list(self.compatible_bands),
            "expected_feature_families": list(self.expected_feature_families),
            "expected_forward_validation_targets": list(self.expected_forward_validation_targets),
            "fallback_policy_hints": to_jsonable(self.fallback_policy_hints),
            "allow_single_state_output": bool(self.allow_single_state_output),
            "label_set": list(self.label_set),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetStateAxisSpec":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "axis_id",
                "purpose",
                "compatible_bands",
                "expected_feature_families",
                "expected_forward_validation_targets",
                "fallback_policy_hints",
                "allow_single_state_output",
            },
            optional={"label_set"},
            context="AssetStateAxisSpec",
        )
        return cls(**obj)


@dataclass(frozen=True)
class AssetStateStudyKey:
    study_id: str
    axis: str | AssetStateAxis
    band: str | AssetStateBand
    pathway: str | AssetStatePathway = AssetStatePathway.ASSET_STATE
    classification: str = "sandbox"
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        pathway = _enum_value(self.pathway, AssetStatePathway, field_name="pathway")
        axis = _enum_value(self.axis, AssetStateAxis, field_name="axis")
        band = _enum_value(self.band, AssetStateBand, field_name="band")
        classification = _non_empty_text(self.classification, field_name="classification").lower()
        if classification in {"production", "prod"}:
            raise ValueError("Asset-state Test study keys cannot use production classification")
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
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetStateStudyKey":
        obj = require_known_fields(
            payload,
            required={"schema_version", "study_id", "pathway", "axis", "band", "classification"},
            optional=set(),
            context="AssetStateStudyKey",
        )
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "AssetStateStudyKey":
        return cls.from_dict(require_json_object(loads_json(text), context="AssetStateStudyKey JSON"))


@dataclass(frozen=True)
class AssetStateOutputSchema:
    key_columns: Sequence[str]
    partition_columns: Sequence[str]
    state_columns: Sequence[str]
    diagnostic_columns: Sequence[str]
    production_flags: Mapping[str, Any]
    pathway: str | AssetStatePathway = AssetStatePathway.ASSET_STATE
    classification: str = "sandbox_non_production"
    artifact_kind: str = "asset_state_test_output_schema"
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        flags = _mapping(self.production_flags, field_name="production_flags")
        if bool(flags.get("production_output")) or bool(flags.get("production_parquet_allowed")):
            raise ValueError("Asset-state output schema must remain non-production")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "pathway", _enum_value(self.pathway, AssetStatePathway, field_name="pathway"))
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
        object.__setattr__(self, "production_flags", to_jsonable(flags))

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.key_columns, *self.state_columns, *self.diagnostic_columns)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "classification": self.classification,
            "key_columns": list(self.key_columns),
            "partition_columns": list(self.partition_columns),
            "state_columns": list(self.state_columns),
            "diagnostic_columns": list(self.diagnostic_columns),
            "required_columns": list(self.required_columns),
            "production_flags": to_jsonable(self.production_flags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetStateOutputSchema":
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
                "production_flags",
            },
            optional={"required_columns"},
            context="AssetStateOutputSchema",
        )
        obj.pop("required_columns", None)
        return cls(**obj)


@dataclass(frozen=True)
class AssetStateTaxonomyManifest:
    axes: Mapping[str, AssetStateAxisSpec | Mapping[str, Any]]
    bands: Mapping[str, AssetStateBandSpec | Mapping[str, Any]]
    output_schema: AssetStateOutputSchema | Mapping[str, Any]
    pathway: str | AssetStatePathway = AssetStatePathway.ASSET_STATE
    artifact_kind: str = "asset_state_taxonomy_manifest"
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axes: dict[str, AssetStateAxisSpec] = {}
        for key, value in self.axes.items():
            spec = value if isinstance(value, AssetStateAxisSpec) else AssetStateAxisSpec.from_dict(value)
            if str(key) != spec.axis_id:
                raise ValueError("Asset-state taxonomy axis keys must match axis_id")
            axes[spec.axis_id] = spec
        bands: dict[str, AssetStateBandSpec] = {}
        for key, value in self.bands.items():
            spec = value if isinstance(value, AssetStateBandSpec) else AssetStateBandSpec.from_dict(value)
            if str(key) != spec.band:
                raise ValueError("Asset-state taxonomy band keys must match band")
            bands[spec.band] = spec
        if set(axes) != set(ASSET_STATE_AXIS_VALUES):
            missing = sorted(set(ASSET_STATE_AXIS_VALUES).difference(axes))
            extra = sorted(set(axes).difference(ASSET_STATE_AXIS_VALUES))
            raise ValueError(f"Asset-state taxonomy axes mismatch; missing={missing}, extra={extra}")
        if set(bands) != set(ASSET_STATE_BAND_VALUES):
            missing = sorted(set(ASSET_STATE_BAND_VALUES).difference(bands))
            extra = sorted(set(bands).difference(ASSET_STATE_BAND_VALUES))
            raise ValueError(f"Asset-state taxonomy bands mismatch; missing={missing}, extra={extra}")
        for axis in axes.values():
            unknown_bands = sorted(set(axis.compatible_bands).difference(bands))
            if unknown_bands:
                raise ValueError(f"Asset-state axis {axis.axis_id!r} declares unknown bands: {unknown_bands}")
        schema = self.output_schema if isinstance(self.output_schema, AssetStateOutputSchema) else AssetStateOutputSchema.from_dict(self.output_schema)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "pathway", _enum_value(self.pathway, AssetStatePathway, field_name="pathway"))
        object.__setattr__(self, "axes", dict(sorted(axes.items())))
        object.__setattr__(self, "bands", dict(sorted(bands.items())))
        object.__setattr__(self, "output_schema", schema)
        object.__setattr__(self, "metadata", to_jsonable(_mapping(self.metadata, field_name="metadata")))

    def axis_spec(self, axis: str | AssetStateAxis) -> AssetStateAxisSpec:
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        return self.axes[axis_value]

    def band_spec(self, band: str | AssetStateBand) -> AssetStateBandSpec:
        band_value = _enum_value(band, AssetStateBand, field_name="band")
        return self.bands[band_value]

    def validate_axis_band(self, axis: str | AssetStateAxis, band: str | AssetStateBand) -> None:
        axis_spec = self.axis_spec(axis)
        band_value = _enum_value(band, AssetStateBand, field_name="band")
        if band_value not in axis_spec.compatible_bands:
            raise ValueError(f"Unsupported asset-state axis/band combination: {axis_spec.axis_id}/{band_value}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "axes": {key: spec.as_dict() for key, spec in sorted(self.axes.items())},
            "bands": {key: spec.as_dict() for key, spec in sorted(self.bands.items())},
            "output_schema": self.output_schema.as_dict(),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetStateTaxonomyManifest":
        obj = require_known_fields(
            payload,
            required={"schema_version", "artifact_kind", "pathway", "axes", "bands", "output_schema"},
            optional={"metadata"},
            context="AssetStateTaxonomyManifest",
        )
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "AssetStateTaxonomyManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="AssetStateTaxonomyManifest JSON"))


__all__ = [
    "ASSET_STATE_AXIS_VALUES",
    "ASSET_STATE_BAND_VALUES",
    "ASSET_STATE_PATHWAY_VALUE",
    "ASSET_STATE_SCHEMA_VERSION",
    "AssetStateAxis",
    "AssetStateAxisSpec",
    "AssetStateBand",
    "AssetStateBandSpec",
    "AssetStateClusterabilityStatus",
    "AssetStateFallbackStatus",
    "AssetStateOutputSchema",
    "AssetStatePathway",
    "AssetStateProfileDecisionStatus",
    "AssetStateSchemaVersion",
    "AssetStateStudyKey",
    "AssetStateTaxonomyManifest",
]
