from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import dumps_json, loads_json, require_known_fields, require_json_object, to_jsonable


class RegimeLayer(str, Enum):
    ASSET_STATE = "asset_state"
    MARKET_STATE = "market_state"
    RELATIVE_STATE = "relative_state"


class RegimeAxis(str, Enum):
    TREND = "trend"
    VOL = "vol"
    ACTIVITY = "activity"
    MARKET = "market"
    BREADTH = "breadth"
    DISPERSION = "dispersion"
    CORRELATION = "correlation"
    MARKET_VOL = "market_vol"
    LEADERSHIP = "leadership"
    RELATIVE = "relative"
    BETA = "beta"
    RELATIVE_STRENGTH = "relative_strength"
    RELATIVE_DISPERSION = "relative_dispersion"


class RegimeBand(str, Enum):
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"
    POOLED = "pooled"


class RegimeClassification(str, Enum):
    PRODUCTION = "production"
    STAGED = "staged"
    SCAFFOLD = "scaffold"
    SANDBOX = "sandbox"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    METADATA_ONLY = "metadata_only"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class RegimeSchemaVersion(IntEnum):
    V1 = 1


CANONICAL_SCHEMA_VERSION = int(RegimeSchemaVersion.V1)
REGIME_LAYER_VALUES: tuple[str, ...] = tuple(layer.value for layer in RegimeLayer)
REGIME_AXIS_VALUES: tuple[str, ...] = tuple(axis.value for axis in RegimeAxis)
REGIME_BAND_VALUES: tuple[str, ...] = tuple(band.value for band in RegimeBand)
REGIME_CLASSIFICATION_VALUES: tuple[str, ...] = tuple(classification.value for classification in RegimeClassification)
REGIME_RUN_STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in RunStatus)

ASSET_STATE_AXES: tuple[str, ...] = (
    RegimeAxis.TREND.value,
    RegimeAxis.VOL.value,
    RegimeAxis.ACTIVITY.value,
)
MARKET_STATE_AXES: tuple[str, ...] = (
    RegimeAxis.MARKET.value,
    RegimeAxis.BREADTH.value,
    RegimeAxis.DISPERSION.value,
    RegimeAxis.CORRELATION.value,
    RegimeAxis.MARKET_VOL.value,
    RegimeAxis.LEADERSHIP.value,
)
RELATIVE_STATE_AXES: tuple[str, ...] = (
    RegimeAxis.RELATIVE.value,
    RegimeAxis.BETA.value,
    RegimeAxis.CORRELATION.value,
    RegimeAxis.RELATIVE_STRENGTH.value,
    RegimeAxis.RELATIVE_DISPERSION.value,
)
REGIME_LAYER_AXIS_VALUES: Mapping[str, tuple[str, ...]] = {
    RegimeLayer.ASSET_STATE.value: ASSET_STATE_AXES,
    RegimeLayer.MARKET_STATE.value: MARKET_STATE_AXES,
    RegimeLayer.RELATIVE_STATE.value: RELATIVE_STATE_AXES,
}


def _enum_values(enum_type: type[Enum]) -> tuple[Any, ...]:
    return tuple(member.value for member in enum_type)


def require_schema_version(value: object, *, field_name: str = "schema_version") -> int:
    try:
        version = int(value)
    except Exception as exc:
        raise ValueError(f"Regime {field_name} must be {CANONICAL_SCHEMA_VERSION}") from exc
    if version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Regime {field_name} {version!r}; expected {CANONICAL_SCHEMA_VERSION}")
    return version


def require_non_empty_string(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime {field_name} must be non-empty")
    return text


def normalize_enum_value(value: object, enum_type: type[Enum], *, field_name: str) -> str:
    text = require_non_empty_string(value.value if isinstance(value, Enum) else value, field_name=field_name).lower()
    valid = _enum_values(enum_type)
    if text not in valid:
        valid_text = ", ".join(str(item) for item in valid)
        raise ValueError(f"Unsupported Regime {field_name} {text!r}; expected one of: {valid_text}")
    return text


def normalize_string_tuple(values: Sequence[object] | None, *, field_name: str, require_non_empty: bool = False) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime {field_name} must be a sequence of strings")
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not normalized:
        raise ValueError(f"Regime {field_name} must include at least one value")
    return normalized


def require_json_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Regime {field_name} must be a JSON object")
    return dict(value)


def validate_layer_axis_band(*, layer: str, axis: str, band: str) -> None:
    layer = normalize_enum_value(layer, RegimeLayer, field_name="layer")
    axis = normalize_enum_value(axis, RegimeAxis, field_name="axis")
    normalize_enum_value(band, RegimeBand, field_name="band")
    allowed_axes = REGIME_LAYER_AXIS_VALUES[layer]
    if axis not in allowed_axes:
        valid_text = ", ".join(allowed_axes)
        raise ValueError(f"Unsupported Regime {layer} axis {axis!r}; expected one of: {valid_text}")


def validate_classification_for_key(*, layer: str, band: str, classification: str) -> None:
    layer = normalize_enum_value(layer, RegimeLayer, field_name="layer")
    band = normalize_enum_value(band, RegimeBand, field_name="band")
    classification = normalize_enum_value(classification, RegimeClassification, field_name="classification")
    if classification == RegimeClassification.PRODUCTION.value and layer != RegimeLayer.ASSET_STATE.value:
        raise ValueError("Regime production classification is currently allowed only for asset_state")
    if classification == RegimeClassification.PRODUCTION.value and band == RegimeBand.POOLED.value:
        raise ValueError("Regime production classification requires a concrete micro/meso/macro band")


def validate_study_key_compatible(left: "StudyKey", right: "StudyKey", *, context: str) -> None:
    if left.as_dict() != right.as_dict():
        raise ValueError(f"{context} study_key must match")


def _parse_orderable_time(value: object, *, field_name: str) -> int | float | datetime:
    if value is None:
        raise ValueError(f"Regime {field_name} must be non-empty")
    if isinstance(value, bool):
        raise ValueError(f"Regime {field_name} must be a timestamp, not bool")
    try:
        return int(value)
    except Exception:
        pass
    text = require_non_empty_string(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime {field_name} must be an integer timestamp or ISO datetime") from exc


def validate_window_order(start_ts: object, end_ts: object) -> None:
    start = _parse_orderable_time(start_ts, field_name="start_ts")
    end = _parse_orderable_time(end_ts, field_name="end_ts")
    try:
        invalid_order = start > end
    except TypeError as exc:
        raise ValueError("Regime dataset window start_ts and end_ts must be comparable") from exc
    if invalid_order:
        raise ValueError("Regime dataset window start_ts must be <= end_ts")


def coerce_study_key(value: "StudyKey | Mapping[str, Any]", *, field_name: str = "study_key") -> "StudyKey":
    if isinstance(value, StudyKey):
        return value
    if isinstance(value, Mapping):
        return StudyKey.from_dict(value)
    raise ValueError(f"Regime {field_name} must be a StudyKey or JSON object")


@dataclass(frozen=True)
class StudyKey:
    study_id: str
    layer: str | RegimeLayer
    axis: str | RegimeAxis
    band: str | RegimeBand
    classification: str | RegimeClassification
    schema_version: int | RegimeSchemaVersion = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_id = require_non_empty_string(self.study_id, field_name="study_id")
        layer = normalize_enum_value(self.layer, RegimeLayer, field_name="layer")
        axis = normalize_enum_value(self.axis, RegimeAxis, field_name="axis")
        band = normalize_enum_value(self.band, RegimeBand, field_name="band")
        classification = normalize_enum_value(self.classification, RegimeClassification, field_name="classification")
        validate_layer_axis_band(layer=layer, axis=axis, band=band)
        validate_classification_for_key(layer=layer, band=band, classification=classification)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "classification", classification)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_id": self.study_id,
            "layer": self.layer,
            "axis": self.axis,
            "band": self.band,
            "classification": self.classification,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudyKey":
        obj = require_known_fields(
            payload,
            required={"schema_version", "study_id", "layer", "axis", "band", "classification"},
            optional=set(),
            context="Regime StudyKey",
        )
        return cls(
            schema_version=obj["schema_version"],
            study_id=obj["study_id"],
            layer=obj["layer"],
            axis=obj["axis"],
            band=obj["band"],
            classification=obj["classification"],
        )

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "StudyKey":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime StudyKey JSON"))


@dataclass(frozen=True)
class DatasetWindowSpec:
    study_key: StudyKey | Mapping[str, Any]
    dataset_id: str
    start_ts: int | str
    end_ts: int | str
    interval_minutes: int
    source_artifacts: Sequence[str]
    feature_columns: Sequence[str]
    horizon_minutes: int | None = None
    asset: str | None = None
    assets: Sequence[str] = ()
    universe: str | None = None
    benchmark: str | None = None
    peer_assets: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | RegimeSchemaVersion = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_key = coerce_study_key(self.study_key)
        dataset_id = require_non_empty_string(self.dataset_id, field_name="dataset_id")
        validate_window_order(self.start_ts, self.end_ts)
        interval_minutes = int(self.interval_minutes)
        if interval_minutes <= 0:
            raise ValueError("Regime interval_minutes must be positive")
        horizon_minutes = None if self.horizon_minutes is None else int(self.horizon_minutes)
        if horizon_minutes is not None and horizon_minutes <= 0:
            raise ValueError("Regime horizon_minutes must be positive when supplied")
        source_artifacts = normalize_string_tuple(
            self.source_artifacts,
            field_name="source_artifacts",
            require_non_empty=True,
        )
        feature_columns = normalize_string_tuple(
            self.feature_columns,
            field_name="feature_columns",
            require_non_empty=True,
        )
        assets = normalize_string_tuple(self.assets, field_name="assets")
        peer_assets = normalize_string_tuple(self.peer_assets, field_name="peer_assets")
        asset = None if self.asset is None else require_non_empty_string(self.asset, field_name="asset")
        universe = None if self.universe is None else require_non_empty_string(self.universe, field_name="universe")
        benchmark = None if self.benchmark is None else require_non_empty_string(self.benchmark, field_name="benchmark")
        if study_key.layer == RegimeLayer.ASSET_STATE.value and not (asset or assets):
            raise ValueError("Regime asset_state dataset window requires asset or assets")
        if study_key.layer == RegimeLayer.MARKET_STATE.value and not (universe or assets):
            raise ValueError("Regime market_state dataset window requires universe or assets")
        if study_key.layer == RegimeLayer.RELATIVE_STATE.value:
            if not asset:
                raise ValueError("Regime relative_state dataset window requires asset")
            if not (benchmark or peer_assets):
                raise ValueError("Regime relative_state dataset window requires benchmark or peer_assets")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_key", study_key)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "interval_minutes", interval_minutes)
        object.__setattr__(self, "horizon_minutes", horizon_minutes)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(self, "feature_columns", feature_columns)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "peer_assets", peer_assets)
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_key": self.study_key.as_dict(),
            "dataset_id": self.dataset_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "interval_minutes": self.interval_minutes,
            "horizon_minutes": self.horizon_minutes,
            "asset": self.asset,
            "assets": list(self.assets),
            "universe": self.universe,
            "benchmark": self.benchmark,
            "peer_assets": list(self.peer_assets),
            "source_artifacts": list(self.source_artifacts),
            "feature_columns": list(self.feature_columns),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetWindowSpec":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "study_key",
                "dataset_id",
                "start_ts",
                "end_ts",
                "interval_minutes",
                "source_artifacts",
                "feature_columns",
            },
            optional={
                "horizon_minutes",
                "asset",
                "assets",
                "universe",
                "benchmark",
                "peer_assets",
                "metadata",
            },
            context="Regime DatasetWindowSpec",
        )
        return cls(
            schema_version=obj["schema_version"],
            study_key=obj["study_key"],
            dataset_id=obj["dataset_id"],
            start_ts=obj["start_ts"],
            end_ts=obj["end_ts"],
            interval_minutes=obj["interval_minutes"],
            horizon_minutes=obj.get("horizon_minutes"),
            asset=obj.get("asset"),
            assets=obj.get("assets", ()),
            universe=obj.get("universe"),
            benchmark=obj.get("benchmark"),
            peer_assets=obj.get("peer_assets", ()),
            source_artifacts=obj["source_artifacts"],
            feature_columns=obj["feature_columns"],
            metadata=obj.get("metadata", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "DatasetWindowSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime DatasetWindowSpec JSON"))


@dataclass(frozen=True)
class TrialSpec:
    study_key: StudyKey | Mapping[str, Any]
    trial_id: str
    dataset_window: DatasetWindowSpec | Mapping[str, Any]
    feature_family: str
    preprocessing_family: str
    clusterer_family: str
    assignment_policy: str
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    requested_artifacts: Sequence[str] = ("trial_result_envelope",)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | RegimeSchemaVersion = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_key = coerce_study_key(self.study_key)
        if isinstance(self.dataset_window, DatasetWindowSpec):
            dataset_window = self.dataset_window
        elif isinstance(self.dataset_window, Mapping):
            dataset_window = DatasetWindowSpec.from_dict(self.dataset_window)
        else:
            raise ValueError("Regime dataset_window must be a DatasetWindowSpec or JSON object")
        validate_study_key_compatible(study_key, dataset_window.study_key, context="Regime TrialSpec")
        trial_id = require_non_empty_string(self.trial_id, field_name="trial_id")
        text_fields: dict[str, str] = {}
        for field_name in ("feature_family", "preprocessing_family", "clusterer_family", "assignment_policy"):
            text_fields[field_name] = require_non_empty_string(getattr(self, field_name), field_name=field_name)
        requested_artifacts = normalize_string_tuple(
            self.requested_artifacts,
            field_name="requested_artifacts",
            require_non_empty=True,
        )
        random_seed = None if self.random_seed is None else int(self.random_seed)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_key", study_key)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "dataset_window", dataset_window)
        for field_name, value in text_fields.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "hyperparameters", to_jsonable(require_json_mapping(self.hyperparameters, field_name="hyperparameters")))
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "requested_artifacts", requested_artifacts)
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_key": self.study_key.as_dict(),
            "trial_id": self.trial_id,
            "dataset_window": self.dataset_window.as_dict(),
            "feature_family": str(self.feature_family),
            "preprocessing_family": str(self.preprocessing_family),
            "clusterer_family": str(self.clusterer_family),
            "assignment_policy": str(self.assignment_policy),
            "hyperparameters": to_jsonable(self.hyperparameters),
            "random_seed": self.random_seed,
            "requested_artifacts": list(self.requested_artifacts),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialSpec":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "study_key",
                "trial_id",
                "dataset_window",
                "feature_family",
                "preprocessing_family",
                "clusterer_family",
                "assignment_policy",
            },
            optional={
                "hyperparameters",
                "random_seed",
                "requested_artifacts",
                "metadata",
            },
            context="Regime TrialSpec",
        )
        return cls(
            schema_version=obj["schema_version"],
            study_key=obj["study_key"],
            trial_id=obj["trial_id"],
            dataset_window=obj["dataset_window"],
            feature_family=obj["feature_family"],
            preprocessing_family=obj["preprocessing_family"],
            clusterer_family=obj["clusterer_family"],
            assignment_policy=obj["assignment_policy"],
            hyperparameters=obj.get("hyperparameters", {}),
            random_seed=obj.get("random_seed"),
            requested_artifacts=obj.get("requested_artifacts", ("trial_result_envelope",)),
            metadata=obj.get("metadata", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "TrialSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime TrialSpec JSON"))


@dataclass(frozen=True)
class TrialArtifacts:
    study_key: StudyKey | Mapping[str, Any]
    trial_id: str
    artifact_paths: Mapping[str, str]
    production_outputs_written: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | RegimeSchemaVersion = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_key = coerce_study_key(self.study_key)
        trial_id = require_non_empty_string(self.trial_id, field_name="trial_id")
        paths = require_json_mapping(self.artifact_paths, field_name="artifact_paths")
        if not paths:
            raise ValueError("Regime artifact_paths must include at least one value")
        normalized_paths = {
            require_non_empty_string(key, field_name="artifact_paths key"): require_non_empty_string(
                value,
                field_name=f"artifact_paths.{key}",
            )
            for key, value in paths.items()
        }
        if study_key.classification != RegimeClassification.PRODUCTION.value and bool(self.production_outputs_written):
            raise ValueError("Regime non-production TrialArtifacts cannot claim production_outputs_written")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_key", study_key)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "artifact_paths", normalized_paths)
        object.__setattr__(self, "production_outputs_written", bool(self.production_outputs_written))
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_key": self.study_key.as_dict(),
            "trial_id": self.trial_id,
            "artifact_paths": dict(self.artifact_paths),
            "production_outputs_written": bool(self.production_outputs_written),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialArtifacts":
        obj = require_known_fields(
            payload,
            required={"schema_version", "study_key", "trial_id", "artifact_paths"},
            optional={"production_outputs_written", "metadata"},
            context="Regime TrialArtifacts",
        )
        return cls(
            schema_version=obj["schema_version"],
            study_key=obj["study_key"],
            trial_id=obj["trial_id"],
            artifact_paths=obj["artifact_paths"],
            production_outputs_written=bool(obj.get("production_outputs_written", False)),
            metadata=obj.get("metadata", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "TrialArtifacts":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime TrialArtifacts JSON"))


@dataclass(frozen=True)
class TrialResultEnvelope:
    study_key: StudyKey | Mapping[str, Any]
    trial: TrialSpec | Mapping[str, Any]
    artifacts: TrialArtifacts | Mapping[str, Any]
    status: str | RunStatus
    metrics: Mapping[str, Any] = field(default_factory=dict)
    errors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | RegimeSchemaVersion = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        study_key = coerce_study_key(self.study_key)
        trial = self.trial if isinstance(self.trial, TrialSpec) else TrialSpec.from_dict(self.trial)
        artifacts = self.artifacts if isinstance(self.artifacts, TrialArtifacts) else TrialArtifacts.from_dict(self.artifacts)
        validate_study_key_compatible(study_key, trial.study_key, context="Regime TrialResultEnvelope trial")
        validate_study_key_compatible(study_key, artifacts.study_key, context="Regime TrialResultEnvelope artifacts")
        if trial.trial_id != artifacts.trial_id:
            raise ValueError("Regime TrialResultEnvelope artifacts trial_id must match trial trial_id")
        status = normalize_enum_value(self.status, RunStatus, field_name="run status")
        errors = normalize_string_tuple(self.errors, field_name="errors")
        if status == RunStatus.SUCCEEDED.value and errors:
            raise ValueError("Regime succeeded TrialResultEnvelope cannot carry errors")
        if status in {RunStatus.FAILED.value, RunStatus.SKIPPED.value, RunStatus.BLOCKED.value} and not errors:
            raise ValueError(f"Regime {status} TrialResultEnvelope requires errors")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "study_key", study_key)
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", to_jsonable(require_json_mapping(self.metrics, field_name="metrics")))
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_key": self.study_key.as_dict(),
            "trial": self.trial.as_dict(),
            "artifacts": self.artifacts.as_dict(),
            "status": self.status,
            "metrics": to_jsonable(self.metrics),
            "errors": list(self.errors),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialResultEnvelope":
        obj = require_known_fields(
            payload,
            required={"schema_version", "study_key", "trial", "artifacts", "status"},
            optional={"metrics", "errors", "metadata"},
            context="Regime TrialResultEnvelope",
        )
        return cls(
            schema_version=obj["schema_version"],
            study_key=obj["study_key"],
            trial=obj["trial"],
            artifacts=obj["artifacts"],
            status=obj["status"],
            metrics=obj.get("metrics", {}),
            errors=obj.get("errors", ()),
            metadata=obj.get("metadata", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "TrialResultEnvelope":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime TrialResultEnvelope JSON"))


__all__ = [
    "ASSET_STATE_AXES",
    "CANONICAL_SCHEMA_VERSION",
    "DatasetWindowSpec",
    "MARKET_STATE_AXES",
    "REGIME_AXIS_VALUES",
    "REGIME_BAND_VALUES",
    "REGIME_CLASSIFICATION_VALUES",
    "REGIME_LAYER_AXIS_VALUES",
    "REGIME_LAYER_VALUES",
    "REGIME_RUN_STATUS_VALUES",
    "RELATIVE_STATE_AXES",
    "RegimeAxis",
    "RegimeBand",
    "RegimeClassification",
    "RegimeLayer",
    "RegimeSchemaVersion",
    "RunStatus",
    "StudyKey",
    "TrialArtifacts",
    "TrialResultEnvelope",
    "TrialSpec",
    "coerce_study_key",
    "normalize_enum_value",
    "normalize_string_tuple",
    "require_json_mapping",
    "require_non_empty_string",
    "require_schema_version",
    "validate_classification_for_key",
    "validate_layer_axis_band",
    "validate_study_key_compatible",
    "validate_window_order",
]
