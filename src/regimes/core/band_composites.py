from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


BAND_COMPOSITE_SCHEMA_VERSION = 1
BAND_COMPOSITE_SPEC_ARTIFACT_KIND = "regime_band_composite_spec"
BAND_COMPOSITE_REGISTRY_ARTIFACT_KIND = "regime_band_composite_registry"

COMPOSITE_BANDS: tuple[str, ...] = ("micro", "meso", "macro")
ALIGNMENT_POLICY_CEILING_BOUNDARY = "ceiling_interval_boundary"
PAIRWISE_RELATIONSHIP_FEATURES = "pairwise_relationship_features"
CROSS_ASSET_SUMMARY_FEATURES = "cross_asset_summary_features"

DEFAULT_COMPOSITE_BAND_INTERVALS: Mapping[str, Mapping[str, Any]] = {
    "micro": {"ceiling_interval": 30, "member_intervals": (1, 5, 15, 30)},
    "meso": {"ceiling_interval": 240, "member_intervals": (60, 240)},
    "macro": {"ceiling_interval": 1440, "member_intervals": (720, 1440)},
}


def default_relationship_feature_permissions() -> dict[str, dict[str, Any]]:
    return {
        PAIRWISE_RELATIONSHIP_FEATURES: {
            "execution_enabled": False,
            "auto_inherit_member_intervals": False,
            "short_interval_execution_enabled": False,
            "requires_explicit_enable": True,
            "allowed_member_intervals": [],
            "status": "scaffold_only",
        },
        CROSS_ASSET_SUMMARY_FEATURES: {
            "execution_enabled": False,
            "cross_asset_clustering_enabled": False,
            "auto_inherit_member_intervals": False,
            "short_interval_execution_enabled": False,
            "requires_explicit_enable": True,
            "allowed_member_intervals": [],
            "status": "scaffold_only",
        },
    }


@dataclass(frozen=True)
class BandCompositeSpec:
    band: str
    ceiling_interval: int
    member_intervals: Sequence[int]
    output_cadence: int
    alignment_policy: str = ALIGNMENT_POLICY_CEILING_BOUNDARY
    allowed_feature_families: Sequence[str] = ()
    relationship_feature_permissions: Mapping[str, Any] = field(default_factory=default_relationship_feature_permissions)
    schema_version: int = BAND_COMPOSITE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        band = _band(self.band)
        ceiling_interval = _positive_int(self.ceiling_interval, field_name="ceiling_interval")
        member_intervals = _interval_tuple(self.member_intervals, field_name="member_intervals")
        for interval in member_intervals:
            if ceiling_interval % interval != 0:
                raise ValueError("Band composite member intervals must divide ceiling interval")
        output_cadence = _positive_int(self.output_cadence, field_name="output_cadence")
        if output_cadence != ceiling_interval:
            raise ValueError("Band composite output_cadence must preserve the ceiling interval cadence")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "ceiling_interval", ceiling_interval)
        object.__setattr__(self, "member_intervals", member_intervals)
        object.__setattr__(self, "output_cadence", output_cadence)
        object.__setattr__(self, "alignment_policy", _text(self.alignment_policy, field_name="alignment_policy"))
        object.__setattr__(
            self,
            "allowed_feature_families",
            _string_tuple(self.allowed_feature_families, field_name="allowed_feature_families"),
        )
        object.__setattr__(
            self,
            "relationship_feature_permissions",
            _relationship_permissions(self.relationship_feature_permissions),
        )
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def ceiling_interval_min(self) -> int:
        return self.ceiling_interval

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": BAND_COMPOSITE_SPEC_ARTIFACT_KIND,
            "band": self.band,
            "ceiling_interval": int(self.ceiling_interval),
            "ceiling_interval_min": int(self.ceiling_interval),
            "member_intervals": list(self.member_intervals),
            "output_cadence": int(self.output_cadence),
            "alignment_policy": self.alignment_policy,
            "allowed_feature_families": list(self.allowed_feature_families),
            "relationship_feature_permissions": to_jsonable(dict(self.relationship_feature_permissions)),
            "composite_band": True,
            "preserve_member_interval_identity": True,
            "preserve_ceiling_interval": True,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self) -> str:
        return dumps_json(self.as_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BandCompositeSpec":
        obj = require_json_object(payload, context="BandCompositeSpec")
        obj.pop("artifact_kind", None)
        obj.pop("composite_band", None)
        obj.pop("preserve_member_interval_identity", None)
        obj.pop("preserve_ceiling_interval", None)
        if "ceiling_interval" not in obj and "ceiling_interval_min" in obj:
            obj["ceiling_interval"] = obj["ceiling_interval_min"]
        obj.pop("ceiling_interval_min", None)
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "BandCompositeSpec":
        return cls.from_dict(loads_json(text))


def default_band_composite_specs(
    *,
    allowed_feature_families_by_band: Mapping[str, Sequence[str]] | None = None,
    relationship_feature_permissions: Mapping[str, Any] | None = None,
) -> tuple[BandCompositeSpec, ...]:
    allowed = allowed_feature_families_by_band or {}
    permissions = relationship_feature_permissions or default_relationship_feature_permissions()
    return tuple(
        BandCompositeSpec(
            band=band,
            ceiling_interval=int(payload["ceiling_interval"]),
            member_intervals=payload["member_intervals"],
            output_cadence=int(payload["ceiling_interval"]),
            alignment_policy=ALIGNMENT_POLICY_CEILING_BOUNDARY,
            allowed_feature_families=allowed.get(band, ()),
            relationship_feature_permissions=permissions,
            metadata={"default_contract": "micro_meso_macro_composite_band_policy"},
        )
        for band, payload in DEFAULT_COMPOSITE_BAND_INTERVALS.items()
    )


def band_composite_registry_as_dict(specs: Sequence[BandCompositeSpec]) -> dict[str, Any]:
    normalized = _specs_by_band(specs)
    return {
        "schema_version": BAND_COMPOSITE_SCHEMA_VERSION,
        "artifact_kind": BAND_COMPOSITE_REGISTRY_ARTIFACT_KIND,
        "bands": {band: normalized[band].as_dict() for band in COMPOSITE_BANDS},
        "production_parquet_allowed": False,
        "production_promotion_allowed": False,
    }


def resolve_band_composite_spec(
    band: str | Enum,
    specs: Sequence[BandCompositeSpec] | None = None,
) -> BandCompositeSpec:
    band_value = _band(band)
    normalized = _specs_by_band(specs or default_band_composite_specs())
    try:
        return normalized[band_value]
    except KeyError as exc:
        raise ValueError(f"Unsupported composite band {band_value!r}; expected one of: {', '.join(COMPOSITE_BANDS)}") from exc


def validate_band_composite_specs(specs: Sequence[BandCompositeSpec | Mapping[str, Any]]) -> tuple[BandCompositeSpec, ...]:
    normalized = tuple(spec if isinstance(spec, BandCompositeSpec) else BandCompositeSpec.from_dict(spec) for spec in specs)
    _specs_by_band(normalized)
    return normalized


def validate_output_timestamps_align_to_ceiling(
    timestamps: Iterable[int | float | datetime],
    spec: BandCompositeSpec | Mapping[str, Any],
    *,
    timestamp_unit: str = "seconds",
) -> None:
    normalized = spec if isinstance(spec, BandCompositeSpec) else BandCompositeSpec.from_dict(spec)
    divisor = _timestamp_divisor(normalized.ceiling_interval, timestamp_unit=timestamp_unit)
    for value in timestamps:
        numeric = _timestamp_to_numeric(value)
        remainder = abs(math.fmod(numeric, divisor)) if math.isfinite(numeric) else math.nan
        if not math.isfinite(numeric) or min(remainder, abs(divisor - remainder)) > 1e-9:
            raise ValueError("Band composite output timestamps must align to ceiling interval")


def validate_source_availability_for_band(
    spec: BandCompositeSpec | Mapping[str, Any],
    available_intervals: Sequence[int],
) -> None:
    normalized = spec if isinstance(spec, BandCompositeSpec) else BandCompositeSpec.from_dict(spec)
    available = set(_interval_tuple(available_intervals, field_name="available_intervals"))
    missing = sorted(set(normalized.member_intervals).difference(available))
    if missing:
        raise ValueError(f"Band composite source availability missing member intervals: {missing}")
    if max(normalized.member_intervals) > max(available):
        raise ValueError("Band composite member interval may not outrun source availability")


def _specs_by_band(specs: Sequence[BandCompositeSpec]) -> dict[str, BandCompositeSpec]:
    by_band = {spec.band: spec for spec in specs}
    missing = sorted(set(COMPOSITE_BANDS).difference(by_band))
    if missing:
        raise ValueError(f"Composite band registry missing bands: {missing}")
    return {band: by_band[band] for band in COMPOSITE_BANDS}


def _relationship_permissions(payload: Mapping[str, Any]) -> dict[str, Any]:
    merged = default_relationship_feature_permissions()
    for key, value in dict(payload).items():
        text_key = _text(key, field_name="relationship_feature_permission_key")
        if not isinstance(value, Mapping):
            raise ValueError("relationship_feature_permissions values must be mappings")
        base = dict(merged.get(text_key, {}))
        base.update(dict(value))
        merged[text_key] = base
    for family in (PAIRWISE_RELATIONSHIP_FEATURES, CROSS_ASSET_SUMMARY_FEATURES):
        permissions = merged[family]
        if permissions.get("auto_inherit_member_intervals") is not False:
            raise ValueError(f"{family} must not auto-inherit composite member intervals")
        if permissions.get("execution_enabled") is not False:
            raise ValueError(f"{family} execution must remain gated")
        if permissions.get("short_interval_execution_enabled") is not False:
            raise ValueError(f"{family} short intervals must remain gated")
    return to_jsonable(merged)


def _band(value: str | Enum) -> str:
    raw = value.value if isinstance(value, Enum) else value
    text = _text(raw, field_name="band").lower()
    if text not in COMPOSITE_BANDS:
        raise ValueError(f"Unsupported composite band {text!r}; expected one of: {', '.join(COMPOSITE_BANDS)}")
    return text


def _interval_tuple(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    intervals = tuple(dict.fromkeys(_positive_int(value, field_name=field_name) for value in values))
    if not intervals:
        raise ValueError(f"{field_name} must be non-empty")
    return intervals


def _string_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_text(value, field_name=field_name) for value in values))


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _timestamp_divisor(ceiling_interval: int, *, timestamp_unit: str) -> float:
    unit = _text(timestamp_unit, field_name="timestamp_unit").lower()
    if unit == "seconds":
        return float(ceiling_interval * 60)
    if unit == "minutes":
        return float(ceiling_interval)
    raise ValueError("timestamp_unit must be either 'seconds' or 'minutes'")


def _timestamp_to_numeric(value: int | float | datetime) -> float:
    if isinstance(value, datetime):
        return float(value.timestamp())
    return float(value)


__all__ = [
    "ALIGNMENT_POLICY_CEILING_BOUNDARY",
    "BAND_COMPOSITE_REGISTRY_ARTIFACT_KIND",
    "BAND_COMPOSITE_SCHEMA_VERSION",
    "BAND_COMPOSITE_SPEC_ARTIFACT_KIND",
    "COMPOSITE_BANDS",
    "CROSS_ASSET_SUMMARY_FEATURES",
    "DEFAULT_COMPOSITE_BAND_INTERVALS",
    "PAIRWISE_RELATIONSHIP_FEATURES",
    "BandCompositeSpec",
    "band_composite_registry_as_dict",
    "default_band_composite_specs",
    "default_relationship_feature_permissions",
    "resolve_band_composite_spec",
    "validate_band_composite_specs",
    "validate_output_timestamps_align_to_ceiling",
    "validate_source_availability_for_band",
]
