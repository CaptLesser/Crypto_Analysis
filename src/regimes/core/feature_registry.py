from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    REGIME_LAYER_AXIS_VALUES,
    RegimeAxis,
    RegimeBand,
    RegimeLayer,
    normalize_enum_value,
    normalize_string_tuple,
    require_non_empty_string,
    require_schema_version,
    validate_layer_axis_band,
)
from src.regimes.core.serialization import dumps_json, loads_json, require_known_fields, require_json_object, to_jsonable


REGIME_FEATURE_FAMILY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
FEATURE_FAMILY_IMPLEMENTATION_STATUSES: tuple[str, ...] = ("metadata_only", "implemented", "deprecated")
FEATURE_FAMILY_LEAKAGE_POLICIES: tuple[str, ...] = (
    "train_window_only",
    "source_features_only",
    "no_forward_target_columns",
    "declared_metadata_only",
)


def _normalize_allowed_values(values: Sequence[object], enum_type: type, *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(normalize_enum_value(value, enum_type, field_name=field_name) for value in values)
    if not normalized:
        raise ValueError(f"Regime feature family {field_name} must include at least one value")
    return tuple(dict.fromkeys(normalized))


def _require_policy(payload: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    policy = dict(payload)
    if "policy" not in policy or not str(policy["policy"]).strip():
        raise ValueError(f"Regime feature family {field_name} requires policy")
    if "fail_closed" not in policy:
        raise ValueError(f"Regime feature family {field_name} requires fail_closed")
    policy["policy"] = str(policy["policy"]).strip()
    policy["fail_closed"] = bool(policy["fail_closed"])
    if not policy["fail_closed"]:
        raise ValueError(f"Regime feature family {field_name} must fail closed")
    if "max_missing_fraction" in policy and policy["max_missing_fraction"] is not None:
        fraction = float(policy["max_missing_fraction"])
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"Regime feature family {field_name}.max_missing_fraction must be between 0 and 1")
        policy["max_missing_fraction"] = fraction
    return policy


def _require_lineage(payload: Mapping[str, Any]) -> dict[str, Any]:
    lineage = dict(payload)
    for field_name in ("artifact_kind", "artifact_path", "produced_by"):
        if field_name not in lineage or not str(lineage[field_name]).strip():
            raise ValueError(f"Regime feature family lineage_metadata requires {field_name}")
        lineage[field_name] = str(lineage[field_name]).strip()
    lineage.setdefault("schema_version", REGIME_FEATURE_FAMILY_SCHEMA_VERSION)
    return lineage


@dataclass(frozen=True)
class FeatureFamilySpec:
    family_name: str
    compatible_layers: Sequence[str | RegimeLayer]
    compatible_axes: Sequence[str | RegimeAxis]
    compatible_bands: Sequence[str | RegimeBand]
    required_source_columns: Sequence[str]
    missingness_policy: Mapping[str, Any]
    leakage_policy: str
    lineage_metadata: Mapping[str, Any]
    implementation_status: str = "metadata_only"
    schema_version: int = REGIME_FEATURE_FAMILY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family_name = require_non_empty_string(self.family_name, field_name="feature family name").lower()
        layers = _normalize_allowed_values(self.compatible_layers, RegimeLayer, field_name="compatible_layers")
        axes = _normalize_allowed_values(self.compatible_axes, RegimeAxis, field_name="compatible_axes")
        bands = _normalize_allowed_values(self.compatible_bands, RegimeBand, field_name="compatible_bands")
        source_columns = normalize_string_tuple(
            self.required_source_columns,
            field_name="required_source_columns",
            require_non_empty=True,
        )
        for layer in layers:
            for axis in axes:
                if axis in REGIME_LAYER_AXIS_VALUES[layer]:
                    continue
                if len(layers) == 1:
                    validate_layer_axis_band(layer=layer, axis=axis, band=bands[0])
        for layer in layers:
            if not any(axis in REGIME_LAYER_AXIS_VALUES[layer] for axis in axes):
                valid_text = ", ".join(REGIME_LAYER_AXIS_VALUES[layer])
                raise ValueError(f"Regime feature family has no compatible axis for {layer}; expected one of: {valid_text}")
        leakage_policy = require_non_empty_string(self.leakage_policy, field_name="leakage_policy").lower()
        if leakage_policy not in FEATURE_FAMILY_LEAKAGE_POLICIES:
            valid_text = ", ".join(FEATURE_FAMILY_LEAKAGE_POLICIES)
            raise ValueError(f"Unsupported Regime feature family leakage_policy {leakage_policy!r}; expected one of: {valid_text}")
        implementation_status = require_non_empty_string(
            self.implementation_status,
            field_name="implementation_status",
        ).lower()
        if implementation_status not in FEATURE_FAMILY_IMPLEMENTATION_STATUSES:
            valid_text = ", ".join(FEATURE_FAMILY_IMPLEMENTATION_STATUSES)
            raise ValueError(
                f"Unsupported Regime feature family implementation_status {implementation_status!r}; expected one of: {valid_text}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family_name)
        object.__setattr__(self, "compatible_layers", layers)
        object.__setattr__(self, "compatible_axes", axes)
        object.__setattr__(self, "compatible_bands", bands)
        object.__setattr__(self, "required_source_columns", source_columns)
        object.__setattr__(self, "missingness_policy", _require_policy(self.missingness_policy, field_name="missingness_policy"))
        object.__setattr__(self, "leakage_policy", leakage_policy)
        object.__setattr__(self, "lineage_metadata", _require_lineage(self.lineage_metadata))
        object.__setattr__(self, "implementation_status", implementation_status)

    def is_compatible(self, *, layer: str, axis: str, band: str) -> bool:
        layer_value = normalize_enum_value(layer, RegimeLayer, field_name="layer")
        axis_value = normalize_enum_value(axis, RegimeAxis, field_name="axis")
        band_value = normalize_enum_value(band, RegimeBand, field_name="band")
        if layer_value not in self.compatible_layers:
            return False
        if axis_value not in self.compatible_axes:
            return False
        if band_value not in self.compatible_bands:
            return False
        validate_layer_axis_band(layer=layer_value, axis=axis_value, band=band_value)
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family_name": self.family_name,
            "compatible_layers": list(self.compatible_layers),
            "compatible_axes": list(self.compatible_axes),
            "compatible_bands": list(self.compatible_bands),
            "required_source_columns": list(self.required_source_columns),
            "missingness_policy": to_jsonable(self.missingness_policy),
            "leakage_policy": self.leakage_policy,
            "lineage_metadata": to_jsonable(self.lineage_metadata),
            "implementation_status": self.implementation_status,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureFamilySpec":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "family_name",
                "compatible_layers",
                "compatible_axes",
                "compatible_bands",
                "required_source_columns",
                "missingness_policy",
                "leakage_policy",
                "lineage_metadata",
                "implementation_status",
            },
            optional=set(),
            context="Regime FeatureFamilySpec",
        )
        return cls(
            schema_version=obj["schema_version"],
            family_name=obj["family_name"],
            compatible_layers=obj["compatible_layers"],
            compatible_axes=obj["compatible_axes"],
            compatible_bands=obj["compatible_bands"],
            required_source_columns=obj["required_source_columns"],
            missingness_policy=obj["missingness_policy"],
            leakage_policy=obj["leakage_policy"],
            lineage_metadata=obj["lineage_metadata"],
            implementation_status=obj["implementation_status"],
        )

    @classmethod
    def from_json(cls, text: str) -> "FeatureFamilySpec":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FeatureFamilySpec JSON"))


@dataclass(frozen=True)
class FeatureFamilyRegistry:
    families: Mapping[str, FeatureFamilySpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, FeatureFamilySpec] = {}
        for name, spec in self.families.items():
            if not isinstance(spec, FeatureFamilySpec):
                spec = FeatureFamilySpec.from_dict(spec)  # type: ignore[arg-type]
            if str(name).strip().lower() != spec.family_name:
                raise ValueError("Regime feature registry keys must match FeatureFamilySpec.family_name")
            normalized[spec.family_name] = spec
        object.__setattr__(self, "families", dict(sorted(normalized.items())))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.families)

    def get(self, family_name: str) -> FeatureFamilySpec:
        key = require_non_empty_string(family_name, field_name="feature family name").lower()
        try:
            return self.families[key]
        except KeyError as exc:
            valid_text = ", ".join(self.names)
            raise ValueError(f"Unsupported Regime feature family {key!r}; expected one of: {valid_text}") from exc

    def register(self, spec: FeatureFamilySpec, *, replace: bool = False) -> "FeatureFamilyRegistry":
        if spec.family_name in self.families and not replace:
            raise ValueError(f"Regime feature family {spec.family_name!r} is already registered")
        next_families = dict(self.families)
        next_families[spec.family_name] = spec
        return FeatureFamilyRegistry(next_families)

    def query(
        self,
        *,
        layer: str | None = None,
        axis: str | None = None,
        band: str | None = None,
        implementation_status: str | None = None,
    ) -> tuple[FeatureFamilySpec, ...]:
        if implementation_status is not None:
            status = require_non_empty_string(implementation_status, field_name="implementation_status").lower()
        else:
            status = None
        results = []
        for spec in self.families.values():
            if status is not None and spec.implementation_status != status:
                continue
            if layer is not None and normalize_enum_value(layer, RegimeLayer, field_name="layer") not in spec.compatible_layers:
                continue
            if axis is not None and normalize_enum_value(axis, RegimeAxis, field_name="axis") not in spec.compatible_axes:
                continue
            if band is not None and normalize_enum_value(band, RegimeBand, field_name="band") not in spec.compatible_bands:
                continue
            if layer is not None and axis is not None and band is not None:
                if not spec.is_compatible(layer=layer, axis=axis, band=band):
                    continue
            results.append(spec)
        return tuple(sorted(results, key=lambda item: item.family_name))

    def as_dict(self) -> dict[str, Any]:
        return {name: spec.as_dict() for name, spec in self.families.items()}


def _asset_metadata_spec(axis: str, columns: Sequence[str]) -> FeatureFamilySpec:
    return FeatureFamilySpec(
        family_name=f"asset_state_{axis}_metadata_only",
        compatible_layers=(RegimeLayer.ASSET_STATE,),
        compatible_axes=(axis,),
        compatible_bands=(RegimeBand.MICRO, RegimeBand.MESO, RegimeBand.MACRO),
        required_source_columns=columns,
        missingness_policy={
            "policy": "train_window_feature_filter_then_drop_nonfinite_rows",
            "max_missing_fraction": 0.2,
            "fail_closed": True,
        },
        leakage_policy="train_window_only",
        lineage_metadata={
            "artifact_kind": "regime_feature_family_declaration",
            "artifact_path": "src/regimes/core/feature_registry.py",
            "produced_by": "default_asset_state_feature_family_registry",
            "axis": axis,
        },
        implementation_status="metadata_only",
    )


def default_asset_state_feature_family_specs() -> tuple[FeatureFamilySpec, ...]:
    return (
        _asset_metadata_spec(
            "activity",
            ("trade_intensity", "avg_trade_size", "vroc_14", "prr"),
        ),
        _asset_metadata_spec(
            "trend",
            ("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"),
        ),
        _asset_metadata_spec(
            "vol",
            ("atr_14", "ret_std_20", "cv_20", "vol_osc_pct_14_28"),
        ),
    )


def default_feature_family_registry() -> FeatureFamilyRegistry:
    registry = FeatureFamilyRegistry()
    for spec in default_asset_state_feature_family_specs():
        registry = registry.register(spec)
    return registry


__all__ = [
    "FEATURE_FAMILY_IMPLEMENTATION_STATUSES",
    "FEATURE_FAMILY_LEAKAGE_POLICIES",
    "FeatureFamilyRegistry",
    "FeatureFamilySpec",
    "REGIME_FEATURE_FAMILY_SCHEMA_VERSION",
    "default_asset_state_feature_family_specs",
    "default_feature_family_registry",
]
