from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.market_state.contracts import MARKET_STATE_BAND_VALUES
from src.regimes.regime_features.contracts import REGIME_FEATURES_SCHEMA_VERSION
from src.regimes.regime_features.market import (
    MARKET_FEATURE_FAMILY_BREADTH,
    MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY,
    MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY,
    MARKET_FEATURE_FAMILY_DISPERSION,
    MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY,
    MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY,
    MARKET_FEATURE_FAMILY_RETURN_SUMMARY,
    MARKET_FEATURE_FAMILY_STRESS,
    PRIMITIVE_MARKET_FEATURE_COLUMNS_BY_FAMILY,
    PRIMITIVE_MARKET_FEATURE_FAMILIES,
    STATUS_UNAVAILABLE_MISSING_TRADES,
    STATUS_UNAVAILABLE_MISSING_VOLUME,
    UNAVAILABLE_MARKET_FEATURE_STATES,
)


MARKET_FEATURE_FAMILY_REGISTRY_ID = "primitive_market_regime_feature_family_registry_v1"
MARKET_FEATURE_FAMILY_DECLARATION_ARTIFACT_KIND = "primitive_market_regime_feature_family_spec"
MARKET_FEATURE_FAMILY_REGISTRY_ARTIFACT_KIND = "primitive_market_regime_feature_family_registry"

FEATURE_FAMILY_INPUT_STATUS_USABLE = "usable"
FEATURE_FAMILY_INPUT_STATUS_USABLE_WITH_UNAVAILABLE_OPTIONAL_INPUTS = "usable_with_unavailable_optional_inputs"
FEATURE_FAMILY_INPUT_STATUS_BLOCKED_MISSING_REQUIRED_INPUTS = "blocked_missing_required_inputs"
FEATURE_FAMILY_INPUT_STATUSES: tuple[str, ...] = (
    FEATURE_FAMILY_INPUT_STATUS_USABLE,
    FEATURE_FAMILY_INPUT_STATUS_USABLE_WITH_UNAVAILABLE_OPTIONAL_INPUTS,
    FEATURE_FAMILY_INPUT_STATUS_BLOCKED_MISSING_REQUIRED_INPUTS,
)

DEFAULT_KNOWN_AT_REQUIREMENTS: Mapping[str, Any] = {
    "requires_known_at_ts": True,
    "requires_source_tail_ts": True,
    "requires_feature_available_at_ts": True,
    "requires_no_lookahead_verified": True,
    "known_at_must_not_precede_source_tail": True,
}
DEFAULT_LINEAGE_REQUIREMENTS: Mapping[str, Any] = {
    "requires_artifact_family": True,
    "requires_feature_set_id": True,
    "requires_source_data_kinds": True,
    "requires_source_partition_lineage": True,
    "requires_universe_snapshot_id_or_hash": True,
    "requires_calculation_policy": True,
    "production_outputs_written": False,
}


@dataclass(frozen=True)
class PrimitiveMarketFeatureFamilySpec:
    feature_family_id: str
    purpose: str
    source_requirements: Mapping[str, Any]
    ohlcvt_required_fields: Sequence[str]
    ohlcvt_optional_fields: Sequence[str] = ()
    scalar_feature_required_fields: Sequence[str] = ()
    scalar_feature_optional_fields: Sequence[str] = ()
    calculation_windows: Mapping[str, Any] = field(default_factory=dict)
    compatible_bands: Sequence[str] = MARKET_STATE_BAND_VALUES
    known_at_requirements: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_KNOWN_AT_REQUIREMENTS))
    lineage_requirements: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_LINEAGE_REQUIREMENTS))
    unavailable_values_allowed: bool = True
    unavailable_feature_states: Sequence[str] = UNAVAILABLE_MARKET_FEATURE_STATES
    output_columns: Sequence[str] = ()
    implementation_reference: str = "src.regimes.regime_features.market.build_primitive_market_regime_features"
    candidate_feature_layer: bool = True
    final_selection_claimed: bool = False
    market_state_clustering_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family_id = _member(self.feature_family_id, PRIMITIVE_MARKET_FEATURE_FAMILIES, field_name="feature_family_id")
        bands = _string_tuple(self.compatible_bands, field_name="compatible_bands", require_non_empty=True)
        unknown_bands = sorted(set(bands).difference(MARKET_STATE_BAND_VALUES))
        if unknown_bands:
            raise ValueError(f"Primitive market feature family compatible_bands contains unsupported values: {unknown_bands}")
        output_columns = _string_tuple(
            self.output_columns or PRIMITIVE_MARKET_FEATURE_COLUMNS_BY_FAMILY[family_id],
            field_name="output_columns",
            require_non_empty=True,
        )
        expected = tuple(PRIMITIVE_MARKET_FEATURE_COLUMNS_BY_FAMILY[family_id])
        if tuple(output_columns) != expected:
            raise ValueError(f"Primitive market feature family {family_id!r} output_columns must match implemented primitive columns")
        if self.candidate_feature_layer is not True:
            raise ValueError("Primitive market feature family declarations must remain candidate_feature_layer=True")
        if self.final_selection_claimed is not False:
            raise ValueError("Primitive market feature family declarations cannot claim final selected features")
        if self.market_state_clustering_enabled is not False:
            raise ValueError("Primitive market feature family declarations cannot enable Market-State clustering")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "feature_family_id", family_id)
        object.__setattr__(self, "purpose", _text(self.purpose, field_name="purpose"))
        object.__setattr__(self, "source_requirements", to_jsonable(dict(self.source_requirements)))
        object.__setattr__(self, "ohlcvt_required_fields", _string_tuple(self.ohlcvt_required_fields, field_name="ohlcvt_required_fields", require_non_empty=True))
        object.__setattr__(self, "ohlcvt_optional_fields", _string_tuple(self.ohlcvt_optional_fields, field_name="ohlcvt_optional_fields", require_non_empty=False))
        object.__setattr__(self, "scalar_feature_required_fields", _string_tuple(self.scalar_feature_required_fields, field_name="scalar_feature_required_fields", require_non_empty=False))
        object.__setattr__(self, "scalar_feature_optional_fields", _string_tuple(self.scalar_feature_optional_fields, field_name="scalar_feature_optional_fields", require_non_empty=False))
        object.__setattr__(self, "calculation_windows", to_jsonable(dict(self.calculation_windows)))
        object.__setattr__(self, "compatible_bands", bands)
        object.__setattr__(self, "known_at_requirements", to_jsonable(dict(self.known_at_requirements)))
        object.__setattr__(self, "lineage_requirements", to_jsonable(dict(self.lineage_requirements)))
        object.__setattr__(self, "unavailable_values_allowed", bool(self.unavailable_values_allowed))
        object.__setattr__(self, "unavailable_feature_states", _string_tuple(self.unavailable_feature_states, field_name="unavailable_feature_states", require_non_empty=False))
        object.__setattr__(self, "output_columns", output_columns)
        object.__setattr__(self, "implementation_reference", _text(self.implementation_reference, field_name="implementation_reference"))
        object.__setattr__(self, "candidate_feature_layer", True)
        object.__setattr__(self, "final_selection_claimed", False)
        object.__setattr__(self, "market_state_clustering_enabled", False)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def supports_band(self, band: str) -> bool:
        return str(band).strip().lower() in set(self.compatible_bands)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": MARKET_FEATURE_FAMILY_DECLARATION_ARTIFACT_KIND,
            "feature_family_id": self.feature_family_id,
            "purpose": self.purpose,
            "source_requirements": to_jsonable(dict(self.source_requirements)),
            "ohlcvt_required_fields": list(self.ohlcvt_required_fields),
            "ohlcvt_optional_fields": list(self.ohlcvt_optional_fields),
            "scalar_feature_required_fields": list(self.scalar_feature_required_fields),
            "scalar_feature_optional_fields": list(self.scalar_feature_optional_fields),
            "calculation_windows": to_jsonable(dict(self.calculation_windows)),
            "compatible_bands": list(self.compatible_bands),
            "known_at_requirements": to_jsonable(dict(self.known_at_requirements)),
            "lineage_requirements": to_jsonable(dict(self.lineage_requirements)),
            "unavailable_values_allowed": bool(self.unavailable_values_allowed),
            "unavailable_feature_states": list(self.unavailable_feature_states),
            "output_columns": list(self.output_columns),
            "implementation_reference": self.implementation_reference,
            "candidate_feature_layer": True,
            "final_selection_claimed": False,
            "market_state_clustering_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveMarketFeatureFamilySpec":
        obj = require_json_object(payload, context="PrimitiveMarketFeatureFamilySpec")
        obj.pop("artifact_kind", None)
        return cls(**obj)


@dataclass(frozen=True)
class PrimitiveMarketFeatureFamilyInputResolution:
    feature_family_id: str
    status: str
    usable: bool
    missing_required_ohlcvt_fields: Sequence[str] = ()
    missing_optional_ohlcvt_fields: Sequence[str] = ()
    missing_required_scalar_feature_fields: Sequence[str] = ()
    missing_optional_scalar_feature_fields: Sequence[str] = ()
    unavailable_feature_states: Sequence[str] = ()

    def __post_init__(self) -> None:
        status = _member(self.status, FEATURE_FAMILY_INPUT_STATUSES, field_name="status")
        object.__setattr__(self, "feature_family_id", _member(self.feature_family_id, PRIMITIVE_MARKET_FEATURE_FAMILIES, field_name="feature_family_id"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "usable", bool(self.usable))
        object.__setattr__(self, "missing_required_ohlcvt_fields", _string_tuple(self.missing_required_ohlcvt_fields, field_name="missing_required_ohlcvt_fields", require_non_empty=False))
        object.__setattr__(self, "missing_optional_ohlcvt_fields", _string_tuple(self.missing_optional_ohlcvt_fields, field_name="missing_optional_ohlcvt_fields", require_non_empty=False))
        object.__setattr__(self, "missing_required_scalar_feature_fields", _string_tuple(self.missing_required_scalar_feature_fields, field_name="missing_required_scalar_feature_fields", require_non_empty=False))
        object.__setattr__(self, "missing_optional_scalar_feature_fields", _string_tuple(self.missing_optional_scalar_feature_fields, field_name="missing_optional_scalar_feature_fields", require_non_empty=False))
        object.__setattr__(self, "unavailable_feature_states", _string_tuple(self.unavailable_feature_states, field_name="unavailable_feature_states", require_non_empty=False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_family_id": self.feature_family_id,
            "status": self.status,
            "usable": bool(self.usable),
            "missing_required_ohlcvt_fields": list(self.missing_required_ohlcvt_fields),
            "missing_optional_ohlcvt_fields": list(self.missing_optional_ohlcvt_fields),
            "missing_required_scalar_feature_fields": list(self.missing_required_scalar_feature_fields),
            "missing_optional_scalar_feature_fields": list(self.missing_optional_scalar_feature_fields),
            "unavailable_feature_states": list(self.unavailable_feature_states),
        }


@dataclass(frozen=True)
class PrimitiveMarketFeatureFamilyRegistry:
    families: Mapping[str, PrimitiveMarketFeatureFamilySpec | Mapping[str, Any]]
    registry_id: str = MARKET_FEATURE_FAMILY_REGISTRY_ID
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        families: dict[str, PrimitiveMarketFeatureFamilySpec] = {}
        for key, value in self.families.items():
            spec = value if isinstance(value, PrimitiveMarketFeatureFamilySpec) else PrimitiveMarketFeatureFamilySpec.from_dict(value)
            if str(key) != spec.feature_family_id:
                raise ValueError("Primitive market feature family registry keys must match feature_family_id")
            families[spec.feature_family_id] = spec
        missing = sorted(set(PRIMITIVE_MARKET_FEATURE_FAMILIES).difference(families))
        if missing:
            raise ValueError(f"Primitive market feature family registry missing families: {missing}")
        object.__setattr__(self, "families", {family_id: families[family_id] for family_id in PRIMITIVE_MARKET_FEATURE_FAMILIES})
        object.__setattr__(self, "registry_id", _text(self.registry_id, field_name="registry_id"))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(self.families)

    @property
    def output_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for spec in self.families.values():
            columns.extend(spec.output_columns)
        return tuple(dict.fromkeys(columns))

    def get(self, feature_family_id: str) -> PrimitiveMarketFeatureFamilySpec:
        key = _member(feature_family_id, PRIMITIVE_MARKET_FEATURE_FAMILIES, field_name="feature_family_id")
        return self.families[key]

    def resolve_inputs(
        self,
        *,
        available_ohlcvt_fields: Sequence[str],
        available_scalar_feature_fields: Sequence[str] = (),
    ) -> tuple[PrimitiveMarketFeatureFamilyInputResolution, ...]:
        ohlcvt = set(_string_tuple(available_ohlcvt_fields, field_name="available_ohlcvt_fields", require_non_empty=False))
        scalar = set(_string_tuple(available_scalar_feature_fields, field_name="available_scalar_feature_fields", require_non_empty=False))
        resolutions: list[PrimitiveMarketFeatureFamilyInputResolution] = []
        for spec in self.families.values():
            missing_required_ohlcvt = tuple(field for field in spec.ohlcvt_required_fields if field not in ohlcvt)
            missing_optional_ohlcvt = tuple(field for field in spec.ohlcvt_optional_fields if field not in ohlcvt)
            missing_required_scalar = tuple(field for field in spec.scalar_feature_required_fields if field not in scalar)
            missing_optional_scalar = tuple(field for field in spec.scalar_feature_optional_fields if field not in scalar)
            missing_required = bool(missing_required_ohlcvt or missing_required_scalar)
            missing_optional = bool(missing_optional_ohlcvt or missing_optional_scalar)
            if missing_required:
                status = FEATURE_FAMILY_INPUT_STATUS_BLOCKED_MISSING_REQUIRED_INPUTS
                usable = False
            elif missing_optional:
                status = FEATURE_FAMILY_INPUT_STATUS_USABLE_WITH_UNAVAILABLE_OPTIONAL_INPUTS
                usable = True
            else:
                status = FEATURE_FAMILY_INPUT_STATUS_USABLE
                usable = True
            resolutions.append(
                PrimitiveMarketFeatureFamilyInputResolution(
                    feature_family_id=spec.feature_family_id,
                    status=status,
                    usable=usable,
                    missing_required_ohlcvt_fields=missing_required_ohlcvt,
                    missing_optional_ohlcvt_fields=missing_optional_ohlcvt,
                    missing_required_scalar_feature_fields=missing_required_scalar,
                    missing_optional_scalar_feature_fields=missing_optional_scalar,
                    unavailable_feature_states=spec.unavailable_feature_states if missing_optional and spec.unavailable_values_allowed else (),
                )
            )
        return tuple(resolutions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": MARKET_FEATURE_FAMILY_REGISTRY_ARTIFACT_KIND,
            "registry_id": self.registry_id,
            "family_count": int(len(self.families)),
            "families": {family_id: spec.as_dict() for family_id, spec in self.families.items()},
            "output_columns": list(self.output_columns),
            "candidate_feature_layer": True,
            "final_selection_claimed": False,
            "market_state_clustering_enabled": False,
            "production_outputs_written": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveMarketFeatureFamilyRegistry":
        obj = require_json_object(payload, context="PrimitiveMarketFeatureFamilyRegistry")
        obj.pop("artifact_kind", None)
        obj.pop("family_count", None)
        obj.pop("output_columns", None)
        obj.pop("candidate_feature_layer", None)
        obj.pop("final_selection_claimed", None)
        obj.pop("market_state_clustering_enabled", None)
        obj.pop("production_outputs_written", None)
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "PrimitiveMarketFeatureFamilyRegistry":
        return cls.from_dict(require_json_object(loads_json(text), context="PrimitiveMarketFeatureFamilyRegistry JSON"))


def default_primitive_market_feature_family_registry() -> PrimitiveMarketFeatureFamilyRegistry:
    families = (
        _family(
            MARKET_FEATURE_FAMILY_RETURN_SUMMARY,
            purpose="Candidate market-level return central-tendency and tail-summary primitives.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("log_return",),
            calculation_windows={"point_in_time_cross_section": True, "rolling_window_configurable": False},
        ),
        _family(
            MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY,
            purpose="Candidate market and asset-level realized volatility summaries.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("ret_std_20", "atr_14", "parkinson_vol_20", "garman_klass_vol_20", "rogers_satchell_vol_20"),
            calculation_windows={"rolling_window": 20, "min_periods": 3, "configurable_by": "PrimitiveMarketFeatureConfig"},
        ),
        _family(
            MARKET_FEATURE_FAMILY_BREADTH,
            purpose="Candidate broad-universe participation breadth and up/down return share primitives.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("log_return", "activity_state_score_20"),
            calculation_windows={"point_in_time_cross_section": True, "rolling_window_configurable": False},
        ),
        _family(
            MARKET_FEATURE_FAMILY_DISPERSION,
            purpose="Candidate cross-sectional return spread and dispersion primitives.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("log_return", "ret_std_20", "range_efficiency_20"),
            calculation_windows={"point_in_time_cross_section": True, "rolling_window_configurable": False},
        ),
        _family(
            MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY,
            purpose="Candidate core-basket rolling pairwise correlation summary primitives, without emitting raw pairwise matrices.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("log_return",),
            calculation_windows={"rolling_window": 20, "min_periods": 3, "configurable_by": "PrimitiveMarketFeatureConfig"},
            metadata={"implemented_by": "src.regimes.regime_features.market._rolling_correlation_summary"},
        ),
        _family(
            MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY,
            purpose="Candidate core-basket rolling covariance and optional shrinkage summary primitives.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("log_return",),
            calculation_windows={"rolling_window": 20, "min_periods": 3, "configurable_by": "PrimitiveMarketFeatureConfig"},
            metadata={
                "implemented_by": "src.regimes.regime_features.market._rolling_covariance_summary",
                "optional_dependency": "sklearn.covariance",
            },
        ),
        _family(
            MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY,
            purpose="Candidate aggregate volume/trade activity, breadth, and concentration primitives.",
            ohlcvt_required=("ts", "asset", "close"),
            ohlcvt_optional=("volume", "trades"),
            scalar_optional=("volume_zscore_20", "trades_zscore_20", "dollar_volume_proxy", "activity_state_score_20"),
            calculation_windows={"point_in_time_cross_section": True, "rolling_window_configurable": False},
            unavailable_states=(STATUS_UNAVAILABLE_MISSING_VOLUME, STATUS_UNAVAILABLE_MISSING_TRADES),
        ),
        _family(
            MARKET_FEATURE_FAMILY_STRESS,
            purpose="Candidate downside participation, high-volatility share, and high-correlation/high-volatility coincidence primitives.",
            ohlcvt_required=("ts", "asset", "close"),
            scalar_optional=("downside_vol_20", "negative_return_share_20", "high_vol_downside_pressure_20"),
            calculation_windows={"rolling_window": 20, "min_periods": 3, "configurable_by": "PrimitiveMarketFeatureConfig"},
        ),
    )
    return PrimitiveMarketFeatureFamilyRegistry(
        families={family.feature_family_id: family for family in families},
        metadata={
            "implemented_primitive_builder": "src.regimes.regime_features.market.build_primitive_market_regime_features",
            "candidate_oriented": True,
            "final_market_state_selection": False,
        },
    )


def _family(
    feature_family_id: str,
    *,
    purpose: str,
    ohlcvt_required: Sequence[str],
    ohlcvt_optional: Sequence[str] = (),
    scalar_required: Sequence[str] = (),
    scalar_optional: Sequence[str] = (),
    calculation_windows: Mapping[str, Any],
    unavailable_states: Sequence[str] = UNAVAILABLE_MARKET_FEATURE_STATES,
    metadata: Mapping[str, Any] | None = None,
) -> PrimitiveMarketFeatureFamilySpec:
    return PrimitiveMarketFeatureFamilySpec(
        feature_family_id=feature_family_id,
        purpose=purpose,
        source_requirements={
            "source_frames": "bounded per-asset OHLCVT frames",
            "universe_scope": "active broad universe plus explicit core basket where required",
            "scalar_features_role": "optional enrichment/compatibility metadata for downstream Market-State consumption",
            "raw_pairwise_matrices_required": False,
        },
        ohlcvt_required_fields=ohlcvt_required,
        ohlcvt_optional_fields=ohlcvt_optional,
        scalar_feature_required_fields=scalar_required,
        scalar_feature_optional_fields=scalar_optional,
        calculation_windows=calculation_windows,
        compatible_bands=MARKET_STATE_BAND_VALUES,
        unavailable_feature_states=unavailable_states,
        output_columns=PRIMITIVE_MARKET_FEATURE_COLUMNS_BY_FAMILY[feature_family_id],
        metadata={
            "no_final_selection": True,
            "no_winner_claims": True,
            "market_state_clustering_not_performed": True,
            **dict(metadata or {}),
        },
    )


def _member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if text not in set(allowed):
        raise ValueError(f"Unsupported primitive market feature family {field_name} {text!r}; expected one of: {', '.join(allowed)}")
    return text


def _string_tuple(values: Sequence[object], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Primitive market feature family {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not out:
        raise ValueError(f"Primitive market feature family {field_name} must include at least one value")
    return tuple(dict.fromkeys(out))


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Primitive market feature family {field_name} must be non-empty")
    return text


__all__ = [
    "FEATURE_FAMILY_INPUT_STATUS_BLOCKED_MISSING_REQUIRED_INPUTS",
    "FEATURE_FAMILY_INPUT_STATUS_USABLE",
    "FEATURE_FAMILY_INPUT_STATUS_USABLE_WITH_UNAVAILABLE_OPTIONAL_INPUTS",
    "FEATURE_FAMILY_INPUT_STATUSES",
    "MARKET_FEATURE_FAMILY_DECLARATION_ARTIFACT_KIND",
    "MARKET_FEATURE_FAMILY_REGISTRY_ARTIFACT_KIND",
    "MARKET_FEATURE_FAMILY_REGISTRY_ID",
    "PrimitiveMarketFeatureFamilyInputResolution",
    "PrimitiveMarketFeatureFamilyRegistry",
    "PrimitiveMarketFeatureFamilySpec",
    "default_primitive_market_feature_family_registry",
]
