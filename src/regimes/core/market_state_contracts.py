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
from src.regimes.core.paths import (
    default_foundation_report_root,
    is_production_adjacent_path,
    is_relative_to,
    resolve_project_path,
    resolve_project_root,
)
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


MARKET_STATE_METADATA_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
MARKET_STATE_METADATA_ARTIFACT_KIND = "regime_market_state_metadata_manifest"
MARKET_STATE_FEATURE_FAMILY_ARTIFACT_KIND = "regime_market_state_feature_family_metadata"
MARKET_STATE_LAYER = RegimeLayer.MARKET_STATE.value

MARKET_STATE_FEATURE_FAMILY_NAMES: tuple[str, ...] = (
    "market_return_summary",
    "realized_volatility_summary",
    "breadth",
    "dispersion",
    "correlation_summary",
    "covariance_summary",
)
MARKET_STATE_SUMMARY_TYPES: tuple[str, ...] = (
    "return_summary",
    "realized_volatility_summary",
    "breadth",
    "dispersion",
    "correlation",
    "covariance",
)


def _token(value: object, *, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    return require_non_empty_string(value, field_name=field_name).lower()


def _unique_tokens(values: Sequence[object], *, field_name: str, require_non_empty: bool = True) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(_token(value, field_name=field_name) for value in values if str(value).strip()))
    if require_non_empty and not out:
        raise ValueError(f"Regime market-state metadata {field_name} must include at least one value")
    return out


def validate_market_state_metadata_report_root(
    report_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_project_path(report_root, project_root=project_root)
    project = resolve_project_root(project_root)
    if is_production_adjacent_path(root, project_root=project):
        raise ValueError("Regime market-state metadata report_root is production-adjacent and is not allowed")
    if is_relative_to(root, project / "reports"):
        classification = "project_report_root"
    elif is_relative_to(root, project / "logs" / "diagnostics"):
        classification = "project_diagnostics_root"
    elif any("pytest" in part.lower() or "tmp" in part.lower() or "temp" in part.lower() for part in root.parts):
        classification = "temporary_report_root"
    else:
        classification = "explicit_non_production_root"
    return {
        "root": str(root),
        "classification": classification,
        "allowed": True,
        "metadata_only": True,
        "production_writes_enabled": False,
        "production_adjacent_roots_rejected": True,
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
        "aggregation_execution_enabled": False,
        "production_readers_enabled": False,
    }


@dataclass(frozen=True)
class MarketStateFeatureFamilyDeclaration:
    family_name: str
    axis: str | RegimeAxis
    summary_type: str
    required_source_columns: Sequence[str]
    derived_feature_columns: Sequence[str]
    aggregation_grain: str = "universe_band_timestamp"
    lookback_windows: Sequence[int] = (20, 60)
    minimum_coverage_threshold: float = 0.8
    lineage_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_only: bool = True
    production_writes_enabled: bool = False
    schema_version: int = MARKET_STATE_METADATA_SCHEMA_VERSION
    artifact_kind: str = MARKET_STATE_FEATURE_FAMILY_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        family = _token(self.family_name, field_name="feature family name")
        if family not in MARKET_STATE_FEATURE_FAMILY_NAMES:
            valid = ", ".join(MARKET_STATE_FEATURE_FAMILY_NAMES)
            raise ValueError(f"Unsupported Regime market-state feature family {family!r}; expected one of: {valid}")
        axis = normalize_enum_value(self.axis, RegimeAxis, field_name="axis")
        validate_layer_axis_band(layer=MARKET_STATE_LAYER, axis=axis, band=RegimeBand.MICRO.value)
        summary_type = _token(self.summary_type, field_name="summary_type")
        if summary_type not in MARKET_STATE_SUMMARY_TYPES:
            valid = ", ".join(MARKET_STATE_SUMMARY_TYPES)
            raise ValueError(f"Unsupported Regime market-state summary_type {summary_type!r}; expected one of: {valid}")
        source_columns = normalize_string_tuple(
            self.required_source_columns,
            field_name="required_source_columns",
            require_non_empty=True,
        )
        derived_columns = normalize_string_tuple(
            self.derived_feature_columns,
            field_name="derived_feature_columns",
            require_non_empty=True,
        )
        lookbacks = tuple(int(window) for window in self.lookback_windows)
        if not lookbacks or any(window <= 0 for window in lookbacks):
            raise ValueError("Regime market-state lookback_windows must be positive")
        coverage = float(self.minimum_coverage_threshold)
        if coverage < 0.0 or coverage > 1.0:
            raise ValueError("Regime market-state minimum_coverage_threshold must be between 0 and 1")
        if self.metadata_only is not True:
            raise ValueError("Regime market-state metadata contracts must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime market-state metadata contracts cannot enable production writes")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "family_name", family)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "summary_type", summary_type)
        object.__setattr__(self, "required_source_columns", source_columns)
        object.__setattr__(self, "derived_feature_columns", derived_columns)
        object.__setattr__(self, "aggregation_grain", require_non_empty_string(self.aggregation_grain, field_name="aggregation_grain"))
        object.__setattr__(self, "lookback_windows", lookbacks)
        object.__setattr__(self, "minimum_coverage_threshold", coverage)
        object.__setattr__(
            self,
            "lineage_metadata",
            {
                "artifact_kind": "regime_market_state_metadata_contract",
                "produced_by": "src.regimes.core.market_state_contracts",
                **to_jsonable(require_json_mapping(self.lineage_metadata, field_name="lineage_metadata")),
            },
        )
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        return _metadata_boundary("market_state_feature_family_metadata_only")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": MARKET_STATE_LAYER,
            "family_name": self.family_name,
            "axis": self.axis,
            "summary_type": self.summary_type,
            "required_source_columns": list(self.required_source_columns),
            "derived_feature_columns": list(self.derived_feature_columns),
            "aggregation_grain": self.aggregation_grain,
            "lookback_windows": list(self.lookback_windows),
            "minimum_coverage_threshold": float(self.minimum_coverage_threshold),
            "metadata_only": True,
            "production_writes_enabled": False,
            "lineage_metadata": to_jsonable(self.lineage_metadata),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateFeatureFamilyDeclaration":
        obj = require_json_object(payload, context="Regime MarketStateFeatureFamilyDeclaration")
        return cls(
            schema_version=obj.get("schema_version", MARKET_STATE_METADATA_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", MARKET_STATE_FEATURE_FAMILY_ARTIFACT_KIND),
            family_name=obj["family_name"],
            axis=obj["axis"],
            summary_type=obj["summary_type"],
            required_source_columns=obj["required_source_columns"],
            derived_feature_columns=obj["derived_feature_columns"],
            aggregation_grain=obj.get("aggregation_grain", "universe_band_timestamp"),
            lookback_windows=obj.get("lookback_windows", (20, 60)),
            minimum_coverage_threshold=obj.get("minimum_coverage_threshold", 0.8),
            lineage_metadata=obj.get("lineage_metadata", {}),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
        )


@dataclass(frozen=True)
class MarketStateMetadataManifest:
    manifest_id: str
    universe: str
    bands: Sequence[str | RegimeBand]
    feature_families: Sequence[MarketStateFeatureFamilyDeclaration | Mapping[str, Any]]
    report_root: str | Path
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_only: bool = True
    production_writes_enabled: bool = False
    classification: str | RegimeClassification = RegimeClassification.METADATA_ONLY.value
    schema_version: int = MARKET_STATE_METADATA_SCHEMA_VERSION
    artifact_kind: str = MARKET_STATE_METADATA_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        manifest_id = require_non_empty_string(self.manifest_id, field_name="manifest_id")
        universe = require_non_empty_string(self.universe, field_name="universe")
        bands = tuple(dict.fromkeys(normalize_enum_value(band, RegimeBand, field_name="band") for band in self.bands))
        if not bands:
            raise ValueError("Regime market-state metadata bands must include at least one value")
        classification = normalize_enum_value(self.classification, RegimeClassification, field_name="classification")
        if classification != RegimeClassification.METADATA_ONLY.value:
            raise ValueError("Regime market-state metadata manifest classification must be metadata_only")
        families = tuple(
            family
            if isinstance(family, MarketStateFeatureFamilyDeclaration)
            else MarketStateFeatureFamilyDeclaration.from_dict(family)
            for family in self.feature_families
        )
        if not families:
            raise ValueError("Regime market-state metadata manifest requires feature families")
        if self.metadata_only is not True:
            raise ValueError("Regime market-state metadata manifest must declare metadata_only=true")
        if self.production_writes_enabled is not False:
            raise ValueError("Regime market-state metadata manifest cannot enable production writes")
        report_policy = validate_market_state_metadata_report_root(self.report_root)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "manifest_id", manifest_id)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "feature_families", families)
        object.__setattr__(self, "report_root", report_policy["root"])
        object.__setattr__(self, "metadata", require_json_mapping(self.metadata, field_name="metadata"))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "production_writes_enabled", False)

    @property
    def report_root_policy(self) -> dict[str, Any]:
        return validate_market_state_metadata_report_root(self.report_root)

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        return _metadata_boundary("market_state_metadata_manifest_json_only")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "manifest_id": self.manifest_id,
            "layer": MARKET_STATE_LAYER,
            "classification": self.classification,
            "universe": self.universe,
            "bands": list(self.bands),
            "metadata_only": True,
            "production_writes_enabled": False,
            "feature_families": [family.as_dict() for family in self.feature_families],
            "report_root": str(self.report_root),
            "report_root_policy": self.report_root_policy,
            "metadata": to_jsonable(self.metadata),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateMetadataManifest":
        obj = require_json_object(payload, context="Regime MarketStateMetadataManifest")
        return cls(
            schema_version=obj.get("schema_version", MARKET_STATE_METADATA_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", MARKET_STATE_METADATA_ARTIFACT_KIND),
            manifest_id=obj["manifest_id"],
            universe=obj["universe"],
            bands=obj["bands"],
            classification=obj.get("classification", RegimeClassification.METADATA_ONLY.value),
            feature_families=obj["feature_families"],
            report_root=obj["report_root"],
            metadata=obj.get("metadata", {}),
            metadata_only=obj.get("metadata_only", True),
            production_writes_enabled=obj.get("production_writes_enabled", False),
        )

    @classmethod
    def from_json(cls, text: str) -> "MarketStateMetadataManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime MarketStateMetadataManifest JSON"))


def default_market_state_feature_family_declarations() -> tuple[MarketStateFeatureFamilyDeclaration, ...]:
    lineage = {"source": "default_market_state_feature_family_declarations"}
    return (
        MarketStateFeatureFamilyDeclaration(
            family_name="market_return_summary",
            axis=RegimeAxis.MARKET,
            summary_type="return_summary",
            required_source_columns=("timestamp", "asset", "log_return"),
            derived_feature_columns=("mean_market_return", "median_market_return", "market_return_quantile_spread"),
            lineage_metadata=lineage,
        ),
        MarketStateFeatureFamilyDeclaration(
            family_name="realized_volatility_summary",
            axis=RegimeAxis.MARKET_VOL,
            summary_type="realized_volatility_summary",
            required_source_columns=("timestamp", "asset", "realized_volatility"),
            derived_feature_columns=("median_realized_volatility", "volatility_iqr", "high_volatility_share"),
            lineage_metadata=lineage,
        ),
        MarketStateFeatureFamilyDeclaration(
            family_name="breadth",
            axis=RegimeAxis.BREADTH,
            summary_type="breadth",
            required_source_columns=("timestamp", "asset", "log_return"),
            derived_feature_columns=("positive_return_share", "advance_decline_ratio", "above_zero_return_count"),
            lineage_metadata=lineage,
        ),
        MarketStateFeatureFamilyDeclaration(
            family_name="dispersion",
            axis=RegimeAxis.DISPERSION,
            summary_type="dispersion",
            required_source_columns=("timestamp", "asset", "log_return"),
            derived_feature_columns=("cross_sectional_return_std", "return_iqr", "return_mad"),
            lineage_metadata=lineage,
        ),
        MarketStateFeatureFamilyDeclaration(
            family_name="correlation_summary",
            axis=RegimeAxis.CORRELATION,
            summary_type="correlation",
            required_source_columns=("timestamp", "asset", "log_return"),
            derived_feature_columns=("median_pairwise_correlation", "correlation_breadth"),
            lineage_metadata=lineage,
        ),
        MarketStateFeatureFamilyDeclaration(
            family_name="covariance_summary",
            axis=RegimeAxis.CORRELATION,
            summary_type="covariance",
            required_source_columns=("timestamp", "asset", "log_return"),
            derived_feature_columns=("covariance_trace", "first_principal_component_concentration"),
            lineage_metadata=lineage,
        ),
    )


def build_market_state_metadata_manifest(
    *,
    manifest_id: str = "market_state_metadata_manifest",
    universe: str = "global",
    bands: Sequence[str | RegimeBand] = (RegimeBand.MICRO,),
    report_root: str | Path = default_foundation_report_root("market_state_metadata"),
    feature_families: Sequence[MarketStateFeatureFamilyDeclaration | Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MarketStateMetadataManifest:
    return MarketStateMetadataManifest(
        manifest_id=manifest_id,
        universe=universe,
        bands=bands,
        feature_families=tuple(feature_families or default_market_state_feature_family_declarations()),
        report_root=report_root,
        metadata=metadata or {},
    )


__all__ = [
    "MARKET_STATE_FEATURE_FAMILY_ARTIFACT_KIND",
    "MARKET_STATE_FEATURE_FAMILY_NAMES",
    "MARKET_STATE_LAYER",
    "MARKET_STATE_METADATA_ARTIFACT_KIND",
    "MARKET_STATE_METADATA_SCHEMA_VERSION",
    "MARKET_STATE_SUMMARY_TYPES",
    "MarketStateFeatureFamilyDeclaration",
    "MarketStateMetadataManifest",
    "build_market_state_metadata_manifest",
    "default_market_state_feature_family_declarations",
    "validate_market_state_metadata_report_root",
]
