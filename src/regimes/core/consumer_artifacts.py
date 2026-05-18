"""Compact downstream Regime artifact scaffolding.

These contracts exist to preserve architecture direction for future consumers.
They do not implement forecast targets, Numerics exports, or broad feature
materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.clamp_policy import RegimeClampPolicy
from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.known_at import KnownAtSpec
from src.regimes.core.lineage import REGIME_LINEAGE_PATHWAYS, RegimeLineageSpec
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


COMPACT_REGIME_ARTIFACT_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
COMPACT_REGIME_ARTIFACT_SPEC_KIND = "compact_regime_artifact_spec"

ASSET_STATE_COMPOSITION_SUMMARY = "asset_state_composition_summary"
MARKET_STATE_FEATURE_VIEW = "market_state_feature_view"
CROSS_ASSET_PEER_SUMMARY = "cross_asset_peer_summary"
REGIME_TRANSITION_SUMMARY = "regime_transition_summary"
REGIME_FORECAST_FEATURE_VIEW = "regime_forecast_feature_view"
NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT = "numerics_candidate_regime_feature_export"

COMPACT_REGIME_ARTIFACT_KINDS: tuple[str, ...] = (
    ASSET_STATE_COMPOSITION_SUMMARY,
    MARKET_STATE_FEATURE_VIEW,
    CROSS_ASSET_PEER_SUMMARY,
    REGIME_TRANSITION_SUMMARY,
    REGIME_FORECAST_FEATURE_VIEW,
    NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT,
)

COMPACT_REGIME_ARTIFACT_KIND_PATHWAYS: Mapping[str, tuple[str, ...]] = {
    ASSET_STATE_COMPOSITION_SUMMARY: ("asset_state",),
    MARKET_STATE_FEATURE_VIEW: ("market_state",),
    CROSS_ASSET_PEER_SUMMARY: ("relative_state",),
    REGIME_TRANSITION_SUMMARY: REGIME_LINEAGE_PATHWAYS,
    REGIME_FORECAST_FEATURE_VIEW: REGIME_LINEAGE_PATHWAYS,
    NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT: REGIME_LINEAGE_PATHWAYS,
}


@dataclass(frozen=True)
class CompactRegimeSchemaStub:
    artifact_kind: str
    pathway_scope: Sequence[str]
    intended_consumer: str
    required_fields: Sequence[str]
    status: str = "placeholder_contract_only"
    forecast_targets_implemented: bool = False
    numerics_export_implemented: bool = False
    broad_feature_materialization_implemented: bool = False
    production_enabled: bool = False
    schema_version: int = COMPACT_REGIME_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        artifact = _artifact_kind(self.artifact_kind)
        if artifact not in COMPACT_REGIME_ARTIFACT_KINDS:
            raise ValueError(f"Unsupported compact Regime artifact_kind {artifact!r}")
        pathways = tuple(dict.fromkeys(_pathway(pathway) for pathway in self.pathway_scope))
        if not pathways:
            raise ValueError("Compact Regime schema stub pathway_scope must be non-empty")
        fields = _string_tuple(self.required_fields, field_name="required_fields")
        if self.production_enabled is not False:
            raise ValueError("Compact Regime schema stubs cannot enable production")
        if self.forecast_targets_implemented or self.numerics_export_implemented or self.broad_feature_materialization_implemented:
            raise ValueError("Compact Regime schema stubs are placeholder contracts only")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", artifact)
        object.__setattr__(self, "pathway_scope", pathways)
        object.__setattr__(self, "intended_consumer", _text(self.intended_consumer, field_name="intended_consumer"))
        object.__setattr__(self, "required_fields", fields)
        object.__setattr__(self, "status", _text(self.status, field_name="status"))
        object.__setattr__(self, "forecast_targets_implemented", False)
        object.__setattr__(self, "numerics_export_implemented", False)
        object.__setattr__(self, "broad_feature_materialization_implemented", False)
        object.__setattr__(self, "production_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway_scope": list(self.pathway_scope),
            "intended_consumer": self.intended_consumer,
            "required_fields": list(self.required_fields),
            "status": self.status,
            "placeholder_contract_only": True,
            "forecast_targets_implemented": False,
            "numerics_export_implemented": False,
            "broad_feature_materialization_implemented": False,
            "production_enabled": False,
            "warnings": [
                "no forecast targets are implemented here",
                "no Numerics export is implemented here",
                "placeholder contracts prevent downstream architecture drift",
            ],
        }


@dataclass(frozen=True)
class CompactRegimeArtifactSpec:
    artifact_kind: str
    pathway: str
    interval: int
    timestamp_key: str
    known_at_policy: KnownAtSpec | Mapping[str, Any]
    lineage: RegimeLineageSpec | Mapping[str, Any]
    clamp_policy: RegimeClampPolicy | Mapping[str, Any]
    intended_consumer: str
    axis: str | None = None
    band: str | None = None
    production_enabled: bool = False
    schema_version: int = COMPACT_REGIME_ARTIFACT_SCHEMA_VERSION
    spec_kind: str = COMPACT_REGIME_ARTIFACT_SPEC_KIND
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifact = _artifact_kind(self.artifact_kind)
        pathway = _pathway(self.pathway)
        allowed_pathways = COMPACT_REGIME_ARTIFACT_KIND_PATHWAYS[artifact]
        if pathway not in allowed_pathways:
            raise ValueError(f"Compact Regime artifact {artifact!r} does not support pathway {pathway!r}")
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Compact Regime artifact interval must be positive")
        lineage = self.lineage if isinstance(self.lineage, RegimeLineageSpec) else RegimeLineageSpec.from_dict(self.lineage)
        known_at = self.known_at_policy if isinstance(self.known_at_policy, KnownAtSpec) else KnownAtSpec.from_dict(self.known_at_policy)
        clamp = self.clamp_policy if isinstance(self.clamp_policy, RegimeClampPolicy) else RegimeClampPolicy.from_dict(self.clamp_policy)
        if lineage.pathway != pathway:
            raise ValueError("Compact Regime artifact lineage pathway must match artifact pathway")
        if int(lineage.interval) != interval:
            raise ValueError("Compact Regime artifact lineage interval must match artifact interval")
        if pathway not in clamp.applies_to_pathways:
            raise ValueError("Compact Regime artifact clamp policy must apply to artifact pathway")
        if self.production_enabled is not False:
            raise ValueError("Compact Regime downstream artifact scaffolding cannot enable production")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "spec_kind", _text(self.spec_kind, field_name="spec_kind"))
        object.__setattr__(self, "artifact_kind", artifact)
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "axis", _optional_text(self.axis))
        object.__setattr__(self, "band", _optional_text(self.band))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "timestamp_key", _text(self.timestamp_key, field_name="timestamp_key"))
        object.__setattr__(self, "known_at_policy", known_at)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "clamp_policy", clamp)
        object.__setattr__(self, "intended_consumer", _text(self.intended_consumer, field_name="intended_consumer"))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "spec_kind": self.spec_kind,
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "axis": self.axis,
            "band": self.band,
            "interval": int(self.interval),
            "timestamp_key": self.timestamp_key,
            "known_at_policy": self.known_at_policy.as_dict(),
            "lineage": self.lineage.as_dict(),
            "clamp_policy": self.clamp_policy.as_dict(),
            "intended_consumer": self.intended_consumer,
            "production_enabled": False,
            "forecast_targets_implemented": False,
            "numerics_export_implemented": False,
            "materialized_feature_view_written": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactRegimeArtifactSpec":
        obj = require_json_object(payload, context="CompactRegimeArtifactSpec")
        return cls(
            schema_version=obj.get("schema_version", COMPACT_REGIME_ARTIFACT_SCHEMA_VERSION),
            spec_kind=obj.get("spec_kind", COMPACT_REGIME_ARTIFACT_SPEC_KIND),
            artifact_kind=obj["artifact_kind"],
            pathway=obj["pathway"],
            axis=obj.get("axis"),
            band=obj.get("band"),
            interval=obj["interval"],
            timestamp_key=obj["timestamp_key"],
            known_at_policy=obj["known_at_policy"],
            lineage=obj["lineage"],
            clamp_policy=obj["clamp_policy"],
            intended_consumer=obj["intended_consumer"],
            production_enabled=bool(obj.get("production_enabled", False)),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "CompactRegimeArtifactSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="CompactRegimeArtifactSpec JSON"))


def compact_regime_schema_stubs() -> tuple[CompactRegimeSchemaStub, ...]:
    base_fields = (
        "artifact_kind",
        "pathway",
        "interval",
        "timestamp_key",
        "known_at_policy",
        "lineage",
        "clamp_policy",
        "intended_consumer",
        "production_enabled",
        "schema_version",
    )
    return (
        CompactRegimeSchemaStub(
            artifact_kind=ASSET_STATE_COMPOSITION_SUMMARY,
            pathway_scope=("asset_state",),
            intended_consumer="future_asset_state_dashboards",
            required_fields=base_fields + ("axis", "band"),
        ),
        CompactRegimeSchemaStub(
            artifact_kind=MARKET_STATE_FEATURE_VIEW,
            pathway_scope=("market_state",),
            intended_consumer="future_market_state_consumers",
            required_fields=base_fields + ("feature_family_id",),
        ),
        CompactRegimeSchemaStub(
            artifact_kind=CROSS_ASSET_PEER_SUMMARY,
            pathway_scope=("relative_state",),
            intended_consumer="future_cross_asset_relative_state_consumers",
            required_fields=base_fields + ("peer_group_id",),
        ),
        CompactRegimeSchemaStub(
            artifact_kind=REGIME_TRANSITION_SUMMARY,
            pathway_scope=REGIME_LINEAGE_PATHWAYS,
            intended_consumer="future_transition_diagnostics",
            required_fields=base_fields + ("transition_count",),
        ),
        CompactRegimeSchemaStub(
            artifact_kind=REGIME_FORECAST_FEATURE_VIEW,
            pathway_scope=REGIME_LINEAGE_PATHWAYS,
            intended_consumer="future_regime_forecasters_placeholder",
            required_fields=base_fields + ("forecast_feature_columns_placeholder",),
        ),
        CompactRegimeSchemaStub(
            artifact_kind=NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT,
            pathway_scope=REGIME_LINEAGE_PATHWAYS,
            intended_consumer="future_numerics_candidate_feature_exports_placeholder",
            required_fields=base_fields + ("numerics_export_columns_placeholder",),
        ),
    )


def compact_regime_schema_stub_by_kind() -> dict[str, CompactRegimeSchemaStub]:
    return {stub.artifact_kind: stub for stub in compact_regime_schema_stubs()}


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Compact Regime artifact {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _artifact_kind(value: object) -> str:
    return _text(value, field_name="artifact_kind").lower()


def _pathway(value: object) -> str:
    text = _text(value, field_name="pathway").lower()
    if text not in REGIME_LINEAGE_PATHWAYS:
        valid = ", ".join(REGIME_LINEAGE_PATHWAYS)
        raise ValueError(f"Unsupported Compact Regime artifact pathway {text!r}; expected one of: {valid}")
    return text


def _string_tuple(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Compact Regime artifact {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if not out:
        raise ValueError(f"Compact Regime artifact {field_name} must be non-empty")
    return out


__all__ = [
    "ASSET_STATE_COMPOSITION_SUMMARY",
    "COMPACT_REGIME_ARTIFACT_KINDS",
    "COMPACT_REGIME_ARTIFACT_KIND_PATHWAYS",
    "COMPACT_REGIME_ARTIFACT_SCHEMA_VERSION",
    "COMPACT_REGIME_ARTIFACT_SPEC_KIND",
    "CROSS_ASSET_PEER_SUMMARY",
    "MARKET_STATE_FEATURE_VIEW",
    "NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT",
    "REGIME_FORECAST_FEATURE_VIEW",
    "REGIME_TRANSITION_SUMMARY",
    "CompactRegimeArtifactSpec",
    "CompactRegimeSchemaStub",
    "compact_regime_schema_stub_by_kind",
    "compact_regime_schema_stubs",
]
