from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import require_schema_version
from src.regimes.core.serialization import dumps_json, to_jsonable


CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION = 1
RELATIONSHIP_FEATURE_CATALOG_ARTIFACT_KIND = "relationship_feature_catalog"
RELATIONSHIP_FEATURE_CATALOG_ID = "relationship_feature_catalog_v1"

FEATURE_CLASS_V1 = "v1"
FEATURE_CLASS_SIDECAR = "sidecar"
FEATURE_CLASS_CONDITIONAL = "conditional"
FEATURE_CLASS_DEFERRED = "deferred"

FEATURE_FAMILY_MARKET_EXPOSURE = "market_exposure"
FEATURE_FAMILY_ISOLATION_STATUS = "isolation_status"
FEATURE_FAMILY_RESIDUAL_PEER_SUMMARY = "residual_peer_summary"
FEATURE_FAMILY_RISK_NEIGHBORHOOD = "risk_neighborhood"
FEATURE_FAMILY_DEFERRED_GROUP = "deferred_group_or_regime"

V1_MARKET_EXPOSURE_FEATURES: tuple[str, ...] = (
    "corr_to_anchor_primary",
    "corr_to_anchor_secondary",
    "corr_to_core_basket",
    "beta_to_core_basket",
    "market_mode_exposure_score",
)

V1_ISOLATION_STATUS_FEATURES: tuple[str, ...] = (
    "isolated_asset_score",
    "peer_signal_availability_status",
    "stable_edge_count",
    "candidate_edge_count",
)

V1_RESIDUAL_PEER_SUMMARY_FEATURES: tuple[str, ...] = (
    "residual_peer_signal_score",
    "relationship_concentration",
    "relationship_entropy",
    "top_peer_count",
    "top_peer_stability_mean",
    "strongest_peer_slot_1_strength",
)

SIDECAR_FEATURES: tuple[str, ...] = (
    "strongest_peer_slot_1_alias",
    "strongest_peer_slot_2_strength",
    "strongest_peer_slot_2_alias",
    "volatility_neighborhood_score",
    "residual_return_vs_core",
)

DEFERRED_FEATURES: tuple[str, ...] = (
    "peer_group_id",
    "peer_group_centroid_distance",
    "graph_centrality",
    "community_membership_confidence",
    "cross_asset_regime_label",
)

V1_FEATURE_FIELDS: tuple[str, ...] = (
    *V1_MARKET_EXPOSURE_FEATURES,
    *V1_ISOLATION_STATUS_FEATURES,
    *V1_RESIDUAL_PEER_SUMMARY_FEATURES,
)


@dataclass(frozen=True)
class CrossAssetRelationshipFeatureCatalogEntry:
    feature_name: str
    feature_family: str
    feature_class: str
    source_artifact: str
    source_field: str | None
    expected_downstream_use: str
    stable_schema: bool = True
    sidecar_availability_flag: str | None = None
    deferred_reason: str | None = None
    schema_version: int = CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "feature_name", _text(self.feature_name, field_name="feature_name"))
        object.__setattr__(self, "feature_family", _text(self.feature_family, field_name="feature_family"))
        feature_class = _text(self.feature_class, field_name="feature_class")
        if feature_class not in {FEATURE_CLASS_V1, FEATURE_CLASS_SIDECAR, FEATURE_CLASS_CONDITIONAL, FEATURE_CLASS_DEFERRED}:
            raise ValueError("Cross-Asset feature catalog feature_class is not supported")
        object.__setattr__(self, "feature_class", feature_class)
        object.__setattr__(self, "source_artifact", _text(self.source_artifact, field_name="source_artifact"))
        object.__setattr__(self, "source_field", _optional_text(self.source_field))
        object.__setattr__(self, "expected_downstream_use", _text(self.expected_downstream_use, field_name="expected_downstream_use"))
        object.__setattr__(self, "stable_schema", bool(self.stable_schema))
        object.__setattr__(self, "sidecar_availability_flag", _optional_text(self.sidecar_availability_flag))
        object.__setattr__(self, "deferred_reason", _optional_text(self.deferred_reason))
        if self.feature_class == FEATURE_CLASS_DEFERRED and self.deferred_reason is None:
            raise ValueError("Cross-Asset deferred feature catalog entries require deferred_reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_feature_catalog_entry",
            "schema_version": int(self.schema_version),
            "feature_name": self.feature_name,
            "feature_family": self.feature_family,
            "feature_class": self.feature_class,
            "source_artifact": self.source_artifact,
            "source_field": self.source_field,
            "expected_downstream_use": self.expected_downstream_use,
            "stable_schema": bool(self.stable_schema),
            "sidecar_availability_flag": self.sidecar_availability_flag,
            "deferred_reason": self.deferred_reason,
            "production_enabled": False,
        }


@dataclass(frozen=True)
class CrossAssetRelationshipFeatureCatalog:
    entries: Sequence[CrossAssetRelationshipFeatureCatalogEntry | Mapping[str, Any]]
    catalog_id: str = RELATIONSHIP_FEATURE_CATALOG_ID
    schema_version: int = CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION
    artifact_kind: str = RELATIONSHIP_FEATURE_CATALOG_ARTIFACT_KIND

    def __post_init__(self) -> None:
        entries = tuple(entry if isinstance(entry, CrossAssetRelationshipFeatureCatalogEntry) else CrossAssetRelationshipFeatureCatalogEntry(**entry) for entry in self.entries)
        names = [entry.feature_name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Cross-Asset feature catalog feature names must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "catalog_id", _text(self.catalog_id, field_name="catalog_id"))
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))

    @property
    def v1_feature_names(self) -> tuple[str, ...]:
        return tuple(entry.feature_name for entry in self.entries if entry.feature_class == FEATURE_CLASS_V1)

    @property
    def sidecar_feature_names(self) -> tuple[str, ...]:
        return tuple(entry.feature_name for entry in self.entries if entry.feature_class == FEATURE_CLASS_SIDECAR)

    @property
    def deferred_feature_names(self) -> tuple[str, ...]:
        return tuple(entry.feature_name for entry in self.entries if entry.feature_class == FEATURE_CLASS_DEFERRED)

    def validate(self) -> None:
        required = set(V1_FEATURE_FIELDS)
        present = set(self.v1_feature_names)
        missing = sorted(required.difference(present))
        if missing:
            raise ValueError(f"Cross-Asset feature catalog missing required v1 features: {missing}")
        missing_sidecars = sorted(set(SIDECAR_FEATURES).difference(self.sidecar_feature_names))
        if missing_sidecars:
            raise ValueError(f"Cross-Asset feature catalog missing sidecar features: {missing_sidecars}")
        missing_deferred = sorted(set(DEFERRED_FEATURES).difference(self.deferred_feature_names))
        if missing_deferred:
            raise ValueError(f"Cross-Asset feature catalog missing deferred features: {missing_deferred}")

    def as_dict(self) -> dict[str, Any]:
        class_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        for entry in self.entries:
            class_counts[entry.feature_class] = class_counts.get(entry.feature_class, 0) + 1
            family_counts[entry.feature_family] = family_counts.get(entry.feature_family, 0) + 1
        return {
            "artifact_kind": self.artifact_kind,
            "schema_version": int(self.schema_version),
            "catalog_id": self.catalog_id,
            "feature_count": len(self.entries),
            "class_counts": dict(sorted(class_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "entries": [entry.as_dict() for entry in self.entries],
            "production_enabled": False,
            "cross_asset_regime_label_written": False,
            "one_column_per_related_asset_allowed": False,
        }

    def to_json(self) -> str:
        return dumps_json(self.as_dict())


def default_cross_asset_relationship_feature_catalog() -> CrossAssetRelationshipFeatureCatalog:
    entries: list[CrossAssetRelationshipFeatureCatalogEntry] = []
    entries.extend(
        CrossAssetRelationshipFeatureCatalogEntry(
            feature_name=name,
            feature_family=FEATURE_FAMILY_MARKET_EXPOSURE,
            feature_class=FEATURE_CLASS_V1,
            source_artifact="asset_relationship_profiles",
            source_field=name,
            expected_downstream_use="market/core exposure context, not peer identity",
        )
        for name in V1_MARKET_EXPOSURE_FEATURES
    )
    entries.extend(
        CrossAssetRelationshipFeatureCatalogEntry(
            feature_name=name,
            feature_family=FEATURE_FAMILY_ISOLATION_STATUS,
            feature_class=FEATURE_CLASS_V1,
            source_artifact="isolated_asset_profiles" if "edge_count" in name or "isolated" in name or "availability" in name else "asset_relationship_profiles",
            source_field=name,
            expected_downstream_use="peer-signal gating and non-peer status handling",
        )
        for name in V1_ISOLATION_STATUS_FEATURES
    )
    entries.extend(
        CrossAssetRelationshipFeatureCatalogEntry(
            feature_name=name,
            feature_family=FEATURE_FAMILY_RESIDUAL_PEER_SUMMARY,
            feature_class=FEATURE_CLASS_V1,
            source_artifact="asset_relationship_profiles" if name in {"residual_peer_signal_score", "relationship_concentration", "relationship_entropy", "top_peer_count", "top_peer_stability_mean"} else "selected_relationship_edges",
            source_field=name,
            expected_downstream_use="stable residual-peer summary without dynamic peer identity columns",
        )
        for name in V1_RESIDUAL_PEER_SUMMARY_FEATURES
    )
    sidecar_sources = {
        "strongest_peer_slot_1_alias": ("edge_alias_manifest", "strongest_peer_slot_1_alias_available", FEATURE_FAMILY_RESIDUAL_PEER_SUMMARY),
        "strongest_peer_slot_2_strength": ("edge_alias_manifest", "strongest_peer_slot_2_available", FEATURE_FAMILY_RESIDUAL_PEER_SUMMARY),
        "strongest_peer_slot_2_alias": ("edge_alias_manifest", "strongest_peer_slot_2_available", FEATURE_FAMILY_RESIDUAL_PEER_SUMMARY),
        "volatility_neighborhood_score": ("asset_relationship_profiles", "volatility_neighborhood_score_available", FEATURE_FAMILY_RISK_NEIGHBORHOOD),
        "residual_return_vs_core": ("asset_relationship_profiles", "residual_return_vs_core_available", FEATURE_FAMILY_MARKET_EXPOSURE),
    }
    entries.extend(
        CrossAssetRelationshipFeatureCatalogEntry(
            feature_name=name,
            feature_family=family,
            feature_class=FEATURE_CLASS_SIDECAR,
            source_artifact=source,
            source_field=name,
            expected_downstream_use="diagnostic or identity sidecar; not required as a core model column",
            stable_schema=True,
            sidecar_availability_flag=flag,
        )
        for name, (source, flag, family) in sidecar_sources.items()
    )
    entries.extend(
        CrossAssetRelationshipFeatureCatalogEntry(
            feature_name=name,
            feature_family=FEATURE_FAMILY_DEFERRED_GROUP,
            feature_class=FEATURE_CLASS_DEFERRED,
            source_artifact="deferred",
            source_field=None,
            expected_downstream_use="not emitted by Cross-Asset Feature Handoff v1",
            stable_schema=False,
            deferred_reason="requires future peer-group, graph, or Cross-Asset regime classification implementation",
        )
        for name in DEFERRED_FEATURES
    )
    catalog = CrossAssetRelationshipFeatureCatalog(entries=tuple(entries))
    catalog.validate()
    return catalog


def validate_cross_asset_relationship_feature_catalog(catalog: CrossAssetRelationshipFeatureCatalog | Mapping[str, Any]) -> None:
    if not isinstance(catalog, CrossAssetRelationshipFeatureCatalog):
        if bool(catalog.get("production_enabled", False)):
            raise ValueError("Cross-Asset feature catalog production_enabled must be false")
        if bool(catalog.get("one_column_per_related_asset_allowed", False)):
            raise ValueError("Cross-Asset feature catalog must not allow one-column-per-related-asset schema")
        if bool(catalog.get("cross_asset_regime_label_written", False)):
            raise ValueError("Cross-Asset feature catalog must not write Cross-Asset regime labels")
    resolved = catalog if isinstance(catalog, CrossAssetRelationshipFeatureCatalog) else CrossAssetRelationshipFeatureCatalog(
        entries=tuple(catalog.get("entries", ())),
        catalog_id=str(catalog.get("catalog_id") or RELATIONSHIP_FEATURE_CATALOG_ID),
        schema_version=int(catalog.get("schema_version") or CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION),
        artifact_kind=str(catalog.get("artifact_kind") or RELATIONSHIP_FEATURE_CATALOG_ARTIFACT_KIND),
    )
    resolved.validate()
    payload = resolved.as_dict()
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Cross-Asset feature catalog production_enabled must be false")
    if bool(payload.get("one_column_per_related_asset_allowed", False)):
        raise ValueError("Cross-Asset feature catalog must not allow one-column-per-related-asset schema")


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Cross-Asset feature catalog {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION",
    "DEFERRED_FEATURES",
    "FEATURE_CLASS_CONDITIONAL",
    "FEATURE_CLASS_DEFERRED",
    "FEATURE_CLASS_SIDECAR",
    "FEATURE_CLASS_V1",
    "FEATURE_FAMILY_DEFERRED_GROUP",
    "FEATURE_FAMILY_ISOLATION_STATUS",
    "FEATURE_FAMILY_MARKET_EXPOSURE",
    "FEATURE_FAMILY_RESIDUAL_PEER_SUMMARY",
    "FEATURE_FAMILY_RISK_NEIGHBORHOOD",
    "RELATIONSHIP_FEATURE_CATALOG_ARTIFACT_KIND",
    "RELATIONSHIP_FEATURE_CATALOG_ID",
    "SIDECAR_FEATURES",
    "V1_FEATURE_FIELDS",
    "V1_ISOLATION_STATUS_FEATURES",
    "V1_MARKET_EXPOSURE_FEATURES",
    "V1_RESIDUAL_PEER_SUMMARY_FEATURES",
    "CrossAssetRelationshipFeatureCatalog",
    "CrossAssetRelationshipFeatureCatalogEntry",
    "default_cross_asset_relationship_feature_catalog",
    "validate_cross_asset_relationship_feature_catalog",
]
