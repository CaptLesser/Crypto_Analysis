from __future__ import annotations

from typing import Any, Mapping

from src.regimes.core.band_composites import (
    ALIGNMENT_POLICY_CEILING_BOUNDARY,
    BAND_COMPOSITE_SCHEMA_VERSION,
    BandCompositeSpec,
    band_composite_registry_as_dict,
    default_relationship_feature_permissions,
    resolve_band_composite_spec,
)
from src.regimes.market_state.taxonomy import default_market_state_taxonomy
from src.regimes.regime_features.feature_families import (
    PrimitiveMarketFeatureFamilyRegistry,
    default_primitive_market_feature_family_registry,
)


MARKET_STATE_BAND_COMPOSITE_POLICY_ID = "market_state_band_composite_policy_v1"


def market_state_band_composite_specs(
    *,
    feature_family_registry: PrimitiveMarketFeatureFamilyRegistry | None = None,
    relationship_feature_permissions: Mapping[str, Any] | None = None,
) -> tuple[BandCompositeSpec, ...]:
    taxonomy = default_market_state_taxonomy()
    allowed_by_band = _allowed_market_feature_families_by_band(
        feature_family_registry or default_primitive_market_feature_family_registry()
    )
    permissions = relationship_feature_permissions or default_relationship_feature_permissions()
    return tuple(
        BandCompositeSpec(
            band=band,
            ceiling_interval=int(spec.ceiling_interval_min),
            member_intervals=spec.member_intervals,
            output_cadence=int(spec.ceiling_interval_min),
            alignment_policy=ALIGNMENT_POLICY_CEILING_BOUNDARY,
            allowed_feature_families=allowed_by_band.get(band, ()),
            relationship_feature_permissions=permissions,
            metadata={
                "source": "market_state_taxonomy",
                "feature_family_source": "primitive_market_regime_feature_family_registry",
                "train_days": int(spec.train_days),
                "validation_horizons_min": list(spec.validation_horizons_min),
                "preserves_existing_taxonomy_band_contract": True,
            },
        )
        for band, spec in sorted(taxonomy.bands.items())
    )


def market_state_band_composite_policy(
    *,
    feature_family_registry: PrimitiveMarketFeatureFamilyRegistry | None = None,
) -> dict[str, Any]:
    specs = market_state_band_composite_specs(feature_family_registry=feature_family_registry)
    payload = band_composite_registry_as_dict(specs)
    payload.update(
        {
            "policy_id": MARKET_STATE_BAND_COMPOSITE_POLICY_ID,
            "pathway": "market_state",
            "schema_version": BAND_COMPOSITE_SCHEMA_VERSION,
            "pairwise_execution_enabled": False,
            "cross_asset_execution_enabled": False,
            "market_state_clustering_enabled": False,
        }
    )
    return payload


def resolve_market_state_band_composite(
    band: str,
    *,
    feature_family_registry: PrimitiveMarketFeatureFamilyRegistry | None = None,
) -> BandCompositeSpec:
    return resolve_band_composite_spec(
        band,
        specs=market_state_band_composite_specs(feature_family_registry=feature_family_registry),
    )


def _allowed_market_feature_families_by_band(registry: PrimitiveMarketFeatureFamilyRegistry) -> dict[str, tuple[str, ...]]:
    payload: dict[str, list[str]] = {"micro": [], "meso": [], "macro": []}
    for spec in registry.families.values():
        for band in spec.compatible_bands:
            payload[str(band)].append(spec.feature_family_id)
    return {band: tuple(sorted(set(family_ids))) for band, family_ids in payload.items()}


__all__ = [
    "MARKET_STATE_BAND_COMPOSITE_POLICY_ID",
    "market_state_band_composite_policy",
    "market_state_band_composite_specs",
    "resolve_market_state_band_composite",
]
