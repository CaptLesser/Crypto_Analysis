from __future__ import annotations

from dataclasses import dataclass, field
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
MARKET_STATE_V1_BAND_POLICY_ID = "market_state_v1_band_horizon_policy"
MARKET_STATE_V1_BLOCKED_SUBHOUR_INTERVALS: tuple[int, ...] = (1, 5, 15)
MARKET_STATE_V1_BANDS: tuple[str, ...] = ("micro", "meso", "macro")


@dataclass(frozen=True)
class MarketStateBandPolicy:
    band: str
    interval_minutes: int
    output_cadence_minutes: int
    feature_lookback_defaults: Mapping[str, int]
    covariance_correlation_allowed: bool
    covariance_correlation_requires_explicit_config: bool = False
    covariance_correlation_min_window: int = 0
    l2_sidecar_allowed: bool = False
    blocked_pairwise_cross_asset_intervals: tuple[int, ...] = MARKET_STATE_V1_BLOCKED_SUBHOUR_INTERVALS
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        band = str(self.band).strip().lower()
        if band not in MARKET_STATE_V1_BANDS:
            raise ValueError(f"Unsupported Market-State v1 band {band!r}")
        interval = int(self.interval_minutes)
        cadence = int(self.output_cadence_minutes)
        if interval <= 0 or cadence <= 0:
            raise ValueError("Market-State v1 band interval and output cadence must be positive")
        if interval < 60:
            raise ValueError("Market-State v1 band intervals must be hourly or higher")
        lookbacks = {str(key): int(value) for key, value in dict(self.feature_lookback_defaults).items()}
        if not lookbacks:
            raise ValueError("Market-State v1 band policy requires feature_lookback_defaults")
        invalid = [key for key, value in lookbacks.items() if value <= 0]
        if invalid:
            raise ValueError(f"Market-State v1 lookback defaults must be positive: {invalid}")
        blocked = tuple(dict.fromkeys(int(value) for value in self.blocked_pairwise_cross_asset_intervals))
        if any(value >= 60 for value in blocked):
            raise ValueError("Market-State v1 blocked pairwise/cross-asset intervals must be sub-hour")
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "interval_minutes", interval)
        object.__setattr__(self, "output_cadence_minutes", cadence)
        object.__setattr__(self, "feature_lookback_defaults", lookbacks)
        object.__setattr__(self, "covariance_correlation_allowed", bool(self.covariance_correlation_allowed))
        object.__setattr__(
            self,
            "covariance_correlation_requires_explicit_config",
            bool(self.covariance_correlation_requires_explicit_config),
        )
        object.__setattr__(self, "covariance_correlation_min_window", max(0, int(self.covariance_correlation_min_window)))
        object.__setattr__(self, "l2_sidecar_allowed", bool(self.l2_sidecar_allowed))
        object.__setattr__(self, "blocked_pairwise_cross_asset_intervals", blocked)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def output_cadence(self) -> int:
        return int(self.output_cadence_minutes)

    def covariance_correlation_permitted(
        self,
        *,
        interval_minutes: int | None = None,
        explicit_config: bool = False,
        window_observations: int | None = None,
    ) -> bool:
        interval = self.interval_minutes if interval_minutes is None else int(interval_minutes)
        if interval < 60 or interval in self.blocked_pairwise_cross_asset_intervals:
            return False
        if interval != self.interval_minutes:
            return False
        if not self.covariance_correlation_allowed:
            return False
        if self.covariance_correlation_requires_explicit_config and not bool(explicit_config):
            return False
        if window_observations is not None and int(window_observations) < int(self.covariance_correlation_min_window):
            return False
        return True

    def pairwise_cross_asset_permitted(self, *, interval_minutes: int) -> bool:
        interval = int(interval_minutes)
        if interval < 60 or interval in self.blocked_pairwise_cross_asset_intervals:
            return False
        return bool(self.metadata.get("pairwise_cross_asset_execution_enabled", False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_v1_band_policy",
            "policy_id": MARKET_STATE_V1_BAND_POLICY_ID,
            "band": self.band,
            "interval_minutes": int(self.interval_minutes),
            "output_cadence_minutes": int(self.output_cadence_minutes),
            "feature_lookback_defaults": dict(self.feature_lookback_defaults),
            "covariance_correlation_allowed": bool(self.covariance_correlation_allowed),
            "covariance_correlation_requires_explicit_config": bool(
                self.covariance_correlation_requires_explicit_config
            ),
            "covariance_correlation_min_window": int(self.covariance_correlation_min_window),
            "l2_sidecar_allowed": bool(self.l2_sidecar_allowed),
            "blocked_pairwise_cross_asset_intervals": list(self.blocked_pairwise_cross_asset_intervals),
            "sub_hour_covariance_correlation_allowed": False,
            "sub_hour_pairwise_cross_asset_allowed": False,
            "metadata": dict(self.metadata),
        }


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


def default_market_state_v1_band_policies() -> dict[str, MarketStateBandPolicy]:
    return {
        "micro": MarketStateBandPolicy(
            band="micro",
            interval_minutes=60,
            output_cadence_minutes=60,
            feature_lookback_defaults={
                "return": 12,
                "volatility": 12,
                "activity": 12,
                "breadth": 12,
                "trend": 24,
                "covariance_correlation": 24,
            },
            covariance_correlation_allowed=True,
            covariance_correlation_requires_explicit_config=True,
            covariance_correlation_min_window=20,
            l2_sidecar_allowed=False,
            metadata={
                "role": "short return/vol/activity/breadth",
                "pairwise_cross_asset_execution_enabled": False,
                "notes": "Covariance/correlation is gated by explicit config and sufficient hourly window.",
            },
        ),
        "meso": MarketStateBandPolicy(
            band="meso",
            interval_minutes=240,
            output_cadence_minutes=240,
            feature_lookback_defaults={
                "return": 18,
                "volatility": 18,
                "activity": 18,
                "breadth": 18,
                "trend": 30,
                "covariance_correlation": 30,
            },
            covariance_correlation_allowed=True,
            covariance_correlation_requires_explicit_config=False,
            covariance_correlation_min_window=20,
            l2_sidecar_allowed=False,
            metadata={
                "role": "main market-state context",
                "pairwise_cross_asset_execution_enabled": False,
            },
        ),
        "macro": MarketStateBandPolicy(
            band="macro",
            interval_minutes=1440,
            output_cadence_minutes=1440,
            feature_lookback_defaults={
                "return": 30,
                "volatility": 30,
                "activity": 30,
                "breadth": 30,
                "drawdown": 60,
                "stress": 60,
                "covariance_correlation": 60,
            },
            covariance_correlation_allowed=True,
            covariance_correlation_requires_explicit_config=False,
            covariance_correlation_min_window=20,
            l2_sidecar_allowed=False,
            metadata={
                "role": "drawdown/stress/correlation backdrop",
                "pairwise_cross_asset_execution_enabled": False,
            },
        ),
    }


def resolve_market_state_v1_band_policy(band: str) -> MarketStateBandPolicy:
    normalized = str(band).strip().lower()
    policies = default_market_state_v1_band_policies()
    try:
        return policies[normalized]
    except KeyError as exc:
        valid = ", ".join(policies)
        raise ValueError(f"Unsupported Market-State v1 band {band!r}; expected one of: {valid}") from exc


def resolve_market_state_v1_interval_minutes(band: str) -> int:
    return int(resolve_market_state_v1_band_policy(band).interval_minutes)


def market_state_v1_band_policy_manifest() -> dict[str, Any]:
    policies = default_market_state_v1_band_policies()
    return {
        "schema_version": 1,
        "artifact_kind": "market_state_v1_band_policy_manifest",
        "policy_id": MARKET_STATE_V1_BAND_POLICY_ID,
        "pathway": "market_state",
        "bands": {band: policy.as_dict() for band, policy in policies.items()},
        "blocked_covariance_correlation_intervals": list(MARKET_STATE_V1_BLOCKED_SUBHOUR_INTERVALS),
        "blocked_pairwise_cross_asset_intervals": list(MARKET_STATE_V1_BLOCKED_SUBHOUR_INTERVALS),
        "asset_state_composite_band_logic_preserved": True,
        "production_writes_enabled": False,
        "l2_sidecar_enabled": False,
    }


def market_state_v1_covariance_correlation_permitted(
    band: str,
    *,
    interval_minutes: int | None = None,
    explicit_config: bool = False,
    window_observations: int | None = None,
) -> bool:
    return resolve_market_state_v1_band_policy(band).covariance_correlation_permitted(
        interval_minutes=interval_minutes,
        explicit_config=explicit_config,
        window_observations=window_observations,
    )


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
    "MARKET_STATE_V1_BAND_POLICY_ID",
    "MARKET_STATE_V1_BANDS",
    "MARKET_STATE_V1_BLOCKED_SUBHOUR_INTERVALS",
    "MarketStateBandPolicy",
    "default_market_state_v1_band_policies",
    "market_state_band_composite_policy",
    "market_state_band_composite_specs",
    "market_state_v1_band_policy_manifest",
    "market_state_v1_covariance_correlation_permitted",
    "resolve_market_state_band_composite",
    "resolve_market_state_v1_band_policy",
    "resolve_market_state_v1_interval_minutes",
]
