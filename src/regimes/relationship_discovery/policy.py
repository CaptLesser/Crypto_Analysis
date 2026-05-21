from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import dumps_json
from src.regimes.relationship_discovery.methods import (
    METHOD_BETA_TO_CORE,
    METHOD_RAW_ROLLING_CORR,
    METHOD_RESIDUAL_CORR,
    METHOD_RESIDUAL_RETURN,
    METHOD_VOLATILITY_SIMILARITY,
    RelationshipMethodComparisonConfig,
)
from src.regimes.relationship_discovery.scope import (
    DEFAULT_CONFIRMATION_INTERVAL,
    DEFAULT_EVIDENCE_INTERVAL,
    DEFAULT_PRIMARY_INTERVAL,
)
from src.regimes.relationship_discovery.selection import RelationshipEdgeSelectionConfig


RELATIONSHIP_POLICY_ARTIFACT_KIND = "relationship_discovery_v1_policy"

INTERVAL_ROLE_PRIMARY_PERSISTENT = "primary_persistent"
INTERVAL_ROLE_CONFIRMATION_EVIDENCE = "confirmation_evidence"
INTERVAL_ROLE_PROBE_EVIDENCE = "probe_evidence"
INTERVAL_ROLE_GATED_DISABLED = "gated_disabled"

K_POLICY_PRIMARY = "k3_primary"
K_POLICY_SENSITIVITY = "k5_sensitivity"
K_POLICY_ADAPTIVE_PROTOTYPE = "adaptive_prototype_only"

METHOD_ROLE_MARKET_MODE_RELATIONSHIP = "market_mode_relationship"
METHOD_ROLE_PEER_LIKE_RELATIONSHIP = "peer_like_relationship"
METHOD_ROLE_MARKET_EXPOSURE = "market_exposure"
METHOD_ROLE_RISK_NEIGHBORHOOD_SIDECAR = "risk_neighborhood_sidecar"
METHOD_ROLE_DIAGNOSTIC = "diagnostic"

SATELLITE_POLICY_EXCLUDED_DEFAULT_PEER_DISCOVERY = "excluded_from_default_peer_discovery"
SATELLITE_POLICY_EXPLICIT_SIDECAR_ONLY = "explicit_sidecar_only"


@dataclass(frozen=True)
class RelationshipIntervalPolicy:
    primary_interval: int = DEFAULT_PRIMARY_INTERVAL
    confirmation_interval: int = DEFAULT_CONFIRMATION_INTERVAL
    probe_interval: int = DEFAULT_EVIDENCE_INTERVAL
    include_probe_interval: bool = True
    sub_hour_enabled: bool = False

    def __post_init__(self) -> None:
        primary = _interval(self.primary_interval, field_name="primary_interval")
        confirmation = _interval(self.confirmation_interval, field_name="confirmation_interval")
        probe = _interval(self.probe_interval, field_name="probe_interval")
        object.__setattr__(self, "primary_interval", primary)
        object.__setattr__(self, "confirmation_interval", confirmation)
        object.__setattr__(self, "probe_interval", probe)
        object.__setattr__(self, "include_probe_interval", bool(self.include_probe_interval))
        object.__setattr__(self, "sub_hour_enabled", bool(self.sub_hour_enabled))
        if self.sub_hour_enabled:
            raise ValueError("Relationship Discovery v1 sub-hour relationships are gated and disabled")

    @property
    def intervals(self) -> tuple[int, ...]:
        values = [int(self.primary_interval), int(self.confirmation_interval)]
        if self.include_probe_interval:
            values.append(int(self.probe_interval))
        return tuple(dict.fromkeys(values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_interval_policy",
            "primary_interval": int(self.primary_interval),
            "primary_interval_role": INTERVAL_ROLE_PRIMARY_PERSISTENT,
            "confirmation_interval": int(self.confirmation_interval),
            "confirmation_interval_role": INTERVAL_ROLE_CONFIRMATION_EVIDENCE,
            "probe_interval": int(self.probe_interval),
            "probe_interval_role": INTERVAL_ROLE_PROBE_EVIDENCE,
            "include_probe_interval": bool(self.include_probe_interval),
            "sub_hour_enabled": False,
            "sub_hour_role": INTERVAL_ROLE_GATED_DISABLED,
            "intervals_to_load": [int(value) for value in self.intervals],
        }


@dataclass(frozen=True)
class RelationshipKPolicy:
    primary_k: int = 3
    sensitivity_k: int = 5
    adaptive_k_enabled: bool = False
    adaptive_k_max: int = 5
    max_k: int = 10

    def __post_init__(self) -> None:
        max_k = _positive_int(self.max_k, field_name="max_k")
        primary = _bounded_k(self.primary_k, max_k=max_k, field_name="primary_k")
        sensitivity = _bounded_k(self.sensitivity_k, max_k=max_k, field_name="sensitivity_k")
        adaptive = _bounded_k(self.adaptive_k_max, max_k=max_k, field_name="adaptive_k_max")
        object.__setattr__(self, "max_k", max_k)
        object.__setattr__(self, "primary_k", primary)
        object.__setattr__(self, "sensitivity_k", sensitivity)
        object.__setattr__(self, "adaptive_k_max", adaptive)
        object.__setattr__(self, "adaptive_k_enabled", bool(self.adaptive_k_enabled))
        if sensitivity < primary:
            raise ValueError("Relationship Discovery v1 sensitivity_k must be greater than or equal to primary_k")
        if self.adaptive_k_enabled:
            raise ValueError("Relationship Discovery v1 adaptive K is prototype-only and disabled by default")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_k_policy",
            "primary_policy": K_POLICY_PRIMARY,
            "primary_k": int(self.primary_k),
            "sensitivity_policy": K_POLICY_SENSITIVITY,
            "sensitivity_k": int(self.sensitivity_k),
            "adaptive_policy": K_POLICY_ADAPTIVE_PROTOTYPE,
            "adaptive_k_enabled": False,
            "adaptive_k_max": int(self.adaptive_k_max),
            "max_k": int(self.max_k),
        }


@dataclass(frozen=True)
class RelationshipMethodPolicy:
    raw_correlation_role: str = METHOD_ROLE_MARKET_MODE_RELATIONSHIP
    residual_correlation_role: str = METHOD_ROLE_PEER_LIKE_RELATIONSHIP
    beta_to_core_role: str = METHOD_ROLE_MARKET_EXPOSURE
    residual_return_role: str = METHOD_ROLE_DIAGNOSTIC
    volatility_similarity_role: str = METHOD_ROLE_RISK_NEIGHBORHOOD_SIDECAR
    residual_min_abs_strength: float = 0.05
    market_mode_min_abs_strength: float = 0.05
    risk_neighborhood_min_similarity: float = 0.05
    min_coverage: float = 0.75
    min_observations: int = 20
    require_window_survival: bool = True
    min_survival_count: int = 2
    min_survival_share: float = 0.5
    include_market_mode_edges: bool = True
    method_order: Sequence[str] = (
        METHOD_RAW_ROLLING_CORR,
        METHOD_BETA_TO_CORE,
        METHOD_RESIDUAL_RETURN,
        METHOD_RESIDUAL_CORR,
        METHOD_VOLATILITY_SIMILARITY,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual_min_abs_strength", _non_negative(self.residual_min_abs_strength, field_name="residual_min_abs_strength"))
        object.__setattr__(self, "market_mode_min_abs_strength", _non_negative(self.market_mode_min_abs_strength, field_name="market_mode_min_abs_strength"))
        object.__setattr__(self, "risk_neighborhood_min_similarity", _non_negative(self.risk_neighborhood_min_similarity, field_name="risk_neighborhood_min_similarity"))
        object.__setattr__(self, "min_coverage", _share(self.min_coverage, field_name="min_coverage"))
        object.__setattr__(self, "min_observations", max(2, int(self.min_observations)))
        object.__setattr__(self, "require_window_survival", bool(self.require_window_survival))
        object.__setattr__(self, "min_survival_count", _positive_int(self.min_survival_count, field_name="min_survival_count"))
        object.__setattr__(self, "min_survival_share", _share(self.min_survival_share, field_name="min_survival_share"))
        object.__setattr__(self, "include_market_mode_edges", bool(self.include_market_mode_edges))
        methods = tuple(dict.fromkeys(str(method).strip() for method in self.method_order if str(method).strip()))
        if METHOD_RESIDUAL_CORR not in methods:
            raise ValueError("Relationship Discovery v1 method_order must include residual correlation")
        object.__setattr__(self, "method_order", methods)

    @property
    def method_roles(self) -> dict[str, str]:
        return {
            METHOD_BETA_TO_CORE: self.beta_to_core_role,
            METHOD_RAW_ROLLING_CORR: self.raw_correlation_role,
            METHOD_RESIDUAL_CORR: self.residual_correlation_role,
            METHOD_RESIDUAL_RETURN: self.residual_return_role,
            METHOD_VOLATILITY_SIMILARITY: self.volatility_similarity_role,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_method_policy",
            "method_roles": dict(sorted(self.method_roles.items())),
            "method_order": list(self.method_order),
            "thresholds": {
                "market_mode_min_abs_strength": float(self.market_mode_min_abs_strength),
                "residual_min_abs_strength": float(self.residual_min_abs_strength),
                "risk_neighborhood_min_similarity": float(self.risk_neighborhood_min_similarity),
            },
            "min_coverage": float(self.min_coverage),
            "min_observations": int(self.min_observations),
            "require_window_survival": bool(self.require_window_survival),
            "min_survival_count": int(self.min_survival_count),
            "min_survival_share": float(self.min_survival_share),
            "include_market_mode_edges": bool(self.include_market_mode_edges),
        }


@dataclass(frozen=True)
class SatelliteRelationshipPolicy:
    include_satellites_in_default_peer_discovery: bool = False
    explicit_sidecar_available: bool = True
    sidecar_requires_explicit_request: bool = True

    def __post_init__(self) -> None:
        include_default = bool(self.include_satellites_in_default_peer_discovery)
        object.__setattr__(self, "include_satellites_in_default_peer_discovery", include_default)
        object.__setattr__(self, "explicit_sidecar_available", bool(self.explicit_sidecar_available))
        object.__setattr__(self, "sidecar_requires_explicit_request", bool(self.sidecar_requires_explicit_request))
        if include_default:
            raise ValueError("Relationship Discovery v1 excludes satellites from default peer discovery")
        if not self.sidecar_requires_explicit_request:
            raise ValueError("Relationship Discovery v1 satellite sidecars must require an explicit request")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "satellite_relationship_policy",
            "default_peer_discovery_policy": SATELLITE_POLICY_EXCLUDED_DEFAULT_PEER_DISCOVERY,
            "sidecar_policy": SATELLITE_POLICY_EXPLICIT_SIDECAR_ONLY,
            "include_satellites_in_default_peer_discovery": False,
            "explicit_sidecar_available": bool(self.explicit_sidecar_available),
            "sidecar_requires_explicit_request": True,
        }


@dataclass(frozen=True)
class RelationshipDiscoveryPolicy:
    interval_policy: RelationshipIntervalPolicy = field(default_factory=RelationshipIntervalPolicy)
    method_policy: RelationshipMethodPolicy = field(default_factory=RelationshipMethodPolicy)
    k_policy: RelationshipKPolicy = field(default_factory=RelationshipKPolicy)
    satellite_policy: SatelliteRelationshipPolicy = field(default_factory=SatelliteRelationshipPolicy)
    production_enabled: bool = False
    broad_all_to_all_enabled: bool = False
    dynamic_peer_clusters_enabled: bool = False
    cross_asset_regime_labels_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "production_enabled", bool(self.production_enabled))
        object.__setattr__(self, "broad_all_to_all_enabled", bool(self.broad_all_to_all_enabled))
        object.__setattr__(self, "dynamic_peer_clusters_enabled", bool(self.dynamic_peer_clusters_enabled))
        object.__setattr__(self, "cross_asset_regime_labels_enabled", bool(self.cross_asset_regime_labels_enabled))
        if self.production_enabled:
            raise ValueError("Relationship Discovery v1 production writes are disabled")
        if self.broad_all_to_all_enabled:
            raise ValueError("Relationship Discovery v1 broad all-to-all computation is disabled")
        if self.dynamic_peer_clusters_enabled:
            raise ValueError("Relationship Discovery v1 dynamic peer clusters are not enabled")
        if self.cross_asset_regime_labels_enabled:
            raise ValueError("Relationship Discovery v1 does not create Cross-Asset regime labels")

    @property
    def intervals(self) -> tuple[int, ...]:
        return self.interval_policy.intervals

    def method_config(
        self,
        *,
        window_days: Sequence[int] = (30, 90, 180),
        max_pair_count: int = 2500,
    ) -> RelationshipMethodComparisonConfig:
        return RelationshipMethodComparisonConfig(
            methods=self.method_policy.method_order,
            window_days=window_days,
            min_observations=self.method_policy.min_observations,
            min_coverage=self.method_policy.min_coverage,
            top_k_per_asset=self.k_policy.sensitivity_k,
            max_pair_count=max_pair_count,
            edge_threshold=self.method_policy.residual_min_abs_strength,
        )

    def residual_selection_config(self) -> RelationshipEdgeSelectionConfig:
        return RelationshipEdgeSelectionConfig(
            min_coverage=self.method_policy.min_coverage,
            min_sample_count=self.method_policy.min_observations,
            min_abs_strength=self.method_policy.residual_min_abs_strength,
            top_k_per_asset=self.k_policy.primary_k,
            require_window_survival=self.method_policy.require_window_survival,
            min_survival_count=self.method_policy.min_survival_count,
            min_survival_share=self.method_policy.min_survival_share,
            relationship_types=(METHOD_RESIDUAL_CORR, METHOD_BETA_TO_CORE, METHOD_RAW_ROLLING_CORR),
            include_market_mode_edges=self.method_policy.include_market_mode_edges,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": RELATIONSHIP_POLICY_ARTIFACT_KIND,
            "interval_policy": self.interval_policy.as_dict(),
            "k_policy": self.k_policy.as_dict(),
            "method_policy": self.method_policy.as_dict(),
            "satellite_policy": self.satellite_policy.as_dict(),
            "production_enabled": False,
            "broad_all_to_all_enabled": False,
            "dynamic_peer_clusters_enabled": False,
            "cross_asset_regime_labels_enabled": False,
        }

    def to_json(self) -> str:
        return dumps_json(self.as_dict())


def relationship_policy_from_mapping(payload: Mapping[str, Any]) -> RelationshipDiscoveryPolicy:
    return RelationshipDiscoveryPolicy(
        interval_policy=RelationshipIntervalPolicy(**dict(payload.get("interval_policy", {}))),
        method_policy=RelationshipMethodPolicy(**dict(payload.get("method_policy", {}))),
        k_policy=RelationshipKPolicy(**dict(payload.get("k_policy", {}))),
        satellite_policy=SatelliteRelationshipPolicy(**dict(payload.get("satellite_policy", {}))),
        production_enabled=bool(payload.get("production_enabled", False)),
        broad_all_to_all_enabled=bool(payload.get("broad_all_to_all_enabled", False)),
        dynamic_peer_clusters_enabled=bool(payload.get("dynamic_peer_clusters_enabled", False)),
        cross_asset_regime_labels_enabled=bool(payload.get("cross_asset_regime_labels_enabled", False)),
    )


def _interval(value: object, *, field_name: str) -> int:
    out = _positive_int(value, field_name=field_name)
    if out < 60:
        raise ValueError(f"Relationship Discovery v1 {field_name} must be 60 minutes or greater")
    return out


def _bounded_k(value: object, *, max_k: int, field_name: str) -> int:
    out = _positive_int(value, field_name=field_name)
    if out > int(max_k):
        raise ValueError(f"Relationship Discovery v1 {field_name} must be less than or equal to max_k")
    return out


def _positive_int(value: object, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery v1 {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Relationship Discovery v1 {field_name} must be positive")
    return out


def _non_negative(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery v1 {field_name} must be numeric") from exc
    if out < 0.0:
        raise ValueError(f"Relationship Discovery v1 {field_name} must be non-negative")
    return out


def _share(value: object, *, field_name: str) -> float:
    out = _non_negative(value, field_name=field_name)
    if out > 1.0:
        raise ValueError(f"Relationship Discovery v1 {field_name} must be between 0 and 1")
    return out


__all__ = [
    "INTERVAL_ROLE_CONFIRMATION_EVIDENCE",
    "INTERVAL_ROLE_GATED_DISABLED",
    "INTERVAL_ROLE_PRIMARY_PERSISTENT",
    "INTERVAL_ROLE_PROBE_EVIDENCE",
    "K_POLICY_ADAPTIVE_PROTOTYPE",
    "K_POLICY_PRIMARY",
    "K_POLICY_SENSITIVITY",
    "METHOD_ROLE_DIAGNOSTIC",
    "METHOD_ROLE_MARKET_EXPOSURE",
    "METHOD_ROLE_MARKET_MODE_RELATIONSHIP",
    "METHOD_ROLE_PEER_LIKE_RELATIONSHIP",
    "METHOD_ROLE_RISK_NEIGHBORHOOD_SIDECAR",
    "RELATIONSHIP_POLICY_ARTIFACT_KIND",
    "SATELLITE_POLICY_EXCLUDED_DEFAULT_PEER_DISCOVERY",
    "SATELLITE_POLICY_EXPLICIT_SIDECAR_ONLY",
    "RelationshipDiscoveryPolicy",
    "RelationshipIntervalPolicy",
    "RelationshipKPolicy",
    "RelationshipMethodPolicy",
    "SatelliteRelationshipPolicy",
    "relationship_policy_from_mapping",
]
