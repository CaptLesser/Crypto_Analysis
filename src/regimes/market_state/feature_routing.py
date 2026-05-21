from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.universe_views import MarketStateUniverseV1Views, MarketStateUniverseView


VIEW_EFFECTIVE_CORE = "effective_core"
VIEW_BROAD_CLEAN_RISK = "broad_clean_risk"
VIEW_BROAD_WITH_SATELLITES = "broad_with_satellites"
VIEW_SPECULATIVE_SATELLITE = "speculative_satellite"
VIEW_STABLE_PEG_PANEL = "stable_peg_panel"
VIEW_EXCLUDED = "excluded"
VIEW_NEEDS_REVIEW = "needs_review"

MARKET_STATE_ROUTABLE_UNIVERSE_VIEWS: tuple[str, ...] = (
    VIEW_EFFECTIVE_CORE,
    VIEW_BROAD_CLEAN_RISK,
    VIEW_BROAD_WITH_SATELLITES,
    VIEW_SPECULATIVE_SATELLITE,
    VIEW_STABLE_PEG_PANEL,
)
MARKET_STATE_IGNORED_UNIVERSE_VIEWS: tuple[str, ...] = (VIEW_EXCLUDED, VIEW_NEEDS_REVIEW)

FEATURE_MARKET_RETURN_SUMMARY = "market_return_summary"
FEATURE_MARKET_TREND = "market_trend"
FEATURE_MARKET_REALIZED_VOLATILITY = "market_realized_volatility"
FEATURE_MARKET_CORE_VOLATILITY = "market_core_volatility"
FEATURE_MARKET_CORRELATION = "market_correlation"
FEATURE_MARKET_CORRELATION_SUMMARY = "market_correlation_summary"
FEATURE_MARKET_COVARIANCE_SUMMARY = "market_covariance_summary"
FEATURE_MARKET_CONCENTRATION = "market_concentration"
FEATURE_MARKET_TURBULENCE = "market_turbulence"
FEATURE_MARKET_CORE_STRESS = "market_core_stress"
FEATURE_MARKET_STRESS = "market_stress"
FEATURE_MARKET_BREADTH = "market_breadth"
FEATURE_MARKET_DISPERSION = "market_dispersion"
FEATURE_MARKET_DRAWDOWN_BREADTH = "market_drawdown_breadth"
FEATURE_MARKET_LIQUIDITY_ACTIVITY = "market_liquidity_activity"
FEATURE_MARKET_SPECULATIVE_EUPHORIA_BREADTH = "market_speculative_euphoria_breadth"
FEATURE_MARKET_SATELLITE_ACTIVITY = "market_satellite_activity"
FEATURE_MARKET_SPECULATIVE_SIDECAR = "market_speculative_sidecar"
FEATURE_MARKET_STABLE_PEG_STRESS = "market_stable_peg_stress"
FEATURE_MARKET_PEG_DEVIATION = "market_peg_deviation"
FEATURE_MARKET_STABLE_ACTIVITY_STRESS = "market_stable_activity_stress"

DEFAULT_MARKET_STATE_ROUTED_FEATURE_FAMILIES: tuple[str, ...] = (
    FEATURE_MARKET_RETURN_SUMMARY,
    FEATURE_MARKET_TREND,
    FEATURE_MARKET_REALIZED_VOLATILITY,
    FEATURE_MARKET_CORE_VOLATILITY,
    FEATURE_MARKET_CORRELATION,
    FEATURE_MARKET_CORRELATION_SUMMARY,
    FEATURE_MARKET_COVARIANCE_SUMMARY,
    FEATURE_MARKET_CONCENTRATION,
    FEATURE_MARKET_TURBULENCE,
    FEATURE_MARKET_CORE_STRESS,
    FEATURE_MARKET_STRESS,
    FEATURE_MARKET_BREADTH,
    FEATURE_MARKET_DISPERSION,
    FEATURE_MARKET_DRAWDOWN_BREADTH,
    FEATURE_MARKET_LIQUIDITY_ACTIVITY,
    FEATURE_MARKET_SPECULATIVE_EUPHORIA_BREADTH,
    FEATURE_MARKET_SATELLITE_ACTIVITY,
    FEATURE_MARKET_SPECULATIVE_SIDECAR,
    FEATURE_MARKET_STABLE_PEG_STRESS,
    FEATURE_MARKET_PEG_DEVIATION,
    FEATURE_MARKET_STABLE_ACTIVITY_STRESS,
)


@dataclass(frozen=True)
class MarketStateFeatureRoute:
    feature_family_id: str
    required_universe_view: str
    purpose: str
    enabled_by_default: bool = True
    forbidden_universe_views: Sequence[str] = ()
    include_needs_review_allowed: bool = False
    include_excluded_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = _text(self.feature_family_id, field_name="feature_family_id")
        view = _view_name(self.required_universe_view)
        forbidden = tuple(_view_name(value) for value in self.forbidden_universe_views)
        if view in MARKET_STATE_IGNORED_UNIVERSE_VIEWS:
            raise ValueError("Market-State feature routes cannot target excluded or needs_review by default")
        object.__setattr__(self, "feature_family_id", family)
        object.__setattr__(self, "required_universe_view", view)
        object.__setattr__(self, "purpose", _text(self.purpose, field_name="purpose"))
        object.__setattr__(self, "enabled_by_default", bool(self.enabled_by_default))
        object.__setattr__(self, "forbidden_universe_views", tuple(dict.fromkeys(forbidden)))
        object.__setattr__(self, "include_needs_review_allowed", bool(self.include_needs_review_allowed))
        object.__setattr__(self, "include_excluded_allowed", bool(self.include_excluded_allowed))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_family_id": self.feature_family_id,
            "required_universe_view": self.required_universe_view,
            "purpose": self.purpose,
            "enabled_by_default": bool(self.enabled_by_default),
            "forbidden_universe_views": list(self.forbidden_universe_views),
            "include_needs_review_allowed": bool(self.include_needs_review_allowed),
            "include_excluded_allowed": bool(self.include_excluded_allowed),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class MarketStateFeatureRouteResolution:
    feature_family_id: str
    required_universe_view: str
    enabled: bool
    assets: tuple[str, ...]
    source_assets: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def asset_count(self) -> int:
        return len(self.assets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_family_id": self.feature_family_id,
            "required_universe_view": self.required_universe_view,
            "enabled": bool(self.enabled),
            "assets": list(self.assets),
            "asset_count": self.asset_count,
            "source_assets": list(self.source_assets),
            "reason_codes": list(self.reason_codes),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class MarketStateFeatureRoutingPolicy:
    routes: Mapping[str, MarketStateFeatureRoute | Mapping[str, Any]]
    policy_id: str = "market_state_feature_routing_policy_v1"
    schema_version: int = 1
    ignored_views: Sequence[str] = MARKET_STATE_IGNORED_UNIVERSE_VIEWS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        routes: dict[str, MarketStateFeatureRoute] = {}
        for key, value in self.routes.items():
            route = value if isinstance(value, MarketStateFeatureRoute) else MarketStateFeatureRoute(**dict(value))
            if str(key) != route.feature_family_id:
                raise ValueError("Market-State feature routing keys must match feature_family_id")
            routes[route.feature_family_id] = route
        if not routes:
            raise ValueError("Market-State feature routing policy requires at least one route")
        object.__setattr__(self, "routes", dict(sorted(routes.items())))
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "ignored_views", tuple(_view_name(view) for view in self.ignored_views))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def feature_family_ids(self) -> tuple[str, ...]:
        return tuple(self.routes)

    def route_for(self, feature_family_id: str) -> MarketStateFeatureRoute:
        family = _text(feature_family_id, field_name="feature_family_id")
        try:
            return self.routes[family]
        except KeyError as exc:
            raise ValueError(f"No Market-State v1 universe route declared for feature family {family!r}") from exc

    def validate_feature_family_declarations(self, feature_family_ids: Sequence[str]) -> None:
        requested = tuple(_text(family, field_name="feature_family_id") for family in feature_family_ids)
        missing = [family for family in requested if family not in self.routes]
        if missing:
            raise ValueError(f"Market-State feature families missing universe routes: {missing}")

    def validate_universe_views(
        self,
        views: MarketStateUniverseV1Views,
        *,
        include_optional: bool = False,
        include_needs_review: bool = False,
    ) -> None:
        problems: list[str] = []
        excluded = set(views.excluded.members)
        needs_review = set(views.needs_review.members)
        for route in self.routes.values():
            if not route.enabled_by_default and not include_optional:
                continue
            if route.required_universe_view == VIEW_NEEDS_REVIEW and not include_needs_review:
                continue
            view = views.view(route.required_universe_view)
            view_members = set(view.members)
            if not route.include_excluded_allowed and view_members.intersection(excluded):
                problems.append(f"{route.feature_family_id} includes excluded assets")
            if not include_needs_review and not route.include_needs_review_allowed and view_members.intersection(needs_review):
                problems.append(f"{route.feature_family_id} includes needs-review assets")
            for forbidden_view_name in route.forbidden_universe_views:
                forbidden_members = set(views.view(forbidden_view_name).members)
                overlap = sorted(view_members.intersection(forbidden_members))
                if overlap:
                    problems.append(
                        f"{route.feature_family_id} route {route.required_universe_view} overlaps forbidden "
                        f"{forbidden_view_name}: {overlap[:5]}"
                    )
        if problems:
            raise ValueError("Market-State feature routing validation failed: " + "; ".join(problems))

    def resolve(
        self,
        feature_family_id: str,
        views: MarketStateUniverseV1Views,
        *,
        include_optional: bool = False,
        include_needs_review: bool = False,
    ) -> MarketStateFeatureRouteResolution:
        route = self.route_for(feature_family_id)
        if not route.enabled_by_default and not include_optional:
            return MarketStateFeatureRouteResolution(
                feature_family_id=route.feature_family_id,
                required_universe_view=route.required_universe_view,
                enabled=False,
                assets=(),
                source_assets=(),
                reason_codes=("optional_route_disabled",),
                metadata=route.as_dict(),
            )
        if route.required_universe_view == VIEW_NEEDS_REVIEW and not include_needs_review:
            return MarketStateFeatureRouteResolution(
                feature_family_id=route.feature_family_id,
                required_universe_view=route.required_universe_view,
                enabled=False,
                assets=(),
                source_assets=(),
                reason_codes=("needs_review_ignored_by_default",),
                metadata=route.as_dict(),
            )
        self.validate_universe_views(
            views,
            include_optional=include_optional,
            include_needs_review=include_needs_review,
        )
        view = views.view(route.required_universe_view)
        return MarketStateFeatureRouteResolution(
            feature_family_id=route.feature_family_id,
            required_universe_view=route.required_universe_view,
            enabled=True,
            assets=tuple(view.members),
            source_assets=tuple(view.source_assets),
            metadata=route.as_dict(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_feature_routing_policy",
            "policy_id": self.policy_id,
            "routes": {family: route.as_dict() for family, route in self.routes.items()},
            "ignored_views": list(self.ignored_views),
            "excluded_ignored_by_default": VIEW_EXCLUDED in self.ignored_views,
            "needs_review_ignored_by_default": VIEW_NEEDS_REVIEW in self.ignored_views,
            "permanent_needs_review_classification": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }


def default_market_state_feature_routing_policy() -> MarketStateFeatureRoutingPolicy:
    return MarketStateFeatureRoutingPolicy(
        routes={route.feature_family_id: route for route in _default_routes()},
        metadata={
            "market_state_v1": True,
            "final_feature_selection_claimed": False,
            "pairwise_cross_asset_enabled": False,
        },
    )


def resolve_market_state_feature_route(
    feature_family_id: str,
    views: MarketStateUniverseV1Views,
    *,
    policy: MarketStateFeatureRoutingPolicy | None = None,
    include_optional: bool = False,
    include_needs_review: bool = False,
) -> MarketStateFeatureRouteResolution:
    routing = policy or default_market_state_feature_routing_policy()
    return routing.resolve(
        feature_family_id,
        views,
        include_optional=include_optional,
        include_needs_review=include_needs_review,
    )


def validate_market_state_feature_routing(
    views: MarketStateUniverseV1Views,
    *,
    feature_family_ids: Sequence[str] = DEFAULT_MARKET_STATE_ROUTED_FEATURE_FAMILIES,
    policy: MarketStateFeatureRoutingPolicy | None = None,
    include_optional: bool = False,
    include_needs_review: bool = False,
) -> None:
    routing = policy or default_market_state_feature_routing_policy()
    routing.validate_feature_family_declarations(feature_family_ids)
    routing.validate_universe_views(views, include_optional=include_optional, include_needs_review=include_needs_review)


def _default_routes() -> tuple[MarketStateFeatureRoute, ...]:
    effective_forbidden = (
        VIEW_BROAD_CLEAN_RISK,
        VIEW_BROAD_WITH_SATELLITES,
        VIEW_SPECULATIVE_SATELLITE,
        VIEW_STABLE_PEG_PANEL,
        VIEW_EXCLUDED,
        VIEW_NEEDS_REVIEW,
    )
    broad_clean_forbidden = (
        VIEW_SPECULATIVE_SATELLITE,
        VIEW_STABLE_PEG_PANEL,
        VIEW_EXCLUDED,
        VIEW_NEEDS_REVIEW,
    )
    broad_with_forbidden = (VIEW_STABLE_PEG_PANEL, VIEW_EXCLUDED, VIEW_NEEDS_REVIEW)
    speculative_forbidden = (
        VIEW_EFFECTIVE_CORE,
        VIEW_BROAD_CLEAN_RISK,
        VIEW_STABLE_PEG_PANEL,
        VIEW_EXCLUDED,
        VIEW_NEEDS_REVIEW,
    )
    stable_forbidden = (
        VIEW_EFFECTIVE_CORE,
        VIEW_BROAD_CLEAN_RISK,
        VIEW_BROAD_WITH_SATELLITES,
        VIEW_SPECULATIVE_SATELLITE,
        VIEW_EXCLUDED,
        VIEW_NEEDS_REVIEW,
    )
    return (
        _route(FEATURE_MARKET_RETURN_SUMMARY, VIEW_EFFECTIVE_CORE, "market return/trend", effective_forbidden),
        _route(FEATURE_MARKET_TREND, VIEW_EFFECTIVE_CORE, "market trend", effective_forbidden),
        _route(FEATURE_MARKET_REALIZED_VOLATILITY, VIEW_EFFECTIVE_CORE, "core volatility", effective_forbidden),
        _route(FEATURE_MARKET_CORE_VOLATILITY, VIEW_EFFECTIVE_CORE, "core volatility", effective_forbidden),
        _route(FEATURE_MARKET_CORRELATION, VIEW_EFFECTIVE_CORE, "covariance/correlation", effective_forbidden),
        _route(FEATURE_MARKET_CORRELATION_SUMMARY, VIEW_EFFECTIVE_CORE, "covariance/correlation", effective_forbidden),
        _route(FEATURE_MARKET_COVARIANCE_SUMMARY, VIEW_EFFECTIVE_CORE, "covariance/correlation", effective_forbidden),
        _route(FEATURE_MARKET_CONCENTRATION, VIEW_EFFECTIVE_CORE, "concentration", effective_forbidden),
        _route(FEATURE_MARKET_TURBULENCE, VIEW_EFFECTIVE_CORE, "turbulence", effective_forbidden),
        _route(FEATURE_MARKET_CORE_STRESS, VIEW_EFFECTIVE_CORE, "core stress", effective_forbidden),
        _route(FEATURE_MARKET_STRESS, VIEW_EFFECTIVE_CORE, "core stress", effective_forbidden),
        _route(FEATURE_MARKET_BREADTH, VIEW_BROAD_CLEAN_RISK, "ordinary breadth", broad_clean_forbidden),
        _route(FEATURE_MARKET_DISPERSION, VIEW_BROAD_CLEAN_RISK, "ordinary dispersion", broad_clean_forbidden),
        _route(FEATURE_MARKET_DRAWDOWN_BREADTH, VIEW_BROAD_CLEAN_RISK, "drawdown breadth", broad_clean_forbidden),
        _route(FEATURE_MARKET_LIQUIDITY_ACTIVITY, VIEW_BROAD_CLEAN_RISK, "broad activity participation", broad_clean_forbidden),
        _route(
            FEATURE_MARKET_SPECULATIVE_EUPHORIA_BREADTH,
            VIEW_BROAD_WITH_SATELLITES,
            "optional speculative/euphoria breadth",
            broad_with_forbidden,
            enabled_by_default=False,
        ),
        _route(
            FEATURE_MARKET_SATELLITE_ACTIVITY,
            VIEW_BROAD_WITH_SATELLITES,
            "optional satellite-inclusive activity",
            broad_with_forbidden,
            enabled_by_default=False,
        ),
        _route(
            FEATURE_MARKET_SPECULATIVE_SIDECAR,
            VIEW_SPECULATIVE_SATELLITE,
            "speculative sidecar features only",
            speculative_forbidden,
            enabled_by_default=False,
        ),
        _route(FEATURE_MARKET_STABLE_PEG_STRESS, VIEW_STABLE_PEG_PANEL, "stable/peg stress", stable_forbidden),
        _route(FEATURE_MARKET_PEG_DEVIATION, VIEW_STABLE_PEG_PANEL, "peg deviation", stable_forbidden),
        _route(FEATURE_MARKET_STABLE_ACTIVITY_STRESS, VIEW_STABLE_PEG_PANEL, "stable activity stress", stable_forbidden),
    )


def _route(
    feature_family_id: str,
    view: str,
    purpose: str,
    forbidden: Sequence[str],
    *,
    enabled_by_default: bool = True,
) -> MarketStateFeatureRoute:
    return MarketStateFeatureRoute(
        feature_family_id=feature_family_id,
        required_universe_view=view,
        purpose=purpose,
        enabled_by_default=enabled_by_default,
        forbidden_universe_views=forbidden,
    )


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Market-State feature routing {field_name} must be non-empty")
    return text


def _view_name(value: object) -> str:
    text = _text(value, field_name="universe_view").lower()
    valid = set(MARKET_STATE_ROUTABLE_UNIVERSE_VIEWS).union(MARKET_STATE_IGNORED_UNIVERSE_VIEWS)
    if text not in valid:
        raise ValueError(f"Unsupported Market-State feature routing universe view {text!r}")
    return text


__all__ = [
    "DEFAULT_MARKET_STATE_ROUTED_FEATURE_FAMILIES",
    "MARKET_STATE_IGNORED_UNIVERSE_VIEWS",
    "MARKET_STATE_ROUTABLE_UNIVERSE_VIEWS",
    "MarketStateFeatureRoute",
    "MarketStateFeatureRouteResolution",
    "MarketStateFeatureRoutingPolicy",
    "default_market_state_feature_routing_policy",
    "resolve_market_state_feature_route",
    "validate_market_state_feature_routing",
]
