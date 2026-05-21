from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.market_state.contracts import (
    MARKET_STATE_AXIS_VALUES,
    MARKET_STATE_BAND_VALUES,
    MARKET_STATE_SCHEMA_VERSION,
    MarketStateAxis,
    MarketStateBand,
    MarketStateSchemaVersion,
    _enum_value,
    _non_empty_text,
    _schema_version,
    _string_tuple,
)


MARKET_FEATURE_SCOPE_CORE = "core_basket"
MARKET_FEATURE_SCOPE_BROAD = "broad_universe"
MARKET_FEATURE_SCOPE_BOTH = "core_and_broad"
MARKET_FEATURE_SCOPES: tuple[str, ...] = (
    MARKET_FEATURE_SCOPE_CORE,
    MARKET_FEATURE_SCOPE_BROAD,
    MARKET_FEATURE_SCOPE_BOTH,
)


@dataclass(frozen=True)
class MarketFeatureFamilySpec:
    family_id: str
    purpose: str
    scope: str
    compatible_axes: Sequence[str | MarketStateAxis]
    compatible_bands: Sequence[str | MarketStateBand]
    required_base_series: Sequence[str]
    output_features: Sequence[str]
    uses_core_basket: bool = False
    uses_broad_universe: bool = False
    requires_rolling_window: bool = False
    requires_covariance_correlation: bool = False
    allow_unavailable: bool = False
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family_id = _non_empty_text(self.family_id, field_name="family_id")
        scope = _non_empty_text(self.scope, field_name="scope")
        if scope not in MARKET_FEATURE_SCOPES:
            raise ValueError(f"Unsupported market-state feature scope {scope!r}")
        axes = tuple(_enum_value(axis, MarketStateAxis, field_name="compatible_axes") for axis in self.compatible_axes)
        bands = tuple(_enum_value(band, MarketStateBand, field_name="compatible_bands") for band in self.compatible_bands)
        if not axes or not bands:
            raise ValueError("Market-state feature family must declare compatible axes and bands")
        output_features = _string_tuple(self.output_features, field_name="output_features", require_non_empty=True)
        required = _string_tuple(self.required_base_series, field_name="required_base_series")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "purpose", _non_empty_text(self.purpose, field_name="purpose"))
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "compatible_axes", tuple(dict.fromkeys(axes)))
        object.__setattr__(self, "compatible_bands", tuple(dict.fromkeys(bands)))
        object.__setattr__(self, "required_base_series", required)
        object.__setattr__(self, "output_features", output_features)
        object.__setattr__(self, "uses_core_basket", bool(self.uses_core_basket))
        object.__setattr__(self, "uses_broad_universe", bool(self.uses_broad_universe))
        object.__setattr__(self, "requires_rolling_window", bool(self.requires_rolling_window))
        object.__setattr__(self, "requires_covariance_correlation", bool(self.requires_covariance_correlation))
        object.__setattr__(self, "allow_unavailable", bool(self.allow_unavailable))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def supports(self, *, axis: str | MarketStateAxis, band: str | MarketStateBand) -> bool:
        axis_value = _enum_value(axis, MarketStateAxis, field_name="axis")
        band_value = _enum_value(band, MarketStateBand, field_name="band")
        return axis_value in self.compatible_axes and band_value in self.compatible_bands

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_feature_family_spec",
            "family_id": self.family_id,
            "purpose": self.purpose,
            "scope": self.scope,
            "compatible_axes": list(self.compatible_axes),
            "compatible_bands": list(self.compatible_bands),
            "required_base_series": list(self.required_base_series),
            "output_features": list(self.output_features),
            "uses_core_basket": bool(self.uses_core_basket),
            "uses_broad_universe": bool(self.uses_broad_universe),
            "requires_rolling_window": bool(self.requires_rolling_window),
            "requires_covariance_correlation": bool(self.requires_covariance_correlation),
            "allow_unavailable": bool(self.allow_unavailable),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketFeatureFamilySpec":
        obj = require_json_object(payload, context="MarketFeatureFamilySpec")
        obj.pop("artifact_kind", None)
        return cls(**obj)


@dataclass(frozen=True)
class MarketFeatureRegistry:
    families: Mapping[str, MarketFeatureFamilySpec | Mapping[str, Any]]
    registry_id: str = "default_market_state_feature_registry"
    artifact_kind: str = "market_state_feature_registry"
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        families: dict[str, MarketFeatureFamilySpec] = {}
        for key, value in self.families.items():
            spec = value if isinstance(value, MarketFeatureFamilySpec) else MarketFeatureFamilySpec.from_dict(value)
            if str(key) != spec.family_id:
                raise ValueError("Market-state feature registry keys must match family_id")
            families[spec.family_id] = spec
        if not families:
            raise ValueError("Market-state feature registry must contain at least one family")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "registry_id", _non_empty_text(self.registry_id, field_name="registry_id"))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "families", dict(sorted(families.items())))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(self.families)

    @property
    def output_features(self) -> tuple[str, ...]:
        out: list[str] = []
        for spec in self.families.values():
            out.extend(spec.output_features)
        return tuple(dict.fromkeys(out))

    def get(self, family_id: str) -> MarketFeatureFamilySpec:
        return self.families[_non_empty_text(family_id, field_name="family_id")]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "registry_id": self.registry_id,
            "pathway": "market_state",
            "families": {key: spec.as_dict() for key, spec in sorted(self.families.items())},
            "output_features": list(self.output_features),
            "metadata": to_jsonable(dict(self.metadata)),
            "raw_asset_dimension_clustering_default": False,
            "production_feature_store_writes": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketFeatureRegistry":
        obj = require_json_object(payload, context="MarketFeatureRegistry")
        obj.pop("pathway", None)
        obj.pop("output_features", None)
        obj.pop("raw_asset_dimension_clustering_default", None)
        obj.pop("production_feature_store_writes", None)
        return cls(**obj)

    @classmethod
    def from_json(cls, text: str) -> "MarketFeatureRegistry":
        return cls.from_dict(require_json_object(loads_json(text), context="MarketFeatureRegistry JSON"))


def default_market_state_feature_family_specs() -> tuple[MarketFeatureFamilySpec, ...]:
    all_axes = tuple(MARKET_STATE_AXIS_VALUES)
    all_bands = tuple(MARKET_STATE_BAND_VALUES)
    return (
        MarketFeatureFamilySpec(
            family_id="market_return_summary",
            purpose="Market-level return direction, central tendency, and core basket activity-weighted confirmation.",
            scope=MARKET_FEATURE_SCOPE_BOTH,
            compatible_axes=(MarketStateAxis.MARKET_RETURN, MarketStateAxis.MARKET_STRESS),
            compatible_bands=all_bands,
            required_base_series=("log_return",),
            output_features=(
                "market_return_equal_weight",
                "market_return_core_activity_weighted",
                "market_return_median",
                "market_return_q10",
                "market_return_q25",
                "market_return_q75",
                "market_return_q90",
            ),
            uses_core_basket=True,
            uses_broad_universe=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_realized_volatility",
            purpose="Market proxy realized volatility and cross-sectional asset volatility levels.",
            scope=MARKET_FEATURE_SCOPE_BROAD,
            compatible_axes=(MarketStateAxis.MARKET_VOLATILITY, MarketStateAxis.MARKET_STRESS),
            compatible_bands=all_bands,
            required_base_series=("log_return", "realized_volatility_proxy"),
            output_features=(
                "market_realized_vol_proxy",
                "market_asset_vol_median",
                "market_asset_vol_q75",
                "market_asset_vol_q90",
                "market_high_vol_share",
            ),
            uses_broad_universe=True,
            requires_rolling_window=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_breadth",
            purpose="Advance/decline, trend participation, and drawdown breadth across the broad universe.",
            scope=MARKET_FEATURE_SCOPE_BROAD,
            compatible_axes=(MarketStateAxis.MARKET_BREADTH, MarketStateAxis.MARKET_RETURN, MarketStateAxis.MARKET_STRESS),
            compatible_bands=all_bands,
            required_base_series=("close", "log_return", "drawdown_proxy"),
            output_features=(
                "market_breadth_up_share",
                "market_breadth_above_trend_share",
                "market_breadth_drawdown_share",
                "market_breadth_stress_share",
            ),
            uses_broad_universe=True,
            requires_rolling_window=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_dispersion",
            purpose="Cross-sectional return and volatility spread across the broad universe.",
            scope=MARKET_FEATURE_SCOPE_BROAD,
            compatible_axes=(MarketStateAxis.MARKET_DISPERSION, MarketStateAxis.MARKET_VOLATILITY),
            compatible_bands=all_bands,
            required_base_series=("log_return", "realized_volatility_proxy"),
            output_features=(
                "market_return_dispersion_std",
                "market_return_quantile_spread_q90_q10",
                "market_tail_return_spread_q95_q05",
                "market_vol_dispersion_std",
                "market_vol_quantile_spread_q90_q10",
            ),
            uses_broad_universe=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_correlation",
            purpose="Core-basket pairwise correlation concentration and first principal component concentration.",
            scope=MARKET_FEATURE_SCOPE_CORE,
            compatible_axes=(MarketStateAxis.MARKET_CORRELATION, MarketStateAxis.MARKET_STRESS),
            compatible_bands=all_bands,
            required_base_series=("log_return",),
            output_features=(
                "market_corr_median_pairwise",
                "market_corr_q25_pairwise",
                "market_corr_q75_pairwise",
                "market_corr_high_share",
                "market_corr_first_pc_concentration",
            ),
            uses_core_basket=True,
            requires_rolling_window=True,
            requires_covariance_correlation=True,
            allow_unavailable=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_covariance_summary",
            purpose="Core-basket shrinkage covariance summaries with explicit fallback status.",
            scope=MARKET_FEATURE_SCOPE_CORE,
            compatible_axes=(MarketStateAxis.MARKET_CORRELATION, MarketStateAxis.MARKET_STRESS),
            compatible_bands=all_bands,
            required_base_series=("log_return",),
            output_features=(
                "market_cov_ledoit_wolf_trace",
                "market_cov_ledoit_wolf_mean_variance",
                "market_cov_ledoit_wolf_mean_covariance",
                "market_cov_oas_trace",
                "market_cov_oas_mean_variance",
                "market_cov_oas_mean_covariance",
            ),
            uses_core_basket=True,
            requires_rolling_window=True,
            requires_covariance_correlation=True,
            allow_unavailable=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_liquidity_activity",
            purpose="Aggregate market activity, participation breadth, and volume/trade concentration.",
            scope=MARKET_FEATURE_SCOPE_BROAD,
            compatible_axes=(MarketStateAxis.MARKET_LIQUIDITY_ACTIVITY,),
            compatible_bands=all_bands,
            required_base_series=("volume", "trades"),
            output_features=(
                "market_volume_total",
                "market_trades_total",
                "market_activity_breadth_share",
                "market_volume_top_asset_share",
                "market_trade_top_asset_share",
            ),
            uses_broad_universe=True,
            allow_unavailable=True,
        ),
        MarketFeatureFamilySpec(
            family_id="market_stress",
            purpose="Downside breadth, high-volatility/high-correlation coincidence, and broad drawdown participation.",
            scope=MARKET_FEATURE_SCOPE_BOTH,
            compatible_axes=(MarketStateAxis.MARKET_STRESS,),
            compatible_bands=all_bands,
            required_base_series=("log_return", "realized_volatility_proxy", "drawdown_proxy"),
            output_features=(
                "market_downside_breadth_share",
                "market_high_vol_high_corr_coincidence",
                "market_broad_drawdown_participation",
                "market_stress_composite",
            ),
            uses_core_basket=True,
            uses_broad_universe=True,
            requires_covariance_correlation=True,
            allow_unavailable=True,
        ),
    )


def default_market_state_feature_registry() -> MarketFeatureRegistry:
    families = {spec.family_id: spec for spec in default_market_state_feature_family_specs()}
    return MarketFeatureRegistry(
        families=families,
        metadata={
            "feature_scope_policy": "aggregate_market_features_only; raw per-asset dimensions are not default clustering inputs",
            "core_basket_usage": "correlation and covariance families",
            "broad_universe_usage": "breadth, dispersion, return summaries, volatility, activity, and stress aggregates",
        },
    )


__all__ = [
    "MARKET_FEATURE_SCOPE_BOTH",
    "MARKET_FEATURE_SCOPE_BROAD",
    "MARKET_FEATURE_SCOPE_CORE",
    "MARKET_FEATURE_SCOPES",
    "MarketFeatureFamilySpec",
    "MarketFeatureRegistry",
    "default_market_state_feature_family_specs",
    "default_market_state_feature_registry",
]
