from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.asset_state.contracts import (
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateBand,
    AssetStateSchemaVersion,
)
from src.regimes.asset_state.contracts import _enum_value, _mapping, _non_empty_text, _schema_version, _string_tuple
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.serialization import dumps_json, to_jsonable


@dataclass(frozen=True)
class AssetStateFeaturePoolSpec:
    pool_id: str
    axis: str | AssetStateAxis
    feature_bases: Sequence[str]
    compatible_bands: Sequence[str | AssetStateBand] = (AssetStateBand.MICRO, AssetStateBand.MESO, AssetStateBand.MACRO)
    expected_source_kind: str = "scalar_feature_parquet"
    missingness_policy: Mapping[str, Any] = field(
        default_factory=lambda: {
            "policy": "train_window_feature_filter_then_drop_nonfinite_rows",
            "max_missing_fraction": 0.35,
            "fail_closed": True,
        }
    )
    leakage_policy: str = "source_features_only_no_forward_targets"
    implementation_status: str = "implemented"
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        axis = _enum_value(self.axis, AssetStateAxis, field_name="axis")
        bands = tuple(_enum_value(band, AssetStateBand, field_name="compatible_bands") for band in self.compatible_bands)
        if not bands:
            raise ValueError("Asset-state feature pool compatible_bands must be non-empty")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "pool_id", _non_empty_text(self.pool_id, field_name="feature pool id"))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "feature_bases", _string_tuple(self.feature_bases, field_name="feature_bases", require_non_empty=True))
        object.__setattr__(self, "compatible_bands", tuple(dict.fromkeys(bands)))
        object.__setattr__(self, "expected_source_kind", _non_empty_text(self.expected_source_kind, field_name="expected_source_kind"))
        policy = _mapping(self.missingness_policy, field_name="missingness_policy")
        if not bool(policy.get("fail_closed", False)):
            raise ValueError("Asset-state feature pool missingness policy must fail closed")
        object.__setattr__(self, "missingness_policy", to_jsonable(policy))
        object.__setattr__(self, "leakage_policy", _non_empty_text(self.leakage_policy, field_name="leakage_policy"))
        object.__setattr__(self, "implementation_status", _non_empty_text(self.implementation_status, field_name="implementation_status"))

    def supports(self, *, axis: str | AssetStateAxis, band: str | AssetStateBand) -> bool:
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        band_value = _enum_value(band, AssetStateBand, field_name="band")
        return axis_value == self.axis and band_value in self.compatible_bands

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "pool_id": self.pool_id,
            "axis": self.axis,
            "feature_bases": list(self.feature_bases),
            "compatible_bands": list(self.compatible_bands),
            "expected_source_kind": self.expected_source_kind,
            "missingness_policy": to_jsonable(self.missingness_policy),
            "leakage_policy": self.leakage_policy,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class AssetStateFeaturePoolRegistry:
    pools: Mapping[str, AssetStateFeaturePoolSpec | Mapping[str, Any]]
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized: dict[str, AssetStateFeaturePoolSpec] = {}
        for key, value in self.pools.items():
            spec = value if isinstance(value, AssetStateFeaturePoolSpec) else AssetStateFeaturePoolSpec(**dict(value))
            if str(key) != spec.pool_id:
                raise ValueError("Asset-state feature pool registry keys must match pool_id")
            normalized[spec.pool_id] = spec
        axes = set(default_asset_state_taxonomy().axes)
        represented_axes = {spec.axis for spec in normalized.values()}
        missing = sorted(axes.difference(represented_axes))
        if missing:
            raise ValueError(f"Asset-state feature pool registry is missing axes: {missing}")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "pools", dict(sorted(normalized.items())))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.pools)

    def get(self, pool_id: str) -> AssetStateFeaturePoolSpec:
        key = _non_empty_text(pool_id, field_name="pool_id")
        try:
            return self.pools[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported asset-state feature pool {key!r}; expected one of: {', '.join(self.names)}") from exc

    def default_for_axis(self, axis: str | AssetStateAxis, *, band: str | AssetStateBand | None = None) -> AssetStateFeaturePoolSpec:
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        band_value = None if band is None else _enum_value(band, AssetStateBand, field_name="band")
        for spec in self.pools.values():
            if spec.axis != axis_value:
                continue
            if band_value is None or band_value in spec.compatible_bands:
                return spec
        raise ValueError(f"No asset-state feature pool registered for axis={axis_value!r} band={band_value!r}")

    def by_axis(self, axis: str | AssetStateAxis) -> tuple[AssetStateFeaturePoolSpec, ...]:
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        return tuple(spec for spec in self.pools.values() if spec.axis == axis_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "asset_state_feature_pool_registry",
            "pools": {key: spec.as_dict() for key, spec in self.pools.items()},
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


def default_asset_state_feature_pool_registry() -> AssetStateFeaturePoolRegistry:
    pools = (
        AssetStateFeaturePoolSpec(
            pool_id="asset_state_trend_core",
            axis=AssetStateAxis.TREND,
            feature_bases=(
                "log_return",
                "macd_hist_12_26_9",
                "rsi_14",
                "adx_14",
                "aroon_osc_25",
                "plus_di_14",
                "minus_di_14",
                "vi_plus_14",
                "vi_minus_14",
                "roc_14",
                "mom_14",
                "range_efficiency_20",
            ),
        ),
        AssetStateFeaturePoolSpec(
            pool_id="asset_state_volatility_core",
            axis=AssetStateAxis.VOLATILITY,
            feature_bases=(
                "atr_14",
                "ret_std_20",
                "cv_20",
                "vol_osc_pct_14_28",
                "true_range",
                "range_hl",
                "range_co",
                "var_20",
                "skew_20",
                "kurt_20",
                "q25_20",
                "q75_20",
                "squeeze_scalar",
            ),
        ),
        AssetStateFeaturePoolSpec(
            pool_id="asset_state_activity_core",
            axis=AssetStateAxis.ACTIVITY,
            feature_bases=(
                "activity_state_score_20",
                "volume_zscore_20",
                "trades_zscore_20",
                "trade_intensity",
                "avg_trade_size",
                "vroc_14",
                "prr",
                "dollar_volume_proxy",
                "volume_share_vs_rolling_20",
                "trade_count_intensity_zscore_20",
                "illiquidity_proxy_20",
                "obv",
                "adl",
                "force_index",
                "chaikin_osc_3_10",
                "vpt",
                "eom_14",
                "pvi",
                "nvi",
            ),
        ),
        AssetStateFeaturePoolSpec(
            pool_id="asset_state_mean_reversion_core",
            axis=AssetStateAxis.MEAN_REVERSION,
            feature_bases=(
                "rsi_14",
                "stoch_k_14",
                "stoch_d_3",
                "cci_20",
                "williams_r_14",
                "zscore_20",
                "bollinger_pct_b_20",
                "bollinger_bandwidth_20",
                "choppiness_14",
                "hurst_100",
                "range_efficiency_20",
                "log_return",
            ),
        ),
        AssetStateFeaturePoolSpec(
            pool_id="asset_state_drawdown_core",
            axis=AssetStateAxis.DRAWDOWN,
            feature_bases=(
                "drawdown_from_rolling_high_20",
                "rolling_max_drawdown_20",
                "downside_excursion_20",
                "downside_vol_20",
                "ulcer_index_14",
                "atr_14",
                "ret_std_20",
                "true_range",
                "log_return",
                "d_close_5",
                "d_close_20",
            ),
        ),
        AssetStateFeaturePoolSpec(
            pool_id="asset_state_range_efficiency_core",
            axis=AssetStateAxis.RANGE_EFFICIENCY,
            feature_bases=(
                "range_efficiency_20",
                "true_range",
                "range_hl",
                "range_co",
                "atr_14",
                "adx_14",
                "log_return",
                "path_efficiency_20",
                "directional_efficiency_20",
                "runup_drawdown_ratio_20",
                "drawdown_from_rolling_high_20",
                "close_location_value",
                "rolling_position_in_range_20",
            ),
        ),
    )
    return AssetStateFeaturePoolRegistry({spec.pool_id: spec for spec in pools})


def feature_bases_for_axis(axis: str | AssetStateAxis, *, pool_id: str | None = None) -> tuple[str, ...]:
    registry = default_asset_state_feature_pool_registry()
    spec = registry.get(pool_id) if pool_id else registry.default_for_axis(axis)
    return tuple(spec.feature_bases)


def canonical_interval_feature_name(interval: int, base: str) -> str:
    return f"i{int(interval)}_{str(base)}"


def select_interval_feature_columns(
    frame: pd.DataFrame,
    *,
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    member_intervals: Sequence[int],
    pool_id: str | None = None,
) -> tuple[str, ...]:
    registry = default_asset_state_feature_pool_registry()
    spec = registry.get(pool_id) if pool_id else registry.default_for_axis(axis, band=band)
    if not spec.supports(axis=axis, band=band):
        raise ValueError(f"Asset-state feature pool {spec.pool_id!r} is not compatible with axis/band")
    selected: list[str] = []
    columns = set(str(column) for column in frame.columns)
    for interval in member_intervals:
        for base in spec.feature_bases:
            canonical = canonical_interval_feature_name(int(interval), base)
            if canonical in columns:
                selected.append(canonical)
    return tuple(dict.fromkeys(selected))


__all__ = [
    "AssetStateFeaturePoolRegistry",
    "AssetStateFeaturePoolSpec",
    "canonical_interval_feature_name",
    "default_asset_state_feature_pool_registry",
    "feature_bases_for_axis",
    "select_interval_feature_columns",
]
