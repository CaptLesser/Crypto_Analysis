from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.features.scalar_features import SCALAR_FEATURE_COLUMNS
from src.regimes.asset_state.dataset import LEAKAGE_COLUMN_TOKENS
from src.regimes.asset_state.dataset_builder import (
    discover_scalar_feature_assets,
    resolve_asset_state_scalar_feature_root,
    resolve_scalar_feature_partitions,
)
from src.regimes.core.serialization import dumps_json, to_jsonable


FEATURE_COLUMN_STATUS_AVAILABLE = "available"
FEATURE_COLUMN_STATUS_MISSING = "missing"
FEATURE_COLUMN_STATUS_PENDING_SCALAR_FEATURE = "pending_scalar_feature"
FEATURE_COLUMN_STATUS_SOURCED_FROM_OHLCVT = "sourced_from_ohlcvt"
FEATURE_COLUMN_STATUS_VALIDATION_TARGET_ONLY = "validation_target_only"
FEATURE_COLUMN_STATUS_UNSUPPORTED_FOR_NOW = "unsupported_for_now"
FEATURE_COLUMN_STATUSES: tuple[str, ...] = (
    FEATURE_COLUMN_STATUS_AVAILABLE,
    FEATURE_COLUMN_STATUS_MISSING,
    FEATURE_COLUMN_STATUS_PENDING_SCALAR_FEATURE,
    FEATURE_COLUMN_STATUS_SOURCED_FROM_OHLCVT,
    FEATURE_COLUMN_STATUS_VALIDATION_TARGET_ONLY,
    FEATURE_COLUMN_STATUS_UNSUPPORTED_FOR_NOW,
)

SCALAR_SOURCE_KIND = "scalar_feature"
OHLCVT_SOURCE_KIND = "ohlcvt"
VALIDATION_SOURCE_KIND = "validation_target"
UNSUPPORTED_SOURCE_KIND = "unsupported"

TREND_DIRECTIONALITY = "trend_directionality"
VOLATILITY_EXPANSION_COMPRESSION = "volatility_expansion_compression"
ACTIVITY_PARTICIPATION_LIQUIDITY = "activity_participation_liquidity"
MEAN_REVERSION_CHOP_RANGE = "mean_reversion_chop_range"
DRAWDOWN_STRESS_DOWNSIDE = "drawdown_stress_downside"
RANGE_EFFICIENCY_RUNUP_DRAWDOWN = "range_efficiency_runup_drawdown"
PRICE_TRANSFORM = "price_transform"
RAW_OHLCVT = "raw_ohlcvt"
VALIDATION_TARGETS = "validation_targets"
UNSUPPORTED_EXTERNAL = "unsupported_external"
MISC_SCALAR = "misc_scalar"

PENDING_SCALAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "trend_persistence_score_20",
    "volatility_compression_score_20",
    "participation_imbalance_20",
    "mean_reversion_pressure_20",
    "stress_recovery_score_20",
    "path_choppiness_ratio_20",
)
SOURCED_FROM_OHLCVT_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "trades")
VALIDATION_TARGET_ONLY_COLUMNS: tuple[str, ...] = (
    "forward_return_60m",
    "forward_return_240m",
    "forward_realized_volatility",
    "forward_drawdown",
    "forward_runup",
)
UNSUPPORTED_FOR_NOW_COLUMNS: tuple[str, ...] = (
    "order_book_depth",
    "funding_rate",
    "open_interest",
    "cross_exchange_basis",
)

FEATURE_CATEGORY_COLUMNS: Mapping[str, tuple[str, ...]] = {
    TREND_DIRECTIONALITY: (
        "log_return",
        "pct_change",
        "delta_close",
        "macd_12_26_9",
        "macd_signal_12_26_9",
        "macd_hist_12_26_9",
        "rsi_14",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "aroon_up_25",
        "aroon_down_25",
        "aroon_osc_25",
        "roc_14",
        "mom_14",
        "cmo_14",
        "trix_15",
        "dpo_20",
        "ultosc_7_14_28",
        "lr_slope_20",
        "lr_slope_norm_20",
        "lr_r2_20",
        "lr_slope_norm_50",
        "lr_r2_50",
        "vi_plus_14",
        "vi_minus_14",
        "dir",
    ),
    VOLATILITY_EXPANSION_COMPRESSION: (
        "true_range",
        "true_range_pct",
        "atr_14",
        "atr_pct_14",
        "ret_std_20",
        "cv_20",
        "chaikin_vol_10_10",
        "vol_osc_14_28",
        "vol_osc_pct_14_28",
        "var_20",
        "skew_20",
        "kurt_20",
        "q25_20",
        "q50_20",
        "q75_20",
        "parkinson_vol_20",
        "garman_klass_vol_20",
        "rogers_satchell_vol_20",
        "bipower_var_20",
        "jump_var_20",
        "vol_of_vol_20",
        "range_hl",
        "range_co",
        "range_expansion_ratio_20",
        "squeeze_scalar",
        "in_squeeze",
        "squeeze_breakout_pressure",
        "vol_expansion_with_negative_return",
        "range_expansion_with_direction",
    ),
    ACTIVITY_PARTICIPATION_LIQUIDITY: (
        "volume_zscore_20",
        "trades_zscore_20",
        "dollar_volume_proxy",
        "volume_share_vs_rolling_20",
        "trade_count_intensity_zscore_20",
        "activity_state_score_20",
        "illiquidity_proxy_20",
        "avg_trade_size",
        "trade_intensity",
        "prr",
        "vroc_14",
        "obv",
        "mfi_14",
        "adl",
        "force_index",
        "chaikin_osc_3_10",
        "vpt",
        "eom_14",
        "pvi",
        "nvi",
        "vpt_vol_14",
        "msv_14",
    ),
    MEAN_REVERSION_CHOP_RANGE: (
        "stoch_k_14",
        "stoch_d_3",
        "williams_r_14",
        "cci_20",
        "bollinger_pct_b_20",
        "bollinger_bandwidth_20",
        "zscore_20",
        "choppiness_14",
        "hurst_100",
        "entropy_20",
        "fractal_100",
        "tir",
        "donchian_hi_20",
        "donchian_lo_20",
        "donchian_width_pct_20",
        "rolling_position_in_range_20",
        "distance_from_mid_band_20",
        "distance_from_vwap_day",
        "prank_20",
    ),
    DRAWDOWN_STRESS_DOWNSIDE: (
        "drawdown_from_rolling_high_20",
        "drawdown_from_rolling_high_50",
        "rolling_max_drawdown_20",
        "rolling_max_drawdown_50",
        "downside_vol_20",
        "downside_vol_50",
        "ulcer_index_14",
        "ulcer_index_50",
        "downside_excursion_20",
        "recovery_ratio_20",
        "drawdown_duration_lookback",
        "negative_return_share_20",
        "omega_ratio_20",
        "upside_downside_vol_ratio_20",
        "ret_q05_50",
        "ret_q95_50",
        "ret_tail_spread_50",
        "high_vol_downside_pressure_20",
        "d_close_2",
        "d_close_3",
        "d_close_5",
        "d_close_10",
        "d_close_14",
        "d_close_20",
    ),
    RANGE_EFFICIENCY_RUNUP_DRAWDOWN: (
        "range_efficiency_20",
        "range_efficiency_50",
        "range_efficiency_100",
        "path_efficiency_20",
        "path_efficiency_50",
        "abs_return_over_true_range_sum_20",
        "directional_efficiency_20",
        "directional_efficiency_50",
        "runup_drawdown_ratio_20",
        "close_location_value",
        "body_to_range_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
    ),
    PRICE_TRANSFORM: (
        "typical_price",
        "median_price",
        "weighted_close",
        "sma_20",
        "ema_20",
        "wma_20",
        "hma_20",
        "wilder_14",
        "ma_env_upper_20_2pct",
        "ma_env_lower_20_2pct",
        "boll_mid_20",
        "boll_up_20",
        "boll_low_20",
        "keltner_mid_20",
        "keltner_up_20",
        "keltner_low_20",
        "keltner_bandwidth_20",
        "vwap_day",
        "psar",
        "tenkan_9",
        "kijun_26",
        "span_a_26",
        "span_b_26",
        "chikou_26",
        "kama_10_2_30",
        "frama_16",
        "ewm_mean_alpha_0_1",
        "ewm_mean_alpha_0_2",
    ),
}


@dataclass(frozen=True)
class AssetStateFeatureColumnCatalogEntry:
    column: str
    category: str
    status: str
    source_kind: str
    axis_affinity: Sequence[str] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        status = str(self.status).strip()
        if status not in FEATURE_COLUMN_STATUSES:
            raise ValueError(f"Unsupported Asset-State feature column status {status!r}")
        object.__setattr__(self, "column", _text(self.column, field_name="column"))
        object.__setattr__(self, "category", _text(self.category, field_name="category"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "source_kind", _text(self.source_kind, field_name="source_kind"))
        object.__setattr__(self, "axis_affinity", tuple(str(axis).strip() for axis in self.axis_affinity if str(axis).strip()))
        object.__setattr__(self, "notes", str(self.notes).strip())

    @property
    def leakage_risk(self) -> bool:
        return _is_leakage_risk_column(self.column)

    @property
    def usable_as_input_feature(self) -> bool:
        return self.status == FEATURE_COLUMN_STATUS_AVAILABLE and not self.leakage_risk

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "category": self.category,
            "status": self.status,
            "source_kind": self.source_kind,
            "axis_affinity": list(self.axis_affinity),
            "leakage_risk": self.leakage_risk,
            "usable_as_input_feature": self.usable_as_input_feature,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AssetStateFeatureColumnCatalog:
    entries: Mapping[str, AssetStateFeatureColumnCatalogEntry | Mapping[str, Any]]
    source: str = "scalar_feature_manifest"

    def __post_init__(self) -> None:
        normalized: dict[str, AssetStateFeatureColumnCatalogEntry] = {}
        for key, value in self.entries.items():
            entry = value if isinstance(value, AssetStateFeatureColumnCatalogEntry) else AssetStateFeatureColumnCatalogEntry(**dict(value))
            if str(key) != entry.column:
                raise ValueError("Asset-State feature column catalog keys must match entry column")
            normalized[entry.column] = entry
        object.__setattr__(self, "entries", dict(sorted(normalized.items())))
        object.__setattr__(self, "source", _text(self.source, field_name="source"))

    @property
    def available_columns(self) -> tuple[str, ...]:
        return tuple(column for column, entry in self.entries.items() if entry.status == FEATURE_COLUMN_STATUS_AVAILABLE)

    @property
    def grouped_by_category(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for entry in self.entries.values():
            grouped.setdefault(entry.category, []).append(entry.column)
        return {category: sorted(columns) for category, columns in sorted(grouped.items())}

    @property
    def status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in FEATURE_COLUMN_STATUSES}
        for entry in self.entries.values():
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts

    def classify(self, column: str) -> AssetStateFeatureColumnCatalogEntry:
        key = _source_base(column)
        if key in self.entries:
            return self.entries[key]
        if _is_leakage_risk_column(key):
            return AssetStateFeatureColumnCatalogEntry(
                column=key,
                category=VALIDATION_TARGETS,
                status=FEATURE_COLUMN_STATUS_VALIDATION_TARGET_ONLY,
                source_kind=VALIDATION_SOURCE_KIND,
                notes="Leakage-risk column treated as validation target only, never as an Asset-State input feature.",
            )
        return AssetStateFeatureColumnCatalogEntry(
            column=key,
            category=MISC_SCALAR,
            status=FEATURE_COLUMN_STATUS_MISSING,
            source_kind=SCALAR_SOURCE_KIND,
            notes="Column is not present in the inspected scalar-feature schema or declared support lists.",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "asset_state_feature_column_catalog",
            "source": self.source,
            "status_counts": self.status_counts,
            "grouped_by_category": self.grouped_by_category,
            "entries": {column: entry.as_dict() for column, entry in self.entries.items()},
            "production_profile_selection_enabled": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class ScalarFeatureSchemaInspection:
    status: str
    available_columns: Sequence[str]
    root_resolution: Mapping[str, Any]
    interval: int | None = None
    asset: str | None = None
    inspected_paths: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _text(self.status, field_name="status"))
        object.__setattr__(self, "available_columns", tuple(sorted(dict.fromkeys(str(column) for column in self.available_columns if str(column).strip()))))
        object.__setattr__(self, "root_resolution", to_jsonable(dict(self.root_resolution)))
        object.__setattr__(self, "interval", None if self.interval is None else int(self.interval))
        object.__setattr__(self, "asset", None if self.asset is None else _text(self.asset, field_name="asset"))
        object.__setattr__(self, "inspected_paths", tuple(str(path) for path in self.inspected_paths))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "asset_state_scalar_feature_schema_inspection",
            "status": self.status,
            "available_columns": list(self.available_columns),
            "available_column_count": int(len(self.available_columns)),
            "root_resolution": to_jsonable(dict(self.root_resolution)),
            "interval": self.interval,
            "asset": self.asset,
            "inspected_paths": list(self.inspected_paths),
            "production_writes_performed": False,
        }


def inspect_scalar_feature_schema(
    source_feature_root: str | Path | None = None,
    *,
    interval: int | None = None,
    asset: str | None = None,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
    sample_partition_limit: int = 3,
) -> ScalarFeatureSchemaInspection:
    resolution = resolve_asset_state_scalar_feature_root(
        source_feature_root=source_feature_root,
        profile=profile,
        env=env,
    )
    if not resolution.found or resolution.root is None:
        return ScalarFeatureSchemaInspection(
            status=resolution.status,
            available_columns=(),
            root_resolution=resolution.as_dict(),
        )

    intervals = (int(interval),) if interval is not None else tuple(int(value) for value in resolution.intervals)
    for selected_interval in intervals:
        selected_asset = asset
        if selected_asset is None:
            assets = discover_scalar_feature_assets(resolution.root, int(selected_interval))
            selected_asset = assets[0] if assets else None
        if selected_asset is None:
            continue
        paths = resolve_scalar_feature_partitions(
            resolution.root,
            interval=int(selected_interval),
            asset=str(selected_asset),
        )
        if not paths:
            continue
        columns: set[str] = set()
        inspected: list[str] = []
        for path in paths[: max(1, int(sample_partition_limit))]:
            frame = pd.read_parquet(path)
            columns.update(str(column) for column in frame.columns if str(column) not in {"ts", "asset"})
            inspected.append(str(path))
        return ScalarFeatureSchemaInspection(
            status="schema_inspected",
            available_columns=tuple(sorted(columns)),
            root_resolution=resolution.as_dict(),
            interval=int(selected_interval),
            asset=str(selected_asset),
            inspected_paths=tuple(inspected),
        )

    return ScalarFeatureSchemaInspection(
        status="schema_partitions_not_found",
        available_columns=(),
        root_resolution=resolution.as_dict(),
        interval=interval,
        asset=asset,
    )


def default_asset_state_feature_column_catalog(
    available_columns: Sequence[str] | None = None,
    *,
    source: str = "scalar_feature_manifest",
) -> AssetStateFeatureColumnCatalog:
    available = set(SCALAR_FEATURE_COLUMNS if available_columns is None else tuple(_source_base(column) for column in available_columns))
    entries: dict[str, AssetStateFeatureColumnCatalogEntry] = {}
    for column in SCALAR_FEATURE_COLUMNS:
        entries[str(column)] = AssetStateFeatureColumnCatalogEntry(
            column=str(column),
            category=_category_for_column(str(column)),
            status=FEATURE_COLUMN_STATUS_AVAILABLE if str(column) in available else FEATURE_COLUMN_STATUS_MISSING,
            source_kind=SCALAR_SOURCE_KIND,
            axis_affinity=_axis_affinity_for_column(str(column)),
            notes="Scalar Feature manifest column.",
        )
    for column in PENDING_SCALAR_FEATURE_COLUMNS:
        entries[column] = AssetStateFeatureColumnCatalogEntry(
            column=column,
            category=_category_for_column(column),
            status=FEATURE_COLUMN_STATUS_PENDING_SCALAR_FEATURE,
            source_kind=SCALAR_SOURCE_KIND,
            axis_affinity=_axis_affinity_for_column(column),
            notes="Plausible Asset-State input candidate pending Scalar Feature support; not consumed as an input feature yet.",
        )
    for column in SOURCED_FROM_OHLCVT_COLUMNS:
        entries[column] = AssetStateFeatureColumnCatalogEntry(
            column=column,
            category=RAW_OHLCVT,
            status=FEATURE_COLUMN_STATUS_SOURCED_FROM_OHLCVT,
            source_kind=OHLCVT_SOURCE_KIND,
            notes="Raw OHLCVT source column used upstream to compute scalar features; not part of Scalar Feature input pools.",
        )
    for column in VALIDATION_TARGET_ONLY_COLUMNS:
        entries[column] = AssetStateFeatureColumnCatalogEntry(
            column=column,
            category=VALIDATION_TARGETS,
            status=FEATURE_COLUMN_STATUS_VALIDATION_TARGET_ONLY,
            source_kind=VALIDATION_SOURCE_KIND,
            notes="Forward validation/economic target only; excluded from Asset-State input features.",
        )
    for column in UNSUPPORTED_FOR_NOW_COLUMNS:
        entries[column] = AssetStateFeatureColumnCatalogEntry(
            column=column,
            category=UNSUPPORTED_EXTERNAL,
            status=FEATURE_COLUMN_STATUS_UNSUPPORTED_FOR_NOW,
            source_kind=UNSUPPORTED_SOURCE_KIND,
            notes="Potentially useful external/microstructure field but unsupported in this sprint.",
        )
    for column in available:
        if column in entries:
            continue
        entries[column] = AssetStateFeatureColumnCatalogEntry(
            column=column,
            category=VALIDATION_TARGETS if _is_leakage_risk_column(column) else MISC_SCALAR,
            status=FEATURE_COLUMN_STATUS_VALIDATION_TARGET_ONLY if _is_leakage_risk_column(column) else FEATURE_COLUMN_STATUS_AVAILABLE,
            source_kind=VALIDATION_SOURCE_KIND if _is_leakage_risk_column(column) else SCALAR_SOURCE_KIND,
            notes="Discovered from inspected schema but not part of the declared Scalar Feature manifest.",
        )
    return AssetStateFeatureColumnCatalog(entries=entries, source=source)


def asset_state_feature_column_catalog_from_root(
    source_feature_root: str | Path | None = None,
    *,
    interval: int | None = None,
    asset: str | None = None,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[AssetStateFeatureColumnCatalog, ScalarFeatureSchemaInspection]:
    inspection = inspect_scalar_feature_schema(
        source_feature_root=source_feature_root,
        interval=interval,
        asset=asset,
        profile=profile,
        env=env,
    )
    catalog = default_asset_state_feature_column_catalog(
        inspection.available_columns,
        source=f"root_inspection:{inspection.status}",
    )
    return catalog, inspection


def _category_for_column(column: str) -> str:
    for category, columns in FEATURE_CATEGORY_COLUMNS.items():
        if column in columns:
            return category
    pending_categories = {
        "trend_persistence_score_20": TREND_DIRECTIONALITY,
        "volatility_compression_score_20": VOLATILITY_EXPANSION_COMPRESSION,
        "participation_imbalance_20": ACTIVITY_PARTICIPATION_LIQUIDITY,
        "mean_reversion_pressure_20": MEAN_REVERSION_CHOP_RANGE,
        "stress_recovery_score_20": DRAWDOWN_STRESS_DOWNSIDE,
        "path_choppiness_ratio_20": RANGE_EFFICIENCY_RUNUP_DRAWDOWN,
    }
    return pending_categories.get(column, MISC_SCALAR)


def _axis_affinity_for_column(column: str) -> tuple[str, ...]:
    category = _category_for_column(column)
    mapping = {
        TREND_DIRECTIONALITY: ("trend",),
        VOLATILITY_EXPANSION_COMPRESSION: ("volatility",),
        ACTIVITY_PARTICIPATION_LIQUIDITY: ("activity",),
        MEAN_REVERSION_CHOP_RANGE: ("mean_reversion",),
        DRAWDOWN_STRESS_DOWNSIDE: ("drawdown",),
        RANGE_EFFICIENCY_RUNUP_DRAWDOWN: ("range_efficiency",),
        PRICE_TRANSFORM: ("trend", "mean_reversion"),
    }
    return mapping.get(category, ())


def _source_base(column: object) -> str:
    text = str(column).strip()
    if text.startswith("i"):
        prefix, sep, suffix = text.partition("_")
        if sep and prefix[1:].isdigit() and suffix:
            return suffix
    return text


def _is_leakage_risk_column(column: object) -> bool:
    text = str(column).lower()
    return any(token in text for token in LEAKAGE_COLUMN_TOKENS)


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Asset-State feature column catalog {field_name} must be non-empty")
    return text


__all__ = [
    "ACTIVITY_PARTICIPATION_LIQUIDITY",
    "DRAWDOWN_STRESS_DOWNSIDE",
    "FEATURE_COLUMN_STATUS_AVAILABLE",
    "FEATURE_COLUMN_STATUS_MISSING",
    "FEATURE_COLUMN_STATUS_PENDING_SCALAR_FEATURE",
    "FEATURE_COLUMN_STATUS_SOURCED_FROM_OHLCVT",
    "FEATURE_COLUMN_STATUS_UNSUPPORTED_FOR_NOW",
    "FEATURE_COLUMN_STATUS_VALIDATION_TARGET_ONLY",
    "FEATURE_COLUMN_STATUSES",
    "MEAN_REVERSION_CHOP_RANGE",
    "PENDING_SCALAR_FEATURE_COLUMNS",
    "RANGE_EFFICIENCY_RUNUP_DRAWDOWN",
    "SOURCED_FROM_OHLCVT_COLUMNS",
    "TREND_DIRECTIONALITY",
    "UNSUPPORTED_FOR_NOW_COLUMNS",
    "VALIDATION_TARGET_ONLY_COLUMNS",
    "VOLATILITY_EXPANSION_COMPRESSION",
    "AssetStateFeatureColumnCatalog",
    "AssetStateFeatureColumnCatalogEntry",
    "ScalarFeatureSchemaInspection",
    "asset_state_feature_column_catalog_from_root",
    "default_asset_state_feature_column_catalog",
    "inspect_scalar_feature_schema",
]
