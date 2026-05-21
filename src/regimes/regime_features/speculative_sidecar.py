from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.source_panel import ASSET_COLUMN, MARKET_STATE_SOURCE_PANEL_STATUS_READY, TIMESTAMP_COLUMN, MarketStateSourcePanelResult
from src.regimes.market_state.universe_views import MarketStateUniverseV1Views
from src.regimes.regime_features.contracts import MARKET_REGIME_FEATURES, REGIME_FEATURES_SCHEMA_VERSION
from src.regimes.regime_features.lineage import RegimeFeatureKnownAtSpec, RegimeFeatureLineageSpec


SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID = "speculative_satellite_sidecar"
SPECULATIVE_SIDECAR_FEATURE_SET_ID = "market_state_v1_speculative_satellite_sidecar"
SPECULATIVE_SIDECAR_FEATURE_STATUS_READY = "ready"
SPECULATIVE_SIDECAR_FEATURE_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class SpeculativeSidecarFeatureConfig:
    feature_set_id: str = SPECULATIVE_SIDECAR_FEATURE_SET_ID
    universe_policy_id: str = "market_state_v1_dual_broad"
    run_id: str = "market_state_v1_speculative_sidecar_bounded"
    min_speculative_assets: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_set_id", _text(self.feature_set_id, "feature_set_id"))
        object.__setattr__(self, "universe_policy_id", _text(self.universe_policy_id, "universe_policy_id"))
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "min_speculative_assets", max(1, int(self.min_speculative_assets)))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "universe_policy_id": self.universe_policy_id,
            "run_id": self.run_id,
            "min_speculative_assets": int(self.min_speculative_assets),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass
class SpeculativeSidecarFeatureResult:
    status: str
    features: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    feature_family_id: str = SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID
    feature_set_id: str = SPECULATIVE_SIDECAR_FEATURE_SET_ID
    interval: int | None = None
    band: str | None = None
    assets_used: Sequence[str] = ()
    coverage: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)
    known_at_metadata: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = str(self.status).strip().lower()
        self.assets_used = tuple(str(asset) for asset in self.assets_used)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "market_state_speculative_sidecar_feature_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "feature_family_id": self.feature_family_id,
            "feature_set_id": self.feature_set_id,
            "interval": self.interval,
            "band": self.band,
            "row_count": self.row_count,
            "assets_used": list(self.assets_used),
            "coverage": to_jsonable(dict(self.coverage)),
            "lineage": to_jsonable(dict(self.lineage)),
            "known_at_metadata": to_jsonable(dict(self.known_at_metadata)),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_writes_enabled": False,
        }


def build_speculative_sidecar_features(
    source_panel: MarketStateSourcePanelResult,
    universe_views: MarketStateUniverseV1Views,
    config: SpeculativeSidecarFeatureConfig | None = None,
) -> SpeculativeSidecarFeatureResult:
    cfg = config or SpeculativeSidecarFeatureConfig()
    if source_panel.status != MARKET_STATE_SOURCE_PANEL_STATUS_READY or source_panel.long_panel.empty:
        return _blocked(source_panel, universe_views, cfg, ("speculative_satellite_unavailable",))
    timestamps = tuple(int(ts) for ts in source_panel.timestamps)
    speculative_assets = _available(universe_views.speculative_satellite.source_assets, source_panel.assets_loaded)
    broad_assets = _available(universe_views.broad_clean_risk.source_assets, source_panel.assets_loaded)
    if len(speculative_assets) < cfg.min_speculative_assets:
        return _blocked(source_panel, universe_views, cfg, ("speculative_satellite_unavailable",))

    spec_returns = _pivot(source_panel.long_panel, assets=speculative_assets, timestamps=timestamps, value_column="log_return")
    spec_volume = _pivot(source_panel.long_panel, assets=speculative_assets, timestamps=timestamps, value_column="dollar_volume")
    broad_returns = _pivot(source_panel.long_panel, assets=broad_assets, timestamps=timestamps, value_column="log_return")
    broad_volume = _pivot(source_panel.long_panel, assets=broad_assets, timestamps=timestamps, value_column="dollar_volume")
    spec_valid = spec_returns.notna()
    spec_volume_total = spec_volume.sum(axis=1, min_count=1)
    broad_volume_total = broad_volume.sum(axis=1, min_count=1)
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: list(timestamps),
            "speculative_advance_fraction": (spec_returns > 0.0).sum(axis=1).divide(spec_valid.sum(axis=1).replace(0, np.nan)).reindex(timestamps).to_numpy(),
            "speculative_activity_breadth": (spec_volume > 0.0).sum(axis=1).divide(spec_volume.notna().sum(axis=1).replace(0, np.nan)).reindex(timestamps).to_numpy(),
            "speculative_return_dispersion": spec_returns.std(axis=1, ddof=0, skipna=True).reindex(timestamps).to_numpy(),
            "speculative_vs_clean_broad_return_spread": (spec_returns.mean(axis=1, skipna=True) - broad_returns.mean(axis=1, skipna=True)).reindex(timestamps).to_numpy(),
            "speculative_volume_share": spec_volume_total.divide(spec_volume_total.add(broad_volume_total, fill_value=np.nan).replace(0.0, np.nan)).reindex(timestamps).to_numpy(),
            "speculative_sample_count": spec_valid.sum(axis=1).reindex(timestamps).astype(int).to_numpy(),
        }
    )
    frame["speculative_coverage"] = frame["speculative_sample_count"] / float(len(speculative_assets))
    frame["speculative_satellite_requested_n"] = int(len(universe_views.speculative_satellite.source_assets))
    frame["speculative_satellite_active_n"] = frame["speculative_sample_count"].astype(int)
    frame["speculative_satellite_coverage_ratio"] = frame["speculative_satellite_active_n"] / float(max(1, frame["speculative_satellite_requested_n"].iloc[0]))
    frame["speculative_satellite_min_active_n"] = int(cfg.min_speculative_assets)
    frame["speculative_satellite_min_coverage_ratio"] = float(cfg.min_speculative_assets) / float(max(1, frame["speculative_satellite_requested_n"].iloc[0]))
    frame["speculative_satellite_coverage_pass"] = (
        (frame["speculative_satellite_active_n"] >= int(cfg.min_speculative_assets))
        & (frame["speculative_satellite_coverage_ratio"] >= frame["speculative_satellite_min_coverage_ratio"])
    )
    frame["speculative_satellite_late_entry_count"] = _late_entry_count_by_ts(
        source_panel.long_panel,
        requested_assets=universe_views.speculative_satellite.source_assets,
        timestamps=timestamps,
    )
    frame["speculative_satellite_warmup_excluded_count"] = 0
    lineage = _feature_lineage(source_panel, universe_views, cfg)
    frame = _attach_row_metadata(frame, source_panel=source_panel, universe_views=universe_views, config=cfg, lineage=lineage)
    return SpeculativeSidecarFeatureResult(
        status=SPECULATIVE_SIDECAR_FEATURE_STATUS_READY,
        features=frame,
        feature_set_id=cfg.feature_set_id,
        interval=source_panel.interval,
        band=source_panel.band,
        assets_used=speculative_assets,
        coverage={"speculative_asset_count": len(speculative_assets), "timestamp_count": len(timestamps), "sidecar_only_not_core": True},
        lineage=lineage.as_dict(),
        known_at_metadata={"row_known_at_policy": "closed_source_tail_per_timestamp"},
        metadata={"config": cfg.as_dict(), "speculative_sidecar_only": True, "not_core_covariance": True, "l2_order_book_fields_used": False, "production_writes_enabled": False},
    )


def _pivot(long_panel: pd.DataFrame, *, assets: Sequence[str], timestamps: Sequence[int], value_column: str) -> pd.DataFrame:
    index = pd.Index(tuple(int(ts) for ts in timestamps), name=TIMESTAMP_COLUMN)
    if not assets or value_column not in long_panel.columns:
        return pd.DataFrame(index=index, columns=tuple(assets), dtype=float)
    frame = long_panel.loc[long_panel[ASSET_COLUMN].astype(str).isin(tuple(assets)), [ASSET_COLUMN, TIMESTAMP_COLUMN, value_column]].copy()
    frame[TIMESTAMP_COLUMN] = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    wide = frame.pivot_table(index=TIMESTAMP_COLUMN, columns=ASSET_COLUMN, values=value_column, aggfunc="last")
    return wide.reindex(index=index, columns=tuple(assets))


def _feature_lineage(source_panel: MarketStateSourcePanelResult, universe_views: MarketStateUniverseV1Views, config: SpeculativeSidecarFeatureConfig) -> RegimeFeatureLineageSpec:
    timestamps = tuple(int(ts) for ts in source_panel.timestamps)
    source_tail = max(timestamps) if timestamps else 0
    return RegimeFeatureLineageSpec(artifact_family=MARKET_REGIME_FEATURES, feature_set_id=config.feature_set_id, interval=int(source_panel.interval), band=str(source_panel.band), source_data_kinds=("ohlcvt",), source_partition_lineage=_source_lineage(source_panel), source_tail_ts=source_tail, feature_window_start=min(timestamps) if timestamps else source_tail, feature_window_end=source_tail, generated_at=source_tail, run_id=config.run_id, universe_snapshot_id=universe_views.manifest_id, calculation_policy={"feature_family_id": SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID, "sidecar_only_not_core": True, "production_writes_enabled": False, "config": config.as_dict()})


def _late_entry_count_by_ts(
    long_panel: pd.DataFrame,
    *,
    requested_assets: Sequence[str],
    timestamps: Sequence[int],
) -> list[int]:
    if long_panel.empty or ASSET_COLUMN not in long_panel.columns or TIMESTAMP_COLUMN not in long_panel.columns:
        return [len(tuple(requested_assets)) for _ in timestamps]
    source = long_panel.loc[long_panel[ASSET_COLUMN].astype(str).isin(tuple(requested_assets))].copy()
    if "source_row_present" in source.columns:
        source = source.loc[source["source_row_present"].astype(bool)]
    first_by_asset = source.groupby(ASSET_COLUMN)[TIMESTAMP_COLUMN].min().to_dict()
    out: list[int] = []
    for ts in timestamps:
        count = 0
        for asset in requested_assets:
            first_ts = first_by_asset.get(str(asset))
            if first_ts is None or int(first_ts) > int(ts):
                count += 1
        out.append(count)
    return out


def _attach_row_metadata(frame: pd.DataFrame, *, source_panel: MarketStateSourcePanelResult, universe_views: MarketStateUniverseV1Views, config: SpeculativeSidecarFeatureConfig, lineage: RegimeFeatureLineageSpec) -> pd.DataFrame:
    out = frame.copy()
    out["interval"] = int(source_panel.interval)
    out["band"] = str(source_panel.band)
    out["feature_family_id"] = SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID
    out["feature_set_id"] = config.feature_set_id
    out["universe_policy_id"] = config.universe_policy_id
    out["universe_snapshot_id"] = universe_views.manifest_id
    out["known_at_ts"] = out[TIMESTAMP_COLUMN].astype("int64")
    out["source_tail_ts"] = out[TIMESTAMP_COLUMN].astype("int64")
    out["lineage_id"] = lineage.lineage_id
    out["schema_version"] = int(REGIME_FEATURES_SCHEMA_VERSION)
    out["known_at"] = [RegimeFeatureKnownAtSpec(ts=int(ts), known_at_ts=int(ts), source_tail_ts=int(ts), feature_available_at_ts=int(ts), alignment_policy="closed_ohlcvt_bar_market_state_v1", latency_policy="same_batch_after_source_tail", no_lookahead_verified=True).as_dict() for ts in out[TIMESTAMP_COLUMN].astype("int64")]
    return out


def _blocked(source_panel: MarketStateSourcePanelResult, universe_views: MarketStateUniverseV1Views, config: SpeculativeSidecarFeatureConfig, reason_codes: Sequence[str]) -> SpeculativeSidecarFeatureResult:
    return SpeculativeSidecarFeatureResult(status=SPECULATIVE_SIDECAR_FEATURE_STATUS_BLOCKED, feature_set_id=config.feature_set_id, interval=source_panel.interval, band=source_panel.band, reason_codes=tuple(reason_codes), metadata={"config": config.as_dict(), "universe_manifest_id": universe_views.manifest_id, "production_writes_enabled": False})


def _available(candidates: Sequence[str], available_assets: Sequence[str]) -> tuple[str, ...]:
    available = {str(asset) for asset in available_assets}
    return tuple(str(asset) for asset in candidates if str(asset) in available)


def _source_lineage(source_panel: MarketStateSourcePanelResult) -> tuple[dict[str, Any], ...]:
    lineage = tuple(dict(item) for item in source_panel.source_lineage)
    return lineage or ({"source_kind": "market_state_source_panel", "path": "in_memory"},)


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Speculative sidecar feature {field_name} must be non-empty")
    return text


__all__ = [
    "SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID",
    "SPECULATIVE_SIDECAR_FEATURE_SET_ID",
    "SPECULATIVE_SIDECAR_FEATURE_STATUS_BLOCKED",
    "SPECULATIVE_SIDECAR_FEATURE_STATUS_READY",
    "SpeculativeSidecarFeatureConfig",
    "SpeculativeSidecarFeatureResult",
    "build_speculative_sidecar_features",
]
