from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.asset_state.dataset_builder import resolve_scalar_feature_partitions
from src.regimes.core.known_at import KnownAtSpec
from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.universe_views import MarketStateUniverseV1Views, MarketStateUniverseView
from src.regimes.regime_features.sources import (
    OHLCVT_SOURCE_KIND,
    SCALAR_FEATURE_SOURCE_KIND,
    load_ohlcvt_frames_for_assets,
    source_partition_lineage,
)


MARKET_STATE_SOURCE_PANEL_STATUS_READY = "ready"
MARKET_STATE_SOURCE_PANEL_STATUS_MISSING_DATA = "missing_data"
MARKET_STATE_SOURCE_PANEL_STATUS_MALFORMED_DATA = "malformed_data"
MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_ASSETS = "too_few_assets"
MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_TIMESTAMPS = "too_few_timestamps"
MARKET_STATE_SOURCE_PANEL_STATUSES: tuple[str, ...] = (
    MARKET_STATE_SOURCE_PANEL_STATUS_READY,
    MARKET_STATE_SOURCE_PANEL_STATUS_MISSING_DATA,
    MARKET_STATE_SOURCE_PANEL_STATUS_MALFORMED_DATA,
    MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_ASSETS,
    MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_TIMESTAMPS,
)

TIMESTAMP_COLUMN = "ts"
ASSET_COLUMN = "asset"
OHLCVT_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "trades")
DERIVED_COLUMNS: tuple[str, ...] = (
    "effective_close",
    "log_return",
    "dollar_volume",
    "high_low_range",
    "range_return",
    "drawdown_input",
    "drawdown_pct",
    "trade_count",
    "source_row_present",
    "close_forward_filled",
    "stale_close",
    "no_trade_bar",
)
RESERVED_SCALAR_COLUMNS: frozenset[str] = frozenset(
    {
        ASSET_COLUMN,
        TIMESTAMP_COLUMN,
        *OHLCVT_COLUMNS,
        *DERIVED_COLUMNS,
        "return",
        "returns",
        "ret",
        "log_return",
        "dollar_volume",
        "quote_volume",
    }
)


@dataclass(frozen=True)
class MarketStateStalenessPolicy:
    allow_close_forward_fill: bool = False
    max_stale_steps: int = 0

    def __post_init__(self) -> None:
        max_steps = int(self.max_stale_steps)
        if max_steps < 0:
            raise ValueError("Market-State source panel max_stale_steps cannot be negative")
        object.__setattr__(self, "allow_close_forward_fill", bool(self.allow_close_forward_fill))
        object.__setattr__(self, "max_stale_steps", max_steps if bool(self.allow_close_forward_fill) else 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow_close_forward_fill": bool(self.allow_close_forward_fill),
            "max_stale_steps": int(self.max_stale_steps),
        }


@dataclass(frozen=True)
class MarketStateSourcePanelRequest:
    ohlcvt_root: str | Path
    interval: int
    band: str
    assets: Sequence[str] = ()
    universe_views: MarketStateUniverseV1Views | None = None
    selected_views: Sequence[str] = ("effective_core", "broad_clean_risk", "stable_peg_panel")
    include_broad_with_satellites: bool = False
    include_speculative_satellite: bool = False
    scalar_feature_root: str | Path | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    min_assets: int = 2
    min_timestamps: int = 8
    staleness_policy: MarketStateStalenessPolicy = field(default_factory=MarketStateStalenessPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Market-State source panel interval must be positive")
        if self.start_ts is not None and self.end_ts is not None and int(self.start_ts) >= int(self.end_ts):
            raise ValueError("Market-State source panel start_ts must be before end_ts")
        selected_views = tuple(str(view).strip() for view in self.selected_views if str(view).strip())
        if not selected_views and not self.assets:
            raise ValueError("Market-State source panel requires explicit assets or selected universe views")
        staleness = (
            self.staleness_policy
            if isinstance(self.staleness_policy, MarketStateStalenessPolicy)
            else MarketStateStalenessPolicy(**dict(self.staleness_policy))
        )
        object.__setattr__(self, "ohlcvt_root", Path(self.ohlcvt_root))
        object.__setattr__(self, "scalar_feature_root", None if self.scalar_feature_root is None else Path(self.scalar_feature_root))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", str(self.band).strip().lower())
        object.__setattr__(self, "assets", _asset_tuple(self.assets))
        object.__setattr__(self, "selected_views", selected_views)
        object.__setattr__(self, "min_assets", max(1, int(self.min_assets)))
        object.__setattr__(self, "min_timestamps", max(1, int(self.min_timestamps)))
        object.__setattr__(self, "staleness_policy", staleness)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def requested_assets(self) -> tuple[str, ...]:
        assets = list(self.assets)
        if self.universe_views is not None:
            selected = list(self.selected_views)
            if self.include_broad_with_satellites and "broad_with_satellites" not in selected:
                selected.append("broad_with_satellites")
            if self.include_speculative_satellite and "speculative_satellite" not in selected:
                selected.append("speculative_satellite")
            for view_name in selected:
                view = self.universe_views.view(view_name)
                assets.extend(view.source_assets)
        return tuple(dict.fromkeys(str(asset).strip() for asset in assets if str(asset).strip()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ohlcvt_root": str(self.ohlcvt_root),
            "scalar_feature_root": str(self.scalar_feature_root) if self.scalar_feature_root is not None else None,
            "interval": int(self.interval),
            "band": self.band,
            "assets": list(self.assets),
            "selected_views": list(self.selected_views),
            "include_broad_with_satellites": bool(self.include_broad_with_satellites),
            "include_speculative_satellite": bool(self.include_speculative_satellite),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "min_assets": int(self.min_assets),
            "min_timestamps": int(self.min_timestamps),
            "staleness_policy": self.staleness_policy.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class MarketStateSourcePanelResult:
    status: str
    interval: int
    band: str
    assets_requested: Sequence[str]
    assets_loaded: Sequence[str] = ()
    timestamps: Sequence[int] = ()
    panel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    long_panel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    coverage_by_asset: Mapping[str, Any] = field(default_factory=dict)
    stale_no_trade_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    source_lineage: Sequence[Mapping[str, Any]] = ()
    known_at_metadata: Mapping[str, Any] = field(default_factory=dict)
    scalar_enrichment_columns: Sequence[str] = ()
    reason_codes: Sequence[str] = ()
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in MARKET_STATE_SOURCE_PANEL_STATUSES:
            raise ValueError(f"Unsupported Market-State source panel status {status!r}")
        self.status = status
        self.interval = int(self.interval)
        self.band = str(self.band).strip().lower()
        self.assets_requested = tuple(str(asset) for asset in self.assets_requested)
        self.assets_loaded = tuple(str(asset) for asset in self.assets_loaded)
        self.timestamps = tuple(int(ts) for ts in self.timestamps)
        self.scalar_enrichment_columns = tuple(str(column) for column in self.scalar_enrichment_columns)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))

    @property
    def usable(self) -> bool:
        return self.status == MARKET_STATE_SOURCE_PANEL_STATUS_READY

    @property
    def asset_count(self) -> int:
        return len(self.assets_loaded)

    @property
    def timestamp_count(self) -> int:
        return len(self.timestamps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "market_state_source_panel_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "interval": int(self.interval),
            "band": self.band,
            "assets_requested": list(self.assets_requested),
            "assets_loaded": list(self.assets_loaded),
            "asset_count": self.asset_count,
            "timestamps": list(self.timestamps),
            "timestamp_count": self.timestamp_count,
            "panel_shape": [int(self.panel.shape[0]), int(self.panel.shape[1])],
            "long_panel_shape": [int(self.long_panel.shape[0]), int(self.long_panel.shape[1])],
            "coverage_by_asset": to_jsonable(dict(self.coverage_by_asset)),
            "stale_no_trade_diagnostics": to_jsonable(dict(self.stale_no_trade_diagnostics)),
            "source_lineage": to_jsonable([dict(item) for item in self.source_lineage]),
            "known_at_metadata": to_jsonable(dict(self.known_at_metadata)),
            "scalar_enrichment_columns": list(self.scalar_enrichment_columns),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_writes_enabled": False,
        }


def build_market_state_source_panel(request: MarketStateSourcePanelRequest) -> MarketStateSourcePanelResult:
    assets = request.requested_assets()
    if not assets:
        return _blocked(request, MARKET_STATE_SOURCE_PANEL_STATUS_MISSING_DATA, ("missing_assets",), "no source panel assets were requested")
    if not request.ohlcvt_root.exists() or not request.ohlcvt_root.is_dir():
        return _blocked(
            request,
            MARKET_STATE_SOURCE_PANEL_STATUS_MISSING_DATA,
            ("missing_ohlcvt_root",),
            f"OHLCVT root does not exist: {request.ohlcvt_root}",
            assets_requested=assets,
        )
    try:
        raw_frames = load_ohlcvt_frames_for_assets(
            request.ohlcvt_root,
            interval=request.interval,
            assets=assets,
            start_ts=request.start_ts,
            end_ts=request.end_ts,
            columns=("asset", "ts", "open", "high", "low", "close", "volume", "trades"),
        )
        _validate_raw_ohlcvt(raw_frames)
    except Exception as exc:
        return _blocked(
            request,
            MARKET_STATE_SOURCE_PANEL_STATUS_MALFORMED_DATA,
            ("malformed_ohlcvt",),
            str(exc),
            assets_requested=assets,
        )

    non_empty = {asset: frame for asset, frame in raw_frames.items() if frame is not None and not frame.empty}
    if len(non_empty) < int(request.min_assets):
        return _blocked(
            request,
            MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_ASSETS,
            ("too_few_assets",),
            f"loaded OHLCVT asset count {len(non_empty)} < min_assets {request.min_assets}",
            assets_requested=assets,
            coverage_by_asset={asset: _empty_asset_coverage(asset) for asset in assets},
        )

    timestamps = _fixed_timestamp_grid(non_empty.values(), interval=request.interval, start_ts=request.start_ts, end_ts=request.end_ts)
    if len(timestamps) < int(request.min_timestamps):
        return _blocked(
            request,
            MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_TIMESTAMPS,
            ("too_few_timestamps",),
            f"timestamp count {len(timestamps)} < min_timestamps {request.min_timestamps}",
            assets_requested=assets,
            assets_loaded=tuple(non_empty),
        )

    scalar_frames = _load_scalar_enrichment_frames(
        request.scalar_feature_root,
        interval=request.interval,
        assets=assets,
        start_ts=request.start_ts,
        end_ts=request.end_ts,
    )
    long_parts: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    scalar_columns: list[str] = []
    for asset in assets:
        frame = raw_frames.get(asset, pd.DataFrame())
        regularized = _regularize_asset_frame(
            asset,
            frame,
            timestamps=timestamps,
            staleness_policy=request.staleness_policy,
        )
        scalar = scalar_frames.get(asset)
        if scalar is not None and not scalar.empty:
            regularized = regularized.merge(scalar, on=[ASSET_COLUMN, TIMESTAMP_COLUMN], how="left")
            scalar_columns.extend(column for column in scalar.columns if column.startswith("scalar__"))
        coverage[asset] = _asset_coverage(regularized)
        long_parts.append(regularized)

    long_panel = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    loaded_assets = tuple(asset for asset, item in coverage.items() if int(item["source_row_count"]) > 0)
    if len(loaded_assets) < int(request.min_assets):
        return _blocked(
            request,
            MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_ASSETS,
            ("too_few_assets_after_regularization",),
            f"regularized loaded asset count {len(loaded_assets)} < min_assets {request.min_assets}",
            assets_requested=assets,
            assets_loaded=loaded_assets,
            coverage_by_asset=coverage,
        )

    panel = _wide_panel(long_panel, timestamps=timestamps, assets=loaded_assets)
    lineage = _source_lineage(request, assets=loaded_assets)
    known_at = _known_at_metadata(timestamps=timestamps, lineage=lineage)
    return MarketStateSourcePanelResult(
        status=MARKET_STATE_SOURCE_PANEL_STATUS_READY,
        interval=request.interval,
        band=request.band,
        assets_requested=assets,
        assets_loaded=loaded_assets,
        timestamps=timestamps,
        panel=panel,
        long_panel=long_panel.loc[long_panel[ASSET_COLUMN].isin(loaded_assets)].reset_index(drop=True),
        coverage_by_asset=coverage,
        stale_no_trade_diagnostics=_stale_no_trade_diagnostics(long_panel),
        source_lineage=lineage,
        known_at_metadata=known_at,
        scalar_enrichment_columns=tuple(sorted(set(scalar_columns))),
        metadata={
            "request": request.as_dict(),
            "ohlcvt_authoritative_fields": list(OHLCVT_COLUMNS),
            "scalar_features_used_for_enrichment_only": bool(scalar_columns),
            "fixed_timestamp_grid": True,
            "production_writes_enabled": False,
        },
    )


def _regularize_asset_frame(
    asset: str,
    frame: pd.DataFrame,
    *,
    timestamps: Sequence[int],
    staleness_policy: MarketStateStalenessPolicy,
) -> pd.DataFrame:
    grid = pd.DataFrame({TIMESTAMP_COLUMN: tuple(int(ts) for ts in timestamps)})
    if frame is None or frame.empty:
        out = grid.copy()
        out[ASSET_COLUMN] = str(asset)
        for column in OHLCVT_COLUMNS:
            out[column] = np.nan
        source_present = pd.Series(False, index=out.index)
    else:
        raw = frame.copy()
        raw[TIMESTAMP_COLUMN] = pd.to_numeric(raw[TIMESTAMP_COLUMN], errors="coerce")
        raw = raw.dropna(subset=[TIMESTAMP_COLUMN]).copy()
        raw[TIMESTAMP_COLUMN] = raw[TIMESTAMP_COLUMN].astype("int64")
        raw = raw.sort_values(TIMESTAMP_COLUMN).drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
        for column in OHLCVT_COLUMNS:
            if column not in raw.columns:
                raw[column] = np.nan
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        out = grid.merge(raw[[TIMESTAMP_COLUMN, *OHLCVT_COLUMNS]], on=TIMESTAMP_COLUMN, how="left")
        out[ASSET_COLUMN] = str(asset)
        source_present = out[list(OHLCVT_COLUMNS)].notna().any(axis=1)

    for column in OHLCVT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    original_close_present = out["close"].notna()
    if staleness_policy.allow_close_forward_fill and int(staleness_policy.max_stale_steps) > 0:
        filled_close = out["close"].ffill(limit=int(staleness_policy.max_stale_steps))
    else:
        filled_close = out["close"]
    out["effective_close"] = filled_close
    out["close_forward_filled"] = (~original_close_present) & out["effective_close"].notna()
    out["stale_close"] = out["close_forward_filled"]
    out["source_row_present"] = source_present.astype(bool)
    out["no_trade_bar"] = source_present & (
        out["volume"].fillna(np.nan).eq(0.0) | out["trades"].fillna(np.nan).eq(0.0)
    )
    close = pd.to_numeric(out["effective_close"], errors="coerce")
    out["log_return"] = np.log(close / close.shift(1))
    out["dollar_volume"] = close * pd.to_numeric(out["volume"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    out["high_low_range"] = high - low
    out["range_return"] = np.where((high > 0.0) & (low > 0.0), high / low - 1.0, np.nan)
    out["drawdown_input"] = close
    running_max = close.cummax()
    out["drawdown_pct"] = close / running_max - 1.0
    out["trade_count"] = pd.to_numeric(out["trades"], errors="coerce")
    return out[[ASSET_COLUMN, TIMESTAMP_COLUMN, *OHLCVT_COLUMNS, *DERIVED_COLUMNS]]


def _load_scalar_enrichment_frames(
    scalar_root: Path | None,
    *,
    interval: int,
    assets: Sequence[str],
    start_ts: int | None,
    end_ts: int | None,
) -> dict[str, pd.DataFrame]:
    if scalar_root is None or not scalar_root.exists() or not scalar_root.is_dir():
        return {}
    frames: dict[str, pd.DataFrame] = {}
    for asset in assets:
        paths = resolve_scalar_feature_partitions(scalar_root, interval=int(interval), asset=str(asset), start_ts=start_ts, end_ts=end_ts)
        parts: list[pd.DataFrame] = []
        for path in paths:
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            if TIMESTAMP_COLUMN not in frame.columns:
                continue
            if ASSET_COLUMN not in frame.columns:
                frame = frame.copy()
                frame[ASSET_COLUMN] = str(asset)
            frame = frame.loc[frame[ASSET_COLUMN].astype(str) == str(asset)].copy()
            frame[TIMESTAMP_COLUMN] = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce")
            frame = frame.dropna(subset=[TIMESTAMP_COLUMN])
            if start_ts is not None:
                frame = frame.loc[frame[TIMESTAMP_COLUMN] >= int(start_ts)]
            if end_ts is not None:
                frame = frame.loc[frame[TIMESTAMP_COLUMN] < int(end_ts)]
            enrich = _scalar_enrichment_columns(frame)
            if enrich:
                out = frame[[ASSET_COLUMN, TIMESTAMP_COLUMN, *enrich]].copy()
                out[TIMESTAMP_COLUMN] = out[TIMESTAMP_COLUMN].astype("int64")
                out = out.rename(columns={column: f"scalar__{column}" for column in enrich})
                parts.append(out)
        if parts:
            frames[str(asset)] = (
                pd.concat(parts, ignore_index=True)
                .sort_values(TIMESTAMP_COLUMN)
                .drop_duplicates(subset=[ASSET_COLUMN, TIMESTAMP_COLUMN], keep="last")
                .reset_index(drop=True)
            )
    return frames


def _validate_raw_ohlcvt(frames: Mapping[str, pd.DataFrame]) -> None:
    for asset, frame in frames.items():
        if frame is None or frame.empty:
            continue
        missing = [column for column in (TIMESTAMP_COLUMN, "close") if column not in frame.columns]
        if missing:
            raise ValueError(f"OHLCVT frame for {asset} missing required columns: {missing}")
        close = pd.to_numeric(frame["close"], errors="coerce")
        if bool((close.dropna() <= 0.0).any()):
            raise ValueError(f"OHLCVT close must be positive for asset {asset}")
        for column in ("volume", "trades"):
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                if bool((values.dropna() < 0.0).any()):
                    raise ValueError(f"OHLCVT {column} cannot be negative for asset {asset}")
        if {"high", "low"}.issubset(frame.columns):
            high = pd.to_numeric(frame["high"], errors="coerce")
            low = pd.to_numeric(frame["low"], errors="coerce")
            bad = (high.notna() & low.notna()) & (high < low)
            if bool(bad.any()):
                raise ValueError(f"OHLCVT high cannot be below low for asset {asset}")


def _fixed_timestamp_grid(
    frames: Sequence[pd.DataFrame],
    *,
    interval: int,
    start_ts: int | None,
    end_ts: int | None,
) -> tuple[int, ...]:
    step = int(interval) * 60
    if start_ts is not None and end_ts is not None:
        return tuple(int(ts) for ts in np.arange(int(start_ts), int(end_ts), step, dtype=np.int64))
    values: list[int] = []
    for frame in frames:
        if frame is not None and not frame.empty and TIMESTAMP_COLUMN in frame.columns:
            ts = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce").dropna()
            values.extend(int(value) for value in ts.astype("int64"))
    if not values:
        return ()
    start = int(start_ts) if start_ts is not None else min(values)
    end = int(end_ts) if end_ts is not None else max(values) + step
    return tuple(int(ts) for ts in np.arange(start, end, step, dtype=np.int64))


def _asset_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    expected = int(frame.shape[0])
    source_rows = int(frame["source_row_present"].sum()) if "source_row_present" in frame.columns else 0
    close_present = int(frame["effective_close"].notna().sum()) if "effective_close" in frame.columns else 0
    return {
        "asset": str(frame[ASSET_COLUMN].iloc[0]) if not frame.empty else None,
        "expected_timestamp_count": expected,
        "source_row_count": source_rows,
        "source_coverage_pct": float(source_rows / expected) if expected else 0.0,
        "effective_close_count": close_present,
        "effective_close_coverage_pct": float(close_present / expected) if expected else 0.0,
        "close_forward_filled_count": int(frame["close_forward_filled"].sum()) if "close_forward_filled" in frame.columns else 0,
        "stale_close_count": int(frame["stale_close"].sum()) if "stale_close" in frame.columns else 0,
        "no_trade_bar_count": int(frame["no_trade_bar"].sum()) if "no_trade_bar" in frame.columns else 0,
        "first_ts": int(frame[TIMESTAMP_COLUMN].min()) if expected else None,
        "last_ts": int(frame[TIMESTAMP_COLUMN].max()) if expected else None,
    }


def _empty_asset_coverage(asset: str) -> dict[str, Any]:
    return {
        "asset": str(asset),
        "expected_timestamp_count": 0,
        "source_row_count": 0,
        "source_coverage_pct": 0.0,
        "effective_close_count": 0,
        "effective_close_coverage_pct": 0.0,
        "close_forward_filled_count": 0,
        "stale_close_count": 0,
        "no_trade_bar_count": 0,
        "first_ts": None,
        "last_ts": None,
    }


def _stale_no_trade_diagnostics(long_panel: pd.DataFrame) -> dict[str, Any]:
    if long_panel.empty:
        return {"stale_close_count": 0, "no_trade_bar_count": 0, "assets_with_no_trade_bars": []}
    grouped = long_panel.groupby(ASSET_COLUMN, dropna=False)
    return {
        "stale_close_count": int(long_panel["stale_close"].sum()),
        "no_trade_bar_count": int(long_panel["no_trade_bar"].sum()),
        "assets_with_no_trade_bars": [
            str(asset)
            for asset, frame in grouped
            if bool(frame["no_trade_bar"].any())
        ],
        "assets_with_stale_close": [
            str(asset)
            for asset, frame in grouped
            if bool(frame["stale_close"].any())
        ],
    }


def _wide_panel(long_panel: pd.DataFrame, *, timestamps: Sequence[int], assets: Sequence[str]) -> pd.DataFrame:
    timestamp_index = pd.Index(tuple(int(ts) for ts in timestamps), name=TIMESTAMP_COLUMN)
    wide_columns: dict[str, Any] = {TIMESTAMP_COLUMN: timestamp_index.to_numpy()}
    for asset in assets:
        asset_frame = long_panel.loc[long_panel[ASSET_COLUMN] == str(asset)].set_index(TIMESTAMP_COLUMN)
        reindexed = asset_frame.reindex(timestamp_index)
        for column in (*OHLCVT_COLUMNS, *DERIVED_COLUMNS):
            wide_columns[f"{column}__{asset}"] = reindexed[column].to_numpy()
    return pd.DataFrame(wide_columns)


def _source_lineage(request: MarketStateSourcePanelRequest, *, assets: Sequence[str]) -> tuple[dict[str, Any], ...]:
    lineage: list[dict[str, Any]] = []
    lineage.extend(
        source_partition_lineage(
            request.ohlcvt_root,
            source_kind=OHLCVT_SOURCE_KIND,
            interval=request.interval,
            assets=assets,
            start_ts=request.start_ts,
            end_ts=request.end_ts,
        )
    )
    if request.scalar_feature_root is not None and request.scalar_feature_root.exists():
        lineage.extend(
            source_partition_lineage(
                request.scalar_feature_root,
                source_kind=SCALAR_FEATURE_SOURCE_KIND,
                interval=request.interval,
                assets=assets,
                start_ts=request.start_ts,
                end_ts=request.end_ts,
            )
        )
    return tuple(lineage)


def _known_at_metadata(*, timestamps: Sequence[int], lineage: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_ts = int(max(timestamps)) if timestamps else 0
    max_lineage_ts = output_ts
    for item in lineage:
        value = item.get("max_ts")
        if value is not None:
            try:
                max_lineage_ts = max(max_lineage_ts, int(value))
            except Exception:
                pass
    known_at_ts = max(output_ts, max_lineage_ts)
    return KnownAtSpec(
        ts=output_ts,
        known_at_ts=known_at_ts,
        source_tail_ts=max_lineage_ts,
        label_available_at_ts=known_at_ts,
        alignment_policy="market_state_source_panel_fixed_grid_source_tail",
        latency_policy="source_panel_metadata_only_no_labels",
        no_lookahead_verified=True,
    ).as_dict()


def _blocked(
    request: MarketStateSourcePanelRequest,
    status: str,
    reason_codes: Sequence[str],
    message: str,
    *,
    assets_requested: Sequence[str] | None = None,
    assets_loaded: Sequence[str] = (),
    coverage_by_asset: Mapping[str, Any] | None = None,
) -> MarketStateSourcePanelResult:
    return MarketStateSourcePanelResult(
        status=status,
        interval=request.interval,
        band=request.band,
        assets_requested=tuple(assets_requested or request.requested_assets()),
        assets_loaded=tuple(assets_loaded),
        coverage_by_asset=coverage_by_asset or {},
        reason_codes=tuple(reason_codes),
        message=message,
        metadata={"request": request.as_dict(), "production_writes_enabled": False},
    )


def _scalar_enrichment_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    out: list[str] = []
    for column in frame.columns:
        text = str(column)
        if text in RESERVED_SCALAR_COLUMNS or text.startswith("scalar__"):
            continue
        if text.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            out.append(text)
    return tuple(dict.fromkeys(out))


def _asset_tuple(values: Sequence[object]) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = (values,)
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


__all__ = [
    "MARKET_STATE_SOURCE_PANEL_STATUS_MALFORMED_DATA",
    "MARKET_STATE_SOURCE_PANEL_STATUS_MISSING_DATA",
    "MARKET_STATE_SOURCE_PANEL_STATUS_READY",
    "MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_ASSETS",
    "MARKET_STATE_SOURCE_PANEL_STATUS_TOO_FEW_TIMESTAMPS",
    "MARKET_STATE_SOURCE_PANEL_STATUSES",
    "MarketStateSourcePanelRequest",
    "MarketStateSourcePanelResult",
    "MarketStateStalenessPolicy",
    "build_market_state_source_panel",
]
