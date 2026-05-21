from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import to_jsonable
from src.regimes.regime_features.sources import (
    load_ohlcvt_frames_for_assets,
    scalar_feature_partition_metadata,
    source_partition_lineage,
)


RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED = "real_data_loaded"
RELATIONSHIP_DATA_STATUS_INSUFFICIENT_ASSETS = "insufficient_assets"
RELATIONSHIP_DATA_STATUS_INSUFFICIENT_OVERLAP = "insufficient_overlap"
RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA = "missing_source_data"

RELATIONSHIP_DATA_PANEL_STATUSES: tuple[str, ...] = (
    RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED,
    RELATIONSHIP_DATA_STATUS_INSUFFICIENT_ASSETS,
    RELATIONSHIP_DATA_STATUS_INSUFFICIENT_OVERLAP,
    RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA,
)


@dataclass(frozen=True)
class RelationshipReturnPanelRequest:
    ohlcvt_root: str | Path
    interval: int
    assets: Sequence[str]
    scalar_feature_root: str | Path | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    min_assets: int = 3
    min_overlap: int = 30
    evidence_probe: bool = False

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval < 60:
            raise ValueError("Relationship Discovery data panel does not support sub-hour intervals")
        if self.start_ts is not None and self.end_ts is not None and int(self.start_ts) >= int(self.end_ts):
            raise ValueError("Relationship Discovery data panel start_ts must be before end_ts")
        assets = tuple(dict.fromkeys(str(asset).strip() for asset in self.assets if str(asset).strip()))
        if not assets:
            raise ValueError("Relationship Discovery data panel requires explicit assets")
        object.__setattr__(self, "ohlcvt_root", Path(self.ohlcvt_root))
        object.__setattr__(self, "scalar_feature_root", None if self.scalar_feature_root is None else Path(self.scalar_feature_root))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "min_assets", max(1, int(self.min_assets)))
        object.__setattr__(self, "min_overlap", max(2, int(self.min_overlap)))
        object.__setattr__(self, "evidence_probe", bool(self.evidence_probe))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ohlcvt_root": str(self.ohlcvt_root),
            "scalar_feature_root": str(self.scalar_feature_root) if self.scalar_feature_root is not None else None,
            "interval": int(self.interval),
            "assets": list(self.assets),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "min_assets": int(self.min_assets),
            "min_overlap": int(self.min_overlap),
            "evidence_probe": bool(self.evidence_probe),
        }


@dataclass
class RelationshipReturnPanelResult:
    status: str
    interval: int
    assets_requested: Sequence[str]
    assets_loaded: Sequence[str] = ()
    overlap_count: int = 0
    return_panel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    coverage_by_asset: Mapping[str, Any] = field(default_factory=dict)
    scalar_enrichment: Mapping[str, Any] = field(default_factory=dict)
    source_lineage: Sequence[Mapping[str, Any]] = ()
    reason_codes: Sequence[str] = ()
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in RELATIONSHIP_DATA_PANEL_STATUSES:
            raise ValueError(f"Unsupported Relationship Discovery data panel status {status!r}")
        self.status = status
        self.interval = int(self.interval)
        self.assets_requested = tuple(str(asset) for asset in self.assets_requested)
        self.assets_loaded = tuple(str(asset) for asset in self.assets_loaded)
        self.overlap_count = int(self.overlap_count)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))

    @property
    def usable(self) -> bool:
        return self.status == RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_return_panel_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "interval": int(self.interval),
            "assets_requested": list(self.assets_requested),
            "assets_loaded": list(self.assets_loaded),
            "asset_count": len(self.assets_loaded),
            "overlap_count": int(self.overlap_count),
            "return_panel_shape": [int(self.return_panel.shape[0]), int(self.return_panel.shape[1])],
            "coverage_by_asset": to_jsonable(dict(self.coverage_by_asset)),
            "scalar_enrichment": to_jsonable(dict(self.scalar_enrichment)),
            "source_lineage": to_jsonable([dict(item) for item in self.source_lineage]),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_writes_enabled": False,
        }


def build_relationship_return_panel(request: RelationshipReturnPanelRequest) -> RelationshipReturnPanelResult:
    assets = tuple(request.assets)
    if not request.ohlcvt_root.exists() or not request.ohlcvt_root.is_dir():
        return _blocked(
            request,
            RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA,
            ("missing_ohlcvt_root",),
            f"OHLCVT root does not exist: {request.ohlcvt_root}",
        )
    if not (request.ohlcvt_root / f"ohlcvt_{request.interval}").exists():
        return _blocked(
            request,
            RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA,
            ("missing_interval_data",),
            f"OHLCVT interval is unavailable: {request.interval}",
        )

    try:
        raw_frames = load_ohlcvt_frames_for_assets(
            request.ohlcvt_root,
            interval=request.interval,
            assets=assets,
            start_ts=request.start_ts,
            end_ts=request.end_ts,
            columns=("asset", "ts", "close", "volume", "trades"),
        )
    except Exception as exc:
        return _blocked(
            request,
            RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA,
            ("ohlcvt_load_failed",),
            str(exc),
        )

    returns: dict[str, pd.Series] = {}
    coverage: dict[str, Any] = {}
    for asset in assets:
        frame = raw_frames.get(asset, pd.DataFrame())
        series, item = _asset_log_returns(asset, frame)
        coverage[asset] = item
        if series is not None and not series.dropna().empty:
            returns[asset] = series

    if len(returns) < int(request.min_assets):
        return _blocked(
            request,
            RELATIONSHIP_DATA_STATUS_INSUFFICIENT_ASSETS,
            ("insufficient_loaded_assets",),
            f"loaded return asset count {len(returns)} < min_assets {request.min_assets}",
            assets_loaded=tuple(returns),
            coverage_by_asset=coverage,
        )

    panel = pd.DataFrame(returns).sort_index()
    panel.index.name = "ts"
    overlap = panel.dropna(how="any")
    if int(overlap.shape[0]) < int(request.min_overlap):
        return _blocked(
            request,
            RELATIONSHIP_DATA_STATUS_INSUFFICIENT_OVERLAP,
            ("insufficient_common_return_overlap",),
            f"common return overlap {overlap.shape[0]} < min_overlap {request.min_overlap}",
            assets_loaded=tuple(returns),
            coverage_by_asset=coverage,
            return_panel=panel.reset_index(),
            overlap_count=int(overlap.shape[0]),
        )

    loaded_assets = tuple(str(column) for column in overlap.columns)
    scalar_enrichment = _scalar_enrichment_summary(
        request.scalar_feature_root,
        interval=request.interval,
        assets=loaded_assets,
        start_ts=request.start_ts,
        end_ts=request.end_ts,
    )
    lineage = source_partition_lineage(
        request.ohlcvt_root,
        source_kind="ohlcvt",
        interval=request.interval,
        assets=loaded_assets,
        start_ts=request.start_ts,
        end_ts=request.end_ts,
    )
    return RelationshipReturnPanelResult(
        status=RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED,
        interval=request.interval,
        assets_requested=assets,
        assets_loaded=loaded_assets,
        overlap_count=int(overlap.shape[0]),
        return_panel=overlap.reset_index(),
        coverage_by_asset=coverage,
        scalar_enrichment=scalar_enrichment,
        source_lineage=lineage,
        metadata={
            "request": request.as_dict(),
            "return_column_policy": "log_close_return",
            "evidence_probe": bool(request.evidence_probe),
            "production_writes_enabled": False,
        },
    )


def _asset_log_returns(asset: str, frame: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    if frame is None or frame.empty:
        return None, _coverage(asset, row_count=0, return_count=0, first_ts=None, last_ts=None)
    work = frame.copy()
    work["ts"] = pd.to_numeric(work["ts"], errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["ts", "close"]).copy()
    work = work.loc[work["close"] > 0].copy()
    if work.empty:
        return None, _coverage(asset, row_count=0, return_count=0, first_ts=None, last_ts=None)
    work["ts"] = work["ts"].astype("int64")
    work = work.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    returns = np.log(work["close"] / work["close"].shift(1))
    series = pd.Series(returns.to_numpy(), index=work["ts"].to_numpy(), name=str(asset), dtype="float64")
    return_count = int(series.notna().sum())
    return (
        series,
        _coverage(
            asset,
            row_count=int(work.shape[0]),
            return_count=return_count,
            first_ts=int(work["ts"].min()),
            last_ts=int(work["ts"].max()),
        ),
    )


def _coverage(asset: str, *, row_count: int, return_count: int, first_ts: int | None, last_ts: int | None) -> dict[str, Any]:
    return {
        "asset": str(asset),
        "source_row_count": int(row_count),
        "return_count": int(return_count),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def _scalar_enrichment_summary(
    scalar_root: Path | None,
    *,
    interval: int,
    assets: Sequence[str],
    start_ts: int | None,
    end_ts: int | None,
) -> dict[str, Any]:
    if scalar_root is None or not scalar_root.exists() or not scalar_root.is_dir():
        return {"available": False, "asset_count": 0, "assets": {}, "policy": "optional_if_already_available"}
    assets_meta: dict[str, Any] = {}
    for asset in assets:
        meta = scalar_feature_partition_metadata(
            scalar_root,
            interval=int(interval),
            asset=str(asset),
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if meta.available:
            assets_meta[str(asset)] = meta.as_dict()
    return {
        "available": bool(assets_meta),
        "asset_count": int(len(assets_meta)),
        "assets": assets_meta,
        "policy": "optional_if_already_available",
    }


def _blocked(
    request: RelationshipReturnPanelRequest,
    status: str,
    reason_codes: Sequence[str],
    message: str,
    *,
    assets_loaded: Sequence[str] = (),
    coverage_by_asset: Mapping[str, Any] | None = None,
    return_panel: pd.DataFrame | None = None,
    overlap_count: int = 0,
) -> RelationshipReturnPanelResult:
    return RelationshipReturnPanelResult(
        status=status,
        interval=request.interval,
        assets_requested=request.assets,
        assets_loaded=tuple(assets_loaded),
        overlap_count=int(overlap_count),
        return_panel=return_panel if return_panel is not None else pd.DataFrame(),
        coverage_by_asset=coverage_by_asset or {},
        reason_codes=tuple(reason_codes),
        message=message,
        metadata={"request": request.as_dict(), "production_writes_enabled": False},
    )


__all__ = [
    "RELATIONSHIP_DATA_PANEL_STATUSES",
    "RELATIONSHIP_DATA_STATUS_INSUFFICIENT_ASSETS",
    "RELATIONSHIP_DATA_STATUS_INSUFFICIENT_OVERLAP",
    "RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA",
    "RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED",
    "RelationshipReturnPanelRequest",
    "RelationshipReturnPanelResult",
    "build_relationship_return_panel",
]
