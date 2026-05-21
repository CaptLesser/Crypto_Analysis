from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.forecasting.common.ohlcvt_source import list_assets_ohlcvt, list_month_partitions, read_ohlcvt
from src.regimes.asset_state.dataset_builder import resolve_scalar_feature_partitions
from src.regimes.core.serialization import to_jsonable


OHLCVT_SOURCE_KIND = "ohlcvt"
SCALAR_FEATURE_SOURCE_KIND = "scalar_features"


@dataclass(frozen=True)
class ScalarFeaturePartitionMetadata:
    asset: str
    interval: int
    partition_count: int
    column_count: int
    row_count: int
    columns: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.partition_count > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "interval": int(self.interval),
            "available": bool(self.available),
            "partition_count": int(self.partition_count),
            "column_count": int(self.column_count),
            "row_count": int(self.row_count),
            "columns": list(self.columns),
        }


def discover_ohlcvt_intervals(source_ohlcvt_root: str | Path) -> tuple[int, ...]:
    root = Path(source_ohlcvt_root)
    intervals: set[int] = set()
    if not root.exists() or not root.is_dir():
        return ()
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("ohlcvt_"):
            continue
        value = child.name.split("_", 1)[1]
        if value.isdigit():
            intervals.add(int(value))
    return tuple(sorted(intervals))


def discover_ohlcvt_assets(source_ohlcvt_root: str | Path, interval: int) -> tuple[str, ...]:
    return tuple(list_assets_ohlcvt(int(interval), root=Path(source_ohlcvt_root)))


def load_ohlcvt_frames_for_assets(
    source_ohlcvt_root: str | Path,
    *,
    interval: int,
    assets: Sequence[str],
    start_ts: int | None = None,
    end_ts: int | None = None,
    columns: Sequence[str] = ("asset", "ts", "open", "high", "low", "close", "volume", "trades"),
) -> dict[str, pd.DataFrame]:
    explicit_assets = tuple(str(asset).strip() for asset in assets if str(asset).strip())
    if not explicit_assets:
        raise ValueError("Regime Feature OHLCVT loading requires explicit assets")
    frames: dict[str, pd.DataFrame] = {}
    for asset in explicit_assets:
        frame = read_ohlcvt(
            asset=asset,
            interval_min=int(interval),
            start_ts=start_ts,
            end_ts=end_ts,
            columns=columns,
            root=Path(source_ohlcvt_root),
        )
        frames[asset] = _normalize_ohlcvt_frame(frame, asset=asset)
    return frames


def scalar_feature_partition_metadata(
    source_feature_root: str | Path | None,
    *,
    interval: int,
    asset: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> ScalarFeaturePartitionMetadata:
    if source_feature_root is None:
        return ScalarFeaturePartitionMetadata(asset=str(asset), interval=int(interval), partition_count=0, column_count=0, row_count=0, columns=())
    paths = resolve_scalar_feature_partitions(
        source_feature_root,
        interval=int(interval),
        asset=str(asset),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    if not paths:
        return ScalarFeaturePartitionMetadata(asset=str(asset), interval=int(interval), partition_count=0, column_count=0, row_count=0, columns=())
    columns: set[str] = set()
    row_count = 0
    for path in paths:
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            columns.update(str(name) for name in parquet_file.schema.names)
            row_count += int(parquet_file.metadata.num_rows)
        except Exception:
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            columns.update(str(column) for column in frame.columns)
            row_count += int(frame.shape[0])
    return ScalarFeaturePartitionMetadata(
        asset=str(asset),
        interval=int(interval),
        partition_count=int(len(paths)),
        column_count=int(len(columns)),
        row_count=int(row_count),
        columns=tuple(sorted(columns)),
    )


def source_partition_lineage(
    source_root: str | Path,
    *,
    source_kind: str,
    interval: int,
    assets: Sequence[str],
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> tuple[dict[str, Any], ...]:
    root = Path(source_root)
    entries: list[dict[str, Any]] = []
    for asset in tuple(str(asset).strip() for asset in assets if str(asset).strip()):
        if source_kind == OHLCVT_SOURCE_KIND:
            paths = list_month_partitions(
                family="ohlcvt",
                interval_min=int(interval),
                asset=asset,
                root=root,
            )
        elif source_kind == SCALAR_FEATURE_SOURCE_KIND:
            paths = list(resolve_scalar_feature_partitions(root, interval=int(interval), asset=asset, start_ts=start_ts, end_ts=end_ts))
        else:
            raise ValueError(f"Unsupported Regime Feature source_kind {source_kind!r}")
        for path in paths:
            entries.append(
                {
                    "source_kind": source_kind,
                    "asset": asset,
                    "interval": int(interval),
                    "path": _safe_path_text(path, root=root),
                }
            )
    return tuple(to_jsonable(entry) for entry in entries)


def _normalize_ohlcvt_frame(frame: pd.DataFrame, *, asset: str) -> pd.DataFrame:
    columns = ["asset", "ts", "open", "high", "low", "close", "volume", "trades"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    if "asset" not in out.columns:
        out["asset"] = str(asset)
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    out["asset"] = out["asset"].fillna(str(asset)).astype(str)
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"]).copy()
    out["ts"] = out["ts"].astype("int64")
    for column in ("open", "high", "low", "close", "volume", "trades"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[columns].drop_duplicates(subset=["asset", "ts"], keep="last").sort_values(["asset", "ts"]).reset_index(drop=True)


def _safe_path_text(path: str | Path, *, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


__all__ = [
    "OHLCVT_SOURCE_KIND",
    "SCALAR_FEATURE_SOURCE_KIND",
    "ScalarFeaturePartitionMetadata",
    "discover_ohlcvt_assets",
    "discover_ohlcvt_intervals",
    "load_ohlcvt_frames_for_assets",
    "scalar_feature_partition_metadata",
    "source_partition_lineage",
]
