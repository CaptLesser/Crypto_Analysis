from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.forecasting.common.path_config import PathConfigError
from src.regimes.core.production_consumer import REGIME_PRODUCTION_BRANCHES
from src.regimes.core.root_resolution import (
    REGIME_PRODUCTION_CONFIGURED_ROOTS_ONLY,
    SOURCE_KIND_OHLCVT,
    resolve_required_regime_production_source_root,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_INPUT_EDGE_ARTIFACT_KIND = "regime_production_input_edge_resolution"
REGIME_PRODUCTION_INPUT_EDGE_SCHEMA_VERSION = 1

INPUT_EDGE_STATUS_READY = "ready"
INPUT_EDGE_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class RegimeProductionInputEdge:
    branch: str
    source_kind: str
    root: str | Path | None
    root_source: str | None
    edge_ts: int | None
    min_ts: int | None = None
    interval_edges: Sequence[Mapping[str, Any]] = ()
    reason_codes: Sequence[str] = ()

    def __post_init__(self) -> None:
        branch = str(self.branch)
        if branch not in REGIME_PRODUCTION_BRANCHES:
            raise ValueError(f"Unsupported Regime Production branch: {branch!r}")
        reasons = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes if str(reason or "").strip()))
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "source_kind", str(self.source_kind))
        object.__setattr__(self, "root", None if self.root in (None, "") else str(self.root))
        object.__setattr__(self, "root_source", None if self.root_source in (None, "") else str(self.root_source))
        object.__setattr__(self, "edge_ts", None if self.edge_ts is None else int(self.edge_ts))
        object.__setattr__(self, "min_ts", None if self.min_ts is None else int(self.min_ts))
        object.__setattr__(self, "interval_edges", tuple(to_jsonable(dict(item)) for item in self.interval_edges))
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def status(self) -> str:
        return INPUT_EDGE_STATUS_READY if self.edge_ts is not None and not self.reason_codes else INPUT_EDGE_STATUS_BLOCKED

    @property
    def passed(self) -> bool:
        return self.status == INPUT_EDGE_STATUS_READY

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_INPUT_EDGE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_INPUT_EDGE_ARTIFACT_KIND,
            "branch": self.branch,
            "source": "shared_regime_production_input_edge_resolver",
            "source_kind": self.source_kind,
            "root": _portable_path(self.root),
            "root_source": self.root_source,
            "configured_root_policy": REGIME_PRODUCTION_CONFIGURED_ROOTS_ONLY,
            "edge_ts": self.edge_ts,
            "min_ts": self.min_ts,
            "interval_edges": [dict(item) for item in self.interval_edges],
            "status": self.status,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "raw_data_edge_drives_output_end": True,
            "clamp_controls_historical_backfill_floor": True,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def resolve_regime_production_input_edge(
    branch: str,
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> RegimeProductionInputEdge:
    branch_name = str(branch)
    if branch_name not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {branch!r}")
    source_env = os.environ if env is None else env
    try:
        root, root_source = resolve_required_regime_production_source_root(
            SOURCE_KIND_OHLCVT,
            env=source_env,
            project_root=project_root,
        )
    except PathConfigError as exc:
        return RegimeProductionInputEdge(
            branch=branch_name,
            source_kind=SOURCE_KIND_OHLCVT,
            root=None,
            root_source=None,
            edge_ts=None,
            reason_codes=(f"source_ohlcvt_root_unresolved:{exc}",),
        )
    except Exception as exc:
        return RegimeProductionInputEdge(
            branch=branch_name,
            source_kind=SOURCE_KIND_OHLCVT,
            root=None,
            root_source=None,
            edge_ts=None,
            reason_codes=(f"source_ohlcvt_edge_resolution_failed:{type(exc).__name__}:{exc}",),
        )

    if not root.exists():
        return RegimeProductionInputEdge(
            branch=branch_name,
            source_kind=SOURCE_KIND_OHLCVT,
            root=root,
            root_source=root_source,
            edge_ts=None,
            reason_codes=("source_ohlcvt_root_missing",),
        )

    intervals = _discover_ohlcvt_interval_edges(root)
    if not intervals:
        return RegimeProductionInputEdge(
            branch=branch_name,
            source_kind=SOURCE_KIND_OHLCVT,
            root=root,
            root_source=root_source,
            edge_ts=None,
            reason_codes=("source_ohlcvt_edge_missing",),
        )

    edge_values = [int(item["edge_ts"]) for item in intervals if item.get("edge_ts") is not None]
    min_values = [int(item["min_ts"]) for item in intervals if item.get("min_ts") is not None]
    return RegimeProductionInputEdge(
        branch=branch_name,
        source_kind=SOURCE_KIND_OHLCVT,
        root=root,
        root_source=root_source,
        edge_ts=max(edge_values) if edge_values else None,
        min_ts=min(min_values) if min_values else None,
        interval_edges=intervals,
        reason_codes=() if edge_values else ("source_ohlcvt_edge_missing",),
    )


def _discover_ohlcvt_interval_edges(root: Path) -> tuple[Mapping[str, Any], ...]:
    interval_dirs: list[tuple[int, Path]] = []
    for interval_dir in sorted(root.glob("ohlcvt_*")):
        if not interval_dir.is_dir():
            continue
        try:
            interval = int(interval_dir.name.split("_", 1)[1])
        except Exception:
            continue
        interval_dirs.append((int(interval), interval_dir))
    for interval, interval_dir in sorted(interval_dirs):
        files = _latest_partition_files(interval_dir)
        if not files:
            continue
        bounds = _ts_bounds(files)
        if bounds is None:
            continue
        min_ts, edge_ts = bounds
        return (
            {
                "interval_min": int(interval),
                "edge_ts": int(edge_ts),
                "min_ts": int(min_ts),
                "latest_partition_file_count": len(files),
                "edge_discovery_strategy": "finest_available_ohlcvt_interval",
            },
        )
    return ()


def _latest_partition_files(interval_dir: Path) -> tuple[Path, ...]:
    month_dirs: dict[tuple[int, int], list[Path]] = {}
    for month_dir in interval_dir.rglob("month=*"):
        if not month_dir.is_dir():
            continue
        year_dir = month_dir.parent
        if not year_dir.name.startswith("year="):
            continue
        try:
            year = int(year_dir.name.split("=", 1)[1])
            month = int(month_dir.name.split("=", 1)[1])
        except Exception:
            continue
        files = sorted(path for path in month_dir.glob("*.parquet") if path.is_file())
        if files:
            month_dirs.setdefault((year, month), []).extend(files)
    if not month_dirs:
        return ()
    latest = max(month_dirs)
    return tuple(sorted(month_dirs[latest]))


def _ts_bounds(files: Sequence[Path]) -> tuple[int, int] | None:
    metadata_bounds = _ts_bounds_from_parquet_metadata(files)
    if metadata_bounds is not None:
        return metadata_bounds
    min_ts: int | None = None
    max_ts: int | None = None
    for path in files:
        try:
            frame = pd.read_parquet(path, columns=["ts"])
        except Exception:
            continue
        if "ts" not in frame.columns or frame.empty:
            continue
        series = frame["ts"].dropna()
        if series.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            values = (series.astype("int64") // 1_000_000_000).astype("int64")
        else:
            values = pd.to_numeric(series, errors="coerce").dropna().astype("int64")
        if values.empty:
            continue
        current_min = int(values.min())
        current_max = int(values.max())
        min_ts = current_min if min_ts is None else min(int(min_ts), current_min)
        max_ts = current_max if max_ts is None else max(int(max_ts), current_max)
    if min_ts is None or max_ts is None:
        return None
    return int(min_ts), int(max_ts)


def _ts_bounds_from_parquet_metadata(files: Sequence[Path]) -> tuple[int, int] | None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None

    min_ts: int | None = None
    max_ts: int | None = None
    for path in files:
        try:
            metadata = pq.ParquetFile(path).metadata
            schema = metadata.schema
            ts_index = schema.names.index("ts")
        except Exception:
            return None
        for row_group_idx in range(metadata.num_row_groups):
            try:
                column = metadata.row_group(row_group_idx).column(ts_index)
                stats = column.statistics
            except Exception:
                return None
            if stats is None or stats.min is None or stats.max is None:
                return None
            current_min = _metadata_ts_value(stats.min)
            current_max = _metadata_ts_value(stats.max)
            if current_min is None or current_max is None:
                return None
            min_ts = current_min if min_ts is None else min(int(min_ts), int(current_min))
            max_ts = current_max if max_ts is None else max(int(max_ts), int(current_max))
    if min_ts is None or max_ts is None:
        return None
    return int(min_ts), int(max_ts)


def _metadata_ts_value(value: Any) -> int | None:
    if value in (None, ""):
        return None


def _portable_path(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    try:
        resolved = path.resolve()
    except Exception:
        return str(path)
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"<external_configured_root>/{resolved.name}"
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    if hasattr(value, "timestamp"):
        try:
            return int(value.timestamp())
        except Exception:
            return None
    try:
        return int(float(value))
    except Exception:
        return None


__all__ = [
    "INPUT_EDGE_STATUS_BLOCKED",
    "INPUT_EDGE_STATUS_READY",
    "REGIME_PRODUCTION_INPUT_EDGE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_INPUT_EDGE_SCHEMA_VERSION",
    "RegimeProductionInputEdge",
    "resolve_regime_production_input_edge",
]
