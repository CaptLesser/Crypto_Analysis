from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.sandbox_paths import (
    SandboxOutputRoots,
    assert_write_allowed,
    resolve_sandbox_output_roots,
)
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.regime_forecast_io import RegimeLabelReadStats, iter_months_between, read_regime_labels
from src.regimes.contracts import REGIME_AXES, REGIME_AXIS_ORDER, REGIME_BANDS, regime_label_month_dir, regime_table_dir
from src.regimes.core import (
    require_pathway_diagnostics_root,
    safe_path_part,
    write_json,
)


CURRENT_LABEL_AUDIT_SCHEMA_VERSION = 1
DEFAULT_AUDIT_RUN_ID_PREFIX = "asset_state_current_label_audit"
AUDIT_JSON_NAME = "current_label_audit.json"
AUDIT_MARKDOWN_NAME = "current_label_audit.md"


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: object) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp must be non-empty")
    try:
        return int(text)
    except ValueError:
        ts = pd.Timestamp(text)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return int(ts.timestamp())


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv(raw))


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if pd.isna(value):
        return None
    return value


@dataclass(frozen=True)
class AssetStateLabelAuditConfig:
    assets: tuple[str, ...]
    interval_minutes: int
    band: str
    start_ts: int
    end_ts: int
    diagnostics_root: Path
    regime_label_root: Path
    run_id: str = ""
    compute_forward_targets: bool = False
    forward_horizon_minutes: tuple[int, ...] = ()
    ohlcvt_root: Path | None = None
    tiny_state_threshold: int = 20
    allow_legacy_unpartitioned: bool = False
    project_root: Path | None = None

    def __post_init__(self) -> None:
        assets = tuple(str(asset).strip() for asset in self.assets if str(asset).strip())
        if not assets:
            raise ValueError("Asset-state current-label audit requires at least one asset")
        band = str(self.band).strip().lower()
        if band not in REGIME_BANDS:
            valid = ", ".join(REGIME_BANDS)
            raise ValueError(f"Unsupported Regime band {self.band!r}; expected one of: {valid}")
        interval = int(self.interval_minutes)
        if interval <= 0:
            raise ValueError("Audit interval_minutes must be positive")
        if interval not in tuple(int(v) for v in REGIME_BANDS[band].member_intervals):
            raise ValueError(f"Audit interval {interval} is not a member interval for band {band}")
        start_ts = int(self.start_ts)
        end_ts = int(self.end_ts)
        if start_ts > end_ts:
            raise ValueError("Audit start_ts must be <= end_ts")
        horizons = tuple(int(h) for h in self.forward_horizon_minutes)
        if bool(self.compute_forward_targets):
            if not horizons:
                raise ValueError("Forward target audit requires at least one forward horizon")
            bad = [h for h in horizons if h <= 0 or h % interval != 0]
            if bad:
                raise ValueError("Forward horizons must be positive multiples of interval_minutes")
        if int(self.tiny_state_threshold) < 1:
            raise ValueError("tiny_state_threshold must be positive")
        run_id = str(self.run_id).strip() or f"{DEFAULT_AUDIT_RUN_ID_PREFIX}_{_utc_now_compact()}"
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "interval_minutes", interval)
        object.__setattr__(self, "start_ts", start_ts)
        object.__setattr__(self, "end_ts", end_ts)
        object.__setattr__(self, "forward_horizon_minutes", horizons)
        object.__setattr__(self, "tiny_state_threshold", int(self.tiny_state_threshold))
        object.__setattr__(self, "run_id", safe_path_part(run_id, context="Regime current-label audit run_id"))
        object.__setattr__(self, "diagnostics_root", Path(self.diagnostics_root))
        object.__setattr__(self, "regime_label_root", Path(self.regime_label_root))
        if self.ohlcvt_root is not None:
            object.__setattr__(self, "ohlcvt_root", Path(self.ohlcvt_root))

    @property
    def ceiling_interval_min(self) -> int:
        return int(REGIME_BANDS[self.band].ceiling_interval_min)

    @property
    def output_dir(self) -> Path:
        return Path(self.diagnostics_root) / "current_label_audit" / self.run_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "assets": list(self.assets),
            "asset_count": int(len(self.assets)),
            "interval_minutes": int(self.interval_minutes),
            "band": self.band,
            "ceiling_interval_min": int(self.ceiling_interval_min),
            "start_ts": int(self.start_ts),
            "end_ts": int(self.end_ts),
            "diagnostics_root": str(self.diagnostics_root),
            "regime_label_root": str(self.regime_label_root),
            "run_id": self.run_id,
            "compute_forward_targets": bool(self.compute_forward_targets),
            "forward_horizon_minutes": list(self.forward_horizon_minutes),
            "ohlcvt_root": None if self.ohlcvt_root is None else str(self.ohlcvt_root),
            "tiny_state_threshold": int(self.tiny_state_threshold),
            "allow_legacy_unpartitioned": bool(self.allow_legacy_unpartitioned),
        }


def _expected_timestamp_count(start_ts: int, end_ts: int, interval_minutes: int) -> int:
    if int(end_ts) < int(start_ts):
        return 0
    step = int(interval_minutes) * 60
    return int((int(end_ts) - int(start_ts)) // step) + 1


def _missing_timestamp_sample(
    *,
    observed_ts: set[int],
    start_ts: int,
    end_ts: int,
    interval_minutes: int,
    max_expected_scan: int = 100_000,
    max_sample: int = 20,
) -> list[int]:
    expected = _expected_timestamp_count(start_ts, end_ts, interval_minutes)
    if expected > int(max_expected_scan):
        return []
    step = int(interval_minutes) * 60
    sample: list[int] = []
    current = int(start_ts)
    while current <= int(end_ts) and len(sample) < int(max_sample):
        if current not in observed_ts:
            sample.append(current)
        current += step
    return sample


def _state_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    values = series.dropna().astype(str).str.strip()
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def _cluster_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype("int64")
    return {str(int(k)): int(v) for k, v in sorted(Counter(numeric.tolist()).items())}


def _share(count: int, total: int) -> float | None:
    return None if int(total) <= 0 else float(int(count) / int(total))


def _tiny_share(counts: Mapping[str, int], *, total: int, threshold: int) -> float | None:
    if int(total) <= 0:
        return None
    tiny_rows = sum(int(v) for v in counts.values() if int(v) <= int(threshold))
    return float(tiny_rows / int(total))


def _singleton_share(counts: Mapping[str, int], *, total: int) -> float | None:
    if int(total) <= 0:
        return None
    singleton_rows = sum(int(v) for v in counts.values() if int(v) == 1)
    return float(singleton_rows / int(total))


def _largest_share(counts: Mapping[str, int], *, total: int) -> float | None:
    if int(total) <= 0 or not counts:
        return None
    return float(max(int(v) for v in counts.values()) / int(total))


def _asset_axis_audit(
    frame: pd.DataFrame,
    *,
    asset: str,
    axis: str,
    band: str,
    expected_rows: int,
    tiny_state_threshold: int,
) -> dict[str, Any]:
    axis_contract = REGIME_AXES[str(axis)]
    label_col = axis_contract.label_column
    cluster_col = axis_contract.cluster_column
    observed_rows = int(len(frame))
    label_series = frame[label_col] if label_col in frame.columns else pd.Series(dtype=object)
    cluster_series = frame[cluster_col] if cluster_col in frame.columns else pd.Series(dtype=object)
    label_counts = _state_counts(label_series)
    cluster_counts = _cluster_counts(cluster_series)
    normalized_labels = label_series.astype("string").str.strip().str.lower() if not label_series.empty else label_series
    unknown_count = int((normalized_labels == "unknown").sum()) if not label_series.empty else 0
    null_label_count = int(label_series.isna().sum()) if not label_series.empty else 0
    cluster_numeric = pd.to_numeric(cluster_series, errors="coerce") if not cluster_series.empty else pd.Series(dtype=float)
    noise_count = int((cluster_numeric == -1).sum()) if not cluster_series.empty else 0
    null_cluster_count = int(cluster_numeric.isna().sum()) if not cluster_series.empty else 0
    non_unknown_label_counts = {
        label: count for label, count in label_counts.items() if str(label).strip().lower() != "unknown"
    }
    non_noise_cluster_counts = {
        cluster: count for cluster, count in cluster_counts.items() if str(cluster).strip() != "-1"
    }
    return {
        "asset": str(asset),
        "axis": str(axis),
        "band": str(band),
        "row_count": int(observed_rows),
        "expected_timestamp_count": int(expected_rows),
        "label_counts": label_counts,
        "cluster_counts": cluster_counts,
        "effective_label_state_count": int(len(non_unknown_label_counts)),
        "effective_cluster_state_count": int(len(non_noise_cluster_counts)),
        "unknown_count": int(unknown_count),
        "unknown_share": _share(unknown_count, observed_rows),
        "null_label_count": int(null_label_count),
        "null_label_share": _share(null_label_count, observed_rows),
        "noise_cluster_count": int(noise_count),
        "noise_cluster_share": _share(noise_count, observed_rows),
        "null_cluster_count": int(null_cluster_count),
        "null_cluster_share": _share(null_cluster_count, observed_rows),
        "singleton_label_share": _singleton_share(non_unknown_label_counts, total=observed_rows),
        "tiny_label_share": _tiny_share(
            non_unknown_label_counts,
            total=observed_rows,
            threshold=int(tiny_state_threshold),
        ),
        "singleton_cluster_share": _singleton_share(non_noise_cluster_counts, total=observed_rows),
        "tiny_cluster_share": _tiny_share(
            non_noise_cluster_counts,
            total=observed_rows,
            threshold=int(tiny_state_threshold),
        ),
        "largest_label_state_share": _largest_share(non_unknown_label_counts, total=observed_rows),
        "largest_cluster_state_share": _largest_share(non_noise_cluster_counts, total=observed_rows),
    }


def _transition_counts(frame: pd.DataFrame, *, axis: str) -> dict[str, Any]:
    label_col = REGIME_AXES[str(axis)].label_column
    if frame.empty or label_col not in frame.columns:
        return {"status": "no_labels", "transition_counts": {}, "transition_count": 0}
    labels = frame.sort_values("ts")[label_col].astype("string").fillna("<null>").tolist()
    counts: Counter[str] = Counter()
    for prev, current in zip(labels, labels[1:]):
        counts[f"{prev}->{current}"] += 1
    return {
        "status": "computed" if counts else "not_enough_rows",
        "transition_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "transition_count": int(sum(counts.values())),
    }


def _label_mean_spread(frame: pd.DataFrame, labels: pd.Series, column: str) -> tuple[float | None, dict[str, float]]:
    if frame.empty or column not in frame.columns:
        return None, {}
    values = pd.to_numeric(frame[column], errors="coerce")
    label_values = labels.astype("string").str.strip().str.lower()
    valid = values.notna() & label_values.notna() & (label_values != "unknown")
    if int(valid.sum()) < 2:
        return None, {}
    means: dict[str, float] = {}
    for label in sorted(set(label_values[valid].tolist())):
        mask = valid & (label_values == label)
        if int(mask.sum()) > 0:
            value = _safe_float(values[mask].mean())
            if value is not None:
                means[str(label)] = value
    if len(means) < 2:
        return None, means
    return float(max(means.values()) - min(means.values())), means


def _forward_metrics_for_asset(
    labels: pd.DataFrame,
    *,
    asset: str,
    interval_minutes: int,
    horizon_minutes: int,
    start_ts: int,
    end_ts: int,
    ohlcvt_root: Path | None,
) -> dict[str, Any]:
    horizon_bars = int(horizon_minutes) // int(interval_minutes)
    target_end_ts = int(end_ts) + int(horizon_minutes) * 60
    try:
        ohlc = read_ohlcvt(
            asset=str(asset),
            interval_min=int(interval_minutes),
            start_ts=int(start_ts),
            end_ts=int(target_end_ts),
            columns=("asset", "ts", "open", "high", "low", "close"),
            root=ohlcvt_root,
        )
    except Exception as exc:
        return {
            "status": "target_read_failed",
            "horizon_minutes": int(horizon_minutes),
            "horizon_bars": int(horizon_bars),
            "error": str(exc),
            "coverage": {"available_rows": 0, "label_rows": int(len(labels)), "coverage_share": 0.0},
            "axis_separability": {},
        }
    if ohlc.empty:
        return {
            "status": "no_target_data",
            "horizon_minutes": int(horizon_minutes),
            "horizon_bars": int(horizon_bars),
            "coverage": {"available_rows": 0, "label_rows": int(len(labels)), "coverage_share": 0.0},
            "axis_separability": {},
        }
    ohlc = ohlc.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last").reset_index(drop=True)
    targets, target_meta = compute_future_labels(
        ohlc,
        int(horizon_bars),
        future_direction_deadzone=0.0,
        target_columns=("future_log_return", "future_realized_vol", "future_max_drawdown", "future_max_runup"),
    )
    target_frame = pd.concat([ohlc[["asset", "ts"]].reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    merged = labels[["asset", "ts", *(REGIME_AXES[axis].label_column for axis in REGIME_AXIS_ORDER)]].merge(
        target_frame,
        on=["asset", "ts"],
        how="left",
    )
    target_columns = ("future_log_return", "future_realized_vol", "future_max_drawdown", "future_max_runup")
    available = merged[list(target_columns)].notna().any(axis=1) if not merged.empty else pd.Series(dtype=bool)
    coverage = {
        "available_rows": int(available.sum()) if not merged.empty else 0,
        "label_rows": int(len(merged)),
        "coverage_share": _share(int(available.sum()) if not merged.empty else 0, int(len(merged))) or 0.0,
    }
    axis_separability: dict[str, Any] = {}
    for axis in REGIME_AXIS_ORDER:
        labels_for_axis = merged[REGIME_AXES[axis].label_column] if REGIME_AXES[axis].label_column in merged.columns else pd.Series(dtype=object)
        metric_payload: dict[str, Any] = {}
        for column in target_columns:
            spread, means = _label_mean_spread(merged, labels_for_axis, column)
            metric_payload[column] = {
                "mean_spread": spread,
                "per_label_mean": means,
            }
        axis_separability[str(axis)] = metric_payload
    return {
        "status": "computed" if coverage["available_rows"] > 0 else "no_forward_rows",
        "horizon_minutes": int(horizon_minutes),
        "horizon_bars": int(horizon_bars),
        "coverage": coverage,
        "axis_separability": axis_separability,
        "target_metadata": _jsonable(target_meta),
    }


def _source_paths_for_asset(config: AssetStateLabelAuditConfig, *, asset: str) -> tuple[str, ...]:
    paths: list[Path] = []
    table_dir = regime_table_dir(config.ceiling_interval_min)
    for year, month in iter_months_between(config.start_ts, config.end_ts):
        month_dir = regime_label_month_dir(
            config.regime_label_root,
            config.ceiling_interval_min,
            asset,
            year,
            month,
        )
        if month_dir.exists():
            paths.extend(sorted(month_dir.glob("*.parquet")))
        if config.allow_legacy_unpartitioned:
            legacy_dir = Path(config.regime_label_root) / table_dir / f"year={int(year)}" / f"month={int(month):02d}"
            if legacy_dir.exists():
                paths.extend(sorted(legacy_dir.glob("*.parquet")))
    return tuple(str(path) for path in paths)


def _empty_axis_payload(asset: str, band: str, expected_rows: int) -> list[dict[str, Any]]:
    return [
        {
            "asset": str(asset),
            "axis": str(axis),
            "band": str(band),
            "row_count": 0,
            "expected_timestamp_count": int(expected_rows),
            "label_counts": {},
            "cluster_counts": {},
            "effective_label_state_count": 0,
            "effective_cluster_state_count": 0,
            "unknown_count": 0,
            "unknown_share": None,
            "null_label_count": 0,
            "null_label_share": None,
            "noise_cluster_count": 0,
            "noise_cluster_share": None,
            "null_cluster_count": 0,
            "null_cluster_share": None,
            "singleton_label_share": None,
            "tiny_label_share": None,
            "singleton_cluster_share": None,
            "tiny_cluster_share": None,
            "largest_label_state_share": None,
            "largest_cluster_state_share": None,
        }
        for axis in REGIME_AXIS_ORDER
    ]


def build_current_label_audit_payload(config: AssetStateLabelAuditConfig) -> dict[str, Any]:
    expected_rows = _expected_timestamp_count(config.start_ts, config.end_ts, config.interval_minutes)
    assets_payload: list[dict[str, Any]] = []
    read_stats_total = Counter()
    source_paths_by_asset: dict[str, list[str]] = {}
    for asset in config.assets:
        stats = RegimeLabelReadStats()
        columns = [
            "ts",
            "asset",
            "band",
            "ceiling_interval_min",
            *(column for axis in REGIME_AXIS_ORDER for column in REGIME_AXES[axis].output_columns),
            "feature_schema_hash",
        ]
        labels = read_regime_labels(
            base_dir=config.regime_label_root,
            ceiling_interval=config.ceiling_interval_min,
            start_ts=config.start_ts,
            end_ts=config.end_ts,
            asset=str(asset),
            columns=columns,
            allow_legacy_unpartitioned=bool(config.allow_legacy_unpartitioned),
            validate_schema=True,
            stats=stats,
        )
        read_stats_total.update(stats.as_dict())
        source_paths = list(_source_paths_for_asset(config, asset=str(asset)))
        source_paths_by_asset[str(asset)] = source_paths
        observed_ts = {int(v) for v in pd.to_numeric(labels.get("ts", pd.Series(dtype=int)), errors="coerce").dropna().tolist()}
        missing_count = max(0, int(expected_rows) - int(len(observed_ts)))
        axis_payload = (
            _empty_axis_payload(str(asset), config.band, expected_rows)
            if labels.empty
            else [
                _asset_axis_audit(
                    labels,
                    asset=str(asset),
                    axis=str(axis),
                    band=config.band,
                    expected_rows=expected_rows,
                    tiny_state_threshold=config.tiny_state_threshold,
                )
                for axis in REGIME_AXIS_ORDER
            ]
        )
        transition_payload = {
            axis: _transition_counts(labels, axis=str(axis))
            for axis in REGIME_AXIS_ORDER
        }
        forward_payload: dict[str, Any] = {"status": "not_requested", "horizons": {}}
        if config.compute_forward_targets:
            horizon_payload = {
                str(horizon): _forward_metrics_for_asset(
                    labels,
                    asset=str(asset),
                    interval_minutes=config.interval_minutes,
                    horizon_minutes=int(horizon),
                    start_ts=config.start_ts,
                    end_ts=config.end_ts,
                    ohlcvt_root=config.ohlcvt_root,
                )
                for horizon in config.forward_horizon_minutes
            }
            forward_payload = {
                "status": "computed" if any(v.get("status") == "computed" for v in horizon_payload.values()) else "not_available",
                "horizons": horizon_payload,
            }
        assets_payload.append(
            {
                "asset": str(asset),
                "row_count": int(len(labels)),
                "expected_timestamp_count": int(expected_rows),
                "missing_timestamp_count": int(missing_count),
                "missing_timestamp_share": _share(missing_count, expected_rows),
                "missing_timestamp_sample": _missing_timestamp_sample(
                    observed_ts=observed_ts,
                    start_ts=config.start_ts,
                    end_ts=config.end_ts,
                    interval_minutes=config.interval_minutes,
                ),
                "source_paths": source_paths,
                "read_stats": stats.as_dict(),
                "axis_metrics": axis_payload,
                "transition_metrics": transition_payload,
                "forward_metrics": forward_payload,
            }
        )
    aggregate = _aggregate_audit(assets_payload)
    payload = {
        "schema_version": CURRENT_LABEL_AUDIT_SCHEMA_VERSION,
        "artifact_kind": "asset_state_current_label_audit",
        "status": "completed",
        "created_at_utc": _utc_now_iso(),
        "config": config.as_dict(),
        "artifact_boundary": {
            "diagnostics_only": True,
            "production_outputs_written": False,
            "production_regime_parquet_written": False,
            "production_definitions_written": False,
            "label_generation_invoked": False,
            "clustering_invoked": False,
        },
        "source_lineage": {
            "artifact_kind": "asset_state_regime_label_parquet",
            "regime_label_root": str(config.regime_label_root),
            "table_dir": regime_table_dir(config.ceiling_interval_min),
            "source_paths_by_asset": source_paths_by_asset,
        },
        "read_stats": {str(k): int(v) for k, v in sorted(read_stats_total.items())},
        "assets": assets_payload,
        "aggregate": aggregate,
    }
    return _jsonable(payload)


def _aggregate_audit(assets_payload: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows_total = sum(int(asset.get("row_count", 0) or 0) for asset in assets_payload)
    expected_total = sum(int(asset.get("expected_timestamp_count", 0) or 0) for asset in assets_payload)
    missing_total = sum(int(asset.get("missing_timestamp_count", 0) or 0) for asset in assets_payload)
    axis_summary: dict[str, Any] = {}
    for axis in REGIME_AXIS_ORDER:
        axis_rows = [
            row
            for asset in assets_payload
            for row in asset.get("axis_metrics", [])
            if row.get("axis") == axis
        ]
        axis_summary[str(axis)] = {
            "assets_with_rows": int(sum(1 for row in axis_rows if int(row.get("row_count", 0) or 0) > 0)),
            "max_effective_label_state_count": max(
                [int(row.get("effective_label_state_count", 0) or 0) for row in axis_rows] or [0]
            ),
            "single_label_asset_count": int(
                sum(1 for row in axis_rows if int(row.get("effective_label_state_count", 0) or 0) == 1)
            ),
            "unknown_rows": int(sum(int(row.get("unknown_count", 0) or 0) for row in axis_rows)),
            "noise_cluster_rows": int(sum(int(row.get("noise_cluster_count", 0) or 0) for row in axis_rows)),
            "null_label_rows": int(sum(int(row.get("null_label_count", 0) or 0) for row in axis_rows)),
        }
    forward_status_counts = Counter()
    for asset in assets_payload:
        forward = asset.get("forward_metrics", {})
        forward_status_counts[str(forward.get("status", "not_requested"))] += 1
    return {
        "asset_count": int(len(assets_payload)),
        "row_count": int(rows_total),
        "expected_timestamp_count": int(expected_total),
        "missing_timestamp_count": int(missing_total),
        "missing_timestamp_share": _share(missing_total, expected_total),
        "axis_summary": axis_summary,
        "forward_status_counts": {str(k): int(v) for k, v in sorted(forward_status_counts.items())},
    }


def build_current_label_audit_markdown(payload: Mapping[str, Any]) -> str:
    config = dict(payload.get("config", {}) or {})
    aggregate = dict(payload.get("aggregate", {}) or {})
    lines = [
        "# Asset-State Current Label Audit",
        "",
        f"- Run ID: `{config.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Band: `{config.get('band')}`",
        f"- Interval minutes: `{config.get('interval_minutes')}`",
        f"- Window: `{config.get('start_ts')}` to `{config.get('end_ts')}`",
        f"- Assets: `{', '.join(config.get('assets', []))}`",
        f"- Rows loaded: `{aggregate.get('row_count')}`",
        f"- Missing timestamp share: `{aggregate.get('missing_timestamp_share')}`",
        "",
        "## Axis Summary",
        "",
        "| Axis | Assets With Rows | Single-State Assets | Max Effective States | Unknown Rows | Noise Rows | Null Label Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for axis, row in dict(aggregate.get("axis_summary", {}) or {}).items():
        lines.append(
            "| {axis} | {assets_with_rows} | {single_label_asset_count} | {max_effective_label_state_count} | "
            "{unknown_rows} | {noise_cluster_rows} | {null_label_rows} |".format(axis=axis, **row)
        )
    lines.extend(["", "## Asset Coverage", ""])
    lines.append("| Asset | Rows | Expected | Missing Share | Source Files | Forward Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for asset in payload.get("assets", []):
        forward = dict(asset.get("forward_metrics", {}) or {})
        lines.append(
            "| {asset} | {rows} | {expected} | {missing} | {sources} | {forward_status} |".format(
                asset=asset.get("asset"),
                rows=asset.get("row_count"),
                expected=asset.get("expected_timestamp_count"),
                missing=asset.get("missing_timestamp_share"),
                sources=len(asset.get("source_paths", []) or []),
                forward_status=forward.get("status", "not_requested"),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Diagnostics only: `true`",
            "- Production outputs written: `false`",
            "- Clustering invoked: `false`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text_atomic(
    path: Path,
    text: str,
    *,
    write_kind: str,
    sandbox_roots: SandboxOutputRoots | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_write_allowed(path, write_kind, roots=sandbox_roots)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, f"{write_kind} temp", roots=sandbox_roots)
    try:
        tmp.write_text(text, encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def write_current_label_audit_outputs(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    project_root: Path | None = None,
    sandbox_roots: SandboxOutputRoots | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    require_pathway_diagnostics_root(output_dir, for_source_probe=True, project_root=project_root)
    json_path = output_dir / AUDIT_JSON_NAME
    markdown_path = output_dir / AUDIT_MARKDOWN_NAME
    artifact_paths = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    serial_payload = dict(payload)
    serial_payload.setdefault("artifact_paths", artifact_paths)
    write_json(
        json_path,
        serial_payload,
        write_kind="Regime asset-state current-label audit JSON",
        sandbox_roots=sandbox_roots,
    )
    _write_text_atomic(
        markdown_path,
        build_current_label_audit_markdown(payload),
        write_kind="Regime asset-state current-label audit Markdown",
        sandbox_roots=sandbox_roots,
    )
    return artifact_paths


def run_current_label_audit(
    config: AssetStateLabelAuditConfig,
    *,
    write_outputs: bool = True,
    sandbox_roots: SandboxOutputRoots | None = None,
) -> dict[str, Any]:
    payload = build_current_label_audit_payload(config)
    artifact_paths: dict[str, str] = {}
    if bool(write_outputs):
        artifact_paths = write_current_label_audit_outputs(
            payload,
            output_dir=config.output_dir,
            project_root=config.project_root,
            sandbox_roots=sandbox_roots,
        )
        payload["artifact_paths"] = artifact_paths
    return {
        "payload": payload,
        "artifact_paths": artifact_paths,
    }


def _default_regime_label_root(profile: str) -> Path:
    return Path(
        resolve_path("source_regime_root", profile=profile, required=False)
        or resolve_path("output_parquet_root", profile=profile, required=False)
        or Path("parquet")
    )


def _default_ohlcvt_root(profile: str) -> Path | None:
    resolved = resolve_path("source_ohlcvt_root", profile=profile, required=False) or resolve_path(
        "output_parquet_root",
        profile=profile,
        required=False,
    )
    return None if resolved is None else Path(resolved)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded audit of existing asset-state Regime labels.")
    parser.add_argument("--assets", required=True, help="Comma-delimited bounded asset panel, e.g. AAVEUSD,XBTUSD")
    parser.add_argument("--interval", type=int, required=True, help="Source/member interval minutes for coverage grid")
    parser.add_argument("--band", choices=tuple(REGIME_BANDS.keys()), required=True)
    parser.add_argument("--start-ts", required=True, help="Inclusive UTC timestamp as epoch seconds or ISO datetime")
    parser.add_argument("--end-ts", required=True, help="Inclusive UTC timestamp as epoch seconds or ISO datetime")
    parser.add_argument("--run-id", default="", help="Optional run id used for diagnostic output path")
    parser.add_argument("--regime-label-root", type=Path, default=None, help="Existing asset-state Regime label root")
    parser.add_argument("--ohlcvt-root", type=Path, default=None, help="Optional OHLCVT root for forward target scoring")
    parser.add_argument("--diagnostics-root", type=Path, default=None, help="Approved diagnostics/report/sandbox root")
    parser.add_argument("--forward-horizons", default="", help="Comma-delimited forward horizons in minutes")
    parser.add_argument("--compute-forward-targets", action="store_true", help="Compute read-only forward target metrics")
    parser.add_argument("--tiny-state-threshold", type=int, default=20)
    parser.add_argument("--allow-legacy-unpartitioned", action="store_true")
    parser.add_argument("--profile", default="", help="Pipeline path profile for default source roots")
    parser.add_argument("--sandbox-output-root", type=Path, default=None, help="Sandbox output root for diagnostics writes")
    return parser


def config_from_args(args: argparse.Namespace) -> tuple[AssetStateLabelAuditConfig, SandboxOutputRoots]:
    profile = str(getattr(args, "profile", "") or selected_profile())
    sandbox_roots = resolve_sandbox_output_roots(args)
    diagnostics_root = (
        Path(getattr(args, "diagnostics_root"))
        if getattr(args, "diagnostics_root", None) is not None
        else sandbox_roots.diagnostics_root / "regime_current_label_audit"
        if sandbox_roots.enabled
        else Path("reports") / "codex_automation" / "regimes" / "current_label_audit"
    )
    regime_label_root = (
        Path(getattr(args, "regime_label_root"))
        if getattr(args, "regime_label_root", None) is not None
        else _default_regime_label_root(profile)
    )
    ohlcvt_root = (
        Path(getattr(args, "ohlcvt_root"))
        if getattr(args, "ohlcvt_root", None) is not None
        else _default_ohlcvt_root(profile)
    )
    config = AssetStateLabelAuditConfig(
        assets=_csv(getattr(args, "assets", "")),
        interval_minutes=int(getattr(args, "interval")),
        band=str(getattr(args, "band")),
        start_ts=parse_timestamp(getattr(args, "start_ts")),
        end_ts=parse_timestamp(getattr(args, "end_ts")),
        diagnostics_root=diagnostics_root,
        regime_label_root=regime_label_root,
        run_id=str(getattr(args, "run_id", "") or ""),
        compute_forward_targets=bool(getattr(args, "compute_forward_targets", False)),
        forward_horizon_minutes=_int_csv(getattr(args, "forward_horizons", "")),
        ohlcvt_root=ohlcvt_root,
        tiny_state_threshold=int(getattr(args, "tiny_state_threshold", 20)),
        allow_legacy_unpartitioned=bool(getattr(args, "allow_legacy_unpartitioned", False)),
        project_root=Path.cwd(),
    )
    return config, sandbox_roots


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config, sandbox_roots = config_from_args(args)
    result = run_current_label_audit(config, write_outputs=True, sandbox_roots=sandbox_roots)
    print(json.dumps(result["artifact_paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
