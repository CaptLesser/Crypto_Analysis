from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.regime_features.contracts import REGIME_FEATURES_SCHEMA_VERSION


TIMESTAMP_COLUMN = "ts"
ASSET_COLUMN = "asset"

ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED = "eligible_observed"
ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED = "likely_flat_or_pegged"
ELIGIBILITY_STATUS_LIKELY_SPARSE = "likely_sparse"
ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY = "likely_low_activity"
ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE = "likely_interval_unreliable"
ELIGIBILITY_STATUS_INSUFFICIENT_DATA = "insufficient_data"
ELIGIBILITY_STATUS_NEEDS_REVIEW = "needs_review"

REGIME_FEATURE_ELIGIBILITY_STATUSES: tuple[str, ...] = (
    ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED,
    ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED,
    ELIGIBILITY_STATUS_LIKELY_SPARSE,
    ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY,
    ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE,
    ELIGIBILITY_STATUS_INSUFFICIENT_DATA,
    ELIGIBILITY_STATUS_NEEDS_REVIEW,
)


class RegimeFeatureEligibilityStatus(str, Enum):
    ELIGIBLE_OBSERVED = ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED
    LIKELY_FLAT_OR_PEGGED = ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED
    LIKELY_SPARSE = ELIGIBILITY_STATUS_LIKELY_SPARSE
    LIKELY_LOW_ACTIVITY = ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY
    LIKELY_INTERVAL_UNRELIABLE = ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE
    INSUFFICIENT_DATA = ELIGIBILITY_STATUS_INSUFFICIENT_DATA
    NEEDS_REVIEW = ELIGIBILITY_STATUS_NEEDS_REVIEW


@dataclass(frozen=True)
class RegimeFeatureEligibilityConfig:
    snapshot_id: str
    interval: int
    band: str
    expected_row_count: int | None = None
    min_observed_rows_hint: int = 8
    sparse_coverage_ratio_hint: float = 0.50
    low_activity_coverage_hint: float = 0.20
    flat_zero_return_share_hint: float = 0.95
    low_movement_abs_return_hint: float = 1e-8
    interval_gap_ratio_hint: float = 0.20
    extreme_return_abs_hint: float = 0.50
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature eligibility interval must be positive")
        if self.expected_row_count is not None and int(self.expected_row_count) <= 0:
            raise ValueError("Regime Feature eligibility expected_row_count must be positive when supplied")
        if int(self.min_observed_rows_hint) <= 0:
            raise ValueError("Regime Feature eligibility min_observed_rows_hint must be positive")
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "expected_row_count", None if self.expected_row_count is None else int(self.expected_row_count))
        object.__setattr__(self, "min_observed_rows_hint", int(self.min_observed_rows_hint))
        object.__setattr__(self, "sparse_coverage_ratio_hint", _coverage(self.sparse_coverage_ratio_hint, field_name="sparse_coverage_ratio_hint"))
        object.__setattr__(self, "low_activity_coverage_hint", _coverage(self.low_activity_coverage_hint, field_name="low_activity_coverage_hint"))
        object.__setattr__(self, "flat_zero_return_share_hint", _coverage(self.flat_zero_return_share_hint, field_name="flat_zero_return_share_hint"))
        object.__setattr__(self, "low_movement_abs_return_hint", float(self.low_movement_abs_return_hint))
        object.__setattr__(self, "interval_gap_ratio_hint", _coverage(self.interval_gap_ratio_hint, field_name="interval_gap_ratio_hint"))
        object.__setattr__(self, "extreme_return_abs_hint", float(self.extreme_return_abs_hint))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_eligibility_config",
            "snapshot_id": self.snapshot_id,
            "interval": int(self.interval),
            "band": self.band,
            "expected_row_count": self.expected_row_count,
            "min_observed_rows_hint": int(self.min_observed_rows_hint),
            "sparse_coverage_ratio_hint": float(self.sparse_coverage_ratio_hint),
            "low_activity_coverage_hint": float(self.low_activity_coverage_hint),
            "flat_zero_return_share_hint": float(self.flat_zero_return_share_hint),
            "low_movement_abs_return_hint": float(self.low_movement_abs_return_hint),
            "interval_gap_ratio_hint": float(self.interval_gap_ratio_hint),
            "extreme_return_abs_hint": float(self.extreme_return_abs_hint),
            "descriptive_only": True,
            "final_filter_encoded": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class RegimeFeatureEligibilityRow:
    asset: str
    interval: int
    band: str
    status: str
    status_reasons: Sequence[str]
    coverage_diagnostics: Mapping[str, Any]
    finite_return_coverage: Mapping[str, Any]
    movement_diagnostics: Mapping[str, Any]
    activity_diagnostics: Mapping[str, Any]
    risk_diagnostics: Mapping[str, Any]
    interval_reliability_hints: Mapping[str, Any]
    scalar_feature_availability: Mapping[str, Any]
    ohlcvt_availability: Mapping[str, Any]
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        status = _status(self.status)
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "interval", int(self.interval))
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "status_reasons", _string_tuple(self.status_reasons, field_name="status_reasons"))
        object.__setattr__(self, "coverage_diagnostics", to_jsonable(dict(self.coverage_diagnostics)))
        object.__setattr__(self, "finite_return_coverage", to_jsonable(dict(self.finite_return_coverage)))
        object.__setattr__(self, "movement_diagnostics", to_jsonable(dict(self.movement_diagnostics)))
        object.__setattr__(self, "activity_diagnostics", to_jsonable(dict(self.activity_diagnostics)))
        object.__setattr__(self, "risk_diagnostics", to_jsonable(dict(self.risk_diagnostics)))
        object.__setattr__(self, "interval_reliability_hints", to_jsonable(dict(self.interval_reliability_hints)))
        object.__setattr__(self, "scalar_feature_availability", to_jsonable(dict(self.scalar_feature_availability)))
        object.__setattr__(self, "ohlcvt_availability", to_jsonable(dict(self.ohlcvt_availability)))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_eligibility_row",
            "asset": self.asset,
            "interval": int(self.interval),
            "band": self.band,
            "status": self.status,
            "status_reasons": list(self.status_reasons),
            "coverage_diagnostics": to_jsonable(dict(self.coverage_diagnostics)),
            "finite_return_coverage": to_jsonable(dict(self.finite_return_coverage)),
            "movement_diagnostics": to_jsonable(dict(self.movement_diagnostics)),
            "activity_diagnostics": to_jsonable(dict(self.activity_diagnostics)),
            "risk_diagnostics": to_jsonable(dict(self.risk_diagnostics)),
            "interval_reliability_hints": to_jsonable(dict(self.interval_reliability_hints)),
            "scalar_feature_availability": to_jsonable(dict(self.scalar_feature_availability)),
            "ohlcvt_availability": to_jsonable(dict(self.ohlcvt_availability)),
            "descriptive_only": True,
            "final_filter_encoded": False,
            "clustering_performed": False,
            "peer_discovery_performed": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureEligibilityRow":
        obj = require_json_object(payload, context="RegimeFeatureEligibilityRow")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            asset=obj["asset"],
            interval=obj["interval"],
            band=obj["band"],
            status=obj["status"],
            status_reasons=obj.get("status_reasons", ()),
            coverage_diagnostics=obj["coverage_diagnostics"],
            finite_return_coverage=obj["finite_return_coverage"],
            movement_diagnostics=obj["movement_diagnostics"],
            activity_diagnostics=obj["activity_diagnostics"],
            risk_diagnostics=obj["risk_diagnostics"],
            interval_reliability_hints=obj["interval_reliability_hints"],
            scalar_feature_availability=obj["scalar_feature_availability"],
            ohlcvt_availability=obj["ohlcvt_availability"],
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureEligibilityRow":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureEligibilityRow JSON"))


@dataclass(frozen=True)
class RegimeFeatureEligibilitySnapshot:
    config: RegimeFeatureEligibilityConfig
    rows: Sequence[RegimeFeatureEligibilityRow]
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def snapshot_hash(self) -> str:
        payload = {
            "config": self.config.as_dict(),
            "rows": [row.as_dict() for row in self.rows],
        }
        return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for row in self.rows:
            status_counts[row.status] = status_counts.get(row.status, 0) + 1
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_eligibility_snapshot",
            "snapshot_id": self.config.snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "config": self.config.as_dict(),
            "interval": int(self.config.interval),
            "band": self.config.band,
            "row_count": int(len(self.rows)),
            "status_counts": dict(sorted(status_counts.items())),
            "statuses": list(REGIME_FEATURE_ELIGIBILITY_STATUSES),
            "rows": [row.as_dict() for row in self.rows],
            "descriptive_only": True,
            "final_filter_encoded": False,
            "clustering_performed": False,
            "peer_discovery_performed": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureEligibilitySnapshot":
        obj = require_json_object(payload, context="RegimeFeatureEligibilitySnapshot")
        config_payload = obj.get("config")
        if isinstance(config_payload, Mapping):
            config = RegimeFeatureEligibilityConfig(
                snapshot_id=config_payload["snapshot_id"],
                interval=config_payload["interval"],
                band=config_payload["band"],
                expected_row_count=config_payload.get("expected_row_count"),
                min_observed_rows_hint=config_payload.get("min_observed_rows_hint", 8),
                sparse_coverage_ratio_hint=config_payload.get("sparse_coverage_ratio_hint", 0.50),
                low_activity_coverage_hint=config_payload.get("low_activity_coverage_hint", 0.20),
                flat_zero_return_share_hint=config_payload.get("flat_zero_return_share_hint", 0.95),
                low_movement_abs_return_hint=config_payload.get("low_movement_abs_return_hint", 1e-8),
                interval_gap_ratio_hint=config_payload.get("interval_gap_ratio_hint", 0.20),
                extreme_return_abs_hint=config_payload.get("extreme_return_abs_hint", 0.50),
                schema_version=config_payload.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
                metadata=config_payload.get("metadata", {}),
            )
        else:
            config = RegimeFeatureEligibilityConfig(
                snapshot_id=obj["snapshot_id"],
                interval=obj["interval"],
                band=obj["band"],
            )
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            config=config,
            rows=tuple(RegimeFeatureEligibilityRow.from_dict(row) for row in obj.get("rows", ())),
            metadata=obj.get("metadata", {}),
        )


def build_regime_feature_eligibility_snapshot(
    *,
    config: RegimeFeatureEligibilityConfig,
    ohlcvt_frames_by_asset: Mapping[str, pd.DataFrame] | None = None,
    scalar_feature_metadata_by_asset: Mapping[str, Mapping[str, Any]] | None = None,
) -> RegimeFeatureEligibilitySnapshot:
    ohlcvt_frames = dict(ohlcvt_frames_by_asset or {})
    scalar_metadata = dict(scalar_feature_metadata_by_asset or {})
    assets = sorted({str(asset) for asset in ohlcvt_frames} | {str(asset) for asset in scalar_metadata})
    if not assets:
        raise ValueError("Regime Feature eligibility snapshot requires OHLCVT or Scalar Feature inputs")
    rows = tuple(
        build_regime_feature_eligibility_row(
            asset=asset,
            config=config,
            ohlcvt_frame=ohlcvt_frames.get(asset),
            scalar_feature_metadata=scalar_metadata.get(asset),
        )
        for asset in assets
    )
    return RegimeFeatureEligibilitySnapshot(config=config, rows=rows)


def build_regime_feature_eligibility_row(
    *,
    asset: str,
    config: RegimeFeatureEligibilityConfig,
    ohlcvt_frame: pd.DataFrame | None = None,
    scalar_feature_metadata: Mapping[str, Any] | None = None,
) -> RegimeFeatureEligibilityRow:
    frame = _normalize_ohlcvt_frame(ohlcvt_frame, asset=asset)
    scalar_summary = _scalar_summary(scalar_feature_metadata)
    ohlcvt_summary = _ohlcvt_summary(frame)
    coverage = _coverage_diagnostics(frame, config=config)
    returns = _return_series(frame, interval=int(config.interval))
    finite_return = returns[np.isfinite(returns)]
    finite_return_coverage = {
        "finite_return_ratio": _finite_ratio(returns),
        "finite_return_count": int(finite_return.shape[0]),
        "return_observation_count": int(returns.shape[0]),
    }
    movement = _movement_diagnostics(finite_return, config=config)
    activity = _activity_diagnostics(frame)
    risk = _risk_diagnostics(coverage=coverage, movement=movement, activity=activity, config=config)
    reliability = _interval_reliability_hints(frame=frame, returns=returns, coverage=coverage, config=config)
    status, reasons = _status_from_diagnostics(
        coverage=coverage,
        movement=movement,
        activity=activity,
        risk=risk,
        reliability=reliability,
        ohlcvt_summary=ohlcvt_summary,
        scalar_summary=scalar_summary,
        config=config,
    )
    return RegimeFeatureEligibilityRow(
        asset=asset,
        interval=int(config.interval),
        band=config.band,
        status=status,
        status_reasons=reasons,
        coverage_diagnostics=coverage,
        finite_return_coverage=finite_return_coverage,
        movement_diagnostics=movement,
        activity_diagnostics=activity,
        risk_diagnostics=risk,
        interval_reliability_hints=reliability,
        scalar_feature_availability=scalar_summary,
        ohlcvt_availability=ohlcvt_summary,
        schema_version=int(config.schema_version),
    )


def _normalize_ohlcvt_frame(frame: pd.DataFrame | None, *, asset: str) -> pd.DataFrame:
    columns = [ASSET_COLUMN, TIMESTAMP_COLUMN, "close", "volume", "trades"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    if ASSET_COLUMN not in out.columns:
        out[ASSET_COLUMN] = str(asset)
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    out[ASSET_COLUMN] = out[ASSET_COLUMN].fillna(str(asset)).astype(str)
    out = out[out[ASSET_COLUMN] == str(asset)].copy()
    out[TIMESTAMP_COLUMN] = pd.to_numeric(out[TIMESTAMP_COLUMN], errors="coerce")
    out = out.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    out[TIMESTAMP_COLUMN] = out[TIMESTAMP_COLUMN].astype("int64")
    for column in ("close", "volume", "trades"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[columns].drop_duplicates([ASSET_COLUMN, TIMESTAMP_COLUMN], keep="last").sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def _coverage_diagnostics(frame: pd.DataFrame, *, config: RegimeFeatureEligibilityConfig) -> dict[str, Any]:
    if frame.empty:
        expected = int(config.expected_row_count or 0)
        return {
            "first_ts": None,
            "last_ts": None,
            "row_count": 0,
            "expected_row_count": expected,
            "coverage_ratio": 0.0,
            "finite_close_ratio": 0.0,
            "missing_row_count_hint": expected,
        }
    first_ts = int(frame[TIMESTAMP_COLUMN].min())
    last_ts = int(frame[TIMESTAMP_COLUMN].max())
    expected = config.expected_row_count
    if expected is None:
        step_seconds = int(config.interval) * 60
        expected = max(1, int((last_ts - first_ts) // step_seconds) + 1)
    row_count = int(frame.shape[0])
    return {
        "first_ts": first_ts,
        "last_ts": last_ts,
        "row_count": row_count,
        "expected_row_count": int(expected),
        "coverage_ratio": _clamp01(row_count / max(1, int(expected))),
        "finite_close_ratio": _finite_ratio(frame["close"]),
        "missing_row_count_hint": max(0, int(expected) - row_count),
    }


def _return_series(frame: pd.DataFrame, *, interval: int) -> pd.Series:
    if frame.empty or "close" not in frame.columns:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns = np.log(close).diff()
    step_seconds = int(interval) * 60
    ts = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce")
    valid_step = ts.diff().fillna(step_seconds).eq(step_seconds)
    return pd.Series(returns).where(valid_step)


def _movement_diagnostics(finite_return: pd.Series, *, config: RegimeFeatureEligibilityConfig) -> dict[str, Any]:
    if finite_return.empty:
        return {
            "log_return_nonzero_share": 0.0,
            "zero_log_return_share": 1.0,
            "near_zero_log_return_share": 1.0,
            "median_abs_log_return": 0.0,
            "mean_abs_log_return": 0.0,
            "realized_volatility": 0.0,
            "flat_or_pegged_risk": 1.0,
        }
    abs_return = np.abs(finite_return)
    zero_share = _share(np.isclose(finite_return, 0.0, atol=0.0))
    near_zero_share = _share(np.isclose(finite_return, 0.0, atol=max(float(config.low_movement_abs_return_hint), 0.0)))
    median_abs = float(np.nanmedian(abs_return))
    realized_vol = float(np.nanstd(finite_return, ddof=0))
    flat_risk = _clamp01(max(zero_share, near_zero_share * 0.8, 1.0 - (median_abs / max(float(config.low_movement_abs_return_hint), 1e-12))))
    return {
        "log_return_nonzero_share": _clamp01(1.0 - zero_share),
        "zero_log_return_share": zero_share,
        "near_zero_log_return_share": near_zero_share,
        "median_abs_log_return": median_abs,
        "mean_abs_log_return": float(np.nanmean(abs_return)),
        "realized_volatility": realized_vol,
        "flat_or_pegged_risk": flat_risk,
    }


def _activity_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "volume_available": False,
            "trades_available": False,
            "finite_volume_ratio": 0.0,
            "finite_trades_ratio": 0.0,
            "zero_volume_share": 1.0,
            "zero_trades_share": 1.0,
            "median_volume": 0.0,
            "median_trades": 0.0,
            "activity_coverage_score": 0.0,
        }
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    trades = pd.to_numeric(frame.get("trades"), errors="coerce")
    volume_available = bool(volume.notna().any())
    trades_available = bool(trades.notna().any())
    finite_volume_ratio = _finite_ratio(volume)
    finite_trades_ratio = _finite_ratio(trades)
    zero_volume_share = _share(volume.fillna(0.0).eq(0.0)) if len(volume) else 1.0
    zero_trades_share = _share(trades.fillna(0.0).eq(0.0)) if len(trades) else 1.0
    volume_activity = finite_volume_ratio * (1.0 - zero_volume_share)
    trades_activity = finite_trades_ratio * (1.0 - zero_trades_share)
    activity_score = _clamp01(max(volume_activity, trades_activity))
    return {
        "volume_available": volume_available,
        "trades_available": trades_available,
        "finite_volume_ratio": finite_volume_ratio,
        "finite_trades_ratio": finite_trades_ratio,
        "zero_volume_share": zero_volume_share,
        "zero_trades_share": zero_trades_share,
        "median_volume": float(np.nanmedian(volume)) if np.isfinite(volume).any() else 0.0,
        "median_trades": float(np.nanmedian(trades)) if np.isfinite(trades).any() else 0.0,
        "activity_score": activity_score,
        "activity_coverage_score": activity_score,
    }


def _risk_diagnostics(
    *,
    coverage: Mapping[str, Any],
    movement: Mapping[str, Any],
    activity: Mapping[str, Any],
    config: RegimeFeatureEligibilityConfig,
) -> dict[str, Any]:
    sparse_risk = _clamp01(1.0 - float(coverage.get("coverage_ratio") or 0.0))
    flat_risk = _clamp01(float(movement.get("flat_or_pegged_risk") or 0.0))
    low_activity_risk = _clamp01(1.0 - (float(activity.get("activity_coverage_score") or 0.0) / max(float(config.low_activity_coverage_hint), 1e-12)))
    problematic_risk = _clamp01(max(sparse_risk, flat_risk * 0.7, low_activity_risk * 0.7))
    return {
        "flat_or_pegged_risk": flat_risk,
        "sparse_problematic_risk": problematic_risk,
        "sparse_coverage_risk": sparse_risk,
        "low_activity_risk": low_activity_risk,
    }


def _interval_reliability_hints(
    *,
    frame: pd.DataFrame,
    returns: pd.Series,
    coverage: Mapping[str, Any],
    config: RegimeFeatureEligibilityConfig,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "high_gap_ratio": 1.0,
            "extreme_return_outlier_share": 0.0,
            "interval_unreliable_hint": True,
        }
    step_seconds = int(config.interval) * 60
    ts = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce").dropna().astype("int64")
    if ts.shape[0] <= 1:
        gap_ratio = 1.0
    else:
        gap_ratio = _share(ts.diff().dropna().gt(step_seconds))
    finite_return = returns[np.isfinite(returns)]
    outlier_share = _share(np.abs(finite_return).gt(float(config.extreme_return_abs_hint))) if not finite_return.empty else 0.0
    unreliable = bool(
        gap_ratio > float(config.interval_gap_ratio_hint)
        or outlier_share > float(config.interval_gap_ratio_hint)
    )
    return {
        "high_gap_ratio": gap_ratio,
        "extreme_return_outlier_share": outlier_share,
        "interval_unreliable_hint": unreliable,
    }


def _scalar_summary(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(metadata or {})
    available = bool(payload.get("available", payload.get("partition_count", 0) or payload.get("column_count", 0)))
    return {
        "available": available,
        "partition_count": int(payload.get("partition_count") or 0),
        "column_count": int(payload.get("column_count") or 0),
        "row_count": int(payload.get("row_count") or 0),
        "columns": list(payload.get("columns") or ()),
    }


def _ohlcvt_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "available": bool(not frame.empty),
        "row_count": int(frame.shape[0]),
        "close_available": bool("close" in frame.columns and pd.to_numeric(frame["close"], errors="coerce").notna().any()),
        "volume_available": bool("volume" in frame.columns and pd.to_numeric(frame["volume"], errors="coerce").notna().any()),
        "trades_available": bool("trades" in frame.columns and pd.to_numeric(frame["trades"], errors="coerce").notna().any()),
    }


def _status_from_diagnostics(
    *,
    coverage: Mapping[str, Any],
    movement: Mapping[str, Any],
    activity: Mapping[str, Any],
    risk: Mapping[str, Any],
    reliability: Mapping[str, Any],
    ohlcvt_summary: Mapping[str, Any],
    scalar_summary: Mapping[str, Any],
    config: RegimeFeatureEligibilityConfig,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    row_count = int(coverage.get("row_count") or 0)
    if row_count <= 0 and not bool(scalar_summary.get("available")):
        return ELIGIBILITY_STATUS_INSUFFICIENT_DATA, ("missing_ohlcvt_and_scalar_features",)
    if row_count < int(config.min_observed_rows_hint):
        reasons.append("observed_rows_below_hint")
    if float(coverage.get("coverage_ratio") or 0.0) < float(config.sparse_coverage_ratio_hint):
        reasons.append("coverage_below_sparse_hint")
    if float(reliability.get("high_gap_ratio") or 0.0) > float(config.interval_gap_ratio_hint):
        reasons.append("high_gap_ratio")
    if bool(reliability.get("interval_unreliable_hint")):
        reasons.append("interval_reliability_hint")
    if float(movement.get("zero_log_return_share") or 0.0) >= float(config.flat_zero_return_share_hint) or float(risk.get("flat_or_pegged_risk") or 0.0) >= 0.75:
        reasons.append("flat_or_pegged_risk")
    if float(activity.get("activity_coverage_score") or 0.0) < float(config.low_activity_coverage_hint):
        reasons.append("low_activity_coverage")
    if not bool(ohlcvt_summary.get("available")) and bool(scalar_summary.get("available")):
        reasons.append("scalar_only_no_ohlcvt")

    unique = tuple(dict.fromkeys(reasons))
    if "observed_rows_below_hint" in unique and "coverage_below_sparse_hint" in unique:
        return ELIGIBILITY_STATUS_INSUFFICIENT_DATA, unique
    if "interval_reliability_hint" in unique or "high_gap_ratio" in unique:
        return ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE, unique
    if "coverage_below_sparse_hint" in unique or "scalar_only_no_ohlcvt" in unique:
        return ELIGIBILITY_STATUS_LIKELY_SPARSE, unique
    if "flat_or_pegged_risk" in unique:
        return ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED, unique
    if "low_activity_coverage" in unique:
        return ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY, unique
    if unique:
        return ELIGIBILITY_STATUS_NEEDS_REVIEW, unique
    return ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED, ("observed_descriptive_inputs_available",)


def _finite_ratio(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(np.isfinite(pd.to_numeric(series, errors="coerce")).mean())


def _share(values: Any) -> float:
    if values is None or len(values) == 0:
        return 0.0
    return float(np.asarray(values, dtype=bool).mean())


def _clamp01(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(0.0, min(1.0, val))


def _coverage(value: Any, *, field_name: str) -> float:
    val = float(value)
    if val < 0.0 or val > 1.0:
        raise ValueError(f"Regime Feature eligibility {field_name} must be within [0, 1]")
    return val


def _status(value: object) -> str:
    text = _text(value, field_name="status")
    if text not in REGIME_FEATURE_ELIGIBILITY_STATUSES:
        valid = ", ".join(REGIME_FEATURE_ELIGIBILITY_STATUSES)
        raise ValueError(f"Unsupported Regime Feature eligibility status {text!r}; expected one of: {valid}")
    return text


def _string_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature eligibility {field_name} must be a sequence of strings")
    return tuple(_text(value, field_name=field_name) for value in values)


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature eligibility {field_name} must be non-empty")
    return text


__all__ = [
    "ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED",
    "ELIGIBILITY_STATUS_INSUFFICIENT_DATA",
    "ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED",
    "ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE",
    "ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY",
    "ELIGIBILITY_STATUS_LIKELY_SPARSE",
    "ELIGIBILITY_STATUS_NEEDS_REVIEW",
    "REGIME_FEATURE_ELIGIBILITY_STATUSES",
    "RegimeFeatureEligibilityConfig",
    "RegimeFeatureEligibilityRow",
    "RegimeFeatureEligibilitySnapshot",
    "RegimeFeatureEligibilityStatus",
    "build_regime_feature_eligibility_row",
    "build_regime_feature_eligibility_snapshot",
]
