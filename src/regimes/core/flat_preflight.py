from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.core.artifacts import safe_path_part, write_json
from src.regimes.core.foundation_contracts import REGIME_LAYER_AXES, REGIME_STUDY_BANDS
from src.regimes.core.pathway_artifacts import require_pathway_diagnostics_root


REGIME_FLAT_PREFLIGHT_SCHEMA_VERSION = 1
FLAT_PREFLIGHT_ARTIFACT_KIND = "regime_flat_pegged_preflight"

FLAT_STATUS_ACTIVE = "active"
FLAT_STATUS_VALID_SINGLE_STATE = "valid_flat_single_state_candidate"
FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE = "near_flat_needs_more_evidence"
FLAT_STATUS_AXIS_NOT_CLUSTERABLE = "axis_not_clusterable_candidate"
FLAT_STATUS_INSUFFICIENT_DATA = "insufficient_data"
FLAT_STATUS_UNKNOWN = "unknown"
FLAT_PREFLIGHT_STATUSES: tuple[str, ...] = (
    FLAT_STATUS_ACTIVE,
    FLAT_STATUS_VALID_SINGLE_STATE,
    FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE,
    FLAT_STATUS_AXIS_NOT_CLUSTERABLE,
    FLAT_STATUS_INSUFFICIENT_DATA,
    FLAT_STATUS_UNKNOWN,
)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, BaseException):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Regime flat preflight {field_name} must be non-empty")
    return text


def _require_member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    token = _normalize_token(value, field_name=field_name)
    if token not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime flat preflight {field_name} {token!r}; expected one of: {valid}")
    return token


def _auto_movement_columns(columns: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if (
            "return" in lowered
            or "log_ret" in lowered
            or "_d_close_" in lowered
            or lowered.endswith("_roc_14")
            or lowered.endswith("_mom_14")
            or "true_range" in lowered
            or "range_" in lowered
            or lowered.endswith("_prr")
        ):
            out.append(str(column))
    return tuple(dict.fromkeys(out))


def _auto_activity_columns(columns: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if (
            "volume" in lowered
            or "trades" in lowered
            or "activity" in lowered
            or "trade_intensity" in lowered
            or "avg_trade_size" in lowered
            or lowered.endswith("_vroc_14")
        ):
            out.append(str(column))
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class FlatPeggedPreflightPolicy:
    min_rows: int = 16
    min_unique_rows: int = 4
    min_effective_diversity_share: float = 0.2
    near_zero_return_abs_threshold: float = 1e-6
    near_zero_movement_fraction_threshold: float = 0.98
    near_flat_more_evidence_fraction_threshold: float = 0.9
    low_activity_fraction_threshold: float = 0.98
    near_constant_variance_threshold: float = 1e-12
    duplicate_row_fraction_threshold: float = 0.95
    extreme_fragmentation_state_share_threshold: float = 0.35
    extreme_fragmentation_singleton_share_threshold: float = 0.25
    mostly_noise_threshold: float = 0.8
    tiny_state_threshold: int = 2
    schema_version: int = REGIME_FLAT_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.min_rows) < 1:
            raise ValueError("Regime flat preflight min_rows must be positive")
        if int(self.min_unique_rows) < 1:
            raise ValueError("Regime flat preflight min_unique_rows must be positive")
        if int(self.tiny_state_threshold) < 1:
            raise ValueError("Regime flat preflight tiny_state_threshold must be positive")
        for name in (
            "min_effective_diversity_share",
            "near_zero_movement_fraction_threshold",
            "near_flat_more_evidence_fraction_threshold",
            "low_activity_fraction_threshold",
            "duplicate_row_fraction_threshold",
            "extreme_fragmentation_state_share_threshold",
            "extreme_fragmentation_singleton_share_threshold",
            "mostly_noise_threshold",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Regime flat preflight {name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        for name in ("near_zero_return_abs_threshold", "near_constant_variance_threshold"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"Regime flat preflight {name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "min_rows", int(self.min_rows))
        object.__setattr__(self, "min_unique_rows", int(self.min_unique_rows))
        object.__setattr__(self, "tiny_state_threshold", int(self.tiny_state_threshold))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "min_rows": int(self.min_rows),
            "min_unique_rows": int(self.min_unique_rows),
            "min_effective_diversity_share": float(self.min_effective_diversity_share),
            "near_zero_return_abs_threshold": float(self.near_zero_return_abs_threshold),
            "near_zero_movement_fraction_threshold": float(self.near_zero_movement_fraction_threshold),
            "near_flat_more_evidence_fraction_threshold": float(self.near_flat_more_evidence_fraction_threshold),
            "low_activity_fraction_threshold": float(self.low_activity_fraction_threshold),
            "near_constant_variance_threshold": float(self.near_constant_variance_threshold),
            "duplicate_row_fraction_threshold": float(self.duplicate_row_fraction_threshold),
            "extreme_fragmentation_state_share_threshold": float(self.extreme_fragmentation_state_share_threshold),
            "extreme_fragmentation_singleton_share_threshold": float(self.extreme_fragmentation_singleton_share_threshold),
            "mostly_noise_threshold": float(self.mostly_noise_threshold),
            "tiny_state_threshold": int(self.tiny_state_threshold),
        }


@dataclass(frozen=True)
class FlatPeggedPreflightResult:
    asset: str
    layer: str
    axis: str
    band: str
    status: str
    confidence_score: float
    diagnostics: Mapping[str, Any]
    policy_decision: Mapping[str, Any]
    policy: FlatPeggedPreflightPolicy
    warnings: tuple[str, ...] = ()
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FLAT_PREFLIGHT_SCHEMA_VERSION
    artifact_kind: str = FLAT_PREFLIGHT_ARTIFACT_KIND

    def __post_init__(self) -> None:
        status = _require_member(self.status, FLAT_PREFLIGHT_STATUSES, field_name="status")
        confidence = float(self.confidence_score)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Regime flat preflight confidence_score must be between 0 and 1")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "confidence_score", confidence)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "asset": self.asset,
            "layer": self.layer,
            "axis": self.axis,
            "band": self.band,
            "status": self.status,
            "confidence_score": float(self.confidence_score),
            "diagnostics": _jsonable(dict(self.diagnostics)),
            "policy_decision": _jsonable(dict(self.policy_decision)),
            "warnings": list(self.warnings),
            "source_metadata": _jsonable(dict(self.source_metadata)),
            "policy": self.policy.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            numeric[str(column)] = pd.to_numeric(frame[str(column)], errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _variance_diagnostics(numeric: pd.DataFrame, policy: FlatPeggedPreflightPolicy) -> dict[str, Any]:
    if numeric.empty or numeric.shape[1] == 0:
        return {
            "feature_count": int(numeric.shape[1]),
            "near_constant_feature_count": int(numeric.shape[1]),
            "zero_variance_feature_count": int(numeric.shape[1]),
            "min_variance": None,
            "median_variance": None,
            "max_variance": None,
            "per_feature_variance": {},
            "low_variance_flag": True,
        }
    variances = numeric.var(ddof=0).replace([np.inf, -np.inf], np.nan)
    finite = variances.dropna()
    near_constant = finite <= float(policy.near_constant_variance_threshold)
    return {
        "feature_count": int(numeric.shape[1]),
        "near_constant_feature_count": int(near_constant.sum()) + int(variances.isna().sum()),
        "zero_variance_feature_count": int((finite == 0.0).sum()) + int(variances.isna().sum()),
        "min_variance": None if finite.empty else float(finite.min()),
        "median_variance": None if finite.empty else float(finite.median()),
        "max_variance": None if finite.empty else float(finite.max()),
        "per_feature_variance": {str(key): float(value) for key, value in finite.items()},
        "low_variance_flag": bool(finite.empty or int(near_constant.sum()) == len(finite)),
    }


def _movement_diagnostics(
    numeric: pd.DataFrame,
    movement_columns: Sequence[str],
    policy: FlatPeggedPreflightPolicy,
) -> dict[str, Any]:
    present = [str(column) for column in movement_columns if str(column) in numeric.columns]
    if not present or numeric.empty:
        return {
            "movement_columns": present,
            "movement_feature_count": int(len(present)),
            "near_zero_movement_fraction": 1.0 if not present else None,
            "zero_movement_fraction": 1.0 if not present else None,
            "median_abs_movement": None,
            "max_abs_movement": None,
            "per_feature_near_zero_fraction": {},
            "near_flat_behavior_flag": True if not present else None,
        }
    movement = numeric[present].abs()
    row_near_zero = (movement <= float(policy.near_zero_return_abs_threshold)).all(axis=1)
    row_zero = (movement <= 0.0).all(axis=1)
    stacked = movement.to_numpy(dtype=float).ravel()
    stacked = stacked[np.isfinite(stacked)]
    near_zero_fraction = float(row_near_zero.mean()) if len(row_near_zero) else None
    return {
        "movement_columns": present,
        "movement_feature_count": int(len(present)),
        "near_zero_movement_fraction": near_zero_fraction,
        "zero_movement_fraction": float(row_zero.mean()) if len(row_zero) else None,
        "median_abs_movement": None if stacked.size == 0 else float(np.median(stacked)),
        "max_abs_movement": None if stacked.size == 0 else float(np.max(stacked)),
        "per_feature_near_zero_fraction": {
            str(column): float((movement[str(column)] <= float(policy.near_zero_return_abs_threshold)).mean())
            for column in present
        },
        "near_flat_behavior_flag": bool(
            near_zero_fraction is not None and near_zero_fraction >= float(policy.near_zero_movement_fraction_threshold)
        ),
    }


def _activity_diagnostics(
    numeric: pd.DataFrame,
    activity_columns: Sequence[str],
    policy: FlatPeggedPreflightPolicy,
) -> dict[str, Any]:
    present = [str(column) for column in activity_columns if str(column) in numeric.columns]
    if not present or numeric.empty:
        return {
            "activity_columns": present,
            "activity_feature_count": int(len(present)),
            "low_activity_fraction": None,
            "low_activity_flag": False,
        }
    activity = numeric[present].fillna(0.0).abs()
    low_activity_fraction = float((activity <= float(policy.near_zero_return_abs_threshold)).all(axis=1).mean())
    return {
        "activity_columns": present,
        "activity_feature_count": int(len(present)),
        "low_activity_fraction": low_activity_fraction,
        "low_activity_flag": bool(low_activity_fraction >= float(policy.low_activity_fraction_threshold)),
    }


def _diversity_diagnostics(numeric: pd.DataFrame, policy: FlatPeggedPreflightPolicy) -> dict[str, Any]:
    finite = numeric.dropna(axis=0, how="any").copy()
    finite = finite[np.isfinite(finite.to_numpy(dtype=float)).all(axis=1)] if not finite.empty else finite
    finite_rows = int(len(finite))
    if finite_rows == 0:
        return {
            "finite_row_count": 0,
            "unique_row_count": 0,
            "effective_diversity_share": None,
            "duplicate_row_count": 0,
            "duplicate_row_fraction": None,
            "duplicate_heavy_flag": False,
            "low_effective_sample_diversity_flag": True,
        }
    duplicate_mask = finite.duplicated(keep=False)
    unique_rows = int(len(finite.drop_duplicates()))
    diversity_share = float(unique_rows / finite_rows)
    duplicate_fraction = float(duplicate_mask.mean())
    return {
        "finite_row_count": finite_rows,
        "unique_row_count": unique_rows,
        "effective_diversity_share": diversity_share,
        "duplicate_row_count": int(duplicate_mask.sum()),
        "duplicate_row_fraction": duplicate_fraction,
        "duplicate_heavy_flag": bool(duplicate_fraction >= float(policy.duplicate_row_fraction_threshold)),
        "low_effective_sample_diversity_flag": bool(
            unique_rows < int(policy.min_unique_rows)
            or diversity_share < float(policy.min_effective_diversity_share)
        ),
    }


def _is_null_label(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_noise_label(value: object, noise_label: object = -1) -> bool:
    if _is_null_label(value):
        return False
    try:
        return int(value) == int(noise_label)
    except Exception:
        return str(value).strip().lower() in {"noise", str(noise_label)}


def _label_key(value: object) -> str:
    if _is_null_label(value):
        return "<null>"
    text = str(value).strip()
    return text if text else "<empty>"


def _label_diagnostics(
    labels: Sequence[object] | pd.Series | np.ndarray | None,
    policy: FlatPeggedPreflightPolicy,
) -> dict[str, Any]:
    if labels is None:
        return {
            "labels_supplied": False,
            "row_count": 0,
            "effective_state_count": None,
            "state_counts": {},
            "noise_count": None,
            "noise_share": None,
            "all_noise_warning": False,
            "mostly_noise_warning": False,
            "singleton_state_share": None,
            "tiny_state_share": None,
            "state_count_share": None,
            "extreme_cluster_fragmentation_warning": False,
        }
    values = list(pd.Series(list(labels), dtype="object").tolist())
    row_count = int(len(values))
    effective_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    noise_count = 0
    null_count = 0
    for value in values:
        key = _label_key(value)
        raw_counts[key] += 1
        if _is_null_label(value):
            null_count += 1
        elif _is_noise_label(value):
            noise_count += 1
        else:
            effective_counts[key] += 1
    effective_state_count = int(len(effective_counts))
    singleton_rows = sum(int(v) for v in effective_counts.values() if int(v) == 1)
    tiny_rows = sum(int(v) for v in effective_counts.values() if int(v) <= int(policy.tiny_state_threshold))
    noise_share = None if row_count == 0 else float(noise_count / row_count)
    singleton_share = None if row_count == 0 else float(singleton_rows / row_count)
    tiny_share = None if row_count == 0 else float(tiny_rows / row_count)
    state_count_share = None if row_count == 0 else float(effective_state_count / row_count)
    fragmentation = bool(
        row_count > 0
        and (
            (state_count_share is not None and state_count_share >= float(policy.extreme_fragmentation_state_share_threshold))
            or (
                singleton_share is not None
                and singleton_share >= float(policy.extreme_fragmentation_singleton_share_threshold)
            )
        )
    )
    return {
        "labels_supplied": True,
        "row_count": row_count,
        "effective_state_count": effective_state_count,
        "state_counts": {str(k): int(v) for k, v in sorted(effective_counts.items())},
        "raw_state_counts": {str(k): int(v) for k, v in sorted(raw_counts.items())},
        "noise_count": int(noise_count),
        "null_count": int(null_count),
        "noise_share": noise_share,
        "all_noise_warning": bool(row_count > 0 and noise_count == row_count),
        "mostly_noise_warning": bool(noise_share is not None and noise_share >= float(policy.mostly_noise_threshold)),
        "singleton_state_share": singleton_share,
        "tiny_state_share": tiny_share,
        "state_count_share": state_count_share,
        "extreme_cluster_fragmentation_warning": fragmentation,
    }


def _confidence_for_status(status: str, diagnostics: Mapping[str, Any]) -> float:
    if status == FLAT_STATUS_ACTIVE:
        return 0.75
    if status == FLAT_STATUS_VALID_SINGLE_STATE:
        movement = diagnostics.get("movement_behavior", {})
        fraction = movement.get("near_zero_movement_fraction")
        return min(0.99, max(0.7, float(fraction or 0.0)))
    if status == FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE:
        movement = diagnostics.get("movement_behavior", {})
        fraction = movement.get("near_zero_movement_fraction")
        return min(0.85, max(0.45, float(fraction or 0.0)))
    if status == FLAT_STATUS_AXIS_NOT_CLUSTERABLE:
        return 0.8
    if status == FLAT_STATUS_INSUFFICIENT_DATA:
        return 0.65
    return 0.25


def run_flat_pegged_preflight(
    frame: pd.DataFrame,
    *,
    asset: str,
    axis: str,
    band: str,
    feature_columns: Sequence[str],
    layer: str = "asset_state",
    movement_columns: Sequence[str] | None = None,
    activity_columns: Sequence[str] | None = None,
    labels: Sequence[object] | pd.Series | np.ndarray | None = None,
    policy: FlatPeggedPreflightPolicy | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> FlatPeggedPreflightResult:
    layer_token = _require_member(layer, ("asset_state",), field_name="layer")
    axis_token = _require_member(axis, REGIME_LAYER_AXES[layer_token], field_name=f"{layer_token} axis")
    band_token = _require_member(band, REGIME_STUDY_BANDS, field_name="band")
    cfg = policy or FlatPeggedPreflightPolicy()
    columns = tuple(dict.fromkeys(str(column) for column in feature_columns if str(column).strip()))
    present_columns = tuple(column for column in columns if column in frame.columns)
    numeric = _numeric_frame(frame, present_columns)
    movement_cols = tuple(movement_columns) if movement_columns is not None else _auto_movement_columns(present_columns)
    activity_cols = tuple(activity_columns) if activity_columns is not None else _auto_activity_columns(present_columns)
    row_count = int(len(frame))
    missing_feature_columns = tuple(column for column in columns if column not in frame.columns)
    missing_fraction = None
    if row_count > 0 and present_columns:
        missing_fraction = float(numeric.isna().to_numpy().mean())
    variance = _variance_diagnostics(numeric, cfg)
    movement = _movement_diagnostics(numeric, movement_cols, cfg)
    activity = _activity_diagnostics(numeric, activity_cols, cfg)
    diversity = _diversity_diagnostics(numeric, cfg)
    label_diag = _label_diagnostics(labels, cfg)
    near_flat_fraction = movement.get("near_zero_movement_fraction")
    near_flat_strong = bool(
        near_flat_fraction is not None and float(near_flat_fraction) >= float(cfg.near_zero_movement_fraction_threshold)
    )
    near_flat_partial = bool(
        near_flat_fraction is not None and float(near_flat_fraction) >= float(cfg.near_flat_more_evidence_fraction_threshold)
    )
    low_variance = bool(variance.get("low_variance_flag"))
    low_activity = bool(activity.get("low_activity_flag"))
    low_diversity = bool(diversity.get("low_effective_sample_diversity_flag"))
    duplicate_heavy = bool(diversity.get("duplicate_heavy_flag"))
    fragmentation = bool(label_diag.get("extreme_cluster_fragmentation_warning"))
    all_noise = bool(label_diag.get("all_noise_warning"))
    mostly_noise = bool(label_diag.get("mostly_noise_warning"))
    effective_state_count = label_diag.get("effective_state_count")
    single_state_labels = bool(label_diag.get("labels_supplied") and effective_state_count == 1 and not all_noise)
    fake_fragmentation_warning = bool((near_flat_strong or near_flat_partial or low_variance) and fragmentation)
    warnings: list[str] = []
    if missing_feature_columns:
        warnings.append("missing_feature_columns")
    if fragmentation:
        warnings.append("extreme_cluster_fragmentation")
    if fake_fragmentation_warning:
        warnings.append("near_flat_fake_fragmentation")
    if all_noise:
        warnings.append("all_noise")
    elif mostly_noise:
        warnings.append("mostly_noise")
    if duplicate_heavy:
        warnings.append("duplicate_heavy_rows")
    if row_count < int(cfg.min_rows) or int(diversity.get("finite_row_count", 0) or 0) < int(cfg.min_rows):
        status = FLAT_STATUS_INSUFFICIENT_DATA
    elif fake_fragmentation_warning:
        status = FLAT_STATUS_AXIS_NOT_CLUSTERABLE
    elif near_flat_strong and (not label_diag.get("labels_supplied") or single_state_labels or effective_state_count in {None, 0}):
        status = FLAT_STATUS_VALID_SINGLE_STATE
    elif (low_diversity or duplicate_heavy) and near_flat_partial:
        status = FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE
    elif near_flat_partial or low_variance or low_activity:
        status = FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE
    elif present_columns and row_count >= int(cfg.min_rows):
        status = FLAT_STATUS_ACTIVE
    else:
        status = FLAT_STATUS_UNKNOWN
    valid_single_state_candidate = bool(status == FLAT_STATUS_VALID_SINGLE_STATE or (near_flat_strong and not fake_fragmentation_warning))
    pegged_candidate = bool(near_flat_strong and (low_variance or duplicate_heavy or low_activity or single_state_labels))
    diagnostics = {
        "row_count": row_count,
        "feature_columns_requested": list(columns),
        "feature_columns_present": list(present_columns),
        "feature_columns_missing": list(missing_feature_columns),
        "missing_fraction": missing_fraction,
        "variance_behavior": variance,
        "movement_behavior": movement,
        "activity_behavior": activity,
        "sample_diversity": diversity,
        "label_fragmentation": label_diag,
        "valid_single_state_candidate": valid_single_state_candidate,
        "pegged_stable_like_candidate": pegged_candidate,
        "near_flat_fake_fragmentation_warning": fake_fragmentation_warning,
        "all_noise_warning": all_noise,
        "mostly_noise_warning": mostly_noise,
    }
    policy_decision = {
        "status": status,
        "allow_single_state_outcome": bool(
            status in {FLAT_STATUS_VALID_SINGLE_STATE, FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE}
            or valid_single_state_candidate
        ),
        "do_not_penalize_missing_directional_states": bool(
            status in {FLAT_STATUS_VALID_SINGLE_STATE, FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE}
        ),
        "penalize_fake_fragmentation": fake_fragmentation_warning,
        "fake_fragmentation_penalty_weight": 1.0 if fake_fragmentation_warning else 0.0,
        "permanent_exclusion": False,
        "exclude_from_all_future_studies": False,
        "recommended_action": {
            FLAT_STATUS_ACTIVE: "include_as_active_candidate",
            FLAT_STATUS_VALID_SINGLE_STATE: "allow_flat_single_state_candidate",
            FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE: "allow_single_state_and_collect_more_evidence",
            FLAT_STATUS_AXIS_NOT_CLUSTERABLE: "candidate_axis_not_clusterable_review",
            FLAT_STATUS_INSUFFICIENT_DATA: "insufficient_data_no_permanent_decision",
            FLAT_STATUS_UNKNOWN: "unknown_manual_review",
        }[status],
        "production_outputs_written": False,
        "production_label_change": False,
    }
    return FlatPeggedPreflightResult(
        asset=str(asset),
        layer=layer_token,
        axis=axis_token,
        band=band_token,
        status=status,
        confidence_score=_confidence_for_status(status, diagnostics),
        diagnostics=diagnostics,
        policy_decision=policy_decision,
        policy=cfg,
        warnings=tuple(warnings),
        source_metadata=dict(source_metadata or {}),
    )


def _diagnostic_path(diagnostics_root: Path, result: FlatPeggedPreflightResult, run_id: str, suffix: str) -> Path:
    stem = "__".join(
        (
            safe_path_part(run_id, context="Regime flat preflight run_id"),
            safe_path_part(result.asset, context="Regime flat preflight asset"),
            safe_path_part(result.axis, context="Regime flat preflight axis"),
            safe_path_part(result.band, context="Regime flat preflight band"),
        )
    )
    return Path(diagnostics_root) / "flat_pegged_preflight" / safe_path_part(run_id) / f"{stem}.{suffix}"


def _markdown_summary(result: FlatPeggedPreflightResult) -> str:
    payload = result.as_dict()
    diagnostics = payload["diagnostics"]
    policy = payload["policy_decision"]
    movement = diagnostics["movement_behavior"]
    diversity = diagnostics["sample_diversity"]
    labels = diagnostics["label_fragmentation"]
    lines = [
        f"# Regime Flat/Pegged Preflight {payload['asset']}",
        "",
        f"- layer/axis/band: {payload['layer']} / {payload['axis']} / {payload['band']}",
        f"- status: {payload['status']}",
        f"- confidence_score: {payload['confidence_score']}",
        f"- recommended_action: {policy['recommended_action']}",
        f"- allow_single_state_outcome: {policy['allow_single_state_outcome']}",
        f"- penalize_fake_fragmentation: {policy['penalize_fake_fragmentation']}",
        f"- near_zero_movement_fraction: {movement.get('near_zero_movement_fraction')}",
        f"- effective_diversity_share: {diversity.get('effective_diversity_share')}",
        f"- effective_state_count: {labels.get('effective_state_count')}",
        f"- all_noise_warning: {payload['diagnostics']['all_noise_warning']}",
        f"- production_outputs_written: {policy['production_outputs_written']}",
        "",
        "This diagnostic is policy evidence only; it does not create a permanent exclusion list or write production labels.",
        "",
    ]
    return "\n".join(lines)


def _write_markdown(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def write_flat_pegged_preflight_diagnostics(
    result: FlatPeggedPreflightResult,
    *,
    diagnostics_root: Path,
    run_id: str,
    project_root: Path | None = None,
) -> dict[str, str]:
    policy = require_pathway_diagnostics_root(Path(diagnostics_root), project_root=project_root)
    payload = result.as_dict()
    payload["diagnostics_root_policy"] = policy.as_dict()
    json_path = _diagnostic_path(Path(diagnostics_root), result, run_id, "json")
    md_path = _diagnostic_path(Path(diagnostics_root), result, run_id, "md")
    write_json(json_path, payload, write_kind="Regime flat/pegged preflight diagnostic")
    _write_markdown(md_path, _markdown_summary(result))
    return {"json": str(json_path), "markdown": str(md_path)}


__all__ = [
    "FLAT_PREFLIGHT_ARTIFACT_KIND",
    "FLAT_PREFLIGHT_STATUSES",
    "FLAT_STATUS_ACTIVE",
    "FLAT_STATUS_AXIS_NOT_CLUSTERABLE",
    "FLAT_STATUS_INSUFFICIENT_DATA",
    "FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE",
    "FLAT_STATUS_UNKNOWN",
    "FLAT_STATUS_VALID_SINGLE_STATE",
    "REGIME_FLAT_PREFLIGHT_SCHEMA_VERSION",
    "FlatPeggedPreflightPolicy",
    "FlatPeggedPreflightResult",
    "run_flat_pegged_preflight",
    "write_flat_pegged_preflight_diagnostics",
]
