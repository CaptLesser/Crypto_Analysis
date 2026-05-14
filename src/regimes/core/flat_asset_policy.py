from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    RegimeAxis,
    RegimeBand,
    RegimeLayer,
    normalize_enum_value,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
    validate_layer_axis_band,
)
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


REGIME_FLAT_ASSET_POLICY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
FLAT_ASSET_POLICY_ARTIFACT_KIND = "regime_flat_asset_policy_diagnostics"

FLAT_POLICY_STATUS_ACTIVE = "active"
FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE = "valid_flat_single_state_candidate"
FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE = "near_flat_needs_more_evidence"
FLAT_POLICY_STATUS_AXIS_NOT_CLUSTERABLE_CANDIDATE = "axis_not_clusterable_candidate"
FLAT_POLICY_STATUS_INSUFFICIENT_DATA = "insufficient_data"
FLAT_POLICY_STATUS_UNKNOWN = "unknown"
FLAT_ASSET_POLICY_STATUSES: tuple[str, ...] = (
    FLAT_POLICY_STATUS_ACTIVE,
    FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE,
    FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE,
    FLAT_POLICY_STATUS_AXIS_NOT_CLUSTERABLE_CANDIDATE,
    FLAT_POLICY_STATUS_INSUFFICIENT_DATA,
    FLAT_POLICY_STATUS_UNKNOWN,
)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _normalize_status(value: object) -> str:
    text = require_non_empty_string(value, field_name="flat asset policy status").lower()
    if text not in FLAT_ASSET_POLICY_STATUSES:
        valid = ", ".join(FLAT_ASSET_POLICY_STATUSES)
        raise ValueError(f"Unsupported Regime flat asset policy status {text!r}; expected one of: {valid}")
    return text


def _string_tuple(values: Sequence[object] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime flat asset policy {field_name} must be a sequence")
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            numeric[str(column)] = pd.to_numeric(frame[str(column)], errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _auto_movement_columns(columns: Sequence[str]) -> tuple[str, ...]:
    tokens = ("return", "log_ret", "ret_", "movement", "delta", "d_close", "pct", "roc", "drawdown")
    selected = []
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in tokens):
            selected.append(str(column))
    return tuple(dict.fromkeys(selected))


def _is_null_label(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_noise_label(value: object, noise_label: object) -> bool:
    if _is_null_label(value):
        return False
    try:
        return int(value) == int(noise_label)
    except Exception:
        return str(value).strip().lower() in {"noise", str(noise_label).lower()}


def _label_key(value: object) -> str:
    if _is_null_label(value):
        return "<null>"
    text = str(value).strip()
    return text if text else "<empty>"


@dataclass(frozen=True)
class FlatAssetPolicyConfig:
    min_rows: int = 6
    min_unique_rows: int = 3
    min_effective_diversity_share: float = 0.25
    low_variance_threshold: float = 1e-10
    near_zero_movement_abs_threshold: float = 1e-6
    mostly_zero_movement_fraction_threshold: float = 0.98
    near_flat_movement_fraction_threshold: float = 0.90
    duplicate_row_fraction_threshold: float = 0.80
    extreme_fragmentation_state_share_threshold: float = 0.35
    extreme_fragmentation_singleton_share_threshold: float = 0.25
    mostly_noise_threshold: float = 0.80
    tiny_state_threshold: int = 1
    schema_version: int = REGIME_FLAT_ASSET_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        min_rows = int(self.min_rows)
        min_unique_rows = int(self.min_unique_rows)
        tiny_state_threshold = int(self.tiny_state_threshold)
        if min_rows < 1:
            raise ValueError("Regime flat asset policy min_rows must be positive")
        if min_unique_rows < 1:
            raise ValueError("Regime flat asset policy min_unique_rows must be positive")
        if tiny_state_threshold < 1:
            raise ValueError("Regime flat asset policy tiny_state_threshold must be positive")
        for name in (
            "min_effective_diversity_share",
            "mostly_zero_movement_fraction_threshold",
            "near_flat_movement_fraction_threshold",
            "duplicate_row_fraction_threshold",
            "extreme_fragmentation_state_share_threshold",
            "extreme_fragmentation_singleton_share_threshold",
            "mostly_noise_threshold",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Regime flat asset policy {name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        for name in ("low_variance_threshold", "near_zero_movement_abs_threshold"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"Regime flat asset policy {name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "min_rows", min_rows)
        object.__setattr__(self, "min_unique_rows", min_unique_rows)
        object.__setattr__(self, "tiny_state_threshold", tiny_state_threshold)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "min_rows": int(self.min_rows),
            "min_unique_rows": int(self.min_unique_rows),
            "min_effective_diversity_share": float(self.min_effective_diversity_share),
            "low_variance_threshold": float(self.low_variance_threshold),
            "near_zero_movement_abs_threshold": float(self.near_zero_movement_abs_threshold),
            "mostly_zero_movement_fraction_threshold": float(self.mostly_zero_movement_fraction_threshold),
            "near_flat_movement_fraction_threshold": float(self.near_flat_movement_fraction_threshold),
            "duplicate_row_fraction_threshold": float(self.duplicate_row_fraction_threshold),
            "extreme_fragmentation_state_share_threshold": float(self.extreme_fragmentation_state_share_threshold),
            "extreme_fragmentation_singleton_share_threshold": float(self.extreme_fragmentation_singleton_share_threshold),
            "mostly_noise_threshold": float(self.mostly_noise_threshold),
            "tiny_state_threshold": int(self.tiny_state_threshold),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FlatAssetPolicyConfig":
        obj = require_json_object(payload, context="Regime FlatAssetPolicyConfig")
        return cls(**obj)


@dataclass(frozen=True)
class FlatAssetPolicyResult:
    status: str
    valid_single_state_allowed: bool
    fragmentation_penalty_flag: bool
    confidence_score: float
    evidence_summary: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    policy_decision: Mapping[str, Any]
    policy: FlatAssetPolicyConfig | Mapping[str, Any] = field(default_factory=FlatAssetPolicyConfig)
    warnings: Sequence[str] = ()
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FLAT_ASSET_POLICY_SCHEMA_VERSION
    artifact_kind: str = FLAT_ASSET_POLICY_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        status = _normalize_status(self.status)
        confidence = float(self.confidence_score)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Regime flat asset policy confidence_score must be between 0 and 1")
        policy = self.policy if isinstance(self.policy, FlatAssetPolicyConfig) else FlatAssetPolicyConfig.from_dict(self.policy)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "valid_single_state_allowed", bool(self.valid_single_state_allowed))
        object.__setattr__(self, "fragmentation_penalty_flag", bool(self.fragmentation_penalty_flag))
        object.__setattr__(self, "confidence_score", confidence)
        object.__setattr__(self, "evidence_summary", require_json_mapping(self.evidence_summary, field_name="evidence_summary"))
        object.__setattr__(self, "diagnostics", require_json_mapping(self.diagnostics, field_name="diagnostics"))
        object.__setattr__(self, "policy_decision", require_json_mapping(self.policy_decision, field_name="policy_decision"))
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, field_name="warnings"))
        object.__setattr__(self, "source_metadata", require_json_mapping(self.source_metadata, field_name="source_metadata"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "valid_single_state_allowed": bool(self.valid_single_state_allowed),
            "fragmentation_penalty_flag": bool(self.fragmentation_penalty_flag),
            "confidence_score": float(self.confidence_score),
            "evidence_summary": to_jsonable(self.evidence_summary),
            "diagnostics": to_jsonable(self.diagnostics),
            "policy_decision": to_jsonable(self.policy_decision),
            "warnings": list(self.warnings),
            "source_metadata": to_jsonable(self.source_metadata),
            "policy": self.policy.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FlatAssetPolicyResult":
        obj = require_json_object(payload, context="Regime FlatAssetPolicyResult")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FLAT_ASSET_POLICY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", FLAT_ASSET_POLICY_ARTIFACT_KIND),
            status=obj["status"],
            valid_single_state_allowed=obj["valid_single_state_allowed"],
            fragmentation_penalty_flag=obj["fragmentation_penalty_flag"],
            confidence_score=obj["confidence_score"],
            evidence_summary=obj["evidence_summary"],
            diagnostics=obj["diagnostics"],
            policy_decision=obj["policy_decision"],
            policy=obj.get("policy", {}),
            warnings=obj.get("warnings", ()),
            source_metadata=obj.get("source_metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "FlatAssetPolicyResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FlatAssetPolicyResult JSON"))


def _variance_diagnostics(numeric: pd.DataFrame, policy: FlatAssetPolicyConfig) -> dict[str, Any]:
    feature_count = int(numeric.shape[1])
    if feature_count == 0:
        return {
            "feature_count": 0,
            "low_variance_feature_count": 0,
            "low_variance_feature_share": None,
            "min_variance": None,
            "median_variance": None,
            "max_variance": None,
            "per_feature_variance": {},
            "low_variance_behavior_flag": None,
        }
    variances = numeric.var(ddof=0).replace([np.inf, -np.inf], np.nan)
    finite = variances.dropna()
    low_count = int((finite <= float(policy.low_variance_threshold)).sum()) + int(variances.isna().sum())
    finite_values = finite.to_numpy(dtype=float)
    return {
        "feature_count": feature_count,
        "low_variance_feature_count": low_count,
        "low_variance_feature_share": float(low_count / feature_count) if feature_count else None,
        "min_variance": None if finite.empty else float(np.min(finite_values)),
        "median_variance": None if finite.empty else float(np.median(finite_values)),
        "max_variance": None if finite.empty else float(np.max(finite_values)),
        "per_feature_variance": {str(key): float(value) for key, value in finite.items()},
        "low_variance_behavior_flag": bool(feature_count > 0 and low_count == feature_count),
    }


def _movement_diagnostics(
    numeric: pd.DataFrame,
    movement_columns: Sequence[str],
    policy: FlatAssetPolicyConfig,
) -> dict[str, Any]:
    present = tuple(column for column in movement_columns if column in numeric.columns)
    if not present:
        return {
            "movement_columns": [],
            "near_zero_movement_fraction": None,
            "zero_movement_fraction": None,
            "mostly_zero_movement_flag": None,
            "near_flat_movement_flag": None,
            "median_abs_movement": None,
            "max_abs_movement": None,
            "per_feature_near_zero_fraction": {},
        }
    movement = numeric.loc[:, list(present)].abs()
    finite = movement.dropna(axis=0, how="any")
    if finite.empty:
        return {
            "movement_columns": list(present),
            "near_zero_movement_fraction": None,
            "zero_movement_fraction": None,
            "mostly_zero_movement_flag": None,
            "near_flat_movement_flag": None,
            "median_abs_movement": None,
            "max_abs_movement": None,
            "per_feature_near_zero_fraction": {},
        }
    row_near_zero = (finite <= float(policy.near_zero_movement_abs_threshold)).all(axis=1)
    row_zero = (finite <= 0.0).all(axis=1)
    values = finite.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    near_zero_fraction = float(row_near_zero.mean())
    zero_fraction = float(row_zero.mean())
    return {
        "movement_columns": list(present),
        "near_zero_movement_fraction": near_zero_fraction,
        "zero_movement_fraction": zero_fraction,
        "mostly_zero_movement_flag": bool(near_zero_fraction >= float(policy.mostly_zero_movement_fraction_threshold)),
        "near_flat_movement_flag": bool(near_zero_fraction >= float(policy.near_flat_movement_fraction_threshold)),
        "median_abs_movement": None if values.size == 0 else float(np.median(values)),
        "max_abs_movement": None if values.size == 0 else float(np.max(values)),
        "per_feature_near_zero_fraction": {
            str(column): float((finite[str(column)] <= float(policy.near_zero_movement_abs_threshold)).mean())
            for column in present
        },
    }


def _sample_diversity_diagnostics(numeric: pd.DataFrame, policy: FlatAssetPolicyConfig) -> dict[str, Any]:
    finite = numeric.dropna(axis=0, how="any")
    if not finite.empty:
        finite = finite[np.isfinite(finite.to_numpy(dtype=float)).all(axis=1)]
    finite_row_count = int(len(finite))
    if finite_row_count == 0:
        return {
            "finite_row_count": 0,
            "unique_row_count": 0,
            "effective_sample_diversity_share": None,
            "duplicate_row_count": 0,
            "duplicate_row_fraction": None,
            "duplicate_heavy_rows_flag": False,
            "low_effective_sample_diversity_flag": True,
        }
    duplicated = finite.duplicated(keep=False)
    unique_rows = int(len(finite.drop_duplicates()))
    diversity_share = float(unique_rows / finite_row_count)
    duplicate_fraction = float(duplicated.mean())
    return {
        "finite_row_count": finite_row_count,
        "unique_row_count": unique_rows,
        "effective_sample_diversity_share": diversity_share,
        "duplicate_row_count": int(duplicated.sum()),
        "duplicate_row_fraction": duplicate_fraction,
        "duplicate_heavy_rows_flag": bool(duplicate_fraction >= float(policy.duplicate_row_fraction_threshold)),
        "low_effective_sample_diversity_flag": bool(
            unique_rows < int(policy.min_unique_rows)
            or diversity_share < float(policy.min_effective_diversity_share)
        ),
    }


def _label_fragmentation_diagnostics(
    labels: Sequence[object] | pd.Series | np.ndarray | None,
    policy: FlatAssetPolicyConfig,
    *,
    noise_label: object,
) -> dict[str, Any]:
    if labels is None:
        return {
            "labels_supplied": False,
            "row_count": 0,
            "effective_state_count": None,
            "state_counts": {},
            "raw_state_counts": {},
            "noise_count": None,
            "noise_share": None,
            "all_noise_flag": False,
            "mostly_noise_flag": False,
            "singleton_state_share": None,
            "tiny_state_share": None,
            "state_count_share": None,
            "extreme_fragmentation_flag": False,
        }
    values = list(pd.Series(list(labels), dtype="object").tolist())
    row_count = int(len(values))
    raw_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    noise_count = 0
    null_count = 0
    for value in values:
        raw_counts[_label_key(value)] += 1
        if _is_null_label(value):
            null_count += 1
        elif _is_noise_label(value, noise_label):
            noise_count += 1
        else:
            effective_counts[_label_key(value)] += 1
    effective_state_count = int(len(effective_counts))
    singleton_rows = int(sum(count for count in effective_counts.values() if int(count) == 1))
    tiny_rows = int(sum(count for count in effective_counts.values() if int(count) <= int(policy.tiny_state_threshold)))
    noise_share = float(noise_count / row_count) if row_count else None
    singleton_share = float(singleton_rows / row_count) if row_count else None
    tiny_share = float(tiny_rows / row_count) if row_count else None
    state_count_share = float(effective_state_count / row_count) if row_count else None
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
        "state_counts": {str(key): int(value) for key, value in sorted(effective_counts.items())},
        "raw_state_counts": {str(key): int(value) for key, value in sorted(raw_counts.items())},
        "noise_count": int(noise_count),
        "null_count": int(null_count),
        "noise_share": noise_share,
        "all_noise_flag": bool(row_count > 0 and noise_count == row_count),
        "mostly_noise_flag": bool(noise_share is not None and noise_share >= float(policy.mostly_noise_threshold)),
        "singleton_state_share": singleton_share,
        "tiny_state_share": tiny_share,
        "state_count_share": state_count_share,
        "extreme_fragmentation_flag": fragmentation,
    }


def _confidence(status: str, evidence_flags: Sequence[str], movement_fraction: float | None) -> float:
    if status == FLAT_POLICY_STATUS_ACTIVE:
        return 0.75
    if status == FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE:
        return min(0.99, max(0.75, float(movement_fraction or 0.0), 0.20 * len(evidence_flags)))
    if status == FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE:
        return min(0.85, max(0.45, float(movement_fraction or 0.0), 0.15 * len(evidence_flags)))
    if status == FLAT_POLICY_STATUS_AXIS_NOT_CLUSTERABLE_CANDIDATE:
        return min(0.95, max(0.65, 0.16 * len(evidence_flags)))
    if status == FLAT_POLICY_STATUS_INSUFFICIENT_DATA:
        return 0.65
    return 0.25


def evaluate_flat_asset_policy(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    labels: Sequence[object] | pd.Series | np.ndarray | None = None,
    movement_columns: Sequence[str] | None = None,
    layer: str | RegimeLayer = RegimeLayer.ASSET_STATE.value,
    axis: str | RegimeAxis = RegimeAxis.TREND.value,
    band: str | RegimeBand = RegimeBand.MICRO.value,
    noise_label: object = -1,
    policy: FlatAssetPolicyConfig | Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> FlatAssetPolicyResult:
    layer_value = normalize_enum_value(layer, RegimeLayer, field_name="layer")
    axis_value = normalize_enum_value(axis, RegimeAxis, field_name="axis")
    band_value = normalize_enum_value(band, RegimeBand, field_name="band")
    validate_layer_axis_band(layer=layer_value, axis=axis_value, band=band_value)
    cfg = policy if isinstance(policy, FlatAssetPolicyConfig) else FlatAssetPolicyConfig.from_dict(policy) if policy else FlatAssetPolicyConfig()
    columns = _string_tuple(feature_columns, field_name="feature_columns")
    present_columns = tuple(column for column in columns if column in frame.columns)
    numeric = _numeric_frame(frame, present_columns)
    movement = tuple(movement_columns) if movement_columns is not None else _auto_movement_columns(present_columns)
    row_count = int(len(frame))
    missing_columns = tuple(column for column in columns if column not in frame.columns)
    variance = _variance_diagnostics(numeric, cfg)
    movement_diag = _movement_diagnostics(numeric, movement, cfg)
    diversity = _sample_diversity_diagnostics(numeric, cfg)
    label_diag = _label_fragmentation_diagnostics(labels, cfg, noise_label=noise_label)

    low_variance = bool(variance.get("low_variance_behavior_flag"))
    mostly_zero = bool(movement_diag.get("mostly_zero_movement_flag"))
    near_flat_movement = bool(movement_diag.get("near_flat_movement_flag"))
    duplicate_heavy = bool(diversity.get("duplicate_heavy_rows_flag"))
    low_diversity = bool(diversity.get("low_effective_sample_diversity_flag"))
    fragmentation = bool(label_diag.get("extreme_fragmentation_flag"))
    all_noise = bool(label_diag.get("all_noise_flag"))
    mostly_noise = bool(label_diag.get("mostly_noise_flag"))
    effective_state_count = label_diag.get("effective_state_count")
    labels_supplied = bool(label_diag.get("labels_supplied"))
    single_state_labels = bool(labels_supplied and effective_state_count == 1 and not all_noise)
    enough_data = bool(row_count >= int(cfg.min_rows) and int(diversity.get("finite_row_count") or 0) >= int(cfg.min_rows))
    feature_evidence_available = bool(present_columns and numeric.shape[1] > 0)
    near_flat = bool(near_flat_movement or low_variance or (duplicate_heavy and low_diversity))
    strong_flat = bool(mostly_zero or (low_variance and duplicate_heavy))
    fragmentation_penalty = bool((near_flat or strong_flat) and fragmentation)

    evidence_flags: list[str] = []
    for name, flag in (
        ("low_variance_behavior", low_variance),
        ("mostly_zero_movement", mostly_zero),
        ("near_flat_movement", near_flat_movement),
        ("duplicate_heavy_rows", duplicate_heavy),
        ("low_effective_sample_diversity", low_diversity),
        ("extreme_fragmentation", fragmentation),
        ("all_noise", all_noise),
        ("mostly_noise", mostly_noise),
        ("single_state_labels", single_state_labels),
    ):
        if flag:
            evidence_flags.append(name)

    warnings: list[str] = []
    if missing_columns:
        warnings.append("missing_feature_columns")
    if fragmentation:
        warnings.append("extreme_fragmentation")
    if fragmentation_penalty:
        warnings.append("fragmentation_penalty")
    if all_noise:
        warnings.append("all_noise")
    elif mostly_noise:
        warnings.append("mostly_noise")
    if duplicate_heavy:
        warnings.append("duplicate_heavy_rows")
    if low_diversity:
        warnings.append("low_effective_sample_diversity")

    if not feature_evidence_available:
        status = FLAT_POLICY_STATUS_UNKNOWN
    elif not enough_data:
        status = FLAT_POLICY_STATUS_INSUFFICIENT_DATA
    elif all_noise or fragmentation_penalty:
        status = FLAT_POLICY_STATUS_AXIS_NOT_CLUSTERABLE_CANDIDATE
    elif strong_flat and (not labels_supplied or single_state_labels or effective_state_count in {None, 0}):
        status = FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE
    elif near_flat:
        status = FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE
    else:
        status = FLAT_POLICY_STATUS_ACTIVE

    valid_single_state_allowed = bool(
        status in {
            FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE,
            FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE,
        }
        and not all_noise
        and not fragmentation_penalty
    )
    movement_fraction = _safe_float(movement_diag.get("near_zero_movement_fraction"))
    confidence = _confidence(status, evidence_flags, movement_fraction)
    evidence_summary = {
        "evidence_flags": evidence_flags,
        "evidence_count": int(len(evidence_flags)),
        "near_zero_movement_fraction": movement_fraction,
        "low_variance_feature_share": variance.get("low_variance_feature_share"),
        "duplicate_row_fraction": diversity.get("duplicate_row_fraction"),
        "effective_sample_diversity_share": diversity.get("effective_sample_diversity_share"),
        "effective_state_count": effective_state_count,
        "confidence_score": confidence,
    }
    diagnostics = {
        "row_count": row_count,
        "layer": layer_value,
        "axis": axis_value,
        "band": band_value,
        "feature_columns_requested": list(columns),
        "feature_columns_present": list(present_columns),
        "feature_columns_missing": list(missing_columns),
        "low_variance_behavior": variance,
        "mostly_zero_movement": movement_diag,
        "duplicate_heavy_rows": {
            "duplicate_row_fraction": diversity.get("duplicate_row_fraction"),
            "duplicate_row_count": diversity.get("duplicate_row_count"),
            "duplicate_heavy_rows_flag": duplicate_heavy,
        },
        "effective_sample_diversity": diversity,
        "label_fragmentation": label_diag,
        "near_flat_behavior_flag": near_flat,
        "strong_flat_behavior_flag": strong_flat,
        "all_noise_behavior_flag": all_noise,
    }
    policy_decision = {
        "status": status,
        "valid_single_state_allowed": valid_single_state_allowed,
        "fragmentation_penalty_flag": fragmentation_penalty,
        "penalize_fake_regime_fragmentation": fragmentation_penalty,
        "production_outputs_written": False,
        "production_label_change": False,
        "permanent_exclusion": False,
        "recommended_action": {
            FLAT_POLICY_STATUS_ACTIVE: "cluster_as_active_candidate",
            FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE: "allow_flat_single_state_candidate",
            FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE: "collect_more_evidence_before_method_selection",
            FLAT_POLICY_STATUS_AXIS_NOT_CLUSTERABLE_CANDIDATE: "treat_axis_as_not_clusterable_for_this_trial",
            FLAT_POLICY_STATUS_INSUFFICIENT_DATA: "collect_more_rows",
            FLAT_POLICY_STATUS_UNKNOWN: "manual_review_missing_policy_evidence",
        }[status],
    }
    return FlatAssetPolicyResult(
        status=status,
        valid_single_state_allowed=valid_single_state_allowed,
        fragmentation_penalty_flag=fragmentation_penalty,
        confidence_score=confidence,
        evidence_summary=evidence_summary,
        diagnostics=diagnostics,
        policy_decision=policy_decision,
        policy=cfg,
        warnings=warnings,
        source_metadata=dict(source_metadata or {}),
    )


__all__ = [
    "FLAT_ASSET_POLICY_ARTIFACT_KIND",
    "FLAT_ASSET_POLICY_STATUSES",
    "FLAT_POLICY_STATUS_ACTIVE",
    "FLAT_POLICY_STATUS_AXIS_NOT_CLUSTERABLE_CANDIDATE",
    "FLAT_POLICY_STATUS_INSUFFICIENT_DATA",
    "FLAT_POLICY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE",
    "FLAT_POLICY_STATUS_UNKNOWN",
    "FLAT_POLICY_STATUS_VALID_FLAT_SINGLE_STATE_CANDIDATE",
    "REGIME_FLAT_ASSET_POLICY_SCHEMA_VERSION",
    "FlatAssetPolicyConfig",
    "FlatAssetPolicyResult",
    "evaluate_flat_asset_policy",
]
