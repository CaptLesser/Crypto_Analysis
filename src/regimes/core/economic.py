from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_non_empty_string, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable

try:
    from scipy.stats import kruskal  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - only exercised in minimal dependency environments
    kruskal = None  # type: ignore[assignment]
    _HAS_SCIPY = False


REGIME_ECONOMIC_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION

ECONOMIC_COMPUTED = "computed"
ECONOMIC_NOT_APPLICABLE = "not_applicable"
ECONOMIC_NOT_AVAILABLE = "not_available"
ECONOMIC_DEPENDENCY_MISSING = "dependency_missing"
ECONOMIC_FAILED = "failed"

FORWARD_TARGET_TYPES: tuple[str, ...] = (
    "forward_return",
    "realized_volatility",
    "drawdown",
)
MISSING_TARGET_POLICIES: tuple[str, ...] = ("not_available", "not_applicable", "fail")


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _label_key(value: object) -> str:
    try:
        if pd.isna(value):
            return "<null>"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "<empty>"


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
        return str(value).strip() == str(noise_label)


def _is_unknown_label(value: object, unknown_tokens: Sequence[str]) -> bool:
    if _is_null_label(value):
        return False
    tokens = {str(token).strip().lower() for token in unknown_tokens}
    return str(value).strip().lower() in tokens


@dataclass(frozen=True)
class ForwardTargetSpec:
    horizon: int
    target_type: str
    missing_target_policy: str = ECONOMIC_NOT_AVAILABLE
    target_column: str | None = None
    schema_version: int = REGIME_ECONOMIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        horizon = int(self.horizon)
        if horizon <= 0:
            raise ValueError("Regime forward target horizon must be positive")
        target_type = require_non_empty_string(self.target_type, field_name="forward target type").lower()
        if target_type not in FORWARD_TARGET_TYPES:
            valid = ", ".join(FORWARD_TARGET_TYPES)
            raise ValueError(f"Unsupported Regime forward target type {target_type!r}; expected one of: {valid}")
        policy = require_non_empty_string(self.missing_target_policy, field_name="missing target policy").lower()
        if policy not in MISSING_TARGET_POLICIES:
            valid = ", ".join(MISSING_TARGET_POLICIES)
            raise ValueError(f"Unsupported Regime missing target policy {policy!r}; expected one of: {valid}")
        column = None if self.target_column is None else str(self.target_column).strip() or None
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "target_type", target_type)
        object.__setattr__(self, "missing_target_policy", policy)
        object.__setattr__(self, "target_column", column)

    @property
    def metric_name(self) -> str:
        return self.target_type

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "horizon": int(self.horizon),
            "target_type": self.target_type,
            "missing_target_policy": self.missing_target_policy,
            "target_column": self.target_column,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ForwardTargetSpec":
        obj = require_json_object(payload, context="Regime ForwardTargetSpec")
        return cls(
            schema_version=obj["schema_version"],
            horizon=obj["horizon"],
            target_type=obj["target_type"],
            missing_target_policy=obj.get("missing_target_policy", ECONOMIC_NOT_AVAILABLE),
            target_column=obj.get("target_column"),
        )

    @classmethod
    def from_json(cls, text: str) -> "ForwardTargetSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime ForwardTargetSpec JSON"))


def default_forward_target_specs(horizon: int = 1) -> tuple[ForwardTargetSpec, ...]:
    return (
        ForwardTargetSpec(horizon=horizon, target_type="forward_return"),
        ForwardTargetSpec(horizon=horizon, target_type="realized_volatility"),
        ForwardTargetSpec(horizon=horizon, target_type="drawdown"),
    )


def _missing_status(spec: ForwardTargetSpec, reason: str) -> dict[str, Any]:
    status = spec.missing_target_policy
    if status == "fail":
        status = ECONOMIC_FAILED
    return {
        "status": status,
        "reason": reason,
        "metrics": {},
        "target_spec": spec.as_dict(),
    }


def _candidate_columns(spec: ForwardTargetSpec) -> tuple[str, ...]:
    horizon = int(spec.horizon)
    suffixes = (f"_{horizon}m", f"_{horizon}", "")
    if spec.target_type == "forward_return":
        bases = ("future_log_return", "future_return", "forward_return", "log_return")
    elif spec.target_type == "realized_volatility":
        bases = (
            "future_realized_volatility",
            "future_realized_vol",
            "future_volatility",
            "future_vol",
            "forward_realized_volatility",
            "forward_realized_vol",
            "forward_volatility",
            "forward_vol",
        )
    else:
        bases = (
            "future_max_drawdown",
            "future_drawdown",
            "future_downside_excursion",
            "max_drawdown",
            "drawdown",
            "downside_excursion",
        )
    return tuple(dict.fromkeys(f"{base}{suffix}" for base in bases for suffix in suffixes))


def find_forward_target_column(frame: pd.DataFrame, spec: ForwardTargetSpec) -> str | None:
    if spec.target_column:
        return spec.target_column if spec.target_column in frame.columns else None
    for column in _candidate_columns(spec):
        if column in frame.columns:
            return str(column)
    tokens_by_type = {
        "forward_return": ("return",),
        "realized_volatility": ("vol",),
        "drawdown": ("drawdown",),
    }
    tokens = tokens_by_type[spec.target_type]
    for column in frame.columns:
        lowered = str(column).lower()
        if all(token in lowered for token in tokens):
            return str(column)
    if spec.target_type == "drawdown":
        for column in frame.columns:
            if "downside" in str(column).lower():
                return str(column)
    return None


def _valid_label_mask(
    labels: Sequence[object],
    *,
    noise_label: object,
    unknown_tokens: Sequence[str],
) -> np.ndarray:
    mask = np.ones((len(labels),), dtype=bool)
    for idx, label in enumerate(labels):
        if _is_null_label(label) or _is_noise_label(label, noise_label) or _is_unknown_label(label, unknown_tokens):
            mask[idx] = False
    return mask


def _cohen_d(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    pooled = math.sqrt((float(np.var(left, ddof=0)) + float(np.var(right, ddof=0))) / 2.0)
    if pooled <= 1e-12:
        return None
    return float((float(np.mean(left)) - float(np.mean(right))) / pooled)


def _effect_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    groups: dict[str, np.ndarray] = {}
    for label in sorted(set(labels.tolist()), key=str):
        group = values[labels == label]
        group = group[np.isfinite(group)]
        if group.size:
            groups[str(label)] = group
            per_label[str(label)] = {
                "count": int(group.size),
                "mean": float(np.mean(group)),
                "median": float(np.median(group)),
                "std": float(np.std(group, ddof=0)),
                "min": float(np.min(group)),
                "max": float(np.max(group)),
            }
    if len(groups) < 2:
        return {"per_label": per_label}
    means = [row["mean"] for row in per_label.values()]
    medians = [row["median"] for row in per_label.values()]
    pairwise: list[dict[str, Any]] = []
    for left_label, right_label in combinations(sorted(groups), 2):
        left = groups[left_label]
        right = groups[right_label]
        pairwise.append(
            {
                "left_label": left_label,
                "right_label": right_label,
                "mean_difference": float(np.mean(left) - np.mean(right)),
                "median_difference": float(np.median(left) - np.median(right)),
                "cohen_d": _cohen_d(left, right),
            }
        )
    cohen_values = [abs(float(row["cohen_d"])) for row in pairwise if row["cohen_d"] is not None]
    return {
        "per_label": per_label,
        "mean_spread": float(max(means) - min(means)),
        "median_spread": float(max(medians) - min(medians)),
        "max_abs_pairwise_cohen_d": None if not cohen_values else float(max(cohen_values)),
        "pairwise_effects": pairwise,
    }


def _nonparametric_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    groups = []
    for label in sorted(set(labels.tolist()), key=str):
        group = values[labels == label]
        group = group[np.isfinite(group)]
        if group.size:
            groups.append(group)
    if len(groups) < 2:
        return {"status": ECONOMIC_NOT_APPLICABLE, "reason": "fewer than two finite label groups", "value": None}
    if not _HAS_SCIPY:
        return {"status": ECONOMIC_DEPENDENCY_MISSING, "reason": "scipy.stats.kruskal is not available", "value": None}
    try:
        statistic, pvalue = kruskal(*groups)  # type: ignore[misc]
    except ValueError as exc:
        return {"status": ECONOMIC_NOT_APPLICABLE, "reason": f"kruskal comparison not valid: {exc}", "value": None}
    except Exception as exc:
        return {"status": ECONOMIC_FAILED, "reason": f"kruskal failed: {exc}", "value": None}
    return {
        "status": ECONOMIC_COMPUTED,
        "reason": None,
        "value": {"statistic": float(statistic), "pvalue": float(pvalue)},
    }


def score_forward_target_separability(
    labels: Sequence[object],
    forward_frame: pd.DataFrame | None,
    spec: ForwardTargetSpec,
    *,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
) -> dict[str, Any]:
    if forward_frame is None:
        return _missing_status(spec, "forward target frame not supplied")
    if len(labels) != len(forward_frame):
        return {
            "status": ECONOMIC_NOT_APPLICABLE,
            "reason": "forward target frame and labels row counts differ",
            "metrics": {},
            "target_spec": spec.as_dict(),
        }
    column = find_forward_target_column(forward_frame, spec)
    if column is None:
        return _missing_status(spec, f"target column for {spec.target_type!r} not found")
    values = pd.to_numeric(forward_frame[column], errors="coerce").to_numpy(dtype=float)
    label_arr = np.asarray(list(labels), dtype=object)
    label_mask = _valid_label_mask(label_arr.tolist(), noise_label=noise_label, unknown_tokens=unknown_tokens)
    finite_mask = np.isfinite(values) & label_mask
    values = values[finite_mask]
    label_arr = label_arr[finite_mask]
    labels_clean = np.asarray([_label_key(label) for label in label_arr.tolist()], dtype=object)
    if values.size < 2:
        return {
            "status": ECONOMIC_NOT_APPLICABLE,
            "reason": "fewer than two finite target rows",
            "metrics": {"target_column": column, "finite_row_count": int(values.size)},
            "target_spec": spec.as_dict(),
        }
    if len(set(labels_clean.tolist())) < 2:
        return {
            "status": ECONOMIC_NOT_APPLICABLE,
            "reason": "fewer than two finite label groups",
            "metrics": {"target_column": column, **_effect_summary(values, labels_clean)},
            "target_spec": spec.as_dict(),
        }
    effect = _effect_summary(values, labels_clean)
    metrics = {
        "target_column": column,
        "finite_row_count": int(values.size),
        **effect,
        "nonparametric": _nonparametric_summary(values, labels_clean),
    }
    return {
        "status": ECONOMIC_COMPUTED,
        "reason": None,
        "metrics": to_jsonable(metrics),
        "target_spec": spec.as_dict(),
    }


def score_forward_return_separability(
    labels: Sequence[object],
    forward_frame: pd.DataFrame | None,
    *,
    horizon: int = 1,
    missing_target_policy: str = ECONOMIC_NOT_AVAILABLE,
    target_column: str | None = None,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
) -> dict[str, Any]:
    return score_forward_target_separability(
        labels,
        forward_frame,
        ForwardTargetSpec(
            horizon=horizon,
            target_type="forward_return",
            missing_target_policy=missing_target_policy,
            target_column=target_column,
        ),
        noise_label=noise_label,
        unknown_tokens=unknown_tokens,
    )


def score_realized_volatility_separability(
    labels: Sequence[object],
    forward_frame: pd.DataFrame | None,
    *,
    horizon: int = 1,
    missing_target_policy: str = ECONOMIC_NOT_AVAILABLE,
    target_column: str | None = None,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
) -> dict[str, Any]:
    return score_forward_target_separability(
        labels,
        forward_frame,
        ForwardTargetSpec(
            horizon=horizon,
            target_type="realized_volatility",
            missing_target_policy=missing_target_policy,
            target_column=target_column,
        ),
        noise_label=noise_label,
        unknown_tokens=unknown_tokens,
    )


def score_drawdown_separability(
    labels: Sequence[object],
    forward_frame: pd.DataFrame | None,
    *,
    horizon: int = 1,
    missing_target_policy: str = ECONOMIC_NOT_AVAILABLE,
    target_column: str | None = None,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
) -> dict[str, Any]:
    return score_forward_target_separability(
        labels,
        forward_frame,
        ForwardTargetSpec(
            horizon=horizon,
            target_type="drawdown",
            missing_target_policy=missing_target_policy,
            target_column=target_column,
        ),
        noise_label=noise_label,
        unknown_tokens=unknown_tokens,
    )


def score_economic_separability(
    labels: Sequence[object],
    forward_frame: pd.DataFrame | None,
    *,
    target_specs: Sequence[ForwardTargetSpec | Mapping[str, Any]] | None = None,
    horizon: int = 1,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
) -> dict[str, Any]:
    specs = tuple(
        spec if isinstance(spec, ForwardTargetSpec) else ForwardTargetSpec.from_dict(spec)
        for spec in (target_specs if target_specs is not None else default_forward_target_specs(horizon=horizon))
    )
    metrics = {
        spec.metric_name: score_forward_target_separability(
            labels,
            forward_frame,
            spec,
            noise_label=noise_label,
            unknown_tokens=unknown_tokens,
        )
        for spec in specs
    }
    if any(row["status"] == ECONOMIC_COMPUTED for row in metrics.values()):
        status = ECONOMIC_COMPUTED
        reason = None
    elif all(row["status"] == ECONOMIC_NOT_AVAILABLE for row in metrics.values()):
        status = ECONOMIC_NOT_AVAILABLE
        reason = "forward targets are not available"
    elif all(row["status"] == ECONOMIC_NOT_APPLICABLE for row in metrics.values()):
        status = ECONOMIC_NOT_APPLICABLE
        reason = "forward targets are not applicable"
    else:
        status = ECONOMIC_NOT_APPLICABLE
        reason = "no economic separability metrics were computed"
    return {
        "schema_version": REGIME_ECONOMIC_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "metrics": metrics,
    }


__all__ = [
    "ECONOMIC_COMPUTED",
    "ECONOMIC_DEPENDENCY_MISSING",
    "ECONOMIC_FAILED",
    "ECONOMIC_NOT_APPLICABLE",
    "ECONOMIC_NOT_AVAILABLE",
    "FORWARD_TARGET_TYPES",
    "ForwardTargetSpec",
    "MISSING_TARGET_POLICIES",
    "REGIME_ECONOMIC_SCHEMA_VERSION",
    "default_forward_target_specs",
    "find_forward_target_column",
    "score_drawdown_separability",
    "score_economic_separability",
    "score_forward_target_separability",
    "score_forward_return_separability",
    "score_realized_volatility_separability",
]
