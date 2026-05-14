from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_non_empty_string, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable

try:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - only exercised in minimal dependency environments
    adjusted_mutual_info_score = None  # type: ignore[assignment]
    adjusted_rand_score = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


REGIME_STABILITY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION

STABILITY_COMPUTED = "computed"
STABILITY_NOT_APPLICABLE = "not_applicable"
STABILITY_DEPENDENCY_MISSING = "dependency_missing"
STABILITY_DEGENERATE_ONE_LABEL = "degenerate_one_label"
STABILITY_NO_OVERLAP = "no_overlap"
STABILITY_FAILED = "failed"


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _is_null_label(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _label_key(value: object) -> str:
    if _is_null_label(value):
        return "<null>"
    text = str(value).strip()
    return text if text else "<empty>"


@dataclass(frozen=True)
class StabilityPlan:
    bootstrap_count: int = 0
    walk_forward_count: int = 0
    minimum_overlap: int = 2
    random_seed: int = 17
    schema_version: int = REGIME_STABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        bootstrap_count = int(self.bootstrap_count)
        walk_forward_count = int(self.walk_forward_count)
        minimum_overlap = int(self.minimum_overlap)
        random_seed = int(self.random_seed)
        if bootstrap_count < 0:
            raise ValueError("Regime stability bootstrap_count must be non-negative")
        if walk_forward_count < 0:
            raise ValueError("Regime stability walk_forward_count must be non-negative")
        if minimum_overlap < 1:
            raise ValueError("Regime stability minimum_overlap must be positive")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "bootstrap_count", bootstrap_count)
        object.__setattr__(self, "walk_forward_count", walk_forward_count)
        object.__setattr__(self, "minimum_overlap", minimum_overlap)
        object.__setattr__(self, "random_seed", random_seed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "bootstrap_count": int(self.bootstrap_count),
            "walk_forward_count": int(self.walk_forward_count),
            "minimum_overlap": int(self.minimum_overlap),
            "random_seed": int(self.random_seed),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StabilityPlan":
        obj = require_json_object(payload, context="Regime StabilityPlan")
        return cls(
            schema_version=obj["schema_version"],
            bootstrap_count=obj["bootstrap_count"],
            walk_forward_count=obj["walk_forward_count"],
            minimum_overlap=obj["minimum_overlap"],
            random_seed=obj["random_seed"],
        )

    @classmethod
    def from_json(cls, text: str) -> "StabilityPlan":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime StabilityPlan JSON"))


@dataclass(frozen=True)
class PerturbedLabelSet:
    name: str
    labels: Sequence[object]
    timestamps: Sequence[object] | None = None
    kind: str = "precomputed"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = require_non_empty_string(self.name, field_name="perturbed label set name")
        labels = tuple(self.labels)
        if self.timestamps is not None and len(tuple(self.timestamps)) != len(labels):
            raise ValueError("Regime perturbed label timestamps must match labels length")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "timestamps", None if self.timestamps is None else tuple(self.timestamps))
        object.__setattr__(self, "kind", str(self.kind).strip() or "precomputed")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, fallback_name: str) -> "PerturbedLabelSet":
        return cls(
            name=str(payload.get("name", fallback_name)),
            labels=tuple(payload.get("labels", ())),
            timestamps=None if payload.get("timestamps") is None else tuple(payload.get("timestamps", ())),
            kind=str(payload.get("kind", "precomputed")),
            metadata=dict(payload.get("metadata", {})),
        )


def align_label_sets(
    baseline_labels: Sequence[object],
    comparison_labels: Sequence[object],
    *,
    baseline_timestamps: Sequence[object] | None = None,
    comparison_timestamps: Sequence[object] | None = None,
) -> dict[str, Any]:
    if baseline_timestamps is None or comparison_timestamps is None:
        overlap = min(len(baseline_labels), len(comparison_labels))
        return {
            "alignment": "row_position",
            "baseline_labels": list(baseline_labels)[:overlap],
            "comparison_labels": list(comparison_labels)[:overlap],
            "overlap_count": int(overlap),
            "first_overlap": 0 if overlap else None,
            "last_overlap": int(overlap - 1) if overlap else None,
        }
    if len(baseline_timestamps) != len(baseline_labels):
        raise ValueError("Regime stability baseline_timestamps must match baseline_labels length")
    if len(comparison_timestamps) != len(comparison_labels):
        raise ValueError("Regime stability comparison_timestamps must match comparison_labels length")
    baseline = pd.DataFrame({"ts": list(baseline_timestamps), "baseline_label": list(baseline_labels)})
    comparison = pd.DataFrame({"ts": list(comparison_timestamps), "comparison_label": list(comparison_labels)})
    merged = baseline.merge(comparison, on="ts", how="inner").sort_values("ts")
    return {
        "alignment": "timestamp",
        "baseline_labels": merged["baseline_label"].tolist(),
        "comparison_labels": merged["comparison_label"].tolist(),
        "overlap_count": int(len(merged)),
        "first_overlap": None if merged.empty else to_jsonable(merged["ts"].iloc[0]),
        "last_overlap": None if merged.empty else to_jsonable(merged["ts"].iloc[-1]),
    }


def _drop_null_pairs(left: Sequence[object], right: Sequence[object]) -> tuple[list[object], list[object]]:
    out_left: list[object] = []
    out_right: list[object] = []
    for left_value, right_value in zip(left, right):
        if _is_null_label(left_value) or _is_null_label(right_value):
            continue
        out_left.append(left_value)
        out_right.append(right_value)
    return out_left, out_right


def compare_label_stability(
    baseline_labels: Sequence[object],
    comparison_labels: Sequence[object],
    *,
    baseline_timestamps: Sequence[object] | None = None,
    comparison_timestamps: Sequence[object] | None = None,
    comparison_name: str = "comparison",
    minimum_overlap: int = 2,
) -> dict[str, Any]:
    aligned = align_label_sets(
        baseline_labels,
        comparison_labels,
        baseline_timestamps=baseline_timestamps,
        comparison_timestamps=comparison_timestamps,
    )
    clean_left, clean_right = _drop_null_pairs(aligned["baseline_labels"], aligned["comparison_labels"])
    valid_overlap = int(len(clean_left))
    baseline_states = len({_label_key(value) for value in clean_left})
    comparison_states = len({_label_key(value) for value in clean_right})
    base = {
        "comparison_name": str(comparison_name),
        "alignment": aligned["alignment"],
        "overlap_count": int(aligned["overlap_count"]),
        "valid_overlap_count": valid_overlap,
        "minimum_overlap": int(minimum_overlap),
        "first_overlap": aligned["first_overlap"],
        "last_overlap": aligned["last_overlap"],
        "baseline_state_count": int(baseline_states),
        "comparison_state_count": int(comparison_states),
    }
    if valid_overlap < int(minimum_overlap):
        return {
            **base,
            "status": STABILITY_NO_OVERLAP if int(aligned["overlap_count"]) == 0 else STABILITY_NOT_APPLICABLE,
            "reason": "insufficient overlapping non-null labels",
            "ari": None,
            "ami": None,
        }
    if baseline_states <= 1 or comparison_states <= 1:
        return {
            **base,
            "status": STABILITY_DEGENERATE_ONE_LABEL,
            "reason": "baseline or comparison has one label state on overlap",
            "ari": None,
            "ami": None,
        }
    if not _HAS_SKLEARN:
        return {
            **base,
            "status": STABILITY_DEPENDENCY_MISSING,
            "reason": "sklearn is not available",
            "ari": None,
            "ami": None,
        }
    try:
        ari = float(adjusted_rand_score(clean_left, clean_right))  # type: ignore[misc]
        ami = float(adjusted_mutual_info_score(clean_left, clean_right))  # type: ignore[misc]
    except Exception as exc:
        return {
            **base,
            "status": STABILITY_FAILED,
            "reason": f"stability metric computation failed: {exc}",
            "ari": None,
            "ami": None,
        }
    return {
        **base,
        "status": STABILITY_COMPUTED,
        "reason": None,
        "ari": ari,
        "ami": ami,
    }


def _value_summary(values: Sequence[object]) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values if _safe_float(value) is not None], dtype=float)
    if clean.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(clean.size),
        "min": float(np.min(clean)),
        "median": float(np.median(clean)),
        "max": float(np.max(clean)),
    }


def score_precomputed_stability(
    baseline_labels: Sequence[object],
    perturbations: Sequence[PerturbedLabelSet | Mapping[str, Any]],
    *,
    baseline_timestamps: Sequence[object] | None = None,
    plan: StabilityPlan | None = None,
) -> dict[str, Any]:
    resolved_plan = plan or StabilityPlan()
    comparisons: list[dict[str, Any]] = []
    for idx, perturbation in enumerate(perturbations):
        item = (
            perturbation
            if isinstance(perturbation, PerturbedLabelSet)
            else PerturbedLabelSet.from_mapping(perturbation, fallback_name=f"perturbation_{idx}")
        )
        comparisons.append(
            compare_label_stability(
                baseline_labels,
                item.labels,
                baseline_timestamps=baseline_timestamps,
                comparison_timestamps=item.timestamps,
                comparison_name=item.name,
                minimum_overlap=resolved_plan.minimum_overlap,
            )
        )
    ari_values = [row.get("ari") for row in comparisons]
    ami_values = [row.get("ami") for row in comparisons]
    computed_count = int(sum(1 for row in comparisons if row["status"] == STABILITY_COMPUTED))
    no_overlap_count = int(sum(1 for row in comparisons if row["status"] == STABILITY_NO_OVERLAP))
    degenerate_count = int(sum(1 for row in comparisons if row["status"] == STABILITY_DEGENERATE_ONE_LABEL))
    if not comparisons:
        status = STABILITY_NOT_APPLICABLE
        reason = "no precomputed perturbations supplied"
    elif computed_count:
        status = STABILITY_COMPUTED
        reason = None
    elif no_overlap_count == len(comparisons):
        status = STABILITY_NO_OVERLAP
        reason = "no comparisons had overlapping labels"
    elif degenerate_count == len(comparisons):
        status = STABILITY_DEGENERATE_ONE_LABEL
        reason = "all comparisons were degenerate one-label cases"
    elif all(row["status"] == STABILITY_DEPENDENCY_MISSING for row in comparisons):
        status = STABILITY_DEPENDENCY_MISSING
        reason = "sklearn is not available"
    else:
        status = STABILITY_NOT_APPLICABLE
        reason = "no stability metrics were computed"
    return {
        "schema_version": REGIME_STABILITY_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "plan": resolved_plan.as_dict(),
        "comparisons": comparisons,
        "summary": {
            "comparison_count": int(len(comparisons)),
            "computed_count": computed_count,
            "no_overlap_count": no_overlap_count,
            "degenerate_one_label_count": degenerate_count,
            "overlap_counts": _value_summary([row.get("valid_overlap_count") for row in comparisons]),
            "ari": _value_summary(ari_values),
            "ami": _value_summary(ami_values),
        },
    }


__all__ = [
    "REGIME_STABILITY_SCHEMA_VERSION",
    "STABILITY_COMPUTED",
    "STABILITY_DEPENDENCY_MISSING",
    "STABILITY_DEGENERATE_ONE_LABEL",
    "STABILITY_FAILED",
    "STABILITY_NO_OVERLAP",
    "STABILITY_NOT_APPLICABLE",
    "PerturbedLabelSet",
    "StabilityPlan",
    "align_label_sets",
    "compare_label_stability",
    "score_precomputed_stability",
]
