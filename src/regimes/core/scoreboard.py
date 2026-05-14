from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_non_empty_string, require_schema_version
from src.regimes.core.economic import ForwardTargetSpec, score_economic_separability
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.core.stability import StabilityPlan, score_precomputed_stability

try:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - only exercised in minimal dependency environments
    calinski_harabasz_score = None  # type: ignore[assignment]
    davies_bouldin_score = None  # type: ignore[assignment]
    silhouette_score = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


REGIME_SCOREBOARD_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
REGIME_SCOREBOARD_ARTIFACT_KIND = "regime_core_scoreboard"

METRIC_COMPUTED = "computed"
METRIC_NOT_APPLICABLE = "not_applicable"
METRIC_DEPENDENCY_MISSING = "dependency_missing"
METRIC_FAILED = "failed"


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _metric(
    status: str,
    value: object = None,
    *,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": str(status),
        "value": to_jsonable(value),
        "reason": reason,
        "metadata": to_jsonable(dict(metadata or {})),
    }


def _not_applicable(reason: str, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _metric(METRIC_NOT_APPLICABLE, reason=reason, metadata=metadata)


def _label_key(value: object) -> str:
    try:
        if pd.isna(value):
            return "<null>"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "<empty>"


def _is_null(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_noise(value: object, noise_label: object) -> bool:
    if _is_null(value):
        return False
    try:
        return int(value) == int(noise_label)
    except Exception:
        return str(value).strip() == str(noise_label)


def _is_unknown(value: object, unknown_tokens: Sequence[str]) -> bool:
    if _is_null(value):
        return False
    tokens = {str(token).strip().lower() for token in unknown_tokens}
    return str(value).strip().lower() in tokens


def _valid_metric_mask(
    labels: Sequence[object],
    *,
    noise_label: object,
    unknown_tokens: Sequence[str],
) -> np.ndarray:
    mask = np.ones((len(labels),), dtype=bool)
    for idx, label in enumerate(labels):
        if _is_null(label) or _is_noise(label, noise_label) or _is_unknown(label, unknown_tokens):
            mask[idx] = False
    return mask


def _metric_inputs(
    features: Sequence[Sequence[float]] | np.ndarray | None,
    labels: Sequence[object],
    *,
    noise_label: object,
    unknown_tokens: Sequence[str],
) -> tuple[bool, str | None, np.ndarray, np.ndarray]:
    if features is None:
        return False, "feature matrix not supplied", np.empty((0, 0)), np.empty((0,), dtype=object)
    try:
        x = np.asarray(features, dtype=float)
    except Exception as exc:
        return False, f"feature matrix conversion failed: {exc}", np.empty((0, 0)), np.empty((0,), dtype=object)
    label_arr = np.asarray(list(labels), dtype=object)
    if x.ndim != 2:
        return False, "feature matrix must be two-dimensional", np.empty((0, 0)), np.empty((0,), dtype=object)
    if x.shape[0] != label_arr.size:
        return False, "feature matrix and labels row counts differ", np.empty((0, 0)), np.empty((0,), dtype=object)
    if not np.isfinite(x).all():
        return False, "feature matrix contains non-finite values", np.empty((0, 0)), np.empty((0,), dtype=object)
    mask = _valid_metric_mask(label_arr.tolist(), noise_label=noise_label, unknown_tokens=unknown_tokens)
    if int(mask.sum()) < 3:
        return False, "fewer than three non-noise labeled rows", x[mask], label_arr[mask]
    x_eval = x[mask]
    labels_eval = label_arr[mask]
    state_count = len({_label_key(label) for label in labels_eval.tolist()})
    if state_count < 2:
        return False, "one effective label state", x_eval, labels_eval
    if state_count >= labels_eval.size:
        return False, "every labeled row is its own state", x_eval, labels_eval
    if not _HAS_SKLEARN:
        return False, "sklearn is not available", x_eval, labels_eval
    return True, None, x_eval, labels_eval


def _call_metric(name: str, fn: Any, x_eval: np.ndarray, labels_eval: np.ndarray) -> dict[str, Any]:
    if fn is None:
        return _metric(METRIC_DEPENDENCY_MISSING, reason="sklearn is not available")
    try:
        return _metric(METRIC_COMPUTED, _safe_float(fn(x_eval, labels_eval)))
    except Exception as exc:
        return _metric(METRIC_FAILED, reason=f"{name} failed: {exc}")


def _extract_outcome_metadata(fit_result: Any | None) -> dict[str, Any]:
    if fit_result is None:
        return {}
    metadata = getattr(fit_result, "metadata", None)
    if isinstance(metadata, Mapping):
        outcome = metadata.get("outcome")
        if isinstance(outcome, Mapping):
            return dict(outcome)
    fit_metadata = getattr(fit_result, "fit_metadata", None)
    if isinstance(fit_metadata, Mapping):
        return dict(fit_metadata)
    return {}


def _information_criterion(
    name: str,
    features: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    model: object | None,
    fit_result: Any | None,
) -> dict[str, Any]:
    outcome = _extract_outcome_metadata(fit_result)
    if name in outcome:
        safe = _safe_float(outcome[name])
        if safe is not None:
            return _metric(METRIC_COMPUTED, safe, metadata={"source": "fit_result"})
    if model is None:
        return _not_applicable(f"model does not provide {name}", metadata={"source": "model"})
    attr = getattr(model, name, None)
    if attr is None:
        return _not_applicable(f"model does not expose {name}", metadata={"source": "model"})
    if features is None:
        return _not_applicable(f"feature matrix required for model {name}", metadata={"source": "model"})
    try:
        x = np.asarray(features, dtype=float)
        value = attr(x) if callable(attr) else attr
    except Exception as exc:
        return _metric(METRIC_FAILED, reason=f"model {name} failed: {exc}", metadata={"source": "model"})
    safe = _safe_float(value)
    if safe is None:
        return _metric(METRIC_FAILED, reason=f"model {name} returned non-finite value", metadata={"source": "model"})
    return _metric(METRIC_COMPUTED, safe, metadata={"source": "model"})


def score_internal_validity(
    features: Sequence[Sequence[float]] | np.ndarray | None,
    labels: Sequence[object],
    *,
    model: object | None = None,
    fit_result: Any | None = None,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
) -> dict[str, Any]:
    possible, reason, x_eval, labels_eval = _metric_inputs(
        features,
        labels,
        noise_label=noise_label,
        unknown_tokens=unknown_tokens,
    )
    if not possible:
        status = METRIC_DEPENDENCY_MISSING if reason == "sklearn is not available" else METRIC_NOT_APPLICABLE
        base = _metric(status, reason=reason)
        return {
            "status": status,
            "metrics": {
                "silhouette": base,
                "calinski_harabasz": base,
                "davies_bouldin": base,
                "aic": _information_criterion("aic", features, model=model, fit_result=fit_result),
                "bic": _information_criterion("bic", features, model=model, fit_result=fit_result),
            },
        }
    metrics = {
        "silhouette": _call_metric("silhouette", silhouette_score, x_eval, labels_eval),
        "calinski_harabasz": _call_metric("calinski_harabasz", calinski_harabasz_score, x_eval, labels_eval),
        "davies_bouldin": _call_metric("davies_bouldin", davies_bouldin_score, x_eval, labels_eval),
        "aic": _information_criterion("aic", features, model=model, fit_result=fit_result),
        "bic": _information_criterion("bic", features, model=model, fit_result=fit_result),
    }
    status = METRIC_COMPUTED if any(row["status"] == METRIC_COMPUTED for row in metrics.values()) else METRIC_NOT_APPLICABLE
    return {"status": status, "metrics": metrics}


def score_coverage_degeneracy(
    labels: Sequence[object],
    *,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
    tiny_cluster_threshold: int = 20,
) -> dict[str, Any]:
    total = int(len(labels))
    raw_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    noise_count = 0
    unknown_count = 0
    null_count = 0
    for label in labels:
        key = _label_key(label)
        raw_counts[key] += 1
        if _is_null(label):
            null_count += 1
        elif _is_noise(label, noise_label):
            noise_count += 1
        elif _is_unknown(label, unknown_tokens):
            unknown_count += 1
        else:
            effective_counts[key] += 1
    cluster_sizes = np.asarray(list(effective_counts.values()), dtype=float)
    singleton_rows = int(sum(size for size in effective_counts.values() if int(size) == 1))
    tiny_rows = int(sum(size for size in effective_counts.values() if int(size) <= int(tiny_cluster_threshold)))
    unknown_or_null_count = int(unknown_count + null_count)
    effective_state_count = int(len(effective_counts))
    return {
        "status": METRIC_COMPUTED,
        "metrics": {
            "row_count": total,
            "effective_state_count": effective_state_count,
            "raw_state_counts": {str(key): int(value) for key, value in sorted(raw_counts.items())},
            "effective_state_counts": {str(key): int(value) for key, value in sorted(effective_counts.items())},
            "noise_count": int(noise_count),
            "noise_share": float(noise_count / total) if total else None,
            "unknown_count": int(unknown_count),
            "unknown_share": float(unknown_count / total) if total else None,
            "null_count": int(null_count),
            "null_share": float(null_count / total) if total else None,
            "unknown_or_null_count": unknown_or_null_count,
            "unknown_or_null_share": float(unknown_or_null_count / total) if total else None,
            "singleton_cluster_share": float(singleton_rows / total) if total else None,
            "tiny_cluster_share": float(tiny_rows / total) if total else None,
            "singleton_or_tiny_cluster_share": float(tiny_rows / total) if total else None,
            "min_cluster_size": int(cluster_sizes.min()) if cluster_sizes.size else None,
            "median_cluster_size": float(np.median(cluster_sizes)) if cluster_sizes.size else None,
            "max_cluster_size": int(cluster_sizes.max()) if cluster_sizes.size else None,
            "one_cluster_flag": bool(total > 0 and effective_state_count == 1),
            "all_noise_flag": bool(total > 0 and noise_count == total),
            "all_unknown_or_null_flag": bool(total > 0 and unknown_or_null_count == total),
            "tiny_cluster_threshold": int(tiny_cluster_threshold),
        },
    }


def _runtime_value(obj: Any, *keys: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj:
                return obj[key]
        return None
    for key in keys:
        if hasattr(obj, key):
            return getattr(obj, key)
    metadata = getattr(obj, "runtime_metadata", None)
    if metadata is not None:
        return _runtime_value(metadata, *keys)
    return None


def score_runtime(
    *,
    fit_result: Any | None = None,
    assignment_result: Any | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    row_count: int | None = None,
    feature_count: int | None = None,
    memory_estimate_mb: float | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime_metadata or {})
    fit_seconds = _safe_float(runtime.get("fit_seconds", runtime.get("fit_time_s", runtime.get("fit_elapsed_s"))))
    assign_seconds = _safe_float(runtime.get("assign_seconds", runtime.get("assign_time_s", runtime.get("assign_elapsed_s"))))
    fit_runtime = getattr(fit_result, "runtime_metadata", None)
    assign_runtime = getattr(assignment_result, "runtime_metadata", None)
    if fit_seconds is None:
        fit_seconds = _safe_float(_runtime_value(fit_runtime, "elapsed_s", "fit_time_s"))
    if assign_seconds is None:
        assign_seconds = _safe_float(_runtime_value(assign_runtime, "elapsed_s", "assign_time_s"))
    if row_count is None:
        row_count = runtime.get("row_count", runtime.get("rows"))
    if feature_count is None:
        feature_count = runtime.get("feature_count", runtime.get("features"))
    if row_count is None:
        row_count = _runtime_value(fit_runtime, "row_count")
    if feature_count is None:
        feature_count = _runtime_value(fit_runtime, "feature_count")
    if memory_estimate_mb is None:
        memory_estimate_mb = _safe_float(runtime.get("memory_estimate_mb", runtime.get("peak_rss_mb")))
    metrics = {
        "fit_seconds": fit_seconds,
        "assign_seconds": assign_seconds,
        "row_count": None if row_count is None else int(row_count),
        "feature_count": None if feature_count is None else int(feature_count),
        "memory_estimate_mb": None if memory_estimate_mb is None else float(memory_estimate_mb),
    }
    status = METRIC_COMPUTED if any(value is not None for value in metrics.values()) else METRIC_NOT_APPLICABLE
    return {
        "status": status,
        "metrics": metrics,
    }


@dataclass(frozen=True)
class RegimeScoreboard:
    trial_id: str
    clusterer_family: str
    internal_validity: Mapping[str, Any]
    coverage_degeneracy: Mapping[str, Any]
    runtime: Mapping[str, Any]
    stability: Mapping[str, Any] = field(default_factory=dict)
    economic_separability: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_SCOREBOARD_SCHEMA_VERSION
    artifact_kind: str = REGIME_SCOREBOARD_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        trial_id = require_non_empty_string(self.trial_id, field_name="scoreboard trial_id")
        clusterer_family = require_non_empty_string(self.clusterer_family, field_name="scoreboard clusterer_family").lower()
        if not self.stability:
            object.__setattr__(
                self,
                "stability",
                score_precomputed_stability((), (), plan=StabilityPlan()),
            )
        if not self.economic_separability:
            object.__setattr__(
                self,
                "economic_separability",
                score_economic_separability((), None),
            )
        for section_name in (
            "internal_validity",
            "coverage_degeneracy",
            "runtime",
            "stability",
            "economic_separability",
        ):
            section = getattr(self, section_name)
            if not isinstance(section, Mapping):
                raise ValueError(f"Regime scoreboard {section_name} section must be a mapping")
            if section_name == "stability":
                if "status" not in section or "summary" not in section:
                    raise ValueError("Regime scoreboard stability section requires status and summary")
            elif section_name == "economic_separability":
                if "status" not in section or "metrics" not in section:
                    raise ValueError("Regime scoreboard economic_separability section requires status and metrics")
            elif "status" not in section or "metrics" not in section:
                raise ValueError(f"Regime scoreboard {section_name} section requires status and metrics")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "clusterer_family", clusterer_family)
        object.__setattr__(self, "internal_validity", dict(self.internal_validity))
        object.__setattr__(self, "coverage_degeneracy", dict(self.coverage_degeneracy))
        object.__setattr__(self, "runtime", dict(self.runtime))
        object.__setattr__(self, "stability", dict(self.stability))
        object.__setattr__(self, "economic_separability", dict(self.economic_separability))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "trial_id": self.trial_id,
            "clusterer_family": self.clusterer_family,
            "sections": {
                "internal_validity": to_jsonable(self.internal_validity),
                "coverage_degeneracy": to_jsonable(self.coverage_degeneracy),
                "runtime": to_jsonable(self.runtime),
                "stability": to_jsonable(self.stability),
                "economic_separability": to_jsonable(self.economic_separability),
            },
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeScoreboard":
        obj = require_json_object(payload, context="RegimeScoreboard")
        require_schema_version(obj.get("schema_version"))
        sections = require_json_object(obj.get("sections"), context="RegimeScoreboard sections")
        return cls(
            schema_version=obj["schema_version"],
            artifact_kind=str(obj.get("artifact_kind", REGIME_SCOREBOARD_ARTIFACT_KIND)),
            trial_id=obj["trial_id"],
            clusterer_family=obj["clusterer_family"],
            internal_validity=sections["internal_validity"],
            coverage_degeneracy=sections["coverage_degeneracy"],
            runtime=sections["runtime"],
            stability=sections.get("stability", {}),
            economic_separability=sections.get("economic_separability", {}),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeScoreboard":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeScoreboard JSON"))


def build_regime_scoreboard(
    *,
    trial_id: str,
    clusterer_family: str,
    labels: Sequence[object],
    features: Sequence[Sequence[float]] | np.ndarray | None = None,
    model: object | None = None,
    fit_result: Any | None = None,
    assignment_result: Any | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    stability_plan: StabilityPlan | Mapping[str, Any] | None = None,
    stability_perturbations: Sequence[Mapping[str, Any]] = (),
    baseline_timestamps: Sequence[object] | None = None,
    forward_frame: pd.DataFrame | None = None,
    forward_target_specs: Sequence[ForwardTargetSpec | Mapping[str, Any]] | None = None,
    economic_horizon: int = 1,
    tiny_cluster_threshold: int = 20,
    memory_estimate_mb: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeScoreboard:
    plan = (
        stability_plan
        if isinstance(stability_plan, StabilityPlan)
        else StabilityPlan.from_dict(stability_plan)
        if stability_plan is not None
        else StabilityPlan()
    )
    return RegimeScoreboard(
        trial_id=trial_id,
        clusterer_family=clusterer_family,
        internal_validity=score_internal_validity(features, labels, model=model, fit_result=fit_result),
        coverage_degeneracy=score_coverage_degeneracy(labels, tiny_cluster_threshold=tiny_cluster_threshold),
        runtime=score_runtime(
            fit_result=fit_result,
            assignment_result=assignment_result,
            runtime_metadata=runtime_metadata,
            row_count=None if features is None else int(np.asarray(features).shape[0]),
            feature_count=None if features is None or np.asarray(features).ndim != 2 else int(np.asarray(features).shape[1]),
            memory_estimate_mb=memory_estimate_mb,
        ),
        stability=score_precomputed_stability(
            labels,
            stability_perturbations,
            baseline_timestamps=baseline_timestamps,
            plan=plan,
        ),
        economic_separability=score_economic_separability(
            labels,
            forward_frame,
            target_specs=forward_target_specs,
            horizon=economic_horizon,
        ),
        metadata=metadata or {},
    )


__all__ = [
    "METRIC_COMPUTED",
    "METRIC_DEPENDENCY_MISSING",
    "METRIC_FAILED",
    "METRIC_NOT_APPLICABLE",
    "REGIME_SCOREBOARD_ARTIFACT_KIND",
    "REGIME_SCOREBOARD_SCHEMA_VERSION",
    "RegimeScoreboard",
    "build_regime_scoreboard",
    "score_coverage_degeneracy",
    "score_economic_separability",
    "score_internal_validity",
    "score_runtime",
]
