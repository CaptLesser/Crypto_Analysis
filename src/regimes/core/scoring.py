"""Legacy scoring compatibility surface.

Canonical foundation studies should use ``src.regimes.core.scoreboard`` for the
score envelope. This module remains importable for older scaffold tests.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only in minimal dependency environments
    adjusted_mutual_info_score = None  # type: ignore[assignment]
    adjusted_rand_score = None  # type: ignore[assignment]
    calinski_harabasz_score = None  # type: ignore[assignment]
    davies_bouldin_score = None  # type: ignore[assignment]
    silhouette_score = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


REGIME_SCORING_SCHEMA_VERSION = 1
METRIC_COMPUTED = "computed"
METRIC_NOT_APPLICABLE = "not_applicable"
METRIC_DEPENDENCY_MISSING = "dependency_missing"
METRIC_MISSING = "missing"
METRIC_FAILED = "failed"


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
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
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def metric_record(
    status: str,
    value: object = None,
    *,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": str(status),
        "value": _jsonable(value),
        "reason": reason,
        "metadata": _jsonable(dict(metadata or {})),
    }


def _label_series(labels: Sequence[object]) -> pd.Series:
    return pd.Series(list(labels), dtype="object")


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


def _is_noise_label(value: object, noise_label: object = -1) -> bool:
    if _is_null_label(value):
        return False
    try:
        return int(value) == int(noise_label)
    except Exception:
        return str(value).strip() == str(noise_label)


def _is_unknown_label(value: object, unknown_tokens: Sequence[str] = ("unknown",)) -> bool:
    if _is_null_label(value):
        return False
    return str(value).strip().lower() in {str(token).strip().lower() for token in unknown_tokens}


def _valid_scoring_mask(
    labels: Sequence[object],
    *,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
    drop_noise: bool = True,
    drop_unknown: bool = True,
) -> np.ndarray:
    series = _label_series(labels)
    mask = np.ones((len(series),), dtype=bool)
    for idx, value in enumerate(series.tolist()):
        if _is_null_label(value):
            mask[idx] = False
        elif drop_noise and _is_noise_label(value, noise_label=noise_label):
            mask[idx] = False
        elif drop_unknown and _is_unknown_label(value, unknown_tokens=unknown_tokens):
            mask[idx] = False
    return mask


def _metric_possible(features: np.ndarray, labels: Sequence[object]) -> tuple[bool, str | None, np.ndarray, np.ndarray]:
    if not _HAS_SKLEARN:
        return False, "sklearn is not available", np.empty((0, 0)), np.empty((0,))
    x = np.asarray(features, dtype=float)
    label_arr = np.asarray(list(labels), dtype=object)
    if x.ndim != 2:
        return False, "features must be a 2D matrix", np.empty((0, 0)), np.empty((0,))
    if x.shape[0] != label_arr.size:
        return False, "features and labels row counts differ", np.empty((0, 0)), np.empty((0,))
    mask = _valid_scoring_mask(label_arr)
    if int(mask.sum()) < 3:
        return False, "fewer than three non-noise labeled rows", np.empty((0, 0)), np.empty((0,))
    x_eval = x[mask]
    labels_eval = label_arr[mask]
    unique = sorted({_label_key(value) for value in labels_eval.tolist()})
    if len(unique) < 2:
        return False, "one effective label state", x_eval, labels_eval
    if len(unique) >= len(labels_eval):
        return False, "every labeled row is its own state", x_eval, labels_eval
    return True, None, x_eval, labels_eval


def _call_sklearn_metric(
    name: str,
    fn: Callable[..., float] | None,
    x_eval: np.ndarray,
    labels_eval: np.ndarray,
) -> dict[str, Any]:
    if fn is None:
        return metric_record(METRIC_DEPENDENCY_MISSING, reason="sklearn is not available")
    try:
        return metric_record(METRIC_COMPUTED, float(fn(x_eval, labels_eval)))
    except Exception as exc:
        return metric_record(METRIC_FAILED, reason=f"{name} failed: {exc}")


def _score_model_information_criterion(
    model: object | None,
    features: np.ndarray,
    name: str,
) -> dict[str, Any]:
    if model is None:
        return metric_record(METRIC_NOT_APPLICABLE, reason="model object not supplied")
    attr = getattr(model, name, None)
    if attr is None:
        return metric_record(METRIC_NOT_APPLICABLE, reason=f"model does not expose {name}")
    try:
        value = attr(features) if callable(attr) else attr
    except Exception as exc:
        return metric_record(METRIC_FAILED, reason=f"model {name} failed: {exc}")
    safe = _safe_float(value)
    if safe is None:
        return metric_record(METRIC_FAILED, reason=f"model {name} returned non-finite value")
    return metric_record(METRIC_COMPUTED, safe)


def _score_density_metric(
    features: np.ndarray,
    labels: Sequence[object],
    *,
    clusterer_family: str,
    density_metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    family = str(clusterer_family or "").strip().lower()
    if family not in {"hdbscan", "optics"} and density_metric_fn is None:
        return metric_record(METRIC_NOT_APPLICABLE, reason="clusterer family is not density-aware")
    x = np.asarray(features, dtype=float)
    labels_arr = np.asarray(list(labels))
    if x.ndim != 2 or x.shape[0] != labels_arr.size:
        return metric_record(METRIC_NOT_APPLICABLE, reason="features and labels are not aligned")
    if density_metric_fn is not None:
        try:
            return metric_record(METRIC_COMPUTED, float(density_metric_fn(x, labels_arr)))
        except Exception as exc:
            return metric_record(METRIC_FAILED, reason=f"density metric hook failed: {exc}")
    try:
        from hdbscan.validity import validity_index  # type: ignore
    except Exception:
        return metric_record(METRIC_DEPENDENCY_MISSING, reason="hdbscan validity_index is not available")
    try:
        return metric_record(METRIC_COMPUTED, float(validity_index(x, labels_arr.astype(int))))
    except Exception as exc:
        return metric_record(METRIC_FAILED, reason=f"hdbscan validity_index failed: {exc}")


def score_internal_validity(
    features: Sequence[Sequence[float]] | np.ndarray | None,
    labels: Sequence[object],
    *,
    model: object | None = None,
    clusterer_family: str = "",
    density_metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    if features is None:
        base = metric_record(METRIC_NOT_APPLICABLE, reason="feature matrix not supplied")
        return {
            "status": METRIC_NOT_APPLICABLE,
            "metrics": {
                "silhouette": base,
                "calinski_harabasz": base,
                "davies_bouldin": base,
                "bic": _score_model_information_criterion(model, np.empty((0, 0)), "bic"),
                "aic": _score_model_information_criterion(model, np.empty((0, 0)), "aic"),
                "density_validity": metric_record(METRIC_NOT_APPLICABLE, reason="feature matrix not supplied"),
            },
        }
    try:
        x_full = np.asarray(features, dtype=float)
    except Exception as exc:
        metric = metric_record(METRIC_FAILED, reason=f"feature matrix conversion failed: {exc}")
        return {
            "status": METRIC_FAILED,
            "metrics": {
                "silhouette": metric,
                "calinski_harabasz": metric,
                "davies_bouldin": metric,
                "bic": _score_model_information_criterion(model, np.empty((0, 0)), "bic"),
                "aic": _score_model_information_criterion(model, np.empty((0, 0)), "aic"),
                "density_validity": metric_record(METRIC_NOT_APPLICABLE, reason="feature matrix conversion failed"),
            },
        }
    possible, reason, x_eval, labels_eval = _metric_possible(x_full, labels)
    if not possible:
        metric = (
            metric_record(METRIC_DEPENDENCY_MISSING, reason=reason)
            if reason == "sklearn is not available"
            else metric_record(METRIC_NOT_APPLICABLE, reason=reason)
        )
        return {
            "status": metric["status"],
            "metrics": {
                "silhouette": metric,
                "calinski_harabasz": metric,
                "davies_bouldin": metric,
                "bic": _score_model_information_criterion(model, x_full, "bic"),
                "aic": _score_model_information_criterion(model, x_full, "aic"),
                "density_validity": _score_density_metric(
                    x_full,
                    labels,
                    clusterer_family=clusterer_family,
                    density_metric_fn=density_metric_fn,
                ),
            },
        }
    metrics = {
        "silhouette": _call_sklearn_metric("silhouette", silhouette_score, x_eval, labels_eval),
        "calinski_harabasz": _call_sklearn_metric("calinski_harabasz", calinski_harabasz_score, x_eval, labels_eval),
        "davies_bouldin": _call_sklearn_metric("davies_bouldin", davies_bouldin_score, x_eval, labels_eval),
        "bic": _score_model_information_criterion(model, x_full, "bic"),
        "aic": _score_model_information_criterion(model, x_full, "aic"),
        "density_validity": _score_density_metric(
            x_full,
            labels,
            clusterer_family=clusterer_family,
            density_metric_fn=density_metric_fn,
        ),
    }
    status = METRIC_COMPUTED if any(row["status"] == METRIC_COMPUTED for row in metrics.values()) else METRIC_NOT_APPLICABLE
    return {"status": status, "metrics": metrics}


def score_coverage_degeneracy(
    labels: Sequence[object],
    *,
    tiny_cluster_threshold: int = 20,
    noise_label: object = -1,
    unknown_tokens: Sequence[str] = ("unknown",),
    mostly_noise_threshold: float = 0.8,
) -> dict[str, Any]:
    series = _label_series(labels)
    total = int(len(series))
    null_count = 0
    noise_count = 0
    unknown_count = 0
    effective_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    for value in series.tolist():
        key = _label_key(value)
        raw_counts[key] += 1
        if _is_null_label(value):
            null_count += 1
        elif _is_noise_label(value, noise_label=noise_label):
            noise_count += 1
        elif _is_unknown_label(value, unknown_tokens=unknown_tokens):
            unknown_count += 1
        else:
            effective_counts[key] += 1
    sizes = np.asarray(list(effective_counts.values()), dtype=float)
    singleton_rows = sum(int(v) for v in effective_counts.values() if int(v) == 1)
    tiny_rows = sum(int(v) for v in effective_counts.values() if int(v) <= int(tiny_cluster_threshold))
    effective_state_count = int(len(effective_counts))
    noise_share = float(noise_count / total) if total else None
    return {
        "status": METRIC_COMPUTED,
        "metrics": {
            "row_count": int(total),
            "effective_state_count": effective_state_count,
            "raw_state_counts": {str(k): int(v) for k, v in sorted(raw_counts.items())},
            "effective_state_counts": {str(k): int(v) for k, v in sorted(effective_counts.items())},
            "noise_count": int(noise_count),
            "noise_share": noise_share,
            "unknown_count": int(unknown_count),
            "unknown_share": float(unknown_count / total) if total else None,
            "null_count": int(null_count),
            "null_share": float(null_count / total) if total else None,
            "singleton_cluster_share": float(singleton_rows / total) if total else None,
            "tiny_cluster_share": float(tiny_rows / total) if total else None,
            "min_cluster_size": int(sizes.min()) if sizes.size else None,
            "median_cluster_size": float(np.median(sizes)) if sizes.size else None,
            "max_cluster_size": int(sizes.max()) if sizes.size else None,
            "one_cluster_outcome": bool(total > 0 and effective_state_count == 1),
            "all_noise": bool(total > 0 and noise_count == total),
            "mostly_noise": bool(noise_share is not None and noise_share >= float(mostly_noise_threshold)),
            "all_null_or_unknown": bool(total > 0 and effective_state_count == 0 and noise_count == 0),
            "tiny_cluster_threshold": int(tiny_cluster_threshold),
            "mostly_noise_threshold": float(mostly_noise_threshold),
        },
    }


def _align_by_timestamps(
    baseline_labels: Sequence[object],
    comparison_labels: Sequence[object],
    baseline_timestamps: Sequence[object] | None,
    comparison_timestamps: Sequence[object] | None,
) -> tuple[list[object], list[object], dict[str, Any]]:
    if baseline_timestamps is None or comparison_timestamps is None:
        n = min(len(baseline_labels), len(comparison_labels))
        return list(baseline_labels)[:n], list(comparison_labels)[:n], {
            "alignment": "position",
            "overlap_count": int(n),
        }
    baseline = pd.DataFrame({"ts": list(baseline_timestamps), "baseline_label": list(baseline_labels)})
    comparison = pd.DataFrame({"ts": list(comparison_timestamps), "comparison_label": list(comparison_labels)})
    merged = baseline.merge(comparison, on="ts", how="inner").sort_values("ts")
    return merged["baseline_label"].tolist(), merged["comparison_label"].tolist(), {
        "alignment": "timestamp",
        "overlap_count": int(len(merged)),
        "first_overlap_ts": None if merged.empty else _jsonable(merged["ts"].iloc[0]),
        "last_overlap_ts": None if merged.empty else _jsonable(merged["ts"].iloc[-1]),
    }


def _drop_null_pair_labels(a: Sequence[object], b: Sequence[object]) -> tuple[list[object], list[object]]:
    left: list[object] = []
    right: list[object] = []
    for av, bv in zip(a, b):
        if _is_null_label(av) or _is_null_label(bv):
            continue
        left.append(av)
        right.append(bv)
    return left, right


def _metric_distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = np.asarray([float(v) for v in values if _safe_float(v) is not None], dtype=float)
    if clean.size == 0:
        return {"count": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "median": float(np.median(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p90": float(np.percentile(clean, 90)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def compare_labels_on_overlap(
    baseline_labels: Sequence[object],
    comparison_labels: Sequence[object],
    *,
    baseline_timestamps: Sequence[object] | None = None,
    comparison_timestamps: Sequence[object] | None = None,
    comparison_name: str = "comparison",
) -> dict[str, Any]:
    aligned_a, aligned_b, alignment = _align_by_timestamps(
        baseline_labels,
        comparison_labels,
        baseline_timestamps,
        comparison_timestamps,
    )
    clean_a, clean_b = _drop_null_pair_labels(aligned_a, aligned_b)
    overlap = int(len(clean_a))
    base_unique = len({_label_key(v) for v in clean_a})
    comp_unique = len({_label_key(v) for v in clean_b})
    exact_match_share = None if overlap == 0 else float(sum(1 for av, bv in zip(clean_a, clean_b) if av == bv) / overlap)
    base = {
        "comparison_name": str(comparison_name),
        **alignment,
        "valid_overlap_count": int(overlap),
        "baseline_state_count": int(base_unique),
        "comparison_state_count": int(comp_unique),
        "exact_match_share": exact_match_share,
    }
    if overlap < 2:
        return {
            **base,
            "status": METRIC_NOT_APPLICABLE,
            "reason": "fewer than two overlapping non-null labels",
            "ari": None,
            "ami": None,
        }
    if base_unique <= 1 or comp_unique <= 1:
        if not _HAS_SKLEARN:
            return {
                **base,
                "status": "degenerate_one_label",
                "reason": "baseline or comparison has one label state on overlap",
                "ari": None,
                "ami": None,
            }
        status = "degenerate_one_label"
        reason = "baseline or comparison has one label state on overlap"
    elif not _HAS_SKLEARN:
        return {
            **base,
            "status": METRIC_DEPENDENCY_MISSING,
            "reason": "sklearn is not available",
            "ari": None,
            "ami": None,
        }
    else:
        status = METRIC_COMPUTED
        reason = None
    try:
        ari = float(adjusted_rand_score(clean_a, clean_b))  # type: ignore[misc]
    except Exception:
        ari = None
        status = METRIC_FAILED
        reason = "adjusted_rand_score failed"
    try:
        ami = float(adjusted_mutual_info_score(clean_a, clean_b))  # type: ignore[misc]
    except Exception:
        ami = None
        status = METRIC_FAILED
        reason = "adjusted_mutual_info_score failed"
    return {
        **base,
        "status": status,
        "reason": reason,
        "ari": ari,
        "ami": ami,
    }


def score_stability(
    baseline_labels: Sequence[object],
    perturbations: Sequence[Mapping[str, Any]],
    *,
    baseline_timestamps: Sequence[object] | None = None,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for idx, perturbation in enumerate(perturbations):
        labels = perturbation.get("labels", ())
        timestamps = perturbation.get("timestamps")
        comparisons.append(
            compare_labels_on_overlap(
                baseline_labels,
                list(labels),
                baseline_timestamps=baseline_timestamps,
                comparison_timestamps=list(timestamps) if timestamps is not None else None,
                comparison_name=str(perturbation.get("name", f"perturbation_{idx}")),
            )
        )
    ari_values = [row["ari"] for row in comparisons if _safe_float(row.get("ari")) is not None]
    ami_values = [row["ami"] for row in comparisons if _safe_float(row.get("ami")) is not None]
    status = METRIC_COMPUTED if comparisons else METRIC_NOT_APPLICABLE
    if comparisons and all(row["status"] == "degenerate_one_label" for row in comparisons):
        status = "degenerate_one_label"
    return {
        "status": status,
        "comparisons": comparisons,
        "summary": {
            "ari": _metric_distribution(ari_values),
            "ami": _metric_distribution(ami_values),
            "comparison_count": int(len(comparisons)),
            "degenerate_comparison_count": int(sum(1 for row in comparisons if row["status"] == "degenerate_one_label")),
        },
    }


def _find_target_column(frame: pd.DataFrame, candidates: Sequence[str], contains: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return str(column)
    tokens = tuple(str(token).lower() for token in contains)
    for column in frame.columns:
        lowered = str(column).lower()
        if all(token in lowered for token in tokens):
            return str(column)
    return None


def _cohen_d_max(values: np.ndarray, labels: np.ndarray) -> float | None:
    groups = []
    for label in sorted(set(labels.tolist()), key=str):
        group = values[labels == label]
        group = group[np.isfinite(group)]
        if group.size:
            groups.append(group)
    if len(groups) < 2:
        return None
    best = 0.0
    for idx, left in enumerate(groups):
        for right in groups[idx + 1 :]:
            pooled = math.sqrt((float(np.var(left)) + float(np.var(right))) / 2.0)
            if pooled <= 1e-12:
                continue
            best = max(best, abs(float(np.mean(left)) - float(np.mean(right))) / pooled)
    return float(best)


def _kruskal_hook(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    try:
        from scipy.stats import kruskal  # type: ignore
    except Exception:
        return metric_record(METRIC_DEPENDENCY_MISSING, reason="scipy.stats.kruskal is not available")
    groups = []
    for label in sorted(set(labels.tolist()), key=str):
        group = values[labels == label]
        group = group[np.isfinite(group)]
        if group.size:
            groups.append(group)
    if len(groups) < 2:
        return metric_record(METRIC_NOT_APPLICABLE, reason="fewer than two finite label groups")
    try:
        statistic, pvalue = kruskal(*groups)
    except Exception as exc:
        return metric_record(METRIC_FAILED, reason=f"kruskal failed: {exc}")
    return metric_record(METRIC_COMPUTED, {"statistic": float(statistic), "pvalue": float(pvalue)})


def _bootstrap_spread_ci(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if int(reps) <= 0:
        return metric_record(METRIC_NOT_APPLICABLE, reason="bootstrap reps not requested")
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    if values.size < 4 or len(set(labels.tolist())) < 2:
        return metric_record(METRIC_NOT_APPLICABLE, reason="insufficient finite grouped rows")
    rng = np.random.default_rng(int(seed))
    spreads: list[float] = []
    n = int(values.size)
    for _ in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        sample_values = values[idx]
        sample_labels = labels[idx]
        means = [
            float(np.mean(sample_values[sample_labels == label]))
            for label in set(sample_labels.tolist())
            if int(np.sum(sample_labels == label)) > 0
        ]
        if len(means) >= 2:
            spreads.append(float(max(means) - min(means)))
    if not spreads:
        return metric_record(METRIC_NOT_APPLICABLE, reason="no bootstrap samples had two groups")
    arr = np.asarray(spreads, dtype=float)
    return metric_record(
        METRIC_COMPUTED,
        {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
            "reps": int(len(arr)),
        },
    )


def _score_target_separation(
    frame: pd.DataFrame,
    labels: Sequence[object],
    column: str | None,
    *,
    bootstrap_reps: int = 0,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    if column is None:
        return {"status": METRIC_MISSING, "reason": "target column not found", "metrics": {}}
    if column not in frame.columns:
        return {"status": METRIC_MISSING, "reason": f"target column {column!r} not in frame", "metrics": {}}
    label_arr = np.asarray(list(labels), dtype=object)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if values.size != label_arr.size:
        return {"status": METRIC_NOT_APPLICABLE, "reason": "target and label row counts differ", "metrics": {}}
    mask = _valid_scoring_mask(label_arr)
    values = values[mask]
    label_arr = label_arr[mask]
    finite = np.isfinite(values)
    values = values[finite]
    label_arr = label_arr[finite]
    labels_clean = np.asarray([_label_key(v) for v in label_arr], dtype=object)
    if values.size < 2:
        return {"status": METRIC_NOT_APPLICABLE, "reason": "fewer than two finite target rows", "metrics": {}}
    per_label: dict[str, Any] = {}
    for label in sorted(set(labels_clean.tolist()), key=str):
        group = values[labels_clean == label]
        if group.size:
            per_label[str(label)] = {
                "count": int(group.size),
                "mean": float(np.mean(group)),
                "median": float(np.median(group)),
                "std": float(np.std(group, ddof=0)),
            }
    if len(per_label) < 2:
        return {
            "status": METRIC_NOT_APPLICABLE,
            "reason": "fewer than two finite label groups",
            "metrics": {"target_column": column, "per_label": per_label},
        }
    means = [row["mean"] for row in per_label.values()]
    medians = [row["median"] for row in per_label.values()]
    metrics = {
        "target_column": column,
        "per_label": per_label,
        "mean_spread": float(max(means) - min(means)),
        "median_spread": float(max(medians) - min(medians)),
        "max_abs_pairwise_cohen_d": _cohen_d_max(values, labels_clean),
        "kruskal": _kruskal_hook(values, labels_clean),
        "bootstrap_mean_spread_ci": _bootstrap_spread_ci(
            values,
            labels_clean,
            reps=int(bootstrap_reps),
            seed=int(bootstrap_seed),
        ),
    }
    return {"status": METRIC_COMPUTED, "reason": None, "metrics": _jsonable(metrics)}


def score_economic_separability(
    labels: Sequence[object],
    forward_frame: pd.DataFrame | None,
    *,
    bootstrap_reps: int = 0,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    if forward_frame is None:
        return {
            "status": METRIC_NOT_APPLICABLE,
            "metrics": {
                "forward_return": {"status": METRIC_MISSING, "reason": "forward frame not supplied", "metrics": {}},
                "forward_realized_volatility": {"status": METRIC_MISSING, "reason": "forward frame not supplied", "metrics": {}},
                "forward_drawdown": {"status": METRIC_MISSING, "reason": "forward frame not supplied", "metrics": {}},
            },
        }
    frame = forward_frame.copy()
    targets = {
        "forward_return": _find_target_column(
            frame,
            ("future_log_return", "future_return", "forward_return"),
            ("return",),
        ),
        "forward_realized_volatility": _find_target_column(
            frame,
            (
                "future_realized_vol",
                "future_realized_volatility",
                "future_vol",
                "future_volatility",
                "forward_realized_vol",
                "forward_realized_volatility",
                "forward_vol",
                "forward_volatility",
            ),
            ("realized", "vol"),
        ),
        "forward_drawdown": _find_target_column(
            frame,
            ("future_max_drawdown", "future_drawdown", "max_drawdown", "downside_excursion"),
            ("drawdown",),
        ),
    }
    metrics = {
        name: _score_target_separation(
            frame,
            labels,
            column,
            bootstrap_reps=int(bootstrap_reps),
            bootstrap_seed=int(bootstrap_seed),
        )
        for name, column in targets.items()
    }
    status = METRIC_COMPUTED if any(row["status"] == METRIC_COMPUTED for row in metrics.values()) else METRIC_NOT_APPLICABLE
    return {"status": status, "metrics": metrics}


def score_engineering_runtime(runtime_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(runtime_metadata or {})

    def first(*names: str) -> Any:
        for name in names:
            if name in metadata:
                return metadata.get(name)
        return None

    failure = first("failure_reason", "error", "exception")
    metrics = {
        "fit_time_s": _safe_float(first("fit_time_s", "fit_elapsed_s", "elapsed_fit_s")),
        "score_time_s": _safe_float(first("score_time_s", "score_elapsed_s", "elapsed_score_s")),
        "elapsed_s": _safe_float(first("elapsed_s", "total_elapsed_s")),
        "row_count": None if first("row_count", "rows", "rows_fit") is None else int(first("row_count", "rows", "rows_fit")),
        "feature_count": None if first("feature_count", "features") is None else int(first("feature_count", "features")),
        "memory_estimate_mb": _safe_float(first("memory_estimate_mb", "peak_rss_mb", "max_rss_mb")),
        "parquet_read_bytes": None if first("parquet_read_bytes", "read_bytes") is None else int(first("parquet_read_bytes", "read_bytes")),
        "parquet_write_bytes": None if first("parquet_write_bytes", "write_bytes") is None else int(first("parquet_write_bytes", "write_bytes")),
        "retry_count": int(first("retry_count", "retries") or 0),
        "failure_reason": None if failure is None else str(failure),
        "status_metadata": _jsonable(metadata),
    }
    return {
        "status": METRIC_FAILED if failure else METRIC_COMPUTED,
        "metrics": metrics,
    }


@dataclass(frozen=True)
class RegimeTrialScoreInput:
    trial_id: str
    labels: Sequence[object]
    features: Sequence[Sequence[float]] | np.ndarray | None = None
    clusterer_family: str = ""
    model: object | None = None
    forward_frame: pd.DataFrame | None = None
    stability_perturbations: tuple[Mapping[str, Any], ...] = ()
    timestamps: Sequence[object] | None = None
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    bootstrap_reps: int = 0


def score_regime_trial(score_input: RegimeTrialScoreInput) -> dict[str, Any]:
    internal = score_internal_validity(
        score_input.features,
        score_input.labels,
        model=score_input.model,
        clusterer_family=score_input.clusterer_family,
    )
    coverage = score_coverage_degeneracy(score_input.labels)
    stability = score_stability(
        score_input.labels,
        score_input.stability_perturbations,
        baseline_timestamps=score_input.timestamps,
    )
    economic = score_economic_separability(
        score_input.labels,
        score_input.forward_frame,
        bootstrap_reps=int(score_input.bootstrap_reps),
    )
    engineering = score_engineering_runtime(score_input.runtime_metadata)
    return {
        "schema_version": REGIME_SCORING_SCHEMA_VERSION,
        "artifact_kind": "regime_trial_scoreboard",
        "trial_id": str(score_input.trial_id),
        "metric_families": {
            "internal_validity": internal,
            "stability": stability,
            "economic_separability": economic,
            "coverage_degeneracy": coverage,
            "engineering_runtime": engineering,
        },
    }


def score_regime_trial_json(score_input: RegimeTrialScoreInput, **json_kwargs: Any) -> str:
    return json.dumps(score_regime_trial(score_input), sort_keys=True, **json_kwargs)


__all__ = [
    "METRIC_COMPUTED",
    "METRIC_DEPENDENCY_MISSING",
    "METRIC_FAILED",
    "METRIC_MISSING",
    "METRIC_NOT_APPLICABLE",
    "REGIME_SCORING_SCHEMA_VERSION",
    "RegimeTrialScoreInput",
    "compare_labels_on_overlap",
    "metric_record",
    "score_coverage_degeneracy",
    "score_economic_separability",
    "score_engineering_runtime",
    "score_internal_validity",
    "score_regime_trial",
    "score_regime_trial_json",
    "score_stability",
]
