from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.regimes.asset_state_test.adapters import ClusterAssignmentResult, ClusterFitResult
from src.regimes.asset_state_test.contracts import ARTIFACT_NAMES, FlatPreflightResult, TrialConfig, safe_float


def _valid_metric_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels, dtype=int)


def _cluster_size_stats(labels: Sequence[int]) -> dict[str, Any]:
    arr = np.asarray(labels, dtype=int)
    total = int(arr.size)
    counts = Counter(int(v) for v in arr.tolist() if int(v) != -1)
    largest = max(counts.values()) if counts else 0
    return {
        "cluster_count": int(len(counts)),
        "noise_count": int(np.sum(arr == -1)),
        "noise_frac": float(np.sum(arr == -1) / total) if total else 1.0,
        "largest_cluster_frac": float(largest / total) if total else 0.0,
        "cluster_sizes": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def _label_support_stats(labels: Sequence[int], *, tiny_threshold: int = 20) -> dict[str, Any]:
    arr = np.asarray(labels, dtype=int)
    counts = Counter(int(v) for v in arr.tolist() if int(v) != -1)
    values = np.asarray(list(counts.values()), dtype=float)
    if values.size == 0:
        return {
            "label_support_counts": {},
            "label_support_min": None,
            "label_support_median": None,
            "non_noise_label_count": 0,
            "tiny_label_fraction": None,
        }
    tiny = values < float(tiny_threshold)
    return {
        "label_support_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "label_support_min": int(values.min()),
        "label_support_median": float(np.median(values)),
        "non_noise_label_count": int(values.size),
        "tiny_label_fraction": float(tiny.mean()),
    }


def _single_label_asset_warning(
    *,
    included_in_fit: bool,
    row_count: int,
    non_noise_label_count: int,
) -> bool:
    return bool(included_in_fit) and int(row_count) > 0 and int(non_noise_label_count) <= 1


def internal_validity_metrics(x: np.ndarray, labels: Sequence[int]) -> dict[str, Optional[float]]:
    arr = _valid_metric_labels(np.asarray(labels, dtype=int))
    x = np.asarray(x, dtype=float)
    non_noise = arr != -1
    if x.shape[0] != arr.size or arr.size < 3 or int(non_noise.sum()) < 3:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    x_eval = x[non_noise]
    labels_eval = arr[non_noise]
    unique = sorted(set(int(v) for v in labels_eval.tolist()))
    if len(unique) < 2 or len(unique) >= len(labels_eval):
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    metrics: dict[str, Optional[float]] = {}
    try:
        metrics["silhouette"] = float(silhouette_score(x_eval, labels_eval))
    except Exception:
        metrics["silhouette"] = None
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, labels_eval))
    except Exception:
        metrics["calinski_harabasz"] = None
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(x_eval, labels_eval))
    except Exception:
        metrics["davies_bouldin"] = None
    return metrics


def _label_distribution(labels: Sequence[int]) -> dict[str, float]:
    arr = np.asarray(labels, dtype=int)
    total = int(arr.size)
    if total <= 0:
        return {}
    counts = Counter(int(v) for v in arr.tolist())
    return {str(k): float(v / total) for k, v in sorted(counts.items())}


def _label_distribution_delta(train_labels: Sequence[int], eval_labels: Sequence[int]) -> dict[str, float]:
    train_dist = _label_distribution(train_labels)
    eval_dist = _label_distribution(eval_labels)
    keys = sorted(set(train_dist) | set(eval_dist), key=lambda v: int(v))
    return {key: float(eval_dist.get(key, 0.0) - train_dist.get(key, 0.0)) for key in keys}


def _safe_asset_mask(frame: Optional[pd.DataFrame], asset: str) -> Optional[np.ndarray]:
    if frame is None or "asset" not in frame.columns:
        return None
    return (frame["asset"].astype(str) == str(asset)).to_numpy(dtype=bool)


def _per_asset_candidate_diagnostics(
    *,
    flat_results: Sequence[FlatPreflightResult],
    train_frame: Optional[pd.DataFrame],
    train_labels: Sequence[int],
    eval_frame: Optional[pd.DataFrame],
    eval_labels: Optional[Sequence[int]],
) -> list[dict[str, Any]]:
    train_arr = np.asarray(train_labels, dtype=int)
    eval_arr = None if eval_labels is None else np.asarray(eval_labels, dtype=int)
    rows: list[dict[str, Any]] = []
    for result in flat_results:
        asset = str(result.asset)
        train_mask = _safe_asset_mask(train_frame, asset)
        eval_mask = _safe_asset_mask(eval_frame, asset)
        asset_train_labels = np.empty(0, dtype=int)
        asset_eval_labels = np.empty(0, dtype=int)
        asset_eval_frame = None
        if train_mask is not None and train_arr.size == int(train_mask.size):
            asset_train_labels = train_arr[train_mask]
        if eval_arr is not None and eval_mask is not None and eval_arr.size == int(eval_mask.size):
            asset_eval_labels = eval_arr[eval_mask]
            asset_eval_frame = eval_frame.loc[eval_mask].copy() if eval_frame is not None else None
        train_support = _label_support_stats(asset_train_labels)
        eval_support = _label_support_stats(asset_eval_labels)
        train_size = _cluster_size_stats(asset_train_labels)
        eval_size = _cluster_size_stats(asset_eval_labels)
        forward = forward_conditional_separability_metrics(
            eval_frame=asset_eval_frame,
            eval_labels=asset_eval_labels if asset_eval_labels.size else None,
            train_labels=asset_train_labels,
        )
        rows.append(
            {
                "asset": asset,
                "axis": result.axis,
                "band": result.band,
                "preflight_reason_code": result.reason_code,
                "included_in_fit": bool(result.included_in_fit),
                "carried_as": result.carried_as,
                "near_zero_movement_fraction": result.near_zero_movement_fraction,
                "near_flat_fraction_threshold": result.near_flat_fraction_threshold,
                "near_flat_distance_to_threshold": result.near_flat_distance_to_threshold,
                "zero_variance_feature_count": result.zero_variance_feature_count,
                "near_zero_variance_feature_count": result.near_zero_variance_feature_count,
                "train_rows_available": int(result.row_count),
                "train_rows_clustered": int(asset_train_labels.size),
                "eval_rows_assigned": int(asset_eval_labels.size),
                "train_label_support_counts": train_support["label_support_counts"],
                "train_label_support_min": train_support["label_support_min"],
                "train_label_support_median": train_support["label_support_median"],
                "train_non_noise_label_count": train_support["non_noise_label_count"],
                "train_cluster_count": train_size["cluster_count"],
                "train_largest_cluster_frac": train_size["largest_cluster_frac"],
                "train_noise_frac": train_size["noise_frac"] if asset_train_labels.size else None,
                "eval_label_support_counts": eval_support["label_support_counts"],
                "eval_label_support_min": eval_support["label_support_min"],
                "eval_label_support_median": eval_support["label_support_median"],
                "eval_non_noise_label_count": eval_support["non_noise_label_count"],
                "eval_cluster_count": eval_size["cluster_count"],
                "eval_largest_cluster_frac": eval_size["largest_cluster_frac"],
                "eval_noise_frac": eval_size["noise_frac"] if asset_eval_labels.size else None,
                "train_single_label": _single_label_asset_warning(
                    included_in_fit=bool(result.included_in_fit),
                    row_count=int(asset_train_labels.size),
                    non_noise_label_count=int(train_support["non_noise_label_count"]),
                ),
                "eval_single_label": _single_label_asset_warning(
                    included_in_fit=bool(result.included_in_fit),
                    row_count=int(asset_eval_labels.size),
                    non_noise_label_count=int(eval_support["non_noise_label_count"]),
                ),
                "forward_conditional_separability": forward,
                "stability_contribution": {
                    "status": "trial_level_only",
                    "row_bootstrap_ari": None,
                    "walk_forward_ari": None,
                },
                "coverage_flags": {
                    "clusterable_candidate": bool(result.clusterable_candidate),
                    "fit_excluded": not bool(result.included_in_fit),
                    "neutral_flat": result.carried_as == "neutral_flat",
                },
            }
        )
        rows[-1]["asset_balance_warning"] = bool(rows[-1]["train_single_label"] or rows[-1]["eval_single_label"])
        rows[-1]["asset_balance_warning_reason"] = (
            "single_label_train_and_eval"
            if rows[-1]["train_single_label"] and rows[-1]["eval_single_label"]
            else "single_label_train"
            if rows[-1]["train_single_label"]
            else "single_label_eval"
            if rows[-1]["eval_single_label"]
            else None
        )
    return rows


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str], *, contains: Sequence[str] = ()) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return str(column)
    if contains:
        tokens = tuple(str(v).lower() for v in contains)
        for column in frame.columns:
            name = str(column).lower()
            if all(token in name for token in tokens):
                return str(column)
    return None


def _label_mean_spread(frame: pd.DataFrame, labels: np.ndarray, column: Optional[str]) -> tuple[Optional[float], dict[str, float]]:
    if column is None or column not in frame.columns or len(frame) != int(labels.size):
        return None, {}
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & (labels != -1)
    if int(finite.sum()) < 2:
        return None, {}
    means: dict[str, float] = {}
    for label in sorted({int(v) for v in labels[finite].tolist()}):
        mask = finite & (labels == label)
        if int(mask.sum()) > 0:
            means[str(label)] = float(np.nanmean(values[mask]))
    if len(means) < 2:
        return None, means
    spread = float(max(means.values()) - min(means.values()))
    return spread, means


def forward_conditional_separability_metrics(
    *,
    eval_frame: Optional[pd.DataFrame],
    eval_labels: Optional[Sequence[int]],
    train_labels: Sequence[int],
) -> dict[str, Any]:
    if eval_frame is None or eval_labels is None:
        return {
            "status": "not_available",
            "future_mean_return_spread": None,
            "future_abs_return_spread": None,
            "future_drawdown_spread": None,
            "future_runup_spread": None,
            "label_distribution_delta": {},
            "per_label_means": {},
        }
    labels = np.asarray(eval_labels, dtype=int)
    if labels.size == 0 or len(eval_frame) != int(labels.size):
        return {
            "status": "not_evaluable",
            "future_mean_return_spread": None,
            "future_abs_return_spread": None,
            "future_drawdown_spread": None,
            "future_runup_spread": None,
            "label_distribution_delta": _label_distribution_delta(train_labels, labels),
            "per_label_means": {},
        }
    return_col = _first_existing_column(
        eval_frame,
        (
            "future_log_return",
            "future_return",
            "forward_return",
            "future_return_30m",
            "future_log_return_30m",
        ),
        contains=("future", "return"),
    )
    abs_col = _first_existing_column(
        eval_frame,
        ("future_abs_return", "abs_future_return", "future_abs_return_30m"),
        contains=("future", "abs", "return"),
    )
    drawdown_col = _first_existing_column(
        eval_frame,
        ("future_drawdown", "max_drawdown", "future_max_drawdown", "future_drawdown_30m"),
        contains=("drawdown",),
    )
    runup_col = _first_existing_column(
        eval_frame,
        ("future_runup", "max_runup", "future_max_runup", "future_runup_30m"),
        contains=("runup",),
    )
    mean_return_spread, mean_return_by_label = _label_mean_spread(eval_frame, labels, return_col)
    if abs_col is None and return_col is not None and return_col in eval_frame.columns:
        tmp = eval_frame.copy()
        tmp["__future_abs_return_from_return"] = pd.to_numeric(tmp[return_col], errors="coerce").abs()
        abs_return_spread, abs_return_by_label = _label_mean_spread(tmp, labels, "__future_abs_return_from_return")
        abs_col_used = "__abs_from_return__"
    else:
        abs_return_spread, abs_return_by_label = _label_mean_spread(eval_frame, labels, abs_col)
        abs_col_used = abs_col
    drawdown_spread, drawdown_by_label = _label_mean_spread(eval_frame, labels, drawdown_col)
    runup_spread, runup_by_label = _label_mean_spread(eval_frame, labels, runup_col)
    any_metric = any(v is not None for v in (mean_return_spread, abs_return_spread, drawdown_spread, runup_spread))
    return {
        "status": "computed" if any_metric else "no_forward_columns",
        "future_mean_return_spread": mean_return_spread,
        "future_abs_return_spread": abs_return_spread,
        "future_drawdown_spread": drawdown_spread,
        "future_runup_spread": runup_spread,
        "label_distribution_delta": _label_distribution_delta(train_labels, labels),
        "columns_used": {
            "future_return": return_col,
            "future_abs_return": abs_col_used,
            "future_drawdown": drawdown_col,
            "future_runup": runup_col,
        },
        "per_label_means": {
            "future_return": mean_return_by_label,
            "future_abs_return": abs_return_by_label,
            "future_drawdown": drawdown_by_label,
            "future_runup": runup_by_label,
        },
    }


def build_cluster_diagnostics(
    *,
    trial_config: TrialConfig,
    fit_result: Optional[ClusterFitResult],
    x: np.ndarray,
    flat_results: Sequence[FlatPreflightResult],
    preprocess_metadata: Mapping[str, Any],
    elapsed_s: float,
    rows_read: int = 0,
    files_read: int = 0,
    error: Optional[str] = None,
    eval_x: Optional[np.ndarray] = None,
    eval_frame: Optional[pd.DataFrame] = None,
    eval_assignment: Optional[ClusterAssignmentResult] = None,
    stability_metrics: Optional[Mapping[str, Any]] = None,
    train_frame: Optional[pd.DataFrame] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    labels = np.asarray([], dtype=int) if fit_result is None else np.asarray(fit_result.labels, dtype=int)
    size_stats = _cluster_size_stats(labels)
    internal = internal_validity_metrics(x, labels)
    fit_meta = {} if fit_result is None else dict(fit_result.fitted_metadata)
    probabilities = None if fit_result is None else fit_result.probabilities
    confidence = None if fit_result is None else fit_result.confidence
    density = {
        "noise_fraction": size_stats["noise_frac"],
        "probability_mean": None,
        "probability_p10": None,
        "probability_p50": None,
        "probability_p90": None,
        "cluster_persistence": fit_meta.get("cluster_persistence"),
        "reachability_mean": fit_meta.get("reachability_mean"),
        "reachability_p90": fit_meta.get("reachability_p90"),
    }
    if confidence is not None:
        finite = np.asarray(confidence, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            density["probability_mean"] = float(finite.mean())
            density["probability_p10"] = float(np.quantile(finite, 0.10))
            density["probability_p50"] = float(np.quantile(finite, 0.50))
            density["probability_p90"] = float(np.quantile(finite, 0.90))
    reason_counts = Counter(r.reason_code for r in flat_results)
    included_assets = [str(r.asset) for r in flat_results if r.included_in_fit]
    excluded_assets = [str(r.asset) for r in flat_results if not r.included_in_fit]
    eval_labels = None if eval_assignment is None else np.asarray(eval_assignment.labels, dtype=int)
    eval_rows = 0 if eval_x is None else int(np.asarray(eval_x).shape[0])
    eval_assigned = 0 if eval_labels is None else int(eval_labels.size)
    eval_noise = 0 if eval_labels is None else int(np.sum(eval_labels == -1))
    train_support = _label_support_stats(labels)
    eval_support = _label_support_stats([] if eval_labels is None else eval_labels)
    density.update(
        {
            "train_label_support_counts": train_support["label_support_counts"],
            "train_label_support_min": train_support["label_support_min"],
            "train_label_support_median": train_support["label_support_median"],
            "train_non_noise_label_count": train_support["non_noise_label_count"],
            "train_tiny_label_fraction": train_support["tiny_label_fraction"],
            "eval_label_support_counts": eval_support["label_support_counts"],
            "eval_label_support_min": eval_support["label_support_min"],
            "eval_label_support_median": eval_support["label_support_median"],
            "eval_non_noise_label_count": eval_support["non_noise_label_count"],
            "eval_tiny_label_fraction": eval_support["tiny_label_fraction"],
        }
    )
    eval_assignment_status = "not_requested"
    eval_assignment_supported = None
    eval_assignment_error = None
    eval_assignment_metadata: Mapping[str, Any] = {}
    eval_probability_summary = {
        "probability_mean": None,
        "probability_p10": None,
        "probability_p50": None,
        "probability_p90": None,
    }
    if eval_assignment is not None:
        eval_assignment_status = eval_assignment.status
        eval_assignment_supported = bool(eval_assignment.supported)
        eval_assignment_error = eval_assignment.error
        eval_assignment_metadata = dict(eval_assignment.metadata)
        if eval_assignment.confidence is not None:
            finite_eval_conf = np.asarray(eval_assignment.confidence, dtype=float)
            finite_eval_conf = finite_eval_conf[np.isfinite(finite_eval_conf)]
            if finite_eval_conf.size:
                eval_probability_summary = {
                    "probability_mean": float(finite_eval_conf.mean()),
                    "probability_p10": float(np.quantile(finite_eval_conf, 0.10)),
                    "probability_p50": float(np.quantile(finite_eval_conf, 0.50)),
                    "probability_p90": float(np.quantile(finite_eval_conf, 0.90)),
                }
    stability = {
        "status": "placeholder",
        "seed_perturbation_ari": None,
        "row_bootstrap_ari": None,
        "feature_perturbation_nmi": None,
        "feature_perturbation_nmi_status": "unsupported_bounded_cycle7",
        "feature_perturbation_nmi_error": None,
        "row_bootstrap_repeat_count": None,
        "row_bootstrap_sample_frac": None,
        "row_bootstrap_assignment_method": None,
        "row_bootstrap_error": None,
        "walk_forward_split_count": 0,
        "walk_forward_assignment_method": None,
        "walk_forward_ari_mean": None,
        "walk_forward_ari_min": None,
        "walk_forward_label_flip_rate": None,
        "walk_forward_status": "not_requested",
        "walk_forward_error": None,
    }
    if stability_metrics:
        stability.update(dict(stability_metrics))
    runtime_meta = dict(runtime_metadata or {})
    forward = forward_conditional_separability_metrics(
        eval_frame=eval_frame,
        eval_labels=eval_labels,
        train_labels=labels,
    )
    per_asset_diagnostics = _per_asset_candidate_diagnostics(
        flat_results=flat_results,
        train_frame=train_frame,
        train_labels=labels,
        eval_frame=eval_frame,
        eval_labels=eval_labels,
    )
    single_label_assets_train = sorted(
        str(row["asset"]) for row in per_asset_diagnostics if bool(row.get("train_single_label"))
    )
    single_label_assets_eval = sorted(
        str(row["asset"]) for row in per_asset_diagnostics if bool(row.get("eval_single_label"))
    )
    asset_balance_warning = bool(single_label_assets_train or single_label_assets_eval)
    return {
        "trial_metadata": trial_config.to_dict(),
        "status": "failed" if error else "ok",
        "error": error,
        "internal_validity": {
            **internal,
            "bic": safe_float(fit_meta.get("bic")),
            "aic": safe_float(fit_meta.get("aic")),
            "inertia": safe_float(fit_meta.get("inertia")),
            **size_stats,
        },
        "density_validity": density,
        "stability": stability,
        "forward_conditional_separability": forward,
        "asset_balance": {
            "train_single_label_asset_count": int(len(single_label_assets_train)),
            "eval_single_label_asset_count": int(len(single_label_assets_eval)),
            "single_label_assets_train": single_label_assets_train,
            "single_label_assets_eval": single_label_assets_eval,
            "asset_balance_warning": asset_balance_warning,
            "asset_balance_warning_reason": "single_label_asset_collapse" if asset_balance_warning else None,
        },
        "coverage": {
            "rows_fit": int(labels.size),
            "rows_read": int(rows_read),
            "files_read": int(files_read),
            "fit_coverage": float(labels.size / rows_read) if int(rows_read) > 0 else None,
            "unknown_frac": None,
            "no_model_frac": 1.0 if fit_result is None else 0.0,
            "assignment_coverage": 1.0 if fit_result is not None and labels.size > 0 else 0.0,
            "eval_rows": eval_rows,
            "eval_assigned_rows": eval_assigned,
            "eval_assignment_coverage": float(eval_assigned / eval_rows) if eval_rows > 0 else None,
            "eval_noise_count": eval_noise,
            "eval_noise_frac": float(eval_noise / eval_assigned) if eval_assigned > 0 else None,
            "eval_assignment_status": eval_assignment_status,
            "eval_assignment_supported": eval_assignment_supported,
            "eval_assignment_error": eval_assignment_error,
            "eval_assignment_metadata": eval_assignment_metadata,
            "eval_assignment_probability_summary": eval_probability_summary,
        },
        "flat_asset_safety": {
            "reason_code_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
            "assets_included": included_assets,
            "assets_excluded": excluded_assets,
            "flat_excluded_count": int(sum(1 for r in flat_results if r.reason_code in {"valid_flat_or_pegged", "low_activity", "insufficient_variance"})),
            "flat_neutral_count": int(sum(1 for r in flat_results if r.carried_as == "neutral_flat")),
            "axis_not_clusterable_count": int(reason_counts.get("axis_not_clusterable", 0)),
            "bad_or_missing_data_count": int(reason_counts.get("bad_or_missing_data", 0)),
        },
        "runtime_io": {
            "elapsed_s": float(elapsed_s),
            "adapter_elapsed_s": None if fit_result is None else safe_float(fit_result.runtime_stats.get("fit_elapsed_s")),
            "workers": int(trial_config.study.workers),
            "cpu_count": int(os.cpu_count() or 1),
            "files_read": int(files_read),
            "rows_read": int(rows_read),
            "peak_rss_mb": safe_float(runtime_meta.get("peak_rss_mb")),
            "peak_rss_status": runtime_meta.get("peak_rss_status", "unavailable_bounded_harness"),
            "peak_rss_source": runtime_meta.get("peak_rss_source"),
            "peak_rss_unavailable_reason": runtime_meta.get("peak_rss_unavailable_reason"),
        },
        "interpretability": {
            "features_used": list(preprocess_metadata.get("selected_columns", [])),
            "dropped_features": list(preprocess_metadata.get("dropped_columns", [])),
            "feature_count": int(preprocess_metadata.get("feature_count", 0) or 0),
            "intended_feature_bases": list(preprocess_metadata.get("intended_feature_bases", [])),
            "selected_feature_bases": list(preprocess_metadata.get("selected_feature_bases", [])),
            "selected_feature_columns": list(preprocess_metadata.get("selected_feature_columns", [])),
            "missing_intended_bases": list(preprocess_metadata.get("missing_intended_bases", [])),
            "feature_availability_status": preprocess_metadata.get("feature_availability_status"),
            "missing_intended_feature_reasons": dict(preprocess_metadata.get("missing_intended_feature_reasons", {}) or {}),
            "preprocessed_selected_columns": list(preprocess_metadata.get("preprocessed_selected_columns", [])),
            "preprocessed_dropped_columns": list(preprocess_metadata.get("preprocessed_dropped_columns", [])),
            "preprocess_metadata": dict(preprocess_metadata),
            "label_mapping_basis": "unmapped_cluster_ids",
            "config_hash": trial_config.config_hash,
            "probability_shape": None if probabilities is None else list(np.asarray(probabilities).shape),
        },
        "method_metadata": {
            "method": trial_config.method,
            "method_params": dict(fit_result.method_params) if fit_result is not None else dict(trial_config.method_params),
            "fitted_metadata": fit_meta,
        },
        "per_asset_diagnostics": per_asset_diagnostics,
    }


def build_candidate_score_row(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    trial = dict(diagnostics.get("trial_metadata", {}))
    study = dict(trial.get("study", {}))
    method_params = dict(trial.get("method_params", {}) or {})
    internal = dict(diagnostics.get("internal_validity", {}))
    density = dict(diagnostics.get("density_validity", {}))
    coverage = dict(diagnostics.get("coverage", {}))
    flat = dict(diagnostics.get("flat_asset_safety", {}))
    runtime = dict(diagnostics.get("runtime_io", {}))
    forward = dict(diagnostics.get("forward_conditional_separability", {}))
    stability = dict(diagnostics.get("stability", {}))
    asset_balance = dict(diagnostics.get("asset_balance", {}))
    interpretability = dict(diagnostics.get("interpretability", {}))
    preprocess_metadata = dict(interpretability.get("preprocess_metadata", {}) or {})
    row = {
        "layer": study.get("layer"),
        "axis": study.get("axis"),
        "band": study.get("band"),
        "method": trial.get("method"),
        "preprocess": trial.get("preprocess"),
        "feature_strategy": trial.get("feature_strategy"),
        "feature_count": interpretability.get("feature_count"),
        "features_used_json": json.dumps(interpretability.get("features_used", []), sort_keys=True),
        "dropped_features_json": json.dumps(interpretability.get("dropped_features", []), sort_keys=True),
        "intended_feature_bases_json": json.dumps(interpretability.get("intended_feature_bases", []), sort_keys=True),
        "selected_feature_bases_json": json.dumps(interpretability.get("selected_feature_bases", []), sort_keys=True),
        "selected_feature_columns_json": json.dumps(interpretability.get("selected_feature_columns", []), sort_keys=True),
        "missing_intended_bases_json": json.dumps(interpretability.get("missing_intended_bases", []), sort_keys=True),
        "feature_availability_status": interpretability.get("feature_availability_status"),
        "missing_intended_feature_reasons_json": json.dumps(interpretability.get("missing_intended_feature_reasons", {}), sort_keys=True),
        "preprocessed_selected_columns_json": json.dumps(interpretability.get("preprocessed_selected_columns", []), sort_keys=True),
        "clipper": preprocess_metadata.get("clipper"),
        "clip_bounds_json": json.dumps(preprocess_metadata.get("clip_bounds", {}), sort_keys=True),
        "asset_count": len(study.get("assets", []) or []),
        "row_count": coverage.get("rows_read"),
        "cluster_count": internal.get("cluster_count"),
        "largest_cluster_frac": internal.get("largest_cluster_frac"),
        "noise_frac": internal.get("noise_frac"),
        "unknown_frac": coverage.get("unknown_frac"),
        "no_model_frac": coverage.get("no_model_frac"),
        "silhouette": internal.get("silhouette"),
        "calinski_harabasz": internal.get("calinski_harabasz"),
        "davies_bouldin": internal.get("davies_bouldin"),
        "bic": internal.get("bic"),
        "aic": internal.get("aic"),
        "forward_mean_return_spread": forward.get("future_mean_return_spread"),
        "forward_abs_return_spread": forward.get("future_abs_return_spread"),
        "forward_drawdown_spread": forward.get("future_drawdown_spread"),
        "forward_runup_spread": forward.get("future_runup_spread"),
        "forward_status": forward.get("status"),
        "stability_ari": stability.get("seed_perturbation_ari"),
        "stability_nmi": stability.get("feature_perturbation_nmi"),
        "feature_perturbation_nmi_status": stability.get("feature_perturbation_nmi_status"),
        "feature_perturbation_nmi_error": stability.get("feature_perturbation_nmi_error"),
        "stability_status": stability.get("status"),
        "walk_forward_split_count": stability.get("walk_forward_split_count"),
        "walk_forward_assignment_method": stability.get("walk_forward_assignment_method"),
        "walk_forward_ari_mean": stability.get("walk_forward_ari_mean"),
        "walk_forward_ari_min": stability.get("walk_forward_ari_min"),
        "walk_forward_label_flip_rate": stability.get("walk_forward_label_flip_rate"),
        "walk_forward_status": stability.get("walk_forward_status"),
        "walk_forward_error": stability.get("walk_forward_error"),
        "eval_rows": coverage.get("eval_rows"),
        "eval_assigned_rows": coverage.get("eval_assigned_rows"),
        "eval_assignment_coverage": coverage.get("eval_assignment_coverage"),
        "eval_noise_frac": coverage.get("eval_noise_frac"),
        "eval_assignment_status": coverage.get("eval_assignment_status"),
        "eval_assignment_supported": coverage.get("eval_assignment_supported"),
        "train_single_label_asset_count": asset_balance.get("train_single_label_asset_count"),
        "eval_single_label_asset_count": asset_balance.get("eval_single_label_asset_count"),
        "single_label_assets_train": "|".join(str(v) for v in (asset_balance.get("single_label_assets_train", []) or [])),
        "single_label_assets_eval": "|".join(str(v) for v in (asset_balance.get("single_label_assets_eval", []) or [])),
        "asset_balance_warning": asset_balance.get("asset_balance_warning"),
        "flat_excluded_count": flat.get("flat_excluded_count"),
        "flat_neutral_count": flat.get("flat_neutral_count"),
        "axis_not_clusterable_count": flat.get("axis_not_clusterable_count"),
        "elapsed_s": runtime.get("elapsed_s"),
        "max_rss_mb": runtime.get("peak_rss_mb"),
        "max_rss_status": runtime.get("peak_rss_status"),
        "max_rss_source": runtime.get("peak_rss_source"),
        "max_rss_unavailable_reason": runtime.get("peak_rss_unavailable_reason"),
        "files_read": runtime.get("files_read"),
        "rows_read": runtime.get("rows_read"),
        "output_bytes": None,
        "diagnostics_bytes": None,
        "trial_config_hash": trial.get("trial_config_hash"),
        "status": diagnostics.get("status"),
        "trial_id": trial.get("trial_id"),
        "grid_family": trial.get("grid_family"),
        "grid_variant_id": trial.get("grid_variant_id"),
        "method_params_json": json.dumps(method_params, sort_keys=True, separators=(",", ":")),
        "fit_rows": coverage.get("rows_fit"),
        "evaluated_rows": coverage.get("eval_rows"),
        "assets_included": "|".join(str(v) for v in (flat.get("assets_included", []) or [])),
        "assets_excluded": "|".join(str(v) for v in (flat.get("assets_excluded", []) or [])),
        "flat_reason_counts_json": json.dumps(flat.get("reason_code_counts", {}), sort_keys=True),
        "seed_perturbation_ari": stability.get("seed_perturbation_ari"),
        "subsample_refit_ari": stability.get("subsample_refit_ari"),
        "row_bootstrap_ari": stability.get("row_bootstrap_ari"),
        "row_bootstrap_repeat_count": stability.get("row_bootstrap_repeat_count"),
        "row_bootstrap_sample_frac": stability.get("row_bootstrap_sample_frac"),
        "row_bootstrap_assignment_method": stability.get("row_bootstrap_assignment_method"),
        "row_bootstrap_error": stability.get("row_bootstrap_error"),
        "feature_perturbation_nmi_repeat_count": stability.get("feature_perturbation_nmi_repeat_count"),
        "feature_perturbation_feature_fraction": stability.get("feature_perturbation_feature_fraction"),
        "feature_perturbation_strategy": stability.get("feature_perturbation_strategy"),
        "stability_repeat_count": stability.get("stability_repeat_count"),
        "stability_sample_frac": stability.get("stability_sample_frac"),
        "stability_assignment_method": stability.get("stability_assignment_method"),
        "stability_error": stability.get("stability_error"),
        "train_label_support_min": density.get("train_label_support_min"),
        "train_label_support_median": density.get("train_label_support_median"),
        "eval_label_support_min": density.get("eval_label_support_min"),
        "eval_label_support_median": density.get("eval_label_support_median"),
        "tiny_label_fraction": density.get("train_tiny_label_fraction"),
        "non_noise_label_count": density.get("train_non_noise_label_count"),
        "per_asset_diagnostics_json": json.dumps(diagnostics.get("per_asset_diagnostics", []), sort_keys=True),
    }
    for key, value in sorted(method_params.items()):
        safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(key).strip().lower()).strip("_")
        row[f"param_{safe_key}"] = value
    return row


def build_per_asset_summary_rows(cluster_diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostics in cluster_diagnostics:
        trial = dict(diagnostics.get("trial_metadata", {}))
        study = dict(trial.get("study", {}))
        coverage = dict(diagnostics.get("coverage", {}))
        stability = dict(diagnostics.get("stability", {}))
        asset_balance = dict(diagnostics.get("asset_balance", {}))
        method_role = method_role_for_trial(trial)
        for asset_row in list(diagnostics.get("per_asset_diagnostics", []) or []):
            asset = dict(asset_row)
            forward = dict(asset.get("forward_conditional_separability", {}) or {})
            coverage_flags = dict(asset.get("coverage_flags", {}) or {})
            rows.append(
                {
                    "layer": study.get("layer"),
                    "axis": study.get("axis"),
                    "band": study.get("band"),
                    "method": trial.get("method"),
                    "method_role": method_role,
                    "preprocess": trial.get("preprocess"),
                    "feature_strategy": trial.get("feature_strategy"),
                    "trial_id": trial.get("trial_id"),
                    "grid_family": trial.get("grid_family"),
                    "grid_variant_id": trial.get("grid_variant_id"),
                    "asset": asset.get("asset"),
                    "preflight_reason_code": asset.get("preflight_reason_code"),
                    "included_in_fit": asset.get("included_in_fit"),
                    "carried_as": asset.get("carried_as"),
                    "near_zero_movement_fraction": asset.get("near_zero_movement_fraction"),
                    "near_flat_fraction_threshold": asset.get("near_flat_fraction_threshold"),
                    "near_flat_distance_to_threshold": asset.get("near_flat_distance_to_threshold"),
                    "zero_variance_feature_count": asset.get("zero_variance_feature_count"),
                    "near_zero_variance_feature_count": asset.get("near_zero_variance_feature_count"),
                    "clusterable_candidate": coverage_flags.get("clusterable_candidate"),
                    "train_rows_available": asset.get("train_rows_available"),
                    "train_rows_clustered": asset.get("train_rows_clustered"),
                    "eval_rows_assigned": asset.get("eval_rows_assigned"),
                    "train_cluster_count": asset.get("train_cluster_count"),
                    "eval_cluster_count": asset.get("eval_cluster_count"),
                    "train_non_noise_label_count": asset.get("train_non_noise_label_count"),
                    "eval_non_noise_label_count": asset.get("eval_non_noise_label_count"),
                    "train_largest_cluster_frac": asset.get("train_largest_cluster_frac"),
                    "eval_largest_cluster_frac": asset.get("eval_largest_cluster_frac"),
                    "train_noise_frac": asset.get("train_noise_frac"),
                    "eval_noise_frac": asset.get("eval_noise_frac"),
                    "train_single_label": asset.get("train_single_label"),
                    "eval_single_label": asset.get("eval_single_label"),
                    "asset_balance_warning": asset.get("asset_balance_warning"),
                    "asset_balance_warning_reason": asset.get("asset_balance_warning_reason"),
                    "candidate_asset_balance_warning": asset_balance.get("asset_balance_warning"),
                    "forward_status": forward.get("status"),
                    "forward_abs_return_spread": forward.get("future_abs_return_spread"),
                    "forward_mean_return_spread": forward.get("future_mean_return_spread"),
                    "eval_assignment_status": coverage.get("eval_assignment_status"),
                    "feature_perturbation_nmi_status": stability.get("feature_perturbation_nmi_status"),
                }
            )
    return rows


def _bool_count(values: Sequence[Any]) -> int:
    return int(sum(1 for value in values if bool(value)))


def _json_counter(values: Sequence[Any]) -> str:
    return json.dumps({str(k): int(v) for k, v in sorted(Counter(str(v) for v in values if v is not None).items())}, sort_keys=True)


def _finite_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = safe_float(value)
        if number is not None:
            out.append(float(number))
    return out


def _first_present(values: Sequence[Any]) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return None


def method_role_for_trial(trial: Mapping[str, Any]) -> str:
    """Classify retained bounded-study arms by decision authority."""
    method = str(trial.get("method") or "").strip().lower()
    variant = str(trial.get("grid_variant_id") or "").strip().lower()
    if method == "gaussian_mixture" or "gaussian_mixture" in variant:
        return "primary_comparator"
    if method == "kmeans" or "kmeans" in variant:
        return "sentinel"
    if method == "hdbscan" or "hdbscan" in variant:
        return "diagnostic_only"
    return "exploratory"


def _role_summary(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    role_rows = [row for row in rows if str(row.get("method_role") or "") == str(role)]
    trial_ids = sorted({str(row.get("trial_id")) for row in role_rows if row.get("trial_id") is not None})
    observed = int(len(trial_ids) or len(role_rows))
    eval_single_count = _bool_count([row.get("eval_single_label") for row in role_rows])
    train_single_count = _bool_count([row.get("train_single_label") for row in role_rows])
    eval_support = _finite_values([row.get("eval_non_noise_label_count") for row in role_rows])
    train_support = _finite_values([row.get("train_non_noise_label_count") for row in role_rows])
    return {
        "trial_count": observed,
        "train_single_label_trial_count": train_single_count,
        "eval_single_label_trial_count": eval_single_count,
        "train_collapse_rate": float(train_single_count / observed) if observed else None,
        "eval_collapse_rate": float(eval_single_count / observed) if observed else None,
        "eval_support_min": int(min(eval_support)) if eval_support else None,
        "eval_support_max": int(max(eval_support)) if eval_support else None,
        "eval_support_values": [int(v) for v in eval_support],
        "train_support_min": int(min(train_support)) if train_support else None,
        "train_support_max": int(max(train_support)) if train_support else None,
    }


def _role_has_multilabel_support(summary: Mapping[str, Any]) -> bool:
    return (safe_float(summary.get("eval_support_max"), default=0.0) or 0.0) >= 2.0


def _role_is_full_eval_collapse(summary: Mapping[str, Any]) -> bool:
    trial_count = int(summary.get("trial_count") or 0)
    collapse_rate = safe_float(summary.get("eval_collapse_rate"))
    support_max = safe_float(summary.get("eval_support_max"), default=0.0) or 0.0
    return trial_count > 0 and collapse_rate is not None and collapse_rate >= 1.0 and support_max <= 1.0


def _asset_decision_for_row(row: Mapping[str, Any]) -> tuple[str, str, str]:
    reason = str(row.get("preflight_reason_code") or "")
    carried_as = str(row.get("carried_as") or "")
    observed = int(row.get("candidate_trials_observed") or 0)
    collapse_rate = safe_float(row.get("retained_method_eval_collapse_rate"))
    support_max = row.get("eval_non_noise_label_support_max")
    support_max_number = safe_float(support_max, default=0.0) or 0.0
    included = bool(row.get("included_in_fit"))
    primary_trials = int(row.get("primary_comparator_trial_count") or 0)
    primary_support_max = safe_float(row.get("primary_comparator_eval_support_max"), default=0.0) or 0.0
    primary_collapse_rate = safe_float(row.get("primary_comparator_eval_collapse_rate"))
    diagnostic_support_max = safe_float(row.get("diagnostic_only_eval_support_max"), default=0.0) or 0.0
    sentinel_support_max = safe_float(row.get("sentinel_eval_support_max"), default=0.0) or 0.0

    if reason == "valid_flat_or_pegged" or carried_as == "neutral_flat":
        return "neutral_flat", "valid_flat_or_pegged", "candidate_only_no_production_label_change"
    if reason == "bad_or_missing_data":
        return "data_quality_no_model", "bad_or_missing_data", "candidate_only_no_production_label_change"
    if reason in {"insufficient_variance", "low_activity"}:
        return "data_quality_no_model", reason, "candidate_only_no_production_label_change"
    if reason == "axis_not_clusterable":
        return "candidate_axis_not_clusterable", "axis_not_clusterable_preflight", "candidate_only_no_production_label_change"
    if observed <= 0 or not included:
        return "needs_more_evidence", "no_candidate_trials_observed", "candidate_only_no_production_label_change"
    if primary_trials > 0:
        if primary_collapse_rate is not None and primary_collapse_rate >= 1.0 and primary_support_max <= 1.0:
            if diagnostic_support_max >= 2.0:
                return (
                    "candidate_axis_not_clusterable",
                    "primary_comparator_eval_collapse_diagnostic_only_support_discounted",
                    "candidate_only_diagnostic_support_warning_no_production_label_change",
                )
            return "candidate_axis_not_clusterable", "primary_comparator_eval_single_label_collapse", "candidate_only_no_production_label_change"
        if primary_support_max >= 2.0:
            return "cluster_candidate", "primary_comparator_multi_label_support_observed", "candidate_only_no_production_label_change"
        if diagnostic_support_max >= 2.0 or sentinel_support_max >= 2.0:
            return "needs_more_evidence", "primary_comparator_ambiguous_non_primary_support_only", "candidate_only_no_production_label_change"
    if collapse_rate is not None and collapse_rate >= 1.0 and support_max_number <= 1.0:
        return "candidate_axis_not_clusterable", "clean_eval_single_label_collapse_across_retained_methods", "candidate_only_no_production_label_change"
    if support_max_number >= 2.0:
        return "cluster_candidate", "multi_label_support_observed", "candidate_only_no_production_label_change"
    return "needs_more_evidence", "incomplete_or_ambiguous_candidate_evidence", "candidate_only_no_production_label_change"


def build_asset_model_decision_rows(per_asset_summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate trial-level per-asset diagnostics into candidate-vs-no-model evidence rows."""
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for raw_row in per_asset_summary:
        row = dict(raw_row)
        key = (
            str(row.get("layer") or "asset_state"),
            str(row.get("axis") or ""),
            str(row.get("band") or ""),
            str(row.get("asset") or ""),
        )
        if key[-1]:
            grouped.setdefault(key, []).append(row)

    decisions: list[dict[str, Any]] = []
    for (layer, axis, band, asset), rows in sorted(grouped.items(), key=lambda item: item[0]):
        trial_ids = sorted({str(row.get("trial_id")) for row in rows if row.get("trial_id") is not None})
        observed = int(len(trial_ids) or len(rows))
        train_single_count = _bool_count([row.get("train_single_label") for row in rows])
        eval_single_count = _bool_count([row.get("eval_single_label") for row in rows])
        eval_support = _finite_values([row.get("eval_non_noise_label_count") for row in rows])
        train_support = _finite_values([row.get("train_non_noise_label_count") for row in rows])
        forward_spreads = _finite_values([row.get("forward_abs_return_spread") for row in rows])
        forward_statuses = [row.get("forward_status") for row in rows]
        near_zero_fractions = _finite_values([row.get("near_zero_movement_fraction") for row in rows])
        near_flat_distances = _finite_values([row.get("near_flat_distance_to_threshold") for row in rows])
        zero_variance_counts = _finite_values([row.get("zero_variance_feature_count") for row in rows])
        near_zero_variance_counts = _finite_values([row.get("near_zero_variance_feature_count") for row in rows])
        preflight_reason = str(_first_present([row.get("preflight_reason_code") for row in rows]) or "")
        carried_as = str(_first_present([row.get("carried_as") for row in rows]) or "")
        included = bool(_first_present([row.get("included_in_fit") for row in rows]))
        clusterable = bool(_first_present([row.get("clusterable_candidate") for row in rows]))
        roles = [str(row.get("method_role") or method_role_for_trial(row)) for row in rows]
        role_counts = Counter(roles)
        primary = _role_summary(rows, "primary_comparator")
        sentinel = _role_summary(rows, "sentinel")
        diagnostic = _role_summary(rows, "diagnostic_only")
        primary_support = _role_has_multilabel_support(primary)
        sentinel_support = _role_has_multilabel_support(sentinel)
        diagnostic_support = _role_has_multilabel_support(diagnostic)
        primary_full_collapse = _role_is_full_eval_collapse(primary)
        primary_collapse_summary = (
            f"{primary['eval_single_label_trial_count']}/{primary['trial_count']} primary comparator eval single-label trials; "
            f"support range {primary['eval_support_min']}..{primary['eval_support_max']}"
            if int(primary["trial_count"]) > 0
            else "0/0 primary comparator trials observed"
        )

        if any(status == "computed" for status in forward_statuses):
            forward_status = "computed"
        elif forward_statuses:
            forward_status = str(_first_present(forward_statuses) or "not_available")
        else:
            forward_status = "not_available"

        row = {
            "layer": layer,
            "axis": axis,
            "band": band,
            "asset": asset,
            "preflight_reason_code": preflight_reason,
            "carried_as": carried_as,
            "included_in_fit": included,
            "clusterable_candidate": clusterable,
            "candidate_trials_observed": observed,
            "train_single_label_trial_count": train_single_count,
            "eval_single_label_trial_count": eval_single_count,
            "retained_method_train_collapse_rate": float(train_single_count / observed) if observed else None,
            "retained_method_eval_collapse_rate": float(eval_single_count / observed) if observed else None,
            "eval_non_noise_label_support_min": int(min(eval_support)) if eval_support else None,
            "eval_non_noise_label_support_max": int(max(eval_support)) if eval_support else None,
            "eval_non_noise_label_support_values_json": json.dumps([int(v) for v in eval_support], sort_keys=True),
            "train_non_noise_label_support_min": int(min(train_support)) if train_support else None,
            "train_non_noise_label_support_max": int(max(train_support)) if train_support else None,
            "near_zero_movement_fraction": max(near_zero_fractions) if near_zero_fractions else None,
            "near_flat_distance_to_threshold": min(near_flat_distances) if near_flat_distances else None,
            "zero_variance_feature_count": int(max(zero_variance_counts)) if zero_variance_counts else None,
            "near_zero_variance_feature_count": int(max(near_zero_variance_counts)) if near_zero_variance_counts else None,
            "forward_separability_status": forward_status,
            "forward_status_counts_json": _json_counter(forward_statuses),
            "forward_abs_return_spread_max": max(forward_spreads) if forward_spreads else None,
            "method_role_counts_json": json.dumps({str(k): int(v) for k, v in sorted(role_counts.items())}, sort_keys=True),
            "primary_comparator_trial_count": primary["trial_count"],
            "primary_comparator_eval_single_label_trial_count": primary["eval_single_label_trial_count"],
            "primary_comparator_eval_collapse_rate": primary["eval_collapse_rate"],
            "primary_comparator_eval_support_min": primary["eval_support_min"],
            "primary_comparator_eval_support_max": primary["eval_support_max"],
            "primary_comparator_eval_support_values_json": json.dumps(primary["eval_support_values"], sort_keys=True),
            "primary_comparator_eval_full_collapse": primary_full_collapse,
            "primary_comparator_eval_collapse_summary": primary_collapse_summary,
            "sentinel_trial_count": sentinel["trial_count"],
            "sentinel_eval_single_label_trial_count": sentinel["eval_single_label_trial_count"],
            "sentinel_eval_collapse_rate": sentinel["eval_collapse_rate"],
            "sentinel_eval_support_min": sentinel["eval_support_min"],
            "sentinel_eval_support_max": sentinel["eval_support_max"],
            "sentinel_eval_support_values_json": json.dumps(sentinel["eval_support_values"], sort_keys=True),
            "diagnostic_only_trial_count": diagnostic["trial_count"],
            "diagnostic_only_eval_single_label_trial_count": diagnostic["eval_single_label_trial_count"],
            "diagnostic_only_eval_collapse_rate": diagnostic["eval_collapse_rate"],
            "diagnostic_only_eval_support_min": diagnostic["eval_support_min"],
            "diagnostic_only_eval_support_max": diagnostic["eval_support_max"],
            "diagnostic_only_eval_support_values_json": json.dumps(diagnostic["eval_support_values"], sort_keys=True),
            "primary_comparator_support_observed": primary_support,
            "sentinel_support_observed": sentinel_support,
            "diagnostic_only_support_observed": diagnostic_support,
            "primary_comparator_full_eval_collapse": primary_full_collapse,
            "diagnostic_only_support_warning": bool(primary_full_collapse and diagnostic_support),
            "support_source_for_proposed_decision": "",
            "method_role_policy": "primary_comparator_support_required_for_cluster_candidate_when_primary_trials_exist",
            "decision_warning": "",
            "reason_code_evidence": "",
            "proposed_decision": "",
            "proposed_reason_code": "",
            "decision_finality": "",
            "production_label_change": False,
        }
        decision, reason_code, finality = _asset_decision_for_row(row)
        row["proposed_decision"] = decision
        row["proposed_reason_code"] = reason_code
        row["decision_finality"] = finality
        if decision == "cluster_candidate" and primary_support:
            row["support_source_for_proposed_decision"] = "primary_comparator"
        elif diagnostic_support and not primary_support:
            row["support_source_for_proposed_decision"] = "diagnostic_only_discounted"
        elif sentinel_support and not primary_support:
            row["support_source_for_proposed_decision"] = "sentinel_only_discounted"
        elif decision == "neutral_flat":
            row["support_source_for_proposed_decision"] = "flat_preflight"
        elif decision in {"data_quality_no_model", "candidate_axis_not_clusterable"}:
            row["support_source_for_proposed_decision"] = "primary_comparator_or_preflight_no_model"
        else:
            row["support_source_for_proposed_decision"] = "no_decisive_support"
        row["decision_warning"] = (
            "diagnostic_only_hdbscan_support_does_not_override_primary_comparator_collapse"
            if bool(row["diagnostic_only_support_warning"])
            else "sentinel_only_support_discounted"
            if sentinel_support and not primary_support
            else ""
        )
        row["reason_code_evidence"] = (
            f"preflight={preflight_reason}; carried_as={carried_as}; "
            f"near_zero_movement_fraction={row['near_zero_movement_fraction']}; "
            f"near_flat_distance_to_threshold={row['near_flat_distance_to_threshold']}; "
            f"eval_single_label_trials={eval_single_count}/{observed}; "
            f"eval_non_noise_support_range={row['eval_non_noise_label_support_min']}..{row['eval_non_noise_label_support_max']}; "
            f"primary_support_range={row['primary_comparator_eval_support_min']}..{row['primary_comparator_eval_support_max']}; "
            f"sentinel_support_range={row['sentinel_eval_support_min']}..{row['sentinel_eval_support_max']}; "
            f"diagnostic_support_range={row['diagnostic_only_eval_support_min']}..{row['diagnostic_only_eval_support_max']}; "
            f"support_source={row['support_source_for_proposed_decision']}; "
            f"forward_status={forward_status}"
        )
        decisions.append(row)
    return decisions


def summarize_asset_model_decisions(asset_model_decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision_counts = Counter(str(row.get("proposed_decision")) for row in asset_model_decisions)
    assets_by_decision: dict[str, list[str]] = {}
    for row in asset_model_decisions:
        decision = str(row.get("proposed_decision"))
        assets_by_decision.setdefault(decision, []).append(str(row.get("asset")))
    return {
        "asset_model_decision_count": int(len(asset_model_decisions)),
        "proposed_decision_counts": {str(k): int(v) for k, v in sorted(decision_counts.items())},
        "assets_by_decision": {key: sorted(values) for key, values in sorted(assets_by_decision.items())},
        "candidate_axis_not_clusterable_assets": sorted(
            str(row.get("asset"))
            for row in asset_model_decisions
            if str(row.get("proposed_decision")) == "candidate_axis_not_clusterable"
        ),
        "neutral_flat_assets": sorted(
            str(row.get("asset")) for row in asset_model_decisions if str(row.get("proposed_decision")) == "neutral_flat"
        ),
        "diagnostic_only_support_warning_assets": sorted(
            str(row.get("asset")) for row in asset_model_decisions if bool(row.get("diagnostic_only_support_warning"))
        ),
        "method_role_policy": "primary_comparator_support_required_for_cluster_candidate_when_primary_trials_exist",
        "support_source_counts": {
            str(k): int(v)
            for k, v in sorted(Counter(str(row.get("support_source_for_proposed_decision")) for row in asset_model_decisions).items())
        },
        "decision_finality": "candidate_only_no_production_label_change",
        "production_label_change": False,
    }


def write_study_artifacts(
    output_root: Path,
    *,
    trial_manifest: Mapping[str, Any],
    candidate_scores: Sequence[Mapping[str, Any]],
    flat_preflight: Sequence[FlatPreflightResult],
    cluster_diagnostics: Sequence[Mapping[str, Any]],
    runtime_summary: Mapping[str, Any],
    aggregate_summary: Optional[Mapping[str, Any]] = None,
    experiment_config_snapshot: Optional[Mapping[str, Any]] = None,
    command_log: Optional[str] = None,
    artifact_validation: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {name: output_root / name for name in ARTIFACT_NAMES}
    per_asset_summary = build_per_asset_summary_rows(cluster_diagnostics)
    asset_model_decisions = build_asset_model_decision_rows(per_asset_summary)
    asset_model_decision_summary = summarize_asset_model_decisions(asset_model_decisions)
    manifest_payload = dict(trial_manifest)
    manifest_payload.setdefault(
        "asset_model_decision_contract",
        {
            "artifact_csv": "asset_model_decisions.csv",
            "artifact_json": "asset_model_decisions.json",
            "decision_values": [
                "cluster_candidate",
                "neutral_flat",
                "candidate_axis_not_clusterable",
                "needs_more_evidence",
                "data_quality_no_model",
            ],
            "method_role_fields": [
                "primary_comparator_eval_support_min",
                "primary_comparator_eval_support_max",
                "sentinel_eval_support_min",
                "sentinel_eval_support_max",
                "diagnostic_only_eval_support_min",
                "diagnostic_only_eval_support_max",
                "support_source_for_proposed_decision",
                "diagnostic_only_support_warning",
            ],
            "method_role_policy": "primary_comparator_support_required_for_cluster_candidate_when_primary_trials_exist",
            "finality": "candidate_only_no_production_label_change",
            "production_label_change": False,
        },
    )
    manifest_payload.setdefault("asset_model_decision_summary", asset_model_decision_summary)
    paths["trial_manifest.json"].write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(list(candidate_scores)).to_csv(paths["candidate_scores.csv"], index=False)
    pd.DataFrame([row.to_dict() for row in flat_preflight]).to_csv(paths["flat_preflight.csv"], index=False)
    paths["cluster_diagnostics.json"].write_text(json.dumps(list(cluster_diagnostics), indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(per_asset_summary).to_csv(paths["per_asset_summary.csv"], index=False)
    paths["per_asset_summary.json"].write_text(json.dumps(per_asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(asset_model_decisions).to_csv(paths["asset_model_decisions.csv"], index=False)
    paths["asset_model_decisions.json"].write_text(json.dumps(asset_model_decisions, indent=2, sort_keys=True), encoding="utf-8")
    summary = dict(runtime_summary)
    summary.setdefault("written_at_monotonic_s", float(time.monotonic()))
    paths["runtime_summary.json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    aggregate = dict(aggregate_summary or {})
    aggregate.setdefault("candidate_count", int(len(candidate_scores)))
    aggregate.setdefault("flat_reason_counts", dict(Counter(row.reason_code for row in flat_preflight)))
    aggregate.setdefault("included_assets", [row.asset for row in flat_preflight if row.included_in_fit])
    aggregate.setdefault("excluded_assets", [row.asset for row in flat_preflight if not row.included_in_fit])
    aggregate.setdefault("asset_model_decision_summary", asset_model_decision_summary)
    aggregate.setdefault("production_outputs_written", False)
    paths["aggregate_summary.json"].write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    snapshot = dict(experiment_config_snapshot or {})
    snapshot.setdefault("trial_manifest", manifest_payload)
    snapshot.setdefault("artifact_contract", list(ARTIFACT_NAMES))
    snapshot.setdefault("asset_model_decision_summary", asset_model_decision_summary)
    paths["experiment_config_snapshot.json"].write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    paths["command_log.txt"].write_text(str(command_log or "run_frame_study synthetic_or_frame invocation\n"), encoding="utf-8")
    validation = dict(artifact_validation or {})
    validation.setdefault("status", "passed")
    validation.setdefault("production_outputs_written", False)
    validation.setdefault("production_regime_parquet_written", False)
    validation.setdefault("production_definitions_written", False)
    paths["artifact_validation.json"].write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    byte_counts = {name: int(path.stat().st_size) for name, path in paths.items()}
    if candidate_scores:
        scores_path = paths["candidate_scores.csv"]
        scores = pd.read_csv(scores_path)
        scores["output_bytes"] = int(sum(byte_counts.values()))
        scores["diagnostics_bytes"] = int(byte_counts["cluster_diagnostics.json"])
        scores.to_csv(scores_path, index=False)
    return {name: str(path) for name, path in paths.items()}
