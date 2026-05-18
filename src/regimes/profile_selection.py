from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.asset_state_benchmark import (
    BENCHMARK_MATRIX_ARTIFACT_KIND,
    LABEL_BENCHMARK_CANDIDATE,
    MATRIX_FILENAME,
    NO_OPTIMIZER,
    OPTUNA_HELPER,
    RESULTS_FILENAME,
    SCOREBOARD_JSON_FILENAME,
    STATUS_SUCCEEDED,
    _bounded_hyperparameters,
    _cell_identity,
    _config_identity,
    _fit_clusterer_result,
    _fit_optuna_helper,
    _jsonable,
)
from src.regimes.clusterability_preflight import (
    PATHWAY_ASSET_STATE,
    PANEL_STRATIFIED,
    build_stratified_feature_frame,
    validate_clusterability_write_root,
)
from src.regimes.core.preprocessing import fit_preprocessing_pipeline

try:
    from sklearn.metrics import adjusted_rand_score

    _HAS_SKLEARN_ARI = True
except Exception:  # pragma: no cover - exercised only in minimal dependency environments
    adjusted_rand_score = None  # type: ignore[assignment]
    _HAS_SKLEARN_ARI = False


PROFILE_SELECTION_SCHEMA_VERSION = 1
CANDIDATE_PROFILE_ARTIFACT_KIND = "regime_asset_state_candidate_profile_manifest"
REJECTED_PROFILE_ARTIFACT_KIND = "regime_asset_state_rejected_profile_manifest"
GAP_REGISTER_ARTIFACT_KIND = "regime_discovery_gap_register"
DECISION_REPORT_ARTIFACT_KIND = "regime_discovery_decision_report"

CANDIDATE_FILENAME = "candidate_profile_manifest.json"
REJECTED_FILENAME = "rejected_profile_manifest.json"
GAP_REGISTER_FILENAME = "regimes_discovery_gap_register.json"
DECISION_REPORT_FILENAME = "regimes_discovery_decision_report.md"

DEFAULT_THRESHOLDS: Mapping[str, float] = {
    "stability_ari_min": 0.70,
    "silhouette_min": 0.12,
    "calinski_harabasz_min": 1.0,
    "davies_bouldin_max": 3.0,
    "excessive_noise_share_max": 0.65,
    "tiny_cluster_min_share": 0.0,
    "tiny_cluster_min_count": 2.0,
    "economic_proxy_min": 0.01,
    "fit_runtime_cap_sec": 120.0,
}

HARD_REJECT_REASON_SEVERITY: Mapping[str, str] = {
    "invalid_status_output": "high",
    "invalid_score_output": "high",
    "one_cluster_output": "high",
    "tiny_cluster_output": "high",
    "all_noise_output": "high",
    "excessive_noise_output": "high",
    "weak_silhouette": "medium",
    "weak_calinski_harabasz": "medium",
    "weak_davies_bouldin": "medium",
    "weak_stability_ari": "high",
    "weak_economic_proxy": "medium",
    "missing_dynamic_evidence": "medium",
    "unsafe_artifact_boundary": "critical",
}


@dataclass(frozen=True)
class ProfileSelectionResult:
    candidate_manifest: Mapping[str, Any]
    rejected_manifest: Mapping[str, Any]
    gap_register: Mapping[str, Any]
    artifact_paths: Mapping[str, str]


@dataclass(frozen=True)
class CloseoutResult:
    report_markdown: str
    artifact_paths: Mapping[str, str]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"expected JSON object rows in {path}")
        rows.append(payload)
    return tuple(rows)


def _source_paths(root: Path) -> dict[str, str]:
    return {
        "benchmark_matrix": str(root / MATRIX_FILENAME),
        "benchmark_results": str(root / RESULTS_FILENAME),
        "benchmark_scoreboard": str(root / SCOREBOARD_JSON_FILENAME),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _artifact_boundary(root: Path, source_paths: Mapping[str, str]) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in source_paths.values()]
    return {
        "pathway": PATHWAY_ASSET_STATE,
        "panel": PANEL_STRATIFIED,
        "stratified_panel_only": True,
        "production_labels_written": False,
        "production_outputs_written": False,
        "promotion_allowed": False,
        "write_root": str(root.resolve()),
        "source_artifacts_under_write_root": all(_is_relative_to(path, root) for path in paths),
        "report_artifacts_only": True,
    }


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def _safe_min(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(min(clean)) if clean else None


def _safe_max(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(max(clean)) if clean else None


def _profile_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        config = row.get("config")
        if isinstance(config, Mapping):
            groups[str(config.get("config_id"))].append(row)
    return dict(sorted(groups.items()))


def _tiny_cluster(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    label_counts = metrics.get("label_counts") or {}
    if not isinstance(label_counts, Mapping):
        return False
    non_noise_counts = [int(value) for key, value in label_counts.items() if str(key) != "-1"]
    if not non_noise_counts:
        return False
    row_count = int(metrics.get("row_count") or sum(non_noise_counts))
    if row_count <= 0:
        return False
    min_count = min(non_noise_counts)
    min_share = float(min_count / row_count)
    return min_count < int(thresholds["tiny_cluster_min_count"]) or min_share < float(thresholds["tiny_cluster_min_share"])


def _base_profile_summary(config_id: str, rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    first = rows[0]
    first_config = first["config"]
    status_counts = Counter(str(row.get("status")) for row in rows)
    final_label_counts = Counter(str(row.get("final_label")) for row in rows)
    metrics = [row.get("metrics") for row in rows if isinstance(row.get("metrics"), Mapping)]
    invalid_status_count = sum(1 for row in rows if row.get("status") != STATUS_SUCCEEDED)
    invalid_score_count = sum(
        1
        for metric in metrics
        if _finite_float(metric.get("silhouette")) is None
        or _finite_float(metric.get("calinski_harabasz")) is None
        or _finite_float(metric.get("davies_bouldin")) is None
    )
    one_cluster_count = sum(1 for metric in metrics if int(metric.get("effective_state_count") or 0) <= 1)
    all_noise_count = sum(1 for metric in metrics if _finite_float(metric.get("noise_share")) == 1.0)
    excessive_noise_count = sum(
        1 for metric in metrics if (_finite_float(metric.get("noise_share")) or 0.0) > float(thresholds["excessive_noise_share_max"])
    )
    tiny_cluster_count = sum(1 for metric in metrics if _tiny_cluster(metric, thresholds))
    runtime_values = [_finite_float(metric.get("runtime_sec")) for metric in metrics]
    silhouette_values = [_finite_float(metric.get("silhouette")) for metric in metrics]
    calinski_values = [_finite_float(metric.get("calinski_harabasz")) for metric in metrics]
    davies_values = [_finite_float(metric.get("davies_bouldin")) for metric in metrics]
    reasons: list[dict[str, Any]] = []
    if invalid_status_count:
        reasons.append({"reason_code": "invalid_status_output", "count": invalid_status_count})
    if invalid_score_count:
        reasons.append({"reason_code": "invalid_score_output", "count": invalid_score_count})
    if one_cluster_count:
        reasons.append({"reason_code": "one_cluster_output", "count": one_cluster_count})
    if tiny_cluster_count:
        reasons.append({"reason_code": "tiny_cluster_output", "count": tiny_cluster_count})
    if all_noise_count:
        reasons.append({"reason_code": "all_noise_output", "count": all_noise_count})
    if excessive_noise_count:
        reasons.append({"reason_code": "excessive_noise_output", "count": excessive_noise_count})
    mean_silhouette = _mean(silhouette_values)
    mean_calinski = _mean(calinski_values)
    mean_davies = _mean(davies_values)
    if mean_silhouette is None or mean_silhouette < float(thresholds["silhouette_min"]):
        reasons.append({"reason_code": "weak_silhouette", "value": mean_silhouette, "threshold": thresholds["silhouette_min"]})
    if mean_calinski is None or mean_calinski < float(thresholds["calinski_harabasz_min"]):
        reasons.append({"reason_code": "weak_calinski_harabasz", "value": mean_calinski, "threshold": thresholds["calinski_harabasz_min"]})
    if mean_davies is None or mean_davies > float(thresholds["davies_bouldin_max"]):
        reasons.append({"reason_code": "weak_davies_bouldin", "value": mean_davies, "threshold": thresholds["davies_bouldin_max"]})
    demotions: list[dict[str, Any]] = []
    max_runtime = _safe_max(runtime_values)
    if max_runtime is not None and max_runtime > float(thresholds["fit_runtime_cap_sec"]):
        demotions.append({"reason_code": "fit_runtime_cap_exceeded", "value": max_runtime, "threshold": thresholds["fit_runtime_cap_sec"]})
    return {
        "profile_id": f"asset_state__{config_id}",
        "config_id": config_id,
        "feature_family": str(first_config.get("feature_family")),
        "preprocessing": str(first_config.get("preprocessing")),
        "clusterer": str(first_config.get("clusterer")),
        "optimizer": str(first_config.get("optimizer", NO_OPTIMIZER)),
        "feature_columns": list(first_config.get("feature_columns") or ()),
        "hyperparameters": dict(first_config.get("resolved_hyperparameters") or first_config.get("hyperparameters") or {}),
        "fit_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "final_label_counts": dict(sorted(final_label_counts.items())),
        "metrics": {
            "mean_silhouette": mean_silhouette,
            "min_silhouette": _safe_min(silhouette_values),
            "mean_calinski_harabasz": mean_calinski,
            "mean_davies_bouldin": mean_davies,
            "max_davies_bouldin": _safe_max(davies_values),
            "mean_runtime_sec": _mean(runtime_values),
            "max_runtime_sec": max_runtime,
            "invalid_status_count": invalid_status_count,
            "invalid_score_count": invalid_score_count,
            "one_cluster_count": one_cluster_count,
            "tiny_cluster_count": tiny_cluster_count,
            "all_noise_count": all_noise_count,
            "excessive_noise_count": excessive_noise_count,
        },
        "demotions": demotions,
        "rejection_reasons": reasons,
    }


def _fit_labels_for_cell_config(
    *,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    data_seed: int,
    fit_seed: int,
    max_runtime_per_fit_sec: float,
) -> tuple[str, np.ndarray, np.ndarray]:
    frame = build_stratified_feature_frame(
        cohort=str(cell["cohort"]),
        asset=str(cell["asset"]),
        axis=str(cell["axis"]),
        band=str(cell["band"]),
        seed=int(data_seed),
    )
    preprocessing = fit_preprocessing_pipeline(
        frame,
        tuple(str(column) for column in config.get("feature_columns") or ()),
        preprocess=str(config["preprocessing"]),
        fit_window={
            "pathway": PATHWAY_ASSET_STATE,
            "panel": PANEL_STRATIFIED,
            "cohort": str(cell["cohort"]),
            "asset": str(cell["asset"]),
            "axis": str(cell["axis"]),
            "band": str(cell["band"]),
            "seed": int(data_seed),
            "fit_seed": int(fit_seed),
        },
        fit_window_role="train",
    )
    x = np.asarray(preprocessing.fitted.x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        return "invalid_input", np.empty(0, dtype=int), x
    clusterer = str(config["clusterer"])
    params = _bounded_hyperparameters(clusterer, dict(config.get("hyperparameters") or {}), seed=int(fit_seed), row_count=int(x.shape[0]))
    if str(config.get("optimizer", NO_OPTIMIZER)) == OPTUNA_HELPER:
        status, labels, _, _, _ = _fit_optuna_helper(
            x=x,
            seed=int(fit_seed),
            base_hyperparameters=params,
            max_runtime_per_fit_sec=float(max_runtime_per_fit_sec),
        )
    else:
        status, labels, _, _, _ = _fit_clusterer_result(
            x=x,
            clusterer=clusterer,
            hyperparameters=params,
            max_runtime_per_fit_sec=float(max_runtime_per_fit_sec),
        )
    return status, np.asarray(labels, dtype=int), x


def _stability_ari_for_profile(
    *,
    matrix: Mapping[str, Any],
    config_id: str,
    seeds: Sequence[int],
    max_runtime_per_fit_sec: float,
) -> dict[str, Any]:
    if not _HAS_SKLEARN_ARI:
        return {"mean_ari": None, "pair_count": 0, "status": "skipped_dependency_unavailable", "dependency": "sklearn.metrics"}
    ari_values: list[float] = []
    status_counts: Counter[str] = Counter()
    cell_count = 0
    for cell in matrix.get("cells") or ():
        configs = [config for config in cell.get("configs") or () if str(config.get("config_id")) == str(config_id)]
        if not configs:
            continue
        cell_count += 1
        config = configs[0]
        label_runs: list[np.ndarray] = []
        for fit_seed in seeds:
            status, labels, _ = _fit_labels_for_cell_config(
                cell=cell,
                config=config,
                data_seed=int(seeds[0]),
                fit_seed=int(fit_seed),
                max_runtime_per_fit_sec=max_runtime_per_fit_sec,
            )
            status_counts[status] += 1
            if status == STATUS_SUCCEEDED and labels.size:
                label_runs.append(labels)
        for left_idx in range(len(label_runs)):
            for right_idx in range(left_idx + 1, len(label_runs)):
                left = label_runs[left_idx]
                right = label_runs[right_idx]
                if left.size == right.size:
                    ari_values.append(float(adjusted_rand_score(left, right)))  # type: ignore[misc]
    return {
        "mean_ari": _mean(ari_values),
        "min_ari": _safe_min(ari_values),
        "pair_count": len(ari_values),
        "cell_count": cell_count,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _separation_ratio(values: Sequence[float], labels: Sequence[int]) -> float | None:
    value_arr = np.asarray(values, dtype=float)
    label_arr = np.asarray(labels, dtype=int)
    mask = np.isfinite(value_arr) & (label_arr != -1)
    if int(np.sum(mask)) < 4:
        return None
    clean_values = value_arr[mask]
    clean_labels = label_arr[mask]
    unique = sorted(set(int(label) for label in clean_labels.tolist()))
    if len(unique) < 2:
        return None
    overall = float(np.mean(clean_values))
    total = float(np.sum((clean_values - overall) ** 2))
    if total <= 0.0:
        return None
    between = 0.0
    for label in unique:
        group_values = clean_values[clean_labels == label]
        between += float(group_values.size) * (float(np.mean(group_values)) - overall) ** 2
    return float(max(0.0, min(1.0, between / total)))


def _economic_proxy_for_cell_config(
    *,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    max_runtime_per_fit_sec: float,
) -> dict[str, Any]:
    status, labels, _ = _fit_labels_for_cell_config(
        cell=cell,
        config=config,
        data_seed=int(seed),
        fit_seed=int(seed),
        max_runtime_per_fit_sec=max_runtime_per_fit_sec,
    )
    if status != STATUS_SUCCEEDED or labels.size == 0:
        return {"status": status, "proxy_score": None, "component_scores": {}}
    frame = build_stratified_feature_frame(
        cohort=str(cell["cohort"]),
        asset=str(cell["asset"]),
        axis=str(cell["axis"]),
        band=str(cell["band"]),
        seed=int(seed),
    )
    holdout_start = max(1, int(len(frame) * 0.75))
    holdout_labels = labels[holdout_start:]
    holdout = frame.iloc[holdout_start:].copy()
    if len(holdout) != int(holdout_labels.size):
        return {"status": "invalid_input", "proxy_score": None, "component_scores": {}}
    component_values = {
        "forward_abs_return": holdout["log_return"].shift(-1).abs().to_numpy(dtype=float),
        "forward_vol": holdout["ret_std_20"].shift(-1).abs().to_numpy(dtype=float),
        "forward_activity": holdout["trade_intensity"].shift(-1).abs().to_numpy(dtype=float),
    }
    component_scores = {
        name: _separation_ratio(values[:-1], holdout_labels[:-1]) if len(values) > 1 else None
        for name, values in component_values.items()
    }
    return {
        "status": STATUS_SUCCEEDED,
        "proxy_score": _mean([_finite_float(value) for value in component_scores.values()]),
        "component_scores": component_scores,
    }


def _economic_proxy_for_profile(
    *,
    matrix: Mapping[str, Any],
    config_id: str,
    seed: int,
    max_runtime_per_fit_sec: float,
) -> dict[str, Any]:
    scores: list[float | None] = []
    component_scores: dict[str, list[float | None]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    cell_count = 0
    for cell in matrix.get("cells") or ():
        configs = [config for config in cell.get("configs") or () if str(config.get("config_id")) == str(config_id)]
        if not configs:
            continue
        cell_count += 1
        result = _economic_proxy_for_cell_config(
            cell=cell,
            config=configs[0],
            seed=int(seed),
            max_runtime_per_fit_sec=max_runtime_per_fit_sec,
        )
        status_counts[str(result["status"])] += 1
        scores.append(_finite_float(result.get("proxy_score")))
        for name, value in dict(result.get("component_scores") or {}).items():
            component_scores[name].append(_finite_float(value))
    return {
        "mean_proxy_score": _mean(scores),
        "min_proxy_score": _safe_min(scores),
        "cell_count": cell_count,
        "status_counts": dict(sorted(status_counts.items())),
        "components": {name: _mean(values) for name, values in sorted(component_scores.items())},
    }


def _profile_score(profile: Mapping[str, Any]) -> float:
    metrics = profile.get("metrics") if isinstance(profile.get("metrics"), Mapping) else {}
    silhouette = _finite_float(metrics.get("mean_silhouette")) or 0.0
    ari = _finite_float(metrics.get("mean_stability_ari")) or 0.0
    economic = _finite_float(metrics.get("mean_economic_proxy")) or 0.0
    davies = _finite_float(metrics.get("mean_davies_bouldin"))
    davies_bonus = 0.0 if davies is None else max(0.0, min(0.2, (3.0 - davies) / 10.0))
    demotion_penalty = 0.10 * len(profile.get("demotions") or ())
    return float(silhouette + 0.25 * ari + 0.50 * economic + davies_bonus - demotion_penalty)


def _with_dynamic_gates(
    *,
    profile: dict[str, Any],
    matrix: Mapping[str, Any],
    thresholds: Mapping[str, float],
    seeds: Sequence[int],
) -> dict[str, Any]:
    if profile["rejection_reasons"]:
        profile["metrics"]["mean_stability_ari"] = None
        profile["metrics"]["mean_economic_proxy"] = None
        profile["dynamic_evidence"] = {"status": "not_computed_after_hard_reject"}
        return profile
    max_runtime = float(thresholds["fit_runtime_cap_sec"])
    stability = _stability_ari_for_profile(matrix=matrix, config_id=profile["config_id"], seeds=seeds, max_runtime_per_fit_sec=max_runtime)
    economic = _economic_proxy_for_profile(matrix=matrix, config_id=profile["config_id"], seed=int(seeds[0]), max_runtime_per_fit_sec=max_runtime)
    mean_ari = _finite_float(stability.get("mean_ari"))
    mean_economic = _finite_float(economic.get("mean_proxy_score"))
    profile["metrics"]["mean_stability_ari"] = mean_ari
    profile["metrics"]["min_stability_ari"] = _finite_float(stability.get("min_ari"))
    profile["metrics"]["mean_economic_proxy"] = mean_economic
    profile["metrics"]["min_economic_proxy"] = _finite_float(economic.get("min_proxy_score"))
    profile["dynamic_evidence"] = {"stability": stability, "economic_proxy": economic}
    if mean_ari is None:
        profile["rejection_reasons"].append({"reason_code": "missing_dynamic_evidence", "metric": "stability_ari"})
    elif mean_ari < float(thresholds["stability_ari_min"]):
        profile["rejection_reasons"].append({"reason_code": "weak_stability_ari", "value": mean_ari, "threshold": thresholds["stability_ari_min"]})
    if mean_economic is None:
        profile["rejection_reasons"].append({"reason_code": "missing_dynamic_evidence", "metric": "economic_proxy"})
    elif mean_economic < float(thresholds["economic_proxy_min"]):
        profile["rejection_reasons"].append({"reason_code": "weak_economic_proxy", "value": mean_economic, "threshold": thresholds["economic_proxy_min"]})
    return profile


def score_profiles(
    *,
    matrix: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float] | None = None,
    seeds: Sequence[int] = (11, 29, 47),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_thresholds = {**dict(DEFAULT_THRESHOLDS), **dict(thresholds or {})}
    if matrix.get("artifact_kind") != BENCHMARK_MATRIX_ARTIFACT_KIND:
        raise ValueError("profile selection requires an asset-state benchmark matrix")
    grouped = _profile_groups(rows)
    profiles: list[dict[str, Any]] = []
    for config_id, items in grouped.items():
        profile = _base_profile_summary(config_id, items, active_thresholds)
        profile = _with_dynamic_gates(profile=profile, matrix=matrix, thresholds=active_thresholds, seeds=seeds)
        profile["score"] = _profile_score(profile)
        profile["decision"] = "rejected" if profile["rejection_reasons"] else "candidate"
        profiles.append(profile)
    profiles.sort(key=lambda item: (-float(item["score"]), str(item["config_id"])))
    candidates = [dict(profile, rank=idx) for idx, profile in enumerate([p for p in profiles if p["decision"] == "candidate"], start=1)]
    rejected = [dict(profile, rank=None) for profile in profiles if profile["decision"] == "rejected"]
    return candidates, rejected


def _gap_entries(
    *,
    candidates: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    artifact_boundary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    idx = 1
    if not artifact_boundary.get("source_artifacts_under_write_root") or artifact_boundary.get("production_outputs_written"):
        gaps.append(
            {
                "gap_id": f"GAP-{idx:03d}",
                "severity": "critical",
                "category": "safety",
                "profile_id": None,
                "config_id": None,
                "reason_code": "unsafe_artifact_boundary",
                "description": "Source or output artifact boundary is not restricted to the discovery report root.",
                "recommended_action": "Block closeout until all artifacts are report-root scoped and production writes are false.",
            }
        )
        idx += 1
    if not candidates:
        gaps.append(
            {
                "gap_id": f"GAP-{idx:03d}",
                "severity": "critical",
                "category": "selection",
                "profile_id": None,
                "config_id": None,
                "reason_code": "no_candidate_profile",
                "description": "No profile passed the bounded discovery gates.",
                "recommended_action": "Broaden preflight-eligible matrix or revisit gate thresholds before any promotion discussion.",
            }
        )
        idx += 1
    for profile in rejected:
        for reason in profile.get("rejection_reasons") or ():
            reason_code = str(reason.get("reason_code"))
            gaps.append(
                {
                    "gap_id": f"GAP-{idx:03d}",
                    "severity": HARD_REJECT_REASON_SEVERITY.get(reason_code, "medium"),
                    "category": "profile_gate",
                    "profile_id": profile.get("profile_id"),
                    "config_id": profile.get("config_id"),
                    "reason_code": reason_code,
                    "details": reason,
                    "recommended_action": _recommended_action(reason_code),
                }
            )
            idx += 1
    return gaps


def _recommended_action(reason_code: str) -> str:
    return {
        "invalid_status_output": "Remove failed or skipped fits from candidate consideration and inspect runner dependency/status handling.",
        "invalid_score_output": "Reject the profile until all bounded outputs expose valid silhouette, CH, and DB scores.",
        "one_cluster_output": "Reject or retune state-count/density parameters; one-cluster outputs are not profile candidates.",
        "tiny_cluster_output": "Retune state count, preprocessing, or feature family to avoid unstable tiny states.",
        "all_noise_output": "Reject density settings that classify all observations as noise.",
        "excessive_noise_output": "Retune density settings or reject the profile for bounded asset-state discovery.",
        "weak_silhouette": "Retain as rejected evidence unless a later bounded matrix improves silhouette.",
        "weak_calinski_harabasz": "Retain as rejected evidence unless a later bounded matrix improves CH separation.",
        "weak_davies_bouldin": "Retain as rejected evidence unless a later bounded matrix improves DB separation.",
        "weak_stability_ari": "Rerun with a stabler clusterer or constrained hyperparameters.",
        "weak_economic_proxy": "Add or adjust economically meaningful features before promotion consideration.",
        "missing_dynamic_evidence": "Compute missing stability/economic evidence after resolving earlier hard rejects.",
    }.get(reason_code, "Review the rejected profile before any later discovery sprint.")


def run_profile_selection_analysis(
    *,
    write_root: str | Path,
    no_write: bool = False,
    project_root: str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    seeds: Sequence[int] = (11, 29, 47),
) -> ProfileSelectionResult:
    root = Path(write_root) if no_write else validate_clusterability_write_root(write_root, project_root=project_root)
    source_paths = _source_paths(root)
    matrix = _load_json(Path(source_paths["benchmark_matrix"]))
    rows = _load_jsonl(Path(source_paths["benchmark_results"]))
    candidates, rejected = score_profiles(matrix=matrix, rows=rows, thresholds=thresholds, seeds=seeds)
    artifact_boundary = _artifact_boundary(root, source_paths)
    candidate_manifest = {
        "schema_version": PROFILE_SELECTION_SCHEMA_VERSION,
        "artifact_kind": CANDIDATE_PROFILE_ARTIFACT_KIND,
        "created_at_utc": _now_utc(),
        "source_artifacts": source_paths,
        "artifact_boundary": artifact_boundary,
        "thresholds": {**dict(DEFAULT_THRESHOLDS), **dict(thresholds or {})},
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    rejected_manifest = {
        "schema_version": PROFILE_SELECTION_SCHEMA_VERSION,
        "artifact_kind": REJECTED_PROFILE_ARTIFACT_KIND,
        "created_at_utc": _now_utc(),
        "source_artifacts": source_paths,
        "artifact_boundary": artifact_boundary,
        "thresholds": {**dict(DEFAULT_THRESHOLDS), **dict(thresholds or {})},
        "rejected_count": len(rejected),
        "rejected_profiles": rejected,
    }
    gap_register = {
        "schema_version": PROFILE_SELECTION_SCHEMA_VERSION,
        "artifact_kind": GAP_REGISTER_ARTIFACT_KIND,
        "created_at_utc": _now_utc(),
        "source_artifacts": source_paths,
        "artifact_boundary": artifact_boundary,
        "gap_count": 0,
        "gaps": [],
    }
    gaps = _gap_entries(candidates=candidates, rejected=rejected, artifact_boundary=artifact_boundary)
    gap_register = {**gap_register, "gap_count": len(gaps), "gaps": gaps}
    artifact_paths: dict[str, str] = {}
    if not no_write:
        candidate_path = root / CANDIDATE_FILENAME
        rejected_path = root / REJECTED_FILENAME
        gap_path = root / GAP_REGISTER_FILENAME
        _write_json(candidate_path, candidate_manifest)
        _write_json(rejected_path, rejected_manifest)
        _write_json(gap_path, gap_register)
        artifact_paths = {
            "candidate_manifest": str(candidate_path),
            "rejected_manifest": str(rejected_path),
            "gap_register": str(gap_path),
        }
    return ProfileSelectionResult(
        candidate_manifest=candidate_manifest,
        rejected_manifest=rejected_manifest,
        gap_register=gap_register,
        artifact_paths=artifact_paths,
    )


def _decision_report(candidate_manifest: Mapping[str, Any], rejected_manifest: Mapping[str, Any], gap_register: Mapping[str, Any]) -> str:
    candidates = list(candidate_manifest.get("candidates") or ())
    rejected = list(rejected_manifest.get("rejected_profiles") or ())
    top = candidates[0] if candidates else None
    lines = [
        "# Regime Discovery Decision Report",
        "",
        f"- Artifact kind: `{DECISION_REPORT_ARTIFACT_KIND}`",
        f"- Candidate profiles: {len(candidates)}",
        f"- Rejected profiles: {len(rejected)}",
        f"- Gap count: {gap_register.get('gap_count', 0)}",
        f"- Production labels written: `{candidate_manifest['artifact_boundary']['production_labels_written']}`",
        f"- Promotion allowed: `{candidate_manifest['artifact_boundary']['promotion_allowed']}`",
        f"- Source artifacts under discovery root: `{candidate_manifest['artifact_boundary']['source_artifacts_under_write_root']}`",
        "",
    ]
    if top:
        metrics = top.get("metrics") or {}
        lines.extend(
            [
                "## Selected Candidate",
                "",
                f"- Profile: `{top['profile_id']}`",
                f"- Config: `{top['config_id']}`",
                f"- Feature/preprocess/clusterer: `{top['feature_family']}` / `{top['preprocessing']}` / `{top['clusterer']}`",
                f"- Score: `{float(top['score']):.4f}`",
                f"- Stability ARI: `{_format_metric(metrics.get('mean_stability_ari'))}`",
                f"- Separability: silhouette `{_format_metric(metrics.get('mean_silhouette'))}`, CH `{_format_metric(metrics.get('mean_calinski_harabasz'))}`, DB `{_format_metric(metrics.get('mean_davies_bouldin'))}`",
                f"- Economic proxy: `{_format_metric(metrics.get('mean_economic_proxy'))}`",
                "",
            ]
        )
    else:
        lines.extend(["## Selected Candidate", "", "No profile passed the bounded discovery gates.", ""])
    lines.extend(
        [
            "## Rejection Summary",
            "",
            "| Profile | Config | Primary reasons |",
            "|---|---|---|",
        ]
    )
    for profile in rejected:
        reasons = ", ".join(str(reason.get("reason_code")) for reason in profile.get("rejection_reasons") or ())
        lines.append(f"| `{profile['profile_id']}` | `{profile['config_id']}` | {reasons or 'n/a'} |")
    lines.extend(
        [
            "",
            "## Closeout Decision",
            "",
            "The bounded discovery sprint is closed as report-only evidence. Candidate and rejected manifests are discovery artifacts only; no production labels, production writes, or promotion actions are authorized by this report.",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    out = _finite_float(value)
    return "n/a" if out is None else f"{out:.4f}"


def run_profile_selection_closeout(
    *,
    write_root: str | Path,
    no_write: bool = False,
    project_root: str | Path | None = None,
) -> CloseoutResult:
    root = Path(write_root) if no_write else validate_clusterability_write_root(write_root, project_root=project_root)
    candidate_path = root / CANDIDATE_FILENAME
    rejected_path = root / REJECTED_FILENAME
    gap_path = root / GAP_REGISTER_FILENAME
    if not (candidate_path.exists() and rejected_path.exists() and gap_path.exists()):
        analysis = run_profile_selection_analysis(write_root=root, no_write=no_write, project_root=project_root)
        candidate_manifest = analysis.candidate_manifest
        rejected_manifest = analysis.rejected_manifest
        gap_register = analysis.gap_register
    else:
        candidate_manifest = _load_json(candidate_path)
        rejected_manifest = _load_json(rejected_path)
        gap_register = _load_json(gap_path)
    report = _decision_report(candidate_manifest, rejected_manifest, gap_register)
    artifact_paths: dict[str, str] = {}
    if not no_write:
        report_path = root / DECISION_REPORT_FILENAME
        _write_text(report_path, report)
        artifact_paths = {"decision_report": str(report_path)}
    return CloseoutResult(report_markdown=report, artifact_paths=artifact_paths)


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--write-root", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main_analysis(argv: Sequence[str] | None = None) -> int:
    args = _build_parser("Score and gate bounded asset-state benchmark profiles.").parse_args(argv)
    started = time.monotonic()
    result = run_profile_selection_analysis(
        write_root=args.write_root,
        no_write=args.no_write,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "artifact_paths": result.artifact_paths,
                "candidate_count": result.candidate_manifest["candidate_count"],
                "rejected_count": result.rejected_manifest["rejected_count"],
                "gap_count": result.gap_register["gap_count"],
                "elapsed_sec": round(time.monotonic() - started, 3),
            },
            sort_keys=True,
        )
    )
    return 0


def main_closeout(argv: Sequence[str] | None = None) -> int:
    args = _build_parser("Write bounded regime discovery decision report.").parse_args(argv)
    result = run_profile_selection_closeout(
        write_root=args.write_root,
        no_write=args.no_write,
        project_root=args.project_root,
    )
    print(json.dumps({"artifact_paths": result.artifact_paths}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile selection analysis and closeout.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analysis_parser = subparsers.add_parser("analysis", parents=[_build_parser("analysis")], add_help=False)
    analysis_parser.set_defaults(command="analysis")
    closeout_parser = subparsers.add_parser("closeout", parents=[_build_parser("closeout")], add_help=False)
    closeout_parser.set_defaults(command="closeout")
    args = parser.parse_args(argv)
    selected = [
        "--write-root",
        args.write_root,
        *(("--project-root", args.project_root) if args.project_root else ()),
        *(("--no-write",) if args.no_write else ()),
    ]
    if args.command == "analysis":
        return main_analysis(selected)
    if args.command == "closeout":
        return main_closeout(selected)
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
