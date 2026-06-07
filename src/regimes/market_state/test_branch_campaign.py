from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.regimes.core.clustering_candidates import (
    CANDIDATE_STATUS_DIAGNOSTIC_ONLY,
    CANDIDATE_STATUS_PRODUCTION_CANDIDATE,
    clusterer_and_embedding_for_method,
)
from src.regimes.core.clusterer_registry import default_clusterer_registry
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.test_branch_contracts import (
    apply_nonproduction_gate_flags,
    score_contract_fields,
    scoring_contract_manifest_section,
    test_branch_consumable_status_fields,
    validate_selected_profile_manifest,
)
from src.regimes.core.window_profiles import RegimeWindowProfile, coerce_window_profile, window_profile_rows
from src.regimes.market_state.axis_contracts import MARKET_STATE_V1_AXIS_IDS
from src.regimes.market_state.clustering_policy import (
    MARKET_STATE_DIAGNOSTIC_CONDITIONAL_METHODS,
    MARKET_STATE_PRODUCTION_CANDIDATE_METHODS,
)
from src.regimes.market_state.runtime import (
    MARKET_STATE_TEST_DEFAULT_WORKERS,
    MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS,
    MarketStateTestRuntimeConfig,
    observed_thread_env,
)
from src.regimes.market_state.window_profiles import (
    MarketStateCoverageGatePolicy,
    default_market_state_window_profiles,
    evaluate_market_state_window_coverage,
    normalize_market_state_coverage_reason,
)


MARKET_STATE_SELECTED_PROFILES_FILENAME = "market_state_selected_profiles.nonprod.json"
MARKET_STATE_CANDIDATE_SCOREBOARD_FILENAME = "market_state_candidate_scoreboard.csv"
MARKET_STATE_MASKED_CELLS_FILENAME = "market_state_masked_or_skipped_cells.csv"
MARKET_STATE_RUNTIME_TELEMETRY_FILENAME = "market_state_runtime_telemetry.csv"
MARKET_STATE_CAMPAIGN_SUMMARY_FILENAME = "market_state_test_branch_campaign_summary.json"

MARKET_STATE_TEST_CAMPAIGN_ARTIFACT_KIND = "market_state_test_branch_selected_profiles_nonprod"
MARKET_STATE_TEST_CAMPAIGN_SCHEMA_VERSION = 1

DEFAULT_BAND_INTERVALS: tuple[tuple[str, int], ...] = (("micro", 60), ("meso", 240), ("macro", 1440))

PREPROCESSING_OPTIONS: tuple[str, ...] = ("standard_scale", "robust_scale", "noop")
REDUCER_OPTIONS: tuple[str, ...] = ("none", "pca", "factor_analysis")


def _candidate_profile(
    method_family: str,
    cluster_count: int,
    preprocessing: str,
    reducer: str,
    production_candidate: bool,
    *,
    label: str = "expanded",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    method = str(method_family)
    status = CANDIDATE_STATUS_PRODUCTION_CANDIDATE if bool(production_candidate) else CANDIDATE_STATUS_DIAGNOSTIC_ONLY
    suffix = f"k{cluster_count}" if int(cluster_count) > 0 else "density"
    payload = {
        "method_family": method,
        "method_profile": f"{method}_{suffix}_{preprocessing}_{reducer}_{label}",
        "n_clusters": int(cluster_count),
        "preprocessing": str(preprocessing),
        "reducer": str(reducer),
        "candidate_status": status,
        "production_candidate": bool(production_candidate),
        "diagnostic_only": not bool(production_candidate),
    }
    payload.update(dict(extra or {}))
    return payload


def default_market_state_candidate_profiles() -> tuple[Mapping[str, Any], ...]:
    profiles: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSING_OPTIONS:
        production_preprocessing = preprocessing != "noop"
        for k in (2, 3, 4):
            profiles.append(_candidate_profile("kmeans", k, preprocessing, "none", production_preprocessing))
            profiles.append(_candidate_profile("gaussian_mixture", k, preprocessing, "none", production_preprocessing))
        for k in (2, 3):
            profiles.append(_candidate_profile("minibatch_kmeans", k, preprocessing, "none", production_preprocessing))
    for preprocessing in ("standard_scale", "robust_scale"):
        for k in (2, 3):
            profiles.append(_candidate_profile("pca_kmeans", k, preprocessing, "pca", True))
            profiles.append(_candidate_profile("factor_analysis_kmeans", k, preprocessing, "factor_analysis", True))
            profiles.append(_candidate_profile("factor_analysis_gaussian_mixture", k, preprocessing, "factor_analysis", True))
        for k in (3, 4):
            profiles.append(_candidate_profile("bayesian_gaussian_mixture", k, preprocessing, "none", True))
            profiles.append(_candidate_profile("birch", k, preprocessing, "none", True))
    profiles.extend(
        (
            _candidate_profile("hdbscan", 0, "standard_scale", "none", False, extra={"min_cluster_size": 8, "min_samples": 3}),
            _candidate_profile("optics", 0, "standard_scale", "none", False, extra={"min_samples": 8, "xi": 0.05}),
            _candidate_profile("agglomerative", 2, "standard_scale", "none", False),
            _candidate_profile("agglomerative", 3, "standard_scale", "none", False),
        )
    )
    return tuple(profiles)


def legacy_market_state_candidate_profiles() -> tuple[Mapping[str, Any], ...]:
    return (
        _candidate_profile("kmeans", 2, "standard_scale", "none", True, label="legacy"),
        _candidate_profile("kmeans", 3, "standard_scale", "none", True, label="legacy"),
        _candidate_profile("minibatch_kmeans", 2, "standard_scale", "none", True, label="legacy"),
        _candidate_profile("gaussian_mixture", 2, "standard_scale", "none", True, label="legacy"),
    )


PROFILE_METHODS: tuple[Mapping[str, Any], ...] = default_market_state_candidate_profiles()

AXIS_FEATURE_POOLS: Mapping[str, Mapping[str, Any]] = {
    "market_return_state": {
        "feature_pool": "market_return_summary",
        "features": ("market_return_equal_weight", "market_return_core_equal_weight", "market_return_median"),
    },
    "market_volatility_state": {
        "feature_pool": "market_realized_volatility",
        "features": ("market_realized_volatility", "market_core_return_realized_volatility", "market_volatility_median"),
    },
    "market_breadth_state": {
        "feature_pool": "market_breadth",
        "features": ("share_assets_up", "share_assets_down", "positive_return_breadth", "negative_return_breadth"),
    },
    "market_dispersion_state": {
        "feature_pool": "market_dispersion",
        "features": ("return_dispersion_std", "return_dispersion_iqr", "return_quantile_spread_q90_q10"),
    },
    "market_correlation_state": {
        "feature_pool": "market_covariance_summary",
        "features": ("core_pairwise_corr_median", "core_pairwise_corr_mean", "covariance_first_pc_concentration"),
    },
    "market_liquidity_activity_state": {
        "feature_pool": "market_liquidity_activity",
        "features": ("aggregate_volume", "aggregate_trades", "activity_breadth", "volume_activity_breadth", "trades_activity_breadth"),
    },
    "market_stress_state": {
        "feature_pool": "market_stress",
        "features": ("stress_down_participation", "downside_breadth", "high_vol_asset_share", "high_corr_high_vol_coincidence"),
    },
    "stable_peg_stress_state": {
        "feature_pool": "stable_peg_stress",
        "features": ("peg_deviation_abs", "stable_stress_breadth", "stable_panel_coverage", "stable_activity_share"),
    },
    "market_speculative_state": {
        "feature_pool": "speculative_satellite_sidecar",
        "features": (
            "speculative_return_dispersion",
            "speculative_vs_clean_broad_return_spread",
            "speculative_volume_share",
            "speculative_activity_breadth",
        ),
    },
}


@dataclass(frozen=True)
class MarketStateTestBranchCampaignConfig:
    output_root: str | Path
    run_id: str
    feature_root: str | Path | None = None
    band_intervals: Sequence[tuple[str, int]] = DEFAULT_BAND_INTERVALS
    window_profiles: Sequence[RegimeWindowProfile | Mapping[str, Any]] | None = None
    candidate_profiles: Sequence[Mapping[str, Any]] | None = None
    max_files_per_band: int = 6
    min_rows: int = 48
    min_finite_share: float = 0.75
    workers: int | None = None
    model_threads: int = 1
    writer_workers: int = MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS
    random_seed: int = 17
    allow_partial: bool = True
    schema_version: int = MARKET_STATE_TEST_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "feature_root", None if self.feature_root is None else Path(self.feature_root))
        object.__setattr__(self, "run_id", _safe_token(self.run_id))
        object.__setattr__(self, "band_intervals", tuple((str(band).lower(), int(interval)) for band, interval in self.band_intervals))
        profiles = self.window_profiles
        if profiles is None:
            profiles = default_market_state_window_profiles()
        object.__setattr__(self, "window_profiles", tuple(coerce_window_profile(profile) for profile in profiles))
        candidate_profiles = self.candidate_profiles
        if candidate_profiles is None:
            candidate_profiles = default_market_state_candidate_profiles()
        object.__setattr__(self, "candidate_profiles", tuple(dict(profile) for profile in candidate_profiles))
        object.__setattr__(self, "max_files_per_band", max(1, int(self.max_files_per_band)))
        object.__setattr__(self, "min_rows", max(1, int(self.min_rows)))
        object.__setattr__(self, "min_finite_share", min(1.0, max(0.0, float(self.min_finite_share))))
        object.__setattr__(self, "workers", None if self.workers is None else max(1, int(self.workers)))
        object.__setattr__(self, "model_threads", max(1, int(self.model_threads)))
        object.__setattr__(self, "writer_workers", max(1, int(self.writer_workers)))
        object.__setattr__(self, "random_seed", int(self.random_seed))


@dataclass(frozen=True)
class MarketStateTestBranchCampaignResult:
    status: str
    output_root: Path
    selected_manifest_path: Path
    candidate_scoreboard_path: Path
    masked_cells_path: Path
    runtime_telemetry_path: Path
    summary_path: Path
    selected_profile_count: int
    masked_or_skipped_count: int
    candidate_count: int
    runtime_seconds: float
    summary: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_root": str(self.output_root),
            "selected_manifest_path": str(self.selected_manifest_path),
            "candidate_scoreboard_path": str(self.candidate_scoreboard_path),
            "masked_cells_path": str(self.masked_cells_path),
            "runtime_telemetry_path": str(self.runtime_telemetry_path),
            "summary_path": str(self.summary_path),
            "selected_profile_count": int(self.selected_profile_count),
            "masked_or_skipped_count": int(self.masked_or_skipped_count),
            "candidate_count": int(self.candidate_count),
            "runtime_seconds": float(self.runtime_seconds),
            "summary": to_jsonable(dict(self.summary)),
        }


def build_market_state_campaign_work_matrix(config: MarketStateTestBranchCampaignConfig) -> tuple[dict[str, Any], ...]:
    """Return the deterministic candidate-level work matrix without loading source frames."""
    rows: list[dict[str, Any]] = []
    for band, interval in tuple(config.band_intervals):
        for window_profile in _profiles_for_band(config, band):
            for axis in MARKET_STATE_V1_AXIS_IDS:
                pool = dict(AXIS_FEATURE_POOLS.get(axis, {}))
                feature_pool = str(pool.get("feature_pool") or "unknown")
                features = tuple(str(item) for item in pool.get("features", ()))
                for method in tuple(config.candidate_profiles or ()):
                    rows.append(
                        {
                            "market_axis": axis,
                            "band": str(band),
                            "interval": int(interval),
                            "window_profile_id": window_profile.window_profile_id,
                            "feature_pool": feature_pool,
                            "feature_set": list(features),
                            "method_family": str(method.get("method_family")),
                            "method_profile": str(method.get("method_profile")),
                            "preprocessing": str(method.get("preprocessing") or "standard_scale"),
                            "reducer": str(method.get("reducer") or "none"),
                            "candidate_status": str(method.get("candidate_status") or CANDIDATE_STATUS_PRODUCTION_CANDIDATE),
                            "production_candidate": bool(method.get("production_candidate", True)),
                            "diagnostic_only": bool(method.get("diagnostic_only", False)),
                            "core_parameters": _core_parameters(method, random_seed=int(config.random_seed)),
                        }
                    )
    return tuple(rows)


def run_market_state_test_branch_campaign(config: MarketStateTestBranchCampaignConfig) -> MarketStateTestBranchCampaignResult:
    started = time.perf_counter()
    _apply_model_thread_caps(config.model_threads)
    output_root = _safe_output_root(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_root, feature_root_source = _resolve_feature_root(config.feature_root)
    phase_timings: dict[str, float] = {}
    runtime_config = MarketStateTestRuntimeConfig.from_manifest(
        type("Manifest", (), {"study_id": config.run_id})(),
        write_root=output_root,
    )
    if config.workers is not None:
        runtime_config = MarketStateTestRuntimeConfig(
            run_id=runtime_config.run_id,
            output_root=runtime_config.output_root,
            market_state_test_workers=int(config.workers),
            writer_workers=int(config.writer_workers),
            thread_caps_enforced=True,
            observed_thread_env=observed_thread_env(),
            worker_source={"source": "explicit_campaign_config", "value": int(config.workers)},
            writer_source={"source": "explicit_campaign_config", "value": int(config.writer_workers)},
        )
    elif int(runtime_config.writer_workers) != int(config.writer_workers):
        runtime_config = MarketStateTestRuntimeConfig(
            run_id=runtime_config.run_id,
            output_root=runtime_config.output_root,
            market_state_test_workers=int(runtime_config.market_state_test_workers),
            writer_workers=int(config.writer_workers),
            thread_caps_enforced=True,
            observed_thread_env=observed_thread_env(),
            worker_source=runtime_config.worker_source,
            writer_source={"source": "campaign_config_default", "value": int(config.writer_workers)},
        )
    if not runtime_config.thread_caps_enforced:
        runtime_config = MarketStateTestRuntimeConfig(
            run_id=runtime_config.run_id,
            output_root=runtime_config.output_root,
            market_state_test_workers=int(runtime_config.market_state_test_workers),
            writer_workers=int(runtime_config.writer_workers),
            thread_caps_enforced=True,
            observed_thread_env=observed_thread_env(),
            worker_source=runtime_config.worker_source,
            writer_source=runtime_config.writer_source,
        )

    matrix_started = time.perf_counter()
    work_matrix = build_market_state_campaign_work_matrix(config)
    phase_timings["build_work_matrix_s"] = max(0.0, time.perf_counter() - matrix_started)
    load_started = time.perf_counter()
    frames, input_rows = _load_feature_frames(feature_root, config=config)
    phase_timings["load_feature_frames_s"] = max(0.0, time.perf_counter() - load_started)
    scoreboard: list[dict[str, Any]] = []
    masked: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    eval_started = time.perf_counter()
    tasks = [
        (cell_idx, band, int(interval), profile, axis, frames.get((band, int(interval), profile.window_profile_id), pd.DataFrame()))
        for cell_idx, (band, interval, profile, axis) in enumerate(
            (band, int(interval), profile, axis)
            for band, interval in config.band_intervals
            for profile in _profiles_for_band(config, band)
            for axis in MARKET_STATE_V1_AXIS_IDS
        )
    ]
    cell_results: dict[int, tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]] = {}
    worker_count = max(1, min(int(runtime_config.market_state_test_workers), len(tasks) or 1))
    if worker_count <= 1:
        for cell_idx, band, interval, profile, axis, frame in tasks:
            cell_results[cell_idx] = _evaluate_cell_task(
                frame,
                axis=axis,
                band=band,
                interval=interval,
                window_profile=profile,
                config=config,
                worker_count=worker_count,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _evaluate_cell_task,
                    frame,
                    axis=axis,
                    band=band,
                    interval=interval,
                    window_profile=profile,
                    config=config,
                    worker_count=worker_count,
                ): cell_idx
                for cell_idx, band, interval, profile, axis, frame in tasks
            }
            for future in as_completed(futures):
                cell_results[futures[future]] = future.result()
    for cell_idx in sorted(cell_results):
        cell_scoreboard, cell_mask, cell_selected, telemetry = cell_results[cell_idx]
        scoreboard.extend(cell_scoreboard)
        if cell_mask is not None:
            masked.append(cell_mask)
        if cell_selected is not None:
            selected.append(cell_selected)
        telemetry_rows.append(telemetry)
    phase_timings["evaluate_cells_s"] = max(0.0, time.perf_counter() - eval_started)
    selected = _select_profiles_by_axis_band(selected)

    selected_manifest = _selected_manifest(
        config=config,
        feature_root=feature_root,
        feature_root_source=feature_root_source,
        runtime_config=runtime_config,
        selected=selected,
        masked=masked,
        scoreboard=scoreboard,
        input_rows=input_rows,
        work_matrix=work_matrix,
        phase_timings=phase_timings,
        elapsed_s=max(0.0, time.perf_counter() - started),
    )
    validation = _validate_campaign_manifest(config, selected_manifest)
    if not validation.passed:
        raise RuntimeError(f"Market-State selected-profile manifest validation failed: {','.join(validation.reason_codes)}")
    selected_manifest["selected_profile_manifest_validation"] = validation.as_dict()
    scoreboard_path = output_root / MARKET_STATE_CANDIDATE_SCOREBOARD_FILENAME
    masked_path = output_root / MARKET_STATE_MASKED_CELLS_FILENAME
    telemetry_path = output_root / MARKET_STATE_RUNTIME_TELEMETRY_FILENAME
    manifest_path = output_root / MARKET_STATE_SELECTED_PROFILES_FILENAME
    summary_path = output_root / MARKET_STATE_CAMPAIGN_SUMMARY_FILENAME
    write_started = time.perf_counter()
    _write_csv(scoreboard_path, scoreboard, fieldnames=_scoreboard_fields())
    _write_csv(masked_path, masked, fieldnames=_masked_fields())
    _write_csv(telemetry_path, telemetry_rows, fieldnames=_telemetry_fields())
    _write_json_atomic(manifest_path, selected_manifest)
    phase_timings["finalizer_write_s"] = max(0.0, time.perf_counter() - write_started)
    selected_manifest["runtime"]["phase_timings"] = dict(phase_timings)
    _write_json_atomic(manifest_path, selected_manifest)
    summary = _summary_payload(selected_manifest, scoreboard, masked, telemetry_rows, phase_timings=phase_timings)
    _write_json_atomic(summary_path, summary)
    return MarketStateTestBranchCampaignResult(
        status=str(summary["status"]),
        output_root=output_root,
        selected_manifest_path=manifest_path,
        candidate_scoreboard_path=scoreboard_path,
        masked_cells_path=masked_path,
        runtime_telemetry_path=telemetry_path,
        summary_path=summary_path,
        selected_profile_count=len(selected),
        masked_or_skipped_count=len(masked),
        candidate_count=len(scoreboard),
        runtime_seconds=float(summary["runtime"]["elapsed_s"]),
        summary=summary,
    )


def _evaluate_cell_task(
    frame: pd.DataFrame,
    *,
    axis: str,
    band: str,
    interval: int,
    window_profile: RegimeWindowProfile,
    config: MarketStateTestBranchCampaignConfig,
    worker_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    cell_started = time.perf_counter()
    pool = dict(AXIS_FEATURE_POOLS.get(axis, {}))
    features = tuple(str(item) for item in pool.get("features", ()))
    feature_pool = str(pool.get("feature_pool") or "unknown")
    cell_scoreboard, cell_mask, cell_selected = _evaluate_cell(
        frame,
        axis=axis,
        band=band,
        interval=int(interval),
        window_profile=window_profile,
        features=features,
        feature_pool=feature_pool,
        config=config,
    )
    telemetry = {
        "run_id": config.run_id,
        "axis": axis,
        "band": band,
        "interval": int(interval),
        "window_profile_id": window_profile.window_profile_id,
        "row_count": int(frame.shape[0]),
        "candidate_count": len(cell_scoreboard),
        "selected": cell_selected is not None,
        "masked_or_skipped": cell_mask is not None,
        "elapsed_s": max(0.0, time.perf_counter() - cell_started),
        "worker_count": int(worker_count),
        "model_threads": int(config.model_threads),
    }
    return cell_scoreboard, cell_mask, cell_selected, telemetry


def _evaluate_cell(
    frame: pd.DataFrame,
    *,
    axis: str,
    band: str,
    interval: int,
    window_profile: RegimeWindowProfile,
    features: Sequence[str],
    feature_pool: str,
    config: MarketStateTestBranchCampaignConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    if frame.empty:
        return [], _mask(axis, band, interval, window_profile.window_profile_id, feature_pool, "missing_feature_rows", "no rows for band/interval/window"), None
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        gate = evaluate_market_state_window_coverage(
            frame,
            market_axis=axis,
            band=band,
            window_profile=window_profile,
            required_features=features,
            policy=MarketStateCoverageGatePolicy(min_rows=config.min_rows),
        )
        return [], _mask(
            axis,
            band,
            interval,
            window_profile.window_profile_id,
            feature_pool,
            normalize_market_state_coverage_reason("missing_required_features"),
            ",".join(missing),
            coverage_gate=gate.as_dict(),
        ), None
    coverage_gate = evaluate_market_state_window_coverage(
        frame,
        market_axis=axis,
        band=band,
        window_profile=window_profile,
        required_features=features,
        policy=MarketStateCoverageGatePolicy(min_rows=config.min_rows),
    )
    if not coverage_gate.passed:
        first_reason = str(coverage_gate.reason_codes[0])
        return [], _mask(
            axis,
            band,
            interval,
            window_profile.window_profile_id,
            feature_pool,
            normalize_market_state_coverage_reason(first_reason),
            "|".join(str(reason) for reason in coverage_gate.reason_codes),
            row_count=coverage_gate.summary.row_count,
            coverage_gate=coverage_gate.as_dict(),
        ), None
    work = frame[["ts", "known_at_ts", "source_tail_ts", *features]].copy()
    for feature in features:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
    finite_mask = np.isfinite(work[list(features)].to_numpy(dtype=float)).all(axis=1)
    clean = work.loc[finite_mask].sort_values("ts").reset_index(drop=True)
    finite_share = float(len(clean) / max(1, len(work)))
    if len(clean) < int(config.min_rows):
        return [], _mask(axis, band, interval, window_profile.window_profile_id, feature_pool, "insufficient_rows", f"finite rows {len(clean)} < min_rows {config.min_rows}", row_count=len(clean), finite_share=finite_share, coverage_gate=coverage_gate.as_dict()), None
    if finite_share < float(config.min_finite_share):
        return [], _mask(axis, band, interval, window_profile.window_profile_id, feature_pool, "insufficient_finite_share", f"finite share {finite_share:.3f} < {config.min_finite_share:.3f}", row_count=len(clean), finite_share=finite_share, coverage_gate=coverage_gate.as_dict()), None
    if _low_variation(clean, features):
        return [], _mask(axis, band, interval, window_profile.window_profile_id, feature_pool, "low_variance_near_flat", "axis feature pool has low variation", row_count=len(clean), finite_share=finite_share, coverage_gate=coverage_gate.as_dict()), None

    rows: list[dict[str, Any]] = []
    for method in tuple(config.candidate_profiles or ()):
        row = _score_method(
            clean,
            axis=axis,
            band=band,
            interval=interval,
            window_profile_id=window_profile.window_profile_id,
            feature_pool=feature_pool,
            features=features,
            method=method,
            row_count=len(clean),
            finite_share=finite_share,
            config=config,
        )
        rows.append(row)
    eligible = [row for row in rows if str(row["status"]) == "candidate" and bool(row.get("production_candidate"))]
    if not eligible:
        reason = rows[0]["reason"] if rows else "no candidate methods"
        return rows, _mask(axis, band, interval, window_profile.window_profile_id, feature_pool, "no_healthy_candidate", reason, row_count=len(clean), finite_share=finite_share, coverage_gate=coverage_gate.as_dict()), None
    winner = sorted(
        eligible,
        key=lambda row: (float(row["semantic_candidate_score"]), float(row["runtime_adjusted_score"]), str(row["candidate_id"])),
        reverse=True,
    )[0]
    source_tail_ts = _max_numeric(clean.get("source_tail_ts"))
    known_at_ts = _max_numeric(clean.get("known_at_ts"))
    selected = {
        "profile_id": _profile_id(axis, band, interval, window_profile.window_profile_id, winner["method_profile"], feature_pool),
        "market_axis": axis,
        "band": band,
        "window_profile_id": window_profile.window_profile_id,
        "window_profile": window_profile.as_dict(),
        "source_interval": int(interval),
        "selected_method_profile": winner["method_profile"],
        "selected_method_family": winner["method_family"],
        "selected_feature_pool": feature_pool,
        "selected_feature_set": list(features),
        "preprocessing": winner["preprocessing"],
        "reducer": winner["reducer"],
        "candidate_status": winner["candidate_status"],
        "production_candidate": bool(winner["production_candidate"]),
        "diagnostic_only": bool(winner["diagnostic_only"]),
        "tuned_core_parameters": json.loads(str(winner["core_parameters_json"])),
        "score_evidence_summary": {
            "semantic_candidate_score": float(winner["semantic_candidate_score"]),
            "runtime_penalty": float(winner["runtime_penalty"]),
            "runtime_adjusted_score": float(winner["runtime_adjusted_score"]),
            "silhouette": _nullable_float(winner["silhouette"]),
            "effective_state_count": int(winner["effective_state_count"]),
            "row_count": int(winner["row_count"]),
            "finite_share": float(winner["finite_share"]),
            "selection_score_policy": "semantic_candidate_score_primary_runtime_adjusted_tiebreak",
        },
        "coverage_summary": coverage_gate.summary.as_dict(),
        "coverage_gate": coverage_gate.as_dict(),
        "label_output_health_gate_summary": {
            "status": "passed",
            "non_degenerate": True,
            "finite_input": True,
            "selected_rows": int(winner["row_count"]),
        },
        "source_tail_ts": source_tail_ts,
        "known_at_ts": known_at_ts,
        "run_id": config.run_id,
        "trial_study_lineage": {
            "candidate_id": winner["candidate_id"],
            "selection_scope": f"{axis}/{band}/{window_profile.window_profile_id}",
        },
        "selection_scope": "market_axis_band",
        "availability_status": "selected",
        "mask_reason_code": None,
        **apply_nonproduction_gate_flags({}, include_canonical=True),
        **test_branch_consumable_status_fields(),
    }
    return rows, None, selected


def _score_method(
    clean: pd.DataFrame,
    *,
    axis: str,
    band: str,
    interval: int,
    window_profile_id: str,
    feature_pool: str,
    features: Sequence[str],
    method: Mapping[str, Any],
    row_count: int,
    finite_share: float,
    config: MarketStateTestBranchCampaignConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    method_family = str(method["method_family"])
    n_clusters = int(method.get("n_clusters") or method.get("n_components") or 0)
    preprocessing = str(method.get("preprocessing") or "standard_scale")
    reducer = str(method.get("reducer") or "none")
    candidate_status = str(method.get("candidate_status") or CANDIDATE_STATUS_PRODUCTION_CANDIDATE)
    production_candidate = candidate_status == CANDIDATE_STATUS_PRODUCTION_CANDIDATE
    diagnostic_only = candidate_status == CANDIDATE_STATUS_DIAGNOSTIC_ONLY
    candidate_id = _profile_id(axis, band, interval, window_profile_id, str(method["method_profile"]), feature_pool)
    core_parameters = _core_parameters(method, random_seed=int(config.random_seed))
    base = {
        "run_id": config.run_id,
        "candidate_id": candidate_id,
        "axis": axis,
        "band": band,
        "interval": int(interval),
        "window_profile_id": str(window_profile_id),
        "feature_pool": feature_pool,
        "feature_set": "|".join(features),
        "method_family": method_family,
        "method_profile": str(method["method_profile"]),
        "preprocessing": preprocessing,
        "reducer": reducer,
        "candidate_status": candidate_status,
        "production_candidate": bool(production_candidate),
        "diagnostic_only": bool(diagnostic_only),
        "core_parameters_json": json.dumps(core_parameters, sort_keys=True),
        "row_count": int(row_count),
        "finite_share": float(finite_share),
        "early_elimination_used": False,
    }
    try:
        if n_clusters and row_count <= n_clusters:
            raise ValueError("row_count_not_greater_than_cluster_count")
        x = _prepare_candidate_matrix(
            clean,
            features=features,
            preprocessing=preprocessing,
            reducer=reducer,
            random_seed=int(config.random_seed),
        )
        labels = _fit_predict(x, method_family=method_family, core_parameters=core_parameters)
        counts = pd.Series(labels).value_counts()
        effective = int(counts.shape[0])
        min_share = float(counts.min() / max(1, len(labels))) if effective else 0.0
        degenerate = effective < 2 or min_share < 0.03
        silhouette = _silhouette(x, labels)
        semantic = _semantic_score(silhouette=silhouette, effective_state_count=effective, min_cluster_share=min_share, degenerate=degenerate)
        elapsed = max(0.0, time.perf_counter() - started)
        runtime_penalty = min(0.05, elapsed / 600.0)
        status = "rejected" if degenerate else "diagnostic" if diagnostic_only else "candidate"
        return {
            **base,
            "status": status,
            "reason": "degenerate_cluster_labels" if degenerate else "diagnostic_only_not_selectable" if diagnostic_only else "",
            **score_contract_fields(semantic_candidate_score=semantic, runtime_penalty=runtime_penalty),
            "elapsed_s": elapsed,
            "silhouette": "" if silhouette is None else float(silhouette),
            "effective_state_count": effective,
            "min_cluster_share": min_share,
            "label_health_gate_pass": not degenerate,
        }
    except Exception as exc:
        elapsed = max(0.0, time.perf_counter() - started)
        return {
            **base,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            **score_contract_fields(semantic_candidate_score=0.0, runtime_penalty=min(0.05, elapsed / 600.0)),
            "elapsed_s": elapsed,
            "silhouette": "",
            "effective_state_count": 0,
            "min_cluster_share": 0.0,
            "label_health_gate_pass": False,
        }


def _fit_predict(x: np.ndarray, *, method_family: str, core_parameters: Mapping[str, Any]) -> np.ndarray:
    clusterer_family, _embedding = clusterer_and_embedding_for_method(method_family)
    adapter = default_clusterer_registry().build(clusterer_family, **dict(core_parameters))
    fit = adapter.fit(x)
    if fit.status != "fitted":
        failure = fit.failure_metadata.reason_code if fit.failure_metadata is not None else fit.status
        raise ValueError(f"{clusterer_family}_fit_{failure}")
    return np.asarray(fit.labels, dtype=int)


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    from sklearn.metrics import silhouette_score

    value = float(silhouette_score(x, labels))
    return value if np.isfinite(value) else None


def _semantic_score(*, silhouette: float | None, effective_state_count: int, min_cluster_share: float, degenerate: bool) -> float:
    if degenerate:
        return 0.0
    sil = 0.0 if silhouette is None else max(-1.0, min(1.0, float(silhouette)))
    sil_scaled = (sil + 1.0) / 2.0
    balance = max(0.0, min(1.0, float(min_cluster_share) / 0.20))
    state_bonus = min(0.10, max(0, int(effective_state_count) - 2) * 0.02)
    return float(max(0.0, min(1.0, 0.75 * sil_scaled + 0.20 * balance + state_bonus)))


def _load_feature_frames(
    feature_root: Path,
    *,
    config: MarketStateTestBranchCampaignConfig,
) -> tuple[dict[tuple[str, int, str], pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[tuple[str, int, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for band, interval in config.band_intervals:
        root = feature_root / f"regime_features_market_{int(interval)}" / f"band={band}"
        paths = sorted(root.glob("year=*/month=*/*.parquet"), key=lambda path: str(path).lower())
        selected_paths = paths
        parts: list[pd.DataFrame] = []
        for path in selected_paths:
            frame = pd.read_parquet(path)
            parts.append(frame)
            rows.append({"band": band, "interval": int(interval), "path": str(path), "row_count": int(frame.shape[0]), "column_count": int(frame.shape[1])})
        base = pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True) if parts else pd.DataFrame()
        for profile in _profiles_for_band(config, band):
            frames[(band, int(interval), profile.window_profile_id)] = _windowed_feature_frame(base, profile)
    return frames, rows


def _selected_manifest(
    *,
    config: MarketStateTestBranchCampaignConfig,
    feature_root: Path,
    feature_root_source: str,
    runtime_config: MarketStateTestRuntimeConfig,
    selected: Sequence[Mapping[str, Any]],
    masked: Sequence[Mapping[str, Any]],
    scoreboard: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    work_matrix: Sequence[Mapping[str, Any]],
    phase_timings: Mapping[str, float],
    elapsed_s: float,
) -> dict[str, Any]:
    work_summary = _work_matrix_summary(work_matrix)
    telemetry_summary = _runtime_telemetry_summary(
        scoreboard=scoreboard,
        selected=selected,
        masked=masked,
        telemetry_rows=(),
        phase_timings=phase_timings,
        runtime_config=runtime_config,
        config=config,
        work_matrix=work_matrix,
    )
    return {
        "artifact_kind": MARKET_STATE_TEST_CAMPAIGN_ARTIFACT_KIND,
        "schema_version": MARKET_STATE_TEST_CAMPAIGN_SCHEMA_VERSION,
        "run_id": config.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "market_axis_band",
        "profile_grain": "market_axis x band",
        "selected_profile_count": int(len(selected)),
        "masked_or_skipped_count": int(len(masked)),
        "candidate_count": int(len(scoreboard)),
        "expected_candidate_work_item_count": int(len(work_matrix)),
        "expected_axis_band_cell_count": int(work_summary["axis_band_cell_count"]),
        "expected_axis_band_window_cell_count": int(work_summary["axis_band_window_cell_count"]),
        "selected_profiles": list(selected),
        "masked_or_skipped_cells": list(masked),
        "source_lineage": {
            "feature_root": str(feature_root),
            "feature_root_source": feature_root_source,
            "input_partitions_read": list(input_rows),
            "window_profiles": list(window_profile_rows(tuple(config.window_profiles or ()))),
            "canonical_market_regime_features": True,
            "stale_sandbox_manifest_used": False,
        },
        "candidate_method_universe": {
            "methods": [dict(item) for item in tuple(config.candidate_profiles or ())],
            "preprocessing_options": list(PREPROCESSING_OPTIONS),
            "reducer_options": list(REDUCER_OPTIONS),
            "optuna_used": False,
            "grid_policy": "expanded_explicit_market_state_grid",
            "early_elimination_used": False,
            "diagnostic_candidates_selectable_for_production": False,
        },
        "deterministic_work_matrix": work_summary,
        "scoring_contract": scoring_contract_manifest_section(),
        "runtime": {
            "elapsed_s": float(elapsed_s),
            "runtime_config": runtime_config.as_dict(),
            "model_threads": int(config.model_threads),
            "writer_workers": int(runtime_config.writer_workers),
            "phase_timings": dict(phase_timings),
            "telemetry_summary": telemetry_summary,
        },
        "production_promotion_performed": False,
        **apply_nonproduction_gate_flags({}, include_canonical=True),
        **test_branch_consumable_status_fields(),
        "single_active_nonproduction_handoff_artifact": MARKET_STATE_SELECTED_PROFILES_FILENAME,
    }


def _summary_payload(
    selected_manifest: Mapping[str, Any],
    scoreboard: Sequence[Mapping[str, Any]],
    masked: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    *,
    phase_timings: Mapping[str, float],
) -> dict[str, Any]:
    selected = list(selected_manifest.get("selected_profiles") or [])
    winners = [
        {
            "axis": item.get("market_axis"),
            "band": item.get("band"),
            "window_profile_id": item.get("window_profile_id"),
            "method_profile": item.get("selected_method_profile"),
            "method_family": item.get("selected_method_family"),
            "preprocessing": item.get("preprocessing"),
            "reducer": item.get("reducer"),
            "feature_pool": item.get("selected_feature_pool"),
            "semantic_candidate_score": (item.get("score_evidence_summary") or {}).get("semantic_candidate_score"),
        }
        for item in selected
    ]
    runtime_config = (selected_manifest.get("runtime") or {}).get("runtime_config") or {}
    return {
        "artifact_kind": "market_state_test_branch_campaign_summary",
        "schema_version": MARKET_STATE_TEST_CAMPAIGN_SCHEMA_VERSION,
        "status": "complete_ready_for_human_review" if selected else "blocked_no_selected_profiles",
        "run_id": selected_manifest.get("run_id"),
        "axes_covered": sorted({str(item.get("market_axis")) for item in selected} | {str(item.get("axis")) for item in masked}),
        "bands_covered": sorted({str(item.get("band")) for item in selected} | {str(item.get("band")) for item in masked}),
        "selected_profile_count": len(selected),
        "masked_or_skipped_count": len(masked),
        "candidate_count": len(scoreboard),
        "expected_candidate_work_item_count": int(selected_manifest.get("expected_candidate_work_item_count") or 0),
        "expected_axis_band_cell_count": int(selected_manifest.get("expected_axis_band_cell_count") or 0),
        "expected_axis_band_window_cell_count": int(selected_manifest.get("expected_axis_band_window_cell_count") or 0),
        "method_profile_winners": winners,
        "runtime": selected_manifest.get("runtime", {}),
        "runtime_telemetry": _runtime_telemetry_summary(
            scoreboard=scoreboard,
            selected=selected,
            masked=masked,
            telemetry_rows=telemetry_rows,
            phase_timings=phase_timings,
            runtime_config=runtime_config,
            config=None,
            work_matrix=(),
        ),
        "selected_profile_manifest_validation": selected_manifest.get("selected_profile_manifest_validation", {}),
        "one_active_manifest_produced": True,
        "production_consumer_contract_ready": bool(selected),
        "remaining_polish_items": ["review masked_or_skipped_cells before production approval"] if masked else [],
        "safety": {
            "production_writes": False,
            "production_labels": False,
            "canonical_production_state_outputs": False,
            "production_promotion": False,
            "broad_cross_asset_run": False,
            "cleanup_quarantine_delete": False,
        },
        "artifacts": {
            "selected_profiles": MARKET_STATE_SELECTED_PROFILES_FILENAME,
            "candidate_scoreboard": MARKET_STATE_CANDIDATE_SCOREBOARD_FILENAME,
            "masked_or_skipped_cells": MARKET_STATE_MASKED_CELLS_FILENAME,
            "runtime_telemetry": MARKET_STATE_RUNTIME_TELEMETRY_FILENAME,
            "summary": MARKET_STATE_CAMPAIGN_SUMMARY_FILENAME,
        },
    }


def _validate_campaign_manifest(config: MarketStateTestBranchCampaignConfig, manifest: Mapping[str, Any]):
    expected = [
        {"market_axis": axis, "band": str(band)}
        for band, _interval in tuple(config.band_intervals)
        for axis in MARKET_STATE_V1_AXIS_IDS
    ]
    return validate_selected_profile_manifest(
        manifest,
        active_filename=MARKET_STATE_SELECTED_PROFILES_FILENAME,
        expected_cells=expected,
        selected_cell_key_fields=("market_axis", "band"),
        masked_cell_key_fields=("axis", "band"),
        require_canonical_gate_field=True,
        expected_requires_human_approval=False,
    )


def _work_matrix_summary(work_matrix: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in work_matrix]
    axis_band = {(str(row.get("market_axis")), str(row.get("band"))) for row in rows}
    axis_band_window = {
        (str(row.get("market_axis")), str(row.get("band")), str(row.get("window_profile_id")))
        for row in rows
    }
    return {
        "matrix_policy": "market_axis x band x window_profile x feature_pool x method_profile x preprocessing x reducer",
        "execution_unit": "market_axis_band_window_cell",
        "candidate_scoring_unit": "method_profile_with_preprocessing_and_reducer",
        "axis_count": len({str(row.get("market_axis")) for row in rows}),
        "band_count": len({str(row.get("band")) for row in rows}),
        "window_profile_count": len({str(row.get("window_profile_id")) for row in rows}),
        "feature_pool_count": len({str(row.get("feature_pool")) for row in rows}),
        "method_profile_count": len({str(row.get("method_profile")) for row in rows}),
        "candidate_work_item_count": len(rows),
        "axis_band_cell_count": len(axis_band),
        "axis_band_window_cell_count": len(axis_band_window),
        "method_counts": _count_by(rows, "method_family"),
        "candidate_status_counts": _count_by(rows, "candidate_status"),
        "preprocessing_counts": _count_by(rows, "preprocessing"),
        "reducer_counts": _count_by(rows, "reducer"),
        "window_counts": _count_by(rows, "window_profile_id"),
        "band_window_counts": _count_by(rows, "band", "window_profile_id"),
        "no_interval_band_cartesian_expansion": True,
        "band_intervals": sorted({f"{row.get('band')}:{row.get('interval')}" for row in rows}),
        "diagnostic_candidates_selectable_for_production": False,
        "work_matrix_sample": rows[:20],
    }


def _runtime_telemetry_summary(
    *,
    scoreboard: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    masked: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    phase_timings: Mapping[str, float],
    runtime_config: Any,
    config: MarketStateTestBranchCampaignConfig | None,
    work_matrix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_rows = [dict(row) for row in scoreboard]
    telemetry = [dict(row) for row in telemetry_rows]
    matrix_rows = [dict(row) for row in work_matrix]
    runtime_payload = runtime_config.as_dict() if hasattr(runtime_config, "as_dict") else dict(runtime_config or {})
    candidate_status_counts = _count_by(score_rows, "status")
    coverage_passed_cells = len(
        {
            (str(row.get("axis")), str(row.get("band")), str(row.get("window_profile_id")))
            for row in score_rows
        }
    )
    coverage_reason_counts: dict[str, int] = {}
    for row in masked:
        reason = str(row.get("mask_reason_code") or "unknown")
        coverage_reason_counts[reason] = coverage_reason_counts.get(reason, 0) + 1
    elapsed_values = [_nullable_float(row.get("elapsed_s")) for row in score_rows]
    elapsed_values = [float(value) for value in elapsed_values if value is not None]
    candidate_count = len(score_rows)
    expected_candidate_count = len(matrix_rows) if matrix_rows else candidate_count
    return {
        "workers": int(runtime_payload.get("market_state_test_workers") or (config.workers if config else 0) or MARKET_STATE_TEST_DEFAULT_WORKERS),
        "writer_workers": int(runtime_payload.get("writer_workers") or (config.writer_workers if config else MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS)),
        "model_threads": int((config.model_threads if config else None) or runtime_payload.get("model_threads") or 1),
        "thread_caps_enforced": bool(runtime_payload.get("thread_caps_enforced", True)),
        "phase_timings": {str(key): float(value) for key, value in sorted(dict(phase_timings).items())},
        "candidate_count": int(candidate_count),
        "expected_candidate_work_item_count": int(expected_candidate_count),
        "candidate_completion_ratio": float(candidate_count / expected_candidate_count) if expected_candidate_count else 0.0,
        "selected_count": int(len(selected)),
        "masked_count": int(len(masked)),
        "telemetry_cell_count": int(len(telemetry)),
        "method_counts": _count_by(score_rows, "method_family") if score_rows else _count_by(matrix_rows, "method_family"),
        "method_status_counts": candidate_status_counts,
        "window_counts": _count_by(score_rows, "window_profile_id") if score_rows else _count_by(matrix_rows, "window_profile_id"),
        "selected_window_counts": _count_by(selected, "window_profile_id"),
        "coverage_stats": {
            "evaluated_axis_band_window_cells": int(len(telemetry)),
            "coverage_passed_axis_band_window_cells": int(coverage_passed_cells),
            "masked_or_skipped_cells": int(len(masked)),
            "coverage_mask_reason_counts": coverage_reason_counts,
        },
        "candidate_elapsed_s": {
            "min": min(elapsed_values) if elapsed_values else None,
            "median": float(np.median(elapsed_values)) if elapsed_values else None,
            "max": max(elapsed_values) if elapsed_values else None,
            "sum": float(np.sum(elapsed_values)) if elapsed_values else 0.0,
        },
        "parent_single_finalizer_writer": True,
        "production_writes": False,
        "production_labels": False,
    }


def _count_by(rows: Sequence[Mapping[str, Any]], *fields: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = "|".join(str(row.get(field)) for field in fields)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _select_profiles_by_axis_band(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in selected:
        row = dict(item)
        key = (str(row.get("market_axis")), str(row.get("band")))
        by_cell.setdefault(key, []).append(row)
    winners: list[dict[str, Any]] = []
    for key in sorted(by_cell):
        rows = by_cell[key]
        winner = sorted(
            rows,
            key=lambda row: (
                float((row.get("score_evidence_summary") or {}).get("semantic_candidate_score") or -1.0),
                float((row.get("score_evidence_summary") or {}).get("runtime_adjusted_score") or -1.0),
                str(row.get("profile_id")),
            ),
            reverse=True,
        )[0]
        winners.append(winner)
    return winners


def _mask(
    axis: str,
    band: str,
    interval: int,
    window_profile_id: str,
    feature_pool: str,
    reason_code: str,
    reason: str,
    *,
    row_count: int = 0,
    finite_share: float = 0.0,
    coverage_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "axis": axis,
        "band": band,
        "interval": int(interval),
        "window_profile_id": str(window_profile_id),
        "feature_pool": feature_pool,
        "availability_status": "masked_unavailable",
        "mask_reason_code": reason_code,
        "reason": reason,
        "profile_id": _profile_id(axis, band, interval, window_profile_id, "masked_unavailable", feature_pool),
        "row_count": int(row_count),
        "finite_share": float(finite_share),
        "coverage_gate": to_jsonable(dict(coverage_gate or {})),
        **apply_nonproduction_gate_flags({}, include_canonical=True),
        **test_branch_consumable_status_fields(),
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    out = (x - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_candidate_matrix(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    preprocessing: str,
    reducer: str,
    random_seed: int,
) -> np.ndarray:
    x = frame[list(features)].to_numpy(dtype=float)
    preprocessed = _preprocess_matrix(x, preprocessing=preprocessing)
    return _reduce_matrix(preprocessed, reducer=reducer, random_seed=random_seed)


def _preprocess_matrix(x: np.ndarray, *, preprocessing: str) -> np.ndarray:
    mode = str(preprocessing).strip().lower()
    if mode == "standard_scale":
        return _standardize(x)
    if mode == "robust_scale":
        median = np.nanmedian(x, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        scale = np.where((q75 - q25) <= 1e-12, 1.0, q75 - q25)
        return np.nan_to_num((x - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "noop":
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"unsupported preprocessing {preprocessing!r}")


def _reduce_matrix(x: np.ndarray, *, reducer: str, random_seed: int) -> np.ndarray:
    mode = str(reducer).strip().lower()
    if mode == "none":
        return x
    components = max(1, min(2, int(x.shape[1]), max(1, int(x.shape[0]) - 1)))
    if mode == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=components, random_state=int(random_seed)).fit_transform(x)
    if mode == "factor_analysis":
        from sklearn.decomposition import FactorAnalysis

        return FactorAnalysis(n_components=components, random_state=int(random_seed)).fit_transform(x)
    raise ValueError(f"unsupported reducer {reducer!r}")


def _core_parameters(method: Mapping[str, Any], *, random_seed: int) -> dict[str, Any]:
    method_family = str(method.get("method_family"))
    n_clusters = int(method.get("n_clusters") or 0)
    if method_family in {"kmeans", "pca_kmeans", "factor_analysis_kmeans"}:
        return {"n_clusters": n_clusters, "n_init": 10, "random_state": int(random_seed)}
    if method_family == "minibatch_kmeans":
        return {"n_clusters": n_clusters, "n_init": 3, "batch_size": 128, "random_state": int(random_seed)}
    if method_family in {"gaussian_mixture", "factor_analysis_gaussian_mixture"}:
        return {"n_components": n_clusters, "covariance_type": "full", "random_state": int(random_seed)}
    if method_family == "bayesian_gaussian_mixture":
        return {"n_components": n_clusters, "covariance_type": "full", "random_state": int(random_seed)}
    if method_family == "birch":
        return {"n_clusters": n_clusters, "threshold": 0.5}
    if method_family == "hdbscan":
        return {
            "min_cluster_size": int(method.get("min_cluster_size") or 8),
            "min_samples": int(method.get("min_samples") or 3),
            "allow_single_cluster": True,
            "prediction_data": True,
        }
    if method_family == "optics":
        return {"min_samples": int(method.get("min_samples") or 8), "xi": float(method.get("xi") or 0.05)}
    if method_family == "agglomerative":
        return {"n_clusters": n_clusters}
    raise ValueError(f"unsupported method_family {method_family!r}")


def _low_variation(frame: pd.DataFrame, features: Sequence[str]) -> bool:
    values = frame[list(features)].to_numpy(dtype=float)
    std = np.nanstd(values, axis=0)
    return bool(np.all(std <= 1e-12))


def _max_numeric(values: Any) -> int | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce").dropna()
    return None if series.empty else int(series.max())


def _nullable_float(value: object) -> float | None:
    if value == "" or value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _profiles_for_band(config: MarketStateTestBranchCampaignConfig, band: str) -> tuple[RegimeWindowProfile, ...]:
    normalized = str(band).strip().lower()
    return tuple(profile for profile in tuple(config.window_profiles or ()) if profile.band == normalized)


def _windowed_feature_frame(frame: pd.DataFrame, profile: RegimeWindowProfile) -> pd.DataFrame:
    if frame.empty or "ts" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["ts"] = pd.to_numeric(work["ts"], errors="coerce")
    work = work.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if work.empty:
        return work
    source_tail = _max_numeric(work.get("source_tail_ts"))
    if source_tail is None:
        source_tail = int(work["ts"].max())
    resolved = profile.resolve(source_tail_ts=source_tail)
    if resolved.start_ts is not None:
        work = work.loc[work["ts"] >= int(resolved.start_ts)]
    if resolved.end_ts is not None:
        work = work.loc[work["ts"] <= int(resolved.end_ts)]
    if resolved.row_cap is not None and int(work.shape[0]) > int(resolved.row_cap):
        work = work.tail(int(resolved.row_cap))
    return work.sort_values("ts").reset_index(drop=True)


def _profile_id(axis: str, band: str, interval: int, window_profile_id: str, method_profile: str, feature_pool: str) -> str:
    return "market_state_" + "_".join(
        _safe_token(part) for part in (axis, band, str(interval), window_profile_id, feature_pool, method_profile)
    )


def _resolve_feature_root(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Market-State feature root does not exist: {root}")
        return root, "explicit_argument"
    env_root = os.environ.get("PIPELINE_MARKET_STATE_FEATURE_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.exists() and root.is_dir():
            return root, "env.PIPELINE_MARKET_STATE_FEATURE_ROOT"
    profile = selected_profile()
    for key in ("source_regime_root", "output_parquet_root"):
        resolved = resolve_path(key, profile=profile, required=False)
        if resolved is None:
            continue
        candidate = (resolved / "regime_features").resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate, f"path_config.{key}/regime_features"
    raise FileNotFoundError("Market-State feature root is not configured; pass --feature-root or PIPELINE_MARKET_STATE_FEATURE_ROOT")


def _safe_output_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    parts = {part.lower() for part in resolved.parts}
    if parts.intersection({"production", "prod", "live", "promoted", "promotion", "regime_labels", "market_state_labels"}):
        raise ValueError("Market-State Test Branch campaign refusing production-like output root")
    return resolved


def _safe_token(value: object) -> str:
    text = str(value).strip().lower() or "unknown"
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in text)


def _apply_model_thread_caps(model_threads: int) -> None:
    value = str(max(1, int(model_threads)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _scoreboard_fields() -> tuple[str, ...]:
    return (
        "run_id",
        "candidate_id",
        "axis",
        "band",
        "interval",
        "window_profile_id",
        "feature_pool",
        "feature_set",
        "method_family",
        "method_profile",
        "preprocessing",
        "reducer",
        "candidate_status",
        "production_candidate",
        "diagnostic_only",
        "core_parameters_json",
        "row_count",
        "finite_share",
        "status",
        "reason",
        "semantic_candidate_score",
        "candidate_score",
        "runtime_penalty",
        "runtime_adjusted_score",
        "score_policy",
        "elapsed_s",
        "silhouette",
        "effective_state_count",
        "min_cluster_share",
        "label_health_gate_pass",
        "early_elimination_used",
    )


def _masked_fields() -> tuple[str, ...]:
    return (
        "axis",
        "band",
        "interval",
        "window_profile_id",
        "feature_pool",
        "availability_status",
        "mask_reason_code",
        "reason",
        "profile_id",
        "row_count",
        "finite_share",
        "production_approved",
        "production_writer_enabled",
        "production_labels_written",
        "production_outputs_written",
        "canonical_production_state_outputs_written",
    )


def _telemetry_fields() -> tuple[str, ...]:
    return (
        "run_id",
        "axis",
        "band",
        "interval",
        "window_profile_id",
        "row_count",
        "candidate_count",
        "selected",
        "masked_or_skipped",
        "elapsed_s",
        "worker_count",
        "model_threads",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the non-production Market-State Test Branch selected-profile campaign.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default=f"market_state_test_branch_campaign_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--band-interval", action="append", default=[])
    parser.add_argument("--max-files-per-band", type=int, default=6)
    parser.add_argument("--min-rows", type=int, default=48)
    parser.add_argument("--workers", type=int, default=MARKET_STATE_TEST_DEFAULT_WORKERS)
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument("--writer-workers", type=int, default=MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    band_intervals = _parse_band_intervals(args.band_interval)
    result = run_market_state_test_branch_campaign(
        MarketStateTestBranchCampaignConfig(
            output_root=args.output_root,
            run_id=args.run_id,
            feature_root=args.feature_root,
            band_intervals=band_intervals,
            max_files_per_band=args.max_files_per_band,
            min_rows=args.min_rows,
            workers=args.workers,
            model_threads=args.model_threads,
            writer_workers=args.writer_workers,
        )
    )
    payload = result.as_dict()
    print(json.dumps(payload if args.json else {key: payload[key] for key in ("status", "selected_profile_count", "masked_or_skipped_count", "summary_path")}, indent=2, sort_keys=True))
    return 0 if result.selected_profile_count > 0 else 2


def _parse_band_intervals(raw: Sequence[str]) -> tuple[tuple[str, int], ...]:
    if not raw:
        return DEFAULT_BAND_INTERVALS
    out: list[tuple[str, int]] = []
    for item in raw:
        if ":" not in str(item):
            raise SystemExit(f"--band-interval must use band:interval form, got {item!r}")
        band, interval = str(item).split(":", 1)
        out.append((band.strip().lower(), int(interval)))
    return tuple(out)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MARKET_STATE_CANDIDATE_SCOREBOARD_FILENAME",
    "MARKET_STATE_MASKED_CELLS_FILENAME",
    "MARKET_STATE_RUNTIME_TELEMETRY_FILENAME",
    "MARKET_STATE_SELECTED_PROFILES_FILENAME",
    "MARKET_STATE_CAMPAIGN_SUMMARY_FILENAME",
    "MarketStateTestBranchCampaignConfig",
    "MarketStateTestBranchCampaignResult",
    "build_market_state_campaign_work_matrix",
    "default_market_state_candidate_profiles",
    "legacy_market_state_candidate_profiles",
    "run_market_state_test_branch_campaign",
]
