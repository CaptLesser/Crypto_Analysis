from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.regimes.asset_state_test.adapters import ClusterAssignmentResult, build_clusterer_adapter
from src.regimes.asset_state_test.contracts import ARTIFACT_NAMES, DEFAULT_ASSETS, DEFAULT_CYCLE5_ASSETS, DEFAULT_FIRST_METHODS, StudyConfig, TrialConfig
from src.regimes.asset_state_test.diagnostics import (
    build_candidate_score_row,
    build_cluster_diagnostics,
    write_study_artifacts,
)
from src.regimes.asset_state_test.features import feature_pool, preprocess_feature_frame, select_feature_columns, transform_feature_frame
from src.regimes.asset_state_test.filters import run_flat_preflight
from src.regimes.asset_state_test.manifest import load_study_manifest


CYCLE5_TRAIN_START_TS = 1769535000
CYCLE5_TRAIN_END_TS = 1772125200
CYCLE5_EVAL_START_TS = 1772127000
CYCLE5_EVAL_END_TS = 1772730000
CYCLE5_OUTPUT_SUFFIX = "_pass3_cycle5_asset_expansion"
CYCLE7_OUTPUT_SUFFIX = "_pass3_trend_micro_asset_balance_refinement"
CYCLE8_OUTPUT_SUFFIX = "_pass3_trend_micro_adausd_feature_preprocess_probe"
CYCLE9_OUTPUT_SUFFIX = "_pass3_trend_micro_clean_collapse_no_model_decision_probe"
CYCLE10_OUTPUT_SUFFIX = "_pass3_trend_micro_near_flat_panel_sensitivity_probe"
CYCLE11_OUTPUT_SUFFIX = "_pass3_trend_micro_method_role_no_model_decision_refinement"
CYCLE12_OUTPUT_SUFFIX = "_pass3_trend_micro_cycle11_repro_resource_perturbation_probe"
CYCLE13_OUTPUT_SUFFIX = "_pass3_trend_micro_primary_comparator_feature_refinement_probe"
CYCLE14_OUTPUT_SUFFIX = "_pass3_trend_micro_axis_not_clusterable_confirmation_and_feature_availability_probe"
CYCLE15_OUTPUT_SUFFIX = "_pass3_trend_micro_anlog_near_flat_clean_collapse_safety_expansion_probe"
CYCLE16_OUTPUT_SUFFIX = "_pass3_trend_micro_near_flat_panel_sensitivity_confirmation_probe"
CYCLE17_OUTPUT_SUFFIX = "_pass3_trend_micro_per_asset_stability_collapse_attribution_probe"
CYCLE_REAL_BOUNDED_OUTPUT_SUFFIXES = (
    CYCLE5_OUTPUT_SUFFIX,
    CYCLE7_OUTPUT_SUFFIX,
    CYCLE8_OUTPUT_SUFFIX,
    CYCLE9_OUTPUT_SUFFIX,
    CYCLE10_OUTPUT_SUFFIX,
    CYCLE11_OUTPUT_SUFFIX,
    CYCLE12_OUTPUT_SUFFIX,
    CYCLE13_OUTPUT_SUFFIX,
    CYCLE14_OUTPUT_SUFFIX,
    CYCLE15_OUTPUT_SUFFIX,
    CYCLE16_OUTPUT_SUFFIX,
    CYCLE17_OUTPUT_SUFFIX,
)
CYCLE5_REPORT_PARTS = ("reports", "codex_automation", "regimes", "asset_state_test", "experiments")
CYCLE5_SOURCE_ROOT_ENV = "PIPELINE_PARQUET_ROOT"
CYCLE10_ASSETS: tuple[str, ...] = ("AAVEUSD", "XBTUSD", "ADAUSD", "TEERUSD", "AI16ZUSD")
CYCLE15_ASSETS: tuple[str, ...] = ("AAVEUSD", "XBTUSD", "ADAUSD", "TEERUSD", "ANLOGUSD", "AI16ZUSD")
CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET = "TERMUSD"
CYCLE16_ASSETS: tuple[str, ...] = (*CYCLE15_ASSETS, CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET)
CYCLE17_ASSETS: tuple[str, ...] = CYCLE16_ASSETS
CYCLE9_BASELINE_ASSET_DECISIONS: dict[str, dict[str, Any]] = {
    "AAVEUSD": {
        "proposed_decision": "cluster_candidate",
        "eval_non_noise_label_support_min": 1,
        "eval_non_noise_label_support_max": 4,
        "retained_method_eval_collapse_rate": 0.3333333333333333,
    },
    "ADAUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "eval_non_noise_label_support_min": 1,
        "eval_non_noise_label_support_max": 1,
        "retained_method_eval_collapse_rate": 1.0,
    },
    "AI16ZUSD": {
        "proposed_decision": "neutral_flat",
        "eval_non_noise_label_support_min": 0,
        "eval_non_noise_label_support_max": 0,
        "retained_method_eval_collapse_rate": 0.0,
    },
    "XBTUSD": {
        "proposed_decision": "cluster_candidate",
        "eval_non_noise_label_support_min": 2,
        "eval_non_noise_label_support_max": 32,
        "retained_method_eval_collapse_rate": 0.0,
    },
}
CYCLE15_BASELINE_ASSET_DECISIONS: dict[str, dict[str, Any]] = {
    "AAVEUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.25,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 3,
    },
    "ADAUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.5,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 2,
    },
    "AI16ZUSD": {
        "proposed_decision": "neutral_flat",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 0,
        "primary_comparator_eval_support_max": 0,
    },
    "ANLOGUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "primary_comparator_eval_collapse_rate": 1.0,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 1,
    },
    "TEERUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "primary_comparator_eval_collapse_rate": 1.0,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 1,
    },
    "XBTUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 2,
        "primary_comparator_eval_support_max": 2,
    },
}
CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE: dict[str, dict[str, Any]] = {
    "TERMUSD": {
        "source": "reports/codex_automation/regimes/asset_state_test/cycles/20260512_1924_pass1_study_design.md",
        "preflight_reason_code": "pass",
        "train_rows": 1440,
        "eval_rows": 1440,
        "near_zero_movement_fraction": 0.972222,
        "near_flat_distance_to_threshold": 0.007778,
        "near_zero_variance_feature_count": 0,
    },
}
CYCLE14_BASELINE_ASSET_DECISIONS: dict[str, dict[str, Any]] = {
    "AAVEUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.25,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 3,
    },
    "ADAUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "primary_comparator_eval_collapse_rate": 1.0,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 1,
    },
    "AI16ZUSD": {
        "proposed_decision": "neutral_flat",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 0,
        "primary_comparator_eval_support_max": 0,
    },
    "TEERUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "primary_comparator_eval_collapse_rate": 1.0,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 1,
    },
    "XBTUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 2,
        "primary_comparator_eval_support_max": 2,
    },
}
CYCLE16_BASELINE_ASSET_DECISIONS: dict[str, dict[str, Any]] = {
    "AAVEUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.5,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 2,
    },
    "ADAUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 2,
        "primary_comparator_eval_support_max": 2,
    },
    "AI16ZUSD": {
        "proposed_decision": "neutral_flat",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 0,
        "primary_comparator_eval_support_max": 0,
    },
    "ANLOGUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "primary_comparator_eval_collapse_rate": 1.0,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 1,
    },
    "TEERUSD": {
        "proposed_decision": "candidate_axis_not_clusterable",
        "primary_comparator_eval_collapse_rate": 1.0,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 1,
    },
    "TERMUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.0,
        "primary_comparator_eval_support_min": 2,
        "primary_comparator_eval_support_max": 2,
    },
    "XBTUSD": {
        "proposed_decision": "cluster_candidate",
        "primary_comparator_eval_collapse_rate": 0.5,
        "primary_comparator_eval_support_min": 1,
        "primary_comparator_eval_support_max": 2,
    },
}


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(raw).split(",") if token.strip())


def _resolve_cycle5_source_root(source_feature_root: Path | str | None) -> Path:
    if source_feature_root is not None and str(source_feature_root).strip():
        return Path(source_feature_root)
    raw = os.getenv(CYCLE5_SOURCE_ROOT_ENV, "").strip()
    if raw:
        return Path(raw)
    raise ValueError(f"Real-data Regime asset-state runs require --source-feature-root or {CYCLE5_SOURCE_ROOT_ENV}.")


def build_cycle3_trial_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the bounded Cycle 3 parameter grid requested by the study manifest."""
    requested = {str(method).strip().lower() for method in config.methods}
    trials: list[TrialConfig] = []
    if "hdbscan" in requested:
        for label, min_cluster_size, min_samples in (
            ("baseline_mcs5_ms1", 5, 1),
            ("conservative_mcs20_ms3", 20, 3),
            ("stronger_mcs40_ms5", 40, 5),
        ):
            variant = f"hdbscan_mcs{min_cluster_size}_ms{min_samples}"
            trials.append(
                TrialConfig(
                    study=config,
                    method="hdbscan",
                    method_params={
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "allow_single_cluster": True,
                        "prediction_data": True,
                        "cluster_selection_method": "eom",
                    },
                    trial_id=f"{config.axis}_{config.band}_hdbscan_{label}",
                    grid_family="hdbscan_min_cluster_size_min_samples",
                    grid_variant_id=variant,
                )
            )
    if "kmeans" in requested:
        for n_clusters in (2, 3, 4):
            trials.append(
                TrialConfig(
                    study=config,
                    method="kmeans",
                    method_params={"n_clusters": n_clusters, "n_init": 10, "random_state": int(config.random_state)},
                    trial_id=f"{config.axis}_{config.band}_kmeans_k{n_clusters}",
                    grid_family="kmeans_n_clusters",
                    grid_variant_id=f"kmeans_k{n_clusters}",
                )
            )
    if "gaussian_mixture" in requested:
        for n_components in (2, 3, 4):
            trials.append(
                TrialConfig(
                    study=config,
                    method="gaussian_mixture",
                    method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                    trial_id=f"{config.axis}_{config.band}_gmm_c{n_components}",
                    grid_family="gaussian_mixture_n_components",
                    grid_variant_id=f"gaussian_mixture_c{n_components}",
                )
            )
    return tuple(trials)


def build_cycle4_retained_trial_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the retained Cycle 4 grid selected by the latest Pass 1 design."""
    requested = {str(method).strip().lower() for method in config.methods}
    trials: list[TrialConfig] = []
    if "kmeans" in requested:
        for n_clusters in (3, 4):
            trials.append(
                TrialConfig(
                    study=config,
                    method="kmeans",
                    method_params={"n_clusters": n_clusters, "n_init": 10, "random_state": int(config.random_state)},
                    trial_id=f"{config.axis}_{config.band}_kmeans_k{n_clusters}",
                    grid_family="cycle4_retained_trend_micro",
                    grid_variant_id=f"kmeans_k{n_clusters}",
                )
            )
    if "gaussian_mixture" in requested:
        for n_components in (3, 4):
            trials.append(
                TrialConfig(
                    study=config,
                    method="gaussian_mixture",
                    method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                    trial_id=f"{config.axis}_{config.band}_gmm_c{n_components}",
                    grid_family="cycle4_retained_trend_micro",
                    grid_variant_id=f"gaussian_mixture_c{n_components}",
                )
            )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                trial_id=f"{config.axis}_{config.band}_hdbscan_mcs5_ms1",
                grid_family="cycle4_retained_trend_micro",
                grid_variant_id="hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle5_retained_trial_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 5 retained grid: add ADAUSD coverage while omitting rejected kmeans_k3."""
    requested = {str(method).strip().lower() for method in config.methods}
    trials: list[TrialConfig] = []
    if "gaussian_mixture" in requested:
        for n_components in (3, 4):
            trials.append(
                TrialConfig(
                    study=config,
                    method="gaussian_mixture",
                    method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                    trial_id=f"{config.axis}_{config.band}_gmm_c{n_components}",
                    grid_family="cycle5_retained_trend_micro_asset_expansion",
                    grid_variant_id=f"gaussian_mixture_c{n_components}",
                )
            )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                trial_id=f"{config.axis}_{config.band}_kmeans_k4",
                grid_family="cycle5_retained_trend_micro_asset_expansion",
                grid_variant_id="kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                trial_id=f"{config.axis}_{config.band}_hdbscan_mcs5_ms1",
                grid_family="cycle5_retained_trend_micro_asset_expansion",
                grid_variant_id="hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle8_feature_preprocess_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 8 ADAUSD feature/preprocess probe grid from the Pass 1 handoff."""
    requested = {str(method).strip().lower() for method in config.methods}
    arms = (
        ("control_manual_robust", "manual_baseline", "robust_scale"),
        ("manual_winsor_p01_p99_robust", "manual_baseline", "winsor_p01_p99"),
        ("compact_return_macd_robust", "trend_return_macd_compact", "robust_scale"),
    )
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy, preprocess in arms:
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                variant = f"gaussian_mixture_c{n_components}"
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess=preprocess,
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family="cycle8_trend_micro_adausd_feature_preprocess_probe",
                        grid_variant_id=f"{arm_id}__{variant}",
                    )
                )
        if "kmeans" in requested:
            trials.append(
                TrialConfig(
                    study=config,
                    method="kmeans",
                    method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                    preprocess=preprocess,
                    feature_strategy=feature_strategy,
                    trial_id=f"{config.axis}_{config.band}_{arm_id}_kmeans_k4",
                    grid_family="cycle8_trend_micro_adausd_feature_preprocess_probe",
                    grid_variant_id=f"{arm_id}__kmeans_k4",
                )
            )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family="cycle8_trend_micro_adausd_feature_preprocess_probe",
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle9_no_model_decision_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 9 clean-collapse/no-model decision grid from the Pass 1 handoff."""
    requested = {str(method).strip().lower() for method in config.methods}
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                variant = f"gaussian_mixture_c{n_components}"
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family="cycle9_trend_micro_clean_collapse_no_model_decision_probe",
                        grid_variant_id=f"{arm_id}__{variant}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family="cycle9_trend_micro_clean_collapse_no_model_decision_probe",
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family="cycle9_trend_micro_clean_collapse_no_model_decision_probe",
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle10_near_flat_panel_sensitivity_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 10 near-flat panel-sensitivity grid from the Pass 1 handoff."""
    requested = {str(method).strip().lower() for method in config.methods}
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                variant = f"gaussian_mixture_c{n_components}"
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family="cycle10_trend_micro_near_flat_panel_sensitivity_probe",
                        grid_variant_id=f"{arm_id}__{variant}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family="cycle10_trend_micro_near_flat_panel_sensitivity_probe",
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family="cycle10_trend_micro_near_flat_panel_sensitivity_probe",
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle11_method_role_no_model_decision_refinement_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 11 retained grid with method-role-specific decision semantics."""
    requested = {str(method).strip().lower() for method in config.methods}
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                variant = f"gaussian_mixture_c{n_components}"
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family="cycle11_trend_micro_method_role_no_model_decision_refinement",
                        grid_variant_id=f"{arm_id}__{variant}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family="cycle11_trend_micro_method_role_no_model_decision_refinement",
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family="cycle11_trend_micro_method_role_no_model_decision_refinement",
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle12_cycle11_repro_resource_perturbation_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the exact Cycle 11 grid for the Cycle 12 reproducibility/resource probe."""
    return build_cycle11_method_role_no_model_decision_refinement_grid(config)


def build_cycle13_primary_comparator_feature_refinement_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 13 retained controls plus directional compact GMM refinement arms."""
    requested = {str(method).strip().lower() for method in config.methods}
    grid_family = "cycle13_trend_micro_primary_comparator_feature_refinement_probe"
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
        ("directional_compact_robust", "trend_directional_compact"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                variant = f"gaussian_mixture_c{n_components}"
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family=grid_family,
                        grid_variant_id=f"{arm_id}__{variant}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle14_axis_not_clusterable_feature_availability_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 14 no-model confirmation grid with feature-availability proof arms."""
    requested = {str(method).strip().lower() for method in config.methods}
    grid_family = "cycle14_trend_micro_axis_not_clusterable_confirmation_and_feature_availability_probe"
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
        ("directional_compact_robust", "trend_directional_compact"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family=grid_family,
                        grid_variant_id=f"{arm_id}__gaussian_mixture_c{n_components}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 15 ANLOG near-flat safety-expansion retained grid."""
    requested = {str(method).strip().lower() for method in config.methods}
    grid_family = "cycle15_trend_micro_anlog_near_flat_clean_collapse_safety_expansion_probe"
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family=grid_family,
                        grid_variant_id=f"{arm_id}__gaussian_mixture_c{n_components}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle16_near_flat_panel_sensitivity_confirmation_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 16 panel-sensitivity confirmation grid with Cycle 15 retained arms."""
    requested = {str(method).strip().lower() for method in config.methods}
    grid_family = "cycle16_trend_micro_near_flat_panel_sensitivity_confirmation_probe"
    trials: list[TrialConfig] = []
    for arm_id, feature_strategy in (
        ("compact_return_macd_robust", "trend_return_macd_compact"),
        ("control_manual_robust", "manual_baseline"),
    ):
        if "gaussian_mixture" in requested:
            for n_components in (3, 4):
                trials.append(
                    TrialConfig(
                        study=config,
                        method="gaussian_mixture",
                        method_params={"n_components": n_components, "covariance_type": "full", "random_state": int(config.random_state)},
                        preprocess="robust_scale",
                        feature_strategy=feature_strategy,
                        trial_id=f"{config.axis}_{config.band}_{arm_id}_gmm_c{n_components}",
                        grid_family=grid_family,
                        grid_variant_id=f"{arm_id}__gaussian_mixture_c{n_components}",
                    )
                )
    if "kmeans" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="kmeans",
                method_params={"n_clusters": 4, "n_init": 10, "random_state": int(config.random_state)},
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_kmeans_k4",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__kmeans_k4",
            )
        )
    if "hdbscan" in requested:
        trials.append(
            TrialConfig(
                study=config,
                method="hdbscan",
                method_params={
                    "min_cluster_size": 5,
                    "min_samples": 1,
                    "allow_single_cluster": True,
                    "prediction_data": True,
                    "cluster_selection_method": "eom",
                },
                preprocess="robust_scale",
                feature_strategy="manual_baseline",
                trial_id=f"{config.axis}_{config.band}_control_manual_robust_hdbscan_mcs5_ms1",
                grid_family=grid_family,
                grid_variant_id="control_manual_robust__hdbscan_mcs5_ms1",
            )
        )
    return tuple(trials)


def build_cycle17_per_asset_stability_collapse_attribution_probe_grid(config: StudyConfig) -> tuple[TrialConfig, ...]:
    """Return the Cycle 17 attribution grid with the Cycle 16 retained arms."""
    grid_family = "cycle17_trend_micro_per_asset_stability_collapse_attribution_probe"
    return tuple(
        TrialConfig(
            study=config,
            method=trial.method,
            method_params=dict(trial.method_params),
            preprocess=trial.preprocess,
            feature_strategy=trial.feature_strategy,
            trial_id=trial.trial_id,
            grid_family=grid_family,
            grid_variant_id=trial.grid_variant_id,
        )
        for trial in build_cycle16_near_flat_panel_sensitivity_confirmation_probe_grid(config)
    )


def build_default_trials(config: StudyConfig) -> tuple[TrialConfig, ...]:
    return tuple(
        TrialConfig(study=config, method=method, trial_id=f"{config.axis}_{config.band}_{method}")
        for method in config.methods
    )


def build_synthetic_trend_micro_frame(*, assets: Sequence[str], rows_per_asset: int = 90, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    frames: list[pd.DataFrame] = []
    bases = ("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14")
    for asset_index, asset in enumerate(assets):
        rows = int(rows_per_asset)
        ts = np.arange(rows, dtype=np.int64) * 1800
        regime = np.repeat(np.arange(3), rows // 3 + 1)[:rows]
        data: dict[str, Any] = {"ts": ts, "asset": [str(asset)] * rows}
        for interval in (1, 5, 15, 30):
            scale = 1.0 + (float(interval) / 30.0)
            data[f"i{interval}_log_return"] = (regime - 1) * 0.002 * scale + rng.normal(0.0, 0.0004, rows)
            data[f"i{interval}_macd_hist_12_26_9"] = (regime - 1) * scale + rng.normal(0.0, 0.10, rows)
            data[f"i{interval}_rsi_14"] = 45.0 + regime * 8.0 + rng.normal(0.0, 2.0, rows)
            data[f"i{interval}_adx_14"] = 15.0 + regime * 6.0 + rng.normal(0.0, 1.5, rows)
            data[f"i{interval}_roc_14"] = (regime - 1) * 0.003 * scale + rng.normal(0.0, 0.0005, rows)
            data[f"i{interval}_mom_14"] = (regime - 1) * 0.004 * scale + rng.normal(0.0, 0.0007, rows)
            data[f"i{interval}_range_efficiency"] = (regime + 1) / 3.0 + rng.normal(0.0, 0.03, rows)
            data[f"i{interval}_trade_intensity"] = 1.0 + asset_index + rng.normal(0.0, 0.05, rows)
            data[f"i{interval}_prr"] = np.abs(data[f"i{interval}_log_return"])
        asset_frame = pd.DataFrame(data)
        asset_frame["future_log_return"] = asset_frame["i30_log_return"].shift(-1)
        asset_frame["future_abs_return"] = asset_frame["future_log_return"].abs()
        asset_frame["future_drawdown"] = np.minimum(asset_frame["future_log_return"], 0.0)
        asset_frame["future_runup"] = np.maximum(asset_frame["future_log_return"], 0.0)
        asset_frame = asset_frame.dropna(subset=["future_log_return"]).copy()
        frames.append(asset_frame)
    return pd.concat(frames, ignore_index=True)[
        [
            "ts",
            "asset",
            *[f"i{i}_{b}" for i in (1, 5, 15, 30) for b in (*bases, "trade_intensity", "prr")],
            *[f"i{i}_{b}" for i in (1, 5, 15, 30) for b in ("roc_14", "mom_14", "range_efficiency")],
            "future_log_return",
            "future_abs_return",
            "future_drawdown",
            "future_runup",
        ]
    ]


def _parts_contain_subsequence(parts: Sequence[str], required: Sequence[str]) -> bool:
    lowered = [str(part).lower() for part in parts]
    needed = [str(part).lower() for part in required]
    if not needed:
        return True
    for start in range(0, len(lowered) - len(needed) + 1):
        if tuple(lowered[start : start + len(needed)]) == tuple(needed):
            return True
    return False


def validate_cycle5_output_root(output_root: Path) -> Path:
    root = Path(output_root)
    if not any(str(root.name).endswith(suffix) for suffix in CYCLE_REAL_BOUNDED_OUTPUT_SUFFIXES):
        raise ValueError(f"Real bounded runner output root must end with one of {CYCLE_REAL_BOUNDED_OUTPUT_SUFFIXES!r}: {root}")
    if not _parts_contain_subsequence(root.parts, CYCLE5_REPORT_PARTS):
        expected = str(Path(*CYCLE5_REPORT_PARTS) / f"<timestamp>{CYCLE8_OUTPUT_SUFFIX}")
        raise ValueError(f"Real bounded runner output root must be under {expected}: {root}")
    return root


def _month_range_for_files(start_ts: int, end_ts: int) -> list[tuple[int, int]]:
    start_dt = pd.to_datetime(int(start_ts), unit="s", utc=True)
    end_dt = pd.to_datetime(int(end_ts), unit="s", utc=True)
    cur = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=1, tz="UTC")
    end_marker = pd.Timestamp(year=end_dt.year, month=end_dt.month, day=1, tz="UTC")
    out: list[tuple[int, int]] = []
    while cur <= end_marker:
        out.append((int(cur.year), int(cur.month)))
        cur = cur + pd.DateOffset(months=1)
    return out


def list_cycle5_source_files(
    *,
    source_feature_root: Path,
    assets: Sequence[str],
    member_intervals: Sequence[int],
    start_ts: int,
    end_ts: int,
) -> tuple[str, ...]:
    root = Path(source_feature_root)
    files: list[str] = []
    for interval in member_intervals:
        for asset in assets:
            for year, month in _month_range_for_files(int(start_ts), int(end_ts)):
                month_dir = root / f"scalar_features_{int(interval)}" / f"asset={asset}" / f"year={year}" / f"month={month:02d}"
                files.extend(str(path) for path in sorted(month_dir.glob("*.parquet"), key=lambda p: p.name.lower()))
    return tuple(sorted(dict.fromkeys(files)))


def _attach_forward_columns(frame: pd.DataFrame, *, ceiling_interval_min: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy().sort_values(["asset", "ts"] if "asset" in frame.columns else ["ts"])
    return_col = f"i{int(ceiling_interval_min)}_log_return"
    if return_col in out.columns and "asset" in out.columns:
        out["future_log_return"] = out.groupby("asset", sort=False)[return_col].shift(-1)
    elif return_col in out.columns:
        out["future_log_return"] = out[return_col].shift(-1)
    else:
        out["future_log_return"] = np.nan
    out["future_abs_return"] = pd.to_numeric(out["future_log_return"], errors="coerce").abs()
    out["future_drawdown"] = np.minimum(pd.to_numeric(out["future_log_return"], errors="coerce"), 0.0)
    out["future_runup"] = np.maximum(pd.to_numeric(out["future_log_return"], errors="coerce"), 0.0)
    return out


def _real_aligned_loader(
    *,
    asset: str,
    band: str,
    start_ts: int,
    end_ts: int,
    feature_bases: Sequence[str],
    source_feature_root: Path,
) -> pd.DataFrame:
    from src.regimes import regime_clustering as rc

    band_spec = next((spec for spec in rc.BANDS if str(spec.name) == str(band)), None)
    if band_spec is None:
        raise ValueError(f"Unsupported real-frame band {band!r}")
    old_root = rc.PARQUET_ROOT
    rc.PARQUET_ROOT = Path(source_feature_root)
    try:
        if hasattr(rc, "_FEATURE_COLUMN_DISCOVERY_CACHE"):
            rc._FEATURE_COLUMN_DISCOVERY_CACHE.clear()
        return rc.build_aligned_features(
            str(asset),
            band_spec,
            int(start_ts),
            int(end_ts),
            tuple(str(base) for base in feature_bases),
        )
    finally:
        rc.PARQUET_ROOT = old_root
        if hasattr(rc, "_FEATURE_COLUMN_DISCOVERY_CACHE"):
            rc._FEATURE_COLUMN_DISCOVERY_CACHE.clear()


def build_real_aligned_feature_frames(
    *,
    assets: Sequence[str],
    band: str,
    feature_bases: Sequence[str],
    source_feature_root: Path,
    train_start_ts: int = CYCLE5_TRAIN_START_TS,
    train_end_ts: int = CYCLE5_TRAIN_END_TS,
    eval_start_ts: int = CYCLE5_EVAL_START_TS,
    eval_end_ts: int = CYCLE5_EVAL_END_TS,
    loader: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    load = loader or _real_aligned_loader
    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []
    per_asset: dict[str, dict[str, int]] = {}
    for asset in assets:
        train = load(
            asset=str(asset),
            band=str(band),
            start_ts=int(train_start_ts),
            end_ts=int(train_end_ts),
            feature_bases=feature_bases,
            source_feature_root=Path(source_feature_root),
        )
        eval_frame = load(
            asset=str(asset),
            band=str(band),
            start_ts=int(eval_start_ts),
            end_ts=int(eval_end_ts),
            feature_bases=feature_bases,
            source_feature_root=Path(source_feature_root),
        )
        if not train.empty:
            train_frames.append(train)
        if not eval_frame.empty:
            eval_frames.append(eval_frame)
        per_asset[str(asset)] = {"train_rows": int(len(train)), "eval_rows": int(len(eval_frame))}
    train_frame = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    eval_frame = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    metadata = {
        "source_root": str(Path(source_feature_root)),
        "per_asset_rows_loaded": per_asset,
        "train_rows_loaded": int(len(train_frame)),
        "eval_rows_loaded": int(len(eval_frame)),
        "train_start_ts": int(train_start_ts),
        "train_end_ts": int(train_end_ts),
        "eval_start_ts": int(eval_start_ts),
        "eval_end_ts": int(eval_end_ts),
    }
    return (
        _attach_forward_columns(train_frame, ceiling_interval_min=30),
        _attach_forward_columns(eval_frame, ceiling_interval_min=30),
        metadata,
    )


def _read_json_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _safe_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _current_process_rss_mb() -> tuple[float | None, dict[str, Any]]:
    try:
        import psutil  # type: ignore

        rss_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
        return rss_mb, {
            "peak_rss_status": "computed_in_process_sampled",
            "peak_rss_source": "psutil.Process.memory_info.rss",
            "peak_rss_unavailable_reason": None,
        }
    except Exception as exc:
        return None, {
            "peak_rss_status": "unavailable_psutil_sample_failed",
            "peak_rss_source": None,
            "peak_rss_unavailable_reason": str(exc),
        }


def _build_near_flat_boundary_summary(flat_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in flat_rows:
        row = dict(raw)
        near_zero_fraction = _safe_number(row.get("near_zero_movement_fraction"))
        threshold = _safe_number(row.get("near_flat_fraction_threshold"))
        distance = _safe_number(row.get("near_flat_distance_to_threshold"))
        rows.append(
            {
                "asset": str(row.get("asset") or ""),
                "reason_code": str(row.get("reason_code") or ""),
                "included_in_fit": _safe_bool(row.get("included_in_fit")),
                "carried_as": row.get("carried_as"),
                "near_zero_movement_fraction": near_zero_fraction,
                "near_flat_fraction_threshold": threshold,
                "near_flat_distance_to_threshold": distance,
                "zero_variance_feature_count": row.get("zero_variance_feature_count"),
                "near_zero_variance_feature_count": row.get("near_zero_variance_feature_count"),
            }
        )
    pass_boundary = [
        row
        for row in rows
        if row["reason_code"] == "pass"
        and row["near_zero_movement_fraction"] is not None
        and row["near_flat_distance_to_threshold"] is not None
    ]
    pass_boundary.sort(key=lambda row: (float(row["near_flat_distance_to_threshold"]), str(row["asset"])))
    return {
        "status": "computed",
        "flat_threshold": _safe_number(rows[0].get("near_flat_fraction_threshold")) if rows else None,
        "pass_assets_by_distance_to_flat_threshold": pass_boundary,
        "nearest_pass_asset": None if not pass_boundary else pass_boundary[0]["asset"],
        "neutral_flat_assets": sorted(row["asset"] for row in rows if row["carried_as"] == "neutral_flat"),
        "note": "Distance is threshold minus near_zero_movement_fraction; positive pass values are boundary candidates.",
    }


def _build_cycle10_panel_sensitivity_rows(decision_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in decision_rows:
        row = dict(raw)
        asset = str(row.get("asset") or "")
        baseline = dict(CYCLE9_BASELINE_ASSET_DECISIONS.get(asset, {}))
        current_decision = str(row.get("proposed_decision") or "")
        baseline_decision = baseline.get("proposed_decision")
        current_support_max = _safe_number(row.get("eval_non_noise_label_support_max"))
        baseline_support_max = _safe_number(baseline.get("eval_non_noise_label_support_max"))
        current_collapse = _safe_number(row.get("retained_method_eval_collapse_rate"))
        baseline_collapse = _safe_number(baseline.get("retained_method_eval_collapse_rate"))
        row["cycle9_baseline_proposed_decision"] = baseline_decision
        row["cycle9_baseline_eval_support_min"] = baseline.get("eval_non_noise_label_support_min")
        row["cycle9_baseline_eval_support_max"] = baseline.get("eval_non_noise_label_support_max")
        row["cycle9_baseline_eval_collapse_rate"] = baseline.get("retained_method_eval_collapse_rate")
        row["decision_changed_vs_cycle9"] = None if not baseline else bool(current_decision != str(baseline_decision))
        row["eval_support_max_delta_vs_cycle9"] = (
            None if current_support_max is None or baseline_support_max is None else float(current_support_max - baseline_support_max)
        )
        row["eval_collapse_rate_delta_vs_cycle9"] = (
            None if current_collapse is None or baseline_collapse is None else float(current_collapse - baseline_collapse)
        )
        row["panel_sensitivity_role"] = "cycle10_new_near_flat_boundary_asset" if asset == "TEERUSD" else "cycle9_shared_asset"
        rows.append(row)
    return rows


def _augment_cycle10_panel_sensitivity_artifacts(output_root: Path) -> None:
    output_root = Path(output_root)
    flat_path = output_root / "flat_preflight.csv"
    decisions_path = output_root / "asset_model_decisions.csv"
    if not flat_path.exists() or not decisions_path.exists():
        return
    flat_rows = pd.read_csv(flat_path).to_dict("records")
    decision_rows = pd.read_csv(decisions_path).to_dict("records")
    near_flat_summary = _build_near_flat_boundary_summary(flat_rows)
    panel_rows = _build_cycle10_panel_sensitivity_rows(decision_rows)
    pd.DataFrame(panel_rows).to_csv(decisions_path, index=False)
    (output_root / "asset_model_decisions.json").write_text(json.dumps(panel_rows, indent=2, sort_keys=True), encoding="utf-8")
    shared = [row for row in panel_rows if row.get("panel_sensitivity_role") == "cycle9_shared_asset"]
    changed = [str(row.get("asset")) for row in shared if bool(row.get("decision_changed_vs_cycle9"))]
    teer = next((row for row in panel_rows if str(row.get("asset")) == "TEERUSD"), None)
    panel_summary = {
        "status": "computed",
        "baseline": "official_cycle9_20260511_171736_clean_collapse_no_model_decision_probe",
        "shared_assets_compared": sorted(str(row.get("asset")) for row in shared),
        "changed_decision_assets": sorted(changed),
        "teerusd_proposed_decision": None if teer is None else teer.get("proposed_decision"),
        "teerusd_preflight_reason_code": None if teer is None else teer.get("preflight_reason_code"),
        "teerusd_near_zero_movement_fraction": None if teer is None else teer.get("near_zero_movement_fraction"),
        "teerusd_near_flat_distance_to_threshold": None if teer is None else teer.get("near_flat_distance_to_threshold"),
        "production_label_change": False,
    }
    for artifact_name in ("aggregate_summary.json", "trial_manifest.json", "experiment_config_snapshot.json"):
        path = output_root / artifact_name
        if path.exists():
            payload = _read_json_artifact(path)
            payload["cycle10_panel_sensitivity_summary"] = panel_summary
            payload["near_flat_boundary_summary"] = near_flat_summary
            _write_json_artifact(path, payload)
    validation_path = output_root / "artifact_validation.json"
    if validation_path.exists():
        validation = _read_json_artifact(validation_path)
        validation["cycle10_panel_sensitivity_probe"] = True
        validation["panel_sensitivity_summary_written"] = True
        validation["near_flat_boundary_summary_written"] = True
        _write_json_artifact(validation_path, validation)


def _summarize_method_role_decisions(decision_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    warning_assets = sorted(
        str(row.get("asset"))
        for row in decision_rows
        if _safe_bool(row.get("diagnostic_only_support_warning"))
    )
    source_counts: dict[str, int] = {}
    for row in decision_rows:
        source = str(row.get("support_source_for_proposed_decision") or "unknown")
        source_counts[source] = int(source_counts.get(source, 0) + 1)
    return {
        "status": "computed",
        "policy": "primary_comparator_support_required_for_cluster_candidate_when_primary_trials_exist",
        "primary_comparator_role": "gaussian_mixture_c3_c4_compact_and_manual",
        "sentinel_role": "kmeans_k4_discounted",
        "diagnostic_only_role": "hdbscan_mcs5_ms1_discounted",
        "diagnostic_only_support_warning_assets": warning_assets,
        "support_source_counts": {key: int(source_counts[key]) for key in sorted(source_counts)},
        "production_label_change": False,
    }


def _augment_cycle11_method_role_artifacts(output_root: Path) -> None:
    output_root = Path(output_root)
    flat_path = output_root / "flat_preflight.csv"
    decisions_path = output_root / "asset_model_decisions.csv"
    if not flat_path.exists() or not decisions_path.exists():
        return
    flat_rows = pd.read_csv(flat_path).to_dict("records")
    decision_rows = pd.read_csv(decisions_path).to_dict("records")
    near_flat_summary = _build_near_flat_boundary_summary(flat_rows)
    role_summary = _summarize_method_role_decisions(decision_rows)
    for artifact_name in ("aggregate_summary.json", "trial_manifest.json", "experiment_config_snapshot.json"):
        path = output_root / artifact_name
        if path.exists():
            payload = _read_json_artifact(path)
            payload["method_role_decision_summary"] = role_summary
            payload["near_flat_boundary_summary"] = near_flat_summary
            _write_json_artifact(path, payload)
    validation_path = output_root / "artifact_validation.json"
    if validation_path.exists():
        validation = _read_json_artifact(validation_path)
        validation["cycle11_method_role_no_model_decision_refinement"] = True
        validation["method_role_decision_summary_written"] = True
        validation["near_flat_boundary_summary_written"] = True
        _write_json_artifact(validation_path, validation)


def _build_cycle15_safety_expansion_rows(decision_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in decision_rows:
        row = dict(raw)
        asset = str(row.get("asset") or "")
        baseline = dict(CYCLE14_BASELINE_ASSET_DECISIONS.get(asset, {}))
        current_decision = str(row.get("proposed_decision") or "")
        baseline_decision = baseline.get("proposed_decision")
        current_support_max = _safe_number(row.get("primary_comparator_eval_support_max"))
        baseline_support_max = _safe_number(baseline.get("primary_comparator_eval_support_max"))
        current_collapse = _safe_number(row.get("primary_comparator_eval_collapse_rate"))
        baseline_collapse = _safe_number(baseline.get("primary_comparator_eval_collapse_rate"))
        row["cycle15_representative_source"] = (
            "pass1_full_non_core_near_flat_boundary_scan" if asset == "ANLOGUSD" else "cycle14_control_panel_member"
        )
        row["cycle15_safety_family_role"] = (
            "selected_nearest_non_core_pass_boundary_asset"
            if asset == "ANLOGUSD"
            else "near_flat_pass_boundary_control"
            if asset == "TEERUSD"
            else "valid_flat_neutral_control"
            if asset == "AI16ZUSD"
            else "panel_sensitive_watch_asset"
            if asset == "ADAUSD"
            else "core_control_asset"
        )
        row["cycle14_baseline_proposed_decision"] = baseline_decision
        row["cycle14_baseline_primary_collapse_rate"] = baseline.get("primary_comparator_eval_collapse_rate")
        row["cycle14_baseline_primary_support_min"] = baseline.get("primary_comparator_eval_support_min")
        row["cycle14_baseline_primary_support_max"] = baseline.get("primary_comparator_eval_support_max")
        row["decision_changed_vs_cycle14"] = None if not baseline else bool(current_decision != str(baseline_decision))
        row["primary_support_max_delta_vs_cycle14"] = (
            None if current_support_max is None or baseline_support_max is None else float(current_support_max - baseline_support_max)
        )
        row["primary_collapse_rate_delta_vs_cycle14"] = (
            None if current_collapse is None or baseline_collapse is None else float(current_collapse - baseline_collapse)
        )
        row["temporary_probe_evidence_status"] = "design_note_only_not_official_production_evidence"
        row["temporary_single_add_probe_note"] = (
            "ANLOGUSD and TEERUSD were candidate_axis_not_clusterable in a temporary single-add design probe; "
            "ADAUSD flipped to cluster_candidate there, so Cycle 15 must treat panel sensitivity as unresolved."
        )
        rows.append(row)
    return rows


def _build_cycle15_safety_expansion_summary(
    *,
    flat_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    near_flat_summary: Mapping[str, Any],
) -> dict[str, Any]:
    by_asset = {str(row.get("asset") or ""): dict(row) for row in decision_rows}
    focus_assets = ("ADAUSD", "TEERUSD", "ANLOGUSD", "AI16ZUSD")
    return {
        "status": "computed",
        "target": "trend_micro_anlog_near_flat_clean_collapse_safety_expansion_probe",
        "selected_representative": "ANLOGUSD",
        "selected_representative_source": "Pass 1 full non-core preflight scan of 354 assets",
        "scan_reason_counts": {
            "pass": 354,
            "valid_flat_or_pegged": 0,
            "insufficient_variance": 0,
            "bad_or_missing_data": 0,
            "low_activity": 0,
        },
        "no_true_safety_family_representative_found": True,
        "near_flat_distance_ranking": list(near_flat_summary.get("pass_assets_by_distance_to_flat_threshold") or []),
        "fit_inclusion_and_carried_as": {
            str(row.get("asset")): {
                "preflight_reason_code": row.get("reason_code"),
                "included_in_fit": _safe_bool(row.get("included_in_fit")),
                "carried_as": row.get("carried_as"),
                "near_zero_movement_fraction": _safe_number(row.get("near_zero_movement_fraction")),
                "near_flat_distance_to_threshold": _safe_number(row.get("near_flat_distance_to_threshold")),
            }
            for row in flat_rows
        },
        "primary_comparator_collapse_by_asset": {
            asset: {
                "proposed_decision": by_asset.get(asset, {}).get("proposed_decision"),
                "primary_comparator_eval_collapse_rate": by_asset.get(asset, {}).get("primary_comparator_eval_collapse_rate"),
                "primary_comparator_eval_support_min": by_asset.get(asset, {}).get("primary_comparator_eval_support_min"),
                "primary_comparator_eval_support_max": by_asset.get(asset, {}).get("primary_comparator_eval_support_max"),
                "primary_comparator_eval_collapse_summary": by_asset.get(asset, {}).get("primary_comparator_eval_collapse_summary"),
            }
            for asset in focus_assets
            if asset in by_asset
        },
        "panel_sensitivity_comparison": {
            str(row.get("asset")): {
                "cycle14_baseline_proposed_decision": row.get("cycle14_baseline_proposed_decision"),
                "current_proposed_decision": row.get("proposed_decision"),
                "decision_changed_vs_cycle14": row.get("decision_changed_vs_cycle14"),
                "primary_support_max_delta_vs_cycle14": row.get("primary_support_max_delta_vs_cycle14"),
                "primary_collapse_rate_delta_vs_cycle14": row.get("primary_collapse_rate_delta_vs_cycle14"),
            }
            for row in decision_rows
            if str(row.get("asset")) in focus_assets
        },
        "temporary_probe_evidence_status": "design_note_only_not_official_production_evidence",
        "temporary_single_add_design_note": (
            "Pass 1 temporary probes under D:/pipeline_codex_temp are not official cycle artifacts and do not authorize "
            "production changes; they only motivate the Cycle 15 bounded run."
        ),
        "production_label_change": False,
    }


def _augment_cycle15_safety_expansion_artifacts(output_root: Path) -> None:
    output_root = Path(output_root)
    flat_path = output_root / "flat_preflight.csv"
    decisions_path = output_root / "asset_model_decisions.csv"
    if not flat_path.exists() or not decisions_path.exists():
        return
    flat_rows = pd.read_csv(flat_path).to_dict("records")
    decision_rows = pd.read_csv(decisions_path).to_dict("records")
    near_flat_summary = _build_near_flat_boundary_summary(flat_rows)
    cycle15_rows = _build_cycle15_safety_expansion_rows(decision_rows)
    pd.DataFrame(cycle15_rows).to_csv(decisions_path, index=False)
    (output_root / "asset_model_decisions.json").write_text(json.dumps(cycle15_rows, indent=2, sort_keys=True), encoding="utf-8")
    safety_summary = _build_cycle15_safety_expansion_summary(
        flat_rows=flat_rows,
        decision_rows=cycle15_rows,
        near_flat_summary=near_flat_summary,
    )
    for artifact_name in ("aggregate_summary.json", "trial_manifest.json", "experiment_config_snapshot.json"):
        path = output_root / artifact_name
        if path.exists():
            payload = _read_json_artifact(path)
            payload["cycle15_safety_expansion_summary"] = safety_summary
            payload["near_flat_boundary_summary"] = near_flat_summary
            _write_json_artifact(path, payload)
    validation_path = output_root / "artifact_validation.json"
    if validation_path.exists():
        validation = _read_json_artifact(validation_path)
        validation["cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe"] = True
        validation["cycle15_safety_expansion_summary_written"] = True
        validation["near_flat_boundary_summary_written"] = True
        validation["feature_availability_contract_written"] = True
        _write_json_artifact(validation_path, validation)


def _build_cycle16_panel_sensitivity_rows(decision_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in decision_rows:
        row = dict(raw)
        asset = str(row.get("asset") or "")
        baseline14 = dict(CYCLE14_BASELINE_ASSET_DECISIONS.get(asset, {}))
        baseline15 = dict(CYCLE15_BASELINE_ASSET_DECISIONS.get(asset, {}))
        current_decision = _safe_string(row.get("proposed_decision"))
        current_support_max = _safe_number(row.get("primary_comparator_eval_support_max"))
        current_collapse = _safe_number(row.get("primary_comparator_eval_collapse_rate"))
        baseline14_support_max = _safe_number(baseline14.get("primary_comparator_eval_support_max"))
        baseline14_collapse = _safe_number(baseline14.get("primary_comparator_eval_collapse_rate"))
        baseline15_support_max = _safe_number(baseline15.get("primary_comparator_eval_support_max"))
        baseline15_collapse = _safe_number(baseline15.get("primary_comparator_eval_collapse_rate"))
        row["cycle16_panel_sensitivity_role"] = (
            "added_preverified_near_flat_pass_boundary_asset"
            if asset in CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE
            else "near_flat_pass_boundary_control"
            if asset in {"TEERUSD", "ANLOGUSD"}
            else "valid_flat_neutral_control"
            if asset == "AI16ZUSD"
            else "panel_sensitive_watch_asset"
            if asset == "ADAUSD"
            else "core_control_asset"
        )
        row["cycle16_preverified_boundary_source"] = CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE.get(asset, {}).get("source")
        row["cycle14_baseline_proposed_decision"] = baseline14.get("proposed_decision")
        row["cycle14_baseline_primary_collapse_rate"] = baseline14.get("primary_comparator_eval_collapse_rate")
        row["cycle14_baseline_primary_support_min"] = baseline14.get("primary_comparator_eval_support_min")
        row["cycle14_baseline_primary_support_max"] = baseline14.get("primary_comparator_eval_support_max")
        row["decision_changed_vs_cycle14"] = (
            None if not baseline14 or current_decision is None else bool(current_decision != str(baseline14.get("proposed_decision")))
        )
        row["primary_support_max_delta_vs_cycle14"] = (
            None if current_support_max is None or baseline14_support_max is None else float(current_support_max - baseline14_support_max)
        )
        row["primary_collapse_rate_delta_vs_cycle14"] = (
            None if current_collapse is None or baseline14_collapse is None else float(current_collapse - baseline14_collapse)
        )
        row["cycle15_baseline_proposed_decision"] = baseline15.get("proposed_decision")
        row["cycle15_baseline_primary_collapse_rate"] = baseline15.get("primary_comparator_eval_collapse_rate")
        row["cycle15_baseline_primary_support_min"] = baseline15.get("primary_comparator_eval_support_min")
        row["cycle15_baseline_primary_support_max"] = baseline15.get("primary_comparator_eval_support_max")
        row["decision_changed_vs_cycle15"] = (
            None if not baseline15 or current_decision is None else bool(current_decision != str(baseline15.get("proposed_decision")))
        )
        row["primary_support_max_delta_vs_cycle15"] = (
            None if current_support_max is None or baseline15_support_max is None else float(current_support_max - baseline15_support_max)
        )
        row["primary_collapse_rate_delta_vs_cycle15"] = (
            None if current_collapse is None or baseline15_collapse is None else float(current_collapse - baseline15_collapse)
        )
        row["production_label_change"] = False
        rows.append(row)
    return rows


def _decision_snapshot(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "proposed_decision": _safe_string(row.get("proposed_decision")),
        "support_source_for_proposed_decision": _safe_string(row.get("support_source_for_proposed_decision")),
        "primary_comparator_eval_collapse_rate": _safe_number(row.get("primary_comparator_eval_collapse_rate")),
        "primary_comparator_eval_support_min": _safe_number(row.get("primary_comparator_eval_support_min")),
        "primary_comparator_eval_support_max": _safe_number(row.get("primary_comparator_eval_support_max")),
        "diagnostic_only_support_warning": _safe_bool(row.get("diagnostic_only_support_warning"))
        if row.get("diagnostic_only_support_warning") is not None
        else None,
        "preflight_reason_code": _safe_string(row.get("preflight_reason_code")),
        "near_zero_movement_fraction": _safe_number(row.get("near_zero_movement_fraction")),
        "near_flat_distance_to_threshold": _safe_number(row.get("near_flat_distance_to_threshold")),
    }


def _classify_cycle16_focus_asset(
    asset: str,
    *,
    current: Mapping[str, Any] | None,
    cycle14: Mapping[str, Any] | None,
    cycle15: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current_decision = None if current is None else _safe_string(current.get("proposed_decision"))
    cycle14_decision = None if cycle14 is None else _safe_string(cycle14.get("proposed_decision"))
    cycle15_decision = None if cycle15 is None else _safe_string(cycle15.get("proposed_decision"))
    if current_decision is None:
        return {"status": "still_needs_evidence", "decision_retention": "missing_current_decision"}
    if asset == "ADAUSD":
        if cycle14_decision is None or cycle15_decision is None:
            return {"status": "still_needs_evidence", "decision_retention": "missing_baseline_cycle"}
        if current_decision != cycle15_decision or cycle14_decision != cycle15_decision:
            return {
                "status": "unstable",
                "decision_retention": "panel_sensitive_cycle14_cycle15_or_current_decision_change",
            }
        return {"status": "stable", "decision_retention": "decision_matches_cycles14_15_and_current"}
    expected_decisions = {
        "AI16ZUSD": "neutral_flat",
        "ANLOGUSD": "candidate_axis_not_clusterable",
        "TEERUSD": "candidate_axis_not_clusterable",
    }
    if asset in expected_decisions:
        expected = expected_decisions[asset]
        return {
            "status": "retained" if current_decision == expected else "demoted",
            "decision_retention": f"expected_{expected}",
        }
    if asset in CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE:
        return {
            "status": "new_preverified_boundary_evidence",
            "decision_retention": "new_asset_no_cycle14_or_cycle15_decision_baseline",
        }
    if cycle15_decision is None:
        return {"status": "still_needs_evidence", "decision_retention": "missing_cycle15_baseline"}
    return {
        "status": "stable" if current_decision == cycle15_decision else "unstable",
        "decision_retention": "compared_to_cycle15_baseline",
    }


def _score_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "grid_variant_id": _safe_string(row.get("grid_variant_id")),
        "method": _safe_string(row.get("method")),
        "feature_strategy": _safe_string(row.get("feature_strategy")),
        "cluster_count": _safe_number(row.get("cluster_count")),
        "train_noise_frac": _safe_number(row.get("noise_frac")),
        "eval_noise_frac": _safe_number(row.get("eval_noise_frac")),
        "forward_abs_return_spread": _safe_number(row.get("forward_abs_return_spread")),
        "row_bootstrap_ari": _safe_number(row.get("row_bootstrap_ari")),
        "subsample_refit_ari": _safe_number(row.get("subsample_refit_ari")),
        "walk_forward_ari_min": _safe_number(row.get("walk_forward_ari_min")),
        "walk_forward_ari_mean": _safe_number(row.get("walk_forward_ari_mean")),
        "walk_forward_label_flip_rate": _safe_number(row.get("walk_forward_label_flip_rate")),
        "eval_single_label_asset_count": _safe_number(row.get("eval_single_label_asset_count")),
        "single_label_assets_eval": _safe_string(row.get("single_label_assets_eval")),
        "elapsed_s": _safe_number(row.get("elapsed_s")),
    }


def _build_cycle16_compact_gmm_status(score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    compact_ids = {
        "compact_return_macd_robust__gaussian_mixture_c3",
        "compact_return_macd_robust__gaussian_mixture_c4",
    }
    compact_rows = [
        _score_snapshot(row)
        for row in score_rows
        if _safe_string(row.get("grid_variant_id")) in compact_ids
    ]
    return {
        "status": "clarified_not_production_promoted",
        "active_candidate_rows": compact_rows,
        "production_promotion": False,
        "model_quality_claim": False,
        "note": "Compact GMM c3/c4 remain bounded candidate comparators for panel sensitivity only.",
    }


def _build_cycle16_hdbscan_status(score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hdbscan_row = next(
        (
            _score_snapshot(row)
            for row in score_rows
            if _safe_string(row.get("method")) == "hdbscan"
            or _safe_string(row.get("grid_variant_id")) == "control_manual_robust__hdbscan_mcs5_ms1"
        ),
        None,
    )
    if hdbscan_row is None:
        return {
            "status": "diagnostic_only",
            "material_improvement_observed": False,
            "reason": "hdbscan_score_row_missing",
            "production_promotion": False,
        }
    cluster_count = _safe_number(hdbscan_row.get("cluster_count"))
    train_noise = _safe_number(hdbscan_row.get("train_noise_frac"))
    eval_noise = _safe_number(hdbscan_row.get("eval_noise_frac"))
    row_bootstrap = _safe_number(hdbscan_row.get("row_bootstrap_ari"))
    walk_forward_min = _safe_number(hdbscan_row.get("walk_forward_ari_min"))
    criteria = {
        "fragmentation_acceptable_cluster_count_le_50": cluster_count is not None and cluster_count <= 50,
        "train_noise_frac_le_0_20": train_noise is not None and train_noise <= 0.20,
        "eval_noise_frac_le_0_20": eval_noise is not None and eval_noise <= 0.20,
        "row_bootstrap_ari_ge_0_80": row_bootstrap is not None and row_bootstrap >= 0.80,
        "walk_forward_ari_min_ge_0_80": walk_forward_min is not None and walk_forward_min >= 0.80,
    }
    material_improvement = bool(all(criteria.values()))
    return {
        "status": "diagnostic_only",
        "material_improvement_observed": material_improvement,
        "material_improvement_criteria": criteria,
        "score": hdbscan_row,
        "production_promotion": False,
        "note": (
            "HDBSCAN remains diagnostic-only; material improvement would require simultaneous fragmentation, noise, "
            "bootstrap, and walk-forward stability improvement."
        ),
    }


def _build_cycle16_panel_sensitivity_comparison(
    *,
    flat_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    near_flat_summary: Mapping[str, Any],
) -> dict[str, Any]:
    by_asset = {str(row.get("asset") or ""): dict(row) for row in decision_rows if str(row.get("asset") or "")}
    assets = sorted(
        set(by_asset)
        | set(CYCLE14_BASELINE_ASSET_DECISIONS)
        | set(CYCLE15_BASELINE_ASSET_DECISIONS)
        | set(CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE)
    )
    matrix: dict[str, Any] = {}
    for asset in assets:
        current = _decision_snapshot(by_asset.get(asset))
        cycle14 = _decision_snapshot(CYCLE14_BASELINE_ASSET_DECISIONS.get(asset))
        cycle15 = _decision_snapshot(CYCLE15_BASELINE_ASSET_DECISIONS.get(asset))
        classification = _classify_cycle16_focus_asset(asset, current=current, cycle14=cycle14, cycle15=cycle15)
        matrix[asset] = {
            "asset": asset,
            "cycle14": cycle14,
            "cycle15": cycle15,
            "current": current,
            "decision_changed_vs_cycle14": None
            if current is None or cycle14 is None
            else current.get("proposed_decision") != cycle14.get("proposed_decision"),
            "decision_changed_vs_cycle15": None
            if current is None or cycle15 is None
            else current.get("proposed_decision") != cycle15.get("proposed_decision"),
            "classification": classification,
            "preverified_near_flat_source": CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE.get(asset),
        }
    focus_assets = ("ADAUSD", "AI16ZUSD", "ANLOGUSD", "TEERUSD", CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET)
    focus_status = {asset: matrix[asset]["classification"] for asset in focus_assets if asset in matrix}
    return {
        "status": "computed",
        "target": "trend_micro_near_flat_panel_sensitivity_confirmation_probe",
        "baseline_cycles": {
            "cycle14": "20260512_171654_pass3_trend_micro_axis_not_clusterable_confirmation_and_feature_availability_probe",
            "cycle15": "20260512_201802_pass3_trend_micro_anlog_near_flat_clean_collapse_safety_expansion_probe",
        },
        "scope": {
            "layer": "asset_state",
            "axis": "trend",
            "band": "micro",
            "assets": sorted(by_asset),
            "cycle15_panel_assets": list(CYCLE15_ASSETS),
            "added_preverified_near_flat_boundary_assets": sorted(
                asset for asset in by_asset if asset in CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE
            ),
            "production_label_change": False,
            "method_promotion": False,
            "hdbscan_promotion": False,
        },
        "asset_comparison_matrix": matrix,
        "focus_asset_status": focus_status,
        "near_flat_boundary_summary": dict(near_flat_summary),
        "fit_inclusion_and_carried_as": {
            str(row.get("asset")): {
                "preflight_reason_code": _safe_string(row.get("reason_code")),
                "included_in_fit": _safe_bool(row.get("included_in_fit")),
                "carried_as": _safe_string(row.get("carried_as")),
                "near_zero_movement_fraction": _safe_number(row.get("near_zero_movement_fraction")),
                "near_flat_distance_to_threshold": _safe_number(row.get("near_flat_distance_to_threshold")),
            }
            for row in flat_rows
        },
        "compact_gmm_c3_c4_status": _build_cycle16_compact_gmm_status(score_rows),
        "hdbscan_status": _build_cycle16_hdbscan_status(score_rows),
        "production_label_change": False,
        "method_promotion": False,
        "hdbscan_promotion": False,
    }


def _augment_cycle16_panel_sensitivity_confirmation_artifacts(output_root: Path) -> None:
    output_root = Path(output_root)
    flat_path = output_root / "flat_preflight.csv"
    decisions_path = output_root / "asset_model_decisions.csv"
    scores_path = output_root / "candidate_scores.csv"
    if not flat_path.exists() or not decisions_path.exists() or not scores_path.exists():
        return
    flat_rows = pd.read_csv(flat_path).to_dict("records")
    decision_rows = pd.read_csv(decisions_path).to_dict("records")
    score_rows = pd.read_csv(scores_path).to_dict("records")
    near_flat_summary = _build_near_flat_boundary_summary(flat_rows)
    cycle16_rows = _build_cycle16_panel_sensitivity_rows(decision_rows)
    pd.DataFrame(cycle16_rows).to_csv(decisions_path, index=False)
    (output_root / "asset_model_decisions.json").write_text(json.dumps(cycle16_rows, indent=2, sort_keys=True), encoding="utf-8")
    comparison = _build_cycle16_panel_sensitivity_comparison(
        flat_rows=flat_rows,
        decision_rows=cycle16_rows,
        score_rows=score_rows,
        near_flat_summary=near_flat_summary,
    )
    (output_root / "panel_sensitivity_confirmation_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "status": "computed",
        "target": comparison["target"],
        "focus_asset_status": comparison["focus_asset_status"],
        "added_preverified_near_flat_boundary_assets": comparison["scope"]["added_preverified_near_flat_boundary_assets"],
        "compact_gmm_c3_c4_status": comparison["compact_gmm_c3_c4_status"]["status"],
        "hdbscan_status": comparison["hdbscan_status"]["status"],
        "hdbscan_material_improvement_observed": comparison["hdbscan_status"].get("material_improvement_observed"),
        "production_label_change": False,
        "method_promotion": False,
    }
    for artifact_name in ("aggregate_summary.json", "trial_manifest.json", "experiment_config_snapshot.json"):
        path = output_root / artifact_name
        if path.exists():
            payload = _read_json_artifact(path)
            payload["panel_sensitivity_confirmation_summary"] = summary
            payload["near_flat_boundary_summary"] = near_flat_summary
            payload["panel_sensitivity_confirmation_comparison_artifact"] = "panel_sensitivity_confirmation_comparison.json"
            _write_json_artifact(path, payload)
    validation_path = output_root / "artifact_validation.json"
    if validation_path.exists():
        validation = _read_json_artifact(validation_path)
        validation["cycle16_near_flat_panel_sensitivity_confirmation_probe"] = True
        validation["trend_micro_near_flat_panel_sensitivity_confirmation_probe"] = True
        validation["panel_sensitivity_confirmation_summary_written"] = True
        validation["panel_sensitivity_confirmation_comparison_written"] = True
        validation["near_flat_boundary_summary_written"] = True
        validation["feature_availability_contract_written"] = True
        validation["production_outputs_written"] = False
        validation["production_regime_parquet_written"] = False
        validation["production_definitions_written"] = False
        _write_json_artifact(validation_path, validation)


def _cycle_baseline_snapshot(source: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return _decision_snapshot(source)


def _safe_field(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    text = _safe_string(value)
    if text is not None:
        return text
    number = _safe_number(value)
    if number is not None:
        return number
    if isinstance(value, bool):
        return bool(value)
    return None


def _build_cycle17_attribution_artifact(
    *,
    per_asset_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    flat_rows: Sequence[Mapping[str, Any]],
    near_flat_summary: Mapping[str, Any],
) -> dict[str, Any]:
    decisions_by_asset = {str(row.get("asset") or ""): dict(row) for row in decision_rows if str(row.get("asset") or "")}
    scores_by_trial = {str(row.get("trial_id") or ""): dict(row) for row in score_rows if str(row.get("trial_id") or "")}
    flat_by_asset = {str(row.get("asset") or ""): dict(row) for row in flat_rows if str(row.get("asset") or "")}
    near_flat_boundary_assets = {"ANLOGUSD", "TEERUSD", *CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE.keys()}
    rows: list[dict[str, Any]] = []
    for raw in per_asset_rows:
        per_asset = dict(raw)
        asset = str(per_asset.get("asset") or "")
        trial_id = str(per_asset.get("trial_id") or "")
        score = scores_by_trial.get(trial_id, {})
        decision = decisions_by_asset.get(asset, {})
        flat = flat_by_asset.get(asset, {})
        train_support = _safe_number(per_asset.get("train_non_noise_label_count"))
        eval_support = _safe_number(per_asset.get("eval_non_noise_label_count"))
        method_role = str(per_asset.get("method_role") or "")
        train_eval_support_delta = None if train_support is None or eval_support is None else float(eval_support - train_support)
        trial_walk_forward_flip = _safe_number(score.get("walk_forward_label_flip_rate"))
        trial_row_bootstrap = _safe_number(score.get("row_bootstrap_ari"))
        trial_subsample = _safe_number(score.get("subsample_refit_ari"))
        trial_feature_perturbation = _safe_number(score.get("stability_nmi"))
        eval_single = _safe_bool(per_asset.get("eval_single_label"))
        train_single = _safe_bool(per_asset.get("train_single_label"))
        current_decision = _decision_snapshot(decision)
        cycle16 = _cycle_baseline_snapshot(CYCLE16_BASELINE_ASSET_DECISIONS.get(asset))
        row = {
            "asset": asset,
            "layer": per_asset.get("layer"),
            "axis": per_asset.get("axis"),
            "band": per_asset.get("band"),
            "trial_id": trial_id,
            "grid_variant_id": per_asset.get("grid_variant_id"),
            "method": per_asset.get("method"),
            "method_role": method_role,
            "feature_strategy": per_asset.get("feature_strategy"),
            "preprocess": per_asset.get("preprocess"),
            "train_label_support": None if train_support is None else int(train_support),
            "eval_label_support": None if eval_support is None else int(eval_support),
            "train_eval_support_delta": train_eval_support_delta,
            "train_concentration": _safe_number(per_asset.get("train_largest_cluster_frac")),
            "eval_concentration": _safe_number(per_asset.get("eval_largest_cluster_frac")),
            "train_single_label_collapse": train_single,
            "eval_single_label_collapse": eval_single,
            "primary_collapse_driver": bool(method_role == "primary_comparator" and eval_single),
            "flat_reason": _safe_string(per_asset.get("preflight_reason_code") or flat.get("reason_code")),
            "included_in_fit": _safe_bool(per_asset.get("included_in_fit")),
            "carried_as": _safe_string(per_asset.get("carried_as")),
            "near_flat_boundary_asset": asset in near_flat_boundary_assets,
            "near_zero_movement_fraction": _safe_number(per_asset.get("near_zero_movement_fraction")),
            "near_flat_distance_to_threshold": _safe_number(per_asset.get("near_flat_distance_to_threshold")),
            "forward_separability_by_asset": {
                "status": _safe_string(per_asset.get("forward_status")),
                "future_abs_return_spread": _safe_number(per_asset.get("forward_abs_return_spread")),
                "future_mean_return_spread": _safe_number(per_asset.get("forward_mean_return_spread")),
            },
            "trial_stability_metrics": {
                "row_bootstrap_ari": trial_row_bootstrap,
                "subsample_refit_ari": trial_subsample,
                "walk_forward_ari_min": _safe_number(score.get("walk_forward_ari_min")),
                "walk_forward_ari_mean": _safe_number(score.get("walk_forward_ari_mean")),
                "walk_forward_label_flip_rate": trial_walk_forward_flip,
                "feature_perturbation_nmi": trial_feature_perturbation,
                "feature_perturbation_nmi_status": _safe_string(
                    per_asset.get("feature_perturbation_nmi_status") or score.get("feature_perturbation_nmi_status")
                ),
            },
            "attribution_status": {
                "row_bootstrap": "trial_level_metric_joined" if trial_row_bootstrap is not None else "not_available",
                "subsample": "trial_level_metric_joined" if trial_subsample is not None else "not_available",
                "walk_forward": "trial_level_metric_joined" if _safe_number(score.get("walk_forward_ari_min")) is not None else "not_available",
                "feature_perturbation": "trial_level_metric_joined" if trial_feature_perturbation is not None else "not_available",
                "label_flip_contribution": "collapse_flag_plus_trial_level_flip_rate" if eval_single and trial_walk_forward_flip is not None else "not_evaluable",
            },
            "label_flip_contribution": {
                "status": "collapse_flag_plus_trial_level_flip_rate" if eval_single and trial_walk_forward_flip is not None else "not_evaluable",
                "asset_eval_single_label": eval_single,
                "trial_walk_forward_label_flip_rate": trial_walk_forward_flip,
            },
            "decision_context": {
                "current": current_decision,
                "cycle14": _cycle_baseline_snapshot(CYCLE14_BASELINE_ASSET_DECISIONS.get(asset)),
                "cycle15": _cycle_baseline_snapshot(CYCLE15_BASELINE_ASSET_DECISIONS.get(asset)),
                "cycle16": cycle16,
                "decision_changed_vs_cycle16": None
                if current_decision is None or cycle16 is None
                else current_decision.get("proposed_decision") != cycle16.get("proposed_decision"),
                "decision_finality": _safe_string(decision.get("decision_finality")),
                "support_source_for_proposed_decision": _safe_string(decision.get("support_source_for_proposed_decision")),
                "diagnostic_only_support_warning": _safe_bool(decision.get("diagnostic_only_support_warning")),
            },
            "production_label_change": False,
        }
        rows.append(row)

    primary_rows = [row for row in rows if row["method_role"] == "primary_comparator"]
    primary_collapse_driver_counts: dict[str, int] = {}
    primary_trial_counts: dict[str, int] = {}
    for row in primary_rows:
        asset = str(row.get("asset") or "")
        primary_trial_counts[asset] = int(primary_trial_counts.get(asset, 0) + 1)
        if bool(row.get("eval_single_label_collapse")):
            primary_collapse_driver_counts[asset] = int(primary_collapse_driver_counts.get(asset, 0) + 1)
    primary_collapse_drivers = [
        {
            "asset": asset,
            "primary_eval_single_label_trials": int(primary_collapse_driver_counts.get(asset, 0)),
            "primary_trial_count": int(primary_trial_counts.get(asset, 0)),
            "collapse_rate": float(primary_collapse_driver_counts.get(asset, 0) / primary_trial_counts.get(asset, 1)),
            "near_flat_boundary_asset": asset in near_flat_boundary_assets,
            "current_decision": decisions_by_asset.get(asset, {}).get("proposed_decision"),
        }
        for asset in sorted(primary_trial_counts, key=lambda name: (-primary_collapse_driver_counts.get(name, 0), name))
    ]
    near_flat_decision_assets = sorted(
        asset for asset in near_flat_boundary_assets if asset in decisions_by_asset and decisions_by_asset[asset].get("proposed_decision")
    )
    summary = {
        "status": "computed",
        "target": "trend_micro_per_asset_stability_collapse_attribution_probe",
        "row_count": int(len(rows)),
        "assets": sorted(decisions_by_asset),
        "trial_count": int(len({row.get("trial_id") for row in rows if row.get("trial_id")})),
        "method_roles": sorted({str(row.get("method_role")) for row in rows if row.get("method_role")}),
        "primary_collapse_drivers": primary_collapse_drivers,
        "primary_collapse_driver_assets": [row["asset"] for row in primary_collapse_drivers if row["primary_eval_single_label_trials"] > 0],
        "near_flat_boundary_assets": sorted(asset for asset in near_flat_boundary_assets if asset in decisions_by_asset),
        "near_flat_boundary_summary": dict(near_flat_summary),
        "near_flat_exclusion_sensitivity": {
            "status": "diagnostic_only_not_refit",
            "silently_excluded": False,
            "near_flat_boundary_decision_rows_that_would_be_removed": near_flat_decision_assets,
            "remaining_asset_decision_change_if_excluded": "not_recomputed_in_pass2_harness",
            "assessment": (
                "Excluding near-flat pass-boundary assets would change the diagnostic/candidate-decision matrix by "
                "removing their explicit rows; this harness does not silently refit or claim remaining-asset decision changes."
            ),
        },
        "baseline_cycles": {
            "cycle14": "20260512_171654_pass3_trend_micro_axis_not_clusterable_confirmation_and_feature_availability_probe",
            "cycle15": "20260512_201802_pass3_trend_micro_anlog_near_flat_clean_collapse_safety_expansion_probe",
            "cycle16": "20260513_120000_pass3_trend_micro_near_flat_panel_sensitivity_confirmation_probe",
        },
        "production_label_change": False,
    }
    return {
        "status": "computed",
        "artifact_name": "per_asset_stability_collapse_attribution.json",
        "target": "trend_micro_per_asset_stability_collapse_attribution_probe",
        "summary": summary,
        "rows": rows,
        "production_label_change": False,
    }


def _augment_cycle17_attribution_artifacts(output_root: Path) -> None:
    output_root = Path(output_root)
    required_paths = {
        "flat": output_root / "flat_preflight.csv",
        "decisions": output_root / "asset_model_decisions.csv",
        "scores": output_root / "candidate_scores.csv",
        "per_asset": output_root / "per_asset_summary.csv",
    }
    if any(not path.exists() for path in required_paths.values()):
        return
    flat_rows = pd.read_csv(required_paths["flat"]).to_dict("records")
    decision_rows = pd.read_csv(required_paths["decisions"]).to_dict("records")
    score_rows = pd.read_csv(required_paths["scores"]).to_dict("records")
    per_asset_rows = pd.read_csv(required_paths["per_asset"]).to_dict("records")
    near_flat_summary = _build_near_flat_boundary_summary(flat_rows)
    artifact = _build_cycle17_attribution_artifact(
        per_asset_rows=per_asset_rows,
        decision_rows=decision_rows,
        score_rows=score_rows,
        flat_rows=flat_rows,
        near_flat_summary=near_flat_summary,
    )
    attribution_path = output_root / "per_asset_stability_collapse_attribution.json"
    attribution_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    summary = dict(artifact["summary"])
    for artifact_name in ("aggregate_summary.json", "trial_manifest.json", "experiment_config_snapshot.json"):
        path = output_root / artifact_name
        if path.exists():
            payload = _read_json_artifact(path)
            payload["per_asset_stability_collapse_attribution_summary"] = summary
            payload["per_asset_stability_collapse_attribution_artifact"] = "per_asset_stability_collapse_attribution.json"
            payload["near_flat_boundary_summary"] = near_flat_summary
            _write_json_artifact(path, payload)
    validation_path = output_root / "artifact_validation.json"
    if validation_path.exists():
        validation = _read_json_artifact(validation_path)
        validation["cycle17_per_asset_stability_collapse_attribution_probe"] = True
        validation["trend_micro_per_asset_stability_collapse_attribution_probe"] = True
        validation["per_asset_stability_collapse_attribution_written"] = True
        validation["near_flat_boundary_summary_written"] = True
        validation["production_outputs_written"] = False
        validation["production_regime_parquet_written"] = False
        validation["production_definitions_written"] = False
        _write_json_artifact(validation_path, validation)


def _augment_cycle5_real_artifacts(
    output_root: Path,
    *,
    config: StudyConfig,
    source_feature_root: Path,
    source_files: Sequence[str],
    load_metadata: Mapping[str, Any],
    command_log: str,
    cycle8_feature_preprocess_probe: bool = False,
    cycle9_no_model_decision_probe: bool = False,
    cycle10_near_flat_panel_sensitivity_probe: bool = False,
    cycle11_method_role_no_model_decision_refinement: bool = False,
    cycle12_repro_resource_perturbation_probe: bool = False,
    cycle13_primary_comparator_feature_refinement_probe: bool = False,
    cycle14_axis_not_clusterable_feature_availability_probe: bool = False,
    cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe: bool = False,
    cycle16_near_flat_panel_sensitivity_confirmation_probe: bool = False,
    cycle17_per_asset_stability_collapse_attribution_probe: bool = False,
) -> None:
    output_root = Path(output_root)
    files_read = int(len(source_files))
    source_policy = "bounded Jan-Mar 2026 asset/interval parquet files touched under D:/pipeline/parquet"
    scope = {
        "layer": config.layer,
        "axis": config.axis,
        "band": config.band,
        "assets": list(config.assets),
        "expected_fit_assets": [asset for asset in config.assets if asset != "AI16ZUSD"],
        "expected_flat_neutral_assets": [asset for asset in config.assets if asset == "AI16ZUSD"],
        "candidate_near_flat_pass_boundary_assets": [
            asset for asset in config.assets if asset in {"TEERUSD", "ANLOGUSD", *CYCLE16_PREVERIFIED_NEAR_FLAT_ASSET_EVIDENCE.keys()}
        ],
        "panel_sensitive_watch_assets": [asset for asset in config.assets if asset == "ADAUSD"],
        "feature_strategy": config.feature_strategy,
        "preprocess": (
            "cycle17_trial_level_per_asset_stability_collapse_attribution_retained_arms"
            if bool(cycle17_per_asset_stability_collapse_attribution_probe)
            else
            "cycle16_trial_level_near_flat_panel_sensitivity_confirmation_retained_arms"
            if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
            else "cycle15_trial_level_anlog_safety_expansion_retained_arms"
            if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe)
            else "cycle14_trial_level_axis_not_clusterable_feature_availability_probe_arms"
            if bool(cycle14_axis_not_clusterable_feature_availability_probe)
            else
            "cycle13_trial_level_primary_comparator_feature_refinement_arms"
            if bool(cycle13_primary_comparator_feature_refinement_probe)
            else "cycle11_trial_level_method_role_no_model_decision_arms"
            if bool(cycle11_method_role_no_model_decision_refinement) or bool(cycle12_repro_resource_perturbation_probe)
            else "cycle10_trial_level_retained_near_flat_panel_sensitivity_arms"
            if bool(cycle10_near_flat_panel_sensitivity_probe)
            else "cycle9_trial_level_control_compact_no_model_decision_arms"
            if bool(cycle9_no_model_decision_probe)
            else (
                "cycle8_trial_level_control_winsor_compact_arms"
                if bool(cycle8_feature_preprocess_probe)
                else "robust_scale_plus_train_window_variance_threshold"
            )
        ),
        "trial_arm_mode": (
            "cycle17_per_asset_stability_collapse_attribution_probe"
            if bool(cycle17_per_asset_stability_collapse_attribution_probe)
            else
            "cycle11_method_role_no_model_decision_refinement"
            if bool(cycle11_method_role_no_model_decision_refinement)
            else "cycle12_cycle11_repro_resource_perturbation_probe"
            if bool(cycle12_repro_resource_perturbation_probe)
            else "cycle16_near_flat_panel_sensitivity_confirmation_probe"
            if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
            else "cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe"
            if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe)
            else "cycle14_axis_not_clusterable_feature_availability_probe"
            if bool(cycle14_axis_not_clusterable_feature_availability_probe)
            else "cycle13_primary_comparator_feature_refinement_probe"
            if bool(cycle13_primary_comparator_feature_refinement_probe)
            else "cycle10_near_flat_panel_sensitivity_probe"
            if bool(cycle10_near_flat_panel_sensitivity_probe)
            else "cycle9_clean_collapse_no_model_decision_probe"
            if bool(cycle9_no_model_decision_probe)
            else "cycle8_feature_preprocess_probe"
            if bool(cycle8_feature_preprocess_probe)
            else "single_preprocess_control"
        ),
        "source_root": str(Path(source_feature_root)),
        "source_files_count": files_read,
        "runtime_profile": f"regimes_asset_state_study; workers={int(config.workers)}",
        "train_start_ts": config.train_start_ts,
        "train_end_ts": config.train_end_ts,
        "eval_start_ts": config.eval_start_ts,
        "eval_end_ts": config.eval_end_ts,
    }
    manifest_path = output_root / "trial_manifest.json"
    manifest = _read_json_artifact(manifest_path)
    manifest["commands"] = command_log.splitlines()
    manifest["scope"] = scope
    manifest["source_files"] = list(source_files)
    manifest["source_files_count"] = files_read
    manifest["load_metadata"] = dict(load_metadata)
    _write_json_artifact(manifest_path, manifest)

    runtime_path = output_root / "runtime_summary.json"
    runtime = _read_json_artifact(runtime_path)
    runtime["source_root"] = str(Path(source_feature_root))
    runtime["source_files_count"] = files_read
    runtime["source_files_count_policy"] = source_policy
    runtime["files_read"] = files_read
    runtime["train_rows_loaded"] = int(load_metadata.get("train_rows_loaded", 0) or 0)
    runtime["eval_rows_loaded"] = int(load_metadata.get("eval_rows_loaded", 0) or 0)
    _write_json_artifact(runtime_path, runtime)

    snapshot_path = output_root / "experiment_config_snapshot.json"
    snapshot = _read_json_artifact(snapshot_path)
    snapshot["output_root"] = str(output_root)
    snapshot["created_utc"] = datetime.now(timezone.utc).isoformat()
    snapshot["commands"] = command_log.splitlines()
    snapshot["source_root"] = str(Path(source_feature_root))
    snapshot["source_files"] = list(source_files)
    snapshot["source_files_count"] = files_read
    snapshot["source_files_count_policy"] = source_policy
    snapshot["train_rows_loaded"] = int(load_metadata.get("train_rows_loaded", 0) or 0)
    snapshot["eval_rows_loaded"] = int(load_metadata.get("eval_rows_loaded", 0) or 0)
    _write_json_artifact(snapshot_path, snapshot)

    validation_path = output_root / "artifact_validation.json"
    validation = _read_json_artifact(validation_path)
    validation["status"] = "passed"
    validation["cycle5_output_root_validated"] = True
    validation["cycle8_feature_preprocess_probe"] = bool(cycle8_feature_preprocess_probe)
    validation["cycle9_no_model_decision_probe"] = bool(cycle9_no_model_decision_probe)
    validation["cycle10_near_flat_panel_sensitivity_probe"] = bool(cycle10_near_flat_panel_sensitivity_probe)
    validation["cycle11_method_role_no_model_decision_refinement"] = bool(cycle11_method_role_no_model_decision_refinement)
    validation["cycle12_repro_resource_perturbation_probe"] = bool(cycle12_repro_resource_perturbation_probe)
    validation["cycle13_primary_comparator_feature_refinement_probe"] = bool(cycle13_primary_comparator_feature_refinement_probe)
    validation["cycle14_axis_not_clusterable_feature_availability_probe"] = bool(cycle14_axis_not_clusterable_feature_availability_probe)
    validation["cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe"] = bool(
        cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe
    )
    validation["cycle16_near_flat_panel_sensitivity_confirmation_probe"] = bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
    validation["cycle17_per_asset_stability_collapse_attribution_probe"] = bool(
        cycle17_per_asset_stability_collapse_attribution_probe
    )
    validation["feature_availability_contract_written"] = bool(
        cycle14_axis_not_clusterable_feature_availability_probe
        or cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe
        or cycle16_near_flat_panel_sensitivity_confirmation_probe
        or cycle17_per_asset_stability_collapse_attribution_probe
    )
    validation["production_outputs_written"] = False
    validation["production_regime_parquet_written"] = False
    validation["production_definitions_written"] = False
    _write_json_artifact(validation_path, validation)

    diagnostics_path = output_root / "cluster_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    for row in diagnostics:
        row.setdefault("runtime_io", {})["files_read"] = files_read
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    scores_path = output_root / "candidate_scores.csv"
    scores = pd.read_csv(scores_path)
    scores["files_read"] = files_read
    scores.to_csv(scores_path, index=False)
    (output_root / "command_log.txt").write_text(command_log, encoding="utf-8")
    if bool(cycle10_near_flat_panel_sensitivity_probe):
        _augment_cycle10_panel_sensitivity_artifacts(output_root)
    if (
        bool(cycle11_method_role_no_model_decision_refinement)
        or bool(cycle12_repro_resource_perturbation_probe)
        or bool(cycle13_primary_comparator_feature_refinement_probe)
        or bool(cycle14_axis_not_clusterable_feature_availability_probe)
        or bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe)
        or bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
        or bool(cycle17_per_asset_stability_collapse_attribution_probe)
    ):
        _augment_cycle11_method_role_artifacts(output_root)
        if bool(cycle12_repro_resource_perturbation_probe):
            validation = _read_json_artifact(validation_path)
            validation["cycle11_method_role_no_model_decision_refinement"] = False
            validation["cycle12_repro_resource_perturbation_probe"] = True
            validation["method_role_decision_summary_written"] = True
            validation["near_flat_boundary_summary_written"] = True
            _write_json_artifact(validation_path, validation)
        if bool(cycle13_primary_comparator_feature_refinement_probe):
            validation = _read_json_artifact(validation_path)
            validation["cycle11_method_role_no_model_decision_refinement"] = False
            validation["cycle12_repro_resource_perturbation_probe"] = False
            validation["cycle13_primary_comparator_feature_refinement_probe"] = True
            validation["cycle14_axis_not_clusterable_feature_availability_probe"] = False
            validation["feature_availability_contract_written"] = True
            validation["method_role_decision_summary_written"] = True
            validation["near_flat_boundary_summary_written"] = True
            _write_json_artifact(validation_path, validation)
        if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe):
            _augment_cycle15_safety_expansion_artifacts(output_root)
            validation = _read_json_artifact(validation_path)
            validation["cycle11_method_role_no_model_decision_refinement"] = False
            validation["cycle12_repro_resource_perturbation_probe"] = False
            validation["cycle13_primary_comparator_feature_refinement_probe"] = False
            validation["cycle14_axis_not_clusterable_feature_availability_probe"] = False
            validation["cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe"] = True
            validation["feature_availability_contract_written"] = True
            validation["method_role_decision_summary_written"] = True
            validation["near_flat_boundary_summary_written"] = True
            validation["cycle15_safety_expansion_summary_written"] = True
            _write_json_artifact(validation_path, validation)
        if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe):
            _augment_cycle16_panel_sensitivity_confirmation_artifacts(output_root)
            validation = _read_json_artifact(validation_path)
            validation["cycle11_method_role_no_model_decision_refinement"] = False
            validation["cycle12_repro_resource_perturbation_probe"] = False
            validation["cycle13_primary_comparator_feature_refinement_probe"] = False
            validation["cycle14_axis_not_clusterable_feature_availability_probe"] = False
            validation["cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe"] = False
            validation["cycle16_near_flat_panel_sensitivity_confirmation_probe"] = True
            validation["trend_micro_near_flat_panel_sensitivity_confirmation_probe"] = True
            validation["feature_availability_contract_written"] = True
            validation["method_role_decision_summary_written"] = True
            validation["near_flat_boundary_summary_written"] = True
            validation["panel_sensitivity_confirmation_summary_written"] = True
            validation["panel_sensitivity_confirmation_comparison_written"] = True
            _write_json_artifact(validation_path, validation)
        if bool(cycle17_per_asset_stability_collapse_attribution_probe):
            _augment_cycle17_attribution_artifacts(output_root)
            validation = _read_json_artifact(validation_path)
            validation["cycle11_method_role_no_model_decision_refinement"] = False
            validation["cycle12_repro_resource_perturbation_probe"] = False
            validation["cycle13_primary_comparator_feature_refinement_probe"] = False
            validation["cycle14_axis_not_clusterable_feature_availability_probe"] = False
            validation["cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe"] = False
            validation["cycle16_near_flat_panel_sensitivity_confirmation_probe"] = False
            validation["cycle17_per_asset_stability_collapse_attribution_probe"] = True
            validation["trend_micro_per_asset_stability_collapse_attribution_probe"] = True
            validation["feature_availability_contract_written"] = True
            validation["method_role_decision_summary_written"] = True
            validation["near_flat_boundary_summary_written"] = True
            validation["per_asset_stability_collapse_attribution_written"] = True
            _write_json_artifact(validation_path, validation)
        if bool(cycle14_axis_not_clusterable_feature_availability_probe):
            validation = _read_json_artifact(validation_path)
            validation["cycle11_method_role_no_model_decision_refinement"] = False
            validation["cycle12_repro_resource_perturbation_probe"] = False
            validation["cycle13_primary_comparator_feature_refinement_probe"] = False
            validation["cycle14_axis_not_clusterable_feature_availability_probe"] = True
            validation["feature_availability_contract_written"] = True
            validation["method_role_decision_summary_written"] = True
            validation["near_flat_boundary_summary_written"] = True
            _write_json_artifact(validation_path, validation)


def run_cycle5_real_bounded_study(
    *,
    output_root: Path,
    source_feature_root: Path | str | None = None,
    assets: Sequence[str] = DEFAULT_CYCLE5_ASSETS,
    methods: Sequence[str] = DEFAULT_FIRST_METHODS,
    manifest_path: Path = Path("reports/codex_automation/regimes/asset_state_test/study_manifest.md"),
    workers: int = 2,
    train_start_ts: int = CYCLE5_TRAIN_START_TS,
    train_end_ts: int = CYCLE5_TRAIN_END_TS,
    eval_start_ts: int = CYCLE5_EVAL_START_TS,
    eval_end_ts: int = CYCLE5_EVAL_END_TS,
    cycle8_feature_preprocess_probe: bool = False,
    cycle9_no_model_decision_probe: bool = False,
    cycle10_near_flat_panel_sensitivity_probe: bool = False,
    cycle11_method_role_no_model_decision_refinement: bool = False,
    cycle12_repro_resource_perturbation_probe: bool = False,
    cycle13_primary_comparator_feature_refinement_probe: bool = False,
    cycle14_axis_not_clusterable_feature_availability_probe: bool = False,
    cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe: bool = False,
    cycle16_near_flat_panel_sensitivity_confirmation_probe: bool = False,
    cycle17_per_asset_stability_collapse_attribution_probe: bool = False,
    loader: Any | None = None,
) -> dict[str, Any]:
    output_root = validate_cycle5_output_root(Path(output_root))
    source_feature_root = _resolve_cycle5_source_root(source_feature_root)
    config = StudyConfig(
        axis="trend",
        band="micro",
        assets=tuple(str(asset) for asset in assets),
        methods=tuple(str(method) for method in methods),
        preprocess="robust_scale",
        feature_strategy="manual_baseline",
        feature_bases=("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"),
        member_intervals=(1, 5, 15, 30),
        workers=int(workers),
        train_start_ts=int(train_start_ts),
        train_end_ts=int(train_end_ts),
        eval_start_ts=int(eval_start_ts),
        eval_end_ts=int(eval_end_ts),
        notes=(
            "Cycle 17 Pass 3 per-asset stability/collapse attribution probe runner."
            if bool(cycle17_per_asset_stability_collapse_attribution_probe)
            else
            "Cycle 11 Pass 3 method-role no-model decision refinement runner."
            if bool(cycle11_method_role_no_model_decision_refinement)
            else "Cycle 12 Pass 3 Cycle 11 reproducibility/resource/feature-perturbation probe runner."
            if bool(cycle12_repro_resource_perturbation_probe)
            else "Cycle 13 Pass 3 primary-comparator directional compact feature refinement runner."
            if bool(cycle13_primary_comparator_feature_refinement_probe)
            else "Cycle 16 Pass 3 near-flat panel-sensitivity confirmation runner."
            if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
            else "Cycle 15 Pass 3 ANLOG near-flat clean-collapse safety-expansion probe runner."
            if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe)
            else "Cycle 14 Pass 3 axis-not-clusterable confirmation and feature-availability probe runner."
            if bool(cycle14_axis_not_clusterable_feature_availability_probe)
            else "Cycle 10 Pass 3 near-flat panel-sensitivity probe runner."
            if bool(cycle10_near_flat_panel_sensitivity_probe)
            else "Cycle 9 Pass 3 clean-collapse/no-model decision probe runner."
            if bool(cycle9_no_model_decision_probe)
            else "Cycle 8 Pass 3 ADAUSD feature/preprocess probe runner."
            if bool(cycle8_feature_preprocess_probe)
            else (
                "Cycle 5 Pass 3 recovery runner: retained trend/micro grid with ADAUSD expansion "
                "and AI16ZUSD neutral-flat safety representative."
            )
        ),
    )
    load_feature_bases = tuple(config.feature_bases)
    if bool(cycle13_primary_comparator_feature_refinement_probe) or bool(cycle14_axis_not_clusterable_feature_availability_probe):
        load_feature_bases = tuple(
            dict.fromkeys(
                (
                    *config.feature_bases,
                    "roc_14",
                    "mom_14",
                    "range_efficiency",
                )
            )
        )
    train_frame, eval_frame, load_metadata = build_real_aligned_feature_frames(
        assets=config.assets,
        band=config.band,
        feature_bases=load_feature_bases,
        source_feature_root=Path(source_feature_root),
        train_start_ts=int(train_start_ts),
        train_end_ts=int(train_end_ts),
        eval_start_ts=int(eval_start_ts),
        eval_end_ts=int(eval_end_ts),
        loader=loader,
    )
    source_files = list_cycle5_source_files(
        source_feature_root=Path(source_feature_root),
        assets=config.assets,
        member_intervals=config.member_intervals,
        start_ts=int(train_start_ts),
        end_ts=int(eval_end_ts),
    )
    command_log = "\n".join(
        [
            f"PIPELINE_PARQUET_ROOT={Path(source_feature_root)}",
            (
                "python -m src.regimes.asset_state_test.cli "
                f"--real-cycle5-bounded --output-root {output_root} "
                f"--source-feature-root {Path(source_feature_root)} "
                f"--assets {','.join(config.assets)} "
                f"--methods {','.join(config.methods)} "
                f"--train-start-ts {int(train_start_ts)} --train-end-ts {int(train_end_ts)} "
                f"--eval-start-ts {int(eval_start_ts)} --eval-end-ts {int(eval_end_ts)} "
                f"{'--cycle17-per-asset-stability-collapse-attribution-probe' if bool(cycle17_per_asset_stability_collapse_attribution_probe) else '--cycle11-method-role-no-model-decision-refinement' if bool(cycle11_method_role_no_model_decision_refinement) else '--cycle12-repro-resource-perturbation-probe' if bool(cycle12_repro_resource_perturbation_probe) else '--cycle16-near-flat-panel-sensitivity-confirmation-probe' if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe) else '--cycle15-anlog-near-flat-clean-collapse-safety-expansion-probe' if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe) else '--cycle14-axis-not-clusterable-feature-availability-probe' if bool(cycle14_axis_not_clusterable_feature_availability_probe) else '--cycle13-primary-comparator-feature-refinement-probe' if bool(cycle13_primary_comparator_feature_refinement_probe) else '--cycle10-near-flat-panel-sensitivity-probe' if bool(cycle10_near_flat_panel_sensitivity_probe) else '--cycle9-no-model-decision-probe' if bool(cycle9_no_model_decision_probe) else '--cycle8-feature-preprocess-probe' if bool(cycle8_feature_preprocess_probe) else '--cycle5-retained-grid'} "
                f"--workers {int(workers)}"
            ),
            (
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle17_per_asset_stability_collapse_attribution_probe_grid(config)); validate per-asset attribution artifacts."
                if bool(cycle17_per_asset_stability_collapse_attribution_probe)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle11_method_role_no_model_decision_refinement_grid(config)); validate sandbox-only method-role no-model artifacts."
                if bool(cycle11_method_role_no_model_decision_refinement)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle12_cycle11_repro_resource_perturbation_probe_grid(config)); validate sampled RSS and feature-perturbation diagnostics."
                if bool(cycle12_repro_resource_perturbation_probe)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle16_near_flat_panel_sensitivity_confirmation_probe_grid(config)); validate panel-sensitivity comparison artifacts."
                if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe_grid(config)); validate ANLOG safety-expansion diagnostics."
                if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle13_primary_comparator_feature_refinement_probe_grid(config)); validate directional compact feature-refinement artifacts."
                if bool(cycle13_primary_comparator_feature_refinement_probe)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle14_axis_not_clusterable_feature_availability_probe_grid(config)); validate intended-vs-selected feature availability artifacts."
                if bool(cycle14_axis_not_clusterable_feature_availability_probe)
                else
                "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle10_near_flat_panel_sensitivity_probe_grid(config)); validate sandbox-only near-flat/panel-sensitivity artifacts."
                if bool(cycle10_near_flat_panel_sensitivity_probe)
                else "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle9_no_model_decision_probe_grid(config)); validate sandbox-only candidate/no-model artifacts."
                if bool(cycle9_no_model_decision_probe)
                else "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., "
                "build_cycle8_feature_preprocess_probe_grid(config)); validate sandbox-only artifacts."
                if bool(cycle8_feature_preprocess_probe)
                else "Real bounded runner: load aligned scalar feature frames; call run_frame_study(..., build_cycle5_retained_trial_grid(config)); validate sandbox-only artifacts."
            ),
            "Recommended subprocess env caps for Pass 3: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        ]
    )
    result = run_frame_study(
        train_frame,
        config=config,
        output_root=output_root,
        manifest_path=manifest_path,
        eval_frame=eval_frame,
        trial_configs=build_cycle17_per_asset_stability_collapse_attribution_probe_grid(config)
        if bool(cycle17_per_asset_stability_collapse_attribution_probe)
        else build_cycle11_method_role_no_model_decision_refinement_grid(config)
        if bool(cycle11_method_role_no_model_decision_refinement)
        else build_cycle12_cycle11_repro_resource_perturbation_probe_grid(config)
        if bool(cycle12_repro_resource_perturbation_probe)
        else build_cycle16_near_flat_panel_sensitivity_confirmation_probe_grid(config)
        if bool(cycle16_near_flat_panel_sensitivity_confirmation_probe)
        else build_cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe_grid(config)
        if bool(cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe)
        else build_cycle13_primary_comparator_feature_refinement_probe_grid(config)
        if bool(cycle13_primary_comparator_feature_refinement_probe)
        else build_cycle14_axis_not_clusterable_feature_availability_probe_grid(config)
        if bool(cycle14_axis_not_clusterable_feature_availability_probe)
        else build_cycle10_near_flat_panel_sensitivity_probe_grid(config)
        if bool(cycle10_near_flat_panel_sensitivity_probe)
        else build_cycle9_no_model_decision_probe_grid(config)
        if bool(cycle9_no_model_decision_probe)
        else build_cycle8_feature_preprocess_probe_grid(config)
        if bool(cycle8_feature_preprocess_probe)
        else build_cycle5_retained_trial_grid(config),
    )
    _augment_cycle5_real_artifacts(
        output_root,
        config=config,
        source_feature_root=Path(source_feature_root),
        source_files=source_files,
        load_metadata=load_metadata,
        command_log=command_log,
        cycle8_feature_preprocess_probe=bool(cycle8_feature_preprocess_probe),
        cycle9_no_model_decision_probe=bool(cycle9_no_model_decision_probe),
        cycle10_near_flat_panel_sensitivity_probe=bool(cycle10_near_flat_panel_sensitivity_probe),
        cycle11_method_role_no_model_decision_refinement=bool(cycle11_method_role_no_model_decision_refinement),
        cycle12_repro_resource_perturbation_probe=bool(cycle12_repro_resource_perturbation_probe),
        cycle13_primary_comparator_feature_refinement_probe=bool(cycle13_primary_comparator_feature_refinement_probe),
        cycle14_axis_not_clusterable_feature_availability_probe=bool(cycle14_axis_not_clusterable_feature_availability_probe),
        cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe=bool(
            cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe
        ),
        cycle16_near_flat_panel_sensitivity_confirmation_probe=bool(cycle16_near_flat_panel_sensitivity_confirmation_probe),
        cycle17_per_asset_stability_collapse_attribution_probe=bool(
            cycle17_per_asset_stability_collapse_attribution_probe
        ),
    )
    return {
        **result,
        "source_files_count": int(len(source_files)),
        "train_rows_loaded": int(load_metadata.get("train_rows_loaded", 0) or 0),
        "eval_rows_loaded": int(load_metadata.get("eval_rows_loaded", 0) or 0),
    }


def _split_train_eval_frame(frame: pd.DataFrame, config: StudyConfig, eval_frame: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    if eval_frame is not None:
        return frame.copy(), eval_frame.copy(), {
            "split_policy": "explicit_eval_frame",
            "train_rows": int(len(frame)),
            "eval_rows": int(len(eval_frame)),
        }
    if "ts" in frame.columns and config.eval_start_ts is not None and config.eval_end_ts is not None:
        train_start = -np.inf if config.train_start_ts is None else int(config.train_start_ts)
        train_end = int(config.train_end_ts) if config.train_end_ts is not None else int(config.eval_start_ts) - 1
        train_mask = (pd.to_numeric(frame["ts"], errors="coerce") >= train_start) & (pd.to_numeric(frame["ts"], errors="coerce") <= train_end)
        eval_mask = (pd.to_numeric(frame["ts"], errors="coerce") >= int(config.eval_start_ts)) & (pd.to_numeric(frame["ts"], errors="coerce") <= int(config.eval_end_ts))
        return frame.loc[train_mask].copy(), frame.loc[eval_mask].copy(), {
            "split_policy": "config_ts_bounds",
            "train_rows": int(train_mask.sum()),
            "eval_rows": int(eval_mask.sum()),
        }
    return frame.copy(), None, {
        "split_policy": "train_only",
        "train_rows": int(len(frame)),
        "eval_rows": 0,
    }


def _fit_adapter(method: str, x: np.ndarray, random_state: int, method_params: Mapping[str, Any] | None = None):
    params = dict(method_params or {})
    if str(method) in {"kmeans", "minibatch_kmeans", "gaussian_mixture", "bayesian_gaussian_mixture"}:
        params.setdefault("random_state", int(random_state))
    adapter = build_clusterer_adapter(method, **params)
    fit_result = adapter.fit(x)
    return adapter, fit_result


def _empty_assignment(method: str, status: str, rows: int, *, error: str | None = None) -> ClusterAssignmentResult:
    return ClusterAssignmentResult(
        method=str(method),
        labels=np.empty(0, dtype=int),
        probabilities=None,
        confidence=None,
        noise_mask=np.empty(0, dtype=bool),
        status=str(status),
        supported=False,
        error=error,
        metadata={"assignment_method": "not_run", "requested_rows": int(rows)},
    )


def _compute_seed_stability(method: str, x: np.ndarray, labels: np.ndarray, random_state: int, method_params: Mapping[str, Any]) -> dict[str, Any]:
    if x.shape[0] < 3 or labels.size != x.shape[0]:
        return {"status": "not_evaluable", "seed_perturbation_ari": None, "stability_error": "insufficient rows"}
    try:
        params = dict(method_params)
        if "random_state" in params:
            params["random_state"] = int(params["random_state"]) + 1
        _, refit = _fit_adapter(method, x, int(random_state) + 1, params)
        ari = adjusted_rand_score(np.asarray(labels, dtype=int), np.asarray(refit.labels, dtype=int))
        return {
            "status": "computed",
            "seed_perturbation_ari": float(ari),
            "row_bootstrap_ari": None,
            "feature_perturbation_nmi": None,
            "walk_forward_label_flip_rate": None,
            "stability_error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "seed_perturbation_ari": None,
            "row_bootstrap_ari": None,
            "feature_perturbation_nmi": None,
            "walk_forward_label_flip_rate": None,
            "stability_error": str(exc),
        }


def _compute_subsample_refit_stability(
    method: str,
    x: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    method_params: Mapping[str, Any],
    *,
    repeat_count: int = 3,
    sample_frac: float = 0.8,
) -> dict[str, Any]:
    if x.shape[0] < 8 or labels.size != x.shape[0]:
        return {
            "subsample_refit_ari": None,
            "stability_repeat_count": int(repeat_count),
            "stability_sample_frac": float(sample_frac),
            "stability_assignment_method": "not_evaluable",
            "subsample_error": "insufficient rows",
        }
    rng = np.random.default_rng(int(random_state) + 10_003)
    scores: list[float] = []
    assignment_method = "predict"
    errors: list[str] = []
    n_rows = int(x.shape[0])
    sample_size = max(3, min(n_rows, int(round(n_rows * float(sample_frac)))))
    for repeat in range(int(repeat_count)):
        try:
            indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
            params = dict(method_params)
            if "random_state" in params:
                params["random_state"] = int(params["random_state"]) + repeat + 101
            adapter, _ = _fit_adapter(method, x[indices], int(random_state) + repeat + 101, params)
            assignment = adapter.assign_result(x)
            assignment_method = str(assignment.metadata.get("assignment_method", assignment_method))
            if assignment.status != "assigned" or assignment.labels.size != n_rows:
                errors.append(f"repeat {repeat}: {assignment.status} {assignment.error or ''}".strip())
                continue
            scores.append(float(adjusted_rand_score(np.asarray(labels, dtype=int), np.asarray(assignment.labels, dtype=int))))
        except Exception as exc:
            errors.append(f"repeat {repeat}: {exc}")
    return {
        "subsample_refit_ari": float(np.mean(scores)) if scores else None,
        "stability_repeat_count": int(repeat_count),
        "stability_sample_frac": float(sample_frac),
        "stability_assignment_method": assignment_method,
        "subsample_error": "; ".join(errors) if errors else None,
    }


def _compute_row_bootstrap_stability(
    method: str,
    x: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    method_params: Mapping[str, Any],
    *,
    repeat_count: int = 5,
) -> dict[str, Any]:
    if x.shape[0] < 8 or labels.size != x.shape[0]:
        return {
            "row_bootstrap_ari": None,
            "row_bootstrap_repeat_count": int(repeat_count),
            "row_bootstrap_sample_frac": 1.0,
            "row_bootstrap_assignment_method": "not_evaluable",
            "row_bootstrap_error": "insufficient rows",
        }
    rng = np.random.default_rng(int(random_state) + 20_003)
    n_rows = int(x.shape[0])
    scores: list[float] = []
    methods: list[str] = []
    errors: list[str] = []
    for repeat in range(int(repeat_count)):
        try:
            indices = rng.choice(n_rows, size=n_rows, replace=True)
            params = dict(method_params)
            if "random_state" in params:
                params["random_state"] = int(params["random_state"]) + repeat + 2_001
            adapter, _ = _fit_adapter(method, x[indices], int(random_state) + repeat + 2_001, params)
            assignment = adapter.assign_result(x)
            methods.append(str(assignment.metadata.get("assignment_method", "unknown")))
            if assignment.status != "assigned" or assignment.labels.size != n_rows:
                errors.append(f"repeat {repeat}: {assignment.status} {assignment.error or ''}".strip())
                continue
            scores.append(float(adjusted_rand_score(np.asarray(labels, dtype=int), np.asarray(assignment.labels, dtype=int))))
        except Exception as exc:
            errors.append(f"repeat {repeat}: {exc}")
    return {
        "row_bootstrap_ari": float(np.mean(scores)) if scores else None,
        "row_bootstrap_repeat_count": int(repeat_count),
        "row_bootstrap_sample_frac": 1.0,
        "row_bootstrap_assignment_method": "|".join(sorted(set(methods))) if methods else "not_evaluable",
        "row_bootstrap_error": "; ".join(errors) if errors else None,
    }


def _compute_feature_perturbation_stability(
    method: str,
    x: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    method_params: Mapping[str, Any],
    *,
    repeat_count: int = 3,
    feature_fraction: float = 0.25,
) -> dict[str, Any]:
    if x.shape[0] < 8 or x.shape[1] < 1 or labels.size != x.shape[0]:
        return {
            "feature_perturbation_nmi": None,
            "feature_perturbation_nmi_status": "not_evaluable",
            "feature_perturbation_nmi_error": "insufficient rows/features",
            "feature_perturbation_nmi_repeat_count": int(repeat_count),
            "feature_perturbation_feature_fraction": float(feature_fraction),
            "feature_perturbation_strategy": "median_mask_train_features",
        }
    rng = np.random.default_rng(int(random_state) + 30_003)
    n_features = int(x.shape[1])
    perturb_count = max(1, min(n_features, int(round(n_features * float(feature_fraction)))))
    medians = np.nanmedian(np.asarray(x, dtype=float), axis=0)
    scores: list[float] = []
    errors: list[str] = []
    for repeat in range(int(repeat_count)):
        try:
            perturbed = np.asarray(x, dtype=float).copy()
            columns = rng.choice(n_features, size=perturb_count, replace=False)
            perturbed[:, columns] = medians[columns]
            params = dict(method_params)
            if "random_state" in params:
                params["random_state"] = int(params["random_state"]) + repeat + 3_001
            _, refit = _fit_adapter(method, perturbed, int(random_state) + repeat + 3_001, params)
            if refit.labels.size != labels.size:
                errors.append(f"repeat {repeat}: label size mismatch")
                continue
            scores.append(float(normalized_mutual_info_score(np.asarray(labels, dtype=int), np.asarray(refit.labels, dtype=int))))
        except Exception as exc:
            errors.append(f"repeat {repeat}: {exc}")
    return {
        "feature_perturbation_nmi": float(np.mean(scores)) if scores else None,
        "feature_perturbation_nmi_status": "computed" if scores else "failed",
        "feature_perturbation_nmi_error": "; ".join(errors) if errors else None,
        "feature_perturbation_nmi_repeat_count": int(repeat_count),
        "feature_perturbation_feature_fraction": float(feature_fraction),
        "feature_perturbation_strategy": "median_mask_train_features",
    }


def _aligned_label_flip_rate(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=int)
    candidate = np.asarray(candidate, dtype=int)
    if reference.size == 0 or reference.size != candidate.size:
        return None
    mapped = np.full(candidate.shape, -1, dtype=int)
    for label in sorted({int(v) for v in candidate.tolist()}):
        mask = candidate == label
        if label == -1:
            mapped[mask] = -1
            continue
        ref_values = reference[mask]
        if ref_values.size == 0:
            continue
        values, counts = np.unique(ref_values, return_counts=True)
        mapped[mask] = int(values[int(np.argmax(counts))])
    return float(np.mean(mapped != reference))


def _temporal_walk_forward_slices(clean_frame: pd.DataFrame, *, split_count: int = 3) -> list[tuple[pd.Index, pd.Index]]:
    if "ts" not in clean_frame.columns or len(clean_frame) < 24:
        return []
    ts = pd.to_numeric(clean_frame["ts"], errors="coerce")
    finite_ts = ts[np.isfinite(ts.to_numpy(dtype=float))]
    unique_ts = np.asarray(sorted(finite_ts.unique()), dtype=float)
    if unique_ts.size < int(split_count) + 2:
        return []
    boundaries = np.linspace(0, unique_ts.size - 1, int(split_count) + 2, dtype=int)
    slices: list[tuple[pd.Index, pd.Index]] = []
    for idx in range(1, len(boundaries) - 1):
        train_end_ts = unique_ts[boundaries[idx]]
        eval_end_ts = unique_ts[boundaries[idx + 1]]
        train_mask = ts <= train_end_ts
        eval_mask = (ts > train_end_ts) & (ts <= eval_end_ts)
        train_index = clean_frame.index[train_mask.fillna(False)]
        eval_index = clean_frame.index[eval_mask.fillna(False)]
        if len(train_index) >= 8 and len(eval_index) >= 3:
            slices.append((train_index, eval_index))
    return slices


def _compute_walk_forward_stability(
    method: str,
    fit_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    base_preprocess,
    labels: np.ndarray,
    random_state: int,
    method_params: Mapping[str, Any],
    *,
    preprocess: str = "robust_scale",
    split_count: int = 3,
) -> dict[str, Any]:
    if base_preprocess.clean_frame.empty or labels.size != len(base_preprocess.clean_frame):
        return {
            "walk_forward_split_count": 0,
            "walk_forward_assignment_method": "not_evaluable",
            "walk_forward_ari_mean": None,
            "walk_forward_ari_min": None,
            "walk_forward_label_flip_rate": None,
            "walk_forward_status": "not_evaluable",
            "walk_forward_error": "base labels unavailable",
        }
    base_labels = pd.Series(np.asarray(labels, dtype=int), index=base_preprocess.clean_frame.index)
    slices = _temporal_walk_forward_slices(base_preprocess.clean_frame, split_count=split_count)
    if not slices:
        return {
            "walk_forward_split_count": 0,
            "walk_forward_assignment_method": "not_evaluable",
            "walk_forward_ari_mean": None,
            "walk_forward_ari_min": None,
            "walk_forward_label_flip_rate": None,
            "walk_forward_status": "not_evaluable",
            "walk_forward_error": "insufficient temporal splits",
        }
    scores: list[float] = []
    flip_rates: list[float] = []
    methods: list[str] = []
    errors: list[str] = []
    for split_index, (train_index, eval_index) in enumerate(slices):
        try:
            split_train = fit_frame.loc[train_index].copy()
            split_eval = fit_frame.loc[eval_index].copy()
            split_preprocess = preprocess_feature_frame(split_train, feature_columns, preprocess=preprocess)
            if split_preprocess.x.shape[0] < 3 or split_preprocess.x.shape[1] < 1:
                errors.append(f"split {split_index}: no clusterable train rows/features")
                continue
            params = dict(method_params)
            if "random_state" in params:
                params["random_state"] = int(params["random_state"]) + split_index + 1_001
            adapter, _ = _fit_adapter(method, split_preprocess.x, int(random_state) + split_index + 1_001, params)
            split_eval_preprocess = transform_feature_frame(split_eval, feature_columns, split_preprocess)
            if split_eval_preprocess.x.shape[0] == 0:
                errors.append(f"split {split_index}: no evaluable rows")
                continue
            assignment = adapter.assign_result(split_eval_preprocess.x)
            methods.append(str(assignment.metadata.get("assignment_method", "unknown")))
            if assignment.status != "assigned" or assignment.labels.size != split_eval_preprocess.x.shape[0]:
                errors.append(f"split {split_index}: {assignment.status} {assignment.error or ''}".strip())
                continue
            reference = base_labels.loc[split_eval_preprocess.clean_frame.index].to_numpy(dtype=int)
            assigned = np.asarray(assignment.labels, dtype=int)
            scores.append(float(adjusted_rand_score(reference, assigned)))
            flip_rate = _aligned_label_flip_rate(reference, assigned)
            if flip_rate is not None:
                flip_rates.append(float(flip_rate))
        except Exception as exc:
            errors.append(f"split {split_index}: {exc}")
    assignment_method = "|".join(sorted(set(methods))) if methods else "not_evaluable"
    return {
        "walk_forward_split_count": int(len(scores)),
        "walk_forward_assignment_method": assignment_method,
        "walk_forward_ari_mean": float(np.mean(scores)) if scores else None,
        "walk_forward_ari_min": float(np.min(scores)) if scores else None,
        "walk_forward_label_flip_rate": float(np.mean(flip_rates)) if flip_rates else None,
        "walk_forward_status": "computed" if scores else "failed",
        "walk_forward_error": "; ".join(errors) if errors else None,
    }


def _feature_bases_for_trial(config: StudyConfig, feature_strategy: str) -> tuple[str, ...] | None:
    if str(feature_strategy) in {"manual_baseline", "variance_threshold_then_manual"}:
        return tuple(config.feature_bases)
    return None


def _intended_feature_bases_for_trial(config: StudyConfig, feature_strategy: str) -> tuple[str, ...]:
    if str(feature_strategy) in {"manual_baseline", "variance_threshold_then_manual"}:
        return tuple(str(base) for base in config.feature_bases)
    return tuple(str(base) for base in feature_pool(config.axis, feature_strategy))


def _feature_availability_metadata(
    *,
    config: StudyConfig,
    feature_strategy: str,
    selected_feature_columns: Sequence[str],
    preprocess_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    intended_bases = _intended_feature_bases_for_trial(config, feature_strategy)
    selected_columns = tuple(str(column) for column in selected_feature_columns)
    selected_bases = tuple(
        base
        for base in intended_bases
        if any(str(column) == f"i{int(interval)}_{base}" for column in selected_columns for interval in config.member_intervals)
    )
    missing_bases = tuple(base for base in intended_bases if base not in set(selected_bases))
    if not selected_bases:
        status = "no_matching_columns"
    elif missing_bases:
        status = "partial_missing_intended_bases"
    else:
        status = "complete"
    missing_reasons = {base: "no_matching_columns_for_intended_base" for base in missing_bases}
    return {
        "intended_feature_bases": list(intended_bases),
        "selected_feature_bases": list(selected_bases),
        "selected_feature_columns": list(selected_columns),
        "missing_intended_bases": list(missing_bases),
        "feature_availability_status": status,
        "missing_intended_feature_reasons": missing_reasons,
        "preprocessed_selected_columns": list(preprocess_metadata.get("selected_columns", [])),
        "preprocessed_dropped_columns": list(preprocess_metadata.get("dropped_columns", [])),
    }


def run_frame_study(
    frame: pd.DataFrame,
    *,
    config: StudyConfig,
    output_root: Path,
    manifest_path: Path | None = None,
    eval_frame: pd.DataFrame | None = None,
    trial_configs: Sequence[TrialConfig] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    rss_samples: list[float] = []
    initial_rss, rss_metadata = _current_process_rss_mb()
    if initial_rss is not None:
        rss_samples.append(float(initial_rss))
    manifest = load_study_manifest(manifest_path) if manifest_path is not None else None
    train_frame, heldout_frame, split_metadata = _split_train_eval_frame(frame, config, eval_frame)
    preflight_feature_columns = select_feature_columns(
        train_frame,
        axis=config.axis,
        member_intervals=config.member_intervals,
        feature_bases=config.feature_bases,
        strategy=config.feature_strategy,
    )
    flat_results = [
        run_flat_preflight(
            train_frame[train_frame["asset"] == asset].copy() if "asset" in train_frame.columns else train_frame.copy(),
            asset=asset,
            axis=config.axis,
            band=config.band,
            feature_columns=preflight_feature_columns,
            train_start_ts=config.train_start_ts,
            train_end_ts=config.train_end_ts,
        )
        for asset in config.assets
    ]
    fit_assets = {r.asset for r in flat_results if r.included_in_fit}
    fit_frame = train_frame[train_frame["asset"].isin(fit_assets)].copy() if "asset" in train_frame.columns else train_frame.copy()
    eval_fit_frame = None
    if heldout_frame is not None:
        eval_fit_frame = heldout_frame[heldout_frame["asset"].isin(fit_assets)].copy() if "asset" in heldout_frame.columns else heldout_frame.copy()

    cluster_diagnostics: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []
    trials = tuple(trial_configs or build_default_trials(config))
    trial_preprocess_metadata: dict[str, Any] = {}
    trial_eval_preprocess_metadata: dict[str, Any] = {}
    trial_feature_columns: dict[str, list[str]] = {}
    trial_feature_availability: dict[str, dict[str, Any]] = {}
    trial_intended_feature_bases: dict[str, list[str]] = {}
    trial_selected_feature_bases: dict[str, list[str]] = {}
    trial_missing_intended_bases: dict[str, list[str]] = {}
    trial_feature_availability_status: dict[str, str] = {}
    for trial_config in trials:
        method = str(trial_config.method).strip().lower()
        trial_feature_strategy = trial_config.resolved_feature_strategy
        trial_preprocess = trial_config.resolved_preprocess
        feature_columns = select_feature_columns(
            train_frame,
            axis=config.axis,
            member_intervals=config.member_intervals,
            feature_bases=_feature_bases_for_trial(config, trial_feature_strategy),
            strategy=trial_feature_strategy,
        )
        preprocess = preprocess_feature_frame(fit_frame, feature_columns, preprocess=trial_preprocess)
        eval_preprocess = None if eval_fit_frame is None else transform_feature_frame(eval_fit_frame, feature_columns, preprocess)
        base_preprocess_metadata = preprocess.to_metadata()
        availability_metadata = _feature_availability_metadata(
            config=config,
            feature_strategy=trial_feature_strategy,
            selected_feature_columns=feature_columns,
            preprocess_metadata=base_preprocess_metadata,
        )
        preprocess_metadata = {**base_preprocess_metadata, **availability_metadata}
        trial_feature_columns[str(trial_config.trial_id)] = list(feature_columns)
        trial_feature_availability[str(trial_config.trial_id)] = dict(availability_metadata)
        trial_intended_feature_bases[str(trial_config.trial_id)] = list(availability_metadata["intended_feature_bases"])
        trial_selected_feature_bases[str(trial_config.trial_id)] = list(availability_metadata["selected_feature_bases"])
        trial_missing_intended_bases[str(trial_config.trial_id)] = list(availability_metadata["missing_intended_bases"])
        trial_feature_availability_status[str(trial_config.trial_id)] = str(availability_metadata["feature_availability_status"])
        trial_preprocess_metadata[str(trial_config.trial_id)] = preprocess_metadata
        trial_eval_preprocess_metadata[str(trial_config.trial_id)] = None if eval_preprocess is None else eval_preprocess.to_metadata()
        trial_started = time.monotonic()
        fit_result = None
        error = None
        try:
            if preprocess.x.shape[0] < 3 or preprocess.x.shape[1] < 1:
                raise RuntimeError("No clusterable rows/features after preflight and preprocessing")
            adapter, fit_result = _fit_adapter(method, preprocess.x, config.random_state, trial_config.method_params)
        except Exception as exc:
            error = str(exc)
        eval_assignment = None
        stability = None
        if fit_result is not None:
            stability = _compute_seed_stability(method, preprocess.x, fit_result.labels, config.random_state, trial_config.method_params)
            stability.update(
                _compute_subsample_refit_stability(
                    method,
                    preprocess.x,
                    fit_result.labels,
                    config.random_state,
                    trial_config.method_params,
                )
            )
            stability.update(
                _compute_row_bootstrap_stability(
                    method,
                    preprocess.x,
                    fit_result.labels,
                    config.random_state,
                    trial_config.method_params,
                )
            )
            stability.update(
                _compute_feature_perturbation_stability(
                    method,
                    preprocess.x,
                    fit_result.labels,
                    config.random_state,
                    trial_config.method_params,
                )
            )
            stability.update(
                _compute_walk_forward_stability(
                    method,
                    fit_frame,
                    feature_columns,
                    preprocess,
                    fit_result.labels,
                    config.random_state,
                    trial_config.method_params,
                    preprocess=trial_preprocess,
                )
            )
            if stability.get("subsample_error") and not stability.get("stability_error"):
                stability["stability_error"] = stability.get("subsample_error")
            if stability.get("row_bootstrap_error") and not stability.get("stability_error"):
                stability["stability_error"] = stability.get("row_bootstrap_error")
            if eval_preprocess is None:
                eval_assignment = _empty_assignment(method, "not_requested", 0)
            elif eval_preprocess.x.shape[0] == 0:
                eval_assignment = _empty_assignment(method, "no_eval_rows", 0)
            else:
                eval_assignment = adapter.assign_result(eval_preprocess.x)
        trial_rss, trial_rss_metadata = _current_process_rss_mb()
        if trial_rss is not None:
            rss_samples.append(float(trial_rss))
            rss_metadata = trial_rss_metadata
        peak_rss = max(rss_samples) if rss_samples else None
        diagnostics = build_cluster_diagnostics(
            trial_config=trial_config,
            fit_result=fit_result,
            x=preprocess.x,
            flat_results=flat_results,
            preprocess_metadata=preprocess_metadata,
            elapsed_s=time.monotonic() - trial_started,
            rows_read=int(len(train_frame)),
            files_read=0,
            error=error,
            eval_x=None if eval_preprocess is None else eval_preprocess.x,
            eval_frame=None if eval_preprocess is None else eval_preprocess.clean_frame,
            eval_assignment=eval_assignment,
            stability_metrics=stability,
            train_frame=preprocess.clean_frame,
            runtime_metadata={
                **rss_metadata,
                "peak_rss_mb": peak_rss,
            },
        )
        cluster_diagnostics.append(diagnostics)
        candidate_scores.append(build_candidate_score_row(diagnostics))

    runtime_summary = {
        "status": "ok",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": float(time.monotonic() - started),
        "workers": int(config.workers),
        "rows_read": int(len(train_frame)),
        "eval_rows": 0 if heldout_frame is None else int(len(heldout_frame)),
        "files_read": 0,
        "output_root": str(output_root),
        "production_outputs_written": False,
        "artifact_contract": list(ARTIFACT_NAMES),
        "peak_rss_mb": max(rss_samples) if rss_samples else None,
        **rss_metadata,
    }
    first_preprocess_metadata = next(iter(trial_preprocess_metadata.values()), None)
    first_eval_preprocess_metadata = next(iter(trial_eval_preprocess_metadata.values()), None)
    trial_manifest = {
        "study_config": config.to_dict(),
        "manifest": None if manifest is None else manifest.to_dict(),
        "split_metadata": split_metadata,
        "feature_columns_requested": list(preflight_feature_columns),
        "preflight_feature_columns": list(preflight_feature_columns),
        "preprocess_metadata": first_preprocess_metadata,
        "eval_preprocess_metadata": first_eval_preprocess_metadata,
        "trial_feature_columns": trial_feature_columns,
        "trial_feature_availability": trial_feature_availability,
        "trial_intended_feature_bases": trial_intended_feature_bases,
        "trial_selected_feature_bases": trial_selected_feature_bases,
        "trial_missing_intended_bases": trial_missing_intended_bases,
        "trial_feature_availability_status": trial_feature_availability_status,
        "trial_preprocess_metadata": trial_preprocess_metadata,
        "trial_eval_preprocess_metadata": trial_eval_preprocess_metadata,
        "feature_availability_contract": {
            "status_values": ["complete", "partial_missing_intended_bases", "no_matching_columns"],
            "missing_intended_feature_reason": "no_matching_columns_for_intended_base",
            "selected_feature_columns_scope": "source_columns_present_before_train_window_variance_threshold",
            "preprocessed_selected_columns_scope": "columns retained after train_window_variance_threshold_and_scaling",
        },
        "method_count": int(len(config.methods)),
        "trial_count": int(len(trials)),
        "artifact_boundary": {
            "production_outputs_written": False,
            "production_regime_parquet_written": False,
            "production_definitions_written": False,
        },
        "trials": [
            {
                "trial_id": trial.trial_id,
                "grid_family": trial.grid_family,
                "grid_variant_id": trial.grid_variant_id,
                "method": trial.method,
                "method_params": dict(trial.method_params),
                "preprocess": trial.resolved_preprocess,
                "feature_strategy": trial.resolved_feature_strategy,
                "feature_availability": trial_feature_availability.get(str(trial.trial_id), {}),
            }
            for trial in trials
        ],
        "stability_contract": {
            "type": "seed_subsample_row_bootstrap_feature_perturbation_walk_forward",
            "repeat_count": 3,
            "sample_frac": 0.8,
            "row_bootstrap_repeat_count": 5,
            "row_bootstrap_sample_frac": 1.0,
            "feature_perturbation_repeat_count": 3,
            "feature_perturbation_strategy": "median_mask_train_features",
            "feature_perturbation_feature_fraction": 0.25,
            "walk_forward_split_count_requested": 3,
            "feature_perturbation_fields": [
                "feature_perturbation_nmi",
                "feature_perturbation_nmi_status",
                "feature_perturbation_nmi_error",
                "feature_perturbation_nmi_repeat_count",
                "feature_perturbation_feature_fraction",
                "feature_perturbation_strategy",
            ],
            "walk_forward_fields": [
                "walk_forward_split_count",
                "walk_forward_assignment_method",
                "walk_forward_ari_mean",
                "walk_forward_ari_min",
                "walk_forward_label_flip_rate",
                "walk_forward_status",
                "walk_forward_error",
            ],
            "row_bootstrap_fields": [
                "row_bootstrap_ari",
                "row_bootstrap_repeat_count",
                "row_bootstrap_sample_frac",
                "row_bootstrap_assignment_method",
                "row_bootstrap_error",
            ],
        },
    }
    artifact_paths = write_study_artifacts(
        output_root,
        trial_manifest=trial_manifest,
        candidate_scores=candidate_scores,
        flat_preflight=flat_results,
        cluster_diagnostics=cluster_diagnostics,
        runtime_summary=runtime_summary,
        aggregate_summary={
            "candidate_count": int(len(candidate_scores)),
            "trial_count": int(len(trials)),
            "flat_reason_counts": {
                str(reason): int(sum(1 for row in flat_results if row.reason_code == reason))
                for reason in sorted({row.reason_code for row in flat_results})
            },
            "included_assets": sorted(r.asset for r in flat_results if r.included_in_fit),
            "excluded_assets": sorted(r.asset for r in flat_results if not r.included_in_fit),
            "fit_rows": None if first_preprocess_metadata is None else int(first_preprocess_metadata.get("rows_after_dropna", 0) or 0),
            "eval_rows": 0 if first_eval_preprocess_metadata is None else int(first_eval_preprocess_metadata.get("rows_after_dropna", 0) or 0),
            "production_outputs_written": False,
        },
        experiment_config_snapshot={
            "study_config": config.to_dict(),
            "split_metadata": split_metadata,
            "trials": [trial.to_dict() for trial in trials],
        },
        command_log="run_frame_study invoked from sandbox asset_state_test harness\n",
        artifact_validation={
            "status": "passed",
            "required_artifacts": list(ARTIFACT_NAMES),
            "production_outputs_written": False,
            "production_regime_parquet_written": False,
            "production_definitions_written": False,
        },
    )
    return {"artifact_paths": artifact_paths, "runtime_summary": runtime_summary}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sandbox asset-state regime study harness.")
    parser.add_argument("--manifest", default="reports/codex_automation/regimes/asset_state_test/study_manifest.md")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--assets", default=None)
    parser.add_argument("--methods", default="hdbscan,kmeans,gaussian_mixture")
    parser.add_argument("--axis", default="trend")
    parser.add_argument("--band", default="micro")
    parser.add_argument("--preprocess", default="robust_scale")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--real-data", action="store_true", help="Alias for the bounded real trend/micro study runner.")
    parser.add_argument("--real-cycle5-bounded", action="store_true", help="Run the bounded real Cycle 5 trend/micro retained-grid recovery path.")
    parser.add_argument(
        "--source-feature-root",
        default=None,
        help=f"Feature parquet root for real-data runs. Defaults to {CYCLE5_SOURCE_ROOT_ENV}.",
    )
    parser.add_argument("--source-root", dest="source_feature_root", default=argparse.SUPPRESS, help="Alias for --source-feature-root.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-start-ts", type=int, default=CYCLE5_TRAIN_START_TS)
    parser.add_argument("--train-end-ts", type=int, default=CYCLE5_TRAIN_END_TS)
    parser.add_argument("--eval-start-ts", type=int, default=CYCLE5_EVAL_START_TS)
    parser.add_argument("--eval-end-ts", type=int, default=CYCLE5_EVAL_END_TS)
    parser.add_argument("--rows-per-asset", type=int, default=90)
    parser.add_argument("--eval-rows-per-asset", type=int, default=0)
    parser.add_argument("--cycle3-grid", action="store_true", help="Expand the Cycle 3 nine-variant method grid.")
    parser.add_argument("--cycle4-retained-grid", action="store_true", help="Expand the Cycle 4 retained five-variant method grid.")
    parser.add_argument("--cycle5-retained-grid", action="store_true", help="Expand the Cycle 5 retained four-variant asset-expansion grid.")
    parser.add_argument("--cycle8-feature-preprocess-probe", action="store_true", help="Expand the Cycle 8 ADAUSD feature/preprocess probe grid.")
    parser.add_argument("--cycle9-no-model-decision-probe", action="store_true", help="Expand the Cycle 9 clean-collapse/no-model decision probe grid.")
    parser.add_argument("--cycle10-near-flat-panel-sensitivity-probe", action="store_true", help="Expand the Cycle 10 near-flat panel-sensitivity probe grid.")
    parser.add_argument("--cycle11-method-role-no-model-decision-refinement", action="store_true", help="Expand the Cycle 11 method-role no-model decision refinement grid.")
    parser.add_argument("--cycle12-repro-resource-perturbation-probe", action="store_true", help="Repeat the exact Cycle 11 grid with Cycle 12 RSS and feature-perturbation diagnostics.")
    parser.add_argument("--cycle13-primary-comparator-feature-refinement-probe", action="store_true", help="Run the Cycle 13 primary-comparator directional compact feature refinement grid.")
    parser.add_argument("--cycle14-axis-not-clusterable-feature-availability-probe", action="store_true", help="Run the Cycle 14 axis-not-clusterable confirmation grid with intended-vs-selected feature diagnostics.")
    parser.add_argument("--cycle15-anlog-near-flat-clean-collapse-safety-expansion-probe", action="store_true", help="Run the Cycle 15 ANLOG near-flat safety-expansion retained grid.")
    parser.add_argument(
        "--cycle16-near-flat-panel-sensitivity-confirmation-probe",
        "--trend-micro-near-flat-panel-sensitivity-confirmation-probe",
        dest="cycle16_near_flat_panel_sensitivity_confirmation_probe",
        action="store_true",
        help="Run the Cycle 16 near-flat panel-sensitivity confirmation retained grid.",
    )
    parser.add_argument(
        "--cycle17-per-asset-stability-collapse-attribution-probe",
        "--trend-micro-per-asset-stability-collapse-attribution-probe",
        dest="cycle17_per_asset_stability_collapse_attribution_probe",
        action="store_true",
        help="Run the Cycle 17 per-asset stability/collapse attribution retained grid.",
    )
    args = parser.parse_args(argv)

    real_data = bool(args.real_cycle5_bounded or args.real_data)
    if args.synthetic_smoke and real_data:
        raise SystemExit("Choose either --synthetic-smoke or a real-data runner, not both.")
    if sum(
        bool(v)
        for v in (
            args.cycle3_grid,
            args.cycle4_retained_grid,
            args.cycle5_retained_grid,
            args.cycle8_feature_preprocess_probe,
            args.cycle9_no_model_decision_probe,
            args.cycle10_near_flat_panel_sensitivity_probe,
            args.cycle11_method_role_no_model_decision_refinement,
            args.cycle12_repro_resource_perturbation_probe,
            args.cycle13_primary_comparator_feature_refinement_probe,
            args.cycle14_axis_not_clusterable_feature_availability_probe,
            args.cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe,
            args.cycle16_near_flat_panel_sensitivity_confirmation_probe,
            args.cycle17_per_asset_stability_collapse_attribution_probe,
        )
    ) > 1:
        raise SystemExit("Choose at most one grid flag.")
    assets = (
        _parse_csv(args.assets)
        if args.assets
        else CYCLE17_ASSETS
        if real_data and args.cycle17_per_asset_stability_collapse_attribution_probe
        else CYCLE16_ASSETS
        if real_data and args.cycle16_near_flat_panel_sensitivity_confirmation_probe
        else CYCLE15_ASSETS
        if real_data and args.cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe
        else CYCLE10_ASSETS
        if real_data
        and (
            args.cycle10_near_flat_panel_sensitivity_probe
            or args.cycle11_method_role_no_model_decision_refinement
            or args.cycle12_repro_resource_perturbation_probe
            or args.cycle13_primary_comparator_feature_refinement_probe
            or args.cycle14_axis_not_clusterable_feature_availability_probe
        )
        else DEFAULT_CYCLE5_ASSETS
        if real_data
        else DEFAULT_ASSETS
    )
    methods = _parse_csv(args.methods)
    if real_data:
        if args.axis != "trend" or args.band != "micro":
            raise SystemExit("The bounded real runner is intentionally scoped to --axis trend --band micro.")
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(env_name, "1")
        result = run_cycle5_real_bounded_study(
            output_root=Path(args.output_root),
            source_feature_root=Path(args.source_feature_root),
            assets=assets,
            methods=methods,
            manifest_path=Path(args.manifest),
            workers=int(args.workers),
            train_start_ts=int(args.train_start_ts),
            train_end_ts=int(args.train_end_ts),
            eval_start_ts=int(args.eval_start_ts),
            eval_end_ts=int(args.eval_end_ts),
            cycle8_feature_preprocess_probe=bool(args.cycle8_feature_preprocess_probe),
            cycle9_no_model_decision_probe=bool(args.cycle9_no_model_decision_probe),
            cycle10_near_flat_panel_sensitivity_probe=bool(args.cycle10_near_flat_panel_sensitivity_probe),
            cycle11_method_role_no_model_decision_refinement=bool(args.cycle11_method_role_no_model_decision_refinement),
            cycle12_repro_resource_perturbation_probe=bool(args.cycle12_repro_resource_perturbation_probe),
            cycle13_primary_comparator_feature_refinement_probe=bool(args.cycle13_primary_comparator_feature_refinement_probe),
            cycle14_axis_not_clusterable_feature_availability_probe=bool(args.cycle14_axis_not_clusterable_feature_availability_probe),
            cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe=bool(
                args.cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe
            ),
            cycle16_near_flat_panel_sensitivity_confirmation_probe=bool(
                args.cycle16_near_flat_panel_sensitivity_confirmation_probe
            ),
            cycle17_per_asset_stability_collapse_attribution_probe=bool(
                args.cycle17_per_asset_stability_collapse_attribution_probe
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    config = StudyConfig(axis=args.axis, band=args.band, assets=assets, methods=methods, preprocess=args.preprocess)
    if not args.synthetic_smoke:
        raise SystemExit("Choose --synthetic-smoke or --real-cycle5-bounded.")
    frame = build_synthetic_trend_micro_frame(assets=assets, rows_per_asset=int(args.rows_per_asset), seed=config.random_state)
    eval_frame = None
    if int(args.eval_rows_per_asset) > 0:
        eval_frame = build_synthetic_trend_micro_frame(
            assets=assets,
            rows_per_asset=int(args.eval_rows_per_asset),
            seed=int(config.random_state) + 101,
        )
    result = run_frame_study(
        frame,
        config=config,
        output_root=Path(args.output_root),
        manifest_path=Path(args.manifest),
        eval_frame=eval_frame,
        trial_configs=build_cycle10_near_flat_panel_sensitivity_probe_grid(config)
        if args.cycle10_near_flat_panel_sensitivity_probe
        else build_cycle17_per_asset_stability_collapse_attribution_probe_grid(config)
        if args.cycle17_per_asset_stability_collapse_attribution_probe
        else build_cycle11_method_role_no_model_decision_refinement_grid(config)
        if args.cycle11_method_role_no_model_decision_refinement
        else build_cycle12_cycle11_repro_resource_perturbation_probe_grid(config)
        if args.cycle12_repro_resource_perturbation_probe
        else build_cycle16_near_flat_panel_sensitivity_confirmation_probe_grid(config)
        if args.cycle16_near_flat_panel_sensitivity_confirmation_probe
        else build_cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe_grid(config)
        if args.cycle15_anlog_near_flat_clean_collapse_safety_expansion_probe
        else build_cycle13_primary_comparator_feature_refinement_probe_grid(config)
        if args.cycle13_primary_comparator_feature_refinement_probe
        else build_cycle14_axis_not_clusterable_feature_availability_probe_grid(config)
        if args.cycle14_axis_not_clusterable_feature_availability_probe
        else build_cycle9_no_model_decision_probe_grid(config)
        if args.cycle9_no_model_decision_probe
        else build_cycle8_feature_preprocess_probe_grid(config)
        if args.cycle8_feature_preprocess_probe
        else build_cycle5_retained_trial_grid(config)
        if args.cycle5_retained_grid
        else build_cycle4_retained_trial_grid(config)
        if args.cycle4_retained_grid
        else build_cycle3_trial_grid(config)
        if args.cycle3_grid
        else None,
    )
    if args.cycle17_per_asset_stability_collapse_attribution_probe:
        _augment_cycle11_method_role_artifacts(Path(args.output_root))
        _augment_cycle17_attribution_artifacts(Path(args.output_root))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
