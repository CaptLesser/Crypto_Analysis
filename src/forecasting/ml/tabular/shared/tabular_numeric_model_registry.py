from __future__ import annotations

import ast
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.features.numeric_forecast_profiles import (
    APPROVED_XGB_DEFAULT_HORIZONS,
    APPROVED_XGB_DEFAULT_INTERVALS,
    APPROVED_XGB_DEFAULT_TASKS,
    filter_approved_xgb_combos,
)


@dataclass(frozen=True)
class TabularNumericModelSpec:
    model_key: str
    display_name: str
    short_label: str
    module_import_path: str
    root_script_name: Optional[str]
    runtime_config_key: str
    log_file_name: str
    forecast_table_tag: str
    eval_table_tag: str
    prediction_prefix: str
    diagnostics_output_dir: Path
    raw_accuracy_output_dir: Path
    feature_experiment_output_dir: Path
    diagnostic_analysis_output_name: str
    progress_log_prefix: str
    parquet_root_env: str
    train_windows_env: str
    progress_seconds_env: str
    source_end_env: str
    work_start_env: str
    forecast_resume_edge_env: str
    eval_resume_edge_env: str
    source_start_env: str


XGBOOST_SPEC = TabularNumericModelSpec(
    model_key="xgboost",
    display_name="XGBoost",
    short_label="XGB",
    module_import_path="src.forecasting.ml.tabular.xgboost.numerics",
    root_script_name="xgboost_numerics.py",
    runtime_config_key="xgboost_numerics",
    log_file_name="xgboost_numerics.log",
    forecast_table_tag="xgboost_numerics",
    eval_table_tag="xgboost_numerics_eval",
    prediction_prefix="xgb",
    diagnostics_output_dir=Path("logs") / "diagnostics" / "xgboost_numeric_scaling_full_test",
    raw_accuracy_output_dir=Path("logs") / "diagnostics" / "xgboost_numeric_scaling_accuracy_raw",
    feature_experiment_output_dir=Path("logs") / "diagnostics" / "xgboost_numeric_feature_experiment",
    diagnostic_analysis_output_name="xgboost_decision_memo.md",
    progress_log_prefix="[xgboost_numeric]",
    parquet_root_env="PIPELINE_PARQUET_XGB_NUMERICS_ROOT",
    train_windows_env="XGB_NUMERIC_TRAIN_WINDOWS",
    progress_seconds_env="XGB_NUMERIC_PROGRESS_SECONDS",
    source_end_env="XGB_NUMERIC_SOURCE_END_TS",
    work_start_env="XGB_NUMERIC_WORK_START_TS",
    forecast_resume_edge_env="XGB_NUMERIC_FORECAST_RESUME_EDGE_TS",
    eval_resume_edge_env="XGB_NUMERIC_EVAL_RESUME_EDGE_TS",
    source_start_env="XGB_NUMERIC_SOURCE_START_TS",
)

LIGHTGBM_SPEC = TabularNumericModelSpec(
    model_key="lightgbm",
    display_name="LightGBM",
    short_label="LGBM",
    module_import_path="src.forecasting.ml.tabular.lightgbm.numerics",
    root_script_name="lightgbm_numerics.py",
    runtime_config_key="lightgbm_numerics",
    log_file_name="lightgbm_numerics.log",
    forecast_table_tag="lightgbm_numerics",
    eval_table_tag="lightgbm_numerics_eval",
    prediction_prefix="lgbm",
    diagnostics_output_dir=Path("logs") / "diagnostics" / "lightgbm_numeric_scaling_full_test",
    raw_accuracy_output_dir=Path("logs") / "diagnostics" / "lightgbm_numeric_scaling_accuracy_raw",
    feature_experiment_output_dir=Path("logs") / "diagnostics" / "lightgbm_numeric_feature_experiment",
    diagnostic_analysis_output_name="lightgbm_decision_memo.md",
    progress_log_prefix="[lightgbm_numeric]",
    parquet_root_env="PIPELINE_PARQUET_LGB_NUMERICS_ROOT",
    train_windows_env="LGB_NUMERIC_TRAIN_WINDOWS",
    progress_seconds_env="LGB_NUMERIC_PROGRESS_SECONDS",
    source_end_env="LGB_NUMERIC_SOURCE_END_TS",
    work_start_env="LGB_NUMERIC_WORK_START_TS",
    forecast_resume_edge_env="LGB_NUMERIC_FORECAST_RESUME_EDGE_TS",
    eval_resume_edge_env="LGB_NUMERIC_EVAL_RESUME_EDGE_TS",
    source_start_env="LGB_NUMERIC_SOURCE_START_TS",
)


CATBOOST_SPEC = TabularNumericModelSpec(
    model_key="catboost",
    display_name="CatBoost",
    short_label="CB",
    module_import_path="src.forecasting.ml.tabular.catboost.numerics",
    root_script_name=None,
    runtime_config_key="catboost_numerics",
    log_file_name="catboost_numerics.log",
    forecast_table_tag="catboost_numerics",
    eval_table_tag="catboost_numerics_eval",
    prediction_prefix="cb",
    diagnostics_output_dir=Path("logs") / "diagnostics" / "catboost_numeric_scaling_full_test",
    raw_accuracy_output_dir=Path("logs") / "diagnostics" / "catboost_numeric_scaling_accuracy_raw",
    feature_experiment_output_dir=Path("logs") / "diagnostics" / "catboost_numeric_feature_experiment",
    diagnostic_analysis_output_name="catboost_decision_memo.md",
    progress_log_prefix="[cb_numeric]",
    parquet_root_env="PIPELINE_PARQUET_CB_NUMERICS_ROOT",
    train_windows_env="CB_NUMERIC_TRAIN_WINDOWS",
    progress_seconds_env="CB_NUMERIC_PROGRESS_SECONDS",
    source_end_env="CB_NUMERIC_SOURCE_END_TS",
    work_start_env="CB_NUMERIC_WORK_START_TS",
    forecast_resume_edge_env="CB_NUMERIC_FORECAST_RESUME_EDGE_TS",
    eval_resume_edge_env="CB_NUMERIC_EVAL_RESUME_EDGE_TS",
    source_start_env="CB_NUMERIC_SOURCE_START_TS",
)

RANDOM_FOREST_SPEC = TabularNumericModelSpec(
    model_key="random_forest",
    display_name="Random Forest",
    short_label="RF",
    module_import_path="src.forecasting.ml.tabular.random_forest.numerics",
    root_script_name=None,
    runtime_config_key="random_forest_numerics",
    log_file_name="random_forest_numerics.log",
    forecast_table_tag="random_forest_numerics",
    eval_table_tag="random_forest_numerics_eval",
    prediction_prefix="rf",
    diagnostics_output_dir=Path("logs") / "diagnostics" / "random_forest_numeric_scaling_full_test",
    raw_accuracy_output_dir=Path("logs") / "diagnostics" / "random_forest_numeric_scaling_accuracy_raw",
    feature_experiment_output_dir=Path("logs") / "diagnostics" / "random_forest_numeric_feature_experiment",
    diagnostic_analysis_output_name="random_forest_decision_memo.md",
    progress_log_prefix="[rf_numeric]",
    parquet_root_env="PIPELINE_PARQUET_RF_NUMERICS_ROOT",
    train_windows_env="RF_NUMERIC_TRAIN_WINDOWS",
    progress_seconds_env="RF_NUMERIC_PROGRESS_SECONDS",
    source_end_env="RF_NUMERIC_SOURCE_END_TS",
    work_start_env="RF_NUMERIC_WORK_START_TS",
    forecast_resume_edge_env="RF_NUMERIC_FORECAST_RESUME_EDGE_TS",
    eval_resume_edge_env="RF_NUMERIC_EVAL_RESUME_EDGE_TS",
    source_start_env="RF_NUMERIC_SOURCE_START_TS",
)

ELASTICNET_SPEC = TabularNumericModelSpec(
    model_key="elasticnet",
    display_name="ElasticNet",
    short_label="EN",
    module_import_path="src.forecasting.ml.tabular.elasticnet.numerics",
    root_script_name=None,
    runtime_config_key="elasticnet_numerics",
    log_file_name="elasticnet_numerics.log",
    forecast_table_tag="elasticnet_numerics",
    eval_table_tag="elasticnet_numerics_eval",
    prediction_prefix="en",
    diagnostics_output_dir=Path("logs") / "diagnostics" / "elasticnet_numeric_scaling_full_test",
    raw_accuracy_output_dir=Path("logs") / "diagnostics" / "elasticnet_numeric_scaling_accuracy_raw",
    feature_experiment_output_dir=Path("logs") / "diagnostics" / "elasticnet_numeric_feature_experiment",
    diagnostic_analysis_output_name="elasticnet_decision_memo.md",
    progress_log_prefix="[en_numeric]",
    parquet_root_env="PIPELINE_PARQUET_EN_NUMERICS_ROOT",
    train_windows_env="EN_NUMERIC_TRAIN_WINDOWS",
    progress_seconds_env="EN_NUMERIC_PROGRESS_SECONDS",
    source_end_env="EN_NUMERIC_SOURCE_END_TS",
    work_start_env="EN_NUMERIC_WORK_START_TS",
    forecast_resume_edge_env="EN_NUMERIC_FORECAST_RESUME_EDGE_TS",
    eval_resume_edge_env="EN_NUMERIC_EVAL_RESUME_EDGE_TS",
    source_start_env="EN_NUMERIC_SOURCE_START_TS",
)

_MODEL_SPECS: Dict[str, TabularNumericModelSpec] = {
    XGBOOST_SPEC.model_key: XGBOOST_SPEC,
    LIGHTGBM_SPEC.model_key: LIGHTGBM_SPEC,
    CATBOOST_SPEC.model_key: CATBOOST_SPEC,
    RANDOM_FOREST_SPEC.model_key: RANDOM_FOREST_SPEC,
    ELASTICNET_SPEC.model_key: ELASTICNET_SPEC,
}


def get_tabular_numeric_model_spec(model_key: str) -> TabularNumericModelSpec:
    key = str(model_key).strip().lower()
    if key not in _MODEL_SPECS:
        raise KeyError(f"Unsupported tabular numeric model: {model_key}")
    return _MODEL_SPECS[key]


def load_model_task_metadata(spec: TabularNumericModelSpec, project_root: Path) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    try:
        module = importlib.import_module(spec.module_import_path)
        return list(module.NUMERIC_TASKS), dict(module.TASK_SHORT), dict(module.TASK_LABEL)
    except Exception:
        if not spec.root_script_name:
            raise
        src = (project_root / spec.root_script_name).read_text(encoding="utf-8-sig").replace("﻿", "")
        tree = ast.parse(src)
        tasks: List[str] = []
        task_short: Dict[str, str] = {}
        task_label: Dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name == "NUMERIC_TASKS":
                tasks = list(ast.literal_eval(node.value))
            elif name == "TASK_SHORT":
                task_short = dict(ast.literal_eval(node.value))
            elif name == "TASK_LABEL":
                task_label = dict(ast.literal_eval(node.value))
        return tasks, task_short, task_label


def resolve_default_tasks(spec: TabularNumericModelSpec, supported_tasks: Sequence[str]) -> List[str]:
    if spec.model_key == "xgboost":
        return [task for task in APPROVED_XGB_DEFAULT_TASKS if task in supported_tasks]
    return list(supported_tasks)


def resolve_default_or_requested_combos(
    spec: TabularNumericModelSpec,
    *,
    requested_intervals: Sequence[int],
    requested_tasks: Sequence[str],
    requested_horizons: Sequence[int],
    runnable_tasks: Sequence[str],
) -> List[Tuple[int, int, str]]:
    if spec.model_key == "xgboost":
        use_default_profile = not requested_intervals and not requested_tasks and not requested_horizons
        if use_default_profile:
            return filter_approved_xgb_combos(tasks=list(runnable_tasks))
        intervals = list(requested_intervals) if requested_intervals else list(APPROVED_XGB_DEFAULT_INTERVALS)
        horizons = list(requested_horizons) if requested_horizons else list(APPROVED_XGB_DEFAULT_HORIZONS)
        tasks = list(runnable_tasks)
        return [(int(interval), int(horizon), str(task)) for interval in intervals for task in tasks for horizon in horizons]
    module = importlib.import_module(spec.module_import_path)
    use_default_profile = not requested_intervals and not requested_tasks and not requested_horizons
    if use_default_profile:
        default_combos = [
            (int(interval), int(horizon), str(task))
            for interval, horizon, task in getattr(module, "PRODUCTION_DEFAULT_COMBOS", [])
            if str(task) in set(str(value) for value in runnable_tasks)
        ]
        if default_combos:
            return default_combos
    default_intervals = [int(v) for v in getattr(module, "DEFAULT_INTERVALS", [])]
    default_horizons = [int(v) for v in getattr(module, "DEFAULT_HORIZON_MINUTES", [])]
    intervals = list(requested_intervals) if requested_intervals else default_intervals
    horizons = list(requested_horizons) if requested_horizons else default_horizons
    tasks = list(runnable_tasks)
    return [(int(interval), int(horizon), str(task)) for interval in intervals for task in tasks for horizon in horizons]


def write_runtime_config_for_model(spec: TabularNumericModelSpec, runtime_cfg_path: Path, original_text: str, model_threads: int) -> None:
    payload = json.loads(original_text)
    modules = payload.get("modules")
    if not isinstance(modules, dict):
        modules = {}
        payload["modules"] = modules
    module_cfg = modules.get(spec.runtime_config_key)
    if not isinstance(module_cfg, dict):
        module_cfg = {}
        modules[spec.runtime_config_key] = module_cfg
    module_cfg["model_threads"] = int(model_threads)
    runtime_cfg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
