from __future__ import annotations

from src.forecasting.ml.tabular.lightgbm.numeric_adapter import fit_regressor, predict_block
from src.forecasting.ml.tabular.lightgbm.numeric_feature_profiles import feature_profile_label, select_feature_columns
from src.forecasting.ml.tabular.lightgbm.numeric_profiles import *
from src.forecasting.ml.tabular.lightgbm.numeric_profiles import (
    default_training_window_months_for_combo,
    normalize_refit_cadence,
    regressor_profile_label,
    resolve_lgbm_default_combo_profile,
    resolve_lgbm_default_refit_policy,
    resolve_regressor_params,
    training_window_bars_for_pair,
    training_window_bars_from_months,
)
from src.forecasting.ml.tabular.shared.numeric_forecast_runner import NumericFamilyModuleSpec, build_numeric_family_module


globals().update(
    build_numeric_family_module(
        NumericFamilyModuleSpec(
            module_slug="lightgbm_numerics",
            family_name="LightGBM",
            prediction_prefix="lgbm",
            log_prefix="[lightgbm_numeric]",
            parquet_root_env="PIPELINE_PARQUET_LGB_NUMERICS_ROOT",
            progress_seconds_env="LGB_NUMERIC_PROGRESS_SECONDS",
            source_start_env="LGB_NUMERIC_SOURCE_START_TS",
            source_end_env="LGB_NUMERIC_SOURCE_END_TS",
            work_start_env="LGB_NUMERIC_WORK_START_TS",
            forecast_resume_edge_env="LGB_NUMERIC_FORECAST_RESUME_EDGE_TS",
            eval_resume_edge_env="LGB_NUMERIC_EVAL_RESUME_EDGE_TS",
            deadzone_env_prefix="LGB_NUMERIC",
            default_unit_workers=8,
            default_model_threads=6,
            max_logical_threads=64,
            thread_env_vars=(),
            thread_param_name="n_jobs",
            numeric_tasks=NUMERIC_TASKS,
            task_short=TASK_SHORT,
            task_label=TASK_LABEL,
            future_label_columns=FUTURE_LABEL_COLUMNS,
            default_intervals=DEFAULT_INTERVALS,
            default_horizon_minutes=DEFAULT_HORIZON_MINUTES,
            active_task_horizon_matrix=ACTIVE_TASK_HORIZON_MATRIX,
            normalize_refit_cadence_fn=normalize_refit_cadence,
            resolve_default_combo_profile_fn=resolve_lgbm_default_combo_profile,
            resolve_default_refit_policy_fn=resolve_lgbm_default_refit_policy,
            select_feature_columns_fn=select_feature_columns,
            fit_model_fn=fit_regressor,
            predict_model_fn=predict_block,
            resolve_regressor_params_fn=resolve_regressor_params,
            regressor_profile_label_fn=regressor_profile_label,
            default_training_window_months_for_combo_fn=default_training_window_months_for_combo,
            training_window_bars_for_pair_fn=training_window_bars_for_pair,
            training_window_bars_from_months_fn=training_window_bars_from_months,
        )
    )
)


if __name__ == "__main__":
    main()
