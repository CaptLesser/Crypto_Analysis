from __future__ import annotations

STATS_NUMERIC_BRANCHES = (
    "sarimax",
    "llt",
    "egarch",
    "quantreg",
)

STATS_NUMERIC_ENTRYPOINTS = {
    "sarimax": "src.forecasting.stats.sarimax.numerics",
    "llt": "src.forecasting.stats.llt.numerics",
    "egarch": "src.forecasting.stats.egarch.numerics",
    "quantreg": "src.forecasting.stats.quantreg.numerics",
}

STATS_STAGE0_ENTRYPOINT = "src.forecasting.stats.shared.stats_numeric_stage0_profile"
STATS_TEST_ORCHESTRATOR_ENTRYPOINT = "src.forecasting.stats.shared.stats_numeric_test_orchestrator"

STATS_STAGE1_ENTRYPOINTS = {
    "sarimax": "src.forecasting.stats.sarimax.sarimax_feature_experiment",
    "llt": "src.forecasting.stats.llt.llt_feature_experiment",
    "egarch": "src.forecasting.stats.egarch.egarch_feature_experiment",
    "quantreg": "src.forecasting.stats.quantreg.quantreg_feature_experiment",
}

STATS_STAGE2_ENTRYPOINTS = {
    "sarimax": "src.forecasting.stats.sarimax.sarimax_numeric_scaling_test",
    "llt": "src.forecasting.stats.llt.llt_numeric_scaling_test",
    "egarch": "src.forecasting.stats.egarch.egarch_numeric_scaling_test",
    "quantreg": "src.forecasting.stats.quantreg.quantreg_numeric_scaling_test",
}

STATS_STAGE3_ENTRYPOINTS = {
    "sarimax": "src.forecasting.stats.sarimax.sarimax_numeric_optuna_tuning",
    "llt": "src.forecasting.stats.llt.llt_numeric_optuna_tuning",
    "egarch": "src.forecasting.stats.egarch.egarch_numeric_optuna_tuning",
    "quantreg": "src.forecasting.stats.quantreg.quantreg_numeric_optuna_tuning",
}

STATS_NUMERIC_FAMILY_ROOT_ENVS = {
    "sarimax": "PIPELINE_PARQUET_SARIMAX_ROOT",
    "llt": "PIPELINE_PARQUET_LLT_ROOT",
    "egarch": "PIPELINE_PARQUET_EGARCH_ROOT",
    "quantreg": "PIPELINE_PARQUET_QR_ROOT",
}

STATS_NUMERIC_FAMILY_ROOT_NAMES = {
    "sarimax": "Stats_SARIMAX",
    "llt": "Stats_LLT",
    "egarch": "Stats_EGARCH",
    "quantreg": "Stats_QuantReg",
}
