from __future__ import annotations

BAYESIAN_NUMERIC_BRANCHES = (
    "dlm_tvp",
    "stochastic_vol",
    "dynamic_regression_shrinkage",
    "copula_dependency",
    "tail_risk",
)

BAYESIAN_NUMERIC_ENTRYPOINTS = {
    "dlm_tvp": "src.forecasting.ml.bayesian.dlm_tvp.numerics",
    "stochastic_vol": "src.forecasting.ml.bayesian.stochastic_vol.numerics",
    "dynamic_regression_shrinkage": "src.forecasting.ml.bayesian.dynamic_regression_shrinkage.numerics",
    "copula_dependency": "src.forecasting.ml.bayesian.copula_dependency.numerics",
    "tail_risk": "src.forecasting.ml.bayesian.tail_risk.numerics",
}

BAYESIAN_STAGE1_ENTRYPOINTS = {
    "dlm_tvp": "src.forecasting.ml.bayesian.dlm_tvp.dlm_tvp_feature_experiment",
    "stochastic_vol": "src.forecasting.ml.bayesian.stochastic_vol.stochastic_vol_feature_experiment",
    "dynamic_regression_shrinkage": "src.forecasting.ml.bayesian.dynamic_regression_shrinkage.dynamic_regression_shrinkage_feature_experiment",
    "copula_dependency": "src.forecasting.ml.bayesian.copula_dependency.copula_dependency_feature_experiment",
    "tail_risk": "src.forecasting.ml.bayesian.tail_risk.tail_risk_feature_experiment",
}

BAYESIAN_STAGE0_ENTRYPOINT = "src.forecasting.ml.bayesian.shared.bayesian_numeric_stage0_profile"

BAYESIAN_NUMERIC_FAMILY_ROOT_ENVS = {
    "dlm_tvp": "PIPELINE_PARQUET_BAYES_DLM_ROOT",
    "stochastic_vol": "PIPELINE_PARQUET_BAYES_SV_ROOT",
    "dynamic_regression_shrinkage": "PIPELINE_PARQUET_BAYES_DYNREG_ROOT",
    "copula_dependency": "PIPELINE_PARQUET_BAYES_COPULA_ROOT",
    "tail_risk": "PIPELINE_PARQUET_BAYES_TAIL_ROOT",
}

BAYESIAN_NUMERIC_FAMILY_ROOT_NAMES = {
    "dlm_tvp": "Stats_Bayes_DLM_TVP",
    "stochastic_vol": "Stats_Bayes_StochasticVol",
    "dynamic_regression_shrinkage": "Stats_Bayes_DynamicRegression",
    "copula_dependency": "Stats_Bayes_CopulaDependency",
    "tail_risk": "Stats_Bayes_TailRisk",
}

BAYESIAN_STAGE3_ENTRYPOINTS = {
    "dlm_tvp": "src.forecasting.ml.bayesian.dlm_tvp.dlm_tvp_numeric_optuna_tuning",
    "stochastic_vol": "src.forecasting.ml.bayesian.stochastic_vol.stochastic_vol_numeric_optuna_tuning",
    "dynamic_regression_shrinkage": "src.forecasting.ml.bayesian.dynamic_regression_shrinkage.dynamic_regression_shrinkage_numeric_optuna_tuning",
    "copula_dependency": "src.forecasting.ml.bayesian.copula_dependency.copula_dependency_numeric_optuna_tuning",
    "tail_risk": "src.forecasting.ml.bayesian.tail_risk.tail_risk_numeric_optuna_tuning",
}

BAYESIAN_STAGE2_ENTRYPOINTS = {
    "dlm_tvp": "src.forecasting.ml.bayesian.dlm_tvp.dlm_tvp_numeric_scaling_test",
    "stochastic_vol": "src.forecasting.ml.bayesian.stochastic_vol.stochastic_vol_numeric_scaling_test",
    "dynamic_regression_shrinkage": "src.forecasting.ml.bayesian.dynamic_regression_shrinkage.dynamic_regression_shrinkage_numeric_scaling_test",
    "copula_dependency": "src.forecasting.ml.bayesian.copula_dependency.copula_dependency_numeric_scaling_test",
    "tail_risk": "src.forecasting.ml.bayesian.tail_risk.tail_risk_numeric_scaling_test",
}
