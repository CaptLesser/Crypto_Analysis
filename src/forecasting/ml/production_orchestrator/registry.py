from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from src.forecasting.ml.bayesian.shared.bayesian_numeric_model_registry import BAYESIAN_NUMERIC_ENTRYPOINTS
from src.forecasting.ml.neural.shared.neural_numeric_model_registry import NEURAL_NUMERIC_ENTRYPOINTS
from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import (
    CATBOOST_SPEC,
    ELASTICNET_SPEC,
    LIGHTGBM_SPEC,
    RANDOM_FOREST_SPEC,
    XGBOOST_SPEC,
)
from src.forecasting.stats.shared.stats_numeric_model_registry import (
    STATS_NUMERIC_ENTRYPOINTS,
    STATS_NUMERIC_FAMILY_ROOT_ENVS,
)


@dataclass(frozen=True)
class ProductionModuleSpec:
    module_key: str
    family: str
    display_name: str
    entrypoint: str
    runtime_config_key: Optional[str]
    module_tag: str
    parquet_root_env: Optional[str] = None

    def command(self, python_exe: str) -> List[str]:
        return [str(python_exe), "-m", str(self.entrypoint)]

    def env_hints(self, env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        source_env = env if env is not None else os.environ
        hints: Dict[str, str] = {}
        if self.parquet_root_env:
            raw = str(source_env.get(str(self.parquet_root_env), "") or "").strip()
            if raw:
                hints[str(self.parquet_root_env)] = str(raw)
        raw_root = str(source_env.get("PIPELINE_ROOT", "") or "").strip()
        if raw_root:
            hints["PIPELINE_ROOT"] = str(raw_root)
        raw_parquet_root = str(source_env.get("PIPELINE_PARQUET_ROOT", "") or "").strip()
        if raw_parquet_root:
            hints["PIPELINE_PARQUET_ROOT"] = str(raw_parquet_root)
        return hints


def mature_ml_modules() -> List[ProductionModuleSpec]:
    return [
        ProductionModuleSpec("xgboost", "tabular", "XGBoost", XGBOOST_SPEC.module_import_path, XGBOOST_SPEC.runtime_config_key, XGBOOST_SPEC.runtime_config_key, XGBOOST_SPEC.parquet_root_env),
        ProductionModuleSpec("lightgbm", "tabular", "LightGBM", LIGHTGBM_SPEC.module_import_path, LIGHTGBM_SPEC.runtime_config_key, LIGHTGBM_SPEC.runtime_config_key, LIGHTGBM_SPEC.parquet_root_env),
        ProductionModuleSpec("catboost", "tabular", "CatBoost", CATBOOST_SPEC.module_import_path, CATBOOST_SPEC.runtime_config_key, CATBOOST_SPEC.runtime_config_key, CATBOOST_SPEC.parquet_root_env),
        ProductionModuleSpec("random_forest", "tabular", "Random Forest", RANDOM_FOREST_SPEC.module_import_path, RANDOM_FOREST_SPEC.runtime_config_key, RANDOM_FOREST_SPEC.runtime_config_key, RANDOM_FOREST_SPEC.parquet_root_env),
        ProductionModuleSpec("elasticnet", "tabular", "ElasticNet", ELASTICNET_SPEC.module_import_path, ELASTICNET_SPEC.runtime_config_key, ELASTICNET_SPEC.runtime_config_key, ELASTICNET_SPEC.parquet_root_env),
        ProductionModuleSpec("dlm_tvp", "bayesian", "DLM TVP", BAYESIAN_NUMERIC_ENTRYPOINTS["dlm_tvp"], "bayesian_numeric_runner", "bayes_dlm_tvp"),
        ProductionModuleSpec("stochastic_vol", "bayesian", "Stochastic Vol", BAYESIAN_NUMERIC_ENTRYPOINTS["stochastic_vol"], "bayesian_numeric_runner", "bayes_stochastic_vol"),
        ProductionModuleSpec("dynamic_regression_shrinkage", "bayesian", "Dynamic Regression Shrinkage", BAYESIAN_NUMERIC_ENTRYPOINTS["dynamic_regression_shrinkage"], "bayesian_numeric_runner", "bayes_dynamic_regression_shrinkage"),
        ProductionModuleSpec("copula_dependency", "bayesian", "Copula Dependency", BAYESIAN_NUMERIC_ENTRYPOINTS["copula_dependency"], "bayesian_numeric_runner", "bayes_copula_dependency"),
        ProductionModuleSpec("tail_risk", "bayesian", "Tail Risk", BAYESIAN_NUMERIC_ENTRYPOINTS["tail_risk"], "bayesian_numeric_runner", "bayes_tail_risk"),
        ProductionModuleSpec("lstm", "neural", "LSTM", NEURAL_NUMERIC_ENTRYPOINTS["lstm"], "neural_numeric_runner", "neural_lstm"),
        ProductionModuleSpec("tcn", "neural", "TCN", NEURAL_NUMERIC_ENTRYPOINTS["tcn"], "neural_numeric_runner", "neural_tcn"),
        ProductionModuleSpec("nbeats", "neural", "N-BEATS", NEURAL_NUMERIC_ENTRYPOINTS["nbeats"], "neural_numeric_runner", "neural_nbeats"),
        ProductionModuleSpec("sarimax", "stats", "SARIMAX", STATS_NUMERIC_ENTRYPOINTS["sarimax"], "stats_numeric_runner", "sarimax_forecaster", STATS_NUMERIC_FAMILY_ROOT_ENVS["sarimax"]),
        ProductionModuleSpec("llt", "stats", "LLT State Space", STATS_NUMERIC_ENTRYPOINTS["llt"], "stats_numeric_runner", "llt_state_space", STATS_NUMERIC_FAMILY_ROOT_ENVS["llt"]),
        ProductionModuleSpec("egarch", "stats", "EGARCH", STATS_NUMERIC_ENTRYPOINTS["egarch"], "stats_numeric_runner", "egarch_vol", STATS_NUMERIC_FAMILY_ROOT_ENVS["egarch"]),
        ProductionModuleSpec("quantreg", "stats", "QuantReg", STATS_NUMERIC_ENTRYPOINTS["quantreg"], "stats_numeric_runner", "linear_quantile_reg", STATS_NUMERIC_FAMILY_ROOT_ENVS["quantreg"]),
    ]
