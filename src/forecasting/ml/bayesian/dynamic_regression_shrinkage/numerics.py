from __future__ import annotations

from src.forecasting.ml.bayesian.shared.bayesian_runtime_bootstrap import configure_bayesian_thread_env

configure_bayesian_thread_env()

from src.forecasting.ml.bayesian.dynamic_regression_shrinkage import numeric_profiles as profiles
from src.forecasting.ml.bayesian.shared.bayesian_numeric_runner import BayesianNumericModuleSpec, run_bayesian_numeric_module

MODULE_SPEC = BayesianNumericModuleSpec(
    module_tag=profiles.MODULE_TAG,
    model_id=profiles.MODEL_ID,
    model_version=profiles.MODEL_VERSION,
    family_root_name=profiles.FAMILY_ROOT_NAME,
    family_root_env=profiles.FAMILY_ROOT_ENV,
    default_intervals=profiles.DEFAULT_INTERVALS,
    default_horizons=profiles.DEFAULT_HORIZONS,
    default_tasks=profiles.DEFAULT_TASKS,
    predict_fn=profiles.predict_fn,
    predict_batch_fn=profiles.predict_batch_fn,
    use_seasonality=profiles.USE_SEASONALITY,
    needs_dynamic_features=profiles.NEEDS_DYNAMIC_FEATURES,
    needs_factor_cache=profiles.NEEDS_FACTOR_CACHE,
    dynamic_feature_candidates=profiles.DYNAMIC_FEATURE_CANDIDATES,
    model_params=profiles.MODEL_PARAMS,
    runtime_params=profiles.RUNTIME_PARAMS,
    resolve_model_params_fn=profiles.resolve_model_params,
    resolve_default_combo_specs_fn=profiles.resolve_default_combo_specs,
)


def main() -> None:
    run_bayesian_numeric_module(MODULE_SPEC)


if __name__ == "__main__":
    main()
