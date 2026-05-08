from __future__ import annotations

from src.forecasting.ml.bayesian.shared.bayesian_optuna_runner import BayesianOptunaModelSpec, main_for_model, parse_args, requested_assets, resolve_combo_specs

MODEL_SPEC = BayesianOptunaModelSpec(
    model_key="dynamic_regression_shrinkage",
    display_name="Bayesian Dynamic Regression Shrinkage",
    numerics_module_import_path="src.forecasting.ml.bayesian.dynamic_regression_shrinkage.numerics",
    optuna_profile_import_path="src.forecasting.ml.bayesian.dynamic_regression_shrinkage.dynamic_regression_shrinkage_optuna_profile",
    default_study_name_prefix="bayesian_dynamic_regression_shrinkage",
)


def main() -> None:
    main_for_model(MODEL_SPEC)


if __name__ == "__main__":
    main()
