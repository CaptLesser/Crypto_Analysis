from __future__ import annotations

from src.forecasting.ml.bayesian.shared.bayesian_optuna_runner import BayesianOptunaModelSpec, main_for_model, parse_args, requested_assets, resolve_combo_specs

MODEL_SPEC = BayesianOptunaModelSpec(
    model_key="stochastic_vol",
    display_name="Bayesian Stochastic Volatility",
    numerics_module_import_path="src.forecasting.ml.bayesian.stochastic_vol.numerics",
    optuna_profile_import_path="src.forecasting.ml.bayesian.stochastic_vol.stochastic_vol_optuna_profile",
    default_study_name_prefix="bayesian_stochastic_vol",
)


def main() -> None:
    main_for_model(MODEL_SPEC)


if __name__ == "__main__":
    main()
