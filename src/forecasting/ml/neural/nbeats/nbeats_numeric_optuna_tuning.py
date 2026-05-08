from __future__ import annotations

from src.forecasting.ml.neural.shared.neural_optuna_runner import NeuralOptunaModelSpec, main_for_model, parse_args, requested_assets, resolve_combo_specs

MODEL_SPEC = NeuralOptunaModelSpec(
    model_key="nbeats",
    display_name="Neural N-BEATS",
    numerics_module_import_path="src.forecasting.ml.neural.nbeats.numerics",
    optuna_profile_import_path="src.forecasting.ml.neural.nbeats.nbeats_optuna_profile",
    default_study_name_prefix="neural_nbeats",
)


def main() -> None:
    main_for_model(MODEL_SPEC)


if __name__ == "__main__":
    main()
