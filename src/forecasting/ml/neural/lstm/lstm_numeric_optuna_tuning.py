from __future__ import annotations

from src.forecasting.ml.neural.shared.neural_optuna_runner import NeuralOptunaModelSpec, main_for_model, parse_args, requested_assets, resolve_combo_specs

MODEL_SPEC = NeuralOptunaModelSpec(
    model_key="lstm",
    display_name="Neural LSTM",
    numerics_module_import_path="src.forecasting.ml.neural.lstm.numerics",
    optuna_profile_import_path="src.forecasting.ml.neural.lstm.lstm_optuna_profile",
    default_study_name_prefix="neural_lstm",
)


def main() -> None:
    main_for_model(MODEL_SPEC)


if __name__ == "__main__":
    main()
