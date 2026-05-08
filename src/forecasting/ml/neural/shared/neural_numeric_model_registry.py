from __future__ import annotations

NEURAL_NUMERIC_BRANCHES = (
    "lstm",
    "tcn",
    "nbeats",
)

NEURAL_NUMERIC_ENTRYPOINTS = {
    "lstm": "src.forecasting.ml.neural.lstm.numerics",
    "tcn": "src.forecasting.ml.neural.tcn.numerics",
    "nbeats": "src.forecasting.ml.neural.nbeats.numerics",
}

NEURAL_NUMERIC_STAGE1_ENTRYPOINTS = {
    "lstm": "src.forecasting.ml.neural.lstm.lstm_feature_experiment",
    "tcn": "src.forecasting.ml.neural.tcn.tcn_feature_experiment",
    "nbeats": "src.forecasting.ml.neural.nbeats.nbeats_feature_experiment",
}

NEURAL_STAGE0_ENTRYPOINT = "src.forecasting.ml.neural.shared.neural_numeric_stage0_profile"

NEURAL_NUMERIC_FAMILY_ROOT_ENVS = {
    "lstm": "PIPELINE_PARQUET_NEURALTS_LSTM_ROOT",
    "tcn": "PIPELINE_PARQUET_NEURALTS_TCN_ROOT",
    "nbeats": "PIPELINE_PARQUET_NEURALTS_NBEATS_ROOT",
}

NEURAL_NUMERIC_FAMILY_ROOT_NAMES = {
    "lstm": "NeuralTS_LSTM",
    "tcn": "NeuralTS_TCN",
    "nbeats": "NeuralTS_NBEATS",
}
