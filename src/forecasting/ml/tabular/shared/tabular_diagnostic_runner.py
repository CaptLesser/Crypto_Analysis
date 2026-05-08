from __future__ import annotations

from src.forecasting.ml.tabular.shared.tabular_numeric_scaling_test import main_for_model

DEFAULT_MODEL_KEY = "xgboost"


def main(model_key: str = DEFAULT_MODEL_KEY) -> None:
    main_for_model(model_key)
