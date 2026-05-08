from __future__ import annotations

from src.forecasting.ml.bayesian.shared.bayesian_feature_experiment import main_for_model


def main() -> None:
    main_for_model("copula_dependency")


if __name__ == "__main__":
    main()
