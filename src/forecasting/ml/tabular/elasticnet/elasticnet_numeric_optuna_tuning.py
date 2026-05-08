from __future__ import annotations

from pathlib import Path

from src.forecasting.ml.tabular.shared.tabular_optuna_runner import (
    ComboSpec,
    Stage2Context,
    TabularOptunaModelSpec,
    configure_for_model,
    load_stage2_contexts,
    main_for_model,
    parse_args,
    resolve_combo_specs,
    resolve_stage2_context_for_combo,
)

MODEL_SPEC = TabularOptunaModelSpec(
    model_key="elasticnet",
    display_name="Elastic Net",
    short_label="EN",
    numerics_module_import_path="src.forecasting.ml.tabular.elasticnet.numerics",
    optuna_profile_import_path="src.forecasting.ml.tabular.elasticnet.elasticnet_optuna_profile",
    output_root=Path("logs") / "diagnostics" / "elasticnet_numeric_optuna_combo_tuning",
    default_study_name_prefix="elasticnet_numeric_combo_rmse",
    default_model_threads=6,
    default_trials_per_combo=40,
)

configure_for_model(MODEL_SPEC)


def main() -> None:
    main_for_model(MODEL_SPEC)


if __name__ == "__main__":
    main()
