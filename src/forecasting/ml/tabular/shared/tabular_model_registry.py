from __future__ import annotations

import importlib
from typing import Any

from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import *  # noqa: F401,F403
from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import TabularNumericModelSpec, get_tabular_numeric_model_spec


def load_model_module(model_key: str) -> Any:
    spec = get_tabular_numeric_model_spec(model_key)
    return importlib.import_module(spec.module_import_path)


def load_model_module_for_spec(spec: TabularNumericModelSpec) -> Any:
    return importlib.import_module(spec.module_import_path)
