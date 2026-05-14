from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.feature_diagnostics import FeatureSelectionDiagnostics
from src.regimes.core.feature_preprocessing import (
    PREPROCESS_FIT_SCOPE_TRAIN_ONLY,
    REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
    FeatureFilterConfig,
    FittedRegimePreprocessor,
    PreprocessorSpec,
    TransformedFeatureMatrix,
    filter_regime_feature_frame,
    fit_regime_preprocessor as _fit_regime_preprocessor,
    get_preprocessor_spec,
    preprocessing_registry,
    transform_regime_preprocessor as _transform_regime_preprocessor,
)


SCORE_WINDOW_ROLES: tuple[str, ...] = ("score", "validation", "holdout", "test")


def _normalize_role(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Regime preprocessing {field_name} must be non-empty")
    return text


def require_train_window_role(role: object) -> str:
    normalized = _normalize_role(role, field_name="fit window role")
    if normalized not in {"train", "training", "train_window"}:
        raise ValueError("Regime preprocessing fit must use a train window")
    return normalized


def require_score_window_role(role: object) -> str:
    normalized = _normalize_role(role, field_name="transform window role")
    if normalized not in SCORE_WINDOW_ROLES:
        valid = ", ".join(SCORE_WINDOW_ROLES)
        raise ValueError(f"Regime preprocessing transform must use a score-like window role; expected one of: {valid}")
    return normalized


def filter_regime_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: FeatureFilterConfig | None = None,
) -> FeatureSelectionDiagnostics:
    result = filter_regime_feature_frame(frame, feature_columns, config=config)
    return FeatureSelectionDiagnostics.from_filter_result(result)


@dataclass(frozen=True)
class PreprocessingPipelineResult:
    fitted: FittedRegimePreprocessor
    diagnostics: FeatureSelectionDiagnostics

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
            "preprocess_name": self.fitted.preprocess_name,
            "fit_scope": PREPROCESS_FIT_SCOPE_TRAIN_ONLY,
            "selected_columns": list(self.fitted.selected_columns),
            "output_feature_names": list(self.fitted.output_feature_names),
            "diagnostics": self.diagnostics.as_dict(),
            "metadata": self.fitted.to_metadata(),
        }


def selection_diagnostics_from_fitted(fitted: FittedRegimePreprocessor) -> FeatureSelectionDiagnostics:
    return FeatureSelectionDiagnostics.from_filter_result(fitted.filter_result)


def fit_train_window_preprocessor(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    preprocess: str = "robust_scale",
    preprocess_params: Mapping[str, Any] | None = None,
    filter_config: FeatureFilterConfig | None = None,
    fit_window: Mapping[str, Any] | None = None,
    fit_window_role: str = "train",
) -> FittedRegimePreprocessor:
    role = require_train_window_role(fit_window_role)
    return _fit_regime_preprocessor(
        train_frame,
        feature_columns,
        preprocess=preprocess,
        preprocess_params=preprocess_params,
        filter_config=filter_config,
        fit_window=fit_window,
        fit_window_role=role,
        allow_full_window_fit=False,
    )


def fit_preprocessing_pipeline(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    preprocess: str = "robust_scale",
    preprocess_params: Mapping[str, Any] | None = None,
    filter_config: FeatureFilterConfig | None = None,
    fit_window: Mapping[str, Any] | None = None,
    fit_window_role: str = "train",
) -> PreprocessingPipelineResult:
    fitted = fit_train_window_preprocessor(
        train_frame,
        feature_columns,
        preprocess=preprocess,
        preprocess_params=preprocess_params,
        filter_config=filter_config,
        fit_window=fit_window,
        fit_window_role=fit_window_role,
    )
    return PreprocessingPipelineResult(fitted=fitted, diagnostics=selection_diagnostics_from_fitted(fitted))


def transform_score_window_preprocessor(
    frame: pd.DataFrame,
    fitted: FittedRegimePreprocessor,
    *,
    window_role: str = "score",
) -> TransformedFeatureMatrix:
    role = require_score_window_role(window_role)
    return _transform_regime_preprocessor(frame, fitted, window_role=role)


fit_regime_preprocessor = fit_train_window_preprocessor
transform_regime_preprocessor = transform_score_window_preprocessor


__all__ = [
    "PREPROCESS_FIT_SCOPE_TRAIN_ONLY",
    "REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION",
    "SCORE_WINDOW_ROLES",
    "FeatureFilterConfig",
    "FittedRegimePreprocessor",
    "PreprocessingPipelineResult",
    "PreprocessorSpec",
    "TransformedFeatureMatrix",
    "filter_regime_features",
    "fit_preprocessing_pipeline",
    "fit_regime_preprocessor",
    "fit_train_window_preprocessor",
    "get_preprocessor_spec",
    "preprocessing_registry",
    "require_score_window_role",
    "require_train_window_role",
    "selection_diagnostics_from_fitted",
    "transform_regime_preprocessor",
    "transform_score_window_preprocessor",
]
