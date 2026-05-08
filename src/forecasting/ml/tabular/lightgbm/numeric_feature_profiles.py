from __future__ import annotations

from typing import List, Sequence, Set

from src.forecasting.common.ml_module_utils import select_numeric_feature_columns
from src.forecasting.ml.shared.feature_profile_common import (
    resolve_feature_profile_label as shared_feature_profile_label,
    resolve_selected_feature_columns,
)

DEFAULT_FEATURE_PROFILE = "broad_numeric"
SUPPORTED_FEATURE_PROFILES = (DEFAULT_FEATURE_PROFILE, "selected_subset")


def normalize_feature_profile(value: str) -> str:
    profile = str(value).strip().lower()
    if profile not in SUPPORTED_FEATURE_PROFILES:
        raise ValueError(f"unsupported feature profile: {value}")
    return profile


def resolve_feature_profile(task: str, horizon_minutes: int, interval_minutes: int) -> str:
    return shared_feature_profile_label(
        DEFAULT_FEATURE_PROFILE,
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        interval_minutes=int(interval_minutes),
    )


def select_feature_columns(
    columns: Sequence[str],
    task: str,
    horizon_minutes: int,
    interval_minutes: int,
    extra_exclude: Set[str],
) -> List[str]:
    return resolve_selected_feature_columns(
        columns=columns,
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        interval_minutes=int(interval_minutes),
        extra_exclude=set(extra_exclude),
        fallback_fn=lambda cols, exclude: list(select_numeric_feature_columns(cols, extra_exclude=set(exclude))),
    )


def feature_profile_label(task: str, horizon_minutes: int, interval_minutes: int) -> str:
    return resolve_feature_profile(task, horizon_minutes, interval_minutes)
