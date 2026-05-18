from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.forecasting.ml.shared.stage1_candidate_universe import STRUCTURAL_TOKEN_NAMES, is_dynamic_input_column


def _combo_key(interval: int, horizon: int, task: str) -> str:
    return f"interval={int(interval)}|horizon={int(horizon)}|task={str(task)}"


@dataclass(frozen=True)
class BayesianStage1ExecutionProfile:
    selection_semantics: str
    selected_feature_blocks: tuple[str, ...]
    selected_features: tuple[str, ...]
    selected_formulation: Dict[str, Any]
    selected_dynamic_feature_columns: tuple[str, ...]
    use_dynamic_features: bool
    use_seasonality: bool


def load_feature_profile(feature_profile_json: Path) -> Dict[str, Any]:
    payload = json.loads(Path(feature_profile_json).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in Stage 1 feature profile: {feature_profile_json}")
    return payload


def combo_payload(
    feature_profile_json: Path | Dict[str, Any],
    *,
    interval: int,
    horizon: int,
    task: str,
) -> Dict[str, Any]:
    payload = (
        feature_profile_json
        if isinstance(feature_profile_json, dict)
        else load_feature_profile(Path(feature_profile_json))
    )
    selections = payload.get("selections") or {}
    entry = selections.get(_combo_key(interval, horizon, task)) or {}
    return dict(entry) if isinstance(entry, dict) else {}


def _fallback_dynamic_columns(
    selected_features: Sequence[str],
    dynamic_feature_candidates: Sequence[str],
) -> tuple[str, ...]:
    candidate_set = {str(value) for value in dynamic_feature_candidates if str(value)}
    selected: list[str] = []
    for value in selected_features:
        feature_name = str(value)
        if feature_name in candidate_set:
            selected.append(feature_name)
            continue
        if feature_name in STRUCTURAL_TOKEN_NAMES:
            continue
        if is_dynamic_input_column(feature_name):
            selected.append(feature_name)
    return tuple(selected)


def resolve_execution_profile(
    feature_profile_json: Path | Dict[str, Any],
    *,
    interval: int,
    horizon: int,
    task: str,
    dynamic_feature_candidates: Sequence[str],
    needs_dynamic_features: bool,
    use_seasonality: bool,
) -> BayesianStage1ExecutionProfile:
    entry = combo_payload(feature_profile_json, interval=interval, horizon=horizon, task=task)
    selected_features = tuple(str(value) for value in (entry.get("selected_features") or []) if str(value))
    selected_feature_blocks = tuple(str(value) for value in (entry.get("selected_feature_blocks") or []) if str(value))
    selected_formulation = dict(entry.get("selected_formulation") or {})
    explicit_dynamic_columns = tuple(str(value) for value in (entry.get("selected_dynamic_feature_columns") or []) if str(value))
    if explicit_dynamic_columns:
        selected_dynamic_feature_columns = explicit_dynamic_columns
    else:
        selected_dynamic_feature_columns = _fallback_dynamic_columns(selected_features, dynamic_feature_candidates)
    seasonality_mode = str(selected_formulation.get("seasonality_mode", "")).strip().lower()
    seasonality_enabled = bool(use_seasonality)
    if seasonality_mode:
        seasonality_enabled = seasonality_enabled and seasonality_mode not in {"none", "off", "disabled", "no_seasonality"}
    return BayesianStage1ExecutionProfile(
        selection_semantics=str(entry.get("selection_semantics") or ""),
        selected_feature_blocks=selected_feature_blocks,
        selected_features=selected_features,
        selected_formulation=selected_formulation,
        selected_dynamic_feature_columns=selected_dynamic_feature_columns,
        use_dynamic_features=bool(needs_dynamic_features and selected_dynamic_feature_columns),
        use_seasonality=bool(seasonality_enabled),
    )
