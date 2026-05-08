from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence


def _combo_key(interval: int, horizon: int, task: str) -> str:
    return f"interval={int(interval)}|horizon={int(horizon)}|task={str(task)}"


@dataclass(frozen=True)
class NeuralStage1ExecutionProfile:
    selection_semantics: str
    selected_feature_blocks: tuple[str, ...]
    selected_features: tuple[str, ...]
    selected_formulation: Dict[str, Any]
    selected_dynamic_feature_columns: tuple[str, ...]
    use_dynamic_features: bool


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


def resolve_execution_profile(
    feature_profile_json: Path | Dict[str, Any],
    *,
    interval: int,
    horizon: int,
    task: str,
    dynamic_feature_candidates: Sequence[str],
    needs_dynamic_features: bool,
) -> NeuralStage1ExecutionProfile:
    entry = combo_payload(feature_profile_json, interval=interval, horizon=horizon, task=task)
    selected_features = tuple(str(value) for value in (entry.get("selected_features") or []) if str(value))
    selected_feature_blocks = tuple(str(value) for value in (entry.get("selected_feature_blocks") or []) if str(value))
    selected_formulation = dict(entry.get("selected_formulation") or {})
    explicit_dynamic_columns = tuple(str(value) for value in (entry.get("selected_dynamic_feature_columns") or []) if str(value))
    if explicit_dynamic_columns:
        selected_dynamic_feature_columns = explicit_dynamic_columns
    else:
        dynamic_candidates = {str(value) for value in dynamic_feature_candidates}
        selected_dynamic_feature_columns = tuple(
            str(value) for value in selected_features if str(value) in dynamic_candidates
        )
    return NeuralStage1ExecutionProfile(
        selection_semantics=str(entry.get("selection_semantics") or ""),
        selected_feature_blocks=selected_feature_blocks,
        selected_features=selected_features,
        selected_formulation=selected_formulation,
        selected_dynamic_feature_columns=selected_dynamic_feature_columns,
        use_dynamic_features=bool(needs_dynamic_features and selected_dynamic_feature_columns),
    )
