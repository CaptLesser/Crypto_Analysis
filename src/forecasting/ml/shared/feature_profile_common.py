from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set

FEATURE_SELECTION_FILE_ENV = "TABULAR_NUMERIC_FEATURE_SELECTION_FILE"


def combo_selection_key(interval_minutes: int, horizon_minutes: int, task: str) -> str:
    return f"interval={int(interval_minutes)}|horizon={int(horizon_minutes)}|task={str(task)}"


@lru_cache(maxsize=8)
def _load_payload(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    return json.loads(path.read_text(encoding="utf-8"))


def _active_payload() -> Dict[str, Any] | None:
    raw = os.getenv(FEATURE_SELECTION_FILE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Feature-selection file not found: {path}")
    return _load_payload(str(path.resolve()))


def resolve_feature_profile_entry(interval_minutes: int, horizon_minutes: int, task: str) -> Dict[str, Any] | None:
    payload = _active_payload()
    if not payload:
        return None
    selections = payload.get("selections") or {}
    entry = selections.get(combo_selection_key(interval_minutes, horizon_minutes, task))
    return dict(entry) if isinstance(entry, dict) else None


def resolve_selected_feature_columns(
    *,
    columns: Sequence[str],
    task: str,
    horizon_minutes: int,
    interval_minutes: int,
    extra_exclude: Set[str],
    fallback_fn: Callable[[Sequence[str], Set[str]], List[str]],
) -> List[str]:
    fallback = list(fallback_fn(columns, set(extra_exclude)))
    entry = resolve_feature_profile_entry(int(interval_minutes), int(horizon_minutes), str(task))
    if not entry:
        return fallback
    configured = [str(value) for value in (entry.get("selected_features") or []) if str(value)]
    if not configured:
        return fallback
    available = [col for col in configured if col in set(columns) and col not in set(extra_exclude)]
    return available or fallback


def resolve_feature_profile_label(default_profile: str, *, task: str, horizon_minutes: int, interval_minutes: int) -> str:
    entry = resolve_feature_profile_entry(int(interval_minutes), int(horizon_minutes), str(task))
    if not entry:
        return str(default_profile)
    label = str(entry.get("feature_profile") or "").strip()
    return label or str(default_profile)
