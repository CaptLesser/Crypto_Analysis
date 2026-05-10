from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.ml.shared.numeric_runner_common import load_stage3_combo_results

Combo = Tuple[int, int, str]


def fallback_combo_specs(default_intervals: Sequence[int], default_horizons: Sequence[int], default_tasks: Sequence[str]) -> List[Combo]:
    combos: List[Combo] = []
    for interval in default_intervals:
        for horizon in default_horizons:
            for task in default_tasks:
                if int(interval) > 0 and int(horizon) > 0 and int(horizon) % int(interval) == 0:
                    combos.append((int(interval), int(horizon), str(task)))
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def diagnostics_root(*, diagnostics_root_name: str, model_key: str) -> Path:
    raw_root = str(os.getenv("PIPELINE_TEST_BRANCH_PROFILE_ROOT") or os.getenv("PIPELINE_SANDBOX_DIAGNOSTICS_ROOT") or "").strip()
    root = Path(raw_root) if raw_root else Path.cwd() / "logs" / "diagnostics"
    return (root / str(diagnostics_root_name) / str(model_key)).resolve()


def latest_combo_results_path(*, diagnostics_root_name: str, model_key: str) -> Optional[Path]:
    direct_root = diagnostics_root(diagnostics_root_name=diagnostics_root_name, model_key=model_key)
    orchestrator_root = (Path.cwd() / "logs" / "diagnostics" / str(diagnostics_root_name)).resolve()
    candidates: List[Path] = []
    canonical_combo = direct_root / "stage3" / "combo_results.csv"
    if canonical_combo.is_file():
        return canonical_combo.resolve()
    if direct_root.exists():
        candidates.extend(path for path in direct_root.rglob("combo_results.csv") if path.is_file())
    cwd_direct_root = (orchestrator_root / str(model_key)).resolve()
    if cwd_direct_root != direct_root and cwd_direct_root.exists():
        candidates.extend(path for path in cwd_direct_root.rglob("combo_results.csv") if path.is_file())
    if orchestrator_root.exists():
        candidates.extend(
            path
            for path in orchestrator_root.glob(f"run=*/{str(model_key)}/stage3/combo_results.csv")
            if path.is_file()
        )
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


@lru_cache(maxsize=64)
def load_latest_stage3_profiles(*, diagnostics_root_name: str, model_key: str) -> Dict[str, Any]:
    combo_results = latest_combo_results_path(diagnostics_root_name=diagnostics_root_name, model_key=model_key)
    if combo_results is None:
        return {"source": None, "combos": [], "params": {}}
    combos, params = load_stage3_combo_results(combo_results)
    return {
        "source": str(combo_results),
        "combos": list(combos),
        "params": params,
    }


def resolve_default_combo_specs(
    *,
    diagnostics_root_name: str,
    model_key: str,
    default_intervals: Sequence[int],
    default_horizons: Sequence[int],
    default_tasks: Sequence[str],
    fallback_combos: Optional[Sequence[Combo]] = None,
) -> List[Combo]:
    payload = load_latest_stage3_profiles(diagnostics_root_name=diagnostics_root_name, model_key=str(model_key))
    combos = list(payload.get("combos") or [])
    if combos:
        return combos
    if fallback_combos:
        return sorted({(int(i), int(h), str(t)) for i, h, t in fallback_combos}, key=lambda item: (item[0], item[1], item[2]))
    return fallback_combo_specs(default_intervals, default_horizons, default_tasks)


def resolve_model_params(
    *,
    diagnostics_root_name: str,
    model_key: str,
    baseline_params: Dict[str, Any],
    interval_minutes: Optional[int],
    horizon_minutes: Optional[int],
    task: Optional[str],
) -> Dict[str, Any]:
    params = dict(baseline_params)
    if interval_minutes is None or horizon_minutes is None or not task:
        return params
    payload = load_latest_stage3_profiles(diagnostics_root_name=diagnostics_root_name, model_key=str(model_key))
    discovered = payload.get("params") or {}
    combo = (int(interval_minutes), int(horizon_minutes), str(task))
    params.update(dict(discovered.get(combo) or {}))
    return params
