from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.ml.bayesian.shared.bayesian_numeric_cohort import resolve_bayesian_cohort_assets
from src.forecasting.ml.bayesian.shared.bayesian_numeric_model_registry import BAYESIAN_NUMERIC_BRANCHES
from src.forecasting.ml.shared.stage1_dynamic_feature_selection import select_stage1_dynamic_feature_columns
from src.forecasting.ml.shared.feature_profile_common import combo_selection_key


@dataclass(frozen=True)
class BayesianStage1Spec:
    model_key: str
    module_tag: str
    model_id: str
    default_intervals: Sequence[int]
    default_horizons: Sequence[int]
    default_tasks: Sequence[str]
    use_seasonality: bool
    needs_dynamic_features: bool
    needs_factor_cache: bool
    dynamic_feature_candidates: Sequence[str]
    stage1_mode: str
    stage1_feature_blocks: Dict[str, Sequence[str]]
    stage1_formulation_options: Dict[str, Sequence[str]]
    default_combo_specs: Sequence[Tuple[int, int, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian numerics Stage 1 feature experiment")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervals", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--horizon-minutes", type=str, default="")
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--asset-count", type=int, default=8)
    parser.add_argument("--train-window-months", type=int, default=6)
    args = parser.parse_args()
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return args


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_int_csv(raw: str, default: Sequence[int]) -> List[int]:
    values = _split_csv(raw)
    if not values:
        return [int(v) for v in default]
    return [int(v) for v in values]


def _parse_str_csv(raw: str, default: Sequence[str]) -> List[str]:
    values = _split_csv(raw)
    if not values:
        return [str(v) for v in default]
    return [str(v) for v in values]


def _parse_combo_list(raw: str) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    for token in _split_csv(raw):
        interval, horizon, task = token.split(":", 2)
        combos.append((int(interval), int(horizon), str(task)))
    return combos


def _load_stage1_spec(model_key: str) -> BayesianStage1Spec:
    module = importlib.import_module(f"src.forecasting.ml.bayesian.{model_key}.numerics")
    profiles = importlib.import_module(f"src.forecasting.ml.bayesian.{model_key}.numeric_profiles")
    spec = module.MODULE_SPEC
    return BayesianStage1Spec(
        model_key=str(model_key),
        module_tag=str(spec.module_tag),
        model_id=str(spec.model_id),
        default_intervals=tuple(int(v) for v in spec.default_intervals),
        default_horizons=tuple(int(v) for v in spec.default_horizons),
        default_tasks=tuple(str(v) for v in spec.default_tasks),
        use_seasonality=bool(spec.use_seasonality),
        needs_dynamic_features=bool(spec.needs_dynamic_features),
        needs_factor_cache=bool(spec.needs_factor_cache),
        dynamic_feature_candidates=tuple(str(v) for v in spec.dynamic_feature_candidates),
        stage1_mode=str(getattr(profiles, "STAGE1_MODE", "full")),
        stage1_feature_blocks={str(k): tuple(str(v) for v in values) for k, values in dict(getattr(profiles, "STAGE1_FEATURE_BLOCKS", {})).items()},
        stage1_formulation_options={str(k): tuple(str(v) for v in values) for k, values in dict(getattr(profiles, "STAGE1_FORMULATION_OPTIONS", {})).items()},
        default_combo_specs=tuple((int(i), int(h), str(t)) for i, h, t in profiles.resolve_default_combo_specs()),
    )


def resolved_combo_universe(
    stage1_spec: BayesianStage1Spec,
    *,
    intervals: Sequence[int],
    tasks: Sequence[str],
    horizons: Sequence[int],
    combo_list: Sequence[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    if combo_list:
        combos = [(int(i), int(h), str(t)) for i, h, t in combo_list]
    else:
        combos = list(stage1_spec.default_combo_specs)
        if intervals:
            allowed_intervals = {int(v) for v in intervals}
            combos = [combo for combo in combos if int(combo[0]) in allowed_intervals]
        if tasks:
            allowed_tasks = {str(v) for v in tasks}
            combos = [combo for combo in combos if str(combo[2]) in allowed_tasks]
        if horizons:
            allowed_horizons = {int(v) for v in horizons}
            combos = [combo for combo in combos if int(combo[1]) in allowed_horizons]
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def _default_feature_blocks(stage1_spec: BayesianStage1Spec) -> Dict[str, List[str]]:
    if stage1_spec.stage1_feature_blocks:
        return {str(name): [str(v) for v in values if str(v)] for name, values in stage1_spec.stage1_feature_blocks.items()}
    if stage1_spec.needs_dynamic_features:
        return {"dynamic_feature_subset": [str(value) for value in stage1_spec.dynamic_feature_candidates if str(value)] or ["target_history"]}
    if stage1_spec.needs_factor_cache:
        return {"factor_augmented_history": ["target_history", "market_factor"]}
    if stage1_spec.use_seasonality:
        return {"seasonal_state_history": ["target_history", "seasonality_state"]}
    return {"target_history": ["target_history"]}


def _default_formulation_options(stage1_spec: BayesianStage1Spec) -> Dict[str, List[str]]:
    if stage1_spec.stage1_formulation_options:
        return {str(name): [str(v) for v in values if str(v)] for name, values in stage1_spec.stage1_formulation_options.items()}
    if stage1_spec.model_key == "stochastic_vol":
        return {"target_schema": ["volatility_state_history"]}
    if stage1_spec.model_key == "tail_risk":
        return {"tail_schema": ["tail_state_history"]}
    return {"history_schema": ["target_history"]}


def _resolve_stage1_payload(stage1_spec: BayesianStage1Spec) -> Dict[str, Any]:
    feature_blocks = _default_feature_blocks(stage1_spec)
    formulation_options = _default_formulation_options(stage1_spec)
    if stage1_spec.stage1_mode == "full":
        selected_block_names = [name for name, values in feature_blocks.items() if values]
        selected_features: List[str] = []
        for values in feature_blocks.values():
            for value in values:
                if value not in selected_features:
                    selected_features.append(str(value))
        formulation_choice = {name: values[0] for name, values in formulation_options.items() if values}
        return {
            "selection_semantics": "feature_relationships",
            "feature_blocks": feature_blocks,
            "selected_feature_blocks": selected_block_names,
            "selected_features": selected_features or ["target_history"],
            "formulation_options": formulation_options,
            "selected_formulation": formulation_choice,
        }
    formulation_choice = {name: values[0] for name, values in formulation_options.items() if values}
    selected_features = ["target_history"]
    if stage1_spec.needs_factor_cache:
        selected_features.append("market_factor")
    if stage1_spec.use_seasonality:
        selected_features.append("seasonality_state")
    return {
        "selection_semantics": "structural_formulation",
        "feature_blocks": feature_blocks,
        "selected_feature_blocks": [name for name, values in feature_blocks.items() if values],
        "selected_features": selected_features,
        "formulation_options": formulation_options,
        "selected_formulation": formulation_choice,
    }


def _cohort_assets(*, parquet_root: Path, intervals: Sequence[int], raw_assets: str, asset_count: int, seed: int = 17) -> List[str]:
    return resolve_bayesian_cohort_assets(
        parquet_root=Path(parquet_root).resolve(),
        intervals=intervals,
        asset_count=int(asset_count),
        explicit_assets=_split_csv(raw_assets),
        seed=int(seed),
    )


def main_for_model(model_key: str) -> Path:
    if str(model_key) not in BAYESIAN_NUMERIC_BRANCHES:
        raise ValueError(f"Unsupported Bayesian model key: {model_key}")
    args = parse_args()
    stage1_spec = _load_stage1_spec(str(model_key))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intervals = _parse_int_csv(args.intervals, stage1_spec.default_intervals)
    tasks = _parse_str_csv(args.tasks, stage1_spec.default_tasks)
    horizons = _parse_int_csv(getattr(args, "horizon_minutes", ""), stage1_spec.default_horizons)
    combo_list = _parse_combo_list(str(args.combo_list))
    combos = resolved_combo_universe(stage1_spec, intervals=intervals, tasks=tasks, horizons=horizons, combo_list=combo_list)

    cohort_assets = _cohort_assets(
        parquet_root=Path(args.parquet_root),
        intervals=intervals,
        raw_assets=str(args.assets),
        asset_count=int(args.asset_count),
        seed=17,
    )
    generated_at = datetime.now(timezone.utc).isoformat()

    selections: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for interval_minutes, horizon_minutes, task in combos:
        payload_bits = _resolve_stage1_payload(stage1_spec)
        selected_dynamic_feature_columns: List[str] = []
        selected_features = list(payload_bits["selected_features"])
        if bool(stage1_spec.needs_dynamic_features) and stage1_spec.dynamic_feature_candidates:
            selected_dynamic_feature_columns = select_stage1_dynamic_feature_columns(
                parquet_root=Path(args.parquet_root).resolve(),
                asset_list=cohort_assets,
                interval_minutes=int(interval_minutes),
                horizon_minutes=int(horizon_minutes),
                task=str(task),
                training_window_months=int(args.train_window_months),
                requested_feature_names=stage1_spec.dynamic_feature_candidates,
            )
            if selected_dynamic_feature_columns:
                selected_features = list(selected_dynamic_feature_columns)
        combo_key = combo_selection_key(int(interval_minutes), int(horizon_minutes), str(task))
        entry = {
            "model_key": str(model_key),
            "model_id": str(stage1_spec.model_id),
            "interval_minutes": int(interval_minutes),
            "horizon_minutes": int(horizon_minutes),
            "task": str(task),
            "stage1_mode": str(stage1_spec.stage1_mode),
            "selection_semantics": str(payload_bits["selection_semantics"]),
            "feature_profile": ("feature_block_profile" if stage1_spec.stage1_mode == "full" else "formulation_profile"),
            "feature_blocks": dict(payload_bits["feature_blocks"]),
            "selected_feature_blocks": list(payload_bits["selected_feature_blocks"]),
            "selected_features": list(selected_features),
            "selected_dynamic_feature_columns": list(selected_dynamic_feature_columns),
            "formulation_options": dict(payload_bits["formulation_options"]),
            "selected_formulation": dict(payload_bits["selected_formulation"]),
            "asset_count_used": int(len(cohort_assets)),
            "cohort_assets": list(cohort_assets),
            "training_window_months": int(args.train_window_months),
            "needs_dynamic_features": bool(stage1_spec.needs_dynamic_features),
            "needs_factor_cache": bool(stage1_spec.needs_factor_cache),
            "uses_seasonality": bool(stage1_spec.use_seasonality),
            "selection_status": "complete",
        }
        selections[combo_key] = entry
        summary_rows.append(entry)

    payload = {
        "selection_file_version": 2,
        "family": "bayesian_numeric",
        "model_key": str(model_key),
        "model_id": str(stage1_spec.model_id),
        "stage1_mode": str(stage1_spec.stage1_mode),
        "generated_at": generated_at,
        "intervals": [int(v) for v in intervals],
        "tasks": [str(v) for v in tasks],
        "horizon_minutes": [int(v) for v in horizons],
        "cohort_assets": list(cohort_assets),
        "selections": selections,
    }
    (output_dir / "feature_profile_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_csv_rows = [
        {
            "model_key": row["model_key"],
            "interval_minutes": row["interval_minutes"],
            "horizon_minutes": row["horizon_minutes"],
            "task": row["task"],
            "stage1_mode": row["stage1_mode"],
            "selection_semantics": row["selection_semantics"],
            "feature_profile": row["feature_profile"],
            "selected_feature_count": len(row["selected_features"]),
            "selected_dynamic_feature_count": len(row.get("selected_dynamic_feature_columns") or []),
            "selected_formulation_count": len(row["selected_formulation"]),
            "asset_count_used": row["asset_count_used"],
            "training_window_months": row["training_window_months"],
        }
        for row in summary_rows
    ]
    if summary_csv_rows:
        import pandas as pd

        pd.DataFrame(summary_csv_rows).to_csv(output_dir / "feature_experiment_summary.csv", index=False)
    else:
        (output_dir / "feature_experiment_summary.csv").write_text("", encoding="utf-8")

    progress_payload = {
        "model_key": str(model_key),
        "stage1_mode": str(stage1_spec.stage1_mode),
        "status": "completed",
        "expected_combo_count": int(len(combos)),
        "completed_combo_count": int(len(combos)),
        "missing_combo_keys": [],
        "generated_at": generated_at,
    }
    (output_dir / "feature_experiment_progress.json").write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")

    meta_payload = {
        "family": "bayesian_numeric",
        "model_key": str(model_key),
        "model_id": str(stage1_spec.model_id),
        "stage1_mode": str(stage1_spec.stage1_mode),
        "expected_combo_count": int(len(combos)),
        "completed_combo_count": int(len(combos)),
        "missing_combo_keys": [],
        "cohort_assets": list(cohort_assets),
        "training_window_months": int(args.train_window_months),
        "generated_at": generated_at,
    }
    (output_dir / "feature_experiment_run_meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-key", type=str, required=True)
    known_args, _ = parser.parse_known_args()
    main_for_model(str(known_args.model_key))


if __name__ == "__main__":
    main()
