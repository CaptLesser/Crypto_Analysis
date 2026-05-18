from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.ml.shared.feature_profile_common import combo_selection_key
from src.forecasting.ml.shared.stage1_candidate_universe import build_stage1_candidate_universe
from src.forecasting.ml.shared.stage1_dynamic_feature_selection import select_stage1_dynamic_feature_columns
from src.forecasting.ml.neural.shared.neural_numeric_cohort import resolve_neural_cohort_assets
from src.forecasting.ml.neural.shared.neural_numeric_model_registry import NEURAL_NUMERIC_BRANCHES
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path


@dataclass(frozen=True)
class NeuralStage1Spec:
    model_key: str
    module_tag: str
    model_id: str
    default_intervals: Sequence[int]
    default_horizons: Sequence[int]
    default_tasks: Sequence[str]
    stage1_mode: str
    stage1_feature_blocks: Dict[str, Sequence[str]]
    stage1_formulation_options: Dict[str, Sequence[str]]
    default_combo_specs: Sequence[Tuple[int, int, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural numerics Stage 1 feature experiment")
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


def _resolved_source_roots(parquet_root: Path) -> Dict[str, str]:
    root = Path(parquet_root).expanduser().resolve()
    return {
        "parquet_root": str(root),
        "ohlcvt_root": str(root),
        "scalar_feature_root": str(root),
        "edge_discovery_root": str(root),
        "target_label_root": str(root),
    }


def _parse_int_csv(raw: str) -> List[int]:
    values = _split_csv(raw)
    return [int(v) for v in values]


def _parse_combo_list(raw: str) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    for token in _split_csv(raw):
        interval, horizon, task = token.split(":", 2)
        combos.append((int(interval), int(horizon), str(task)))
    return combos


def _load_stage1_spec(model_key: str) -> NeuralStage1Spec:
    module = importlib.import_module(f"src.forecasting.ml.neural.{model_key}.numerics")
    profiles = importlib.import_module(f"src.forecasting.ml.neural.{model_key}.numeric_profiles")
    spec = module.MODULE_SPEC
    return NeuralStage1Spec(
        model_key=str(model_key),
        module_tag=str(spec.module_tag),
        model_id=str(spec.model_id),
        default_intervals=tuple(int(v) for v in spec.default_intervals),
        default_horizons=tuple(int(v) for v in spec.default_horizons),
        default_tasks=tuple(str(v) for v in spec.default_tasks),
        stage1_mode=str(getattr(profiles, "STAGE1_MODE", "full")),
        stage1_feature_blocks={str(k): tuple(str(v) for v in values) for k, values in dict(getattr(profiles, "STAGE1_FEATURE_BLOCKS", {})).items()},
        stage1_formulation_options={str(k): tuple(str(v) for v in values) for k, values in dict(getattr(profiles, "STAGE1_FORMULATION_OPTIONS", {})).items()},
        default_combo_specs=tuple((int(i), int(h), str(t)) for i, h, t in profiles.resolve_default_combo_specs()),
    )


def resolved_combo_universe(stage1_spec: NeuralStage1Spec, *, intervals: Sequence[int], tasks: Sequence[str], horizons: Sequence[int], combo_list: Sequence[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
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


def _default_feature_blocks(stage1_spec: NeuralStage1Spec) -> Dict[str, List[str]]:
    return {str(name): [str(v) for v in values if str(v)] for name, values in stage1_spec.stage1_feature_blocks.items()}


def _default_formulation_options(stage1_spec: NeuralStage1Spec) -> Dict[str, List[str]]:
    return {str(name): [str(v) for v in values if str(v)] for name, values in stage1_spec.stage1_formulation_options.items()}


def _structural_stage1_contract(stage1_spec: NeuralStage1Spec) -> Dict[str, Any]:
    if str(stage1_spec.model_key) == "nbeats":
        return {
            "scalar_feature_search_performed": False,
            "scalar_feature_search_reason": "N-BEATS is configured as target-history-only backcast/forecast; scalar exogenous columns are not part of its active input contract.",
            "stage1_decision_basis": {
                "kind": "fixed_default_recorded",
                "data_derived_in_stage1": False,
                "deferred_to": ["stage2_validation", "stage3_validation"],
                "note": "Stage 1 records the active target-history-only input schema and cohort/combo scope; model-quality selection is deferred to later validation stages.",
            },
            "stage1_selected_instead": ["target_history_only", "input_schema", "lookback_design"],
            "candidates_options_considered": _default_formulation_options(stage1_spec),
            "model_specific_stage1_intent": {
                "target_history_only": "confirm univariate target-history input path",
                "input_schema": "choose the univariate backcast/forecast schema",
                "lookback_design": "choose the basis-aligned lookback/input schema family",
            },
            "data_derived_evidence_used": {
                "evidence_kind": "cohort_and_combo_scope_only",
                "note": "Slim Stage 1 records the univariate input contract; model performance is validated downstream, not through scalar feature search.",
            },
        }
    return {
        "scalar_feature_search_performed": True,
        "scalar_feature_search_reason": "dynamic NeuralTS model searches scalar/raw sequence inputs",
        "stage1_selected_instead": ["dynamic_sequence_feature_subset"],
        "candidates_options_considered": _default_formulation_options(stage1_spec),
        "model_specific_stage1_intent": {"dynamic_features": "select sequence-capable scalar/raw inputs for this task/interval/horizon"},
        "data_derived_evidence_used": {"evidence_kind": "dynamic_feature_relationship_scores"},
        "stage1_decision_basis": {
            "kind": "data_derived_feature_relationship_scores",
            "data_derived_in_stage1": True,
            "deferred_to": [],
            "note": "Dynamic NeuralTS Stage 1 scores scalar/raw sequence inputs for the requested task/interval/horizon.",
        },
    }


def _resolve_stage1_payload(stage1_spec: NeuralStage1Spec) -> Dict[str, Any]:
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
            **_structural_stage1_contract(stage1_spec),
        }
    formulation_choice = {name: values[0] for name, values in formulation_options.items() if values}
    return {
        "selection_semantics": "input_schema",
        "feature_blocks": feature_blocks,
        "selected_feature_blocks": [name for name, values in feature_blocks.items() if values],
        "selected_features": ["target_history"],
        "formulation_options": formulation_options,
        "selected_formulation": formulation_choice,
        **_structural_stage1_contract(stage1_spec),
    }


def _cohort_assets(raw_assets: str, asset_count: int, *, parquet_root: Path, intervals: Sequence[int]) -> List[str]:
    assets = [asset for asset in _split_csv(raw_assets)]
    if assets:
        return assets
    return resolve_neural_cohort_assets(
        parquet_root=Path(parquet_root).resolve(),
        intervals=intervals,
        asset_count=int(asset_count),
        explicit_assets=(),
        seed=17,
    )


def main_for_model(model_key: str) -> Path:
    if str(model_key) not in NEURAL_NUMERIC_BRANCHES:
        raise ValueError(f"Unsupported Neural model key: {model_key}")
    args = parse_args()
    args.parquet_root = Path(args.parquet_root).expanduser().resolve()
    source_roots = _resolved_source_roots(args.parquet_root)
    stage1_spec = _load_stage1_spec(str(model_key))
    module = importlib.import_module(f"src.forecasting.ml.neural.{model_key}.numerics")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intervals = _parse_int_csv(args.intervals)
    tasks = _split_csv(args.tasks)
    horizons = _parse_int_csv(getattr(args, "horizon_minutes", ""))
    combo_list = _parse_combo_list(str(args.combo_list))
    combos = resolved_combo_universe(stage1_spec, intervals=intervals, tasks=tasks, horizons=horizons, combo_list=combo_list)

    cohort_assets = _cohort_assets(
        str(args.assets),
        int(args.asset_count),
        parquet_root=Path(args.parquet_root),
        intervals=sorted({int(interval) for interval, _, _ in combos}),
    )
    emit_event_for_path(
        output_dir,
        family="Neural_Numeric",
        model=str(model_key),
        stage="stage1",
        function_name="_cohort_assets",
        module_name=__name__,
        phase_name="asset_planning",
        status="completed",
        asset_count=len(cohort_assets),
        output_rows=len(cohort_assets),
        reason_code=("no_assets" if not cohort_assets else ""),
        source_path=str(args.parquet_root),
        **source_roots,
    )
    generated_at = datetime.now(timezone.utc).isoformat()

    selections: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for interval_minutes, horizon_minutes, task in combos:
        payload_bits = _resolve_stage1_payload(stage1_spec)
        selected_dynamic_feature_columns: List[str] = []
        selected_features = list(payload_bits["selected_features"])
        candidate_universe = None
        dynamic_selection_report = None
        if bool(getattr(module.MODULE_SPEC, "needs_dynamic_features", False)):
            candidate_universe = build_stage1_candidate_universe(
                model_family="neural_numeric",
                model_key=str(model_key),
                preferred_feature_names=tuple(str(value) for value in getattr(module.MODULE_SPEC, "dynamic_feature_candidates", ())),
                feature_blocks=stage1_spec.stage1_feature_blocks,
                include_raw_source=True,
            )
            selection_result = select_stage1_dynamic_feature_columns(
                parquet_root=Path(args.parquet_root).resolve(),
                asset_list=cohort_assets,
                interval_minutes=int(interval_minutes),
                horizon_minutes=int(horizon_minutes),
                task=str(task),
                training_window_months=int(args.train_window_months),
                requested_feature_names=candidate_universe.candidate_columns,
                telemetry_path=output_dir,
                family="Neural_Numeric",
                model=str(model_key),
                stage="stage1",
                combo_key=combo_selection_key(int(interval_minutes), int(horizon_minutes), str(task)),
                return_report=True,
            )
            if hasattr(selection_result, "selected_features"):
                dynamic_selection_report = selection_result
                selected_dynamic_feature_columns = [str(value) for value in getattr(selection_result, "selected_features")]
            else:
                selected_dynamic_feature_columns = [str(value) for value in selection_result]
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
            "feature_profile": ("feature_block_profile" if stage1_spec.stage1_mode == "full" else "input_schema_profile"),
            "feature_blocks": dict(payload_bits["feature_blocks"]),
            "selected_feature_blocks": list(payload_bits["selected_feature_blocks"]),
            "selected_features": list(selected_features),
            "selected_dynamic_feature_columns": list(selected_dynamic_feature_columns),
            "formulation_options": dict(payload_bits["formulation_options"]),
            "selected_formulation": dict(payload_bits["selected_formulation"]),
            "scalar_feature_search_performed": bool(payload_bits.get("scalar_feature_search_performed", False)),
            "scalar_feature_search_reason": str(payload_bits.get("scalar_feature_search_reason", "")),
            "stage1_selected_instead": list(payload_bits.get("stage1_selected_instead") or []),
            "candidates_options_considered": dict(payload_bits.get("candidates_options_considered") or payload_bits.get("formulation_options") or {}),
            "stage1_decision_basis": dict(payload_bits.get("stage1_decision_basis") or {}),
            "data_derived_evidence_used": {
                **dict(payload_bits.get("data_derived_evidence_used") or {}),
                "asset_count": int(len(cohort_assets)),
                "training_window_months": int(args.train_window_months),
                "combo_key": combo_selection_key(int(interval_minutes), int(horizon_minutes), str(task)),
            },
            "final_selected_formulation_settings": dict(payload_bits["selected_formulation"]),
            "model_specific_stage1_intent": dict(payload_bits.get("model_specific_stage1_intent") or {}),
            "asset_count_used": int(len(cohort_assets)),
            "cohort_assets": list(cohort_assets),
            "training_window_months": int(args.train_window_months),
            "selection_status": "complete",
            "resolved_roots": source_roots,
        }
        if candidate_universe is not None:
            entry["candidate_universe"] = candidate_universe.to_artifact()
            entry["stale_or_missing_candidates"] = list(candidate_universe.stale_or_missing_candidates)
            entry["alias_resolutions"] = list(candidate_universe.alias_resolutions)
        if dynamic_selection_report is not None:
            report_payload = dynamic_selection_report.to_artifact()
            entry["dynamic_selection_report"] = report_payload
            entry["dynamic_feature_scores"] = list(report_payload.get("feature_scores") or [])
            entry["dynamic_dropped_candidates"] = list(report_payload.get("dropped_candidates") or [])
            entry["dynamic_redundancy_groups"] = list(report_payload.get("redundancy_groups") or [])
        selections[combo_key] = entry
        summary_rows.append(entry)

    payload = {
        "selection_file_version": 2,
        "family": "neural_numeric",
        "model_key": str(model_key),
        "model_id": str(stage1_spec.model_id),
        "stage1_mode": str(stage1_spec.stage1_mode),
        "generated_at": generated_at,
        "intervals": sorted({int(interval) for interval, _, _ in combos}),
        "tasks": sorted({str(task) for _, _, task in combos}),
        "horizon_minutes": sorted({int(horizon) for _, horizon, _ in combos}),
        "cohort_assets": list(cohort_assets),
        "resolved_roots": source_roots,
        "selections": selections,
    }
    (output_dir / "feature_profile_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    emit_event_for_path(
        output_dir,
        family="Neural_Numeric",
        model=str(model_key),
        stage="stage1",
        function_name="write_feature_profile_selection",
        module_name=__name__,
        phase_name="artifact_handoff",
        status="completed",
        asset_count=len(cohort_assets),
        output_rows=len(selections),
        output_path=str(output_dir / "feature_profile_selection.json"),
    )

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
        "resolved_roots": source_roots,
        "generated_at": generated_at,
    }
    (output_dir / "feature_experiment_progress.json").write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")

    meta_payload = {
        "family": "neural_numeric",
        "model_key": str(model_key),
        "model_id": str(stage1_spec.model_id),
        "stage1_mode": str(stage1_spec.stage1_mode),
        "expected_combo_count": int(len(combos)),
        "completed_combo_count": int(len(combos)),
        "missing_combo_keys": [],
        "cohort_assets": list(cohort_assets),
        "training_window_months": int(args.train_window_months),
        "resolved_roots": source_roots,
        "generated_at": generated_at,
    }
    (output_dir / "feature_experiment_run_meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    emit_event_for_path(
        output_dir,
        family="Neural_Numeric",
        model=str(model_key),
        stage="stage1",
        function_name="write_feature_experiment_artifacts",
        module_name=__name__,
        phase_name="write",
        status="completed",
        output_rows=len(summary_rows),
        output_path=str(output_dir),
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-key", type=str, required=True)
    known_args, _ = parser.parse_known_args()
    main_for_model(str(known_args.model_key))


if __name__ == "__main__":
    main()


