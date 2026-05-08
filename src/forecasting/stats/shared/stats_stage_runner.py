from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.stats.shared.stats_numeric_model_registry import (
    STATS_NUMERIC_ENTRYPOINTS,
    STATS_NUMERIC_FAMILY_ROOT_ENVS,
    STATS_NUMERIC_FAMILY_ROOT_NAMES,
)

STATS_MANIFEST_FILES = {
    "sarimax": "sarimax_run_manifest.json",
    "llt": "llt_run_manifest.json",
    "egarch": "egarch_run_manifest.json",
    "quantreg": "quantreg_run_manifest.json",
}

STATS_SKIPPED_FILES = {
    "sarimax": "sarimax_skipped.json",
    "llt": "llt_skipped.json",
    "egarch": "egarch_skipped.json",
    "quantreg": "quantreg_skipped.json",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequentist stats staged branch runner")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backfill-days", type=int, default=14)
    parser.add_argument("--fit-days", type=int, default=180)
    parser.add_argument("--model-threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-latest-only", action="store_true")
    parser.add_argument("--fill-to-edge", action="store_true")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--feature-profile-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return args


def _combo_key(interval: int, horizon_minutes: int, task: str) -> str:
    return f"{int(interval)}:{int(horizon_minutes)}:{task}"


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _load_feature_profile(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    return load_json_dict(Path(path))


def _combos_from_profile(profile: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    selections = profile.get("selections")
    if not isinstance(selections, dict):
        return []
    combos: List[Tuple[int, int, str]] = []
    for item in selections.values():
        if not isinstance(item, dict):
            continue
        try:
            combos.append((int(item["interval_minutes"]), int(item["horizon_minutes"]), str(item["task"])))
        except Exception:
            continue
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def _combo_list_arg(combos: Sequence[Tuple[int, int, str]]) -> str:
    return ",".join(_combo_key(interval, horizon, task) for interval, horizon, task in combos)


def _unit_combo_from_ukey(ukey: str) -> Optional[Tuple[str, int, int, str]]:
    parts = str(ukey).split("|")
    if len(parts) < 6:
        return None
    task = str(parts[2])
    horizon = int(str(parts[3]).removesuffix("m"))
    asset = str(parts[4])
    interval = int(parts[5])
    return asset, interval, horizon, task


def _stage_combo_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for entry in list(manifest.get("unit_entries") or []):
        if not isinstance(entry, dict):
            continue
        parsed = _unit_combo_from_ukey(str(entry.get("ukey", "")))
        if parsed is None:
            continue
        asset, interval, horizon, task = parsed
        key = (int(interval), int(horizon), str(task))
        row = grouped.setdefault(
            key,
            {
                "interval_minutes": int(interval),
                "horizon_minutes": int(horizon),
                "task": str(task),
                "assets": set(),
                "forecast_rows": 0,
                "eval_rows": 0,
                "unit_count": 0,
                "done_units": 0,
                "converged_units": 0,
                "convergence_warning_count": 0,
                "nonconverged_fit_count": 0,
                "fit_retry_count": 0,
                "fit_retry_resolved_count": 0,
                "unit_elapsed_s": [],
                "selected_window_bars": [],
                "aic_values": [],
                "bic_values": [],
                "first_ts": None,
                "last_ts": None,
            },
        )
        row["assets"].add(str(asset))
        row["forecast_rows"] += int(entry.get("forecast_rows", 0) or 0)
        row["eval_rows"] += int(entry.get("eval_rows", 0) or 0)
        row["unit_count"] += 1
        if str(entry.get("status", "")) == "done":
            row["done_units"] += 1
        fit_meta = entry.get("fit_meta") if isinstance(entry.get("fit_meta"), dict) else {}
        if bool(fit_meta.get("converged", False)):
            row["converged_units"] += 1
        row["convergence_warning_count"] += int(fit_meta.get("convergence_warning_count", 0) or 0)
        row["nonconverged_fit_count"] += int(fit_meta.get("nonconverged_fit_count", 0) or 0)
        row["fit_retry_count"] += int(fit_meta.get("fit_retry_count", 0) or 0)
        row["fit_retry_resolved_count"] += int(fit_meta.get("fit_retry_resolved_count", 0) or 0)
        elapsed_s = entry.get("elapsed_s")
        if elapsed_s is not None:
            row["unit_elapsed_s"].append(elapsed_s)
        for metric_name, target_key in (("selected_window_bars", "selected_window_bars"), ("aic", "aic_values"), ("bic", "bic_values")):
            value = fit_meta.get(metric_name)
            if value is not None:
                row[target_key].append(value)
    for part in list(manifest.get("parts") or []):
        if not isinstance(part, dict) or str(part.get("store", "forecast")) != "forecast":
            continue
        try:
            key = (int(part["interval"]), int(part["horizon_minutes"]), str(part["task"]))
        except Exception:
            continue
        row = grouped.get(key)
        if row is None:
            continue
        min_ts = part.get("min_ts")
        max_ts = part.get("max_ts")
        if min_ts is not None:
            row["first_ts"] = int(min_ts) if row["first_ts"] is None else min(int(row["first_ts"]), int(min_ts))
        if max_ts is not None:
            row["last_ts"] = int(max_ts) if row["last_ts"] is None else max(int(row["last_ts"]), int(max_ts))
    out: List[Dict[str, Any]] = []
    for row in grouped.values():
        unit_count = max(1, int(row["unit_count"]))
        converged_rate = float(row["converged_units"]) / float(unit_count)
        skipped_origin_rate = 1.0 - (float(row["done_units"]) / float(unit_count))
        selected_windows = [int(v) for v in row["selected_window_bars"] if v is not None]
        unit_elapsed = [float(v) for v in row["unit_elapsed_s"] if v is not None]
        aic_values = [float(v) for v in row["aic_values"] if v is not None]
        bic_values = [float(v) for v in row["bic_values"] if v is not None]
        out.append(
            {
                "interval_minutes": int(row["interval_minutes"]),
                "horizon_minutes": int(row["horizon_minutes"]),
                "task": str(row["task"]),
                "assets": sorted(str(asset) for asset in row["assets"]),
                "forecast_rows": int(row["forecast_rows"]),
                "eval_rows": int(row["eval_rows"]),
                "unit_count": int(row["unit_count"]),
                "done_units": int(row["done_units"]),
                "convergence_rate": converged_rate,
                "convergence_warning_count": int(row["convergence_warning_count"]),
                "nonconverged_fit_count": int(row["nonconverged_fit_count"]),
                "fit_retry_count": int(row["fit_retry_count"]),
                "fit_retry_resolved_count": int(row["fit_retry_resolved_count"]),
                "unit_elapsed_s_total": round(float(sum(unit_elapsed)), 3) if unit_elapsed else None,
                "unit_elapsed_s_mean": round(float(sum(unit_elapsed) / len(unit_elapsed)), 3) if unit_elapsed else None,
                "unit_elapsed_s_max": round(float(max(unit_elapsed)), 3) if unit_elapsed else None,
                "skipped_origin_rate": skipped_origin_rate,
                "selected_window_bars": selected_windows,
                "median_selected_window_bars": int(sorted(selected_windows)[len(selected_windows) // 2]) if selected_windows else None,
                "mean_aic": (sum(aic_values) / len(aic_values)) if aic_values else None,
                "mean_bic": (sum(bic_values) / len(bic_values)) if bic_values else None,
                "first_prediction_ts": row["first_ts"],
                "last_prediction_ts": row["last_ts"],
                "status": "passed" if int(row["forecast_rows"]) > 0 and int(row["done_units"]) > 0 else "failed",
            }
        )
    return sorted(out, key=lambda item: (int(item["interval_minutes"]), int(item["horizon_minutes"]), str(item["task"])))


def _stage_eval_quality_rows(manifest: Dict[str, Any]) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    grouped_frames: Dict[Tuple[int, int, str], List[pd.DataFrame]] = {}
    eval_part_counts: Dict[Tuple[int, int, str], int] = {}
    eval_declared_rows: Dict[Tuple[int, int, str], int] = {}
    eval_expected_cols: Dict[Tuple[int, int, str], List[str]] = {}
    eval_paths: Dict[Tuple[int, int, str], List[str]] = {}
    for part in list(manifest.get("parts") or []):
        if not isinstance(part, dict) or str(part.get("store", "forecast")) != "eval":
            continue
        try:
            key = (int(part["interval"]), int(part["horizon_minutes"]), str(part["task"]))
            path = Path(str(part["path"]))
        except Exception:
            continue
        eval_part_counts[key] = int(eval_part_counts.get(key, 0)) + 1
        eval_paths.setdefault(key, []).append(str(path))
        expected_cols = [str(col) for col in list(part.get("expected_cols") or []) if str(col)]
        if expected_cols:
            merged_expected = eval_expected_cols.setdefault(key, [])
            for col in expected_cols:
                if col not in merged_expected:
                    merged_expected.append(col)
        try:
            eval_declared_rows[key] = int(eval_declared_rows.get(key, 0)) + int(part.get("rows", 0) or 0)
        except Exception:
            pass
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if not frame.empty:
            grouped_frames.setdefault(key, []).append(frame)
    out: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

    def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(dtype="float64")
        values = pd.to_numeric(frame[column], errors="coerce")
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values, dtype="float64")

    def _contract_metric_column(key: Tuple[int, int, str], generic: str) -> Optional[str]:
        expected_cols = eval_expected_cols.get(key, [])
        candidates = [
            str(col)
            for col in expected_cols
            if str(col) == generic or str(col).endswith(f"_{generic}") or f"_{generic}_" in str(col)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    for key, part_count in eval_part_counts.items():
        if key not in grouped_frames:
            out[key] = {
                "quality_status": "ineligible_no_eval_rows",
                "quality_reason": "eval parts were declared but no readable non-empty eval parquet rows were found",
                "eval_part_count": int(part_count),
                "eval_declared_rows": int(eval_declared_rows.get(key, 0)),
                "quality_rows": 0,
            }

    for key, frames in grouped_frames.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            out[key] = {
                "quality_status": "ineligible_no_eval_rows",
                "quality_reason": "eval parts were readable but contained zero rows",
                "eval_part_count": int(eval_part_counts.get(key, 0)),
                "eval_declared_rows": int(eval_declared_rows.get(key, 0)),
                "quality_rows": 0,
            }
            continue
        if not eval_expected_cols.get(key):
            raise ValueError(
                "stats eval quality contract error: eval part is missing expected_cols "
                f"for interval={key[0]} horizon={key[1]} task={key[2]} paths={eval_paths.get(key, [])}"
            )
        squared_col = _contract_metric_column(key, "squared_error_p50")
        absolute_col = _contract_metric_column(key, "abs_error_p50")
        if squared_col is None or absolute_col is None:
            raise ValueError(
                "stats eval quality contract error: missing or ambiguous p50 error columns in declared expected_cols "
                f"for interval={key[0]} horizon={key[1]} task={key[2]} "
                f"squared_col={squared_col!r} abs_col={absolute_col!r} expected_cols={eval_expected_cols.get(key, [])}"
            )
        missing_contract_cols = [col for col in (squared_col, absolute_col) if col not in frame.columns]
        if missing_contract_cols:
            raise ValueError(
                "stats eval quality contract error: declared eval metric columns are absent from eval parquet "
                f"for interval={key[0]} horizon={key[1]} task={key[2]} "
                f"missing={missing_contract_cols} expected_cols={eval_expected_cols.get(key, [])} columns={list(map(str, frame.columns))}"
            )
        squared = _numeric_series(frame, squared_col)
        absolute = _numeric_series(frame, absolute_col)
        valid_squared = squared.dropna()
        valid_absolute = absolute.dropna()
        if valid_squared.empty:
            out[key] = {
                "quality_status": "ineligible_no_valid_quality_rows",
                "quality_reason": f"eval rows exist but {squared_col} contains no finite numeric values",
                "eval_part_count": int(eval_part_counts.get(key, 0)),
                "eval_declared_rows": int(eval_declared_rows.get(key, 0)),
                "eval_observed_rows": int(len(frame)),
                "quality_rows": 0,
            }
            continue
        ts_values = _numeric_series(frame, "ts").dropna()
        out[key] = {
            "quality_status": "passed",
            "squared_error_column": str(squared_col),
            "abs_error_column": str(absolute_col),
            "quality_metric": "rmse_p50",
            "rmse_p50": float(valid_squared.mean() ** 0.5),
            "mae_p50": float(valid_absolute.mean()) if not valid_absolute.empty else None,
            "quality_rows": int(len(valid_squared)),
            "eval_part_count": int(eval_part_counts.get(key, 0)),
            "eval_declared_rows": int(eval_declared_rows.get(key, 0)),
            "eval_observed_rows": int(len(frame)),
            "first_prediction_ts": int(ts_values.min()) if not ts_values.empty else None,
            "last_prediction_ts": int(ts_values.max()) if not ts_values.empty else None,
        }
    return out


def _write_stage1_profile(output_dir: Path, model_key: str, manifest: Dict[str, Any], args: argparse.Namespace) -> None:
    combo_rows = _stage_combo_rows(manifest)
    assets = sorted({asset for row in combo_rows for asset in list(row.get("assets") or [])}) or _split_csv(str(args.assets))
    selections: Dict[str, Dict[str, Any]] = {}
    for row in combo_rows:
        key = _combo_key(int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"]))
        selections[key] = {
            "model_key": str(model_key),
            "interval_minutes": int(row["interval_minutes"]),
            "horizon_minutes": int(row["horizon_minutes"]),
            "task": str(row["task"]),
            "selection_semantics": "stats_relationship_profile",
            "feature_profile": "endogenous_history_and_distribution_form",
            "selected_features": ["target_history"],
            "selected_formulation": {
                "model_family": str(model_key),
                "selected_window_bars": row.get("median_selected_window_bars"),
            },
            "cohort_assets": list(row.get("assets") or assets),
            "forecast_rows": int(row.get("forecast_rows", 0) or 0),
            "eval_rows": int(row.get("eval_rows", 0) or 0),
            "convergence_rate": float(row.get("convergence_rate", 0.0) or 0.0),
            "convergence_warning_count": int(row.get("convergence_warning_count", 0) or 0),
            "skipped_origin_rate": float(row.get("skipped_origin_rate", 0.0) or 0.0),
            "selection_status": str(row.get("status", "failed")),
        }
    payload = {
        "selection_file_version": 2,
        "family": "stats_numeric",
        "model_key": str(model_key),
        "stage1_mode": "relationship_confirmation",
        "generated_at": utc_now_iso(),
        "cohort_assets": list(assets),
        "combo_count": int(len(selections)),
        "selections": selections,
    }
    write_json_atomic(output_dir / "feature_profile_selection.json", payload)
    write_json_atomic(
        output_dir / "feature_experiment_run_meta.json",
        {
            "family": "stats_numeric",
            "model_key": str(model_key),
            "stage1_mode": "relationship_confirmation",
            "status": "completed" if selections else "failed",
            "expected_combo_count": int(len(selections)),
            "completed_combo_count": int(len([row for row in selections.values() if row.get("selection_status") == "passed"])),
            "cohort_assets": list(assets),
            "generated_at": utc_now_iso(),
        },
    )


def _write_stage2_handoff(output_dir: Path, model_key: str, manifest: Dict[str, Any], summary: Dict[str, Any], args: argparse.Namespace) -> None:
    profile_path = Path(args.feature_profile_json).resolve() if args.feature_profile_json is not None else None
    profile = _load_feature_profile(profile_path)
    combo_rows = _stage_combo_rows(manifest)
    quality_by_combo = _stage_eval_quality_rows(manifest)
    survivors = [
        {
            "model_key": str(model_key),
            "interval_minutes": int(row["interval_minutes"]),
            "horizon_minutes": int(row["horizon_minutes"]),
            "task": str(row["task"]),
            "status": str(row["status"]),
            "forecast_rows": int(row.get("forecast_rows", 0) or 0),
            "eval_rows": int(row.get("eval_rows", 0) or 0),
            "convergence_rate": float(row.get("convergence_rate", 0.0) or 0.0),
            "convergence_warning_count": int(row.get("convergence_warning_count", 0) or 0),
            "skipped_origin_rate": float(row.get("skipped_origin_rate", 0.0) or 0.0),
            "median_selected_window_bars": row.get("median_selected_window_bars"),
            "mean_aic": row.get("mean_aic"),
            "mean_bic": row.get("mean_bic"),
            **quality_by_combo.get((int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"])), {}),
            "cohort_assets": list(row.get("assets") or []),
        }
        for row in combo_rows
        if str(row.get("status")) == "passed" and (int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"])) in quality_by_combo
    ]
    assets = sorted({asset for row in combo_rows for asset in list(row.get("assets") or [])})
    manifest_payload = {
        "generated_utc": utc_now_iso(),
        "family": "stats_numeric",
        "model_key": str(model_key),
        "feature_profile_json": str(profile_path) if profile_path is not None else None,
        "feature_profile_cohort_assets": list(profile.get("cohort_assets") or assets),
        "selected_assets": list(assets),
        "stage2_summary_json": str((output_dir / "stage2_summary.json").resolve()),
        "run_manifest_json": str(summary.get("manifest_path", "")),
        "combo_count": int(len(combo_rows)),
        "survivor_count": int(len(survivors)),
        "stage0_profile": {
            "workers": int(args.workers),
            "model_threads": int(args.model_threads),
        },
        "runs": [
            {
                "combo": _combo_key(int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"])),
                "paths": {"run_summary": str((output_dir / "stage2_summary.json").resolve())},
                "metrics": row,
            }
            for row in combo_rows
        ],
    }
    write_json_atomic(output_dir / "diagnostic_manifest.json", manifest_payload)
    write_json_atomic(
        output_dir / "stage3_survivor_handoff.json",
        {
            "family": "stats_numeric",
            "model_key": str(model_key),
            "handoff_kind": "stage2_survivor_handoff",
            "feature_profile_json": str(profile_path) if profile_path is not None else None,
            "diagnostic_manifest_json": str((output_dir / "diagnostic_manifest.json").resolve()),
            "cohort_assets": list(profile.get("cohort_assets") or assets),
            "survivors": survivors,
            "generated_at": utc_now_iso(),
        },
    )


def _clear_stage_locks(branch_root: Path, model_key: str) -> None:
    state_root = Path(branch_root) / STATS_NUMERIC_FAMILY_ROOT_NAMES[str(model_key)] / "state"
    if not state_root.exists():
        return
    for lock_path in state_root.glob("*.lock"):
        try:
            lock_path.unlink()
        except Exception:
            continue


def _stage_manifest_paths(branch_root: Path, model_key: str) -> tuple[Path, Path]:
    state_root = Path(branch_root) / STATS_NUMERIC_FAMILY_ROOT_NAMES[str(model_key)] / "state"
    return state_root / STATS_MANIFEST_FILES[str(model_key)], state_root / STATS_SKIPPED_FILES[str(model_key)]


def _validate_stage_artifacts(branch_root: Path, model_key: str) -> Dict[str, Any]:
    manifest_path, skipped_path = _stage_manifest_paths(branch_root, model_key)
    manifest = load_json_dict(manifest_path)
    skipped = load_json_dict(skipped_path)
    parts = list(manifest.get("parts") or []) if isinstance(manifest.get("parts"), list) else []
    forecast_parts = [part for part in parts if str(part.get("store", "forecast")) == "forecast"]
    eval_parts = [part for part in parts if str(part.get("store", "forecast")) == "eval"]
    forecast_rows = sum(int(part.get("rows", 0) or 0) for part in forecast_parts)
    eval_rows = sum(int(part.get("rows", 0) or 0) for part in eval_parts)
    unit_entries = list(manifest.get("unit_entries") or []) if isinstance(manifest.get("unit_entries"), list) else []
    skipped_units = int(manifest.get("skipped_units", 0) or 0) if manifest else 0
    ok = bool(manifest) and bool(skipped) and bool(unit_entries) and int(forecast_rows) > 0
    reason = None
    if not manifest:
        reason = "missing_run_manifest"
    elif not skipped:
        reason = "missing_skipped_manifest"
    elif not unit_entries:
        reason = "no_unit_entries"
    elif int(forecast_rows) <= 0:
        reason = "no_forecast_rows"
    return {
        "artifact_status": "passed" if ok else "failed",
        "artifact_failure_reason": reason,
        "manifest_path": str(manifest_path),
        "skipped_path": str(skipped_path),
        "forecast_parts": int(len(forecast_parts)),
        "eval_parts": int(len(eval_parts)),
        "forecast_rows": int(forecast_rows),
        "eval_rows": int(eval_rows),
        "unit_entries": int(len(unit_entries)),
        "skipped_units": int(skipped_units),
    }


def run_stage_for_model(model_key: str, stage_name: str, argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entrypoint = STATS_NUMERIC_ENTRYPOINTS[str(model_key)]
    branch_root = output_dir / "parquet" / str(model_key)
    feature_profile = _load_feature_profile(args.feature_profile_json)
    profile_combos = _combos_from_profile(feature_profile)
    effective_combo_list = _combo_list_arg(profile_combos) if bool(args.staged) and profile_combos else str(args.combo_list)
    profile_assets = [str(asset) for asset in list(feature_profile.get("cohort_assets") or []) if str(asset)]
    effective_assets = str(args.assets).strip() or (",".join(profile_assets) if bool(args.staged) and profile_assets else "")
    command = [
        sys.executable,
        "-m",
        entrypoint,
        "--parquet-root",
        str(Path(args.parquet_root).resolve()),
        "--workers",
        str(int(args.workers)),
        "--backfill_days",
        str(int(args.backfill_days)),
        "--fit_days",
        str(int(args.fit_days)),
        "--model_threads",
        str(max(1, int(args.model_threads))),
    ]
    if str(effective_combo_list).strip():
        command.extend(["--combo-list", str(effective_combo_list)])
    if str(effective_assets).strip():
        command.extend(["--assets", str(effective_assets)])
    if bool(args.force):
        command.append("--force")
    if bool(args.predict_latest_only) or str(stage_name) == "stage3":
        command.append("--predict_latest_only")
    if bool(args.fill_to_edge):
        command.append("--fill_to_edge")

    env = dict(os.environ)
    env[STATS_NUMERIC_FAMILY_ROOT_ENVS[str(model_key)]] = str(branch_root)
    if bool(args.force):
        _clear_stage_locks(branch_root, str(model_key))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(max(1, int(args.model_threads)))
    env["STATS_NUMERIC_MODEL_THREADS"] = str(max(1, int(args.model_threads)))
    log_path = output_dir / f"{stage_name}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(command, cwd=str(Path(args.project_root)), env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    artifact_summary = (
        _validate_stage_artifacts(branch_root, str(model_key))
        if int(proc.returncode) == 0
        else {"artifact_status": "failed", "artifact_failure_reason": "subprocess_failed"}
    )
    status = "passed" if int(proc.returncode) == 0 and artifact_summary.get("artifact_status") == "passed" else "failed"
    summary = {
        "stage": str(stage_name),
        "model_key": str(model_key),
        "entrypoint": entrypoint,
        "family_root_env": STATS_NUMERIC_FAMILY_ROOT_ENVS[str(model_key)],
        "family_root_name": STATS_NUMERIC_FAMILY_ROOT_NAMES[str(model_key)],
        "command": command,
        "returncode": int(proc.returncode),
        "status": status,
        "log_path": str(log_path),
        "output_root": str(branch_root),
        **artifact_summary,
        "feature_profile_json": str(Path(args.feature_profile_json).resolve()) if args.feature_profile_json is not None else None,
        "staged": bool(args.staged),
        "finished_at": utc_now_iso(),
    }
    write_json_atomic(output_dir / f"{stage_name}_summary.json", summary)
    if status == "passed":
        manifest = load_json_dict(Path(str(artifact_summary.get("manifest_path", ""))))
        if str(stage_name) == "stage1":
            _write_stage1_profile(output_dir, str(model_key), manifest, args)
        if str(stage_name) == "stage2":
            _write_stage2_handoff(output_dir, str(model_key), manifest, summary, args)
    if int(proc.returncode) != 0 or status != "passed":
        raise SystemExit(int(proc.returncode) if int(proc.returncode) != 0 else 1)
