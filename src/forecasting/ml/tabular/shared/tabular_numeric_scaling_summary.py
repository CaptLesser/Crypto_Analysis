from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import get_tabular_numeric_model_spec


DEFAULT_MODEL_KEY = "xgboost"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def avg(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def profile_sort_key(profile: str) -> Tuple[int, int]:
    try:
        left, right = str(profile).lower().split("x", 1)
        return (int(left), int(right))
    except Exception:
        return (999, 999)


def collect_artifacts(output_dir: Path) -> Tuple[List[dict], List[dict], dict]:
    metrics = [load_json(path) for path in output_dir.rglob("metrics.json")]
    run_summaries = [load_json(path) for path in output_dir.rglob("run_summary.json")]
    manifest_path = output_dir / "diagnostic_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return metrics, run_summaries, manifest


def build_runtime_summary(metrics: List[dict], run_summaries: List[dict]) -> dict:
    by_profile_metrics: Dict[str, List[dict]] = defaultdict(list)
    by_profile_runs: Dict[str, List[dict]] = defaultdict(list)
    for row in metrics:
        by_profile_metrics[str(row.get("runtime_profile"))].append(row)
    for row in run_summaries:
        by_profile_runs[str(row.get("runtime_profile"))].append(row)
    out: Dict[str, dict] = {}
    for profile in sorted(set(by_profile_metrics) | set(by_profile_runs), key=profile_sort_key):
        rows = by_profile_metrics.get(profile, [])
        runs = by_profile_runs.get(profile, [])
        training = [v for v in (safe_float(r.get("training_runtime_seconds")) for r in rows) if v is not None]
        forecast = [v for v in (safe_float(r.get("forecast_runtime_seconds")) for r in rows) if v is not None]
        wall = [v for v in (safe_float((r.get("timing") or {}).get("wall_clock_s")) for r in runs) if v is not None]
        total_wall = float(sum(wall)) if wall else None
        units = len(rows)
        throughput = (float(units) / (total_wall / 3600.0)) if units > 0 and total_wall and total_wall > 0 else None
        out[profile] = {
            "runtime_profile": profile,
            "run_count": len(runs),
            "successful_runs": int(sum(1 for r in runs if r.get("success"))),
            "unit_artifact_count": units,
            "training_runtime_seconds_avg": avg(training),
            "training_runtime_seconds_median": median(training),
            "forecast_runtime_seconds_avg": avg(forecast),
            "forecast_runtime_seconds_median": median(forecast),
            "wall_clock_seconds_total": total_wall,
            "wall_clock_seconds_avg": avg(wall),
            "wall_clock_seconds_median": median(wall),
            "throughput_units_per_hour": throughput,
        }
    return {"generated_utc": utc_now_iso(), "by_runtime_profile": out}


def weighted_mean(rows: Sequence[Tuple[Optional[float], int]]) -> Optional[float]:
    num = 0.0
    den = 0
    for value, weight in rows:
        if value is None or weight <= 0:
            continue
        num += float(value) * int(weight)
        den += int(weight)
    return float(num / den) if den > 0 else None


def build_accuracy_summary(metrics: List[dict]) -> dict:
    by_task_horizon: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        key = f"{row.get('task')}:{int(row.get('horizon_minutes'))}m"
        by_task_horizon[key][str(row.get("runtime_profile"))].append(row)
    summary: Dict[str, dict] = {}
    overall_by_profile: Dict[str, List[dict]] = defaultdict(list)
    for task_horizon, profile_rows in sorted(by_task_horizon.items()):
        task, horizon = task_horizon.split(":", 1)
        profile_summary: Dict[str, dict] = {}
        for profile, rows in sorted(profile_rows.items(), key=lambda item: profile_sort_key(item[0])):
            weighted_rows = [(safe_float(r.get("rmse")), int(safe_int(r.get("forecast_count")) or 0)) for r in rows]
            weighted_mae = [(safe_float(r.get("mae")), int(safe_int(r.get("forecast_count")) or 0)) for r in rows]
            weighted_dir = [
                (safe_float(r.get("directional_accuracy")), int(safe_int(r.get("forecast_count")) or 0)) for r in rows
            ]
            entry = {
                "runtime_profile": profile,
                "artifact_count": len(rows),
                "forecast_count_total": int(sum(int(safe_int(r.get("forecast_count")) or 0) for r in rows)),
                "rmse_weighted_mean": weighted_mean(weighted_rows),
                "mae_weighted_mean": weighted_mean(weighted_mae),
                "directional_accuracy_weighted_mean": weighted_mean(weighted_dir),
            }
            profile_summary[profile] = entry
            overall_by_profile[profile].append(entry)
        summary[task_horizon] = {
            "task": task,
            "horizon_minutes": int(horizon[:-1]),
            "by_runtime_profile": profile_summary,
        }
    overall: Dict[str, dict] = {}
    for profile, rows in sorted(overall_by_profile.items(), key=lambda item: profile_sort_key(item[0])):
        overall[profile] = {
            "runtime_profile": profile,
            "task_horizon_cells": len(rows),
            "rmse_mean": avg([v for v in (safe_float(r.get("rmse_weighted_mean")) for r in rows) if v is not None]),
            "mae_mean": avg([v for v in (safe_float(r.get("mae_weighted_mean")) for r in rows) if v is not None]),
            "directional_accuracy_mean": avg(
                [v for v in (safe_float(r.get("directional_accuracy_weighted_mean")) for r in rows) if v is not None]
            ),
        }
    return {"generated_utc": utc_now_iso(), "by_task_and_horizon": summary, "overall_by_runtime_profile": overall}


def build_resource_summary(metrics: List[dict], run_summaries: List[dict]) -> dict:
    by_profile_runs: Dict[str, List[dict]] = defaultdict(list)
    by_profile_metrics: Dict[str, List[dict]] = defaultdict(list)
    for row in run_summaries:
        by_profile_runs[str(row.get("runtime_profile"))].append(row)
    for row in metrics:
        by_profile_metrics[str(row.get("runtime_profile"))].append(row)
    out: Dict[str, dict] = {}
    for profile in sorted(set(by_profile_runs) | set(by_profile_metrics), key=profile_sort_key):
        runs = by_profile_runs.get(profile, [])
        rows = by_profile_metrics.get(profile, [])
        avg_cpu = [
            v for v in (safe_float((r.get("resources") or {}).get("avg_proc_cpu_pct")) for r in runs) if v is not None
        ]
        peak_mem = [v for v in (safe_float(r.get("peak_memory_mb")) for r in rows) if v is not None]
        peak_threads = [
            v for v in (safe_float((r.get("resources") or {}).get("peak_proc_threads")) for r in runs) if v is not None
        ]
        out[profile] = {
            "runtime_profile": profile,
            "avg_cpu_utilization_pct": avg(avg_cpu),
            "max_cpu_utilization_pct": max(avg_cpu) if avg_cpu else None,
            "avg_peak_memory_mb": avg(peak_mem),
            "max_peak_memory_mb": max(peak_mem) if peak_mem else None,
            "avg_peak_process_threads": avg(peak_threads),
            "run_count": len(runs),
        }
    return {"generated_utc": utc_now_iso(), "by_runtime_profile": out}


def build_best_profiles(runtime_summary: dict, accuracy_summary: dict) -> dict:
    runtime_profiles = runtime_summary.get("by_runtime_profile") or {}
    accuracy_profiles = accuracy_summary.get("overall_by_runtime_profile") or {}
    throughput_candidates = [
        row for row in runtime_profiles.values() if safe_float(row.get("throughput_units_per_hour")) is not None
    ]
    best_throughput = None
    if throughput_candidates:
        best_throughput = max(throughput_candidates, key=lambda row: float(row["throughput_units_per_hour"]))

    pair_summaries = accuracy_summary.get("by_task_and_horizon") or {}
    score_by_profile: Dict[str, List[float]] = defaultdict(list)
    for pair_entry in pair_summaries.values():
        profile_map = pair_entry.get("by_runtime_profile") or {}
        rmse_rankable = [(p, safe_float(v.get("rmse_weighted_mean"))) for p, v in profile_map.items()]
        rmse_rankable = [(p, v) for p, v in rmse_rankable if v is not None]
        mae_rankable = [(p, safe_float(v.get("mae_weighted_mean"))) for p, v in profile_map.items()]
        mae_rankable = [(p, v) for p, v in mae_rankable if v is not None]
        dir_rankable = [(p, safe_float(v.get("directional_accuracy_weighted_mean"))) for p, v in profile_map.items()]
        dir_rankable = [(p, v) for p, v in dir_rankable if v is not None]
        for rank, (profile, _) in enumerate(sorted(rmse_rankable, key=lambda item: item[1]), start=1):
            score_by_profile[profile].append(float(rank))
        for rank, (profile, _) in enumerate(sorted(mae_rankable, key=lambda item: item[1]), start=1):
            score_by_profile[profile].append(float(rank))
        for rank, (profile, _) in enumerate(sorted(dir_rankable, key=lambda item: item[1], reverse=True), start=1):
            score_by_profile[profile].append(float(rank))
    scored_profiles = []
    for profile, ranks in score_by_profile.items():
        scored_profiles.append(
            {
                "runtime_profile": profile,
                "accuracy_rank_score": avg(ranks),
                "rmse_mean": (accuracy_profiles.get(profile) or {}).get("rmse_mean"),
                "mae_mean": (accuracy_profiles.get(profile) or {}).get("mae_mean"),
                "directional_accuracy_mean": (accuracy_profiles.get(profile) or {}).get("directional_accuracy_mean"),
            }
        )
    scored_profiles.sort(key=lambda row: (float(row["accuracy_rank_score"]) if row["accuracy_rank_score"] is not None else 9999.0, profile_sort_key(row["runtime_profile"])))
    best_accuracy = scored_profiles[0] if scored_profiles else None
    return {
        "generated_utc": utc_now_iso(),
        "best_runtime_profile_for_throughput": best_throughput,
        "best_runtime_profile_for_accuracy": best_accuracy,
        "accuracy_rank_table": scored_profiles,
    }


def build_refit_projections(runtime_summary: dict, manifest: dict, best_profiles: dict) -> dict:
    runtime_profiles = runtime_summary.get("by_runtime_profile") or {}
    runnable_tasks = manifest.get("runnable_tasks") or sorted(
        {str(path_key.split(":")[0]) for path_key in (manifest.get("pair_keys") or [])}
    )
    training_windows = manifest.get("training_windows") or []
    horizons = manifest.get("horizon_minutes") or []
    units_per_asset = int(len(runnable_tasks)) * int(len(training_windows)) * int(len(horizons))

    def profile_projection(profile_name: Optional[str]) -> Optional[dict]:
        if not profile_name:
            return None
        row = runtime_profiles.get(profile_name)
        throughput = safe_float((row or {}).get("throughput_units_per_hour"))
        if throughput is None or throughput <= 0:
            return None
        projections: Dict[str, dict] = {}
        for asset_count in (8, 20, 50):
            total_units = int(asset_count) * int(units_per_asset)
            hours = float(total_units) / float(throughput)
            projections[str(asset_count)] = {
                "asset_count": int(asset_count),
                "estimated_hours": hours,
                "estimated_minutes": hours * 60.0,
                "estimated_seconds": hours * 3600.0,
                "estimated_total_units": total_units,
            }
        return {
            "runtime_profile": profile_name,
            "throughput_units_per_hour": throughput,
            "units_per_asset": units_per_asset,
            "projections": projections,
        }

    throughput_profile = ((best_profiles.get("best_runtime_profile_for_throughput") or {}).get("runtime_profile"))
    accuracy_profile = ((best_profiles.get("best_runtime_profile_for_accuracy") or {}).get("runtime_profile"))
    return {
        "generated_utc": utc_now_iso(),
        "units_per_asset": units_per_asset,
        "runnable_task_count": len(runnable_tasks),
        "training_window_count": len(training_windows),
        "horizon_count": len(horizons),
        "best_throughput_profile_projection": profile_projection(throughput_profile),
        "best_accuracy_profile_projection": profile_projection(accuracy_profile),
    }


def write_markdown_report(
    output_dir: Path,
    manifest: dict,
    runtime_summary: dict,
    accuracy_summary: dict,
    resource_summary: dict,
    best_profiles: dict,
    refit_projections: dict,
    *,
    model_key: str,
) -> None:
    best_throughput = best_profiles.get("best_runtime_profile_for_throughput") or {}
    best_accuracy = best_profiles.get("best_runtime_profile_for_accuracy") or {}
    runtime_table = runtime_summary.get("by_runtime_profile") or {}
    resource_table = resource_summary.get("by_runtime_profile") or {}
    lines: List[str] = []
    spec = get_tabular_numeric_model_spec(model_key)
    lines.append(f"# {spec.short_label} Numeric Scaling Summary")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now_iso()}")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- Selected assets: {', '.join(manifest.get('selected_assets') or [])}")
    if manifest.get("unsupported_not_run_tasks"):
        lines.append(f"- Unsupported requested tasks not run: {', '.join(manifest.get('unsupported_not_run_tasks') or [])}")
    lines.append("")
    lines.append("## Best Profiles")
    lines.append("")
    if best_throughput:
        lines.append(
            f"- Throughput: `{best_throughput.get('runtime_profile')}` at "
            f"{(safe_float(best_throughput.get('throughput_units_per_hour')) or 0.0):.2f} units/hour"
        )
    if best_accuracy:
        lines.append(
            f"- Accuracy: `{best_accuracy.get('runtime_profile')}` with rank score "
            f"{(safe_float(best_accuracy.get('accuracy_rank_score')) or 0.0):.3f}"
        )
    lines.append("")
    lines.append("## Runtime Summary")
    lines.append("")
    lines.append("| Profile | Runs | Units | Avg Wall (s) | Units/Hour | Avg CPU % | Avg Peak MB |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in sorted(runtime_table, key=profile_sort_key):
        run_row = runtime_table[profile]
        res_row = resource_table.get(profile) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    profile,
                    str(run_row.get("run_count")),
                    str(run_row.get("unit_artifact_count")),
                    f"{(safe_float(run_row.get('wall_clock_seconds_avg')) or 0.0):.2f}",
                    f"{(safe_float(run_row.get('throughput_units_per_hour')) or 0.0):.2f}",
                    f"{(safe_float(res_row.get('avg_cpu_utilization_pct')) or 0.0):.2f}",
                    f"{(safe_float(res_row.get('avg_peak_memory_mb')) or 0.0):.2f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Accuracy Summary")
    lines.append("")
    lines.append("| Profile | Mean RMSE | Mean MAE | Mean Directional Accuracy |")
    lines.append("| --- | ---: | ---: | ---: |")
    for profile, row in sorted((accuracy_summary.get("overall_by_runtime_profile") or {}).items(), key=lambda item: profile_sort_key(item[0])):
        lines.append(
            "| "
            + " | ".join(
                [
                    profile,
                    f"{(safe_float(row.get('rmse_mean')) or 0.0):.6f}",
                    f"{(safe_float(row.get('mae_mean')) or 0.0):.6f}",
                    f"{(safe_float(row.get('directional_accuracy_mean')) or 0.0):.6f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Refit Projections")
    lines.append("")
    for label, key in (
        ("Best Throughput", "best_throughput_profile_projection"),
        ("Best Accuracy", "best_accuracy_profile_projection"),
    ):
        row = refit_projections.get(key)
        if not row:
            continue
        lines.append(f"### {label} `{row.get('runtime_profile')}`")
        lines.append("")
        for asset_count, proj in (row.get("projections") or {}).items():
            lines.append(
                f"- {asset_count} assets: {float(proj.get('estimated_minutes') or 0.0):.2f} minutes "
                f"({float(proj.get('estimated_hours') or 0.0):.2f} hours)"
            )
        lines.append("")
    (output_dir / "scaling_summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(model_key: str) -> argparse.Namespace:
    spec = get_tabular_numeric_model_spec(model_key)
    parser = argparse.ArgumentParser(description=f"Summarize {spec.short_label} numeric scaling diagnostics artifacts.")
    parser.add_argument("--output-dir", type=Path, default=spec.diagnostics_output_dir)
    return parser.parse_args()


def main_for_model(model_key: str = DEFAULT_MODEL_KEY) -> None:
    args = parse_args(model_key)
    output_dir = args.output_dir.resolve()
    metrics, run_summaries, manifest = collect_artifacts(output_dir)
    runtime_summary = build_runtime_summary(metrics, run_summaries)
    accuracy_summary = build_accuracy_summary(metrics)
    resource_summary = build_resource_summary(metrics, run_summaries)
    best_profiles = build_best_profiles(runtime_summary, accuracy_summary)
    refit_projections = build_refit_projections(runtime_summary, manifest, best_profiles)
    (output_dir / "scaling_runtime_summary.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    (output_dir / "scaling_accuracy_summary.json").write_text(json.dumps(accuracy_summary, indent=2), encoding="utf-8")
    (output_dir / "scaling_resource_summary.json").write_text(json.dumps(resource_summary, indent=2), encoding="utf-8")
    (output_dir / "scaling_best_profiles.json").write_text(json.dumps(best_profiles, indent=2), encoding="utf-8")
    (output_dir / "scaling_refit_projections.json").write_text(json.dumps(refit_projections, indent=2), encoding="utf-8")
    write_markdown_report(
        output_dir=output_dir,
        manifest=manifest,
        runtime_summary=runtime_summary,
        accuracy_summary=accuracy_summary,
        resource_summary=resource_summary,
        best_profiles=best_profiles,
        refit_projections=refit_projections,
        model_key=model_key,
    )


def main() -> None:
    main_for_model(DEFAULT_MODEL_KEY)


if __name__ == "__main__":
    main()
