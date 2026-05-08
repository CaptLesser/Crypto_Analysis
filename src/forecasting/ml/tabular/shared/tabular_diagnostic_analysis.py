from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import get_tabular_numeric_model_spec


TASK_LABEL = {
    "log_return": "future_log_return",
    "realized_vol": "future_realized_vol",
    "true_range": "future_true_range",
    "max_drawdown": "future_max_drawdown",
    "max_runup": "future_max_runup",
    "range_efficiency": "future_range_efficiency",
}

DEFAULT_MODEL_KEY = "xgboost"

TASK_FAMILY = {
    "log_return": "returns",
    "realized_vol": "vol_range",
    "true_range": "vol_range",
    "max_drawdown": "path_extremes",
    "max_runup": "path_extremes",
    "range_efficiency": "path_extremes",
}

STAGE3_MIN_POSITIVE_ASSETS = 2
STAGE3_BROAD_POSITIVE_ASSETS = 4
STAGE3_MIN_MEAN_SKILL = 0.0
STAGE3_BROAD_MIN_MEDIAN_SKILL = 0.01
STAGE3_NARROW_MIN_MEDIAN_SKILL = 0.03
STAGE3_WINDOW_NEAR_BEST_TOLERANCE = 0.05


@dataclass
class StageContext:
    run_summary_path: Path
    run_dir: Path
    module_root: Path
    interval: str
    interval_minutes: int
    training_window: str
    training_window_months: Optional[int]
    wall_clock_s: Optional[float]
    module_run_span_s: Optional[float]
    est_unit_wall_clock_s: Optional[float]
    peak_rss_mb: Optional[float]
    avg_cpu_pct: Optional[float]
    peak_threads: Optional[float]
    forecast_year: int
    forecast_month: int
    assets: List[str]
    stage_accuracy: Dict[str, Any]
    stage_unit_diag: Dict[str, Any]
    stage_output_verification: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def percentile(values: pd.Series, q: float) -> Optional[float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    return safe_float(vals.quantile(q))


def baseline_metrics(y: pd.Series) -> Dict[str, Optional[float]]:
    yv = pd.to_numeric(y, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if len(yv) >= 2:
        baseline = yv.shift(1)
        baseline_type = "persistence"
    elif len(yv) == 1:
        baseline = pd.Series([float(yv.mean())] * len(yv), dtype=float)
        baseline_type = "constant_mean"
    else:
        return {
            "baseline_type": None,
            "baseline_mae": None,
            "baseline_rmse": None,
            "y_std": None,
        }
    compare = pd.DataFrame({"y": yv, "baseline": baseline}).dropna()
    if compare.empty:
        return {
            "baseline_type": baseline_type,
            "baseline_mae": None,
            "baseline_rmse": None,
            "y_std": safe_float(yv.std(ddof=0)),
        }
    err = compare["baseline"] - compare["y"]
    return {
        "baseline_type": baseline_type,
        "baseline_mae": safe_float(err.abs().mean()),
        "baseline_rmse": safe_float(math.sqrt(float((err.pow(2.0)).mean()))),
        "y_std": safe_float(yv.std(ddof=0)),
    }


def strength_cutoffs(skill_values: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(skill_values, errors="coerce").dropna()
    abs_nonzero = vals.abs()
    abs_nonzero = abs_nonzero[abs_nonzero > 0.0]
    noise_cutoff = max(0.01, safe_float(abs_nonzero.quantile(0.25)) or 0.01)
    positives = vals[vals > noise_cutoff]
    strong_cutoff = max(0.05, safe_float(positives.quantile(0.75)) or 0.05)
    return float(noise_cutoff), float(strong_cutoff)


def bucket_skill(skill: Optional[float], noise_cutoff: float, strong_cutoff: float) -> str:
    if skill is None:
        return "neutral"
    if skill >= strong_cutoff:
        return "strong_positive"
    if skill > noise_cutoff:
        return "weak_positive"
    if skill < -noise_cutoff:
        return "negative"
    return "neutral"


def find_latest_manifest(root: Path) -> Path:
    manifests = sorted(root.glob("run=*/diagnostic_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No diagnostic_manifest.json found under {root}")
    return manifests[-1].resolve()


def stage_contexts(manifest_path: Path) -> List[StageContext]:
    manifest = load_json(manifest_path)
    contexts: List[StageContext] = []
    for run in manifest.get("runs", []):
        paths = run.get("paths") or {}
        run_summary_path = Path(paths.get("run_summary", ""))
        if not run_summary_path.exists():
            continue
        summary = load_json(run_summary_path)
        cfg = summary.get("config") or {}
        timing = summary.get("timing") or {}
        resources = summary.get("resources") or {}
        forecast_dt = datetime.fromisoformat(cfg["forecast_target_month_start_utc"])
        units_ok = ((summary.get("output_verification") or {}).get("units_ok") or 0)
        wall_clock_s = safe_float(timing.get("wall_clock_s"))
        contexts.append(
            StageContext(
                run_summary_path=run_summary_path,
                run_dir=run_summary_path.parent,
                module_root=Path((summary.get("paths") or {}).get("run_root", "")),
                interval=str(cfg.get("interval")),
                interval_minutes=int(str(cfg.get("interval", "0m")).rstrip("m") or 0),
                training_window=str(summary.get("training_window_label")),
                training_window_months=summary.get("training_window_months"),
                wall_clock_s=wall_clock_s,
                module_run_span_s=safe_float(timing.get("module_run_span_s")),
                est_unit_wall_clock_s=safe_float(wall_clock_s / units_ok) if wall_clock_s is not None and units_ok else None,
                peak_rss_mb=safe_float(resources.get("peak_proc_tree_rss_mb")),
                avg_cpu_pct=safe_float(resources.get("avg_proc_tree_cpu_pct")),
                peak_threads=safe_float(resources.get("peak_proc_threads")),
                forecast_year=int(forecast_dt.year),
                forecast_month=int(forecast_dt.month),
                assets=list(cfg.get("assets") or []),
                stage_accuracy=summary.get("accuracy") or {},
                stage_unit_diag=summary.get("unit_diag") or {},
                stage_output_verification=summary.get("output_verification") or {},
            )
        )
    return contexts


def eval_month_frame(module_root: Path, eval_table_tag: str, interval_minutes: int, asset: str, year: int, month: int) -> Optional[pd.DataFrame]:
    base = module_root / "parquet" / f"{str(eval_table_tag)}_{int(interval_minutes)}" / f"asset={asset}" / f"year={int(year)}" / f"month={int(month):02d}"
    files = sorted(base.glob("*.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(p) for p in files]
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    if "ts" in frame.columns:
        frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    return frame


def build_asset_combo_detail(manifest_path: Path, eval_table_tag: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    contexts = stage_contexts(manifest_path)
    for ctx in contexts:
        accuracy_by_asset = (ctx.stage_accuracy.get("by_asset_target_horizon") or {})
        verification_by_unit = (ctx.stage_output_verification.get("per_unit") or {})
        for asset in ctx.assets:
            asset_acc = accuracy_by_asset.get(asset) or {}
            for key, metric_block in asset_acc.items():
                task, horizon_text = str(key).split(":")
                horizon_minutes = int(str(horizon_text).rstrip("m"))
                rmse = safe_float(metric_block.get("rmse"))
                mae = safe_float(metric_block.get("mae"))
                baseline_rmse = safe_float(metric_block.get("baseline_rmse"))
                baseline_mae = safe_float(metric_block.get("baseline_mae"))
                rmse_skill = safe_float(1.0 - (rmse / baseline_rmse)) if rmse is not None and baseline_rmse not in (None, 0.0) else None
                mae_skill = safe_float(1.0 - (mae / baseline_mae)) if mae is not None and baseline_mae not in (None, 0.0) else None
                unit_key = f"{asset}|{int(ctx.interval_minutes)}|{int(horizon_minutes)}|{task}"
                unit_diag = ctx.stage_unit_diag.get(unit_key) or {}
                verify = verification_by_unit.get(unit_key) or {}
                rows.append(
                    {
                        "asset": asset,
                        "task": task,
                        "task_family": TASK_FAMILY.get(task, "other"),
                        "horizon_minutes": int(horizon_minutes),
                        "interval": ctx.interval,
                        "interval_minutes": int(ctx.interval_minutes),
                        "training_window": ctx.training_window,
                        "training_window_months": ctx.training_window_months,
                        "unit_key": unit_key,
                        "mae": mae,
                        "rmse": rmse,
                        "forecast_count": metric_block.get("forecast_count"),
                        "directional_accuracy": safe_float(metric_block.get("directional_accuracy")),
                        "baseline_type": metric_block.get("baseline_type"),
                        "baseline_mae": baseline_mae,
                        "baseline_rmse": baseline_rmse,
                        "mae_skill": mae_skill,
                        "rmse_skill": rmse_skill,
                        "y_std": safe_float(metric_block.get("y_std")),
                        "training_runtime_seconds": safe_float(unit_diag.get("fit_s")),
                        "forecast_runtime_seconds": safe_float(unit_diag.get("predict_s")),
                        "total_runtime_seconds": safe_float((safe_float(unit_diag.get("fit_s")) or 0.0) + (safe_float(unit_diag.get("predict_s")) or 0.0)),
                        "feature_count": unit_diag.get("feature_count"),
                        "training_rows": unit_diag.get("selected_window_bars"),
                        "run_wall_clock_s": ctx.wall_clock_s,
                        "estimated_per_unit_wall_clock_s": ctx.est_unit_wall_clock_s,
                        "peak_rss_mb": ctx.peak_rss_mb,
                        "avg_cpu_pct": ctx.avg_cpu_pct,
                        "peak_threads": ctx.peak_threads,
                        "eval_rows": verify.get("eval_rows"),
                        "eval_non_null": verify.get("eval_non_null"),
                        "forecast_rows": verify.get("forecast_rows"),
                        "forecast_non_null": verify.get("forecast_non_null"),
                        "failure_reason": None if verify else "missing_output_verification",
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No asset-level rows built from {manifest_path}")
    noise_cutoff, strong_cutoff = strength_cutoffs(df["rmse_skill"])
    df["strength_bucket"] = [
        bucket_skill(safe_float(v), noise_cutoff=noise_cutoff, strong_cutoff=strong_cutoff) for v in df["rmse_skill"]
    ]
    df["positive_flag"] = df["rmse_skill"].fillna(-999.0) > 0.0
    df["negative_flag"] = df["rmse_skill"].fillna(0.0) < 0.0
    df.attrs["noise_cutoff"] = noise_cutoff
    df.attrs["strong_cutoff"] = strong_cutoff
    return df


def efficiency_score(grp: pd.DataFrame) -> Optional[float]:
    median_skill = percentile(grp["rmse_skill"], 0.5)
    positive_share = safe_float((pd.to_numeric(grp["rmse_skill"], errors="coerce") > 0.0).mean())
    runtime = percentile(grp["estimated_per_unit_wall_clock_s"], 0.5) or percentile(grp["total_runtime_seconds"], 0.5)
    rss = percentile(grp["peak_rss_mb"], 0.5) or 0.0
    cpu = percentile(grp["avg_cpu_pct"], 0.5) or 0.0
    if median_skill is None or runtime in (None, 0.0):
        return None
    cost_scale = float(runtime) * (1.0 + (float(rss) / 4096.0)) * (1.0 + (float(cpu) / 400.0))
    if cost_scale <= 0.0:
        return None
    return safe_float((max(0.0, float(median_skill)) * max(0.0, float(positive_share or 0.0))) / cost_scale)


def build_window_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["task", "horizon_minutes", "interval", "interval_minutes", "training_window", "training_window_months", "task_family"]
    for keys, grp in detail_df.groupby(group_cols, dropna=False, sort=False):
        task, horizon_minutes, interval, interval_minutes, training_window, training_window_months, task_family = keys
        skill = pd.to_numeric(grp["rmse_skill"], errors="coerce")
        positive_asset_count = int((skill > 0.0).sum())
        row = {
            "task": task,
            "task_family": task_family,
            "horizon_minutes": int(horizon_minutes),
            "interval": interval,
            "interval_minutes": int(interval_minutes),
            "training_window": training_window,
            "training_window_months": training_window_months,
            "asset_count": int(len(grp)),
            "positive_asset_count": int(positive_asset_count),
            "median_skill": percentile(skill, 0.5),
            "mean_skill": safe_float(skill.mean()),
            "skill_std": safe_float(skill.std(ddof=0)),
            "positive_asset_share": safe_float((skill > 0.0).mean()),
            "strong_positive_share": safe_float((grp["strength_bucket"] == "strong_positive").mean()),
            "runtime_median": percentile(grp["estimated_per_unit_wall_clock_s"], 0.5) or percentile(grp["total_runtime_seconds"], 0.5),
            "runtime_mean": safe_float(pd.to_numeric(grp["estimated_per_unit_wall_clock_s"], errors="coerce").mean()),
            "rss_median": percentile(grp["peak_rss_mb"], 0.5),
            "cpu_median": percentile(grp["avg_cpu_pct"], 0.5),
            "efficiency_score": efficiency_score(grp),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["recommendation_rank_within_combo"] = out.groupby(["task", "horizon_minutes", "interval_minutes"], dropna=False)["efficiency_score"].rank(
        method="min", ascending=False
    )
    return out.sort_values(["task", "interval_minutes", "horizon_minutes", "training_window_months"], kind="stable").reset_index(drop=True)


def stage3_filter_policy() -> Dict[str, float]:
    return {
        "min_positive_assets": float(STAGE3_MIN_POSITIVE_ASSETS),
        "broad_positive_assets": float(STAGE3_BROAD_POSITIVE_ASSETS),
        "min_mean_skill": float(STAGE3_MIN_MEAN_SKILL),
        "broad_min_median_skill": float(STAGE3_BROAD_MIN_MEDIAN_SKILL),
        "narrow_min_median_skill": float(STAGE3_NARROW_MIN_MEDIAN_SKILL),
        "window_near_best_tolerance": float(STAGE3_WINDOW_NEAR_BEST_TOLERANCE),
    }


def stage3_window_gate(window_row: pd.Series) -> Tuple[bool, str, float]:
    positive_asset_count = int(window_row.get("positive_asset_count") or 0)
    mean_skill = safe_float(window_row.get("mean_skill"))
    median_skill = safe_float(window_row.get("median_skill"))
    min_required_median_skill = (
        float(STAGE3_BROAD_MIN_MEDIAN_SKILL)
        if positive_asset_count >= int(STAGE3_BROAD_POSITIVE_ASSETS)
        else float(STAGE3_NARROW_MIN_MEDIAN_SKILL)
    )
    if positive_asset_count < int(STAGE3_MIN_POSITIVE_ASSETS):
        return False, "fewer than 2 positive assets", min_required_median_skill
    if mean_skill is None or float(mean_skill) <= float(STAGE3_MIN_MEAN_SKILL):
        return False, "mean skill is not positive", min_required_median_skill
    if median_skill is None or float(median_skill) < float(min_required_median_skill):
        if positive_asset_count >= int(STAGE3_BROAD_POSITIVE_ASSETS):
            return False, "broad positive coverage but RMSE-skill edge is negligible", min_required_median_skill
        return False, "narrow positive coverage needs stronger RMSE-skill edge", min_required_median_skill
    return True, "approved", min_required_median_skill



def build_stage3_window_candidates(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in window_df.to_dict(orient="records"):
        passed, reason, min_required_median_skill = stage3_window_gate(pd.Series(row))
        enriched = dict(row)
        enriched["stage3_window_pass"] = bool(passed)
        enriched["stage3_window_reason"] = str(reason)
        enriched["stage3_min_required_median_skill"] = float(min_required_median_skill)
        rows.append(enriched)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["interval_minutes", "horizon_minutes", "task", "training_window_months"], kind="stable").reset_index(drop=True)



def select_stage3_window(combo_windows: pd.DataFrame) -> Optional[pd.Series]:
    eligible = combo_windows[combo_windows["stage3_window_pass"] == True].copy()
    if eligible.empty:
        return None
    best_score = safe_float(eligible["median_skill"].max())
    if best_score is None:
        return None
    cutoff = float(best_score) * (1.0 - float(STAGE3_WINDOW_NEAR_BEST_TOLERANCE))
    near_best = eligible[pd.to_numeric(eligible["median_skill"], errors="coerce") >= cutoff].copy()
    if near_best.empty:
        near_best = eligible.copy()
    near_best = near_best.sort_values(
        ["runtime_median", "training_window_months", "positive_asset_count", "mean_skill"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    return near_best.iloc[0]



def build_stage3_survivor_handoff(window_df: pd.DataFrame, manifest_path: Path, *, model_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    manifest = load_json(manifest_path)
    contexts = stage_contexts(manifest_path)
    context_map: Dict[Tuple[int, int], StageContext] = {}
    for ctx in contexts:
        key = (int(ctx.interval_minutes), int(ctx.training_window_months or 0))
        if key not in context_map:
            context_map[key] = ctx
    candidate_df = build_stage3_window_candidates(window_df)
    survivors: List[Dict[str, Any]] = []
    for keys, grp in candidate_df.groupby(["task", "horizon_minutes", "interval", "interval_minutes"], dropna=False, sort=False):
        task, horizon_minutes, interval, interval_minutes = keys
        selected = select_stage3_window(grp)
        if selected is None:
            continue
        training_window_months = int(selected.get("training_window_months") or 0)
        ctx = context_map.get((int(interval_minutes), int(training_window_months)))
        survivors.append(
            {
                "task": str(task),
                "horizon_minutes": int(horizon_minutes),
                "interval": str(interval),
                "interval_minutes": int(interval_minutes),
                "training_window": str(selected.get("training_window")),
                "training_window_months": int(training_window_months),
                "asset_count": int(selected.get("asset_count") or 0),
                "positive_asset_count": int(selected.get("positive_asset_count") or 0),
                "positive_asset_share": safe_float(selected.get("positive_asset_share")),
                "median_skill": safe_float(selected.get("median_skill")),
                "mean_skill": safe_float(selected.get("mean_skill")),
                "runtime_median": safe_float(selected.get("runtime_median")),
                "stage3_window_reason": str(selected.get("stage3_window_reason") or "approved"),
                "stage3_min_required_median_skill": safe_float(selected.get("stage3_min_required_median_skill")),
                "combo_list_entry": f"{int(interval_minutes)}:{int(horizon_minutes)}:{str(task)}",
                "combo_window_entry": f"{int(interval_minutes)}:{int(horizon_minutes)}:{str(task)}@{int(training_window_months)}m",
                "combo_profile_entry": f"{int(interval_minutes)}:{int(horizon_minutes)}:{str(task)}@{int(training_window_months)}m",
                "stage2_run_summary_path": (str(ctx.run_summary_path) if ctx is not None else None),
                "stage2_assets": (list(ctx.assets) if ctx is not None else []),
                "stage2_forecast_target_month_start_utc": (f"{int(ctx.forecast_year):04d}-{int(ctx.forecast_month):02d}-01T00:00:00+00:00" if ctx is not None else None),
            }
        )
    survivors_df = pd.DataFrame(survivors)
    if not survivors_df.empty:
        survivors_df = survivors_df.sort_values(["interval_minutes", "horizon_minutes", "task"], kind="stable").reset_index(drop=True)
    payload = {
        "generated_utc": utc_now_iso(),
        "model_key": str(model_key),
        "manifest_path": str(manifest_path),
        "run_dir": str(manifest_path.parent),
        "feature_profile_json": manifest.get("feature_profile_json"),
        "cohort_assets": list(manifest.get("feature_profile_cohort_assets") or manifest.get("selected_assets") or []),
        "filter_policy": stage3_filter_policy(),
        "survivor_count": int(len(survivors_df)),
        "candidate_combo_count": int(candidate_df[["interval_minutes", "horizon_minutes", "task"]].drop_duplicates().shape[0]) if not candidate_df.empty else 0,
        "survivors": survivors_df.to_dict(orient="records"),
    }
    return survivors_df, payload


def choose_recommended_window(combo_windows: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    by_median = combo_windows.sort_values(
        ["median_skill", "positive_asset_share", "efficiency_score", "training_window_months"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    by_eff = combo_windows.sort_values(
        ["efficiency_score", "median_skill", "positive_asset_share", "training_window_months"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    recommended = by_median
    if (
        pd.notna(by_eff.get("efficiency_score"))
        and pd.notna(by_median.get("median_skill"))
        and pd.notna(by_eff.get("median_skill"))
        and float(by_eff["median_skill"]) >= (0.95 * float(by_median["median_skill"]))
        and float(by_eff.get("positive_asset_share") or 0.0) >= float(by_median.get("positive_asset_share") or 0.0) - 0.03
    ):
        recommended = by_eff
    return recommended, by_median, by_eff


def combo_status(
    positive_share: float,
    median_skill: Optional[float],
    strong_positive_share: float,
    skill_std: Optional[float],
    skill_p25: Optional[float],
    skill_p75: Optional[float],
) -> str:
    median_skill = median_skill or 0.0
    skill_std = skill_std or 0.0
    skill_p25 = skill_p25 or 0.0
    skill_p75 = skill_p75 or 0.0
    if positive_share >= 0.50 and median_skill >= 0.05 and skill_p25 > 0.0 and skill_std <= max(0.30, 4.0 * median_skill):
        return "keep"
    if positive_share >= 0.20 and (
        median_skill > 0.0
        or skill_p75 >= 0.10
        or (positive_share >= 0.60 and median_skill >= -0.02)
    ):
        return "borderline"
    return "drop"


def build_combo_summary(detail_df: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, grp in detail_df.groupby(["task", "horizon_minutes", "interval", "interval_minutes", "task_family"], dropna=False, sort=False):
        task, horizon_minutes, interval, interval_minutes, task_family = keys
        combo_windows = window_df[
            (window_df["task"] == task)
            & (window_df["horizon_minutes"] == horizon_minutes)
            & (window_df["interval_minutes"] == interval_minutes)
        ].copy()
        if combo_windows.empty:
            continue
        recommended_window, by_median, by_eff = choose_recommended_window(combo_windows)
        chosen = grp[grp["training_window"] == recommended_window["training_window"]].copy()
        skill = pd.to_numeric(chosen["rmse_skill"], errors="coerce")
        positive_share = safe_float((skill > 0.0).mean()) or 0.0
        strong_positive_share = safe_float((chosen["strength_bucket"] == "strong_positive").mean()) or 0.0
        weak_positive_share = safe_float((chosen["strength_bucket"] == "weak_positive").mean()) or 0.0
        neutral_share = safe_float((chosen["strength_bucket"] == "neutral").mean()) or 0.0
        negative_share = safe_float((chosen["strength_bucket"] == "negative").mean()) or 0.0
        skill_p25 = percentile(skill, 0.25)
        skill_p75 = percentile(skill, 0.75)
        status = combo_status(
            positive_share=positive_share,
            median_skill=percentile(skill, 0.5),
            strong_positive_share=strong_positive_share,
            skill_std=safe_float(skill.std(ddof=0)),
            skill_p25=skill_p25,
            skill_p75=skill_p75,
        )
        rows.append(
            {
                "task": task,
                "task_family": task_family,
                "horizon_minutes": int(horizon_minutes),
                "interval": interval,
                "interval_minutes": int(interval_minutes),
                "asset_count": int(len(chosen)),
                "positive_asset_share": positive_share,
                "negative_asset_share": negative_share,
                "neutral_asset_share": neutral_share,
                "strong_positive_share": strong_positive_share,
                "weak_positive_share": weak_positive_share,
                "median_skill": percentile(skill, 0.5),
                "mean_skill": safe_float(skill.mean()),
                "skill_std": safe_float(skill.std(ddof=0)),
                "skill_p10": percentile(skill, 0.10),
                "skill_p25": skill_p25,
                "skill_p50": percentile(skill, 0.50),
                "skill_p75": skill_p75,
                "skill_p90": percentile(skill, 0.90),
                "dispersion_iqr": safe_float((percentile(skill, 0.75) or 0.0) - (percentile(skill, 0.25) or 0.0)),
                "best_window_by_median_skill": by_median["training_window"],
                "best_window_by_efficiency": by_eff["training_window"],
                "recommended_window": recommended_window["training_window"],
                "recommended_window_months": recommended_window["training_window_months"],
                "runtime_median": recommended_window.get("runtime_median"),
                "rss_median": recommended_window.get("rss_median"),
                "cpu_median": recommended_window.get("cpu_median"),
                "overall_pass_20pct_rule": bool(positive_share >= 0.20),
                "recommended_status": status,
            }
        )
    out = pd.DataFrame(rows).sort_values(["mean_skill", "median_skill", "positive_asset_share"], ascending=[False, False, False], kind="stable").reset_index(drop=True)
    return out


def build_task_family_summary(combo_df: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family, grp in combo_df.groupby("task_family", dropna=False, sort=False):
        active = grp[grp["recommended_status"] != "drop"].copy()
        kept_intervals = sorted(active["interval_minutes"].dropna().astype(int).unique().tolist())
        interval_split_warranted = len(kept_intervals) > 1
        best_window = None
        if not active.empty:
            merged = active.merge(
                window_df,
                how="left",
                left_on=["task", "horizon_minutes", "interval_minutes", "recommended_window"],
                right_on=["task", "horizon_minutes", "interval_minutes", "training_window"],
                suffixes=("", "_window"),
            )
            if not merged.empty and merged["efficiency_score"].notna().any():
                best_window = merged.sort_values(["efficiency_score", "median_skill_window"], ascending=[False, False], kind="stable").iloc[0]["recommended_window"]
        rows.append(
            {
                "task_family": family,
                "combo_count": int(len(grp)),
                "keep_count": int((grp["recommended_status"] == "keep").sum()),
                "borderline_count": int((grp["recommended_status"] == "borderline").sum()),
                "drop_count": int((grp["recommended_status"] == "drop").sum()),
                "median_combo_skill": percentile(grp["median_skill"], 0.5),
                "mean_combo_skill": safe_float(pd.to_numeric(grp["median_skill"], errors="coerce").mean()),
                "positive_combo_share": safe_float((grp["recommended_status"] != "drop").mean()),
                "recommended_intervals": ",".join(f"{int(v)}m" for v in kept_intervals),
                "interval_split_warranted": bool(interval_split_warranted),
                "default_window_hint": best_window,
            }
        )
    return pd.DataFrame(rows).sort_values(["keep_count", "borderline_count", "median_combo_skill"], ascending=[False, False, False], kind="stable").reset_index(drop=True)


def build_drop_list(combo_df: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, combo in combo_df.iterrows():
        if combo["recommended_status"] != "drop":
            continue
        reason_parts: List[str] = []
        if float(combo.get("positive_asset_share") or 0.0) < 0.20:
            reason_parts.append("weak asset coverage")
        if float(combo.get("median_skill") or 0.0) <= 0.0:
            reason_parts.append("long-horizon collapse" if int(combo["horizon_minutes"]) >= 720 else "noisy/low-value positives")
        combo_windows = window_df[
            (window_df["task"] == combo["task"])
            & (window_df["horizon_minutes"] == combo["horizon_minutes"])
            & (window_df["interval_minutes"] == combo["interval_minutes"])
        ]
        if not combo_windows.empty:
            best_eff = combo_windows["efficiency_score"].max()
            if pd.notna(best_eff) and float(best_eff) <= 0.0:
                reason_parts.append("runtime too expensive for gain")
        rows.append(
            {
                "task": combo["task"],
                "horizon_minutes": int(combo["horizon_minutes"]),
                "interval": combo["interval"],
                "interval_minutes": int(combo["interval_minutes"]),
                "recommended_window": combo["recommended_window"],
                "reason": "; ".join(dict.fromkeys(reason_parts)) or "dominated by another interval/window",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["task", "horizon_minutes", "interval", "interval_minutes", "recommended_window", "reason"])
    return out.sort_values(["interval_minutes", "horizon_minutes", "task"], kind="stable").reset_index(drop=True)


def memo_text(combo_df: pd.DataFrame, window_df: pd.DataFrame, family_df: pd.DataFrame, thresholds: Dict[str, float]) -> str:
    keep = combo_df[combo_df["recommended_status"] == "keep"].copy()
    borderline = combo_df[combo_df["recommended_status"] == "borderline"].copy()
    drop = combo_df[combo_df["recommended_status"] == "drop"].copy()

    def fmt_combo(row: pd.Series) -> str:
        median_skill = safe_float(row.get("median_skill"))
        positive_share = safe_float(row.get("positive_asset_share")) or 0.0
        median_text = f"{float(median_skill):.3f}" if median_skill is not None else "NA"
        return (
            f"{row['interval']} / {int(row['horizon_minutes'])}m / {row['task']} "
            f"(window={row['recommended_window']}, positive_share={float(positive_share):.1%}, "
            f"median_skill={median_text})"
        )

    fantastic = keep[
        (pd.to_numeric(keep["positive_asset_share"], errors="coerce") >= 0.35)
        & (pd.to_numeric(keep["median_skill"], errors="coerce") >= 0.05)
        & (pd.to_numeric(keep["strong_positive_share"], errors="coerce") >= 0.08)
    ].copy()
    acceptable = keep.drop(fantastic.index, errors="ignore")

    window_picks = []
    for _, row in combo_df.iterrows():
        combo_windows = window_df[
            (window_df["task"] == row["task"])
            & (window_df["horizon_minutes"] == row["horizon_minutes"])
            & (window_df["interval_minutes"] == row["interval_minutes"])
        ]
        if combo_windows.empty:
            continue
        eff = combo_windows.sort_values(["efficiency_score", "median_skill"], ascending=[False, False], kind="stable").iloc[0]
        window_picks.append(
            f"{row['interval']} / {int(row['horizon_minutes'])}m / {row['task']}: "
            f"best_eff={eff['training_window']} median_skill={float(eff['median_skill'] or 0.0):.3f} "
            f"runtime={float(eff['runtime_median'] or 0.0):.3f}s rss={float(eff['rss_median'] or 0.0):.0f}MB"
        )

    interval_split_lines = []
    for _, row in family_df.iterrows():
        interval_split_lines.append(
            f"{row['task_family']}: intervals={row['recommended_intervals'] or 'none'} split_warranted={bool(row['interval_split_warranted'])}"
        )

    lines = [
        "# {MODEL_TITLE} Numeric Diagnostic Decision Memo",
        "",
        f"Run analyzed: `{combo_df.attrs.get('run_dir', '')}`",
        f"Asset universe: `{int(combo_df['asset_count'].max())}` per combo at the recommended window when fully populated.",
        f"Strength-bucket rule: `neutral` if |skill| <= {thresholds['noise_cutoff']:.4f}; `strong_positive` if skill >= {thresholds['strong_cutoff']:.4f}; otherwise positive rows above noise are `weak_positive` and rows below -noise are `negative`.",
        "",
        "## Survival",
        f"Clearly survive (`keep`): {len(keep)} combos.",
    ]
    lines.extend(f"- {fmt_combo(row)}" for _, row in keep.iterrows())
    lines.append("")
    lines.append(f"Borderline/questionable: {len(borderline)} combos.")
    lines.extend(f"- {fmt_combo(row)}" for _, row in borderline.iterrows())
    lines.append("")
    lines.append(f"Fail/drop: {len(drop)} combos.")
    lines.extend(f"- {fmt_combo(row)}" for _, row in drop.iterrows())
    lines.append("")
    lines.append("## Quality Tiers")
    lines.append("Fantastic: broad benefit plus strong median and strong-winner share.")
    lines.extend(f"- {fmt_combo(row)}" for _, row in fantastic.iterrows())
    lines.append("Acceptable: passes survival with usable breadth, but not dominant.")
    lines.extend(f"- {fmt_combo(row)}" for _, row in acceptable.iterrows())
    lines.append("Noise / dead: fails the 20% asset-benefit rule or stays too weak to justify production.")
    lines.extend(f"- {fmt_combo(row)}" for _, row in pd.concat([borderline[borderline['positive_asset_share'] < 0.20], drop]).head(12).iterrows())
    lines.append("")
    lines.append("## Window Efficiency")
    lines.extend(f"- {line}" for line in window_picks[:20])
    lines.append("")
    lines.append("## Interval Splits By Task Family")
    lines.extend(f"- {line}" for line in interval_split_lines)
    lines.append("")
    lines.append("## Production Shortlist")
    lines.extend(
        f"- {fmt_combo(row)}"
        for _, row in combo_df[combo_df["recommended_status"].isin(["keep", "borderline"])].iterrows()
    )
    lines.append("")
    lines.append("## Drop Now")
    lines.extend(f"- {fmt_combo(row)}" for _, row in drop.iterrows())
    lines.append("")
    lines.append("## Tuning Guidance")
    lines.append("- Tuning is justified only for `keep` combos and the strongest `borderline` rows that already clear the 20% asset-benefit rule or sit just above it with positive median skill.")
    lines.append("- Do not spend tuning cycles on `drop` combos; the current diagnostic already shows insufficient breadth or unstable/noisy gains.")
    return "\n".join(lines) + "\n"


def write_dual(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


def analyze_manifest_for_model(model_key: str, manifest_path: Path) -> Dict[str, str]:
    spec = get_tabular_numeric_model_spec(model_key)
    manifest_path = manifest_path.resolve()
    output_dir = manifest_path.parent

    detail_df = build_asset_combo_detail(manifest_path, spec.eval_table_tag)
    detail_df.attrs["run_dir"] = str(output_dir)
    window_df = build_window_summary(detail_df)
    combo_df = build_combo_summary(detail_df, window_df)
    combo_df.attrs["run_dir"] = str(output_dir)
    family_df = build_task_family_summary(combo_df, window_df)
    drop_df = build_drop_list(combo_df, window_df)
    survivors_df = combo_df[combo_df["recommended_status"].isin(["keep", "borderline"])].copy()
    stage3_survivors_df, stage3_payload = build_stage3_survivor_handoff(window_df, manifest_path, model_key=model_key)

    thresholds = {
        "noise_cutoff": float(detail_df.attrs["noise_cutoff"]),
        "strong_cutoff": float(detail_df.attrs["strong_cutoff"]),
    }
    memo = memo_text(combo_df, window_df, family_df, thresholds)

    combo_csv = output_dir / "combo_summary.csv"
    combo_json = output_dir / "combo_summary.json"
    window_csv = output_dir / "window_summary.csv"
    window_json = output_dir / "window_summary.json"
    detail_csv = output_dir / "asset_combo_detail.csv"
    detail_json = output_dir / "asset_combo_detail.json"
    family_csv = output_dir / "task_family_summary.csv"
    family_json = output_dir / "task_family_summary.json"
    drop_csv = output_dir / "drop_list.csv"
    drop_json = output_dir / "drop_list.json"
    survivor_csv = output_dir / "survivor_list.csv"
    survivor_json = output_dir / "survivor_list.json"
    stage3_csv = output_dir / "stage3_survivor_handoff.csv"
    stage3_json = output_dir / "stage3_survivor_handoff.json"
    thresholds_json = output_dir / "analysis_thresholds.json"
    memo_name = spec.diagnostic_analysis_output_name
    memo_path = output_dir / memo_name

    write_dual(combo_df, combo_csv, combo_json)
    write_dual(window_df, window_csv, window_json)
    write_dual(detail_df, detail_csv, detail_json)
    write_dual(family_df, family_csv, family_json)
    write_dual(drop_df, drop_csv, drop_json)
    write_dual(survivors_df, survivor_csv, survivor_json)
    write_dual(stage3_survivors_df, stage3_csv, stage3_json)
    stage3_json.write_text(json.dumps(stage3_payload, indent=2), encoding="utf-8")
    memo = memo.replace("{MODEL_TITLE}", str(spec.short_label))
    memo_path.write_text(memo, encoding="utf-8")
    thresholds_json.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    return {
        "combo_summary_csv": str(combo_csv),
        "window_summary_csv": str(window_csv),
        "asset_combo_detail_csv": str(detail_csv),
        "task_family_summary_csv": str(family_csv),
        "drop_list_csv": str(drop_csv),
        "survivor_list_csv": str(survivor_csv),
        "stage3_survivor_handoff_json": str(stage3_json),
        "stage3_survivor_handoff_csv": str(stage3_csv),
        "decision_memo": str(memo_path),
        "analysis_thresholds_json": str(thresholds_json),
    }



def main_for_model(model_key: str = DEFAULT_MODEL_KEY) -> None:
    parser = argparse.ArgumentParser(description="Analyze the latest tabular numeric diagnostic run.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to diagnostic_manifest.json. Defaults to the latest run under logs/diagnostics/xgboost_numeric_scaling_full_test.",
    )
    args = parser.parse_args()

    spec = get_tabular_numeric_model_spec(model_key)
    root = spec.diagnostics_output_dir
    manifest_path = args.manifest.resolve() if args.manifest else find_latest_manifest(root)
    outputs = analyze_manifest_for_model(model_key, manifest_path)

    print(outputs["combo_summary_csv"])
    print(outputs["window_summary_csv"])
    print(outputs["asset_combo_detail_csv"])
    print(outputs["task_family_summary_csv"])
    print(outputs["drop_list_csv"])
    print(outputs["survivor_list_csv"])
    print(outputs["stage3_survivor_handoff_json"])
    print(outputs["decision_memo"])


def main() -> None:
    main_for_model(DEFAULT_MODEL_KEY)


if __name__ == "__main__":
    main()
