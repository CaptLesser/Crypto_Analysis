from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

STAGE3_MIN_POSITIVE_ASSETS = 2
STAGE3_BROAD_POSITIVE_ASSETS = 4
STAGE3_MIN_MEAN_SKILL = 0.0
STAGE3_BROAD_MIN_MEDIAN_SKILL = 0.01
STAGE3_NARROW_MIN_MEDIAN_SKILL = 0.03
STAGE3_WINDOW_NEAR_BEST_TOLERANCE = 0.05


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


@dataclass
class StageContext:
    run_summary_path: Path
    interval: str
    interval_minutes: int
    training_window: str
    training_window_months: Optional[int]
    assets: List[str]
    stage_accuracy: Dict[str, Any]


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
        contexts.append(
            StageContext(
                run_summary_path=run_summary_path,
                interval=str(cfg.get("interval")),
                interval_minutes=int(str(cfg.get("interval", "0m")).rstrip("m") or 0),
                training_window=str(summary.get("training_window_label")),
                training_window_months=summary.get("training_window_months"),
                assets=list(cfg.get("assets") or []),
                stage_accuracy=summary.get("accuracy") or {},
            )
        )
    return contexts


def build_asset_combo_detail(manifest_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ctx in stage_contexts(manifest_path):
        accuracy_by_asset = (ctx.stage_accuracy.get("by_asset_target_horizon") or {})
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
                rows.append(
                    {
                        "asset": asset,
                        "task": task,
                        "horizon_minutes": int(horizon_minutes),
                        "interval": ctx.interval,
                        "interval_minutes": int(ctx.interval_minutes),
                        "training_window": ctx.training_window,
                        "training_window_months": ctx.training_window_months,
                        "mae": mae,
                        "rmse": rmse,
                        "baseline_mae": baseline_mae,
                        "baseline_rmse": baseline_rmse,
                        "mae_skill": mae_skill,
                        "rmse_skill": rmse_skill,
                        "forecast_count": metric_block.get("forecast_count"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No asset-level rows built from {manifest_path}")
    df["positive_flag"] = pd.to_numeric(df["rmse_skill"], errors="coerce").fillna(-999.0) > 0.0
    return df


def build_window_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["task", "horizon_minutes", "interval", "interval_minutes", "training_window", "training_window_months"]
    for keys, grp in detail_df.groupby(group_cols, dropna=False, sort=False):
        task, horizon_minutes, interval, interval_minutes, training_window, training_window_months = keys
        skill = pd.to_numeric(grp["rmse_skill"], errors="coerce")
        rows.append(
            {
                "task": task,
                "horizon_minutes": int(horizon_minutes),
                "interval": interval,
                "interval_minutes": int(interval_minutes),
                "training_window": training_window,
                "training_window_months": training_window_months,
                "asset_count": int(len(grp)),
                "positive_asset_count": int((skill > 0.0).sum()),
                "positive_asset_share": safe_float((skill > 0.0).mean()),
                "median_skill": percentile(skill, 0.5),
                "mean_skill": safe_float(skill.mean()),
                "median_mae": percentile(pd.to_numeric(grp["mae"], errors="coerce"), 0.5),
                "median_rmse": percentile(pd.to_numeric(grp["rmse"], errors="coerce"), 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values(["interval_minutes", "horizon_minutes", "task", "training_window_months"], kind="stable").reset_index(drop=True)


def stage3_window_gate(window_row: pd.Series) -> Tuple[bool, str, float]:
    positive_asset_count = int(window_row.get("positive_asset_count") or 0)
    mean_skill = safe_float(window_row.get("mean_skill"))
    median_skill = safe_float(window_row.get("median_skill"))
    min_required_median_skill = float(STAGE3_BROAD_MIN_MEDIAN_SKILL) if positive_asset_count >= int(STAGE3_BROAD_POSITIVE_ASSETS) else float(STAGE3_NARROW_MIN_MEDIAN_SKILL)
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
    return pd.DataFrame(rows)


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
    near_best = near_best.sort_values(["training_window_months", "median_rmse", "median_mae"], ascending=[True, True, True], kind="stable")
    return near_best.iloc[0]


def build_stage3_survivor_handoff(window_df: pd.DataFrame, manifest_path: Path, *, model_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    manifest = load_json(manifest_path)
    contexts = stage_contexts(manifest_path)
    context_map = {(int(ctx.interval_minutes), int(ctx.training_window_months or 0)): ctx for ctx in contexts}
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
                "median_mae": safe_float(selected.get("median_mae")),
                "median_rmse": safe_float(selected.get("median_rmse")),
                "stage2_run_summary_path": str(ctx.run_summary_path) if ctx else None,
                "combo_window_entry": f"{int(interval_minutes)}:{int(horizon_minutes)}:{str(task)}@{int(training_window_months)}m",
            }
        )
    survivors_df = pd.DataFrame(survivors).sort_values(["interval_minutes", "horizon_minutes", "task"], kind="stable").reset_index(drop=True) if survivors else pd.DataFrame(columns=["task", "horizon_minutes", "interval", "interval_minutes", "training_window", "training_window_months", "asset_count", "positive_asset_count", "positive_asset_share", "median_skill", "mean_skill", "median_mae", "median_rmse", "stage2_run_summary_path", "combo_window_entry"])
    payload = {
        "generated_utc": utc_now_iso(),
        "model_key": str(model_key),
        "manifest_path": str(manifest_path),
        "feature_profile_json": manifest.get("feature_profile_json"),
        "cohort_assets": list(manifest.get("feature_profile_cohort_assets") or manifest.get("selected_assets") or []),
        "survivor_count": int(len(survivors_df)),
        "survivors": survivors_df.to_dict(orient="records"),
    }
    return survivors_df, payload


def write_dual(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


def analyze_manifest_for_model(model_key: str, manifest_path: Path) -> Dict[str, str]:
    manifest_path = manifest_path.resolve()
    output_dir = manifest_path.parent
    detail_df = build_asset_combo_detail(manifest_path)
    window_df = build_window_summary(detail_df)
    stage3_survivors_df, stage3_payload = build_stage3_survivor_handoff(window_df, manifest_path, model_key=model_key)
    detail_csv = output_dir / "asset_combo_detail.csv"
    detail_json = output_dir / "asset_combo_detail.json"
    window_csv = output_dir / "window_summary.csv"
    window_json = output_dir / "window_summary.json"
    stage3_csv = output_dir / "stage3_survivor_handoff.csv"
    stage3_json = output_dir / "stage3_survivor_handoff.json"
    write_dual(detail_df, detail_csv, detail_json)
    write_dual(window_df, window_csv, window_json)
    write_dual(stage3_survivors_df, stage3_csv, stage3_json)
    stage3_json.write_text(json.dumps(stage3_payload, indent=2), encoding="utf-8")
    return {"asset_combo_detail_csv": str(detail_csv), "window_summary_csv": str(window_csv), "stage3_survivor_handoff_json": str(stage3_json), "stage3_survivor_handoff_csv": str(stage3_csv)}

