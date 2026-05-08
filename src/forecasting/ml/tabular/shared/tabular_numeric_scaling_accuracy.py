from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import (
    get_tabular_numeric_model_spec,
    load_model_task_metadata,
)


DEFAULT_MODEL_KEY = "xgboost"
TASK_SHORT: Dict[str, str] = {}
TASK_LABEL: Dict[str, str] = {}



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def month_token_from_ts(ts: int) -> str:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def month_tokens_inclusive(start_ts: int, end_ts: int) -> set[str]:
    start = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
    end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
    y = int(start.year)
    m = int(start.month)
    out: set[str] = set()
    while (y, m) <= (int(end.year), int(end.month)):
        out.add(f"{y:04d}-{m:02d}")
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


def profile_sort_key(profile: str) -> Tuple[int, int]:
    try:
        left, right = str(profile).lower().split("x", 1)
        return (int(left), int(right))
    except Exception:
        return (999, 999)


def window_sort_key(label: str) -> Tuple[int, str]:
    raw = str(label).strip().upper()
    if raw == "FULL":
        return (999999, raw)
    if raw.endswith("M"):
        try:
            return (int(raw[:-1]), raw)
        except Exception:
            pass
    return (999998, raw)


def pair_sort_key(pair_key: str) -> Tuple[int, str]:
    task, _, horizon = str(pair_key).partition(":")
    minutes = 0
    if horizon.endswith("m"):
        try:
            minutes = int(horizon[:-1])
        except Exception:
            minutes = 0
    return (minutes, task)


def is_monotonic_non_decreasing(values: Sequence[float], tol: float = 1e-12) -> bool:
    if len(values) < 2:
        return True
    for prev, cur in zip(values[:-1], values[1:]):
        if float(cur) + tol < float(prev):
            return False
    return True


def weighted_mean(pairs: Sequence[Tuple[Optional[float], int]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0
    for value, weight in pairs:
        if value is None or int(weight) <= 0:
            continue
        numerator += float(value) * int(weight)
        denominator += int(weight)
    return float(numerator / denominator) if denominator > 0 else None


def compute_metrics(df: pd.DataFrame, *, y_col: str, pred_col: str, dir_col: Optional[str]) -> dict:
    y = pd.to_numeric(df[y_col], errors="coerce").astype(float)
    pred = pd.to_numeric(df[pred_col], errors="coerce").astype(float)
    valid = y.notna() & pred.notna()
    if not valid.any():
        return {"forecast_count": 0, "rmse": None, "mae": None, "directional_accuracy": None}
    yv = y[valid]
    pv = pred[valid]
    err = pv - yv
    out = {
        "forecast_count": int(valid.sum()),
        "rmse": float(math.sqrt(float((err * err).mean()))),
        "mae": float(err.abs().mean()),
        "directional_accuracy": None,
    }
    if dir_col and dir_col in df.columns:
        true_dir = pd.to_numeric(df.loc[valid, dir_col], errors="coerce").fillna(0).astype(int)
        pred_dir = np.where(pv > 0.0001, 1, np.where(pv < -0.0001, -1, 0))
        out["directional_accuracy"] = float((pred_dir == true_dir.to_numpy()).mean()) if len(true_dir) else None
    return out


def _bin_frame(pair_df: pd.DataFrame, bins: int) -> List[dict]:
    pair_df = pair_df.sort_values("ts").reset_index(drop=True)
    n = len(pair_df)
    if n == 0:
        return []
    if bins <= 1:
        bin_ids = np.zeros(n, dtype=int)
    else:
        bin_ids = np.minimum((np.arange(n) * int(bins)) // n, int(bins) - 1).astype(int)
    pair_df = pair_df.copy()
    pair_df["progress_bin"] = bin_ids
    out: List[dict] = []
    for bin_id, grp in pair_df.groupby("progress_bin", sort=True):
        out.append(
            {
                "bin_index": int(bin_id),
                "rows": int(len(grp)),
                "start_ts": int(grp["ts"].iloc[0]),
                "end_ts": int(grp["ts"].iloc[-1]),
            }
        )
    return out


def _metric_trend_from_bins(bin_metrics: List[dict], metric_key: str) -> dict:
    values = [safe_float(row.get(metric_key)) for row in bin_metrics]
    usable = [(idx, val) for idx, val in enumerate(values) if val is not None]
    if not usable:
        return {
            "first": None,
            "last": None,
            "delta": None,
            "delta_pct": None,
            "slope_per_bin": None,
            "monotonic_worsening": None,
        }
    xs = np.array([idx for idx, _ in usable], dtype=float)
    ys = np.array([val for _, val in usable], dtype=float)
    if len(xs) >= 2:
        slope = float(np.polyfit(xs, ys, 1)[0])
        monotonic = is_monotonic_non_decreasing(list(ys))
    else:
        slope = 0.0
        monotonic = True
    first = float(ys[0])
    last = float(ys[-1])
    delta = last - first
    denom = abs(first) if abs(first) > 1e-12 else None
    delta_pct = (delta / denom) if denom else None
    return {
        "first": first,
        "last": last,
        "delta": float(delta),
        "delta_pct": (float(delta_pct) if delta_pct is not None else None),
        "slope_per_bin": slope,
        "monotonic_worsening": bool(monotonic),
    }


def compute_bin_metrics(
    pair_df: pd.DataFrame,
    *,
    y_col: str,
    pred_col: str,
    dir_col: Optional[str],
    bins: int,
) -> Tuple[List[dict], dict]:
    pair_df = pair_df.sort_values("ts").reset_index(drop=True)
    n = len(pair_df)
    if n == 0:
        return [], {
            "rmse_trend": _metric_trend_from_bins([], "rmse"),
            "mae_trend": _metric_trend_from_bins([], "mae"),
            "directional_accuracy_trend": _metric_trend_from_bins([], "directional_accuracy"),
        }
    if bins <= 1:
        bin_ids = np.zeros(n, dtype=int)
    else:
        bin_ids = np.minimum((np.arange(n) * int(bins)) // n, int(bins) - 1).astype(int)
    work = pair_df.copy()
    work["progress_bin"] = bin_ids
    bin_rows: List[dict] = []
    for bin_id, grp in work.groupby("progress_bin", sort=True):
        metrics = compute_metrics(grp, y_col=y_col, pred_col=pred_col, dir_col=dir_col)
        bin_rows.append(
            {
                "bin_index": int(bin_id),
                "rows": int(len(grp)),
                "start_ts": int(grp["ts"].iloc[0]),
                "end_ts": int(grp["ts"].iloc[-1]),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "directional_accuracy": metrics.get("directional_accuracy"),
            }
        )
    trends = {
        "rmse_trend": _metric_trend_from_bins(bin_rows, "rmse"),
        "mae_trend": _metric_trend_from_bins(bin_rows, "mae"),
        "directional_accuracy_trend": _metric_trend_from_bins(bin_rows, "directional_accuracy"),
    }
    return bin_rows, trends


def read_partitioned_asset_range(
    root: Path,
    *,
    asset: str,
    start_exclusive_ts: int,
    end_inclusive_ts: int,
    columns: Sequence[str],
) -> pd.DataFrame:
    asset_root = root / f"asset={asset}"
    if not asset_root.exists():
        return pd.DataFrame(columns=list(columns))
    month_filter = month_tokens_inclusive(int(start_exclusive_ts) + 1, int(end_inclusive_ts))
    paths: List[Path] = []
    for part in asset_root.rglob("*.parquet"):
        try:
            year = int(part.parent.parent.name.split("=", 1)[1])
            month = int(part.parent.name.split("=", 1)[1])
            token = f"{year:04d}-{month:02d}"
        except Exception:
            token = ""
        if token in month_filter:
            paths.append(part)
    if not paths:
        return pd.DataFrame(columns=list(columns))
    frames: List[pd.DataFrame] = []
    wanted = list(dict.fromkeys(columns))
    for path in sorted(paths):
        df = pd.read_parquet(path, columns=wanted)
        if df.empty:
            continue
        ts = pd.to_numeric(df["ts"], errors="coerce")
        df = df[(ts > int(start_exclusive_ts)) & (ts <= int(end_inclusive_ts))].copy()
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=wanted)
    out = pd.concat(frames, ignore_index=True)
    if "asset" in out.columns:
        out["asset"] = out["asset"].astype(str)
    return out.sort_values("ts").reset_index(drop=True)


def discover_completed_runs(scaling_dir: Path, runtime_profile: str) -> List[dict]:
    runs_root = scaling_dir / "_runs" / f"runtime_profile={runtime_profile}"
    if not runs_root.exists():
        return []
    out: List[dict] = []
    for run_summary_path in sorted(runs_root.rglob("run_summary.json")):
        summary = load_json(run_summary_path)
        if not bool(summary.get("success")):
            continue
        run_dir = run_summary_path.parent
        parquet_root = run_dir / "module_root" / "parquet"
        spec = get_tabular_numeric_model_spec(DEFAULT_MODEL_KEY)
        pred_root = parquet_root / f"{spec.forecast_table_tag}_1"
        eval_root = parquet_root / f"{spec.eval_table_tag}_1"
        if not pred_root.exists() or not eval_root.exists():
            continue
        config = summary.get("config") or {}
        out.append(
            {
                "training_window": str(summary.get("training_window_label")),
                "run_dir": run_dir,
                "summary": summary,
                "config": config,
                "pred_root": pred_root,
                "eval_root": eval_root,
            }
        )
    out.sort(key=lambda row: window_sort_key(str(row["training_window"])))
    return out


def parse_window_filter(raw: str) -> Optional[set[str]]:
    tokens = [str(tok).strip() for tok in str(raw).split(",") if str(tok).strip()]
    if not tokens:
        return None
    return {tok for tok in tokens}


def analyze_run(run: dict, *, bins: int) -> List[dict]:
    config = run.get("config") or {}
    pred_root = Path(run["pred_root"])
    eval_root = Path(run["eval_root"])
    assets = [str(v) for v in (config.get("assets") or [])]
    tasks = [str(v) for v in (config.get("tasks") or []) if str(v) in TASK_SHORT]
    horizons = [int(v) for v in (config.get("horizons") or [])]
    start_exclusive_ts = int(config.get("seed_ts"))
    end_inclusive_ts = int(config.get("accuracy_end_ts"))
    if not assets or not tasks or not horizons:
        return []
    pred_cols = ["asset", "ts"]
    eval_cols = ["asset", "ts"]
    for task in tasks:
        short = TASK_SHORT[task]
        label = TASK_LABEL[task]
        for horizon in horizons:
            pred_cols.append(f"xgb_pred_mean_{short}_{int(horizon)}m")
            eval_cols.append(f"{label}_{int(horizon)}m")
            if task == "log_return":
                eval_cols.append(f"future_direction_{int(horizon)}m")

    results: List[dict] = []
    for asset in assets:
        pred = read_partitioned_asset_range(
            pred_root,
            asset=asset,
            start_exclusive_ts=start_exclusive_ts,
            end_inclusive_ts=end_inclusive_ts,
            columns=pred_cols,
        )
        ev = read_partitioned_asset_range(
            eval_root,
            asset=asset,
            start_exclusive_ts=start_exclusive_ts,
            end_inclusive_ts=end_inclusive_ts,
            columns=eval_cols,
        )
        if pred.empty or ev.empty:
            continue
        merged = pred.merge(ev, on=["asset", "ts"], how="inner")
        if merged.empty:
            continue
        merged = merged.sort_values("ts").reset_index(drop=True)
        for task in tasks:
            short = TASK_SHORT[task]
            label = TASK_LABEL[task]
            for horizon in horizons:
                pred_col = f"xgb_pred_mean_{short}_{int(horizon)}m"
                y_col = f"{label}_{int(horizon)}m"
                dir_col = f"future_direction_{int(horizon)}m" if task == "log_return" else None
                if pred_col not in merged.columns or y_col not in merged.columns:
                    continue
                pair_df = merged[["asset", "ts", pred_col, y_col] + ([dir_col] if dir_col and dir_col in merged.columns else [])].copy()
                metrics = compute_metrics(pair_df, y_col=y_col, pred_col=pred_col, dir_col=dir_col)
                if int(metrics.get("forecast_count", 0) or 0) <= 0:
                    continue
                bins_out, trends = compute_bin_metrics(pair_df, y_col=y_col, pred_col=pred_col, dir_col=dir_col, bins=bins)
                results.append(
                    {
                        "runtime_profile": str(run["summary"].get("runtime_profile")),
                        "training_window": str(run["training_window"]),
                        "training_window_months": run["summary"].get("training_window_months"),
                        "asset": asset,
                        "task": task,
                        "horizon_minutes": int(horizon),
                        "pair_key": f"{task}:{int(horizon)}m",
                        "forecast_count": int(metrics["forecast_count"]),
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "directional_accuracy": metrics.get("directional_accuracy"),
                        "start_ts": start_exclusive_ts + 1,
                        "end_ts": end_inclusive_ts,
                        "progress_bins": bins_out,
                        "trends": trends,
                    }
                )
    return results


def build_pair_summaries(rows: List[dict]) -> dict:
    by_pair: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_pair[str(row["pair_key"])][str(row["training_window"])].append(row)

    pair_summary: Dict[str, dict] = {}
    best_window_votes: Dict[str, Dict[str, int]] = {}
    for pair_key, by_window in sorted(by_pair.items(), key=lambda item: pair_sort_key(item[0])):
        windows: Dict[str, dict] = {}
        rmse_best_per_asset: Dict[str, Tuple[str, float, Optional[float]]] = {}
        mae_best_per_asset: Dict[str, Tuple[str, float, Optional[float]]] = {}
        dir_best_per_asset: Dict[str, Tuple[str, float]] = {}
        for window, entries in sorted(by_window.items(), key=lambda item: window_sort_key(item[0])):
            weighted_rmse = weighted_mean([(safe_float(e.get("rmse")), int(e.get("forecast_count", 0) or 0)) for e in entries])
            weighted_mae = weighted_mean([(safe_float(e.get("mae")), int(e.get("forecast_count", 0) or 0)) for e in entries])
            weighted_dir = weighted_mean(
                [(safe_float(e.get("directional_accuracy")), int(e.get("forecast_count", 0) or 0)) for e in entries]
            )
            rmse_slopes = [safe_float(((e.get("trends") or {}).get("rmse_trend") or {}).get("slope_per_bin")) for e in entries]
            rmse_slopes = [v for v in rmse_slopes if v is not None]
            mae_slopes = [safe_float(((e.get("trends") or {}).get("mae_trend") or {}).get("slope_per_bin")) for e in entries]
            mae_slopes = [v for v in mae_slopes if v is not None]
            monotonic_rmse = [
                ((e.get("trends") or {}).get("rmse_trend") or {}).get("monotonic_worsening") for e in entries
                if ((e.get("trends") or {}).get("rmse_trend") or {}).get("monotonic_worsening") is not None
            ]
            windows[window] = {
                "training_window": window,
                "asset_count": len(entries),
                "forecast_count_total": int(sum(int(e.get("forecast_count", 0) or 0) for e in entries)),
                "rmse_weighted_mean": weighted_rmse,
                "mae_weighted_mean": weighted_mae,
                "directional_accuracy_weighted_mean": weighted_dir,
                "rmse_slope_mean_per_bin": mean(rmse_slopes),
                "mae_slope_mean_per_bin": mean(mae_slopes),
                "rmse_monotonic_worsening_share": (
                    float(sum(1 for v in monotonic_rmse if bool(v)) / len(monotonic_rmse)) if monotonic_rmse else None
                ),
            }
            for entry in entries:
                asset = str(entry["asset"])
                rmse = safe_float(entry.get("rmse"))
                mae = safe_float(entry.get("mae"))
                if rmse is not None:
                    prev = rmse_best_per_asset.get(asset)
                    if prev is None or rmse < prev[1] or (math.isclose(rmse, prev[1]) and (mae or 0.0) < (prev[2] or 0.0)):
                        rmse_best_per_asset[asset] = (window, rmse, mae)
                if mae is not None:
                    prev_mae = mae_best_per_asset.get(asset)
                    if prev_mae is None or mae < prev_mae[1] or (
                        math.isclose(mae, prev_mae[1]) and (rmse or 0.0) < (prev_mae[2] or 0.0)
                    ):
                        mae_best_per_asset[asset] = (window, mae, rmse)
                dir_acc = safe_float(entry.get("directional_accuracy"))
                if dir_acc is not None:
                    prev_dir = dir_best_per_asset.get(asset)
                    if prev_dir is None or dir_acc > prev_dir[1]:
                        dir_best_per_asset[asset] = (window, dir_acc)
        best_window_votes[pair_key] = {
            "rmse": dict(sorted(((w, sum(1 for v in rmse_best_per_asset.values() if v[0] == w)) for w in windows), key=lambda item: window_sort_key(item[0]))),
            "mae": dict(sorted(((w, sum(1 for v in mae_best_per_asset.values() if v[0] == w)) for w in windows), key=lambda item: window_sort_key(item[0]))),
        }
        if dir_best_per_asset:
            best_window_votes[pair_key]["directional_accuracy"] = dict(
                sorted(((w, sum(1 for v in dir_best_per_asset.values() if v[0] == w)) for w in windows), key=lambda item: window_sort_key(item[0]))
            )
        pair_summary[pair_key] = {
            "pair_key": pair_key,
            "task": str(pair_key.split(":", 1)[0]),
            "horizon_minutes": int(pair_key.split(":", 1)[1][:-1]),
            "windows": windows,
            "best_window_votes_by_asset": best_window_votes[pair_key],
        }
    return {"by_pair": pair_summary, "best_window_votes_by_pair": best_window_votes}


def build_best_window_tables(rows: List[dict]) -> dict:
    by_asset_pair: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        by_asset_pair[(str(row["asset"]), str(row["pair_key"]))].append(row)
    by_asset_pair_out: Dict[str, dict] = {}
    aggregate_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (asset, pair_key), entries in sorted(by_asset_pair.items(), key=lambda item: (item[0][0], pair_sort_key(item[0][1]))):
        valid_rmse = [e for e in entries if safe_float(e.get("rmse")) is not None]
        valid_mae = [e for e in entries if safe_float(e.get("mae")) is not None]
        best_rmse = min(
            valid_rmse,
            key=lambda e: (
                float(e["rmse"]),
                float(e["mae"]) if safe_float(e.get("mae")) is not None else float("inf"),
                window_sort_key(str(e["training_window"])),
            ),
        ) if valid_rmse else None
        best_mae = min(
            valid_mae,
            key=lambda e: (
                float(e["mae"]),
                float(e["rmse"]) if safe_float(e.get("rmse")) is not None else float("inf"),
                window_sort_key(str(e["training_window"])),
            ),
        ) if valid_mae else None
        best_dir = None
        valid_dir = [e for e in entries if safe_float(e.get("directional_accuracy")) is not None]
        if valid_dir:
            best_dir = max(valid_dir, key=lambda e: (float(e["directional_accuracy"]), -window_sort_key(str(e["training_window"]))[0]))
        key = f"{asset}|{pair_key}"
        by_asset_pair_out[key] = {
            "asset": asset,
            "pair_key": pair_key,
            "task": pair_key.split(":", 1)[0],
            "horizon_minutes": int(pair_key.split(":", 1)[1][:-1]),
            "best_window_by_rmse": (
                {
                    "training_window": str(best_rmse["training_window"]),
                    "rmse": best_rmse["rmse"],
                    "mae": best_rmse["mae"],
                    "directional_accuracy": best_rmse.get("directional_accuracy"),
                }
                if best_rmse
                else None
            ),
            "best_window_by_mae": (
                {
                    "training_window": str(best_mae["training_window"]),
                    "mae": best_mae["mae"],
                    "rmse": best_mae["rmse"],
                    "directional_accuracy": best_mae.get("directional_accuracy"),
                }
                if best_mae
                else None
            ),
            "best_window_by_directional_accuracy": (
                {
                    "training_window": str(best_dir["training_window"]),
                    "directional_accuracy": best_dir["directional_accuracy"],
                    "rmse": best_dir["rmse"],
                    "mae": best_dir["mae"],
                }
                if best_dir
                else None
            ),
        }
        if best_rmse:
            aggregate_votes[pair_key][str(best_rmse["training_window"])] += 1
    return {
        "by_asset_pair": by_asset_pair_out,
        "aggregate_rmse_votes_by_pair": {
            pair: dict(sorted(window_counts.items(), key=lambda item: window_sort_key(item[0])))
            for pair, window_counts in sorted(aggregate_votes.items(), key=lambda item: pair_sort_key(item[0]))
        },
    }


def build_overall_summary(rows: List[dict]) -> dict:
    by_window: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_window[str(row["training_window"])].append(row)
    out: Dict[str, dict] = {}
    for window, entries in sorted(by_window.items(), key=lambda item: window_sort_key(item[0])):
        out[window] = {
            "training_window": window,
            "pair_count": len(entries),
            "forecast_count_total": int(sum(int(e.get("forecast_count", 0) or 0) for e in entries)),
            "rmse_weighted_mean": weighted_mean([(safe_float(e.get("rmse")), int(e.get("forecast_count", 0) or 0)) for e in entries]),
            "mae_weighted_mean": weighted_mean([(safe_float(e.get("mae")), int(e.get("forecast_count", 0) or 0)) for e in entries]),
            "directional_accuracy_weighted_mean": weighted_mean(
                [(safe_float(e.get("directional_accuracy")), int(e.get("forecast_count", 0) or 0)) for e in entries]
            ),
            "rmse_slope_mean_per_bin": mean(
                [
                    v
                    for v in (
                        safe_float(((e.get("trends") or {}).get("rmse_trend") or {}).get("slope_per_bin")) for e in entries
                    )
                    if v is not None
                ]
            ),
            "mae_slope_mean_per_bin": mean(
                [
                    v
                    for v in (
                        safe_float(((e.get("trends") or {}).get("mae_trend") or {}).get("slope_per_bin")) for e in entries
                    )
                    if v is not None
                ]
            ),
        }
    return {"by_training_window": out}


def build_markdown_report(*, rows: List[dict], pair_summary: dict, overall_summary: dict) -> str:
    lines: List[str] = []
    spec = get_tabular_numeric_model_spec(DEFAULT_MODEL_KEY)
    lines.append(f"# {spec.short_label} Scaling Raw Accuracy Analysis")
    lines.append("")
    lines.append(f"Generated UTC: `{utc_now_iso()}`")
    lines.append("")
    lines.append("This report is derived from saved prediction/eval parquet, not the pre-aggregated `metrics.json` files.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Window | Pair Count | Weighted RMSE | Weighted MAE | Weighted Directional Acc | Mean RMSE Slope/ Bin |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for window, row in (overall_summary.get("by_training_window") or {}).items():
        lines.append(
            f"| {window} | {int(row.get('pair_count', 0))} | "
            f"{(row.get('rmse_weighted_mean') if row.get('rmse_weighted_mean') is not None else 'na')} | "
            f"{(row.get('mae_weighted_mean') if row.get('mae_weighted_mean') is not None else 'na')} | "
            f"{(row.get('directional_accuracy_weighted_mean') if row.get('directional_accuracy_weighted_mean') is not None else 'na')} | "
            f"{(row.get('rmse_slope_mean_per_bin') if row.get('rmse_slope_mean_per_bin') is not None else 'na')} |"
        )
    lines.append("")
    lines.append("## Best Windows By Pair")
    lines.append("")
    lines.append("| Pair | Best RMSE Window Vote Leader | Vote Count | Notes |")
    lines.append("|---|---|---:|---|")
    for pair_key, payload in (pair_summary.get("by_pair") or {}).items():
        votes = ((payload.get("best_window_votes_by_asset") or {}).get("rmse")) or {}
        if votes:
            best_window, best_count = max(votes.items(), key=lambda item: (item[1], -window_sort_key(item[0])[0]))
            distinct = sum(1 for count in votes.values() if int(count) > 0)
            note = "shared best" if distinct > 1 else "uniform"
            lines.append(f"| {pair_key} | {best_window} | {int(best_count)} | {note} |")
        else:
            lines.append(f"| {pair_key} | na | 0 | no completed rows |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `rmse_slope_mean_per_bin > 0` means error tended to worsen as the forecast span advanced.")
    lines.append("- A pair with split vote leaders across assets indicates there is no single clean best window for that pair.")
    lines.append("- Use the JSON outputs for exact per-asset and per-bin detail.")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main_for_model(model_key: str = DEFAULT_MODEL_KEY) -> None:
    global DEFAULT_MODEL_KEY, TASK_SHORT, TASK_LABEL
    DEFAULT_MODEL_KEY = str(model_key)
    spec = get_tabular_numeric_model_spec(model_key)
    project_root = Path(__file__).resolve().parents[5]
    _tasks, TASK_SHORT, TASK_LABEL = load_model_task_metadata(spec, project_root)
    parser = argparse.ArgumentParser(description=f"Analyze raw {spec.short_label} scaling parquet for time-sliced accuracy trends.")
    parser.add_argument("--scaling-dir", type=Path, default=spec.diagnostics_output_dir)
    parser.add_argument("--output-dir", type=Path, default=spec.raw_accuracy_output_dir)
    parser.add_argument("--runtime-profile", type=str, default="8x8")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--training-windows", type=str, default="")
    args = parser.parse_args()

    runs = discover_completed_runs(args.scaling_dir, args.runtime_profile)
    window_filter = parse_window_filter(args.training_windows)
    if window_filter is not None:
        runs = [run for run in runs if str(run["training_window"]) in window_filter]
    all_rows: List[dict] = []
    for run in runs:
        all_rows.extend(analyze_run(run, bins=max(1, int(args.bins))))

    pair_summary = build_pair_summaries(all_rows)
    best_window_summary = build_best_window_tables(all_rows)
    overall_summary = build_overall_summary(all_rows)
    manifest = {
        "generated_utc": utc_now_iso(),
        "scaling_dir": str(args.scaling_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "runtime_profile": str(args.runtime_profile),
        "bins": int(args.bins),
        "completed_training_windows": [str(run["training_window"]) for run in runs],
        "row_count": len(all_rows),
    }
    row_payload = {
        "generated_utc": utc_now_iso(),
        "runtime_profile": str(args.runtime_profile),
        "rows": all_rows,
    }
    pair_payload = {"generated_utc": utc_now_iso(), **pair_summary}
    best_payload = {"generated_utc": utc_now_iso(), **best_window_summary}
    overall_payload = {"generated_utc": utc_now_iso(), **overall_summary}
    report = build_markdown_report(rows=all_rows, pair_summary=pair_summary, overall_summary=overall_summary)

    write_json(args.output_dir / "analysis_manifest.json", manifest)
    write_json(args.output_dir / "accuracy_rows.json", row_payload)
    write_json(args.output_dir / "accuracy_pair_summary.json", pair_payload)
    write_json(args.output_dir / "accuracy_best_windows.json", best_payload)
    write_json(args.output_dir / "accuracy_overall_summary.json", overall_payload)
    (args.output_dir / "accuracy_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    main_for_model(DEFAULT_MODEL_KEY)


if __name__ == "__main__":
    main()
