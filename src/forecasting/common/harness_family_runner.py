from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.forecasting.common.forecast_family_core import (
    DEFAULT_BACKFILL_DAYS,
    default_common_roots,
    default_task_map,
    discover_edge_and_min,
    forecast_output_tail_ts,
    iter_months_between,
    load_assets,
    make_files,
    parse_int_csv,
    parse_str_csv,
    read_feature_window_columns,
    task_target_col,
    utc_now_iso,
    write_json_atomic,
    write_partitioned_predictions,
)
from src.forecasting.common.runtime_config import get_workers, log_resolved_runtime


@dataclass
class HarnessConfig:
    module_tag: str
    family_root_name: str
    family_root_env: str
    source_roots: Sequence[str]
    default_intervals: Sequence[int]
    default_horizons: Sequence[int]
    default_tasks: Sequence[str]


def _pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    err = y - yhat
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def _maybe_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    cols = {str(c): c for c in df.columns}
    for n in names:
        if n in cols:
            return n
    return None


def _coerce_forecast_schema(df: pd.DataFrame, default_model_id: str, default_version: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out_cols = set(str(c) for c in out.columns)

    p50 = _maybe_col(out, ["pred_p50", "pred_mean"])
    if p50 is None:
        yhat_cols = [c for c in out.columns if str(c).lower().endswith("_yhat")]
        if yhat_cols:
            p50 = str(yhat_cols[0])
            out["pred_p50"] = pd.to_numeric(out[p50], errors="coerce")
        else:
            return pd.DataFrame()
    elif p50 != "pred_p50":
        out["pred_p50"] = pd.to_numeric(out[p50], errors="coerce")

    if "pred_p10" not in out_cols:
        out["pred_p10"] = pd.to_numeric(out["pred_p50"], errors="coerce")
    else:
        out["pred_p10"] = pd.to_numeric(out["pred_p10"], errors="coerce")

    if "pred_p90" not in out_cols:
        out["pred_p90"] = pd.to_numeric(out["pred_p50"], errors="coerce")
    else:
        out["pred_p90"] = pd.to_numeric(out["pred_p90"], errors="coerce")

    if "model_id" not in out.columns:
        out["model_id"] = str(default_model_id)
    if "model_version" not in out.columns:
        out["model_version"] = str(default_version)
    if "task" not in out.columns:
        out["task"] = ""
    if "horizon_min" not in out.columns and "horizon_minutes" in out.columns:
        out["horizon_min"] = pd.to_numeric(out["horizon_minutes"], errors="coerce")
    if "interval_min" not in out.columns and "interval" in out.columns:
        out["interval_min"] = pd.to_numeric(out["interval"], errors="coerce")

    for c in ["ts", "horizon_min", "interval_min"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["asset"] = out.get("asset", "").astype(str)
    out["task"] = out.get("task", "").astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["model_version"] = out["model_version"].astype(str)

    keep = [
        "asset",
        "ts",
        "interval_min",
        "horizon_min",
        "task",
        "pred_p10",
        "pred_p50",
        "pred_p90",
        "run_id",
        "model_id",
        "model_version",
        "train_start_ts",
        "train_end_ts",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan

    out = out[keep].dropna(subset=["ts", "horizon_min"]).copy()
    if out.empty:
        return out
    out["ts"] = out["ts"].astype("int64")
    out["horizon_min"] = out["horizon_min"].astype("int64")
    out["interval_min"] = pd.to_numeric(out["interval_min"], errors="coerce").fillna(-1).astype("int64")
    return out


def _read_family_forecasts(
    *,
    parquet_root: Path,
    source_root: str,
    interval: int,
    start_ts: int,
    end_ts: int,
    asset: str,
    task: str,
    horizon_min: int,
) -> pd.DataFrame:
    root = parquet_root / str(source_root)
    if not root.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for y, m in iter_months_between(int(start_ts), int(end_ts)):
        month_dir = root / f"{int(interval)}" / f"year={int(y)}" / f"month={int(m):02d}"
        if not month_dir.exists():
            continue
        for p in sorted(month_dir.glob("*.parquet")):
            try:
                d = pd.read_parquet(p)
            except Exception:
                continue
            if d.empty:
                continue
            d = _coerce_forecast_schema(d, default_model_id=source_root, default_version="legacy")
            if d.empty:
                continue
            d = d[
                (d["asset"].astype(str) == str(asset))
                & (d["task"].astype(str) == str(task))
                & (pd.to_numeric(d["horizon_min"], errors="coerce").astype("int64") == int(horizon_min))
                & (pd.to_numeric(d["interval_min"], errors="coerce").astype("int64") == int(interval))
                & (pd.to_numeric(d["ts"], errors="coerce").astype("int64") >= int(start_ts))
                & (pd.to_numeric(d["ts"], errors="coerce").astype("int64") <= int(end_ts))
            ].copy()
            if d.empty:
                continue
            frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["model_id", "ts"]).drop_duplicates(subset=["model_id", "asset", "ts", "horizon_min", "task"], keep="last")
    return out


def _score_models(pred: pd.DataFrame, actual: pd.DataFrame, edge_ts: int) -> Dict[str, Dict[str, float]]:
    if pred.empty or actual.empty:
        return {}
    joined = pred.merge(actual, on=["asset", "ts"], how="inner")
    if joined.empty:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    six_start = int(edge_ts) - 180 * 86400
    two_start = int(edge_ts) - 730 * 86400

    for mid, grp in joined.groupby("model_id", sort=True):
        g6 = grp[grp["ts"] >= int(six_start)]
        g24 = grp[grp["ts"] >= int(two_start)]
        if g24.empty:
            continue

        def _metrics(g: pd.DataFrame) -> Tuple[float, float]:
            y = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
            p10 = pd.to_numeric(g["pred_p10"], errors="coerce").to_numpy(dtype=float)
            p50 = pd.to_numeric(g["pred_p50"], errors="coerce").to_numpy(dtype=float)
            p90 = pd.to_numeric(g["pred_p90"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(p50)
            if int(mask.sum()) == 0:
                return float("inf"), float("inf")
            y = y[mask]
            p10 = p10[mask]
            p50 = p50[mask]
            p90 = p90[mask]
            mae = float(np.mean(np.abs(y - p50)))
            pb = float(np.mean([_pinball(y, p10, 0.1), _pinball(y, p50, 0.5), _pinball(y, p90, 0.9)]))
            return mae, pb

        mae6, pb6 = _metrics(g6 if not g6.empty else g24)
        mae24, pb24 = _metrics(g24)
        score = 0.4 * mae6 + 0.2 * mae24 + 0.3 * pb6 + 0.1 * pb24
        out[str(mid)] = {
            "mae_6m": float(mae6),
            "mae_24m": float(mae24),
            "pinball_6m": float(pb6),
            "pinball_24m": float(pb24),
            "score": float(score),
        }

    return out


def run_harness(cfg: HarnessConfig) -> None:
    parser = argparse.ArgumentParser(description=f"{cfg.module_tag} family harness")
    parser.add_argument("--intervals", type=str, default=",".join(str(x) for x in cfg.default_intervals))
    parser.add_argument("--horizons_minutes", type=str, default=",".join(str(x) for x in cfg.default_horizons))
    parser.add_argument("--tasks", type=str, default=",".join(str(x) for x in cfg.default_tasks))
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--assets-file", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["incremental", "backfill"], default="incremental")
    parser.add_argument("--backfill_days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict_latest_only", action="store_true")
    parser.add_argument("--workers", type=int, default=get_workers("harnesses", "score_workers", max(1, (os.cpu_count() or 4) // 2)))
    args = parser.parse_args()
    log_resolved_runtime(
        "harnesses",
        resolved={
            "score_workers": max(1, int(args.workers)),
            "writer_workers": 1,
            "model_threads": "n/a",
        },
    )

    parquet_root, feature_root, out_root = default_common_roots(cfg.family_root_env, cfg.family_root_name)
    files = make_files(out_root, "harness_run_manifest.json", "harness_skipped.json")

    intervals = parse_int_csv(args.intervals, cfg.default_intervals)
    horizons = parse_int_csv(args.horizons_minutes, cfg.default_horizons)
    task_map = default_task_map()
    tasks = [t for t in parse_str_csv(args.tasks, cfg.default_tasks) if t in task_map]
    if not tasks:
        tasks = [t for t in cfg.default_tasks if t in task_map]

    assets = load_assets(intervals=intervals, assets_arg=args.assets, assets_file=args.assets_file)
    run_id = os.getenv("RUN_ID", "") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    parts: List[Dict[str, Any]] = []
    skipped: Dict[str, Dict[str, Any]] = {}

    for interval in intervals:
        for hm in horizons:
            if int(hm) % int(interval) != 0:
                continue
            for task in tasks:
                target_col = task_target_col(task, task_map)
                if not target_col:
                    continue

                frames_by_month: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
                updates: List[Tuple[str, Dict[str, Any]]] = []
                max_workers = max(1, int(args.workers))

                def _score_asset(asset: str) -> Tuple[str, Dict[str, Any], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
                    edge_ts, min_ts = discover_edge_and_min(asset=asset, interval_minutes=int(interval))
                    ukey = f"{cfg.module_tag}|{task}|{int(hm)}m|{asset}|{int(interval)}"

                    if edge_ts is None or min_ts is None:
                        return ukey, {"status": "skipped"}, None, {
                            "reason": "missing_edge_or_min_ts",
                            "asset": str(asset),
                            "interval_min": int(interval),
                            "horizon_min": int(hm),
                            "task": str(task),
                        }

                    step = int(interval) * 60
                    target_tail = int(edge_ts) - int(hm) * 60
                    if int(target_tail) < int(min_ts):
                        return ukey, {"status": "done", "reason": "no_closed_target"}, None, None
                    dst_tail = None if args.force else forecast_output_tail_ts(
                        out_root=out_root,
                        interval_minutes=int(interval),
                        task=str(task),
                        horizon_minutes=int(hm),
                        asset=str(asset),
                    )
                    if dst_tail is not None and int(dst_tail) > int(target_tail):
                        raise RuntimeError(
                            f"[hard-stop] unit={ukey} dst_tail={int(dst_tail)} ahead_of_target_tail={int(target_tail)}"
                        )
                    if dst_tail is not None and int(dst_tail) == int(target_tail):
                        return ukey, {"status": "done", "reason": "at_edge"}, None, None

                    eval_start = max(int(min_ts), int(edge_ts) - 730 * 86400)

                    pred_frames = []
                    for src in cfg.source_roots:
                        pred_frames.append(
                            _read_family_forecasts(
                                parquet_root=parquet_root,
                                source_root=str(src),
                                interval=int(interval),
                                start_ts=int(eval_start),
                                end_ts=int(edge_ts),
                                asset=str(asset),
                                task=str(task),
                                horizon_min=int(hm),
                            )
                    )
                    pred = pd.concat([d for d in pred_frames if d is not None and not d.empty], ignore_index=True) if pred_frames else pd.DataFrame()
                    if pred.empty:
                        return ukey, {"status": "skipped"}, None, {
                            "reason": "no_source_forecasts",
                            "asset": str(asset),
                            "interval_min": int(interval),
                            "horizon_min": int(hm),
                            "task": str(task),
                        }

                    actual = read_feature_window_columns(
                        root=feature_root,
                        interval_minutes=int(interval),
                        asset=str(asset),
                        columns=[str(target_col)],
                        start_ts=int(eval_start),
                        end_ts=int(edge_ts),
                    )
                    if actual.empty:
                        return ukey, {"status": "skipped"}, None, {
                            "reason": "missing_actuals",
                            "asset": str(asset),
                            "interval_min": int(interval),
                            "horizon_min": int(hm),
                            "task": str(task),
                        }
                    actual = actual.rename(columns={target_col: "y_true"})[["asset", "ts", "y_true"]]
                    actual["y_true"] = pd.to_numeric(actual["y_true"], errors="coerce")

                    scores = _score_models(pred=pred, actual=actual, edge_ts=int(edge_ts))
                    if not scores:
                        return ukey, {"status": "skipped"}, None, {
                            "reason": "no_scored_models",
                            "asset": str(asset),
                            "interval_min": int(interval),
                            "horizon_min": int(hm),
                            "task": str(task),
                        }

                    selected_model = min(scores.items(), key=lambda kv: kv[1].get("score", float("inf")))[0]
                    selected = pred[pred["model_id"].astype(str) == str(selected_model)].copy()
                    if selected.empty:
                        return ukey, {"status": "skipped"}, None, {
                            "reason": "selected_model_empty",
                            "asset": str(asset),
                            "interval_min": int(interval),
                            "horizon_min": int(hm),
                            "task": str(task),
                        }

                    start_ts = (
                        max(int(min_ts), int(target_tail) - int(args.backfill_days) * 86400)
                        if str(args.mode) == "backfill"
                        else (int(min_ts) if dst_tail is None else int(dst_tail) + int(step))
                    )
                    ts_vals = pd.to_numeric(selected["ts"], errors="coerce").dropna().astype("int64").tolist()
                    finalized = sorted(set(int(t) for t in ts_vals if int(start_ts) <= int(t) <= int(target_tail)))
                    if bool(args.predict_latest_only) and finalized:
                        finalized = [finalized[-1]]
                    if not finalized:
                        return ukey, {"status": "done", "reason": "no_origins"}, None, None

                    score_blob = scores.get(selected_model, {})
                    sel_final = selected[selected["ts"].astype("int64").isin(set(int(t) for t in finalized))].copy()
                    if sel_final.empty:
                        return ukey, {"status": "done", "reason": "no_rows_after_filter"}, None, None

                    sel_final["interval_min"] = int(interval)
                    sel_final["horizon_min"] = int(hm)
                    sel_final["task"] = str(task)
                    sel_final["run_id"] = run_id
                    sel_final["model_id"] = str(cfg.module_tag)
                    sel_final["model_version"] = "1.0.0"
                    sel_final["selected_model"] = str(selected_model)
                    sel_final["selected_score"] = float(score_blob.get("score", float("inf")))
                    sel_final["mae_6m"] = float(score_blob.get("mae_6m", float("nan")))
                    sel_final["mae_24m"] = float(score_blob.get("mae_24m", float("nan")))
                    sel_final["pinball_6m"] = float(score_blob.get("pinball_6m", float("nan")))
                    sel_final["pinball_24m"] = float(score_blob.get("pinball_24m", float("nan")))
                    sel_final["confidence_score"] = float(1.0 / (1.0 + max(0.0, score_blob.get("score", 0.0))))
                    if "train_start_ts" not in sel_final.columns:
                        sel_final["train_start_ts"] = np.nan
                    if "train_end_ts" not in sel_final.columns:
                        sel_final["train_end_ts"] = np.nan

                    return (
                        ukey,
                        {
                            "status": "done",
                            "metadata": {
                                "selected_model": str(selected_model),
                                "selected_score": float(score_blob.get("score", float("inf"))),
                                "mae_6m": float(score_blob.get("mae_6m", float("nan"))),
                                "mae_24m": float(score_blob.get("mae_24m", float("nan"))),
                                "pinball_6m": float(score_blob.get("pinball_6m", float("nan"))),
                                "pinball_24m": float(score_blob.get("pinball_24m", float("nan"))),
                                "start_ts": int(start_ts),
                                "target_tail_ts": int(target_tail),
                                "dst_tail_ts": int(dst_tail) if dst_tail is not None else None,
                            },
                        },
                        sel_final,
                        None,
                    )

                if max_workers == 1 or len(assets) <= 1:
                    asset_results = [_score_asset(str(asset)) for asset in assets]
                else:
                    asset_results = []
                    with ThreadPoolExecutor(max_workers=min(max_workers, len(assets))) as ex:
                        fut_map = {ex.submit(_score_asset, str(asset)): str(asset) for asset in assets}
                        for fut in as_completed(fut_map):
                            asset_results.append(fut.result())

                for ukey, upd, sel_final, skip_info in asset_results:
                    if skip_info:
                        skipped[ukey] = skip_info
                    updates.append((ukey, upd))
                    if sel_final is not None and not sel_final.empty:
                        dt = pd.to_datetime(sel_final["ts"], unit="s", utc=True)
                        sel_final = sel_final.copy()
                        sel_final["year"] = dt.dt.year.astype(int)
                        sel_final["month"] = dt.dt.month.astype(int)
                        for (y, m), grp in sel_final.groupby(["year", "month"], sort=True):
                            frames_by_month.setdefault((int(y), int(m)), []).append(grp.drop(columns=["year", "month"]))

                out_df = pd.concat([pd.concat(v, ignore_index=True) for v in frames_by_month.values()], ignore_index=True) if frames_by_month else pd.DataFrame()
                written = write_partitioned_predictions(
                    out_root=out_root,
                    interval_minutes=int(interval),
                    run_id=run_id,
                    module_tag=cfg.module_tag,
                    task=str(task),
                    horizon_minutes=int(hm),
                    df=out_df,
                )
                parts.extend(written)

    write_json_atomic(files.skipped_file, {"run_id": run_id, "generated_at": utc_now_iso(), "units": skipped})
    write_json_atomic(
        files.manifest_file,
        {
            "run_id": run_id,
            "module": cfg.module_tag,
            "family": cfg.family_root_name,
            "source_roots": list(cfg.source_roots),
            "intervals": intervals,
            "horizon_minutes": horizons,
            "tasks": tasks,
            "mode": str(args.mode),
            "backfill_days": int(args.backfill_days),
            "predict_latest_only": bool(args.predict_latest_only),
            "force": bool(args.force),
            "default_run_profile": "production_full",
            "parts": parts,
            "skipped_units": int(len(skipped)),
            "evaluation_windows": {"window_6m_days": 180, "window_24m_days": 730},
            "finished_at": utc_now_iso(),
        },
    )
