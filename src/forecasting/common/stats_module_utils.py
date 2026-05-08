from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.common.ohlcvt_source import ohlcvt_bounds
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.ml_module_utils import make_unit_key
from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.numeric_runner_common import runtime_target_label_window
from src.features.scalar_features import (
    OHLCVT_PARQUET_ROOT,
    PARQUET_COMPRESSION,
    PARQUET_ROW_GROUP,
    PARQUET_ROOT as SCALAR_PARQUET_ROOT,
    list_assets_from_ohlcvt,

)


PIPELINE_PROFILE = selected_profile()
PIPELINE_ROOT = Path(".")
DEFAULT_PARQUET_ROOT = Path(
    resolve_path("source_ohlcvt_root", profile=PIPELINE_PROFILE, required=False)
    or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or Path("parquet")
)
DEFAULT_FEATURE_ROOT = Path(
    resolve_path("source_feature_root", profile=PIPELINE_PROFILE, required=False)
    or SCALAR_PARQUET_ROOT
)
DEFAULT_BACKFILL_DAYS = int(os.getenv("STATS_BACKFILL_DAYS", "730"))
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) // 2)
DEFAULT_CONF_ALPHA = float(os.getenv("STATS_CONF_ALPHA", "0.1"))
DEFAULT_SEASONALITY_REFRESH_DAYS = int(os.getenv("PIPELINE_SEASONALITY_REFRESH_DAYS", "30"))

NUMERIC_TASK_TO_TARGET_COLUMN: Dict[str, str] = {
    "log_return": "future_log_return",
    "realized_vol": "future_realized_vol",
    "true_range": "future_true_range",
    "max_drawdown": "future_max_drawdown",
    "max_runup": "future_max_runup",
    "range_efficiency": "future_range_efficiency",
}

REGIME_TASK_TO_TARGET_COLUMN: Dict[str, str] = {
    "regime_state": "future_regime_state",
}

CAPABILITY_MATRIX: Dict[str, Dict[str, Sequence[str]]] = {
    "llt": {
        "numerics": ("log_return", "realized_vol", "true_range"),
        "regimes": (),
    },
    "sarimax": {
        "numerics": ("log_return", "realized_vol", "true_range"),
        "regimes": (),
    },
    "egarch": {
        "numerics": ("realized_vol", "true_range"),
        "regimes": (),
    },
    "quantreg": {
        "numerics": tuple(NUMERIC_TASK_TO_TARGET_COLUMN.keys()),
        "regimes": (),
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sandbox_resolution_env() -> Dict[str, str]:
    env = {str(key): str(value) for key, value in os.environ.items()}
    raw_root = str(env.get("PIPELINE_SANDBOX_OUTPUT_ROOT", "") or "").strip()
    raw_pipeline_root = str(env.get("PIPELINE_ROOT", "") or "").strip()
    if raw_root and raw_pipeline_root:
        try:
            if Path(raw_root).expanduser().resolve() == Path(raw_pipeline_root).expanduser().resolve():
                env.pop("PIPELINE_ROOT", None)
        except Exception:
            pass
    return env


def _sandbox_roots() -> SandboxOutputRoots:
    return resolve_sandbox_output_roots(env=_sandbox_resolution_env())


def _sandbox_env_path(roots: SandboxOutputRoots, env_name: str, fallback: Path, kind: str) -> Path:
    raw = str(os.getenv(env_name, "") or "").strip()
    path = Path(raw) if raw else Path(fallback)
    assert_write_allowed(path, kind, roots=roots)
    return path


def default_stats_source_parquet_root(family_root_env: str, fallback: Path) -> Path:
    roots = _sandbox_roots()
    if roots.enabled:
        return Path(os.getenv("PIPELINE_SOURCE_PARQUET_ROOT") or os.getenv("PIPELINE_PARQUET_ROOT", str(fallback))).resolve()
    return Path(os.getenv(str(family_root_env), str(fallback)))


def default_stats_forecast_root(family_root_env: str, family_root_name: str, fallback: Path) -> Path:
    roots = _sandbox_roots()
    if roots.enabled:
        parquet_root = _sandbox_env_path(
            roots,
            "PIPELINE_SANDBOX_PARQUET_ROOT",
            roots.parquet_root,
            "Stats forecast parquet root",
        )
        root = parquet_root / str(family_root_name)
        assert_write_allowed(root, "Stats forecast root", roots=roots)
        return root.resolve()
    return Path(os.getenv(str(family_root_env), str(fallback))) / str(family_root_name)


def default_stats_state_root(forecast_root: Path, family_tag: str) -> Path:
    roots = _sandbox_roots()
    if roots.enabled:
        state_root = _sandbox_env_path(
            roots,
            "PIPELINE_SANDBOX_STATE_ROOT",
            roots.state_root,
            "Stats state root",
        ) / "stats_numeric_runner" / str(family_tag)
        assert_write_allowed(state_root, "Stats state root", roots=roots)
        return state_root.resolve()
    return Path(forecast_root) / "state"


def default_stats_log_file(filename: str) -> Path:
    roots = _sandbox_roots()
    if roots.enabled:
        log_root = _sandbox_env_path(
            roots,
            "PIPELINE_SANDBOX_LOG_ROOT",
            roots.log_root,
            "Stats log root",
        ) / "stats_numeric_runner"
        assert_write_allowed(log_root, "Stats log root", roots=roots)
        log_root.mkdir(parents=True, exist_ok=True)
        return (log_root / str(filename)).resolve()
    log_dir = PIPELINE_ROOT / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return log_dir / str(filename)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    roots = _sandbox_roots()
    assert_write_allowed(path, "Stats JSON", roots=roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "Stats JSON temp", roots=roots)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def iter_months_between(start_ts: int, end_ts: int) -> Iterable[Tuple[int, int]]:
    if end_ts is None or start_ts is None or int(end_ts) < int(start_ts):
        return
    cur = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
    y, m = int(cur.year), int(cur.month)
    while True:
        yield (y, m)
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        if int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp()) > int(end_ts):
            break


def make_stats_unit_key(
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    asset: str,
    interval_minutes: int,
) -> str:
    return make_unit_key(
        family=family,
        domain=domain,
        task=task,
        horizon_minutes=int(horizon_minutes),
        asset=asset,
        interval=int(interval_minutes),
    )



def list_assets_from_features(interval_minutes: int, root: Optional[Path] = None) -> List[str]:
    base = Path(root or DEFAULT_FEATURE_ROOT) / f"scalar_features_{int(interval_minutes)}"
    if not base.exists():
        return []
    assets: set[str] = set()
    for child in base.glob("asset=*"):
        if child.is_dir() and child.name.startswith("asset="):
            asset = child.name.split("=", 1)[1].strip()
            if asset:
                assets.add(asset)
    return sorted(assets)
def resolve_assets(intervals: Sequence[int], assets_arg: str) -> List[str]:
    if str(assets_arg or "").strip():
        return sorted({a.strip() for a in str(assets_arg).split(",") if a.strip()})
    assets: set[str] = set()
    for interval in intervals:
        assets.update(list_assets_from_ohlcvt(int(interval)))
        assets.update(list_assets_from_features(int(interval)))
    return sorted(assets)


def interval_edge_ts(asset: str, interval_minutes: int) -> Optional[int]:
    _mn, mx = ohlcvt_bounds(int(interval_minutes), str(asset), root=OHLCVT_PARQUET_ROOT)
    return int(mx) if mx is not None else None


def interval_min_ts(asset: str, interval_minutes: int) -> Optional[int]:
    mn, _mx = ohlcvt_bounds(int(interval_minutes), str(asset), root=OHLCVT_PARQUET_ROOT)
    return int(mn) if mn is not None else None


def scalar_feature_month_path(root: Path, interval_minutes: int, asset: str, year: int, month: int) -> Path:
    return (
        root
        / f"scalar_features_{int(interval_minutes)}"
        / f"asset={str(asset)}"
        / f"year={int(year)}"
        / f"month={int(month):02d}"
        / f"part-scalar_features_{int(interval_minutes)}-{str(asset)}-{int(year)}{int(month):02d}.parquet"
    )


def scalar_feature_month_path_legacy(root: Path, interval_minutes: int, year: int, month: int) -> Path:
    return (
        root
        / f"scalar_features_{int(interval_minutes)}"
        / f"year={int(year)}"
        / f"month={int(month):02d}"
        / f"part-scalar_features_{int(interval_minutes)}-{int(year)}{int(month):02d}.parquet"
    )


def read_feature_series_window(
    root: Path,
    interval_minutes: int,
    asset: str,
    column: str,
    start_ts: int,
    end_ts: int,
    horizon_bars: Optional[int] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    cols = ["ts", "asset", str(column)]
    for y, m in iter_months_between(int(start_ts), int(end_ts)):
        p = scalar_feature_month_path(
            root=root,
            interval_minutes=int(interval_minutes),
            asset=str(asset),
            year=int(y),
            month=int(m),
        )
        if not p.exists():
            p = scalar_feature_month_path_legacy(root=root, interval_minutes=int(interval_minutes), year=int(y), month=int(m))
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            continue
        if df.empty:
            continue
        ts = pd.to_numeric(df.get("ts"), errors="coerce")
        mask = (
            (df.get("asset").astype(str) == str(asset))
            & ts.notna()
            & (ts.astype("int64") >= int(start_ts))
            & (ts.astype("int64") <= int(end_ts))
        )
        d = df.loc[mask, cols].copy()
        if not d.empty:
            frames.append(d)
    if not frames:
        out = pd.DataFrame(columns=cols)
    else:
        out = pd.concat(frames, ignore_index=True)
    if out.empty:
        out = _runtime_future_label_window(
            interval_minutes=int(interval_minutes),
            asset=str(asset),
            column=str(column),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            horizon_bars=horizon_bars,
        )
        if out.empty:
            return pd.DataFrame(columns=cols)
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out["asset"] = out["asset"].astype(str)
    out = out.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")
    if str(column) in NUMERIC_TASK_TO_TARGET_COLUMN.values() and not pd.to_numeric(out[column], errors="coerce").notna().any():
        runtime = _runtime_future_label_window(
            interval_minutes=int(interval_minutes),
            asset=str(asset),
            column=str(column),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            horizon_bars=horizon_bars,
        )
        if not runtime.empty:
            out = runtime
    return out


def _runtime_future_label_window(
    *,
    interval_minutes: int,
    asset: str,
    column: str,
    start_ts: int,
    end_ts: int,
    horizon_bars: Optional[int] = None,
) -> pd.DataFrame:
    if str(column) not in set(NUMERIC_TASK_TO_TARGET_COLUMN.values()):
        return pd.DataFrame(columns=["ts", "asset", str(column)])
    resolved_horizon_bars = max(1, int(horizon_bars if horizon_bars is not None else os.getenv("STATS_RUNTIME_LABEL_HORIZON_BARS", "1")))
    return runtime_target_label_window(
        parquet_root=OHLCVT_PARQUET_ROOT,
        asset=str(asset),
        interval=int(interval_minutes),
        horizon_bars=int(resolved_horizon_bars),
        target_col=str(column),
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        read_ohlcvt_fn=read_ohlcvt,
        compute_future_labels_fn=compute_future_labels,
    )


def parse_interval_labels(interval_minutes: int) -> str:
    iv = int(interval_minutes)
    canonical = {
        60: "1H",
        240: "4H",
        1440: "1D",
    }
    return canonical.get(iv, f"{iv}m")


def interval_label_candidates(interval_minutes: int) -> List[str]:
    iv = int(interval_minutes)
    canonical = parse_interval_labels(iv)
    legacy = f"{iv}m"
    if canonical == legacy:
        return [canonical]
    return [canonical, legacy]


@dataclass
class SeasonalityProfile:
    source: str
    usable: bool
    seasonal_period_bars: Optional[int]
    path: Optional[str]
    details: Dict[str, Any]


def _seasonality_root(parquet_root: Path) -> Path:
    roots = _sandbox_roots()
    if roots.enabled:
        raw = str(os.getenv("PIPELINE_SEASONALITY_ROOT", "") or "").strip()
        root = Path(raw) if raw else _sandbox_env_path(
            roots,
            "PIPELINE_SANDBOX_STATE_ROOT",
            roots.state_root,
            "Stats seasonality state root",
        ) / "stats_numeric_runner" / "seasonality"
        assert_write_allowed(root, "Stats seasonality root", roots=roots)
        return root.resolve()
    return Path(os.getenv("PIPELINE_SEASONALITY_ROOT", str(parquet_root / "seasonality")))


def _profile_path(root: Path, interval_label: str, asset: Optional[str]) -> Path:
    if asset:
        return root / interval_label / "assets" / str(asset) / "seasonality.parquet"
    return root / interval_label / "global" / "seasonality.parquet"


def _selection_path(root: Path, interval_label: str, asset: str) -> Path:
    return root / interval_label / "selection" / str(asset) / "seasonality_selection.json"


def _selection_lock_path(selection_path: Path) -> Path:
    return selection_path.with_name(f"{selection_path.name}.lock")


def _utc_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _is_fresh(path: Path, refresh_days: int) -> bool:
    if int(refresh_days) <= 0:
        return False
    if not path.exists():
        return False
    try:
        age = _utc_timestamp() - int(path.stat().st_mtime)
    except Exception:
        return False
    return age < int(refresh_days) * 86400


def _load_profile(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return df if not df.empty else None
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_local(path: Path, payload: Dict[str, Any]) -> None:
    roots = _sandbox_roots()
    assert_write_allowed(path, "Stats seasonality JSON", roots=roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "Stats seasonality JSON temp", roots=roots)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def _acquire_file_lock(lock_path: Path, *, timeout_sec: float, stale_sec: float) -> bool:
    roots = _sandbox_roots()
    assert_write_allowed(lock_path, "Stats seasonality lock", roots=roots)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    poll = 0.05
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                payload = json.dumps({"pid": os.getpid(), "created_at": utc_now_iso()}, sort_keys=True).encode("utf-8")
                os.write(fd, payload)
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            try:
                age = _utc_timestamp() - int(lock_path.stat().st_mtime)
                if age > max(1.0, float(stale_sec)):
                    lock_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)
            poll = min(1.0, poll * 1.5)


def _release_file_lock(lock_path: Path) -> None:
    assert_write_allowed(lock_path, "Stats seasonality lock release", roots=_sandbox_roots())
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _infer_profile(df: pd.DataFrame) -> Tuple[bool, Optional[int], Dict[str, Any]]:
    cols = set(str(c) for c in df.columns)
    usable = False
    for c in ("overall_usable", "usable"):
        if c in cols:
            try:
                vals = df[c].dropna().astype(bool)
                if not vals.empty:
                    usable = bool(vals.iloc[0])
                    break
            except Exception:
                pass
    if not usable and "overall_recommended_use" in cols:
        try:
            rec = str(df["overall_recommended_use"].dropna().iloc[0]).lower()
            usable = rec in {"high", "med", "medium", "use"}
        except Exception:
            pass

    period = None
    for c in ("recommended_period_bars", "period_bars", "vol_recommended_period", "return_recommended_period"):
        if c in cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if not s.empty and int(s.iloc[0]) > 1:
                period = int(s.iloc[0])
                break
    details = {
        "columns": sorted(cols),
        "rows": int(len(df)),
        "inferred_usable": bool(usable),
        "inferred_period_bars": period,
    }
    return bool(usable), period, details


def _candidate_periods_from_profile(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = set(str(c) for c in df.columns)
    out: List[Dict[str, Any]] = []
    if {"record_type", "period_bars"}.issubset(cols):
        rows = df[df["record_type"].astype(str).eq("period_candidate")].copy()
        for _, row in rows.iterrows():
            try:
                period = int(float(row.get("period_bars")))
            except Exception:
                continue
            if period <= 1:
                continue
            out.append(
                {
                    "period_bars": period,
                    "period_channel": row.get("period_channel"),
                    "quality_score": float(row.get("quality_score", 0.0) or 0.0),
                    "stability_score": float(row.get("stability_score", 0.0) or 0.0),
                    "recommended_use": str(row.get("recommended_use", "")),
                    "usable": bool(row.get("usable", False)),
                }
            )
    for c in ("recommended_period_bars", "period_bars", "vol_recommended_period", "return_recommended_period"):
        if c not in cols:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if vals.empty:
            continue
        period = int(vals.iloc[0])
        if period > 1 and not any(int(v["period_bars"]) == period for v in out):
            out.append(
                {
                    "period_bars": period,
                    "period_channel": None,
                    "quality_score": 0.0,
                    "stability_score": 0.0,
                    "recommended_use": "artifact",
                    "usable": True,
                }
            )
    return sorted(out, key=lambda d: (float(d["quality_score"]), float(d["stability_score"])), reverse=True)


def _ensure_asset_profile(
    *,
    parquet_root: Path,
    seasonality_root: Path,
    interval_label: str,
    asset: str,
    refresh_days: int,
) -> None:
    asset_path = _profile_path(seasonality_root, interval_label, asset=asset)
    if _is_fresh(asset_path, refresh_days):
        return
    if os.getenv("PIPELINE_SEASONALITY_LAZY_BUILD", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        from src.features.calendar_seasonality import artifact_to_dataframe, build_asset_artifact
        from src.features.calendar_seasonality import INTERVAL_TO_MIN
        from src.features.calendar_seasonality import asset_bounds_for_interval

        interval_min = int(INTERVAL_TO_MIN[interval_label])
        bounds = asset_bounds_for_interval(parquet_root, interval_min).get(str(asset))
        if not bounds:
            return
        end_ts = int(bounds["max_ts"])
        lookback_days = int(os.getenv("PIPELINE_SEASONALITY_ASSET_LOOKBACK_DAYS", "730"))
        start_ts = max(int(bounds["min_ts"]), end_ts - lookback_days * 86400)
        art = build_asset_artifact(
            parquet_root=parquet_root,
            interval_label=interval_label,
            asset=str(asset),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            prefer_scalar_features_true_range=True,
            smoothing_window=int(os.getenv("PIPELINE_SEASONALITY_SMOOTHING_WINDOW", "5")),
            top_k=int(os.getenv("PIPELINE_SEASONALITY_TOP_K", "10")),
            baseline_method=os.getenv("PIPELINE_SEASONALITY_BASELINE_METHOD", "median"),
        )
        if art is None:
            return
        roots = _sandbox_roots()
        assert_write_allowed(asset_path, "Stats seasonality profile parquet", roots=roots)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = sibling_temp_path(asset_path)
        assert_write_allowed(tmp, "Stats seasonality profile parquet temp", roots=roots)
        artifact_to_dataframe(art).to_parquet(tmp, index=False)
        atomic_replace(tmp, asset_path)
    except Exception:
        return


def _rolling_origin_score(values: pd.Series, period: Optional[int]) -> Optional[float]:
    y = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(y) < 96:
        return None
    arr = y.astype(float).to_numpy()
    if period is not None and int(period) > 1:
        p = int(period)
        if len(arr) <= p + 32:
            return None
        actual = arr[p:]
        pred = arr[:-p]
    else:
        start = min(64, max(8, len(arr) // 10))
        if len(arr) <= start + 8:
            return None
        actual = arr[start:]
        baseline = float(pd.Series(arr[:start]).median())
        pred = pd.Series(arr).expanding().median().shift(1).to_numpy()[start:]
        pred = pd.Series(pred).fillna(baseline).to_numpy(dtype=float)
    err = actual - pred
    denom = float(pd.Series(actual).mad()) if hasattr(pd.Series(actual), "mad") else float(abs(actual - pd.Series(actual).median()).mean())
    denom = max(denom, 1e-12)
    return float((pd.Series(err).abs().median()) / denom)


def _score_period_candidate(parquet_root: Path, interval_label: str, asset: str, period: Optional[int], channel: Optional[str]) -> Optional[float]:
    try:
        from src.features.calendar_seasonality import INTERVAL_TO_MIN, _prepare_asset_series
        from src.features.calendar_seasonality import asset_bounds_for_interval

        interval_min = int(INTERVAL_TO_MIN[interval_label])
        bounds = asset_bounds_for_interval(parquet_root, interval_min).get(str(asset))
        if not bounds:
            return None
        end_ts = int(bounds["max_ts"])
        lookback_days = int(os.getenv("PIPELINE_SEASONALITY_VALIDATION_LOOKBACK_DAYS", "365"))
        start_ts = max(int(bounds["min_ts"]), end_ts - lookback_days * 86400)
        df = _prepare_asset_series(
            parquet_root=parquet_root,
            interval_label=interval_label,
            asset=str(asset),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            prefer_scalar_features_true_range=True,
        )
        if df.empty:
            return None
        score_cols = ["true_range"] if channel == "vol" else (["log_return"] if channel == "return" else ["true_range", "log_return"])
        scores = [_rolling_origin_score(df[c], period) for c in score_cols if c in df.columns]
        scores = [s for s in scores if s is not None]
        return float(sum(scores) / len(scores)) if scores else None
    except Exception:
        return None


def _selection_from_payload(payload: Dict[str, Any]) -> Optional[SeasonalityProfile]:
    if not payload:
        return None
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        return None
    mode = str(selected.get("mode", "none"))
    if mode not in {"asset", "global", "none"}:
        return None
    usable = bool(selected.get("usable", False))
    period = selected.get("period_bars")
    try:
        period_i = int(period) if period is not None and int(period) > 1 else None
    except Exception:
        period_i = None
    return SeasonalityProfile(
        source=mode,
        usable=usable and mode != "none" and period_i is not None,
        seasonal_period_bars=period_i,
        path=selected.get("profile_path"),
        details={"selection": payload},
    )


def _build_selection(
    *,
    parquet_root: Path,
    seasonality_root: Path,
    interval_label: str,
    asset: str,
    refresh_days: int,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = [
        {"mode": "none", "period_bars": None, "period_channel": None, "profile_path": None, "artifact_quality": 0.0}
    ]
    for mode, path in (
        ("asset", _profile_path(seasonality_root, interval_label, asset=asset)),
        ("global", _profile_path(seasonality_root, interval_label, asset=None)),
    ):
        df = _load_profile(path)
        if df is None:
            continue
        for cand in _candidate_periods_from_profile(df)[:4]:
            candidates.append(
                {
                    "mode": mode,
                    "period_bars": int(cand["period_bars"]),
                    "period_channel": cand.get("period_channel"),
                    "profile_path": str(path),
                    "artifact_quality": float(cand.get("quality_score", 0.0) or 0.0),
                    "artifact_usable": bool(cand.get("usable", False)),
                    "artifact_recommended_use": cand.get("recommended_use"),
                }
            )

    for cand in candidates:
        cand["validation_score"] = _score_period_candidate(
            parquet_root=parquet_root,
            interval_label=interval_label,
            asset=asset,
            period=cand.get("period_bars"),
            channel=cand.get("period_channel"),
        )

    scored = [c for c in candidates if c.get("validation_score") is not None]
    artifact_candidates = [c for c in candidates if c.get("mode") in {"asset", "global"} and c.get("period_bars") is not None]
    if scored:
        selected = sorted(
            scored,
            key=lambda c: (
                float(c["validation_score"]),
                0 if c["mode"] == "asset" else (1 if c["mode"] == "global" else 2),
                -float(c.get("artifact_quality", 0.0) or 0.0),
            ),
        )[0]
    elif artifact_candidates:
        selected = sorted(
            artifact_candidates,
            key=lambda c: (
                0 if c["mode"] == "asset" else 1,
                -float(c.get("artifact_quality", 0.0) or 0.0),
            ),
        )[0]
    else:
        selected = candidates[0]
    selected = dict(selected)
    selected["usable"] = selected.get("mode") in {"asset", "global"} and selected.get("period_bars") is not None
    return {
        "schema_version": 1,
        "computed_at": utc_now_iso(),
        "interval": interval_label,
        "asset": str(asset),
        "refresh_days": int(refresh_days),
        "selection_method": "rolling_origin_median_absolute_scaled_error",
        "selected": selected,
        "candidates": candidates,
    }


def resolve_seasonality_profile(parquet_root: Path, interval_minutes: int, asset: str) -> SeasonalityProfile:
    sroot = _seasonality_root(parquet_root)
    refresh_days = int(os.getenv("PIPELINE_SEASONALITY_REFRESH_DAYS", str(DEFAULT_SEASONALITY_REFRESH_DAYS)))
    lock_timeout_sec = float(os.getenv("PIPELINE_SEASONALITY_LOCK_TIMEOUT_SEC", "120"))
    lock_stale_sec = float(os.getenv("PIPELINE_SEASONALITY_LOCK_STALE_SEC", "900"))
    fallback_selection: Optional[SeasonalityProfile] = None
    for ilabel in interval_label_candidates(int(interval_minutes)):
        if ilabel in {"1H", "4H", "1D"}:
            sel_path = _selection_path(sroot, ilabel, asset)
            if _is_fresh(sel_path, refresh_days):
                selected = _selection_from_payload(_load_json(sel_path))
                if selected is not None and selected.usable:
                    return selected
                if selected is not None:
                    fallback_selection = selected
                    continue

            lock_path = _selection_lock_path(sel_path)
            if not _acquire_file_lock(lock_path, timeout_sec=lock_timeout_sec, stale_sec=lock_stale_sec):
                raise TimeoutError(
                    "timed out waiting for seasonality selection writer "
                    f"interval={ilabel} asset={asset} path={sel_path} lock={lock_path}"
                )
            try:
                if _is_fresh(sel_path, refresh_days):
                    selected = _selection_from_payload(_load_json(sel_path))
                    if selected is not None and selected.usable:
                        return selected
                    if selected is not None:
                        fallback_selection = selected
                        continue

                _ensure_asset_profile(
                    parquet_root=parquet_root,
                    seasonality_root=sroot,
                    interval_label=ilabel,
                    asset=str(asset),
                    refresh_days=refresh_days,
                )
                payload = _build_selection(
                    parquet_root=parquet_root,
                    seasonality_root=sroot,
                    interval_label=ilabel,
                    asset=str(asset),
                    refresh_days=refresh_days,
                )
                _write_json_local(sel_path, payload)
                selected = _selection_from_payload(payload)
                if selected is not None and selected.usable:
                    return selected
                if selected is not None:
                    fallback_selection = selected
                    continue
            finally:
                _release_file_lock(lock_path)

        asset_path = _profile_path(sroot, ilabel, asset=asset)
        g_path = _profile_path(sroot, ilabel, asset=None)

        asset_df = _load_profile(asset_path)
        if asset_df is not None:
            usable, period, details = _infer_profile(asset_df)
            if usable:
                return SeasonalityProfile(
                    source="asset",
                    usable=True,
                    seasonal_period_bars=period,
                    path=str(asset_path),
                    details=details,
                )

        global_df = _load_profile(g_path)
        if global_df is not None:
            usable, period, details = _infer_profile(global_df)
            if usable:
                return SeasonalityProfile(
                    source="global",
                    usable=True,
                    seasonal_period_bars=period,
                    path=str(g_path),
                    details=details,
                )

    return fallback_selection or SeasonalityProfile(source="none", usable=False, seasonal_period_bars=None, path=None, details={})


def warm_seasonality_profiles(parquet_root: Path, interval_minutes: Sequence[int], assets: Sequence[str]) -> Dict[str, Any]:
    warmed = 0
    failures: Dict[str, str] = {}
    for interval in sorted({int(v) for v in interval_minutes}):
        for asset in sorted({str(v) for v in assets}):
            key = f"{interval}:{asset}"
            try:
                resolve_seasonality_profile(parquet_root=parquet_root, interval_minutes=int(interval), asset=str(asset))
                warmed += 1
            except Exception as exc:
                failures[key] = f"{type(exc).__name__}: {exc}"
    return {"enabled": True, "warmed": int(warmed), "failed": failures}


def write_forecast_parts(
    monthly_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
    out_root: Path,
    family_tag: str,
    interval_minutes: int,
    task: str,
    horizon_minutes: int,
    run_id: str,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []

    def _row_needs_recompute(frame: pd.DataFrame) -> pd.Series:
        if "needs_recompute" in frame.columns:
            return frame["needs_recompute"].fillna(False).astype(bool)
        if "is_forward_filled" in frame.columns:
            return frame["is_forward_filled"].fillna(False).astype(bool)
        return pd.Series(False, index=frame.index)

    for (y, m), frames in sorted(monthly_frames.items()):
        if not frames:
            continue
        chunk = pd.concat(frames, ignore_index=True)
        for flag_col in ("is_forward_filled", "needs_recompute"):
            if flag_col not in chunk.columns:
                chunk[flag_col] = False
            else:
                chunk[flag_col] = chunk[flag_col].fillna(False).astype(bool)
        key_cols = ["asset", "ts"]
        for optional_col in ("horizon_min", "task"):
            if optional_col in chunk.columns:
                key_cols.append(optional_col)
        chunk = chunk.sort_values(["asset", "ts"]).drop_duplicates(subset=key_cols, keep="last")
        month_dir = out_root / f"{int(interval_minutes)}" / f"year={int(y)}" / f"month={int(m):02d}"
        existing_files = sorted(month_dir.glob("*.parquet")) if month_dir.exists() else []
        replacement_keys: set[tuple[str, int, int, str]] = set()
        if existing_files:
            seen_finalized_keys: set[tuple[str, int, int, str]] = set()
            seen_all_keys: set[tuple[str, int, int, str]] = set()
            for p in existing_files:
                try:
                    ex = pd.read_parquet(p)
                except Exception:
                    continue
                if ex.empty or not {"asset", "ts"}.issubset(str(c) for c in ex.columns):
                    continue
                ex["asset"] = ex["asset"].astype(str)
                ex["ts"] = pd.to_numeric(ex["ts"], errors="coerce")
                ex["horizon_min"] = pd.to_numeric(ex["horizon_min"], errors="coerce") if "horizon_min" in ex.columns else int(horizon_minutes)
                ex["task"] = ex["task"].astype(str) if "task" in ex.columns else str(task)
                ex = ex.dropna(subset=["ts", "horizon_min"]).copy()
                for a, t, h, tk in zip(ex["asset"], ex["ts"], ex["horizon_min"], ex["task"]):
                    seen_all_keys.add((str(a), int(t), int(h), str(tk)))
                if "needs_recompute" in ex.columns or "is_forward_filled" in ex.columns:
                    ex = ex.loc[~_row_needs_recompute(ex)].copy()
                for a, t, h, tk in zip(ex["asset"], ex["ts"], ex["horizon_min"], ex["task"]):
                    seen_finalized_keys.add((str(a), int(t), int(h), str(tk)))
            if seen_finalized_keys or seen_all_keys:
                incoming_recompute = _row_needs_recompute(chunk)
                chunk_h = pd.to_numeric(chunk["horizon_min"], errors="coerce").fillna(int(horizon_minutes)).astype("int64") if "horizon_min" in chunk.columns else pd.Series(int(horizon_minutes), index=chunk.index, dtype="int64")
                chunk_task = chunk["task"].astype(str) if "task" in chunk.columns else pd.Series(str(task), index=chunk.index)
                keep_mask = []
                for a, t, h, tk, needs_recompute in zip(
                    chunk["asset"].astype(str),
                    pd.to_numeric(chunk["ts"], errors="coerce").astype("int64"),
                    chunk_h,
                    chunk_task,
                    incoming_recompute.astype(bool),
                ):
                    key = (str(a), int(t), int(h), str(tk))
                    keep_mask.append(key not in (seen_all_keys if bool(needs_recompute) else seen_finalized_keys))
                chunk = chunk.loc[keep_mask].copy()
                if chunk.empty:
                    continue
                incoming_recompute = _row_needs_recompute(chunk)
                chunk_h = pd.to_numeric(chunk["horizon_min"], errors="coerce").fillna(int(horizon_minutes)).astype("int64") if "horizon_min" in chunk.columns else pd.Series(int(horizon_minutes), index=chunk.index, dtype="int64")
                chunk_task = chunk["task"].astype(str) if "task" in chunk.columns else pd.Series(str(task), index=chunk.index)
                replacement_keys = {
                    (str(a), int(t), int(h), str(tk))
                    for a, t, h, tk, needs_recompute in zip(
                        chunk["asset"].astype(str),
                        pd.to_numeric(chunk["ts"], errors="coerce").astype("int64"),
                        chunk_h,
                        chunk_task,
                        incoming_recompute.astype(bool),
                    )
                    if not bool(needs_recompute)
                }
        if replacement_keys and existing_files:
            preserved_frames: List[pd.DataFrame] = []
            rewritten_any = False
            for p in existing_files:
                assert_write_allowed(p, "Stats forecast parquet read/delete", roots=_sandbox_roots())
                try:
                    ex_full = pd.read_parquet(p)
                except Exception:
                    continue
                if ex_full.empty or not {"asset", "ts"}.issubset(str(c) for c in ex_full.columns):
                    continue
                ex_full["asset"] = ex_full["asset"].astype(str)
                ex_full["ts"] = pd.to_numeric(ex_full["ts"], errors="coerce")
                ex_full["horizon_min"] = pd.to_numeric(ex_full["horizon_min"], errors="coerce") if "horizon_min" in ex_full.columns else int(horizon_minutes)
                ex_full["task"] = ex_full["task"].astype(str) if "task" in ex_full.columns else str(task)
                ex_full = ex_full.dropna(subset=["ts", "horizon_min"]).copy()
                ex_keys = [
                    (str(a), int(t), int(h), str(tk))
                    for a, t, h, tk in zip(ex_full["asset"], ex_full["ts"], ex_full["horizon_min"], ex_full["task"])
                ]
                keep_existing = [key not in replacement_keys for key in ex_keys]
                if not all(keep_existing):
                    rewritten_any = True
                kept = ex_full.loc[keep_existing].copy()
                if not kept.empty:
                    preserved_frames.append(kept)
            if rewritten_any:
                if preserved_frames:
                    chunk = pd.concat([*preserved_frames, chunk], ignore_index=True)
                    chunk = chunk.sort_values(["asset", "ts"]).drop_duplicates(subset=key_cols, keep="last")
                for p in existing_files:
                    assert_write_allowed(p, "Stats forecast parquet delete", roots=_sandbox_roots())
                    try:
                        p.unlink()
                    except Exception:
                        pass
        dst = (
            out_root
            / f"{int(interval_minutes)}"
            / f"year={int(y)}"
            / f"month={int(m):02d}"
            / (
                f"part-{family_tag}_{int(interval_minutes)}-{int(y)}{int(m):02d}-"
                f"{task}-h{int(horizon_minutes)}m-{run_id}.parquet"
            )
        )
        roots = _sandbox_roots()
        assert_write_allowed(dst, "Stats forecast parquet", roots=roots)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
        assert_write_allowed(tmp, "Stats forecast parquet temp", roots=roots)
        chunk.to_parquet(
            tmp,
            engine="pyarrow",
            compression=PARQUET_COMPRESSION,
            index=False,
            row_group_size=PARQUET_ROW_GROUP,
        )
        atomic_replace(tmp, dst)
        parts.append(
            {
                "path": str(dst),
                "rows": int(len(chunk)),
                "interval": int(interval_minutes),
                "task": str(task),
                "horizon_minutes": int(horizon_minutes),
                "year": int(y),
                "month": int(m),
                "min_ts": int(chunk["ts"].min()) if not chunk.empty else None,
                "max_ts": int(chunk["ts"].max()) if not chunk.empty else None,
                "assets": sorted(set(chunk["asset"].astype(str).tolist())),
            }
        )
    return parts


def _latest_interval_month_dir(out_root: Path, interval_minutes: int) -> Optional[Path]:
    interval_dir = Path(out_root) / f"{int(interval_minutes)}"
    if not interval_dir.exists():
        return None
    years: List[Tuple[int, Path]] = []
    for ydir in interval_dir.glob("year=*"):
        try:
            y = int(str(ydir.name).split("=", 1)[1])
        except Exception:
            continue
        years.append((int(y), ydir))
    years.sort(key=lambda x: x[0], reverse=True)
    for _y, ydir in years:
        months: List[Tuple[int, Path]] = []
        for mdir in ydir.glob("month=*"):
            try:
                m = int(str(mdir.name).split("=", 1)[1])
            except Exception:
                continue
            if 1 <= int(m) <= 12:
                months.append((int(m), mdir))
        months.sort(key=lambda x: x[0], reverse=True)
        for _m, mdir in months:
            if any(mdir.glob("*.parquet")):
                return mdir
    return None


def forecast_parts_tail_ts(
    *,
    out_root: Path,
    interval_minutes: int,
    family_tag: str,
    task: str,
    horizon_minutes: int,
    asset: str,
) -> Optional[int]:
    """
    Resolve the completed destination tail for one forecast unit.

    New stats outputs include the shared numeric metadata columns, so they can
    use the same gap/recompute semantics as the mature numeric families. Older
    legacy stats partitions are still tolerated by falling back to filename and
    asset filtering.
    """
    interval_root = Path(out_root) / f"{int(interval_minutes)}"
    if not interval_root.exists():
        return None
    pat = f"part-{str(family_tag)}_{int(interval_minutes)}-*-{str(task)}-h{int(horizon_minutes)}m-*.parquet"
    completed_ts: List[int] = []
    legacy_ts: List[int] = []
    key_cols = {"ts", "asset", "task", "horizon_min", "interval_min", "run_id", "model_id", "model_version"}
    for p in sorted(interval_root.glob("year=*/month=*/*.parquet"), key=lambda q: str(q).lower()):
        if not p.match(f"**/{pat}"):
            continue
        try:
            d = pd.read_parquet(p)
        except Exception:
            continue
        if d.empty:
            continue
        cols = set(str(c) for c in d.columns)
        if not {"asset", "ts"}.issubset(cols):
            continue
        d = d[d["asset"].astype(str) == str(asset)].copy()
        if d.empty:
            continue
        has_shared_meta = {"task", "horizon_min"}.issubset(cols)
        if has_shared_meta:
            d = d[
                (d["task"].astype(str) == str(task))
                & (pd.to_numeric(d["horizon_min"], errors="coerce").fillna(-1).astype("int64") == int(horizon_minutes))
            ]
            if d.empty:
                continue
            valid_mask = pd.to_numeric(d["ts"], errors="coerce").notna()
            if "needs_recompute" in d.columns:
                valid_mask = valid_mask & ~d["needs_recompute"].astype(bool)
            if "is_forward_filled" in d.columns:
                valid_mask = valid_mask & ~d["is_forward_filled"].astype(bool)
            value_cols = [
                str(col)
                for col in d.columns
                if str(col) not in key_cols and pd.api.types.is_numeric_dtype(d[col])
            ]
            if value_cols:
                valid_mask = valid_mask & d.loc[:, value_cols].replace([float("inf"), float("-inf")], pd.NA).notna().all(axis=1)
            ts = pd.to_numeric(d.loc[valid_mask, "ts"], errors="coerce").dropna().astype("int64")
            completed_ts.extend(int(x) for x in ts.tolist())
            continue
        ts = pd.to_numeric(d["ts"], errors="coerce").dropna().astype("int64")
        if ts.empty:
            continue
        legacy_ts.extend(int(x) for x in ts.tolist())
    if completed_ts:
        ordered = sorted(set(int(x) for x in completed_ts))
        step = int(interval_minutes) * 60
        expected = int(ordered[0])
        last_complete: Optional[int] = None
        for ts_i in ordered:
            if int(ts_i) != int(expected):
                break
            last_complete = int(ts_i)
            expected += int(step)
        return int(last_complete) if last_complete is not None else None
    return max(legacy_ts) if legacy_ts else None


def select_make_do_window(
    valid_train_points: int,
    train_windows_bars: Sequence[int],
    min_train_bars: int,
) -> Optional[int]:
    n = int(valid_train_points)
    ladder = sorted({int(w) for w in train_windows_bars if int(w) > 0})
    if n < int(min_train_bars):
        return None
    choices = [w for w in ladder if w <= n]
    if choices:
        return int(choices[-1])
    return int(n)
