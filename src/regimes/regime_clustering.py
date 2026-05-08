from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import hashlib
import inspect
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.runtime_config import get_workers, log_resolved_runtime
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

from src.features.scalar_features import PARQUET_ROOT, PARQUET_COMPRESSION, PARQUET_ROW_GROUP, feature_max_ts_from_parquet, log as base_log


def _patch_check_array_compat() -> None:
    """
    Compatibility shim for environments where sklearn renamed force_all_finite
    to ensure_all_finite, while hdbscan still calls force_all_finite.
    """
    try:
        from sklearn.utils import validation as sk_validation  # type: ignore
        import sklearn.utils as sk_utils  # type: ignore
    except Exception:
        return
    try:
        sig = inspect.signature(sk_validation.check_array)
    except Exception:
        return
    if "force_all_finite" in sig.parameters:
        return
    orig = sk_validation.check_array
    try:
        orig_sig = inspect.signature(orig)
    except Exception:
        orig_sig = None

    def check_array_compat(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and (orig_sig is None or "ensure_all_finite" in orig_sig.parameters):
            kwargs.setdefault("ensure_all_finite", force_all_finite)
        return orig(*args, **kwargs)

    sk_validation.check_array = check_array_compat  # type: ignore
    try:
        sk_utils.check_array = check_array_compat  # type: ignore
    except Exception:
        pass
    if hdbscan is not None:
        for mod_name in ("hdbscan_", "prediction", "flat"):
            try:
                mod = getattr(hdbscan, mod_name, None)
                if mod is not None and hasattr(mod, "check_array"):
                    setattr(mod, "check_array", check_array_compat)
            except Exception:
                continue


_patch_check_array_compat()


PIPELINE_PROFILE = selected_profile(default="production")
LOG_DIR = Path(resolve_path("log_root", profile=PIPELINE_PROFILE, required=False) or Path("logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "regime_clustering.log"

REGIME_PARQUET_ROOT = Path(
    resolve_path("source_regime_root", profile=PIPELINE_PROFILE, required=False)
    or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or Path("parquet")
)
DEFINITION_ROOT = Path(
    resolve_path("regime_definition_root", profile=PIPELINE_PROFILE, required=False)
    or Path("regime_definitions")
)
DEFINITION_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_INTERVALS = [1, 5, 15, 30, 60, 240, 720, 1440]
SECONDS_PER_DAY = 86400
HARD_OUTPUT_START_TS = int(pd.Timestamp("2021-01-01T00:00:00Z").timestamp())

DEFAULT_FEATURE_SUBSET = [
    "log_return",
    "atr_14",
    "rsi_14",
    "macd_hist_12_26_9",
    "adx_14",
    "ret_std_20",
    "cv_20",
    "vol_osc_pct_14_28",
    "vroc_14",
    "avg_trade_size",
    "trade_intensity",
    "prr",
]

CATEGORY_SPECS: Dict[str, Dict[str, object]] = {
    "trend": {
        "bases": ["log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"],
        "cluster_col": "trend_cluster_id",
        "label_col": "trend_label",
        "confidence_col": "trend_confidence_pct",
        "intensity_col": "trend_intensity_pct",
        "labels": ["down", "flat", "up"],
    },
    "vol": {
        "bases": ["atr_14", "ret_std_20", "cv_20", "vol_osc_pct_14_28"],
        "cluster_col": "vol_cluster_id",
        "label_col": "vol_label",
        "confidence_col": "vol_confidence_pct",
        "intensity_col": "vol_intensity_pct",
        "labels": ["low", "normal", "high"],
    },
    "activity": {
        "bases": ["trade_intensity", "avg_trade_size", "vroc_14", "prr"],
        "cluster_col": "activity_cluster_id",
        "label_col": "activity_label",
        "confidence_col": "activity_confidence_pct",
        "intensity_col": "activity_intensity_pct",
        "labels": ["low", "normal", "high"],
    },
}
CATEGORY_ORDER = ["trend", "vol", "activity"]
ALL_CATEGORY_BASES = sorted({b for spec in CATEGORY_SPECS.values() for b in spec["bases"]})
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_EPSILON_MICRO = 0.0
HDBSCAN_EPSILON_MESO = 0.0
HDBSCAN_EPSILON_MACRO = 0.0
FLAT_ACTIVITY_TOL = 1e-12
EPSILON_OVERRIDE: Optional[float] = None
CONFIDENCE_MAPPING = "soft"


@dataclass
class BandSpec:
    name: str
    ceiling: int
    member_intervals: List[int]
    train_days: int


BANDS = [
    BandSpec("micro", 30, [1, 5, 15, 30], 30),
    BandSpec("meso", 240, [60, 240], 180),
    BandSpec("macro", 1440, [720, 1440], 360),
]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    base_log(f"[regimes] {msg}")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    tmp = sibling_temp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def month_range(start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
    start_dt = pd.to_datetime(start_ts, unit="s", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="s", utc=True)
    cur = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=1, tz="UTC")
    end_marker = pd.Timestamp(year=end_dt.year, month=end_dt.month, day=1, tz="UTC")
    out: List[Tuple[int, int]] = []
    while cur <= end_marker:
        out.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return out


def assets_present_in_features(interval_min: int, root: Optional[Path] = None) -> List[str]:
    root_dir = Path(root) if root else PARQUET_ROOT
    base = root_dir / f"scalar_features_{int(interval_min)}"
    if not base.exists():
        return []
    assets: List[str] = []
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith("asset="):
            continue
        asset = child.name.split("=", 1)[1]
        if asset:
            assets.append(asset)
    return sorted(set(assets))


def ceiling_seconds(ceiling_min: int) -> int:
    return int(ceiling_min * 60)


def floor_to_ceiling(ts: int, ceiling_min: int) -> int:
    step = ceiling_seconds(ceiling_min)
    return int(ts // step * step)


def ceil_to_ceiling(ts: int, ceiling_min: int) -> int:
    step = ceiling_seconds(ceiling_min)
    if ts % step == 0:
        return int(ts)
    return int((ts // step + 1) * step)


def utc_week_key(ts: int) -> str:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    iso = dt.isocalendar()
    return f"{int(iso.year):04d}-{int(iso.week):02d}"


def utc_month_key(ts: int) -> str:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    return f"{int(dt.year):04d}-{int(dt.month):02d}"


def next_weekly_boundary_start_ts(ts: int) -> int:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    week_start = dt.normalize() - pd.Timedelta(days=int(dt.weekday()))
    boundary = week_start + pd.Timedelta(days=7)
    while boundary <= dt:
        boundary = boundary + pd.Timedelta(days=7)
    return int(boundary.timestamp())


def next_monthly_boundary_start_ts(ts: int) -> int:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    month_start = pd.Timestamp(year=dt.year, month=dt.month, day=1, tz="UTC")
    boundary = month_start + pd.DateOffset(months=1)
    while boundary <= dt:
        boundary = boundary + pd.DateOffset(months=1)
    return int(boundary.timestamp())


def cadence_refit_key(ts: int, band: BandSpec) -> str:
    if str(band.name).lower() == "micro":
        return utc_week_key(ts)
    return utc_month_key(ts)


def next_boundary_start_ts(ts: int, band: BandSpec) -> int:
    step = ceiling_seconds(band.ceiling)
    if str(band.name).lower() == "micro":
        raw = next_weekly_boundary_start_ts(ts)
    else:
        raw = next_monthly_boundary_start_ts(ts)
    boundary = ceil_to_ceiling(int(raw), band.ceiling)
    if boundary <= int(ts):
        boundary += int(step)
    return int(boundary)


def definition_valid_until_for_refit(refit_ts: int, band: BandSpec) -> int:
    return int(next_boundary_start_ts(int(refit_ts), band) - ceiling_seconds(band.ceiling))


def epsilon_for_band(band_name: str) -> float:
    if EPSILON_OVERRIDE is not None:
        return float(EPSILON_OVERRIDE)
    name = str(band_name).lower()
    if name == "micro":
        return float(HDBSCAN_EPSILON_MICRO)
    if name == "meso":
        return float(HDBSCAN_EPSILON_MESO)
    return float(HDBSCAN_EPSILON_MACRO)


def mcs_for_band(band_name: str) -> int:
    name = str(band_name).lower()
    if name == "micro":
        return 10
    return 15


def flat_activity_mask(df: pd.DataFrame, band: BandSpec, tol: float = FLAT_ACTIVITY_TOL) -> Tuple[pd.Series, bool]:
    c_log = f"i{band.ceiling}_log_return"
    c_ti = f"i{band.ceiling}_trade_intensity"
    c_prr = f"i{band.ceiling}_prr"
    if not all(c in df.columns for c in (c_log, c_ti, c_prr)):
        return pd.Series(False, index=df.index, dtype=bool), False
    activity = df[[c_log, c_ti, c_prr]].astype(float).abs()
    inactive_mask = (activity[c_log] <= tol) & (activity[c_ti] <= tol) & (activity[c_prr] <= tol)
    return inactive_mask.fillna(False).astype(bool), True


def _safe_float(val: Any) -> float:
    try:
        x = float(val)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x


def _series_to_float_dict(s: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in s.items():
        out[str(k)] = _safe_float(v)
    return out


def _pairwise_corr_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not cols:
        return {}
    sub = df[list(cols)].astype(float)
    corr = sub.corr()
    out: Dict[str, Dict[str, float]] = {}
    for r in corr.index:
        row: Dict[str, float] = {}
        for c in corr.columns:
            row[str(c)] = _safe_float(corr.loc[r, c])
        out[str(r)] = row
    return out


def _hist_quantile_from_counts(counts: Sequence[int], q: float) -> float:
    if not counts:
        return 0.0
    total = int(sum(int(x) for x in counts))
    if total <= 0:
        return 0.0
    q = float(min(1.0, max(0.0, q)))
    target = int(math.ceil(q * total))
    if target <= 0:
        target = 1
    csum = 0
    for i, c in enumerate(counts):
        csum += int(c)
        if csum >= target:
            return float(i)
    return float(len(counts) - 1)


class DiagnosticCollector:
    def __init__(self, asset: str):
        self.asset = str(asset)
        self._rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _entry(self, band: str, category: str) -> Dict[str, Any]:
        key = (str(band), str(category))
        if key not in self._rows:
            self._rows[key] = {
                "asset": self.asset,
                "band": str(band),
                "category": str(category),
                "bars_total": 0,
                "flat_bar_count": 0,
                "centroid_candidate_rows": 0,
                "overridden_to_neutral_rows": 0,
                "final_unknown_rows": 0,
                "centroid_assigned_rows": 0,
                "centroid_left_unknown_rows": 0,
                "no_model_or_no_cluster_rows": 0,
                "incomplete_feature_rows": 0,
                "confidence_count": 0,
                "confidence_sum": 0.0,
                "confidence_hist": [0] * 101,
                "feature_variance": {},
                "feature_std": {},
                "pairwise_correlation_matrix": {},
                "clustering": {
                    "clusters_found": 0,
                    "training_noise_fraction": 1.0,
                    "cluster_sizes": {},
                },
            }
        return self._rows[key]

    def record_fit(self, band: BandSpec, category: str, train_df: pd.DataFrame, feature_cols: Sequence[str], model: Optional[dict]) -> None:
        entry = self._entry(band.name, category)
        if feature_cols and not train_df.empty:
            cols_present = [c for c in feature_cols if c in train_df.columns]
            if cols_present:
                numeric = train_df[cols_present].astype(float)
                entry["feature_variance"] = _series_to_float_dict(numeric.var(ddof=0))
                entry["feature_std"] = _series_to_float_dict(numeric.std(ddof=0))
                entry["pairwise_correlation_matrix"] = _pairwise_corr_matrix(numeric, cols_present)
        if model is None:
            entry["clustering"] = {
                "clusters_found": 0,
                "training_noise_fraction": 1.0,
                "cluster_sizes": {},
            }
            return
        train_meta = model.get("train_meta", {}) if isinstance(model, dict) else {}
        labels_raw = train_meta.get("labels", np.array([], dtype=int))
        labels = np.asarray(labels_raw, dtype=int) if labels_raw is not None else np.array([], dtype=int)
        n = int(labels.size)
        noise_count = int(np.sum(labels == -1)) if n > 0 else 0
        cluster_sizes: Dict[str, int] = {}
        for cid in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
            cluster_sizes[str(cid)] = int(np.sum(labels == cid))
        entry["clustering"] = {
            "clusters_found": int(len(cluster_sizes)),
            "training_noise_fraction": float(noise_count / n) if n > 0 else 1.0,
            "cluster_sizes": cluster_sizes,
        }

    def record_assign(
        self,
        band: BandSpec,
        category: str,
        bars_total: int,
        flat_bar_count: int,
        centroid_candidate_rows: int,
        overridden_to_neutral_rows: int,
        final_unknown_rows: int,
        centroid_assigned_rows: int = 0,
        centroid_left_unknown_rows: int = 0,
        no_model_or_no_cluster_rows: int = 0,
        incomplete_feature_rows: int = 0,
        confidence_values: Optional[np.ndarray] = None,
    ) -> None:
        entry = self._entry(band.name, category)
        entry["bars_total"] += int(bars_total)
        entry["flat_bar_count"] += int(flat_bar_count)
        entry["centroid_candidate_rows"] += int(centroid_candidate_rows)
        entry["overridden_to_neutral_rows"] += int(overridden_to_neutral_rows)
        entry["final_unknown_rows"] += int(final_unknown_rows)
        entry["centroid_assigned_rows"] += int(centroid_assigned_rows)
        entry["centroid_left_unknown_rows"] += int(centroid_left_unknown_rows)
        entry["no_model_or_no_cluster_rows"] += int(no_model_or_no_cluster_rows)
        entry["incomplete_feature_rows"] += int(incomplete_feature_rows)
        if confidence_values is not None and int(len(confidence_values)) > 0:
            vals = np.asarray(confidence_values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                vals = np.clip(np.rint(vals), 0, 100).astype(int)
                hist = np.bincount(vals, minlength=101)
                entry["confidence_count"] += int(vals.size)
                entry["confidence_sum"] += float(vals.sum())
                cur_hist = entry.get("confidence_hist", [0] * 101)
                if not isinstance(cur_hist, list) or len(cur_hist) != 101:
                    cur_hist = [0] * 101
                entry["confidence_hist"] = [int(cur_hist[i]) + int(hist[i]) for i in range(101)]

    def to_json_ready(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (_band, _category), entry in sorted(self._rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            bars_total = int(entry["bars_total"])
            if bars_total <= 0:
                continue
            conf_count = int(entry.get("confidence_count", 0))
            conf_sum = float(entry.get("confidence_sum", 0.0))
            conf_hist = entry.get("confidence_hist", [0] * 101)
            if not isinstance(conf_hist, list) or len(conf_hist) != 101:
                conf_hist = [0] * 101
            rows.append(
                {
                    "asset": entry["asset"],
                    "band": entry["band"],
                    "category": entry["category"],
                    "bars_total": bars_total,
                    "flat_bar_fraction": float(entry["flat_bar_count"] / bars_total),
                    "feature_variance": dict(entry.get("feature_variance", {})),
                    "feature_std": dict(entry.get("feature_std", {})),
                    "pairwise_correlation_matrix": dict(entry.get("pairwise_correlation_matrix", {})),
                    "clustering": dict(entry.get("clustering", {})),
                    "assignments": {
                        "centroid_candidate_fraction": float(entry["centroid_candidate_rows"] / bars_total),
                        "overridden_to_neutral_fraction": float(entry["overridden_to_neutral_rows"] / bars_total),
                        "final_unknown_fraction": float(entry["final_unknown_rows"] / bars_total),
                        "centroid_assigned_fraction": float(entry["centroid_assigned_rows"] / bars_total),
                        "centroid_left_unknown_fraction": float(entry["centroid_left_unknown_rows"] / bars_total),
                        "no_model_or_no_cluster_fraction": float(entry["no_model_or_no_cluster_rows"] / bars_total),
                        "incomplete_feature_fraction": float(entry["incomplete_feature_rows"] / bars_total),
                        "mean_confidence": float(conf_sum / conf_count) if conf_count > 0 else 0.0,
                        "median_confidence": _hist_quantile_from_counts(conf_hist, 0.50) if conf_count > 0 else 0.0,
                        "p10_confidence": _hist_quantile_from_counts(conf_hist, 0.10) if conf_count > 0 else 0.0,
                        "p90_confidence": _hist_quantile_from_counts(conf_hist, 0.90) if conf_count > 0 else 0.0,
                    },
                }
            )
        return rows


def parquet_path_for(ceiling_interval: int, asset: str, year: int, month: int, root: Path) -> Path:
    return (
        root
        / f"regimes_{ceiling_interval}"
        / f"asset={asset}"
        / f"year={year}"
        / f"month={month:02d}"
        / "part-000.parquet"
    )


def _lock_path_for(dst: Path) -> Path:
    return dst.with_suffix(dst.suffix + ".lock")


def _acquire_lock(lock_path: Path, retries: int = 40, sleep_sec: float = 0.25) -> Optional[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            time.sleep(sleep_sec)
        except Exception:
            time.sleep(sleep_sec)
    return None


def _release_lock(fd: Optional[int], lock_path: Path) -> None:
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def write_parquet_atomic(df: pd.DataFrame, dst: Path, retries: int = 6, sleep_base_sec: float = 0.25) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path_for(dst)
    lock_fd = _acquire_lock(lock_path)
    if lock_fd is None:
        raise TimeoutError(f"Could not acquire parquet lock: {lock_path}")
    last_exc: Optional[Exception] = None
    try:
        for attempt in range(retries):
            tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
            try:
                out_df = df
                if dst.exists():
                    try:
                        existing = pd.read_parquet(dst)
                        out_df = pd.concat([existing, df], ignore_index=True)
                        if set(["asset", "ts", "band"]).issubset(out_df.columns):
                            out_df = out_df.drop_duplicates(subset=["asset", "ts", "band"], keep="last")
                    except Exception:
                        out_df = df
                out_df.to_parquet(tmp, engine="pyarrow", compression=PARQUET_COMPRESSION, index=False, row_group_size=PARQUET_ROW_GROUP)
                atomic_replace(tmp, dst)
                return
            except PermissionError as exc:
                last_exc = exc
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                if attempt == retries - 1:
                    break
                time.sleep(sleep_base_sec * (attempt + 1))
            except Exception:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                raise
        if last_exc is not None:
            raise last_exc
    finally:
        _release_lock(lock_fd, lock_path)


def definition_paths(asset: str, band: str, category: str) -> Tuple[Path, Path]:
    safe_asset = asset.replace("/", "_")
    safe_band = str(band).replace("/", "_")
    safe_category = str(category).replace("/", "_")
    stem = f"{safe_asset}__{safe_band}__{safe_category}"
    return DEFINITION_ROOT / f"{stem}.pkl", DEFINITION_ROOT / f"{stem}.json"


def load_definition(asset: str, band: str, category: str) -> Optional[dict]:
    model_path, meta_path = definition_paths(asset, band, category)
    if not model_path.exists() or not meta_path.exists():
        return None
    try:
        with model_path.open("rb") as f:
            model_obj = pickle.load(f)
        meta = read_json(meta_path)
        if not isinstance(model_obj, dict):
            return None
        model_obj["meta"] = meta
        return model_obj
    except Exception as exc:
        log(f"[definition][warn] failed loading asset={asset} band={band} category={category}: {exc}")
        return None


def save_definition(asset: str, band: str, category: str, model_obj: dict, meta: dict) -> None:
    model_path, meta_path = definition_paths(asset, band, category)
    with model_path.open("wb") as f:
        pickle.dump(model_obj, f)
    write_json(meta_path, meta)


def load_scalar_features_window(
    interval_min: int,
    asset: str,
    start_ts: int,
    end_ts: int,
    columns: Sequence[str],
    root: Optional[Path] = None,
) -> pd.DataFrame:
    root = Path(root) if root else PARQUET_ROOT
    base = root / f"scalar_features_{interval_min}"
    if not base.exists():
        raise RuntimeError(
            f"scalar_features base missing for interval={interval_min}; expected {base} "
            f"(asset-partitioned required)"
        )
    asset_base = base / f"asset={asset}"
    if not asset_base.exists():
        raise RuntimeError(
            f"scalar_features asset partition missing for interval={interval_min} asset={asset}; expected "
            f"{asset_base / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    scan_base = asset_base
    dfs: List[pd.DataFrame] = []
    for y, m in month_range(start_ts, end_ts):
        p = scan_base / f"year={y}/month={m:02d}"
        if not p.exists():
            continue
        for pq in p.glob("*.parquet"):
            try:
                df = pd.read_parquet(pq, columns=columns)
            except Exception:
                try:
                    read_cols = [c for c in columns if c != "asset"]
                    df = pd.read_parquet(pq, columns=read_cols)
                except Exception:
                    continue
            if "asset" not in df.columns:
                df["asset"] = asset
            df = df[df["asset"] == asset]
            if df.empty:
                continue
            df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
            if not df.empty:
                dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("ts")


def _latest_year_month_dir(base: Path) -> Optional[Tuple[int, int, Path]]:
    if not base.exists():
        return None
    years: List[int] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.split("=", 1)[1]))
            except Exception:
                continue
    if not years:
        return None
    year = max(years)
    year_dir = base / f"year={year}"
    months: List[int] = []
    if year_dir.exists():
        for child in year_dir.iterdir():
            if child.is_dir() and child.name.startswith("month="):
                try:
                    months.append(int(child.name.split("=", 1)[1]))
                except Exception:
                    continue
    if not months:
        return None
    month = max(months)
    return (int(year), int(month), year_dir / f"month={month:02d}")


def _oldest_year_month_dir(base: Path) -> Optional[Tuple[int, int, Path]]:
    if not base.exists():
        return None
    years: List[int] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.split("=", 1)[1]))
            except Exception:
                continue
    if not years:
        return None
    year = min(years)
    year_dir = base / f"year={year}"
    months: List[int] = []
    if year_dir.exists():
        for child in year_dir.iterdir():
            if child.is_dir() and child.name.startswith("month="):
                try:
                    months.append(int(child.name.split("=", 1)[1]))
                except Exception:
                    continue
    if not months:
        return None
    month = min(months)
    return (int(year), int(month), year_dir / f"month={month:02d}")


def _previous_year_month(year: int, month: int) -> Tuple[int, int]:
    if int(month) > 1:
        return int(year), int(month) - 1
    return int(year) - 1, 12


def _next_year_month(year: int, month: int) -> Tuple[int, int]:
    if int(month) < 12:
        return int(year), int(month) + 1
    return int(year) + 1, 1


def _one_parquet_in_month_dir(month_dir: Path) -> Optional[Path]:
    if not month_dir.exists():
        return None
    files = sorted(month_dir.glob("*.parquet"), key=lambda p: p.name.lower())
    return files[0] if files else None


def feature_bounds_from_parquet(interval_min: int, asset: str, root: Optional[Path] = None) -> Tuple[Optional[int], Optional[int]]:
    root_dir = Path(root) if root else PARQUET_ROOT
    asset_dir = root_dir / f"scalar_features_{int(interval_min)}" / f"asset={asset}"
    if not asset_dir.exists():
        return (None, None)

    max_ts = feature_max_ts_from_parquet(interval_min=int(interval_min), asset=str(asset), root=root_dir)
    oldest = _oldest_year_month_dir(asset_dir)
    if oldest is None:
        return (None, int(max_ts) if max_ts is not None else None)

    year, month, _ = oldest
    first_ts: Optional[int] = None
    for attempt in range(2):
        month_dir = asset_dir / f"year={year}" / f"month={month:02d}"
        part = _one_parquet_in_month_dir(month_dir)
        if part is not None and part.exists():
            try:
                df = pd.read_parquet(part, columns=["ts"])
                if not df.empty:
                    s = pd.to_numeric(df["ts"], errors="coerce").dropna().astype("int64")
                    if not s.empty:
                        first_ts = int(s.min())
                        break
            except Exception:
                pass
        if attempt == 0:
            year, month = _next_year_month(year, month)
    return (first_ts, int(max_ts) if max_ts is not None else None)


def regime_bounds_from_parquet(ceiling_interval: int, asset: str, root: Optional[Path] = None) -> Tuple[Optional[int], Optional[int]]:
    root = Path(root) if root else REGIME_PARQUET_ROOT
    asset_dir = root / f"regimes_{ceiling_interval}" / f"asset={asset}"
    latest = _latest_year_month_dir(asset_dir)
    if latest is None:
        return (None, None)
    year, month, _month_dir = latest
    for attempt in range(2):
        part = asset_dir / f"year={year}" / f"month={month:02d}" / "part-000.parquet"
        if part.exists():
            try:
                df = pd.read_parquet(part, columns=["ts"])
                if not df.empty:
                    return (None, int(df["ts"].max()))
            except Exception:
                pass
        if attempt == 0:
            year, month = _previous_year_month(year, month)
    return (None, None)


def select_feature_columns(
    df: pd.DataFrame,
    strategy: str,
    subset: Sequence[str],
    corr_thresh: float,
    member_intervals: Sequence[int],
) -> List[str]:
    numeric_cols = [c for c in df.columns if c not in ("ts", "asset")]
    if strategy == "subset":
        selected: List[str] = []
        for interval in member_intervals:
            prefix = f"i{interval}_"
            for base in subset:
                col = f"{prefix}{base}"
                if col in numeric_cols:
                    selected.append(col)
        return selected
    cols = numeric_cols
    if not cols:
        return []
    corr = df[cols].corr().abs()
    keep: List[str] = []
    for col in cols:
        if not keep:
            keep.append(col)
            continue
        if all(corr.loc[col, k] < corr_thresh for k in keep):
            keep.append(col)
    return keep


def discover_feature_columns(interval_min: int, asset: str) -> List[str]:
    base = PARQUET_ROOT / f"scalar_features_{interval_min}"
    if not base.exists():
        raise RuntimeError(
            f"scalar_features base missing for interval={interval_min}; expected {base} "
            f"(asset-partitioned required: {base / f'asset={asset}' / 'year=YYYY' / 'month=MM' / '<one parquet>'})"
        )
    root_dir = base / f"asset={asset}"
    if not root_dir.exists():
        raise RuntimeError(
            f"scalar_features asset partition missing for interval={interval_min} asset={asset}; expected "
            f"{root_dir / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    latest = _latest_year_month_dir(root_dir)
    if latest is None:
        raise RuntimeError(
            f"scalar_features asset partition has no year/month partitions for interval={interval_min} asset={asset}; "
            f"expected pattern {root_dir / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    year, month, _ = latest
    d0 = root_dir / f"year={year}" / f"month={month:02d}"
    py, pm = _previous_year_month(year, month)
    d1 = root_dir / f"year={py}" / f"month={pm:02d}"
    p0 = _one_parquet_in_month_dir(d0)
    p1 = _one_parquet_in_month_dir(d1)
    checked = [str(d0), str(d1)]
    pq = p0 if p0 is not None else p1
    if pq is None:
        raise RuntimeError(
            f"scalar_features parquet missing for interval={interval_min} asset={asset}; checked={checked}; "
            f"regime clustering requires asset-partitioned scalar_features."
        )
    try:
        cols = list(pd.read_parquet(pq, columns=None).columns)
    except Exception as exc:
        raise RuntimeError(
            f"scalar_features parquet unreadable for interval={interval_min} asset={asset}; path={pq}; error={exc}; "
            f"regime clustering requires asset-partitioned scalar_features."
        ) from exc
    discovered = [c for c in cols if c not in ("ts", "asset")]
    if not discovered:
        raise RuntimeError(
            f"scalar_features schema empty for interval={interval_min} asset={asset}; checked={checked}; "
            f"expected feature columns beyond ts/asset under asset-partitioned layout."
        )
    return discovered


def discover_feature_columns_for_intervals(intervals: Sequence[int], asset: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for interval in intervals:
        out[interval] = discover_feature_columns(interval, asset=asset)
    return out


def build_aligned_features(
    asset: str,
    band: BandSpec,
    start_ts: int,
    end_ts: int,
    feature_bases: Sequence[str],
) -> pd.DataFrame:
    if start_ts > end_ts:
        return pd.DataFrame()
    lookback_seconds = ceiling_seconds(band.ceiling)
    window_start = max(HARD_OUTPUT_START_TS, start_ts - lookback_seconds)
    available = discover_feature_columns_for_intervals(band.member_intervals, asset=asset)
    if not any(len(available.get(k, [])) > 0 for k in band.member_intervals):
        raise RuntimeError(
            f"no discovered scalar feature columns for asset={asset} band={band.name}; "
            f"expected asset-partitioned scalar_features under "
            f"{PARQUET_ROOT / f'scalar_features_{band.ceiling}' / f'asset={asset}' / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    interval_cols = {k: [c for c in feature_bases if c in available.get(k, [])] for k in band.member_intervals}
    if not any(len(interval_cols.get(k, [])) > 0 for k in band.member_intervals):
        raise RuntimeError(
            f"no usable discovered feature columns for asset={asset} band={band.name}; requested={list(feature_bases)}; "
            f"discovered={{{', '.join([f'{k}:{len(v)}' for k, v in available.items()])}}}"
        )
    ceiling_cols = interval_cols.get(band.ceiling, [])
    base_df = load_scalar_features_window(
        band.ceiling,
        asset,
        window_start,
        end_ts,
        columns=["ts", "asset", *ceiling_cols],
    )
    if base_df.empty:
        return pd.DataFrame()
    base_df = base_df[(base_df["ts"] >= HARD_OUTPUT_START_TS) & (base_df["ts"] >= start_ts) & (base_df["ts"] <= end_ts)]
    if base_df.empty:
        return pd.DataFrame()
    base_df = base_df.sort_values("ts")[["ts", "asset"] + [c for c in base_df.columns if c not in ("ts", "asset")]]
    base_df = base_df.rename(columns={c: f"i{band.ceiling}_{c}" for c in base_df.columns if c not in ("ts", "asset")})
    aligned = base_df[["ts", "asset"]].copy()
    aligned = aligned.sort_values("ts")
    if band.ceiling in band.member_intervals:
        for c in base_df.columns:
            if c not in ("ts", "asset"):
                aligned[c] = base_df[c].values
    for interval in band.member_intervals:
        if interval == band.ceiling:
            continue
        cols = interval_cols.get(interval, [])
        if not cols:
            continue
        df = load_scalar_features_window(
            interval,
            asset,
            window_start,
            end_ts,
            columns=["ts", "asset", *cols],
        )
        if df.empty:
            continue
        df = df[df["ts"] >= HARD_OUTPUT_START_TS]
        if df.empty:
            continue
        df = df.sort_values("ts")
        rename_cols = {c: f"i{interval}_{c}" for c in cols}
        df = df.rename(columns=rename_cols)
        merge_cols = ["ts"] + [rename_cols[c] for c in cols]
        aligned = pd.merge_asof(
            aligned.sort_values("ts"),
            df[merge_cols],
            on="ts",
            direction="backward",
            tolerance=lookback_seconds,
        )
    return aligned


def robust_scale_fit(df: pd.DataFrame, feature_cols: Sequence[str], standardize: bool = False) -> Tuple[object, np.ndarray, pd.DataFrame]:
    clean = df.dropna(subset=list(feature_cols)).copy()
    if clean.empty:
        return RobustScaler(), np.empty((0, len(feature_cols))), clean
    scaler = StandardScaler() if bool(standardize) else RobustScaler()
    x = scaler.fit_transform(clean[list(feature_cols)].astype(float).values)
    return scaler, x, clean


def fit_cluster_model(
    asset: str,
    band: BandSpec,
    category: str,
    train_start: int,
    train_end: int,
    feature_strategy: str,
    feature_subset: Sequence[str],
    corr_thresh: float,
    n_per_interval: int,
    min_cluster_size: int,
    min_samples: int,
    raw_train: Optional[pd.DataFrame] = None,
    diagnostics: Optional[DiagnosticCollector] = None,
    standardize: bool = False,
) -> Optional[dict]:
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for regime clustering.")
    if category not in CATEGORY_SPECS:
        raise RuntimeError(f"unknown category: {category}")

    # Keep API compatibility with existing call sites; category partition controls actual columns.
    _ = (feature_strategy, feature_subset, corr_thresh)
    if raw_train is None:
        raw_train = build_aligned_features(asset, band, train_start, train_end, ALL_CATEGORY_BASES)
    if raw_train.empty:
        log(f"[fit][warn] no training samples asset={asset} band={band.name} category={category}")
        return None

    effective_min_samples = int(HDBSCAN_MIN_SAMPLES)
    effective_epsilon = float(epsilon_for_band(band.name))
    bases = list(CATEGORY_SPECS[category]["bases"])
    feature_cols = [
        f"i{int(iv)}_{base}"
        for iv in band.member_intervals
        for base in bases
        if f"i{int(iv)}_{base}" in raw_train.columns
    ]
    if not feature_cols:
        log(f"[fit][warn] no feature columns selected asset={asset} band={band.name} category={category}")
        return None

    # Training-only inactivity filter using the existing flatness logic.
    rows_total = int(len(raw_train))
    rows_inactive = 0
    rows_active = rows_total
    filter_applied = False
    train_df_for_fit = raw_train
    inactive_mask, has_flat_cols = flat_activity_mask(raw_train, band)
    if has_flat_cols:
        rows_inactive = int(inactive_mask.fillna(False).sum())
        rows_active = int(rows_total - rows_inactive)
        filtered = raw_train.loc[~inactive_mask].copy()
        min_rows_needed = int(max(int(min_cluster_size) * 3, int(effective_min_samples) * 3, 100))
        if int(len(filtered)) >= int(min_rows_needed):
            train_df_for_fit = filtered
            filter_applied = True
            rows_active = int(len(filtered))
        else:
            train_df_for_fit = raw_train
            filter_applied = False
    log(
        f"[fit] asset={asset} band={band.name} category={category} refit_ts={int(train_end)} "
        f"rows={rows_total} inactive={rows_inactive} active={rows_active} filter_applied={filter_applied}"
    )
    log(
        f"[fit] asset={asset} band={band.name} category={category} refit_ts={int(train_end)} "
        f"min_cluster_size={int(min_cluster_size)} min_samples={int(effective_min_samples)} "
        f"cluster_selection_method=eom epsilon={effective_epsilon} metric=euclidean prediction_data=True"
    )

    if len(train_df_for_fit) > n_per_interval:
        train_df_for_fit = train_df_for_fit.sample(n=n_per_interval, random_state=17)

    scaler, x_scaled, clean = robust_scale_fit(train_df_for_fit, feature_cols, standardize=bool(standardize))
    if x_scaled.size == 0:
        log(f"[fit][warn] no clean samples after scaling asset={asset} band={band.name} category={category}")
        if diagnostics is not None:
            diagnostics.record_fit(band, category, train_df_for_fit, feature_cols, None)
        return None
    n_points = int(x_scaled.shape[0])
    min_required = int(max(2, effective_min_samples, min_cluster_size))
    if n_points < min_required:
        log(
            f"[fit][warn] insufficient points asset={asset} band={band.name} category={category} "
            f"n={n_points} required={min_required}"
        )
        if diagnostics is not None:
            diagnostics.record_fit(band, category, train_df_for_fit, feature_cols, None)
        return None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        cluster_selection_method="eom",
        cluster_selection_epsilon=effective_epsilon,
        metric="euclidean",
        allow_single_cluster=True,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x_scaled)
    has_clusters = bool(np.any(labels != -1))
    probabilities = clusterer.probabilities_
    outlier_scores = getattr(clusterer, "outlier_scores_", np.full(len(labels), np.nan))

    mapping = build_cluster_mapping(category, labels, clean, feature_cols)
    feature_schema_hash = schema_hash(feature_cols)
    depth_stats = compute_cluster_depth_stats(labels, x_scaled)
    centroids: Dict[int, np.ndarray] = {}
    for cid in sorted({int(l) for l in labels if int(l) != -1}):
        mask = labels == cid
        if np.any(mask):
            centroids[int(cid)] = np.nanmean(x_scaled[mask], axis=0)

    model_out = {
        "category": category,
        "band": str(band.name),
        "clusterer": clusterer,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "feature_schema_hash": feature_schema_hash,
        "mapping": mapping,
        "depth_stats": depth_stats,
        "centroids": centroids,
        "has_clusters": has_clusters,
        "train_meta": {
            "train_start": int(train_start),
            "train_end": int(train_end),
            "n_samples": int(len(labels)),
            "labels": labels,
            "probabilities": probabilities,
            "outlier_scores": outlier_scores,
        },
        "hdbscan_params": {
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "cluster_selection_epsilon": float(effective_epsilon),
            "min_samples": int(effective_min_samples),
            "min_cluster_size": int(min_cluster_size),
            "prediction_data": True,
            "standardize": bool(standardize),
        },
    }
    if diagnostics is not None:
        diagnostics.record_fit(band, category, train_df_for_fit, feature_cols, model_out)
    return model_out


def schema_hash(feature_cols: Sequence[str]) -> str:
    joined = ",".join(feature_cols).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def build_cluster_mapping(
    category: str,
    labels: np.ndarray,
    raw_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[int, str]:
    cluster_ids = sorted({int(l) for l in labels if l != -1})
    if not cluster_ids:
        return {}
    scores: Dict[int, float] = {}
    for cid in cluster_ids:
        mask = labels == cid
        if not np.any(mask):
            scores[cid] = float("nan")
            continue
        vals = raw_df.iloc[mask][list(feature_cols)].to_numpy(dtype=float)
        scores[cid] = float(np.nanmedian(vals))

    valid = [(cid, s) for cid, s in scores.items() if np.isfinite(s)]
    if not valid:
        return {cid: "unknown" for cid in scores.keys()}
    valid = sorted(valid, key=lambda x: x[1])
    out: Dict[int, str] = {cid: "unknown" for cid in scores.keys()}
    labels_vocab = list(CATEGORY_SPECS[category]["labels"])
    n = len(valid)
    if n == 1:
        out[valid[0][0]] = labels_vocab[1]
        return out
    if n == 2:
        low_cid = valid[0][0]
        high_cid = valid[1][0]
        out[low_cid] = labels_vocab[0]
        out[high_cid] = labels_vocab[2]
        return out
    t1 = int(math.floor(n / 3))
    t2 = int(math.floor(2 * n / 3))
    for idx, (cid, _score) in enumerate(valid):
        if idx < t1:
            rank = 0
        elif idx < t2:
            rank = 1
        else:
            rank = 2
        out[cid] = labels_vocab[rank]
    return out


def compute_cluster_depth_stats(labels: np.ndarray, x_scaled: np.ndarray) -> Dict[int, Dict[str, object]]:
    stats: Dict[int, Dict[str, object]] = {}
    for cid in sorted({int(l) for l in labels if l != -1}):
        mask = labels == cid
        if not np.any(mask):
            continue
        cluster_points = x_scaled[mask]
        center = np.nanmean(cluster_points, axis=0)
        dists = np.linalg.norm(cluster_points - center, axis=1)
        p50 = float(np.nanpercentile(dists, 50)) if dists.size else float("nan")
        p95 = float(np.nanpercentile(dists, 95)) if dists.size else float("nan")
        stats[cid] = {"center": center, "p50": p50, "p95": p95}
    return stats


def assign_category_clusters(
    df: pd.DataFrame,
    model: dict,
) -> pd.DataFrame:
    category = str(model.get("category"))
    spec = CATEGORY_SPECS[category]
    cluster_col = str(spec["cluster_col"])
    label_col = str(spec["label_col"])
    confidence_col = str(spec["confidence_col"])
    intensity_col = str(spec["intensity_col"])
    feature_cols = model["feature_cols"]
    scaler = model["scaler"]
    mapping = model["mapping"]
    depth_stats = model.get("depth_stats", {})
    x = df[feature_cols].astype(float).values
    x_scaled = scaler.transform(x)
    centroids: Dict[int, np.ndarray] = model.get("centroids", {}) or {}
    p50_by_cid: Dict[int, float] = {}
    p95_by_cid: Dict[int, float] = {}
    for cid, info in (depth_stats.items() if isinstance(depth_stats, dict) else []):
        try:
            p50 = float((info or {}).get("p50", np.nan))
        except Exception:
            p50 = float("nan")
        try:
            p95 = float((info or {}).get("p95", np.nan))
        except Exception:
            p95 = float("nan")
        if np.isfinite(p50) and p50 > 0:
            p50_by_cid[int(cid)] = p50
        if np.isfinite(p95) and p95 > 0:
            p95_by_cid[int(cid)] = p95
    if not centroids:
        labels = np.full((x_scaled.shape[0],), -1, dtype=int)
        nearest_dist = np.full((x_scaled.shape[0],), np.nan, dtype=float)
    else:
        cids = sorted([int(k) for k in centroids.keys()])
        cmat = np.vstack([np.asarray(centroids[cid], dtype=float) for cid in cids])
        d2 = np.sum((x_scaled[:, None, :] - cmat[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        labels = np.asarray([cids[int(i)] for i in idx], dtype=int)
        nearest_dist = np.sqrt(d2[np.arange(x_scaled.shape[0]), idx])
    out = df[["ts", "asset"]].copy()
    out[cluster_col] = labels.astype(int)
    out[confidence_col] = 0
    out[intensity_col] = 0
    # Centroid-distance confidence/intensity: normalized nearest distance in scaled space.
    band_name = str(model.get("band", "")).strip().lower()
    conf_mode = str(CONFIDENCE_MAPPING).strip().lower()
    # Calibration exception from discovery: micro/activity retains current mapping.
    if band_name == "micro" and str(category).strip().lower() == "activity":
        conf_mode = "current"
    for cid in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
        mask = labels == cid
        if not np.any(mask):
            continue
        dists = nearest_dist[mask]
        p50 = float(p50_by_cid.get(int(cid), np.nan))
        p95 = float(p95_by_cid.get(int(cid), np.nan))
        if not np.isfinite(p50) or p50 <= 0:
            p50 = 1.0
        if not np.isfinite(p95) or p95 <= 0:
            p95 = 1.0
        if conf_mode in {"current", "linear"}:
            conf = 1.0 - (dists / p95)
            conf = np.clip(conf, 0.0, 1.0)
        elif conf_mode == "soft":
            conf = 1.0 / (1.0 + (dists / p50))
            conf = np.clip(conf, 0.0, 1.0)
        elif conf_mode == "exponential":
            conf = np.exp(-(dists / p50))
            conf = np.clip(conf, 0.0, 1.0)
        else:
            conf = 1.0 - (dists / p95)
            conf = np.clip(conf, 0.0, 1.0)
        score = (100 * conf).round().astype(int)
        out.loc[mask, confidence_col] = score
        out.loc[mask, intensity_col] = score
    out[label_col] = "unknown"
    for cid, label in mapping.items():
        out.loc[out[cluster_col] == int(cid), label_col] = str(label)
    out.loc[out[cluster_col] == -1, [label_col, confidence_col, intensity_col]] = ["unknown", 0, 0]
    return out


def unknown_category_frame(
    base: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    spec = CATEGORY_SPECS[category]
    cluster_col = str(spec["cluster_col"])
    label_col = str(spec["label_col"])
    confidence_col = str(spec["confidence_col"])
    intensity_col = str(spec["intensity_col"])
    out = base[["ts", "asset"]].copy()
    out[cluster_col] = -1
    out[label_col] = "unknown"
    out[confidence_col] = 0
    out[intensity_col] = 0
    return out


def combined_feature_schema_hash(models_by_category: Dict[str, Optional[dict]]) -> str:
    chunks = []
    for category in CATEGORY_ORDER:
        model = models_by_category.get(category)
        chunks.append(f"{category}:{(model or {}).get('feature_schema_hash', 'unknown')}")
    return schema_hash(chunks)


def assign_range_outputs(
    asset: str,
    band: BandSpec,
    models_by_category: Dict[str, Optional[dict]],
    start_ts: int,
    end_ts: int,
    diagnostics: Optional[DiagnosticCollector] = None,
) -> pd.DataFrame:
    step = ceiling_seconds(band.ceiling)
    ts0 = floor_to_ceiling(int(start_ts), band.ceiling)
    if ts0 < int(start_ts):
        ts0 += step
    if ts0 > int(end_ts):
        return pd.DataFrame()
    base = pd.DataFrame({"ts": np.arange(ts0, int(end_ts) + step, step, dtype=np.int64)})
    base["asset"] = asset
    df = build_aligned_features(asset, band, start_ts, end_ts, feature_bases=ALL_CATEGORY_BASES)
    if df.empty:
        merged = base
    else:
        merged = base.merge(df, on=["ts", "asset"], how="left")
    flat_mask, _has_flat_cols = flat_activity_mask(merged, band)
    flat_bar_count = int(flat_mask.sum())
    assigned = base[["ts", "asset"]].copy()
    for category in CATEGORY_ORDER:
        model = models_by_category.get(category)
        spec = CATEGORY_SPECS[category]
        cluster_col = str(spec["cluster_col"])
        label_col = str(spec["label_col"])
        centroid_candidate_rows = 0
        centroid_assigned_rows = 0
        centroid_left_unknown_rows = 0
        no_model_or_no_cluster_rows = 0
        incomplete_feature_rows = 0
        if model is None or not bool(model.get("has_clusters", False)):
            category_frame = unknown_category_frame(base, category)
            no_model_or_no_cluster_rows = int(len(base))
        else:
            feature_cols = list(model["feature_cols"])
            for c in feature_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
            complete_mask = ~merged[feature_cols].isna().any(axis=1)
            category_parts: List[pd.DataFrame] = []
            if complete_mask.any():
                pred_frame = assign_category_clusters(merged.loc[complete_mask].copy(), model)
                centroid_candidate_rows = int(len(pred_frame))
                centroid_assigned_rows = int((pred_frame[cluster_col] != -1).sum())
                centroid_left_unknown_rows = int((pred_frame[cluster_col] == -1).sum())
                category_parts.append(pred_frame)
            if (~complete_mask).any():
                incomplete_feature_rows = int((~complete_mask).sum())
                category_parts.append(unknown_category_frame(merged.loc[~complete_mask, ["ts", "asset"]], category))
            category_frame = pd.concat(category_parts, ignore_index=True).sort_values("ts").reset_index(drop=True)
        final_unknown_rows = int((category_frame[label_col].astype(str) == "unknown").sum())
        spec_conf_col = str(spec["confidence_col"])
        known_conf = category_frame.loc[
            category_frame[label_col].astype(str) != "unknown",
            spec_conf_col,
        ].to_numpy(dtype=float)
        if diagnostics is not None:
            diagnostics.record_assign(
                band=band,
                category=category,
                bars_total=int(len(base)),
                flat_bar_count=flat_bar_count,
                centroid_candidate_rows=centroid_candidate_rows,
                overridden_to_neutral_rows=0,
                final_unknown_rows=final_unknown_rows,
                centroid_assigned_rows=centroid_assigned_rows,
                centroid_left_unknown_rows=centroid_left_unknown_rows,
                no_model_or_no_cluster_rows=no_model_or_no_cluster_rows,
                incomplete_feature_rows=incomplete_feature_rows,
                confidence_values=known_conf,
            )
        assigned = assigned.merge(category_frame, on=["ts", "asset"], how="left")
    assigned["band"] = band.name
    assigned["ceiling_interval_min"] = band.ceiling
    assigned["feature_schema_hash"] = combined_feature_schema_hash(models_by_category)
    return assigned


def build_unknown_outputs(asset: str, band: BandSpec, start_ts: int, end_ts: int, feature_schema_hash: str = "unknown") -> pd.DataFrame:
    step = ceiling_seconds(band.ceiling)
    ts0 = floor_to_ceiling(int(start_ts), band.ceiling)
    if ts0 < int(start_ts):
        ts0 += step
    if ts0 > int(end_ts):
        return pd.DataFrame()
    out = pd.DataFrame({"ts": np.arange(ts0, int(end_ts) + step, step, dtype=np.int64)})
    out["asset"] = asset
    out["band"] = band.name
    out["ceiling_interval_min"] = band.ceiling
    for category in CATEGORY_ORDER:
        spec = CATEGORY_SPECS[category]
        out[str(spec["cluster_col"])] = -1
        out[str(spec["label_col"])] = "unknown"
        out[str(spec["confidence_col"])] = 0
        out[str(spec["intensity_col"])] = 0
    out["feature_schema_hash"] = feature_schema_hash
    return out


def write_regime_outputs(
    df: pd.DataFrame,
    ceiling_interval: int,
    root: Path,
    on_chunk_committed: Optional[Callable[[int], None]] = None,
    expected_start_ts: Optional[int] = None,
    expected_end_ts: Optional[int] = None,
    expected_asset: Optional[str] = None,
) -> int:
    if df.empty:
        return 0
    df = df[df["ts"] >= HARD_OUTPUT_START_TS]
    if df.empty:
        return 0
    ts_dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.copy()
    df["year"] = ts_dt.dt.year
    df["month"] = ts_dt.dt.month
    rows = 0
    step = ceiling_seconds(int(ceiling_interval))
    for (y, m), grp in df.groupby(["year", "month"]):
        out = grp.drop(columns=["year", "month"]).drop_duplicates(subset=["asset", "ts", "band"], keep="last")
        out = out.sort_values("ts").reset_index(drop=True)
        if expected_asset is not None:
            out = out[out["asset"].astype(str) == str(expected_asset)].copy()
        if out.empty:
            continue
        ts_arr = pd.to_numeric(out["ts"], errors="coerce").dropna().astype("int64").to_numpy()
        if ts_arr.size == 0:
            continue
        if ts_arr.size > 1:
            d = np.diff(ts_arr)
            if not np.all(d == int(step)):
                first_bad = int(ts_arr[np.where(d != int(step))[0][0] + 1])
                raise RuntimeError(
                    f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                    f"month={int(y):04d}-{int(m):02d} first_bad_ts={first_bad} error=non_step_grid"
                )
            if not np.all(d > 0):
                first_bad = int(ts_arr[np.where(d <= 0)[0][0] + 1])
                raise RuntimeError(
                    f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                    f"month={int(y):04d}-{int(m):02d} first_bad_ts={first_bad} error=non_increasing_ts"
                )
        month_start = int(pd.Timestamp(year=int(y), month=int(m), day=1, tz="UTC").timestamp())
        if int(m) == 12:
            month_end = int(pd.Timestamp(year=int(y) + 1, month=1, day=1, tz="UTC").timestamp()) - int(step)
        else:
            month_end = int(pd.Timestamp(year=int(y), month=int(m) + 1, day=1, tz="UTC").timestamp()) - int(step)
        intended_start = max(int(ts_arr[0]), int(expected_start_ts)) if expected_start_ts is not None else int(ts_arr[0])
        intended_end = min(int(ts_arr[-1]), int(expected_end_ts)) if expected_end_ts is not None else int(ts_arr[-1])
        intended_start = max(intended_start, month_start)
        intended_end = min(intended_end, month_end)
        if int(ts_arr[0]) != int(intended_start):
            raise RuntimeError(
                f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                f"month={int(y):04d}-{int(m):02d} first_bad_ts={int(ts_arr[0])} "
                f"error=head_mismatch expected={int(intended_start)}"
            )
        if int(ts_arr[-1]) != int(intended_end):
            raise RuntimeError(
                f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                f"month={int(y):04d}-{int(m):02d} first_bad_ts={int(ts_arr[-1])} "
                f"error=tail_mismatch expected={int(intended_end)}"
            )
        out_asset = str(out["asset"].iloc[0])
        dst = parquet_path_for(ceiling_interval, out_asset, int(y), int(m), root)
        write_parquet_atomic(out, dst)
        validate_attempt = 0
        while True:
            validate_attempt += 1
            try:
                chk = pd.read_parquet(dst, columns=["asset", "ts", "band"])
                if expected_asset is not None:
                    chk = chk[chk["asset"].astype(str) == str(expected_asset)]
                chk = chk[chk["band"].astype(str) == str(out["band"].iloc[0])]
                chk = chk.sort_values("ts")
                if chk.empty:
                    raise RuntimeError("empty_post_write")
                chk_ts = pd.to_numeric(chk["ts"], errors="coerce").dropna().astype("int64").to_numpy()
                if chk_ts.size == 0:
                    raise RuntimeError("empty_ts_post_write")
                if chk_ts.size > 1:
                    dchk = np.diff(chk_ts)
                    if not np.all(dchk > 0):
                        first_bad = int(chk_ts[np.where(dchk <= 0)[0][0] + 1])
                        raise RuntimeError(f"non_increasing_post_write first_bad_ts={first_bad}")
                    if not np.all(dchk == int(step)):
                        first_bad = int(chk_ts[np.where(dchk != int(step))[0][0] + 1])
                        raise RuntimeError(f"non_step_post_write first_bad_ts={first_bad}")
                if int(chk_ts[-1]) != int(ts_arr[-1]):
                    raise RuntimeError(f"tail_post_write_mismatch got={int(chk_ts[-1])} expected={int(ts_arr[-1])}")
                break
            except Exception as exc:
                if validate_attempt >= 2:
                    raise RuntimeError(
                        f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                        f"month={int(y):04d}-{int(m):02d} first_bad_ts={int(ts_arr[0])} error={exc}"
                    )
                write_parquet_atomic(out, dst)
        if on_chunk_committed is not None and not out.empty:
            on_chunk_committed(int(out["ts"].max()))
        rows += len(out)
        log(f"[parquet] regimes_{ceiling_interval} {y}-{int(m):02d}: wrote {len(out):,} rows -> {dst}")
    return rows


def definition_is_valid(model: Optional[dict], cursor_ts: int, cursor_refit_key: str, category: str) -> bool:
    if model is None or not isinstance(model, dict):
        return False
    if str(model.get("category", "")) != str(category):
        return False
    if "clusterer" not in model or "scaler" not in model:
        return False
    feature_cols = model.get("feature_cols", [])
    if not feature_cols:
        return False
    meta = model.get("meta", {})
    if not isinstance(meta, dict):
        return False
    valid_until = int(meta.get("valid_until", -1))
    refit_key = str(meta.get("refit_key", ""))
    if valid_until < int(cursor_ts):
        return False
    if refit_key != str(cursor_refit_key):
        return False
    return True


def walk_forward_asset_band(
    asset: str,
    band: BandSpec,
    feature_strategy: str,
    feature_subset: Sequence[str],
    corr_thresh: float,
    n_per_interval: int,
    min_cluster_size_override: Optional[int],
    min_samples: int,
    diagnostics: Optional[DiagnosticCollector] = None,
    standardize: bool = False,
) -> dict:
    min_ts, max_ts = feature_bounds_from_parquet(band.ceiling, asset)
    if min_ts is None or max_ts is None:
        log(f"[walk][skip] no source data asset={asset} band={band.name}")
        return {}
    src_min = max(int(min_ts), HARD_OUTPUT_START_TS)
    source_tail = floor_to_ceiling(int(max_ts), band.ceiling)
    if int(source_tail) < int(src_min):
        log(f"[walk][skip] no source data at/after hard edge asset={asset} band={band.name}")
        return {}

    _dst_min, dst_tail_raw = regime_bounds_from_parquet(band.ceiling, asset)
    if dst_tail_raw is not None and int(dst_tail_raw) > int(source_tail):
        raise RuntimeError(
            f"[hard-stop] asset={asset} band={band.name} dst_tail={int(dst_tail_raw)} ahead_of_source_tail={int(source_tail)}"
        )
    if dst_tail_raw is not None and int(dst_tail_raw) == int(source_tail):
        log(f"[walk][skip] at edge asset={asset} band={band.name} tail={int(source_tail)}")
        return {"last_assigned_ceiling_ts": int(source_tail), "last_refit_ceiling_ts": int(source_tail)}

    step = ceiling_seconds(band.ceiling)
    start_ts = ceil_to_ceiling(src_min, band.ceiling) if dst_tail_raw is None else int(dst_tail_raw) + int(step)
    end_ts = int(source_tail)
    if int(start_ts) > int(end_ts):
        log(f"[walk][skip] empty range asset={asset} band={band.name} start={int(start_ts)} end={int(end_ts)}")
        return {"last_assigned_ceiling_ts": int(dst_tail_raw or 0), "last_refit_ceiling_ts": int(dst_tail_raw or 0)}

    models_by_category: Dict[str, Optional[dict]] = {}
    for category in CATEGORY_ORDER:
        models_by_category[category] = load_definition(asset, band.name, category)

    cursor = int(start_ts)
    total_rows = 0
    last_refit_ts = int(dst_tail_raw) if dst_tail_raw is not None else int(start_ts)
    last_reason = "reused"
    band_mcs = int(min_cluster_size_override) if min_cluster_size_override is not None else int(mcs_for_band(band.name))

    while int(cursor) <= int(end_ts):
        cursor_refit_key = cadence_refit_key(int(cursor), band)
        boundary = next_boundary_start_ts(int(cursor), band)
        chunk_end = min(int(end_ts), int(boundary) - int(step))
        if int(chunk_end) < int(cursor):
            cursor = int(cursor) + int(step)
            continue

        reason = "reused"
        categories_to_refit = [
            category
            for category in CATEGORY_ORDER
            if not definition_is_valid(models_by_category.get(category), int(cursor), str(cursor_refit_key), category)
        ]
        if categories_to_refit:
            refit_ts = int(cursor)
            train_end = floor_to_ceiling(int(refit_ts), band.ceiling)
            train_start_raw = max(HARD_OUTPUT_START_TS, int(train_end) - int(band.train_days) * SECONDS_PER_DAY)
            train_start = ceil_to_ceiling(int(train_start_raw), band.ceiling)
            if int(train_start) > int(train_end):
                train_start = int(train_end)
            raw_train = build_aligned_features(asset, band, train_start, train_end, ALL_CATEGORY_BASES)
            reason = "fit_unavailable"
            if not raw_train.empty:
                refit_key = cadence_refit_key(int(refit_ts), band)
                model_valid_until = definition_valid_until_for_refit(int(refit_ts), band)
                any_ok = False
                for category in categories_to_refit:
                    model = fit_cluster_model(
                        asset,
                        band,
                        category,
                        train_start,
                        train_end,
                        feature_strategy,
                        feature_subset,
                        corr_thresh,
                        n_per_interval,
                        band_mcs,
                        min_samples,
                        raw_train=raw_train,
                        diagnostics=diagnostics,
                        standardize=bool(standardize),
                    )
                    if model is not None:
                        any_ok = True
                        meta = {
                            "asset": asset,
                            "band": band.name,
                            "category": category,
                            "valid_from": int(refit_ts),
                            "last_refit_ts": int(refit_ts),
                            "refit_key": str(refit_key),
                            "valid_until": int(model_valid_until),
                            "feature_cols": model["feature_cols"],
                            "feature_schema_hash": model["feature_schema_hash"],
                            "hdbscan_params": model.get("hdbscan_params", {}),
                        }
                        model["meta"] = meta
                        models_by_category[category] = model
                        save_definition(asset, band.name, category, model, meta)
                    else:
                        models_by_category[category] = None
                if any_ok:
                    reason = "ok"
                    last_refit_ts = int(refit_ts)
                else:
                    reason = "no_clusters"

        if any(m is not None and bool(m.get("has_clusters", False)) for m in models_by_category.values()):
            outputs = assign_range_outputs(
                asset,
                band,
                models_by_category,
                int(cursor),
                int(chunk_end),
                diagnostics=diagnostics,
            )
        else:
            outputs = build_unknown_outputs(
                asset,
                band,
                int(cursor),
                int(chunk_end),
                feature_schema_hash=combined_feature_schema_hash(models_by_category),
            )
            if diagnostics is not None and not outputs.empty:
                bars = int(len(outputs))
                for category in CATEGORY_ORDER:
                    diagnostics.record_assign(
                        band=band,
                        category=category,
                        bars_total=bars,
                        flat_bar_count=0,
                        centroid_candidate_rows=0,
                        overridden_to_neutral_rows=0,
                        final_unknown_rows=bars,
                        centroid_assigned_rows=0,
                        centroid_left_unknown_rows=bars,
                        no_model_or_no_cluster_rows=bars,
                        incomplete_feature_rows=0,
                        confidence_values=None,
                    )
        rows = write_regime_outputs(
            outputs,
            band.ceiling,
            REGIME_PARQUET_ROOT,
            expected_start_ts=int(cursor),
            expected_end_ts=int(chunk_end),
            expected_asset=asset,
        )
        total_rows += int(rows)
        tail_written = int(outputs["ts"].max()) if not outputs.empty else int(cursor) - int(step)
        if int(tail_written) != int(chunk_end):
            raise RuntimeError(
                f"[hard-stop] asset={asset} band={band.name} write_tail={int(tail_written)} expected_tail={int(chunk_end)}"
            )
        last_reason = reason
        cursor = int(chunk_end) + int(step)

    log(
        f"[walk] asset={asset} band={band.name} source_tail={int(source_tail)} dst_tail={dst_tail_raw} "
        f"range=[{int(start_ts)},{int(end_ts)}] rows={int(total_rows)} reason={last_reason}"
    )
    return {"last_assigned_ceiling_ts": int(end_ts), "last_refit_ceiling_ts": int(last_refit_ts)}


def process_asset(
    asset: str,
    band_specs: List[BandSpec],
    feature_strategy: str,
    feature_subset: Optional[List[str]],
    corr_thresh: float,
    n_per_interval: int,
    min_cluster_size_override: Optional[int],
    min_samples: int,
    standardize: bool = False,
) -> dict:
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for regime clustering.")
    band_states: dict = {}
    diagnostics = DiagnosticCollector(asset=str(asset))
    for band in band_specs:
        try:
            band_state = walk_forward_asset_band(
                asset,
                band,
                feature_strategy,
                feature_subset or DEFAULT_FEATURE_SUBSET,
                corr_thresh,
                n_per_interval,
                min_cluster_size_override,
                min_samples,
                diagnostics=diagnostics,
                standardize=bool(standardize),
            )
            if band_state:
                band_states[band.name] = band_state
        except Exception as exc:
            log(f"[worker][error] asset={asset} band={band.name}: {exc}")
    diagnostics_rows = diagnostics.to_json_ready()
    if diagnostics_rows:
        payload = {
            "asset": str(asset),
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": diagnostics_rows,
        }
        diagnostics_dir = REGIME_PARQUET_ROOT / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = diagnostics_dir / f"{asset}_regime_diagnostics.json"
        write_json(diagnostics_path, payload)
        for row in diagnostics_rows:
            log(
                "[diag] "
                f"asset={row['asset']} band={row['band']} category={row['category']} "
                f"bars_total={row['bars_total']} flat_bar_fraction={float(row['flat_bar_fraction']):.6f} "
                f"clusters_found={int(row.get('clustering', {}).get('clusters_found', 0))} "
                f"final_unknown_fraction={float(row.get('assignments', {}).get('final_unknown_fraction', 1.0)):.6f}"
            )
        log(f"[diag] asset={asset} wrote {len(diagnostics_rows)} category records -> {diagnostics_path}")
    return {"asset": asset, "band_states": band_states}


def run(
    assets: Optional[List[str]] = None,
    bands: Optional[List[str]] = None,
    feature_strategy: str = "subset",
    feature_subset: Optional[List[str]] = None,
    corr_thresh: float = 0.9,
    n_per_interval: int = 5000,
    min_cluster_size_override: Optional[int] = None,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    standardize: bool = False,
    workers: int = 1,
) -> None:
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for regime clustering.")
    feature_subset = feature_subset or DEFAULT_FEATURE_SUBSET
    band_specs = [b for b in BANDS if not bands or b.name in bands]
    if not assets:
        assets = sorted(assets_present_in_features(DEFAULT_INTERVALS[-1]))
    if workers <= 1:
        for asset in assets:
            process_asset(
                asset,
                band_specs,
                feature_strategy,
                feature_subset,
                corr_thresh,
                n_per_interval,
                min_cluster_size_override,
                min_samples,
                standardize,
            )
    else:
        pending_assets: List[str] = list(assets)
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_map = {}
                for asset in assets:
                    fut = pool.submit(
                        process_asset,
                        asset,
                        band_specs,
                        feature_strategy,
                        feature_subset,
                        corr_thresh,
                        n_per_interval,
                        min_cluster_size_override,
                        min_samples,
                        standardize,
                    )
                    future_map[fut] = asset
                for fut in as_completed(future_map):
                    asset = future_map[fut]
                    try:
                        fut.result()
                        if asset in pending_assets:
                            pending_assets.remove(asset)
                    except BrokenProcessPool as exc:
                        log(f"[run][warn] process pool broke ({exc}); retrying remaining assets serially")
                        break
                    except Exception as exc:
                        log(f"[run][worker-error] asset={asset} err={exc}")
                        if asset in pending_assets:
                            pending_assets.remove(asset)
        except BrokenProcessPool as exc:
            log(f"[run][warn] process pool failed to initialize ({exc}); running serially")
        if pending_assets:
            for asset in pending_assets:
                try:
                    process_asset(
                        asset,
                        band_specs,
                        feature_strategy,
                        feature_subset,
                        corr_thresh,
                        n_per_interval,
                        min_cluster_size_override,
                        min_samples,
                        standardize,
                    )
                except Exception as exc:
                    log(f"[run][serial-error] asset={asset} err={exc}")


def parse_list(val: Optional[str]) -> Optional[List[str]]:
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster scalar features into regime labels per asset/band.")
    parser.add_argument("--assets", type=str, default=None, help="Comma-separated assets (default: discover)")
    parser.add_argument("--bands", type=str, default=None, help="Comma-separated band names (micro,meso,macro)")
    parser.add_argument("--feature-strategy", type=str, default="subset", choices=["subset", "decorrelate"])
    parser.add_argument("--feature-subset", type=str, default=None, help="Comma-separated feature list")
    parser.add_argument("--corr-thresh", type=float, default=0.9)
    parser.add_argument("--n-per-interval", type=int, default=5000)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None, help="Alias override for min cluster size")
    parser.add_argument("--min-samples", type=int, default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--epsilon-override", type=float, default=None, help="Temporary global epsilon override for all bands")
    parser.add_argument(
        "--confidence-mapping",
        type=str,
        default=None,
        choices=["current", "linear", "soft", "exponential"],
        help="Centroid confidence mapping rule",
    )
    parser.add_argument("--standardize", action="store_true", help="Use z-score standardization instead of robust scaling")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count")
    args = parser.parse_args()
    env_mcs = os.getenv("REGIME_MCS", "").strip()
    resolved_mcs: Optional[int] = None
    if env_mcs:
        try:
            resolved_mcs = int(env_mcs)
        except Exception:
            resolved_mcs = None
    if args.min_cluster_size is not None:
        resolved_mcs = int(args.min_cluster_size)
    if args.hdbscan_min_cluster_size is not None:
        resolved_mcs = int(args.hdbscan_min_cluster_size)
    env_eps = os.getenv("REGIME_EPSILON_OVERRIDE", "").strip()
    resolved_eps: Optional[float] = None
    if env_eps:
        try:
            resolved_eps = float(env_eps)
        except Exception:
            resolved_eps = None
    if args.epsilon_override is not None:
        resolved_eps = float(args.epsilon_override)
    global EPSILON_OVERRIDE
    EPSILON_OVERRIDE = resolved_eps
    env_conf_map = str(os.getenv("REGIME_CONFIDENCE_MAPPING", "")).strip().lower()
    resolved_conf_map = "current"
    if env_conf_map in {"current", "linear", "soft", "exponential"}:
        resolved_conf_map = env_conf_map
    if args.confidence_mapping is not None:
        resolved_conf_map = str(args.confidence_mapping).strip().lower()
    global CONFIDENCE_MAPPING
    CONFIDENCE_MAPPING = resolved_conf_map
    env_standardize = str(os.getenv("REGIME_STANDARDIZE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    resolved_standardize = bool(args.standardize) or bool(env_standardize)
    workers = int(args.workers) if args.workers is not None else get_workers("regime_clustering", "asset_workers")
    log_resolved_runtime(
        "regime_clustering",
        resolved={
            "asset_workers": int(max(1, workers)),
            "writer_workers": 1,
            "model_threads": "n/a",
        },
    )
    run(
        assets=parse_list(args.assets),
        bands=parse_list(args.bands),
        feature_strategy=args.feature_strategy,
        feature_subset=parse_list(args.feature_subset),
        corr_thresh=args.corr_thresh,
        n_per_interval=args.n_per_interval,
        min_cluster_size_override=resolved_mcs,
        min_samples=args.min_samples,
        standardize=resolved_standardize,
        workers=workers,
    )


if __name__ == "__main__":
    main()
