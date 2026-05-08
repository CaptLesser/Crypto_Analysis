from __future__ import annotations
import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.pipeline_parquet_utils import (
    PipelineValidationError,
    decide_range_from_disk_edges as shared_decide_range_from_disk_edges,
    find_partition_month_dir as shared_find_partition_month_dir,
    partition_max_ts,
    validate_expected_grid as shared_validate_expected_grid,
    validate_no_nan_columns as shared_validate_no_nan_columns,
    validate_strict_timegrid as shared_validate_strict_timegrid,
)
from src.forecasting.common.ohlcvt_source import list_assets_ohlcvt, ohlcvt_bounds, read_ohlcvt
from src.forecasting.common.path_config import require_pipeline_io, resolve_path, selected_profile
from src.forecasting.common.runtime_config import get_workers, log_resolved_runtime
try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    njit = None  # type: ignore
    _HAS_NUMBA = False
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # psutil is optional; wait_for_resources will no-op if missing

_HAS_PYARROW_PQ = None  # resolved lazily to avoid hard import errors in some envs


# ------------------------------------------------
# Config & Paths
# ------------------------------------------------
PIPELINE_PROFILE = selected_profile()
LOG_DIR = resolve_path("log_root", profile=PIPELINE_PROFILE, required=False) or Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "scalar_features.log"
MANIFEST_DIR = LOG_DIR / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
STABILITY_REPORT_FILE = LOG_DIR / "feature_stability_report.json"

PARQUET_ROOT = Path(
    resolve_path("source_feature_root", profile=PIPELINE_PROFILE, required=False)
    or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or Path("parquet")
)
# Root for OHLCVT Parquets produced by live.py
OHLCVT_PARQUET_ROOT = Path(
    resolve_path("source_ohlcvt_root", profile=PIPELINE_PROFILE, required=False)
    or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or Path("parquet")
)
PARQUET_COMPRESSION = os.getenv("PIPELINE_PARQUET_COMPRESSION", "snappy")
PARQUET_ROW_GROUP = int(os.getenv("PIPELINE_PARQUET_ROW_GROUP", "500000"))
DENOM_FLOOR = float(os.getenv("SCALAR_DENOM_FLOOR", "1.0"))
MAX_FEATURE_ABS = float(os.getenv("SCALAR_MAX_FEATURE_ABS", "1e6"))
CHAIKIN_VOL_DENOM_FLOOR = float(os.getenv("SCALAR_CHAIKIN_VOL_DENOM_FLOOR", "50.0"))
CHAIKIN_VOL_MAX_ABS = float(os.getenv("SCALAR_CHAIKIN_VOL_MAX_ABS", "20000.0"))
STABILITY_WARN_ABS = float(os.getenv("SCALAR_STABILITY_WARN_ABS", str(MAX_FEATURE_ABS)))

DEFAULT_INTERVALS = [1, 5, 15, 30, 60, 240, 720, 1440]
TEST_CRASH_AFTER_MONTH_COMMITS = int(os.getenv("SCALAR_FEATURES_TEST_CRASH_AFTER_MONTH_COMMITS", "0"))
TEST_CRASH_ASSET = os.getenv("SCALAR_FEATURES_TEST_CRASH_ASSET", "").strip()
TEST_CRASH_INTERVAL = int(os.getenv("SCALAR_FEATURES_TEST_CRASH_INTERVAL", "0"))


# ------------------------------------------------
# Logging & JSON helpers
# ------------------------------------------------
def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def read_json(path: Optional[Path]) -> dict:
    if not path or not Path(path).exists():
        return {}
    # Accept optional BOM so external edits (e.g., PowerShell UTF-8) don't break startup.
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    tmp = sibling_temp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def _fmt_bytes(num: Optional[int]) -> str:
    if num is None:
        return "n/a"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def wait_for_resources(max_ram_pct: float = 85.0, max_cpu_pct: float = 95.0, timeout_sec: int = 300, sleep_sec: int = 5) -> bool:
    """Back off if system is under load; returns False if timeout exceeded."""
    if psutil is None:
        return True
    start = time.monotonic()
    proc = psutil.Process()
    warned = False
    while True:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.2)
        swap = psutil.swap_memory()
        disks = psutil.disk_io_counters(perdisk=True)
        # Try to extract C and D disk counters
        def _disk_stats(letter: str) -> str:
            letter = letter.upper()
            for name, stats in disks.items():
                if letter in name.upper():
                    return f"{_fmt_bytes(getattr(stats, 'read_bytes', 0))}/{_fmt_bytes(getattr(stats, 'write_bytes', 0))}"
            return "n/a"
        c_stats = _disk_stats("C")
        d_stats = _disk_stats("D")
        io_proc = proc.io_counters() if hasattr(proc, "io_counters") else None
        rss = proc.memory_info().rss if hasattr(proc, "memory_info") else None
        if vm.percent < max_ram_pct and cpu < max_cpu_pct:
            if warned:
                log(f"[resources] cleared to proceed cpu={cpu:.1f}% ram={vm.percent:.1f}% swap={swap.percent:.1f}% proc_rss={_fmt_bytes(rss)} proc_io={io_proc}")
            return True
        warned = True
        log(
            f"[resources][wait] cpu={cpu:.1f}% ram={vm.percent:.1f}% swap={swap.percent:.1f}% "
            f"c_io={c_stats} d_io={d_stats} proc_rss={_fmt_bytes(rss)} proc_io={io_proc}; sleeping {sleep_sec}s"
        )
        time.sleep(sleep_sec)
        if time.monotonic() - start >= timeout_sec:
            log(f"[resources][wait][timeout] exceeded {timeout_sec}s; giving up")
            return False


def _acquire_lockfile(lock_path: Path, timeout_sec: float = 60.0, poll_sec: float = 0.05) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            try:
                # Recover from stale lockfiles left by crashed writers.
                age = time.time() - lock_path.stat().st_mtime
                if age > max(120.0, float(timeout_sec) * 2.0):
                    lock_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring lock {lock_path}")
            time.sleep(max(0.01, float(poll_sec)))


def _release_lockfile(lock_path: Path, fd: Optional[int]) -> None:
    try:
        if fd is not None:
            os.close(fd)
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


def _partition_lock_path(dst: Path) -> Path:
    return dst.with_name(f"{dst.name}.lock")


# ------------------------------------------------
# Parquet discovery helpers
# ------------------------------------------------
def list_assets_from_ohlcvt(interval_min: int, root: Optional[Path] = None) -> List[str]:
    """List distinct assets present in OHLCVT Parquet files for an interval."""
    return list_assets_ohlcvt(interval_min=int(interval_min), root=Path(root) if root else OHLCVT_PARQUET_ROOT)


def feature_max_ts_from_parquet(interval_min: int, asset: str, root: Optional[Path] = None) -> Optional[int]:
    root_dir = Path(root) if root else PARQUET_ROOT
    base = root_dir / f"scalar_features_{int(interval_min)}" / f"asset={asset}"
    return partition_max_ts(base, ts_column="ts")


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


def ohlcvt_bounds_from_parquet(interval_min: int, asset: str, root: Optional[Path] = None) -> Tuple[Optional[int], Optional[int]]:
    return ohlcvt_bounds(interval_min=int(interval_min), asset=str(asset), root=Path(root) if root else OHLCVT_PARQUET_ROOT)
def ohlcvt_max_ts_from_parquet(interval_min: int, asset: str, root: Optional[Path] = None) -> Optional[int]:
    root_dir = Path(root) if root else OHLCVT_PARQUET_ROOT
    base = root_dir / f"ohlcvt_{int(interval_min)}" / f"asset={asset}"
    return partition_max_ts(base, ts_column="ts")


def _find_partition_month_dir(base: Path, newest: bool) -> Optional[Path]:
    return shared_find_partition_month_dir(base=base, newest=bool(newest))


def first_ohlcvt_ts_from_disk(interval_min: int, asset: str, root: Optional[Path] = None) -> Optional[int]:
    """
    Resolve the first on-disk OHLCVT timestamp for one (asset, interval) by:
    - selecting earliest year/month partition under asset path
    - opening one parquet file in that month
    - reading only `ts` and returning min(ts)
    """
    root_dir = Path(root) if root else OHLCVT_PARQUET_ROOT
    base = root_dir / f"ohlcvt_{int(interval_min)}" / f"asset={asset}"
    if not base.exists():
        return None

    earliest_month_dir = _find_partition_month_dir(base=base, newest=False)

    if earliest_month_dir is None:
        return None

    month_files = sorted(earliest_month_dir.glob("*.parquet"), key=lambda p: p.name.lower())
    if not month_files:
        return None

    first_file = month_files[0]
    try:
        d = pd.read_parquet(first_file, columns=["ts"])
    except Exception:
        return None
    if d.empty:
        return None
    ts = pd.to_numeric(d["ts"], errors="coerce").dropna().astype("int64")
    if ts.empty:
        return None
    return int(ts.min())


def _scan_interval_max(root: Path, table_prefix: str, interval_min: int, assets_filter: Optional[set[str]] = None) -> dict[str, Optional[int]]:
    """Return max ts per asset using newest-month-only lookup."""
    if not assets_filter:
        return {}

    prefix = str(table_prefix).strip().lower()
    if prefix not in {"scalar_features", "ohlcvt"}:
        return {}

    out: dict[str, Optional[int]] = {}
    for asset in sorted(str(a) for a in assets_filter):
        try:
            if prefix == "ohlcvt":
                out[asset] = ohlcvt_max_ts_from_parquet(interval_min=int(interval_min), asset=str(asset), root=root)
            else:
                out[asset] = feature_max_ts_from_parquet(interval_min=int(interval_min), asset=str(asset), root=root)
        except Exception:
            out[asset] = None
    return out


def _scan_interval_task(payload: tuple[int, List[str], str, str]) -> tuple[int, dict[str, Optional[int]], dict[str, Optional[int]]]:
    """
    Worker payload for interval-level startup scan.
    Returns (interval, feature_bounds_by_asset, ohlc_bounds_by_asset).
    """
    interval_min, assets, feature_root_str, ohlc_root_str = payload
    assets_filter = set(assets)
    feature_max = _scan_interval_max(Path(feature_root_str), "scalar_features", interval_min, assets_filter=assets_filter)
    ohlc_max = _scan_interval_max(Path(ohlc_root_str), "ohlcvt", interval_min, assets_filter=assets_filter)
    return interval_min, feature_max, ohlc_max


def scan_bounds_for_tasks(
    tasks: List[Tuple[str, int]],
    workers: int,
    context: str = "[features][scan]",
) -> tuple[dict[int, dict[str, Optional[int]]], dict[int, dict[str, Optional[int]]]]:
    """
    Build per-interval asset max-ts indexes using newest-month-only lookup.
    Returns (feature_max_by_interval, ohlc_max_by_interval).
    """
    assets_by_interval: dict[int, set[str]] = {}
    for asset, interval in tasks:
        assets_by_interval.setdefault(int(interval), set()).add(str(asset))
    if not assets_by_interval:
        return {}, {}
    intervals = sorted(assets_by_interval.keys())
    scan_workers = max(1, min(int(workers), len(intervals)))
    total_assets = sum(len(v) for v in assets_by_interval.values())
    log(f"{context} starting interval_scan intervals={len(intervals)} assets={total_assets} workers={scan_workers}")
    payloads = [
        (k, sorted(assets_by_interval[k]), str(PARQUET_ROOT), str(OHLCVT_PARQUET_ROOT))
        for k in intervals
    ]
    feat_idx: dict[int, dict[str, Optional[int]]] = {}
    ohlc_idx: dict[int, dict[str, Optional[int]]] = {}
    done = 0
    if scan_workers == 1:
        for payload in payloads:
            k, feat_max, ohlc_max = _scan_interval_task(payload)
            feat_idx[int(k)] = feat_max
            ohlc_idx[int(k)] = ohlc_max
            done += 1
            log(f"{context} progress {done}/{len(payloads)} k={k} feature_assets={len(feat_max)} ohlc_assets={len(ohlc_max)}")
        return feat_idx, ohlc_idx

    fallback_payloads: list[tuple[int, List[str], str, str]] = []
    try:
        with ProcessPoolExecutor(max_workers=scan_workers) as pool:
            future_map = {pool.submit(_scan_interval_task, p): p for p in payloads}
            for fut in as_completed(future_map):
                payload = future_map[fut]
                k = int(payload[0])
                try:
                    k2, feat_max, ohlc_max = fut.result()
                except BrokenProcessPool as exc:
                    log(f"{context}[warn] process pool broke during interval_scan ({exc}); falling back to serial for remaining intervals")
                    fallback_payloads = [p for f, p in future_map.items() if not f.done()]
                    fallback_payloads.append(payload)
                    break
                except Exception as exc:
                    log(f"{context}[error] k={k}: {exc}")
                    feat_max = {}
                    ohlc_max = {}
                    k2 = k
                feat_idx[int(k2)] = feat_max
                ohlc_idx[int(k2)] = ohlc_max
                done += 1
                log(f"{context} progress {done}/{len(payloads)} k={k2} feature_assets={len(feat_max)} ohlc_assets={len(ohlc_max)}")
    except BrokenProcessPool as exc:
        log(f"{context}[warn] process pool failed to initialize ({exc}); running interval_scan serially")
        fallback_payloads = list(payloads)

    if fallback_payloads:
        for payload in fallback_payloads:
            k, feat_max, ohlc_max = _scan_interval_task(payload)
            feat_idx[int(k)] = feat_max
            ohlc_idx[int(k)] = ohlc_max
            done += 1
            log(f"{context} progress {done}/{len(payloads)} k={k} feature_assets={len(feat_max)} ohlc_assets={len(ohlc_max)}")
    return feat_idx, ohlc_idx


# ------------------------------------------------
# Feature calculations
# ------------------------------------------------
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_div_floor(a: pd.Series, b: pd.Series, floor_abs: float) -> pd.Series:
    floor = max(0.0, float(floor_abs))
    if floor <= 0.0:
        return _safe_div(a, b)
    b_num = pd.to_numeric(b, errors="coerce")
    sign = np.sign(b_num).replace(0, 1.0)
    b_safe = b_num.where(b_num.abs() >= floor, sign * floor)
    out = pd.to_numeric(a, errors="coerce") / b_safe
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_pct_change_floor(s: pd.Series, periods: int, floor_abs: float) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    prev = s_num.shift(int(periods))
    delta = s_num - prev
    return _safe_div_floor(delta, prev, floor_abs=floor_abs)


def _light_clip(s: pd.Series, clip_abs: float) -> pd.Series:
    c = max(0.0, float(clip_abs))
    if c <= 0.0:
        return s
    return pd.to_numeric(s, errors="coerce").clip(lower=-c, upper=c)


if _HAS_NUMBA:
    @njit(cache=True)  # type: ignore[misc]
    def _kama_numba(arr: np.ndarray, n: int, fast: int, slow: int) -> np.ndarray:
        out = arr.copy()
        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        for i in range(1, arr.size):
            if i < n:
                out[i] = arr[i]
                continue
            change = abs(arr[i] - arr[i - n])
            volatility = 0.0
            for j in range(i - n + 1, i + 1):
                volatility += abs(arr[j] - arr[j - 1])
            er = 0.0 if volatility == 0.0 else change / volatility
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            prev = out[i - 1]
            out[i] = prev + sc * (arr[i] - prev)
        return out

    @njit(cache=True)  # type: ignore[misc]
    def _frama_numba(arr: np.ndarray, n: int) -> np.ndarray:
        out = np.empty(arr.size, dtype=np.float64)
        if arr.size == 0:
            return out
        out[0] = arr[0]
        half = n // 2
        ln2 = math.log(2.0)
        for i in range(1, arr.size):
            if i < n:
                out[i] = arr[i]
                continue
            start = i - n + 1
            end = i + 1
            N1 = 0.0
            for j in range(start + half, end):
                N1 += abs(arr[j] - arr[j - half])
            N1 /= max(half, 1)
            N2 = 0.0
            if half > 1:
                for j in range(start + 1, end - half + 1):
                    N2 += abs(arr[j] - arr[j - 1])
                N2 /= half
            else:
                N2 = N1
            wmin = arr[start]
            wmax = arr[start]
            for j in range(start + 1, end):
                v = arr[j]
                if v < wmin:
                    wmin = v
                if v > wmax:
                    wmax = v
            N3 = wmax - wmin
            if N3 > 0.0 and N1 > 0.0 and N2 > 0.0:
                D = (math.log(N1 + N2) - math.log(N3)) / ln2
            else:
                D = 1.0
            alpha = math.exp(-4.6 * (D - 1.0))
            if alpha > 1.0:
                alpha = 1.0
            if alpha < 0.01:
                alpha = 0.01
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    @njit(cache=True)  # type: ignore[misc]
    def _psar_numba(high_arr: np.ndarray, low_arr: np.ndarray, af: float, af_max: float) -> np.ndarray:
        n = high_arr.size
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        uptrend = True
        ep = low_arr[0]
        sar = low_arr[0]
        af_local = af
        out[0] = sar
        for i in range(1, n):
            sar = sar + af_local * (ep - sar)
            if uptrend:
                if sar > low_arr[i - 1]:
                    sar = low_arr[i - 1]
                if high_arr[i] > ep:
                    ep = high_arr[i]
                    af_local = min(af_local + af, af_max)
                if low_arr[i] < sar:
                    uptrend = False
                    sar = ep
                    ep = low_arr[i]
                    af_local = af
            else:
                if sar < high_arr[i - 1]:
                    sar = high_arr[i - 1]
                if low_arr[i] < ep:
                    ep = low_arr[i]
                    af_local = min(af_local + af, af_max)
                if high_arr[i] > sar:
                    uptrend = True
                    sar = ep
                    ep = high_arr[i]
                    af_local = af
            out[i] = sar
        return out

    @njit(cache=True)  # type: ignore[misc]
    def _lr_metrics_numba(arr: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = arr.size
        slope = np.full(m, np.nan, dtype=np.float64)
        intercept = np.full(m, np.nan, dtype=np.float64)
        hi = np.full(m, np.nan, dtype=np.float64)
        lo = np.full(m, np.nan, dtype=np.float64)
        if m < n or n <= 1:
            return slope, intercept, hi, lo
        x_mean = (n - 1) / 2.0
        x_var = 0.0
        sum_x = 0.0
        for i in range(n):
            dx = i - x_mean
            x_var += dx * dx
            sum_x += i
        for end in range(n - 1, m):
            start = end - n + 1
            sum_y = 0.0
            sum_xy = 0.0
            for i in range(n):
                y = arr[start + i]
                sum_y += y
                sum_xy += i * y
            y_mean = sum_y / n
            b = (sum_xy - x_mean * sum_y) / x_var if x_var != 0.0 else 0.0
            a = y_mean - b * x_mean
            fit_end = a + b * (n - 1)
            rmax = -1e308
            rmin = 1e308
            for i in range(n):
                resid = arr[start + i] - (a + b * i)
                if resid > rmax:
                    rmax = resid
                if resid < rmin:
                    rmin = resid
            slope[end] = b
            intercept[end] = a
            hi[end] = fit_end + rmax
            lo[end] = fit_end + rmin
        return slope, intercept, hi, lo

    @njit(cache=True)  # type: ignore[misc]
    def _roll_entropy_numba(arr: np.ndarray, window: int, bins: int) -> np.ndarray:
        n = arr.size
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = 0 if i - window + 1 < 0 else i - window + 1
            count = 0
            wmin = 0.0
            wmax = 0.0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isnan(v):
                    continue
                if count == 0:
                    wmin = v
                    wmax = v
                else:
                    if v < wmin:
                        wmin = v
                    if v > wmax:
                        wmax = v
                count += 1
            if count == 0:
                out[i] = 0.0
                continue
            if wmax == wmin:
                out[i] = 0.0
                continue
            hist = np.zeros(bins, dtype=np.int64)
            width = (wmax - wmin) / bins
            if width <= 0.0:
                out[i] = 0.0
                continue
            for j in range(start, i + 1):
                v = arr[j]
                if np.isnan(v):
                    continue
                b = int((v - wmin) / width)
                if b < 0:
                    b = 0
                elif b >= bins:
                    b = bins - 1
                hist[b] += 1
            ent = 0.0
            for b in range(bins):
                c = hist[b]
                if c <= 0:
                    continue
                p = c / count
                ent -= p * math.log(p)
            out[i] = ent
        return out

    @njit(cache=True)  # type: ignore[misc]
    def _roll_hurst_numba(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        n = arr.size
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = 0 if i - window + 1 < 0 else i - window + 1
            valid = 0
            for j in range(start, i + 1):
                if not np.isnan(arr[j]):
                    valid += 1
            if valid < min_periods:
                out[i] = np.nan
                continue
            vals = np.empty(valid, dtype=np.float64)
            k = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isnan(v):
                    continue
                vals[k] = v
                k += 1
            if valid < 2:
                out[i] = 0.5
                continue
            mean_v = 0.0
            for j in range(valid):
                mean_v += vals[j]
            mean_v /= valid
            z = np.empty(valid, dtype=np.float64)
            csum = 0.0
            s2 = 0.0
            for j in range(valid):
                d = vals[j] - mean_v
                s2 += d * d
                csum += d
                z[j] = csum
            rmax = z[0]
            rmin = z[0]
            for j in range(1, valid):
                if z[j] > rmax:
                    rmax = z[j]
                if z[j] < rmin:
                    rmin = z[j]
            R = rmax - rmin
            S = math.sqrt(s2 / valid) if valid > 0 else 0.0
            if S == 0.0 or R == 0.0:
                out[i] = 0.5
            else:
                out[i] = math.log(R / S) / math.log(valid)
        return out

    @njit(cache=True)  # type: ignore[misc]
    def _higuchi_fd_numba(vals: np.ndarray, kmax: int) -> float:
        N = vals.size
        if N < 2:
            return 1.0
        L = np.zeros(kmax, dtype=np.float64)
        L_count = np.zeros(kmax, dtype=np.int64)
        for k in range(1, kmax + 1):
            idx_k = k - 1
            for m in range(k):
                points = 0
                prev = 0.0
                total = 0.0
                first = True
                t = m
                while t < N:
                    cur = vals[t]
                    if first:
                        prev = cur
                        first = False
                    else:
                        total += abs(cur - prev)
                        prev = cur
                    points += 1
                    t += k
                if points >= 2:
                    Lm = total * (N - 1) / (points * k)
                    L[idx_k] += Lm
                    L_count[idx_k] += 1
        used = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        for i in range(kmax):
            if L_count[i] <= 0:
                continue
            Li = L[i] / L_count[i]
            if Li <= 0.0:
                continue
            x = math.log(i + 1.0)
            y = math.log(Li)
            used += 1
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y
        if used < 2:
            return 1.0
        denom = used * sum_xx - sum_x * sum_x
        if denom == 0.0:
            return 1.0
        slope = (used * sum_xy - sum_x * sum_y) / denom
        return 1.0 - slope

    @njit(cache=True)  # type: ignore[misc]
    def _roll_fractal_numba(arr: np.ndarray, window: int, min_periods: int, kmax: int) -> np.ndarray:
        n = arr.size
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = 0 if i - window + 1 < 0 else i - window + 1
            valid = 0
            for j in range(start, i + 1):
                if not np.isnan(arr[j]):
                    valid += 1
            if valid < min_periods:
                out[i] = np.nan
                continue
            vals = np.empty(valid, dtype=np.float64)
            p = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isnan(v):
                    continue
                vals[p] = v
                p += 1
            out[i] = _higuchi_fd_numba(vals, kmax)
        return out

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: ts, open, high, low, close, volume, trades
    Returns DataFrame with feature columns aligned to input ts (no ts column inside).
    """
    # Ensure numeric types
    for c in ["open", "high", "low", "close", "volume", "trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df["volume"].astype(float)
    trades = pd.to_numeric(df.get("trades", pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(0.0)

    # Basic prices
    typical = (high + low + close) / 3.0
    median = (high + low) / 2.0
    wclose = (high + low + 2 * close) / 4.0
    delta_c = close.diff()
    log_ret = (close / close.shift(1)).apply(lambda x: math.log(x) if pd.notnull(x) and x > 0 else np.nan)
    pct_chg = close.pct_change()
    hl_range = high - low
    co_range = close - open_

    # True range and ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()

    # Moving averages
    sma20 = close.rolling(20, min_periods=1).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    # Weighted moving average (linear weights)
    def wma(s: pd.Series, n: int) -> pd.Series:
        weights = np.arange(1, n + 1, dtype=float)
        return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    wma20 = wma(close, 20)

    # HMA(20)
    wma10 = wma(close, 10)
    wma20_base = wma(close, 20)
    hma20 = wma(2 * wma10 - wma20_base, int(np.sqrt(20)))

    # Wilder's smoothing on close (14)
    wilder14 = close.ewm(alpha=1/14, adjust=False).mean()

    # MA Envelope: SMA(20) +/- 2%
    sma20_env = sma20
    ma_env_upper_20_2pct = sma20_env * 1.02
    ma_env_lower_20_2pct = sma20_env * 0.98

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))

    # Stochastic %K(14) and %D(3)
    low14 = low.rolling(14, min_periods=1).min()
    high14 = high.rolling(14, min_periods=1).max()
    stoch_k14 = 100 * _safe_div((close - low14), (high14 - low14))
    stoch_d3 = stoch_k14.rolling(3, min_periods=1).mean()

    # Williams %R(14)
    willr14 = -100 * _safe_div((high14 - close), (high14 - low14))

    # CCI(20)
    tp = typical
    sma_tp = tp.rolling(20, min_periods=1).mean()
    mad = tp.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci20 = _safe_div(tp - sma_tp, 0.015 * mad)

    # DMI/ADX(14)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)
    tr14 = tr.rolling(14).sum()
    plus_di14 = 100 * _safe_div(plus_dm.rolling(14).sum(), tr14)
    minus_di14 = 100 * _safe_div(minus_dm.rolling(14).sum(), tr14)
    dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan)
    adx14 = dx.rolling(14, min_periods=14).mean()

    # Bollinger Bands(20, k=2)
    mid = sma20
    std20 = close.rolling(20, min_periods=1).std()
    boll_up = mid + 2 * std20
    boll_low = mid - 2 * std20

    # Keltner Channel (EMA20 + ATR10*2)
    ema20_mid = ema20
    atr10 = tr.ewm(alpha=1/10, adjust=False).mean()
    kelt_up = ema20_mid + 2 * atr10
    kelt_low = ema20_mid - 2 * atr10

    # For TTM squeeze: Keltner with 1.5x ATR10
    kelt_up_1_5 = ema20_mid + 1.5 * atr10
    kelt_low_1_5 = ema20_mid - 1.5 * atr10

    # Aroon(25)
    def aroon_up(series: pd.Series, n: int) -> pd.Series:
        return series.rolling(n, min_periods=1).apply(lambda x: 100 * (n - 1 - np.argmax(x[::-1])) / (n - 1) if n > 1 else 0, raw=True)
    def aroon_down(series: pd.Series, n: int) -> pd.Series:
        return series.rolling(n, min_periods=1).apply(lambda x: 100 * (n - 1 - np.argmin(x[::-1])) / (n - 1) if n > 1 else 0, raw=True)
    a_up25 = aroon_up(high, 25)
    a_dn25 = aroon_down(low, 25)
    a_osc25 = a_up25 - a_dn25

    # Volume-derived
    # Keep original cumulative definitions available; test variants can remap these below.
    obv_orig = (np.sign(delta_c.fillna(0)) * vol).cumsum()

    # MFI(14)
    mf_typ = tp
    mf_raw = mf_typ * vol
    mf_pos = mf_raw.where(mf_typ >= mf_typ.shift(1), 0.0)
    mf_neg = mf_raw.where(mf_typ < mf_typ.shift(1), 0.0)
    mr = (mf_pos.rolling(14).sum()) / (mf_neg.rolling(14).sum().replace(0, np.nan))
    mfi14 = 100 - (100 / (1 + mr))

    # ADL
    clv = _safe_div((close - low) - (high - close), (high - low))
    adl_orig = (clv.fillna(0) * vol).cumsum()

    # Force Index
    force = delta_c.fillna(0) * vol

    # Trade-level
    avg_trade_size = _safe_div(vol, trades.replace(0, np.nan))
    trade_intensity = trades.fillna(0).astype("Int64")

    # Statistical rolling
    ret = close.pct_change()
    ret_mean20 = ret.rolling(20, min_periods=1).mean()
    ret_std20 = ret.rolling(20, min_periods=1).std()
    sharpe20 = (ret_mean20 / ret_std20.replace(0, np.nan)) * np.sqrt(20)

    # Momentum & ROC family
    roc_14 = _safe_div(close - close.shift(14), close.shift(14))
    mom_14 = close - close.shift(14)
    sum_up_14 = gain.rolling(14).sum()
    sum_dn_14 = loss.rolling(14).sum()
    cmo_14 = 100 * _safe_div((sum_up_14 - sum_dn_14), (sum_up_14 + sum_dn_14))

    # TRIX(15): triple EMA on close and 1-period ROC (percent)
    te1 = close.ewm(span=15, adjust=False).mean()
    te2 = te1.ewm(span=15, adjust=False).mean()
    te3 = te2.ewm(span=15, adjust=False).mean()
    trix_15 = 100 * _safe_div(te3 - te3.shift(1), te3.shift(1))

    # DPO(20): close - SMA(20) shifted by 11
    dpo_20 = close - sma20.shift(11)

    # Ultimate Oscillator 7/14/28
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr_uo = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
    ultosc_7_14_28 = 100 * ((4*(_safe_div(bp.rolling(7).sum(), tr_uo.rolling(7).sum()))
                            + 2*(_safe_div(bp.rolling(14).sum(), tr_uo.rolling(14).sum()))
                            + (_safe_div(bp.rolling(28).sum(), tr_uo.rolling(28).sum()))) / 7)

    # Volatility & range-based
    cv_20 = _safe_div(close.rolling(20).std(), close.rolling(20).mean())
    donchian_hi_20 = high.rolling(20, min_periods=1).max()
    donchian_lo_20 = low.rolling(20, min_periods=1).min()
    prr = _safe_div((high - low), close)

    # Volume-derived extensions
    chaikin_osc_3_10 = adl_orig.ewm(span=3, adjust=False).mean() - adl_orig.ewm(span=10, adjust=False).mean()
    vpt = (close.pct_change().fillna(0) * vol).cumsum()
    mid_price = (high + low) / 2.0
    mid_prev = (high.shift(1) + low.shift(1)) / 2.0
    eom_raw = (mid_price - mid_prev) * _safe_div((high - low), vol)
    eom_14 = eom_raw.rolling(14, min_periods=1).mean()
    r = close.pct_change().fillna(0)
    vol_up = vol > vol.shift(1)
    vol_down = vol < vol.shift(1)
    pvi_orig = pd.Series(1000 * (np.where(vol_up, 1 + r, 1)).cumprod(), index=close.index, dtype=float)
    nvi_orig = pd.Series(1000 * (np.where(vol_down, 1 + r, 1)).cumprod(), index=close.index, dtype=float)

    # VWAP by day (UTC) using typical price
    ts_dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    day_key = ts_dt.dt.floor("D")
    tp = (high + low + close) / 3.0
    vwap_day = (tp * vol).groupby(day_key).cumsum() / vol.groupby(day_key).cumsum().replace(0, np.nan)

    # Volume oscillator 14/28
    ema_v_14 = vol.ewm(span=14, adjust=False).mean()
    ema_v_28 = vol.ewm(span=28, adjust=False).mean()
    vol_osc_14_28 = ema_v_14 - ema_v_28
    vol_osc_pct_14_28 = 100 * _safe_div(ema_v_14, ema_v_28) - 100

    # Statistical extensions
    var_20 = close.rolling(20, min_periods=1).var()
    skew_20 = close.rolling(20, min_periods=1).skew()
    kurt_20 = close.rolling(20, min_periods=1).kurt()
    zscore_20 = _safe_div((close - close.rolling(20, min_periods=1).mean()), std20.replace(0, np.nan))

    # Sortino(20)
    downside = r.clip(upper=0)
    downside_std = (downside.pow(2).rolling(20, min_periods=1).mean()).pow(0.5)
    sortino_20 = _safe_div(ret_mean20, downside_std)

    # TTM squeeze
    bb_width = (boll_up - boll_low).abs()
    kelt_width = (kelt_up_1_5 - kelt_low_1_5).abs()
    in_squeeze = (_safe_div(bb_width, kelt_width) < 1).astype("Int64")

    # PSAR (AF=0.02, AF_max=0.2)
    if _HAS_NUMBA:
        psar = pd.Series(
            _psar_numba(high.to_numpy(dtype=np.float64, copy=False), low.to_numpy(dtype=np.float64, copy=False), 0.02, 0.2),
            index=high.index,
            dtype=float,
        )
    else:
        psar_arr = np.zeros(len(high), dtype=float)
        if len(high) > 0:
            uptrend = True
            ep = float(low.iloc[0])
            sar = float(low.iloc[0])
            af_local = 0.02
            psar_arr[0] = sar
            for i in range(1, len(high)):
                sar = sar + af_local * (ep - sar)
                if uptrend:
                    sar = min(sar, float(low.iloc[i - 1]))
                    if float(high.iloc[i]) > ep:
                        ep = float(high.iloc[i])
                        af_local = min(af_local + 0.02, 0.2)
                    if float(low.iloc[i]) < sar:
                        uptrend = False
                        sar = ep
                        ep = float(low.iloc[i])
                        af_local = 0.02
                else:
                    sar = max(sar, float(high.iloc[i - 1]))
                    if float(low.iloc[i]) < ep:
                        ep = float(low.iloc[i])
                        af_local = min(af_local + 0.02, 0.2)
                    if float(high.iloc[i]) > sar:
                        uptrend = True
                        sar = ep
                        ep = float(high.iloc[i])
                        af_local = 0.02
                psar_arr[i] = sar
        psar = pd.Series(psar_arr, index=high.index, dtype=float)

    # Linear Regression (n=20)
    n_lr = 20
    if _HAS_NUMBA:
        _lr_slope_arr, _lr_intercept_arr, _lr_hi_arr, _lr_lo_arr = _lr_metrics_numba(
            close.to_numpy(dtype=np.float64, copy=False), n_lr
        )
        lr_slope_20 = pd.Series(_lr_slope_arr, index=close.index, dtype=float)
        lr_intercept_20 = pd.Series(_lr_intercept_arr, index=close.index, dtype=float)
        lr_channel_hi_20 = pd.Series(_lr_hi_arr, index=close.index, dtype=float)
        lr_channel_lo_20 = pd.Series(_lr_lo_arr, index=close.index, dtype=float)
    else:
        x = np.arange(n_lr, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        def _lr_slope_win(y):
            y = np.asarray(y, dtype=float)
            y_mean = y.mean()
            return float(((x - x_mean) * (y - y_mean)).sum() / x_var) if x_var != 0 else 0.0

        def _lr_intercept_win(y):
            y = np.asarray(y, dtype=float)
            y_mean = y.mean()
            slope = _lr_slope_win(y)
            return float(y_mean - slope * x_mean)

        def _lr_channel_hi(y):
            y = np.asarray(y, dtype=float)
            slope = _lr_slope_win(y)
            intercept = _lr_intercept_win(y)
            resid = y - (slope * x + intercept)
            return float(intercept + slope * (n_lr - 1) + resid.max())

        def _lr_channel_lo(y):
            y = np.asarray(y, dtype=float)
            slope = _lr_slope_win(y)
            intercept = _lr_intercept_win(y)
            resid = y - (slope * x + intercept)
            return float(intercept + slope * (n_lr - 1) + resid.min())

        lr_slope_20 = close.rolling(n_lr).apply(lambda w: _lr_slope_win(w), raw=True)
        lr_intercept_20 = close.rolling(n_lr).apply(lambda w: _lr_intercept_win(w), raw=True)
        lr_channel_hi_20 = close.rolling(n_lr).apply(lambda w: _lr_channel_hi(w), raw=True)
        lr_channel_lo_20 = close.rolling(n_lr).apply(lambda w: _lr_channel_lo(w), raw=True)

    # Ichimoku (aligned)
    tenkan_9 = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2.0
    kijun_26 = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2.0
    span_a_26 = (tenkan_9 + kijun_26) / 2.0
    span_b_26 = (high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2.0
    chikou_26 = close

    # RVI(10)
    hl_ema_10 = (high - low).ewm(span=10, adjust=False).mean()
    chaikin_vol_10_10 = _light_clip(
        100.0 * _safe_pct_change_floor(hl_ema_10, periods=10, floor_abs=CHAIKIN_VOL_DENOM_FLOOR),
        CHAIKIN_VOL_MAX_ABS,
    )
    vroc_14 = _light_clip(
        _safe_div_floor(vol - vol.shift(14), vol.shift(14), floor_abs=DENOM_FLOOR),
        MAX_FEATURE_ABS,
    )
    rvi_10 = _light_clip(
        _safe_div_floor(ret.rolling(10).std(), close.rolling(10).std(), floor_abs=DENOM_FLOOR),
        MAX_FEATURE_ABS,
    )
    squeeze_scalar = _light_clip(
        _safe_div_floor(bb_width, kelt_width, floor_abs=DENOM_FLOOR),
        MAX_FEATURE_ABS,
    )

    obv = _safe_pct_change_floor(obv_orig, periods=1, floor_abs=DENOM_FLOOR).fillna(0.0)
    adl = _safe_pct_change_floor(adl_orig, periods=1, floor_abs=DENOM_FLOOR).fillna(0.0)
    nvi = _safe_pct_change_floor(nvi_orig, periods=1, floor_abs=DENOM_FLOOR).fillna(0.0)
    pvi = _safe_pct_change_floor(pvi_orig, periods=1, floor_abs=DENOM_FLOOR).fillna(0.0)

    # Elder Ray (13)
    ema13 = close.ewm(span=13, adjust=False).mean()
    elder_bull_13 = high - ema13
    elder_bear_13 = low - ema13

    # Connors RSI (3,2,100)
    def rsi_of_series(s: pd.Series, n: int) -> pd.Series:
        d = s.diff()
        up = d.clip(lower=0)
        down = -d.clip(upper=0)
        rs = (up.rolling(n).mean()) / (down.rolling(n).mean().replace(0, np.nan))
        return 100 - (100 / (1 + rs))
    rsi3 = rsi_of_series(close, 3)
    delta1 = close.diff()
    streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(streak)):
        if delta1.iloc[i] > 0:
            streak.iloc[i] = max(0, streak.iloc[i-1]) + 1
        elif delta1.iloc[i] < 0:
            streak.iloc[i] = min(0, streak.iloc[i-1]) - 1
        else:
            streak.iloc[i] = 0
    rsi_streak2 = rsi_of_series(streak, 2)
    def percent_rank(window):
        w = np.asarray(window, dtype=float)
        if len(w) <= 1:
            return 50.0
        last = w[-1]
        count = (w <= last).sum()
        return 100.0 * (count - 1) / (len(w) - 1)
    prank100 = ret.rolling(100, min_periods=1).apply(lambda w: percent_rank(w), raw=True)
    crsi_3_2_100 = (rsi3 + rsi_streak2 + prank100) / 3.0

    # KAMA(10,2,30)
    if _HAS_NUMBA:
        kama_10_2_30 = pd.Series(
            _kama_numba(close.to_numpy(dtype=np.float64, copy=False), 10, 2, 30),
            index=close.index,
            dtype=float,
        )
    else:
        er = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum().replace(0, np.nan))
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_10_2_30 = close.copy()
        for i in range(1, len(close)):
            prev = kama_10_2_30.iloc[i - 1] if not pd.isna(kama_10_2_30.iloc[i - 1]) else close.iloc[i - 1]
            kama_10_2_30.iloc[i] = prev + (sc.iloc[i] if not pd.isna(sc.iloc[i]) else 0) * (close.iloc[i] - prev)

    # FRAMA(16)
    if _HAS_NUMBA:
        frama_16 = pd.Series(
            _frama_numba(close.to_numpy(dtype=np.float64, copy=False), 16),
            index=close.index,
            dtype=float,
        )
    else:
        frama_16 = pd.Series(index=close.index, dtype=float)
        if len(close) > 0:
            frama_16.iloc[0] = close.iloc[0]
        for i in range(1, len(close)):
            if i < 16:
                frama_16.iloc[i] = close.iloc[i]
                continue
            win = close.iloc[i - 15:i + 1]
            n2 = 8
            N1 = (win.iloc[n2:] - win.iloc[:-n2]).abs().sum() / n2
            N2 = (win.iloc[n2 - 1:-1] - win.iloc[:-n2]).abs().sum() / n2
            N3 = win.max() - win.min()
            D = (np.log(N1 + N2) - np.log(N3)) / np.log(2) if (np.isfinite(N3) and N3 != 0 and N1 > 0 and N2 > 0) else 1.0
            alpha = np.exp(-4.6 * (D - 1))
            alpha = max(min(alpha, 1.0), 0.01)
            frama_16.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * frama_16.iloc[i - 1]

    # Vortex(14)
    vi_plus_14 = (high - low.shift(1)).abs().rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    vi_minus_14 = (low - high.shift(1)).abs().rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)

    # Entropy(20)
    if _HAS_NUMBA:
        entropy_20 = pd.Series(
            _roll_entropy_numba(ret.to_numpy(dtype=np.float64, copy=False), 20, 10),
            index=ret.index,
            dtype=float,
        )
    else:
        def _entropy_window(arr):
            a = np.asarray(arr, dtype=float)
            a = a[~np.isnan(a)]
            if len(a) == 0:
                return 0.0
            hist, _ = np.histogram(a, bins=10)
            p = hist / hist.sum() if hist.sum() > 0 else hist
            p = p[p > 0]
            return float(-(p * np.log(p)).sum())
        entropy_20 = ret.rolling(20, min_periods=1).apply(lambda w: _entropy_window(w), raw=True)

    # Hurst(100)
    if _HAS_NUMBA:
        hurst_100 = pd.Series(
            _roll_hurst_numba(close.pct_change().to_numpy(dtype=np.float64, copy=False), 100, 2),
            index=close.index,
            dtype=float,
        )
    else:
        def _hurst_window(arr):
            a = np.asarray(arr, dtype=float)
            a = a[~np.isnan(a)]
            n = len(a)
            if n < 2:
                return 0.5
            y = a - a.mean()
            z = np.cumsum(y)
            R = z.max() - z.min()
            S = y.std()
            if S == 0 or R == 0:
                return 0.5
            return float(np.log(R / S) / np.log(n))
        hurst_100 = close.pct_change().rolling(100, min_periods=2).apply(lambda w: _hurst_window(w), raw=True)

    # Fractal(100) Higuchi
    if _HAS_NUMBA:
        fractal_100 = pd.Series(
            _roll_fractal_numba(close.to_numpy(dtype=np.float64, copy=False), 100, 2, 5),
            index=close.index,
            dtype=float,
        )
    else:
        def _higuchi_window(arr):
            a = np.asarray(arr, dtype=float)
            a = a[~np.isnan(a)]
            N = len(a)
            if N < 2:
                return 1.0
            L = []
            for k in range(1, 6):
                Lk = []
                for m in range(k):
                    idx = np.arange(m, N, k)
                    if len(idx) < 2:
                        continue
                    Lm = np.abs(np.diff(a[idx])).sum() * (N - 1) / (len(idx) * k)
                    Lk.append(Lm)
                if Lk:
                    L.append(np.mean(Lk))
            if len(L) < 2:
                return 1.0
            lnL = np.log(L)
            lnk = np.log(np.arange(1, len(L) + 1))
            slope = np.polyfit(lnk, lnL, 1)[0]
            return float(1 - slope)
        fractal_100 = close.rolling(100, min_periods=2).apply(lambda w: _higuchi_window(w), raw=True)

    # Microstructure & lags
    dir_sign = np.sign(delta_c.fillna(0))
    tir = dir_sign
    v_t = _safe_div(vol, trades.replace(0, np.nan))
    vpt_vol_14 = v_t.rolling(14, min_periods=1).std()
    msv_14 = delta_c.rolling(14, min_periods=1).std() * _safe_div(trades, vol)
    d_close_2 = close - close.shift(2)
    d_close_3 = close - close.shift(3)
    d_close_5 = close - close.shift(5)
    d_close_10 = close - close.shift(10)
    d_close_14 = close - close.shift(14)
    d_close_20 = close - close.shift(20)
    ewm_mean_alpha_0_1 = close.ewm(alpha=0.1, adjust=False).mean()
    ewm_mean_alpha_0_2 = close.ewm(alpha=0.2, adjust=False).mean()
    q25_20 = close.rolling(20, min_periods=1).quantile(0.25)
    q50_20 = close.rolling(20, min_periods=1).quantile(0.50)
    q75_20 = close.rolling(20, min_periods=1).quantile(0.75)
    prank_20 = ret.rolling(20, min_periods=1).apply(lambda w: percent_rank(w), raw=True)
    dir = dir_sign

    base_feats = pd.DataFrame({
        # 1) Basic
        "typical_price": typical,
        "median_price": median,
        "weighted_close": wclose,
        "delta_close": delta_c,
        "log_return": log_ret,
        "pct_change": pct_chg,
        "true_range": tr,
        "atr_14": atr14,
        "range_hl": hl_range,
        "range_co": co_range,
        # 2) Trend
        "sma_20": sma20,
        "ema_20": ema20,
        "wma_20": wma20,
        "hma_20": hma20,
        "wilder_14": wilder14,
        "ma_env_upper_20_2pct": ma_env_upper_20_2pct,
        "ma_env_lower_20_2pct": ma_env_lower_20_2pct,
        "macd_12_26_9": macd,
        "macd_signal_12_26_9": macd_signal,
        "macd_hist_12_26_9": macd_hist,
        "rsi_14": rsi14,
        "stoch_k_14": stoch_k14,
        "stoch_d_3": stoch_d3,
        "williams_r_14": willr14,
        "cci_20": cci20,
        "plus_di_14": plus_di14,
        "minus_di_14": minus_di14,
        "adx_14": adx14,
        "boll_mid_20": mid,
        "boll_up_20": boll_up,
        "boll_low_20": boll_low,
        "keltner_mid_20": ema20_mid,
        "keltner_up_20": kelt_up,
        "keltner_low_20": kelt_low,
        "aroon_up_25": a_up25,
        "aroon_down_25": a_dn25,
        "aroon_osc_25": a_osc25,
        # 5) Volume-derived
        "obv": obv,
        "mfi_14": mfi14,
        "adl": adl,
        "force_index": force,
        # 8) Trade-level
        "avg_trade_size": avg_trade_size,
        "trade_intensity": trade_intensity,
        # 6) Statistical rolling
        "ret_mean_20": ret_mean20,
        "ret_std_20": ret_std20,
        "sharpe_20": sharpe20,
        "sortino_20": sortino_20,
        # Momentum & ROC
        "roc_14": roc_14,
        "mom_14": mom_14,
        "cmo_14": cmo_14,
        "trix_15": trix_15,
        "dpo_20": dpo_20,
        "ultosc_7_14_28": ultosc_7_14_28,
        # Volatility & range-based
        "cv_20": cv_20,
        "chaikin_vol_10_10": chaikin_vol_10_10,
        "donchian_hi_20": donchian_hi_20,
        "donchian_lo_20": donchian_lo_20,
        "prr": prr,
        # Volume-derived ext
        "vroc_14": vroc_14,
        "chaikin_osc_3_10": chaikin_osc_3_10,
        "vpt": vpt,
        "eom_14": eom_14,
        "pvi": pvi,
        "nvi": nvi,
        "vwap_day": vwap_day,
        "vol_osc_14_28": vol_osc_14_28,
        "vol_osc_pct_14_28": vol_osc_pct_14_28,
        # Statistical ext
        "var_20": var_20,
        "skew_20": skew_20,
        "kurt_20": kurt_20,
        "zscore_20": zscore_20,
        # TTM squeeze
        "squeeze_scalar": squeeze_scalar,
        "in_squeeze": in_squeeze,
    })

    advanced_feats = pd.DataFrame({
        "psar": psar,
        "lr_slope_20": lr_slope_20,
        "lr_intercept_20": lr_intercept_20,
        "lr_channel_hi_20": lr_channel_hi_20,
        "lr_channel_lo_20": lr_channel_lo_20,
        "tenkan_9": tenkan_9,
        "kijun_26": kijun_26,
        "span_a_26": span_a_26,
        "span_b_26": span_b_26,
        "chikou_26": chikou_26,
        "rvi_10": rvi_10,
        "elder_bull_13": elder_bull_13,
        "elder_bear_13": elder_bear_13,
        "crsi_3_2_100": crsi_3_2_100,
        "kama_10_2_30": kama_10_2_30,
        "frama_16": frama_16,
        "vi_plus_14": vi_plus_14,
        "vi_minus_14": vi_minus_14,
        "entropy_20": entropy_20,
        "hurst_100": hurst_100,
        "fractal_100": fractal_100,
        "tir": tir,
        "vpt_vol_14": vpt_vol_14,
        "msv_14": msv_14,
        "d_close_2": d_close_2,
        "d_close_3": d_close_3,
        "d_close_5": d_close_5,
        "d_close_10": d_close_10,
        "d_close_14": d_close_14,
        "d_close_20": d_close_20,
        "ewm_mean_alpha_0_1": ewm_mean_alpha_0_1,
        "ewm_mean_alpha_0_2": ewm_mean_alpha_0_2,
        "q25_20": q25_20,
        "q50_20": q50_20,
        "q75_20": q75_20,
        "prank_20": prank_20,
        "dir": dir,
    })

    feats = pd.concat([base_feats, advanced_feats], axis=1)

    return feats


def sanitize_features_no_nan(feats: pd.DataFrame) -> pd.DataFrame:
    """Deterministic stabilize pass before parquet export."""
    out = feats.apply(pd.to_numeric, errors="coerce").astype("float64")
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    out = out.clip(lower=-float(MAX_FEATURE_ABS), upper=float(MAX_FEATURE_ABS))
    if out.isna().any().any():
        raise ValueError("sanitize_features_no_nan: NaNs remain after sanitize pass")
    return out


def _scan_feature_instability(feats: pd.DataFrame) -> List[dict]:
    if feats is None or feats.empty:
        return []
    cols = list(feats.columns)
    if not cols:
        return []
    arr = feats.to_numpy(dtype=np.float64, copy=False)
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    finite_mask = np.isfinite(arr)
    abs_arr = np.abs(arr)
    abs_arr[~finite_mask] = np.nan
    with np.errstate(all="ignore"):
        max_abs_by_col = np.nanmax(abs_arr, axis=0)
    nan_count = np.sum(nan_mask, axis=0)
    inf_count = np.sum(inf_mask, axis=0)
    cap_hits = np.sum((abs_arr >= float(MAX_FEATURE_ABS)) & finite_mask, axis=0)

    out: List[dict] = []
    warn_abs = float(STABILITY_WARN_ABS)
    for i, c in enumerate(cols):
        mx = float(max_abs_by_col[i]) if np.isfinite(max_abs_by_col[i]) else 0.0
        n_nan = int(nan_count[i])
        n_inf = int(inf_count[i])
        n_cap = int(cap_hits[i])
        if not (n_nan > 0 or n_inf > 0 or n_cap > 0 or mx >= warn_abs):
            continue
        notes: List[str] = []
        if mx >= warn_abs:
            notes.append("extreme magnitude")
        if n_cap > 0:
            notes.append("cap hits observed")
        if n_nan > 0 or n_inf > 0:
            notes.append("nan/inf repaired")
        out.append(
            {
                "feature_name": str(c),
                "max_abs_value_observed": float(mx),
                "nan_count_before_sanitation": n_nan,
                "inf_count_before_sanitation": n_inf,
                "cap_hit_count": n_cap,
                "note": "; ".join(notes) if notes else "",
            }
        )
    return out


def _merge_stability_records(accum: dict, records: List[dict]) -> None:
    for rec in records or []:
        name = str(rec.get("feature_name") or "")
        if not name:
            continue
        cur = accum.get(name)
        if cur is None:
            accum[name] = {
                "feature_name": name,
                "max_abs_value_observed": float(rec.get("max_abs_value_observed") or 0.0),
                "nan_count_before_sanitation": int(rec.get("nan_count_before_sanitation") or 0),
                "inf_count_before_sanitation": int(rec.get("inf_count_before_sanitation") or 0),
                "cap_hit_count": int(rec.get("cap_hit_count") or 0),
                "note_set": set([x.strip() for x in str(rec.get("note") or "").split(";") if x.strip()]),
            }
            continue
        cur["max_abs_value_observed"] = max(
            float(cur.get("max_abs_value_observed") or 0.0),
            float(rec.get("max_abs_value_observed") or 0.0),
        )
        cur["nan_count_before_sanitation"] = int(cur.get("nan_count_before_sanitation") or 0) + int(
            rec.get("nan_count_before_sanitation") or 0
        )
        cur["inf_count_before_sanitation"] = int(cur.get("inf_count_before_sanitation") or 0) + int(
            rec.get("inf_count_before_sanitation") or 0
        )
        cur["cap_hit_count"] = int(cur.get("cap_hit_count") or 0) + int(rec.get("cap_hit_count") or 0)
        cur["note_set"].update([x.strip() for x in str(rec.get("note") or "").split(";") if x.strip()])


def _finalize_stability_records(accum: dict) -> List[dict]:
    out: List[dict] = []
    for name in sorted(accum.keys()):
        cur = accum[name]
        notes = sorted(list(cur.get("note_set") or []))
        out.append(
            {
                "feature_name": str(name),
                "max_abs_value_observed": float(cur.get("max_abs_value_observed") or 0.0),
                "nan_count_before_sanitation": int(cur.get("nan_count_before_sanitation") or 0),
                "inf_count_before_sanitation": int(cur.get("inf_count_before_sanitation") or 0),
                "cap_hit_count": int(cur.get("cap_hit_count") or 0),
                "note": "; ".join(notes) if notes else "",
            }
        )
    return out

# ------------------------------------------------
# Parquet export helpers
# ------------------------------------------------
def _validate_strict_timegrid(ts: pd.Series, interval_min: int, context: str) -> None:
    shared_validate_strict_timegrid(ts, interval_min=int(interval_min), context=context)


def month_start_ts(year: int, month: int) -> int:
    return int(datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def _next_month(y: int, m: int) -> tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def next_month_start_ts(year: int, month: int) -> int:
    y2, m2 = _next_month(year, month)
    return month_start_ts(y2, m2)


def iter_months_between(start_ts: int, end_ts: int):
    if end_ts is None or start_ts is None or end_ts < start_ts:
        return
    dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    y, m = dt.year, dt.month
    while True:
        yield (y, m)
        y, m = _next_month(y, m)
        if month_start_ts(y, m) > end_ts:
            break


def parquet_path_for(interval_min: int, asset: str, year: int, month: int, root: Path) -> Path:
    return (
        root
        / f"scalar_features_{interval_min}"
        / f"asset={asset}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"part-scalar_features_{interval_min}-{asset}-{year}{month:02d}.parquet"
    )


def ohlcvt_parquet_path_for(interval_min: int, year: int, month: int, root: Path) -> Path:
    raise RuntimeError("legacy non-asset OHLCVT path is disabled; use ohlcvt_source/read_ohlcvt canonical asset-partitioned readers.")


def coerce_ohlc_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "asset" in df.columns:
        df["asset"] = df["asset"].astype("string")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")
    if "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    return df


def normalize_ohlc_schema(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["asset", "ts", "open", "high", "low", "close", "volume", "trades"])
    out = df.copy()
    ren = {}
    if "timestamp" in out.columns and "ts" not in out.columns:
        ren["timestamp"] = "ts"
    if "count" in out.columns and "trades" not in out.columns:
        ren["count"] = "trades"
    if "vol" in out.columns and "volume" not in out.columns:
        ren["vol"] = "volume"
    if ren:
        out = out.rename(columns=ren)
    required = ["asset", "ts", "open", "high", "low", "close", "volume", "trades"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{context}: missing canonical OHLCVT columns: {missing}")
    return out[required]


def read_ohlc_parquet_window(interval_min: int, start_ts: int, end_ts: int, root: Optional[Path] = None, asset: Optional[str] = None) -> pd.DataFrame:
    """Read OHLCVT rows over [start_ts, end_ts].
    If `asset` provided and pyarrow is available, push down filters to reduce memory.
    Fallback: per-month read with immediate per-asset filter to avoid loading all assets.
    """
    if asset is None:
        return pd.DataFrame(columns=["asset", "ts", "open", "high", "low", "close", "volume", "trades"])

    df = read_ohlcvt(
        asset=str(asset),
        interval_min=int(interval_min),
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        columns=["asset", "ts", "open", "high", "low", "close", "volume", "trades"],
        root=Path(root) if root else OHLCVT_PARQUET_ROOT,
    )
    df = normalize_ohlc_schema(df, context=f"read_ohlcvt asset={asset} k={interval_min}")
    df = coerce_ohlc_dtypes(df)
    if not df.empty:
        df = df.drop_duplicates(subset=["asset", "ts"], keep="last").sort_values(["asset", "ts"]).reset_index(drop=True)
    try:
        log(f"[parquet][read][ohlcvt] k={interval_min} asset={asset}: window_rows={len(df):,} window=[{start_ts},{end_ts}]")
    except Exception:
        pass
    return df


def write_parquet_atomic(
    df: pd.DataFrame,
    dst: Path,
    dedupe_keys: Optional[List[str]] = None,
    replace_asset: Optional[str] = None,
    replace_start_ts: Optional[int] = None,
    replace_end_ts: Optional[int] = None,
    replace_entire_file: bool = False,
) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df = df
    if dedupe_keys:
        out_df = out_df.drop_duplicates(subset=dedupe_keys, keep="last")

    if replace_entire_file:
        merged = out_df.copy()
        if dedupe_keys:
            merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")
        if "asset" in merged.columns and "ts" in merged.columns:
            merged = merged.sort_values(["asset", "ts"]).reset_index(drop=True)
        tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
        merged.to_parquet(tmp, engine="pyarrow", compression=PARQUET_COMPRESSION, index=False, row_group_size=PARQUET_ROW_GROUP)
        atomic_replace(tmp, dst)
        min_ts = int(out_df["ts"].min()) if "ts" in out_df.columns and not out_df.empty else None
        max_ts = int(out_df["ts"].max()) if "ts" in out_df.columns and not out_df.empty else None
        return {
            "path": str(dst),
            "rows": int(len(out_df)),
            "month_rows": int(len(merged)),
            "min_ts": min_ts,
            "max_ts": max_ts,
        }

    if dst.exists():
        try:
            existing = pd.read_parquet(dst)
        except Exception:
            existing = pd.DataFrame(columns=out_df.columns)
    else:
        existing = pd.DataFrame(columns=out_df.columns)

    if existing.empty:
        merged = out_df.copy()
    else:
        existing = existing.copy()
        # Upsert-by-ts semantics:
        # keep all existing rows except keys present in out_df, then append out_df (out_df wins).
        if "ts" in existing.columns and "ts" in out_df.columns and not out_df.empty:
            existing_ts = pd.to_numeric(existing["ts"], errors="coerce")
            new_ts = set(pd.to_numeric(out_df["ts"], errors="coerce").dropna().astype("int64").tolist())
            if new_ts:
                if "asset" in existing.columns and "asset" in out_df.columns:
                    new_assets = set(out_df["asset"].astype(str).tolist())
                    if replace_asset is not None:
                        new_assets.add(str(replace_asset))
                    mask_asset = existing["asset"].astype(str).isin(new_assets)
                else:
                    mask_asset = pd.Series(True, index=existing.index)
                mask_ts = existing_ts.astype("Int64").isin(new_ts)
                existing = existing[~(mask_asset & mask_ts)]
        merged = pd.concat([existing, out_df], ignore_index=True)

    if dedupe_keys:
        merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")
    if "asset" in merged.columns and "ts" in merged.columns:
        merged = merged.sort_values(["asset", "ts"]).reset_index(drop=True)

    tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
    merged.to_parquet(tmp, engine="pyarrow", compression=PARQUET_COMPRESSION, index=False, row_group_size=PARQUET_ROW_GROUP)
    atomic_replace(tmp, dst)
    min_ts = int(out_df["ts"].min()) if "ts" in out_df.columns and not out_df.empty else None
    max_ts = int(out_df["ts"].max()) if "ts" in out_df.columns and not out_df.empty else None
    return {
        "path": str(dst),
        "rows": int(len(out_df)),
        "month_rows": int(len(merged)),
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


# ------------------------------------------------
# Main compute/update
# ------------------------------------------------
# Lookback must cover the largest window used by indicators in compute_features.
# Current maximum explicit window is 100 (e.g., Hurst(100), Fractal(100), percent-rank(100)).
MAX_LOOKBACK_BARS = 100
CHUNK_MAX_ROWS = int(os.getenv("SCALAR_FEATURES_CHUNK_MAX_ROWS", "200000"))
CHUNK_MAX_DAYS = int(os.getenv("SCALAR_FEATURES_CHUNK_MAX_DAYS", "365"))
WRITE_QUEUE_MAX_MONTHS = int(os.getenv("SCALAR_FEATURES_WRITE_QUEUE_MAX_MONTHS", "6"))
VALIDATE_WRITES_ON_DISK = str(os.getenv("SCALAR_FEATURES_VALIDATE_WRITE_ON_DISK", "0")).strip().lower() in {"1", "true", "yes", "on"}


class MonthValidationError(PipelineValidationError):
    pass


def _validate_expected_grid(
    ts_series: pd.Series,
    interval_min: int,
    expected_start: int,
    expected_end: int,
    context: str,
) -> None:
    try:
        shared_validate_expected_grid(
            ts_series,
            interval_min=int(interval_min),
            expected_start=int(expected_start),
            expected_end=int(expected_end),
            context=context,
        )
    except PipelineValidationError as exc:
        raise MonthValidationError(str(exc), first_bad_ts=exc.first_bad_ts) from exc


def _validate_no_nan_features(df: pd.DataFrame, context: str) -> None:
    cols = [c for c in df.columns if c not in {"asset", "ts"}]
    try:
        shared_validate_no_nan_columns(df, columns=cols, context=context, ts_column="ts")
    except PipelineValidationError as exc:
        raise MonthValidationError(str(exc), first_bad_ts=exc.first_bad_ts) from exc


def _validate_written_month_partition(
    *,
    asset: str,
    interval_min: int,
    year: int,
    month: int,
    expected_start: int,
    expected_end: int,
) -> None:
    dst = parquet_path_for(interval_min, asset, year, month, PARQUET_ROOT)
    if not dst.exists():
        raise MonthValidationError(
            f"post-write missing file asset={asset} k={interval_min} month={year:04d}-{month:02d}",
            first_bad_ts=int(expected_start),
        )
    d = pd.read_parquet(dst)
    if d.empty:
        raise MonthValidationError(
            f"post-write empty file asset={asset} k={interval_min} month={year:04d}-{month:02d}",
            first_bad_ts=int(expected_start),
        )
    d = d[d["asset"].astype(str) == str(asset)].copy()
    if d.empty:
        raise MonthValidationError(
            f"post-write missing asset rows asset={asset} k={interval_min} month={year:04d}-{month:02d}",
            first_bad_ts=int(expected_start),
        )
    d = d.drop_duplicates(subset=["asset", "ts"], keep="last").sort_values(["asset", "ts"]).reset_index(drop=True)
    _validate_no_nan_features(d, f"post-write asset={asset} k={interval_min} month={year:04d}-{month:02d}")
    w = d[(d["ts"].astype("int64") >= int(expected_start)) & (d["ts"].astype("int64") <= int(expected_end))]
    _validate_expected_grid(
        w["ts"],
        interval_min,
        int(expected_start),
        int(expected_end),
        f"post-write grid asset={asset} k={interval_min} month={year:04d}-{month:02d}",
    )


def _plan_compute_chunks(
    *,
    start_ts: int,
    end_ts: int,
    interval_min: int,
    max_rows: int,
    max_days: int,
) -> List[Tuple[int, int]]:
    """Plan contiguous chunk windows over [start_ts, end_ts] with bounded row count/memory."""
    step = int(interval_min) * 60
    if int(end_ts) < int(start_ts):
        return []
    bounded_rows = max(int(max_rows), int(MAX_LOOKBACK_BARS) + 1)
    by_rows_span = (bounded_rows - 1) * step
    by_days_span = max(step, int(max_days) * 86400 - 1)
    span = max(step, min(int(by_rows_span), int(by_days_span)))
    span = (span // step) * step
    if span < step:
        span = step

    out: List[Tuple[int, int]] = []
    cur = int(start_ts)
    end = int(end_ts)
    while cur <= end:
        nxt = min(end, cur + span)
        nxt = cur + ((nxt - cur) // step) * step
        out.append((int(cur), int(nxt)))
        cur = int(nxt) + step
    return out


def _build_feature_span(
    *,
    asset: str,
    interval_min: int,
    span_start: int,
    span_end: int,
    lookback_secs: int,
) -> Tuple[pd.DataFrame, int, int]:
    feats_out, eff_start, eff_end, _timing = _build_feature_span_timed(
        asset=asset,
        interval_min=interval_min,
        span_start=span_start,
        span_end=span_end,
        lookback_secs=lookback_secs,
    )
    return feats_out, int(eff_start), int(eff_end)


def _build_feature_span_timed(
    *,
    asset: str,
    interval_min: int,
    span_start: int,
    span_end: int,
    lookback_secs: int,
) -> Tuple[pd.DataFrame, int, int, dict]:
    read_s = 0.0
    compute_s = 0.0
    validation_s = 0.0

    t0 = time.perf_counter()
    read_start = max(0, int(span_start) - int(lookback_secs))
    df_src = read_ohlc_parquet_window(interval_min, read_start, int(span_end), root=OHLCVT_PARQUET_ROOT, asset=asset)
    if df_src.empty:
        raise MonthValidationError(
            f"no source rows asset={asset} k={interval_min} span_start={span_start}",
            first_bad_ts=int(span_start),
        )
    df_src = df_src[["ts", "open", "high", "low", "close", "volume", "trades"]].copy()
    df_src = df_src.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    _validate_strict_timegrid(df_src["ts"], interval_min, f"source ohlc asset={asset} k={interval_min} start={span_start}")

    ts_vals = pd.to_numeric(df_src["ts"], errors="coerce").dropna().astype("int64")
    in_span = ts_vals[(ts_vals >= int(span_start)) & (ts_vals <= int(span_end))]
    if in_span.empty:
        src_min = int(ts_vals.min()) if not ts_vals.empty else None
        src_max = int(ts_vals.max()) if not ts_vals.empty else None
        raise MonthValidationError(
            f"no source overlap asset={asset} k={interval_min} window=[{span_start},{span_end}] src=[{src_min},{src_max}]",
            first_bad_ts=int(span_start),
        )
    eff_start = int(in_span.iloc[0])
    eff_end = int(in_span.iloc[-1])
    if eff_start != int(span_start) or eff_end != int(span_end):
        log(
            f"[window][clip] asset={asset} k={interval_min} requested=[{span_start},{span_end}] "
            f"effective=[{eff_start},{eff_end}] src=[{int(ts_vals.min())},{int(ts_vals.max())}]"
        )
    read_s += max(0.0, time.perf_counter() - t0)

    t0 = time.perf_counter()
    feats = compute_features(df_src)
    stability_chunk = _scan_feature_instability(feats)
    feats = sanitize_features_no_nan(feats)
    feats["ts"] = df_src["ts"].astype("int64")
    feats["asset"] = asset
    compute_s += max(0.0, time.perf_counter() - t0)

    t0 = time.perf_counter()
    feats_out = feats[(feats["ts"] >= int(eff_start)) & (feats["ts"] <= int(eff_end))].copy()
    feats_out = feats_out.drop_duplicates(subset=["asset", "ts"], keep="last").sort_values(["asset", "ts"]).reset_index(drop=True)
    _validate_no_nan_features(feats_out, f"pre-write asset={asset} k={interval_min} start={span_start}")
    _validate_expected_grid(
        feats_out["ts"],
        interval_min,
        int(eff_start),
        int(eff_end),
        f"pre-write grid asset={asset} k={interval_min} start={eff_start}",
    )
    validation_s += max(0.0, time.perf_counter() - t0)
    return feats_out, int(eff_start), int(eff_end), {
        "ohlc_read_s": float(read_s),
        "compute_features_s": float(compute_s),
        "validation_s": float(validation_s),
        "chunk_min_ts": int(eff_start),
        "chunk_max_ts": int(eff_end),
        "stability_chunk": stability_chunk,
    }


def _build_month_features(
    *,
    asset: str,
    interval_min: int,
    month_start: int,
    month_end: int,
    lookback_secs: int,
) -> Tuple[pd.DataFrame, int, int]:
    return _build_feature_span(
        asset=asset,
        interval_min=interval_min,
        span_start=month_start,
        span_end=month_end,
        lookback_secs=lookback_secs,
    )


def _write_month_with_validation(
    *,
    asset: str,
    interval_min: int,
    year: int,
    month: int,
    win_start: int,
    win_end: int,
    feats_out: pd.DataFrame,
    full_month_regen: bool = False,
) -> dict:
    validation_s = 0.0
    write_s = 0.0
    t0 = time.perf_counter()
    _validate_no_nan_features(feats_out, f"write buffer asset={asset} k={interval_min} month={year:04d}-{month:02d}")
    _validate_expected_grid(
        feats_out["ts"],
        interval_min,
        int(win_start),
        int(win_end),
        f"write buffer grid asset={asset} k={interval_min} month={year:04d}-{month:02d}",
    )
    validation_s += max(0.0, time.perf_counter() - t0)
    dst = parquet_path_for(interval_min, asset, year, month, PARQUET_ROOT)
    lock_path = _partition_lock_path(dst)
    lock_fd: Optional[int] = None
    try:
        lock_fd = _acquire_lockfile(lock_path, timeout_sec=120.0, poll_sec=0.05)
        t0 = time.perf_counter()
        part_info = write_parquet_atomic(
            feats_out,
            dst,
            dedupe_keys=["asset", "ts"],
            replace_asset=asset,
            replace_start_ts=int(win_start),
            replace_end_ts=int(win_end),
            replace_entire_file=bool(full_month_regen),
        )
        write_s += max(0.0, time.perf_counter() - t0)
        if VALIDATE_WRITES_ON_DISK:
            t0 = time.perf_counter()
            _validate_written_month_partition(
                asset=asset,
                interval_min=interval_min,
                year=year,
                month=month,
                expected_start=int(win_start),
                expected_end=int(win_end),
            )
            validation_s += max(0.0, time.perf_counter() - t0)
    finally:
        _release_lockfile(lock_path, lock_fd)
    part_info["_write_s"] = float(write_s)
    part_info["_validation_s"] = float(validation_s)
    return part_info


def _regenerate_full_month(
    *,
    asset: str,
    interval_min: int,
    year: int,
    month: int,
    ohlc_edge_ts: int,
) -> dict:
    step = int(interval_min) * 60
    lookback_secs = (MAX_LOOKBACK_BARS - 1) * step
    full_start = month_start_ts(year, month)
    full_end = min(next_month_start_ts(year, month) - 1, int(ohlc_edge_ts))
    if full_end < full_start:
        raise MonthValidationError(
            f"regen empty month span asset={asset} k={interval_min} month={year:04d}-{month:02d}",
            first_bad_ts=int(full_start),
        )
    feats_full, eff_start, eff_end = _build_month_features(
        asset=asset,
        interval_min=interval_min,
        month_start=int(full_start),
        month_end=int(full_end),
        lookback_secs=int(lookback_secs),
    )
    return _write_month_with_validation(
        asset=asset,
        interval_min=interval_min,
        year=year,
        month=month,
        win_start=int(eff_start),
        win_end=int(eff_end),
        feats_out=feats_full,
        full_month_regen=True,
    )


def _hard_stop_month_failure(
    *,
    asset: str,
    interval_min: int,
    year: int,
    month: int,
    err: Exception,
) -> None:
    first_bad_ts = getattr(err, "first_bad_ts", None)
    msg = (
        f"[hard-stop] asset={asset} k={interval_min} month={year:04d}-{month:02d} "
        f"first_bad_ts={first_bad_ts} error={err}"
    )
    log(msg)
    raise RuntimeError(msg) from err


def _decide_range_from_disk_edges(
    *,
    asset: str,
    interval_min: int,
    feat_max_ts: Optional[int],
    ohlc_min_ts: Optional[int],
    ohlc_max_ts: Optional[int],
    mode: str,
    backfill_range: Optional[Tuple[int, int]],
) -> Tuple[Optional[int], Optional[int], str]:
    start_ts, end_ts, reason = shared_decide_range_from_disk_edges(
        asset=str(asset),
        interval_min=int(interval_min),
        downstream_max_ts=(int(feat_max_ts) if feat_max_ts is not None else None),
        upstream_min_ts=(int(ohlc_min_ts) if ohlc_min_ts is not None else None),
        upstream_max_ts=(int(ohlc_max_ts) if ohlc_max_ts is not None else None),
        mode=str(mode),
        backfill_range=backfill_range,
    )
    if reason == "no_upstream":
        return None, None, "no_ohlc"
    if reason == "no_upstream_head":
        return None, None, "no_ohlc_head"
    return start_ts, end_ts, reason


def compute_for_asset_interval(
    *,
    asset: str,
    interval_min: int,
    start_ts: int,
    end_ts: int,
    ohlc_edge_ts: int,
) -> Tuple[int, List[dict], dict]:
    if not wait_for_resources():
        raise RuntimeError("resource wait exceeded; aborting task")
    step = int(interval_min) * 60
    lookback_secs = (MAX_LOOKBACK_BARS - 1) * step
    run_end = min(int(end_ts), int(ohlc_edge_ts))
    if run_end < int(start_ts):
        return 0, [], {"read_s": 0.0, "compute_s": 0.0, "write_s": 0.0, "months_committed": 0}
    chunks = _plan_compute_chunks(
        start_ts=int(start_ts),
        end_ts=int(run_end),
        interval_min=int(interval_min),
        max_rows=int(CHUNK_MAX_ROWS),
        max_days=int(CHUNK_MAX_DAYS),
    )
    log(
        f"[compute][chunk-plan] asset={asset} k={interval_min} "
        f"range=[{start_ts},{run_end}] chunks={len(chunks)} "
        f"max_rows={CHUNK_MAX_ROWS} max_days={CHUNK_MAX_DAYS}"
    )
    inserted = 0
    read_time_s = 0.0
    compute_time_s = 0.0
    partition_time_s = 0.0
    write_time_s = 0.0
    validation_time_s = 0.0
    committed_months = 0
    parts_written: List[dict] = []
    stability_accum: dict = {}
    span_min_ts: Optional[int] = None
    span_max_ts: Optional[int] = None

    month_buffers: dict[Tuple[int, int], List[pd.DataFrame]] = {}
    month_windows: dict[Tuple[int, int], Tuple[int, int]] = {}

    def _flush_month(y: int, m: int) -> None:
        nonlocal inserted, write_time_s, validation_time_s, committed_months
        key = (int(y), int(m))
        frames = month_buffers.pop(key, None)
        if not frames:
            return
        win = month_windows.pop(key, None)
        if win is None:
            return
        win_start, win_end = int(win[0]), int(win[1])
        month_df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        month_df = (
            month_df.drop_duplicates(subset=["asset", "ts"], keep="last")
            .sort_values(["asset", "ts"])
            .reset_index(drop=True)
        )
        try:
            part_info = _write_month_with_validation(
                asset=asset,
                interval_min=interval_min,
                year=y,
                month=m,
                win_start=int(win_start),
                win_end=int(win_end),
                feats_out=month_df,
                full_month_regen=False,
            )
        except Exception as first_err:
            try:
                log(f"[repair] regenerate full month asset={asset} k={interval_min} month={y:04d}-{m:02d}")
                t0 = time.perf_counter()
                part_info = _regenerate_full_month(
                    asset=asset,
                    interval_min=interval_min,
                    year=y,
                    month=m,
                    ohlc_edge_ts=int(ohlc_edge_ts),
                )
                write_time_s += max(0.0, time.perf_counter() - t0)
            except Exception as second_err:
                _hard_stop_month_failure(asset=asset, interval_min=interval_min, year=y, month=m, err=second_err)
                raise second_err
            else:
                log(
                    f"[repair] month recovered asset={asset} k={interval_min} month={y:04d}-{m:02d} "
                    f"after_error={first_err}"
                )
        part_info.update({"asset": asset, "interval": interval_min, "year": y, "month": m})
        write_time_s += float(part_info.pop("_write_s", 0.0) or 0.0)
        validation_time_s += float(part_info.pop("_validation_s", 0.0) or 0.0)
        parts_written.append(part_info)
        inserted += int(part_info.get("rows") or 0)
        committed_months += 1
        if (
            TEST_CRASH_AFTER_MONTH_COMMITS > 0
            and committed_months >= int(TEST_CRASH_AFTER_MONTH_COMMITS)
            and (not TEST_CRASH_ASSET or str(asset) == str(TEST_CRASH_ASSET))
            and (TEST_CRASH_INTERVAL <= 0 or int(interval_min) == int(TEST_CRASH_INTERVAL))
        ):
            raise RuntimeError(
                f"test_crash_after_month_commit asset={asset} k={interval_min} committed_months={committed_months}"
            )

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        feats_out, eff_start, eff_end, phase = _build_feature_span_timed(
            asset=asset,
            interval_min=interval_min,
            span_start=int(chunk_start),
            span_end=int(chunk_end),
            lookback_secs=int(lookback_secs),
        )
        read_time_s += float(phase.get("ohlc_read_s") or 0.0)
        compute_time_s += float(phase.get("compute_features_s") or 0.0)
        validation_time_s += float(phase.get("validation_s") or 0.0)
        chunk_min = phase.get("chunk_min_ts")
        chunk_max = phase.get("chunk_max_ts")
        if chunk_min is not None:
            cmin = int(chunk_min)
            span_min_ts = cmin if span_min_ts is None else min(span_min_ts, cmin)
        if chunk_max is not None:
            cmax = int(chunk_max)
            span_max_ts = cmax if span_max_ts is None else max(span_max_ts, cmax)
        _merge_stability_records(stability_accum, list(phase.get("stability_chunk") or []))

        t_part = time.perf_counter()
        for y, m in iter_months_between(int(eff_start), int(eff_end)):
            m_start = max(month_start_ts(y, m), int(eff_start))
            m_end = min(next_month_start_ts(y, m) - 1, int(eff_end))
            if m_end < m_start:
                continue
            month_slice = feats_out[(feats_out["ts"] >= int(m_start)) & (feats_out["ts"] <= int(m_end))].copy()
            if month_slice.empty:
                continue
            actual_start = int(pd.to_numeric(month_slice["ts"], errors="coerce").min())
            actual_end = int(pd.to_numeric(month_slice["ts"], errors="coerce").max())
            key = (int(y), int(m))
            month_buffers.setdefault(key, []).append(month_slice)
            prev = month_windows.get(key)
            if prev is None:
                month_windows[key] = (int(actual_start), int(actual_end))
            else:
                month_windows[key] = (min(int(prev[0]), int(actual_start)), max(int(prev[1]), int(actual_end)))

        cur_dt = datetime.fromtimestamp(int(eff_end), tz=timezone.utc)
        current_month_key = (int(cur_dt.year), int(cur_dt.month))
        ready_keys = sorted([k for k in month_buffers.keys() if k < current_month_key])
        partition_time_s += max(0.0, time.perf_counter() - t_part)
        for y, m in ready_keys:
            _flush_month(y, m)

        while len(month_buffers) > int(max(1, WRITE_QUEUE_MAX_MONTHS)):
            oldest = sorted(month_buffers.keys())[0]
            _flush_month(int(oldest[0]), int(oldest[1]))

        log(
            f"[compute][chunk] asset={asset} k={interval_min} idx={idx}/{len(chunks)} "
            f"chunk=[{chunk_start},{chunk_end}] effective=[{eff_start},{eff_end}] "
            f"rows={len(feats_out):,} buffered_months={len(month_buffers)}"
        )

    for y, m in sorted(month_buffers.keys()):
        _flush_month(int(y), int(m))

    timing = {
        "read_s": float(read_time_s),
        "compute_s": float(compute_time_s),
        "partition_s": float(partition_time_s),
        "write_s": float(write_time_s),
        "validation_s": float(validation_time_s),
        "months_committed": int(committed_months),
        "chunks": int(len(chunks)),
        "span_min_ts": int(span_min_ts) if span_min_ts is not None else None,
        "span_max_ts": int(span_max_ts) if span_max_ts is not None else None,
        "stability": _finalize_stability_records(stability_accum),
    }
    return inserted, parts_written, timing


def main():
    require_pipeline_io(profile=PIPELINE_PROFILE)
    parser = argparse.ArgumentParser(description="Compute scalar feature tables per interval, aligned with OHLCVT disk edges.")
    parser.add_argument("--intervals", type=str, default="1,5,15,30,60,240,720,1440", help="Comma list of intervals to compute")
    parser.add_argument("--assets", type=str, default="", help="Comma list of assets to limit")
    parser.add_argument("--workers", type=int, default=None, help="Legacy alias for --compute-workers")
    parser.add_argument("--scan-workers", type=int, default=None, help="Number of scan worker processes")
    parser.add_argument("--compute-workers", type=int, default=None, help="Number of compute worker processes")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--incremental", action="store_true", help="Incremental mode (default): disk-edge scheduling behavior")
    mode_group.add_argument("--backfill", nargs=2, metavar=("START", "END"), help="Explicit repair range [START, END] unix-ts")
    args = parser.parse_args()

    intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
    if not intervals:
        intervals = DEFAULT_INTERVALS

    default_scan_workers = get_workers("scalar_features", "scan_workers")
    default_compute_workers = get_workers("scalar_features", "compute_workers")
    scan_workers = int(args.scan_workers) if args.scan_workers is not None else default_scan_workers
    compute_workers = int(args.compute_workers) if args.compute_workers is not None else (
        int(args.workers) if args.workers is not None else default_compute_workers
    )
    scan_workers = max(1, int(scan_workers))
    compute_workers = max(1, int(compute_workers))
    log_resolved_runtime(
        "scalar_features",
        resolved={
            "scan_workers": int(scan_workers),
            "compute_workers": int(compute_workers),
            "commit_workers": 1,
            "model_threads": "n/a",
        },
    )

    mode = "backfill" if args.backfill else "incremental"
    backfill_range: Optional[Tuple[int, int]] = None
    if args.backfill:
        try:
            backfill_range = (int(args.backfill[0]), int(args.backfill[1]))
        except Exception:
            log("[features][error] --backfill requires START END as unix timestamps")
            sys.exit(1)

    all_assets: set[str] = set()
    for k in intervals:
        all_assets.update(list_assets_from_ohlcvt(k))
    if args.assets:
        sel = {a.strip() for a in args.assets.split(",") if a.strip()}
        assets = sorted(a for a in all_assets if a in sel)
    else:
        assets = sorted(all_assets)
    if not assets:
        log("No assets found; exiting.")
        sys.exit(0)

    tasks = [(asset, k) for asset in assets for k in intervals]
    asset_count = len({a for a, _ in tasks})
    interval_count = len({k for _, k in tasks})
    log(
        f"[features] starting compute assets={asset_count} intervals={interval_count} "
        f"scan_workers={scan_workers} compute_workers={compute_workers} mode={mode}"
    )
    log(f"[features][stability] denom_floor={DENOM_FLOOR} max_feature_abs={MAX_FEATURE_ABS}")

    feat_idx, ohlc_idx = scan_bounds_for_tasks(tasks, scan_workers, context="[features][scan][edges]")

    scheduled: List[Tuple[str, int, int, int, int, str]] = []
    skipped = 0
    for asset, k in tasks:
        feat_max_raw = feat_idx.get(int(k), {}).get(str(asset))
        ohlc_max_raw = ohlc_idx.get(int(k), {}).get(str(asset))
        feat_max = int(feat_max_raw) if feat_max_raw is not None else None
        ohlc_max = int(ohlc_max_raw) if ohlc_max_raw is not None else None
        need_ohlc_min = bool(mode == "backfill" or feat_max is None)
        ohlc_min = first_ohlcvt_ts_from_disk(int(k), str(asset), root=OHLCVT_PARQUET_ROOT) if need_ohlc_min else None
        start_ts, end_ts, reason = _decide_range_from_disk_edges(
            asset=str(asset),
            interval_min=int(k),
            feat_max_ts=feat_max,
            ohlc_min_ts=ohlc_min,
            ohlc_max_ts=ohlc_max,
            mode=mode,
            backfill_range=backfill_range,
        )
        if start_ts is None or end_ts is None:
            skipped += 1
            continue
        scheduled.append((asset, int(k), int(start_ts), int(end_ts), int(ohlc_max or end_ts), reason))

    log(f"[features][resume] scheduled={len(scheduled)} skipped={skipped} total={len(tasks)}")

    total_rows = 0
    run_parts: List[dict] = []
    task_timings: dict[str, dict] = {}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_ts_iso = datetime.now(timezone.utc).isoformat()

    if not scheduled:
        log("[features] nothing to do; all tasks are at OHLC edge")
    elif compute_workers == 1:
        for asset, k, start_ts, end_ts, ohlc_edge, reason in scheduled:
            log(f"[compute] asset={asset} k={k} reason={reason} range=[{start_ts},{end_ts}] ohlc_edge={ohlc_edge}")
            rows, parts, timing = compute_for_asset_interval(
                asset=asset,
                interval_min=k,
                start_ts=start_ts,
                end_ts=end_ts,
                ohlc_edge_ts=ohlc_edge,
            )
            total_rows += int(rows)
            run_parts.extend(parts)
            task_timings[f"{asset}|{k}"] = dict(timing or {})
    else:
        with ProcessPoolExecutor(max_workers=compute_workers) as pool:
            future_map = {
                pool.submit(
                    compute_for_asset_interval,
                    asset=asset,
                    interval_min=k,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    ohlc_edge_ts=ohlc_edge,
                ): (asset, k)
                for asset, k, start_ts, end_ts, ohlc_edge, _reason in scheduled
            }
            for fut in as_completed(future_map):
                asset, k = future_map[fut]
                rows, parts, timing = fut.result()
                total_rows += int(rows)
                run_parts.extend(parts)
                task_timings[f"{asset}|{k}"] = dict(timing or {})

    stability_entries: List[dict] = []
    for asset, k, _start_ts, _end_ts, _ohlc_edge, _reason in scheduled:
        key = f"{asset}|{k}"
        timing = task_timings.get(key) or {}
        issues = list(timing.get("stability") or [])
        if not issues:
            continue
        stability_entries.append(
            {
                "asset": str(asset),
                "interval": int(k),
                "run_timestamp": str(run_ts_iso),
                "min_ts": timing.get("span_min_ts"),
                "max_ts": timing.get("span_max_ts"),
                "issues": issues,
            }
        )
    stability_report = {
        "run_id": run_id,
        "run_timestamp": run_ts_iso,
        "mode": mode,
        "warning_threshold_abs": float(STABILITY_WARN_ABS),
        "global_cap_abs": float(MAX_FEATURE_ABS),
        "tasks_scheduled": int(len(scheduled)),
        "tasks_with_issues": int(len(stability_entries)),
        "entries": stability_entries,
    }
    write_json(STABILITY_REPORT_FILE, stability_report)
    log(
        f"[features][stability] wrote report path={STABILITY_REPORT_FILE} "
        f"tasks_with_issues={len(stability_entries)}"
    )

    manifest = {
        "run_id": run_id,
        "mode": mode,
        "assets": sorted({p["asset"] for p in run_parts}),
        "intervals": sorted({p["interval"] for p in run_parts}),
        "parts": run_parts,
        "task_timings": task_timings,
    }
    manifest_path = MANIFEST_DIR / f"run_manifest_{run_id}.json"
    write_json(manifest_path, manifest)
    log(f"[features] total rows inserted: {total_rows}")


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        try:
            log("[features] interrupted by user")
        except Exception:
            pass
        sys.exit(130)
