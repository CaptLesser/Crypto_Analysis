from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.ml_module_utils import make_unit_key
from src.forecasting.common.pipeline_parquet_utils import validate_strict_timegrid, validate_no_nan_columns
from src.features.scalar_features import PARQUET_COMPRESSION, PARQUET_ROW_GROUP
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots
from src.forecasting.common.stats_module_utils import (
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PARQUET_ROOT,
    DEFAULT_WORKERS,
    NUMERIC_TASK_TO_TARGET_COLUMN,
    interval_edge_ts,
    interval_min_ts,
    resolve_assets,
    resolve_seasonality_profile,
)


SEED = int(os.getenv("FORECAST_FAMILY_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)

_PARQUET_SCHEMA_NAMES_CACHE: Dict[Tuple[str, int, int, int], set[str]] = {}
_PARQUET_SCHEMA_NAMES_CACHE_MAX = 4096
_FEATURE_WINDOW_READ_CACHE_MAX_ENTRIES = 128
_FEATURE_WINDOW_READ_CACHE_MAX_BYTES = 256 * 1024 * 1024

_FeatureWindowFileStatKey = Tuple[str, int, int, int]
_FeatureWindowReadKey = Tuple[_FeatureWindowFileStatKey, Tuple[str, ...]]


@dataclass
class _FeatureWindowReadCacheEntry:
    file_key: _FeatureWindowFileStatKey
    columns: Tuple[str, ...]
    frame: pd.DataFrame
    byte_size: int


class _FeatureWindowReadCache:
    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._entries: "OrderedDict[_FeatureWindowReadKey, _FeatureWindowReadCacheEntry]" = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._oversize_skips = 0
        self._lock = RLock()

    @staticmethod
    def file_key(path: Path) -> _FeatureWindowFileStatKey:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        ctime_ns = getattr(stat, "st_ctime_ns", None)
        if ctime_ns is None:
            ctime_ns = int(float(getattr(stat, "st_ctime", 0.0)) * 1_000_000_000)
        return (
            str(resolved).lower(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(ctime_ns),
        )

    @staticmethod
    def _frame_bytes(frame: pd.DataFrame) -> int:
        try:
            return int(frame.memory_usage(index=True, deep=True).sum())
        except Exception:
            return int(getattr(frame, "size", 0)) * 8

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0
            self._puts = 0
            self._evictions = 0
            self._oversize_skips = 0

    def get(self, file_key: _FeatureWindowFileStatKey, present_cols: Sequence[str]) -> Optional[pd.DataFrame]:
        requested = tuple(str(col) for col in present_cols)
        requested_set = set(requested)
        with self._lock:
            for key, entry in reversed(self._entries.items()):
                if entry.file_key != file_key:
                    continue
                if not requested_set.issubset(set(entry.columns)):
                    continue
                self._entries.move_to_end(key)
                self._hits += 1
                return entry.frame.loc[:, list(requested)].copy()
            self._misses += 1
        return None

    def put(self, file_key: _FeatureWindowFileStatKey, present_cols: Sequence[str], frame: pd.DataFrame) -> None:
        columns = tuple(str(col) for col in present_cols)
        if not columns:
            return
        stored = frame.loc[:, list(columns)].copy()
        byte_size = self._frame_bytes(stored)
        if byte_size > self._max_bytes:
            with self._lock:
                self._oversize_skips += 1
            return
        key = (file_key, columns)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= int(old.byte_size)
            self._entries[key] = _FeatureWindowReadCacheEntry(
                file_key=file_key,
                columns=columns,
                frame=stored,
                byte_size=byte_size,
            )
            self._bytes += byte_size
            self._puts += 1
            self._evict()

    def _evict(self) -> None:
        while self._entries and (len(self._entries) > self._max_entries or self._bytes > self._max_bytes):
            _, entry = self._entries.popitem(last=False)
            self._bytes -= int(entry.byte_size)
            self._evictions += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "cache_entries": int(len(self._entries)),
                "cache_bytes_estimate": int(self._bytes),
                "cache_hit_count": int(self._hits),
                "cache_miss_count": int(self._misses),
                "cache_put_count": int(self._puts),
                "cache_eviction_count": int(self._evictions),
                "cache_oversize_skip_count": int(self._oversize_skips),
            }


_FEATURE_WINDOW_READ_CACHE = _FeatureWindowReadCache(
    max_entries=_FEATURE_WINDOW_READ_CACHE_MAX_ENTRIES,
    max_bytes=_FEATURE_WINDOW_READ_CACHE_MAX_BYTES,
)


def _clear_feature_window_columns_cache_for_tests() -> None:
    _FEATURE_WINDOW_READ_CACHE.clear()


def feature_window_read_cache_stats() -> Dict[str, int]:
    return _FEATURE_WINDOW_READ_CACHE.stats()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


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


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    roots = _sandbox_roots()
    assert_write_allowed(path, "forecast family JSON", roots=roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "forecast family JSON temp", roots=roots)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def parse_int_csv(raw: str, default_vals: Sequence[int]) -> List[int]:
    vals = []
    for x in str(raw or "").split(","):
        s = x.strip()
        if not s:
            continue
        vals.append(int(s))
    return sorted(set(vals)) if vals else sorted(set(int(v) for v in default_vals))


def parse_str_csv(raw: str, default_vals: Sequence[str]) -> List[str]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return sorted(set(vals)) if vals else sorted(set(str(v) for v in default_vals))


def parse_quantiles(raw: str, default_vals: Sequence[float] = (0.1, 0.5, 0.9)) -> List[float]:
    vals = []
    for x in str(raw or "").split(","):
        s = x.strip()
        if not s:
            continue
        q = float(s)
        if 0.0 < q < 1.0:
            vals.append(q)
    out = sorted(set(vals)) if vals else sorted(set(float(v) for v in default_vals))
    return out if out else [0.1, 0.5, 0.9]


def horizon_bars(horizon_minutes: int, interval_minutes: int) -> int:
    hm = int(horizon_minutes)
    iv = int(interval_minutes)
    if hm <= 0 or iv <= 0 or hm % iv != 0:
        raise ValueError(f"invalid horizon/interval pair: horizon={hm} interval={iv}")
    return hm // iv


def month_path_for_features(root: Path, interval_minutes: int, asset: str, year: int, month: int) -> Path:
    return (
        root
        / f"scalar_features_{int(interval_minutes)}"
        / f"asset={str(asset)}"
        / f"year={int(year)}"
        / f"month={int(month):02d}"
        / f"part-scalar_features_{int(interval_minutes)}-{str(asset)}-{int(year)}{int(month):02d}.parquet"
    )


def month_path_for_features_legacy(root: Path, interval_minutes: int, year: int, month: int) -> Path:
    return (
        root
        / f"scalar_features_{int(interval_minutes)}"
        / f"year={int(year)}"
        / f"month={int(month):02d}"
        / f"part-scalar_features_{int(interval_minutes)}-{int(year)}{int(month):02d}.parquet"
    )


def iter_months_between(start_ts: int, end_ts: int) -> Iterable[Tuple[int, int]]:
    if int(end_ts) < int(start_ts):
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


def _read_parquet_present_columns(path: Path, needed: Sequence[str], required: Sequence[str]) -> Optional[pd.DataFrame]:
    read_cols = list(dict.fromkeys(str(c) for c in needed))
    required_cols = [str(c) for c in required]
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return _read_parquet_present_columns_without_schema(path, read_cols, required_cols)

    path = Path(path)
    file_key = _FEATURE_WINDOW_READ_CACHE.file_key(path)
    cache_key = file_key
    schema_names = _PARQUET_SCHEMA_NAMES_CACHE.get(cache_key)
    if schema_names is None:
        schema_names = set(str(name) for name in pq.read_schema(path).names)
        if len(_PARQUET_SCHEMA_NAMES_CACHE) >= _PARQUET_SCHEMA_NAMES_CACHE_MAX:
            _PARQUET_SCHEMA_NAMES_CACHE.clear()
        _PARQUET_SCHEMA_NAMES_CACHE[cache_key] = set(schema_names)
    present_cols = [c for c in read_cols if c in schema_names]
    if any(c not in present_cols for c in required_cols):
        return None
    cached = _FEATURE_WINDOW_READ_CACHE.get(file_key, present_cols)
    if cached is not None:
        return cached
    frame = pd.read_parquet(path, columns=present_cols)
    _FEATURE_WINDOW_READ_CACHE.put(file_key, present_cols, frame)
    return frame


def _read_parquet_present_columns_without_schema(
    path: Path,
    read_cols: Sequence[str],
    required_cols: Sequence[str],
) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path, columns=list(read_cols))
    except Exception:
        pass

    try:
        out = pd.read_parquet(path, columns=list(required_cols))
    except Exception:
        return None

    for col in read_cols:
        if col in required_cols:
            continue
        try:
            part = pd.read_parquet(path, columns=[str(col)])
        except Exception:
            continue
        if str(col) in part.columns and len(part) == len(out):
            out[str(col)] = part[str(col)].to_numpy()
    return out


def read_feature_window_columns(
    *,
    root: Path,
    interval_minutes: int,
    asset: str,
    columns: Sequence[str],
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    if int(end_ts) < int(start_ts):
        return pd.DataFrame(columns=["ts", "asset", *columns])

    needed = ["ts", "asset", *[str(c) for c in columns]]
    required = ["ts", "asset"]
    frames: List[pd.DataFrame] = []
    for y, m in iter_months_between(int(start_ts), int(end_ts)):
        p = month_path_for_features(
            root=root,
            interval_minutes=int(interval_minutes),
            asset=str(asset),
            year=int(y),
            month=int(m),
        )
        if not p.exists():
            p = month_path_for_features_legacy(root=root, interval_minutes=int(interval_minutes), year=int(y), month=int(m))
        if not p.exists():
            continue
        try:
            df = _read_parquet_present_columns(p, needed=needed, required=required)
            if df is None:
                continue
        except Exception:
            continue
        if df.empty:
            continue
        have_cols = [c for c in needed if c in df.columns]
        if "ts" not in have_cols or "asset" not in have_cols:
            continue
        d = df[have_cols].copy()
        d["ts"] = pd.to_numeric(d["ts"], errors="coerce")
        d["asset"] = d["asset"].astype(str)
        d = d[(d["asset"] == str(asset)) & d["ts"].notna()].copy()
        if d.empty:
            continue
        d["ts"] = d["ts"].astype("int64")
        d = d[(d["ts"] >= int(start_ts)) & (d["ts"] <= int(end_ts))].copy()
        if d.empty:
            continue
        frames.append(d)

    if not frames:
        out = pd.DataFrame(columns=needed)
        return out

    out = pd.concat(frames, ignore_index=True)
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[["ts", "asset", *columns]].sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")
    return out.reset_index(drop=True)


def load_assets(intervals: Sequence[int], assets_arg: str, assets_file: str) -> List[str]:
    if str(assets_file or "").strip():
        p = Path(str(assets_file).strip())
        if p.exists():
            vals = []
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                vals.append(s)
            if vals:
                return sorted(set(vals))
    return resolve_assets(intervals=intervals, assets_arg=assets_arg)


def default_task_map() -> Dict[str, str]:
    return dict(NUMERIC_TASK_TO_TARGET_COLUMN)


@dataclass
class FamilyStateFiles:
    manifest_file: Path
    skipped_file: Path


@dataclass
class UnitContext:
    family: str
    domain: str
    module_tag: str
    model_id: str
    model_version: str
    interval_minutes: int
    horizon_minutes: int
    task: str
    target_col: str
    asset: str
    run_id: str

    @property
    def ukey(self) -> str:
        return make_unit_key(
            family=self.family,
            domain=self.domain,
            task=self.task,
            horizon_minutes=int(self.horizon_minutes),
            asset=self.asset,
            interval=int(self.interval_minutes),
        )


def quantiles_from_samples(samples: np.ndarray, quantiles: Sequence[float]) -> Dict[float, float]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {float(q): float("nan") for q in quantiles}
    out: Dict[float, float] = {}
    for q in quantiles:
        out[float(q)] = float(np.quantile(arr, float(q)))
    return out


def monotonic_quantiles(qvals: Dict[float, float], quantiles: Sequence[float]) -> Dict[float, float]:
    qs = sorted(float(q) for q in quantiles)
    vals = [float(qvals.get(q, float("nan"))) for q in qs]
    vals = [v if math.isfinite(v) else (vals[i - 1] if i > 0 else 0.0) for i, v in enumerate(vals)]
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return {q: float(v) for q, v in zip(qs, vals)}


def qcol(q: float) -> str:
    return f"pred_p{int(round(float(q) * 100)):02d}"


def write_partitioned_prediction_month_frames(
    *,
    out_root: Path,
    interval_minutes: int,
    run_id: str,
    module_tag: str,
    task: str,
    horizon_minutes: int,
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
    existing_key_cache: Optional[Dict[Tuple[str, int, int, int], set[Tuple[str, int, int, str]]]] = None,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    sandbox_roots = _sandbox_roots()
    assert_write_allowed(out_root, "forecast family output root", roots=sandbox_roots)

    def _row_needs_recompute(frame: pd.DataFrame) -> pd.Series:
        if "needs_recompute" in frame.columns:
            return frame["needs_recompute"].fillna(False).astype(bool)
        if "is_forward_filled" in frame.columns:
            return frame["is_forward_filled"].fillna(False).astype(bool)
        return pd.Series(False, index=frame.index)

    for (y, m), frames in sorted(month_frames.items()):
        valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not valid_frames:
            continue
        chunk = pd.concat(valid_frames, ignore_index=True)
        for flag_col in ("is_forward_filled", "needs_recompute"):
            if flag_col not in chunk.columns:
                chunk[flag_col] = False
            else:
                chunk[flag_col] = chunk[flag_col].fillna(False).astype(bool)
        chunk = chunk.sort_values(["asset", "ts"]).drop_duplicates(
            subset=["asset", "ts", "horizon_min", "task"], keep="last"
        )
        required_cols = [str(col) for col in chunk.columns if str(col) != "ts"]
        validate_no_nan_columns(
            chunk,
            columns=required_cols,
            context=(
                f"prediction write asset-scope task={str(task)} "
                f"h={int(horizon_minutes)}m month={int(y):04d}-{int(m):02d}"
            ),
            ts_column="ts",
        )
        for asset_name, asset_chunk in chunk.groupby("asset", sort=False):
            validate_strict_timegrid(
                asset_chunk["ts"],
                interval_min=int(interval_minutes),
                context=(
                    f"prediction write asset={str(asset_name)} "
                    f"task={str(task)} h={int(horizon_minutes)}m month={int(y):04d}-{int(m):02d}"
                ),
            )

        month_dir = out_root / f"{int(interval_minutes)}" / f"year={int(y)}" / f"month={int(m):02d}"
        assert_write_allowed(month_dir, "forecast family month output directory", roots=sandbox_roots)
        cache_key = (str(Path(out_root).resolve()), int(interval_minutes), int(y), int(m))
        incoming_recompute_mask = _row_needs_recompute(chunk).astype(bool)
        incoming_has_recompute = bool(incoming_recompute_mask.any())
        all_seen_keys: set[Tuple[str, int, int, str]] = set()
        if existing_key_cache is not None and cache_key in existing_key_cache and not incoming_has_recompute:
            seen_keys = existing_key_cache[cache_key]
        else:
            existing_files = sorted(month_dir.glob("*.parquet")) if month_dir.exists() else []
            seen_keys = set()
            for p in existing_files:
                try:
                    ex = pd.read_parquet(p)
                except Exception:
                    continue
                if ex.empty:
                    continue
                missing_key_cols = {"asset", "ts", "horizon_min", "task"}.difference(str(c) for c in ex.columns)
                if missing_key_cols:
                    continue
                ex = ex.loc[:, [c for c in ["asset", "ts", "horizon_min", "task", "needs_recompute", "is_forward_filled"] if c in ex.columns]].copy()
                ex["asset"] = ex["asset"].astype(str)
                ex["ts"] = pd.to_numeric(ex["ts"], errors="coerce")
                ex["horizon_min"] = pd.to_numeric(ex["horizon_min"], errors="coerce")
                ex["task"] = ex["task"].astype(str)
                ex = ex.dropna(subset=["ts", "horizon_min"])
                for a, t, h, task_name in zip(ex["asset"], ex["ts"], ex["horizon_min"], ex["task"]):
                    all_seen_keys.add((str(a), int(t), int(h), str(task_name)))
                if "needs_recompute" in ex.columns or "is_forward_filled" in ex.columns:
                    ex = ex.loc[~_row_needs_recompute(ex)].copy()
                for a, t, h, task_name in zip(ex["asset"], ex["ts"], ex["horizon_min"], ex["task"]):
                    seen_keys.add((str(a), int(t), int(h), str(task_name)))
            if existing_key_cache is not None and not incoming_has_recompute:
                existing_key_cache[cache_key] = seen_keys

        if seen_keys or (incoming_has_recompute and all_seen_keys):
            keep_mask = [
                (
                    (str(a), int(t), int(h), str(tk)) not in (all_seen_keys if bool(needs_recompute) else seen_keys)
                )
                for a, t, h, tk, needs_recompute in zip(
                    chunk["asset"].astype(str),
                    pd.to_numeric(chunk["ts"], errors="coerce").astype("int64"),
                    pd.to_numeric(chunk["horizon_min"], errors="coerce").astype("int64"),
                    chunk["task"].astype(str),
                    incoming_recompute_mask.astype(bool),
                )
            ]
            chunk = chunk.loc[keep_mask].copy()
            if chunk.empty:
                continue
        replacement_keys = {
            (str(a), int(t), int(h), str(tk))
            for a, t, h, tk, needs_recompute in zip(
                chunk["asset"].astype(str),
                pd.to_numeric(chunk["ts"], errors="coerce").astype("int64"),
                pd.to_numeric(chunk["horizon_min"], errors="coerce").astype("int64"),
                chunk["task"].astype(str),
                _row_needs_recompute(chunk).astype(bool),
            )
            if not bool(needs_recompute)
        }
        if replacement_keys and month_dir.exists():
            same_combo_files = [
                p
                for p in sorted(month_dir.glob("*.parquet"), key=lambda path: path.name.lower())
                if f"-{task}-h{int(horizon_minutes)}m-" in p.name
            ]
            if same_combo_files:
                preserved_frames: List[pd.DataFrame] = []
                rewritten_any = False
                for p in same_combo_files:
                    try:
                        ex_full = pd.read_parquet(p)
                    except Exception:
                        continue
                    if ex_full.empty:
                        continue
                    ex_full["asset"] = ex_full["asset"].astype(str)
                    ex_full["ts"] = pd.to_numeric(ex_full["ts"], errors="coerce")
                    ex_full["horizon_min"] = pd.to_numeric(ex_full["horizon_min"], errors="coerce")
                    ex_full["task"] = ex_full["task"].astype(str)
                    ex_full = ex_full.dropna(subset=["ts", "horizon_min"]).copy()
                    ex_full["ts"] = ex_full["ts"].astype("int64")
                    ex_full["horizon_min"] = ex_full["horizon_min"].astype("int64")
                    ex_keys = [
                        (str(a), int(t), int(h), str(tk))
                        for a, t, h, tk in zip(ex_full["asset"], ex_full["ts"], ex_full["horizon_min"], ex_full["task"])
                    ]
                    keep = [key not in replacement_keys for key in ex_keys]
                    if not all(keep):
                        rewritten_any = True
                    kept = ex_full.loc[keep].copy()
                    if not kept.empty:
                        preserved_frames.append(kept)
                if rewritten_any:
                    if preserved_frames:
                        chunk = pd.concat([*preserved_frames, chunk], ignore_index=True)
                        for flag_col in ("is_forward_filled", "needs_recompute"):
                            if flag_col not in chunk.columns:
                                chunk[flag_col] = False
                            else:
                                chunk[flag_col] = chunk[flag_col].fillna(False).astype(bool)
                        chunk = chunk.sort_values(["asset", "ts"]).drop_duplicates(
                            subset=["asset", "ts", "horizon_min", "task"],
                            keep="last",
                        )
                    for p in same_combo_files:
                        assert_write_allowed(p, "forecast family parquet delete", roots=sandbox_roots)
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    if existing_key_cache is not None:
                        existing_key_cache.pop(cache_key, None)
        new_keys = {
            (str(a), int(t), int(h), str(tk))
            for a, t, h, tk in zip(
                chunk["asset"].astype(str),
                pd.to_numeric(chunk["ts"], errors="coerce").astype("int64"),
                pd.to_numeric(chunk["horizon_min"], errors="coerce").astype("int64"),
                chunk["task"].astype(str),
            )
        }

        dst = (
            month_dir
            / f"part-{module_tag}_{int(interval_minutes)}-{int(y)}{int(m):02d}-{task}-h{int(horizon_minutes)}m-{run_id}.parquet"
        )
        assert_write_allowed(dst, "forecast family parquet", roots=sandbox_roots)
        month_dir.mkdir(parents=True, exist_ok=True)
        tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
        assert_write_allowed(tmp, "forecast family parquet temp", roots=sandbox_roots)
        chunk.to_parquet(
            tmp,
            engine="pyarrow",
            compression=PARQUET_COMPRESSION,
            row_group_size=PARQUET_ROW_GROUP,
            index=False,
        )
        atomic_replace(tmp, dst)
        if existing_key_cache is not None:
            existing_key_cache.setdefault(cache_key, seen_keys).update(new_keys)
        parts.append(
            {
                "path": str(dst),
                "rows": int(len(chunk)),
                "interval_min": int(interval_minutes),
                "task": str(task),
                "horizon_min": int(horizon_minutes),
                "year": int(y),
                "month": int(m),
                "min_ts": int(chunk["ts"].min()) if not chunk.empty else None,
                "max_ts": int(chunk["ts"].max()) if not chunk.empty else None,
            }
        )
    return parts


def write_partitioned_predictions(
    *,
    out_root: Path,
    interval_minutes: int,
    run_id: str,
    module_tag: str,
    task: str,
    horizon_minutes: int,
    df: pd.DataFrame,
    existing_key_cache: Optional[Dict[Tuple[str, int, int, int], set[Tuple[str, int, int, str]]]] = None,
) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    out = df.copy()
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"]).copy()
    if out.empty:
        return []
    out["ts"] = out["ts"].astype("int64")
    dt = pd.to_datetime(out["ts"], unit="s", utc=True)
    out["year"] = dt.dt.year.astype(int)
    out["month"] = dt.dt.month.astype(int)
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
    for (y, m), grp in out.groupby(["year", "month"], sort=True):
        month_frames.setdefault((int(y), int(m)), []).append(grp.drop(columns=["year", "month"], errors="ignore").copy())
    return write_partitioned_prediction_month_frames(
        out_root=out_root,
        interval_minutes=int(interval_minutes),
        run_id=str(run_id),
        module_tag=str(module_tag),
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        month_frames=month_frames,
        existing_key_cache=existing_key_cache,
    )


def _find_latest_forecast_month_dir(interval_root: Path) -> Optional[Path]:
    if not interval_root.exists():
        return None
    years: List[Tuple[int, Path]] = []
    for ydir in interval_root.glob("year=*"):
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


def forecast_output_tail_ts(
    *,
    out_root: Path,
    interval_minutes: int,
    task: str,
    horizon_minutes: int,
    asset: str,
    include_recompute: bool = False,
) -> Optional[int]:
    """
    Resolve the contiguous populated destination tail for one forecast unit.

    Newer listings and model warmup can legitimately shorten the head. Once a
    unit has a populated head, interior gaps or bad prediction values stop the
    completed tail so resume/backfill can repair them instead of edge-skipping.
    """
    assert_write_allowed(out_root, "forecast family generated output tail root", roots=_sandbox_roots())
    interval_root = Path(out_root) / f"{int(interval_minutes)}"
    if not interval_root.exists():
        return None
    completed_ts: List[np.ndarray] = []
    key_cols = {"ts", "asset", "task", "horizon_min", "interval_min", "run_id", "model_id", "model_version"}
    for p in sorted(interval_root.glob("year=*/month=*/*.parquet"), key=lambda q: str(q).lower()):
        try:
            d = pd.read_parquet(p)
        except Exception:
            continue
        if d.empty:
            continue
        required = {"asset", "ts", "task", "horizon_min"}
        if not required.issubset(set(str(c) for c in d.columns)):
            continue
        d = d[
            (d["asset"].astype(str) == str(asset))
            & (d["task"].astype(str) == str(task))
            & (pd.to_numeric(d["horizon_min"], errors="coerce").fillna(-1).astype("int64") == int(horizon_minutes))
        ]
        if d.empty:
            continue
        value_cols = [
            str(col)
            for col in d.columns
            if str(col) not in key_cols and pd.api.types.is_numeric_dtype(d[col])
        ]
        valid_mask = pd.to_numeric(d["ts"], errors="coerce").notna()
        if not bool(include_recompute):
            if "needs_recompute" in d.columns:
                valid_mask = valid_mask & ~d["needs_recompute"].astype(bool)
            if "is_forward_filled" in d.columns:
                valid_mask = valid_mask & ~d["is_forward_filled"].astype(bool)
        if value_cols:
            valid_mask = valid_mask & d.loc[:, value_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        ts = pd.to_numeric(d.loc[valid_mask, "ts"], errors="coerce").dropna().astype("int64")
        if ts.empty:
            continue
        completed_ts.append(ts.to_numpy(dtype=np.int64))
    if not completed_ts:
        return None
    merged = np.unique(np.concatenate(completed_ts))
    if merged.size == 0:
        return None
    step = int(interval_minutes) * 60
    expected = int(merged[0])
    last_complete: Optional[int] = None
    for ts_i in merged:
        cur = int(ts_i)
        if cur != expected:
            break
        last_complete = cur
        expected += int(step)
    return int(last_complete) if last_complete is not None else None


def seasonality_info(parquet_root: Path, interval_minutes: int, asset: str) -> Dict[str, Any]:
    prof = resolve_seasonality_profile(parquet_root=parquet_root, interval_minutes=int(interval_minutes), asset=str(asset))
    label = str(prof.source or "none")
    if isinstance(prof.path, str) and prof.path:
        p = Path(prof.path)
        try:
            label = p.parents[2].name
        except Exception:
            label = str(prof.source or "none")
    return {
        "seasonality_mode": str(prof.source),
        "seasonality_usable": bool(prof.usable),
        "seasonality_period_bars": int(prof.seasonal_period_bars) if prof.seasonal_period_bars is not None else None,
        "seasonality_label": label,
        "seasonality_path": prof.path,
    }


def fit_window_start(edge_ts: int, fit_days: int, min_ts: int) -> int:
    return max(int(min_ts), int(edge_ts) - int(fit_days) * 86400)


def robust_sigma(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 1e-6
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return max(1e-6, 1.4826 * mad)


def task_target_col(task: str, task_map: Dict[str, str]) -> Optional[str]:
    return task_map.get(str(task))


def make_files(root: Path, manifest_name: str, skipped_name: str) -> FamilyStateFiles:
    sroot = root / "state"
    assert_write_allowed(sroot, "forecast family state directory", roots=_sandbox_roots())
    sroot.mkdir(parents=True, exist_ok=True)
    return FamilyStateFiles(
        manifest_file=sroot / manifest_name,
        skipped_file=sroot / skipped_name,
    )


def default_common_roots(family_root_env: str, family_root_name: str) -> Tuple[Path, Path, Path]:
    roots = _sandbox_roots()
    if roots.enabled:
        parquet_root = Path(os.getenv("PIPELINE_SOURCE_PARQUET_ROOT") or os.getenv("PIPELINE_PARQUET_ROOT", str(DEFAULT_PARQUET_ROOT)))
        feature_root = Path(os.getenv("PIPELINE_SOURCE_FEATURES_ROOT") or os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(DEFAULT_FEATURE_ROOT)))
        write_parquet_root = _sandbox_env_path(roots, "PIPELINE_SANDBOX_PARQUET_ROOT", roots.parquet_root, "forecast family parquet root")
        forecast_root = write_parquet_root / family_root_name
        assert_write_allowed(forecast_root, "forecast family root", roots=roots)
    else:
        parquet_root = Path(os.getenv(family_root_env, str(DEFAULT_PARQUET_ROOT)))
        feature_root = Path(os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(DEFAULT_FEATURE_ROOT)))
        forecast_root = parquet_root / family_root_name
    forecast_root.mkdir(parents=True, exist_ok=True)
    return parquet_root, feature_root, forecast_root


def build_unit_context(
    *,
    family: str,
    domain: str,
    module_tag: str,
    model_id: str,
    model_version: str,
    interval_minutes: int,
    horizon_minutes: int,
    task: str,
    target_col: str,
    asset: str,
    run_id: str,
) -> UnitContext:
    return UnitContext(
        family=family,
        domain=domain,
        module_tag=module_tag,
        model_id=model_id,
        model_version=model_version,
        interval_minutes=int(interval_minutes),
        horizon_minutes=int(horizon_minutes),
        task=str(task),
        target_col=str(target_col),
        asset=str(asset),
        run_id=str(run_id),
    )


def discover_edge_and_min(
    asset: str,
    interval_minutes: int,
    root: Optional[Path] = None,
) -> Tuple[Optional[int], Optional[int]]:
    return interval_edge_ts(asset=str(asset), interval_minutes=int(interval_minutes), root=root), interval_min_ts(
        asset=str(asset), interval_minutes=int(interval_minutes), root=root
    )


def supported_tasks(default_tasks: Sequence[str], task_map: Dict[str, str]) -> List[str]:
    out = []
    for t in default_tasks:
        if str(t) in task_map:
            out.append(str(t))
    return out


def choose_feature_columns(df: pd.DataFrame, target_col: str, max_features: int) -> List[str]:
    cols = []
    for c in df.columns:
        cs = str(c)
        csl = cs.lower()
        if cs in {"ts", "asset", str(target_col)}:
            continue
        if csl.startswith("future_"):
            continue
        if csl.startswith("regime_") or csl.endswith("_label"):
            continue
        cols.append(cs)
    cols = sorted(cols)
    if max_features > 0 and len(cols) > int(max_features):
        cols = cols[: int(max_features)]
    return cols


def merge_target_with_factor(target_df: pd.DataFrame, factor_df: pd.DataFrame, factor_col: str = "market_factor") -> pd.DataFrame:
    if target_df.empty:
        return target_df
    f = factor_df[["ts", factor_col]].copy() if factor_col in factor_df.columns else pd.DataFrame(columns=["ts", factor_col])
    f["ts"] = pd.to_numeric(f.get("ts"), errors="coerce")
    f = f.dropna(subset=["ts"]).copy()
    if f.empty:
        out = target_df.copy()
        out[factor_col] = np.nan
        return out
    f["ts"] = f["ts"].astype("int64")
    return target_df.merge(f, on="ts", how="left")


def interval_label(interval_minutes: int) -> str:
    iv = int(interval_minutes)
    if iv == 60:
        return "1H"
    if iv == 240:
        return "4H"
    if iv == 1440:
        return "1D"
    return f"{iv}m"
