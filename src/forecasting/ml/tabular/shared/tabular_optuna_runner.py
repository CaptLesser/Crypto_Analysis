from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

import importlib

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE
from src.forecasting.ml.shared.test_branch_function_telemetry import (
    emit_event_for_path,
    emit_stage3_study_summary_for_path,
    telemetry_scope_for_path,
)


CLAMP_START_YEAR = 2025
CLAMP_START_MONTH = 1
TABULAR_REPOSITORY_CACHE_MAX_ENTRIES = 64
TABULAR_REPOSITORY_CACHE_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class TabularOptunaModelSpec:
    model_key: str
    display_name: str
    short_label: str
    numerics_module_import_path: str
    optuna_profile_import_path: str
    output_root: Path
    default_study_name_prefix: str
    default_model_threads: int = 6
    default_trials_per_combo: int = 40
    default_search_back_months: int = 12
    default_history_window_months: int = 13


CURRENT_MODEL_SPEC: Optional[TabularOptunaModelSpec] = None
CURRENT_NUMERICS: Any = None
CURRENT_OPTUNA_PROFILE: Any = None


def configure_for_model(model_spec: TabularOptunaModelSpec) -> None:
    global CURRENT_MODEL_SPEC, CURRENT_NUMERICS, CURRENT_OPTUNA_PROFILE
    CURRENT_MODEL_SPEC = model_spec
    CURRENT_NUMERICS = importlib.import_module(model_spec.numerics_module_import_path)
    CURRENT_OPTUNA_PROFILE = importlib.import_module(model_spec.optuna_profile_import_path)


@dataclass(frozen=True)
class MonthKey:
    year: int
    month: int


@dataclass(frozen=True)
class Stage2Context:
    interval: int
    training_window_months: int
    assets: Tuple[str, ...]
    seed_ts: int
    accuracy_end_ts: int
    forecast_target_month_start_utc: str
    run_summary_path: Path


@dataclass(frozen=True)
class ComboSpec:
    interval: int
    horizon_minutes: int
    task: str
    training_window_months: int
    refit_cadence: Optional[str]

    @property
    def horizon_bars(self) -> int:
        return CURRENT_NUMERICS.horizon_bars_from_minutes(int(self.horizon_minutes), int(self.interval))

    @property
    def training_window_bars(self) -> int:
        return CURRENT_NUMERICS.training_window_bars_from_months(int(self.training_window_months), int(self.interval))

    @property
    def tuple_label(self) -> str:
        base = f"{int(self.interval)}:{int(self.horizon_minutes)}:{self.task}@{int(self.training_window_months)}m"
        return f"{base}@{self.refit_cadence}" if self.refit_cadence else base


@dataclass
class Dataset:
    asset: str
    spec: ComboSpec
    df: pd.DataFrame
    source_start_ts: int
    source_end_ts: int
    seed_ts: int
    accuracy_end_ts: int
    x_cols: Tuple[str, ...]
    x: np.ndarray
    ts: np.ndarray
    y: np.ndarray


@dataclass
class MetricResult:
    combo: str
    asset: str
    rows: int
    rmse: Optional[float]
    mae: Optional[float]
    baseline_rmse: Optional[float]
    baseline_mae: Optional[float]
    first_prediction_ts: Optional[int]
    last_prediction_ts: Optional[int]
    pending_tail_rows: int
    params_label: str


@dataclass(frozen=True)
class ConcurrencyPlan:
    logical_cpus: int
    combo_workers: int
    trial_workers: int
    model_threads: int

    @property
    def active_fit_slots(self) -> int:
        return int(self.combo_workers) * int(self.trial_workers)

    @property
    def cpu_budget(self) -> int:
        return int(self.active_fit_slots) * int(self.model_threads)


class TimingBook:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seconds: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def add_time(self, key: str, elapsed: float) -> None:
        with self._lock:
            self._seconds[str(key)] += float(elapsed)

    def add_count(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counts[str(key)] += int(value)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            for key, value in self._seconds.items():
                out[str(key)] = float(value)
            for key, value in self._counts.items():
                out[str(key)] = int(value)
            return out


class DatasetRepository:
    def __init__(
        self,
        *,
        max_entries: int = TABULAR_REPOSITORY_CACHE_MAX_ENTRIES,
        max_bytes: int = TABULAR_REPOSITORY_CACHE_MAX_BYTES,
    ) -> None:
        self._lock = threading.Lock()
        self.timings = TimingBook()
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._window_cache: OrderedDict[Tuple[Any, ...], Tuple[int, int, List[str], MonthKey]] = OrderedDict()
        self._feature_cache: OrderedDict[Tuple[Any, ...], pd.DataFrame] = OrderedDict()
        self._feature_cache_bytes = 0
        self._label_cache: OrderedDict[Tuple[Any, ...], pd.DataFrame] = OrderedDict()
        self._label_cache_bytes = 0
        self._dataset_cache: OrderedDict[Tuple[Any, ...], Dataset] = OrderedDict()
        self._dataset_cache_bytes = 0
        self._cache_evictions = 0

    @staticmethod
    def _frame_bytes(frame: pd.DataFrame) -> int:
        return int(frame.memory_usage(index=True, deep=True).sum())

    @classmethod
    def _dataset_bytes(cls, dataset: Dataset) -> int:
        return (
            cls._frame_bytes(dataset.df)
            + int(dataset.x.nbytes)
            + int(dataset.ts.nbytes)
            + int(dataset.y.nbytes)
        )

    @staticmethod
    def _clone_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy(deep=True)

    @classmethod
    def _clone_dataset(cls, dataset: Dataset) -> Dataset:
        return Dataset(
            asset=str(dataset.asset),
            spec=dataset.spec,
            df=cls._clone_frame(dataset.df),
            source_start_ts=int(dataset.source_start_ts),
            source_end_ts=int(dataset.source_end_ts),
            seed_ts=int(dataset.seed_ts),
            accuracy_end_ts=int(dataset.accuracy_end_ts),
            x_cols=tuple(dataset.x_cols),
            x=np.array(dataset.x, copy=True),
            ts=np.array(dataset.ts, copy=True),
            y=np.array(dataset.y, copy=True),
        )

    @staticmethod
    def _path_identity(path: Any) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
        if path is None:
            return ("", None, None, None)
        resolved = Path(path).expanduser()
        try:
            resolved = resolved.resolve()
        except Exception:
            resolved = resolved.absolute()
        try:
            stat = resolved.stat()
            return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
        except Exception:
            return (str(resolved), None, None, None)

    @classmethod
    def _source_identity(cls) -> Tuple[Any, ...]:
        io_config = getattr(CURRENT_NUMERICS, "IO_CONFIG", None)
        scalar_root = getattr(io_config, "scalar_root", None) or getattr(CURRENT_NUMERICS, "PARQUET_ROOT", "")
        ohlc_root = getattr(io_config, "ohlc_root", None) or getattr(CURRENT_NUMERICS, "PARQUET_ROOT", "")
        parquet_root = getattr(CURRENT_NUMERICS, "PARQUET_ROOT", "")
        return (
            cls._path_identity(parquet_root),
            cls._path_identity(scalar_root),
            cls._path_identity(ohlc_root),
        )

    @staticmethod
    def _model_identity() -> Tuple[Any, ...]:
        return (
            "tabular",
            str(getattr(CURRENT_MODEL_SPEC, "model_key", "")),
            str(getattr(CURRENT_MODEL_SPEC, "display_name", "")),
            str(getattr(CURRENT_NUMERICS, "__name__", CURRENT_NUMERICS.__class__.__name__)),
        )

    @classmethod
    def _feature_profile_identity(cls) -> Tuple[Any, ...]:
        fn = getattr(CURRENT_NUMERICS, "select_feature_columns", None)
        fn_identity = getattr(fn, "__func__", fn)
        module = str(getattr(fn_identity, "__module__", ""))
        qualname = str(getattr(fn_identity, "__qualname__", ""))
        code = getattr(fn_identity, "__code__", None)
        code_fingerprint = (
            getattr(code, "co_filename", None),
            getattr(code, "co_firstlineno", None),
            getattr(code, "co_code", b""),
        )
        return (module, qualname, id(fn_identity), code_fingerprint)

    @staticmethod
    def _label_identity() -> Tuple[Any, ...]:
        fn = getattr(CURRENT_NUMERICS, "_compute_future_labels", None)
        fn_identity = getattr(fn, "__func__", fn)
        code = getattr(fn_identity, "__code__", None)
        return (
            tuple(str(col) for col in getattr(CURRENT_NUMERICS, "FUTURE_LABEL_COLUMNS", ())),
            tuple(sorted((str(k), str(v)) for k, v in getattr(CURRENT_NUMERICS, "TASK_LABEL", {}).items())),
            str(getattr(fn_identity, "__module__", "")),
            str(getattr(fn_identity, "__qualname__", "")),
            id(fn_identity),
            (
                getattr(code, "co_filename", None),
                getattr(code, "co_firstlineno", None),
                getattr(code, "co_code", b""),
            ),
        )

    def _evict_lru(
        self,
        cache: OrderedDict[Tuple[Any, ...], Any],
        byte_attr: str,
        size_fn,
    ) -> None:
        while len(cache) > self._max_entries or int(getattr(self, byte_attr)) > self._max_bytes:
            _old_key, old_value = cache.popitem(last=False)
            setattr(self, byte_attr, max(0, int(getattr(self, byte_attr)) - int(size_fn(old_value))))
            self._cache_evictions += 1

    def _cache_get(self, cache: OrderedDict[Tuple[Any, ...], Any], key: Tuple[Any, ...]) -> Any:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _cache_put(
        self,
        cache: OrderedDict[Tuple[Any, ...], Any],
        key: Tuple[Any, ...],
        value: Any,
        *,
        byte_attr: Optional[str] = None,
        size_fn=None,
    ) -> None:
        if key in cache:
            old_value = cache.pop(key)
            if byte_attr is not None and size_fn is not None:
                setattr(self, byte_attr, max(0, int(getattr(self, byte_attr)) - int(size_fn(old_value))))
        cache[key] = value
        if byte_attr is not None and size_fn is not None:
            setattr(self, byte_attr, int(getattr(self, byte_attr)) + int(size_fn(value)))
            self._evict_lru(cache, byte_attr, size_fn)
        else:
            while len(cache) > self._max_entries:
                cache.popitem(last=False)
                self._cache_evictions += 1

    def cache_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "window_cache_entries": int(len(self._window_cache)),
                "feature_cache_entries": int(len(self._feature_cache)),
                "feature_cache_bytes_estimate": int(self._feature_cache_bytes),
                "label_cache_entries": int(len(self._label_cache)),
                "label_cache_bytes_estimate": int(self._label_cache_bytes),
                "dataset_cache_entries": int(len(self._dataset_cache)),
                "dataset_cache_bytes_estimate": int(self._dataset_cache_bytes),
                "cache_evictions": int(self._cache_evictions),
            }

    def clear(self) -> None:
        with self._lock:
            self._window_cache.clear()
            self._feature_cache.clear()
            self._label_cache.clear()
            self._dataset_cache.clear()
            self._feature_cache_bytes = 0
            self._label_cache_bytes = 0
            self._dataset_cache_bytes = 0

    def eval_window_for_interval(
        self,
        *,
        interval_minutes: int,
        explicit_assets: Sequence[str],
        history_window_months: int,
        search_back_months: int,
    ) -> Tuple[int, int, List[str], MonthKey]:
        cache_key = (
            self._model_identity(),
            self._source_identity(),
            int(interval_minutes),
            tuple(str(asset) for asset in explicit_assets),
            int(history_window_months),
            int(search_back_months),
        )
        with self._lock:
            cached = self._cache_get(self._window_cache, cache_key)
        if cached is not None:
            self.timings.add_count('eval_window_cache_hits')
            seed_ts, accuracy_end_ts, assets, month = cached
            return int(seed_ts), int(accuracy_end_ts), list(assets), month
        t0 = time.monotonic()
        value = compute_eval_window(
            interval_minutes=int(interval_minutes),
            explicit_assets=list(explicit_assets),
            history_window_months=int(history_window_months),
            search_back_months=int(search_back_months),
        )
        self.timings.add_time('eval_window_s', time.monotonic() - t0)
        self.timings.add_count('eval_window_builds')
        with self._lock:
            self._cache_put(
                self._window_cache,
                cache_key,
                (int(value[0]), int(value[1]), list(value[2]), value[3]),
            )
        return value

    def _load_feature_frame(self, asset: str, spec: ComboSpec, source_start_ts: int, accuracy_end_ts: int) -> pd.DataFrame:
        cache_key = (
            self._model_identity(),
            self._source_identity(),
            self._feature_profile_identity(),
            str(asset),
            int(spec.interval),
            int(source_start_ts),
            int(accuracy_end_ts),
        )
        with self._lock:
            cached = self._cache_get(self._feature_cache, cache_key)
        if cached is not None:
            self.timings.add_count('feature_cache_hits')
            return self._clone_frame(cached)
        t0 = time.monotonic()
        feature_df, _stats = CURRENT_NUMERICS._load_unit_feature_frame(
            asset=asset,
            interval=int(spec.interval),
            start_ts=int(source_start_ts),
            stop_ts=int(accuracy_end_ts),
        )
        if feature_df.empty:
            raise RuntimeError(f'Empty feature frame for asset={asset} combo={spec.tuple_label}')
        self.timings.add_time('feature_load_s', time.monotonic() - t0)
        self.timings.add_count('feature_loads')
        stored = self._clone_frame(feature_df)
        with self._lock:
            self._cache_put(
                self._feature_cache,
                cache_key,
                stored,
                byte_attr="_feature_cache_bytes",
                size_fn=self._frame_bytes,
            )
        return self._clone_frame(stored)

    def _load_labels(self, asset: str, spec: ComboSpec, source_start_ts: int, accuracy_end_ts: int, feature_df: pd.DataFrame) -> pd.DataFrame:
        cache_key = (
            self._model_identity(),
            self._source_identity(),
            self._label_identity(),
            str(asset),
            int(spec.interval),
            int(spec.horizon_minutes),
            int(spec.horizon_bars),
            str(spec.task),
            int(source_start_ts),
            int(accuracy_end_ts),
        )
        with self._lock:
            cached = self._cache_get(self._label_cache, cache_key)
        if cached is not None:
            self.timings.add_count('label_cache_hits')
            return self._clone_frame(cached)
        t0 = time.monotonic()
        labels, _detail = CURRENT_NUMERICS._compute_future_labels(
            feature_df.loc[:, ['open', 'high', 'low', 'close', 'volume', 'trades']].reset_index(drop=True),
            horizon_bars=int(spec.horizon_bars),
        )
        self.timings.add_time('label_build_s', time.monotonic() - t0)
        self.timings.add_count('label_builds')
        stored = self._clone_frame(labels)
        with self._lock:
            self._cache_put(
                self._label_cache,
                cache_key,
                stored,
                byte_attr="_label_cache_bytes",
                size_fn=self._frame_bytes,
            )
        return self._clone_frame(stored)

    def build_dataset(self, asset: str, spec: ComboSpec, seed_ts: int, accuracy_end_ts: int) -> Dataset:
        source_start_ts = trailing_source_start_ts(
            seed_ts=int(seed_ts),
            interval_minutes=int(spec.interval),
            train_window_bars=int(spec.training_window_bars),
            max_horizon_bars=int(spec.horizon_bars),
        )
        cache_key = (
            self._model_identity(),
            self._source_identity(),
            self._feature_profile_identity(),
            self._label_identity(),
            str(asset),
            spec.tuple_label,
            int(spec.interval),
            int(spec.horizon_minutes),
            int(spec.horizon_bars),
            str(spec.task),
            int(spec.training_window_months),
            spec.refit_cadence,
            int(seed_ts),
            int(source_start_ts),
            int(accuracy_end_ts),
        )
        with self._lock:
            cached = self._cache_get(self._dataset_cache, cache_key)
        if cached is not None:
            self.timings.add_count('dataset_cache_hits')
            return self._clone_dataset(cached)
        t0 = time.monotonic()
        feature_df = self._load_feature_frame(asset, spec, int(source_start_ts), int(accuracy_end_ts))
        labels = self._load_labels(asset, spec, int(source_start_ts), int(accuracy_end_ts), feature_df)
        feature_base = feature_df.drop(columns=[col for col in CURRENT_NUMERICS.FUTURE_LABEL_COLUMNS if col in feature_df.columns], errors='ignore').reset_index(drop=True)
        label_frame = labels.loc[:, CURRENT_NUMERICS.FUTURE_LABEL_COLUMNS].reset_index(drop=True)
        df = pd.concat([feature_base, label_frame], axis=1)
        label_base = CURRENT_NUMERICS.TASK_LABEL[str(spec.task)]
        x_cols = tuple(CURRENT_NUMERICS.select_feature_columns(df.columns, str(spec.task), int(spec.horizon_minutes), int(spec.interval), {label_base, 'future_direction'}))
        dataset = Dataset(
            asset=str(asset),
            spec=spec,
            df=df,
            source_start_ts=int(source_start_ts),
            source_end_ts=int(accuracy_end_ts),
            seed_ts=int(seed_ts),
            accuracy_end_ts=int(accuracy_end_ts),
            x_cols=x_cols,
            x=df.loc[:, list(x_cols)].to_numpy(dtype=DEFAULT_FLOAT_DTYPE),
            ts=df['ts'].to_numpy(dtype=np.int64),
            y=pd.to_numeric(df[label_base], errors='coerce').to_numpy(dtype=DEFAULT_FLOAT_DTYPE),
        )
        self.timings.add_time('dataset_build_s', time.monotonic() - t0)
        self.timings.add_count('dataset_builds')
        stored = self._clone_dataset(dataset)
        with self._lock:
            self._cache_put(
                self._dataset_cache,
                cache_key,
                stored,
                byte_attr="_dataset_cache_bytes",
                size_fn=self._dataset_bytes,
            )
        return self._clone_dataset(stored)


def is_memory_failure(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    name = exc.__class__.__name__.lower()
    return 'arraymemoryerror' in name or 'memoryerror' in name


def is_transient_failure(exc: BaseException) -> bool:
    if is_memory_failure(exc):
        return False
    if isinstance(exc, (BrokenProcessPool, sqlite3.OperationalError)):
        return True
    name = exc.__class__.__name__
    if name in {'StorageInternalError'}:
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        return any(token in msg for token in ('temporarily unavailable', 'resource temporarily unavailable', 'database is locked', 'disk i/o error', 'sharing violation'))
    return False


def run_with_transient_retry(fn, *, label: str, max_attempts: int = 2):
    last_exc: Optional[BaseException] = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= int(max_attempts) or not is_transient_failure(exc):
                raise
            time.sleep(0.5 * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f'Failed retry loop for {label}')


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def month_start_utc_ts(year: int, month: int) -> int:
    return int(datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def add_months(year: int, month: int, delta: int) -> MonthKey:
    idx = year * 12 + (month - 1) + int(delta)
    return MonthKey(idx // 12, idx % 12 + 1)


def month_seq(end_month: MonthKey, count: int) -> List[MonthKey]:
    return [add_months(end_month.year, end_month.month, -(count - 1 - i)) for i in range(count)]


def months_between(start: MonthKey, end_exclusive: MonthKey) -> int:
    return (end_exclusive.year * 12 + (end_exclusive.month - 1)) - (start.year * 12 + (start.month - 1))


def parse_asset_name(dirname: str) -> Optional[str]:
    if not dirname.startswith("asset="):
        return None
    value = dirname.split("=", 1)[1].strip()
    return value or None


def list_assets(table_root: Path) -> List[str]:
    if not table_root.exists():
        return []
    out: List[str] = []
    for path in table_root.iterdir():
        if path.is_dir():
            asset = parse_asset_name(path.name)
            if asset:
                out.append(asset)
    return sorted(set(out))


def asset_months(table_root: Path, asset: str) -> set[MonthKey]:
    base = table_root / f"asset={asset}"
    out: set[MonthKey] = set()
    if not base.exists():
        return out
    for ydir in base.glob("year=*"):
        if not ydir.is_dir():
            continue
        try:
            year = int(ydir.name.split("=", 1)[1])
        except Exception:
            continue
        for mdir in ydir.glob("month=*"):
            if not mdir.is_dir():
                continue
            try:
                month = int(mdir.name.split("=", 1)[1])
            except Exception:
                continue
            out.add(MonthKey(year, month))
    return out


def common_recent_window(
    *,
    ohlc_root: Path,
    scalar_root: Path,
    min_assets: int,
    window_months: int,
    search_back_months: int,
    clamp_start: MonthKey,
) -> Tuple[MonthKey, List[str]]:
    all_assets = sorted(set(list_assets(ohlc_root)).intersection(list_assets(scalar_root)))
    month_map: Dict[str, set[MonthKey]] = {}
    for asset in all_assets:
        months = asset_months(ohlc_root, asset).intersection(asset_months(scalar_root, asset))
        months = {m for m in months if months_between(clamp_start, add_months(m.year, m.month, 1)) > 0}
        if months:
            month_map[asset] = months
    now = datetime.now(timezone.utc)
    start_end_month = add_months(now.year, now.month, -1)
    for back in range(max(1, int(search_back_months))):
        end_month = add_months(start_end_month.year, start_end_month.month, -back)
        required = {
            m for m in month_seq(end_month, int(window_months))
            if months_between(clamp_start, add_months(m.year, m.month, 1)) > 0
        }
        eligible = [asset for asset, months in month_map.items() if required.issubset(months)]
        if len(eligible) >= int(min_assets):
            return end_month, sorted(eligible)
    raise RuntimeError(
        f"Could not find a common {int(window_months)}-month window with at least {int(min_assets)} assets."
    )


def max_asset_ts_for_month(table_root: Path, asset: str, month: MonthKey) -> Optional[int]:
    base = table_root / f"asset={asset}" / f"year={month.year}" / f"month={month.month:02d}"
    if not base.exists():
        return None
    max_ts: Optional[int] = None
    for parquet_path in base.glob("*.parquet"):
        try:
            frame = pd.read_parquet(parquet_path, columns=["ts"])
        except Exception:
            continue
        if frame.empty or "ts" not in frame.columns:
            continue
        values = pd.to_numeric(frame["ts"], errors="coerce").dropna()
        if values.empty:
            continue
        current = int(values.astype("int64").max())
        if max_ts is None or current > max_ts:
            max_ts = current
    return max_ts


def trailing_source_start_ts(seed_ts: int, interval_minutes: int, train_window_bars: int, max_horizon_bars: int) -> int:
    step_seconds = int(interval_minutes) * 60
    lookback_bars = max(1, int(train_window_bars) + int(max_horizon_bars) - 1)
    return int(seed_ts) - (int(lookback_bars) * int(step_seconds))


def parse_args() -> argparse.Namespace:
    if CURRENT_MODEL_SPEC is None:
        raise RuntimeError("Optuna runner is not configured for a model.")
    parser = argparse.ArgumentParser(
        description=f"Single-objective {CURRENT_MODEL_SPEC.display_name} Optuna tuning on the survivor-style recent diagnostic slice."
    )
    parser.add_argument("--assets", type=str, default="", help="Optional comma-delimited asset list.")
    parser.add_argument("--combo-profile-list", type=str, default="", help="interval:horizon:task@Nm@cadence tuples.")
    parser.add_argument("--combo-window-list", type=str, default="", help="interval:horizon:task@Nm tuples.")
    parser.add_argument("--combo-list", type=str, default="", help="interval:horizon:task tuples.")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Strict staged mode. Requires explicit approved Stage-2 survivors via CLI and forbids silent staged fallback.",
    )
    parser.add_argument("--stage2-manifest", type=Path, default=None, help=f"Stage-2 diagnostic_manifest.json used as the authoritative staged handoff for {CURRENT_MODEL_SPEC.display_name} stage 3.")
    parser.add_argument("--stage2-survivor-json", type=Path, default=None, help="Stage-2 stage3_survivor_handoff.json used as the approved combo/window handoff for staged Stage 3.")
    parser.add_argument("--train-window-months", type=str, default="")
    parser.add_argument("--refit-cadence", type=str, default="")
    parser.add_argument("--trials-per-combo", type=int, default=int(CURRENT_MODEL_SPEC.default_trials_per_combo))
    parser.add_argument("--model-threads", type=int, default=int(CURRENT_MODEL_SPEC.default_model_threads))
    parser.add_argument("--sampler-seed", type=int, default=17)
    parser.add_argument("--study-name-prefix", type=str, default=str(CURRENT_MODEL_SPEC.default_study_name_prefix))
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--resume-study", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=8, help="Concurrent combo-study worker processes. Default standard 8.")
    parser.add_argument("--trial-workers", type=int, default=1, help="Concurrent Optuna trial workers within each combo study. Default standard 1.")
    parser.add_argument("--allow-oversubscribe", action="store_true", default=True, help="Allow worker budget to exceed logical CPU capacity.")
    parser.add_argument("--strict-cpu-budget", action="store_true", help="Disable default oversubscription and enforce logical CPU budget.")
    parser.add_argument("--pruner-startup-trials", type=int, default=8)
    parser.add_argument("--pruner-warmup-steps", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--search-back-months", type=int, default=int(CURRENT_MODEL_SPEC.default_search_back_months))
    parser.add_argument("--history-window-months", type=int, default=int(CURRENT_MODEL_SPEC.default_history_window_months))
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def requested_assets(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage2_contexts(manifest_path: Path) -> Dict[Tuple[int, int], Stage2Context]:
    manifest = load_json(manifest_path)
    contexts: Dict[Tuple[int, int], Stage2Context] = {}
    duplicates: List[str] = []
    for run in manifest.get("runs", []):
        paths = run.get("paths") or {}
        run_summary_path = Path(paths.get("run_summary", ""))
        if not run_summary_path.exists():
            continue
        summary = load_json(run_summary_path)
        cfg = summary.get("config") or {}
        interval_label = str(cfg.get("interval", "0m"))
        interval_minutes = int(interval_label.rstrip("m") or 0)
        training_window_months = int(summary.get("training_window_months") or 0)
        assets = tuple(str(asset) for asset in (cfg.get("assets") or []))
        seed_ts = int(cfg.get("seed_ts") or 0)
        accuracy_end_ts = int(cfg.get("accuracy_end_ts") or 0)
        forecast_target_month_start_utc = str(cfg.get("forecast_target_month_start_utc") or "")
        if interval_minutes <= 0 or training_window_months <= 0 or not assets or seed_ts <= 0 or accuracy_end_ts <= 0:
            continue
        key = (int(interval_minutes), int(training_window_months))
        context = Stage2Context(
            interval=int(interval_minutes),
            training_window_months=int(training_window_months),
            assets=assets,
            seed_ts=int(seed_ts),
            accuracy_end_ts=int(accuracy_end_ts),
            forecast_target_month_start_utc=forecast_target_month_start_utc,
            run_summary_path=run_summary_path.resolve(),
        )
        if key in contexts:
            duplicates.append(f"interval={interval_minutes}m window={training_window_months}m")
            continue
        contexts[key] = context
    if duplicates:
        raise RuntimeError(
            f"Stage-2 manifest is ambiguous for staged {CURRENT_MODEL_SPEC.display_name} stage 3. Duplicate contexts found for: "
            + ", ".join(sorted(set(duplicates)))
        )
    return contexts


def resolve_stage2_context_for_combo(combo: ComboSpec, contexts: Dict[Tuple[int, int], Stage2Context]) -> Stage2Context:
    key = (int(combo.interval), int(combo.training_window_months))
    if key not in contexts:
        raise RuntimeError(
            f"Staged {CURRENT_MODEL_SPEC.display_name} stage-3 run blocked: no Stage-2 context matches "
            f"combo={combo.tuple_label}. Expected interval={int(combo.interval)}m "
            f"training_window={int(combo.training_window_months)}m in the supplied Stage-2 manifest."
        )
    return contexts[key]


def resolve_stage2_survivor_json_path(args: argparse.Namespace) -> Optional[Path]:
    explicit = getattr(args, "stage2_survivor_json", None)
    if explicit is not None:
        return Path(explicit).resolve()
    manifest = getattr(args, "stage2_manifest", None)
    if manifest is None:
        return None
    return Path(manifest).resolve().parent / "stage3_survivor_handoff.json"



def load_stage2_survivor_specs(path: Path) -> List[ComboSpec]:
    payload = load_json(path)
    rows = list(payload.get("survivors") or [])
    specs: List[ComboSpec] = []
    for row in rows:
        try:
            interval = int(row.get("interval_minutes") or 0)
            horizon = int(row.get("horizon_minutes") or 0)
            task = str(row.get("task") or "").strip()
            months = int(row.get("training_window_months") or 0)
        except Exception:
            continue
        if interval <= 0 or horizon <= 0 or months <= 0 or not task:
            continue
        specs.append(ComboSpec(interval, horizon, task, months, None))
    deduped: Dict[Tuple[int, int, str, int, Optional[str]], ComboSpec] = {}
    for spec in specs:
        deduped[(int(spec.interval), int(spec.horizon_minutes), str(spec.task), int(spec.training_window_months), spec.refit_cadence)] = spec
    return list(deduped.values())


def resolve_combo_specs(args: argparse.Namespace) -> List[ComboSpec]:
    combo_list = CURRENT_NUMERICS._parse_combo_list(args.combo_list)
    combo_windows = CURRENT_NUMERICS._parse_combo_window_list(args.combo_window_list)
    combo_profiles = CURRENT_NUMERICS._parse_combo_profile_list(args.combo_profile_list)
    explicit_train_window_months = CURRENT_NUMERICS._parse_train_window_months(args.train_window_months)
    explicit_refit_cadence = CURRENT_NUMERICS._parse_refit_cadence(args.refit_cadence) if str(args.refit_cadence).strip() else None
    if not combo_profiles and not combo_windows and not combo_list:
        if bool(getattr(args, "staged", False)):
            survivor_json = resolve_stage2_survivor_json_path(args)
            if survivor_json is not None and survivor_json.exists():
                resolved = load_stage2_survivor_specs(survivor_json)
                if resolved:
                    return resolved
            raise SystemExit(
                f"Staged {CURRENT_MODEL_SPEC.display_name} stage-3 run blocked: missing approved Stage-2 survivor input. "
                "Pass --combo-profile-list, --combo-window-list, or --combo-list explicitly, or provide a valid Stage-2 survivor handoff artifact. "
                "Stage 3 must not rediscover or widen survivors during staged execution."
            )
        raise SystemExit(
            f"{CURRENT_MODEL_SPEC.display_name} Optuna tuning requires explicit CLI combo targeting. "
            "Pass --combo-profile-list, --combo-window-list, or --combo-list."
        )
    resolved: List[ComboSpec] = []
    if combo_profiles:
        for interval, horizon, task, months, cadence in combo_profiles:
            resolved.append(ComboSpec(int(interval), int(horizon), str(task), int(months), str(cadence)))
    elif combo_windows:
        for interval, horizon, task, months in combo_windows:
            cadence = explicit_refit_cadence
            resolved.append(ComboSpec(int(interval), int(horizon), str(task), int(months), (str(cadence) if cadence else None)))
    else:
        for interval, horizon, task in combo_list:
            months = (
                int(explicit_train_window_months)
                if explicit_train_window_months is not None
                else int(CURRENT_NUMERICS.default_training_window_months_for_combo(int(interval), int(horizon), str(task)))
            )
            cadence = explicit_refit_cadence
            resolved.append(ComboSpec(int(interval), int(horizon), str(task), int(months), (str(cadence) if cadence else None)))
    return resolved


def compute_eval_window(
    *,
    interval_minutes: int,
    explicit_assets: Sequence[str],
    history_window_months: int,
    search_back_months: int,
) -> Tuple[int, int, List[str], MonthKey]:
    parquet_root = CURRENT_NUMERICS.PARQUET_ROOT
    ohlc_root = parquet_root / f"ohlcvt_{int(interval_minutes)}"
    scalar_root = parquet_root / f"scalar_features_{int(interval_minutes)}"
    clamp_start = MonthKey(CLAMP_START_YEAR, CLAMP_START_MONTH)
    final_month, eligible_assets = common_recent_window(
        ohlc_root=ohlc_root,
        scalar_root=scalar_root,
        min_assets=max(1, len(explicit_assets)) if explicit_assets else 1,
        window_months=int(history_window_months),
        search_back_months=int(search_back_months),
        clamp_start=clamp_start,
    )
    selected_assets = list(explicit_assets) if explicit_assets else list(eligible_assets)
    if explicit_assets:
        missing = [asset for asset in explicit_assets if asset not in eligible_assets]
        if missing:
            raise RuntimeError(
                f"Explicit assets are not eligible for interval={int(interval_minutes)}m recent diagnostic window: {','.join(missing)}"
            )
    if not selected_assets:
        raise RuntimeError(f"No eligible assets found for interval={int(interval_minutes)}m")
    edges: List[int] = []
    for asset in selected_assets:
        edge = max_asset_ts_for_month(ohlc_root, asset, final_month)
        if edge is None:
            raise RuntimeError(
                f"Could not resolve final-month source timestamp for asset={asset} interval={int(interval_minutes)}m"
            )
        edges.append(int(edge))
    next_month = add_months(final_month.year, final_month.month, 1)
    bar_seconds = int(interval_minutes) * 60
    expected_accuracy_end_ts = int(month_start_utc_ts(next_month.year, next_month.month) - bar_seconds)
    actual_accuracy_end_ts = int(min(edges))
    if actual_accuracy_end_ts < expected_accuracy_end_ts:
        raise RuntimeError(
            "Recent diagnostic month is incomplete across the selected cohort: "
            f"interval={int(interval_minutes)}m month={final_month.year:04d}-{final_month.month:02d}"
        )
    accuracy_end_ts = int(expected_accuracy_end_ts)
    seed_ts = int(month_start_utc_ts(final_month.year, final_month.month) - bar_seconds)
    return seed_ts, accuracy_end_ts, selected_assets, final_month


def build_dataset(asset: str, spec: ComboSpec, seed_ts: int, accuracy_end_ts: int) -> Dataset:
    repository = DatasetRepository()
    return repository.build_dataset(asset, spec, int(seed_ts), int(accuracy_end_ts))


def baseline_params_with_threads(spec: ComboSpec, model_threads: int) -> Dict[str, Any]:
    return dict(CURRENT_OPTUNA_PROFILE.resolve_baseline_params(task=spec.task, model_threads=int(model_threads), combo=spec))


def _cost_param_telemetry(params: Mapping[str, Any]) -> Dict[str, Any]:
    def pick(*names: str) -> Any:
        for name in names:
            if name in params:
                return params.get(name)
        return None

    return {
        "model_iterations": pick("iterations", "n_estimators", "num_boost_round"),
        "model_depth": pick("depth", "max_depth"),
        "learning_rate": pick("learning_rate", "eta"),
        "bootstrap_type": pick("bootstrap_type"),
        "subsample": pick("subsample", "bagging_fraction", "rsm"),
        "thread_count": pick("thread_count", "thread_count_", "nthread", "n_jobs"),
        "early_stopping_rounds": pick("early_stopping_rounds", "od_wait"),
    }


def trial_params(trial: optuna.Trial, spec: ComboSpec) -> Dict[str, Any]:
    return dict(CURRENT_OPTUNA_PROFILE.suggest_trial_params(trial, spec))


def finalize_model_params(params: Dict[str, Any], spec: ComboSpec) -> Dict[str, Any]:
    finalize_fn = getattr(CURRENT_OPTUNA_PROFILE, "finalize_params", None)
    if callable(finalize_fn):
        return dict(finalize_fn(dict(params), spec))
    return dict(params)


def metric_from_predictions(
    dataset: Dataset,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    tail_fill_ts: Sequence[int],
    meta: Dict[str, Any],
    params_label: str,
) -> MetricResult:
    label_col = CURRENT_NUMERICS.TASK_LABEL[dataset.spec.task]
    first_real_prediction_ts = meta.get("first_real_prediction_ts") if isinstance(meta, dict) else None
    if first_real_prediction_ts is None:
        return MetricResult(dataset.spec.tuple_label, dataset.asset, 0, None, None, None, None, None, None, len(tail_fill_ts), params_label)
    pending_set = {int(ts_i) for ts_i in tail_fill_ts}
    effective_first_ts = max(int(dataset.seed_ts) + int(dataset.spec.interval) * 60, int(first_real_prediction_ts))
    mask = (
        (pred_df["ts"].astype("int64") >= int(effective_first_ts))
        & (pred_df["ts"].astype("int64") <= int(dataset.accuracy_end_ts))
        & (~pred_df["ts"].astype("int64").isin(pending_set))
    )
    pred_rows = pred_df.loc[mask, ["ts", "pred_mean"]].copy()
    eval_rows = eval_df.loc[mask, ["ts", label_col]].copy()
    if pred_rows.empty or eval_rows.empty:
        return MetricResult(dataset.spec.tuple_label, dataset.asset, 0, None, None, None, None, int(effective_first_ts), None, len(tail_fill_ts), params_label)
    merged = pred_rows.merge(eval_rows, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    y_true = pd.to_numeric(merged[label_col], errors="coerce").astype(float)
    y_pred = pd.to_numeric(merged["pred_mean"], errors="coerce").astype(float)
    baseline = y_true.shift(1)
    valid = baseline.notna() & y_true.notna() & y_pred.notna()
    if not valid.any():
        return MetricResult(dataset.spec.tuple_label, dataset.asset, int(len(merged)), None, None, None, None, int(merged["ts"].min()), int(merged["ts"].max()), len(tail_fill_ts), params_label)
    y_true_valid = y_true.loc[valid].to_numpy(dtype=float)
    y_pred_valid = y_pred.loc[valid].to_numpy(dtype=float)
    baseline_valid = baseline.loc[valid].to_numpy(dtype=float)
    err = y_pred_valid - y_true_valid
    baseline_err = baseline_valid - y_true_valid
    return MetricResult(
        combo=dataset.spec.tuple_label,
        asset=dataset.asset,
        rows=int(valid.sum()),
        rmse=float(math.sqrt(np.mean(np.square(err)))),
        mae=float(np.mean(np.abs(err))),
        baseline_rmse=float(math.sqrt(np.mean(np.square(baseline_err)))),
        baseline_mae=float(np.mean(np.abs(baseline_err))),
        first_prediction_ts=int(merged["ts"].min()),
        last_prediction_ts=int(merged["ts"].max()),
        pending_tail_rows=len(tail_fill_ts),
        params_label=params_label,
    )


def evaluate_dataset(dataset: Dataset, params: Dict[str, Any], params_label: str, quiet_progress: bool) -> MetricResult:
    pred_df, eval_df, tail_fill_ts, meta = CURRENT_NUMERICS._walk_forward_predict(
        df=dataset.df,
        task=dataset.spec.task,
        horizon_minutes=int(dataset.spec.horizon_minutes),
        interval_minutes=int(dataset.spec.interval),
        horizon_bars=int(dataset.spec.horizon_bars),
        selected_window_bars=int(dataset.spec.training_window_bars),
        refit_cadence=(str(dataset.spec.refit_cadence) if dataset.spec.refit_cadence else None),
        initial_state=None,
        process_from_ts=int(dataset.seed_ts),
        progress_label=f"asset={dataset.asset} combo={dataset.spec.tuple_label}",
        progress_every_seconds=(10**9 if quiet_progress else CURRENT_NUMERICS.PROGRESS_EVERY_SECONDS),
        regressor_params=dict(params),
        prepared_x_cols=dataset.x_cols,
        prepared_x=dataset.x,
        prepared_ts=dataset.ts,
        prepared_y=dataset.y,
    )
    return metric_from_predictions(dataset, pred_df, eval_df, tail_fill_ts, meta, params_label)


def summarize_metrics(metrics: Sequence[MetricResult]) -> Dict[str, Optional[float]]:
    rmse_num = 0.0
    mae_num = 0.0
    baseline_rmse_num = 0.0
    baseline_mae_num = 0.0
    total_rows = 0
    for metric in metrics:
        if metric.rows <= 0:
            continue
        if metric.rmse is not None and math.isfinite(float(metric.rmse)):
            rmse_num += float(metric.rmse) * int(metric.rows)
            total_rows += int(metric.rows)
        if metric.mae is not None and math.isfinite(float(metric.mae)):
            mae_num += float(metric.mae) * int(metric.rows)
        if metric.baseline_rmse is not None and math.isfinite(float(metric.baseline_rmse)):
            baseline_rmse_num += float(metric.baseline_rmse) * int(metric.rows)
        if metric.baseline_mae is not None and math.isfinite(float(metric.baseline_mae)):
            baseline_mae_num += float(metric.baseline_mae) * int(metric.rows)
    return {
        "rows": int(total_rows),
        "weighted_rmse": float(rmse_num / total_rows) if total_rows > 0 else None,
        "weighted_mae": float(mae_num / total_rows) if total_rows > 0 else None,
        "baseline_weighted_rmse": float(baseline_rmse_num / total_rows) if total_rows > 0 else None,
        "baseline_weighted_mae": float(baseline_mae_num / total_rows) if total_rows > 0 else None,
    }


def normalize_worker_inputs(
    combo_count: int,
    model_threads: int,
    combo_workers: int,
    trial_workers: int,
    allow_oversubscribe: bool,
) -> ConcurrencyPlan:
    if int(combo_count) <= 0:
        raise SystemExit("No combos requested.")
    if int(model_threads) <= 0:
        raise SystemExit("--model-threads must be positive")
    if int(combo_workers) < 0:
        raise SystemExit("--parallel-workers must be zero or positive")
    if int(trial_workers) < 0:
        raise SystemExit("--trial-workers must be zero or positive")
    logical_cpus = max(1, int(os.cpu_count() or 1))
    oversub_fit_slots = max(1, int(math.ceil((logical_cpus * 1.5) / max(1, int(model_threads)))))
    if int(combo_workers) == 0 and int(trial_workers) == 0:
        if oversub_fit_slots <= 1:
            resolved_combo_workers = 1
            resolved_trial_workers = 1
        elif oversub_fit_slots == 2:
            resolved_combo_workers = 1
            resolved_trial_workers = 2
        else:
            resolved_combo_workers = min(int(combo_count), 2)
            resolved_trial_workers = max(2, min(4, oversub_fit_slots // max(1, resolved_combo_workers)))
    elif int(combo_workers) == 0:
        resolved_trial_workers = int(trial_workers)
        resolved_combo_workers = max(1, min(int(combo_count), oversub_fit_slots // max(1, resolved_trial_workers)))
    elif int(trial_workers) == 0:
        resolved_combo_workers = int(combo_workers)
        resolved_trial_workers = max(1, oversub_fit_slots // max(1, resolved_combo_workers))
    else:
        resolved_combo_workers = int(combo_workers)
        resolved_trial_workers = int(trial_workers)
    plan = ConcurrencyPlan(
        logical_cpus=logical_cpus,
        combo_workers=max(1, min(int(combo_count), int(resolved_combo_workers))),
        trial_workers=max(1, int(resolved_trial_workers)),
        model_threads=int(model_threads),
    )
    if plan.cpu_budget > plan.logical_cpus and not allow_oversubscribe:
        raise SystemExit(
            f"Invalid concurrency budget: combo_workers={plan.combo_workers}, trial_workers={plan.trial_workers}, model_threads={plan.model_threads}, cpu_budget={plan.cpu_budget}, logical_cpus={plan.logical_cpus}. Reduce workers or pass --allow-oversubscribe."
        )
    return plan


def task_storage_for(base_storage: Optional[str], combo: ComboSpec) -> Optional[str]:
    if not base_storage:
        return None
    raw = str(base_storage).strip()
    combo_tag = f"k{int(combo.interval)}_h{int(combo.horizon_minutes)}_{combo.task}"
    if raw.startswith("sqlite:///"):
        base_path = Path(raw.replace("sqlite:///", "", 1))
        task_path = base_path.with_name(f"{base_path.stem}_{combo_tag}{base_path.suffix or '.db'}")
        task_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{task_path}"
    return raw


def run_study_for_combo(
    *,
    combo: ComboSpec,
    datasets: Sequence[Dataset],
    trials_per_combo: int,
    sampler_seed: int,
    storage: Optional[str],
    study_name_prefix: str,
    resume_study: bool,
    model_threads: int,
    trial_workers: int,
    pruner_startup_trials: int,
    pruner_warmup_steps: int,
    timeout_seconds: int,
    quiet_progress: bool,
    telemetry_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[MetricResult], List[MetricResult], List[Dict[str, Any]]]:
    trial_lock = threading.Lock()
    baseline_params = baseline_params_with_threads(combo, int(model_threads))
    t_baseline = time.monotonic()
    with telemetry_scope_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="evaluate_dataset",
        module_name=__name__,
        phase_name="fit",
        parent_phase="baseline_evaluation",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=sum(len(dataset.df) for dataset in datasets),
        asset_count=len(datasets),
    ) as baseline_scope:
        baseline_metrics = [evaluate_dataset(dataset, baseline_params, "baseline", quiet_progress) for dataset in datasets]
        baseline_scope.update(output_rows=sum(int(metric.rows) for metric in baseline_metrics))
    baseline_eval_s = time.monotonic() - t_baseline
    baseline_summary = summarize_metrics(baseline_metrics)
    trial_rows: List[Dict[str, Any]] = []
    combo_input_rows = sum(len(dataset.df) for dataset in datasets)
    combo_feature_count = max((len(dataset.x_cols) for dataset in datasets), default=0)

    def objective(trial: optuna.Trial) -> float:
        params = baseline_params_with_threads(combo, int(model_threads))
        params.update(trial_params(trial, combo))
        params = finalize_model_params(params, combo)
        rmse_num = 0.0
        mae_num = 0.0
        rows = 0
        started = time.monotonic()
        for step_idx, dataset in enumerate(datasets, start=1):
            metric = evaluate_dataset(dataset, params, f"trial_{trial.number}", quiet_progress)
            if metric.rmse is not None and metric.rows > 0:
                rmse_num += float(metric.rmse) * int(metric.rows)
                mae_num += float(metric.mae or 0.0) * int(metric.rows)
                rows += int(metric.rows)
            interim_rmse = float(rmse_num / rows) if rows > 0 else float("inf")
            trial.report(interim_rmse, step=step_idx)
            if trial.should_prune():
                elapsed_s = float(time.monotonic() - started)
                with trial_lock:
                    trial_rows.append(
                        {
                            "combo": combo.tuple_label,
                            "trial_number": int(trial.number),
                            "state": "PRUNED",
                            "objective_rmse": (None if not math.isfinite(interim_rmse) else float(interim_rmse)),
                            "weighted_mae": (float(mae_num / rows) if rows > 0 else None),
                            "rows": int(rows),
                            "elapsed_s": elapsed_s,
                            "params": dict(params),
                        }
                    )
                emit_event_for_path(
                    telemetry_path,
                    family="Tabular_Numeric",
                    model=str(CURRENT_MODEL_SPEC.model_key),
                    stage="stage3",
                    function_name="optuna.trial",
                    module_name=__name__,
                    phase_name="trial",
                    parent_phase="tuning",
                    event_type="trial",
                    status="pruned",
                    reason_code="trial_pruned",
                    combo_key=combo.tuple_label,
                    interval_minutes=int(combo.interval),
                    horizon_minutes=int(combo.horizon_minutes),
                    training_window_months=int(combo.training_window_months),
                    task=str(combo.task),
                    trial_number=int(trial.number),
                    trial_state="PRUNED",
                    elapsed_seconds=elapsed_s,
                    eval_elapsed_seconds=elapsed_s,
                    input_rows=int(combo_input_rows),
                    output_rows=int(rows),
                    eval_rows=int(rows),
                    asset_count=len(datasets),
                    selected_feature_count=int(combo_feature_count),
                    baseline_metric=baseline_summary.get("weighted_rmse"),
                    trial_metric=(None if not math.isfinite(interim_rmse) else float(interim_rmse)),
                    metric_delta=(
                        float(interim_rmse) - float(baseline_summary["weighted_rmse"])
                        if math.isfinite(interim_rmse) and baseline_summary.get("weighted_rmse") is not None
                        else None
                    ),
                    objective_direction="minimize",
                    **_cost_param_telemetry(params),
                )
                raise optuna.TrialPruned()
        objective_value = float(rmse_num / rows) if rows > 0 else float("inf")
        elapsed_s = float(time.monotonic() - started)
        with trial_lock:
            trial_rows.append(
                {
                    "combo": combo.tuple_label,
                    "trial_number": int(trial.number),
                    "state": "COMPLETE",
                    "objective_rmse": (None if not math.isfinite(objective_value) else float(objective_value)),
                    "weighted_mae": (float(mae_num / rows) if rows > 0 else None),
                    "rows": int(rows),
                    "elapsed_s": elapsed_s,
                    "params": dict(params),
                }
            )
        emit_event_for_path(
            telemetry_path,
            family="Tabular_Numeric",
            model=str(CURRENT_MODEL_SPEC.model_key),
            stage="stage3",
            function_name="optuna.trial",
            module_name=__name__,
            phase_name="trial",
            parent_phase="tuning",
            event_type="trial",
            status="completed" if rows > 0 else "skipped",
            reason_code="" if rows > 0 else "predict_returned_empty",
            combo_key=combo.tuple_label,
            interval_minutes=int(combo.interval),
            horizon_minutes=int(combo.horizon_minutes),
            training_window_months=int(combo.training_window_months),
            task=str(combo.task),
            trial_number=int(trial.number),
            trial_state="COMPLETE",
            elapsed_seconds=elapsed_s,
            eval_elapsed_seconds=elapsed_s,
            input_rows=int(combo_input_rows),
            output_rows=int(rows),
            eval_rows=int(rows),
            asset_count=len(datasets),
            selected_feature_count=int(combo_feature_count),
            baseline_metric=baseline_summary.get("weighted_rmse"),
            trial_metric=(None if not math.isfinite(objective_value) else float(objective_value)),
            metric_delta=(
                float(objective_value) - float(baseline_summary["weighted_rmse"])
                if math.isfinite(objective_value) and baseline_summary.get("weighted_rmse") is not None
                else None
            ),
            objective_direction="minimize",
            **_cost_param_telemetry(params),
        )
        return objective_value

    storage_url = task_storage_for(storage, combo)
    with telemetry_scope_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="optuna.create_study",
        module_name=__name__,
        phase_name="study_setup",
        parent_phase="tuning",
        event_type="setup_control",
        setup_control=True,
        row_producing=False,
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        source_path=str(storage_url or ""),
    ) as study_scope:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(sampler_seed)),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=int(pruner_startup_trials),
                n_warmup_steps=int(pruner_warmup_steps),
            ),
            study_name=f"{study_name_prefix}_{combo.interval}_{combo.horizon_minutes}_{combo.task}",
            storage=storage_url,
            load_if_exists=bool(resume_study or storage),
        )
        study_scope.update(output_rows=len(study.trials))
    t_study = time.monotonic()
    if int(trials_per_combo) > 0:
        with telemetry_scope_for_path(
            telemetry_path,
            family="Tabular_Numeric",
            model=str(CURRENT_MODEL_SPEC.model_key),
            stage="stage3",
            function_name="study.optimize",
            module_name=__name__,
            phase_name="fit",
            parent_phase="tuning",
            combo_key=combo.tuple_label,
            interval_minutes=int(combo.interval),
            horizon_minutes=int(combo.horizon_minutes),
            task=str(combo.task),
            input_rows=int(trials_per_combo),
            asset_count=len(datasets),
        ) as optimize_scope:
            study.optimize(
                objective,
                n_trials=int(trials_per_combo),
                timeout=(int(timeout_seconds) if int(timeout_seconds) > 0 else None),
                show_progress_bar=False,
                n_jobs=int(trial_workers),
            )
            optimize_scope.update(
                output_rows=len(trial_rows),
                trial_count=len(trial_rows),
                completed_count=sum(1 for row in trial_rows if str(row.get("state")) == "COMPLETE"),
                pruned_count=sum(1 for row in trial_rows if str(row.get("state")) == "PRUNED"),
                failed_count=sum(1 for row in trial_rows if str(row.get("state")) == "FAIL"),
                asset_count=len(datasets),
                selected_feature_count=int(combo_feature_count),
                thread_count=int(model_threads),
            )
    study_elapsed_s = time.monotonic() - t_study
    best_params = baseline_params_with_threads(combo, int(model_threads))
    if study.trials:
        best_params.update(dict(study.best_trial.params))
    best_params = finalize_model_params(best_params, combo)
    t_tuned = time.monotonic()
    with telemetry_scope_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="evaluate_dataset",
        module_name=__name__,
        phase_name="predict",
        parent_phase="best_trial_evaluation",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=sum(len(dataset.df) for dataset in datasets),
        asset_count=len(datasets),
    ) as tuned_scope:
        tuned_metrics = [evaluate_dataset(dataset, best_params, "tuned", quiet_progress) for dataset in datasets]
        tuned_scope.update(output_rows=sum(int(metric.rows) for metric in tuned_metrics))
    tuned_eval_s = time.monotonic() - t_tuned
    tuned_summary = summarize_metrics(tuned_metrics)
    emit_event_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="summarize_metrics",
        module_name=__name__,
        phase_name="metric_calculation",
        parent_phase="best_trial_evaluation",
        status="completed" if int(tuned_summary.get("rows", 0) or 0) > 0 else "skipped",
        reason_code="" if int(tuned_summary.get("rows", 0) or 0) > 0 else "predict_returned_empty",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=len(tuned_metrics),
        output_rows=int(tuned_summary.get("rows", 0) or 0),
    )
    completed_trials = sum(1 for row in trial_rows if str(row.get("state")) == "COMPLETE")
    pruned_trials = sum(1 for row in trial_rows if str(row.get("state")) == "PRUNED")
    failed_trials = sum(1 for row in trial_rows if str(row.get("state")) == "FAIL")
    rmse_delta = (
        float(tuned_summary["weighted_rmse"]) - float(baseline_summary["weighted_rmse"])
        if baseline_summary.get("weighted_rmse") is not None and tuned_summary.get("weighted_rmse") is not None
        else None
    )
    emit_event_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="run_study_for_combo",
        module_name=__name__,
        phase_name="combo_summary",
        parent_phase="tuning",
        event_type="combo_summary",
        status="completed" if int(tuned_summary.get("rows", 0) or 0) > 0 else "skipped",
        reason_code="" if int(tuned_summary.get("rows", 0) or 0) > 0 else "predict_returned_empty",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        training_window_months=int(combo.training_window_months),
        task=str(combo.task),
        elapsed_seconds=float(study_elapsed_s + tuned_eval_s + baseline_eval_s),
        fit_elapsed_seconds=float(study_elapsed_s),
        eval_elapsed_seconds=float(tuned_eval_s),
        input_rows=int(combo_input_rows),
        output_rows=int(tuned_summary.get("rows", 0) or 0),
        eval_rows=int(tuned_summary.get("rows", 0) or 0),
        asset_count=len(datasets),
        selected_feature_count=int(combo_feature_count),
        trial_count=len(trial_rows),
        completed_count=int(completed_trials),
        pruned_count=int(pruned_trials),
        failed_count=int(failed_trials),
        baseline_metric=baseline_summary.get("weighted_rmse"),
        tuned_metric=tuned_summary.get("weighted_rmse"),
        metric_delta=rmse_delta,
        best_value=(float(study.best_value) if study.trials else None),
        objective_direction="minimize",
        **_cost_param_telemetry(best_params),
    )
    emit_stage3_study_summary_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        function_name="run_study_for_combo",
        module_name=__name__,
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        elapsed_seconds=float(study_elapsed_s),
        trial_rows=trial_rows,
        study_trials=list(study.trials),
        best_value=(float(study.best_value) if study.trials else None),
        input_rows=int(trials_per_combo),
        output_rows=sum(int(metric.rows) for metric in tuned_metrics),
        source_path=str(storage_url or ""),
    )
    combo_row = {
        "combo": combo.tuple_label,
        "interval": int(combo.interval),
        "horizon_minutes": int(combo.horizon_minutes),
        "task": combo.task,
        "training_window_months": int(combo.training_window_months),
        "refit_cadence": combo.refit_cadence,
        "baseline_rmse": baseline_summary.get("weighted_rmse"),
        "baseline_mae": baseline_summary.get("weighted_mae"),
        "tuned_rmse": tuned_summary.get("weighted_rmse"),
        "tuned_mae": tuned_summary.get("weighted_mae"),
        "baseline_rows": baseline_summary.get("rows"),
        "tuned_rows": tuned_summary.get("rows"),
        "rmse_delta": rmse_delta,
        "mae_delta": (
            float(tuned_summary["weighted_mae"]) - float(baseline_summary["weighted_mae"])
            if baseline_summary.get("weighted_mae") is not None and tuned_summary.get("weighted_mae") is not None
            else None
        ),
        "best_params": dict(best_params),
        "best_value": (float(study.best_value) if study.trials else None),
        "baseline_eval_s": float(baseline_eval_s),
        "study_elapsed_s": float(study_elapsed_s),
        "tuned_eval_s": float(tuned_eval_s),
    }
    return combo_row, baseline_metrics, tuned_metrics, trial_rows


def write_outputs(
    output_dir: Path,
    sample_rows: Sequence[Dict[str, Any]],
    combo_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[MetricResult],
    trial_rows: Sequence[Dict[str, Any]],
    runtime_rows: Sequence[Dict[str, Any]],
    concurrency_plan: ConcurrencyPlan,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(list(sample_rows)).to_csv(output_dir / "representative_samples.csv", index=False)
    pd.DataFrame(list(combo_rows)).to_csv(output_dir / "combo_results.csv", index=False)
    pd.DataFrame([asdict(metric) for metric in metric_rows]).to_csv(output_dir / "unit_metrics.csv", index=False)
    pd.DataFrame(list(trial_rows)).to_json(output_dir / "optuna_trials.json", orient="records", indent=2)
    if runtime_rows:
        pd.DataFrame(list(runtime_rows)).to_csv(output_dir / "runtime_summary.csv", index=False)
    emit_event_for_path(
        output_dir,
        family="Tabular_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="write_outputs",
        module_name=__name__,
        phase_name="artifact_handoff",
        status="completed",
        input_rows=len(metric_rows),
        output_rows=len(combo_rows),
        output_path=str(output_dir),
        reason_code=("predict_returned_empty" if not metric_rows else ""),
    )
    lines = [
        f"# {CURRENT_MODEL_SPEC.display_name} Combo-Level Optuna Tuning",
        "",
        "Objective: minimize validation RMSE on the same recent diagnostic month used by survivor evaluation.",
        "",
        "Concurrency:",
        f"- combo_workers={concurrency_plan.combo_workers}",
        f"- trial_workers={concurrency_plan.trial_workers}",
        f"- model_threads={concurrency_plan.model_threads}",
        f"- cpu_budget={concurrency_plan.cpu_budget}/{concurrency_plan.logical_cpus}",
        "",
    ]
    for row in combo_rows:
        lines.append(
            f"- `{row['combo']}`: baseline_rmse={row['baseline_rmse']}, tuned_rmse={row['tuned_rmse']}, "
            f"baseline_mae={row['baseline_mae']}, tuned_mae={row['tuned_mae']}, rmse_delta={row['rmse_delta']}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_combo(
    combo: ComboSpec,
    args_dict: Dict[str, Any],
    model_spec: TabularOptunaModelSpec,
    repository: Optional[DatasetRepository] = None,
) -> Dict[str, Any]:
    configure_for_model(model_spec)
    local_repository = repository if repository is not None else DatasetRepository()
    telemetry_path = Path(str(args_dict["telemetry_path"])) if str(args_dict.get("telemetry_path") or "").strip() else None

    def _run_once() -> Dict[str, Any]:
        explicit_assets = list(args_dict.get('explicit_assets', []))
        if bool(args_dict.get('staged')):
            context = resolve_stage2_context_for_combo(combo, dict(args_dict['stage2_contexts']))
            if explicit_assets:
                requested = [asset for asset in explicit_assets if asset in set(context.assets)]
                if len(requested) != len(explicit_assets):
                    missing = [asset for asset in explicit_assets if asset not in set(context.assets)]
                    raise RuntimeError(
                        f"Staged {CURRENT_MODEL_SPEC.display_name} stage-3 run blocked: explicit assets are outside the authoritative Stage-2 cohort for "
                        f"combo={combo.tuple_label}: {','.join(missing)}"
                    )
                assets = requested
            else:
                assets = list(context.assets)
            seed_ts = int(context.seed_ts)
            accuracy_end_ts = int(context.accuracy_end_ts)
            final_month_dt = datetime.fromisoformat(str(context.forecast_target_month_start_utc))
            final_month = MonthKey(int(final_month_dt.year), int(final_month_dt.month))
        else:
            seed_ts, accuracy_end_ts, assets, final_month = local_repository.eval_window_for_interval(
                interval_minutes=int(combo.interval),
                explicit_assets=explicit_assets,
                history_window_months=int(args_dict['history_window_months']),
                search_back_months=int(args_dict['search_back_months']),
            )
        combo_datasets: List[Dataset] = []
        sample_rows: List[Dict[str, Any]] = []
        with telemetry_scope_for_path(
            telemetry_path,
            family="Tabular_Numeric",
            model=str(CURRENT_MODEL_SPEC.model_key),
            stage="stage3",
            function_name="DatasetRepository.build_dataset",
            module_name=__name__,
            phase_name="dataset_construction",
            parent_phase="objective_setup",
            combo_key=combo.tuple_label,
            interval_minutes=int(combo.interval),
            horizon_minutes=int(combo.horizon_minutes),
            task=str(combo.task),
            input_rows=len(assets),
            asset_count=len(assets),
        ) as dataset_scope:
            for asset in assets:
                dataset = local_repository.build_dataset(asset, combo, int(seed_ts), int(accuracy_end_ts))
                combo_datasets.append(dataset)
                sample_rows.append(
                    {
                        'combo': combo.tuple_label,
                        'asset': asset,
                        'seed_ts': int(seed_ts),
                        'accuracy_end_ts': int(accuracy_end_ts),
                        'source_start_ts': int(dataset.source_start_ts),
                        'source_end_ts': int(dataset.source_end_ts),
                        'rows': int(len(dataset.df)),
                        'diagnostic_month': f"{final_month.year:04d}-{final_month.month:02d}",
                    }
                )
            dataset_scope.update(
                output_rows=sum(int(len(dataset.df)) for dataset in combo_datasets),
                reason_code="" if combo_datasets else "objective_dataset_empty",
            )
        combo_row, baseline_metrics, tuned_metrics, combo_trials = run_study_for_combo(
            combo=combo,
            datasets=combo_datasets,
            trials_per_combo=int(args_dict['trials_per_combo']),
            sampler_seed=int(args_dict['sampler_seed']),
            storage=args_dict.get('storage'),
            study_name_prefix=str(args_dict['study_name_prefix']),
            resume_study=bool(args_dict['resume_study']),
            model_threads=int(args_dict['model_threads']),
            trial_workers=int(args_dict['trial_workers']),
            pruner_startup_trials=int(args_dict['pruner_startup_trials']),
            pruner_warmup_steps=int(args_dict['pruner_warmup_steps']),
            timeout_seconds=int(args_dict['timeout_seconds']),
            quiet_progress=bool(args_dict['quiet_progress']),
            telemetry_path=telemetry_path,
        )
        return {
            'sample_rows': sample_rows,
            'combo_row': combo_row,
            'metric_rows': [asdict(metric) for metric in baseline_metrics] + [asdict(metric) for metric in tuned_metrics],
            'trial_rows': combo_trials,
            'runtime_row': {'combo': combo.tuple_label, 'asset_count': len(assets), 'trial_workers': int(args_dict['trial_workers']), 'model_threads': int(args_dict['model_threads']), **local_repository.timings.snapshot()},
        }

    return run_with_transient_retry(_run_once, label=combo.tuple_label)


def main_for_model(model_spec: TabularOptunaModelSpec) -> None:
    configure_for_model(model_spec)
    args = parse_args()
    combos = resolve_combo_specs(args)
    plan = normalize_worker_inputs(
        combo_count=len(combos),
        model_threads=int(args.model_threads),
        combo_workers=int(args.parallel_workers),
        trial_workers=int(args.trial_workers),
        allow_oversubscribe=bool(args.allow_oversubscribe) and not bool(args.strict_cpu_budget),
    )
    storage = str(args.storage).strip() or None
    if bool(args.staged):
        if args.stage2_manifest is None:
            raise SystemExit(
                f"Staged {CURRENT_MODEL_SPEC.display_name} stage-3 run blocked: missing authoritative Stage-2 artifact. "
                "Pass --stage2-manifest <diagnostic_manifest.json> together with explicit approved survivor combos."
            )
        stage2_manifest = args.stage2_manifest.resolve()
        if not stage2_manifest.exists():
            raise SystemExit(
                f"Staged {CURRENT_MODEL_SPEC.display_name} stage-3 run blocked: Stage-2 manifest not found at "
                f"{stage2_manifest}."
            )
    else:
        stage2_manifest = None
    if storage and storage.startswith('sqlite:///'):
        db_path = Path(storage.replace('sqlite:///', '', 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.resolve() if args.output_dir else (CURRENT_MODEL_SPEC.output_root / f"run={utc_now_stamp()}").resolve()
    emit_event_for_path(
        output_dir,
        family="Tabular_Numeric",
        model=str(model_spec.model_key),
        stage="stage3",
        function_name="main_for_model",
        module_name=__name__,
        phase_name="combo_planning",
        status="completed",
        output_rows=len(combos),
        artifact_profile_source=(str(stage2_manifest) if stage2_manifest is not None else ""),
    )
    sample_rows: List[Dict[str, Any]] = []
    combo_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    stage2_contexts = load_stage2_contexts(stage2_manifest) if stage2_manifest is not None else {}

    args_dict = {
        'trials_per_combo': int(args.trials_per_combo),
        'sampler_seed': int(args.sampler_seed),
        'storage': storage,
        'study_name_prefix': str(args.study_name_prefix),
        'resume_study': bool(args.resume_study),
        'model_threads': int(plan.model_threads),
        'trial_workers': int(plan.trial_workers),
        'quiet_progress': bool(args.quiet_progress),
        'pruner_startup_trials': int(args.pruner_startup_trials),
        'pruner_warmup_steps': int(args.pruner_warmup_steps),
        'timeout_seconds': int(args.timeout_seconds),
        'explicit_assets': list(requested_assets(args.assets)),
        'history_window_months': int(args.history_window_months),
        'search_back_months': int(args.search_back_months),
        'staged': bool(args.staged),
        'stage2_contexts': stage2_contexts,
        'telemetry_path': str(output_dir),
    }
    if int(plan.combo_workers) <= 1:
        repository = DatasetRepository()
        try:
            for combo in combos:
                payload = run_single_combo(combo, args_dict, CURRENT_MODEL_SPEC, repository=repository)
                sample_rows.extend(list(payload['sample_rows']))
                combo_rows.append(dict(payload['combo_row']))
                metric_rows.extend(list(payload['metric_rows']))
                trial_rows.extend(list(payload['trial_rows']))
                runtime_rows.append(dict(payload['runtime_row']))
        finally:
            repository.clear()
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(plan.combo_workers)) as executor:
                futures = {executor.submit(run_single_combo, combo, args_dict, CURRENT_MODEL_SPEC): combo.tuple_label for combo in combos}
                for future in as_completed(futures):
                    payload = future.result()
                    sample_rows.extend(list(payload['sample_rows']))
                    combo_rows.append(dict(payload['combo_row']))
                    metric_rows.extend(list(payload['metric_rows']))
                    trial_rows.extend(list(payload['trial_rows']))
                    runtime_rows.append(dict(payload['runtime_row']))
        except (PermissionError, BrokenProcessPool):
            repository = DatasetRepository()
            try:
                for combo in combos:
                    payload = run_single_combo(combo, args_dict, CURRENT_MODEL_SPEC, repository=repository)
                    sample_rows.extend(list(payload['sample_rows']))
                    combo_rows.append(dict(payload['combo_row']))
                    metric_rows.extend(list(payload['metric_rows']))
                    trial_rows.extend(list(payload['trial_rows']))
                    runtime_rows.append(dict(payload['runtime_row']))
            finally:
                repository.clear()
    write_outputs(
        output_dir=output_dir,
        sample_rows=sample_rows,
        combo_rows=combo_rows,
        metric_rows=[MetricResult(**row) for row in metric_rows],
        trial_rows=trial_rows,
        runtime_rows=runtime_rows,
        concurrency_plan=plan,
    )
    print(json.dumps({'output_dir': str(output_dir), 'concurrency': asdict(plan), 'combo_count': len(combo_rows)}, indent=2))


