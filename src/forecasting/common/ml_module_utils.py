from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path


DEFAULT_MODEL_OUTPUT_PREFIXES: Tuple[str, ...] = (
    "xgb_",
    "lgb_",
    "lgbm_",
    "cb_",
    "rf_",
    "en_",
    "enet_",
)

DEFAULT_REGIME_LABEL_COLUMNS: Tuple[str, ...] = (
    "regime_3",
    "regime_3_idx",
    "trend_label",
    "vol_label",
)

_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_LOGGER_WARNED_BASELOG: set[str] = set()


def get_ts_floor_2021() -> int:
    return int(datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def get_module_logger(
    logger_name: str,
    log_file: Path,
    base_log_fn: Optional[Callable[[str], None]] = None,
    *,
    console: bool = True,
) -> Callable[[str], None]:
    name = str(logger_name)
    logger = _LOGGER_CACHE.get(name)
    if logger is None:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter("[%(asctime)s UTC] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        if console:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            max_bytes = int(os.getenv("ML_LOG_MAX_BYTES", "10485760"))
            backups = int(os.getenv("ML_LOG_BACKUPS", "5"))
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning("Failed to configure rotating file handler at %s: %s", log_file, e)
        _LOGGER_CACHE[name] = logger

    def _log(msg: str) -> None:
        logger.info(str(msg))
        if base_log_fn is not None:
            try:
                base_log_fn(str(msg))
            except Exception as e:
                key = f"{name}:base_log"
                if key not in _LOGGER_WARNED_BASELOG:
                    _LOGGER_WARNED_BASELOG.add(key)
                    logger.warning("base_log forwarding failed and will be muted: %s", e)

    return _log


def get_model_output_prefixes(extra_prefixes: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    cfg_raw = os.getenv("ML_MODEL_OUTPUT_PREFIXES", "")
    cfg = [p.strip() for p in cfg_raw.split(",") if p.strip()]
    items = list(DEFAULT_MODEL_OUTPUT_PREFIXES)
    items.extend(cfg)
    if extra_prefixes:
        items.extend(str(p).strip() for p in extra_prefixes if str(p).strip())
    seen = set()
    out = []
    for p in items:
        p2 = p.lower()
        if p2 in seen:
            continue
        seen.add(p2)
        out.append(p2)
    return tuple(out)


def horizon_bars_from_minutes(interval_minutes: int, horizon_minutes: int) -> int:
    hm = int(horizon_minutes)
    im = int(interval_minutes)
    if hm <= 0 or im <= 0:
        raise ValueError(f"horizon/interval must be positive: horizon_minutes={hm} interval_minutes={im}")
    if hm % im != 0:
        raise ValueError(f"horizon_minutes must be divisible by interval_minutes: horizon_minutes={hm} interval_minutes={im}")
    return hm // im


def tail_start_index(n_rows: int, horizon_bars: int) -> int:
    n = int(n_rows)
    h = int(horizon_bars)
    return max(0, n - max(0, h))


def is_tail_index(i: int, n_rows: int, horizon_bars: int) -> bool:
    return int(i) >= tail_start_index(n_rows=n_rows, horizon_bars=horizon_bars)


def make_unit_key(
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    asset: str,
    interval: int,
) -> str:
    return f"{str(family)}|{str(domain)}|{str(task)}|{int(horizon_minutes)}m|{str(asset)}|{int(interval)}"


def _has_prefix(value: str, prefixes: Sequence[str]) -> bool:
    v = str(value).lower()
    return any(v.startswith(p) for p in prefixes)


def select_numeric_feature_columns(
    columns: Iterable[str],
    extra_exclude: Optional[Sequence[str]] = None,
    model_output_prefixes: Optional[Sequence[str]] = None,
) -> list[str]:
    prefixes = get_model_output_prefixes(model_output_prefixes)
    deny = {"ts", "asset"}
    if extra_exclude:
        deny.update(str(x) for x in extra_exclude)

    out: list[str] = []
    for c in columns:
        cs = str(c)
        csl = cs.lower()
        if cs in deny:
            continue
        if csl.startswith("future_"):
            continue
        if csl.startswith("regime_"):
            continue
        if csl.endswith("_label"):
            continue
        if _has_prefix(csl, prefixes):
            continue
        out.append(cs)
    return out


def select_regime_feature_columns(
    columns: Iterable[str],
    label_source_columns: Optional[Sequence[str]] = None,
    extra_exclude: Optional[Sequence[str]] = None,
    model_output_prefixes: Optional[Sequence[str]] = None,
) -> list[str]:
    prefixes = get_model_output_prefixes(model_output_prefixes)
    deny = {"ts", "asset"}
    deny.update(DEFAULT_REGIME_LABEL_COLUMNS)
    if label_source_columns:
        deny.update(str(x) for x in label_source_columns)
    if extra_exclude:
        deny.update(str(x) for x in extra_exclude)

    out: list[str] = []
    for c in columns:
        cs = str(c)
        csl = cs.lower()
        if cs in deny:
            continue
        if csl.startswith("future_"):
            continue
        if _has_prefix(csl, prefixes):
            continue
        out.append(cs)
    return out


def apply_head_floor(
    df: pd.DataFrame,
    ts_floor: int,
    *,
    asset_col: str = "asset",
    ts_col: str = "ts",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or df.empty or ts_col not in df.columns:
        return df, {"rows_dropped_pre_floor": 0, "ts_start_by_asset": {}}

    out = df.copy()
    added_asset_col = False
    if asset_col not in out.columns:
        out[asset_col] = "__single_asset__"
        added_asset_col = True

    out[ts_col] = pd.to_numeric(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).copy()
    if out.empty:
        if added_asset_col:
            out = out.drop(columns=[asset_col], errors="ignore")
        return out, {"rows_dropped_pre_floor": 0, "ts_start_by_asset": {}}

    out[ts_col] = out[ts_col].astype("int64")
    out[asset_col] = out[asset_col].astype(str)
    out = out.sort_values([asset_col, ts_col]).reset_index(drop=True)

    asset_min = out.groupby(asset_col, dropna=True)[ts_col].min()
    ts_start_by_asset: Dict[str, int] = {
        str(a): int(max(int(ts_floor), int(min_ts))) for a, min_ts in asset_min.items()
    }

    start_series = out[asset_col].map(ts_start_by_asset).astype("int64")
    keep_mask = out[ts_col].astype("int64") >= start_series
    dropped = int((~keep_mask).sum())
    out = out.loc[keep_mask].reset_index(drop=True)

    if added_asset_col:
        out = out.drop(columns=[asset_col], errors="ignore")

    return out, {"rows_dropped_pre_floor": dropped, "ts_start_by_asset": ts_start_by_asset}


def _read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def acquire_single_run_lock(state_root: Path, run_name: str) -> Path:
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / f'{str(run_name)}.lock'
    try:
        with lock_path.open('x', encoding='utf-8') as f:
            json.dump({'run_name': str(run_name), 'pid': os.getpid(), 'created_at': datetime.now(timezone.utc).isoformat()}, f)
    except FileExistsError:
        raise RuntimeError(f'{run_name} already appears to be running: {lock_path}')
    return lock_path


def read_ml_state(state_root: Path, default_compaction: Optional[Dict[str, Any]] = None) -> tuple[dict, dict, dict]:
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    combined = _read_json_dict(state_root / 'ml_state.json')
    watermarks = dict(combined.get('watermarks', {})) if isinstance(combined.get('watermarks', {}), dict) else {}
    pending = dict(combined.get('pending', {})) if isinstance(combined.get('pending', {}), dict) else {}
    progress = dict(combined.get('progress', {})) if isinstance(combined.get('progress', {}), dict) else {}

    if not watermarks:
        watermarks = _read_json_dict(state_root / 'ml_watermarks.json')
    if not pending:
        pending = _read_json_dict(state_root / 'ml_pending.json')
    if not progress:
        progress = _read_json_dict(state_root / 'ml_progress.json')

    if not isinstance(watermarks.get('units', {}), dict):
        watermarks['units'] = {}
    if not isinstance(pending.get('entries', {}), dict):
        pending['entries'] = {}
    if default_compaction is not None:
        pending.setdefault('compaction', default_compaction)
    return watermarks, pending, progress


def write_ml_state(state_root: Path, watermarks: dict, pending: dict, progress: dict) -> None:
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    combined = {
        'watermarks': watermarks if isinstance(watermarks, dict) else {},
        'pending': pending if isinstance(pending, dict) else {},
        'progress': progress if isinstance(progress, dict) else {},
    }
    _write_json_atomic(state_root / 'ml_state.json', combined)
    _write_json_atomic(state_root / 'ml_watermarks.json', combined['watermarks'])
    _write_json_atomic(state_root / 'ml_pending.json', combined['pending'])
    _write_json_atomic(state_root / 'ml_progress.json', combined['progress'])


def replace_pending_entries_for_unit(
    pending: dict,
    unit_key: str,
    unit_meta: Optional[Dict[str, Any]],
    ts_list: Sequence[int],
    reason: str,
) -> None:
    entries = pending.setdefault('entries', {})
    if not isinstance(entries, dict):
        entries = {}
        pending['entries'] = entries
    prefix = f'{str(unit_key)}|'
    stale = [k for k in entries.keys() if str(k).startswith(prefix)]
    for key in stale:
        entries.pop(key, None)
    meta = dict(unit_meta or {})
    for ts in sorted({int(x) for x in ts_list}):
        entries[f'{prefix}{int(ts)}'] = {'ts': int(ts), 'reason': str(reason), **meta}


def prune_pending_ts(
    ts_list: Sequence[int],
    *,
    last_written_ts: Optional[int] = None,
    buffer_seconds: int = 0,
    source_ts_index: Optional[set[int]] = None,
    max_entries: Optional[int] = None,
) -> list[int]:
    keep = sorted({int(x) for x in ts_list})
    if source_ts_index is not None:
        keep = [ts for ts in keep if int(ts) in source_ts_index]
    if last_written_ts is not None:
        floor = int(last_written_ts) - max(0, int(buffer_seconds))
        keep = [ts for ts in keep if int(ts) >= floor]
    if max_entries is not None and int(max_entries) > 0:
        keep = keep[-int(max_entries):]
    return keep
