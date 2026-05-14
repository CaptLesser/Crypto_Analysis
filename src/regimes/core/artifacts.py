from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed
from src.regimes.contracts import regime_label_month_dir

DEFAULT_PARQUET_COMPRESSION = os.getenv("PIPELINE_PARQUET_COMPRESSION", "snappy")
DEFAULT_PARQUET_ROW_GROUP = int(os.getenv("PIPELINE_PARQUET_ROW_GROUP", "500000"))


def _assert_allowed(path: Path, write_kind: Optional[str], roots: Optional[SandboxOutputRoots] = None) -> None:
    if write_kind:
        assert_write_allowed(path, write_kind, roots=roots)


def safe_path_part(value: object, *, context: str = "Regime artifact path parts") -> str:
    cleaned = str(value).strip().replace("/", "_").replace("\\", "_")
    if not cleaned:
        raise ValueError(f"{context} must be non-empty")
    return cleaned


def validate_partition_month(month: int, *, context: str = "Regime artifact month") -> int:
    month_int = int(month)
    if month_int < 1 or month_int > 12:
        raise ValueError(f"{context} must be between 1 and 12, got {month!r}")
    return month_int


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def write_json(
    path: Path,
    obj: dict[str, Any],
    *,
    write_kind: Optional[str] = None,
    sandbox_roots: Optional[SandboxOutputRoots] = None,
) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_allowed(path, write_kind, sandbox_roots)
    tmp = sibling_temp_path(path)
    _assert_allowed(tmp, f"{write_kind} temp" if write_kind else None, sandbox_roots)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def label_part_path(root: Path, ceiling_interval: int, asset: str, year: int, month: int) -> Path:
    month_int = validate_partition_month(month, context="Regime label month")
    return regime_label_month_dir(
        Path(root),
        int(ceiling_interval),
        safe_path_part(asset),
        int(year),
        month_int,
    ) / "part-000.parquet"


def parquet_path_for(ceiling_interval: int, asset: str, year: int, month: int, root: Path) -> Path:
    return label_part_path(root, ceiling_interval, asset, year, month)


def _lock_path_for(dst: Path) -> Path:
    return Path(dst).with_suffix(Path(dst).suffix + ".lock")


def _acquire_lock(lock_path: Path, retries: int = 40, sleep_sec: float = 0.25) -> Optional[int]:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(int(retries)):
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            time.sleep(float(sleep_sec))
        except Exception:
            time.sleep(float(sleep_sec))
    return None


def _release_lock(fd: Optional[int], lock_path: Path) -> None:
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        if Path(lock_path).exists():
            Path(lock_path).unlink()
    except Exception:
        pass


def write_parquet_atomic(
    df: pd.DataFrame,
    dst: Path,
    *,
    retries: int = 6,
    sleep_base_sec: float = 0.25,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
    row_group_size: int = DEFAULT_PARQUET_ROW_GROUP,
    write_kind: Optional[str] = None,
    sandbox_roots: Optional[SandboxOutputRoots] = None,
) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _assert_allowed(dst, write_kind, sandbox_roots)
    lock_path = _lock_path_for(dst)
    lock_fd = _acquire_lock(lock_path)
    if lock_fd is None:
        raise TimeoutError(f"Could not acquire parquet lock: {lock_path}")
    last_exc: Optional[Exception] = None
    try:
        for attempt in range(int(retries)):
            tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
            _assert_allowed(tmp, f"{write_kind} temp" if write_kind else None, sandbox_roots)
            try:
                out_df = df
                if dst.exists():
                    try:
                        existing = pd.read_parquet(dst)
                        out_df = pd.concat([existing, df], ignore_index=True)
                        if {"asset", "ts", "band"}.issubset(out_df.columns):
                            out_df = out_df.drop_duplicates(subset=["asset", "ts", "band"], keep="last")
                    except Exception:
                        out_df = df
                out_df.to_parquet(
                    tmp,
                    engine="pyarrow",
                    compression=compression,
                    index=False,
                    row_group_size=row_group_size,
                )
                atomic_replace(tmp, dst)
                return
            except PermissionError as exc:
                last_exc = exc
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                if attempt == int(retries) - 1:
                    break
                time.sleep(float(sleep_base_sec) * (attempt + 1))
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


def definition_paths(definition_root: Path, asset: str, band: str, category: str) -> Tuple[Path, Path]:
    stem = "__".join(
        (
            safe_path_part(asset),
            safe_path_part(band),
            safe_path_part(category),
        )
    )
    root = Path(definition_root)
    return root / f"{stem}.pkl", root / f"{stem}.json"


def load_definition(
    definition_root: Path,
    asset: str,
    band: str,
    category: str,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[dict[str, Any]]:
    model_path, meta_path = definition_paths(definition_root, asset, band, category)
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
        if log_fn is not None:
            log_fn(f"[definition][warn] failed loading asset={asset} band={band} category={category}: {exc}")
        return None


def save_definition(
    definition_root: Path,
    asset: str,
    band: str,
    category: str,
    model_obj: dict[str, Any],
    meta: dict[str, Any],
    *,
    write_kind: Optional[str] = None,
    sandbox_roots: Optional[SandboxOutputRoots] = None,
) -> None:
    model_path, meta_path = definition_paths(definition_root, asset, band, category)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_allowed(model_path, write_kind, sandbox_roots)
    tmp = sibling_temp_path(model_path, suffix=".pkl.tmp")
    _assert_allowed(tmp, f"{write_kind} temp" if write_kind else None, sandbox_roots)
    try:
        with tmp.open("wb") as f:
            pickle.dump(model_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_replace(tmp, model_path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    write_json(meta_path, meta, write_kind=write_kind, sandbox_roots=sandbox_roots)
