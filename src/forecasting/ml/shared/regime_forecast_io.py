from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.regimes.contracts import REGIME_AXES, REGIME_AXIS_ORDER, band_for_ceiling, regime_table_dir


class RegimeLabelReadError(RuntimeError):
    pass


@dataclass
class RegimeLabelReadStats:
    month_partitions_checked: int = 0
    missing_month_partitions: int = 0
    files_discovered: int = 0
    files_read: int = 0
    unreadable_files: int = 0
    raw_rows_loaded: int = 0
    rows_after_filter: int = 0
    duplicate_rows_dropped: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "month_partitions_checked": int(self.month_partitions_checked),
            "missing_month_partitions": int(self.missing_month_partitions),
            "files_discovered": int(self.files_discovered),
            "files_read": int(self.files_read),
            "unreadable_files": int(self.unreadable_files),
            "raw_rows_loaded": int(self.raw_rows_loaded),
            "rows_after_filter": int(self.rows_after_filter),
            "duplicate_rows_dropped": int(self.duplicate_rows_dropped),
        }


def _next_month(year: int, month: int) -> Tuple[int, int]:
    return (int(year) + 1, 1) if int(month) == 12 else (int(year), int(month) + 1)


def _month_start_ts(year: int, month: int) -> int:
    return int(datetime(int(year), int(month), 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def iter_months_between(start_ts: int, end_ts: int) -> Iterable[Tuple[int, int]]:
    if int(end_ts) < int(start_ts):
        return
    dt = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
    year, month = dt.year, dt.month
    while True:
        yield int(year), int(month)
        year, month = _next_month(year, month)
        if _month_start_ts(year, month) > int(end_ts):
            break


def _empty_frame(columns: Optional[Sequence[str]]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns) if columns else ["ts", "asset"])


def _requested_required_columns(columns: Optional[Sequence[str]]) -> Optional[List[str]]:
    if columns is None:
        return None
    requested = [str(c) for c in columns]
    required = ["ts", "asset"]
    for col in requested:
        if col not in required:
            required.append(col)
    return required


def _month_dirs(
    *,
    base_dir: Path,
    table_dir: str,
    asset: str,
    year: int,
    month: int,
    allow_legacy_unpartitioned: bool,
) -> List[Path]:
    dirs = [Path(base_dir) / table_dir / f"asset={str(asset)}" / f"year={int(year)}" / f"month={int(month):02d}"]
    if allow_legacy_unpartitioned:
        dirs.append(Path(base_dir) / table_dir / f"year={int(year)}" / f"month={int(month):02d}")
    return dirs


def _validate_regime_label_schema(
    df: pd.DataFrame,
    *,
    asset: str,
    ceiling_interval: int,
    path: Path,
) -> None:
    missing = [col for col in ("ts", "asset") if col not in df.columns]
    if missing:
        raise RegimeLabelReadError(f"Regime label parquet missing key columns {missing}: {path}")
    if "ceiling_interval_min" in df.columns:
        bad_ceiling = pd.to_numeric(df["ceiling_interval_min"], errors="coerce").dropna()
        if not bad_ceiling.empty and not bool((bad_ceiling.astype("int64") == int(ceiling_interval)).all()):
            raise RegimeLabelReadError(f"Regime label ceiling mismatch for regimes_{int(ceiling_interval)}: {path}")
    if "band" in df.columns:
        expected_band = band_for_ceiling(int(ceiling_interval)).name
        bad_band = df["band"].dropna().astype(str)
        if not bad_band.empty and not bool((bad_band == expected_band).all()):
            raise RegimeLabelReadError(f"Regime label band mismatch for {expected_band}: {path}")
    asset_values = df["asset"].dropna().astype(str)
    if not asset_values.empty and not bool((asset_values == str(asset)).all()):
        raise RegimeLabelReadError(f"Regime label asset mismatch for asset={asset}: {path}")
    for axis in REGIME_AXIS_ORDER:
        label_col = REGIME_AXES[axis].label_column
        if label_col not in df.columns:
            continue
        allowed = set(REGIME_AXES[axis].labels) | {"unknown"}
        values = df[label_col].dropna().astype(str).str.lower().str.strip()
        bad = sorted(set(values) - allowed)
        if bad:
            raise RegimeLabelReadError(f"Regime label column {label_col} has unsupported values {bad}: {path}")
        for pct_col in (REGIME_AXES[axis].confidence_column, REGIME_AXES[axis].intensity_column):
            if pct_col not in df.columns:
                continue
            pct = pd.to_numeric(df[pct_col], errors="coerce").dropna()
            if not pct.empty and not bool(((pct >= 0) & (pct <= 100)).all()):
                raise RegimeLabelReadError(f"Regime label column {pct_col} outside [0, 100]: {path}")


def read_regime_labels(
    *,
    base_dir: Path,
    ceiling_interval: int,
    start_ts: int,
    end_ts: int,
    asset: str,
    columns: Optional[Sequence[str]] = None,
    allow_legacy_unpartitioned: bool = False,
    validate_schema: bool = True,
    stats: Optional[RegimeLabelReadStats] = None,
) -> pd.DataFrame:
    table_dir = regime_table_dir(int(ceiling_interval))
    read_columns = _requested_required_columns(columns)
    local_stats = stats if stats is not None else RegimeLabelReadStats()
    frames: List[pd.DataFrame] = []

    for year, month in iter_months_between(int(start_ts), int(end_ts)):
        month_files: List[Path] = []
        for month_dir in _month_dirs(
            base_dir=Path(base_dir),
            table_dir=table_dir,
            asset=str(asset),
            year=int(year),
            month=int(month),
            allow_legacy_unpartitioned=bool(allow_legacy_unpartitioned),
        ):
            local_stats.month_partitions_checked += 1
            if month_dir.exists():
                month_files.extend(sorted(month_dir.glob("*.parquet")))
            else:
                local_stats.missing_month_partitions += 1
        local_stats.files_discovered += len(month_files)
        for path in month_files:
            try:
                df = pd.read_parquet(path, columns=read_columns)
            except Exception:
                local_stats.unreadable_files += 1
                continue
            local_stats.files_read += 1
            local_stats.raw_rows_loaded += int(len(df))
            if validate_schema:
                _validate_regime_label_schema(df, asset=str(asset), ceiling_interval=int(ceiling_interval), path=path)
            if "asset" not in df.columns or "ts" not in df.columns:
                continue
            ts_num = pd.to_numeric(df["ts"], errors="coerce")
            scoped = df[
                (df["asset"].astype(str) == str(asset))
                & ts_num.notna()
                & ts_num.ge(int(start_ts))
                & ts_num.le(int(end_ts))
            ].copy()
            if not scoped.empty:
                scoped["ts"] = pd.to_numeric(scoped["ts"], errors="coerce").astype("int64")
                frames.append(scoped)

    if not frames:
        return _empty_frame(columns)

    out = pd.concat(frames, ignore_index=True)
    before = int(len(out))
    out = out.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last").reset_index(drop=True)
    local_stats.rows_after_filter += int(len(out))
    local_stats.duplicate_rows_dropped += max(0, before - int(len(out)))
    return out
