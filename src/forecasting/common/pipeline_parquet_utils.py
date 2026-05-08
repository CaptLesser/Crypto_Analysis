from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class PipelineValidationError(RuntimeError):
    def __init__(self, message: str, first_bad_ts: Optional[int] = None):
        super().__init__(message)
        self.first_bad_ts = int(first_bad_ts) if first_bad_ts is not None else None


def find_partition_month_dir(base: Path, *, newest: bool) -> Optional[Path]:
    if not base.exists():
        return None
    years: list[tuple[int, Path]] = []
    for year_dir in base.glob("year=*"):
        y_txt = year_dir.name.replace("year=", "")
        if y_txt.isdigit():
            years.append((int(y_txt), year_dir))
    years.sort(key=lambda item: item[0], reverse=bool(newest))
    for _year, year_dir in years:
        months: list[tuple[int, Path]] = []
        for month_dir in year_dir.glob("month=*"):
            m_txt = month_dir.name.replace("month=", "")
            if not m_txt.isdigit():
                continue
            month_val = int(m_txt)
            if 1 <= month_val <= 12:
                months.append((month_val, month_dir))
        months.sort(key=lambda item: item[0], reverse=bool(newest))
        for _month, month_dir in months:
            if any(month_dir.glob("*.parquet")):
                return month_dir
    return None


def partition_max_ts(base: Path, *, ts_column: str = "ts") -> Optional[int]:
    latest_month_dir = find_partition_month_dir(base, newest=True)
    if latest_month_dir is None:
        return None
    try:
        latest_files = sorted(latest_month_dir.glob("*.parquet"), key=lambda path: path.name.lower())
        if not latest_files:
            return None
        frame = pd.read_parquet(latest_files[-1], columns=[str(ts_column)])
    except Exception:
        return None
    if str(ts_column) not in frame.columns:
        return None
    ts = pd.to_numeric(frame[str(ts_column)], errors="coerce").dropna().astype("int64")
    if ts.empty:
        return None
    return int(ts.max())


def decide_range_from_disk_edges(
    *,
    asset: str,
    interval_min: int,
    downstream_max_ts: Optional[int],
    upstream_min_ts: Optional[int],
    upstream_max_ts: Optional[int],
    mode: str = "incremental",
    backfill_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[int], Optional[int], str]:
    step = int(interval_min) * 60
    if upstream_max_ts is None:
        return None, None, "no_upstream"
    if downstream_max_ts is not None and int(downstream_max_ts) > int(upstream_max_ts):
        raise RuntimeError(
            f"[edge-corrupt] asset={asset} interval={interval_min} "
            f"downstream_max_ts={int(downstream_max_ts)} upstream_max_ts={int(upstream_max_ts)}"
        )
    if str(mode) == "backfill":
        if backfill_range is None:
            return None, None, "no_backfill_range"
        start_ts, end_ts = int(backfill_range[0]), int(backfill_range[1])
        if upstream_min_ts is not None:
            start_ts = max(int(start_ts), int(upstream_min_ts))
        end_ts = min(int(end_ts), int(upstream_max_ts))
        if end_ts < start_ts:
            return None, None, "empty_backfill"
        return int(start_ts), int(end_ts), "explicit_repair"
    if downstream_max_ts is None:
        if upstream_min_ts is None:
            return None, None, "no_upstream_head"
        return int(upstream_min_ts), int(upstream_max_ts), "first_run"
    if int(downstream_max_ts) >= int(upstream_max_ts):
        return None, None, "at_edge"
    start_ts = int(downstream_max_ts) + int(step)
    return int(start_ts), int(upstream_max_ts), "incremental"


def validate_strict_timegrid(ts: pd.Series, *, interval_min: int, context: str) -> None:
    if ts is None or ts.empty:
        return
    step = int(interval_min) * 60
    arr = pd.to_numeric(ts, errors="coerce").dropna().astype("int64").to_numpy(dtype=np.int64, copy=False)
    if arr.size == 0:
        return
    if arr.size != len(ts):
        raise PipelineValidationError(f"{context}: timestamp parse failure detected.")
    if not np.all(arr % max(step, 1) == 0):
        raise PipelineValidationError(f"{context}: timestamps are not aligned to {step}-second step.")
    if arr.size <= 1:
        return
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        raise PipelineValidationError(f"{context}: timestamps are not strictly increasing.")
    if np.all(diffs == step):
        return
    bad_idx = int(np.where(diffs != step)[0][0])
    prev_ts = int(arr[bad_idx])
    cur_ts = int(arr[bad_idx + 1])
    if cur_ts > prev_ts + step:
        missing_ts = int(prev_ts + step)
        raise PipelineValidationError(
            f"{context}: non-uniform timestamp step; expected {step}. "
            f"first_missing_ts={missing_ts} between prev_ts={prev_ts} and cur_ts={cur_ts}.",
            first_bad_ts=int(missing_ts),
        )
    raise PipelineValidationError(
        f"{context}: non-uniform timestamp step; expected {step}. prev_ts={prev_ts} cur_ts={cur_ts}.",
        first_bad_ts=int(cur_ts),
    )


def validate_expected_grid(
    ts_series: pd.Series,
    *,
    interval_min: int,
    expected_start: int,
    expected_end: int,
    context: str,
) -> None:
    step = int(interval_min) * 60
    arr = pd.to_numeric(ts_series, errors="coerce").dropna().astype("int64").to_numpy(dtype=np.int64, copy=False)
    expected_count = ((int(expected_end) - int(expected_start)) // max(step, 1)) + 1
    if arr.size != expected_count:
        raise PipelineValidationError(f"{context}: row_count={arr.size} expected={expected_count}", first_bad_ts=int(expected_start))
    if arr.size == 0:
        raise PipelineValidationError(f"{context}: no rows", first_bad_ts=int(expected_start))
    if int(arr[0]) != int(expected_start):
        raise PipelineValidationError(
            f"{context}: start_ts={int(arr[0])} expected_start={int(expected_start)}",
            first_bad_ts=int(expected_start),
        )
    if int(arr[-1]) != int(expected_end):
        raise PipelineValidationError(
            f"{context}: end_ts={int(arr[-1])} expected_end={int(expected_end)}",
            first_bad_ts=int(expected_end),
        )
    if arr.size <= 1:
        return
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        bad_idx = int(np.where(diffs <= 0)[0][0])
        raise PipelineValidationError(f"{context}: timestamps not strictly increasing", first_bad_ts=int(arr[bad_idx + 1]))
    if np.all(diffs == step):
        return
    bad_idx = int(np.where(diffs != step)[0][0])
    raise PipelineValidationError(
        f"{context}: non-uniform step expected={step}",
        first_bad_ts=int(arr[bad_idx] + step),
    )


def validate_no_nan_columns(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    context: str,
    ts_column: str = "ts",
) -> None:
    check_cols = [str(col) for col in columns if str(col) in frame.columns]
    if not check_cols:
        return
    bad_mask = frame.loc[:, check_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    if not bool(bad_mask.any()):
        return
    first_bad_idx = int(np.where(bad_mask.to_numpy(dtype=bool, copy=False))[0][0])
    first_bad_ts = int(pd.to_numeric(frame.iloc[first_bad_idx][str(ts_column)], errors="coerce"))
    raise PipelineValidationError(f"{context}: NaN detected in output rows", first_bad_ts=first_bad_ts)


def validate_time_partition_window(
    frame: pd.DataFrame,
    *,
    interval_min: int,
    expected_start: int,
    expected_end: int,
    required_value_cols: Sequence[str],
    context: str,
    ts_column: str = "ts",
) -> None:
    validate_no_nan_columns(frame, columns=required_value_cols, context=context, ts_column=ts_column)
    validate_expected_grid(
        frame[str(ts_column)],
        interval_min=int(interval_min),
        expected_start=int(expected_start),
        expected_end=int(expected_end),
        context=f"{context} grid",
    )
