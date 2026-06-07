from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class RegimeFrameSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    metadata: dict[str, Any]


def split_train_score_by_rows(
    frame: pd.DataFrame,
    split_policy: Mapping[str, Any],
    *,
    train_rows_key: str = "train_rows",
    default_train_fraction: float = 2.0 / 3.0,
    min_train_rows: int = 2,
    min_total_rows: int | None = None,
    min_total_rows_error: str = "Regime study split requires more rows",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = int(len(frame))
    if min_total_rows is not None and row_count < int(min_total_rows):
        raise ValueError(str(min_total_rows_error))
    if split_policy.get(train_rows_key) is not None:
        train_rows = int(split_policy[train_rows_key])
    else:
        train_rows = int(row_count * float(split_policy.get("train_fraction", default_train_fraction)))
    train_rows = max(int(min_train_rows), min(row_count - 1, train_rows))
    return frame.iloc[:train_rows].copy(), frame.iloc[train_rows:].copy()


def split_panel_by_timestamp_train_rows(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_timestamps = list(dict.fromkeys(frame[timestamp_column].tolist()))
    rows_per_timestamp = int(frame.groupby(timestamp_column).size().median())
    train_timestamp_count = max(1, min(len(unique_timestamps) - 1, int(train_rows) // max(rows_per_timestamp, 1)))
    train_timestamps = set(unique_timestamps[:train_timestamp_count])
    train = frame[frame[timestamp_column].isin(train_timestamps)].copy()
    score = frame[~frame[timestamp_column].isin(train_timestamps)].copy()
    return train, score


def split_train_validation_frame(
    frame: pd.DataFrame,
    split_policy: Mapping[str, Any],
) -> RegimeFrameSplit:
    if frame.empty:
        raise ValueError("Regime single-trial runner requires a non-empty feature frame")
    policy = dict(split_policy)
    if "train_row_count" in policy:
        train_rows = int(policy["train_row_count"])
    elif "train_fraction" in policy:
        fraction = float(policy["train_fraction"])
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("Regime split_policy.train_fraction must be between 0 and 1")
        train_rows = max(1, min(len(frame) - 1, int(math.floor(len(frame) * fraction))))
    elif {"train_start_ts", "train_end_ts"}.issubset(policy):
        ts_col = str(policy.get("timestamp_column", "ts"))
        if ts_col not in frame.columns:
            raise ValueError(f"Regime split timestamp column {ts_col!r} not found")
        ts = pd.to_numeric(frame[ts_col], errors="coerce")
        train_mask = (ts >= int(policy["train_start_ts"])) & (ts <= int(policy["train_end_ts"]))
        validation_start = policy.get("validation_start_ts")
        validation_end = policy.get("validation_end_ts")
        if validation_start is not None and validation_end is not None:
            validation_mask = (ts >= int(validation_start)) & (ts <= int(validation_end))
        else:
            validation_mask = ~train_mask
        train = frame.loc[train_mask].copy()
        validation = frame.loc[validation_mask].copy()
        if train.empty:
            raise ValueError("Regime split produced an empty train frame")
        return RegimeFrameSplit(
            train=train,
            validation=validation,
            metadata={
                "split_policy": "timestamp_window",
                "timestamp_column": ts_col,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
            },
        )
    else:
        raise ValueError("Unsupported Regime split_policy")
    if train_rows <= 0:
        raise ValueError("Regime train row count must be positive")
    if train_rows >= len(frame):
        raise ValueError("Regime train row count must leave at least one validation row")
    train = frame.iloc[:train_rows].copy()
    validation = frame.iloc[train_rows:].copy()
    return RegimeFrameSplit(
        train=train,
        validation=validation,
        metadata={
            "split_policy": "row_order",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
        },
    )


def three_way_fraction_bounds(
    row_count: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    holdout_fraction: float,
) -> tuple[int, int]:
    n = int(row_count)
    total = float(train_fraction) + float(validation_fraction) + float(holdout_fraction)
    train_n = int(math.floor(n * (float(train_fraction) / total)))
    validation_n = int(math.floor(n * (float(validation_fraction) / total)))
    if train_n >= n and n > 0:
        train_n = n - 1
    validation_start = train_n
    validation_end = min(n, validation_start + validation_n)
    return int(validation_start), int(validation_end)


def window_values(
    values: Sequence[int],
    start_ts: int | None,
    end_ts: int | None,
    *,
    end_inclusive: bool = False,
) -> tuple[int, ...]:
    if start_ts is None or end_ts is None:
        return ()
    if end_inclusive:
        return tuple(int(value) for value in values if int(start_ts) <= int(value) <= int(end_ts))
    return tuple(int(value) for value in values if int(start_ts) <= int(value) < int(end_ts))
