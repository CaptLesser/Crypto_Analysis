from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.regimes.contracts import REGIME_AXIS_ORDER, RegimeAxisTarget, axis_target


def axis_label_id_column(axis: str) -> str:
    return f"{str(axis)}_label_id"


def axis_label_available_column(axis: str) -> str:
    return f"{str(axis)}_label_available"


@dataclass(frozen=True)
class FutureAxisTargets:
    label_ids: np.ndarray
    available: np.ndarray
    tail_pending: np.ndarray


def append_axis_label_columns(
    frame: pd.DataFrame,
    axes: Sequence[str] = REGIME_AXIS_ORDER,
) -> pd.DataFrame:
    out = frame.copy()
    for axis in axes:
        target = axis_target(str(axis))
        label_col = target.label_column
        if label_col in out.columns:
            normalized = out[label_col].map(
                lambda value: target.unknown_label if pd.isna(value) else target.normalize_label(value, allow_unknown=True)
            )
        else:
            normalized = pd.Series([target.unknown_label] * len(out), index=out.index, dtype="string")
        available = normalized.astype(str) != target.unknown_label
        out[axis_label_available_column(target.axis)] = available.astype(bool)
        out[axis_label_id_column(target.axis)] = normalized.map(target.label_id).astype("int64")
    return out


def future_axis_targets(
    frame: pd.DataFrame,
    target: RegimeAxisTarget | str,
    horizon_bars: int,
) -> FutureAxisTargets:
    resolved = axis_target(target) if isinstance(target, str) else target
    h = int(horizon_bars)
    if h <= 0:
        raise ValueError(f"horizon_bars must be positive, got {h}")

    n = len(frame)
    current_ids = (
        pd.to_numeric(frame.get(axis_label_id_column(resolved.axis)), errors="coerce")
        .fillna(resolved.unknown_id)
        .to_numpy(dtype=np.int64)
    )
    current_available = (
        pd.Series(frame.get(axis_label_available_column(resolved.axis), False), index=frame.index)
        .fillna(False)
        .astype(bool)
        .to_numpy(dtype=bool)
    )

    future_ids = np.full((n,), int(resolved.unknown_id), dtype=np.int64)
    available = np.zeros((n,), dtype=bool)
    tail_pending = np.zeros((n,), dtype=bool)
    for i in range(n):
        j = i + h
        if j >= n:
            tail_pending[i] = True
            continue
        future_ids[i] = int(current_ids[j])
        available[i] = bool(current_available[j])
    return FutureAxisTargets(label_ids=future_ids, available=available, tail_pending=tail_pending)
