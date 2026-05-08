from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np

DEFAULT_FLOAT_DTYPE = np.float32
FALLBACK_FLOAT_DTYPE = np.float64

T = TypeVar("T")


def as_default_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=DEFAULT_FLOAT_DTYPE)


def default_float_full(shape: Any, fill_value: float) -> np.ndarray:
    return np.full(shape, fill_value, dtype=DEFAULT_FLOAT_DTYPE)


def default_float_nan_full(shape: Any) -> np.ndarray:
    return np.full(shape, np.nan, dtype=DEFAULT_FLOAT_DTYPE)


def is_dtype_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (TypeError, ValueError)):
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                'dtype',
                'float32',
                'float64',
                'unsupported type',
                'unsupported dtype',
                'buffer dtype mismatch',
                'expected scalar type',
                'input type',
            )
        )
    return False


def run_with_float_dtype_retry(fn: Callable[[Any], T]) -> T:
    try:
        return fn(DEFAULT_FLOAT_DTYPE)
    except Exception as exc:  # noqa: BLE001
        if not is_dtype_retryable_error(exc):
            raise
        return fn(FALLBACK_FLOAT_DTYPE)
