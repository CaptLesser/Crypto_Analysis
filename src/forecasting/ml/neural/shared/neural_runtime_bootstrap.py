from __future__ import annotations

import os
from typing import Dict

from src.forecasting.common.runtime_config import get_model_threads

NEURAL_NUMERIC_MODEL_THREADS_ENV = "NEURAL_NUMERIC_MODEL_THREADS"
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def resolve_neural_model_threads() -> int:
    raw = os.getenv(NEURAL_NUMERIC_MODEL_THREADS_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            pass
    return get_model_threads("neural_numeric_runner")


def configure_neural_thread_env() -> int:
    threads = resolve_neural_model_threads()
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = str(threads)
    return threads


def neural_thread_env(threads: int) -> Dict[str, str]:
    resolved = max(1, int(threads))
    env = {NEURAL_NUMERIC_MODEL_THREADS_ENV: str(resolved)}
    for key in _THREAD_ENV_KEYS:
        env[key] = str(resolved)
    return env
