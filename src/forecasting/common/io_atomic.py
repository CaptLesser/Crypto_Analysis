from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional


DEFAULT_REPLACE_ATTEMPTS = int(os.getenv("PIPELINE_IO_REPLACE_ATTEMPTS", "16"))
DEFAULT_REPLACE_BACKOFF_SEC = float(os.getenv("PIPELINE_IO_REPLACE_BACKOFF_SEC", "0.1"))

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: Dict[str, threading.Lock] = {}


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve()).lower()
    with _LOCKS_GUARD:
        lk = _PATH_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _PATH_LOCKS[key] = lk
        return lk


def sibling_temp_path(path: Path, *, suffix: str = ".tmp") -> Path:
    """Return a unique temporary sibling path for an atomic write target."""
    token = f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    return path.with_name(f"{path.name}.{token}{suffix}")


def atomic_replace(
    src: Path,
    dst: Path,
    *,
    attempts: Optional[int] = None,
    backoff_sec: Optional[float] = None,
) -> None:
    """Retrying os.replace with per-destination in-process serialization.

    This mitigates transient Windows file-lock contention from AV/indexing/readers.
    """
    n = max(1, int(DEFAULT_REPLACE_ATTEMPTS if attempts is None else attempts))
    backoff = max(0.01, float(DEFAULT_REPLACE_BACKOFF_SEC if backoff_sec is None else backoff_sec))
    last_err: Optional[Exception] = None
    with _path_lock(dst):
        for i in range(1, n + 1):
            try:
                os.replace(src, dst)
                return
            except (PermissionError, OSError) as e:
                last_err = e
                if i >= n:
                    break
                time.sleep(backoff)
                backoff *= 1.8
    if last_err is not None:
        raise last_err
