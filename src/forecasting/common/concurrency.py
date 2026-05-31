from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

from src.forecasting.common.runtime_config import cap_model_threads, resolve_worker_setting


CONCURRENCY_SCHEMA_VERSION = 1

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

PROFILE_NAMES = ("outer_parallel", "balanced", "inner_parallel", "benchmark_only")

_THREADPOOL_LIMIT_HANDLES: list[Any] = []
_LOGGED_KEYS: set[tuple[str, str, str]] = set()


def _clean_profile_name(profile: str) -> str:
    name = str(profile or "").strip().lower()
    if name not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILE_NAMES)}")
    return name


def _safe_int(value: Any, fallback: int = 1) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(fallback))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class ConcurrencyProfile:
    module_name: str
    profile_name: str
    requested_workers: int
    effective_workers: int
    requested_model_threads: int
    effective_model_threads: int
    requested_helper_threads: int
    effective_helper_threads: int
    max_logical_threads: int = 32
    worker_source: str = ""
    worker_source_detail: str = ""
    model_threads_source: str = ""
    model_threads_source_detail: str = ""
    pyarrow_cpu_threads: Optional[int] = None
    pyarrow_io_threads: Optional[int] = None
    numba_threads: Optional[int] = None
    set_thread_env: bool = True
    use_threadpoolctl: bool = True
    apply_numba: bool = True
    apply_pyarrow: bool = True
    notes: tuple[str, ...] = ()

    @property
    def effective_thread_product(self) -> int:
        return int(self.effective_workers) * int(self.effective_model_threads)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreadCapSnapshot:
    schema_version: int
    module_name: str
    profile_name: str
    requested: Dict[str, Any]
    effective: Dict[str, Any]
    env: Dict[str, Optional[str]]
    status: Dict[str, Any]
    effective_snapshot: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_concurrency_profile(
    module_name: str,
    *,
    profile: str = "balanced",
    worker_key: str = "unit_workers",
    model_threads_key: str = "model_threads",
    requested_workers: Optional[int] = None,
    requested_model_threads: Optional[int] = None,
    fallback_workers: Optional[int] = None,
    fallback_model_threads: Optional[int] = None,
    max_logical_threads: int = 32,
    helper_threads: Optional[int] = None,
    pyarrow_cpu_threads: Optional[int] = None,
    pyarrow_io_threads: Optional[int] = None,
    numba_threads: Optional[int] = None,
) -> ConcurrencyProfile:
    """Resolve requested and effective concurrency settings without changing process state."""
    profile_name = _clean_profile_name(profile)
    worker_source = "explicit"
    worker_source_detail = "requested_workers"
    model_source = "explicit"
    model_source_detail = "requested_model_threads"

    if requested_workers is None:
        resolved_workers = resolve_worker_setting(module_name, worker_key, fallback=fallback_workers)
        requested_workers = int(resolved_workers["value"])
        worker_source = str(resolved_workers.get("source", ""))
        worker_source_detail = str(resolved_workers.get("source_detail", ""))
    if requested_model_threads is None:
        resolved_model_threads = resolve_worker_setting(module_name, model_threads_key, fallback=fallback_model_threads)
        requested_model_threads = int(resolved_model_threads["value"])
        model_source = str(resolved_model_threads.get("source", ""))
        model_source_detail = str(resolved_model_threads.get("source_detail", ""))

    req_workers = _safe_int(requested_workers)
    req_model_threads = _safe_int(requested_model_threads)
    req_helper_threads = _safe_int(helper_threads if helper_threads is not None else req_model_threads)
    max_threads = _safe_int(max_logical_threads, 32)
    notes: list[str] = []

    if profile_name == "outer_parallel":
        effective_workers = req_workers
        effective_model_threads = 1
        effective_helper_threads = 1
        notes.append("outer_parallel caps model/native/helper threads to 1")
    elif profile_name == "balanced":
        effective_workers = req_workers
        effective_model_threads = cap_model_threads(
            workers=req_workers,
            model_threads=req_model_threads,
            max_logical_threads=max_threads,
        )
        effective_helper_threads = cap_model_threads(
            workers=req_workers,
            model_threads=req_helper_threads,
            max_logical_threads=max_threads,
        )
        if effective_model_threads != req_model_threads or effective_helper_threads != req_helper_threads:
            notes.append("balanced capped inner/helper threads to max_logical_threads budget")
    elif profile_name == "inner_parallel":
        effective_workers = max(1, min(req_workers, max(1, max_threads // max(1, req_model_threads))))
        effective_model_threads = req_model_threads
        effective_helper_threads = req_helper_threads
        if effective_workers != req_workers:
            notes.append("inner_parallel reduced outer workers to preserve requested inner threads")
    else:
        effective_workers = req_workers
        effective_model_threads = req_model_threads
        effective_helper_threads = req_helper_threads
        notes.append("benchmark_only preserves requested stress settings")

    helper_cap = effective_helper_threads
    return ConcurrencyProfile(
        module_name=str(module_name),
        profile_name=profile_name,
        requested_workers=req_workers,
        effective_workers=max(1, int(effective_workers)),
        requested_model_threads=req_model_threads,
        effective_model_threads=max(1, int(effective_model_threads)),
        requested_helper_threads=req_helper_threads,
        effective_helper_threads=max(1, int(helper_cap)),
        max_logical_threads=max_threads,
        worker_source=worker_source,
        worker_source_detail=worker_source_detail,
        model_threads_source=model_source,
        model_threads_source_detail=model_source_detail,
        pyarrow_cpu_threads=_safe_int(pyarrow_cpu_threads, helper_cap) if pyarrow_cpu_threads is not None else helper_cap,
        pyarrow_io_threads=_safe_int(pyarrow_io_threads, helper_cap) if pyarrow_io_threads is not None else helper_cap,
        numba_threads=_safe_int(numba_threads, helper_cap) if numba_threads is not None else helper_cap,
        notes=tuple(notes),
    )


def _optional_module_status(name: str, importer: Callable[[str], Any]) -> tuple[Optional[Any], Dict[str, Any]]:
    try:
        module = importer(name)
    except Exception as exc:
        return None, {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return module, {"available": True, "version": getattr(module, "__version__", None)}


def apply_thread_caps(
    profile: ConcurrencyProfile,
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    importer: Callable[[str], Any] = import_module,
    include_effective_snapshot: bool = True,
) -> ThreadCapSnapshot:
    """Apply native/helper thread caps for the current parent or worker process."""
    env = os.environ if environ is None else environ
    status: Dict[str, Any] = {"env": {}, "threadpoolctl": {}, "numba": {}, "pyarrow": {}}
    helper_threads = _safe_int(profile.effective_helper_threads)

    if profile.set_thread_env:
        for name in THREAD_ENV_VARS:
            env[name] = str(helper_threads)
            status["env"][name] = {"status": "set", "value": str(helper_threads)}
    else:
        for name in THREAD_ENV_VARS:
            status["env"][name] = {"status": "skipped", "value": env.get(name)}

    if profile.use_threadpoolctl:
        threadpoolctl, module_status = _optional_module_status("threadpoolctl", importer)
        status["threadpoolctl"].update(module_status)
        if threadpoolctl is not None:
            try:
                limiter = threadpoolctl.threadpool_limits(limits=helper_threads)
                enter = getattr(limiter, "__enter__", None)
                if callable(enter):
                    enter()
                    _THREADPOOL_LIMIT_HANDLES.append(limiter)
                status["threadpoolctl"].update({"status": "applied", "limits": helper_threads})
            except Exception as exc:
                status["threadpoolctl"].update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    else:
        status["threadpoolctl"] = {"available": None, "status": "skipped"}

    if profile.apply_numba and profile.numba_threads is not None:
        numba, module_status = _optional_module_status("numba", importer)
        status["numba"].update(module_status)
        if numba is not None:
            try:
                numba.set_num_threads(int(profile.numba_threads))
                status["numba"].update({"status": "applied", "num_threads": int(profile.numba_threads)})
            except Exception as exc:
                status["numba"].update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    else:
        status["numba"] = {"available": None, "status": "skipped"}

    if profile.apply_pyarrow:
        pyarrow, module_status = _optional_module_status("pyarrow", importer)
        status["pyarrow"].update(module_status)
        if pyarrow is not None:
            applied: Dict[str, Any] = {}
            try:
                if profile.pyarrow_cpu_threads is not None:
                    pyarrow.set_cpu_count(int(profile.pyarrow_cpu_threads))
                    applied["cpu_threads"] = int(profile.pyarrow_cpu_threads)
                if profile.pyarrow_io_threads is not None:
                    pyarrow.set_io_thread_count(int(profile.pyarrow_io_threads))
                    applied["io_threads"] = int(profile.pyarrow_io_threads)
                status["pyarrow"].update({"status": "applied", **applied})
            except Exception as exc:
                status["pyarrow"].update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    else:
        status["pyarrow"] = {"available": None, "status": "skipped"}

    env_snapshot = {name: env.get(name) for name in THREAD_ENV_VARS}
    effective = {
        "workers": int(profile.effective_workers),
        "model_threads": int(profile.effective_model_threads),
        "helper_threads": helper_threads,
        "pyarrow_cpu_threads": profile.pyarrow_cpu_threads,
        "pyarrow_io_threads": profile.pyarrow_io_threads,
        "numba_threads": profile.numba_threads,
    }
    requested = {
        "workers": int(profile.requested_workers),
        "model_threads": int(profile.requested_model_threads),
        "helper_threads": int(profile.requested_helper_threads),
    }
    snapshot = (
        effective_thread_snapshot(environ=env, importer=importer)
        if include_effective_snapshot
        else {}
    )
    return ThreadCapSnapshot(
        schema_version=CONCURRENCY_SCHEMA_VERSION,
        module_name=profile.module_name,
        profile_name=profile.profile_name,
        requested=requested,
        effective=effective,
        env=env_snapshot,
        status=_json_safe(status),
        effective_snapshot=snapshot,
    )


def worker_thread_initializer(
    profile: ConcurrencyProfile,
    next_initializer: Optional[Callable[..., Any]] = None,
    *initargs: Any,
    importer: Callable[[str], Any] = import_module,
    **initkwargs: Any,
) -> ThreadCapSnapshot:
    """Process-pool initializer that applies caps before a module-specific initializer."""
    snapshot = apply_thread_caps(profile, importer=importer, include_effective_snapshot=False)
    if next_initializer is not None:
        next_initializer(*initargs, **initkwargs)
    return snapshot


def effective_thread_snapshot(
    *,
    environ: Optional[Mapping[str, str]] = None,
    importer: Callable[[str], Any] = import_module,
) -> Dict[str, Any]:
    env = os.environ if environ is None else environ
    snapshot: Dict[str, Any] = {
        "schema_version": CONCURRENCY_SCHEMA_VERSION,
        "env": {name: env.get(name) for name in THREAD_ENV_VARS},
        "modules": {},
    }

    threadpoolctl, status = _optional_module_status("threadpoolctl", importer)
    snapshot["modules"]["threadpoolctl"] = status
    if threadpoolctl is not None:
        try:
            snapshot["threadpool_info"] = _json_safe(threadpoolctl.threadpool_info())
        except Exception as exc:
            snapshot["threadpool_info_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    numba, status = _optional_module_status("numba", importer)
    snapshot["modules"]["numba"] = status
    if numba is not None:
        try:
            snapshot["numba_num_threads"] = int(numba.get_num_threads())
        except Exception as exc:
            snapshot["numba_threads_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    pyarrow, status = _optional_module_status("pyarrow", importer)
    snapshot["modules"]["pyarrow"] = status
    if pyarrow is not None:
        try:
            snapshot["pyarrow_cpu_count"] = int(pyarrow.cpu_count())
            snapshot["pyarrow_io_thread_count"] = int(pyarrow.io_thread_count())
        except Exception as exc:
            snapshot["pyarrow_threads_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    psutil, status = _optional_module_status("psutil", importer)
    snapshot["modules"]["psutil"] = status
    if psutil is not None:
        try:
            proc = psutil.Process()
            mem = proc.memory_info()
            snapshot["process"] = {
                "pid": int(proc.pid),
                "num_threads": int(proc.num_threads()),
                "rss_mb": round(float(mem.rss) / (1024.0 * 1024.0), 3),
            }
            try:
                full = proc.memory_full_info()
                uss = getattr(full, "uss", None)
                if uss is not None:
                    snapshot["process"]["uss_mb"] = round(float(uss) / (1024.0 * 1024.0), 3)
            except Exception:
                pass
        except Exception as exc:
            snapshot["process_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    return _json_safe(snapshot)


def log_concurrency_once(
    module_name: str,
    payload: Mapping[str, Any] | ConcurrencyProfile | ThreadCapSnapshot,
    *,
    scope: str = "parent",
    log_fn: Callable[[str], None] = print,
    key: Optional[str] = None,
) -> bool:
    """Emit a compact one-time concurrency log line for a module/scope/key."""
    log_key = (str(module_name), str(scope), str(key or "default"))
    if log_key in _LOGGED_KEYS:
        return False
    _LOGGED_KEYS.add(log_key)

    if isinstance(payload, ConcurrencyProfile):
        body = payload.as_dict()
    elif isinstance(payload, ThreadCapSnapshot):
        body = payload.as_dict()
    else:
        body = dict(payload)
    log_fn(f"[concurrency:{module_name}:{scope}] {json.dumps(_json_safe(body), sort_keys=True)}")
    return True


__all__ = [
    "CONCURRENCY_SCHEMA_VERSION",
    "PROFILE_NAMES",
    "THREAD_ENV_VARS",
    "ConcurrencyProfile",
    "ThreadCapSnapshot",
    "apply_thread_caps",
    "effective_thread_snapshot",
    "log_concurrency_once",
    "resolve_concurrency_profile",
    "worker_thread_initializer",
]
