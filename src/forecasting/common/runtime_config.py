from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


RUNTIME_CONFIG_PATH = Path(__file__).with_name("pipeline_runtime.json")


def _safe_default_workers() -> int:
    return max(1, (os.cpu_count() or 4) // 2)


def _derived_default(module_name: str, key: str) -> int:
    cpu = os.cpu_count() or 4
    if key in {"writer_workers", "commit_workers"}:
        return 1
    if key == "model_threads":
        return max(1, min(8, cpu // 2))
    if key in {"asset_workers", "ingest_workers", "scan_workers", "score_workers", "derive_workers", "compute_workers", "unit_workers"}:
        return _safe_default_workers()
    return 1


def load_runtime_config() -> Dict[str, Any]:
    try:
        raw = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {"modules": {}}


def _worker_env_names(module_name: str, key: str) -> tuple[str, ...]:
    module_env = "".join(ch if ch.isalnum() else "_" for ch in str(module_name).upper())
    key_env = "".join(ch if ch.isalnum() else "_" for ch in str(key).upper())
    return (f"PIPELINE_{module_env}_{key_env}", f"{module_env}_{key_env}")


def resolve_worker_setting(
    module_name: str,
    key: str,
    fallback: Optional[int] = None,
    *,
    env_names: Optional[tuple[str, ...]] = None,
) -> Dict[str, Any]:
    for env_name in env_names or _worker_env_names(module_name, key):
        raw_env = os.getenv(env_name)
        if raw_env is None or not str(raw_env).strip():
            continue
        try:
            value = max(1, int(raw_env))
        except Exception:
            continue
        return {
            "value": int(value),
            "source": "env",
            "source_detail": env_name,
            "runtime_config_path": str(RUNTIME_CONFIG_PATH),
        }

    cfg = load_runtime_config()
    modules = cfg.get("modules", {}) if isinstance(cfg, dict) else {}
    module_cfg = modules.get(module_name, {}) if isinstance(modules, dict) else {}
    val = module_cfg.get(key) if isinstance(module_cfg, dict) else None
    if val is not None:
        try:
            return {
                "value": max(1, int(val)),
                "source": "active config",
                "source_detail": str(RUNTIME_CONFIG_PATH),
                "runtime_config_path": str(RUNTIME_CONFIG_PATH),
            }
        except Exception:
            pass

    default_val = fallback if fallback is not None else _derived_default(module_name, key)
    try:
        out = int(default_val)
    except Exception:
        out = _derived_default(module_name, key)
    return {
        "value": max(1, int(out)),
        "source": "derived default",
        "source_detail": ("fallback" if fallback is not None else "cpu-derived"),
        "runtime_config_path": str(RUNTIME_CONFIG_PATH),
    }


def get_workers(module_name: str, key: str, fallback: Optional[int] = None) -> int:
    return int(resolve_worker_setting(module_name, key, fallback=fallback)["value"])


def get_model_threads(module_name: str, fallback: Optional[int] = None) -> int:
    return get_workers(module_name, "model_threads", fallback=fallback)


def cap_model_threads(*, workers: int, model_threads: int, max_logical_threads: int = 32) -> int:
    w = max(1, int(workers))
    mt = max(1, int(model_threads))
    cap = max(1, int(max_logical_threads) // w)
    return max(1, min(mt, cap))


@dataclass
class DispatchPressureGuard:
    module_name: str
    sample_interval_seconds: float = 5.0
    cpu_enter_pct: float = 92.0
    cpu_exit_pct: float = 80.0
    cpu_enter_samples: int = 3
    cpu_exit_samples: int = 3
    ram_enter_pct: float = 90.0
    ram_exit_pct: float = 85.0
    ram_enter_samples: int = 2
    ram_exit_samples: int = 3
    log_fn: Optional[Callable[[str], None]] = None
    throttled: bool = False
    last_sample_monotonic: float = field(default_factory=lambda: 0.0)
    cpu_high_streak: int = 0
    cpu_low_streak: int = 0
    ram_high_streak: int = 0
    ram_low_streak: int = 0

    def __post_init__(self) -> None:
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def _log(self, message: str) -> None:
        if self.log_fn is not None:
            self.log_fn(message)

    def seconds_until_next_sample(self) -> float:
        if self.sample_interval_seconds <= 0:
            return 0.0
        if self.last_sample_monotonic <= 0:
            return 0.0
        elapsed = time.monotonic() - float(self.last_sample_monotonic)
        return max(0.0, float(self.sample_interval_seconds) - float(elapsed))

    def _sample(self) -> Optional[Dict[str, float]]:
        if psutil is None:
            return None
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
            ram_percent = float(psutil.virtual_memory().percent)
            return {"cpu_percent": cpu_percent, "ram_percent": ram_percent}
        except Exception:
            return None

    def refresh_if_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (not force) and self.last_sample_monotonic > 0 and (now - float(self.last_sample_monotonic)) < float(self.sample_interval_seconds):
            return
        sample = self._sample()
        self.last_sample_monotonic = now
        if sample is None:
            return
        cpu_percent = float(sample["cpu_percent"])
        ram_percent = float(sample["ram_percent"])

        self.cpu_high_streak = self.cpu_high_streak + 1 if cpu_percent > float(self.cpu_enter_pct) else 0
        self.ram_high_streak = self.ram_high_streak + 1 if ram_percent > float(self.ram_enter_pct) else 0
        self.cpu_low_streak = self.cpu_low_streak + 1 if cpu_percent < float(self.cpu_exit_pct) else 0
        self.ram_low_streak = self.ram_low_streak + 1 if ram_percent < float(self.ram_exit_pct) else 0

        if not self.throttled:
            enter_reason: Optional[str] = None
            if self.cpu_high_streak >= int(self.cpu_enter_samples):
                enter_reason = "cpu dwell exceeded"
            elif self.ram_high_streak >= int(self.ram_enter_samples):
                enter_reason = "ram dwell exceeded"
            if enter_reason is not None:
                self.throttled = True
                self._log(f"pressure_guard: ENTER throttled ({enter_reason})")
        else:
            if self.cpu_low_streak >= int(self.cpu_exit_samples) and self.ram_low_streak >= int(self.ram_exit_samples):
                self.throttled = False
                self._log("pressure_guard: EXIT throttled (pressure normalized)")

    def should_admit_new_work(self, *, force_sample: bool = False) -> bool:
        self.refresh_if_due(force=force_sample)
        return not bool(self.throttled)


def _memory_summary() -> str:
    if psutil is None:
        return "mem=n/a"
    try:
        vm = psutil.virtual_memory()
        total_gb = float(vm.total) / (1024 ** 3)
        avail_gb = float(vm.available) / (1024 ** 3)
        return f"mem_total_gb={total_gb:.1f}, mem_avail_gb={avail_gb:.1f}, mem_used_pct={vm.percent:.1f}"
    except Exception:
        return "mem=n/a"


def log_resolved_runtime(module_name: str, resolved: Optional[Dict[str, Any]] = None) -> None:
    cfg = load_runtime_config()
    modules = cfg.get("modules", {}) if isinstance(cfg, dict) else {}
    module_cfg = modules.get(module_name, {}) if isinstance(modules, dict) else {}
    cpu_count = os.cpu_count() or 1
    mem_text = _memory_summary()
    if isinstance(module_cfg, dict) and module_cfg:
        items = ", ".join(f"{k}={module_cfg[k]}" for k in sorted(module_cfg))
        extra = ""
        if isinstance(resolved, dict) and resolved:
            resolved_items = ", ".join(f"{k}={resolved[k]}" for k in sorted(resolved))
            extra = f"; resolved: {resolved_items}"
        print(f"[runtime:{module_name}] config_path={RUNTIME_CONFIG_PATH}; {items}{extra}; cpu_count={cpu_count}; {mem_text}")
    else:
        extra = ""
        if isinstance(resolved, dict) and resolved:
            resolved_items = ", ".join(f"{k}={resolved[k]}" for k in sorted(resolved))
            extra = f"; resolved: {resolved_items}"
        print(f"[runtime:{module_name}] config_path={RUNTIME_CONFIG_PATH}; using derived defaults{extra}; cpu_count={cpu_count}; {mem_text}")
