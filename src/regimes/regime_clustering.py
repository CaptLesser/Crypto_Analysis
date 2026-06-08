from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import hashlib
import inspect
import time
import uuid
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.runtime_config import RUNTIME_CONFIG_PATH, get_workers, log_resolved_runtime
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, resolve_sandbox_output_roots
from src.forecasting.ml.shared.numeric_runner_diagnostics import resource_snapshot, worker_resource_telemetry_record
from src.regimes.core import artifacts as regime_artifacts
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

from src.features.scalar_features import PARQUET_ROOT, PARQUET_COMPRESSION, PARQUET_ROW_GROUP, feature_max_ts_from_parquet, log as base_log


def _patch_check_array_compat() -> None:
    """
    Compatibility shim for environments where sklearn renamed force_all_finite
    to ensure_all_finite, while hdbscan still calls force_all_finite.
    """
    try:
        from sklearn.utils import validation as sk_validation  # type: ignore
        import sklearn.utils as sk_utils  # type: ignore
    except Exception:
        return
    try:
        sig = inspect.signature(sk_validation.check_array)
    except Exception:
        return
    if "force_all_finite" in sig.parameters:
        return
    orig = sk_validation.check_array
    try:
        orig_sig = inspect.signature(orig)
    except Exception:
        orig_sig = None

    def check_array_compat(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and (orig_sig is None or "ensure_all_finite" in orig_sig.parameters):
            kwargs.setdefault("ensure_all_finite", force_all_finite)
        return orig(*args, **kwargs)

    sk_validation.check_array = check_array_compat  # type: ignore
    try:
        sk_utils.check_array = check_array_compat  # type: ignore
    except Exception:
        pass
    if hdbscan is not None:
        for mod_name in ("hdbscan_", "prediction", "flat"):
            try:
                mod = getattr(hdbscan, mod_name, None)
                if mod is not None and hasattr(mod, "check_array"):
                    setattr(mod, "check_array", check_array_compat)
            except Exception:
                continue


_patch_check_array_compat()


PIPELINE_PROFILE = selected_profile(default="production")
REGIME_THREAD_CAP_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
REGIME_PARALLEL_THREAD_CAP_RECOMMENDATION = "1"


@dataclass(frozen=True)
class RegimeLabelGenerationIOConfig:
    output_root: Path
    definition_root: Path
    diagnostic_root: Path
    log_root: Path
    tmp_root: Path
    sandbox_enabled: bool = False

    def to_json_ready(self) -> Dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "definition_root": str(self.definition_root),
            "diagnostic_root": str(self.diagnostic_root),
            "log_root": str(self.log_root),
            "tmp_root": str(self.tmp_root),
            "sandbox_enabled": bool(self.sandbox_enabled),
        }


def resolve_regime_label_generation_io_config(
    args: Optional[argparse.Namespace] = None,
    *,
    sandbox_roots: Optional[SandboxOutputRoots] = None,
) -> RegimeLabelGenerationIOConfig:
    roots = sandbox_roots if sandbox_roots is not None else resolve_sandbox_output_roots(args)
    if roots.enabled:
        return RegimeLabelGenerationIOConfig(
            output_root=roots.parquet_root,
            definition_root=roots.regime_definition_root,
            diagnostic_root=roots.diagnostics_root,
            log_root=roots.log_root,
            tmp_root=roots.tmp_root,
            sandbox_enabled=True,
        )
    return RegimeLabelGenerationIOConfig(
        output_root=Path(
            resolve_path("source_regime_root", profile=PIPELINE_PROFILE, required=False)
            or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
            or Path("parquet")
        ),
        definition_root=Path(
            resolve_path("regime_definition_root", profile=PIPELINE_PROFILE, required=False)
            or Path("regime_definitions")
        ),
        diagnostic_root=Path(resolve_path("log_root", profile=PIPELINE_PROFILE, required=False) or Path("logs")),
        log_root=Path(resolve_path("log_root", profile=PIPELINE_PROFILE, required=False) or Path("logs")),
        tmp_root=Path(resolve_path("tmp_root", profile=PIPELINE_PROFILE, required=False) or Path("tmp")),
        sandbox_enabled=False,
    )


def configure_regime_label_generation_io(
    args: Optional[argparse.Namespace] = None,
    *,
    sandbox_roots: Optional[SandboxOutputRoots] = None,
) -> RegimeLabelGenerationIOConfig:
    roots = sandbox_roots if sandbox_roots is not None else resolve_sandbox_output_roots(args)
    config = resolve_regime_label_generation_io_config(args, sandbox_roots=roots)
    global REGIME_LABEL_IO_CONFIG, REGIME_LABEL_SANDBOX_ROOTS, REGIME_PARQUET_ROOT, DEFINITION_ROOT, LOG_DIR, LOG_FILE
    REGIME_LABEL_IO_CONFIG = config
    REGIME_LABEL_SANDBOX_ROOTS = roots if roots.enabled else None
    REGIME_PARQUET_ROOT = Path(config.output_root)
    DEFINITION_ROOT = Path(config.definition_root)
    LOG_DIR = Path(config.log_root)
    LOG_FILE = LOG_DIR / "regime_clustering.log"
    for root in (REGIME_PARQUET_ROOT, DEFINITION_ROOT, LOG_DIR, Path(config.diagnostic_root), Path(config.tmp_root)):
        root.mkdir(parents=True, exist_ok=True)
    return config


def _configure_worker_regime_label_generation_io(
    config: RegimeLabelGenerationIOConfig,
    sandbox_roots: Optional[SandboxOutputRoots],
) -> None:
    global REGIME_LABEL_IO_CONFIG, REGIME_LABEL_SANDBOX_ROOTS, REGIME_PARQUET_ROOT, DEFINITION_ROOT, LOG_DIR, LOG_FILE
    REGIME_LABEL_IO_CONFIG = config
    REGIME_LABEL_SANDBOX_ROOTS = sandbox_roots if sandbox_roots is not None and sandbox_roots.enabled else None
    REGIME_PARQUET_ROOT = Path(config.output_root)
    DEFINITION_ROOT = Path(config.definition_root)
    LOG_DIR = Path(config.log_root)
    LOG_FILE = LOG_DIR / "regime_clustering.log"


REGIME_LABEL_IO_CONFIG = resolve_regime_label_generation_io_config()
REGIME_LABEL_SANDBOX_ROOTS = None
REGIME_PARQUET_ROOT = Path(REGIME_LABEL_IO_CONFIG.output_root)
DEFINITION_ROOT = Path(REGIME_LABEL_IO_CONFIG.definition_root)
LOG_DIR = Path(REGIME_LABEL_IO_CONFIG.log_root)
LOG_FILE = LOG_DIR / "regime_clustering.log"

DEFAULT_INTERVALS = [1, 5, 15, 30, 60, 240, 720, 1440]
SECONDS_PER_DAY = 86400
HARD_OUTPUT_START_TS = int(pd.Timestamp("2021-01-01T00:00:00Z").timestamp())


@dataclass(frozen=True)
class RegimeClusteringRuntimeProfile:
    name: str
    asset_workers: Optional[int] = None
    max_asset_workers: Optional[int] = None
    max_output_months: Optional[int] = None
    requires_bounded_output: bool = False
    required_parallel_thread_cap: Optional[str] = None
    diagnostics_verbosity: str = "normal"
    feature_window_cache_policy: str = "column_discovery_cache"
    description: str = ""


REGIME_CLUSTERING_RUNTIME_PROFILES: Dict[str, RegimeClusteringRuntimeProfile] = {
    "runtime_config": RegimeClusteringRuntimeProfile(
        name="runtime_config",
        asset_workers=None,
        description="Use module-keyed pipeline_runtime.json defaults.",
    ),
    "regimes_architecture_validation": RegimeClusteringRuntimeProfile(
        name="regimes_architecture_validation",
        asset_workers=1,
        diagnostics_verbosity="test",
        description="Conservative import and contract validation profile.",
    ),
    "regimes_asset_state_preflight": RegimeClusteringRuntimeProfile(
        name="regimes_asset_state_preflight",
        asset_workers=2,
        diagnostics_verbosity="preflight",
        description="Explicit-asset source validation profile; override workers to match the planned candidate.",
    ),
    "regimes_asset_state_study": RegimeClusteringRuntimeProfile(
        name="regimes_asset_state_study",
        asset_workers=2,
        description="Current bounded small-study profile selected from available evidence.",
    ),
    "regimes_asset_state_study_bounded": RegimeClusteringRuntimeProfile(
        name="regimes_asset_state_study_bounded",
        asset_workers=2,
        max_output_months=2,
        diagnostics_verbosity="benchmark",
        description="Bounded label-generation probe profile for controlled worker sweeps.",
    ),
    "regimes_asset_state_backfill_interim": RegimeClusteringRuntimeProfile(
        name="regimes_asset_state_backfill_interim",
        asset_workers=None,
        max_asset_workers=6,
        requires_bounded_output=True,
        required_parallel_thread_cap=REGIME_PARALLEL_THREAD_CAP_RECOMMENDATION,
        description="Interim historical backfill profile using current runtime-config worker defaults after preflight.",
    ),
}

DEFAULT_FEATURE_SUBSET = [
    "log_return",
    "atr_14",
    "rsi_14",
    "macd_hist_12_26_9",
    "adx_14",
    "ret_std_20",
    "cv_20",
    "vol_osc_pct_14_28",
    "vroc_14",
    "avg_trade_size",
    "trade_intensity",
    "prr",
]


def _runtime_memory_summary() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return {
            "mem_total_gb": round(float(vm.total) / (1024 ** 3), 1),
            "mem_avail_gb": round(float(vm.available) / (1024 ** 3), 1),
            "mem_used_pct": round(float(vm.percent), 1),
        }
    except Exception:
        return {"mem": "n/a"}


def resolve_regime_clustering_runtime_profile(args: argparse.Namespace) -> RegimeClusteringRuntimeProfile:
    raw_name = str(getattr(args, "runtime_profile", "") or "").strip() or "runtime_config"
    profile = REGIME_CLUSTERING_RUNTIME_PROFILES.get(raw_name)
    if profile is None:
        valid = ", ".join(sorted(REGIME_CLUSTERING_RUNTIME_PROFILES))
        raise ValueError(f"unknown Regime clustering runtime profile {raw_name!r}; expected one of: {valid}")
    return profile


def resolve_regime_clustering_workers(
    args: argparse.Namespace,
    *,
    runtime_profile: Optional[RegimeClusteringRuntimeProfile] = None,
) -> int:
    profile = runtime_profile or resolve_regime_clustering_runtime_profile(args)
    raw_workers = getattr(args, "workers", None)
    if raw_workers is not None:
        workers = _optional_positive_int(raw_workers) or 1
        return _validate_regime_clustering_worker_cap(profile, workers)
    if profile.asset_workers is not None:
        workers = max(1, int(profile.asset_workers))
        return _validate_regime_clustering_worker_cap(profile, workers)
    workers = get_workers("regime_clustering", "asset_workers")
    return _validate_regime_clustering_worker_cap(profile, workers)


def _validate_regime_clustering_worker_cap(profile: RegimeClusteringRuntimeProfile, workers: int) -> int:
    resolved = max(1, int(workers))
    if profile.max_asset_workers is not None and resolved > int(profile.max_asset_workers):
        raise ValueError(
            f"{profile.name} supports at most {int(profile.max_asset_workers)} asset workers; "
            f"got {resolved}"
        )
    return int(resolved)


def _runtime_profile_snapshot(profile: RegimeClusteringRuntimeProfile) -> Dict[str, Any]:
    return {
        "name": str(profile.name),
        "asset_workers": int(profile.asset_workers) if profile.asset_workers is not None else None,
        "max_asset_workers": int(profile.max_asset_workers) if profile.max_asset_workers is not None else None,
        "max_output_months": int(profile.max_output_months) if profile.max_output_months is not None else None,
        "requires_bounded_output": bool(profile.requires_bounded_output),
        "required_parallel_thread_cap": (
            str(profile.required_parallel_thread_cap) if profile.required_parallel_thread_cap is not None else None
        ),
        "diagnostics_verbosity": str(profile.diagnostics_verbosity),
        "feature_window_cache_policy": str(profile.feature_window_cache_policy),
        "description": str(profile.description),
    }


def _environment_thread_cap_snapshot(asset_workers: int) -> Dict[str, Any]:
    caps = {name: str(os.environ.get(name, "")) for name in REGIME_THREAD_CAP_ENV_VARS}
    missing_caps = [name for name, value in caps.items() if not str(value).strip()]
    return {
        "caps": caps,
        "policy": "record_only",
        "recommended_for_parallel_runs": {
            name: REGIME_PARALLEL_THREAD_CAP_RECOMMENDATION for name in REGIME_THREAD_CAP_ENV_VARS
        },
        "missing_caps": missing_caps,
        "all_caps_set": not missing_caps,
        "parallel_label_generation_warning": bool(int(asset_workers) > 1 and missing_caps),
    }


def resolve_regime_clustering_runtime_guardrails(
    args: argparse.Namespace,
    *,
    workers: int,
    runtime_profile: Optional[RegimeClusteringRuntimeProfile] = None,
    output_limits: Optional["RegimeOutputLimits"] = None,
) -> Dict[str, Any]:
    profile = runtime_profile or resolve_regime_clustering_runtime_profile(args)
    asset_workers = max(1, int(workers))
    resolved_output_limits = output_limits or resolve_regime_output_limits(args, runtime_profile=profile)
    output_limit_snapshot = resolved_output_limits.to_json_ready()
    thread_caps = _environment_thread_cap_snapshot(asset_workers)
    required_thread_cap = profile.required_parallel_thread_cap
    cap_mismatches: List[Dict[str, str]] = []
    if required_thread_cap is not None and asset_workers > 1:
        expected = str(required_thread_cap)
        for env_name, value in thread_caps["caps"].items():
            if str(value).strip() != expected:
                cap_mismatches.append(
                    {
                        "name": str(env_name),
                        "expected": expected,
                        "actual": str(value),
                    }
                )

    failed_checks: List[str] = []
    if bool(profile.requires_bounded_output) and not bool(output_limit_snapshot["active"]):
        failed_checks.append(
            f"{profile.name} requires --max-output-months or both --output-start and --output-end"
        )
    if profile.max_asset_workers is not None and asset_workers > int(profile.max_asset_workers):
        failed_checks.append(
            f"{profile.name} supports at most {int(profile.max_asset_workers)} asset workers; got {asset_workers}"
        )
    if cap_mismatches:
        failed_checks.append(
            f"{profile.name} parallel runs require {', '.join(REGIME_THREAD_CAP_ENV_VARS)}="
            f"{str(required_thread_cap)} before launch"
        )

    return {
        "profile_requires_guardrails": bool(
            profile.requires_bounded_output
            or profile.max_asset_workers is not None
            or profile.required_parallel_thread_cap is not None
        ),
        "safe_for_execute": not failed_checks,
        "requires_bounded_output": bool(profile.requires_bounded_output),
        "bounded_output_scope": bool(output_limit_snapshot["active"]),
        "max_asset_workers": int(profile.max_asset_workers) if profile.max_asset_workers is not None else None,
        "worker_cap_satisfied": bool(
            profile.max_asset_workers is None or asset_workers <= int(profile.max_asset_workers)
        ),
        "required_parallel_thread_cap": str(required_thread_cap) if required_thread_cap is not None else None,
        "thread_cap_satisfied": not cap_mismatches,
        "thread_cap_mismatches": cap_mismatches,
        "failed_checks": failed_checks,
        "notes": [
            "native BLAS/thread caps must be set in the launcher environment before Python imports numeric libraries"
        ]
        if required_thread_cap is not None
        else [],
    }


def validate_regime_clustering_runtime_guardrails(
    args: argparse.Namespace,
    *,
    workers: int,
    runtime_profile: Optional[RegimeClusteringRuntimeProfile] = None,
    output_limits: Optional["RegimeOutputLimits"] = None,
) -> Dict[str, Any]:
    status = resolve_regime_clustering_runtime_guardrails(
        args,
        workers=workers,
        runtime_profile=runtime_profile,
        output_limits=output_limits,
    )
    failed_checks = list(status.get("failed_checks", []))
    if failed_checks:
        raise ValueError("; ".join(str(check) for check in failed_checks))
    return status


def resolve_regime_clustering_runtime_snapshot(
    args: argparse.Namespace,
    *,
    workers: int,
    runtime_profile: Optional[RegimeClusteringRuntimeProfile] = None,
    output_limits: Optional["RegimeOutputLimits"] = None,
) -> Dict[str, Any]:
    profile = runtime_profile or resolve_regime_clustering_runtime_profile(args)
    asset_workers = max(1, int(workers))
    cpu_count = os.cpu_count() or 1
    resolved_output_limits = output_limits or resolve_regime_output_limits(args, runtime_profile=profile)
    output_limits_snapshot = resolved_output_limits.to_json_ready()
    thread_caps = _environment_thread_cap_snapshot(asset_workers)
    guardrails = resolve_regime_clustering_runtime_guardrails(
        args,
        workers=asset_workers,
        runtime_profile=profile,
        output_limits=resolved_output_limits,
    )
    return {
        "profile_name": str(profile.name),
        "path_profile": str(PIPELINE_PROFILE),
        "module_slug": "regime_clustering",
        "runtime_profile": _runtime_profile_snapshot(profile),
        "asset_workers": int(asset_workers),
        "writer_workers": 1,
        "model_threads": "n/a",
        "trial_workers": 0,
        "cpu_count": int(cpu_count),
        "cpu_budget": int(asset_workers),
        "oversubscription_warning": bool(asset_workers > cpu_count),
        "thread_pressure_warning": bool(thread_caps["parallel_label_generation_warning"]),
        "thread_pressure_warning_reason": (
            "parallel label generation has unset BLAS/thread caps"
            if bool(thread_caps["parallel_label_generation_warning"])
            else ""
        ),
        "process_start_method": str(mp.get_start_method(allow_none=True) or "default"),
        "runtime_config_path": str(RUNTIME_CONFIG_PATH),
        "environment_thread_caps": thread_caps["caps"],
        "environment_thread_cap_policy": {
            "policy": str(thread_caps["policy"]),
            "recommended_for_parallel_runs": dict(thread_caps["recommended_for_parallel_runs"]),
            "missing_caps": list(thread_caps["missing_caps"]),
            "all_caps_set": bool(thread_caps["all_caps_set"]),
        },
        "runtime_guardrails": guardrails,
        "output_root": str(REGIME_PARQUET_ROOT),
        "diagnostic_root": str(LOG_DIR),
        "label_generation_io": REGIME_LABEL_IO_CONFIG.to_json_ready(),
        "memory": _runtime_memory_summary(),
        "cli_overrides": {
            "assets": str(getattr(args, "assets", "") or ""),
            "bands": str(getattr(args, "bands", "") or ""),
            "feature_strategy": str(getattr(args, "feature_strategy", "") or ""),
            "n_per_interval": int(getattr(args, "n_per_interval", 0) or 0),
            "workers": int(asset_workers),
            "preflight_only": bool(getattr(args, "preflight_only", False)),
            "output_start": str(getattr(args, "output_start", "") or ""),
            "output_end": str(getattr(args, "output_end", "") or ""),
            "max_output_months": _optional_positive_int(getattr(args, "max_output_months", None)),
        },
        "output_limits": output_limits_snapshot,
    }

CATEGORY_SPECS: Dict[str, Dict[str, object]] = {
    "trend": {
        "bases": ["log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"],
        "cluster_col": "trend_cluster_id",
        "label_col": "trend_label",
        "confidence_col": "trend_confidence_pct",
        "intensity_col": "trend_intensity_pct",
        "labels": ["down", "flat", "up"],
    },
    "vol": {
        "bases": ["atr_14", "ret_std_20", "cv_20", "vol_osc_pct_14_28"],
        "cluster_col": "vol_cluster_id",
        "label_col": "vol_label",
        "confidence_col": "vol_confidence_pct",
        "intensity_col": "vol_intensity_pct",
        "labels": ["low", "normal", "high"],
    },
    "activity": {
        "bases": ["trade_intensity", "avg_trade_size", "vroc_14", "prr"],
        "cluster_col": "activity_cluster_id",
        "label_col": "activity_label",
        "confidence_col": "activity_confidence_pct",
        "intensity_col": "activity_intensity_pct",
        "labels": ["low", "normal", "high"],
    },
}
CATEGORY_ORDER = ["trend", "vol", "activity"]
ALL_CATEGORY_BASES = sorted({b for spec in CATEGORY_SPECS.values() for b in spec["bases"]})
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_EPSILON_MICRO = 0.0
HDBSCAN_EPSILON_MESO = 0.0
HDBSCAN_EPSILON_MACRO = 0.0
FLAT_ACTIVITY_TOL = 1e-12
EPSILON_OVERRIDE: Optional[float] = None
CONFIDENCE_MAPPING = "soft"


@dataclass
class BandSpec:
    name: str
    ceiling: int
    member_intervals: List[int]
    train_days: int


@dataclass(frozen=True)
class RegimeOutputLimits:
    output_start_ts: Optional[int] = None
    output_end_ts: Optional[int] = None
    max_output_months: Optional[int] = None

    def to_json_ready(self) -> Dict[str, Any]:
        return {
            "output_start_ts": int(self.output_start_ts) if self.output_start_ts is not None else None,
            "output_end_ts": int(self.output_end_ts) if self.output_end_ts is not None else None,
            "max_output_months": int(self.max_output_months) if self.max_output_months is not None else None,
            "active": bool(
                self.output_start_ts is not None
                or self.output_end_ts is not None
                or self.max_output_months is not None
            ),
        }


def _optional_positive_int(value: object) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"expected a positive integer, got {value!r}") from exc
    if out <= 0:
        raise ValueError(f"expected a positive integer, got {value!r}")
    return int(out)


def _optional_utc_ts(value: object) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        pass
    try:
        ts = pd.Timestamp(text)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    except Exception as exc:
        raise ValueError(f"expected a UTC timestamp or parseable date, got {value!r}") from exc


def resolve_regime_output_limits(
    args: argparse.Namespace,
    *,
    runtime_profile: Optional[RegimeClusteringRuntimeProfile] = None,
) -> RegimeOutputLimits:
    start_ts = _optional_utc_ts(getattr(args, "output_start", None))
    end_ts = _optional_utc_ts(getattr(args, "output_end", None))
    profile = runtime_profile or resolve_regime_clustering_runtime_profile(args)
    max_months = _optional_positive_int(getattr(args, "max_output_months", None))
    if max_months is None and profile.max_output_months is not None:
        max_months = max(1, int(profile.max_output_months))
    if start_ts is not None and end_ts is not None and int(start_ts) > int(end_ts):
        raise ValueError(f"output_start must be <= output_end, got {int(start_ts)} > {int(end_ts)}")
    return RegimeOutputLimits(
        output_start_ts=start_ts,
        output_end_ts=end_ts,
        max_output_months=max_months,
    )


def _month_limited_end_ts(start_ts: int, max_months: int, ceiling_min: int) -> int:
    start_dt = pd.to_datetime(int(start_ts), unit="s", utc=True)
    month_start = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=1, tz="UTC")
    next_window_start = month_start + pd.DateOffset(months=int(max_months))
    raw_end = int(next_window_start.timestamp()) - ceiling_seconds(int(ceiling_min))
    return int(floor_to_ceiling(raw_end, int(ceiling_min)))


def apply_regime_output_limits(
    start_ts: int,
    end_ts: int,
    band: BandSpec,
    limits: Optional[RegimeOutputLimits],
) -> Tuple[int, int, Dict[str, Any]]:
    original_start = int(start_ts)
    original_end = int(end_ts)
    resolved_start = int(start_ts)
    resolved_end = int(end_ts)
    limit_state = limits or RegimeOutputLimits()

    if limit_state.output_start_ts is not None:
        resolved_start = max(resolved_start, ceil_to_ceiling(int(limit_state.output_start_ts), int(band.ceiling)))
    if limit_state.output_end_ts is not None:
        resolved_end = min(resolved_end, floor_to_ceiling(int(limit_state.output_end_ts), int(band.ceiling)))
    if limit_state.max_output_months is not None:
        resolved_end = min(
            resolved_end,
            _month_limited_end_ts(
                start_ts=int(resolved_start),
                max_months=int(limit_state.max_output_months),
                ceiling_min=int(band.ceiling),
            ),
        )

    meta = {
        **limit_state.to_json_ready(),
        "original_start_ts": int(original_start),
        "original_end_ts": int(original_end),
        "resolved_start_ts": int(resolved_start),
        "resolved_end_ts": int(resolved_end),
        "clamped": bool(resolved_start != original_start or resolved_end != original_end),
        "empty_after_limits": bool(int(resolved_start) > int(resolved_end)),
    }
    return int(resolved_start), int(resolved_end), meta


BANDS = [
    BandSpec("micro", 30, [1, 5, 15, 30], 30),
    BandSpec("meso", 240, [60, 240], 180),
    BandSpec("macro", 1440, [720, 1440], 360),
]

_FEATURE_COLUMN_DISCOVERY_CACHE: Dict[Tuple[str, str, int, str, int, int], List[str]] = {}
_FEATURE_COLUMN_DISCOVERY_CACHE_MAX = 512


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    base_log(f"[regimes] {msg}")


def read_json(path: Path) -> dict:
    return regime_artifacts.read_json(path)


def write_json(path: Path, obj: dict) -> None:
    regime_artifacts.write_json(
        path,
        obj,
        write_kind="diagnostics",
        sandbox_roots=REGIME_LABEL_SANDBOX_ROOTS,
    )


def month_range(start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
    start_dt = pd.to_datetime(start_ts, unit="s", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="s", utc=True)
    cur = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=1, tz="UTC")
    end_marker = pd.Timestamp(year=end_dt.year, month=end_dt.month, day=1, tz="UTC")
    out: List[Tuple[int, int]] = []
    while cur <= end_marker:
        out.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return out


def assets_present_in_features(interval_min: int, root: Optional[Path] = None) -> List[str]:
    root_dir = Path(root) if root else PARQUET_ROOT
    base = root_dir / f"scalar_features_{int(interval_min)}"
    if not base.exists():
        return []
    assets: List[str] = []
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith("asset="):
            continue
        asset = child.name.split("=", 1)[1]
        if asset:
            assets.append(asset)
    return sorted(set(assets))


def _band_specs_for_names(bands: Optional[Sequence[str]]) -> List[BandSpec]:
    requested = {str(b).strip().lower() for b in bands or [] if str(b).strip()}
    if not requested:
        return list(BANDS)
    known = {str(b.name).lower(): b for b in BANDS}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"unknown regime band(s): {unknown}; expected one or more of {sorted(known)}")
    return [known[name] for name in sorted(requested, key=lambda n: [b.name for b in BANDS].index(n))]


def _required_feature_intervals_for_bands(band_specs: Sequence[BandSpec]) -> List[int]:
    intervals: Set[int] = set()
    for band in band_specs:
        intervals.update(int(v) for v in band.member_intervals)
        intervals.add(int(band.ceiling))
    return sorted(intervals)


def _asset_interval_feature_summary(asset: str, interval_min: int, root: Optional[Path] = None) -> Dict[str, Any]:
    root_dir = Path(root) if root else PARQUET_ROOT
    asset_dir = root_dir / f"scalar_features_{int(interval_min)}" / f"asset={asset}"
    month_partitions = 0
    parquet_files = 0
    row_count_estimate = 0
    row_count_complete = True
    if asset_dir.exists():
        for year_dir in asset_dir.glob("year=*"):
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.glob("month=*"):
                if not month_dir.is_dir():
                    continue
                files = sorted(month_dir.glob("*.parquet"))
                if files:
                    month_partitions += 1
                    parquet_files += len(files)
                for part in files:
                    rows = _parquet_metadata_rows(part)
                    if rows is None:
                        row_count_complete = False
                    else:
                        row_count_estimate += int(rows)
    first_ts, last_ts = feature_bounds_from_parquet(int(interval_min), str(asset), root=root_dir)
    return {
        "interval_minutes": int(interval_min),
        "asset_partition": str(asset_dir),
        "asset_partition_exists": bool(asset_dir.exists()),
        "month_partitions": int(month_partitions),
        "parquet_files": int(parquet_files),
        "row_count_estimate": int(row_count_estimate) if row_count_complete else None,
        "row_count_estimate_complete": bool(row_count_complete),
        "first_ts": int(first_ts) if first_ts is not None else None,
        "last_ts": int(last_ts) if last_ts is not None else None,
    }


def _parquet_metadata_rows(path: Path) -> Optional[int]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        metadata = pq.ParquetFile(path).metadata
        return int(metadata.num_rows) if metadata is not None else None
    except Exception:
        return None


def resolve_regime_clustering_preflight(
    args: argparse.Namespace,
    *,
    workers: int,
    assets: Optional[Sequence[str]] = None,
    bands: Optional[Sequence[str]] = None,
    runtime_profile: Optional[RegimeClusteringRuntimeProfile] = None,
) -> Dict[str, Any]:
    profile = runtime_profile or resolve_regime_clustering_runtime_profile(args)
    band_specs = _band_specs_for_names(bands)
    required_intervals = _required_feature_intervals_for_bands(band_specs)
    if assets:
        requested_assets = sorted({str(a).strip() for a in assets if str(a).strip()})
        asset_source = "cli"
    else:
        discovery_interval = max(required_intervals) if required_intervals else DEFAULT_INTERVALS[-1]
        requested_assets = assets_present_in_features(discovery_interval)
        asset_source = f"discovered:scalar_features_{int(discovery_interval)}"

    rows: List[Dict[str, Any]] = []
    missing_assets: List[Dict[str, Any]] = []
    total_parquet_files = 0
    total_month_partitions = 0
    total_rows: Optional[int] = 0
    for asset in requested_assets:
        interval_summaries = [
            _asset_interval_feature_summary(asset=str(asset), interval_min=int(interval_min))
            for interval_min in required_intervals
        ]
        missing_intervals = [
            int(row["interval_minutes"])
            for row in interval_summaries
            if not bool(row["asset_partition_exists"]) or int(row["parquet_files"]) <= 0
        ]
        for row in interval_summaries:
            total_parquet_files += int(row["parquet_files"])
            total_month_partitions += int(row["month_partitions"])
            if total_rows is not None:
                if row["row_count_estimate"] is None:
                    total_rows = None
                else:
                    total_rows += int(row["row_count_estimate"])
        asset_row = {
            "asset": str(asset),
            "ok": not bool(missing_intervals),
            "missing_intervals": missing_intervals,
            "intervals": interval_summaries,
        }
        rows.append(asset_row)
        if missing_intervals:
            missing_assets.append({"asset": str(asset), "missing_intervals": missing_intervals})

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "module_slug": "regime_clustering",
        "status": "ok" if not missing_assets else "missing_source_partitions",
        "asset_source": asset_source,
        "requested_assets": requested_assets,
        "bands": [str(b.name) for b in band_specs],
        "required_feature_intervals": required_intervals,
        "workers": max(1, int(workers)),
        "source_feature_root": str(PARQUET_ROOT),
        "output_root": str(REGIME_PARQUET_ROOT),
        "diagnostic_root": str(LOG_DIR),
        "label_generation_io": REGIME_LABEL_IO_CONFIG.to_json_ready(),
        "runtime_profile": _runtime_profile_snapshot(profile),
        "output_limits": resolve_regime_output_limits(args, runtime_profile=profile).to_json_ready(),
        "feature_window_cache_policy": {
            "column_discovery_cache_max_entries": int(_FEATURE_COLUMN_DISCOVERY_CACHE_MAX),
        },
        "totals": {
            "assets": len(requested_assets),
            "ok_assets": len(requested_assets) - len(missing_assets),
            "missing_assets": len(missing_assets),
            "month_partitions": int(total_month_partitions),
            "parquet_files": int(total_parquet_files),
            "row_count_estimate": int(total_rows) if total_rows is not None else None,
            "row_count_estimate_complete": total_rows is not None,
        },
        "missing_assets": missing_assets,
        "assets_detail": rows,
        "runtime_snapshot": resolve_regime_clustering_runtime_snapshot(
            args,
            workers=max(1, int(workers)),
            runtime_profile=profile,
        ),
    }


def ceiling_seconds(ceiling_min: int) -> int:
    return int(ceiling_min * 60)


def floor_to_ceiling(ts: int, ceiling_min: int) -> int:
    step = ceiling_seconds(ceiling_min)
    return int(ts // step * step)


def ceil_to_ceiling(ts: int, ceiling_min: int) -> int:
    step = ceiling_seconds(ceiling_min)
    if ts % step == 0:
        return int(ts)
    return int((ts // step + 1) * step)


def utc_week_key(ts: int) -> str:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    iso = dt.isocalendar()
    return f"{int(iso.year):04d}-{int(iso.week):02d}"


def utc_month_key(ts: int) -> str:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    return f"{int(dt.year):04d}-{int(dt.month):02d}"


def next_weekly_boundary_start_ts(ts: int) -> int:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    week_start = dt.normalize() - pd.Timedelta(days=int(dt.weekday()))
    boundary = week_start + pd.Timedelta(days=7)
    while boundary <= dt:
        boundary = boundary + pd.Timedelta(days=7)
    return int(boundary.timestamp())


def next_monthly_boundary_start_ts(ts: int) -> int:
    dt = pd.to_datetime(int(ts), unit="s", utc=True)
    month_start = pd.Timestamp(year=dt.year, month=dt.month, day=1, tz="UTC")
    boundary = month_start + pd.DateOffset(months=1)
    while boundary <= dt:
        boundary = boundary + pd.DateOffset(months=1)
    return int(boundary.timestamp())


def cadence_refit_key(ts: int, band: BandSpec) -> str:
    if str(band.name).lower() == "micro":
        return utc_week_key(ts)
    return utc_month_key(ts)


def next_boundary_start_ts(ts: int, band: BandSpec) -> int:
    step = ceiling_seconds(band.ceiling)
    if str(band.name).lower() == "micro":
        raw = next_weekly_boundary_start_ts(ts)
    else:
        raw = next_monthly_boundary_start_ts(ts)
    boundary = ceil_to_ceiling(int(raw), band.ceiling)
    if boundary <= int(ts):
        boundary += int(step)
    return int(boundary)


def definition_valid_until_for_refit(refit_ts: int, band: BandSpec) -> int:
    return int(next_boundary_start_ts(int(refit_ts), band) - ceiling_seconds(band.ceiling))


def epsilon_for_band(band_name: str) -> float:
    if EPSILON_OVERRIDE is not None:
        return float(EPSILON_OVERRIDE)
    name = str(band_name).lower()
    if name == "micro":
        return float(HDBSCAN_EPSILON_MICRO)
    if name == "meso":
        return float(HDBSCAN_EPSILON_MESO)
    return float(HDBSCAN_EPSILON_MACRO)


def mcs_for_band(band_name: str) -> int:
    name = str(band_name).lower()
    if name == "micro":
        return 10
    return 15


def flat_activity_mask(df: pd.DataFrame, band: BandSpec, tol: float = FLAT_ACTIVITY_TOL) -> Tuple[pd.Series, bool]:
    c_log = f"i{band.ceiling}_log_return"
    c_ti = f"i{band.ceiling}_trade_intensity"
    c_prr = f"i{band.ceiling}_prr"
    if not all(c in df.columns for c in (c_log, c_ti, c_prr)):
        return pd.Series(False, index=df.index, dtype=bool), False
    activity = df[[c_log, c_ti, c_prr]].astype(float).abs()
    inactive_mask = (activity[c_log] <= tol) & (activity[c_ti] <= tol) & (activity[c_prr] <= tol)
    return inactive_mask.fillna(False).astype(bool), True


def _safe_float(val: Any) -> float:
    try:
        x = float(val)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x


def _series_to_float_dict(s: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in s.items():
        out[str(k)] = _safe_float(v)
    return out


def _pairwise_corr_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not cols:
        return {}
    sub = df[list(cols)].astype(float)
    corr = sub.corr()
    out: Dict[str, Dict[str, float]] = {}
    for r in corr.index:
        row: Dict[str, float] = {}
        for c in corr.columns:
            row[str(c)] = _safe_float(corr.loc[r, c])
        out[str(r)] = row
    return out


def _hist_quantile_from_counts(counts: Sequence[int], q: float) -> float:
    if not counts:
        return 0.0
    total = int(sum(int(x) for x in counts))
    if total <= 0:
        return 0.0
    q = float(min(1.0, max(0.0, q)))
    target = int(math.ceil(q * total))
    if target <= 0:
        target = 1
    csum = 0
    for i, c in enumerate(counts):
        csum += int(c)
        if csum >= target:
            return float(i)
    return float(len(counts) - 1)


class DiagnosticCollector:
    def __init__(self, asset: str):
        self.asset = str(asset)
        self._rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _entry(self, band: str, category: str) -> Dict[str, Any]:
        key = (str(band), str(category))
        if key not in self._rows:
            self._rows[key] = {
                "asset": self.asset,
                "band": str(band),
                "category": str(category),
                "bars_total": 0,
                "flat_bar_count": 0,
                "centroid_candidate_rows": 0,
                "overridden_to_neutral_rows": 0,
                "final_unknown_rows": 0,
                "centroid_assigned_rows": 0,
                "centroid_left_unknown_rows": 0,
                "no_model_or_no_cluster_rows": 0,
                "incomplete_feature_rows": 0,
                "confidence_count": 0,
                "confidence_sum": 0.0,
                "confidence_hist": [0] * 101,
                "feature_variance": {},
                "feature_std": {},
                "pairwise_correlation_matrix": {},
                "clustering": {
                    "clusters_found": 0,
                    "training_noise_fraction": 1.0,
                    "cluster_sizes": {},
                },
            }
        return self._rows[key]

    def record_fit(self, band: BandSpec, category: str, train_df: pd.DataFrame, feature_cols: Sequence[str], model: Optional[dict]) -> None:
        entry = self._entry(band.name, category)
        if feature_cols and not train_df.empty:
            cols_present = [c for c in feature_cols if c in train_df.columns]
            if cols_present:
                numeric = train_df[cols_present].astype(float)
                entry["feature_variance"] = _series_to_float_dict(numeric.var(ddof=0))
                entry["feature_std"] = _series_to_float_dict(numeric.std(ddof=0))
                entry["pairwise_correlation_matrix"] = _pairwise_corr_matrix(numeric, cols_present)
        if model is None:
            entry["clustering"] = {
                "clusters_found": 0,
                "training_noise_fraction": 1.0,
                "cluster_sizes": {},
            }
            return
        train_meta = model.get("train_meta", {}) if isinstance(model, dict) else {}
        labels_raw = train_meta.get("labels", np.array([], dtype=int))
        labels = np.asarray(labels_raw, dtype=int) if labels_raw is not None else np.array([], dtype=int)
        n = int(labels.size)
        noise_count = int(np.sum(labels == -1)) if n > 0 else 0
        cluster_sizes: Dict[str, int] = {}
        for cid in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
            cluster_sizes[str(cid)] = int(np.sum(labels == cid))
        entry["clustering"] = {
            "clusters_found": int(len(cluster_sizes)),
            "training_noise_fraction": float(noise_count / n) if n > 0 else 1.0,
            "cluster_sizes": cluster_sizes,
        }

    def record_assign(
        self,
        band: BandSpec,
        category: str,
        bars_total: int,
        flat_bar_count: int,
        centroid_candidate_rows: int,
        overridden_to_neutral_rows: int,
        final_unknown_rows: int,
        centroid_assigned_rows: int = 0,
        centroid_left_unknown_rows: int = 0,
        no_model_or_no_cluster_rows: int = 0,
        incomplete_feature_rows: int = 0,
        confidence_values: Optional[np.ndarray] = None,
    ) -> None:
        entry = self._entry(band.name, category)
        entry["bars_total"] += int(bars_total)
        entry["flat_bar_count"] += int(flat_bar_count)
        entry["centroid_candidate_rows"] += int(centroid_candidate_rows)
        entry["overridden_to_neutral_rows"] += int(overridden_to_neutral_rows)
        entry["final_unknown_rows"] += int(final_unknown_rows)
        entry["centroid_assigned_rows"] += int(centroid_assigned_rows)
        entry["centroid_left_unknown_rows"] += int(centroid_left_unknown_rows)
        entry["no_model_or_no_cluster_rows"] += int(no_model_or_no_cluster_rows)
        entry["incomplete_feature_rows"] += int(incomplete_feature_rows)
        if confidence_values is not None and int(len(confidence_values)) > 0:
            vals = np.asarray(confidence_values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                vals = np.clip(np.rint(vals), 0, 100).astype(int)
                hist = np.bincount(vals, minlength=101)
                entry["confidence_count"] += int(vals.size)
                entry["confidence_sum"] += float(vals.sum())
                cur_hist = entry.get("confidence_hist", [0] * 101)
                if not isinstance(cur_hist, list) or len(cur_hist) != 101:
                    cur_hist = [0] * 101
                entry["confidence_hist"] = [int(cur_hist[i]) + int(hist[i]) for i in range(101)]

    def to_json_ready(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (_band, _category), entry in sorted(self._rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            bars_total = int(entry["bars_total"])
            if bars_total <= 0:
                continue
            conf_count = int(entry.get("confidence_count", 0))
            conf_sum = float(entry.get("confidence_sum", 0.0))
            conf_hist = entry.get("confidence_hist", [0] * 101)
            if not isinstance(conf_hist, list) or len(conf_hist) != 101:
                conf_hist = [0] * 101
            rows.append(
                {
                    "asset": entry["asset"],
                    "band": entry["band"],
                    "category": entry["category"],
                    "bars_total": bars_total,
                    "flat_bar_fraction": float(entry["flat_bar_count"] / bars_total),
                    "feature_variance": dict(entry.get("feature_variance", {})),
                    "feature_std": dict(entry.get("feature_std", {})),
                    "pairwise_correlation_matrix": dict(entry.get("pairwise_correlation_matrix", {})),
                    "clustering": dict(entry.get("clustering", {})),
                    "assignments": {
                        "centroid_candidate_fraction": float(entry["centroid_candidate_rows"] / bars_total),
                        "overridden_to_neutral_fraction": float(entry["overridden_to_neutral_rows"] / bars_total),
                        "final_unknown_fraction": float(entry["final_unknown_rows"] / bars_total),
                        "centroid_assigned_fraction": float(entry["centroid_assigned_rows"] / bars_total),
                        "centroid_left_unknown_fraction": float(entry["centroid_left_unknown_rows"] / bars_total),
                        "no_model_or_no_cluster_fraction": float(entry["no_model_or_no_cluster_rows"] / bars_total),
                        "incomplete_feature_fraction": float(entry["incomplete_feature_rows"] / bars_total),
                        "mean_confidence": float(conf_sum / conf_count) if conf_count > 0 else 0.0,
                        "median_confidence": _hist_quantile_from_counts(conf_hist, 0.50) if conf_count > 0 else 0.0,
                        "p10_confidence": _hist_quantile_from_counts(conf_hist, 0.10) if conf_count > 0 else 0.0,
                        "p90_confidence": _hist_quantile_from_counts(conf_hist, 0.90) if conf_count > 0 else 0.0,
                    },
                }
            )
        return rows


def parquet_path_for(ceiling_interval: int, asset: str, year: int, month: int, root: Path) -> Path:
    return regime_artifacts.parquet_path_for(ceiling_interval, asset, year, month, root)


def _lock_path_for(dst: Path) -> Path:
    return dst.with_suffix(dst.suffix + ".lock")


def _acquire_lock(lock_path: Path, retries: int = 40, sleep_sec: float = 0.25) -> Optional[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            time.sleep(sleep_sec)
        except Exception:
            time.sleep(sleep_sec)
    return None


def _release_lock(fd: Optional[int], lock_path: Path) -> None:
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def write_parquet_atomic(df: pd.DataFrame, dst: Path, retries: int = 6, sleep_base_sec: float = 0.25) -> None:
    regime_artifacts.write_parquet_atomic(
        df,
        dst,
        retries=retries,
        sleep_base_sec=sleep_base_sec,
        compression=PARQUET_COMPRESSION,
        row_group_size=PARQUET_ROW_GROUP,
        write_kind="parquet",
        sandbox_roots=REGIME_LABEL_SANDBOX_ROOTS,
    )


def definition_paths(asset: str, band: str, category: str) -> Tuple[Path, Path]:
    return regime_artifacts.definition_paths(DEFINITION_ROOT, asset, band, category)


def load_definition(asset: str, band: str, category: str) -> Optional[dict]:
    return regime_artifacts.load_definition(DEFINITION_ROOT, asset, band, category, log_fn=log)


def save_definition(asset: str, band: str, category: str, model_obj: dict, meta: dict) -> None:
    regime_artifacts.save_definition(
        DEFINITION_ROOT,
        asset,
        band,
        category,
        model_obj,
        meta,
        write_kind="regime_definition",
        sandbox_roots=REGIME_LABEL_SANDBOX_ROOTS,
    )


def load_scalar_features_window(
    interval_min: int,
    asset: str,
    start_ts: int,
    end_ts: int,
    columns: Sequence[str],
    root: Optional[Path] = None,
) -> pd.DataFrame:
    root = Path(root) if root else PARQUET_ROOT
    base = root / f"scalar_features_{interval_min}"
    if not base.exists():
        raise RuntimeError(
            f"scalar_features base missing for interval={interval_min}; expected {base} "
            f"(asset-partitioned required)"
        )
    asset_base = base / f"asset={asset}"
    if not asset_base.exists():
        raise RuntimeError(
            f"scalar_features asset partition missing for interval={interval_min} asset={asset}; expected "
            f"{asset_base / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    scan_base = asset_base
    dfs: List[pd.DataFrame] = []
    for y, m in month_range(start_ts, end_ts):
        p = scan_base / f"year={y}/month={m:02d}"
        if not p.exists():
            continue
        for pq in p.glob("*.parquet"):
            try:
                df = pd.read_parquet(pq, columns=columns)
            except Exception:
                try:
                    read_cols = [c for c in columns if c != "asset"]
                    df = pd.read_parquet(pq, columns=read_cols)
                except Exception:
                    continue
            if "asset" not in df.columns:
                df["asset"] = asset
            df = df[df["asset"] == asset]
            if df.empty:
                continue
            df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
            if not df.empty:
                dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("ts")


def _latest_year_month_dir(base: Path) -> Optional[Tuple[int, int, Path]]:
    if not base.exists():
        return None
    years: List[int] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.split("=", 1)[1]))
            except Exception:
                continue
    if not years:
        return None
    year = max(years)
    year_dir = base / f"year={year}"
    months: List[int] = []
    if year_dir.exists():
        for child in year_dir.iterdir():
            if child.is_dir() and child.name.startswith("month="):
                try:
                    months.append(int(child.name.split("=", 1)[1]))
                except Exception:
                    continue
    if not months:
        return None
    month = max(months)
    return (int(year), int(month), year_dir / f"month={month:02d}")


def _oldest_year_month_dir(base: Path) -> Optional[Tuple[int, int, Path]]:
    if not base.exists():
        return None
    years: List[int] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.split("=", 1)[1]))
            except Exception:
                continue
    if not years:
        return None
    year = min(years)
    year_dir = base / f"year={year}"
    months: List[int] = []
    if year_dir.exists():
        for child in year_dir.iterdir():
            if child.is_dir() and child.name.startswith("month="):
                try:
                    months.append(int(child.name.split("=", 1)[1]))
                except Exception:
                    continue
    if not months:
        return None
    month = min(months)
    return (int(year), int(month), year_dir / f"month={month:02d}")


def _previous_year_month(year: int, month: int) -> Tuple[int, int]:
    if int(month) > 1:
        return int(year), int(month) - 1
    return int(year) - 1, 12


def _next_year_month(year: int, month: int) -> Tuple[int, int]:
    if int(month) < 12:
        return int(year), int(month) + 1
    return int(year) + 1, 1


def _one_parquet_in_month_dir(month_dir: Path) -> Optional[Path]:
    if not month_dir.exists():
        return None
    files = sorted(month_dir.glob("*.parquet"), key=lambda p: p.name.lower())
    return files[0] if files else None


def feature_bounds_from_parquet(interval_min: int, asset: str, root: Optional[Path] = None) -> Tuple[Optional[int], Optional[int]]:
    root_dir = Path(root) if root else PARQUET_ROOT
    asset_dir = root_dir / f"scalar_features_{int(interval_min)}" / f"asset={asset}"
    if not asset_dir.exists():
        return (None, None)

    max_ts = feature_max_ts_from_parquet(interval_min=int(interval_min), asset=str(asset), root=root_dir)
    oldest = _oldest_year_month_dir(asset_dir)
    if oldest is None:
        return (None, int(max_ts) if max_ts is not None else None)

    year, month, _ = oldest
    first_ts: Optional[int] = None
    for attempt in range(2):
        month_dir = asset_dir / f"year={year}" / f"month={month:02d}"
        part = _one_parquet_in_month_dir(month_dir)
        if part is not None and part.exists():
            try:
                df = pd.read_parquet(part, columns=["ts"])
                if not df.empty:
                    s = pd.to_numeric(df["ts"], errors="coerce").dropna().astype("int64")
                    if not s.empty:
                        first_ts = int(s.min())
                        break
            except Exception:
                pass
        if attempt == 0:
            year, month = _next_year_month(year, month)
    return (first_ts, int(max_ts) if max_ts is not None else None)


def regime_bounds_from_parquet(ceiling_interval: int, asset: str, root: Optional[Path] = None) -> Tuple[Optional[int], Optional[int]]:
    root = Path(root) if root else REGIME_PARQUET_ROOT
    asset_dir = root / f"regimes_{ceiling_interval}" / f"asset={asset}"
    latest = _latest_year_month_dir(asset_dir)
    if latest is None:
        return (None, None)
    year, month, _month_dir = latest
    for attempt in range(2):
        part = asset_dir / f"year={year}" / f"month={month:02d}" / "part-000.parquet"
        if part.exists():
            try:
                df = pd.read_parquet(part, columns=["ts"])
                if not df.empty:
                    return (None, int(df["ts"].max()))
            except Exception:
                pass
        if attempt == 0:
            year, month = _previous_year_month(year, month)
    return (None, None)


def select_feature_columns(
    df: pd.DataFrame,
    strategy: str,
    subset: Sequence[str],
    corr_thresh: float,
    member_intervals: Sequence[int],
) -> List[str]:
    numeric_cols = [c for c in df.columns if c not in ("ts", "asset")]
    if strategy == "subset":
        selected: List[str] = []
        for interval in member_intervals:
            prefix = f"i{interval}_"
            for base in subset:
                col = f"{prefix}{base}"
                if col in numeric_cols:
                    selected.append(col)
        return selected
    cols = numeric_cols
    if not cols:
        return []
    corr = df[cols].corr().abs()
    keep: List[str] = []
    for col in cols:
        if not keep:
            keep.append(col)
            continue
        if all(corr.loc[col, k] < corr_thresh for k in keep):
            keep.append(col)
    return keep


def _feature_column_cache_key(interval_min: int, asset: str, parquet_path: Path) -> Tuple[str, str, int, str, int, int]:
    stat = parquet_path.stat()
    return (
        str(Path(PARQUET_ROOT).resolve()).lower(),
        str(asset),
        int(interval_min),
        str(parquet_path.resolve()).lower(),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def _read_feature_columns_from_probe(parquet_path: Path) -> List[str]:
    cols = list(pd.read_parquet(parquet_path, columns=None).columns)
    return [c for c in cols if c not in ("ts", "asset")]


def discover_feature_columns(interval_min: int, asset: str) -> List[str]:
    base = PARQUET_ROOT / f"scalar_features_{interval_min}"
    if not base.exists():
        raise RuntimeError(
            f"scalar_features base missing for interval={interval_min}; expected {base} "
            f"(asset-partitioned required: {base / f'asset={asset}' / 'year=YYYY' / 'month=MM' / '<one parquet>'})"
        )
    root_dir = base / f"asset={asset}"
    if not root_dir.exists():
        raise RuntimeError(
            f"scalar_features asset partition missing for interval={interval_min} asset={asset}; expected "
            f"{root_dir / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    latest = _latest_year_month_dir(root_dir)
    if latest is None:
        raise RuntimeError(
            f"scalar_features asset partition has no year/month partitions for interval={interval_min} asset={asset}; "
            f"expected pattern {root_dir / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    year, month, _ = latest
    d0 = root_dir / f"year={year}" / f"month={month:02d}"
    py, pm = _previous_year_month(year, month)
    d1 = root_dir / f"year={py}" / f"month={pm:02d}"
    p0 = _one_parquet_in_month_dir(d0)
    p1 = _one_parquet_in_month_dir(d1)
    checked = [str(d0), str(d1)]
    pq = p0 if p0 is not None else p1
    if pq is None:
        raise RuntimeError(
            f"scalar_features parquet missing for interval={interval_min} asset={asset}; checked={checked}; "
            f"regime clustering requires asset-partitioned scalar_features."
        )
    cache_key = _feature_column_cache_key(int(interval_min), str(asset), pq)
    cached = _FEATURE_COLUMN_DISCOVERY_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        discovered = _read_feature_columns_from_probe(pq)
    except Exception as exc:
        raise RuntimeError(
            f"scalar_features parquet unreadable for interval={interval_min} asset={asset}; path={pq}; error={exc}; "
            f"regime clustering requires asset-partitioned scalar_features."
        ) from exc
    if not discovered:
        raise RuntimeError(
            f"scalar_features schema empty for interval={interval_min} asset={asset}; checked={checked}; "
            f"expected feature columns beyond ts/asset under asset-partitioned layout."
        )
    if len(_FEATURE_COLUMN_DISCOVERY_CACHE) >= _FEATURE_COLUMN_DISCOVERY_CACHE_MAX:
        _FEATURE_COLUMN_DISCOVERY_CACHE.clear()
    _FEATURE_COLUMN_DISCOVERY_CACHE[cache_key] = list(discovered)
    return discovered


def discover_feature_columns_for_intervals(intervals: Sequence[int], asset: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for interval in intervals:
        out[interval] = discover_feature_columns(interval, asset=asset)
    return out


def build_aligned_features(
    asset: str,
    band: BandSpec,
    start_ts: int,
    end_ts: int,
    feature_bases: Sequence[str],
) -> pd.DataFrame:
    if start_ts > end_ts:
        return pd.DataFrame()
    lookback_seconds = ceiling_seconds(band.ceiling)
    window_start = max(HARD_OUTPUT_START_TS, start_ts - lookback_seconds)
    available = discover_feature_columns_for_intervals(band.member_intervals, asset=asset)
    if not any(len(available.get(k, [])) > 0 for k in band.member_intervals):
        raise RuntimeError(
            f"no discovered scalar feature columns for asset={asset} band={band.name}; "
            f"expected asset-partitioned scalar_features under "
            f"{PARQUET_ROOT / f'scalar_features_{band.ceiling}' / f'asset={asset}' / 'year=YYYY' / 'month=MM' / '<one parquet>'}"
        )
    interval_cols = {k: [c for c in feature_bases if c in available.get(k, [])] for k in band.member_intervals}
    if not any(len(interval_cols.get(k, [])) > 0 for k in band.member_intervals):
        raise RuntimeError(
            f"no usable discovered feature columns for asset={asset} band={band.name}; requested={list(feature_bases)}; "
            f"discovered={{{', '.join([f'{k}:{len(v)}' for k, v in available.items()])}}}"
        )
    ceiling_cols = interval_cols.get(band.ceiling, [])
    base_df = load_scalar_features_window(
        band.ceiling,
        asset,
        window_start,
        end_ts,
        columns=["ts", "asset", *ceiling_cols],
    )
    if base_df.empty:
        return pd.DataFrame()
    base_df = base_df[(base_df["ts"] >= HARD_OUTPUT_START_TS) & (base_df["ts"] >= start_ts) & (base_df["ts"] <= end_ts)]
    if base_df.empty:
        return pd.DataFrame()
    base_df = base_df.sort_values("ts")[["ts", "asset"] + [c for c in base_df.columns if c not in ("ts", "asset")]]
    base_df = base_df.rename(columns={c: f"i{band.ceiling}_{c}" for c in base_df.columns if c not in ("ts", "asset")})
    aligned = base_df[["ts", "asset"]].copy()
    aligned = aligned.sort_values("ts")
    if band.ceiling in band.member_intervals:
        for c in base_df.columns:
            if c not in ("ts", "asset"):
                aligned[c] = base_df[c].values
    for interval in band.member_intervals:
        if interval == band.ceiling:
            continue
        cols = interval_cols.get(interval, [])
        if not cols:
            continue
        df = load_scalar_features_window(
            interval,
            asset,
            window_start,
            end_ts,
            columns=["ts", "asset", *cols],
        )
        if df.empty:
            continue
        df = df[df["ts"] >= HARD_OUTPUT_START_TS]
        if df.empty:
            continue
        df = df.sort_values("ts")
        rename_cols = {c: f"i{interval}_{c}" for c in cols}
        df = df.rename(columns=rename_cols)
        merge_cols = ["ts"] + [rename_cols[c] for c in cols]
        aligned = pd.merge_asof(
            aligned.sort_values("ts"),
            df[merge_cols],
            on="ts",
            direction="backward",
            tolerance=lookback_seconds,
        )
    return aligned


def robust_scale_fit(df: pd.DataFrame, feature_cols: Sequence[str], standardize: bool = False) -> Tuple[object, np.ndarray, pd.DataFrame]:
    clean = df.dropna(subset=list(feature_cols)).copy()
    if clean.empty:
        return RobustScaler(), np.empty((0, len(feature_cols))), clean
    scaler = StandardScaler() if bool(standardize) else RobustScaler()
    x = scaler.fit_transform(clean[list(feature_cols)].astype(float).values)
    return scaler, x, clean


def fit_cluster_model(
    asset: str,
    band: BandSpec,
    category: str,
    train_start: int,
    train_end: int,
    feature_strategy: str,
    feature_subset: Sequence[str],
    corr_thresh: float,
    n_per_interval: int,
    min_cluster_size: int,
    min_samples: int,
    raw_train: Optional[pd.DataFrame] = None,
    diagnostics: Optional[DiagnosticCollector] = None,
    standardize: bool = False,
) -> Optional[dict]:
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for regime clustering.")
    if category not in CATEGORY_SPECS:
        raise RuntimeError(f"unknown category: {category}")

    # Keep API compatibility with existing call sites; category partition controls actual columns.
    _ = (feature_strategy, feature_subset, corr_thresh)
    if raw_train is None:
        raw_train = build_aligned_features(asset, band, train_start, train_end, ALL_CATEGORY_BASES)
    if raw_train.empty:
        log(f"[fit][warn] no training samples asset={asset} band={band.name} category={category}")
        return None

    effective_min_samples = int(HDBSCAN_MIN_SAMPLES)
    effective_epsilon = float(epsilon_for_band(band.name))
    bases = list(CATEGORY_SPECS[category]["bases"])
    feature_cols = [
        f"i{int(iv)}_{base}"
        for iv in band.member_intervals
        for base in bases
        if f"i{int(iv)}_{base}" in raw_train.columns
    ]
    if not feature_cols:
        log(f"[fit][warn] no feature columns selected asset={asset} band={band.name} category={category}")
        return None

    # Training-only inactivity filter using the existing flatness logic.
    rows_total = int(len(raw_train))
    rows_inactive = 0
    rows_active = rows_total
    filter_applied = False
    train_df_for_fit = raw_train
    inactive_mask, has_flat_cols = flat_activity_mask(raw_train, band)
    if has_flat_cols:
        rows_inactive = int(inactive_mask.fillna(False).sum())
        rows_active = int(rows_total - rows_inactive)
        filtered = raw_train.loc[~inactive_mask].copy()
        min_rows_needed = int(max(int(min_cluster_size) * 3, int(effective_min_samples) * 3, 100))
        if int(len(filtered)) >= int(min_rows_needed):
            train_df_for_fit = filtered
            filter_applied = True
            rows_active = int(len(filtered))
        else:
            train_df_for_fit = raw_train
            filter_applied = False
    log(
        f"[fit] asset={asset} band={band.name} category={category} refit_ts={int(train_end)} "
        f"rows={rows_total} inactive={rows_inactive} active={rows_active} filter_applied={filter_applied}"
    )
    log(
        f"[fit] asset={asset} band={band.name} category={category} refit_ts={int(train_end)} "
        f"min_cluster_size={int(min_cluster_size)} min_samples={int(effective_min_samples)} "
        f"cluster_selection_method=eom epsilon={effective_epsilon} metric=euclidean prediction_data=True"
    )

    if len(train_df_for_fit) > n_per_interval:
        train_df_for_fit = train_df_for_fit.sample(n=n_per_interval, random_state=17)

    scaler, x_scaled, clean = robust_scale_fit(train_df_for_fit, feature_cols, standardize=bool(standardize))
    if x_scaled.size == 0:
        log(f"[fit][warn] no clean samples after scaling asset={asset} band={band.name} category={category}")
        if diagnostics is not None:
            diagnostics.record_fit(band, category, train_df_for_fit, feature_cols, None)
        return None
    n_points = int(x_scaled.shape[0])
    min_required = int(max(2, effective_min_samples, min_cluster_size))
    if n_points < min_required:
        log(
            f"[fit][warn] insufficient points asset={asset} band={band.name} category={category} "
            f"n={n_points} required={min_required}"
        )
        if diagnostics is not None:
            diagnostics.record_fit(band, category, train_df_for_fit, feature_cols, None)
        return None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        cluster_selection_method="eom",
        cluster_selection_epsilon=effective_epsilon,
        metric="euclidean",
        allow_single_cluster=True,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x_scaled)
    has_clusters = bool(np.any(labels != -1))
    probabilities = clusterer.probabilities_
    outlier_scores = getattr(clusterer, "outlier_scores_", np.full(len(labels), np.nan))

    mapping = build_cluster_mapping(category, labels, clean, feature_cols)
    feature_schema_hash = schema_hash(feature_cols)
    depth_stats = compute_cluster_depth_stats(labels, x_scaled)
    centroids: Dict[int, np.ndarray] = {}
    for cid in sorted({int(l) for l in labels if int(l) != -1}):
        mask = labels == cid
        if np.any(mask):
            centroids[int(cid)] = np.nanmean(x_scaled[mask], axis=0)

    model_out = {
        "category": category,
        "band": str(band.name),
        "clusterer": clusterer,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "feature_schema_hash": feature_schema_hash,
        "mapping": mapping,
        "depth_stats": depth_stats,
        "centroids": centroids,
        "has_clusters": has_clusters,
        "train_meta": {
            "train_start": int(train_start),
            "train_end": int(train_end),
            "n_samples": int(len(labels)),
            "labels": labels,
            "probabilities": probabilities,
            "outlier_scores": outlier_scores,
        },
        "hdbscan_params": {
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "cluster_selection_epsilon": float(effective_epsilon),
            "min_samples": int(effective_min_samples),
            "min_cluster_size": int(min_cluster_size),
            "prediction_data": True,
            "standardize": bool(standardize),
        },
    }
    if diagnostics is not None:
        diagnostics.record_fit(band, category, train_df_for_fit, feature_cols, model_out)
    return model_out


def schema_hash(feature_cols: Sequence[str]) -> str:
    joined = ",".join(feature_cols).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def build_cluster_mapping(
    category: str,
    labels: np.ndarray,
    raw_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[int, str]:
    cluster_ids = sorted({int(l) for l in labels if l != -1})
    if not cluster_ids:
        return {}
    scores: Dict[int, float] = {}
    for cid in cluster_ids:
        mask = labels == cid
        if not np.any(mask):
            scores[cid] = float("nan")
            continue
        vals = raw_df.iloc[mask][list(feature_cols)].to_numpy(dtype=float)
        scores[cid] = float(np.nanmedian(vals))

    valid = [(cid, s) for cid, s in scores.items() if np.isfinite(s)]
    if not valid:
        return {cid: "unknown" for cid in scores.keys()}
    valid = sorted(valid, key=lambda x: x[1])
    out: Dict[int, str] = {cid: "unknown" for cid in scores.keys()}
    labels_vocab = list(CATEGORY_SPECS[category]["labels"])
    n = len(valid)
    if n == 1:
        out[valid[0][0]] = labels_vocab[1]
        return out
    if n == 2:
        low_cid = valid[0][0]
        high_cid = valid[1][0]
        out[low_cid] = labels_vocab[0]
        out[high_cid] = labels_vocab[2]
        return out
    t1 = int(math.floor(n / 3))
    t2 = int(math.floor(2 * n / 3))
    for idx, (cid, _score) in enumerate(valid):
        if idx < t1:
            rank = 0
        elif idx < t2:
            rank = 1
        else:
            rank = 2
        out[cid] = labels_vocab[rank]
    return out


def compute_cluster_depth_stats(labels: np.ndarray, x_scaled: np.ndarray) -> Dict[int, Dict[str, object]]:
    stats: Dict[int, Dict[str, object]] = {}
    for cid in sorted({int(l) for l in labels if l != -1}):
        mask = labels == cid
        if not np.any(mask):
            continue
        cluster_points = x_scaled[mask]
        center = np.nanmean(cluster_points, axis=0)
        dists = np.linalg.norm(cluster_points - center, axis=1)
        p50 = float(np.nanpercentile(dists, 50)) if dists.size else float("nan")
        p95 = float(np.nanpercentile(dists, 95)) if dists.size else float("nan")
        stats[cid] = {"center": center, "p50": p50, "p95": p95}
    return stats


def assign_category_clusters(
    df: pd.DataFrame,
    model: dict,
) -> pd.DataFrame:
    category = str(model.get("category"))
    spec = CATEGORY_SPECS[category]
    cluster_col = str(spec["cluster_col"])
    label_col = str(spec["label_col"])
    confidence_col = str(spec["confidence_col"])
    intensity_col = str(spec["intensity_col"])
    feature_cols = model["feature_cols"]
    scaler = model["scaler"]
    mapping = model["mapping"]
    depth_stats = model.get("depth_stats", {})
    x = df[feature_cols].astype(float).values
    x_scaled = scaler.transform(x)
    centroids: Dict[int, np.ndarray] = model.get("centroids", {}) or {}
    p50_by_cid: Dict[int, float] = {}
    p95_by_cid: Dict[int, float] = {}
    for cid, info in (depth_stats.items() if isinstance(depth_stats, dict) else []):
        try:
            p50 = float((info or {}).get("p50", np.nan))
        except Exception:
            p50 = float("nan")
        try:
            p95 = float((info or {}).get("p95", np.nan))
        except Exception:
            p95 = float("nan")
        if np.isfinite(p50) and p50 > 0:
            p50_by_cid[int(cid)] = p50
        if np.isfinite(p95) and p95 > 0:
            p95_by_cid[int(cid)] = p95
    if not centroids:
        labels = np.full((x_scaled.shape[0],), -1, dtype=int)
        nearest_dist = np.full((x_scaled.shape[0],), np.nan, dtype=float)
    else:
        cids = sorted([int(k) for k in centroids.keys()])
        cmat = np.vstack([np.asarray(centroids[cid], dtype=float) for cid in cids])
        d2 = np.sum((x_scaled[:, None, :] - cmat[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        labels = np.asarray([cids[int(i)] for i in idx], dtype=int)
        nearest_dist = np.sqrt(d2[np.arange(x_scaled.shape[0]), idx])
    out = df[["ts", "asset"]].copy()
    out[cluster_col] = labels.astype(int)
    out[confidence_col] = 0
    out[intensity_col] = 0
    # Centroid-distance confidence/intensity: normalized nearest distance in scaled space.
    band_name = str(model.get("band", "")).strip().lower()
    conf_mode = str(CONFIDENCE_MAPPING).strip().lower()
    # Calibration exception from discovery: micro/activity retains current mapping.
    if band_name == "micro" and str(category).strip().lower() == "activity":
        conf_mode = "current"
    for cid in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
        mask = labels == cid
        if not np.any(mask):
            continue
        dists = nearest_dist[mask]
        p50 = float(p50_by_cid.get(int(cid), np.nan))
        p95 = float(p95_by_cid.get(int(cid), np.nan))
        if not np.isfinite(p50) or p50 <= 0:
            p50 = 1.0
        if not np.isfinite(p95) or p95 <= 0:
            p95 = 1.0
        if conf_mode in {"current", "linear"}:
            conf = 1.0 - (dists / p95)
            conf = np.clip(conf, 0.0, 1.0)
        elif conf_mode == "soft":
            conf = 1.0 / (1.0 + (dists / p50))
            conf = np.clip(conf, 0.0, 1.0)
        elif conf_mode == "exponential":
            conf = np.exp(-(dists / p50))
            conf = np.clip(conf, 0.0, 1.0)
        else:
            conf = 1.0 - (dists / p95)
            conf = np.clip(conf, 0.0, 1.0)
        score = (100 * conf).round().astype(int)
        out.loc[mask, confidence_col] = score
        out.loc[mask, intensity_col] = score
    out[label_col] = "unknown"
    for cid, label in mapping.items():
        out.loc[out[cluster_col] == int(cid), label_col] = str(label)
    out.loc[out[cluster_col] == -1, [label_col, confidence_col, intensity_col]] = ["unknown", 0, 0]
    return out


def unknown_category_frame(
    base: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    spec = CATEGORY_SPECS[category]
    cluster_col = str(spec["cluster_col"])
    label_col = str(spec["label_col"])
    confidence_col = str(spec["confidence_col"])
    intensity_col = str(spec["intensity_col"])
    out = base[["ts", "asset"]].copy()
    out[cluster_col] = -1
    out[label_col] = "unknown"
    out[confidence_col] = 0
    out[intensity_col] = 0
    return out


def combined_feature_schema_hash(models_by_category: Dict[str, Optional[dict]]) -> str:
    chunks = []
    for category in CATEGORY_ORDER:
        model = models_by_category.get(category)
        chunks.append(f"{category}:{(model or {}).get('feature_schema_hash', 'unknown')}")
    return schema_hash(chunks)


def assign_range_outputs(
    asset: str,
    band: BandSpec,
    models_by_category: Dict[str, Optional[dict]],
    start_ts: int,
    end_ts: int,
    diagnostics: Optional[DiagnosticCollector] = None,
) -> pd.DataFrame:
    step = ceiling_seconds(band.ceiling)
    ts0 = floor_to_ceiling(int(start_ts), band.ceiling)
    if ts0 < int(start_ts):
        ts0 += step
    if ts0 > int(end_ts):
        return pd.DataFrame()
    base = pd.DataFrame({"ts": np.arange(ts0, int(end_ts) + step, step, dtype=np.int64)})
    base["asset"] = asset
    df = build_aligned_features(asset, band, start_ts, end_ts, feature_bases=ALL_CATEGORY_BASES)
    if df.empty:
        merged = base
    else:
        merged = base.merge(df, on=["ts", "asset"], how="left")
    flat_mask, _has_flat_cols = flat_activity_mask(merged, band)
    flat_bar_count = int(flat_mask.sum())
    assigned = base[["ts", "asset"]].copy()
    for category in CATEGORY_ORDER:
        model = models_by_category.get(category)
        spec = CATEGORY_SPECS[category]
        cluster_col = str(spec["cluster_col"])
        label_col = str(spec["label_col"])
        centroid_candidate_rows = 0
        centroid_assigned_rows = 0
        centroid_left_unknown_rows = 0
        no_model_or_no_cluster_rows = 0
        incomplete_feature_rows = 0
        if model is None or not bool(model.get("has_clusters", False)):
            category_frame = unknown_category_frame(base, category)
            no_model_or_no_cluster_rows = int(len(base))
        else:
            feature_cols = list(model["feature_cols"])
            for c in feature_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
            complete_mask = ~merged[feature_cols].isna().any(axis=1)
            category_parts: List[pd.DataFrame] = []
            if complete_mask.any():
                pred_frame = assign_category_clusters(merged.loc[complete_mask].copy(), model)
                centroid_candidate_rows = int(len(pred_frame))
                centroid_assigned_rows = int((pred_frame[cluster_col] != -1).sum())
                centroid_left_unknown_rows = int((pred_frame[cluster_col] == -1).sum())
                category_parts.append(pred_frame)
            if (~complete_mask).any():
                incomplete_feature_rows = int((~complete_mask).sum())
                category_parts.append(unknown_category_frame(merged.loc[~complete_mask, ["ts", "asset"]], category))
            category_frame = pd.concat(category_parts, ignore_index=True).sort_values("ts").reset_index(drop=True)
        final_unknown_rows = int((category_frame[label_col].astype(str) == "unknown").sum())
        spec_conf_col = str(spec["confidence_col"])
        known_conf = category_frame.loc[
            category_frame[label_col].astype(str) != "unknown",
            spec_conf_col,
        ].to_numpy(dtype=float)
        if diagnostics is not None:
            diagnostics.record_assign(
                band=band,
                category=category,
                bars_total=int(len(base)),
                flat_bar_count=flat_bar_count,
                centroid_candidate_rows=centroid_candidate_rows,
                overridden_to_neutral_rows=0,
                final_unknown_rows=final_unknown_rows,
                centroid_assigned_rows=centroid_assigned_rows,
                centroid_left_unknown_rows=centroid_left_unknown_rows,
                no_model_or_no_cluster_rows=no_model_or_no_cluster_rows,
                incomplete_feature_rows=incomplete_feature_rows,
                confidence_values=known_conf,
            )
        assigned = assigned.merge(category_frame, on=["ts", "asset"], how="left")
    assigned["band"] = band.name
    assigned["ceiling_interval_min"] = band.ceiling
    assigned["feature_schema_hash"] = combined_feature_schema_hash(models_by_category)
    return assigned


def build_unknown_outputs(asset: str, band: BandSpec, start_ts: int, end_ts: int, feature_schema_hash: str = "unknown") -> pd.DataFrame:
    step = ceiling_seconds(band.ceiling)
    ts0 = floor_to_ceiling(int(start_ts), band.ceiling)
    if ts0 < int(start_ts):
        ts0 += step
    if ts0 > int(end_ts):
        return pd.DataFrame()
    out = pd.DataFrame({"ts": np.arange(ts0, int(end_ts) + step, step, dtype=np.int64)})
    out["asset"] = asset
    out["band"] = band.name
    out["ceiling_interval_min"] = band.ceiling
    for category in CATEGORY_ORDER:
        spec = CATEGORY_SPECS[category]
        out[str(spec["cluster_col"])] = -1
        out[str(spec["label_col"])] = "unknown"
        out[str(spec["confidence_col"])] = 0
        out[str(spec["intensity_col"])] = 0
    out["feature_schema_hash"] = feature_schema_hash
    return out


def write_regime_outputs(
    df: pd.DataFrame,
    ceiling_interval: int,
    root: Path,
    on_chunk_committed: Optional[Callable[[int], None]] = None,
    expected_start_ts: Optional[int] = None,
    expected_end_ts: Optional[int] = None,
    expected_asset: Optional[str] = None,
) -> int:
    if df.empty:
        return 0
    df = df[df["ts"] >= HARD_OUTPUT_START_TS]
    if df.empty:
        return 0
    ts_dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.copy()
    df["year"] = ts_dt.dt.year
    df["month"] = ts_dt.dt.month
    rows = 0
    step = ceiling_seconds(int(ceiling_interval))
    for (y, m), grp in df.groupby(["year", "month"]):
        out = grp.drop(columns=["year", "month"]).drop_duplicates(subset=["asset", "ts", "band"], keep="last")
        out = out.sort_values("ts").reset_index(drop=True)
        if expected_asset is not None:
            out = out[out["asset"].astype(str) == str(expected_asset)].copy()
        if out.empty:
            continue
        ts_arr = pd.to_numeric(out["ts"], errors="coerce").dropna().astype("int64").to_numpy()
        if ts_arr.size == 0:
            continue
        if ts_arr.size > 1:
            d = np.diff(ts_arr)
            if not np.all(d == int(step)):
                first_bad = int(ts_arr[np.where(d != int(step))[0][0] + 1])
                raise RuntimeError(
                    f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                    f"month={int(y):04d}-{int(m):02d} first_bad_ts={first_bad} error=non_step_grid"
                )
            if not np.all(d > 0):
                first_bad = int(ts_arr[np.where(d <= 0)[0][0] + 1])
                raise RuntimeError(
                    f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                    f"month={int(y):04d}-{int(m):02d} first_bad_ts={first_bad} error=non_increasing_ts"
                )
        month_start = int(pd.Timestamp(year=int(y), month=int(m), day=1, tz="UTC").timestamp())
        if int(m) == 12:
            month_end = int(pd.Timestamp(year=int(y) + 1, month=1, day=1, tz="UTC").timestamp()) - int(step)
        else:
            month_end = int(pd.Timestamp(year=int(y), month=int(m) + 1, day=1, tz="UTC").timestamp()) - int(step)
        intended_start = max(int(ts_arr[0]), int(expected_start_ts)) if expected_start_ts is not None else int(ts_arr[0])
        intended_end = min(int(ts_arr[-1]), int(expected_end_ts)) if expected_end_ts is not None else int(ts_arr[-1])
        intended_start = max(intended_start, month_start)
        intended_end = min(intended_end, month_end)
        if int(ts_arr[0]) != int(intended_start):
            raise RuntimeError(
                f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                f"month={int(y):04d}-{int(m):02d} first_bad_ts={int(ts_arr[0])} "
                f"error=head_mismatch expected={int(intended_start)}"
            )
        if int(ts_arr[-1]) != int(intended_end):
            raise RuntimeError(
                f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                f"month={int(y):04d}-{int(m):02d} first_bad_ts={int(ts_arr[-1])} "
                f"error=tail_mismatch expected={int(intended_end)}"
            )
        out_asset = str(out["asset"].iloc[0])
        dst = parquet_path_for(ceiling_interval, out_asset, int(y), int(m), root)
        write_parquet_atomic(out, dst)
        validate_attempt = 0
        while True:
            validate_attempt += 1
            try:
                chk = pd.read_parquet(dst, columns=["asset", "ts", "band"])
                if expected_asset is not None:
                    chk = chk[chk["asset"].astype(str) == str(expected_asset)]
                chk = chk[chk["band"].astype(str) == str(out["band"].iloc[0])]
                chk = chk.sort_values("ts")
                if chk.empty:
                    raise RuntimeError("empty_post_write")
                chk_ts = pd.to_numeric(chk["ts"], errors="coerce").dropna().astype("int64").to_numpy()
                if chk_ts.size == 0:
                    raise RuntimeError("empty_ts_post_write")
                if chk_ts.size > 1:
                    dchk = np.diff(chk_ts)
                    if not np.all(dchk > 0):
                        first_bad = int(chk_ts[np.where(dchk <= 0)[0][0] + 1])
                        raise RuntimeError(f"non_increasing_post_write first_bad_ts={first_bad}")
                    if not np.all(dchk == int(step)):
                        first_bad = int(chk_ts[np.where(dchk != int(step))[0][0] + 1])
                        raise RuntimeError(f"non_step_post_write first_bad_ts={first_bad}")
                if int(chk_ts[-1]) != int(ts_arr[-1]):
                    raise RuntimeError(f"tail_post_write_mismatch got={int(chk_ts[-1])} expected={int(ts_arr[-1])}")
                break
            except Exception as exc:
                if validate_attempt >= 2:
                    raise RuntimeError(
                        f"[hard-stop] asset={expected_asset or 'unknown'} k={ceiling_interval} "
                        f"month={int(y):04d}-{int(m):02d} first_bad_ts={int(ts_arr[0])} error={exc}"
                    )
                write_parquet_atomic(out, dst)
        if on_chunk_committed is not None and not out.empty:
            on_chunk_committed(int(out["ts"].max()))
        rows += len(out)
        log(f"[parquet] regimes_{ceiling_interval} {y}-{int(m):02d}: wrote {len(out):,} rows -> {dst}")
    return rows


def definition_is_valid(model: Optional[dict], cursor_ts: int, cursor_refit_key: str, category: str) -> bool:
    if model is None or not isinstance(model, dict):
        return False
    if str(model.get("category", "")) != str(category):
        return False
    if "clusterer" not in model or "scaler" not in model:
        return False
    feature_cols = model.get("feature_cols", [])
    if not feature_cols:
        return False
    meta = model.get("meta", {})
    if not isinstance(meta, dict):
        return False
    valid_until = int(meta.get("valid_until", -1))
    refit_key = str(meta.get("refit_key", ""))
    if valid_until < int(cursor_ts):
        return False
    if refit_key != str(cursor_refit_key):
        return False
    return True


def walk_forward_asset_band(
    asset: str,
    band: BandSpec,
    feature_strategy: str,
    feature_subset: Sequence[str],
    corr_thresh: float,
    n_per_interval: int,
    min_cluster_size_override: Optional[int],
    min_samples: int,
    diagnostics: Optional[DiagnosticCollector] = None,
    standardize: bool = False,
    output_limits: Optional[RegimeOutputLimits] = None,
) -> dict:
    wall_start = time.perf_counter()
    min_ts, max_ts = feature_bounds_from_parquet(band.ceiling, asset)
    if min_ts is None or max_ts is None:
        log(f"[walk][skip] no source data asset={asset} band={band.name}")
        return {}
    src_min = max(int(min_ts), HARD_OUTPUT_START_TS)
    source_tail = floor_to_ceiling(int(max_ts), band.ceiling)
    if int(source_tail) < int(src_min):
        log(f"[walk][skip] no source data at/after hard edge asset={asset} band={band.name}")
        return {}

    _dst_min, dst_tail_raw = regime_bounds_from_parquet(band.ceiling, asset)
    if dst_tail_raw is not None and int(dst_tail_raw) > int(source_tail):
        raise RuntimeError(
            f"[hard-stop] asset={asset} band={band.name} dst_tail={int(dst_tail_raw)} ahead_of_source_tail={int(source_tail)}"
        )
    if dst_tail_raw is not None and int(dst_tail_raw) == int(source_tail):
        log(f"[walk][skip] at edge asset={asset} band={band.name} tail={int(source_tail)}")
        return {"last_assigned_ceiling_ts": int(source_tail), "last_refit_ceiling_ts": int(source_tail)}

    step = ceiling_seconds(band.ceiling)
    start_ts = ceil_to_ceiling(src_min, band.ceiling) if dst_tail_raw is None else int(dst_tail_raw) + int(step)
    end_ts = int(source_tail)
    start_ts, end_ts, output_limit_state = apply_regime_output_limits(start_ts, end_ts, band, output_limits)
    if int(start_ts) > int(end_ts):
        log(f"[walk][skip] empty range asset={asset} band={band.name} start={int(start_ts)} end={int(end_ts)}")
        return {
            "last_assigned_ceiling_ts": int(dst_tail_raw or 0),
            "last_refit_ceiling_ts": int(dst_tail_raw or 0),
            "rows_written": 0,
            "refit_events": 0,
            "definition_write_events": 0,
            "parquet_write_events": 0,
            "timings_s": {
                "total": round(float(time.perf_counter() - wall_start), 6),
                "feature_read": 0.0,
                "fit": 0.0,
                "definition_write": 0.0,
                "assign": 0.0,
                "parquet_write": 0.0,
            },
            "output_limits": output_limit_state,
        }

    models_by_category: Dict[str, Optional[dict]] = {}
    for category in CATEGORY_ORDER:
        models_by_category[category] = load_definition(asset, band.name, category)

    cursor = int(start_ts)
    total_rows = 0
    last_refit_ts = int(dst_tail_raw) if dst_tail_raw is not None else int(start_ts)
    last_reason = "reused"
    band_mcs = int(min_cluster_size_override) if min_cluster_size_override is not None else int(mcs_for_band(band.name))
    refit_events = 0
    definition_write_events = 0
    parquet_write_events = 0
    feature_read_s = 0.0
    fit_s = 0.0
    definition_write_s = 0.0
    assign_s = 0.0
    parquet_write_s = 0.0

    while int(cursor) <= int(end_ts):
        cursor_refit_key = cadence_refit_key(int(cursor), band)
        boundary = next_boundary_start_ts(int(cursor), band)
        chunk_end = min(int(end_ts), int(boundary) - int(step))
        if int(chunk_end) < int(cursor):
            cursor = int(cursor) + int(step)
            continue

        reason = "reused"
        categories_to_refit = [
            category
            for category in CATEGORY_ORDER
            if not definition_is_valid(models_by_category.get(category), int(cursor), str(cursor_refit_key), category)
        ]
        if categories_to_refit:
            refit_ts = int(cursor)
            train_end = floor_to_ceiling(int(refit_ts), band.ceiling)
            train_start_raw = max(HARD_OUTPUT_START_TS, int(train_end) - int(band.train_days) * SECONDS_PER_DAY)
            train_start = ceil_to_ceiling(int(train_start_raw), band.ceiling)
            if int(train_start) > int(train_end):
                train_start = int(train_end)
            read_start = time.perf_counter()
            raw_train = build_aligned_features(asset, band, train_start, train_end, ALL_CATEGORY_BASES)
            feature_read_s += float(time.perf_counter() - read_start)
            reason = "fit_unavailable"
            if not raw_train.empty:
                refit_key = cadence_refit_key(int(refit_ts), band)
                model_valid_until = definition_valid_until_for_refit(int(refit_ts), band)
                any_ok = False
                for category in categories_to_refit:
                    refit_events += 1
                    fit_start = time.perf_counter()
                    model = fit_cluster_model(
                        asset,
                        band,
                        category,
                        train_start,
                        train_end,
                        feature_strategy,
                        feature_subset,
                        corr_thresh,
                        n_per_interval,
                        band_mcs,
                        min_samples,
                        raw_train=raw_train,
                        diagnostics=diagnostics,
                        standardize=bool(standardize),
                    )
                    fit_s += float(time.perf_counter() - fit_start)
                    if model is not None:
                        any_ok = True
                        meta = {
                            "asset": asset,
                            "band": band.name,
                            "category": category,
                            "valid_from": int(refit_ts),
                            "last_refit_ts": int(refit_ts),
                            "refit_key": str(refit_key),
                            "valid_until": int(model_valid_until),
                            "feature_cols": model["feature_cols"],
                            "feature_schema_hash": model["feature_schema_hash"],
                            "hdbscan_params": model.get("hdbscan_params", {}),
                        }
                        model["meta"] = meta
                        models_by_category[category] = model
                        definition_write_start = time.perf_counter()
                        save_definition(asset, band.name, category, model, meta)
                        definition_write_s += float(time.perf_counter() - definition_write_start)
                        definition_write_events += 1
                    else:
                        models_by_category[category] = None
                if any_ok:
                    reason = "ok"
                    last_refit_ts = int(refit_ts)
                else:
                    reason = "no_clusters"

        if any(m is not None and bool(m.get("has_clusters", False)) for m in models_by_category.values()):
            assign_start = time.perf_counter()
            outputs = assign_range_outputs(
                asset,
                band,
                models_by_category,
                int(cursor),
                int(chunk_end),
                diagnostics=diagnostics,
            )
            assign_s += float(time.perf_counter() - assign_start)
        else:
            assign_start = time.perf_counter()
            outputs = build_unknown_outputs(
                asset,
                band,
                int(cursor),
                int(chunk_end),
                feature_schema_hash=combined_feature_schema_hash(models_by_category),
            )
            assign_s += float(time.perf_counter() - assign_start)
            if diagnostics is not None and not outputs.empty:
                bars = int(len(outputs))
                for category in CATEGORY_ORDER:
                    diagnostics.record_assign(
                        band=band,
                        category=category,
                        bars_total=bars,
                        flat_bar_count=0,
                        centroid_candidate_rows=0,
                        overridden_to_neutral_rows=0,
                        final_unknown_rows=bars,
                        centroid_assigned_rows=0,
                        centroid_left_unknown_rows=bars,
                        no_model_or_no_cluster_rows=bars,
                        incomplete_feature_rows=0,
                        confidence_values=None,
                    )
        def _record_parquet_commit(_tail_ts: int) -> None:
            nonlocal parquet_write_events
            parquet_write_events += 1

        parquet_write_start = time.perf_counter()
        rows = write_regime_outputs(
            outputs,
            band.ceiling,
            REGIME_PARQUET_ROOT,
            on_chunk_committed=_record_parquet_commit,
            expected_start_ts=int(cursor),
            expected_end_ts=int(chunk_end),
            expected_asset=asset,
        )
        parquet_write_s += float(time.perf_counter() - parquet_write_start)
        total_rows += int(rows)
        tail_written = int(outputs["ts"].max()) if not outputs.empty else int(cursor) - int(step)
        if int(tail_written) != int(chunk_end):
            raise RuntimeError(
                f"[hard-stop] asset={asset} band={band.name} write_tail={int(tail_written)} expected_tail={int(chunk_end)}"
            )
        last_reason = reason
        cursor = int(chunk_end) + int(step)

    log(
        f"[walk] asset={asset} band={band.name} source_tail={int(source_tail)} dst_tail={dst_tail_raw} "
        f"range=[{int(start_ts)},{int(end_ts)}] rows={int(total_rows)} refits={int(refit_events)} "
        f"writes={int(parquet_write_events)} reason={last_reason}"
    )
    return {
        "last_assigned_ceiling_ts": int(end_ts),
        "last_refit_ceiling_ts": int(last_refit_ts),
        "rows_written": int(total_rows),
        "refit_events": int(refit_events),
        "definition_write_events": int(definition_write_events),
        "parquet_write_events": int(parquet_write_events),
        "reason": str(last_reason),
        "timings_s": {
            "total": round(float(time.perf_counter() - wall_start), 6),
            "feature_read": round(float(feature_read_s), 6),
            "fit": round(float(fit_s), 6),
            "definition_write": round(float(definition_write_s), 6),
            "assign": round(float(assign_s), 6),
            "parquet_write": round(float(parquet_write_s), 6),
        },
        "output_limits": output_limit_state,
    }


def process_asset(
    asset: str,
    band_specs: List[BandSpec],
    feature_strategy: str,
    feature_subset: Optional[List[str]],
    corr_thresh: float,
    n_per_interval: int,
    min_cluster_size_override: Optional[int],
    min_samples: int,
    standardize: bool = False,
    output_limits: Optional[RegimeOutputLimits] = None,
) -> dict:
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for regime clustering.")
    asset_start = time.perf_counter()
    worker_start_epoch = time.time()
    worker_start_utc = datetime.now(timezone.utc).isoformat()
    worker_start_resource = resource_snapshot(include_thread_snapshot=True)
    band_states: dict = {}
    band_errors: List[Dict[str, str]] = []
    diagnostics = DiagnosticCollector(asset=str(asset))
    diagnostic_write_events = 0
    diagnostic_write_s = 0.0
    for band in band_specs:
        try:
            band_state = walk_forward_asset_band(
                asset,
                band,
                feature_strategy,
                feature_subset or DEFAULT_FEATURE_SUBSET,
                corr_thresh,
                n_per_interval,
                min_cluster_size_override,
                min_samples,
                diagnostics=diagnostics,
                standardize=bool(standardize),
                output_limits=output_limits,
            )
            if band_state:
                band_states[band.name] = band_state
        except Exception as exc:
            log(f"[worker][error] asset={asset} band={band.name}: {exc}")
            band_errors.append({"band": str(band.name), "error": str(exc)})
    diagnostics_rows = diagnostics.to_json_ready()
    if diagnostics_rows:
        payload = {
            "asset": str(asset),
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": diagnostics_rows,
        }
        diagnostics_dir = Path(REGIME_LABEL_IO_CONFIG.diagnostic_root) / "regime_clustering"
        diagnostics_path = diagnostics_dir / f"{asset}_regime_diagnostics.json"
        diagnostic_write_start = time.perf_counter()
        write_json(diagnostics_path, payload)
        diagnostic_write_s += float(time.perf_counter() - diagnostic_write_start)
        diagnostic_write_events += 1
        for row in diagnostics_rows:
            log(
                "[diag] "
                f"asset={row['asset']} band={row['band']} category={row['category']} "
                f"bars_total={row['bars_total']} flat_bar_fraction={float(row['flat_bar_fraction']):.6f} "
                f"clusters_found={int(row.get('clustering', {}).get('clusters_found', 0))} "
                f"final_unknown_fraction={float(row.get('assignments', {}).get('final_unknown_fraction', 1.0)):.6f}"
            )
        log(f"[diag] asset={asset} wrote {len(diagnostics_rows)} category records -> {diagnostics_path}")
    total_s = round(float(time.perf_counter() - asset_start), 6)
    worker_end_epoch = time.time()
    worker_end_utc = datetime.now(timezone.utc).isoformat()
    worker_end_resource = resource_snapshot(include_thread_snapshot=True)
    return {
        "asset": asset,
        "band_states": band_states,
        "errors": band_errors,
        "timings_s": {
            "total": total_s,
            "diagnostic_write": round(float(diagnostic_write_s), 6),
        },
        "rows_written": int(sum(int(state.get("rows_written", 0)) for state in band_states.values())),
        "refit_events": int(sum(int(state.get("refit_events", 0)) for state in band_states.values())),
        "definition_write_events": int(
            sum(int(state.get("definition_write_events", 0)) for state in band_states.values())
        ),
        "parquet_write_events": int(sum(int(state.get("parquet_write_events", 0)) for state in band_states.values())),
        "diagnostic_write_events": int(diagnostic_write_events),
        "worker_telemetry": worker_resource_telemetry_record(
            module="regime_clustering",
            run_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(asset))),
            task_id=f"regime_clustering|asset={asset}",
            work_unit_id=f"regime_clustering|{asset}",
            status="failed" if band_errors else "completed",
            start_time_utc=worker_start_utc,
            end_time_utc=worker_end_utc,
            start_epoch_s=worker_start_epoch,
            end_epoch_s=worker_end_epoch,
            resource_start=worker_start_resource,
            resource_end=worker_end_resource,
            phase_timings={"asset_total_s": total_s, "diagnostic_write_s": float(diagnostic_write_s)},
            identity={
                "asset": str(asset),
                "target": "regime_labels",
                "model_family": "regime_clustering",
                "profile": str(feature_strategy),
                "band": ",".join(str(band.name) for band in band_specs),
            },
            error=("; ".join(str(item.get("error", "")) for item in band_errors) if band_errors else None),
        ),
    }


def run(
    assets: Optional[List[str]] = None,
    bands: Optional[List[str]] = None,
    feature_strategy: str = "subset",
    feature_subset: Optional[List[str]] = None,
    corr_thresh: float = 0.9,
    n_per_interval: int = 5000,
    min_cluster_size_override: Optional[int] = None,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    standardize: bool = False,
    workers: int = 1,
    output_limits: Optional[RegimeOutputLimits] = None,
    label_io_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for regime clustering.")
    configure_regime_label_generation_io(label_io_args)
    run_start = time.perf_counter()
    feature_subset = feature_subset or DEFAULT_FEATURE_SUBSET
    band_specs = _band_specs_for_names(bands)
    if not assets:
        assets = sorted(assets_present_in_features(DEFAULT_INTERVALS[-1]))
    asset_results: List[Dict[str, Any]] = []
    if workers <= 1:
        for asset in assets:
            asset_results.append(
                process_asset(
                    asset,
                    band_specs,
                    feature_strategy,
                    feature_subset,
                    corr_thresh,
                    n_per_interval,
                    min_cluster_size_override,
                    min_samples,
                    standardize,
                    output_limits,
                )
            )
    else:
        pending_assets: List[str] = list(assets)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_configure_worker_regime_label_generation_io,
                initargs=(REGIME_LABEL_IO_CONFIG, REGIME_LABEL_SANDBOX_ROOTS),
            ) as pool:
                future_map = {}
                for asset in assets:
                    fut = pool.submit(
                        process_asset,
                        asset,
                        band_specs,
                        feature_strategy,
                        feature_subset,
                        corr_thresh,
                        n_per_interval,
                        min_cluster_size_override,
                        min_samples,
                        standardize,
                        output_limits,
                    )
                    future_map[fut] = asset
                for fut in as_completed(future_map):
                    asset = future_map[fut]
                    try:
                        asset_results.append(fut.result())
                        if asset in pending_assets:
                            pending_assets.remove(asset)
                    except BrokenProcessPool as exc:
                        log(f"[run][warn] process pool broke ({exc}); retrying remaining assets serially")
                        break
                    except Exception as exc:
                        log(f"[run][worker-error] asset={asset} err={exc}")
                        if asset in pending_assets:
                            pending_assets.remove(asset)
        except BrokenProcessPool as exc:
            log(f"[run][warn] process pool failed to initialize ({exc}); running serially")
        if pending_assets:
            for asset in pending_assets:
                try:
                    asset_results.append(
                        process_asset(
                            asset,
                            band_specs,
                            feature_strategy,
                            feature_subset,
                            corr_thresh,
                            n_per_interval,
                            min_cluster_size_override,
                            min_samples,
                            standardize,
                            output_limits,
                        )
                    )
                except Exception as exc:
                    log(f"[run][serial-error] asset={asset} err={exc}")
                    asset_results.append({"asset": str(asset), "band_states": {}, "errors": [{"error": str(exc)}]})
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "module_slug": "regime_clustering",
        "assets": [str(asset) for asset in assets],
        "bands": [str(band.name) for band in band_specs],
        "workers": max(1, int(workers)),
        "label_generation_io": REGIME_LABEL_IO_CONFIG.to_json_ready(),
        "output_limits": (output_limits or RegimeOutputLimits()).to_json_ready(),
        "elapsed_seconds": round(float(time.perf_counter() - run_start), 6),
        "rows_written": int(sum(int(result.get("rows_written", 0)) for result in asset_results)),
        "refit_events": int(sum(int(result.get("refit_events", 0)) for result in asset_results)),
        "definition_write_events": int(
            sum(int(result.get("definition_write_events", 0)) for result in asset_results)
        ),
        "parquet_write_events": int(sum(int(result.get("parquet_write_events", 0)) for result in asset_results)),
        "diagnostic_write_events": int(
            sum(int(result.get("diagnostic_write_events", 0)) for result in asset_results)
        ),
        "worker_resource_telemetry": {
            "count": int(sum(1 for result in asset_results if isinstance(result.get("worker_telemetry"), dict))),
            "records": [result["worker_telemetry"] for result in asset_results if isinstance(result.get("worker_telemetry"), dict)][:100],
        },
        "asset_results": asset_results,
    }
    log(
        "[summary] "
        f"assets={len(assets)} bands={len(band_specs)} workers={max(1, int(workers))} "
        f"elapsed_s={float(summary['elapsed_seconds']):.3f} rows={int(summary['rows_written'])} "
        f"refits={int(summary['refit_events'])} parquet_writes={int(summary['parquet_write_events'])}"
    )
    return summary


def parse_list(val: Optional[str]) -> Optional[List[str]]:
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster scalar features into regime labels per asset/band.")
    parser.add_argument("--assets", type=str, default=None, help="Comma-separated assets (default: discover)")
    parser.add_argument("--bands", type=str, default=None, help="Comma-separated band names (micro,meso,macro)")
    parser.add_argument("--feature-strategy", type=str, default="subset", choices=["subset", "decorrelate"])
    parser.add_argument("--feature-subset", type=str, default=None, help="Comma-separated feature list")
    parser.add_argument("--corr-thresh", type=float, default=0.9)
    parser.add_argument("--n-per-interval", type=int, default=5000)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None, help="Alias override for min cluster size")
    parser.add_argument("--min-samples", type=int, default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--epsilon-override", type=float, default=None, help="Temporary global epsilon override for all bands")
    parser.add_argument(
        "--confidence-mapping",
        type=str,
        default=None,
        choices=["current", "linear", "soft", "exponential"],
        help="Centroid confidence mapping rule",
    )
    parser.add_argument("--standardize", action="store_true", help="Use z-score standardization instead of robust scaling")
    parser.add_argument(
        "--runtime-profile",
        type=str,
        default="runtime_config",
        choices=sorted(REGIME_CLUSTERING_RUNTIME_PROFILES),
        help="Named Regime label-generation runtime profile; defaults to module-keyed runtime config.",
    )
    parser.add_argument(
        "--sandbox-output-root",
        type=str,
        default=None,
        help="Route Regime label-generation writes under this sandbox root.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count")
    parser.add_argument(
        "--output-start",
        type=str,
        default=None,
        help="Optional UTC timestamp/date lower bound for labels written by this run.",
    )
    parser.add_argument(
        "--output-end",
        type=str,
        default=None,
        help="Optional UTC timestamp/date upper bound for labels written by this run.",
    )
    parser.add_argument(
        "--max-output-months",
        type=int,
        default=None,
        help="Optional calendar-month cap per asset/band, counted from the resolved output start.",
    )
    parser.add_argument(
        "--runtime-summary-json",
        type=str,
        default=None,
        help="Optional path for resolved runtime and in-process timing/write counters.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate requested assets/bands and estimate scalar feature inputs without fitting or writing regimes.",
    )
    parser.add_argument(
        "--preflight-json",
        type=str,
        default=None,
        help="Optional path for the preflight JSON report.",
    )
    parser.add_argument(
        "--allow-missing-preflight-assets",
        action="store_true",
        help="Allow preflight-only mode to exit successfully when requested assets are missing source partitions.",
    )
    args = parser.parse_args()
    io_config = configure_regime_label_generation_io(args)
    try:
        runtime_profile = resolve_regime_clustering_runtime_profile(args)
        workers = resolve_regime_clustering_workers(args, runtime_profile=runtime_profile)
        output_limits = resolve_regime_output_limits(args, runtime_profile=runtime_profile)
        validate_regime_clustering_runtime_guardrails(
            args,
            workers=workers,
            runtime_profile=runtime_profile,
            output_limits=output_limits,
        )
    except ValueError as exc:
        parser.error(str(exc))
    env_mcs = os.getenv("REGIME_MCS", "").strip()
    resolved_mcs: Optional[int] = None
    if env_mcs:
        try:
            resolved_mcs = int(env_mcs)
        except Exception:
            resolved_mcs = None
    if args.min_cluster_size is not None:
        resolved_mcs = int(args.min_cluster_size)
    if args.hdbscan_min_cluster_size is not None:
        resolved_mcs = int(args.hdbscan_min_cluster_size)
    env_eps = os.getenv("REGIME_EPSILON_OVERRIDE", "").strip()
    resolved_eps: Optional[float] = None
    if env_eps:
        try:
            resolved_eps = float(env_eps)
        except Exception:
            resolved_eps = None
    if args.epsilon_override is not None:
        resolved_eps = float(args.epsilon_override)
    global EPSILON_OVERRIDE
    EPSILON_OVERRIDE = resolved_eps
    env_conf_map = str(os.getenv("REGIME_CONFIDENCE_MAPPING", "")).strip().lower()
    resolved_conf_map = "current"
    if env_conf_map in {"current", "linear", "soft", "exponential"}:
        resolved_conf_map = env_conf_map
    if args.confidence_mapping is not None:
        resolved_conf_map = str(args.confidence_mapping).strip().lower()
    global CONFIDENCE_MAPPING
    CONFIDENCE_MAPPING = resolved_conf_map
    env_standardize = str(os.getenv("REGIME_STANDARDIZE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    resolved_standardize = bool(args.standardize) or bool(env_standardize)
    runtime_snapshot = resolve_regime_clustering_runtime_snapshot(
        args,
        workers=workers,
        runtime_profile=runtime_profile,
        output_limits=output_limits,
    )
    log_resolved_runtime("regime_clustering", resolved=runtime_snapshot)
    requested_assets = parse_list(args.assets)
    requested_bands = parse_list(args.bands)
    preflight: Optional[Dict[str, Any]] = None
    if args.preflight_only or args.preflight_json or requested_assets:
        preflight = resolve_regime_clustering_preflight(
            args,
            workers=workers,
            assets=requested_assets,
            bands=requested_bands,
            runtime_profile=runtime_profile,
        )
    if (args.preflight_only or args.preflight_json) and preflight is not None:
        payload_text = json.dumps(preflight, indent=2, sort_keys=True)
        if args.preflight_json:
            preflight_path = Path(args.preflight_json)
            regime_artifacts.write_json(
                preflight_path,
                preflight,
                write_kind="diagnostics",
                sandbox_roots=REGIME_LABEL_SANDBOX_ROOTS,
            )
            log(f"[preflight] wrote runtime preflight report -> {preflight_path}")
        if args.preflight_only:
            print(payload_text)
            if preflight["status"] != "ok" and not bool(args.allow_missing_preflight_assets):
                raise SystemExit(2)
            return
    if requested_assets and preflight is not None and preflight["status"] != "ok":
        log(f"[preflight][error] missing requested source partitions: {preflight['missing_assets']}")
        raise SystemExit(2)
    run_summary = run(
        assets=requested_assets,
        bands=requested_bands,
        feature_strategy=args.feature_strategy,
        feature_subset=parse_list(args.feature_subset),
        corr_thresh=args.corr_thresh,
        n_per_interval=args.n_per_interval,
        min_cluster_size_override=resolved_mcs,
        min_samples=args.min_samples,
        standardize=resolved_standardize,
        workers=workers,
        output_limits=output_limits,
        label_io_args=args,
    )
    if args.runtime_summary_json:
        summary_path = Path(args.runtime_summary_json)
        payload = {
            "runtime_snapshot": runtime_snapshot,
            "run_summary": run_summary,
            "label_generation_io": io_config.to_json_ready(),
        }
        regime_artifacts.write_json(
            summary_path,
            payload,
            write_kind="runtime",
            sandbox_roots=REGIME_LABEL_SANDBOX_ROOTS,
        )
        log(f"[summary] wrote runtime summary -> {summary_path}")


if __name__ == "__main__":
    main()
