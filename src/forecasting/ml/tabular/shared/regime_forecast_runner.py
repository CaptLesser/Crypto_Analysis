from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.features.scalar_features import (
    OHLCVT_PARQUET_ROOT,
    PARQUET_COMPRESSION,
    PARQUET_ROW_GROUP,
    PARQUET_ROOT as SCALAR_PARQUET_ROOT,
)
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.runtime_config import get_workers
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots
from src.regimes.contracts import REGIME_AXIS_ORDER, REGIME_DEFAULT_FORECAST_INTERVALS, axis_target, forecast_ceiling_interval


@dataclass(frozen=True)
class RegimeFamilyModuleSpec:
    module_slug: str
    family_name: str
    prediction_prefix: str
    log_prefix: str
    parquet_root_env: str
    progress_seconds_env: str = ""
    source_start_env: str = ""
    source_end_env: str = ""
    work_start_env: str = ""
    forecast_resume_edge_env: str = ""
    default_unit_workers: int = 1
    default_model_threads: int = 1
    max_logical_threads: int = 1
    thread_env_vars: Sequence[str] = ()
    thread_param_name: str = "n_jobs"
    default_intervals: Sequence[int] = REGIME_DEFAULT_FORECAST_INTERVALS
    default_horizon_minutes: Sequence[int] = (30, 240, 720)
    axes: Sequence[str] = REGIME_AXIS_ORDER
    select_feature_columns_fn: Optional[Callable[..., List[str]]] = None
    fit_classifier_fn: Optional[Callable[..., Any]] = None
    predict_classifier_fn: Optional[Callable[..., Any]] = None
    resolve_classifier_params_fn: Optional[Callable[..., Dict[str, Any]]] = None
    classifier_profile_label_fn: Optional[Callable[..., str]] = None
    default_training_window_months_for_combo_fn: Optional[Callable[[int, int, str], int]] = None
    training_window_bars_for_pair_fn: Optional[Callable[[str, int, int], int]] = None
    training_window_bars_from_months_fn: Optional[Callable[[int, int], int]] = None


@dataclass(frozen=True)
class RegimeForecastIOConfig:
    parquet_root: Path
    staging_root: Path
    state_root: Path
    scalar_root: Path
    ohlc_root: Path
    regime_label_root: Path
    parquet_compression: str = PARQUET_COMPRESSION
    parquet_row_group: int = PARQUET_ROW_GROUP
    log_fn: Callable[[str], None] = print


@dataclass(frozen=True)
class RegimeForecastPlan:
    intervals: tuple[int, ...]
    resolved_regime_ceiling_intervals: tuple[int, ...]
    horizon_minutes: tuple[int, ...]
    axes: tuple[str, ...]
    mode: str
    unit_workers: int


def _csv_ints(raw: object, default: Sequence[int]) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(int(x) for x in default)
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return values or tuple(int(x) for x in default)


def _csv_axes(raw: object, default: Sequence[str]) -> tuple[str, ...]:
    text = str(raw or "").strip()
    axes = tuple(str(part.strip()) for part in text.split(",") if part.strip()) if text else tuple(str(axis) for axis in default)
    for axis in axes:
        axis_target(axis)
    return axes


def build_regime_arg_parser(spec: RegimeFamilyModuleSpec, *, description: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or f"Walk-forward {spec.family_name} regime forecasting from canonical axis labels."
    )
    parser.add_argument("--profile", type=str, default=selected_profile())
    parser.add_argument("--intervals", type=str, default="")
    parser.add_argument("--assets", type=str, default="", help="Comma-delimited assets")
    parser.add_argument("--horizon-minutes", type=str, default="")
    parser.add_argument("--axes", type=str, default="", help="Comma-delimited Regime axes")
    parser.add_argument("--mode", type=str, default="incremental", choices=["incremental", "backfill"])
    parser.add_argument("--unit-workers", type=int, default=get_workers(spec.module_slug, "unit_workers", spec.default_unit_workers))
    return parser


def resolve_regime_run_plan(spec: RegimeFamilyModuleSpec, args: argparse.Namespace) -> RegimeForecastPlan:
    intervals = _csv_ints(getattr(args, "intervals", ""), spec.default_intervals)
    horizons = _csv_ints(getattr(args, "horizon_minutes", ""), spec.default_horizon_minutes)
    axes = _csv_axes(getattr(args, "axes", ""), spec.axes)
    return RegimeForecastPlan(
        intervals=tuple(int(interval) for interval in intervals),
        resolved_regime_ceiling_intervals=tuple(sorted({int(forecast_ceiling_interval(interval)) for interval in intervals})),
        horizon_minutes=tuple(int(horizon) for horizon in horizons),
        axes=tuple(str(axis) for axis in axes),
        mode=str(getattr(args, "mode", "incremental") or "incremental"),
        unit_workers=max(1, int(getattr(args, "unit_workers", spec.default_unit_workers) or spec.default_unit_workers)),
    )


def regime_unit_meta(
    spec: RegimeFamilyModuleSpec,
    *,
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    horizon_bars: int,
    asset: str,
    interval: int,
) -> dict:
    return {
        "family": str(family),
        "domain": str(domain),
        "task": str(task),
        "module_slug": str(spec.module_slug),
        "prediction_prefix": str(spec.prediction_prefix),
        "axes": [str(axis) for axis in spec.axes],
        "horizon_minutes": int(horizon_minutes),
        "horizon_bars": int(horizon_bars),
        "asset": str(asset),
        "interval": int(interval),
        "requested_interval": int(interval),
        "resolved_regime_ceiling_interval": int(forecast_ceiling_interval(interval)),
    }


def regime_manifest_base(
    spec: RegimeFamilyModuleSpec,
    plan: RegimeForecastPlan,
    *,
    run_id: str,
    family: str,
    domain: str,
    task: str,
) -> dict:
    return {
        "run_id": str(run_id),
        "mode": str(plan.mode),
        "family": str(family),
        "domain": str(domain),
        "task": str(task),
        "module_slug": str(spec.module_slug),
        "prediction_prefix": str(spec.prediction_prefix),
        "axes": list(plan.axes),
        "intervals": list(plan.intervals),
        "resolved_regime_ceiling_intervals": list(plan.resolved_regime_ceiling_intervals),
        "horizon_minutes": list(plan.horizon_minutes),
    }


def _sandbox_env_path(roots: SandboxOutputRoots, env_name: str, fallback: Path, kind: str, env: Mapping[str, str]) -> Path:
    raw = str(env.get(env_name, "") or "").strip()
    path = Path(raw) if raw else Path(fallback)
    assert_write_allowed(path, kind, roots=roots)
    return path


def _sandbox_resolution_env(env: Mapping[str, str]) -> Dict[str, str]:
    out = {str(key): str(value) for key, value in env.items()}
    raw_root = str(out.get("PIPELINE_SANDBOX_OUTPUT_ROOT", "") or "").strip()
    raw_pipeline_root = str(out.get("PIPELINE_ROOT", "") or "").strip()
    if raw_root and raw_pipeline_root:
        try:
            if Path(raw_root).expanduser().resolve() == Path(raw_pipeline_root).expanduser().resolve():
                out.pop("PIPELINE_ROOT", None)
        except Exception:
            pass
    return out


def resolve_regime_io_config(
    spec: RegimeFamilyModuleSpec,
    *,
    env: Optional[Mapping[str, str]] = None,
    profile: Optional[str] = None,
    log_fn: Callable[[str], None] = print,
) -> RegimeForecastIOConfig:
    source_env = env if env is not None else os.environ
    pipeline_profile = str(profile or selected_profile(env=source_env))
    path_config_env = dict(source_env)
    sandbox_roots = resolve_sandbox_output_roots(env=_sandbox_resolution_env(source_env))

    if sandbox_roots.enabled:
        parquet_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_PARQUET_ROOT", sandbox_roots.parquet_root, "tabular regime parquet root", source_env)
        state_base = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_STATE_ROOT", sandbox_roots.state_root, "tabular regime state root", source_env)
        tmp_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_TMP_ROOT", sandbox_roots.tmp_root, "tabular regime tmp root", source_env)
        state_root = state_base / "model_states" / spec.module_slug
        staging_root = tmp_root / f"{spec.module_slug}_stage"
        for path, kind in (
            (state_root, "tabular regime model state root"),
            (staging_root, "tabular regime staging root"),
        ):
            assert_write_allowed(path, kind, roots=sandbox_roots)
    else:
        legacy_pipeline_root = Path(source_env["PIPELINE_ROOT"]) if str(source_env.get("PIPELINE_ROOT", "")).strip() else None
        state_base = resolve_path("state_root", profile=pipeline_profile, env=path_config_env, required=False) or (
            legacy_pipeline_root / "model_states" if legacy_pipeline_root is not None else Path("model_states")
        )
        tmp_root = resolve_path("tmp_root", profile=pipeline_profile, env=path_config_env, required=False) or (
            legacy_pipeline_root / "tmp" if legacy_pipeline_root is not None else Path("tmp")
        )
        parquet_root = Path(
            source_env.get(spec.parquet_root_env)
            or resolve_path("output_parquet_root", profile=pipeline_profile, env=path_config_env, required=False)
            or source_env.get("PIPELINE_PARQUET_ROOT", "parquet")
        )
        state_root = Path(state_base) / spec.module_slug
        staging_root = Path(tmp_root) / f"{spec.module_slug}_stage"

    scalar_root = Path(
        source_env.get("PIPELINE_SOURCE_FEATURES_ROOT")
        or source_env.get("PIPELINE_PARQUET_FEATURES_ROOT")
        or resolve_path("source_feature_root", profile=pipeline_profile, env=path_config_env, required=False)
        or SCALAR_PARQUET_ROOT
    )
    ohlc_root = Path(
        source_env.get("PIPELINE_SOURCE_OHLCVT_ROOT")
        or source_env.get("PIPELINE_SOURCE_PARQUET_ROOT")
        or resolve_path("source_ohlcvt_root", profile=pipeline_profile, env=path_config_env, required=False)
        or OHLCVT_PARQUET_ROOT
    )
    regime_label_root = Path(
        source_env.get("PIPELINE_SOURCE_REGIME_ROOT")
        or source_env.get("PIPELINE_PARQUET_REGIME_ROOT")
        or resolve_path("source_regime_root", profile=pipeline_profile, env=path_config_env, required=False)
        or parquet_root
    )

    for path in (state_root, staging_root):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return RegimeForecastIOConfig(
        parquet_root=Path(parquet_root),
        staging_root=Path(staging_root),
        state_root=Path(state_root),
        scalar_root=Path(scalar_root),
        ohlc_root=Path(ohlc_root),
        regime_label_root=Path(regime_label_root),
        log_fn=log_fn,
    )
