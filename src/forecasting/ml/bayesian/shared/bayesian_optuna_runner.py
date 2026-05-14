from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

from src.forecasting.common.forecast_family_core import discover_edge_and_min, read_feature_window_columns
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.stats_module_utils import NUMERIC_TASK_TO_TARGET_COLUMN
from src.forecasting.ml.bayesian.shared.bayesian_stage1_profile import resolve_execution_profile
from src.forecasting.ml.bayesian.shared.bayesian_numeric_cohort import FIXED_BAYESIAN_NUMERIC_COHORT
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.test_branch_function_telemetry import (
    emit_event_for_path,
    emit_stage3_study_summary_for_path,
    telemetry_scope_for_path,
)


@dataclass(frozen=True)
class BayesianOptunaModelSpec:
    model_key: str
    display_name: str
    numerics_module_import_path: str
    optuna_profile_import_path: str
    default_study_name_prefix: str
    default_trials_per_combo: int = 24
    default_recent_eval_days: int = 30
    default_history_window_months: int = 12
    default_model_threads: int = 6


CURRENT_MODEL_SPEC: Optional[BayesianOptunaModelSpec] = None
CURRENT_NUMERICS: Any = None
CURRENT_OPTUNA_PROFILE: Any = None
_STAGE3_SETUP_CACHE_MAX_ENTRIES = 64
_STAGE3_SETUP_CACHE_MAX_BYTES = 512 * 1024 * 1024


def configure_for_model(model_spec: BayesianOptunaModelSpec) -> None:
    global CURRENT_MODEL_SPEC, CURRENT_NUMERICS, CURRENT_OPTUNA_PROFILE
    CURRENT_MODEL_SPEC = model_spec
    CURRENT_NUMERICS = importlib.import_module(model_spec.numerics_module_import_path)
    CURRENT_OPTUNA_PROFILE = importlib.import_module(model_spec.optuna_profile_import_path)


@dataclass(frozen=True)
class ComboSpec:
    interval: int
    horizon_minutes: int
    task: str

    @property
    def tuple_label(self) -> str:
        return f"{int(self.interval)}:{int(self.horizon_minutes)}:{self.task}"

    @property
    def horizon_bars(self) -> int:
        hm = int(self.horizon_minutes)
        iv = int(self.interval)
        if hm <= 0 or iv <= 0 or hm % iv != 0:
            raise ValueError(f"invalid horizon pair {hm}/{iv}")
        return hm // iv


@dataclass
class Dataset:
    asset: str
    combo: ComboSpec
    frame: pd.DataFrame
    target_col: str
    history_start_ts: int
    eval_start_ts: int
    eval_end_ts: int
    origins: List[int]
    factor_map: Dict[int, float]
    selected_dynamic_feature_columns: Tuple[str, ...] = ()
    use_dynamic_features: bool = False
    use_seasonality: bool = False


@dataclass
class MetricResult:
    combo: str
    asset: str
    rows: int
    rmse: Optional[float]
    mae: Optional[float]
    first_prediction_ts: Optional[int]
    last_prediction_ts: Optional[int]
    params_label: str


class _Stage3SetupCache:
    def __init__(self, *, max_entries: int = _STAGE3_SETUP_CACHE_MAX_ENTRIES, max_bytes: int = _STAGE3_SETUP_CACHE_MAX_BYTES) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._entries: "OrderedDict[Tuple[Any, ...], Tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._oversize_skips = 0

    @staticmethod
    def _object_bytes(value: Any) -> int:
        if isinstance(value, pd.DataFrame):
            try:
                return int(value.memory_usage(index=True, deep=True).sum())
            except Exception:
                return int(getattr(value, "size", 0)) * 8
        if isinstance(value, dict):
            return max(4096, sum(len(inner) for inner in value.values() if isinstance(inner, dict)) * 24)
        return 4096

    @staticmethod
    def _copy_value(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return value.copy(deep=True)
        if isinstance(value, dict):
            return {
                key: (dict(inner) if isinstance(inner, dict) else inner)
                for key, inner in value.items()
            }
        return value

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._oversize_skips = 0

    def get(self, key: Tuple[Any, ...]) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, _byte_size = entry
        self._entries.move_to_end(key)
        self._hits += 1
        return self._copy_value(value)

    def put(self, key: Tuple[Any, ...], value: Any) -> None:
        stored = self._copy_value(value)
        byte_size = self._object_bytes(stored)
        if byte_size > self._max_bytes:
            self._oversize_skips += 1
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= int(old[1])
        self._entries[key] = (stored, byte_size)
        self._bytes += byte_size
        self._puts += 1
        while self._entries and (len(self._entries) > self._max_entries or self._bytes > self._max_bytes):
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self._bytes -= int(evicted_size)
            self._evictions += 1

    def stats(self) -> Dict[str, int]:
        return {
            "cache_entries": int(len(self._entries)),
            "cache_bytes_estimate": int(self._bytes),
            "cache_hit_count": int(self._hits),
            "cache_miss_count": int(self._misses),
            "cache_put_count": int(self._puts),
            "cache_eviction_count": int(self._evictions),
            "cache_oversize_skip_count": int(self._oversize_skips),
        }


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    if CURRENT_MODEL_SPEC is None:
        raise RuntimeError("Bayesian optuna runner is not configured for a model.")
    parser = argparse.ArgumentParser(description=f"{CURRENT_MODEL_SPEC.display_name} Bayesian Stage 3 tuning")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--intervals", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--horizon-minutes", type=str, default="")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--stage2-manifest", type=Path, default=None)
    parser.add_argument("--stage2-survivor-json", type=Path, default=None)
    parser.add_argument("--trials-per-combo", type=int, default=int(CURRENT_MODEL_SPEC.default_trials_per_combo))
    parser.add_argument("--model-threads", type=int, default=int(CURRENT_MODEL_SPEC.default_model_threads))
    parser.add_argument("--sampler-seed", type=int, default=17)
    parser.add_argument("--study-name-prefix", type=str, default=str(CURRENT_MODEL_SPEC.default_study_name_prefix))
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--resume-study", action="store_true")
    parser.add_argument("--recent-eval-days", type=int, default=int(CURRENT_MODEL_SPEC.default_recent_eval_days))
    parser.add_argument("--history-window-months", type=int, default=int(CURRENT_MODEL_SPEC.default_history_window_months))
    parser.add_argument("--max-eval-origins", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _parse_int_csv(raw: str, default: Sequence[int]) -> List[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [int(v) for v in values] if values else [int(v) for v in default]


def _parse_str_csv(raw: str, default: Sequence[str]) -> List[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [str(v) for v in values] if values else [str(v) for v in default]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_stage2_survivor_json(args: argparse.Namespace) -> Optional[Path]:
    if args.stage2_survivor_json is not None:
        return Path(args.stage2_survivor_json).resolve()
    if args.stage2_manifest is not None:
        candidate = Path(args.stage2_manifest).resolve().parent / "stage3_survivor_handoff.json"
        if candidate.exists():
            return candidate
    return None


def _load_stage2_survivors(args: argparse.Namespace) -> Dict[str, Any]:
    survivor_json = _resolve_stage2_survivor_json(args)
    if survivor_json is None or not survivor_json.exists():
        raise RuntimeError("staged Bayesian Stage-3 run missing stage2 survivor artifact")
    return _load_json(survivor_json)


def _resolve_stage1_feature_profile_json(args: argparse.Namespace) -> Optional[Path]:
    if not bool(getattr(args, "staged", False)):
        return None
    payload = _load_stage2_survivors(args)
    raw = str(payload.get("feature_profile_json") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path.resolve() if path.is_absolute() else path.resolve()


def resolve_combo_specs(args: argparse.Namespace) -> List[ComboSpec]:
    if str(args.combo_list).strip():
        combos: List[ComboSpec] = []
        for token in [part.strip() for part in str(args.combo_list).split(",") if part.strip()]:
            interval, horizon, task = token.split(":", 2)
            combos.append(ComboSpec(interval=int(interval), horizon_minutes=int(horizon), task=str(task)))
        return combos
    if bool(getattr(args, "staged", False)):
        payload = _load_stage2_survivors(args)
        combos = [
            ComboSpec(interval=int(item["interval_minutes"]), horizon_minutes=int(item["horizon_minutes"]), task=str(item["task"]))
            for item in (payload.get("survivors") or [])
        ]
        return sorted({(combo.interval, combo.horizon_minutes, combo.task): combo for combo in combos}.values(), key=lambda combo: (combo.interval, combo.horizon_minutes, combo.task))
    intervals = _parse_int_csv(args.intervals, CURRENT_NUMERICS.MODULE_SPEC.default_intervals)
    horizons = _parse_int_csv(getattr(args, "horizon_minutes"), CURRENT_NUMERICS.MODULE_SPEC.default_horizons)
    tasks = _parse_str_csv(args.tasks, CURRENT_NUMERICS.MODULE_SPEC.default_tasks)
    combos = [ComboSpec(interval=int(interval), horizon_minutes=int(horizon), task=str(task)) for interval in intervals for horizon in horizons for task in tasks if int(horizon) % int(interval) == 0]
    return combos


def requested_assets(raw: str, args: Optional[argparse.Namespace] = None) -> List[str]:
    explicit = [part.strip() for part in str(raw).split(",") if part.strip()]
    if explicit:
        return explicit
    if args is not None and bool(getattr(args, "staged", False)):
        payload = _load_stage2_survivors(args)
        assets = [str(asset) for asset in (payload.get("cohort_assets") or []) if str(asset)]
        if assets:
            return assets
    return list(FIXED_BAYESIAN_NUMERIC_COHORT)


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (Path.cwd() / "logs" / "diagnostics" / CURRENT_MODEL_SPEC.model_key / f"run={utc_now_stamp()}").resolve()


def _source_ohlcvt_root() -> Path:
    profile = selected_profile(default="pipeline_test")
    return Path(resolve_path("source_ohlcvt_root", profile=profile, required=False) or Path("parquet"))


def _source_feature_root(fallback: Optional[Path] = None) -> Path:
    profile = selected_profile(default="pipeline_test")
    return Path(resolve_path("source_feature_root", profile=profile, required=False) or fallback or _source_ohlcvt_root())


def _path_stat_identity(path: Optional[Path]) -> Tuple[str, int, int, int]:
    if path is None:
        return ("", 0, 0, 0)
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        ctime_ns = getattr(stat, "st_ctime_ns", None)
        if ctime_ns is None:
            ctime_ns = int(float(getattr(stat, "st_ctime", 0.0)) * 1_000_000_000)
        return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns), int(ctime_ns))
    except Exception:
        return (str(path), 0, 0, 0)


def _setup_common_key(
    *,
    combo: ComboSpec,
    args: argparse.Namespace,
    feature_profile_json: Optional[Path],
    selected_feature_columns: Optional[Sequence[str]],
    history_start_ts: int,
    eval_end_ts: int,
    use_dynamic_features: bool,
    use_seasonality: bool,
) -> Tuple[Any, ...]:
    ohlc_root = _source_ohlcvt_root().resolve()
    feature_root = _source_feature_root(fallback=ohlc_root).resolve()
    selected_key = None if selected_feature_columns is None else tuple(str(col) for col in selected_feature_columns)
    return (
        "Bayesian_Numeric",
        str(CURRENT_MODEL_SPEC.model_key),
        str(getattr(CURRENT_NUMERICS.MODULE_SPEC, "module_key", CURRENT_MODEL_SPEC.model_key)),
        int(combo.interval),
        int(combo.horizon_minutes),
        str(combo.task),
        int(history_start_ts),
        int(eval_end_ts),
        int(getattr(args, "recent_eval_days", 0)),
        int(getattr(args, "history_window_months", 0)),
        int(getattr(args, "max_eval_origins", 0)),
        str(ohlc_root),
        str(feature_root),
        _path_stat_identity(feature_profile_json),
        selected_key,
        bool(use_dynamic_features),
        bool(use_seasonality),
        bool(CURRENT_NUMERICS.MODULE_SPEC.needs_factor_cache),
    )


def _evaluate_window(edge_ts: int, recent_eval_days: int, history_window_months: int) -> Tuple[int, int]:
    eval_end_ts = int(edge_ts)
    eval_start_ts = int(edge_ts) - int(recent_eval_days) * 86400
    history_start_ts = int(edge_ts) - int(history_window_months) * 31 * 86400
    return history_start_ts, eval_end_ts


def _load_asset_frame(asset: str, combo: ComboSpec, history_start_ts: int, eval_end_ts: int, selected_dynamic_feature_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    ohlc_columns = ["open", "high", "low", "close", "volume", "trades"]
    feature_columns = [NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]]
    if selected_dynamic_feature_columns is None:
        feature_columns.extend([str(c) for c in CURRENT_NUMERICS.MODULE_SPEC.dynamic_feature_candidates])
    else:
        feature_columns.extend([str(c) for c in selected_dynamic_feature_columns if str(c)])
    ohlc_frame = read_ohlcvt(
        root=_source_ohlcvt_root().resolve(),
        asset=str(asset),
        interval_min=int(combo.interval),
        start_ts=int(history_start_ts),
        end_ts=int(eval_end_ts),
        columns=["ts", "asset", *ohlc_columns],
    )
    feature_frame = read_feature_window_columns(
        root=_source_feature_root(fallback=_source_ohlcvt_root()).resolve(),
        interval_minutes=int(combo.interval),
        asset=str(asset),
        columns=list(dict.fromkeys(feature_columns)),
        start_ts=int(history_start_ts),
        end_ts=int(eval_end_ts),
    )
    merged = ohlc_frame.merge(feature_frame, on=["ts", "asset"], how="outer", sort=True)
    for column in [*ohlc_columns, *feature_columns]:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged.sort_values("ts").reset_index(drop=True)


def _label_frame(frame: pd.DataFrame, combo: ComboSpec) -> pd.DataFrame:
    labels, _stats = compute_future_labels(frame.loc[:, ["high", "low", "close"]].reset_index(drop=True), int(combo.horizon_bars), future_direction_deadzone=0.0)
    return labels


def _sample_origins(ts_values: Sequence[int], *, eval_start_ts: int, eval_end_ts: int, max_eval_origins: int) -> List[int]:
    eligible = [int(ts) for ts in ts_values if int(eval_start_ts) <= int(ts) <= int(eval_end_ts)]
    if len(eligible) <= int(max_eval_origins):
        return eligible
    idx = np.linspace(0, len(eligible) - 1, int(max_eval_origins)).astype(int)
    return [eligible[int(i)] for i in idx]


def _build_factor_maps(frames: Dict[str, pd.DataFrame], combo: ComboSpec) -> Dict[str, Dict[int, float]]:
    if not CURRENT_NUMERICS.MODULE_SPEC.needs_factor_cache:
        return {asset: {} for asset in frames}
    target_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]
    asset_maps = {asset: {int(ts): float(val) for ts, val in zip(pd.to_numeric(frame["ts"], errors="coerce"), pd.to_numeric(frame[target_col], errors="coerce")) if pd.notna(ts) and pd.notna(val)} for asset, frame in frames.items()}
    factor_maps: Dict[str, Dict[int, float]] = {}
    for asset, frame in frames.items():
        factor_map: Dict[int, float] = {}
        for ts in pd.to_numeric(frame["ts"], errors="coerce").dropna().astype("int64"):
            peer_vals = [mapping[int(ts)] for other_asset, mapping in asset_maps.items() if other_asset != asset and int(ts) in mapping]
            if peer_vals:
                factor_map[int(ts)] = float(np.mean(peer_vals))
        factor_maps[asset] = factor_map
    return factor_maps


def build_datasets(
    combo: ComboSpec,
    assets: Sequence[str],
    args: argparse.Namespace,
    telemetry_path: Optional[Path] = None,
    setup_cache: Optional[_Stage3SetupCache] = None,
) -> List[Dataset]:
    with telemetry_scope_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="build_datasets",
        module_name=__name__,
        phase_name="dataset_construction",
        parent_phase="objective_setup",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=len(assets),
        asset_count=len(assets),
    ) as scope:
        edges = []
        for asset in assets:
            edge_ts, _min_ts = discover_edge_and_min(asset=str(asset), interval_minutes=int(combo.interval))
            if edge_ts is not None:
                edges.append(int(edge_ts))
        if not edges:
            scope.update(reason_code="objective_dataset_empty", output_rows=0)
            raise RuntimeError(f"No edge timestamps available for combo={combo.tuple_label}")
        common_edge = min(edges)
        history_start_ts, eval_end_ts = _evaluate_window(common_edge, int(args.recent_eval_days), int(args.history_window_months))
        eval_start_ts = int(common_edge) - int(args.recent_eval_days) * 86400
        feature_profile_json = _resolve_stage1_feature_profile_json(args)
        combo_profile = (
            resolve_execution_profile(
                feature_profile_json,
                interval=int(combo.interval),
                horizon=int(combo.horizon_minutes),
                task=str(combo.task),
                dynamic_feature_candidates=CURRENT_NUMERICS.MODULE_SPEC.dynamic_feature_candidates,
                needs_dynamic_features=bool(CURRENT_NUMERICS.MODULE_SPEC.needs_dynamic_features),
                use_seasonality=bool(CURRENT_NUMERICS.MODULE_SPEC.use_seasonality),
            )
            if feature_profile_json is not None
            else None
        )
        frames: Dict[str, pd.DataFrame] = {}
        frame_keys: Dict[str, Tuple[Any, ...]] = {}
        for asset in assets:
            selected_dynamic_feature_columns = (
                tuple(str(value) for value in combo_profile.selected_dynamic_feature_columns)
                if combo_profile is not None and combo_profile.use_dynamic_features
                else ()
            ) if combo_profile is not None else None
            use_dynamic_features = (
                bool(combo_profile.use_dynamic_features)
                if combo_profile is not None
                else bool(CURRENT_NUMERICS.MODULE_SPEC.needs_dynamic_features)
            )
            use_seasonality = (
                bool(combo_profile.use_seasonality)
                if combo_profile is not None
                else bool(CURRENT_NUMERICS.MODULE_SPEC.use_seasonality)
            )
            common_key = _setup_common_key(
                combo=combo,
                args=args,
                feature_profile_json=feature_profile_json,
                selected_feature_columns=selected_dynamic_feature_columns,
                history_start_ts=int(history_start_ts),
                eval_end_ts=int(eval_end_ts),
                use_dynamic_features=use_dynamic_features,
                use_seasonality=use_seasonality,
            )
            frame_key = ("asset_frame", str(asset), common_key)
            frame_keys[str(asset)] = frame_key
            frame = setup_cache.get(frame_key) if setup_cache is not None else None
            if frame is None:
                frame = _load_asset_frame(
                    str(asset),
                    combo,
                    int(history_start_ts),
                    int(eval_end_ts),
                    selected_dynamic_feature_columns=selected_dynamic_feature_columns,
                )
                if setup_cache is not None and not frame.empty:
                    setup_cache.put(frame_key, frame)
            if frame.empty:
                continue
            label_key = ("labels", str(asset), common_key)
            labels = setup_cache.get(label_key) if setup_cache is not None else None
            if labels is None:
                labels = _label_frame(frame, combo)
                if setup_cache is not None:
                    setup_cache.put(label_key, labels)
            label_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]
            if label_col in frame.columns and label_col in labels.columns:
                frame = frame.drop(columns=[label_col])
            merged = pd.concat([frame.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
            if merged.columns.has_duplicates:
                merged = merged.loc[:, ~merged.columns.duplicated(keep="last")].copy()
            frames[str(asset)] = merged
        factor_key = (
            "factor_maps",
            tuple(str(asset) for asset in assets),
            tuple(frame_keys.get(str(asset)) for asset in assets),
            bool(CURRENT_NUMERICS.MODULE_SPEC.needs_factor_cache),
        )
        factor_maps = setup_cache.get(factor_key) if setup_cache is not None else None
        if factor_maps is None:
            factor_started = time.perf_counter()
            factor_maps = _build_factor_maps(frames, combo)
            emit_event_for_path(
                telemetry_path,
                family="Bayesian_Numeric",
                model=str(CURRENT_MODEL_SPEC.model_key),
                stage="stage3",
                function_name="_build_factor_maps",
                module_name=__name__,
                phase_name="factor_map_build",
                parent_phase="dataset_construction",
                combo_key=combo.tuple_label,
                interval_minutes=int(combo.interval),
                horizon_minutes=int(combo.horizon_minutes),
                training_window_months=int(combo.training_window_months),
                task=str(combo.task),
                elapsed_seconds=time.perf_counter() - factor_started,
                input_rows=sum(len(frame) for frame in frames.values()),
                output_rows=sum(len(mapping) for mapping in factor_maps.values()),
                asset_count=len(frames),
                cache_hit_count=0,
                cache_miss_count=1,
            )
            if setup_cache is not None:
                setup_cache.put(factor_key, factor_maps)
        else:
            emit_event_for_path(
                telemetry_path,
                family="Bayesian_Numeric",
                model=str(CURRENT_MODEL_SPEC.model_key),
                stage="stage3",
                function_name="_build_factor_maps",
                module_name=__name__,
                phase_name="factor_map_build",
                parent_phase="dataset_construction",
                combo_key=combo.tuple_label,
                interval_minutes=int(combo.interval),
                horizon_minutes=int(combo.horizon_minutes),
                training_window_months=int(combo.training_window_months),
                task=str(combo.task),
                elapsed_seconds=0.0,
                input_rows=sum(len(frame) for frame in frames.values()),
                output_rows=sum(len(mapping) for mapping in factor_maps.values()),
                asset_count=len(frames),
                cache_hit_count=1,
                cache_miss_count=0,
            )
        datasets: List[Dataset] = []
        label_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]
        for asset, frame in frames.items():
            label_values = frame[label_col]
            if isinstance(label_values, pd.DataFrame):
                label_values = label_values.iloc[:, -1]
            valid = frame[np.isfinite(pd.to_numeric(label_values, errors="coerce"))].copy()
            origins = _sample_origins(
                valid["ts"].astype("int64").tolist(),
                eval_start_ts=int(eval_start_ts),
                eval_end_ts=int(eval_end_ts - int(combo.horizon_minutes) * 60),
                max_eval_origins=int(args.max_eval_origins),
            )
            if not origins:
                continue
            datasets.append(
                Dataset(
                    asset=str(asset),
                    combo=combo,
                    frame=frame,
                    target_col=str(label_col),
                    history_start_ts=int(history_start_ts),
                    eval_start_ts=int(eval_start_ts),
                    eval_end_ts=int(eval_end_ts),
                    origins=origins,
                    factor_map=factor_maps.get(str(asset), {}),
                    selected_dynamic_feature_columns=(
                        tuple(str(value) for value in combo_profile.selected_dynamic_feature_columns)
                        if combo_profile is not None
                        else tuple(str(value) for value in CURRENT_NUMERICS.MODULE_SPEC.dynamic_feature_candidates)
                    ),
                    use_dynamic_features=(
                        bool(combo_profile.use_dynamic_features)
                        if combo_profile is not None
                        else bool(CURRENT_NUMERICS.MODULE_SPEC.needs_dynamic_features)
                    ),
                    use_seasonality=(
                        bool(combo_profile.use_seasonality)
                        if combo_profile is not None
                        else bool(CURRENT_NUMERICS.MODULE_SPEC.use_seasonality)
                    ),
                )
            )
        scope.update(
            output_rows=sum(len(dataset.frame) for dataset in datasets),
            reason_code="" if datasets else "objective_dataset_empty",
            artifact_profile_source=str(feature_profile_json or ""),
            selected_feature_count=max((len(dataset.selected_dynamic_feature_columns) for dataset in datasets), default=0),
            dynamic_feature_count=max((len(dataset.selected_dynamic_feature_columns) for dataset in datasets), default=0),
            **(setup_cache.stats() if setup_cache is not None else {}),
        )
        return datasets


def _build_datasets_with_telemetry(
    combo: ComboSpec,
    assets: Sequence[str],
    args: argparse.Namespace,
    telemetry_path: Optional[Path],
    setup_cache: Optional[_Stage3SetupCache] = None,
) -> List[Dataset]:
    try:
        return build_datasets(combo, assets, args, telemetry_path=telemetry_path, setup_cache=setup_cache)
    except TypeError as exc:
        if "telemetry_path" not in str(exc) and "setup_cache" not in str(exc):
            raise
        try:
            return build_datasets(combo, assets, args, telemetry_path=telemetry_path)
        except TypeError as retry_exc:
            if "telemetry_path" not in str(retry_exc):
                raise
            return build_datasets(combo, assets, args)


def evaluate_dataset(dataset: Dataset, params: Dict[str, Any], params_label: str) -> MetricResult:
    frame = dataset.frame.reset_index(drop=True)
    ts_vec = pd.to_numeric(frame["ts"], errors="coerce").fillna(-1).astype("int64").to_numpy()
    y_vec = pd.to_numeric(frame[dataset.target_col], errors="coerce").to_numpy(dtype=float)
    valid_target_idx = np.flatnonzero(np.isfinite(y_vec))
    feat_cols = [str(c) for c in dataset.selected_dynamic_feature_columns if str(c) in frame.columns]
    feat_matrix = None
    if dataset.use_dynamic_features and feat_cols:
        feat_frame = frame.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce")
        feat_cols = [str(col) for col in feat_cols if feat_frame[str(col)].notna().any()]
        if feat_cols:
            feat_matrix = feat_frame.loc[:, feat_cols].to_numpy(dtype=float)
    seasonal_period_bars = None
    if dataset.use_seasonality:
        from src.forecasting.common.forecast_family_core import seasonality_info

        seas = seasonality_info(
            parquet_root=_source_ohlcvt_root().resolve(),
            interval_minutes=int(dataset.combo.interval),
            asset=str(dataset.asset),
        )
        seasonal_period_bars = seas.get("seasonality_period_bars") if seas.get("seasonality_usable") else None
    predictions: List[float] = []
    actuals: List[float] = []
    pred_ts: List[int] = []
    for idx_origin, origin_ts in enumerate(dataset.origins):
        idx = int(np.searchsorted(ts_vec, int(origin_ts), side="right") - 1)
        if idx < 0:
            continue
        valid_pos = int(np.searchsorted(valid_target_idx, int(idx), side="right")) - 1
        if valid_pos < 47:
            continue
        hist_idx = valid_target_idx[: valid_pos + 1]
        y_hist = y_vec[hist_idx]
        x_hist = None
        x_last = None
        if dataset.use_dynamic_features:
            if feat_matrix is None:
                continue
            fmat = feat_matrix[hist_idx]
            med = np.nanmedian(fmat, axis=0)
            fmat = np.where(np.isfinite(fmat), fmat, med)
            x_hist = fmat
            x_last = fmat[-1]
        factor_hist = None
        factor_last = None
        if CURRENT_NUMERICS.MODULE_SPEC.needs_factor_cache:
            ts_hist = ts_vec[hist_idx]
            values = np.asarray([dataset.factor_map.get(int(ts), np.nan) for ts in ts_hist], dtype=float)
            if not np.isfinite(values).any():
                continue
            med = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            factor_hist = np.where(np.isfinite(values), values, med)
            factor_last = float(factor_hist[-1])
        try:
            qvals, _meta = CURRENT_NUMERICS.MODULE_SPEC.predict_fn(y_hist=y_hist, horizon_bars=int(dataset.combo.horizon_bars), quantiles=[0.1, 0.5, 0.9], seasonal_period_bars=(int(seasonal_period_bars) if seasonal_period_bars is not None else None), seed=17 + idx_origin, model_params=dict(params), x_hist=x_hist, x_last=x_last, factor_hist=factor_hist, factor_last=factor_last)
        except Exception:
            continue
        y_true = y_vec[idx]
        if not math.isfinite(float(y_true)):
            continue
        predictions.append(float(qvals.get(0.5, np.nan)))
        actuals.append(float(y_true))
        pred_ts.append(int(origin_ts))
    if not predictions:
        return MetricResult(combo=dataset.combo.tuple_label, asset=dataset.asset, rows=0, rmse=None, mae=None, first_prediction_ts=None, last_prediction_ts=None, params_label=params_label)
    pred = np.asarray(predictions, dtype=float)
    act = np.asarray(actuals, dtype=float)
    return MetricResult(combo=dataset.combo.tuple_label, asset=dataset.asset, rows=int(len(predictions)), rmse=float(np.sqrt(np.mean((pred - act) ** 2))), mae=float(np.mean(np.abs(pred - act))), first_prediction_ts=min(pred_ts), last_prediction_ts=max(pred_ts), params_label=params_label)


def summarize_metrics(metrics: Sequence[MetricResult]) -> Dict[str, Any]:
    total_rows = sum(int(metric.rows) for metric in metrics if metric.rmse is not None and metric.rows > 0)
    if total_rows <= 0:
        return {"rows": 0, "weighted_rmse": None, "weighted_mae": None}
    rmse_num = sum(float(metric.rmse) * int(metric.rows) for metric in metrics if metric.rmse is not None and metric.rows > 0)
    mae_num = sum(float(metric.mae) * int(metric.rows) for metric in metrics if metric.mae is not None and metric.rows > 0)
    return {"rows": int(total_rows), "weighted_rmse": float(rmse_num / total_rows), "weighted_mae": float(mae_num / total_rows)}


def baseline_params_with_threads(combo: ComboSpec, model_threads: int) -> Dict[str, Any]:
    return CURRENT_OPTUNA_PROFILE.resolve_baseline_params(task=str(combo.task), model_threads=int(model_threads), combo=combo)


def trial_params(trial: optuna.Trial, combo: ComboSpec) -> Dict[str, Any]:
    return CURRENT_OPTUNA_PROFILE.suggest_trial_params(trial, combo)


def finalize_model_params(params: Dict[str, Any], combo: ComboSpec) -> Dict[str, Any]:
    finalize = getattr(CURRENT_OPTUNA_PROFILE, "finalize_params", None)
    if callable(finalize):
        return dict(finalize(dict(params), combo))
    return dict(params)


def run_study_for_combo(combo: ComboSpec, datasets: Sequence[Dataset], *, trials_per_combo: int, sampler_seed: int, storage: Optional[str], study_name_prefix: str, resume_study: bool, model_threads: int, telemetry_path: Optional[Path] = None) -> Tuple[Dict[str, Any], List[MetricResult], List[MetricResult], List[Dict[str, Any]]]:
    baseline_params = finalize_model_params(baseline_params_with_threads(combo, model_threads), combo)
    with telemetry_scope_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="evaluate_dataset",
        module_name=__name__,
        phase_name="fit",
        parent_phase="baseline_evaluation",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=sum(len(dataset.frame) for dataset in datasets),
        asset_count=len(datasets),
    ) as baseline_scope:
        baseline_metrics = [evaluate_dataset(dataset, baseline_params, "baseline") for dataset in datasets]
        baseline_scope.update(output_rows=sum(int(metric.rows) for metric in baseline_metrics))
    baseline_summary = summarize_metrics(baseline_metrics)
    if int(baseline_summary.get("rows", 0) or 0) <= 0:
        emit_event_for_path(
            telemetry_path,
            family="Bayesian_Numeric",
            model=str(CURRENT_MODEL_SPEC.model_key),
            stage="stage3",
            function_name="run_study_for_combo",
            module_name=__name__,
            phase_name="metric_calculation",
            parent_phase="baseline_evaluation",
            status="skipped",
            reason_code="objective_dataset_empty",
            combo_key=combo.tuple_label,
            interval_minutes=int(combo.interval),
            horizon_minutes=int(combo.horizon_minutes),
            task=str(combo.task),
            input_rows=len(datasets),
            output_rows=0,
        )
        combo_row = {
            "combo": combo.tuple_label,
            "interval": int(combo.interval),
            "horizon_minutes": int(combo.horizon_minutes),
            "task": combo.task,
            "status": "ineligible",
            "reason": "no_finite_evaluation_rows",
            "baseline_rmse": None,
            "baseline_mae": None,
            "tuned_rmse": None,
            "tuned_mae": None,
            "baseline_rows": 0,
            "tuned_rows": 0,
            "rmse_delta": None,
            "mae_delta": None,
            "best_params": dict(baseline_params),
            "best_value": None,
        }
        return combo_row, baseline_metrics, [], []
    trial_rows: List[Dict[str, Any]] = []
    pruner_startup_trials = 8
    pruner_warmup_steps = 2

    def objective(trial: optuna.Trial) -> float:
        params = dict(baseline_params)
        params.update(trial_params(trial, combo))
        params = finalize_model_params(params, combo)
        rmse_num = 0.0
        mae_num = 0.0
        rows = 0
        for step_idx, dataset in enumerate(datasets, start=1):
            metric = evaluate_dataset(dataset, params, f"trial_{trial.number}")
            if metric.rmse is not None and metric.rows > 0:
                rmse_num += float(metric.rmse) * int(metric.rows)
                mae_num += float(metric.mae or 0.0) * int(metric.rows)
                rows += int(metric.rows)
            interim_rmse = float(rmse_num / rows) if rows > 0 else float("inf")
            trial.report(interim_rmse, step=step_idx)
            if trial.should_prune():
                trial_rows.append(
                    {
                        "combo": combo.tuple_label,
                        "trial_number": int(trial.number),
                        "objective_rmse": (None if not math.isfinite(interim_rmse) else float(interim_rmse)),
                        "weighted_mae": (float(mae_num / rows) if rows > 0 else None),
                        "rows": int(rows),
                        "params": dict(params),
                        "state": "PRUNED",
                        "reason": ("no_finite_evaluation_rows" if rows <= 0 else "trial_pruned"),
                    }
                )
                raise optuna.TrialPruned()
        objective_value = float(rmse_num / rows) if rows > 0 else float("inf")
        trial_rows.append(
            {
                "combo": combo.tuple_label,
                "trial_number": int(trial.number),
                "objective_rmse": (None if not math.isfinite(objective_value) else float(objective_value)),
                "weighted_mae": (float(mae_num / rows) if rows > 0 else None),
                "rows": int(rows),
                "params": dict(params),
                "state": "COMPLETE",
            }
        )
        return objective_value
    storage_url = (str(storage).strip() if storage is not None else "")
    with telemetry_scope_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="optuna.create_study",
        module_name=__name__,
        phase_name="study_setup",
        parent_phase="tuning",
        event_type="setup_control",
        setup_control=True,
        row_producing=False,
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        source_path=str(storage_url or ""),
    ) as study_scope:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(sampler_seed)),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=int(pruner_startup_trials),
                n_warmup_steps=int(pruner_warmup_steps),
            ),
            study_name=f"{study_name_prefix}_{combo.interval}_{combo.horizon_minutes}_{combo.task}",
            storage=(storage_url or None),
            load_if_exists=bool(resume_study or storage_url),
        )
        study_scope.update(output_rows=len(study.trials))
    study_started = time.perf_counter()
    if int(trials_per_combo) > 0:
        with telemetry_scope_for_path(
            telemetry_path,
            family="Bayesian_Numeric",
            model=str(CURRENT_MODEL_SPEC.model_key),
            stage="stage3",
            function_name="study.optimize",
            module_name=__name__,
            phase_name="fit",
            parent_phase="tuning",
            combo_key=combo.tuple_label,
            interval_minutes=int(combo.interval),
            horizon_minutes=int(combo.horizon_minutes),
            task=str(combo.task),
            input_rows=int(trials_per_combo),
            asset_count=len(datasets),
        ) as optimize_scope:
            study.optimize(objective, n_trials=int(trials_per_combo), show_progress_bar=False)
            optimize_scope.update(output_rows=len(trial_rows))
    study_elapsed_s = time.perf_counter() - study_started
    best_params = dict(baseline_params)
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if complete_trials:
        best_trial = min(
            complete_trials,
            key=lambda trial: float(trial.value) if trial.value is not None else float("inf"),
        )
        best_params.update(dict(best_trial.params))
    best_params = finalize_model_params(best_params, combo)
    with telemetry_scope_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="evaluate_dataset",
        module_name=__name__,
        phase_name="predict",
        parent_phase="best_trial_evaluation",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=sum(len(dataset.frame) for dataset in datasets),
        asset_count=len(datasets),
    ) as tuned_scope:
        tuned_metrics = [evaluate_dataset(dataset, best_params, "tuned") for dataset in datasets]
        tuned_scope.update(output_rows=sum(int(metric.rows) for metric in tuned_metrics))
    tuned_summary = summarize_metrics(tuned_metrics)
    emit_event_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="summarize_metrics",
        module_name=__name__,
        phase_name="metric_calculation",
        parent_phase="best_trial_evaluation",
        status="completed" if int(tuned_summary.get("rows", 0) or 0) > 0 else "skipped",
        reason_code="" if int(tuned_summary.get("rows", 0) or 0) > 0 else "predict_returned_empty",
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        input_rows=len(tuned_metrics),
        output_rows=int(tuned_summary.get("rows", 0) or 0),
    )
    emit_stage3_study_summary_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        function_name="run_study_for_combo",
        module_name=__name__,
        combo_key=combo.tuple_label,
        interval_minutes=int(combo.interval),
        horizon_minutes=int(combo.horizon_minutes),
        task=str(combo.task),
        elapsed_seconds=float(study_elapsed_s),
        trial_rows=trial_rows,
        study_trials=list(study.trials),
        best_value=(
            min(float(trial.value) for trial in complete_trials if trial.value is not None)
            if complete_trials
            else None
        ),
        input_rows=int(trials_per_combo),
        output_rows=sum(int(metric.rows) for metric in tuned_metrics),
        source_path=str(storage_url or ""),
    )
    combo_row = {
        "combo": combo.tuple_label,
        "interval": int(combo.interval),
        "horizon_minutes": int(combo.horizon_minutes),
        "task": combo.task,
        "baseline_rmse": baseline_summary.get("weighted_rmse"),
        "baseline_mae": baseline_summary.get("weighted_mae"),
        "tuned_rmse": tuned_summary.get("weighted_rmse"),
        "tuned_mae": tuned_summary.get("weighted_mae"),
        "baseline_rows": baseline_summary.get("rows"),
        "tuned_rows": tuned_summary.get("rows"),
        "rmse_delta": (
            float(tuned_summary["weighted_rmse"]) - float(baseline_summary["weighted_rmse"])
            if baseline_summary.get("weighted_rmse") is not None and tuned_summary.get("weighted_rmse") is not None
            else None
        ),
        "mae_delta": (
            float(tuned_summary["weighted_mae"]) - float(baseline_summary["weighted_mae"])
            if baseline_summary.get("weighted_mae") is not None and tuned_summary.get("weighted_mae") is not None
            else None
        ),
        "best_params": dict(best_params),
        "best_value": (min(float(trial.value) for trial in complete_trials if trial.value is not None) if complete_trials else None),
    }
    return combo_row, baseline_metrics, tuned_metrics, trial_rows


def write_outputs(output_dir: Path, sample_rows: Sequence[Dict[str, Any]], combo_rows: Sequence[Dict[str, Any]], metric_rows: Sequence[MetricResult], trial_rows: Sequence[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(list(sample_rows)).to_csv(output_dir / "representative_samples.csv", index=False)
    pd.DataFrame(list(combo_rows)).to_csv(output_dir / "combo_results.csv", index=False)
    pd.DataFrame([asdict(metric) for metric in metric_rows]).to_csv(output_dir / "unit_metrics.csv", index=False)
    pd.DataFrame(list(trial_rows)).to_json(output_dir / "optuna_trials.json", orient="records", indent=2)
    pd.DataFrame(list(combo_rows)).to_csv(output_dir / "runtime_summary.csv", index=False)
    emit_event_for_path(
        output_dir,
        family="Bayesian_Numeric",
        model=str(CURRENT_MODEL_SPEC.model_key),
        stage="stage3",
        function_name="write_outputs",
        module_name=__name__,
        phase_name="artifact_handoff",
        status="completed",
        input_rows=len(metric_rows),
        output_rows=len(combo_rows),
        output_path=str(output_dir),
        reason_code=("predict_returned_empty" if not metric_rows else ""),
    )
    lines = [f"# {CURRENT_MODEL_SPEC.display_name} Bayesian Stage 3", "", "Objective: minimize validation RMSE on a recent evaluation slice.", ""]
    for row in combo_rows:
        if str(row.get("status", "")).strip() == "ineligible":
            lines.append(f"- `{row['combo']}`: status=ineligible, reason={row.get('reason')}")
        else:
            lines.append(
                f"- `{row['combo']}`: baseline_rmse={row['baseline_rmse']}, tuned_rmse={row['tuned_rmse']}, "
                f"baseline_mae={row['baseline_mae']}, tuned_mae={row['tuned_mae']}, rmse_delta={row.get('rmse_delta')}"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_for_model(model_spec: BayesianOptunaModelSpec) -> None:
    configure_for_model(model_spec)
    args = parse_args()
    combos = resolve_combo_specs(args)
    assets = requested_assets(args.assets, args)
    output_dir = _output_dir(args)
    emit_event_for_path(
        output_dir,
        family="Bayesian_Numeric",
        model=str(model_spec.model_key),
        stage="stage3",
        function_name="main_for_model",
        module_name=__name__,
        phase_name="combo_planning",
        status="completed",
        asset_count=len(assets),
        output_rows=len(combos),
        artifact_profile_source=str(_resolve_stage2_survivor_json(args) or _resolve_stage1_feature_profile_json(args) or ""),
    )
    sample_rows: List[Dict[str, Any]] = []
    combo_rows: List[Dict[str, Any]] = []
    metric_rows: List[MetricResult] = []
    trial_rows: List[Dict[str, Any]] = []
    setup_cache = _Stage3SetupCache()
    try:
        for combo in combos:
            datasets = _build_datasets_with_telemetry(combo, assets, args, output_dir, setup_cache=setup_cache)
            if not datasets:
                combo_rows.append(
                    {
                        "combo": combo.tuple_label,
                        "status": "ineligible",
                        "reason": "no_evaluation_datasets",
                        "baseline_rmse": None,
                        "tuned_rmse": None,
                        "baseline_mae": None,
                        "tuned_mae": None,
                        "rmse_delta": None,
                        "mae_delta": None,
                        "trial_count": 0,
                    }
                )
                continue
            for dataset in datasets:
                sample_rows.append({"combo": combo.tuple_label, "asset": dataset.asset, "eval_start_ts": int(dataset.eval_start_ts), "eval_end_ts": int(dataset.eval_end_ts), "rows": int(len(dataset.frame)), "origin_count": int(len(dataset.origins))})
            combo_row, baseline_metrics, tuned_metrics, combo_trials = run_study_for_combo(combo, datasets, trials_per_combo=int(args.trials_per_combo), sampler_seed=int(args.sampler_seed), storage=(str(args.storage).strip() or None), study_name_prefix=str(args.study_name_prefix), resume_study=bool(args.resume_study), model_threads=int(args.model_threads), telemetry_path=output_dir)
            combo_rows.append(combo_row)
            metric_rows.extend(list(baseline_metrics))
            metric_rows.extend(list(tuned_metrics))
            trial_rows.extend(combo_trials)
        write_outputs(output_dir, sample_rows, combo_rows, metric_rows, trial_rows)
    finally:
        setup_cache.clear()
    print(json.dumps({"output_dir": str(output_dir), "combo_count": len(combo_rows)}, indent=2))
