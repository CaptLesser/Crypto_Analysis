from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.forecasting.common.forecast_family_core import read_feature_window_columns
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.stats_module_utils import NUMERIC_TASK_TO_TARGET_COLUMN
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path


@dataclass(frozen=True)
class Stage3FeatureProfile:
    selected_dynamic_feature_columns: Tuple[str, ...] = ()
    use_dynamic_features: bool = False
    use_seasonality: bool = False


@dataclass(frozen=True)
class Stage3DatasetBuildHooks:
    setup_common_key: Callable[..., Tuple[Any, ...]]
    load_asset_frame: Callable[..., pd.DataFrame]
    label_frame: Callable[[pd.DataFrame, Any], pd.DataFrame]
    sample_origins: Callable[..., List[int]]
    dataset_factory: Callable[..., Any]
    factor_maps_factory: Optional[Callable[[Dict[str, pd.DataFrame], Any], Dict[str, Dict[int, float]]]] = None


@dataclass(frozen=True)
class Stage3DatasetBuildConfig:
    family: str
    model: str
    module_name: str
    telemetry_path: Optional[Path]
    history_window_months: int
    history_start_ts: int
    eval_start_ts: int
    eval_end_ts: int
    max_eval_origins: int
    feature_profile_json: Optional[Path]
    feature_profile: Optional[Stage3FeatureProfile]
    default_dynamic_feature_columns: Tuple[str, ...]
    default_use_dynamic_features: bool
    default_use_seasonality: bool = False
    factor_cache_enabled: bool = False


@dataclass(frozen=True)
class Stage3EvaluationArrays:
    ts_vec: np.ndarray
    y_vec: np.ndarray
    feat_cols: Tuple[str, ...]
    feat_matrix: Optional[np.ndarray]


def evaluate_window(edge_ts: int, recent_eval_days: int, history_window_months: int) -> Tuple[int, int]:
    eval_end_ts = int(edge_ts)
    history_start_ts = int(edge_ts) - int(history_window_months) * 31 * 86400
    return history_start_ts, eval_end_ts


def sample_origins(ts_values: Sequence[int], *, eval_start_ts: int, eval_end_ts: int, max_eval_origins: int) -> List[int]:
    eligible = [int(ts) for ts in ts_values if int(eval_start_ts) <= int(ts) <= int(eval_end_ts)]
    if len(eligible) <= int(max_eval_origins):
        return eligible
    idx = np.linspace(0, len(eligible) - 1, int(max_eval_origins)).astype(int)
    return [eligible[int(i)] for i in idx]


def label_frame(frame: pd.DataFrame, combo: Any) -> pd.DataFrame:
    labels, _stats = compute_future_labels(
        frame.loc[:, ["high", "low", "close"]].reset_index(drop=True),
        int(combo.horizon_bars),
        future_direction_deadzone=0.0,
    )
    return labels


def load_stage3_asset_frame(
    *,
    asset: str,
    combo: Any,
    history_start_ts: int,
    eval_end_ts: int,
    ohlcvt_root: Path,
    feature_root: Path,
    selected_feature_columns: Optional[Sequence[str]],
    dynamic_feature_candidates: Sequence[str],
    include_dynamic_default: bool,
    merge_how: str,
    add_missing_columns: bool,
    return_empty_ohlcvt: bool,
    read_ohlcvt_fn: Callable[..., pd.DataFrame] = read_ohlcvt,
    read_feature_window_columns_fn: Callable[..., pd.DataFrame] = read_feature_window_columns,
) -> pd.DataFrame:
    ohlc_columns = ["open", "high", "low", "close", "volume", "trades"]
    feature_columns = [NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]]
    if selected_feature_columns is None:
        if bool(include_dynamic_default):
            feature_columns.extend(str(col) for col in dynamic_feature_candidates if str(col))
    else:
        feature_columns.extend(str(col) for col in selected_feature_columns if str(col))
    ohlc_frame = read_ohlcvt_fn(
        root=Path(ohlcvt_root).resolve(),
        asset=str(asset),
        interval_min=int(combo.interval),
        start_ts=int(history_start_ts),
        end_ts=int(eval_end_ts),
        columns=["ts", "asset", *ohlc_columns],
    )
    feature_frame = read_feature_window_columns_fn(
        root=Path(feature_root).resolve(),
        interval_minutes=int(combo.interval),
        asset=str(asset),
        columns=list(dict.fromkeys(feature_columns)),
        start_ts=int(history_start_ts),
        end_ts=int(eval_end_ts),
    )
    if bool(return_empty_ohlcvt) and ohlc_frame.empty:
        return ohlc_frame.sort_values("ts").reset_index(drop=True)
    merged = ohlc_frame.merge(
        feature_frame,
        on=["ts", "asset"],
        how=str(merge_how),
        sort=str(merge_how).lower() == "outer",
        suffixes=("", "_feature"),
    )
    if add_missing_columns:
        for column in [*ohlc_columns, *feature_columns]:
            if column not in merged.columns:
                merged[column] = np.nan
    if merged.columns.has_duplicates:
        keep = "last" if str(merge_how).lower() == "outer" else "first"
        merged = merged.loc[:, ~merged.columns.duplicated(keep=keep)].copy()
    return merged.sort_values("ts").reset_index(drop=True)


def build_evaluation_arrays(frame: pd.DataFrame, *, target_col: str, selected_feature_columns: Sequence[str], use_dynamic_features: bool) -> Stage3EvaluationArrays:
    normalized = frame.reset_index(drop=True)
    ts_vec = pd.to_numeric(normalized["ts"], errors="coerce").fillna(-1).astype("int64").to_numpy(copy=True)
    y_vec = pd.to_numeric(normalized[target_col], errors="coerce").to_numpy(dtype=float, copy=True)
    feat_cols = [str(col) for col in selected_feature_columns if str(col) in normalized.columns]
    feat_matrix = None
    if bool(use_dynamic_features) and feat_cols:
        feat_frame = normalized.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce")
        feat_cols = [str(col) for col in feat_cols if feat_frame[str(col)].notna().any()]
        if feat_cols:
            feat_matrix = feat_frame.loc[:, feat_cols].to_numpy(dtype=float, copy=True)
    return Stage3EvaluationArrays(ts_vec=ts_vec, y_vec=y_vec, feat_cols=tuple(feat_cols), feat_matrix=feat_matrix)


def _profile_selected_for_load(profile: Optional[Stage3FeatureProfile]) -> Optional[Tuple[str, ...]]:
    if profile is None:
        return None
    if bool(profile.use_dynamic_features):
        return tuple(str(value) for value in profile.selected_dynamic_feature_columns)
    return ()


def _profile_selected_for_dataset(config: Stage3DatasetBuildConfig) -> Tuple[str, ...]:
    if config.feature_profile is not None:
        return tuple(str(value) for value in config.feature_profile.selected_dynamic_feature_columns)
    return tuple(str(value) for value in config.default_dynamic_feature_columns)


def _profile_use_dynamic(config: Stage3DatasetBuildConfig) -> bool:
    if config.feature_profile is not None:
        return bool(config.feature_profile.use_dynamic_features)
    return bool(config.default_use_dynamic_features)


def _profile_use_seasonality(config: Stage3DatasetBuildConfig) -> bool:
    if config.feature_profile is not None:
        return bool(config.feature_profile.use_seasonality)
    return bool(config.default_use_seasonality)


def build_stage3_datasets(
    *,
    combo: Any,
    assets: Sequence[str],
    args: Any,
    setup_cache: Optional[Any],
    config: Stage3DatasetBuildConfig,
    hooks: Stage3DatasetBuildHooks,
) -> List[Any]:
    frames: Dict[str, pd.DataFrame] = {}
    frame_keys: Dict[str, Tuple[Any, ...]] = {}
    selected_for_load = _profile_selected_for_load(config.feature_profile)
    use_dynamic_features = _profile_use_dynamic(config)
    use_seasonality = _profile_use_seasonality(config)
    for asset in assets:
        common_key = hooks.setup_common_key(
            combo=combo,
            args=args,
            feature_profile_json=config.feature_profile_json,
            selected_feature_columns=selected_for_load,
            history_start_ts=int(config.history_start_ts),
            eval_end_ts=int(config.eval_end_ts),
            history_window_months=int(config.history_window_months),
            use_dynamic_features=use_dynamic_features,
            use_seasonality=use_seasonality,
        )
        frame_key = ("asset_frame", str(asset), common_key)
        frame_keys[str(asset)] = frame_key
        frame = setup_cache.get(frame_key) if setup_cache is not None else None
        if frame is None:
            frame = hooks.load_asset_frame(
                str(asset),
                combo,
                int(config.history_start_ts),
                int(config.eval_end_ts),
                selected_for_load,
            )
            if setup_cache is not None and not frame.empty:
                setup_cache.put(frame_key, frame)
        if frame.empty:
            continue
        label_key = ("labels", str(asset), common_key)
        labels = setup_cache.get(label_key) if setup_cache is not None else None
        if labels is None:
            labels = hooks.label_frame(frame, combo)
            if setup_cache is not None:
                setup_cache.put(label_key, labels)
        label_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]
        merged = frame.reset_index(drop=True).copy()
        if label_col in merged.columns and label_col in labels.columns:
            merged = merged.drop(columns=[label_col])
        merged = pd.concat([merged, labels.reset_index(drop=True)], axis=1)
        if merged.columns.has_duplicates:
            merged = merged.loc[:, ~merged.columns.duplicated(keep="last")].copy()
        frames[str(asset)] = merged

    factor_maps: Dict[str, Dict[int, float]]
    if hooks.factor_maps_factory is None:
        factor_maps = {asset: {} for asset in frames}
    else:
        factor_key = (
            "factor_maps",
            tuple(str(asset) for asset in assets),
            tuple(frame_keys.get(str(asset)) for asset in assets),
            bool(config.factor_cache_enabled),
        )
        factor_maps = setup_cache.get(factor_key) if setup_cache is not None else None
        if factor_maps is None:
            factor_started = time.perf_counter()
            factor_maps = hooks.factor_maps_factory(frames, combo)
            emit_event_for_path(
                config.telemetry_path,
                family=config.family,
                model=config.model,
                stage="stage3",
                function_name="_build_factor_maps",
                module_name=config.module_name,
                phase_name="factor_map_build",
                parent_phase="dataset_construction",
                combo_key=combo.tuple_label,
                interval_minutes=int(combo.interval),
                horizon_minutes=int(combo.horizon_minutes),
                training_window_months=int(config.history_window_months),
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
                config.telemetry_path,
                family=config.family,
                model=config.model,
                stage="stage3",
                function_name="_build_factor_maps",
                module_name=config.module_name,
                phase_name="factor_map_build",
                parent_phase="dataset_construction",
                combo_key=combo.tuple_label,
                interval_minutes=int(combo.interval),
                horizon_minutes=int(combo.horizon_minutes),
                training_window_months=int(config.history_window_months),
                task=str(combo.task),
                elapsed_seconds=0.0,
                input_rows=sum(len(frame) for frame in frames.values()),
                output_rows=sum(len(mapping) for mapping in factor_maps.values()),
                asset_count=len(frames),
                cache_hit_count=1,
                cache_miss_count=0,
            )

    datasets: List[Any] = []
    label_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(combo.task)]
    selected_for_dataset = _profile_selected_for_dataset(config)
    for asset, frame in frames.items():
        label_values = frame[label_col]
        if isinstance(label_values, pd.DataFrame):
            label_values = label_values.iloc[:, -1]
        valid = frame[np.isfinite(pd.to_numeric(label_values, errors="coerce"))].copy()
        origins = hooks.sample_origins(
            valid["ts"].astype("int64").tolist(),
            eval_start_ts=int(config.eval_start_ts),
            eval_end_ts=int(config.eval_end_ts - int(combo.horizon_minutes) * 60),
            max_eval_origins=int(config.max_eval_origins),
        )
        if not origins:
            continue
        datasets.append(
            hooks.dataset_factory(
                asset=str(asset),
                combo=combo,
                frame=frame,
                target_col=str(label_col),
                history_start_ts=int(config.history_start_ts),
                eval_start_ts=int(config.eval_start_ts),
                eval_end_ts=int(config.eval_end_ts),
                origins=origins,
                factor_map=factor_maps.get(str(asset), {}),
                selected_dynamic_feature_columns=selected_for_dataset,
                use_dynamic_features=use_dynamic_features,
                use_seasonality=use_seasonality,
            )
        )
    return datasets
