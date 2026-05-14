from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.ml_module_utils import select_numeric_feature_columns
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import ConstantRegressor
from src.forecasting.ml.shared.feature_profile_common import combo_selection_key
from src.forecasting.ml.shared.numeric_cohort_common import (
    CLAMP_START_MONTH,
    CLAMP_START_YEAR,
    DEFAULT_COHORT_WINDOW_MONTHS,
    DEFAULT_SEARCH_BACK_MONTHS,
    MonthKey,
    common_recent_window,
    select_representative_assets,
)
from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import (
    get_tabular_numeric_model_spec,
    load_model_task_metadata,
)
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path, telemetry_scope_for_path

DEFAULT_MODEL_KEY = "xgboost"
STAGE1_DEFAULT_INTERVALS: Tuple[int, ...] = (15, 30, 60, 240, 720, 1440)
STAGE1_DEFAULT_ASSET_COUNT = 8
STAGE1_DEFAULT_WINDOW_MONTHS = 3
STAGE1_SLOW_WINDOW_MONTHS = 6
STAGE1_DEFAULT_WORKERS = 8
STAGE1_DEFAULT_MODEL_THREADS = 6

TASK_FAMILY = {
    "log_return": "returns",
    "realized_vol": "vol_range",
    "true_range": "vol_range",
    "max_drawdown": "path_extremes",
    "max_runup": "path_extremes",
    "range_efficiency": "path_extremes",
}
FAMILY_DEFS: List[Tuple[str, set[str]]] = [
    ("raw_ohlcvt", {"open", "high", "low", "close", "volume", "trades"}),
    ("price_transforms", {"typical_price", "median_price", "weighted_close", "delta_close", "log_return", "pct_change", "true_range", "range_hl", "range_co", "vwap_day", "dir"}),
    ("moving_averages", {"sma_20", "ema_20", "wma_20", "hma_20", "wilder_14", "ma_env_upper_20_2pct", "ma_env_lower_20_2pct", "kama_10_2_30", "frama_16", "ewm_mean_alpha_0_1", "ewm_mean_alpha_0_2"}),
    ("trend_channels", {"macd_12_26_9", "macd_signal_12_26_9", "macd_hist_12_26_9", "plus_di_14", "minus_di_14", "adx_14", "aroon_up_25", "aroon_down_25", "aroon_osc_25", "psar", "lr_slope_20", "lr_intercept_20", "lr_channel_hi_20", "lr_channel_lo_20", "tenkan_9", "kijun_26", "span_a_26", "span_b_26", "chikou_26", "vi_plus_14", "vi_minus_14"}),
    ("oscillators", {"rsi_14", "stoch_k_14", "stoch_d_3", "williams_r_14", "cci_20", "mfi_14", "roc_14", "mom_14", "cmo_14", "trix_15", "dpo_20", "ultosc_7_14_28", "rvi_10", "elder_bull_13", "elder_bear_13", "crsi_3_2_100"}),
    ("volatility_distribution", {"atr_14", "boll_mid_20", "boll_up_20", "boll_low_20", "keltner_mid_20", "keltner_up_20", "keltner_low_20", "ret_mean_20", "ret_std_20", "sharpe_20", "sortino_20", "cv_20", "chaikin_vol_10_10", "vol_osc_14_28", "vol_osc_pct_14_28", "var_20", "skew_20", "kurt_20", "zscore_20", "squeeze_scalar", "in_squeeze", "entropy_20", "hurst_100", "fractal_100", "msv_14", "q25_20", "q50_20", "q75_20", "prank_20"}),
    ("volume_flow", {"obv", "adl", "force_index", "avg_trade_size", "trade_intensity", "vroc_14", "chaikin_osc_3_10", "vpt", "eom_14", "pvi", "nvi", "vpt_vol_14"}),
    ("range_path_extremes", {"donchian_hi_20", "donchian_lo_20", "prr", "tir"}),
    ("lagged_change", {"d_close_2", "d_close_3", "d_close_5", "d_close_10", "d_close_14", "d_close_20"}),
]
FAMILY_LOOKUP: Dict[str, str] = {}
for family_name, members in FAMILY_DEFS:
    for member in members:
        FAMILY_LOOKUP[member] = family_name


@dataclass(frozen=True)
class ComboSpec:
    interval_minutes: int
    horizon_minutes: int
    task: str
    training_window_months: int
    assets: Tuple[str, ...]

    @property
    def key(self) -> str:
        return combo_selection_key(self.interval_minutes, self.horizon_minutes, self.task)


@dataclass
class FoldResult:
    spec_key: str
    asset: str
    fold_index: int
    train_rows: int
    val_rows: int
    full_feature_count: int
    selected_feature_count: int
    full_rmse: Optional[float]
    subset_rmse: Optional[float]
    full_skill: Optional[float]
    subset_skill: Optional[float]
    selected_features: str
    top_families: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def family_for_column(column: str) -> str:
    return FAMILY_LOOKUP.get(str(column), "other")


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(np.asarray(vals, dtype=float))) if vals else None


def parse_csv_int(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(','):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    return sorted(set(out))


def parse_csv_str(raw: str) -> List[str]:
    return sorted(set(token.strip() for token in str(raw).split(',') if token.strip()))


def parse_combo_list(raw: str) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    for token in parse_csv_str(raw):
        parts = token.split(':')
        if len(parts) != 3:
            raise ValueError(f'invalid combo token: {token}')
        interval, horizon, task = parts
        combos.append((int(interval), int(horizon), str(task)))
    return combos


def resolve_stage1_cohort_assets(args: argparse.Namespace, combos: Sequence[Tuple[int, int, str]]) -> Tuple[List[str], Dict[str, str]]:
    explicit_assets = parse_csv_str(getattr(args, "assets", ""))
    if explicit_assets:
        return [str(asset) for asset in explicit_assets], {}
    intervals = sorted({int(interval) for interval, _, _ in combos})
    if not intervals:
        raise RuntimeError("No intervals resolved for Stage-1 cohort selection.")
    parquet_root = Path(args.parquet_root).resolve()
    clamp_start = MonthKey(CLAMP_START_YEAR, CLAMP_START_MONTH)
    eligible_sets: List[set[str]] = []
    for interval_minutes in intervals:
        _end_month, eligible_assets = common_recent_window(
            ohlc_root=parquet_root / f"ohlcvt_{int(interval_minutes)}",
            scalar_root=parquet_root / f"scalar_features_{int(interval_minutes)}",
            min_assets=1,
            window_months=DEFAULT_COHORT_WINDOW_MONTHS,
            search_back_months=DEFAULT_SEARCH_BACK_MONTHS,
            clamp_start=clamp_start,
        )
        eligible_sets.append({str(asset) for asset in eligible_assets})
    common_assets = sorted(set.intersection(*eligible_sets)) if eligible_sets else []
    if not common_assets:
        raise RuntimeError("No shared eligible assets were found across the requested Stage-1 intervals.")
    return select_representative_assets(common_assets, seed=int(args.seed), asset_count=int(args.asset_count))


def full_combo_universe(module: Any, intervals: Sequence[int], tasks: Sequence[str], horizons: Sequence[int]) -> List[Tuple[int, int, str]]:
    active = getattr(module, 'ACTIVE_TASK_HORIZON_MATRIX')
    combos: List[Tuple[int, int, str]] = []
    use_tasks = list(tasks) if tasks else list(getattr(module, 'NUMERIC_TASKS'))
    use_intervals = list(intervals) if intervals else list(STAGE1_DEFAULT_INTERVALS)
    for interval in use_intervals:
        for task in use_tasks:
            supported_horizons = [int(v) for v in active.get(str(task), [])]
            for horizon in supported_horizons:
                if horizons and int(horizon) not in set(int(v) for v in horizons):
                    continue
                if int(horizon) < int(interval) or int(horizon) % int(interval) != 0:
                    continue
                combos.append((int(interval), int(horizon), str(task)))
    return combos


def stage1_training_window_months(explicit_months: int, interval_minutes: int, horizon_minutes: int) -> int:
    if int(explicit_months) > 0:
        return int(explicit_months)
    if int(interval_minutes) >= 240 or int(horizon_minutes) >= 720:
        return int(STAGE1_SLOW_WINDOW_MONTHS)
    return int(STAGE1_DEFAULT_WINDOW_MONTHS)


def stage1_cv_params(args: argparse.Namespace, interval_minutes: int, horizon_minutes: int) -> Tuple[int, int, int]:
    n_splits = int(args.n_splits)
    min_train_rows = int(args.min_train_rows)
    min_val_rows = int(args.min_val_rows)
    if int(interval_minutes) >= 720:
        return max(2, min(n_splits, 2)), min(min_train_rows, 64), min(min_val_rows, 16)
    if int(interval_minutes) >= 240 or int(horizon_minutes) >= 720:
        return max(2, min(n_splits, 3)), min(min_train_rows, 128), min(min_val_rows, 32)
    return n_splits, min_train_rows, min_val_rows


def baseline_rmse(y_train: np.ndarray, y_val: np.ndarray) -> Optional[float]:
    if len(y_val) == 0:
        return None
    prev = np.empty_like(y_val, dtype=float)
    prev[0] = float(y_train[-1]) if len(y_train) else 0.0
    if len(y_val) > 1:
        prev[1:] = y_val[:-1]
    err = prev - y_val
    return float(np.sqrt(np.mean(err * err))) if len(err) else None


def companion_module_base(module: Any) -> str:
    name = str(module.__name__)
    if name in ("xgboost_numerics", "lightgbm_numerics"):
        return f"src.forecasting.ml.tabular.{name.replace('_numerics', '')}"
    return name.rsplit('.', 1)[0]


def fit_stage1_model(module: Any, task: str, x_train: pd.DataFrame, y_train: np.ndarray, model_threads: int) -> Any:
    base = companion_module_base(module)
    profile_module = importlib.import_module(base + '.numeric_profiles')
    adapter_module = importlib.import_module(base + '.numeric_adapter')
    params = profile_module.resolve_regressor_params(str(task), model_threads=int(model_threads), overrides=None, interval_minutes=None, horizon_minutes=None, training_window_months=None)
    estimator_cls = getattr(adapter_module, 'STAGE1_REGRESSOR_CLASS', None)
    if estimator_cls is None:
        for class_name in (
            'XGBRegressor',
            'LGBMRegressor',
            'CatBoostRegressor',
            'RandomForestRegressor',
            'ElasticNetRegressor',
            'SkElasticNet',
        ):
            estimator_cls = getattr(adapter_module, class_name, None)
            if estimator_cls is not None:
                break
    x_train = x_train.copy()
    y_train = np.asarray(y_train, dtype=float)
    finite_mask = np.isfinite(y_train) & np.all(np.isfinite(x_train.to_numpy(dtype=float)), axis=1)
    if estimator_cls is None or x_train.empty or x_train.shape[1] == 0 or int(finite_mask.sum()) < 2:
        base = float(np.nanmean(y_train)) if len(y_train) else 0.0
        return ConstantRegressor(base)
    x_fit = x_train.loc[finite_mask].reset_index(drop=True)
    y_fit = y_train[finite_mask]
    if int(len(y_fit)) < 2 or not np.isfinite(y_fit).all() or float(np.nanstd(y_fit)) <= 0.0:
        base = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
        return ConstantRegressor(base)
    model = estimator_cls(**dict(params or {}))
    model.fit(x_fit, y_fit)
    return model


def predict_stage1_model(model: Any, x_val: pd.DataFrame) -> np.ndarray:
    if isinstance(model, ConstantRegressor):
        return np.asarray(model.predict(np.zeros((len(x_val), 1), dtype=float)), dtype=float)
    return np.asarray(model.predict(x_val), dtype=float)


def fit_and_predict(
    module: Any,
    task: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    model_threads: int,
    *,
    telemetry_path: Optional[Path] = None,
    telemetry_base: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Any]:
    base = dict(telemetry_base or {})
    with telemetry_scope_for_path(
        telemetry_path,
        input_obj=x_train,
        **base,
        function_name="fit_stage1_model",
        module_name=__name__,
        phase_name="fit",
        parent_phase="feature_selection",
    ) as scope:
        model = fit_stage1_model(module, task, x_train, y_train, model_threads)
        scope.set_output(output_rows=len(y_train), output_columns_count=x_train.shape[1])
    with telemetry_scope_for_path(
        telemetry_path,
        input_obj=x_val,
        **base,
        function_name="predict_stage1_model",
        module_name=__name__,
        phase_name="predict",
        parent_phase="feature_selection",
    ) as scope:
        pred = predict_stage1_model(model, x_val)
        scope.set_output(output_rows=len(pred), reason_code=("predict_returned_empty" if len(pred) == 0 else ""))
    return np.asarray(pred, dtype=float), model


def model_gain_map(model: Any, columns: Sequence[str]) -> Dict[str, float]:
    if hasattr(model, 'feature_importances_'):
        raw = getattr(model, 'feature_importances_')
        return {str(name): float(val) for name, val in zip(columns, list(raw))}
    coef = getattr(model, 'coef_', None)
    if coef is not None:
        arr = np.asarray(coef, dtype=float).reshape(-1)
        return {str(name): float(abs(val)) for name, val in zip(columns, arr.tolist())}
    return {str(name): 0.0 for name in columns}


def permutation_scores(model: Any, x_val: pd.DataFrame, y_val: np.ndarray, base_rmse: float, repeats: int, columns: Sequence[str]) -> Dict[str, float]:
    rng = np.random.default_rng(17)
    scores: Dict[str, List[float]] = defaultdict(list)
    for _ in range(max(1, int(repeats))):
        for col in columns:
            shuffled = x_val.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            pred = predict_stage1_model(model, shuffled)
            rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            scores[str(col)].append(float(rmse - base_rmse))
    return {col: float(np.mean(vals)) for col, vals in scores.items() if vals}


def build_labeled_frame(
    module: Any,
    asset: str,
    interval_minutes: int,
    horizon_minutes: int,
    task: str,
    training_window_months: int,
    max_rows: int,
    *,
    telemetry_path: Optional[Path] = None,
    model_key: str = "",
    combo_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    base_event = {
        "family": "Tabular_Numeric",
        "model": str(model_key),
        "stage": "stage1",
        "combo_key": combo_key,
        "interval_minutes": int(interval_minutes),
        "horizon_minutes": int(horizon_minutes),
        "task": str(task),
        "asset": str(asset),
        "module_name": __name__,
    }
    horizon_bars = int(module.horizon_bars_from_minutes(horizon_minutes, interval_minutes))
    train_bars = int(module.training_window_bars_from_months(training_window_months, interval_minutes))
    stop_ts = int(module._get_stop_ts(asset, interval_minutes))
    start_ts = int(stop_ts - ((train_bars + horizon_bars + 32) * int(interval_minutes) * 60))
    with telemetry_scope_for_path(
        telemetry_path,
        **base_event,
        function_name="_load_unit_feature_frame",
        phase_name="feature_load",
        parent_phase="feature_selection",
    ) as scope:
        feature_df, _stats = module._load_unit_feature_frame(asset, interval_minutes, start_ts, stop_ts)
        scope.set_output(feature_df, reason_code=("feature_load_empty" if feature_df.empty else ""))
    if feature_df.empty:
        return pd.DataFrame(), []
    with telemetry_scope_for_path(
        telemetry_path,
        input_obj=feature_df,
        required_columns=["high", "low", "close"],
        **base_event,
        function_name="_compute_future_labels",
        phase_name="label_build",
        parent_phase="feature_selection",
    ) as scope:
        label_df, _label_stats = module._compute_future_labels(feature_df.loc[:, ['high', 'low', 'close']], horizon_bars)
        scope.set_output(label_df, reason_code=("labels_empty" if label_df.empty else ""))
    task_label = getattr(module, 'TASK_LABEL')[str(task)]
    y_col = str(task_label)
    frame = pd.concat([feature_df.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)
    x_cols = list(select_numeric_feature_columns(frame.columns, extra_exclude={y_col, 'future_direction'}))
    needed = [col for col in x_cols if col in frame.columns] + [y_col]
    with telemetry_scope_for_path(
        telemetry_path,
        input_obj=frame,
        required_columns=[y_col],
        key_columns=[y_col],
        **base_event,
        function_name="build_labeled_frame",
        phase_name="join",
        parent_phase="feature_selection",
    ) as scope:
        df = frame.loc[:, needed].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').reset_index(drop=True)
        if max_rows > 0 and len(df) > max_rows:
            df = df.iloc[-int(max_rows):].reset_index(drop=True)
        scope.set_output(df, reason_code=("train_frame_empty" if df.empty else ""))
    return df, [col for col in x_cols if col in df.columns]


def rank_candidate_columns(importances: Dict[str, float], mi_map: Dict[str, float], top_k: int) -> List[str]:
    cols = sorted(set(importances) | set(mi_map))
    ranked = []
    for col in cols:
        ranked.append((float(importances.get(col, 0.0)) + float(mi_map.get(col, 0.0)), col))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [col for _, col in ranked[: max(1, int(top_k))]]


def run_fold(
    module: Any,
    spec: ComboSpec,
    asset: str,
    df: pd.DataFrame,
    x_cols: Sequence[str],
    args: argparse.Namespace,
    model_threads: int,
    *,
    telemetry_path: Optional[Path] = None,
    model_key: str = "",
) -> Tuple[List[FoldResult], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[FoldResult] = []
    top_feature_stats: Counter[str] = Counter()
    family_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    n_splits, min_train_rows, min_val_rows = stage1_cv_params(args, int(spec.interval_minutes), int(spec.horizon_minutes))
    tscv = TimeSeriesSplit(n_splits=max(2, int(n_splits)))
    y_col = str(module.TASK_LABEL[str(spec.task)])
    folds_done = 0
    for fold_index, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        train = df.iloc[train_idx].reset_index(drop=True)
        val = df.iloc[val_idx].reset_index(drop=True)
        if len(train) < int(min_train_rows) or len(val) < int(min_val_rows):
            emit_event_for_path(
                telemetry_path,
                family="Tabular_Numeric",
                model=str(model_key),
                stage="stage1",
                combo_key=spec.key,
                interval_minutes=int(spec.interval_minutes),
                horizon_minutes=int(spec.horizon_minutes),
                task=str(spec.task),
                asset=str(asset),
                function_name="TimeSeriesSplit",
                module_name=__name__,
                phase_name="split",
                parent_phase="feature_selection",
                status="skipped",
                reason_code="insufficient_history",
                input_rows=len(df),
                output_rows=len(train) + len(val),
            )
            continue
        emit_event_for_path(
            telemetry_path,
            family="Tabular_Numeric",
            model=str(model_key),
            stage="stage1",
            combo_key=spec.key,
            interval_minutes=int(spec.interval_minutes),
            horizon_minutes=int(spec.horizon_minutes),
            task=str(spec.task),
            asset=str(asset),
            function_name="TimeSeriesSplit",
            module_name=__name__,
            phase_name="split",
            parent_phase="feature_selection",
            status="completed",
            input_rows=len(df),
            output_rows=len(train) + len(val),
        )
        x_train_raw = train.loc[:, x_cols]
        x_val_raw = val.loc[:, x_cols]
        selector = VarianceThreshold(threshold=float(args.variance_threshold))
        with telemetry_scope_for_path(
            telemetry_path,
            input_obj=x_train_raw,
            family="Tabular_Numeric",
            model=str(model_key),
            stage="stage1",
            combo_key=spec.key,
            interval_minutes=int(spec.interval_minutes),
            horizon_minutes=int(spec.horizon_minutes),
            task=str(spec.task),
            asset=str(asset),
            training_window_months=int(spec.training_window_months),
            function_name="VarianceThreshold.fit_transform",
            module_name=__name__,
            phase_name="candidate_feature_prep",
            parent_phase="feature_selection",
            input_columns_count=len(x_cols),
            candidate_feature_count=len(x_cols),
        ) as variance_scope:
            x_train_screen = pd.DataFrame(
                selector.fit_transform(x_train_raw),
                columns=x_train_raw.columns[selector.get_support()].tolist(),
            )
            x_val_screen = pd.DataFrame(selector.transform(x_val_raw), columns=x_train_screen.columns.tolist())
            variance_scope.set_output(
                x_train_screen,
                output_columns_count=len(x_train_screen.columns),
                candidate_feature_count=len(x_train_screen.columns),
                reason_code=("required_columns_missing" if x_train_screen.empty else ""),
            )
        if x_train_screen.empty:
            emit_event_for_path(
                telemetry_path,
                family="Tabular_Numeric",
                model=str(model_key),
                stage="stage1",
                combo_key=spec.key,
                interval_minutes=int(spec.interval_minutes),
                horizon_minutes=int(spec.horizon_minutes),
                task=str(spec.task),
                asset=str(asset),
                function_name="VarianceThreshold.fit_transform",
                module_name=__name__,
                phase_name="candidate_feature_prep",
                parent_phase="feature_selection",
                status="skipped",
                reason_code="required_columns_missing",
                input_rows=len(train),
                output_rows=0,
                input_columns_count=len(x_cols),
                output_columns_count=0,
                candidate_feature_count=0,
            )
            continue
        y_train = pd.to_numeric(train[y_col], errors='coerce').to_numpy(dtype=float)
        y_val = pd.to_numeric(val[y_col], errors='coerce').to_numpy(dtype=float)
        telemetry_base = {
            "family": "Tabular_Numeric",
            "model": str(model_key),
            "stage": "stage1",
            "combo_key": spec.key,
            "interval_minutes": int(spec.interval_minutes),
            "horizon_minutes": int(spec.horizon_minutes),
            "task": str(spec.task),
            "asset": str(asset),
            "training_window_months": int(spec.training_window_months),
            "thread_count": int(model_threads),
            "candidate_feature_count": int(len(x_train_screen.columns)),
            "dynamic_feature_count": int(len(x_train_screen.columns)),
        }
        full_pred, full_model = fit_and_predict(
            module,
            spec.task,
            x_train_screen,
            y_train,
            x_val_screen,
            model_threads,
            telemetry_path=telemetry_path,
            telemetry_base={
                **telemetry_base,
                "fit_role": "full_model",
                "selected_feature_count": int(len(x_train_screen.columns)),
            },
        )
        full_rmse = float(np.sqrt(np.mean((full_pred - y_val) ** 2)))
        base_rmse = baseline_rmse(y_train, y_val)
        full_skill = float(1.0 - (full_rmse / base_rmse)) if base_rmse not in (None, 0.0) else None
        with telemetry_scope_for_path(
            telemetry_path,
            input_obj=x_train_screen,
            **telemetry_base,
            function_name="mutual_info_regression",
            module_name=__name__,
            phase_name="feature_scoring",
            parent_phase="feature_selection",
            input_rows=len(x_train_screen),
            input_columns_count=len(x_train_screen.columns),
        ) as mi_scope:
            mi_arr = mutual_info_regression(x_train_screen, y_train, random_state=17)
            mi_scope.set_output(
                output_rows=len(mi_arr),
                output_columns_count=len(mi_arr),
                selected_feature_count=0,
            )
        mi_map = {str(col): float(val) for col, val in zip(x_train_screen.columns.tolist(), mi_arr.tolist())}
        gains = model_gain_map(full_model, x_train_screen.columns.tolist())
        candidate_cols = rank_candidate_columns(gains, mi_map, top_k=min(int(args.top_k_features) * 2, max(4, len(x_train_screen.columns))))
        with telemetry_scope_for_path(
            telemetry_path,
            input_obj=x_val_screen.loc[:, candidate_cols] if candidate_cols else x_val_screen.iloc[:, 0:0],
            **telemetry_base,
            function_name="permutation_scores",
            module_name=__name__,
            phase_name="feature_scoring",
            parent_phase="feature_selection",
            input_rows=len(x_val_screen),
            input_columns_count=len(candidate_cols),
            selected_feature_count=len(candidate_cols),
            output_rows=0,
        ) as permutation_scope:
            perm_map = (
                permutation_scores(
                    full_model,
                    x_val_screen,
                    y_val,
                    full_rmse,
                    repeats=int(args.permutation_repeats),
                    columns=candidate_cols,
                )
                if candidate_cols
                else {}
            )
            permutation_scope.set_output(
                output_rows=len(perm_map),
                output_columns_count=len(perm_map),
                selected_feature_count=len(candidate_cols),
            )
        selected_cols = [col for col, score in sorted(perm_map.items(), key=lambda item: (-item[1], item[0])) if float(score) > 0.0]
        if not selected_cols:
            selected_cols = candidate_cols[: max(1, int(args.top_k_features))]
        selected_cols = selected_cols[: max(1, int(args.top_k_features))]
        subset_pred, _subset_model = fit_and_predict(
            module,
            spec.task,
            x_train_screen.loc[:, selected_cols],
            y_train,
            x_val_screen.loc[:, selected_cols],
            model_threads,
            telemetry_path=telemetry_path,
            telemetry_base={
                **telemetry_base,
                "fit_role": "subset_model",
                "selected_feature_count": int(len(selected_cols)),
            },
        )
        subset_rmse = float(np.sqrt(np.mean((subset_pred - y_val) ** 2)))
        subset_skill = float(1.0 - (subset_rmse / base_rmse)) if base_rmse not in (None, 0.0) else None
        top_feature_stats.update(selected_cols)
        family_counts = Counter(family_for_column(col) for col in selected_cols)
        rows.append(FoldResult(
            spec_key=spec.key,
            asset=str(asset),
            fold_index=int(fold_index),
            train_rows=int(len(train)),
            val_rows=int(len(val)),
            full_feature_count=int(len(x_train_screen.columns)),
            selected_feature_count=int(len(selected_cols)),
            full_rmse=float(full_rmse),
            subset_rmse=float(subset_rmse),
            full_skill=full_skill,
            subset_skill=subset_skill,
            selected_features=','.join(selected_cols),
            top_families=','.join(fam for fam, _ in family_counts.most_common(3)),
        ))
        fold_rows.append({
            'spec_key': spec.key,
            'asset': str(asset),
            'fold_index': int(fold_index),
            'full_rmse': float(full_rmse),
            'subset_rmse': float(subset_rmse),
            'full_skill': full_skill,
            'subset_skill': subset_skill,
            'selected_features': ','.join(selected_cols),
        })
        for family in sorted({family_for_column(col) for col in x_train_screen.columns}):
            family_cols = [col for col in x_train_screen.columns if family_for_column(col) == family]
            remaining = [col for col in selected_cols if col not in set(family_cols)]
            if not remaining:
                continue
            ablated_pred, _ablated_model = fit_and_predict(
                module,
                spec.task,
                x_train_screen.loc[:, remaining],
                y_train,
                x_val_screen.loc[:, remaining],
                model_threads,
                telemetry_path=telemetry_path,
                telemetry_base={
                    **telemetry_base,
                    "fit_role": f"ablation:{family}",
                    "selected_feature_count": int(len(remaining)),
                },
            )
            ablated_rmse = float(np.sqrt(np.mean((ablated_pred - y_val) ** 2)))
            ablated_skill = float(1.0 - (ablated_rmse / base_rmse)) if base_rmse not in (None, 0.0) else None
            family_rows.append({
                'spec_key': spec.key,
                'asset': str(asset),
                'fold_index': int(fold_index),
                'family': family,
                'family_feature_count': int(len(family_cols)),
                'subset_skill': subset_skill,
                'ablated_skill': ablated_skill,
                'skill_delta': (None if subset_skill is None or ablated_skill is None else float(ablated_skill - subset_skill)),
            })
        folds_done += 1
    return rows, dict(top_feature_stats), family_rows, fold_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Canonical tabular numeric feature experiment.')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument('--profile', type=str, default=selected_profile(default='pipeline_test'))
    parser.add_argument('--parquet-root', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--intervals', type=str, default='')
    parser.add_argument('--tasks', type=str, default='')
    parser.add_argument('--horizon-minutes', type=str, default='')
    parser.add_argument('--combo-list', type=str, default='')
    parser.add_argument('--assets', type=str, default='')
    parser.add_argument('--asset-count', type=int, default=STAGE1_DEFAULT_ASSET_COUNT)
    parser.add_argument('--seed', type=int, default=20260327)
    parser.add_argument('--train-window-months', type=int, default=0, help='Override Stage 1 screening window months. Default policy is 3m, extended to 6m for slower interval/horizon combos.')
    parser.add_argument('--max-rows', type=int, default=24000)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--min-train-rows', type=int, default=512)
    parser.add_argument('--min-val-rows', type=int, default=64)
    parser.add_argument('--variance-threshold', type=float, default=0.0)
    parser.add_argument('--permutation-repeats', type=int, default=2)
    parser.add_argument('--top-k-features', type=int, default=12)
    parser.add_argument('--workers', type=int, default=STAGE1_DEFAULT_WORKERS)
    parser.add_argument('--model-threads', type=int, default=STAGE1_DEFAULT_MODEL_THREADS)
    args = parser.parse_args()
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path('source_ohlcvt_root', profile=str(args.profile), required=False) or Path('parquet'))
    return args


def run_combo_spec(module: Any, args: argparse.Namespace, interval_minutes: int, horizon_minutes: int, task: str, cohort_assets: Sequence[str]) -> Optional[Dict[str, Any]]:
    assets = [str(asset) for asset in cohort_assets]
    telemetry_path = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) is not None else None
    model_key = str(getattr(args, "model_key", ""))
    combo_key = combo_selection_key(int(interval_minutes), int(horizon_minutes), str(task))
    if not assets:
        emit_event_for_path(
            telemetry_path,
            family="Tabular_Numeric",
            model=model_key,
            stage="stage1",
            combo_key=combo_key,
            interval_minutes=int(interval_minutes),
            horizon_minutes=int(horizon_minutes),
            task=str(task),
            function_name="run_combo_spec",
            module_name=__name__,
            phase_name="asset_planning",
            status="skipped",
            reason_code="no_assets",
            output_rows=0,
            asset_count=0,
        )
        return None
    training_window_months = stage1_training_window_months(int(args.train_window_months), int(interval_minutes), int(horizon_minutes))
    combo_spec = ComboSpec(
        interval_minutes=int(interval_minutes),
        horizon_minutes=int(horizon_minutes),
        task=str(task),
        training_window_months=int(training_window_months),
        assets=tuple(assets),
    )
    all_fold_results: List[FoldResult] = []
    feature_votes: Counter[str] = Counter()
    family_rows_all: List[Dict[str, Any]] = []
    fold_rows_all: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for asset in assets:
        df, x_cols = build_labeled_frame(
            module,
            asset,
            int(interval_minutes),
            int(horizon_minutes),
            str(task),
            int(training_window_months),
            int(args.max_rows),
            telemetry_path=telemetry_path,
            model_key=model_key,
            combo_key=combo_spec.key,
        )
        if df.empty or len(x_cols) < 2:
            continue
        fold_results, feature_stats, family_rows, fold_rows = run_fold(
            module,
            combo_spec,
            asset,
            df,
            x_cols,
            args,
            int(args.model_threads),
            telemetry_path=telemetry_path,
            model_key=model_key,
        )
        all_fold_results.extend(fold_results)
        feature_votes.update(feature_stats)
        family_rows_all.extend(family_rows)
        fold_rows_all.extend(fold_rows)
    if not all_fold_results:
        emit_event_for_path(
            telemetry_path,
            family="Tabular_Numeric",
            model=model_key,
            stage="stage1",
            combo_key=combo_spec.key,
            interval_minutes=int(interval_minutes),
            horizon_minutes=int(horizon_minutes),
            task=str(task),
            function_name="run_combo_spec",
            module_name=__name__,
            phase_name="feature_selection",
            status="completed",
            reason_code="train_frame_empty",
            elapsed_seconds=time.perf_counter() - started,
            input_rows=len(assets),
            output_rows=0,
            asset_count=len(assets),
            training_window_months=int(training_window_months),
            thread_count=int(args.model_threads),
            candidate_feature_count=0,
            selected_feature_count=0,
            dynamic_feature_count=0,
        )
        return None
    top_features = [name for name, _count in feature_votes.most_common(int(args.top_k_features))]
    top_family_counts = Counter(family_for_column(col) for col in top_features)
    summary_row = {
        'spec_key': combo_spec.key,
        'interval_minutes': int(interval_minutes),
        'horizon_minutes': int(horizon_minutes),
        'task': str(task),
        'task_family': TASK_FAMILY.get(str(task), 'other'),
        'training_window_months': int(training_window_months),
        'asset_count_used': int(len(assets)),
        'folds_total': int(len(all_fold_results)),
        'full_set_mean_skill': mean(row.full_skill for row in all_fold_results),
        'selected_subset_mean_skill': mean(row.subset_skill for row in all_fold_results),
        'full_set_median_skill': median(row.full_skill for row in all_fold_results),
        'selected_subset_median_skill': median(row.subset_skill for row in all_fold_results),
        'mean_full_feature_count': mean(row.full_feature_count for row in all_fold_results),
        'mean_selected_feature_count': mean(row.selected_feature_count for row in all_fold_results),
        'stable_top_features': ','.join(top_features),
        'top_feature_families': ','.join(fam for fam, _ in top_family_counts.most_common(4)),
        'subset_vs_full': 'improved_or_equal' if (mean(row.subset_skill for row in all_fold_results) or -1e9) >= (mean(row.full_skill for row in all_fold_results) or -1e9) else 'worse',
    }
    selection = {
        'feature_profile': 'selected_subset',
        'selected_features': top_features,
        'top_feature_families': [fam for fam, _ in top_family_counts.most_common(4)],
        'training_window_months': int(training_window_months),
        'asset_count_used': int(len(assets)),
    }
    emit_event_for_path(
        telemetry_path,
        family="Tabular_Numeric",
        model=model_key,
        stage="stage1",
        combo_key=combo_spec.key,
        interval_minutes=int(interval_minutes),
        horizon_minutes=int(horizon_minutes),
        task=str(task),
        function_name="run_combo_spec",
        module_name=__name__,
        phase_name="feature_selection",
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        input_rows=len(assets),
        output_rows=len(all_fold_results),
        asset_count=len(assets),
        output_columns_count=len(top_features),
        training_window_months=int(training_window_months),
        thread_count=int(args.model_threads),
        candidate_feature_count=int(round(mean(row.full_feature_count for row in all_fold_results) or 0)),
        selected_feature_count=int(len(top_features)),
        dynamic_feature_count=int(round(mean(row.full_feature_count for row in all_fold_results) or 0)),
    )
    return {
        'combo_key': combo_spec.key,
        'summary_row': summary_row,
        'family_rows': family_rows_all,
        'fold_rows': fold_rows_all,
        'selection': selection,
    }


def append_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode='a', header=write_header, index=False)


def write_selection_payload(path: Path, *, model_key: str, selections: Dict[str, Dict[str, Any]], cohort_assets: Sequence[str], cohort_asset_aliases: Dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                'model_key': str(model_key),
                'generated_utc': utc_now_iso(),
                'selection_file_version': 2,
                'cohort_assets': [str(asset) for asset in cohort_assets],
                'cohort_asset_aliases': {str(key): str(value) for key, value in cohort_asset_aliases.items()},
                'selections': selections,
            },
            indent=2,
        ),
        encoding='utf-8',
    )


def write_progress_payload(
    path: Path,
    *,
    model_key: str,
    total_combos: int,
    completed_combos: int,
    last_completed_combo: Optional[str],
    summary_path: Path,
    family_path: Path,
    folds_path: Path,
    selection_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                'model_key': str(model_key),
                'generated_utc': utc_now_iso(),
                'total_combos': int(total_combos),
                'completed_combos': int(completed_combos),
                'remaining_combos': int(max(0, int(total_combos) - int(completed_combos))),
                'last_completed_combo': last_completed_combo,
                'outputs': {
                    'summary_csv': str(summary_path),
                    'family_ablations_csv': str(family_path),
                    'fold_scores_csv': str(folds_path),
                    'selection_json': str(selection_path),
                },
            },
            indent=2,
        ),
        encoding='utf-8',
    )


def main_for_model(model_key: str = DEFAULT_MODEL_KEY) -> None:
    args = parse_args()
    setattr(args, "model_key", str(model_key))
    if args.parquet_root:
        os.environ.setdefault('PIPELINE_PARQUET_ROOT', str(args.parquet_root.resolve()))
        os.environ.setdefault('PIPELINE_PARQUET_FEATURES_ROOT', str(args.parquet_root.resolve()))
    spec = get_tabular_numeric_model_spec(model_key)
    module = importlib.import_module(spec.module_import_path)
    supported_tasks, _task_short, _task_label = load_model_task_metadata(spec, args.project_root.resolve())
    combos = parse_combo_list(args.combo_list) if str(args.combo_list).strip() else full_combo_universe(
        module,
        intervals=parse_csv_int(args.intervals),
        tasks=[task for task in parse_csv_str(args.tasks) if task in supported_tasks],
        horizons=parse_csv_int(args.horizon_minutes),
    )
    if not combos:
        raise RuntimeError('No combos resolved for feature experiment.')
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output_dir = spec.feature_experiment_output_dir / f'run={stamp}'
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / 'feature_experiment_summary.csv'
    family_path = output_dir / 'feature_experiment_family_ablations.csv'
    folds_path = output_dir / 'feature_experiment_fold_scores.csv'
    selection_path = output_dir / 'feature_profile_selection.json'
    progress_path = output_dir / 'feature_experiment_progress.json'
    meta_path = output_dir / 'feature_experiment_run_meta.json'
    cohort_assets, cohort_asset_aliases = resolve_stage1_cohort_assets(args, combos)
    emit_event_for_path(
        output_dir,
        family="Tabular_Numeric",
        model=str(model_key),
        stage="stage1",
        function_name="resolve_stage1_cohort_assets",
        module_name=__name__,
        phase_name="asset_planning",
        status="completed",
        asset_count=len(cohort_assets),
        output_rows=len(cohort_assets),
        reason_code=("no_assets" if not cohort_assets else ""),
        source_path=str(args.parquet_root),
    )
    summary_rows: List[Dict[str, Any]] = []
    family_rows_all: List[Dict[str, Any]] = []
    fold_rows_all: List[Dict[str, Any]] = []
    selections: Dict[str, Dict[str, Any]] = {}
    completed_combos = 0
    write_progress_payload(
        progress_path,
        model_key=str(model_key),
        total_combos=len(combos),
        completed_combos=0,
        last_completed_combo=None,
        summary_path=summary_path,
        family_path=family_path,
        folds_path=folds_path,
        selection_path=selection_path,
    )
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for interval_minutes, horizon_minutes, task in combos:
            future = executor.submit(run_combo_spec, module, args, int(interval_minutes), int(horizon_minutes), str(task), tuple(cohort_assets))
            futures[future] = (int(interval_minutes), int(horizon_minutes), str(task))
        for future in as_completed(futures):
            payload = future.result()
            if not payload:
                continue
            combo_key = str(payload['combo_key'])
            summary_row = dict(payload['summary_row'])
            family_rows = list(payload['family_rows'])
            fold_rows = list(payload['fold_rows'])
            selection = dict(payload['selection'])
            summary_rows.append(summary_row)
            family_rows_all.extend(family_rows)
            fold_rows_all.extend(fold_rows)
            selections[combo_key] = selection
            append_rows_csv(summary_path, [summary_row])
            append_rows_csv(family_path, family_rows)
            append_rows_csv(folds_path, fold_rows)
            completed_combos += 1
            write_selection_payload(selection_path, model_key=str(model_key), selections=selections, cohort_assets=cohort_assets, cohort_asset_aliases=cohort_asset_aliases)
            write_progress_payload(
                progress_path,
                model_key=str(model_key),
                total_combos=len(combos),
                completed_combos=completed_combos,
                last_completed_combo=combo_key,
                summary_path=summary_path,
                family_path=family_path,
                folds_path=folds_path,
                selection_path=selection_path,
            )
    if not summary_rows:
        raise RuntimeError('Feature experiment did not produce any usable combo summaries.')
    summary_df = pd.DataFrame(summary_rows).sort_values(['interval_minutes', 'horizon_minutes', 'task'], kind='stable')
    family_df = pd.DataFrame(family_rows_all).sort_values(['spec_key', 'asset', 'fold_index', 'family'], kind='stable') if family_rows_all else pd.DataFrame()
    fold_df = pd.DataFrame(fold_rows_all).sort_values(['spec_key', 'asset', 'fold_index'], kind='stable') if fold_rows_all else pd.DataFrame()
    summary_df.to_csv(summary_path, index=False)
    family_df.to_csv(family_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    write_selection_payload(selection_path, model_key=str(model_key), selections=selections, cohort_assets=cohort_assets, cohort_asset_aliases=cohort_asset_aliases)
    emit_event_for_path(
        output_dir,
        family="Tabular_Numeric",
        model=str(model_key),
        stage="stage1",
        function_name="write_selection_payload",
        module_name=__name__,
        phase_name="artifact_handoff",
        status="completed",
        output_rows=len(selections),
        output_path=str(selection_path),
    )
    write_progress_payload(
        progress_path,
        model_key=str(model_key),
        total_combos=len(combos),
        completed_combos=completed_combos,
        last_completed_combo=(summary_rows[-1]['spec_key'] if summary_rows else None),
        summary_path=summary_path,
        family_path=family_path,
        folds_path=folds_path,
        selection_path=selection_path,
    )
    expected_combo_keys = [combo_selection_key(int(interval), int(horizon), str(task)) for interval, horizon, task in combos]
    missing_combo_keys = [key for key in expected_combo_keys if key not in selections]
    meta_path.write_text(json.dumps({'model_key': str(model_key), 'generated_utc': utc_now_iso(), 'args': {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}, 'cohort_assets': list(cohort_assets), 'cohort_asset_aliases': dict(cohort_asset_aliases), 'outputs': {'summary_csv': str(summary_path), 'family_ablations_csv': str(family_path), 'fold_scores_csv': str(folds_path), 'selection_json': str(selection_path), 'progress_json': str(progress_path)}, 'expected_combo_count': int(len(expected_combo_keys)), 'completed_combo_count': int(len(selections)), 'missing_combo_keys': missing_combo_keys}, indent=2), encoding='utf-8')
    emit_event_for_path(
        output_dir,
        family="Tabular_Numeric",
        model=str(model_key),
        stage="stage1",
        function_name="write_feature_experiment_artifacts",
        module_name=__name__,
        phase_name="write",
        status="completed",
        output_rows=len(summary_rows),
        output_path=str(output_dir),
    )
    if missing_combo_keys:
        raise RuntimeError(f'Feature experiment completed with missing combo profiles: {len(missing_combo_keys)} missing')
    print(summary_path)
    print(family_path)
    print(folds_path)
    print(selection_path)
    print(meta_path)


def main() -> None:
    main_for_model(DEFAULT_MODEL_KEY)


if __name__ == '__main__':
    main()
