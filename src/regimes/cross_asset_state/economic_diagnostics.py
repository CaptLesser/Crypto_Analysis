from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CROSS_ASSET_STATE_ECONOMIC_DIAGNOSTIC_SCHEMA_VERSION = 1
ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED = "computed"
ECONOMIC_DIAGNOSTIC_STATUS_PENDING_NOT_COMPUTED = "pending_not_computed"
ECONOMIC_DIAGNOSTIC_STATUS_BLOCKED_MISSING_OUTCOME_PANEL = "blocked_missing_future_outcome_panel"
ECONOMIC_DIAGNOSTIC_STATUS_LEAKAGE_BLOCKED = "leakage_blocked"
ECONOMIC_DIAGNOSTIC_STATUS_ALIGNMENT_BLOCKED = "alignment_blocked"
ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE = "not_available"
ECONOMIC_DIAGNOSTIC_STATUS_NOT_APPLICABLE = "not_applicable"

DEFAULT_ECONOMIC_DIAGNOSTICS: tuple[str, ...] = (
    "future_relative_return_vs_anchor_or_core_basket",
    "future_realized_volatility",
    "future_drawdown",
    "future_beta_or_correlation_shift",
    "future_residual_continuation_or_reversal",
)

ECONOMIC_TARGET_RELATIVE_RETURN = "future_relative_return_vs_anchor_or_core_basket"
ECONOMIC_TARGET_REALIZED_VOLATILITY = "future_realized_volatility"
ECONOMIC_TARGET_DRAWDOWN = "future_drawdown"
ECONOMIC_TARGET_BETA_CORRELATION_SHIFT = "future_beta_or_correlation_shift"
ECONOMIC_TARGET_RESIDUAL_CONTINUATION = "future_residual_continuation_or_reversal"

LEAKAGE_RISK_TOKENS: tuple[str, ...] = ("future_", "forward_", "_target", "target_", "outcome_", "_label")
DEFAULT_OUTCOME_HORIZON_STEPS_BY_BAND: Mapping[str, int] = {"meso": 3, "macro": 2}


@dataclass(frozen=True)
class CrossAssetStateOutcomeHorizonPolicy:
    outcome_window_policy_id: str = "cross_asset_state_outcome_horizon_v1"
    horizon_steps_by_band: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_OUTCOME_HORIZON_STEPS_BY_BAND))
    future_missing_row_policy_id: str = "mask_tail_rows_and_require_min_overlap_v1"
    max_missing_future_row_share: float = 0.25
    min_finite_rows_per_target: int = 4
    min_state_groups: int = 2
    timestamp_column: str = "ts"
    asset_id_column: str = "asset_id"
    band_column: str = "band"
    state_label_column: str = "state_label"
    source_tail_column: str = "source_tail_ts"
    known_at_column: str = "known_at_ts"
    close_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    log_return_column: str = "log_return"
    residual_signal_column: str = "residual_peer_signal_score"
    schema_version: int = CROSS_ASSET_STATE_ECONOMIC_DIAGNOSTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        horizons = {str(key): int(value) for key, value in dict(self.horizon_steps_by_band).items()}
        if not horizons:
            raise ValueError("Cross-Asset-State outcome horizon policy requires at least one band horizon")
        for band, horizon in horizons.items():
            if horizon < 1:
                raise ValueError(f"Cross-Asset-State outcome horizon for {band!r} must be positive")
        if int(self.min_finite_rows_per_target) < 1:
            raise ValueError("Cross-Asset-State economic diagnostics min_finite_rows_per_target must be positive")
        if int(self.min_state_groups) < 2:
            raise ValueError("Cross-Asset-State economic diagnostics require at least two state groups")
        missing_share = float(self.max_missing_future_row_share)
        if missing_share < 0.0 or missing_share > 1.0:
            raise ValueError("Cross-Asset-State max_missing_future_row_share must be in [0, 1]")
        object.__setattr__(self, "horizon_steps_by_band", horizons)
        object.__setattr__(self, "max_missing_future_row_share", missing_share)
        object.__setattr__(self, "min_finite_rows_per_target", int(self.min_finite_rows_per_target))
        object.__setattr__(self, "min_state_groups", int(self.min_state_groups))

    def horizon_steps_for_band(self, band: object) -> int:
        key = str(band)
        if key not in self.horizon_steps_by_band:
            raise ValueError(f"Unsupported Cross-Asset-State outcome horizon band {key!r}")
        return int(self.horizon_steps_by_band[key])

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_outcome_horizon_policy",
            "schema_version": int(self.schema_version),
            "outcome_window_policy_id": self.outcome_window_policy_id,
            "horizon_steps_by_band": dict(self.horizon_steps_by_band),
            "future_missing_row_policy_id": self.future_missing_row_policy_id,
            "max_missing_future_row_share": float(self.max_missing_future_row_share),
            "min_finite_rows_per_target": int(self.min_finite_rows_per_target),
            "min_state_groups": int(self.min_state_groups),
            "timestamp_column": self.timestamp_column,
            "asset_id_column": self.asset_id_column,
            "band_column": self.band_column,
            "state_label_column": self.state_label_column,
            "source_tail_column": self.source_tail_column,
            "known_at_column": self.known_at_column,
            "close_column": self.close_column,
            "high_column": self.high_column,
            "low_column": self.low_column,
            "log_return_column": self.log_return_column,
            "residual_signal_column": self.residual_signal_column,
            "leakage_safe": True,
            "production_approved": False,
            "production_writer_enabled": False,
        }


@dataclass(frozen=True)
class CrossAssetStateFutureOutcomePanelResult:
    status: str
    panel_frame: pd.DataFrame = field(repr=False)
    target_statuses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    mask_reason_counts: Mapping[str, int] = field(default_factory=dict)
    policy: CrossAssetStateOutcomeHorizonPolicy = field(default_factory=CrossAssetStateOutcomeHorizonPolicy)
    reason: str | None = None
    schema_version: int = CROSS_ASSET_STATE_ECONOMIC_DIAGNOSTIC_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_future_outcome_panel_result",
            "schema_version": int(self.schema_version),
            "status": self.status,
            "reason": self.reason,
            "row_count": int(len(self.panel_frame)),
            "target_columns": [
                column for column in self.panel_frame.columns if str(column) in DEFAULT_ECONOMIC_DIAGNOSTICS
            ],
            "target_statuses": {str(key): dict(value) for key, value in self.target_statuses.items()},
            "mask_reason_counts": {str(key): int(value) for key, value in self.mask_reason_counts.items()},
            "policy": self.policy.as_dict(),
            "production_approved": False,
            "production_writer_enabled": False,
        }


@dataclass(frozen=True)
class CrossAssetStateEconomicDiagnosticResult:
    status: str
    outcome_panel: CrossAssetStateFutureOutcomePanelResult
    target_scores: Mapping[str, Mapping[str, Any]]
    summary: Mapping[str, Any]
    policy: CrossAssetStateOutcomeHorizonPolicy = field(default_factory=CrossAssetStateOutcomeHorizonPolicy)
    economic_diagnostic_score: float | None = None
    reason: str | None = None
    schema_version: int = CROSS_ASSET_STATE_ECONOMIC_DIAGNOSTIC_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_economic_diagnostic_result",
            "schema_version": int(self.schema_version),
            "status": self.status,
            "economic_diagnostic_status": self.status,
            "economic_diagnostic_score": self.economic_diagnostic_score,
            "reason": self.reason,
            "outcome_panel": self.outcome_panel.as_dict(),
            "target_scores": {str(key): dict(value) for key, value in self.target_scores.items()},
            "summary": dict(self.summary),
            "policy": self.policy.as_dict(),
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
        }


@dataclass(frozen=True)
class CrossAssetStateEconomicDiagnosticContract:
    economic_diagnostic_status: str
    economic_diagnostic_score: float | None
    diagnostics: Sequence[str] = DEFAULT_ECONOMIC_DIAGNOSTICS
    horizon_steps: int | None = None
    outcome_window_policy_id: str | None = None
    missing_requirements: Sequence[str] = ()
    safe_to_compute_now: bool = False
    reason: str | None = None
    schema_version: int = CROSS_ASSET_STATE_ECONOMIC_DIAGNOSTIC_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_economic_diagnostic_contract",
            "schema_version": int(self.schema_version),
            "economic_diagnostic_status": self.economic_diagnostic_status,
            "economic_diagnostic_score": self.economic_diagnostic_score,
            "diagnostics": list(self.diagnostics),
            "horizon_steps": self.horizon_steps,
            "outcome_window_policy_id": self.outcome_window_policy_id,
            "safe_to_compute_now": bool(self.safe_to_compute_now),
            "reason": self.reason,
            "missing_requirements": list(self.missing_requirements),
            "leakage_policy": {
                "future_rows_must_be_joined_only_after_state_known_at_ts": True,
                "state_source_tail_ts_must_be_less_than_or_equal_to_known_at_ts": True,
                "outcome_known_at_ts_must_be_strictly_after_state_known_at_ts": True,
                "future_outcome_horizon_required": True,
                "missing_future_rows_handled_explicitly": True,
                "unsafe_forward_labels_invented": False,
            },
            "production_approved": False,
            "production_writer_enabled": False,
        }


def build_cross_asset_state_economic_diagnostic_contract(
    handoff: object | Mapping[str, Any],
    *,
    horizon_steps: int | None = None,
    outcome_window_policy_id: str | None = None,
) -> CrossAssetStateEconomicDiagnosticContract:
    refs = _future_outcome_panel_refs(handoff)
    missing: list[str] = []
    if not refs:
        missing.append("future_outcome_panel_refs")
    if horizon_steps is None:
        missing.append("explicit_horizon_steps")
    if outcome_window_policy_id is None:
        missing.append("outcome_window_policy_id")
    missing.extend(
        [
            "future_outcome_panel_constructor_run",
            "known_at_ts_source_tail_ts_alignment_validation",
            "future_missing_row_policy_validation",
        ]
    )
    if missing:
        return CrossAssetStateEconomicDiagnosticContract(
            economic_diagnostic_status=ECONOMIC_DIAGNOSTIC_STATUS_PENDING_NOT_COMPUTED,
            economic_diagnostic_score=None,
            horizon_steps=horizon_steps,
            outcome_window_policy_id=outcome_window_policy_id,
            missing_requirements=tuple(dict.fromkeys(missing)),
            safe_to_compute_now=False,
            reason="Leakage-safe future outcome panel inputs or validation artifacts are not present in the active non-production handoff.",
        )
    return CrossAssetStateEconomicDiagnosticContract(
        economic_diagnostic_status=ECONOMIC_DIAGNOSTIC_STATUS_BLOCKED_MISSING_OUTCOME_PANEL,
        economic_diagnostic_score=None,
        horizon_steps=horizon_steps,
        outcome_window_policy_id=outcome_window_policy_id,
        missing_requirements=("implementation_not_wired_for_future_outcome_panel",),
        safe_to_compute_now=False,
        reason="Future outcome refs are declared, but a computed Cross-Asset-State economic outcome panel artifact was not supplied.",
    )


def default_cross_asset_state_outcome_horizon_policy() -> CrossAssetStateOutcomeHorizonPolicy:
    return CrossAssetStateOutcomeHorizonPolicy()


def construct_cross_asset_state_future_outcome_panel(
    state_frame: pd.DataFrame,
    asset_ohlcv_frame: pd.DataFrame,
    *,
    anchor_ohlcv_frame: pd.DataFrame | None = None,
    core_ohlcv_frame: pd.DataFrame | None = None,
    policy: CrossAssetStateOutcomeHorizonPolicy | None = None,
) -> CrossAssetStateFutureOutcomePanelResult:
    cfg = policy or default_cross_asset_state_outcome_horizon_policy()
    leakage = _leakage_risk_columns(asset_ohlcv_frame.columns)
    if anchor_ohlcv_frame is not None:
        leakage.extend(f"anchor:{column}" for column in _leakage_risk_columns(anchor_ohlcv_frame.columns))
    if core_ohlcv_frame is not None:
        leakage.extend(f"core:{column}" for column in _leakage_risk_columns(core_ohlcv_frame.columns))
    leakage.extend(f"state:{column}" for column in _leakage_risk_columns(state_frame.columns, allowed={cfg.state_label_column}))
    if leakage:
        return CrossAssetStateFutureOutcomePanelResult(
            status=ECONOMIC_DIAGNOSTIC_STATUS_LEAKAGE_BLOCKED,
            reason=f"leakage-risk columns present: {list(leakage)}",
            panel_frame=_empty_panel(cfg),
            target_statuses={target: {"status": ECONOMIC_DIAGNOSTIC_STATUS_LEAKAGE_BLOCKED, "reason": "leakage-risk input column"} for target in DEFAULT_ECONOMIC_DIAGNOSTICS},
            policy=cfg,
        )
    required_state = (cfg.timestamp_column, cfg.band_column, cfg.state_label_column, cfg.source_tail_column, cfg.known_at_column)
    missing_state = [column for column in required_state if column not in state_frame.columns]
    if missing_state:
        return CrossAssetStateFutureOutcomePanelResult(
            status=ECONOMIC_DIAGNOSTIC_STATUS_ALIGNMENT_BLOCKED,
            reason=f"state frame missing required alignment columns: {missing_state}",
            panel_frame=_empty_panel(cfg),
            target_statuses={target: {"status": ECONOMIC_DIAGNOSTIC_STATUS_ALIGNMENT_BLOCKED, "reason": "state alignment columns missing"} for target in DEFAULT_ECONOMIC_DIAGNOSTICS},
            policy=cfg,
        )
    asset = _prepare_price_frame(asset_ohlcv_frame, cfg)
    benchmark = _prepare_price_frame(anchor_ohlcv_frame if anchor_ohlcv_frame is not None else core_ohlcv_frame, cfg)
    states = state_frame.copy()
    for column in (cfg.timestamp_column, cfg.source_tail_column, cfg.known_at_column):
        states[column] = pd.to_numeric(states[column], errors="coerce")
    states = states.dropna(subset=[cfg.timestamp_column, cfg.source_tail_column, cfg.known_at_column]).sort_values(cfg.timestamp_column)
    rows: list[dict[str, Any]] = []
    for _, state in states.iterrows():
        row = _outcome_row(state, asset=asset, benchmark=benchmark, policy=cfg)
        rows.append(row)
    panel = pd.DataFrame(rows)
    mask_reason_counts = _value_counts(panel.get("mask_reason", pd.Series(dtype=object)))
    target_statuses = _target_statuses(panel)
    missing_share = 0.0 if len(panel) == 0 else float(mask_reason_counts.get("future_horizon_incomplete", 0)) / float(len(panel))
    if len(panel) == 0:
        status = ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE
        reason = "no state rows available after alignment filtering"
    elif any(row.get("status") == ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED for row in target_statuses.values()):
        status = ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED
        reason = None
    else:
        status = ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE
        reason = "no future outcome targets could be computed"
    if missing_share > cfg.max_missing_future_row_share:
        reason = f"future missing-row share {missing_share:.6f} exceeds policy max {cfg.max_missing_future_row_share:.6f}"
    return CrossAssetStateFutureOutcomePanelResult(
        status=status,
        panel_frame=panel,
        target_statuses=target_statuses,
        mask_reason_counts=mask_reason_counts,
        policy=cfg,
        reason=reason,
    )


def validate_cross_asset_state_economic_usefulness(
    state_frame: pd.DataFrame,
    asset_ohlcv_frame: pd.DataFrame,
    *,
    anchor_ohlcv_frame: pd.DataFrame | None = None,
    core_ohlcv_frame: pd.DataFrame | None = None,
    policy: CrossAssetStateOutcomeHorizonPolicy | None = None,
    outcome_panel: CrossAssetStateFutureOutcomePanelResult | None = None,
) -> CrossAssetStateEconomicDiagnosticResult:
    cfg = policy or default_cross_asset_state_outcome_horizon_policy()
    panel = outcome_panel or construct_cross_asset_state_future_outcome_panel(
        state_frame,
        asset_ohlcv_frame,
        anchor_ohlcv_frame=anchor_ohlcv_frame,
        core_ohlcv_frame=core_ohlcv_frame,
        policy=cfg,
    )
    if panel.status in {ECONOMIC_DIAGNOSTIC_STATUS_LEAKAGE_BLOCKED, ECONOMIC_DIAGNOSTIC_STATUS_ALIGNMENT_BLOCKED}:
        return CrossAssetStateEconomicDiagnosticResult(
            status=panel.status,
            outcome_panel=panel,
            target_scores={},
            summary={"computed_target_count": 0, "status_counts": {panel.status: len(DEFAULT_ECONOMIC_DIAGNOSTICS)}},
            policy=cfg,
            reason=panel.reason,
        )
    target_scores = {target: _score_target(panel.panel_frame, target, cfg) for target in DEFAULT_ECONOMIC_DIAGNOSTICS}
    summary = _economic_summary(target_scores)
    score = _economic_score(target_scores)
    if summary["computed_target_count"]:
        status = ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED
        reason = None
    elif summary["status_counts"].get(ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE) == len(target_scores):
        status = ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE
        reason = "no requested Cross-Asset future outcome targets are available"
    else:
        status = ECONOMIC_DIAGNOSTIC_STATUS_NOT_APPLICABLE
        reason = "future outcomes were available but not separable across at least two state groups"
    return CrossAssetStateEconomicDiagnosticResult(
        status=status,
        outcome_panel=panel,
        target_scores=target_scores,
        summary=summary,
        policy=cfg,
        economic_diagnostic_score=score,
        reason=reason,
    )


def economic_status_fields(contract: CrossAssetStateEconomicDiagnosticContract | Mapping[str, Any]) -> dict[str, Any]:
    data = contract.as_dict() if hasattr(contract, "as_dict") else dict(contract)
    return {
        "economic_diagnostic_status": data.get("economic_diagnostic_status", ECONOMIC_DIAGNOSTIC_STATUS_PENDING_NOT_COMPUTED),
        "economic_diagnostic_score": data.get("economic_diagnostic_score"),
    }


def _outcome_row(state: pd.Series, *, asset: pd.DataFrame, benchmark: pd.DataFrame | None, policy: CrossAssetStateOutcomeHorizonPolicy) -> dict[str, Any]:
    ts = float(state[policy.timestamp_column])
    state_tail = float(state[policy.source_tail_column])
    state_known = float(state[policy.known_at_column])
    band = str(state[policy.band_column])
    horizon = policy.horizon_steps_for_band(band)
    out: dict[str, Any] = {
        policy.timestamp_column: int(ts) if ts.is_integer() else ts,
        policy.asset_id_column: state.get(policy.asset_id_column),
        policy.band_column: band,
        policy.state_label_column: state.get(policy.state_label_column),
        "state_source_tail_ts": state_tail,
        "state_known_at_ts": state_known,
        "horizon_steps": horizon,
        "outcome_window_policy_id": policy.outcome_window_policy_id,
        "future_missing_row_policy_id": policy.future_missing_row_policy_id,
        "outcome_status": ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE,
        "mask_reason": None,
    }
    if ts > state_tail or state_tail > state_known:
        out["mask_reason"] = "known_at_source_tail_alignment_invalid"
        return out
    current = asset[asset[policy.timestamp_column] <= state_tail].tail(1)
    future = asset[asset[policy.timestamp_column] > state_tail].head(horizon)
    if current.empty:
        out["mask_reason"] = "current_price_missing"
        return out
    if len(future) < horizon:
        out["mask_reason"] = "future_horizon_incomplete"
        return out
    outcome_known = _max_numeric(future, policy.known_at_column, fallback_column=policy.timestamp_column)
    out["outcome_known_at_ts"] = outcome_known
    if outcome_known <= state_known:
        out["mask_reason"] = "outcome_known_at_not_after_state_known_at"
        return out
    current_close = _finite_float(current.iloc[-1].get(policy.close_column))
    future_close = _finite_float(future.iloc[-1].get(policy.close_column))
    if current_close is None or future_close is None or current_close <= 0.0 or future_close <= 0.0:
        out["mask_reason"] = "price_missing_or_invalid"
        return out
    asset_return = float(math.log(future_close / current_close))
    out["future_asset_return"] = asset_return
    returns = _future_returns(current_close, future[policy.close_column])
    if returns.size >= 2:
        out[ECONOMIC_TARGET_REALIZED_VOLATILITY] = float(np.std(returns, ddof=0))
    low_col = policy.low_column if policy.low_column in future.columns else policy.close_column
    future_low = pd.to_numeric(future[low_col], errors="coerce").to_numpy(dtype=float)
    future_low = future_low[np.isfinite(future_low)]
    if future_low.size:
        out[ECONOMIC_TARGET_DRAWDOWN] = min(0.0, float(np.min(future_low) / current_close - 1.0))
    benchmark_return = None
    benchmark_returns = np.asarray([], dtype=float)
    if benchmark is not None:
        bench_current = benchmark[benchmark[policy.timestamp_column] <= state_tail].tail(1)
        bench_future = benchmark[benchmark[policy.timestamp_column].isin(list(future[policy.timestamp_column]))].head(horizon)
        if not bench_current.empty and len(bench_future) >= horizon:
            bench_current_close = _finite_float(bench_current.iloc[-1].get(policy.close_column))
            bench_future_close = _finite_float(bench_future.iloc[-1].get(policy.close_column))
            if bench_current_close is not None and bench_future_close is not None and bench_current_close > 0.0 and bench_future_close > 0.0:
                benchmark_return = float(math.log(bench_future_close / bench_current_close))
                benchmark_returns = _future_returns(bench_current_close, bench_future[policy.close_column])
                out[ECONOMIC_TARGET_RELATIVE_RETURN] = asset_return - benchmark_return
                shift = _future_correlation_shift(asset, benchmark, state_tail=state_tail, horizon=horizon, policy=policy, future_asset_returns=returns, future_benchmark_returns=benchmark_returns)
                if shift is not None:
                    out[ECONOMIC_TARGET_BETA_CORRELATION_SHIFT] = shift
    residual = _finite_float(state.get(policy.residual_signal_column))
    continuation_base = _finite_float(out.get(ECONOMIC_TARGET_RELATIVE_RETURN))
    if continuation_base is None:
        continuation_base = asset_return
    if residual is not None:
        out[ECONOMIC_TARGET_RESIDUAL_CONTINUATION] = residual * continuation_base
    out["outcome_status"] = ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED
    return out


def _prepare_price_frame(frame: pd.DataFrame | None, policy: CrossAssetStateOutcomeHorizonPolicy) -> pd.DataFrame | None:
    if frame is None:
        return None
    out = frame.copy()
    if policy.timestamp_column not in out.columns:
        return pd.DataFrame(columns=[policy.timestamp_column])
    out[policy.timestamp_column] = pd.to_numeric(out[policy.timestamp_column], errors="coerce")
    if policy.known_at_column in out.columns:
        out[policy.known_at_column] = pd.to_numeric(out[policy.known_at_column], errors="coerce")
    out = out.dropna(subset=[policy.timestamp_column]).sort_values(policy.timestamp_column).reset_index(drop=True)
    return out


def _future_returns(current_close: float, future_close: pd.Series) -> np.ndarray:
    closes = [float(current_close), *pd.to_numeric(future_close, errors="coerce").astype(float).tolist()]
    values = np.asarray(closes, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.any(values <= 0.0):
        return np.asarray([], dtype=float)
    return np.diff(np.log(values))


def _future_correlation_shift(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    state_tail: float,
    horizon: int,
    policy: CrossAssetStateOutcomeHorizonPolicy,
    future_asset_returns: np.ndarray,
    future_benchmark_returns: np.ndarray,
) -> float | None:
    if future_asset_returns.size < 2 or future_benchmark_returns.size < 2:
        return None
    future_corr = _corr(future_asset_returns, future_benchmark_returns)
    if future_corr is None:
        return None
    asset_past = asset[asset[policy.timestamp_column] <= state_tail].tail(horizon + 1)
    benchmark_past = benchmark[benchmark[policy.timestamp_column].isin(list(asset_past[policy.timestamp_column]))].tail(horizon + 1)
    if len(asset_past) < horizon + 1 or len(benchmark_past) < horizon + 1:
        return future_corr
    asset_returns = _future_returns(float(asset_past.iloc[0][policy.close_column]), asset_past.iloc[1:][policy.close_column])
    benchmark_returns = _future_returns(float(benchmark_past.iloc[0][policy.close_column]), benchmark_past.iloc[1:][policy.close_column])
    past_corr = _corr(asset_returns, benchmark_returns)
    return future_corr if past_corr is None else float(future_corr - past_corr)


def _score_target(panel: pd.DataFrame, target: str, policy: CrossAssetStateOutcomeHorizonPolicy) -> dict[str, Any]:
    if target not in panel.columns:
        return {"status": ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE, "reason": "target column not available", "target_column": None}
    frame = panel[[policy.state_label_column, target, "outcome_status"]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame[(frame["outcome_status"] == ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED) & np.isfinite(frame[target])]
    if len(frame) < policy.min_finite_rows_per_target:
        return {
            "status": ECONOMIC_DIAGNOSTIC_STATUS_NOT_APPLICABLE,
            "reason": "fewer than minimum finite outcome rows",
            "target_column": target,
            "finite_row_count": int(len(frame)),
        }
    labels = frame[policy.state_label_column].astype(str).to_numpy(dtype=object)
    values = frame[target].to_numpy(dtype=float)
    if len(set(labels.tolist())) < policy.min_state_groups:
        return {
            "status": ECONOMIC_DIAGNOSTIC_STATUS_NOT_APPLICABLE,
            "reason": "fewer than two finite state groups",
            "target_column": target,
            "finite_row_count": int(len(frame)),
            "per_state_distribution": _per_state_distribution(values, labels),
        }
    effect = _effect_summary(values, labels)
    return {
        "status": ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED,
        "reason": None,
        "target_column": target,
        "finite_row_count": int(len(frame)),
        "per_state_distribution": _per_state_distribution(values, labels),
        "effect_size": effect,
    }


def _target_statuses(panel: pd.DataFrame) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for target in DEFAULT_ECONOMIC_DIAGNOSTICS:
        if target in panel.columns and int(np.isfinite(pd.to_numeric(panel[target], errors="coerce")).sum()) > 0:
            statuses[target] = {"status": ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED, "finite_row_count": int(np.isfinite(pd.to_numeric(panel[target], errors="coerce")).sum())}
        else:
            statuses[target] = {"status": ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE, "reason": "target could not be computed from supplied frames"}
    return statuses


def _economic_summary(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in scores.values():
        status = str(row.get("status"))
        counts[status] = int(counts.get(status, 0) + 1)
    return {
        "target_count": int(len(scores)),
        "computed_target_count": int(counts.get(ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED, 0)),
        "not_available_target_count": int(counts.get(ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE, 0)),
        "not_applicable_target_count": int(counts.get(ECONOMIC_DIAGNOSTIC_STATUS_NOT_APPLICABLE, 0)),
        "status_counts": counts,
    }


def _economic_score(scores: Mapping[str, Mapping[str, Any]]) -> float | None:
    values = []
    for row in scores.values():
        if row.get("status") != ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED:
            continue
        effect = dict(row.get("effect_size") or {})
        cohen = effect.get("max_abs_pairwise_cohen_d")
        if cohen is not None and math.isfinite(float(cohen)):
            values.append(min(1.0, abs(float(cohen)) / 2.0))
    return None if not values else float(sum(values) / len(values))


def _effect_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    groups = {label: values[labels == label] for label in sorted(set(labels.tolist()), key=str)}
    means = [float(np.mean(group)) for group in groups.values() if group.size]
    medians = [float(np.median(group)) for group in groups.values() if group.size]
    pairwise = []
    for left, right in combinations(sorted(groups), 2):
        left_values = groups[left]
        right_values = groups[right]
        pairwise.append(
            {
                "left_label": str(left),
                "right_label": str(right),
                "mean_difference": float(np.mean(left_values) - np.mean(right_values)),
                "median_difference": float(np.median(left_values) - np.median(right_values)),
                "cohen_d": _cohen_d(left_values, right_values),
            }
        )
    cohen_values = [abs(float(row["cohen_d"])) for row in pairwise if row["cohen_d"] is not None]
    return {
        "mean_spread": None if not means else float(max(means) - min(means)),
        "median_spread": None if not medians else float(max(medians) - min(medians)),
        "max_abs_pairwise_cohen_d": None if not cohen_values else float(max(cohen_values)),
        "pairwise": pairwise,
    }


def _per_state_distribution(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    out = {}
    for label in sorted(set(labels.tolist()), key=str):
        group = values[labels == label]
        group = group[np.isfinite(group)]
        if group.size:
            out[str(label)] = {
                "count": int(group.size),
                "mean": float(np.mean(group)),
                "median": float(np.median(group)),
                "std": float(np.std(group, ddof=0)),
                "q25": float(np.quantile(group, 0.25)),
                "q75": float(np.quantile(group, 0.75)),
            }
    return out


def _cohen_d(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    pooled = math.sqrt((float(np.var(left, ddof=0)) + float(np.var(right, ddof=0))) / 2.0)
    if pooled <= 1e-12:
        return None
    return float((float(np.mean(left)) - float(np.mean(right))) / pooled)


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    size = min(left.size, right.size)
    if size < 2:
        return None
    left = left[:size]
    right = right[:size]
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    left = left[mask]
    right = right[mask]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _empty_panel(policy: CrossAssetStateOutcomeHorizonPolicy) -> pd.DataFrame:
    return pd.DataFrame(columns=[policy.timestamp_column, policy.asset_id_column, policy.band_column, policy.state_label_column, "mask_reason"])


def _leakage_risk_columns(columns: Sequence[object], *, allowed: set[str] | None = None) -> list[str]:
    allowed = set(allowed or set())
    out = []
    for column in columns:
        text = str(column).lower()
        if str(column) in allowed:
            continue
        if any(token in text for token in LEAKAGE_RISK_TOKENS):
            out.append(str(column))
    return out


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _max_numeric(frame: pd.DataFrame, column: str, *, fallback_column: str) -> float:
    source = column if column in frame.columns else fallback_column
    values = pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series.fillna("").astype(str):
        if not value:
            continue
        counts[value] = int(counts.get(value, 0) + 1)
    return counts


def _future_outcome_panel_refs(handoff: object | Mapping[str, Any]) -> tuple[Any, ...]:
    if isinstance(handoff, Mapping):
        refs = handoff.get("future_outcome_panel_refs") or ()
    else:
        refs = getattr(handoff, "future_outcome_panel_refs", ()) or ()
    if isinstance(refs, (str, bytes)):
        return (refs,)
    return tuple(refs)


__all__ = [
    "CROSS_ASSET_STATE_ECONOMIC_DIAGNOSTIC_SCHEMA_VERSION",
    "DEFAULT_ECONOMIC_DIAGNOSTICS",
    "ECONOMIC_DIAGNOSTIC_STATUS_BLOCKED_MISSING_OUTCOME_PANEL",
    "ECONOMIC_DIAGNOSTIC_STATUS_COMPUTED",
    "ECONOMIC_DIAGNOSTIC_STATUS_ALIGNMENT_BLOCKED",
    "ECONOMIC_DIAGNOSTIC_STATUS_LEAKAGE_BLOCKED",
    "ECONOMIC_DIAGNOSTIC_STATUS_NOT_APPLICABLE",
    "ECONOMIC_DIAGNOSTIC_STATUS_NOT_AVAILABLE",
    "ECONOMIC_DIAGNOSTIC_STATUS_PENDING_NOT_COMPUTED",
    "ECONOMIC_TARGET_BETA_CORRELATION_SHIFT",
    "ECONOMIC_TARGET_DRAWDOWN",
    "ECONOMIC_TARGET_REALIZED_VOLATILITY",
    "ECONOMIC_TARGET_RELATIVE_RETURN",
    "ECONOMIC_TARGET_RESIDUAL_CONTINUATION",
    "CrossAssetStateEconomicDiagnosticContract",
    "CrossAssetStateEconomicDiagnosticResult",
    "CrossAssetStateFutureOutcomePanelResult",
    "CrossAssetStateOutcomeHorizonPolicy",
    "build_cross_asset_state_economic_diagnostic_contract",
    "construct_cross_asset_state_future_outcome_panel",
    "default_cross_asset_state_outcome_horizon_policy",
    "economic_status_fields",
    "validate_cross_asset_state_economic_usefulness",
]
