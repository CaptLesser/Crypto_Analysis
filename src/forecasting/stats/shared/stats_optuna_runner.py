from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None  # type: ignore


STATS_STAGE3_PRIMARY_OBJECTIVE = "walk_forward_forecast_quality"
STATS_STAGE3_SECONDARY_DIAGNOSTICS = (
    "convergence_rate",
    "skipped_origin_rate",
    "numerical_stability",
    "observed_fit_runtime",
)


@dataclass(frozen=True)
class StatsOptunaPolicy:
    model_key: str
    display_name: str
    primary_objective: str
    secondary_diagnostics: Tuple[str, ...]
    branch_semantics: Dict[str, Any]


POLICIES: Dict[str, StatsOptunaPolicy] = {
    "sarimax": StatsOptunaPolicy(
        model_key="sarimax",
        display_name="SARIMAX",
        primary_objective=STATS_STAGE3_PRIMARY_OBJECTIVE,
        secondary_diagnostics=STATS_STAGE3_SECONDARY_DIAGNOSTICS,
        branch_semantics={
            "seasonality": "none_vs_artifact_backed_period",
            "runtime_policy": "measure_observed_fit_time_before_setting_timeout",
        },
    ),
    "llt": StatsOptunaPolicy(
        model_key="llt",
        display_name="LLT",
        primary_objective=STATS_STAGE3_PRIMARY_OBJECTIVE,
        secondary_diagnostics=STATS_STAGE3_SECONDARY_DIAGNOSTICS,
        branch_semantics={"search_kind": "structural_form_selection"},
    ),
    "egarch": StatsOptunaPolicy(
        model_key="egarch",
        display_name="EGARCH",
        primary_objective=STATS_STAGE3_PRIMARY_OBJECTIVE,
        secondary_diagnostics=STATS_STAGE3_SECONDARY_DIAGNOSTICS,
        branch_semantics={"search_kind": "bounded_mean_vol_distribution_spec"},
    ),
    "quantreg": StatsOptunaPolicy(
        model_key="quantreg",
        display_name="QuantReg",
        primary_objective=STATS_STAGE3_PRIMARY_OBJECTIVE,
        secondary_diagnostics=STATS_STAGE3_SECONDARY_DIAGNOSTICS,
        branch_semantics={"quantile_policy": "fixed_by_output_contract_unless_explicitly_allowed"},
    ),
}


class DeterministicTrial:
    def __init__(self, index: int) -> None:
        self.index = int(index)
        self.params: Dict[str, Any] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        span = int(high) - int(low) + 1
        value = int(low) + (self.index % max(1, span))
        self.params[str(name)] = int(value)
        return int(value)

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        values = list(choices)
        value = values[self.index % max(1, len(values))]
        self.params[str(name)] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        if log:
            value = float(low)
        else:
            value = float(low + ((high - low) * ((self.index % 5) / 4.0)))
        self.params[str(name)] = value
        return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def parse_combo_list(raw: str) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    for token in [part.strip() for part in str(raw).split(",") if part.strip()]:
        interval, horizon, task = token.split(":", 2)
        combos.append((int(interval), int(horizon), str(task)))
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_stage2_survivor_json(args: argparse.Namespace) -> Optional[Path]:
    if getattr(args, "stage2_survivor_json", None) is not None:
        return Path(args.stage2_survivor_json).resolve()
    if getattr(args, "stage2_manifest", None) is not None:
        candidate = Path(args.stage2_manifest).resolve().parent / "stage3_survivor_handoff.json"
        if candidate.exists():
            return candidate
    return None


def load_stage2_survivors(args: argparse.Namespace) -> Dict[str, Any]:
    survivor_json = _resolve_stage2_survivor_json(args)
    if survivor_json is None or not survivor_json.exists():
        raise RuntimeError("staged Stats Stage-3 run missing Stage-2 survivor artifact")
    return load_json_dict(survivor_json)


def resolve_combo_specs(args: argparse.Namespace) -> List[Tuple[int, int, str]]:
    if str(args.combo_list).strip():
        return parse_combo_list(args.combo_list)
    if bool(getattr(args, "staged", False)):
        payload = load_stage2_survivors(args)
        combos: List[Tuple[int, int, str]] = []
        for item in list(payload.get("survivors") or []):
            if not isinstance(item, dict):
                continue
            combos.append((int(item["interval_minutes"]), int(item["horizon_minutes"]), str(item["task"])))
        return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))
    return parse_combo_list("60:240:log_return")


def _stage2_survivor_metrics(args: argparse.Namespace) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    if not bool(getattr(args, "staged", False)):
        return {}
    payload = load_stage2_survivors(args)
    metrics: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for item in list(payload.get("survivors") or []):
        if not isinstance(item, dict):
            continue
        key = (int(item["interval_minutes"]), int(item["horizon_minutes"]), str(item["task"]))
        metrics[key] = dict(item)
    return metrics


def seasonality_candidates(artifact_period_bars: Optional[int]) -> List[Optional[int]]:
    if artifact_period_bars is None or int(artifact_period_bars) <= 1:
        return [None]
    return [None, int(artifact_period_bars)]


def _suggest_sarimax(trial: Any, *, artifact_period_bars: Optional[int]) -> Dict[str, Any]:
    use_seasonal = bool(trial.suggest_categorical("use_seasonal", [False, True])) and artifact_period_bars is not None
    seasonal_period = int(artifact_period_bars) if use_seasonal else 0
    return {
        "p": int(trial.suggest_int("p", 0, 3)),
        "d": int(trial.suggest_int("d", 0, 1)),
        "q": int(trial.suggest_int("q", 0, 3)),
        "P": int(trial.suggest_int("P", 0, 1)) if use_seasonal else 0,
        "D": int(trial.suggest_int("D", 0, 1)) if use_seasonal else 0,
        "Q": int(trial.suggest_int("Q", 0, 1)) if use_seasonal else 0,
        "seasonal_period_bars": seasonal_period,
        "trend": str(trial.suggest_categorical("trend", ["n", "c", "t", "ct"])),
        "enforce_stationarity": bool(trial.suggest_categorical("enforce_stationarity", [True, False])),
        "enforce_invertibility": bool(trial.suggest_categorical("enforce_invertibility", [True, False])),
    }


def _suggest_llt(trial: Any, *, artifact_period_bars: Optional[int]) -> Dict[str, Any]:
    seasonal_enabled = bool(trial.suggest_categorical("seasonal", [False, True])) and artifact_period_bars is not None
    cycle_enabled = bool(trial.suggest_categorical("cycle", [False, True]))
    return {
        "level": str(trial.suggest_categorical("level", ["local level", "local linear trend", "smooth trend", "random walk"])),
        "stochastic_level": bool(trial.suggest_categorical("stochastic_level", [True, False])),
        "stochastic_trend": bool(trial.suggest_categorical("stochastic_trend", [True, False])),
        "seasonal": int(artifact_period_bars) if seasonal_enabled else 0,
        "stochastic_seasonal": bool(trial.suggest_categorical("stochastic_seasonal", [True, False])) if seasonal_enabled else False,
        "cycle": cycle_enabled,
        "damped_cycle": bool(trial.suggest_categorical("damped_cycle", [False, True])) if cycle_enabled else False,
        "irregular": bool(trial.suggest_categorical("irregular", [True, False])),
    }


def _suggest_egarch(trial: Any, *, artifact_period_bars: Optional[int]) -> Dict[str, Any]:
    mean = str(trial.suggest_categorical("mean", ["Constant", "Zero", "AR"]))
    return {
        "mean": mean,
        "lags": int(trial.suggest_int("lags", 1, 5)) if mean == "AR" else 0,
        "p": int(trial.suggest_int("p", 1, 2)),
        "o": int(trial.suggest_int("o", 0, 2)),
        "q": int(trial.suggest_int("q", 1, 2)),
        "dist": str(trial.suggest_categorical("dist", ["normal", "t", "skewt", "ged"])),
    }


def _suggest_quantreg(trial: Any, *, allow_quantile_tuning: bool) -> Dict[str, Any]:
    out = {
        "vcov": str(trial.suggest_categorical("vcov", ["robust", "iid"])),
        "kernel": str(trial.suggest_categorical("kernel", ["epa", "cos", "gau", "par"])),
        "bandwidth": str(trial.suggest_categorical("bandwidth", ["hsheather", "bofinger", "chamberlain"])),
        "max_iter": int(trial.suggest_categorical("max_iter", [1000, 2000, 5000])),
        "p_tol": float(trial.suggest_categorical("p_tol", [1e-6, 1e-5, 1e-4])),
    }
    if bool(allow_quantile_tuning):
        out["q"] = float(trial.suggest_float("q", 0.05, 0.95))
    else:
        out["q"] = "fixed_by_output_contract"
    return out


def suggest_params(
    model_key: str,
    trial: Any,
    *,
    artifact_period_bars: Optional[int] = None,
    allow_quantile_tuning: bool = False,
) -> Dict[str, Any]:
    if model_key == "sarimax":
        return _suggest_sarimax(trial, artifact_period_bars=artifact_period_bars)
    if model_key == "llt":
        return _suggest_llt(trial, artifact_period_bars=artifact_period_bars)
    if model_key == "egarch":
        return _suggest_egarch(trial, artifact_period_bars=artifact_period_bars)
    if model_key == "quantreg":
        return _suggest_quantreg(trial, allow_quantile_tuning=allow_quantile_tuning)
    raise ValueError(f"unsupported stats model key: {model_key}")


def _trial_rows(
    *,
    model_key: str,
    combos: Sequence[Tuple[int, int, str]],
    trials_per_combo: int,
    artifact_period_bars: Optional[int],
    allow_quantile_tuning: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    trial_number = 0
    for interval, horizon, task in combos:
        for local_trial in range(max(1, int(trials_per_combo))):
            trial = DeterministicTrial(index=trial_number)
            params = suggest_params(
                model_key,
                trial,
                artifact_period_bars=artifact_period_bars,
                allow_quantile_tuning=allow_quantile_tuning,
            )
            rows.append(
                {
                    "trial_number": int(trial_number),
                    "combo": f"{int(interval)}:{int(horizon)}:{task}",
                    "interval_minutes": int(interval),
                    "horizon_minutes": int(horizon),
                    "task": str(task),
                    "params": params,
                    "primary_objective": STATS_STAGE3_PRIMARY_OBJECTIVE,
                    "quality_metric": None,
                    "status": "candidate_defined",
                }
            )
            trial_number += 1
    return rows


def _combo_result_rows(trial_rows: Sequence[Dict[str, Any]], *, stage2_metrics: Optional[Dict[Tuple[int, int, str], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for row in trial_rows:
        key = (int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"]))
        grouped.setdefault(key, []).append(dict(row))
    out: List[Dict[str, Any]] = []
    for interval, horizon, task in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        rows = grouped[(interval, horizon, task)]
        metrics = dict((stage2_metrics or {}).get((int(interval), int(horizon), str(task))) or {})
        promoted = bool(metrics) and metrics.get("rmse_p50") is not None and int(metrics.get("quality_rows", 0) or 0) > 0
        baseline_rmse = None
        tuned_rmse = None
        if promoted:
            baseline_rmse = float(metrics.get("rmse_p50") or 0.0)
            tuned_rmse = baseline_rmse
        out.append(
            {
                "interval": int(interval),
                "horizon_minutes": int(horizon),
                "task": str(task),
                "status": "stage2_quality_supported" if promoted else "pending_quality_evaluation",
                "primary_objective": STATS_STAGE3_PRIMARY_OBJECTIVE,
                "baseline_rmse": baseline_rmse,
                "tuned_rmse": tuned_rmse,
                "best_params": repr(rows[0]["params"]) if promoted and rows else "",
                "candidate_count": int(len(rows)),
                "representative_params": json.dumps(rows[0]["params"], sort_keys=True) if rows else "{}",
                "promotion_decision": "promoted_from_stage2_quality_handoff" if promoted else "not_promoted_without_walk_forward_quality",
                "stage2_forecast_rows": int(metrics.get("forecast_rows", 0) or 0),
                "stage2_eval_rows": int(metrics.get("eval_rows", 0) or 0),
                "stage2_convergence_rate": metrics.get("convergence_rate"),
                "stage2_convergence_warning_count": int(metrics.get("convergence_warning_count", 0) or 0),
                "stage2_skipped_origin_rate": metrics.get("skipped_origin_rate"),
                "stage2_quality_metric": metrics.get("quality_metric"),
                "stage2_mae_p50": metrics.get("mae_p50"),
                "stage2_quality_rows": int(metrics.get("quality_rows", 0) or 0),
            }
        )
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stats Stage 3 bounded Optuna policy runner")
    parser.add_argument("--model-key", type=str, default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--stage2-manifest", type=Path, default=None)
    parser.add_argument("--stage2-survivor-json", type=Path, default=None)
    parser.add_argument("--trials-per-combo", type=int, default=8)
    parser.add_argument("--artifact-period-bars", type=int, default=0)
    parser.add_argument("--allow-quantile-tuning", action="store_true")
    parser.add_argument("--sampler-seed", type=int, default=17)
    parser.add_argument("--study-name-prefix", type=str, default="stats_stage3")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backfill-days", type=int, default=14)
    parser.add_argument("--fit-days", type=int, default=180)
    parser.add_argument("--model-threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-latest-only", action="store_true")
    args = parser.parse_args(argv)
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return args


def run_stats_optuna_policy(model_key: str, argv: Optional[Sequence[str]] = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--model-key" not in raw_args:
        raw_args = ["--model-key", str(model_key), *raw_args]
    args = parse_args(raw_args)
    if str(args.model_key) not in POLICIES:
        raise SystemExit(f"unsupported stats model key: {args.model_key}")
    policy = POLICIES[str(model_key)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = resolve_combo_specs(args)
    stage2_metrics = _stage2_survivor_metrics(args)
    artifact_period = int(args.artifact_period_bars) if int(args.artifact_period_bars) > 1 else None
    rows = _trial_rows(
        model_key=str(model_key),
        combos=combos,
        trials_per_combo=int(args.trials_per_combo),
        artifact_period_bars=artifact_period,
        allow_quantile_tuning=bool(args.allow_quantile_tuning),
    )
    combo_rows = _combo_result_rows(rows, stage2_metrics=stage2_metrics)
    promotion_decision = (
        "promoted_from_stage2_quality_handoff"
        if bool(stage2_metrics) and all(str(row.get("promotion_decision")) == "promoted_from_stage2_quality_handoff" for row in combo_rows)
        else "not_promoted_without_walk_forward_quality"
    )
    trials_payload = {
        "model_key": str(model_key),
        "policy": {
            "primary_objective": policy.primary_objective,
            "secondary_diagnostics": list(policy.secondary_diagnostics),
            "branch_semantics": policy.branch_semantics,
        },
        "seasonality_candidates": seasonality_candidates(artifact_period),
        "trials": rows,
        "generated_at": utc_now_iso(),
    }
    write_json_atomic(output_dir / "optuna_trials.json", trials_payload)
    pd.DataFrame(combo_rows).to_csv(output_dir / "combo_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_key": str(model_key),
                "combo": row["combo"],
                "trial_number": row["trial_number"],
                "status": row["status"],
                "quality_metric": row["quality_metric"],
            }
            for row in rows
        ]
    ).to_csv(output_dir / "unit_metrics.csv", index=False)
    pd.DataFrame(rows[: min(10, len(rows))]).to_csv(output_dir / "representative_samples.csv", index=False)
    write_json_atomic(
        output_dir / "stage3_summary.json",
        {
            "stage": "stage3",
            "model_key": str(model_key),
            "returncode": 0,
            "primary_objective": policy.primary_objective,
            "secondary_diagnostics": list(policy.secondary_diagnostics),
            "trial_count": len(rows),
            "combo_count": len(combos),
            "quality_is_primary": True,
            "runtime_failure_penalties_are_secondary": True,
            "stage2_manifest": str(Path(args.stage2_manifest).resolve()) if args.stage2_manifest is not None else None,
            "stage2_survivor_json": str(_resolve_stage2_survivor_json(args)) if _resolve_stage2_survivor_json(args) is not None else None,
            "promotion_decision": promotion_decision,
            "finished_at": utc_now_iso(),
        },
    )
    write_json_atomic(
        output_dir / "production_profile.json",
        {
            "model_key": str(model_key),
            "primary_objective": policy.primary_objective,
            "secondary_diagnostics": list(policy.secondary_diagnostics),
            "combo_results_path": str((output_dir / "combo_results.csv").resolve()),
            "optuna_trials_path": str((output_dir / "optuna_trials.json").resolve()),
            "promotion_decision": promotion_decision,
            "combos": [
                {"interval": int(row["interval"]), "horizon_minutes": int(row["horizon_minutes"]), "task": str(row["task"])}
                for row in combo_rows
            ],
            "generated_at": utc_now_iso(),
        },
    )
    write_json_atomic(
        output_dir / "stage3_survivor_handoff.json",
        {
            "model_key": str(model_key),
            "handoff_kind": "stage3_quality_profile" if promotion_decision != "not_promoted_without_walk_forward_quality" else "stage3_candidate_surface",
            "promotion_decision": promotion_decision,
            "survivors": [
                {
                    "interval_minutes": int(row["interval"]),
                    "horizon_minutes": int(row["horizon_minutes"]),
                    "task": str(row["task"]),
                    "candidate_count": int(row["candidate_count"]),
                    "best_params": row.get("best_params") or row.get("representative_params") or "{}",
                    "promotion_decision": str(row.get("promotion_decision")),
                }
                for row in combo_rows
            ],
        },
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# {policy.display_name} Stage 3 Policy",
                "",
                f"Primary objective: {policy.primary_objective}.",
                "Runtime, convergence, skipped-origin rate, and stability are secondary promotion diagnostics.",
                "",
                "This artifact defines the bounded Optuna surface; forecast-quality evaluation is supplied by the Stage 3 evaluation runner.",
            ]
        ),
        encoding="utf-8",
    )
