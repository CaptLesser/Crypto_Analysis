from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import PathConfigError, pipeline_io_env, require_pipeline_io, selected_profile
from src.forecasting.common.sandbox_paths import (
    SandboxOutputRoots,
    assert_write_allowed,
    resolve_sandbox_output_roots,
    sandbox_env_for_subprocess,
)
from src.forecasting.ml.shared.numeric_runner_diagnostics import summarize_diagnostics
from src.forecasting.ml.shared.test_branch_function_telemetry import (
    FUNCTION_EVENTS_JSONL,
    write_rollups as write_function_telemetry_rollups,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    assert_write_allowed(path, "test branch json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "test branch json temp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def family_logs_dir(run_root: Path) -> Path:
    return run_root / "logs"


def stage0_dir(run_root: Path) -> Path:
    return run_root / "stage0"


def stage0_log_path(run_root: Path) -> Path:
    return family_logs_dir(run_root) / "stage0.log"


def stage0_profile_path(run_root: Path, profile_json_name: str) -> Path:
    return stage0_dir(run_root) / str(profile_json_name)


def stage0_candidates_path(run_root: Path, candidates_csv_name: str) -> Path:
    return stage0_dir(run_root) / str(candidates_csv_name)


def stage0_complete(run_root: Path, profile_json_name: str, candidates_csv_name: str) -> bool:
    return stage0_profile_path(run_root, profile_json_name).exists() and stage0_candidates_path(run_root, candidates_csv_name).exists()


def stage1_selection_path(paths: Any) -> Path:
    return paths.stage1_dir / "feature_profile_selection.json"


def stage1_meta_path(paths: Any) -> Path:
    return paths.stage1_dir / "feature_experiment_run_meta.json"


def stage1_complete(paths: Any) -> bool:
    return stage1_selection_path(paths).exists() and stage1_meta_path(paths).exists()


def discover_latest_stage2_manifest(stage2_root: Path) -> Optional[Path]:
    manifests = sorted(stage2_root.glob("run=*/diagnostic_manifest.json"))
    if not manifests:
        return None
    return manifests[-1].resolve()


def stage2_survivor_json_from_manifest(manifest_path: Optional[Path]) -> Optional[Path]:
    if manifest_path is None:
        return None
    candidate = manifest_path.parent / "stage3_survivor_handoff.json"
    return candidate if candidate.exists() else None


def stage2_complete(paths: Any) -> bool:
    manifest_path = discover_latest_stage2_manifest(paths.stage2_root)
    survivor_path = stage2_survivor_json_from_manifest(manifest_path)
    return manifest_path is not None and survivor_path is not None


def stage3_complete(paths: Any, required_stage3_files: Sequence[str]) -> bool:
    return all((paths.stage3_dir / name).exists() for name in required_stage3_files)


def run_complete(
    run_root: Path,
    *,
    model_order: Sequence[str],
    model_paths_fn: Callable[[Path, str], Any],
    stage0_profile_json_name: str,
    stage0_candidates_csv_name: str,
    required_stage3_files: Sequence[str],
) -> bool:
    return stage0_complete(run_root, stage0_profile_json_name, stage0_candidates_csv_name) and all(
        stage1_complete(model_paths_fn(run_root, key))
        and stage2_complete(model_paths_fn(run_root, key))
        and stage3_complete(model_paths_fn(run_root, key), required_stage3_files)
        for key in model_order
    )


def latest_incomplete_run(base_output_dir: Path, run_complete_fn: Callable[[Path], bool]) -> Optional[Path]:
    if not base_output_dir.exists():
        return None
    runs = sorted((path for path in base_output_dir.glob("run=*") if path.is_dir()), key=lambda path: path.name)
    for path in reversed(runs):
        if not run_complete_fn(path):
            return path.resolve()
    return None


def resolve_run_root(args: Any, latest_incomplete_run_fn: Callable[[Path], Optional[Path]]) -> Path:
    project_root = args.project_root.resolve()
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()
    if args.resume_run:
        run_name = str(args.resume_run).strip()
        if not run_name.startswith("run="):
            run_name = f"run={run_name}"
        run_root = (base_output_dir / run_name).resolve()
        if not run_root.exists():
            raise RuntimeError(f"Requested orchestrator resume run does not exist: {run_root}")
        return run_root
    if args.run_id:
        run_name = str(args.run_id).strip()
        if not run_name.startswith("run="):
            run_name = f"run={run_name}"
        return (base_output_dir / run_name).resolve()
    if not bool(args.no_resume_latest):
        latest = latest_incomplete_run_fn(base_output_dir)
        if latest is not None:
            return latest
    return (base_output_dir / f"run={utc_now_stamp()}").resolve()


def command_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    command.extend([flag, str(value)])


def remove_tree(path: Path) -> None:
    if path.exists():
        assert_write_allowed(path, "test branch cleanup")
        shutil.rmtree(path)


def add_sandbox_output_args(parser: Any) -> None:
    parser.add_argument(
        "--sandbox-output-root",
        type=Path,
        default=None,
        help="Enable sandbox output mode and redirect write-class artifacts under this root.",
    )


def _argv_tokens(argv: Optional[Sequence[str]]) -> Sequence[str]:
    return tuple(sys.argv[1:] if argv is None else argv)


def _has_cli_flag(argv: Optional[Sequence[str]], flag: str) -> bool:
    prefix = f"{flag}="
    return any(str(token) == flag or str(token).startswith(prefix) for token in _argv_tokens(argv))


def finalize_sandbox_output_args(
    args: Any,
    argv: Optional[Sequence[str]],
    *,
    default_output_dir: Path,
    family_key: str,
) -> Any:
    roots = resolve_sandbox_output_roots(args)
    if not roots.enabled:
        return args
    output_dir_explicit = _has_cli_flag(argv, "--output-dir")
    if not output_dir_explicit and Path(args.output_dir) == Path(default_output_dir):
        args.output_dir = roots.runtime_artifact_root / "test_branch_numeric" / str(family_key)
    project_root = Path(getattr(args, "project_root", Path.cwd())).resolve()
    output_dir = Path(args.output_dir)
    resolved_output_dir = output_dir.resolve() if output_dir.is_absolute() else (project_root / output_dir).resolve()
    assert_write_allowed(resolved_output_dir, f"{family_key} test branch output dir", roots=roots)
    return args


def test_branch_child_env(args: Any, source_env: Dict[str, str]) -> Dict[str, str]:
    profile = str(getattr(args, "profile", "") or selected_profile(default="pipeline_test", env=source_env))
    parquet_root = getattr(args, "parquet_root", None)
    try:
        io_config = require_pipeline_io(profile=profile)
        base_env = pipeline_io_env(io_config, source_env)
    except PathConfigError:
        if parquet_root is None or not str(parquet_root).strip():
            raise
        base_env = {str(key): str(value) for key, value in source_env.items()}
    if parquet_root is not None and str(parquet_root).strip():
        source_root = str(Path(parquet_root).resolve())
        base_env.setdefault("PIPELINE_PARQUET_ROOT", source_root)
        base_env.setdefault("PIPELINE_SOURCE_PARQUET_ROOT", source_root)
        base_env.setdefault("PIPELINE_SOURCE_OHLCVT_ROOT", source_root)
    roots = resolve_sandbox_output_roots(args, env=base_env)
    return sandbox_env_for_subprocess(roots, base_env)


def assert_test_branch_sandbox_launch(
    args: Any,
    run_root: Path,
    env: Dict[str, str],
    *,
    family_key: str,
) -> SandboxOutputRoots:
    roots = resolve_sandbox_output_roots(args)
    if not roots.enabled:
        return roots
    assert_write_allowed(Path(run_root), f"{family_key} test branch run root", roots=roots)
    for key in (
        "PIPELINE_SANDBOX_OUTPUT_ROOT",
        "PIPELINE_SANDBOX_PARQUET_ROOT",
        "PIPELINE_SANDBOX_LOG_ROOT",
        "PIPELINE_SANDBOX_STATE_ROOT",
        "PIPELINE_SANDBOX_TMP_ROOT",
        "PIPELINE_SANDBOX_DIAGNOSTICS_ROOT",
        "PIPELINE_SANDBOX_MANIFEST_ROOT",
        "PIPELINE_SANDBOX_OPTUNA_ROOT",
        "PIPELINE_SANDBOX_CATBOOST_TRAIN_DIR",
        "PIPELINE_SANDBOX_RUNTIME_ARTIFACT_ROOT",
    ):
        raw = str(env.get(key, "") or "").strip()
        if not raw:
            raise RuntimeError(f"Sandbox mode requires {key} for {family_key} test branch children")
        assert_write_allowed(Path(raw), f"{family_key} child env {key}", roots=roots)
    return roots


def test_branch_stage_tmp_root(
    env: Dict[str, str],
    *,
    family_key: str,
    run_name: str,
    model_key: str,
    stage_name: str,
    fallback_root: Path,
) -> Path:
    roots = _sandbox_roots_from_child_env(env)
    if roots.enabled:
        temp_root = (
            roots.tmp_root
            / "test_branch_numeric"
            / str(family_key)
            / str(run_name)
            / str(model_key)
            / str(stage_name)
        ).resolve()
        assert_write_allowed(temp_root, f"{family_key} test branch temp", roots=roots)
    else:
        temp_root = Path(fallback_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


def _sandbox_roots_from_child_env(env: Dict[str, str]) -> SandboxOutputRoots:
    if str(env.get("PIPELINE_SANDBOX_MODE", "") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return resolve_sandbox_output_roots(env=env)
    root = Path(str(env["PIPELINE_SANDBOX_OUTPUT_ROOT"])).resolve()
    return SandboxOutputRoots(
        enabled=True,
        root=root,
        parquet_root=Path(str(env.get("PIPELINE_SANDBOX_PARQUET_ROOT", root / "parquet"))).resolve(),
        log_root=Path(str(env.get("PIPELINE_SANDBOX_LOG_ROOT", root / "logs"))).resolve(),
        diagnostics_root=Path(str(env.get("PIPELINE_SANDBOX_DIAGNOSTICS_ROOT", root / "diagnostics"))).resolve(),
        manifest_root=Path(str(env.get("PIPELINE_SANDBOX_MANIFEST_ROOT", root / "manifests"))).resolve(),
        state_root=Path(str(env.get("PIPELINE_SANDBOX_STATE_ROOT", root / "state"))).resolve(),
        tmp_root=Path(str(env.get("PIPELINE_SANDBOX_TMP_ROOT", root / "tmp"))).resolve(),
        optuna_root=Path(str(env.get("PIPELINE_SANDBOX_OPTUNA_ROOT", root / "optuna"))).resolve(),
        catboost_train_dir=Path(str(env.get("PIPELINE_SANDBOX_CATBOOST_TRAIN_DIR", root / "tmp" / "catboost_train"))).resolve(),
        regime_definition_root=Path(str(env.get("PIPELINE_SANDBOX_REGIME_DEFINITION_ROOT", root / "regime_definitions"))).resolve(),
        runtime_artifact_root=Path(str(env.get("PIPELINE_SANDBOX_RUNTIME_ARTIFACT_ROOT", root / "runtime"))).resolve(),
    )


def collect_stage3_outputs(paths: Any, required_stage3_files: Sequence[str]) -> Dict[str, str]:
    return {name: str((paths.stage3_dir / name).resolve()) for name in required_stage3_files if (paths.stage3_dir / name).exists()}


def canonical_profile_model_dir(args: Any, *, diagnostics_root_name: str, model_key: str) -> Path:
    roots = resolve_sandbox_output_roots(args)
    if roots.enabled:
        root = roots.diagnostics_root
    else:
        project_root = Path(getattr(args, "project_root", Path.cwd())).resolve()
        root = project_root / "logs" / "diagnostics"
    return (root / str(diagnostics_root_name) / str(model_key)).resolve()


def _copy_file_atomic(source: Path, destination: Path) -> None:
    assert_write_allowed(destination, "canonical test branch profile")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(destination)
    assert_write_allowed(tmp, "canonical test branch profile temp")
    shutil.copy2(source, tmp)
    atomic_replace(tmp, destination)


def publish_canonical_model_profile(
    args: Any,
    *,
    diagnostics_root_name: str,
    model_key: str,
    paths: Any,
    required_stage3_files: Sequence[str],
    stage2_survivor_json: Optional[Path],
    run_root: Path,
    family: str,
) -> Dict[str, Any]:
    if not stage1_complete(paths) or not stage3_complete(paths, required_stage3_files):
        return {}
    survivor_source = Path(stage2_survivor_json).resolve() if stage2_survivor_json is not None else None
    if survivor_source is None or not survivor_source.exists():
        return {}

    model_root = canonical_profile_model_dir(args, diagnostics_root_name=diagnostics_root_name, model_key=model_key)
    stage1_root = model_root / "stage1"
    stage2_root = model_root / "stage2"
    stage3_root = model_root / "stage3"

    source_feature_profile = stage1_selection_path(paths).resolve()
    canonical_feature_profile = stage1_root / "feature_profile_selection.json"
    _copy_file_atomic(source_feature_profile, canonical_feature_profile)
    if stage1_meta_path(paths).exists():
        _copy_file_atomic(stage1_meta_path(paths).resolve(), stage1_root / "feature_experiment_run_meta.json")

    survivor_payload = load_json_dict(survivor_source)
    survivor_payload["feature_profile_json"] = str(canonical_feature_profile.resolve())
    survivor_payload["source_handoff_json"] = str(survivor_source)
    survivor_payload["source_run_root"] = str(Path(run_root).resolve())
    write_json_atomic(stage2_root / "stage3_survivor_handoff.json", survivor_payload)

    stage3_artifacts: Dict[str, str] = {}
    for name in required_stage3_files:
        source = (paths.stage3_dir / str(name)).resolve()
        if source.exists():
            destination = stage3_root / str(name)
            _copy_file_atomic(source, destination)
            stage3_artifacts[str(name)] = str(destination.resolve())

    manifest = {
        "generated_utc": utc_now_iso(),
        "family": str(family),
        "model_key": str(model_key),
        "source_run_root": str(Path(run_root).resolve()),
        "canonical_root": str(model_root),
        "stage1": {
            "feature_profile_json": str(canonical_feature_profile.resolve()),
            "source_feature_profile_json": str(source_feature_profile),
        },
        "stage2": {
            "stage3_survivor_handoff_json": str((stage2_root / "stage3_survivor_handoff.json").resolve()),
            "source_stage3_survivor_handoff_json": str(survivor_source),
        },
        "stage3": {
            "output_dir": str(stage3_root.resolve()),
            "artifacts": stage3_artifacts,
        },
    }
    write_json_atomic(model_root / "current_profile_manifest.json", manifest)
    return manifest


def publish_canonical_family_profiles(
    args: Any,
    *,
    run_root: Path,
    diagnostics_root_name: str,
    model_order: Sequence[str],
    model_paths_fn: Callable[[Path, str], Any],
    required_stage3_files: Sequence[str],
    family: str,
) -> Dict[str, Any]:
    published: Dict[str, Any] = {}
    for model_key in model_order:
        paths = model_paths_fn(Path(run_root), str(model_key))
        stage2_root = getattr(paths, "stage2_root", getattr(paths, "stage2_dir", None))
        if stage2_root is None:
            continue
        stage2_path = Path(stage2_root)
        manifest = discover_latest_stage2_manifest(stage2_path)
        if manifest is None and (stage2_path / "diagnostic_manifest.json").exists():
            manifest = (stage2_path / "diagnostic_manifest.json").resolve()
        survivor = stage2_survivor_json_from_manifest(manifest)
        model_manifest = publish_canonical_model_profile(
            args,
            diagnostics_root_name=diagnostics_root_name,
            model_key=str(model_key),
            paths=paths,
            required_stage3_files=required_stage3_files,
            stage2_survivor_json=survivor,
            run_root=Path(run_root),
            family=str(family),
        )
        if model_manifest:
            published[str(model_key)] = model_manifest
    return published


HEALTH_REPORT_FILE = "test_branch_health.json"
_LOG_TS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<message>.*)$")
_EXIT_RE = re.compile(r"\bEXIT\s+(?P<returncode>-?\d+)\b")
_LOG_PATTERNS: Dict[str, re.Pattern[str]] = {
    "traceback": re.compile(r"\bTraceback \(most recent call last\):"),
    "convergence_warning": re.compile(r"\bConvergenceWarning\b"),
    "user_warning": re.compile(r"\bUserWarning\b"),
    "future_warning": re.compile(r"\bFutureWarning\b"),
    "runtime_warning": re.compile(r"\bRuntimeWarning\b"),
    "schema_error": re.compile(r"\bschema-error\b|schema contract error|schema mismatch", re.IGNORECASE),
    "contract_error": re.compile(r"\bcontract error\b", re.IGNORECASE),
    "no_predictions": re.compile(r"\bno-predictions\b|no eval metrics generated", re.IGNORECASE),
    "ineligible": re.compile(r"\bineligible[_ -]", re.IGNORECASE),
    "atomic_replace_error": re.compile(r"atomic replace|\.tmp|FileExistsError|PermissionError", re.IGNORECASE),
    "exception": re.compile(r"\b(?:Exception|Error):\s+"),
}


def _parse_utc(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _duration_seconds(started: Any, finished: Any) -> Optional[float]:
    start_dt = _parse_utc(started)
    finish_dt = _parse_utc(finished)
    if start_dt is None or finish_dt is None:
        return None
    return max(0.0, float((finish_dt - start_dt).total_seconds()))


def scan_log_health(log_path: Path, *, sample_limit: int = 8) -> Dict[str, Any]:
    path = Path(log_path)
    counts: Dict[str, int] = {key: 0 for key in _LOG_PATTERNS}
    samples: List[Dict[str, Any]] = []
    exit_codes: List[int] = []
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    line_count = 0
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "line_count": 0,
            "size_bytes": 0,
            "pattern_counts": counts,
            "exit_codes": [],
            "exit_nonzero_count": 0,
            "samples": [],
            "duration_s": None,
        }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line_count = int(line_no)
            stripped = line.rstrip("\r\n")
            ts_match = _LOG_TS_RE.match(stripped)
            if ts_match:
                ts_value = str(ts_match.group("ts"))
                first_ts = first_ts or ts_value
                last_ts = ts_value
            exit_match = _EXIT_RE.search(stripped)
            if exit_match:
                try:
                    exit_codes.append(int(exit_match.group("returncode")))
                except Exception:
                    pass
            matched_keys: List[str] = []
            for key, pattern in _LOG_PATTERNS.items():
                if pattern.search(stripped):
                    counts[key] = int(counts.get(key, 0)) + 1
                    matched_keys.append(str(key))
            if matched_keys and len(samples) < int(sample_limit):
                samples.append({"line": int(line_no), "patterns": matched_keys, "text": stripped[:500]})
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "line_count": int(line_count),
        "size_bytes": int(stat.st_size),
        "pattern_counts": counts,
        "exit_codes": exit_codes,
        "exit_nonzero_count": int(sum(1 for code in exit_codes if int(code) != 0)),
        "samples": samples,
        "first_timestamp_utc": first_ts,
        "last_timestamp_utc": last_ts,
        "duration_s": _duration_seconds(first_ts, last_ts),
    }


def process_health_from_log(log_path: Path) -> Dict[str, Any]:
    meta_path = Path(log_path).with_suffix(".process.json")
    payload = load_json_dict(meta_path)
    if not payload:
        return {"process_meta_path": str(meta_path), "exists": False}
    duration_s = _duration_seconds(payload.get("started_utc"), payload.get("finished_utc"))
    return {
        "process_meta_path": str(meta_path),
        "exists": True,
        "status": payload.get("status"),
        "pid": payload.get("pid"),
        "returncode": payload.get("returncode"),
        "started_utc": payload.get("started_utc"),
        "finished_utc": payload.get("finished_utc"),
        "duration_s": duration_s,
        "error": payload.get("error"),
        "env_hints": payload.get("env_hints") if isinstance(payload.get("env_hints"), dict) else {},
    }


def _iter_stage2_summaries(manifest_path: Optional[Path]) -> List[Dict[str, Any]]:
    if manifest_path is None or not Path(manifest_path).exists():
        return []
    manifest = load_json_dict(Path(manifest_path))
    summaries: List[Dict[str, Any]] = []
    for row in list(manifest.get("runs") or []):
        if not isinstance(row, dict):
            continue
        paths = row.get("paths")
        if not isinstance(paths, dict):
            continue
        raw_path = paths.get("run_summary")
        if not raw_path:
            continue
        summary_path = Path(str(raw_path))
        payload = load_json_dict(summary_path)
        if payload:
            payload["_summary_path"] = str(summary_path)
            summaries.append(payload)
    return summaries


def _summarize_manifest_runner_diagnostics(manifest: Dict[str, Any]) -> Dict[str, Any]:
    run_manifest_raw = str(manifest.get("run_manifest_json") or "").strip()
    if not run_manifest_raw:
        return {}
    run_manifest = load_json_dict(Path(run_manifest_raw))
    diagnostics_raw = str(run_manifest.get("diagnostics_jsonl") or "").strip()
    if not diagnostics_raw:
        return {}
    diagnostics_path = Path(diagnostics_raw)
    if not diagnostics_path.exists():
        return {"path": str(diagnostics_path), "exists": False}
    try:
        payload = summarize_diagnostics(diagnostics_path, top_n=10)
    except Exception:
        return {"path": str(diagnostics_path), "exists": False}
    payload["exists"] = True
    return payload


def summarize_stage2_quality(manifest_path: Optional[Path]) -> Dict[str, Any]:
    summaries = _iter_stage2_summaries(manifest_path)
    manifest = load_json_dict(Path(manifest_path)) if manifest_path is not None and Path(manifest_path).exists() else {}
    manifest_run_metrics = [
        row.get("metrics")
        for row in list(manifest.get("runs") or [])
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict)
    ]
    if manifest_run_metrics:
        status_counts: Dict[str, int] = {}
        nonfinite_metric_count = 0
        forecast_count_total = 0
        eval_count_total = 0
        metric_cells = 0
        convergence_warning_count = 0
        nonconverged_fit_count = 0
        fit_retry_count = 0
        fit_retry_resolved_count = 0
        for metrics in manifest_run_metrics:
            status = str(metrics.get("quality_status") or metrics.get("status") or "unknown")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            forecast_count_total += int(metrics.get("forecast_rows", 0) or 0)
            eval_count_total += int(metrics.get("eval_rows", 0) or 0)
            convergence_warning_count += int(metrics.get("convergence_warning_count", 0) or 0)
            nonconverged_fit_count += int(metrics.get("nonconverged_fit_count", 0) or 0)
            fit_retry_count += int(metrics.get("fit_retry_count", 0) or 0)
            fit_retry_resolved_count += int(metrics.get("fit_retry_resolved_count", 0) or 0)
            if metrics.get("quality_rows") is not None or metrics.get("rmse_p50") is not None or metrics.get("mae_p50") is not None:
                metric_cells += 1
            for key in ("rmse_p50", "mae_p50", "quality_rows", "forecast_rows", "eval_rows"):
                if metrics.get(key) is not None and _safe_float(metrics.get(key)) is None:
                    nonfinite_metric_count += 1
        return {
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "summary_count": int(len(summaries)),
            "manifest_run_metrics_count": int(len(manifest_run_metrics)),
            "status_counts": status_counts,
            "metric_cells": int(metric_cells),
            "forecast_count_total": int(forecast_count_total),
            "eval_count_total": int(eval_count_total),
            "nonfinite_metric_count": int(nonfinite_metric_count),
            "empty_unclassified_count": 0,
            "empty_unclassified_summaries": [],
            "ineligible_count": int(sum(int(count) for key, count in status_counts.items() if str(key).startswith("ineligible"))),
            "ineligible_samples": [],
            "baseline_comparable_count": 0,
            "model_better_than_baseline_count": 0,
            "model_worse_than_baseline_count": 0,
            "model_equal_to_baseline_count": 0,
            "rmse_to_baseline_ratio_p50": None,
            "rmse_to_baseline_ratio_p90": None,
            "convergence_warning_count": int(convergence_warning_count),
            "nonconverged_fit_count": int(nonconverged_fit_count),
            "fit_retry_count": int(fit_retry_count),
            "fit_retry_resolved_count": int(fit_retry_resolved_count),
            "fit_diagnostics": {},
            "runner_diagnostics": _summarize_manifest_runner_diagnostics(manifest),
        }
    if not summaries and isinstance(manifest.get("survivors"), list):
        status_counts: Dict[str, int] = {}
        nonfinite_metric_count = 0
        forecast_count_total = 0
        metric_cells = 0
        convergence_warning_count = 0
        for row in list(manifest.get("survivors") or []):
            if not isinstance(row, dict):
                continue
            status = str(row.get("quality_status") or row.get("status") or "unknown")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            forecast_count_total += int(row.get("forecast_rows", 0) or 0)
            convergence_warning_count += int(row.get("convergence_warning_count", 0) or 0)
            if row.get("quality_rows") is not None or row.get("rmse_p50") is not None or row.get("mae_p50") is not None:
                metric_cells += 1
            for key in ("rmse_p50", "mae_p50", "quality_rows"):
                if row.get(key) is not None and _safe_float(row.get(key)) is None:
                    nonfinite_metric_count += 1
        return {
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "summary_count": 0,
            "survivor_count": int(len(manifest.get("survivors") or [])),
            "status_counts": status_counts,
            "metric_cells": int(metric_cells),
            "forecast_count_total": int(forecast_count_total),
            "nonfinite_metric_count": int(nonfinite_metric_count),
            "empty_unclassified_count": 0,
            "empty_unclassified_summaries": [],
            "ineligible_count": int(sum(int(count) for key, count in status_counts.items() if str(key).startswith("ineligible"))),
            "ineligible_samples": [],
            "baseline_comparable_count": 0,
            "model_better_than_baseline_count": 0,
            "model_worse_than_baseline_count": 0,
            "model_equal_to_baseline_count": 0,
            "rmse_to_baseline_ratio_p50": None,
            "rmse_to_baseline_ratio_p90": None,
            "convergence_warning_count": int(convergence_warning_count),
            "fit_diagnostics": {},
        }
    status_counts: Dict[str, int] = {}
    metric_cells = 0
    forecast_count_total = 0
    nonfinite_metric_count = 0
    baseline_comparable = 0
    model_better = 0
    model_worse = 0
    model_equal = 0
    rmse_ratios: List[float] = []
    empty_unclassified: List[str] = []
    ineligible: List[Dict[str, Any]] = []
    fit_diagnostics: Dict[str, int] = {}
    for summary in summaries:
        diag_aggregate = summary.get("diag_aggregate") if isinstance(summary.get("diag_aggregate"), dict) else {}
        for key, value in diag_aggregate.items():
            if not str(key).startswith("fit_diag_"):
                continue
            try:
                fit_diagnostics[str(key)] = int(fit_diagnostics.get(str(key), 0) + int(value))
            except Exception:
                continue
        quality = summary.get("quality") if isinstance(summary.get("quality"), dict) else {}
        status = str(summary.get("quality_status") or quality.get("status") or "")
        accuracy = summary.get("accuracy") if isinstance(summary.get("accuracy"), dict) else {}
        by_asset = accuracy.get("by_asset_target_horizon") if isinstance(accuracy, dict) else None
        has_metrics = isinstance(by_asset, dict) and any(bool(v) for v in by_asset.values() if isinstance(v, dict))
        if not status:
            status = "metrics_present" if has_metrics else "empty_unclassified"
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        if not has_metrics:
            if str(status).startswith("ineligible"):
                ineligible.append({"summary_path": summary.get("_summary_path"), "status": status, "reason": summary.get("quality_reason") or quality.get("reason")})
            else:
                empty_unclassified.append(str(summary.get("_summary_path")))
            continue
        for combo_map in by_asset.values():
            if not isinstance(combo_map, dict):
                continue
            for metrics in combo_map.values():
                if not isinstance(metrics, dict):
                    continue
                metric_cells += 1
                forecast_count_total += int(metrics.get("forecast_count", 0) or 0)
                rmse = _safe_float(metrics.get("rmse"))
                mae = _safe_float(metrics.get("mae"))
                if rmse is None or mae is None:
                    nonfinite_metric_count += 1
                baseline_rmse = _safe_float(metrics.get("baseline_rmse"))
                if rmse is None or baseline_rmse is None or baseline_rmse <= 0.0:
                    continue
                baseline_comparable += 1
                ratio = float(rmse) / float(baseline_rmse)
                rmse_ratios.append(float(ratio))
                if ratio < 0.999:
                    model_better += 1
                elif ratio > 1.001:
                    model_worse += 1
                else:
                    model_equal += 1
    rmse_ratios_sorted = sorted(rmse_ratios)
    ratio_p50 = rmse_ratios_sorted[len(rmse_ratios_sorted) // 2] if rmse_ratios_sorted else None
    ratio_p90 = rmse_ratios_sorted[min(len(rmse_ratios_sorted) - 1, int(len(rmse_ratios_sorted) * 0.9))] if rmse_ratios_sorted else None
    return {
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "summary_count": int(len(summaries)),
        "status_counts": status_counts,
        "metric_cells": int(metric_cells),
        "forecast_count_total": int(forecast_count_total),
        "nonfinite_metric_count": int(nonfinite_metric_count),
        "empty_unclassified_count": int(len(empty_unclassified)),
        "empty_unclassified_summaries": empty_unclassified[:10],
        "ineligible_count": int(len(ineligible)),
        "ineligible_samples": ineligible[:10],
        "baseline_comparable_count": int(baseline_comparable),
        "model_better_than_baseline_count": int(model_better),
        "model_worse_than_baseline_count": int(model_worse),
        "model_equal_to_baseline_count": int(model_equal),
        "rmse_to_baseline_ratio_p50": ratio_p50,
        "rmse_to_baseline_ratio_p90": ratio_p90,
        "convergence_warning_count": 0,
        "fit_diagnostics": fit_diagnostics,
    }


def summarize_stage3_combo_results(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"path": str(path) if path is not None else None, "exists": False}
    rows = 0
    status_counts: Dict[str, int] = {}
    nonfinite_numeric = 0
    better = 0
    worse = 0
    equal = 0
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            status = str(row.get("status") or row.get("promotion_decision") or "unknown")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            for key, value in row.items():
                if key is None:
                    continue
                lowered = str(key).lower()
                if not any(token in lowered for token in ("rmse", "mae", "score", "ratio", "delta")):
                    continue
                raw = str(value or "").strip()
                if not raw:
                    continue
                if _safe_float(raw) is None:
                    nonfinite_numeric += 1
            baseline = _safe_float(row.get("baseline_rmse"))
            tuned = _safe_float(row.get("tuned_rmse"))
            delta = _safe_float(row.get("rmse_delta"))
            if delta is None and baseline is not None and tuned is not None:
                delta = float(tuned) - float(baseline)
            if delta is None:
                continue
            if delta < -1e-12:
                better += 1
            elif delta > 1e-12:
                worse += 1
            else:
                equal += 1
    return {
        "path": str(path),
        "exists": True,
        "rows": int(rows),
        "status_counts": status_counts,
        "nonfinite_numeric_count": int(nonfinite_numeric),
        "tuned_better_than_baseline_count": int(better),
        "tuned_worse_than_baseline_count": int(worse),
        "tuned_equal_to_baseline_count": int(equal),
    }


def _compare_stage2_work_across_models(model_health: Dict[str, Any], *, limit: int = 20) -> List[Dict[str, Any]]:
    by_combo: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for model_key, payload in model_health.items():
        if not isinstance(payload, dict):
            continue
        quality = payload.get("stage2_quality") if isinstance(payload.get("stage2_quality"), dict) else {}
        diagnostics = quality.get("runner_diagnostics") if isinstance(quality.get("runner_diagnostics"), dict) else {}
        for row in list(diagnostics.get("slowest_combos") or []):
            if not isinstance(row, dict):
                continue
            try:
                key = (int(row["interval"]), int(row["horizon_minutes"]), str(row["task"]))
            except Exception:
                continue
            elapsed = _safe_float(row.get("shard_elapsed_s"))
            if elapsed is None:
                continue
            by_combo.setdefault(key, []).append(
                {
                    "model_key": str(model_key),
                    "work_elapsed_s": round(float(elapsed), 3),
                    "forecast_rows": int(row.get("forecast_rows", 0) or 0),
                    "done_units": int(row.get("done_units", 0) or 0),
                    "seconds_per_forecast_row": row.get("shard_seconds_per_forecast_row"),
                    "mean_unit_elapsed_s": row.get("mean_unit_elapsed_s"),
                }
            )
    comparisons: List[Dict[str, Any]] = []
    for key, rows in by_combo.items():
        if len(rows) < 2:
            continue
        sorted_rows = sorted(rows, key=lambda row: float(row.get("work_elapsed_s", 0.0) or 0.0), reverse=True)
        positive_elapsed = [float(row.get("work_elapsed_s", 0.0) or 0.0) for row in sorted_rows if float(row.get("work_elapsed_s", 0.0) or 0.0) > 0.0]
        fastest = min(positive_elapsed) if positive_elapsed else None
        slowest = max(positive_elapsed) if positive_elapsed else None
        for row in sorted_rows:
            elapsed = float(row.get("work_elapsed_s", 0.0) or 0.0)
            row["ratio_to_fastest_model"] = round(elapsed / fastest, 3) if fastest and elapsed > 0.0 else None
        comparisons.append(
            {
                "combo": f"{key[0]}:{key[1]}:{key[2]}",
                "interval_minutes": int(key[0]),
                "horizon_minutes": int(key[1]),
                "task": str(key[2]),
                "slowest_to_fastest_ratio": round(float(slowest) / float(fastest), 3) if fastest and slowest else None,
                "models": sorted_rows,
            }
        )
    return sorted(
        comparisons,
        key=lambda row: float(row.get("slowest_to_fastest_ratio", 0.0) or 0.0),
        reverse=True,
    )[: max(1, int(limit))]


def collect_test_branch_health(*, run_root: Path, family: str, stage0_status: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    bottlenecks: List[Dict[str, Any]] = []

    def _stage_health(model_key: str, stage_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        log_path_raw = payload.get("log_path") if isinstance(payload, dict) else None
        log_scan = scan_log_health(Path(str(log_path_raw))) if log_path_raw else {}
        process_scan = process_health_from_log(Path(str(log_path_raw))) if log_path_raw else {}
        elapsed = process_scan.get("duration_s") if process_scan.get("duration_s") is not None else log_scan.get("duration_s")
        if elapsed is not None:
            bottlenecks.append({"model_key": model_key, "stage": stage_name, "elapsed_s": round(float(elapsed), 3), "log_path": log_path_raw})
        pattern_counts = log_scan.get("pattern_counts") if isinstance(log_scan.get("pattern_counts"), dict) else {}
        if int(log_scan.get("exit_nonzero_count", 0) or 0) > 0 or int(pattern_counts.get("traceback", 0) or 0) > 0:
            issues.append({"severity": "error", "model_key": model_key, "stage": stage_name, "kind": "stage_failure_signal", "log_path": log_path_raw, "counts": pattern_counts})
        for key in ("contract_error", "schema_error", "atomic_replace_error"):
            if int(pattern_counts.get(key, 0) or 0) > 0:
                issues.append({"severity": "error", "model_key": model_key, "stage": stage_name, "kind": key, "log_path": log_path_raw, "count": int(pattern_counts.get(key, 0) or 0)})
        if int(pattern_counts.get("convergence_warning", 0) or 0) > 0:
            issues.append({"severity": "warning", "model_key": model_key, "stage": stage_name, "kind": "convergence_warning", "log_path": log_path_raw, "count": int(pattern_counts.get("convergence_warning", 0) or 0)})
        return {"log": log_scan, "process": process_scan}

    stage0_health = _stage_health("family", "stage0", stage0_status)
    model_health: Dict[str, Any] = {}
    for model_key, model_status in models.items():
        if not isinstance(model_status, dict):
            continue
        stage_payloads = {name: model_status.get(name) for name in ("stage1", "stage2", "stage3") if isinstance(model_status.get(name), dict)}
        model_entry: Dict[str, Any] = {"stages": {}}
        for stage_name, stage_payload in stage_payloads.items():
            model_entry["stages"][stage_name] = _stage_health(str(model_key), stage_name, stage_payload)
        stage2_manifest_raw = (stage_payloads.get("stage2") or {}).get("diagnostic_manifest_json")
        stage2_quality = summarize_stage2_quality(Path(str(stage2_manifest_raw))) if stage2_manifest_raw else summarize_stage2_quality(None)
        model_entry["stage2_quality"] = stage2_quality
        if int(stage2_quality.get("empty_unclassified_count", 0) or 0) > 0:
            issues.append({"severity": "error", "model_key": str(model_key), "stage": "stage2", "kind": "unclassified_empty_eval_summary", "count": int(stage2_quality.get("empty_unclassified_count", 0) or 0)})
        if int(stage2_quality.get("nonfinite_metric_count", 0) or 0) > 0:
            issues.append({"severity": "error", "model_key": str(model_key), "stage": "stage2", "kind": "nonfinite_stage2_metric", "count": int(stage2_quality.get("nonfinite_metric_count", 0) or 0)})
        if int(stage2_quality.get("convergence_warning_count", 0) or 0) > 0:
            issues.append({"severity": "warning", "model_key": str(model_key), "stage": "stage2", "kind": "stage2_convergence_warning_count", "count": int(stage2_quality.get("convergence_warning_count", 0) or 0)})
        if int(stage2_quality.get("nonconverged_fit_count", 0) or 0) > 0:
            issues.append({"severity": "warning", "model_key": str(model_key), "stage": "stage2", "kind": "stage2_nonconverged_fit_count", "count": int(stage2_quality.get("nonconverged_fit_count", 0) or 0)})
        fit_diagnostics = stage2_quality.get("fit_diagnostics") if isinstance(stage2_quality.get("fit_diagnostics"), dict) else {}
        if int(fit_diagnostics.get("fit_diag_convergence_warning_count", 0) or 0) > 0:
            issues.append({"severity": "warning", "model_key": str(model_key), "stage": "stage2", "kind": "stage2_fit_convergence_warning_count", "count": int(fit_diagnostics.get("fit_diag_convergence_warning_count", 0) or 0)})
        if int(stage2_quality.get("model_worse_than_baseline_count", 0) or 0) > int(stage2_quality.get("model_better_than_baseline_count", 0) or 0):
            issues.append({"severity": "warning", "model_key": str(model_key), "stage": "stage2", "kind": "stage2_underperforms_persistence_majority", "counts": {"better": stage2_quality.get("model_better_than_baseline_count"), "worse": stage2_quality.get("model_worse_than_baseline_count")}})
        stage3_artifacts = (stage_payloads.get("stage3") or {}).get("artifacts")
        combo_path = None
        if isinstance(stage3_artifacts, dict) and stage3_artifacts.get("combo_results.csv"):
            combo_path = Path(str(stage3_artifacts.get("combo_results.csv")))
        elif isinstance(stage_payloads.get("stage3"), dict) and stage_payloads["stage3"].get("path"):
            candidate = Path(str(stage_payloads["stage3"].get("path"))) / "combo_results.csv"
            combo_path = candidate if candidate.exists() else None
        stage3_quality = summarize_stage3_combo_results(combo_path)
        model_entry["stage3_quality"] = stage3_quality
        if int(stage3_quality.get("nonfinite_numeric_count", 0) or 0) > 0:
            issues.append({"severity": "error", "model_key": str(model_key), "stage": "stage3", "kind": "nonfinite_stage3_numeric", "count": int(stage3_quality.get("nonfinite_numeric_count", 0) or 0)})
        if int(stage3_quality.get("tuned_worse_than_baseline_count", 0) or 0) > 0:
            issues.append({"severity": "warning", "model_key": str(model_key), "stage": "stage3", "kind": "tuned_combo_worse_than_baseline", "count": int(stage3_quality.get("tuned_worse_than_baseline_count", 0) or 0)})
        model_health[str(model_key)] = model_entry

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return {
        "generated_utc": utc_now_iso(),
        "family": str(family),
        "run_root": str(run_root),
        "status": "error" if error_count else "warning" if warning_count else "ok",
        "error_count": int(error_count),
        "warning_count": int(warning_count),
        "issues": issues,
        "slowest_stages": sorted(bottlenecks, key=lambda row: float(row.get("elapsed_s", 0.0) or 0.0), reverse=True)[:10],
        "stage2_cross_model_work": _compare_stage2_work_across_models(model_health),
        "stage0": stage0_health,
        "models": model_health,
    }


def write_test_branch_health(run_root: Path, *, family: str, stage0_status: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    health = collect_test_branch_health(run_root=Path(run_root), family=str(family), stage0_status=stage0_status, models=models)
    try:
        telemetry_rollups = write_function_telemetry_rollups(Path(run_root))
    except Exception as exc:
        telemetry_rollups = {
            "event_count": 0,
            "warning": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    health["function_telemetry"] = {
        "events_jsonl": str((Path(run_root) / FUNCTION_EVENTS_JSONL).resolve()),
        **telemetry_rollups,
    }
    write_json_atomic(Path(run_root) / HEALTH_REPORT_FILE, health)
    return health
