from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.sandbox_paths import assert_write_allowed


FUNCTION_EVENTS_JSONL = "function_events.jsonl"
FUNCTION_HEALTH_SUMMARY_CSV = "function_health_summary.csv"
PHASE_TIMING_SUMMARY_CSV = "phase_timing_summary.csv"
EMPTY_OUTPUT_SUMMARY_CSV = "empty_output_summary.csv"
PRODUCTION_ALIGNMENT_SUMMARY_MD = "production_alignment_summary.md"
TELEMETRY_WARNING_JSONL = "function_telemetry_warnings.jsonl"

STABLE_REASON_CODES = {
    "missing_source_root",
    "missing_source_table",
    "no_assets",
    "no_mature_assets",
    "insufficient_history",
    "cold_start_not_reached",
    "source_read_empty",
    "feature_load_empty",
    "regime_load_empty",
    "seasonality_load_empty",
    "required_columns_missing",
    "labels_empty",
    "join_empty",
    "train_frame_empty",
    "validation_frame_empty",
    "fit_failed",
    "predict_returned_empty",
    "postprocess_dropped_all_rows",
    "validation_dropped_all_rows",
    "write_skipped_empty_output",
    "artifact_missing",
    "profile_missing",
    "exception",
    "stage_already_complete",
    "stage_artifact_missing",
    "study_missing",
    "study_load_failed",
    "objective_dataset_empty",
    "trial_pruned",
    "trial_failed",
    "no_completed_trials",
    "best_trial_missing",
    "profile_validation_failed",
    "profile_write_failed",
}

EVENT_FIELDS = (
    "timestamp_utc",
    "run_id",
    "family",
    "model",
    "stage",
    "combo_key",
    "interval_minutes",
    "horizon_minutes",
    "target",
    "task",
    "asset",
    "asset_count",
    "cohort_id",
    "function_name",
    "module_name",
    "phase_name",
    "parent_phase",
    "status",
    "reason_code",
    "elapsed_seconds",
    "input_rows",
    "output_rows",
    "input_columns_count",
    "output_columns_count",
    "required_columns_missing",
    "key_null_counts",
    "exception_type",
    "exception_message",
    "source_path",
    "output_path",
    "artifact_profile_source",
    "trial_count",
    "pruned_count",
    "failed_count",
    "completed_count",
    "best_value",
    "failure_samples",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _clean_reason_code(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw if raw in STABLE_REASON_CODES else "exception" if "exception" in raw.lower() else raw


def _compact_message(value: Any, *, limit: int = 240) -> str:
    raw = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return raw[:limit]


def event_path(run_root: Path) -> Path:
    return Path(run_root) / FUNCTION_EVENTS_JSONL


def _warning_path(run_root: Path) -> Path:
    return Path(run_root) / TELEMETRY_WARNING_JSONL


def _safe_append_jsonl(path: Path, payload: Mapping[str, Any], *, warning_root: Optional[Path] = None) -> None:
    try:
        assert_write_allowed(path, "test branch function telemetry")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")
    except Exception as exc:
        if warning_root is None:
            return
        try:
            warning = {
                "timestamp_utc": utc_now_iso(),
                "status": "failed",
                "reason_code": "exception",
                "exception_type": type(exc).__name__,
                "exception_message": _compact_message(exc),
                "target_path": str(path),
            }
            warning_path = _warning_path(Path(warning_root))
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            with warning_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(warning, sort_keys=True) + "\n")
        except Exception:
            pass


def object_shape_summary(obj: Any, *, prefix: str) -> Dict[str, Any]:
    if obj is None:
        return {}
    rows = None
    columns = None
    shape = getattr(obj, "shape", None)
    if isinstance(shape, tuple):
        if len(shape) >= 1:
            rows = _safe_int(shape[0])
        if len(shape) >= 2:
            columns = _safe_int(shape[1])
    if rows is None:
        try:
            rows = len(obj)  # type: ignore[arg-type]
        except Exception:
            rows = None
    if columns is None:
        cols = getattr(obj, "columns", None)
        if cols is not None:
            try:
                columns = len(cols)
            except Exception:
                columns = None
    out: Dict[str, Any] = {}
    if rows is not None:
        out[f"{prefix}_rows"] = int(rows)
    if columns is not None:
        out[f"{prefix}_columns_count"] = int(columns)
    return out


def required_columns_missing(obj: Any, required_columns: Optional[Sequence[str]]) -> List[str]:
    if not required_columns:
        return []
    cols = getattr(obj, "columns", None)
    if cols is None:
        return list(required_columns)
    try:
        present = {str(col) for col in cols}
    except Exception:
        return list(required_columns)
    return [str(col) for col in required_columns if str(col) not in present]


def compact_null_summary(obj: Any, key_columns: Optional[Sequence[str]], *, limit: int = 12) -> Dict[str, int]:
    if obj is None or not key_columns:
        return {}
    cols = getattr(obj, "columns", None)
    if cols is None or not hasattr(obj, "__getitem__"):
        return {}
    try:
        present = {str(col) for col in cols}
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for col in list(key_columns)[:limit]:
        if str(col) not in present:
            continue
        try:
            out[str(col)] = int(obj[col].isna().sum())  # type: ignore[index]
        except Exception:
            continue
    return out


def normalize_event(payload: Mapping[str, Any]) -> Dict[str, Any]:
    event: Dict[str, Any] = {field: None for field in EVENT_FIELDS}
    event.update({str(key): value for key, value in payload.items() if str(key) in EVENT_FIELDS})
    event["timestamp_utc"] = str(event.get("timestamp_utc") or utc_now_iso())
    event["run_id"] = str(event.get("run_id") or "")
    event["family"] = str(event.get("family") or "")
    event["model"] = str(event.get("model") or "")
    event["stage"] = str(event.get("stage") or "")
    event["function_name"] = str(event.get("function_name") or "")
    event["module_name"] = str(event.get("module_name") or "")
    event["phase_name"] = str(event.get("phase_name") or "")
    event["parent_phase"] = str(event.get("parent_phase") or "")
    event["status"] = str(event.get("status") or "completed")
    event["reason_code"] = _clean_reason_code(event.get("reason_code"))
    for key in ("elapsed_seconds", "best_value"):
        value = _safe_float(event.get(key))
        event[key] = round(value, 6) if value is not None else None
    for key in (
        "interval_minutes",
        "horizon_minutes",
        "asset_count",
        "input_rows",
        "output_rows",
        "input_columns_count",
        "output_columns_count",
        "trial_count",
        "pruned_count",
        "failed_count",
        "completed_count",
    ):
        event[key] = _safe_int(event.get(key))
    missing = event.get("required_columns_missing")
    if isinstance(missing, (list, tuple)):
        event["required_columns_missing"] = ",".join(str(item) for item in missing[:40])
    elif missing is None:
        event["required_columns_missing"] = ""
    else:
        event["required_columns_missing"] = str(missing)
    null_counts = event.get("key_null_counts")
    if isinstance(null_counts, Mapping):
        event["key_null_counts"] = json.dumps(dict(null_counts), sort_keys=True)
    elif null_counts is None:
        event["key_null_counts"] = ""
    else:
        event["key_null_counts"] = str(null_counts)
    samples = event.get("failure_samples")
    if isinstance(samples, (list, tuple)):
        event["failure_samples"] = json.dumps(list(samples)[:10], sort_keys=True, default=str)
    elif samples is None:
        event["failure_samples"] = ""
    else:
        event["failure_samples"] = _compact_message(samples, limit=500)
    for key in ("exception_message", "source_path", "output_path", "artifact_profile_source"):
        if event.get(key) is not None:
            event[key] = _compact_message(event.get(key), limit=500 if key.endswith("path") else 240)
        else:
            event[key] = ""
    return event


def emit_event(run_root: Path, **payload: Any) -> Dict[str, Any]:
    event = normalize_event(payload)
    _safe_append_jsonl(event_path(Path(run_root)), event, warning_root=Path(run_root))
    return event


@dataclass
class TelemetryScope:
    run_root: Path
    payload: Dict[str, Any]
    started: float = field(default_factory=time.perf_counter)

    def set_output(self, obj: Any = None, **fields: Any) -> None:
        self.payload.update(object_shape_summary(obj, prefix="output"))
        self.payload.update(fields)

    def update(self, **fields: Any) -> None:
        self.payload.update(fields)


@dataclass
class NoopTelemetryScope:
    payload: Dict[str, Any] = field(default_factory=dict)

    def set_output(self, obj: Any = None, **fields: Any) -> None:
        self.payload.update(object_shape_summary(obj, prefix="output"))
        self.payload.update(fields)

    def update(self, **fields: Any) -> None:
        self.payload.update(fields)


@contextmanager
def telemetry_scope(
    run_root: Path,
    *,
    input_obj: Any = None,
    required_columns: Optional[Sequence[str]] = None,
    key_columns: Optional[Sequence[str]] = None,
    **payload: Any,
) -> Iterator[TelemetryScope]:
    base = dict(payload)
    base.update(object_shape_summary(input_obj, prefix="input"))
    missing = required_columns_missing(input_obj, required_columns)
    if missing:
        base["required_columns_missing"] = missing
    nulls = compact_null_summary(input_obj, key_columns)
    if nulls:
        base["key_null_counts"] = nulls
    scope = TelemetryScope(Path(run_root), base)
    try:
        yield scope
    except Exception as exc:
        scope.payload.update(
            {
                "status": "failed",
                "reason_code": scope.payload.get("reason_code") or "exception",
                "exception_type": type(exc).__name__,
                "exception_message": _compact_message(exc),
            }
        )
        raise
    finally:
        elapsed = time.perf_counter() - scope.started
        scope.payload.setdefault("status", "completed")
        scope.payload["elapsed_seconds"] = elapsed
        emit_event(scope.run_root, **scope.payload)


def infer_run_root_from_path(path: Path) -> Optional[Path]:
    path = Path(path)
    parts = path.parts
    for idx in range(len(parts) - 1, -1, -1):
        if str(parts[idx]).startswith("run="):
            return Path(*parts[: idx + 1])
    return None


def infer_run_root_from_log_path(log_path: Path) -> Optional[Path]:
    return infer_run_root_from_path(log_path)


def emit_event_for_path(path: Optional[Path], **payload: Any) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    run_root = infer_run_root_from_path(Path(path))
    if run_root is None:
        return None
    payload.setdefault("run_id", run_root.name)
    return emit_event(run_root, **payload)


def _trial_state_name(value: Any) -> str:
    raw = getattr(value, "state", value)
    name = getattr(raw, "name", raw)
    return str(name or "").upper()


def compact_trial_state_summary(
    *,
    trial_rows: Sequence[Mapping[str, Any]] = (),
    study_trials: Sequence[Any] = (),
    failure_limit: int = 5,
) -> Dict[str, Any]:
    states: List[str] = []
    failure_samples: List[Dict[str, Any]] = []
    for trial in study_trials:
        state = _trial_state_name(trial)
        if state:
            states.append(state)
        if state in {"FAIL", "FAILED"} and len(failure_samples) < int(failure_limit):
            failure_samples.append(
                {
                    "trial_number": _safe_int(getattr(trial, "number", None)),
                    "state": state,
                }
            )
    if not states:
        for row in trial_rows:
            state = _trial_state_name(row.get("state") if isinstance(row, Mapping) else "")
            if state:
                states.append(state)
            if state in {"FAIL", "FAILED"} and len(failure_samples) < int(failure_limit):
                failure_samples.append(
                    {
                        "trial_number": _safe_int(row.get("trial_number") if isinstance(row, Mapping) else None),
                        "state": state,
                        "reason": _compact_message(row.get("reason") if isinstance(row, Mapping) else ""),
                    }
                )
    counter = Counter(states)
    trial_count = len(study_trials) if study_trials else len(trial_rows)
    return {
        "trial_count": int(trial_count),
        "completed_count": int(counter.get("COMPLETE", 0) + counter.get("COMPLETED", 0)),
        "pruned_count": int(counter.get("PRUNED", 0)),
        "failed_count": int(counter.get("FAIL", 0) + counter.get("FAILED", 0)),
        "failure_samples": failure_samples,
    }


def emit_stage3_study_summary_for_path(
    path: Optional[Path],
    *,
    family: str,
    model: str,
    function_name: str,
    module_name: str,
    combo_key: str = "",
    interval_minutes: Optional[int] = None,
    horizon_minutes: Optional[int] = None,
    task: str = "",
    elapsed_seconds: Optional[float] = None,
    trial_rows: Sequence[Mapping[str, Any]] = (),
    study_trials: Sequence[Any] = (),
    best_value: Any = None,
    input_rows: Optional[int] = None,
    output_rows: Optional[int] = None,
    source_path: str = "",
    output_path: str = "",
    status: str = "completed",
    reason_code: str = "",
) -> Optional[Dict[str, Any]]:
    summary = compact_trial_state_summary(trial_rows=trial_rows, study_trials=study_trials)
    if not reason_code and int(summary.get("failed_count") or 0) > 0:
        reason_code = "trial_failed"
    if not reason_code and int(summary.get("completed_count") or 0) <= 0 and int(summary.get("trial_count") or 0) > 0:
        reason_code = "no_completed_trials"
    return emit_event_for_path(
        path,
        family=str(family),
        model=str(model),
        stage="stage3",
        function_name=str(function_name),
        module_name=str(module_name),
        phase_name="trial_summary",
        parent_phase="tuning",
        status=str(status),
        reason_code=str(reason_code or ""),
        combo_key=str(combo_key or ""),
        interval_minutes=interval_minutes,
        horizon_minutes=horizon_minutes,
        task=str(task or ""),
        elapsed_seconds=elapsed_seconds,
        input_rows=input_rows,
        output_rows=output_rows,
        source_path=str(source_path or ""),
        output_path=str(output_path or ""),
        best_value=best_value,
        **summary,
    )


@contextmanager
def telemetry_scope_for_path(
    path: Optional[Path],
    *,
    input_obj: Any = None,
    required_columns: Optional[Sequence[str]] = None,
    key_columns: Optional[Sequence[str]] = None,
    **payload: Any,
) -> Iterator[TelemetryScope | NoopTelemetryScope]:
    if path is None:
        yield NoopTelemetryScope(dict(payload))
        return
    run_root = infer_run_root_from_path(Path(path))
    if run_root is None:
        yield NoopTelemetryScope(dict(payload))
        return
    payload.setdefault("run_id", run_root.name)
    with telemetry_scope(
        run_root,
        input_obj=input_obj,
        required_columns=required_columns,
        key_columns=key_columns,
        **payload,
    ) as scope:
        yield scope


def infer_stage_model_from_log_path(log_path: Path) -> Tuple[str, str]:
    path = Path(log_path)
    stage = path.stem
    run_root = infer_run_root_from_log_path(path)
    model = "family"
    if run_root is not None:
        try:
            rel = path.relative_to(run_root)
            if len(rel.parts) >= 3 and rel.parts[1] == "logs":
                model = rel.parts[0]
        except Exception:
            pass
    return stage, model


def module_from_command(command: Sequence[str]) -> str:
    tokens = [str(part) for part in command]
    for idx, token in enumerate(tokens):
        if token == "-m" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return tokens[0] if tokens else ""


def emit_subprocess_event(
    *,
    log_path: Path,
    command: Sequence[str],
    status: str,
    family: str,
    elapsed_seconds: Optional[float] = None,
    reason_code: str = "",
    exception: Optional[BaseException] = None,
    output_path: Optional[Path] = None,
) -> None:
    run_root = infer_run_root_from_log_path(log_path)
    if run_root is None:
        return
    stage, model = infer_stage_model_from_log_path(log_path)
    payload: Dict[str, Any] = {
        "run_id": run_root.name,
        "family": family,
        "model": model,
        "stage": stage,
        "function_name": "run_logged_subprocess",
        "module_name": module_from_command(command),
        "phase_name": stage,
        "parent_phase": "test_orchestrator",
        "status": status,
        "reason_code": reason_code,
        "elapsed_seconds": elapsed_seconds,
        "output_path": str(output_path or log_path),
    }
    if exception is not None:
        payload["exception_type"] = type(exception).__name__
        payload["exception_message"] = _compact_message(exception)
        payload["reason_code"] = reason_code or "exception"
    emit_event(run_root, **payload)


def read_events(path: Path, *, limit: int = 200_000) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return events
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no > int(limit):
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(normalize_event(payload))
    return events


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    assert_write_allowed(path, "test branch function telemetry rollup")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "test branch function telemetry rollup temp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    atomic_replace(tmp, path)


def _dominant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def build_function_health_rows(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    reason_counts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for event in events:
        key = tuple(str(event.get(field) or "") for field in ("family", "model", "stage", "combo_key", "function_name", "phase_name"))
        row = grouped.setdefault(
            key,
            {
                "family": key[0],
                "model": key[1],
                "stage": key[2],
                "combo_key": key[3],
                "function_name": key[4],
                "phase_name": key[5],
                "call_count": 0,
                "success_count": 0,
                "skip_count": 0,
                "empty_output_count": 0,
                "failure_count": 0,
                "total_seconds": 0.0,
                "max_seconds": 0.0,
                "total_input_rows": 0,
                "total_output_rows": 0,
                "first_error_type": "",
            },
        )
        row["call_count"] += 1
        status = str(event.get("status") or "")
        reason = str(event.get("reason_code") or "")
        if reason:
            reason_counts[key][reason] += 1
        if status == "completed":
            row["success_count"] += 1
        elif status == "skipped":
            row["skip_count"] += 1
        elif status == "failed":
            row["failure_count"] += 1
            if not row["first_error_type"]:
                row["first_error_type"] = str(event.get("exception_type") or "")
        output_rows = _safe_int(event.get("output_rows"))
        if output_rows == 0 or reason in {"predict_returned_empty", "postprocess_dropped_all_rows", "validation_dropped_all_rows", "write_skipped_empty_output", "join_empty", "labels_empty"}:
            row["empty_output_count"] += 1
        elapsed = _safe_float(event.get("elapsed_seconds")) or 0.0
        row["total_seconds"] += elapsed
        row["max_seconds"] = max(float(row["max_seconds"]), elapsed)
        row["total_input_rows"] += _safe_int(event.get("input_rows")) or 0
        row["total_output_rows"] += _safe_int(event.get("output_rows")) or 0
    rows: List[Dict[str, Any]] = []
    for key, row in grouped.items():
        calls = max(1, int(row["call_count"]))
        row["total_seconds"] = round(float(row["total_seconds"]), 6)
        row["mean_seconds"] = round(float(row["total_seconds"]) / calls, 6)
        row["max_seconds"] = round(float(row["max_seconds"]), 6)
        row["dominant_reason_code"] = _dominant(reason_counts[key])
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["family"]), str(row["model"]), str(row["stage"]), -float(row["total_seconds"])))


def build_phase_timing_rows(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for event in events:
        key = tuple(str(event.get(field) or "") for field in ("family", "model", "stage", "phase_name"))
        row = grouped.setdefault(
            key,
            {
                "family": key[0],
                "model": key[1],
                "stage": key[2],
                "phase_name": key[3],
                "call_count": 0,
                "failure_count": 0,
                "total_seconds": 0.0,
                "mean_seconds": 0.0,
                "max_seconds": 0.0,
                "empty_output_count": 0,
            },
        )
        row["call_count"] += 1
        if str(event.get("status") or "") == "failed":
            row["failure_count"] += 1
        if _safe_int(event.get("output_rows")) == 0:
            row["empty_output_count"] += 1
        elapsed = _safe_float(event.get("elapsed_seconds")) or 0.0
        row["total_seconds"] += elapsed
        row["max_seconds"] = max(float(row["max_seconds"]), elapsed)
    rows: List[Dict[str, Any]] = []
    for row in grouped.values():
        calls = max(1, int(row["call_count"]))
        row["total_seconds"] = round(float(row["total_seconds"]), 6)
        row["mean_seconds"] = round(float(row["total_seconds"]) / calls, 6)
        row["max_seconds"] = round(float(row["max_seconds"]), 6)
        rows.append(row)
    return sorted(rows, key=lambda row: -float(row["total_seconds"]))


def build_empty_output_rows(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    empty_reasons = {
        "source_read_empty",
        "feature_load_empty",
        "regime_load_empty",
        "seasonality_load_empty",
        "labels_empty",
        "join_empty",
        "train_frame_empty",
        "validation_frame_empty",
        "predict_returned_empty",
        "postprocess_dropped_all_rows",
        "validation_dropped_all_rows",
        "write_skipped_empty_output",
    }
    for event in events:
        reason = str(event.get("reason_code") or "")
        output_rows = _safe_int(event.get("output_rows"))
        if output_rows != 0 and reason not in empty_reasons:
            continue
        rows.append(
            {
                "family": event.get("family"),
                "model": event.get("model"),
                "stage": event.get("stage"),
                "combo_key": event.get("combo_key"),
                "function_name": event.get("function_name"),
                "phase_name": event.get("phase_name"),
                "reason_code": reason or ("empty_output" if output_rows == 0 else ""),
                "upstream_rows": event.get("input_rows"),
                "downstream_rows": event.get("output_rows"),
                "missing_columns": event.get("required_columns_missing"),
                "where_detected": event.get("phase_name"),
                "source_path": event.get("source_path"),
                "output_path": event.get("output_path"),
            }
        )
    return rows[:1000]


def _write_alignment_md(run_root: Path, events: Sequence[Mapping[str, Any]], health_rows: Sequence[Mapping[str, Any]], empty_rows: Sequence[Mapping[str, Any]]) -> None:
    slowest = sorted(health_rows, key=lambda row: float(row.get("total_seconds", 0.0) or 0.0), reverse=True)[:10]
    failures = [event for event in events if str(event.get("status") or "") == "failed"][:10]
    handoffs = [
        event
        for event in events
        if str(event.get("phase_name") or "") == "artifact_handoff"
        or str(event.get("artifact_profile_source") or "")
        or str(event.get("output_path") or "")
    ][:20]
    lines = [
        "# Production Alignment Summary",
        "",
        f"Generated: {utc_now_iso()}",
        f"Run root: `{Path(run_root)}`",
        "",
        "## Selected Profile And Artifact Sources",
    ]
    if handoffs:
        for event in handoffs:
            source = event.get("artifact_profile_source") or event.get("output_path") or event.get("source_path")
            lines.append(f"- {event.get('family')}/{event.get('model')}/{event.get('stage')}: {event.get('function_name')} -> `{source}`")
    else:
        lines.append("- No profile or artifact handoff events were present in `function_events.jsonl`.")
    lines.extend(["", "## Promotion Blocking Warnings"])
    if failures:
        for event in failures:
            lines.append(f"- {event.get('family')}/{event.get('model')}/{event.get('stage')}: {event.get('function_name')} failed with `{event.get('exception_type')}`")
    if empty_rows:
        for row in empty_rows[:10]:
            lines.append(f"- Empty output: {row.get('family')}/{row.get('model')}/{row.get('stage')} `{row.get('function_name')}` reason=`{row.get('reason_code')}`")
    if not failures and not empty_rows:
        lines.append("- None detected by function telemetry rollups.")
    lines.extend(["", "## Informational Warnings", "- Function-level telemetry is bounded and summarizes phases; full stack traces remain in existing stage logs."])
    lines.extend(["", "## Top Function-Level Concerns"])
    if slowest:
        for row in slowest:
            lines.append(f"- {row.get('family')}/{row.get('model')}/{row.get('stage')} `{row.get('function_name')}` phase=`{row.get('phase_name')}` total_seconds={row.get('total_seconds')}")
    else:
        lines.append("- No timing events were present.")
    path = Path(run_root) / PRODUCTION_ALIGNMENT_SUMMARY_MD
    assert_write_allowed(path, "test branch production alignment summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "test branch production alignment summary temp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def write_rollups(run_root: Path) -> Dict[str, Any]:
    root = Path(run_root)
    events = read_events(event_path(root))
    health_rows = build_function_health_rows(events)
    phase_rows = build_phase_timing_rows(events)
    empty_rows = build_empty_output_rows(events)
    _write_csv_atomic(
        root / FUNCTION_HEALTH_SUMMARY_CSV,
        (
            "family",
            "model",
            "stage",
            "combo_key",
            "function_name",
            "phase_name",
            "call_count",
            "success_count",
            "skip_count",
            "empty_output_count",
            "failure_count",
            "total_seconds",
            "mean_seconds",
            "max_seconds",
            "total_input_rows",
            "total_output_rows",
            "dominant_reason_code",
            "first_error_type",
        ),
        health_rows,
    )
    _write_csv_atomic(
        root / PHASE_TIMING_SUMMARY_CSV,
        ("family", "model", "stage", "phase_name", "call_count", "failure_count", "empty_output_count", "total_seconds", "mean_seconds", "max_seconds"),
        phase_rows,
    )
    _write_csv_atomic(
        root / EMPTY_OUTPUT_SUMMARY_CSV,
        (
            "family",
            "model",
            "stage",
            "combo_key",
            "function_name",
            "phase_name",
            "reason_code",
            "upstream_rows",
            "downstream_rows",
            "missing_columns",
            "where_detected",
            "source_path",
            "output_path",
        ),
        empty_rows,
    )
    _write_alignment_md(root, events, health_rows, empty_rows)
    return {
        "event_count": int(len(events)),
        "function_health_summary_csv": str((root / FUNCTION_HEALTH_SUMMARY_CSV).resolve()),
        "phase_timing_summary_csv": str((root / PHASE_TIMING_SUMMARY_CSV).resolve()),
        "empty_output_summary_csv": str((root / EMPTY_OUTPUT_SUMMARY_CSV).resolve()),
        "production_alignment_summary_md": str((root / PRODUCTION_ALIGNMENT_SUMMARY_MD).resolve()),
    }
