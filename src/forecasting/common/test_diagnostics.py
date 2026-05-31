from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path


TEST_DIAGNOSTIC_SCHEMA_VERSION = 1

RUN_SUMMARY_JSON = "run_summary.json"
DIAGNOSTIC_MANIFEST_JSON = "diagnostic_manifest.json"
DIAGNOSTIC_EVENTS_JSONL = "diagnostic_events.jsonl"
DIAGNOSTIC_SAMPLES_CSV = "diagnostic_samples.csv"
TOP_OFFENDERS_JSON = "top_offenders.json"
OUTPUT_PARITY_JSON = "output_parity.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _csv_safe(value: Any) -> str:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (str, int, float, bool)):
        return str(safe)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path, suffix=".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path, suffix=".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    atomic_replace(tmp, path)


def _module_snapshot(name: str, importer: Callable[[str], Any]) -> Dict[str, Any]:
    try:
        module = importer(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {"available": True, "version": getattr(module, "__version__", None)}


def collect_optional_dependency_snapshot(
    *,
    importer: Callable[[str], Any] = import_module,
) -> Dict[str, Any]:
    """Collect optional runtime diagnostics without making dependencies required."""
    snapshot: Dict[str, Any] = {"modules": {}}
    for module_name in ("psutil", "threadpoolctl", "numba", "pyarrow"):
        snapshot["modules"][module_name] = _module_snapshot(module_name, importer)

    if snapshot["modules"]["threadpoolctl"].get("available"):
        try:
            threadpoolctl = importer("threadpoolctl")
            snapshot["threadpool_info"] = _json_safe(threadpoolctl.threadpool_info())
        except Exception as exc:
            snapshot["threadpool_info_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    if snapshot["modules"]["numba"].get("available"):
        try:
            numba = importer("numba")
            snapshot["numba_num_threads"] = int(numba.get_num_threads())
        except Exception as exc:
            snapshot["numba_threads_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    if snapshot["modules"]["pyarrow"].get("available"):
        try:
            pyarrow = importer("pyarrow")
            snapshot["pyarrow_cpu_count"] = int(pyarrow.cpu_count())
            snapshot["pyarrow_io_thread_count"] = int(pyarrow.io_thread_count())
        except Exception as exc:
            snapshot["pyarrow_threads_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    if snapshot["modules"]["psutil"].get("available"):
        try:
            psutil = importer("psutil")
            proc = psutil.Process()
            mem = proc.memory_info()
            snapshot["process"] = {
                "pid": int(proc.pid),
                "num_threads": int(proc.num_threads()),
                "rss_mb": round(float(mem.rss) / (1024.0 * 1024.0), 3),
            }
            if hasattr(proc, "memory_full_info"):
                try:
                    full = proc.memory_full_info()
                    uss = getattr(full, "uss", None)
                    if uss is not None:
                        snapshot["process"]["uss_mb"] = round(float(uss) / (1024.0 * 1024.0), 3)
                except Exception:
                    pass
        except Exception as exc:
            snapshot["process_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    return snapshot


@dataclass(frozen=True)
class DiagnosticPacketPaths:
    root: Path
    run_summary: Path
    diagnostic_manifest: Path
    diagnostic_events: Path
    diagnostic_samples: Path
    top_offenders: Path
    output_parity: Path

    @classmethod
    def from_root(cls, root: Path) -> "DiagnosticPacketPaths":
        return cls(
            root=Path(root),
            run_summary=Path(root) / RUN_SUMMARY_JSON,
            diagnostic_manifest=Path(root) / DIAGNOSTIC_MANIFEST_JSON,
            diagnostic_events=Path(root) / DIAGNOSTIC_EVENTS_JSONL,
            diagnostic_samples=Path(root) / DIAGNOSTIC_SAMPLES_CSV,
            top_offenders=Path(root) / TOP_OFFENDERS_JSON,
            output_parity=Path(root) / OUTPUT_PARITY_JSON,
        )


@dataclass
class DiagnosticPacketConfig:
    root: Path
    module_name: str
    run_id: str
    mode: str = "test"
    max_events: int = 1000
    max_samples: int = 1000
    max_top_offenders: int = 25
    include_optional_dependency_snapshot: bool = True
    created_at_utc: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"test", "staged", "production"}:
            raise ValueError("mode must be one of: test, staged, production")
        self.mode = mode
        self.root = Path(self.root)
        self.max_events = max(0, int(self.max_events))
        self.max_samples = max(0, int(self.max_samples))
        self.max_top_offenders = max(0, int(self.max_top_offenders))

    @property
    def is_test_packet(self) -> bool:
        return self.mode in {"test", "staged"}


class TestDiagnosticPacket:
    """Bounded diagnostic packet writer for Test/staged runs and compact Production summaries."""

    __test__ = False

    def __init__(
        self,
        config: DiagnosticPacketConfig,
        *,
        dependency_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config
        self.paths = DiagnosticPacketPaths.from_root(config.root)
        self._events: List[Dict[str, Any]] = []
        self._samples: List[Dict[str, Any]] = []
        self._top_offenders: List[Dict[str, Any]] = []
        self._event_overflow_count = 0
        self._sample_overflow_count = 0
        self._summary_updates: Dict[str, Any] = {}
        self._output_parity: Dict[str, Any] = {
            "status": "not_provided",
            "row_counts": {},
            "checksums": {},
            "notes": [],
        }
        self._dependency_snapshot = (
            dict(dependency_snapshot)
            if dependency_snapshot is not None
            else collect_optional_dependency_snapshot()
            if config.include_optional_dependency_snapshot
            else {}
        )

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        module_name: str,
        run_id: str,
        mode: str = "test",
        max_events: int = 1000,
        max_samples: int = 1000,
        max_top_offenders: int = 25,
        dependency_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> "TestDiagnosticPacket":
        return cls(
            DiagnosticPacketConfig(
                root=Path(root),
                module_name=str(module_name),
                run_id=str(run_id),
                mode=str(mode),
                max_events=max_events,
                max_samples=max_samples,
                max_top_offenders=max_top_offenders,
            ),
            dependency_snapshot=dependency_snapshot,
        )

    @property
    def event_overflow_count(self) -> int:
        return self._event_overflow_count

    @property
    def sample_overflow_count(self) -> int:
        return self._sample_overflow_count

    def record_event(self, event_type: str, payload: Optional[Mapping[str, Any]] = None, **fields: Any) -> bool:
        if not self.config.is_test_packet:
            return False
        if len(self._events) >= self.config.max_events:
            self._event_overflow_count += 1
            return False
        event = {
            "timestamp_utc": utc_now_iso(),
            "run_id": self.config.run_id,
            "module_name": self.config.module_name,
            "event_type": str(event_type),
        }
        if payload:
            event.update(dict(payload))
        if fields:
            event.update(fields)
        self._events.append(_json_safe(event))
        return True

    def record_sample(self, sample: Optional[Mapping[str, Any]] = None, **fields: Any) -> bool:
        if not self.config.is_test_packet:
            return False
        if len(self._samples) >= self.config.max_samples:
            self._sample_overflow_count += 1
            return False
        row = {
            "timestamp_utc": utc_now_iso(),
            "run_id": self.config.run_id,
            "module_name": self.config.module_name,
        }
        if sample:
            row.update(dict(sample))
        if fields:
            row.update(fields)
        self._samples.append(_json_safe(row))
        return True

    def record_top_offender(
        self,
        name: str,
        score: float,
        *,
        category: str = "unit",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.config.is_test_packet:
            return
        item = {
            "name": str(name),
            "category": str(category),
            "score": float(score),
            "metadata": dict(metadata or {}),
        }
        self._top_offenders.append(_json_safe(item))

    def update_run_summary(self, **sections: Any) -> None:
        self._summary_updates.update(sections)

    def set_output_parity(
        self,
        *,
        status: str,
        row_counts: Optional[Mapping[str, Any]] = None,
        checksums: Optional[Mapping[str, Any]] = None,
        notes: Optional[Sequence[Any]] = None,
        **fields: Any,
    ) -> None:
        parity = {
            "status": str(status),
            "row_counts": dict(row_counts or {}),
            "checksums": dict(checksums or {}),
            "notes": list(notes or []),
        }
        parity.update(fields)
        self._output_parity = _json_safe(parity)

    def finalize(
        self,
        *,
        status: str = "completed",
        run_summary: Optional[Mapping[str, Any]] = None,
    ) -> DiagnosticPacketPaths:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        finalized_at = utc_now_iso()
        top_offenders = sorted(
            self._top_offenders,
            key=lambda row: float(row.get("score", 0.0) or 0.0),
            reverse=True,
        )[: self.config.max_top_offenders]

        summary: Dict[str, Any] = {
            "schema_version": TEST_DIAGNOSTIC_SCHEMA_VERSION,
            "packet_kind": "test_diagnostic_packet",
            "mode": self.config.mode,
            "status": str(status),
            "run_id": self.config.run_id,
            "module_name": self.config.module_name,
            "created_at_utc": self.config.created_at_utc,
            "finalized_at_utc": finalized_at,
            "bounds": {
                "max_events": self.config.max_events,
                "max_samples": self.config.max_samples,
                "max_top_offenders": self.config.max_top_offenders,
            },
            "counts": {
                "events_recorded": len(self._events),
                "events_overflow": self._event_overflow_count,
                "samples_recorded": len(self._samples),
                "samples_overflow": self._sample_overflow_count,
                "top_offenders_recorded": len(self._top_offenders),
            },
            "optional_dependency_snapshot": self._dependency_snapshot,
        }
        if run_summary:
            summary.update(_json_safe(dict(run_summary)))
        if self._summary_updates:
            summary.update(_json_safe(self._summary_updates))

        parity = dict(self._output_parity)
        top_payload = {
            "schema_version": TEST_DIAGNOSTIC_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "module_name": self.config.module_name,
            "top_offenders": top_offenders,
            "total_offenders_seen": len(self._top_offenders),
            "truncated_count": max(0, len(self._top_offenders) - len(top_offenders)),
        }

        self._write_events()
        self._write_samples()
        _write_json_atomic(self.paths.top_offenders, top_payload)
        _write_json_atomic(self.paths.output_parity, parity)
        _write_json_atomic(self.paths.run_summary, summary)
        _write_json_atomic(self.paths.diagnostic_manifest, self._manifest(finalized_at=finalized_at, status=str(status)))
        return self.paths

    def _write_events(self) -> None:
        if not self.config.is_test_packet:
            _write_text_atomic(self.paths.diagnostic_events, "")
            return
        text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self._events)
        _write_text_atomic(self.paths.diagnostic_events, text)

    def _write_samples(self) -> None:
        if not self.config.is_test_packet:
            _write_text_atomic(self.paths.diagnostic_samples, "")
            return
        fieldnames = self._sample_fieldnames(self._samples)
        if not fieldnames:
            _write_text_atomic(self.paths.diagnostic_samples, "")
            return
        tmp = sibling_temp_path(self.paths.diagnostic_samples, suffix=".tmp")
        self.paths.diagnostic_samples.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._samples:
                writer.writerow({field: _csv_safe(row.get(field)) for field in fieldnames})
        atomic_replace(tmp, self.paths.diagnostic_samples)

    @staticmethod
    def _sample_fieldnames(rows: Iterable[Mapping[str, Any]]) -> List[str]:
        preferred = ["timestamp_utc", "run_id", "module_name", "sample_type", "name", "value"]
        found: List[str] = []
        seen = set()
        for field in preferred:
            seen.add(field)
        for row in rows:
            for key in row:
                skey = str(key)
                if skey not in seen:
                    seen.add(skey)
                    found.append(skey)
        fieldnames = [field for field in preferred if any(field in row for row in rows)]
        fieldnames.extend(found)
        return fieldnames

    def _manifest(self, *, finalized_at: str, status: str) -> Dict[str, Any]:
        artifacts = {
            "run_summary": self._artifact_entry(self.paths.run_summary, emitted=True),
            "diagnostic_manifest": self._artifact_entry(self.paths.diagnostic_manifest, emitted=True),
            "diagnostic_events": self._artifact_entry(
                self.paths.diagnostic_events,
                emitted=self.config.is_test_packet,
                record_count=len(self._events),
                overflow_count=self._event_overflow_count,
                max_records=self.config.max_events,
            ),
            "diagnostic_samples": self._artifact_entry(
                self.paths.diagnostic_samples,
                emitted=self.config.is_test_packet,
                record_count=len(self._samples),
                overflow_count=self._sample_overflow_count,
                max_records=self.config.max_samples,
            ),
            "top_offenders": self._artifact_entry(
                self.paths.top_offenders,
                emitted=self.config.is_test_packet,
                record_count=min(len(self._top_offenders), self.config.max_top_offenders),
                max_records=self.config.max_top_offenders,
            ),
            "output_parity": self._artifact_entry(self.paths.output_parity, emitted=True),
        }
        return {
            "schema_version": TEST_DIAGNOSTIC_SCHEMA_VERSION,
            "packet_kind": "test_diagnostic_packet",
            "mode": self.config.mode,
            "run_id": self.config.run_id,
            "module_name": self.config.module_name,
            "status": status,
            "created_at_utc": self.config.created_at_utc,
            "finalized_at_utc": finalized_at,
            "artifacts": artifacts,
            "bounds": {
                "max_events": self.config.max_events,
                "max_samples": self.config.max_samples,
                "max_top_offenders": self.config.max_top_offenders,
            },
            "overflow": {
                "events": self._event_overflow_count,
                "samples": self._sample_overflow_count,
            },
            "retention": {
                "cleanup_policy": "none",
                "safe_cleanup_candidate": True,
                "notes": "Dedicated diagnostic packet files only; no deletion behavior is implemented.",
            },
        }

    @staticmethod
    def _artifact_entry(
        path: Path,
        *,
        emitted: bool,
        record_count: Optional[int] = None,
        overflow_count: Optional[int] = None,
        max_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "path": str(path),
            "filename": path.name,
            "emitted": bool(emitted),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        if record_count is not None:
            entry["record_count"] = int(record_count)
        if overflow_count is not None:
            entry["overflow_count"] = int(overflow_count)
        if max_records is not None:
            entry["max_records"] = int(max_records)
        return entry


__all__ = [
    "DIAGNOSTIC_EVENTS_JSONL",
    "DIAGNOSTIC_MANIFEST_JSON",
    "DIAGNOSTIC_SAMPLES_CSV",
    "OUTPUT_PARITY_JSON",
    "RUN_SUMMARY_JSON",
    "TEST_DIAGNOSTIC_SCHEMA_VERSION",
    "TOP_OFFENDERS_JSON",
    "DiagnosticPacketConfig",
    "DiagnosticPacketPaths",
    "TestDiagnosticPacket",
    "collect_optional_dependency_snapshot",
]
