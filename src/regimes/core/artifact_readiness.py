from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_refs import contains_machine_local_marker, is_unsafe_serialized_path


ROOT_CLEANUP_CANDIDATES: tuple[str, ...] = (
    ".pytest_tmp",
    ".tmp_copula_state",
    "diagnostics",
    "logs",
)

CROSS_ASSET_FEATURE_ROW_DIR_NAME = "cross_asset_feature_rows"

MACHINE_LOCAL_PATH_TOKENS: tuple[str, ...] = (
    "D:" + "\\",
    "D:" + "/",
    "C:" + "\\",
    "C:" + "/",
    "/" + "Users/",
    "/" + "home/",
    "/" + "mnt/data",
    "project" + "_cohorts",
)

DEFAULT_MACHINE_LOCAL_MARKER_ALLOWED_FIXTURES: tuple[str, ...] = (
    "tests/regimes/test_regime_artifact_refs.py",
)


def classify_artifact_portability(
    artifact_paths: Sequence[str | Path],
    *,
    report_root: str | Path,
    handoff_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root = Path(report_root).resolve()
    resolved_handoff_roots = tuple(Path(path).resolve() for path in handoff_roots)
    findings: list[dict[str, Any]] = []
    scanned = 0

    for item in sorted({str(path) for path in artifact_paths}):
        path = Path(item)
        if not path.is_file():
            continue
        scanned += 1
        unsafe_fields = find_unsafe_serialized_strings(path)
        if not unsafe_fields:
            continue
        path_class = "handoff_critical" if _is_within_any(path.resolve(), resolved_handoff_roots) else "historical"
        findings.append(
            {
                "path": _relative_or_posix(path.resolve(), root),
                "path_class": path_class,
                "unsafe_string_count": len(unsafe_fields),
                "unsafe_fields": unsafe_fields[:20],
            }
        )

    handoff_findings = [finding for finding in findings if finding["path_class"] == "handoff_critical"]
    historical_findings = [finding for finding in findings if finding["path_class"] == "historical"]
    return {
        "artifact_kind": "regime_artifact_portability_classification",
        "schema_version": 1,
        "scanned_file_count": scanned,
        "unsafe_file_count": len(findings),
        "handoff_critical_unsafe_file_count": len(handoff_findings),
        "historical_unsafe_file_count": len(historical_findings),
        "handoff_critical_ready": not handoff_findings,
        "historical_debt_present": bool(historical_findings),
        "findings": findings,
    }


def find_unsafe_serialized_strings(path: str | Path) -> list[str]:
    artifact_path = Path(path)
    if artifact_path.suffix.lower() == ".json":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        return _unsafe_payload_fields(payload)
    if artifact_path.suffix.lower() == ".jsonl":
        findings: list[str] = []
        for idx, line in enumerate(artifact_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            findings.extend(_unsafe_payload_fields(payload, path=f"$.line[{idx}]"))
        return findings
    text = artifact_path.read_text(encoding="utf-8", errors="ignore")
    return ["$"] if contains_machine_local_marker(text) else []


def discover_cross_asset_feature_row_paths(
    artifact_root: str | Path,
    *,
    row_dir_name: str = CROSS_ASSET_FEATURE_ROW_DIR_NAME,
) -> tuple[Path, ...]:
    root = Path(artifact_root)
    if not root.exists():
        return ()
    paths = {
        path
        for path in root.rglob("*.jsonl")
        if path.is_file() and row_dir_name in path.parts
    }
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def detect_cross_asset_row_metadata_gaps(
    row_paths: Sequence[str | Path],
    *,
    artifact_root: str | Path | None = None,
    required_fields: Sequence[str] = ("lineage_id", "known_at_ts", "source_tail_ts"),
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    root = Path(artifact_root).resolve() if artifact_root is not None else None
    scanned_files = 0
    scanned_rows = 0

    for item in sorted({str(path) for path in row_paths}):
        path = Path(item)
        if not path.is_file():
            continue
        scanned_files += 1
        missing_counts = {field: 0 for field in required_fields}
        ordering_violations = 0
        row_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            for field in required_fields:
                if not str(row.get(field, "")).strip():
                    missing_counts[field] += 1
            if "known_at_ts" in row and "source_tail_ts" in row and _orderable(row["source_tail_ts"]) > _orderable(row["known_at_ts"]):
                ordering_violations += 1
        scanned_rows += row_count
        if any(missing_counts.values()) or ordering_violations:
            findings.append(
                {
                    "path": _relative_or_posix(path.resolve(), root) if root is not None else path.as_posix(),
                    "row_count": row_count,
                    "missing_field_counts": {field: count for field, count in missing_counts.items() if count},
                    "source_tail_after_known_at_count": ordering_violations,
                }
            )

    return {
        "artifact_kind": "cross_asset_row_metadata_gap_scan",
        "schema_version": 1,
        "scanned_file_count": scanned_files,
        "scanned_row_count": scanned_rows,
        "stale_file_count": len(findings),
        "stale_rows_detected": bool(findings),
        "findings": findings,
    }


def scan_machine_local_path_markers(
    paths: Sequence[str | Path],
    *,
    root: str | Path | None = None,
    tokens: Sequence[str] = MACHINE_LOCAL_PATH_TOKENS,
    allowed_fixture_paths: Sequence[str | Path] = DEFAULT_MACHINE_LOCAL_MARKER_ALLOWED_FIXTURES,
) -> dict[str, Any]:
    base = Path(root).resolve() if root is not None else None
    allowed = {_normalized_scan_path(path, base=base) for path in allowed_fixture_paths}
    findings: list[dict[str, Any]] = []
    ignored: list[str] = []
    scanned = 0

    for item in sorted({str(path) for path in paths}):
        path = Path(item)
        if not path.is_file():
            continue
        normalized = _normalized_scan_path(path, base=base)
        if normalized in allowed:
            ignored.append(normalized)
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        matched = [token for token in tokens if token in text]
        if matched:
            findings.append({"path": normalized, "tokens": matched})

    return {
        "artifact_kind": "machine_local_path_marker_scan",
        "schema_version": 1,
        "scanned_file_count": scanned,
        "ignored_fixture_count": len(ignored),
        "finding_count": len(findings),
        "findings": findings,
        "ignored_fixtures": ignored,
    }


def build_root_cleanup_manifest(
    root: str | Path,
    *,
    candidate_names: Sequence[str] = ROOT_CLEANUP_CANDIDATES,
    manifest_id: str = "root_cleanup_manifest",
) -> dict[str, Any]:
    base = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    for name in candidate_names:
        target = (base / name).resolve()
        if not _is_within(target, base):
            raise ValueError(f"cleanup candidate resolves outside root: {name!r}")
        exists = target.exists()
        files = [path for path in target.rglob("*") if path.is_file()] if exists and target.is_dir() else []
        dirs = [path for path in target.rglob("*") if path.is_dir()] if exists and target.is_dir() else []
        entries.append(
            {
                "relative_path": _relative_or_posix(target, base),
                "exists": exists,
                "path_type": "directory" if target.is_dir() else "file" if target.is_file() else "missing",
                "file_count": len(files),
                "directory_count": len(dirs),
                "byte_count": sum(path.stat().st_size for path in files),
                "recommended_action": "needs_human_decision_before_quarantine" if exists else "none",
            }
        )
    return {
        "artifact_kind": "regime_root_cleanup_manifest",
        "schema_version": 1,
        "manifest_id": manifest_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_policy": "manifest_only_no_moves_or_deletes",
        "production_data_move_performed": False,
        "production_delete_performed": False,
        "entries": entries,
    }


def _unsafe_payload_fields(payload: Any, *, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            findings.extend(_unsafe_payload_fields(value, path=f"{path}.{key}"))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for idx, value in enumerate(payload):
            findings.extend(_unsafe_payload_fields(value, path=f"{path}[{idx}]"))
    elif isinstance(payload, str) and (is_unsafe_serialized_path(payload) or contains_machine_local_marker(payload)):
        findings.append(path)
    return findings


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_within_any(path: Path, roots: Sequence[Path]) -> bool:
    return any(_is_within(path, root) for root in roots)


def _relative_or_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalized_scan_path(path: str | Path, *, base: Path | None) -> str:
    candidate = Path(path)
    if base is not None:
        try:
            return candidate.resolve().relative_to(base).as_posix()
        except ValueError:
            pass
    return candidate.as_posix()


def _orderable(value: object) -> float:
    if value is None or isinstance(value, bool):
        return float("inf")
    try:
        return float(value)
    except Exception:
        return float("inf")


__all__ = [
    "CROSS_ASSET_FEATURE_ROW_DIR_NAME",
    "DEFAULT_MACHINE_LOCAL_MARKER_ALLOWED_FIXTURES",
    "MACHINE_LOCAL_PATH_TOKENS",
    "ROOT_CLEANUP_CANDIDATES",
    "build_root_cleanup_manifest",
    "classify_artifact_portability",
    "detect_cross_asset_row_metadata_gaps",
    "discover_cross_asset_feature_row_paths",
    "find_unsafe_serialized_strings",
    "scan_machine_local_path_markers",
]
