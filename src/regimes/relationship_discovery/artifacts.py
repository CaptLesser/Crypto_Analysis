from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.paths import resolve_project_path
from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.schemas import (
    process1_artifact_schema_manifest,
    process1_artifact_schemas,
    validate_process1_artifact_rows,
)


RELATIONSHIP_DISCOVERY_OUTPUT_ENV = "PIPELINE_RELATIONSHIP_DISCOVERY_REPORT_ROOT"
RELATIONSHIP_DISCOVERY_REPORT_SUBDIR = "relationship_discovery_prototype"

RELATIONSHIP_ARTIFACT_WRITE_STATUS_WRITTEN = "written"
RELATIONSHIP_ARTIFACT_WRITE_STATUS_NO_ROWS = "no_rows"


@dataclass(frozen=True)
class RelationshipArtifactWriteResult:
    status: str
    output_root: Path
    written_paths: Sequence[Path] = ()
    row_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_artifact_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "written_paths": [str(path) for path in self.written_paths],
            "row_count": int(self.row_count),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_writes_enabled": False,
        }


def validate_relationship_discovery_report_root(
    output_root: str | Path,
    *,
    project_root: str | Path | None = None,
    production_enabled: bool = False,
) -> Path:
    if production_enabled is not False:
        raise ValueError("Relationship Discovery production writes are disabled")
    return validate_report_root(output_root, project_root=project_root, allow_foundation_descendant=True)


def default_relationship_discovery_report_root(
    *,
    report_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    if report_root is not None:
        return validate_relationship_discovery_report_root(report_root, project_root=project_root)
    source_env = env if env is not None else os.environ
    raw = str(source_env.get(RELATIONSHIP_DISCOVERY_OUTPUT_ENV, "") or "").strip()
    if raw:
        return validate_relationship_discovery_report_root(raw, project_root=project_root)
    project = resolve_project_path(".", project_root=project_root).resolve()
    return validate_relationship_discovery_report_root(
        project / "reports" / "regimes" / "foundation" / RELATIONSHIP_DISCOVERY_REPORT_SUBDIR,
        project_root=project_root,
    )


def write_relationship_json_artifact(
    artifact: Any,
    *,
    output_root: str | Path,
    relative_path: str | Path,
    production_enabled: bool = False,
) -> RelationshipArtifactWriteResult:
    root = validate_relationship_discovery_report_root(output_root, production_enabled=production_enabled)
    path = _resolve_relative_output(root, relative_path)
    payload = _artifact_payload(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RelationshipArtifactWriteResult(
        status=RELATIONSHIP_ARTIFACT_WRITE_STATUS_WRITTEN,
        output_root=root,
        written_paths=(path,),
        row_count=1,
        metadata={"format": "json"},
    )


def write_relationship_jsonl_rows(
    rows: Sequence[Any],
    *,
    output_root: str | Path,
    relative_path: str | Path,
    production_enabled: bool = False,
) -> RelationshipArtifactWriteResult:
    root = validate_relationship_discovery_report_root(output_root, production_enabled=production_enabled)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Relationship Discovery JSONL writer requires a sequence of rows")
    path = _resolve_relative_output(root, relative_path)
    payloads = [_artifact_payload(row) for row in rows]
    if not payloads:
        return RelationshipArtifactWriteResult(
            status=RELATIONSHIP_ARTIFACT_WRITE_STATUS_NO_ROWS,
            output_root=root,
            written_paths=(),
            row_count=0,
            metadata={"format": "jsonl"},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads)
    path.write_text(text, encoding="utf-8")
    return RelationshipArtifactWriteResult(
        status=RELATIONSHIP_ARTIFACT_WRITE_STATUS_WRITTEN,
        output_root=root,
        written_paths=(path,),
        row_count=len(payloads),
        metadata={"format": "jsonl"},
    )


def _artifact_payload(artifact: Any) -> dict[str, Any]:
    if hasattr(artifact, "as_dict") and callable(artifact.as_dict):
        payload = artifact.as_dict()
    elif isinstance(artifact, Mapping):
        payload = dict(artifact)
    else:
        raise ValueError("Relationship Discovery artifact writer requires an as_dict-capable object or mapping")
    obj = to_jsonable(payload)
    if not isinstance(obj, Mapping) or not obj:
        raise ValueError("Relationship Discovery artifact writer requires a non-empty JSON object")
    obj = dict(obj)
    obj.setdefault("production_enabled", False)
    if bool(obj.get("production_enabled", False)):
        raise ValueError("Relationship Discovery artifact writer refuses production-enabled artifacts")
    return obj


def _resolve_relative_output(root: Path, relative_path: str | Path) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError("Relationship Discovery artifact relative_path must not be absolute")
    if any(part in {"..", ""} for part in rel.parts):
        raise ValueError("Relationship Discovery artifact relative_path must stay within output root")
    path = root / rel
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Relationship Discovery artifact writer refusing to write outside output root: {path}") from exc
    return path


__all__ = [
    "RELATIONSHIP_ARTIFACT_WRITE_STATUS_NO_ROWS",
    "RELATIONSHIP_ARTIFACT_WRITE_STATUS_WRITTEN",
    "RELATIONSHIP_DISCOVERY_OUTPUT_ENV",
    "RELATIONSHIP_DISCOVERY_REPORT_SUBDIR",
    "RelationshipArtifactWriteResult",
    "default_relationship_discovery_report_root",
    "validate_relationship_discovery_report_root",
    "process1_artifact_schema_manifest",
    "process1_artifact_schemas",
    "validate_process1_artifact_rows",
    "write_relationship_json_artifact",
    "write_relationship_jsonl_rows",
]
