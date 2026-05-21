from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from src.regimes.core.serialization import to_jsonable


ARTIFACT_REF_SCHEMA_VERSION = 1

_LOCAL_ABSOLUTE_PREFIXES: tuple[str, ...] = (
    "/users/",
    "/" + "home/",
    "/mnt/",
    "/volumes/",
)

MACHINE_LOCAL_MARKERS: tuple[str, ...] = (
    "D:" + "\\",
    "D:" + "/",
    "C:" + "\\",
    "C:" + "/",
    "/" + "Users/",
    "/" + "home/",
    "/" + "mnt/data",
    "project" + "_cohorts",
)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_kind: str
    relative_path_from_report_root: str | None = None
    relative_path_from_artifact_root: str | None = None
    content_hash: str | None = None
    schema_version: int = ARTIFACT_REF_SCHEMA_VERSION
    producer: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    known_at_ts: int | float | str | None = None
    source_tail_ts: int | float | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "producer", _text(self.producer, field_name="producer"))
        object.__setattr__(self, "created_at", _text(self.created_at, field_name="created_at"))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if int(self.schema_version) != ARTIFACT_REF_SCHEMA_VERSION:
            raise ValueError("ArtifactRef schema_version is unsupported")
        rel_report = _optional_relative_path(self.relative_path_from_report_root, field_name="relative_path_from_report_root")
        rel_artifact = _optional_relative_path(self.relative_path_from_artifact_root, field_name="relative_path_from_artifact_root")
        if rel_report is None and rel_artifact is None:
            raise ValueError("ArtifactRef requires at least one relative path")
        object.__setattr__(self, "relative_path_from_report_root", rel_report)
        object.__setattr__(self, "relative_path_from_artifact_root", rel_artifact)
        if self.content_hash is not None:
            object.__setattr__(self, "content_hash", _content_hash(self.content_hash))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "relative_path_from_report_root": self.relative_path_from_report_root,
            "relative_path_from_artifact_root": self.relative_path_from_artifact_root,
            "content_hash": self.content_hash,
            "schema_version": int(self.schema_version),
            "producer": self.producer,
            "created_at": self.created_at,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
        }


def make_artifact_ref(
    path: str | Path,
    *,
    artifact_id: str | None = None,
    artifact_kind: str,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    producer: str,
    created_at: str | None = None,
    known_at_ts: int | float | str | None = None,
    source_tail_ts: int | float | str | None = None,
    content_hash: str | None = None,
) -> ArtifactRef:
    resolved_path = Path(path).resolve()
    rel_artifact = _relative_to_root(resolved_path, artifact_root) if artifact_root is not None else None
    rel_report = _relative_to_root(resolved_path, report_root) if report_root is not None else None
    if rel_artifact is None and rel_report is None:
        raise ValueError("ArtifactRef path must be inside artifact_root or report_root")
    digest = content_hash
    if digest is None and resolved_path.exists() and resolved_path.is_file():
        digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    inferred_known_at, inferred_source_tail = _infer_artifact_timestamps(resolved_path)
    ref_id = artifact_id or _default_artifact_id(artifact_kind, rel_report or rel_artifact or str(resolved_path))
    return ArtifactRef(
        artifact_id=ref_id,
        artifact_kind=artifact_kind,
        relative_path_from_report_root=rel_report,
        relative_path_from_artifact_root=rel_artifact,
        content_hash=digest,
        producer=producer,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        known_at_ts=known_at_ts if known_at_ts is not None else inferred_known_at,
        source_tail_ts=source_tail_ts if source_tail_ts is not None else inferred_source_tail,
    )


def validate_portable_artifact_ref(ref: ArtifactRef | Mapping[str, Any]) -> ArtifactRef:
    resolved = ref if isinstance(ref, ArtifactRef) else ArtifactRef(**dict(ref))
    for field_name in ("relative_path_from_report_root", "relative_path_from_artifact_root"):
        value = getattr(resolved, field_name)
        if value is not None:
            validate_portable_relative_path(value, field_name=field_name)
    return resolved


def resolve_artifact_ref(
    ref: ArtifactRef | Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    resolved = validate_portable_artifact_ref(ref)
    if resolved.relative_path_from_artifact_root is not None:
        if artifact_root is None:
            raise ValueError("artifact_root is required to resolve relative_path_from_artifact_root")
        return _resolve_within_root(resolved.relative_path_from_artifact_root, artifact_root, must_exist=must_exist)
    if resolved.relative_path_from_report_root is not None:
        if report_root is None:
            raise ValueError("report_root is required to resolve relative_path_from_report_root")
        return _resolve_within_root(resolved.relative_path_from_report_root, report_root, must_exist=must_exist)
    raise ValueError("ArtifactRef does not contain a resolvable relative path")


def validate_portable_relative_path(value: str | Path, *, field_name: str = "path") -> str:
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if _looks_machine_local_or_absolute(text):
        raise ValueError(f"{field_name} must be portable and relative, got {value!r}")
    pure = PurePosixPath(text)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{field_name} must stay within its artifact root")
    return pure.as_posix()


def is_unsafe_serialized_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    return _looks_machine_local_or_absolute(text)


def contains_machine_local_marker(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(marker.lower() in lowered for marker in MACHINE_LOCAL_MARKERS)


def portable_ref_dict(ref: ArtifactRef | Mapping[str, Any]) -> dict[str, Any]:
    return validate_portable_artifact_ref(ref).as_dict()


def refs_by_artifact_id(refs: Mapping[str, ArtifactRef | Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(name): portable_ref_dict(ref) for name, ref in refs.items()}


def _relative_to_root(path: Path, root: str | Path | None) -> str | None:
    if root is None:
        return None
    try:
        rel = path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return None
    return validate_portable_relative_path(rel.as_posix(), field_name="relative_path")


def _resolve_within_root(relative_path: str, root: str | Path, *, must_exist: bool) -> Path:
    rel = validate_portable_relative_path(relative_path)
    base = Path(root).resolve()
    path = (base / rel).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError("ArtifactRef resolved outside its root") from exc
    if must_exist and not path.is_file():
        raise ValueError(f"ArtifactRef target is missing: {rel}")
    return path


def _looks_machine_local_or_absolute(text: str) -> bool:
    normalized = text.replace("\\", "/")
    lowered = normalized.lower()
    if normalized.startswith("~"):
        return True
    if PurePosixPath(normalized).is_absolute():
        return True
    win = PureWindowsPath(text)
    if win.is_absolute() or win.drive or win.anchor.startswith("\\\\"):
        return True
    if re.match(r"^[A-Za-z]:", text):
        return True
    return lowered.startswith(_LOCAL_ABSOLUTE_PREFIXES)


def _optional_relative_path(value: str | Path | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return validate_portable_relative_path(text, field_name=field_name)


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"ArtifactRef {field_name} must be non-empty")
    return text


def _content_hash(value: object) -> str:
    text = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError("ArtifactRef content_hash must be a sha256 hex digest")
    return text


def _infer_artifact_timestamps(path: Path) -> tuple[Any | None, Any | None]:
    if not path.exists() or not path.is_file():
        return None, None
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        payload = json.loads(line)
                        if isinstance(payload, Mapping):
                            return payload.get("known_at_ts"), payload.get("source_tail_ts")
                        return None, None
            return None, None
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                known_at = payload.get("known_at_ts")
                source_tail = payload.get("source_tail_ts")
                if known_at is not None or source_tail is not None:
                    return known_at, source_tail
                refs = payload.get("output_artifact_refs")
                if isinstance(refs, Mapping):
                    for ref in refs.values():
                        if isinstance(ref, Mapping) and (
                            ref.get("known_at_ts") is not None or ref.get("source_tail_ts") is not None
                        ):
                            return ref.get("known_at_ts"), ref.get("source_tail_ts")
    except Exception:
        return None, None
    return None, None


def _default_artifact_id(artifact_kind: str, relative_path: str) -> str:
    payload = f"{artifact_kind}|{relative_path}".encode("utf-8")
    return f"{artifact_kind}_{hashlib.sha256(payload).hexdigest()[:16]}"


__all__ = [
    "ARTIFACT_REF_SCHEMA_VERSION",
    "ArtifactRef",
    "contains_machine_local_marker",
    "is_unsafe_serialized_path",
    "make_artifact_ref",
    "portable_ref_dict",
    "refs_by_artifact_id",
    "resolve_artifact_ref",
    "validate_portable_artifact_ref",
    "validate_portable_relative_path",
]
