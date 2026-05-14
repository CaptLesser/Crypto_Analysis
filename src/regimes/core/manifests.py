from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.contracts import TrialResultEnvelope, require_schema_version
from src.regimes.core.paths import default_foundation_report_root, is_relative_to, path_ends_with_parts
from src.regimes.core.serialization import require_json_object, to_jsonable


FOUNDATION_MANIFEST_ROOT = default_foundation_report_root()
FOUNDATION_MANIFEST_KIND = "regime_foundation_manifest"


def is_foundation_manifest_root(root: Path | str) -> bool:
    return path_ends_with_parts(Path(root), ("reports", "regimes", "foundation"))


def require_foundation_manifest_root(root: Path | str) -> Path:
    root_path = Path(root)
    if not is_foundation_manifest_root(root_path):
        raise ValueError("Regime foundation manifests must use a reports/regimes/foundation root")
    return root_path


def resolve_foundation_manifest_path(
    path: Path | str,
    *,
    root: Path | str = FOUNDATION_MANIFEST_ROOT,
) -> Path:
    root_path = require_foundation_manifest_root(root)
    requested = Path(path)
    if requested.is_absolute():
        candidate = requested
    else:
        if ".." in requested.parts:
            raise ValueError("Regime foundation manifest paths cannot traverse parent directories")
        candidate = root_path / requested
    if candidate.suffix.lower() != ".json":
        raise ValueError("Regime foundation manifest paths must end in .json")
    root_resolved = root_path.resolve()
    candidate_resolved = candidate.resolve()
    if not is_relative_to(candidate_resolved, root_resolved):
        raise ValueError("Regime foundation manifest path must stay under reports/regimes/foundation")
    return candidate


def validate_foundation_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    obj = require_json_object(payload, context="Regime foundation manifest")
    if "schema_version" not in obj:
        raise ValueError("Regime foundation manifest missing required fields: schema_version")
    require_schema_version(obj["schema_version"])
    return to_jsonable(obj)


def save_foundation_manifest(
    payload: Mapping[str, Any] | Any,
    path: Path | str,
    *,
    root: Path | str = FOUNDATION_MANIFEST_ROOT,
) -> Path:
    target = resolve_foundation_manifest_path(path, root=root)
    obj = validate_foundation_manifest_payload(to_jsonable(payload))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    return target


def load_foundation_manifest(
    path: Path | str,
    *,
    root: Path | str = FOUNDATION_MANIFEST_ROOT,
) -> dict[str, Any]:
    target = resolve_foundation_manifest_path(path, root=root)
    if not target.exists():
        raise FileNotFoundError(f"Regime foundation manifest not found: {target}")
    with target.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    return validate_foundation_manifest_payload(payload)


def save_trial_result_envelope_manifest(
    envelope: TrialResultEnvelope,
    path: Path | str,
    *,
    root: Path | str = FOUNDATION_MANIFEST_ROOT,
) -> Path:
    return save_foundation_manifest(envelope.as_dict(), path, root=root)


def load_trial_result_envelope_manifest(
    path: Path | str,
    *,
    root: Path | str = FOUNDATION_MANIFEST_ROOT,
) -> TrialResultEnvelope:
    return TrialResultEnvelope.from_dict(load_foundation_manifest(path, root=root))


__all__ = [
    "FOUNDATION_MANIFEST_KIND",
    "FOUNDATION_MANIFEST_ROOT",
    "is_foundation_manifest_root",
    "load_foundation_manifest",
    "load_trial_result_envelope_manifest",
    "require_foundation_manifest_root",
    "resolve_foundation_manifest_path",
    "save_foundation_manifest",
    "save_trial_result_envelope_manifest",
    "validate_foundation_manifest_payload",
]
