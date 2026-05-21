from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_refs import (
    ArtifactRef,
    is_unsafe_serialized_path,
    make_artifact_ref,
    portable_ref_dict,
    validate_portable_artifact_ref,
)
from src.regimes.core.disk_safety import (
    artifact_record_from_ref,
    build_disk_safety_report,
    validate_artifact_records,
    validate_disk_safety_report,
)
from src.regimes.core.serialization import to_jsonable


def build_artifact_inventory(
    artifact_paths: Sequence[str | Path],
    *,
    artifact_root: str | Path,
    inventory_id: str,
    producer: str,
    report_root: str | Path | None = None,
) -> dict[str, Any]:
    refs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in sorted({str(Path(item)) for item in artifact_paths if str(item).strip()}):
        resolved = Path(path)
        if not resolved.exists() or not resolved.is_file():
            continue
        ref = make_artifact_ref(
            resolved,
            artifact_kind=_artifact_kind_for_path(resolved),
            artifact_root=artifact_root,
            report_root=report_root,
            producer=producer,
        ).as_dict()
        refs.append(ref)
        records.append(
            artifact_record_from_ref(
                ref,
                path=resolved,
                artifact_root=artifact_root,
                producer=producer,
            )
        )
    disk_safety_report = build_disk_safety_report(records, report_id=f"{inventory_id}_disk_safety")
    payload = {
        "artifact_kind": "regime_pathway_artifact_inventory",
        "schema_version": 1,
        "inventory_id": inventory_id,
        "artifact_count": len(refs),
        "artifact_refs": refs,
        "artifact_records": records,
        "disk_safety_report": disk_safety_report,
        "disk_safety_validation": {
            "portable_refs_validated": True,
            "absolute_paths_serialized": False,
            "production_outputs_written": False,
            "production_promotion_performed": False,
            "broad_all_to_all_pairwise_blocked": True,
            "full_rolling_matrices_not_persisted": True,
            "one_column_per_related_asset_schema_not_introduced": True,
        },
    }
    validate_artifact_inventory(payload)
    return payload


def validate_artifact_inventory(inventory: Mapping[str, Any]) -> None:
    payload = dict(inventory)
    refs = payload.get("artifact_refs", ())
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        raise ValueError("artifact inventory artifact_refs must be a sequence")
    for ref in refs:
        validate_portable_artifact_ref(ref)
        if _row_producing_ref_has_null_timestamps(ref):
            raise ValueError("artifact inventory row-producing refs require known_at_ts and source_tail_ts")
    records = payload.get("artifact_records", ())
    if records:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("artifact inventory artifact_records must be a sequence")
        validate_artifact_records(records)
    disk_safety = payload.get("disk_safety_report")
    if disk_safety:
        if not isinstance(disk_safety, Mapping):
            raise ValueError("artifact inventory disk_safety_report must be a mapping")
        validate_disk_safety_report(disk_safety)
    unsafe = find_unsafe_path_strings(payload)
    if unsafe:
        raise ValueError(f"artifact inventory contains unsafe serialized path strings: {unsafe[:5]}")
    safety = payload.get("disk_safety_validation")
    if not isinstance(safety, Mapping):
        raise ValueError("artifact inventory requires disk_safety_validation")
    for flag in ("absolute_paths_serialized", "production_outputs_written", "production_promotion_performed"):
        if bool(safety.get(flag, False)):
            raise ValueError(f"artifact inventory {flag} must be false")


def find_unsafe_path_strings(payload: Any, *, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            findings.extend(find_unsafe_path_strings(value, path=f"{path}.{key}"))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for idx, value in enumerate(payload):
            findings.extend(find_unsafe_path_strings(value, path=f"{path}[{idx}]"))
    elif is_unsafe_serialized_path(payload):
        findings.append(path)
    return findings


def _row_producing_ref_has_null_timestamps(ref: Mapping[str, Any]) -> bool:
    path = str(ref.get("relative_path_from_artifact_root") or ref.get("relative_path_from_report_root") or "")
    if not _is_row_producing_ref_path(path):
        return False
    return ref.get("known_at_ts") is None or ref.get("source_tail_ts") is None


def _is_row_producing_ref_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith(".jsonl") or any(
        token in normalized
        for token in (
            "/cross_asset_feature_rows/",
            "/selected_relationship_edges/",
            "/asset_relationship_profiles/",
            "/relationship_stability_scores/",
            "/isolated_asset_profiles/",
            "/edge_alias_manifest/",
            "market_state_axis_panel_",
            "regime_features_market_",
            "/asset_state_regime_labels.",
        )
    )


def refs_mapping_from_paths(
    paths: Mapping[str, str | Path],
    *,
    artifact_root: str | Path,
    producer: str,
    report_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        out[str(name)] = portable_ref_dict(
            make_artifact_ref(
                path,
                artifact_kind=str(name),
                artifact_root=artifact_root,
                report_root=report_root,
                producer=producer,
            )
        )
    return to_jsonable(out)


def _artifact_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "artifact"
    stem = path.stem.lower()
    if stem in {"manifest", "feature_manifest", "method_manifest", "snapshot"}:
        return stem
    if suffix in {"json", "jsonl", "parquet", "csv", "md"}:
        return f"{suffix}_artifact"
    return "file_artifact"


__all__ = [
    "build_artifact_inventory",
    "find_unsafe_path_strings",
    "refs_mapping_from_paths",
    "validate_artifact_inventory",
]
