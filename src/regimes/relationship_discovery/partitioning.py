from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from src.regimes.relationship_discovery.schemas import (
    ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
    EDGE_ALIAS_MANIFEST_SCHEMA_ID,
    ISOLATED_ASSET_PROFILES_SCHEMA_ID,
    METHOD_MANIFEST_SCHEMA_ID,
    REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
    RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
    SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
)


RELATIONSHIP_DISCOVERY_PARTITION_ROOT = "relationship_discovery"

_PARTITION_VALUE_RE = re.compile(r"^[A-Za-z0-9_.=-]+$")

_ROW_ARTIFACT_SCHEMA_IDS: frozenset[str] = frozenset(
    {
        SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
        ISOLATED_ASSET_PROFILES_SCHEMA_ID,
        EDGE_ALIAS_MANIFEST_SCHEMA_ID,
    }
)


def relationship_discovery_partition_path(
    schema_id: str,
    row: Mapping[str, Any],
    *,
    run_id: str | None = None,
    format: str = "parquet",
) -> Path:
    schema_id = _schema_id(schema_id)
    format = _format(format)
    if schema_id == METHOD_MANIFEST_SCHEMA_ID:
        run = _partition_value(run_id if run_id is not None else row.get("run_id"), field_name="run_id")
        return Path(RELATIONSHIP_DISCOVERY_PARTITION_ROOT) / schema_id / f"run_id={run}" / "method_manifest.json"
    if schema_id == REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID:
        refit = _partition_value(row.get("refit_key"), field_name="refit_key")
        interval = _partition_int(row.get("interval"), field_name="interval")
        return (
            Path(RELATIONSHIP_DISCOVERY_PARTITION_ROOT)
            / schema_id
            / f"refit_key={refit}"
            / f"interval={interval}"
            / "snapshot.json"
        )
    if schema_id in _ROW_ARTIFACT_SCHEMA_IDS:
        interval = _partition_int(row.get("interval"), field_name="interval")
        window = _partition_int(row.get("window"), field_name="window")
        refit = _partition_value(row.get("refit_key"), field_name="refit_key")
        return (
            Path(RELATIONSHIP_DISCOVERY_PARTITION_ROOT)
            / schema_id
            / f"interval={interval}"
            / f"window={window}"
            / f"refit_key={refit}"
            / f"part-000.{format}"
        )
    raise ValueError(f"Unknown Relationship Discovery v1 schema_id {schema_id!r}")


def relationship_discovery_partition_group(
    schema_id: str,
    row: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> tuple[str, str, str, str]:
    schema_id = _schema_id(schema_id)
    if schema_id == METHOD_MANIFEST_SCHEMA_ID:
        run = _partition_value(run_id if run_id is not None else row.get("run_id"), field_name="run_id")
        return (schema_id, f"run_id={run}", "", "")
    if schema_id == REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID:
        refit = _partition_value(row.get("refit_key"), field_name="refit_key")
        interval = _partition_int(row.get("interval"), field_name="interval")
        return (schema_id, f"refit_key={refit}", f"interval={interval}", "")
    if schema_id in _ROW_ARTIFACT_SCHEMA_IDS:
        interval = _partition_int(row.get("interval"), field_name="interval")
        window = _partition_int(row.get("window"), field_name="window")
        refit = _partition_value(row.get("refit_key"), field_name="refit_key")
        return (schema_id, f"interval={interval}", f"window={window}", f"refit_key={refit}")
    raise ValueError(f"Unknown Relationship Discovery v1 schema_id {schema_id!r}")


def relationship_discovery_partition_contract() -> dict[str, str]:
    return {
        METHOD_MANIFEST_SCHEMA_ID: "relationship_discovery/method_manifest/run_id=<run_id>/method_manifest.json",
        REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID: "relationship_discovery/refit_snapshot_manifest/refit_key=<refit_key>/interval=<interval>/snapshot.json",
        SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID: "relationship_discovery/selected_relationship_edges/interval=<interval>/window=<window>/refit_key=<refit_key>/part-000.parquet",
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID: "relationship_discovery/asset_relationship_profiles/interval=<interval>/window=<window>/refit_key=<refit_key>/part-000.parquet",
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID: "relationship_discovery/relationship_stability_scores/interval=<interval>/window=<window>/refit_key=<refit_key>/part-000.parquet",
        ISOLATED_ASSET_PROFILES_SCHEMA_ID: "relationship_discovery/isolated_asset_profiles/interval=<interval>/window=<window>/refit_key=<refit_key>/part-000.parquet",
        EDGE_ALIAS_MANIFEST_SCHEMA_ID: "relationship_discovery/edge_alias_manifest/interval=<interval>/window=<window>/refit_key=<refit_key>/part-000.parquet",
    }


def _schema_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Relationship Discovery v1 schema_id must be non-empty")
    return text


def _format(value: object) -> str:
    text = str(value).strip().lower()
    if text not in {"parquet", "jsonl"}:
        raise ValueError("Relationship Discovery v1 partition format must be parquet or jsonl")
    return text


def _partition_int(value: object, *, field_name: str) -> str:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery v1 partition {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Relationship Discovery v1 partition {field_name} must be positive")
    return str(out)


def _partition_value(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text == ".." or "/" in text or "\\" in text:
        raise ValueError(f"Relationship Discovery v1 partition {field_name} must be a safe path segment")
    if not _PARTITION_VALUE_RE.match(text):
        raise ValueError(f"Relationship Discovery v1 partition {field_name} contains unsupported characters")
    return text


__all__ = [
    "RELATIONSHIP_DISCOVERY_PARTITION_ROOT",
    "relationship_discovery_partition_contract",
    "relationship_discovery_partition_group",
    "relationship_discovery_partition_path",
]
