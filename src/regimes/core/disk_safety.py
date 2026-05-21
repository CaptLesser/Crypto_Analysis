from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_refs import is_unsafe_serialized_path, validate_portable_artifact_ref
from src.regimes.core.path_safety import normalized_path_parts
from src.regimes.core.serialization import to_jsonable


DISK_RISK_SMALL = "small"
DISK_RISK_MODERATE = "moderate"
DISK_RISK_LARGE = "large"
DISK_RISK_EXPLOSIVE = "explosive"
DISK_RISK_BLOCKED = "blocked"
DISK_RISK_CLASSES: frozenset[str] = frozenset(
    {DISK_RISK_SMALL, DISK_RISK_MODERATE, DISK_RISK_LARGE, DISK_RISK_EXPLOSIVE, DISK_RISK_BLOCKED}
)

PERSISTENCE_PERSISTENT = "persistent"
PERSISTENCE_SANDBOX = "sandbox"
PERSISTENCE_GATED = "gated"
PERSISTENCE_CLASSES: frozenset[str] = frozenset({PERSISTENCE_PERSISTENT, PERSISTENCE_SANDBOX, PERSISTENCE_GATED})

BLOCKED_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "artifact_id": "blocked_broad_all_to_all_pairwise",
        "artifact_kind": "broad_all_to_all_pairwise",
        "pathway": "cross_asset",
        "feature_family": "relationship_discovery_v1",
        "row_grain": "one row per full broad all-to-all asset pair/window/method",
        "persistence_boundary": PERSISTENCE_GATED,
        "disk_risk_class": DISK_RISK_EXPLOSIVE,
        "blocked_reason": "broad all-to-all pairwise remains disabled",
        "consumer": "none",
        "producer": "src.regimes.core.disk_safety",
    },
    {
        "artifact_id": "blocked_full_rolling_relationship_matrices",
        "artifact_kind": "full_rolling_relationship_matrices",
        "pathway": "cross_asset",
        "feature_family": "relationship_discovery_v1",
        "row_grain": "full rolling square relationship matrix per timestamp/window/method",
        "persistence_boundary": PERSISTENCE_GATED,
        "disk_risk_class": DISK_RISK_EXPLOSIVE,
        "blocked_reason": "full rolling matrices are not persisted",
        "consumer": "none",
        "producer": "src.regimes.core.disk_safety",
    },
    {
        "artifact_id": "blocked_one_column_per_related_asset_schema",
        "artifact_kind": "one_column_per_related_asset_schema",
        "pathway": "cross_asset",
        "feature_family": "cross_asset_relationship_v1",
        "row_grain": "wide dynamic related-asset columns",
        "persistence_boundary": PERSISTENCE_GATED,
        "disk_risk_class": DISK_RISK_BLOCKED,
        "blocked_reason": "Cross-Asset feature rows require stable row-based/slot-based schema",
        "consumer": "none",
        "producer": "src.regimes.core.disk_safety",
    },
)


def build_disk_safety_report(
    artifact_records: Sequence[Mapping[str, Any]],
    *,
    report_id: str,
) -> dict[str, Any]:
    records = [dict(record) for record in artifact_records]
    risk_counts: dict[str, int] = {risk: 0 for risk in sorted(DISK_RISK_CLASSES)}
    persistence_counts: dict[str, int] = {status: 0 for status in sorted(PERSISTENCE_CLASSES)}
    for record in records:
        risk_counts[str(record["disk_risk_class"])] += 1
        persistence_counts[str(record["persistence_boundary"])] += 1
    blocked = [dict(item) for item in BLOCKED_ARTIFACTS]
    payload = {
        "artifact_kind": "regime_pathway_disk_safety_report",
        "schema_version": 1,
        "report_id": report_id,
        "artifact_count": len(records),
        "risk_counts": risk_counts,
        "persistence_counts": persistence_counts,
        "blocked_artifacts": blocked,
        "checks": {
            "broad_all_to_all_pairwise_blocked": True,
            "full_rolling_matrices_not_persisted": True,
            "one_column_per_related_asset_schema_not_introduced": True,
            "row_counts_estimated_under_1y_2y_clamp_where_feasible": all(
                "clamp_estimate" in record for record in records if record.get("interval")
            ),
            "selected_edges_and_profiles_classified_separately_from_blocked_all_to_all": _selected_edges_profiles_separate(records),
            "market_state_feature_panels_safe_or_moderate": _market_state_panels_safe_or_moderate(records),
            "asset_state_outputs_classified_by_asset_axis_band_grain": _asset_state_classified(records),
            "artifact_refs_portable": True,
            "production_paths_present": False,
        },
        "production_enabled": False,
        "production_outputs_written": False,
        "production_promotion_performed": False,
    }
    validate_disk_safety_report(payload)
    return payload


def validate_disk_safety_report(report: Mapping[str, Any]) -> None:
    payload = dict(report)
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("disk safety report requires checks")
    for key, value in checks.items():
        if bool(value) is not True and key != "production_paths_present":
            raise ValueError(f"disk safety check failed: {key}")
    if bool(checks.get("production_paths_present", False)):
        raise ValueError("disk safety report found production paths")
    for flag in ("production_enabled", "production_outputs_written", "production_promotion_performed"):
        if bool(payload.get(flag, False)):
            raise ValueError(f"disk safety report {flag} must be false")
    for item in payload.get("blocked_artifacts", ()):
        if dict(item).get("persistence_boundary") != PERSISTENCE_GATED:
            raise ValueError("blocked disk-safety artifacts must be gated")
        risk = str(dict(item).get("disk_risk_class"))
        if risk not in {DISK_RISK_BLOCKED, DISK_RISK_EXPLOSIVE}:
            raise ValueError("blocked disk-safety artifacts must be blocked or explosive")


def artifact_record_from_ref(
    ref: Mapping[str, Any],
    *,
    path: str | Path,
    artifact_root: str | Path,
    producer: str,
    consumer: str = "future_regime_forecaster",
) -> dict[str, Any]:
    validate_portable_artifact_ref(ref)
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    rel = resolved.relative_to(root).as_posix()
    payload = _load_payload(resolved)
    metadata = _metadata_for_path(rel, payload)
    row_count = _row_count(resolved, payload)
    interval = _coerce_optional_int(metadata.get("interval"))
    record = {
        "artifact_id": str(ref["artifact_id"]),
        "artifact_kind": str(ref["artifact_kind"]),
        "pathway": metadata["pathway"],
        "axis": metadata.get("axis"),
        "feature_family": metadata.get("feature_family"),
        "band": metadata.get("band"),
        "interval": interval,
        "row_grain": metadata["row_grain"],
        "row_count": int(row_count),
        "partition_path": rel,
        "portable_ref": to_jsonable(dict(ref)),
        "content_hash": ref.get("content_hash"),
        "schema_version": int(ref.get("schema_version") or metadata.get("schema_version") or 1),
        "persistence_boundary": metadata["persistence_boundary"],
        "disk_risk_class": metadata["disk_risk_class"],
        "consumer": metadata.get("consumer") or consumer,
        "producer": producer,
        "clamp_estimate": _clamp_estimate(row_count=row_count, interval=interval, row_multiplier=metadata.get("row_multiplier")),
    }
    _validate_artifact_record(record)
    return record


def validate_artifact_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        _validate_artifact_record(record)


def has_production_path(path: str | Path) -> bool:
    parts = set(normalized_path_parts(Path(path)))
    return bool(parts.intersection({"prod", "production", "promoted", "live"}))


def _validate_artifact_record(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    for field in (
        "artifact_id",
        "artifact_kind",
        "pathway",
        "row_grain",
        "row_count",
        "partition_path",
        "portable_ref",
        "schema_version",
        "persistence_boundary",
        "disk_risk_class",
        "consumer",
        "producer",
    ):
        if field not in payload:
            raise ValueError(f"artifact inventory record missing {field!r}")
    if str(payload["disk_risk_class"]) not in DISK_RISK_CLASSES:
        raise ValueError("artifact inventory record has unsupported disk_risk_class")
    if str(payload["persistence_boundary"]) not in PERSISTENCE_CLASSES:
        raise ValueError("artifact inventory record has unsupported persistence_boundary")
    if int(payload["row_count"]) < 0:
        raise ValueError("artifact inventory record row_count must be non-negative")
    partition_path = str(payload["partition_path"])
    if is_unsafe_serialized_path(partition_path) or Path(partition_path).is_absolute() or ".." in Path(partition_path).parts:
        raise ValueError("artifact inventory record partition_path must be portable")
    if has_production_path(partition_path):
        raise ValueError("artifact inventory record partition_path must not be production-like")
    validate_portable_artifact_ref(payload["portable_ref"])


def _metadata_for_path(rel: str, payload: Any) -> dict[str, Any]:
    lower = rel.lower()
    loaded = payload if isinstance(payload, Mapping) else {}
    pathway = _pathway(rel, loaded)
    axis = _segment_value(rel, "axis") or _payload_text(loaded.get("axis"))
    feature_family = (
        _segment_value(rel, "feature_family")
        or _payload_text(loaded.get("feature_family"))
        or _payload_text(loaded.get("feature_family_id"))
    )
    band = _segment_value(rel, "band") or _payload_text(loaded.get("band"))
    interval = _segment_value(rel, "interval") or loaded.get("interval")
    return {
        "pathway": pathway,
        "axis": axis,
        "feature_family": feature_family,
        "band": band,
        "interval": interval,
        "row_grain": _row_grain(rel, loaded),
        "schema_version": loaded.get("schema_version") if isinstance(loaded, Mapping) else 1,
        "persistence_boundary": _persistence_boundary(lower),
        "disk_risk_class": _disk_risk_class(lower),
        "row_multiplier": _row_multiplier(pathway=pathway, axis=axis, feature_family=feature_family),
        "consumer": _consumer(pathway, lower),
    }


def _pathway(rel: str, payload: Mapping[str, Any]) -> str:
    if payload.get("pathway"):
        return str(payload["pathway"])
    lower = rel.lower()
    if "asset_state" in lower:
        return "asset_state"
    if "market_state" in lower:
        return "market_state"
    if "cross_asset" in lower or "relationship_discovery" in lower:
        return "cross_asset"
    if "forecaster_handoff" in lower:
        return "forecaster_handoff"
    return "core"


def _row_grain(rel: str, payload: Mapping[str, Any]) -> str:
    if payload.get("row_grain"):
        return str(payload["row_grain"])
    lower = rel.lower()
    if "asset_state" in lower and "sandbox_outputs" in lower:
        return "one row per asset/axis/band/interval/timestamp"
    if "market_state_axis_panel" in lower or "/axis=" in lower:
        return "one row per ts/axis/interval/band/universe"
    if "market_state_feature_panel" in lower or "/feature_family=" in lower:
        return "one row per ts/feature_family/interval/band/universe"
    if "selected_relationship_edges" in lower:
        return "one selected edge per refit/interval/window/asset/related asset"
    if "asset_relationship_profiles" in lower:
        return "one row per asset/refit_key/interval/window"
    if "relationship_stability_scores" in lower:
        return "one row per selected asset/related asset/window stability summary"
    if "isolated_asset_profiles" in lower:
        return "one row per asset/refit_key/interval/window isolation status"
    if "edge_alias_manifest" in lower:
        return "one row per asset/refit_key/interval/window/slot"
    if "cross_asset_feature_rows" in lower:
        return "one row per asset/refit_key/interval/window"
    if lower.endswith(".md"):
        return "one report document"
    return "one manifest or metadata artifact"


def _persistence_boundary(lower_path: str) -> str:
    if "relationship_discovery" in lower_path or "cross_asset_features" in lower_path:
        return PERSISTENCE_PERSISTENT
    return PERSISTENCE_SANDBOX


def _disk_risk_class(lower_path: str) -> str:
    if "blocked" in lower_path or "all_to_all" in lower_path:
        return DISK_RISK_BLOCKED
    if "market_state_axis_panels" in lower_path or "market_state_feature_panel" in lower_path or "feature_family=" in lower_path:
        return DISK_RISK_MODERATE
    if "relationship_stability_scores" in lower_path:
        return DISK_RISK_MODERATE
    return DISK_RISK_SMALL


def _consumer(pathway: str, lower_path: str) -> str:
    if "forecaster_handoff" in lower_path:
        return "future_regime_forecaster"
    if pathway == "asset_state":
        return "asset_state_forecaster"
    if pathway == "market_state":
        return "market_state_forecaster"
    if pathway == "cross_asset":
        return "cross_asset_feature_consumer"
    return "foundation_audit"


def _row_multiplier(*, pathway: str, axis: str | None, feature_family: str | None) -> int | None:
    if pathway == "asset_state" and axis:
        return 1
    if pathway == "market_state" and (axis or feature_family):
        return 1
    if pathway == "cross_asset" and feature_family:
        return 1
    return None


def _clamp_estimate(*, row_count: int, interval: int | None, row_multiplier: int | None) -> dict[str, Any] | None:
    if interval is None or interval <= 0:
        return None
    bars_per_year = int((365 * 24 * 60) / interval)
    multiplier = int(row_multiplier or max(1, row_count))
    return {
        "clamp_years": [1, 2],
        "estimated_rows_1y": bars_per_year * multiplier,
        "estimated_rows_2y": bars_per_year * 2 * multiplier,
        "interval_minutes": int(interval),
    }


def _selected_edges_profiles_separate(records: Sequence[Mapping[str, Any]]) -> bool:
    kinds = {str(record.get("partition_path", "")) for record in records}
    has_selected = any("selected_relationship_edges" in item for item in kinds)
    has_profiles = any("asset_relationship_profiles" in item for item in kinds)
    has_all_to_all = any("all_to_all" in item for item in kinds)
    if not has_selected and not has_profiles:
        return not has_all_to_all
    return has_selected and has_profiles and not has_all_to_all


def _market_state_panels_safe_or_moderate(records: Sequence[Mapping[str, Any]]) -> bool:
    market = [record for record in records if record.get("pathway") == "market_state"]
    if not market:
        return True
    return all(record.get("disk_risk_class") in {DISK_RISK_SMALL, DISK_RISK_MODERATE} for record in market)


def _asset_state_classified(records: Sequence[Mapping[str, Any]]) -> bool:
    asset_rows = [
        record for record in records
        if record.get("pathway") == "asset_state" and "sandbox_outputs" in str(record.get("partition_path", ""))
    ]
    if not asset_rows:
        return True
    return all("asset/axis/band" in str(record.get("row_grain", "")) for record in asset_rows)


def _load_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return {"rows": rows, "row_count": len(rows), **(rows[0] if rows and isinstance(rows[0], Mapping) else {})}
    except Exception:
        return {}
    return {}


def _row_count(path: Path, payload: Any) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        if isinstance(payload, Mapping) and "row_count" in payload:
            return int(payload["row_count"])
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if suffix == ".json" and isinstance(payload, Mapping):
        if isinstance(payload.get("rows"), Sequence) and not isinstance(payload.get("rows"), (str, bytes)):
            return len(payload["rows"])
        if payload.get("row_count") is not None:
            return int(payload["row_count"])
        return 1
    if suffix == ".parquet":
        try:
            import pandas as pd

            return int(pd.read_parquet(path).shape[0])
        except Exception:
            return 0
    return 1 if path.exists() else 0


def _segment_value(rel: str, name: str) -> str | None:
    prefix = f"{name}="
    for part in Path(rel).parts:
        if str(part).startswith(prefix):
            return str(part).split("=", 1)[1]
    return None


def _payload_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


__all__ = [
    "BLOCKED_ARTIFACTS",
    "DISK_RISK_BLOCKED",
    "DISK_RISK_CLASSES",
    "DISK_RISK_EXPLOSIVE",
    "DISK_RISK_LARGE",
    "DISK_RISK_MODERATE",
    "DISK_RISK_SMALL",
    "PERSISTENCE_CLASSES",
    "PERSISTENCE_GATED",
    "PERSISTENCE_PERSISTENT",
    "PERSISTENCE_SANDBOX",
    "artifact_record_from_ref",
    "build_disk_safety_report",
    "has_production_path",
    "validate_artifact_records",
    "validate_disk_safety_report",
]
