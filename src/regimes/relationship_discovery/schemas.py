from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.regimes.relationship_discovery.contracts import (
    RelationshipColumnSpec,
    RelationshipOutputSchema,
)


METHOD_MANIFEST_SCHEMA_ID = "method_manifest"
REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID = "refit_snapshot_manifest"
SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID = "selected_relationship_edges"
ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID = "asset_relationship_profiles"
RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID = "relationship_stability_scores"
EDGE_ALIAS_MANIFEST_SCHEMA_ID = "edge_alias_manifest"
ISOLATED_ASSET_PROFILES_SCHEMA_ID = "isolated_asset_profiles"

METHOD_MANIFEST_ROW_GRAIN = "one row per method/policy/run or build"
REFIT_SNAPSHOT_MANIFEST_ROW_GRAIN = "one row per refit_key/interval/universe scope"
SELECTED_RELATIONSHIP_EDGES_ROW_GRAIN = (
    "one selected edge per refit_key/interval/window/method_id/asset/related asset/relationship family"
)
ASSET_RELATIONSHIP_PROFILES_ROW_GRAIN = "one row per asset/refit_key/interval/window"
RELATIONSHIP_STABILITY_SCORES_ROW_GRAIN = (
    "one row per asset/related asset/method_id/interval/window/refit_key or score window"
)
EDGE_ALIAS_MANIFEST_ROW_GRAIN = "one row per asset/refit_key/interval/window/slot"
ISOLATED_ASSET_PROFILES_ROW_GRAIN = "one row per asset/refit_key/interval/window"


def method_manifest_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=METHOD_MANIFEST_SCHEMA_ID,
        row_grain=METHOD_MANIFEST_ROW_GRAIN,
        columns=(
            _col("method_id", "string"),
            _col("method_family", "string"),
            _col("relationship_family", "string"),
            _col("source_data", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("k_policy", "string"),
            _col("residualization_policy", "string"),
            _col("normalization_policy", "string"),
            _col("thresholds", "string"),
            _col("universe_scope", "string"),
            _col("schema_version", "int64"),
            _col("generated_at", "string"),
            _col("run_id", "string"),
        ),
    )


def refit_snapshot_manifest_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
        row_grain=REFIT_SNAPSHOT_MANIFEST_ROW_GRAIN,
        columns=(
            _col("refit_key", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("effective_start_ts", "string"),
            _col("effective_end_ts", "string"),
            _col("known_at_ts", "string"),
            _col("source_tail_ts", "string"),
            _col("anchors", "string"),
            _col("core_assets", "string"),
            _col("broad_sample_assets", "string"),
            _col("excluded_assets_with_reasons", "string"),
            _col("universe_manifest_ref", "string", required=False),
            _col("universe_manifest_hash", "string", required=False),
            _col("policy_id", "string"),
            _col("schema_version", "int64"),
        ),
    )


def selected_relationship_edges_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
        row_grain=SELECTED_RELATIONSHIP_EDGES_ROW_GRAIN,
        columns=(
            _col("refit_key", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("asset", "string"),
            _col("related_asset_or_benchmark", "string"),
            _col("relationship_family", "string"),
            _col("relationship_type", "string"),
            _col("method_id", "string"),
            _col("value", "double"),
            _col("abs_value", "double"),
            _col("direction", "string"),
            _col("rank", "int64"),
            _col("slot", "string"),
            _col("selected_by_policy", "bool"),
            _col("sample_count", "int64"),
            _col("coverage", "double"),
            _col("stability_score", "double"),
            _col("activation_status", "string"),
            _col("known_at_ts", "string"),
            _col("source_tail_ts", "string"),
            _col("lineage_id", "string"),
            _col("schema_version", "int64"),
        ),
    )


def asset_relationship_profiles_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
        row_grain=ASSET_RELATIONSHIP_PROFILES_ROW_GRAIN,
        columns=(
            _col("asset", "string"),
            _col("refit_key", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("corr_to_anchor_primary", "double"),
            _col("corr_to_anchor_secondary", "double"),
            _col("corr_to_core_basket", "double"),
            _col("beta_to_core_basket", "double"),
            _col("market_mode_exposure_score", "double"),
            _col("residual_peer_signal_score", "double"),
            _col("relationship_concentration", "double"),
            _col("relationship_entropy", "double"),
            _col("top_peer_count", "int64"),
            _col("top_peer_stability_mean", "double"),
            _col("isolated_asset_score", "double"),
            _col("peer_signal_availability_status", "string"),
            _col("known_at_ts", "string"),
            _col("source_tail_ts", "string"),
            _col("lineage_id", "string"),
            _col("schema_version", "int64"),
        ),
    )


def relationship_stability_scores_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
        row_grain=RELATIONSHIP_STABILITY_SCORES_ROW_GRAIN,
        columns=(
            _col("refit_key", "string", required=False),
            _col("asset", "string"),
            _col("related_asset_or_benchmark", "string"),
            _col("method_id", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("survival_count", "int64"),
            _col("survival_share", "double"),
            _col("mean_strength", "double"),
            _col("strength_std", "double"),
            _col("sign_stability", "double"),
            _col("rank_stability", "double"),
            _col("activation_status", "string"),
            _col("enough_history", "bool"),
            _col("stability_reason", "string"),
            _col("known_at_ts", "string"),
            _col("source_tail_ts", "string"),
            _col("lineage_id", "string"),
            _col("schema_version", "int64"),
        ),
    )


def edge_alias_manifest_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=EDGE_ALIAS_MANIFEST_SCHEMA_ID,
        row_grain=EDGE_ALIAS_MANIFEST_ROW_GRAIN,
        columns=(
            _col("asset", "string"),
            _col("refit_key", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("slot", "string"),
            _col("alias_name", "string"),
            _col("related_asset", "string"),
            _col("relationship_family", "string"),
            _col("method_id", "string"),
            _col("strength", "double"),
            _col("stability_score", "double"),
            _col("activation_status", "string"),
            _col("effective_start_ts", "string"),
            _col("effective_end_ts", "string"),
            _col("known_at_ts", "string"),
            _col("source_tail_ts", "string"),
            _col("lineage_id", "string"),
            _col("schema_version", "int64"),
        ),
    )


def isolated_asset_profiles_schema() -> RelationshipOutputSchema:
    return RelationshipOutputSchema(
        schema_id=ISOLATED_ASSET_PROFILES_SCHEMA_ID,
        row_grain=ISOLATED_ASSET_PROFILES_ROW_GRAIN,
        columns=(
            _col("asset", "string"),
            _col("refit_key", "string"),
            _col("interval", "int64"),
            _col("window", "int64"),
            _col("isolation_status", "string"),
            _col("isolated_asset_score", "double"),
            _col("max_relationship_strength", "double"),
            _col("stable_edge_count", "int64"),
            _col("candidate_edge_count", "int64"),
            _col("coverage", "double"),
            _col("reason_codes", "string"),
            _col("known_at_ts", "string"),
            _col("source_tail_ts", "string"),
            _col("lineage_id", "string"),
            _col("schema_version", "int64"),
        ),
    )


def process1_artifact_schemas() -> dict[str, RelationshipOutputSchema]:
    return {
        METHOD_MANIFEST_SCHEMA_ID: method_manifest_schema(),
        REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID: refit_snapshot_manifest_schema(),
        SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID: selected_relationship_edges_schema(),
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID: asset_relationship_profiles_schema(),
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID: relationship_stability_scores_schema(),
        EDGE_ALIAS_MANIFEST_SCHEMA_ID: edge_alias_manifest_schema(),
        ISOLATED_ASSET_PROFILES_SCHEMA_ID: isolated_asset_profiles_schema(),
    }


def process1_artifact_schema_manifest() -> dict[str, Any]:
    return {
        "artifact_kind": "relationship_discovery_v1_process1_schema_manifest",
        "schemas": {schema_id: schema.as_dict() for schema_id, schema in process1_artifact_schemas().items()},
        "parquet_compatible": True,
        "json_fallback_allowed_for_tests": True,
        "production_enabled": False,
    }


def validate_process1_artifact_rows(rows_by_schema_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    schemas = process1_artifact_schemas()
    for schema_id, rows in rows_by_schema_id.items():
        if schema_id not in schemas:
            raise ValueError(f"Unknown Relationship Discovery v1 schema_id {schema_id!r}")
        schemas[schema_id].validate_rows(rows)


def _col(name: str, logical_type: str, *, required: bool = True) -> RelationshipColumnSpec:
    return RelationshipColumnSpec(name=name, logical_type=logical_type, required=required)


__all__ = [
    "ASSET_RELATIONSHIP_PROFILES_ROW_GRAIN",
    "ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID",
    "EDGE_ALIAS_MANIFEST_ROW_GRAIN",
    "EDGE_ALIAS_MANIFEST_SCHEMA_ID",
    "ISOLATED_ASSET_PROFILES_ROW_GRAIN",
    "ISOLATED_ASSET_PROFILES_SCHEMA_ID",
    "METHOD_MANIFEST_ROW_GRAIN",
    "METHOD_MANIFEST_SCHEMA_ID",
    "REFIT_SNAPSHOT_MANIFEST_ROW_GRAIN",
    "REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID",
    "RELATIONSHIP_STABILITY_SCORES_ROW_GRAIN",
    "RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID",
    "SELECTED_RELATIONSHIP_EDGES_ROW_GRAIN",
    "SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID",
    "asset_relationship_profiles_schema",
    "edge_alias_manifest_schema",
    "isolated_asset_profiles_schema",
    "method_manifest_schema",
    "process1_artifact_schema_manifest",
    "process1_artifact_schemas",
    "refit_snapshot_manifest_schema",
    "relationship_stability_scores_schema",
    "selected_relationship_edges_schema",
    "validate_process1_artifact_rows",
]
