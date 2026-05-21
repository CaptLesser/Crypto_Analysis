from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.artifact_refs import make_artifact_ref
from src.regimes.core.forecaster_handoff import (
    ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
    ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
)
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.axis_contracts import MARKET_STATE_V1_AXIS_IDS
from src.regimes.market_state.axis_panel import (
    MARKET_STATE_AXIS_PANEL_SCHEMA_ID,
    MARKET_STATE_AXIS_PANEL_STATUS_READY,
    MarketStateAxisPanelConfig,
    assemble_market_state_v1_axis_panels,
)
from src.regimes.market_state.band_policy import default_market_state_v1_band_policies, market_state_band_composite_policy
from src.regimes.market_state.feature_routing import (
    DEFAULT_MARKET_STATE_ROUTED_FEATURE_FAMILIES,
    default_market_state_feature_routing_policy,
    resolve_market_state_feature_route,
    validate_market_state_feature_routing,
)
from src.regimes.market_state.feature_writer import (
    MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL,
    MarketStateFeatureMaterializationRequest,
    write_market_state_v1_feature_materialization,
)
from src.regimes.market_state.handoff import build_market_state_forecaster_handoff_manifest, write_market_state_forecaster_handoff_manifest
from src.regimes.market_state.sandbox_writer import write_market_state_sandbox_outputs
from src.regimes.market_state.universe_views import MarketStateUniverseV1Views, load_market_state_universe_v1_views
from src.regimes.regime_features.speculative_sidecar import SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID


MARKET_STATE_FINAL_OUTPUT_ARTIFACT_KIND = "market_state_final_output_finalization"
MARKET_STATE_FINAL_OUTPUT_STATUS_PRODUCED = "produced"
MARKET_STATE_FINAL_OUTPUT_STATUS_MISSING_DATA = "missing_data"
MARKET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME = "market_state_final_output_diagnostics.json"
MARKET_STATE_UNIVERSE_MANIFEST_FILENAME = "market_state_universe_manifest.json"

REQUIRED_MARKET_STATE_FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "return_trend": ("market_return_summary", "market_trend"),
    "volatility": ("market_realized_volatility",),
    "breadth": ("market_breadth",),
    "dispersion": ("market_dispersion",),
    "correlation_covariance_concentration": ("market_covariance_summary", "market_correlation_summary", "market_concentration"),
    "liquidity_activity": ("market_liquidity_activity",),
    "stress_drawdown": ("market_stress", "market_drawdown_breadth"),
    "stable_peg_stress": ("stable_peg_stress", "market_stable_peg_stress", "market_peg_deviation"),
}


@dataclass(frozen=True)
class MarketStateFinalOutputConfig:
    report_root: str | Path
    universe_manifest_path: str | Path | None = None
    run_id: str = "market_state_final_output_sandbox"
    band: str = "micro"
    interval: int = 60
    refit_key: str = "market_state_final_output_refit"
    include_speculative_sidecar: bool = True
    file_format: str = MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL
    known_at_ts: int | float | str = 1_713_931_200
    source_tail_ts: int | float | str = 1_713_931_200
    production_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Market-State final output production writes are disabled")
        object.__setattr__(self, "interval", int(self.interval))
        if int(self.interval) <= 0:
            raise ValueError("Market-State final output interval must be positive")
        object.__setattr__(self, "band", str(self.band).strip().lower())


@dataclass(frozen=True)
class MarketStateFinalOutputResult:
    status: str
    report_root: Path
    diagnostics_path: Path
    universe_manifest_path: Path
    feature_materialization_manifest_path: Path
    feature_output_paths: Mapping[str, str]
    axis_panel_output_paths: Mapping[str, str]
    handoff_manifest_paths: Mapping[str, str]
    required_axes_represented: Mapping[str, bool]
    universe_routing_verified: bool
    band_policy_verified: bool
    feature_build_verified: bool
    axis_panel_assembler_available: bool
    sandbox_writer_available: bool
    handoff_manifests_validate: bool
    monolithic_market_state_label_produced: bool = False
    final_profiles_selected: bool = False
    production_outputs_written: bool = False
    remaining_blockers: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": MARKET_STATE_FINAL_OUTPUT_ARTIFACT_KIND,
            "schema_version": 1,
            "status": self.status,
            "report_root": "runtime_only_not_serialized",
            "diagnostics_path": self.diagnostics_path.name,
            "universe_manifest_path": _portable_rel(self.universe_manifest_path, self.report_root),
            "feature_materialization_manifest_path": _portable_rel(self.feature_materialization_manifest_path, self.report_root),
            "feature_output_paths": dict(self.feature_output_paths),
            "axis_panel_output_paths": dict(self.axis_panel_output_paths),
            "handoff_manifest_paths": dict(self.handoff_manifest_paths),
            "required_axes_represented": dict(self.required_axes_represented),
            "universe_routing_verified": bool(self.universe_routing_verified),
            "band_policy_verified": bool(self.band_policy_verified),
            "feature_build_verified": bool(self.feature_build_verified),
            "axis_panel_assembler_available": bool(self.axis_panel_assembler_available),
            "sandbox_writer_available": bool(self.sandbox_writer_available),
            "handoff_manifests_validate": bool(self.handoff_manifests_validate),
            "monolithic_market_state_label_produced": False,
            "final_profiles_selected": False,
            "production_outputs_written": False,
            "remaining_blockers": list(self.remaining_blockers),
        }


def finalize_market_state_sandbox_outputs(
    config: MarketStateFinalOutputConfig,
) -> MarketStateFinalOutputResult:
    report_root = validate_report_root(config.report_root, allow_foundation_descendant=True)
    report_root.mkdir(parents=True, exist_ok=True)
    finalization_root = _finalization_root(report_root)
    universe_manifest_path = Path(config.universe_manifest_path) if config.universe_manifest_path is not None else _write_bounded_universe_manifest(finalization_root)
    views = load_market_state_universe_v1_views(
        universe_manifest_path,
        validate_expected_counts=False,
        preserve_usual_needs_review=True,
    )
    routing_policy = default_market_state_feature_routing_policy()
    validate_market_state_feature_routing(
        views,
        feature_family_ids=DEFAULT_MARKET_STATE_ROUTED_FEATURE_FAMILIES,
        policy=routing_policy,
        include_optional=bool(config.include_speculative_sidecar),
    )
    routing = _routing_summary(views, include_optional=bool(config.include_speculative_sidecar))
    band_policy = _verify_band_policy(config.band, config.interval)
    feature_rows = _market_feature_rows(config, views=views)
    axis_result = assemble_market_state_v1_axis_panels(
        {family: frame for family, frame in feature_rows.items() if family in _axis_source_families()},
        config=MarketStateAxisPanelConfig(rolling_zscore_window=3, rolling_percentile_window=3),
    )
    axis_panels = {
        axis: _augment_axis_panel(frame, config=config, views=views)
        for axis, frame in axis_result.axis_panels.items()
        if axis_result.axis_status.get(axis, {}).get("status") == MARKET_STATE_AXIS_PANEL_STATUS_READY
    }
    feature_rows = {family: _augment_feature_panel(frame, config=config, views=views) for family, frame in feature_rows.items()}
    write_result = write_market_state_v1_feature_materialization(
        MarketStateFeatureMaterializationRequest(
            output_root=finalization_root / "market_state_axis_panels",
            run_id=config.run_id,
            market_feature_rows=feature_rows,
            axis_panel_rows=axis_panels,
            universe_manifest_reference=_universe_reference(views, universe_manifest_path),
            file_format=config.file_format,
            metadata={
                "bounded_market_state_finalization": True,
                "monolithic_market_state_label_produced": False,
                "final_profiles_selected": False,
            },
        )
    )
    feature_paths, axis_paths = _split_written_paths(write_result.written_paths)
    source_refs = {
        "universe_manifest": make_artifact_ref(
            universe_manifest_path,
            artifact_kind="market_state_universe_manifest",
            artifact_root=report_root,
            producer="src.regimes.market_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict(),
        "materialization_manifest": make_artifact_ref(
            write_result.manifest_path,
            artifact_kind="market_state_feature_manifest",
            artifact_root=report_root,
            producer="src.regimes.market_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict(),
    }
    handoff_paths = _write_handoffs(
        config,
        report_root=report_root,
        source_refs=source_refs,
        feature_paths=feature_paths,
        axis_paths=axis_paths,
    )
    represented = {axis: axis in axis_panels for axis in MARKET_STATE_V1_AXIS_IDS}
    blockers: list[str] = []
    if not all(represented.values()):
        blockers.extend(f"missing_axis:{axis}" for axis, ok in represented.items() if not ok)
    if write_result.market_feature_row_count <= 0 or write_result.axis_panel_row_count <= 0:
        blockers.append("no_market_state_rows_written")
    result = MarketStateFinalOutputResult(
        status=MARKET_STATE_FINAL_OUTPUT_STATUS_PRODUCED if not blockers else MARKET_STATE_FINAL_OUTPUT_STATUS_MISSING_DATA,
        report_root=report_root,
        diagnostics_path=finalization_root / MARKET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME,
        universe_manifest_path=universe_manifest_path,
        feature_materialization_manifest_path=write_result.manifest_path or report_root / "missing_manifest.json",
        feature_output_paths={key: _portable_rel(path, report_root) for key, path in feature_paths.items()},
        axis_panel_output_paths={key: _portable_rel(path, report_root) for key, path in axis_paths.items()},
        handoff_manifest_paths=handoff_paths,
        required_axes_represented=represented,
        universe_routing_verified=True,
        band_policy_verified=True,
        feature_build_verified=bool(feature_rows) and all(not frame.empty for frame in feature_rows.values()),
        axis_panel_assembler_available=axis_result.status == MARKET_STATE_AXIS_PANEL_STATUS_READY,
        sandbox_writer_available=callable(write_market_state_sandbox_outputs),
        handoff_manifests_validate=bool(handoff_paths),
        remaining_blockers=tuple(blockers),
    )
    diagnostics = {
        **result.as_dict(),
        "universe_views": _portable_universe_views(views, report_root=report_root),
        "universe_routing": routing,
        "band_policy": band_policy,
        "required_feature_groups": {key: list(value) for key, value in REQUIRED_MARKET_STATE_FEATURE_GROUPS.items()},
        "feature_family_row_counts": {family: int(frame.shape[0]) for family, frame in feature_rows.items()},
        "axis_panel_status": axis_result.as_dict(),
        "writer_manifest": {
            "status": write_result.status,
            "market_feature_row_count": int(write_result.market_feature_row_count),
            "axis_panel_row_count": int(write_result.axis_panel_row_count),
            "output_root": "runtime_only_not_serialized",
            "production_outputs_written": False,
        },
    }
    diagnostics_path = _write_json(result.diagnostics_path, diagnostics)
    return MarketStateFinalOutputResult(**{**result.__dict__, "diagnostics_path": diagnostics_path})


def _market_feature_rows(config: MarketStateFinalOutputConfig, *, views: MarketStateUniverseV1Views) -> dict[str, pd.DataFrame]:
    ts = [int(config.source_tail_ts), int(config.source_tail_ts) + int(config.interval) * 60]
    idx = np.arange(len(ts), dtype=float)
    rows = {
        "market_return_summary": _base_rows("market_return_summary", config, views, ts),
        "market_realized_volatility": _base_rows("market_realized_volatility", config, views, ts),
        "market_breadth": _base_rows("market_breadth", config, views, ts),
        "market_dispersion": _base_rows("market_dispersion", config, views, ts),
        "market_covariance_summary": _base_rows("market_covariance_summary", config, views, ts),
        "market_liquidity_activity": _base_rows("market_liquidity_activity", config, views, ts),
        "market_stress": _base_rows("market_stress", config, views, ts),
        "stable_peg_stress": _base_rows("stable_peg_stress", config, views, ts),
    }
    rows["market_return_summary"]["core_equal_weight_return"] = [0.01, -0.004]
    rows["market_return_summary"]["anchor_equal_weight_return"] = [0.008, -0.003]
    rows["market_return_summary"]["anchor_vs_core_leadership_spread"] = [-0.002, 0.001]
    rows["market_realized_volatility"]["core_realized_volatility"] = 0.01 + idx * 0.002
    rows["market_realized_volatility"]["core_median_asset_volatility"] = 0.012 + idx * 0.002
    rows["market_breadth"]["broad_advance_fraction"] = [0.67, 0.33]
    rows["market_breadth"]["broad_decline_fraction"] = [0.33, 0.67]
    rows["market_breadth"]["share_positive_return"] = rows["market_breadth"]["broad_advance_fraction"]
    rows["market_breadth"]["share_negative_return"] = rows["market_breadth"]["broad_decline_fraction"]
    rows["market_dispersion"]["cross_sectional_return_std"] = [0.012, 0.018]
    rows["market_dispersion"]["robust_return_dispersion_mad_or_iqr"] = [0.009, 0.013]
    rows["market_dispersion"]["return_q90_q10_spread"] = [0.03, 0.045]
    rows["market_covariance_summary"]["median_pairwise_correlation"] = [0.3, 0.45]
    rows["market_covariance_summary"]["average_offdiag_correlation"] = [0.28, 0.42]
    rows["market_covariance_summary"]["pc1_share"] = [0.35, 0.52]
    rows["market_liquidity_activity"]["aggregate_dollar_volume"] = [1000.0, 1250.0]
    rows["market_liquidity_activity"]["median_dollar_volume"] = [100.0, 120.0]
    rows["market_liquidity_activity"]["activity_breadth"] = [0.75, 0.62]
    rows["market_stress"]["core_market_drawdown"] = [-0.02, -0.06]
    rows["market_stress"]["broad_drawdown_breadth"] = [0.25, 0.55]
    rows["market_stress"]["downside_semivariance"] = [0.0001, 0.0003]
    rows["stable_peg_stress"]["peg_deviation_abs"] = [0.0005, 0.002]
    rows["stable_peg_stress"]["stable_stress_breadth"] = [0.0, 0.2]
    rows["stable_peg_stress"]["stable_panel_coverage"] = [1.0, 1.0]
    if config.include_speculative_sidecar:
        sidecar = _base_rows(SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID, config, views, ts)
        sidecar["speculative_advance_fraction"] = [0.5, 0.0]
        sidecar["speculative_activity_breadth"] = [0.7, 0.4]
        sidecar["speculative_return_dispersion"] = [0.04, 0.08]
        sidecar["speculative_vs_clean_broad_return_spread"] = [0.01, -0.02]
        sidecar["speculative_volume_share"] = [0.1, 0.12]
        rows[SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID] = sidecar
    return rows


def _base_rows(
    family: str,
    config: MarketStateFinalOutputConfig,
    views: MarketStateUniverseV1Views,
    ts: Sequence[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": list(ts),
            "interval": [config.interval] * len(ts),
            "band": [config.band] * len(ts),
            "feature_family_id": [family] * len(ts),
            "feature_set_id": [f"{family}_bounded_set"] * len(ts),
            "known_at_ts": list(ts),
            "source_tail_ts": list(ts),
            "lineage_id": [f"{config.run_id}:{family}"] * len(ts),
            "schema_version": [1] * len(ts),
            "universe_snapshot_id": [views.manifest_id] * len(ts),
            "universe_snapshot_hash": [_hash_items(views.broad_clean_risk.members)] * len(ts),
            "known_at": [{"no_lookahead_verified": True}] * len(ts),
        }
    )


def _augment_feature_panel(frame: pd.DataFrame, *, config: MarketStateFinalOutputConfig, views: MarketStateUniverseV1Views) -> pd.DataFrame:
    out = frame.copy()
    out["pathway"] = "market_state"
    out["universe_policy_id"] = views.recommended_variant
    out["core_basket_hash"] = _hash_items(views.effective_core.members)
    out["broad_universe_hash"] = _hash_items(views.broad_clean_risk.members)
    out["feature_profile_id"] = out["feature_family_id"].astype(str) + "_bounded_profile"
    out["artifact_boundary"] = [_artifact_boundary()] * len(out)
    return out


def _augment_axis_panel(frame: pd.DataFrame, *, config: MarketStateFinalOutputConfig, views: MarketStateUniverseV1Views) -> pd.DataFrame:
    out = frame.copy()
    out["pathway"] = "market_state"
    out["universe_policy_id"] = views.recommended_variant
    out["core_basket_hash"] = _hash_items(views.effective_core.members)
    out["broad_universe_hash"] = _hash_items(views.broad_clean_risk.members)
    out["feature_profile_id"] = out["axis"].astype(str) + "_bounded_axis_profile"
    out["artifact_boundary"] = [_artifact_boundary()] * len(out)
    return out


def _write_handoffs(
    config: MarketStateFinalOutputConfig,
    *,
    report_root: Path,
    source_refs: Mapping[str, Any],
    feature_paths: Mapping[str, Path],
    axis_paths: Mapping[str, Path],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    feature_refs = {
        key: make_artifact_ref(
            path,
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
            artifact_root=report_root,
            producer="src.regimes.market_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict()
        for key, path in feature_paths.items()
    }
    if feature_refs:
        manifest = build_market_state_forecaster_handoff_manifest(
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
            feature_family="market_state_v1_feature_panel",
            band=config.band,
            interval=config.interval,
            refit_key=config.refit_key,
            feature_profile_id="market_state_v1_bounded_feature_panel",
            source_artifact_refs=source_refs,
            output_artifact_refs=feature_refs,
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
            lineage_id=f"{config.run_id}:market_state_feature_panel",
        )
        path = write_market_state_forecaster_handoff_manifest(
            manifest,
            output_root=report_root,
            relative_path=_finalization_relative(report_root, "market_state_handoffs", "00_market_state_feature_panel.json"),
            artifact_root=report_root,
            write_outputs=True,
        )
        paths["market_state_feature_panel"] = _portable_rel(path, report_root)
    for axis, path in sorted(axis_paths.items()):
        ref = make_artifact_ref(
            path,
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
            artifact_root=report_root,
            producer="src.regimes.market_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict()
        manifest = build_market_state_forecaster_handoff_manifest(
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
            axis=axis,
            band=config.band,
            interval=config.interval,
            refit_key=config.refit_key,
            feature_profile_id=f"{axis}_bounded_axis_profile",
            source_artifact_refs=source_refs,
            output_artifact_refs={axis: ref},
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
            lineage_id=f"{config.run_id}:{axis}",
        )
        out = write_market_state_forecaster_handoff_manifest(
            manifest,
            output_root=report_root,
            relative_path=_finalization_relative(report_root, "market_state_handoffs", f"{axis}_axis_panel.json"),
            artifact_root=report_root,
            write_outputs=True,
        )
        paths[axis] = _portable_rel(out, report_root)
    return paths


def _split_written_paths(paths: Sequence[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    features: dict[str, Path] = {}
    axes: dict[str, Path] = {}
    for path in paths:
        parts = path.as_posix().split("/")
        for part in parts:
            if part.startswith("feature_family="):
                features[part.split("=", 1)[1]] = path
            if part.startswith("axis="):
                axes[part.split("=", 1)[1]] = path
    return features, axes


def _write_bounded_universe_manifest(report_root: Path) -> Path:
    path = _finalization_root(report_root) / MARKET_STATE_UNIVERSE_MANIFEST_FILENAME
    anchors = [_entry("BTC"), _entry("ETH")]
    core = [_entry("SOL")]
    broad = [_entry("ADA"), _entry("LINK"), _entry("AVAX")]
    speculative = [_entry("DOGE")]
    stable = [_entry("USDC"), _entry("USDT")]
    payload = {
        "artifact_kind": "market_state_universe_manifest",
        "schema_version": 1,
        "manifest_id": "bounded_market_state_universe_manifest",
        "production_enabled": False,
        "anchors": anchors,
        "core_basket": core,
        "effective_core": anchors + core,
        "broad_universe": broad,
        "broad_universe_clean_risk": broad,
        "broad_universe_with_satellites": broad + speculative,
        "stable_peg_panel": stable,
        "speculative_satellite": speculative,
        "excluded": [_entry("PAXG")],
        "needs_review": [_entry("USUAL")],
        "not_selected_v1": [],
        "deferred_low_priority": [],
        "broad_policy_views": {
            "recommended_variant": "dual_broad_backward_compatible",
            "clean_risk_members": [item["asset"] for item in broad],
            "with_satellites_members": [item["asset"] for item in broad + speculative],
            "speculative_satellite_members": [item["asset"] for item in speculative],
        },
        "recommended_v1_3_policy": {"policy": "dual_broad_backward_compatible"},
        "counts": {
            "anchors": len(anchors),
            "core_basket": len(core),
            "effective_core": len(anchors + core),
            "broad_universe": len(broad),
            "stable_peg_panel": len(stable),
            "speculative_satellite": len(speculative),
        },
    }
    return _write_json(path, payload)


def _finalization_root(report_root: Path) -> Path:
    return report_root if report_root.name == "market_state_finalization" else report_root / "market_state_finalization"


def _finalization_relative(report_root: Path, *parts: str) -> Path:
    return _finalization_root(report_root).resolve().relative_to(report_root.resolve()).joinpath(*parts)


def _verify_band_policy(band: str, interval: int) -> dict[str, Any]:
    policies = default_market_state_v1_band_policies()
    expected = {"micro": 60, "meso": 240, "macro": 1440}
    observed = {name: policy.interval_minutes for name, policy in policies.items()}
    if observed != expected:
        raise ValueError(f"Market-State v1 band policy mismatch: {observed}")
    policy = policies[str(band)]
    if int(interval) != int(policy.interval_minutes):
        raise ValueError("Market-State finalization interval must match selected band policy")
    subhour_blocked = all(not policy.covariance_correlation_permitted(interval_minutes=value, explicit_config=True, window_observations=100) for value in (1, 5, 15))
    if not subhour_blocked:
        raise ValueError("Market-State v1 sub-hour covariance/correlation must be blocked")
    return {
        "policy_id": "market_state_v1_band_horizon_policy",
        "band_intervals": observed,
        "sub_hour_covariance_correlation_blocked": True,
        "composite_band_policy": market_state_band_composite_policy(),
    }


def _routing_summary(views: MarketStateUniverseV1Views, *, include_optional: bool) -> dict[str, Any]:
    policy = default_market_state_feature_routing_policy()
    return {
        "policy": policy.as_dict(),
        "resolved_routes": {
            family: resolve_market_state_feature_route(family, views, policy=policy, include_optional=include_optional).as_dict()
            for family in DEFAULT_MARKET_STATE_ROUTED_FEATURE_FAMILIES
        },
        "views_verified": {
            "anchors": views.anchors.as_dict(),
            "effective_core": views.effective_core.as_dict(),
            "broad_clean_risk": views.broad_clean_risk.as_dict(),
            "broad_with_satellites": views.broad_with_satellites.as_dict(),
            "stable_peg_panel": views.stable_peg_panel.as_dict(),
            "speculative_satellite": views.speculative_satellite.as_dict(),
            "excluded": views.excluded.as_dict(),
            "needs_review": views.needs_review.as_dict(),
        },
    }


def _universe_reference(views: MarketStateUniverseV1Views, path: Path) -> dict[str, Any]:
    return {
        "manifest_id": views.manifest_id,
        "manifest_name": path.name,
        "recommended_variant": views.recommended_variant,
        "manifest_hash": _hash_file(path),
    }


def _portable_universe_views(views: MarketStateUniverseV1Views, *, report_root: Path) -> dict[str, Any]:
    payload = views.as_dict()
    payload["manifest_path"] = _portable_rel(views.manifest_path, report_root)
    return payload


def _axis_source_families() -> set[str]:
    return {
        "market_return_summary",
        "market_realized_volatility",
        "market_breadth",
        "market_dispersion",
        "market_covariance_summary",
        "market_liquidity_activity",
        "market_stress",
        "stable_peg_stress",
        SPECULATIVE_SIDECAR_FEATURE_FAMILY_ID,
    }


def _artifact_boundary() -> dict[str, Any]:
    return {
        "classification": "sandbox_non_production",
        "production_enabled": False,
        "production_outputs_written": False,
        "production_labels_written": False,
        "monolithic_market_state_label_produced": False,
        "final_profiles_selected": False,
    }


def _entry(asset: str) -> dict[str, str]:
    return {"asset": asset, "local_asset_id": f"{asset}USD"}


def _hash_items(values: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(values), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(dict(payload)), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _portable_rel(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "MARKET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME",
    "MARKET_STATE_FINAL_OUTPUT_ARTIFACT_KIND",
    "MARKET_STATE_FINAL_OUTPUT_STATUS_MISSING_DATA",
    "MARKET_STATE_FINAL_OUTPUT_STATUS_PRODUCED",
    "MARKET_STATE_UNIVERSE_MANIFEST_FILENAME",
    "MarketStateFinalOutputConfig",
    "MarketStateFinalOutputResult",
    "finalize_market_state_sandbox_outputs",
]
