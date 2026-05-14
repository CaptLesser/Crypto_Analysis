from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.regimes.core.paths import default_foundation_report_root
from src.regimes.core import (
    PathwayDiagnosticsRootPolicy,
    read_json,
    require_pathway_diagnostics_root,
    write_market_aggregation_source_read_precondition,
    write_market_source_coverage_diagnostic,
    write_market_universe_membership_snapshot,
    write_pathway_dry_run_diagnostic,
    write_pathway_source_probe,
)
from src.regimes.core.market_state_contracts import (
    MarketStateMetadataManifest,
    build_market_state_metadata_manifest,
    default_market_state_feature_family_declarations,
    validate_market_state_metadata_report_root,
)
from src.regimes.pathways import (
    MARKET_STATE_PATHWAY,
    MarketStateConfig,
    SourceProbeRecord,
    load_market_universe_membership_input_file,
    market_aggregation_source_read_precondition,
    market_membership_snapshot_provenance,
    market_state_aggregation_source_probe,
    market_state_input_contract,
    market_state_run_manifest,
    market_state_scalar_partition_source_probe,
    market_universe_membership_snapshot,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Market-state Regime pathway scaffold.")
    parser.add_argument("--universe", type=str, default="global", help="Market-state universe key")
    parser.add_argument("--member-assets", type=str, default="", help="Comma-delimited universe member assets")
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Validated scaffold JSON file containing explicit universe member assets",
    )
    parser.add_argument("--bands", type=str, default="", help="Comma-delimited Regime bands")
    parser.add_argument("--timestamp-column", type=str, default="ts")
    parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--run-id", type=str, default="market_state_scaffold")
    parser.add_argument(
        "--membership-source",
        type=str,
        default="",
        help="Optional scaffold universe membership provenance for source-probe snapshot metadata",
    )
    parser.add_argument(
        "--snapshot-timestamp-utc",
        type=str,
        default="",
        help="Optional UTC timestamp to use for source-probe and universe snapshot metadata",
    )
    parser.add_argument(
        "--source-feature-root",
        type=Path,
        default=None,
        help=(
            "Optional scalar-feature source root for a read-only real partition probe. "
            "Requires explicit --member-assets or --universe-file."
        ),
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=None,
        help="Explicit report/sandbox diagnostics root for scaffold-only dry-run JSON output",
    )
    return parser


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _optional_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def membership_input_from_args(args: argparse.Namespace) -> dict | None:
    input_path = getattr(args, "universe_file", None)
    if input_path is None:
        return None
    return load_market_universe_membership_input_file(
        Path(input_path),
        expected_universe=str(getattr(args, "universe", "global") or "global"),
    )


def _member_assets_from_args(args: argparse.Namespace, membership_input: dict | None) -> tuple[str, ...]:
    cli_members = _csv(getattr(args, "member_assets", ""))
    if membership_input is None:
        return cli_members
    file_members = tuple(str(asset) for asset in membership_input["member_assets"])
    if cli_members and tuple(sorted(cli_members, key=str.lower)) != file_members:
        raise ValueError(
            "Market-state --member-assets must match --universe-file member_assets when both are supplied"
        )
    return file_members


def config_from_args(args: argparse.Namespace) -> MarketStateConfig:
    bands = _csv(getattr(args, "bands", ""))
    membership_input = membership_input_from_args(args)
    member_assets = _member_assets_from_args(args, membership_input)
    kwargs = {
        "universe": str(getattr(args, "universe", "global") or "global"),
        "member_assets": member_assets,
        "timestamp_column": str(getattr(args, "timestamp_column", "ts") or "ts"),
        "min_assets": int(getattr(args, "min_assets", 10) or 10),
    }
    if bands:
        kwargs["bands"] = bands
    return MarketStateConfig(**kwargs)


def run(
    config: MarketStateConfig,
    *,
    run_id: str = "market_state_scaffold",
    membership_source: str | None = None,
) -> dict:
    return market_state_run_manifest(
        config,
        run_id=run_id,
        membership_source=_membership_snapshot_source(membership_source),
    )


def metadata_manifest(
    *,
    manifest_id: str = "market_state_metadata_manifest",
    universe: str = "global",
    bands: Sequence[str] = ("micro",),
    report_root: str | Path = default_foundation_report_root("market_state_metadata"),
) -> MarketStateMetadataManifest:
    return build_market_state_metadata_manifest(
        manifest_id=manifest_id,
        universe=universe,
        bands=bands,
        report_root=report_root,
    )


def _synthetic_member_assets(config: MarketStateConfig) -> tuple[str, ...]:
    return config.member_assets or tuple(
        f"SYNTH_MEMBER_{idx:02d}" for idx in range(max(int(config.min_assets), 1))
    )


def synthetic_aggregation_frame(config: MarketStateConfig, *, band: str | None = None) -> pd.DataFrame:
    band_name = str(band or config.bands[0])
    member_assets = _synthetic_member_assets(config)
    contract = market_state_input_contract(config, band=band_name, member_assets=member_assets)
    ceiling = int(contract.band_contract.ceiling_interval_min)
    frame = pd.DataFrame(
        {
            config.timestamp_column: [100, 100 + ceiling * 60],
            "universe": [config.universe, config.universe],
            "band": [band_name, band_name],
            "ceiling_interval_min": [ceiling, ceiling],
            "member_asset_count": [len(member_assets), len(member_assets)],
            "contributing_asset_count": [len(member_assets), int(config.min_assets)],
            "coverage_pct": [1.0, float(config.min_assets) / float(len(member_assets))],
        }
    )
    for group, columns in contract.candidate_columns_by_group.items():
        frame[str(columns[0])] = [1.0, 0.5]
    return frame


def synthetic_validation_results(config: MarketStateConfig) -> tuple[dict, ...]:
    band = str(config.bands[0])
    member_assets = _synthetic_member_assets(config)
    contract = market_state_input_contract(config, band=band, member_assets=member_assets)
    frame = synthetic_aggregation_frame(config, band=band)
    return (contract.validate_frame(frame).as_dict(),)


def source_probe(
    config: MarketStateConfig,
    *,
    run_id: str = "market_state_source_probe",
    membership_source: str | None = None,
    snapshot_timestamp_utc: str | None = None,
) -> SourceProbeRecord:
    band = str(config.bands[0])
    frame = synthetic_aggregation_frame(config, band=band)
    return market_state_aggregation_source_probe(
        config,
        frame,
        band=band,
        run_id=str(run_id),
        created_at_utc=_optional_text(snapshot_timestamp_utc),
        membership_source=_optional_text(membership_source),
    )


def scalar_partition_source_probe(
    config: MarketStateConfig,
    *,
    source_feature_root: Path,
    run_id: str = "market_state_scalar_partition_probe",
    membership_source: str | None = None,
    snapshot_timestamp_utc: str | None = None,
) -> SourceProbeRecord:
    return market_state_scalar_partition_source_probe(
        config,
        source_feature_root=Path(source_feature_root),
        band=str(config.bands[0]),
        run_id=str(run_id),
        created_at_utc=_optional_text(snapshot_timestamp_utc),
        membership_source=_membership_snapshot_source(membership_source),
    )


def _membership_snapshot_source(membership_source: str | None) -> str:
    return _optional_text(membership_source) or "cli.explicit_members"


def membership_source_from_args(args: argparse.Namespace, membership_input: dict | None) -> str | None:
    explicit_source = _optional_text(getattr(args, "membership_source", None))
    if membership_input is None:
        return explicit_source
    if explicit_source and explicit_source != "cli.universe_file":
        raise ValueError(
            "Market-state --membership-source must be 'cli.universe_file' when --universe-file is supplied"
        )
    return "cli.universe_file"


def _membership_source_detail(membership_input: dict | None) -> str:
    if membership_input is None:
        return "market_state_cli_explicit_membership"
    return str(membership_input.get("source_path") or "market_state_cli_universe_file")


def _membership_extra_provenance(membership_input: dict | None) -> dict:
    provenance = {"source": "market_state_cli"}
    if membership_input is not None:
        provenance["universe_file_path"] = str(membership_input.get("source_path"))
        provenance["universe_file_created_at_utc"] = str(membership_input.get("created_at_utc"))
        provenance["universe_file_artifact_kind"] = str(membership_input.get("artifact_kind"))
    return provenance


def _membership_snapshot_provenance(
    config: MarketStateConfig,
    *,
    run_id: str,
    membership_source: str | None = None,
    source_feature_root: Path | None = None,
    membership_input: dict | None = None,
) -> dict:
    return market_membership_snapshot_provenance(
        config,
        run_id=str(run_id),
        membership_source=_membership_snapshot_source(membership_source),
        source_detail=_membership_source_detail(membership_input),
        source_feature_root=source_feature_root,
        extra_provenance=_membership_extra_provenance(membership_input),
    )


def write_membership_snapshot(
    config: MarketStateConfig,
    *,
    diagnostics_root: Path,
    run_id: str = "market_state_scaffold",
    membership_source: str | None = None,
    snapshot_timestamp_utc: str | None = None,
    source_feature_root: Path | None = None,
    membership_input: dict | None = None,
) -> Path:
    if not config.member_assets:
        raise ValueError("Market-state membership snapshot requires explicit member_assets")
    return write_market_universe_membership_snapshot(
        Path(diagnostics_root),
        universe=config.universe,
        run_id=str(run_id),
        member_assets=config.member_assets,
        membership_source=_membership_snapshot_source(membership_source),
        provenance=_membership_snapshot_provenance(
            config,
            run_id=str(run_id),
            membership_source=membership_source,
            source_feature_root=source_feature_root,
            membership_input=membership_input,
        ),
        snapshot_timestamp_utc=_optional_text(snapshot_timestamp_utc),
        snapshot_scope=(
            "market_state_cli_universe_file"
            if membership_input is not None
            else "market_state_cli_explicit_membership"
        ),
        min_assets=config.min_assets,
    )


def write_dry_run_diagnostic(
    config: MarketStateConfig,
    *,
    diagnostics_root: Path,
    run_id: str = "market_state_scaffold",
    manifest: dict | None = None,
) -> Path:
    payload = manifest or run(config, run_id=run_id)
    config_summary = dict(payload["config"])
    if "lifecycle_policy" in payload:
        config_summary["lifecycle_policy"] = dict(payload["lifecycle_policy"])
    return write_pathway_dry_run_diagnostic(
        Path(diagnostics_root),
        pathway=MARKET_STATE_PATHWAY,
        run_id=str(run_id),
        config_summary=config_summary,
        input_frame_contract=payload["input_frame_contract"],
        artifact_boundary=payload["artifact_boundary"],
        validation_results=synthetic_validation_results(config),
    )


def write_source_probe_diagnostic(
    config: MarketStateConfig,
    *,
    diagnostics_root: Path,
    run_id: str = "market_state_scaffold",
    probe: SourceProbeRecord | None = None,
    membership_source: str | None = None,
    snapshot_timestamp_utc: str | None = None,
) -> Path:
    payload = probe or source_probe(
        config,
        run_id=run_id,
        membership_source=membership_source,
        snapshot_timestamp_utc=snapshot_timestamp_utc,
    )
    record = payload.as_dict()
    return write_pathway_source_probe(
        Path(diagnostics_root),
        pathway=MARKET_STATE_PATHWAY,
        run_id=str(run_id),
        source_summary=record["source_summary"],
        input_validation=record["input_validation"],
        artifact_boundary=record["artifact_boundary"],
        created_at_utc=record["created_at_utc"],
    )


def write_source_coverage_diagnostic(
    config: MarketStateConfig,
    *,
    diagnostics_root: Path,
    membership_snapshot_path: Path,
    run_id: str = "market_state_scaffold",
    membership_source: str | None = None,
    scalar_source_probe: SourceProbeRecord | None = None,
    scalar_source_probe_path: Path | None = None,
    snapshot_timestamp_utc: str | None = None,
) -> Path:
    if not config.member_assets:
        raise ValueError("Market-state source coverage diagnostic requires explicit member_assets")
    source_probe_payload = scalar_source_probe.as_dict() if scalar_source_probe is not None else None
    return write_market_source_coverage_diagnostic(
        Path(diagnostics_root),
        run_id=str(run_id),
        universe=config.universe,
        band=str(config.bands[0]),
        member_assets=config.member_assets,
        min_assets=config.min_assets,
        membership_source=_membership_snapshot_source(membership_source),
        membership_snapshot=read_json(Path(membership_snapshot_path)),
        membership_snapshot_path=Path(membership_snapshot_path),
        source_probe=source_probe_payload,
        source_probe_path=Path(scalar_source_probe_path) if source_probe_payload is not None else None,
        created_at_utc=_optional_text(snapshot_timestamp_utc),
    )


def source_read_precondition(
    config: MarketStateConfig,
    *,
    source_probe: SourceProbeRecord,
    run_id: str = "market_state_scaffold",
    membership_source: str | None = None,
    source_probe_path: Path | None = None,
    membership_snapshot: dict | None = None,
    membership_snapshot_path: Path | None = None,
    source_coverage_diagnostic: dict | None = None,
    source_coverage_diagnostic_path: Path | None = None,
    snapshot_timestamp_utc: str | None = None,
) -> dict:
    if not config.member_assets:
        raise ValueError("Market-state source-read precondition requires explicit member_assets")
    return market_aggregation_source_read_precondition(
        config,
        run_id=str(run_id),
        membership_source=_membership_snapshot_source(membership_source),
        source_probe=source_probe,
        source_probe_path=source_probe_path,
        membership_snapshot=membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        source_coverage_diagnostic=source_coverage_diagnostic,
        source_coverage_diagnostic_path=source_coverage_diagnostic_path,
        band=str(config.bands[0]),
        created_at_utc=_optional_text(snapshot_timestamp_utc),
    )


def write_source_read_precondition(
    config: MarketStateConfig,
    *,
    diagnostics_root: Path,
    source_probe: SourceProbeRecord,
    source_probe_path: Path,
    membership_snapshot_path: Path,
    source_coverage_diagnostic: dict | None = None,
    source_coverage_diagnostic_path: Path | None = None,
    run_id: str = "market_state_scaffold",
    membership_source: str | None = None,
    snapshot_timestamp_utc: str | None = None,
) -> Path:
    if not config.member_assets:
        raise ValueError("Market-state source-read precondition requires explicit member_assets")
    return write_market_aggregation_source_read_precondition(
        Path(diagnostics_root),
        run_id=str(run_id),
        universe=config.universe,
        band=str(config.bands[0]),
        member_assets=config.member_assets,
        min_assets=config.min_assets,
        membership_source=_membership_snapshot_source(membership_source),
        source_probe=source_probe.as_dict(),
        source_probe_path=Path(source_probe_path),
        membership_snapshot=read_json(Path(membership_snapshot_path)),
        membership_snapshot_path=Path(membership_snapshot_path),
        source_coverage_diagnostic=source_coverage_diagnostic,
        source_coverage_diagnostic_path=source_coverage_diagnostic_path,
        created_at_utc=_optional_text(snapshot_timestamp_utc),
    )


def diagnostics_root_policy(diagnostics_root: Path) -> PathwayDiagnosticsRootPolicy:
    return require_pathway_diagnostics_root(Path(diagnostics_root), for_source_probe=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    membership_input = membership_input_from_args(args)
    membership_source = membership_source_from_args(args, membership_input)
    config = config_from_args(args)
    manifest = run(config, run_id=str(args.run_id), membership_source=membership_source)
    if membership_input is not None:
        manifest["membership_input"] = membership_input
    policy: PathwayDiagnosticsRootPolicy | None = None
    if args.diagnostics_root is not None:
        policy = diagnostics_root_policy(Path(args.diagnostics_root))
    source_feature_root = getattr(args, "source_feature_root", None)
    probe: SourceProbeRecord | None = None
    if source_feature_root is not None:
        probe = scalar_partition_source_probe(
            config,
            source_feature_root=Path(source_feature_root),
            run_id=str(args.run_id),
            membership_source=membership_source,
            snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
        )
        manifest["source_probe"] = probe.as_dict()
        manifest["source_read_precondition"] = source_read_precondition(
            config,
            source_probe=probe,
            run_id=str(args.run_id),
            membership_source=membership_source,
            membership_snapshot=market_universe_membership_snapshot(
                config,
                membership_source=_membership_snapshot_source(membership_source),
                provenance=_membership_snapshot_provenance(
                    config,
                    run_id=str(args.run_id),
                    membership_source=membership_source,
                    source_feature_root=Path(source_feature_root),
                    membership_input=membership_input,
                ),
                snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
                snapshot_scope=(
                    "market_state_cli_universe_file"
                    if membership_input is not None
                    else "market_state_cli_explicit_membership"
                ),
            ),
            snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
        )
    if args.diagnostics_root is not None:
        assert policy is not None
        manifest["diagnostics_root_policy"] = policy.as_dict()
        path = write_dry_run_diagnostic(
            config,
            diagnostics_root=Path(args.diagnostics_root),
            run_id=str(args.run_id),
            manifest=manifest,
        )
        manifest["dry_run_diagnostic_path"] = str(path)
        source_probe_path = write_source_probe_diagnostic(
            config,
            diagnostics_root=Path(args.diagnostics_root),
            run_id=str(args.run_id),
            probe=probe,
            membership_source=membership_source,
            snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
        )
        manifest["source_probe_diagnostic_path"] = str(source_probe_path)
        if config.member_assets:
            membership_snapshot_path = write_membership_snapshot(
                config,
                diagnostics_root=Path(args.diagnostics_root),
                run_id=str(args.run_id),
                membership_source=membership_source,
                snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
                source_feature_root=Path(source_feature_root) if source_feature_root is not None else None,
                membership_input=membership_input,
            )
            manifest["membership_snapshot_path"] = str(membership_snapshot_path)
            coverage_diagnostic_path = write_source_coverage_diagnostic(
                config,
                diagnostics_root=Path(args.diagnostics_root),
                membership_snapshot_path=membership_snapshot_path,
                run_id=str(args.run_id),
                membership_source=membership_source,
                scalar_source_probe=probe,
                scalar_source_probe_path=source_probe_path if probe is not None else None,
                snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
            )
            manifest["market_source_coverage_diagnostic_path"] = str(coverage_diagnostic_path)
            if probe is not None:
                coverage_diagnostic_payload = read_json(coverage_diagnostic_path)
                source_read_precondition_path = write_source_read_precondition(
                    config,
                    diagnostics_root=Path(args.diagnostics_root),
                    source_probe=probe,
                    source_probe_path=source_probe_path,
                    membership_snapshot_path=membership_snapshot_path,
                    source_coverage_diagnostic=coverage_diagnostic_payload,
                    source_coverage_diagnostic_path=coverage_diagnostic_path,
                    run_id=str(args.run_id),
                    membership_source=membership_source,
                    snapshot_timestamp_utc=_optional_text(args.snapshot_timestamp_utc),
                )
                manifest["source_read_precondition_path"] = str(source_read_precondition_path)
                manifest["source_read_precondition"] = read_json(source_read_precondition_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
