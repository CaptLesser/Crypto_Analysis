from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.regimes.core import write_pathway_dry_run_diagnostic
from src.regimes.core.relative_state_contracts import (
    RelativeStateMetadataManifest,
    build_relative_state_metadata_manifest,
    default_relative_feature_family_declarations,
    validate_relative_state_metadata_report_root,
)
from src.regimes.pathways import (
    RELATIVE_STATE_PATHWAY,
    RelativeStateConfig,
    relative_state_input_contract,
    relative_state_run_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relative-state Regime pathway scaffold.")
    parser.add_argument("--universe", type=str, default="global", help="Relative-state universe key")
    parser.add_argument("--benchmark", type=str, default="benchmark", help="Benchmark or basket key")
    parser.add_argument("--assets", type=str, default="", help="Comma-delimited asset subset")
    parser.add_argument("--peer-assets", type=str, default="", help="Comma-delimited peer basket assets")
    parser.add_argument("--bands", type=str, default="", help="Comma-delimited Regime bands")
    parser.add_argument("--timestamp-column", type=str, default="ts")
    parser.add_argument("--min-peer-assets", type=int, default=5)
    parser.add_argument(
        "--missing-benchmark-policy",
        type=str,
        default="require",
        choices=("require", "drop_timestamp", "carry_forward", "use_universe_proxy"),
    )
    parser.add_argument("--run-id", type=str, default="relative_state_scaffold")
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=None,
        help="Explicit report/sandbox diagnostics root for scaffold-only dry-run JSON output",
    )
    return parser


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def config_from_args(args: argparse.Namespace) -> RelativeStateConfig:
    bands = _csv(getattr(args, "bands", ""))
    assets = _csv(getattr(args, "assets", ""))
    peer_assets = _csv(getattr(args, "peer_assets", ""))
    kwargs = {
        "universe": str(getattr(args, "universe", "global") or "global"),
        "benchmark": str(getattr(args, "benchmark", "benchmark") or "benchmark"),
        "assets": assets,
        "peer_assets": peer_assets,
        "timestamp_column": str(getattr(args, "timestamp_column", "ts") or "ts"),
        "min_peer_assets": int(getattr(args, "min_peer_assets", 5) or 5),
        "missing_benchmark_policy": str(getattr(args, "missing_benchmark_policy", "require") or "require"),
    }
    if bands:
        kwargs["bands"] = bands
    return RelativeStateConfig(**kwargs)


def run(config: RelativeStateConfig, *, run_id: str = "relative_state_scaffold") -> dict:
    return relative_state_run_manifest(config, run_id=run_id)


def metadata_manifest(
    *,
    manifest_id: str = "relative_state_metadata_manifest",
    primary_asset: str = "ETHUSD",
    report_root: str | Path = "reports/regimes/foundation/relative_state_metadata",
) -> RelativeStateMetadataManifest:
    return build_relative_state_metadata_manifest(
        manifest_id=manifest_id,
        primary_asset=primary_asset,
        report_root=report_root,
    )


def synthetic_validation_results(config: RelativeStateConfig) -> tuple[dict, ...]:
    band = str(config.bands[0])
    asset = str(config.assets[0]) if config.assets else "SYNTH_ASSET"
    peer_assets = config.peer_assets or tuple(
        f"SYNTH_PEER_{idx:02d}" for idx in range(max(int(config.min_peer_assets), 1))
    )
    contract = relative_state_input_contract(config, asset=asset, band=band, peer_assets=peer_assets)
    ceiling = int(contract.band_contract.ceiling_interval_min)
    frame = pd.DataFrame(
        {
            config.timestamp_column: [100, 100 + ceiling * 60],
            "asset": [asset, asset],
            "universe": [config.universe, config.universe],
            "benchmark": [config.benchmark, config.benchmark],
            "band": [band, band],
            "ceiling_interval_min": [ceiling, ceiling],
            "peer_asset_count": [len(peer_assets), len(peer_assets)],
            "contributing_peer_asset_count": [len(peer_assets), int(config.min_peer_assets)],
            "benchmark_present": [True, True],
            "coverage_pct": [1.0, float(config.min_peer_assets) / float(len(peer_assets))],
        }
    )
    for group, columns in contract.candidate_columns_by_group.items():
        frame[str(columns[0])] = [1.0, 0.5]
    return (contract.validate_frame(frame).as_dict(),)


def write_dry_run_diagnostic(
    config: RelativeStateConfig,
    *,
    diagnostics_root: Path,
    run_id: str = "relative_state_scaffold",
    manifest: dict | None = None,
) -> Path:
    payload = manifest or run(config, run_id=run_id)
    config_summary = dict(payload["config"])
    for key in (
        "benchmark_source_policy",
        "peer_basket_lifecycle_policy",
        "peer_basket_source_policy",
        "source_read_precondition",
    ):
        if key in payload:
            config_summary[key] = dict(payload[key])
    return write_pathway_dry_run_diagnostic(
        Path(diagnostics_root),
        pathway=RELATIVE_STATE_PATHWAY,
        run_id=str(run_id),
        config_summary=config_summary,
        input_frame_contract=payload["input_frame_contract"],
        artifact_boundary=payload["artifact_boundary"],
        validation_results=synthetic_validation_results(config),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    manifest = run(config, run_id=str(args.run_id))
    if args.diagnostics_root is not None:
        path = write_dry_run_diagnostic(
            config,
            diagnostics_root=Path(args.diagnostics_root),
            run_id=str(args.run_id),
            manifest=manifest,
        )
        manifest["dry_run_diagnostic_path"] = str(path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
