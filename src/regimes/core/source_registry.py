from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.asset_state.dataset_builder import discover_scalar_feature_assets, discover_scalar_feature_intervals
from src.regimes.core.root_resolution import (
    READ_SOURCE_KINDS,
    SOURCE_KIND_MARKET_STATE_UNIVERSE,
    SOURCE_KIND_OHLCVT,
    SOURCE_KIND_REGIME_FEATURES,
    SOURCE_KIND_RELATIONSHIP_DISCOVERY,
    SOURCE_KIND_REPORT_OUTPUT_ROOT,
    SOURCE_KIND_SCALAR_FEATURES,
    SOURCE_KIND_UNIVERSE_ELIGIBILITY,
    WRITE_SOURCE_KINDS,
    resolve_regime_source_root_candidates,
    resolve_regime_write_root,
)
from src.regimes.core.serialization import to_jsonable


SOURCE_STATUS_FOUND = "found"
SOURCE_STATUS_MISSING = "missing"
SOURCE_STATUS_PARTIAL = "partial"
SOURCE_STATUS_FIXTURE_ONLY = "fixture_only"
SOURCE_STATUS_NOT_APPLICABLE = "not_applicable"

SOURCE_STATUSES: frozenset[str] = frozenset(
    {
        SOURCE_STATUS_FOUND,
        SOURCE_STATUS_MISSING,
        SOURCE_STATUS_PARTIAL,
        SOURCE_STATUS_FIXTURE_ONLY,
        SOURCE_STATUS_NOT_APPLICABLE,
    }
)


@dataclass(frozen=True)
class RegimeSourceRegistryConfig:
    explicit_roots: Mapping[str, Any] = field(default_factory=dict)
    cli_args: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] | str | Path | None = None
    runtime_config: Mapping[str, Any] | str | Path | None = None
    env: Mapping[str, str] | None = None
    profile: str | None = None
    project_root: str | Path | None = None
    output_root: str | Path | None = None
    assets: Sequence[str] = ()
    intervals: Sequence[int] = ()
    include_project_defaults: bool = True


@dataclass(frozen=True)
class RegimeSourceDiagnostic:
    source_kind: str
    status: str
    access_mode: str
    root: Path | None
    resolved_by: str
    exists: bool
    sampled_assets: Sequence[str] = ()
    sampled_intervals: Sequence[int] = ()
    missing_partitions: Sequence[str] = ()
    schema_availability: Mapping[str, Any] = field(default_factory=dict)
    candidates: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        if self.status not in SOURCE_STATUSES:
            raise ValueError(f"Unknown Regime source status: {self.status!r}")
        if self.access_mode not in {"read", "write"}:
            raise ValueError("Regime source diagnostic access_mode must be read or write")

    @property
    def found(self) -> bool:
        return self.status == SOURCE_STATUS_FOUND

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "status": self.status,
            "access_mode": self.access_mode,
            "root": "runtime_only_not_serialized" if self.root is not None else None,
            "root_name": self.root.name if self.root is not None else None,
            "resolved_by": self.resolved_by,
            "exists": bool(self.exists),
            "sampled_assets": list(self.sampled_assets),
            "sampled_intervals": [int(interval) for interval in self.sampled_intervals],
            "missing_partitions": list(self.missing_partitions),
            "schema_availability": to_jsonable(dict(self.schema_availability)),
            "candidates": to_jsonable(list(self.candidates)),
        }


@dataclass(frozen=True)
class RegimeSourceRegistry:
    diagnostics: Mapping[str, RegimeSourceDiagnostic]

    def source(self, source_kind: str) -> RegimeSourceDiagnostic:
        return self.diagnostics[str(source_kind)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "regime_source_registry",
            "schema_version": 1,
            "sources": {kind: diagnostic.as_dict() for kind, diagnostic in self.diagnostics.items()},
            "read_roots_may_point_to_project_data": True,
            "write_roots_sandbox_report_only": True,
        }


def build_regime_source_registry(config: RegimeSourceRegistryConfig | None = None) -> RegimeSourceRegistry:
    cfg = config or RegimeSourceRegistryConfig()
    diagnostics: dict[str, RegimeSourceDiagnostic] = {}
    for kind in sorted(READ_SOURCE_KINDS):
        diagnostics[kind] = resolve_regime_source(kind, cfg)
    diagnostics[SOURCE_KIND_REPORT_OUTPUT_ROOT] = resolve_regime_source(SOURCE_KIND_REPORT_OUTPUT_ROOT, cfg)
    return RegimeSourceRegistry(diagnostics=diagnostics)


def resolve_regime_source(source_kind: str, config: RegimeSourceRegistryConfig) -> RegimeSourceDiagnostic:
    if source_kind in WRITE_SOURCE_KINDS:
        root, resolved_by = resolve_regime_write_root(
            config.output_root or config.explicit_roots.get(source_kind),
            env=config.env,
            project_root=config.project_root,
            subdir=None,
        )
        return RegimeSourceDiagnostic(
            source_kind=source_kind,
            status=SOURCE_STATUS_FOUND,
            access_mode="write",
            root=root,
            resolved_by=resolved_by,
            exists=root.exists(),
            schema_availability={"write_root_policy": "reports/regimes/foundation descendant only"},
        )

    candidates = resolve_regime_source_root_candidates(
        source_kind,
        explicit_roots=config.explicit_roots,
        cli_args=config.cli_args,
        manifest=config.manifest,
        runtime_config=config.runtime_config,
        env=config.env,
        profile=config.profile,
        project_root=config.project_root,
        include_project_defaults=config.include_project_defaults,
    )
    candidate_diags = [_candidate_diag(path, source, source_kind=source_kind) for path, source in candidates]
    for path, source in candidates:
        diagnostic = _diagnose_read_root(
            source_kind,
            path,
            resolved_by=source,
            assets=tuple(config.assets),
            intervals=tuple(int(interval) for interval in config.intervals),
            candidates=candidate_diags,
        )
        if diagnostic.status in {SOURCE_STATUS_FOUND, SOURCE_STATUS_PARTIAL, SOURCE_STATUS_FIXTURE_ONLY}:
            return diagnostic
    if candidates:
        path, source = candidates[0]
        return _diagnose_read_root(
            source_kind,
            path,
            resolved_by=source,
            assets=tuple(config.assets),
            intervals=tuple(int(interval) for interval in config.intervals),
            candidates=candidate_diags,
        )
    return RegimeSourceDiagnostic(
        source_kind=source_kind,
        status=SOURCE_STATUS_MISSING,
        access_mode="read",
        root=None,
        resolved_by="unconfigured",
        exists=False,
        candidates=(),
    )


def _diagnose_read_root(
    source_kind: str,
    root: Path,
    *,
    resolved_by: str,
    assets: Sequence[str],
    intervals: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
) -> RegimeSourceDiagnostic:
    exists = root.exists()
    if not exists:
        return RegimeSourceDiagnostic(
            source_kind=source_kind,
            status=SOURCE_STATUS_MISSING,
            access_mode="read",
            root=root,
            resolved_by=resolved_by,
            exists=False,
            candidates=candidates,
        )
    sampled_intervals = _sample_intervals(source_kind, root, intervals)
    sampled_assets = _sample_assets(source_kind, root, assets, sampled_intervals)
    missing = _missing_partitions(source_kind, root, assets=assets, intervals=intervals)
    schema = _schema_availability(source_kind, root)
    status = _status(source_kind, root, sampled_intervals=sampled_intervals, sampled_assets=sampled_assets, schema=schema, missing=missing)
    return RegimeSourceDiagnostic(
        source_kind=source_kind,
        status=status,
        access_mode="read",
        root=root,
        resolved_by=resolved_by,
        exists=True,
        sampled_assets=sampled_assets,
        sampled_intervals=sampled_intervals,
        missing_partitions=missing,
        schema_availability=schema,
        candidates=candidates,
    )


def _sample_intervals(source_kind: str, root: Path, requested: Sequence[int]) -> tuple[int, ...]:
    if source_kind == SOURCE_KIND_OHLCVT:
        discovered = _interval_dirs(root, "ohlcvt_")
    elif source_kind == SOURCE_KIND_SCALAR_FEATURES:
        discovered = discover_scalar_feature_intervals(root)
    elif source_kind == SOURCE_KIND_REGIME_FEATURES:
        discovered = _interval_dirs(root, "regime_features_market_")
    elif source_kind == SOURCE_KIND_RELATIONSHIP_DISCOVERY:
        discovered = _partition_values(root, "interval")
    else:
        discovered = ()
    if requested:
        return tuple(interval for interval in requested if interval in set(discovered))
    return tuple(discovered[:8])


def _sample_assets(source_kind: str, root: Path, requested: Sequence[str], intervals: Sequence[int]) -> tuple[str, ...]:
    if requested:
        return tuple(str(asset) for asset in requested[:8])
    assets: set[str] = set()
    if source_kind == SOURCE_KIND_OHLCVT:
        for interval in intervals:
            assets.update(_asset_dirs(root / f"ohlcvt_{int(interval)}"))
    elif source_kind == SOURCE_KIND_SCALAR_FEATURES:
        for interval in intervals:
            assets.update(discover_scalar_feature_assets(root, int(interval)))
    elif source_kind == SOURCE_KIND_RELATIONSHIP_DISCOVERY:
        for path in root.rglob("*.jsonl"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if '"asset"' in text:
                assets.update(_rough_jsonl_assets(text))
                if len(assets) >= 8:
                    break
    return tuple(sorted(assets)[:8])


def _missing_partitions(source_kind: str, root: Path, *, assets: Sequence[str], intervals: Sequence[int]) -> tuple[str, ...]:
    if not assets or not intervals:
        return ()
    missing: list[str] = []
    for interval in intervals:
        for asset in assets:
            if source_kind == SOURCE_KIND_OHLCVT:
                path = root / f"ohlcvt_{int(interval)}" / f"asset={asset}"
            elif source_kind == SOURCE_KIND_SCALAR_FEATURES:
                path = root / f"scalar_features_{int(interval)}" / f"asset={asset}"
            else:
                continue
            if not path.exists():
                missing.append(f"interval={int(interval)}/asset={asset}")
    return tuple(missing)


def _schema_availability(source_kind: str, root: Path) -> dict[str, Any]:
    if source_kind == SOURCE_KIND_OHLCVT:
        return {"partition_layout": "ohlcvt_<interval>/asset=<asset>/year=<YYYY>/month=<MM>/part-000.parquet"}
    if source_kind == SOURCE_KIND_SCALAR_FEATURES:
        return {"partition_layout": "scalar_features_<interval>/asset=<asset>/year=<YYYY>/month=<MM>/part-000.parquet"}
    if source_kind == SOURCE_KIND_REGIME_FEATURES:
        return {"market_feature_dirs": [path.name for path in root.glob("regime_features_market_*") if path.is_dir()][:8]}
    if source_kind == SOURCE_KIND_MARKET_STATE_UNIVERSE:
        manifests = sorted(path.name for path in root.glob("market_state_universe_manifest*.json"))
        return {"manifest_files": manifests[:8], "schema_available": bool(manifests)}
    if source_kind == SOURCE_KIND_RELATIONSHIP_DISCOVERY:
        return {
            "handoff_manifest": (root / "process1_to_process2_handoff_manifest.json").is_file(),
            "relationship_discovery_dir": (root / "relationship_discovery").is_dir(),
            "cross_asset_features_dir": (root / "cross_asset_features").is_dir(),
        }
    if source_kind == SOURCE_KIND_UNIVERSE_ELIGIBILITY:
        snapshots = sorted(path.name for path in root.glob("*eligibility*.json"))
        return {"snapshot_files": snapshots[:8], "schema_available": bool(snapshots)}
    return {}


def _status(
    source_kind: str,
    root: Path,
    *,
    sampled_intervals: Sequence[int],
    sampled_assets: Sequence[str],
    schema: Mapping[str, Any],
    missing: Sequence[str],
) -> str:
    lowered_parts = {part.lower() for part in root.parts}
    if "fixture" in lowered_parts or "fixtures" in lowered_parts:
        return SOURCE_STATUS_FIXTURE_ONLY
    if source_kind in {SOURCE_KIND_MARKET_STATE_UNIVERSE, SOURCE_KIND_UNIVERSE_ELIGIBILITY}:
        return SOURCE_STATUS_FOUND if schema.get("schema_available") else SOURCE_STATUS_PARTIAL
    if source_kind == SOURCE_KIND_RELATIONSHIP_DISCOVERY:
        return SOURCE_STATUS_FOUND if schema.get("handoff_manifest") or schema.get("relationship_discovery_dir") else SOURCE_STATUS_PARTIAL
    if sampled_intervals and (sampled_assets or source_kind == SOURCE_KIND_REGIME_FEATURES) and not missing:
        return SOURCE_STATUS_FOUND
    if sampled_intervals or sampled_assets or schema:
        return SOURCE_STATUS_PARTIAL
    return SOURCE_STATUS_MISSING


def _candidate_diag(path: Path, source: str, *, source_kind: str) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "resolved_by": source,
        "exists": path.exists(),
    }


def _interval_dirs(root: Path, prefix: str) -> tuple[int, ...]:
    values: set[int] = set()
    if not root.exists() or not root.is_dir():
        return ()
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            raw = child.name.removeprefix(prefix)
            if raw.isdigit():
                values.add(int(raw))
    return tuple(sorted(values))


def _partition_values(root: Path, name: str) -> tuple[int, ...]:
    prefix = f"{name}="
    values: set[int] = set()
    if not root.exists():
        return ()
    for path in root.rglob(f"{prefix}*"):
        raw = path.name.removeprefix(prefix)
        if raw.isdigit():
            values.add(int(raw))
    return tuple(sorted(values))


def _asset_dirs(root: Path) -> tuple[str, ...]:
    if not root.exists() or not root.is_dir():
        return ()
    return tuple(sorted(child.name.split("=", 1)[1] for child in root.iterdir() if child.is_dir() and child.name.startswith("asset=")))


def _rough_jsonl_assets(text: str) -> set[str]:
    assets: set[str] = set()
    for line in text.splitlines()[:64]:
        marker = '"asset"'
        if marker not in line:
            continue
        try:
            after = line.split(marker, 1)[1].split(":", 1)[1].lstrip()
            if after.startswith('"'):
                assets.add(after.split('"', 2)[1])
        except Exception:
            continue
    return assets


__all__ = [
    "RegimeSourceDiagnostic",
    "RegimeSourceRegistry",
    "RegimeSourceRegistryConfig",
    "SOURCE_STATUS_FIXTURE_ONLY",
    "SOURCE_STATUS_FOUND",
    "SOURCE_STATUS_MISSING",
    "SOURCE_STATUS_NOT_APPLICABLE",
    "SOURCE_STATUS_PARTIAL",
    "build_regime_source_registry",
    "resolve_regime_source",
]
