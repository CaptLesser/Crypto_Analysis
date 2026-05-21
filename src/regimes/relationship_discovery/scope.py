from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.paths import resolve_project_path, resolve_project_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.universe_views import MarketStateUniverseV1Views, load_market_state_universe_v1_views
from src.regimes.regime_features.pathing import resolve_regime_feature_source_roots
from src.regimes.relationship_discovery.data_panel import (
    RELATIONSHIP_DATA_STATUS_INSUFFICIENT_ASSETS,
    RELATIONSHIP_DATA_STATUS_INSUFFICIENT_OVERLAP,
    RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA,
    RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED,
    RelationshipReturnPanelRequest,
    RelationshipReturnPanelResult,
    build_relationship_return_panel,
)


RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED = "real_data_loaded"
RELATIONSHIP_SCOPE_STATUS_MANIFEST_NOT_FOUND = "manifest_not_found"
RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_ASSETS = "insufficient_assets"
RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_OVERLAP = "insufficient_overlap"
RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA = "missing_source_data"
RELATIONSHIP_SCOPE_STATUS_FIXTURE_ONLY = "fixture_only"

RELATIONSHIP_SCOPE_STATUSES: tuple[str, ...] = (
    RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED,
    RELATIONSHIP_SCOPE_STATUS_MANIFEST_NOT_FOUND,
    RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_ASSETS,
    RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_OVERLAP,
    RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA,
    RELATIONSHIP_SCOPE_STATUS_FIXTURE_ONLY,
)

DEFAULT_MARKET_STATE_UNIVERSE_ROOT = Path("reports") / "regimes" / "foundation" / "market_state_universe"
DEFAULT_PRIMARY_INTERVAL = 240
DEFAULT_CONFIRMATION_INTERVAL = 1440
DEFAULT_EVIDENCE_INTERVAL = 60


@dataclass(frozen=True)
class RelationshipDiscoveryScopeRequest:
    manifest_path: str | Path | None = None
    market_state_universe_root: str | Path | None = None
    fixture_manifest: Mapping[str, Any] | None = None
    ohlcvt_root: str | Path | None = None
    scalar_feature_root: str | Path | None = None
    primary_interval: int = DEFAULT_PRIMARY_INTERVAL
    confirmation_interval: int = DEFAULT_CONFIRMATION_INTERVAL
    include_60m_evidence_probe: bool = False
    broad_sample_size: int = 40
    min_assets: int = 4
    min_overlap: int = 30
    start_ts: int | None = None
    end_ts: int | None = None
    include_excluded_assets: bool = False
    include_needs_review_assets: bool = False
    load_real_data_for_fixture: bool = False
    project_root: str | Path | None = None

    def __post_init__(self) -> None:
        for field_name in ("primary_interval", "confirmation_interval"):
            interval = int(getattr(self, field_name))
            if interval < 60:
                raise ValueError("Relationship Discovery scope does not support sub-hour relationship intervals")
            object.__setattr__(self, field_name, interval)
        if self.start_ts is not None and self.end_ts is not None and int(self.start_ts) >= int(self.end_ts):
            raise ValueError("Relationship Discovery scope start_ts must be before end_ts")
        object.__setattr__(self, "broad_sample_size", max(0, int(self.broad_sample_size)))
        object.__setattr__(self, "min_assets", max(1, int(self.min_assets)))
        object.__setattr__(self, "min_overlap", max(2, int(self.min_overlap)))
        object.__setattr__(self, "include_60m_evidence_probe", bool(self.include_60m_evidence_probe))
        object.__setattr__(self, "include_excluded_assets", bool(self.include_excluded_assets))
        object.__setattr__(self, "include_needs_review_assets", bool(self.include_needs_review_assets))
        object.__setattr__(self, "load_real_data_for_fixture", bool(self.load_real_data_for_fixture))

    def intervals_to_load(self) -> tuple[int, ...]:
        intervals = [int(self.primary_interval), int(self.confirmation_interval)]
        if self.include_60m_evidence_probe and DEFAULT_EVIDENCE_INTERVAL not in intervals:
            intervals.append(DEFAULT_EVIDENCE_INTERVAL)
        return tuple(dict.fromkeys(intervals))


@dataclass(frozen=True)
class RelationshipDiscoveryUniverse:
    manifest_path: Path | None
    manifest_id: str
    anchors: Sequence[str]
    core_assets: Sequence[str]
    broad_sample_assets: Sequence[str]
    selected_assets: Sequence[str]
    stable_peg_assets_excluded: Sequence[str]
    excluded_assets_blocked: Sequence[str]
    needs_review_assets_blocked: Sequence[str]
    selection_metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_discovery_universe_scope",
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "manifest_id": self.manifest_id,
            "anchors": list(self.anchors),
            "core_assets": list(self.core_assets),
            "broad_sample_assets": list(self.broad_sample_assets),
            "selected_assets": list(self.selected_assets),
            "stable_peg_assets_excluded": list(self.stable_peg_assets_excluded),
            "excluded_assets_blocked": list(self.excluded_assets_blocked),
            "needs_review_assets_blocked": list(self.needs_review_assets_blocked),
            "selection_metadata": to_jsonable(dict(self.selection_metadata)),
            "dynamic_peer_discovery_enabled": False,
            "peer_clusters_enabled": False,
            "production_writes_enabled": False,
        }


@dataclass
class RelationshipDiscoveryScopeResult:
    status: str
    universe: RelationshipDiscoveryUniverse | None = None
    manifest_path: Path | None = None
    panel_results: Mapping[int, RelationshipReturnPanelResult] = field(default_factory=dict)
    reason_codes: Sequence[str] = ()
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in RELATIONSHIP_SCOPE_STATUSES:
            raise ValueError(f"Unsupported Relationship Discovery scope status {status!r}")
        self.status = status
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))

    @property
    def usable(self) -> bool:
        return self.status == RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_discovery_scope_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "universe": self.universe.as_dict() if self.universe is not None else None,
            "panel_results": {str(interval): result.as_dict() for interval, result in sorted(self.panel_results.items())},
            "metadata": to_jsonable(dict(self.metadata)),
            "production_writes_enabled": False,
        }


def build_relationship_discovery_scope(request: RelationshipDiscoveryScopeRequest) -> RelationshipDiscoveryScopeResult:
    manifest_path, payload, from_fixture = _load_scope_manifest(request)
    if payload is None:
        return RelationshipDiscoveryScopeResult(
            status=RELATIONSHIP_SCOPE_STATUS_MANIFEST_NOT_FOUND,
            manifest_path=manifest_path,
            reason_codes=("manifest_not_found",),
            message="Market-State universe manifest was not found and no explicit fixture_manifest was supplied",
            metadata={"request": _request_metadata(request), "production_writes_enabled": False},
        )

    try:
        views = _views_from_payload_or_path(payload, manifest_path=manifest_path)
    except Exception as exc:
        return RelationshipDiscoveryScopeResult(
            status=RELATIONSHIP_SCOPE_STATUS_MANIFEST_NOT_FOUND,
            manifest_path=manifest_path,
            reason_codes=("manifest_unreadable",),
            message=str(exc),
            metadata={"request": _request_metadata(request), "production_writes_enabled": False},
        )

    universe = _build_universe(payload, views, manifest_path=manifest_path, request=request)
    if len(universe.selected_assets) < int(request.min_assets):
        return RelationshipDiscoveryScopeResult(
            status=RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_ASSETS,
            universe=universe,
            manifest_path=manifest_path,
            reason_codes=("insufficient_selected_assets",),
            message=f"selected asset count {len(universe.selected_assets)} < min_assets {request.min_assets}",
            metadata={"request": _request_metadata(request), "production_writes_enabled": False},
        )

    if from_fixture and not request.load_real_data_for_fixture:
        return RelationshipDiscoveryScopeResult(
            status=RELATIONSHIP_SCOPE_STATUS_FIXTURE_ONLY,
            universe=universe,
            manifest_path=manifest_path,
            reason_codes=("fixture_manifest_supplied",),
            message="scope was built from an explicit fixture manifest without real-data loading",
            metadata={"request": _request_metadata(request), "production_writes_enabled": False},
        )

    ohlcvt_root, scalar_root, root_metadata = _resolve_source_roots(request, payload)
    if ohlcvt_root is None:
        return RelationshipDiscoveryScopeResult(
            status=RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA,
            universe=universe,
            manifest_path=manifest_path,
            reason_codes=("missing_ohlcvt_root",),
            message="OHLCVT root could not be resolved from explicit args, manifest source_roots, or accepted environment conventions",
            metadata={"request": _request_metadata(request), "source_roots": root_metadata, "production_writes_enabled": False},
        )

    panels = _build_panels(request, universe=universe, ohlcvt_root=ohlcvt_root, scalar_feature_root=scalar_root)
    status, reason_codes, message = _scope_status_from_panels(request, panels)
    return RelationshipDiscoveryScopeResult(
        status=status,
        universe=universe,
        manifest_path=manifest_path,
        panel_results=panels,
        reason_codes=reason_codes,
        message=message,
        metadata={
            "request": _request_metadata(request),
            "source_roots": root_metadata,
            "required_intervals": [int(request.primary_interval), int(request.confirmation_interval)],
            "evidence_intervals": [DEFAULT_EVIDENCE_INTERVAL] if request.include_60m_evidence_probe else [],
            "production_writes_enabled": False,
        },
    )


def _load_scope_manifest(
    request: RelationshipDiscoveryScopeRequest,
) -> tuple[Path | None, Mapping[str, Any] | None, bool]:
    if request.fixture_manifest is not None:
        return None, dict(request.fixture_manifest), True
    path = _explicit_or_latest_manifest_path(request)
    if path is None or not path.exists() or not path.is_file():
        return path, None, False
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Relationship Discovery manifest JSON root must be an object")
    return path, payload, False


def _explicit_or_latest_manifest_path(request: RelationshipDiscoveryScopeRequest) -> Path | None:
    project_root = resolve_project_root(request.project_root)
    if request.manifest_path is not None and str(request.manifest_path).strip():
        return resolve_project_path(request.manifest_path, project_root=project_root)
    root = (
        resolve_project_path(request.market_state_universe_root, project_root=project_root)
        if request.market_state_universe_root is not None
        else project_root / DEFAULT_MARKET_STATE_UNIVERSE_ROOT
    )
    if not root.exists() or not root.is_dir():
        return None
    candidates = [
        path
        for path in root.glob("market_state_universe_manifest*.json")
        if path.is_file() and "semantic_policy" not in path.name and "comparison" not in path.name
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def _views_from_payload_or_path(payload: Mapping[str, Any], *, manifest_path: Path | None) -> MarketStateUniverseV1Views:
    if manifest_path is not None:
        return load_market_state_universe_v1_views(manifest_path, validate_expected_counts=False, preserve_usual_needs_review=False)
    temp_path = Path("__fixture_manifest_not_written__.json")
    return _views_from_payload(payload, manifest_path=temp_path)


def _views_from_payload(payload: Mapping[str, Any], *, manifest_path: Path) -> MarketStateUniverseV1Views:
    from src.regimes.market_state.universe_views import MarketStateUniverseView

    entry_index = _entry_index(payload)
    anchors = _members_from_entries(payload.get("anchors", ()))
    core = _members_from_entries(payload.get("core_basket", ()))
    effective_core = _members_from_any(payload.get("effective_core")) or tuple(dict.fromkeys((*anchors, *core)))
    broad_policy = payload.get("broad_policy_views") if isinstance(payload.get("broad_policy_views"), Mapping) else {}
    broad_clean = _members_from_any(broad_policy.get("clean_risk_members")) or _members_from_any(payload.get("broad_universe_clean_risk")) or _members_from_entries(
        payload.get("broad_universe", ())
    )
    stable = _members_from_entries(payload.get("stable_peg_panel", ()))
    excluded = _members_from_entries(payload.get("excluded", ()))
    needs_review = _members_from_entries(payload.get("needs_review", ()))
    empty = MarketStateUniverseView("empty", (), (), "fixture_missing_view", False)

    def view(name: str, members: Sequence[str], enabled: bool) -> MarketStateUniverseView:
        clean = tuple(dict.fromkeys(_asset(member) for member in members if str(member).strip()))
        return MarketStateUniverseView(
            name=name,
            members=clean,
            source_assets=tuple(dict.fromkeys(_source_asset_for_member(member, entry_index) for member in clean)),
            purpose=f"relationship discovery fixture {name}",
            enabled_for_v1_features=enabled,
        )

    return MarketStateUniverseV1Views(
        manifest_path=manifest_path,
        manifest_id=str(payload.get("manifest_id") or "fixture_manifest"),
        recommended_variant=str(payload.get("recommended_variant") or "fixture"),
        anchors=view("anchors", anchors, True),
        core_basket=view("core_basket", core, True),
        effective_core=view("effective_core", effective_core, True),
        broad_clean_risk=view("broad_clean_risk", broad_clean, True),
        broad_with_satellites=empty,
        speculative_satellite=empty,
        stable_peg_panel=view("stable_peg_panel", stable, False),
        excluded=view("excluded", excluded, False),
        needs_review=view("needs_review", needs_review, False),
    )


def _build_universe(
    payload: Mapping[str, Any],
    views: MarketStateUniverseV1Views,
    *,
    manifest_path: Path | None,
    request: RelationshipDiscoveryScopeRequest,
) -> RelationshipDiscoveryUniverse:
    entry_index = _entry_index(payload)
    anchors = _anchor_source_assets(views)
    core_assets = tuple(dict.fromkeys((*anchors, *views.effective_core.source_assets)))
    stable_assets = set(views.stable_peg_panel.source_assets)
    excluded_assets = set(views.excluded.source_assets)
    needs_review_assets = set(views.needs_review.source_assets)

    blocked = set(stable_assets)
    if not request.include_excluded_assets:
        blocked.update(excluded_assets)
    if not request.include_needs_review_assets:
        blocked.update(needs_review_assets)

    broad_candidates = []
    for member, source_asset in zip(views.broad_clean_risk.members, views.broad_clean_risk.source_assets):
        if source_asset in core_assets or source_asset in blocked:
            continue
        broad_candidates.append((member, source_asset, _candidate_sort_key(member, entry_index)))
    broad_sample = tuple(source_asset for _, source_asset, _ in sorted(broad_candidates, key=lambda item: item[2])[: request.broad_sample_size])
    selected = tuple(dict.fromkeys((*anchors, *core_assets, *broad_sample)))
    return RelationshipDiscoveryUniverse(
        manifest_path=manifest_path,
        manifest_id=str(views.manifest_id or payload.get("manifest_id") or "unknown_manifest"),
        anchors=anchors,
        core_assets=core_assets,
        broad_sample_assets=broad_sample,
        selected_assets=selected,
        stable_peg_assets_excluded=tuple(sorted(stable_assets)),
        excluded_assets_blocked=tuple(sorted(excluded_assets)) if not request.include_excluded_assets else (),
        needs_review_assets_blocked=tuple(sorted(needs_review_assets)) if not request.include_needs_review_assets else (),
        selection_metadata={
            "anchor_policy": "BTC_ETH_if_present_in_manifest",
            "core_policy": "effective_core_else_core_basket",
            "broad_policy": "deterministic_clean_risk_by_coverage_activity_movement",
            "broad_sample_size": int(request.broad_sample_size),
            "stable_peg_as_peer_candidates": False,
            "excluded_assets_included": bool(request.include_excluded_assets),
            "needs_review_assets_included": bool(request.include_needs_review_assets),
        },
    )


def _resolve_source_roots(
    request: RelationshipDiscoveryScopeRequest,
    payload: Mapping[str, Any],
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    project_root = resolve_project_root(request.project_root)
    manifest_roots = payload.get("source_roots") if isinstance(payload.get("source_roots"), Mapping) else {}
    candidates: list[tuple[Path | None, str]] = []
    if request.ohlcvt_root is not None:
        candidates.append((resolve_project_path(request.ohlcvt_root, project_root=project_root), "explicit_argument"))
        resolved = None
    else:
        manifest_ohlcvt = str(manifest_roots.get("ohlcvt_root") or "").strip()
        if manifest_ohlcvt:
            candidates.append((Path(manifest_ohlcvt).expanduser(), "manifest.source_roots.ohlcvt_root"))
        resolved = resolve_regime_feature_source_roots(project_root=project_root)
        candidates.append((resolved.ohlcvt_root, resolved.ohlcvt_source))

    inspected: list[dict[str, Any]] = []
    ohlcvt_root: Path | None = None
    seen: set[Path] = set()
    for root, source in candidates:
        if root is None:
            continue
        path = root.resolve()
        if path in seen:
            continue
        seen.add(path)
        exists = path.exists() and path.is_dir()
        inspected.append({"source": source, "root": str(path), "exists": bool(exists)})
        if exists and ohlcvt_root is None:
            ohlcvt_root = path

    scalar_candidates: list[tuple[Path | None, str]] = []
    if request.scalar_feature_root is not None:
        scalar_candidates.append((resolve_project_path(request.scalar_feature_root, project_root=project_root), "explicit_argument"))
    manifest_scalar = str(manifest_roots.get("scalar_feature_root") or "").strip()
    if manifest_scalar:
        scalar_candidates.append((Path(manifest_scalar).expanduser(), "manifest.source_roots.scalar_feature_root"))
    if resolved is not None:
        scalar_candidates.append((resolved.scalar_feature_root, resolved.scalar_feature_source))
    scalar_root = next((root.resolve() for root, _ in scalar_candidates if root is not None and root.exists() and root.is_dir()), None)
    return ohlcvt_root, scalar_root, {"ohlcvt_candidates": inspected, "scalar_root": str(scalar_root) if scalar_root else None}


def _build_panels(
    request: RelationshipDiscoveryScopeRequest,
    *,
    universe: RelationshipDiscoveryUniverse,
    ohlcvt_root: Path,
    scalar_feature_root: Path | None,
) -> dict[int, RelationshipReturnPanelResult]:
    panels: dict[int, RelationshipReturnPanelResult] = {}
    for interval in request.intervals_to_load():
        panels[int(interval)] = build_relationship_return_panel(
            RelationshipReturnPanelRequest(
                ohlcvt_root=ohlcvt_root,
                scalar_feature_root=scalar_feature_root,
                interval=int(interval),
                assets=universe.selected_assets,
                start_ts=request.start_ts,
                end_ts=request.end_ts,
                min_assets=request.min_assets,
                min_overlap=request.min_overlap,
                evidence_probe=(int(interval) == DEFAULT_EVIDENCE_INTERVAL),
            )
        )
    return panels


def _scope_status_from_panels(
    request: RelationshipDiscoveryScopeRequest,
    panels: Mapping[int, RelationshipReturnPanelResult],
) -> tuple[str, tuple[str, ...], str | None]:
    required = (int(request.primary_interval), int(request.confirmation_interval))
    for interval in required:
        panel = panels.get(interval)
        if panel is None:
            return RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA, (f"missing_panel_interval_{interval}",), f"required interval {interval} was not attempted"
        if panel.status == RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED:
            continue
        status_map = {
            RELATIONSHIP_DATA_STATUS_MISSING_SOURCE_DATA: RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA,
            RELATIONSHIP_DATA_STATUS_INSUFFICIENT_ASSETS: RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_ASSETS,
            RELATIONSHIP_DATA_STATUS_INSUFFICIENT_OVERLAP: RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_OVERLAP,
        }
        return (
            status_map.get(panel.status, RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA),
            tuple(panel.reason_codes),
            panel.message,
        )
    return RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED, ("real_data_loaded",), None


def _anchor_source_assets(views: MarketStateUniverseV1Views) -> tuple[str, ...]:
    anchors: list[str] = []
    for member, source_asset in zip(views.anchors.members, views.anchors.source_assets):
        if member in {"BTC", "ETH"}:
            anchors.append(source_asset)
    return tuple(dict.fromkeys(anchors))


def _candidate_sort_key(member: str, entry_index: Mapping[str, Mapping[str, Any]]) -> tuple[float, float, float, float, float, int, str]:
    entry = entry_index.get(_asset(member), {})
    diagnostics = entry.get("eligibility_diagnostics_summary") if isinstance(entry.get("eligibility_diagnostics_summary"), Mapping) else {}
    coverage = _float(diagnostics.get("coverage_ratio"), default=0.0)
    activity = _float(diagnostics.get("activity_score"), default=0.0)
    movement = _float(diagnostics.get("movement_score"), default=0.0)
    history = _float(diagnostics.get("history_days"), default=0.0)
    flat_peg = _float(diagnostics.get("flat_peg_risk_score"), default=1.0)
    data_rank = int(_float(entry.get("data_rank"), default=0.0))
    return (-coverage, -activity, -movement, -history, flat_peg, -data_rank, _asset(member))


def _entry_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for field_name, values in payload.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            asset = _asset(item.get("asset") or item.get("base_asset") or item.get("local_asset_id") or "")
            if asset:
                index.setdefault(asset, item)
    return index


def _members_from_entries(values: Any) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    members: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            members.append(_asset(item.get("asset") or item.get("base_asset") or item.get("local_asset_id") or ""))
        else:
            members.append(_asset(item))
    return tuple(dict.fromkeys(member for member in members if member))


def _members_from_any(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return _members_from_entries(value)


def _source_asset_for_member(member: str, entry_index: Mapping[str, Mapping[str, Any]]) -> str:
    entry = entry_index.get(_asset(member), {})
    local = str(entry.get("local_asset_id") or entry.get("canonical_pair") or "").strip().upper()
    return local or _asset(member)


def _asset(value: object) -> str:
    text = str(value).strip().upper()
    for token in ("/", "-", "_", " "):
        text = text.replace(token, "")
    return text


def _float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _request_metadata(request: RelationshipDiscoveryScopeRequest) -> dict[str, Any]:
    return {
        "manifest_path": str(request.manifest_path) if request.manifest_path is not None else None,
        "market_state_universe_root": str(request.market_state_universe_root) if request.market_state_universe_root is not None else None,
        "fixture_manifest_supplied": request.fixture_manifest is not None,
        "ohlcvt_root": str(request.ohlcvt_root) if request.ohlcvt_root is not None else None,
        "scalar_feature_root": str(request.scalar_feature_root) if request.scalar_feature_root is not None else None,
        "primary_interval": int(request.primary_interval),
        "confirmation_interval": int(request.confirmation_interval),
        "include_60m_evidence_probe": bool(request.include_60m_evidence_probe),
        "broad_sample_size": int(request.broad_sample_size),
        "min_assets": int(request.min_assets),
        "min_overlap": int(request.min_overlap),
        "start_ts": request.start_ts,
        "end_ts": request.end_ts,
        "include_excluded_assets": bool(request.include_excluded_assets),
        "include_needs_review_assets": bool(request.include_needs_review_assets),
        "load_real_data_for_fixture": bool(request.load_real_data_for_fixture),
    }


__all__ = [
    "DEFAULT_CONFIRMATION_INTERVAL",
    "DEFAULT_EVIDENCE_INTERVAL",
    "DEFAULT_MARKET_STATE_UNIVERSE_ROOT",
    "DEFAULT_PRIMARY_INTERVAL",
    "RELATIONSHIP_SCOPE_STATUS_FIXTURE_ONLY",
    "RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_ASSETS",
    "RELATIONSHIP_SCOPE_STATUS_INSUFFICIENT_OVERLAP",
    "RELATIONSHIP_SCOPE_STATUS_MANIFEST_NOT_FOUND",
    "RELATIONSHIP_SCOPE_STATUS_MISSING_SOURCE_DATA",
    "RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED",
    "RELATIONSHIP_SCOPE_STATUSES",
    "RelationshipDiscoveryScopeRequest",
    "RelationshipDiscoveryScopeResult",
    "RelationshipDiscoveryUniverse",
    "build_relationship_discovery_scope",
]
