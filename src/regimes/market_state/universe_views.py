from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


MARKET_STATE_V1_MANIFEST_SCHEMA_VERSION = 1
MARKET_STATE_V1_RECOMMENDED_VARIANTS: frozenset[str] = frozenset(
    {"C_dual_broad", "dual_broad_backward_compatible"}
)
MARKET_STATE_V1_EXPECTED_COUNTS: Mapping[str, int] = {
    "anchors": 2,
    "core_basket": 13,
    "effective_core": 15,
    "stable_peg_panel": 11,
    "speculative_satellite": 22,
    "broad_clean_risk": 128,
    "broad_with_satellites": 150,
}


@dataclass(frozen=True)
class MarketStateUniverseView:
    name: str
    members: tuple[str, ...]
    source_assets: tuple[str, ...]
    purpose: str
    enabled_for_v1_features: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def source_asset_count(self) -> int:
        return len(self.source_assets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "members": list(self.members),
            "member_count": self.count,
            "source_assets": list(self.source_assets),
            "source_asset_count": self.source_asset_count,
            "purpose": self.purpose,
            "enabled_for_v1_features": bool(self.enabled_for_v1_features),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MarketStateUniverseV1Views:
    manifest_path: Path
    manifest_id: str
    recommended_variant: str
    anchors: MarketStateUniverseView
    core_basket: MarketStateUniverseView
    effective_core: MarketStateUniverseView
    broad_clean_risk: MarketStateUniverseView
    broad_with_satellites: MarketStateUniverseView
    speculative_satellite: MarketStateUniverseView
    stable_peg_panel: MarketStateUniverseView
    excluded: MarketStateUniverseView
    needs_review: MarketStateUniverseView
    raw_counts: Mapping[str, int] = field(default_factory=dict)
    validation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def view(self, name: str) -> MarketStateUniverseView:
        normalized = str(name).strip().lower()
        aliases = {
            "broad_universe": "broad_clean_risk",
            "broad_clean": "broad_clean_risk",
            "clean_risk_members": "broad_clean_risk",
            "with_satellites": "broad_with_satellites",
            "speculative_satellite_members": "speculative_satellite",
            "stable": "stable_peg_panel",
            "stable_panel": "stable_peg_panel",
            "true_needs_review": "needs_review",
        }
        attr = aliases.get(normalized, normalized)
        if attr not in _VIEW_NAMES:
            valid = ", ".join(_VIEW_NAMES)
            raise ValueError(f"Unknown Market-State v1 universe view {name!r}; expected one of: {valid}")
        return getattr(self, attr)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "market_state_v1_universe_views",
            "manifest_path": str(self.manifest_path),
            "manifest_id": self.manifest_id,
            "recommended_variant": self.recommended_variant,
            "views": {name: getattr(self, name).as_dict() for name in _VIEW_NAMES},
            "counts": {name: getattr(self, name).count for name in _VIEW_NAMES},
            "raw_counts": dict(self.raw_counts),
            "validation_metadata": dict(self.validation_metadata),
            "production_writes_enabled": False,
            "dynamic_peer_discovery_enabled": False,
        }


def load_market_state_universe_v1_views(
    manifest_path: str | Path,
    *,
    validate_expected_counts: bool = True,
    preserve_usual_needs_review: bool = True,
) -> MarketStateUniverseV1Views:
    path = Path(manifest_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Market-State universe manifest path does not exist: {path}")
    payload = _load_json_object(path)
    _validate_manifest_identity(payload)

    broad_policy_views = _required_mapping(payload, "broad_policy_views")
    recommended_variant = _recommended_variant(payload, broad_policy_views)
    if recommended_variant not in MARKET_STATE_V1_RECOMMENDED_VARIANTS:
        valid = ", ".join(sorted(MARKET_STATE_V1_RECOMMENDED_VARIANTS))
        raise ValueError(
            f"Market-State universe manifest recommended_variant must be a dual-broad policy; "
            f"got {recommended_variant!r}, expected one of: {valid}"
        )

    entry_index = _entry_index(payload)
    views = _build_views(payload, broad_policy_views, entry_index)
    raw_counts = _raw_counts(payload)
    if validate_expected_counts:
        _validate_expected_counts(views, raw_counts)
    if preserve_usual_needs_review:
        _validate_usual_needs_review(views)

    return MarketStateUniverseV1Views(
        manifest_path=path,
        manifest_id=str(payload.get("manifest_id", "")),
        recommended_variant=recommended_variant,
        anchors=views["anchors"],
        core_basket=views["core_basket"],
        effective_core=views["effective_core"],
        broad_clean_risk=views["broad_clean_risk"],
        broad_with_satellites=views["broad_with_satellites"],
        speculative_satellite=views["speculative_satellite"],
        stable_peg_panel=views["stable_peg_panel"],
        excluded=views["excluded"],
        needs_review=views["needs_review"],
        raw_counts=raw_counts,
        validation_metadata={
            "expected_counts_validated": bool(validate_expected_counts),
            "usual_preserved_as_needs_review": bool(preserve_usual_needs_review),
            "pairwise_policy": "core_only_no_broad_all_to_all",
            "cross_asset_policy": "disabled",
        },
    )


def _build_views(
    payload: Mapping[str, Any],
    broad_policy_views: Mapping[str, Any],
    entry_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, MarketStateUniverseView]:
    anchors = _members_from_entries(_required_sequence(payload, "anchors"))
    core = _members_from_entries(_required_sequence(payload, "core_basket"))
    effective_core = _members_from_any(payload.get("effective_core")) or tuple(dict.fromkeys((*anchors, *core)))
    broad_clean = (
        _members_from_any(broad_policy_views.get("clean_risk_members"))
        or _members_from_any(payload.get("broad_universe_clean_risk"))
        or _members_from_entries(_required_sequence(payload, "broad_universe"))
    )
    broad_with_satellites = (
        _members_from_any(broad_policy_views.get("with_satellites_members"))
        or _members_from_any(payload.get("broad_universe_with_satellites"))
    )
    speculative = (
        _members_from_any(broad_policy_views.get("speculative_satellite_members"))
        or _members_from_entries(_required_sequence(payload, "speculative_satellite"))
    )
    stable = _members_from_entries(_required_sequence(payload, "stable_peg_panel"))
    excluded = _members_from_entries(_required_sequence(payload, "excluded"))
    needs_review = (
        _members_from_entries(_required_sequence(payload, "needs_review"))
        or _members_from_entries(_members_from_any(payload.get("true_needs_review", ())))
    )

    if not broad_with_satellites:
        broad_with_satellites = tuple(dict.fromkeys((*broad_clean, *speculative)))

    return {
        "anchors": _view("anchors", anchors, entry_index, "BTC/ETH anchor assets.", True),
        "core_basket": _view("core_basket", core, entry_index, "Seeded directional core basket.", True),
        "effective_core": _view(
            "effective_core",
            effective_core,
            entry_index,
            "Anchors plus core basket for return, volatility, covariance, and concentration features.",
            True,
        ),
        "broad_clean_risk": _view(
            "broad_clean_risk",
            broad_clean,
            entry_index,
            "Broad clean-risk universe for ordinary breadth, dispersion, activity, and drawdown breadth.",
            True,
        ),
        "broad_with_satellites": _view(
            "broad_with_satellites",
            broad_with_satellites,
            entry_index,
            "Broad universe including speculative satellites for explicitly gated sidecar use.",
            False,
        ),
        "speculative_satellite": _view(
            "speculative_satellite",
            speculative,
            entry_index,
            "Speculative satellite sidecar; not part of clean-risk default features.",
            False,
        ),
        "stable_peg_panel": _view(
            "stable_peg_panel",
            stable,
            entry_index,
            "Stable/peg panel used only for stable stress/depeg features.",
            True,
        ),
        "excluded": _view("excluded", excluded, entry_index, "Excluded from v1 feature production.", False),
        "needs_review": _view(
            "needs_review",
            needs_review,
            entry_index,
            "Manual-review assets excluded from v1 feature production until explicitly decided.",
            False,
        ),
    }


def _view(
    name: str,
    members: Sequence[str],
    entry_index: Mapping[str, Mapping[str, Any]],
    purpose: str,
    enabled: bool,
) -> MarketStateUniverseView:
    clean_members = tuple(dict.fromkeys(_asset(member) for member in members if str(member).strip()))
    source_assets = tuple(dict.fromkeys(_source_asset_for_member(member, entry_index) for member in clean_members))
    return MarketStateUniverseView(
        name=name,
        members=clean_members,
        source_assets=source_assets,
        purpose=purpose,
        enabled_for_v1_features=enabled,
        metadata={"source_asset_policy": "local_asset_id_when_available_else_manifest_member"},
    )


def _validate_manifest_identity(payload: Mapping[str, Any]) -> None:
    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    if artifact_kind != "market_state_universe_manifest":
        raise ValueError("Market-State universe manifest artifact_kind must be market_state_universe_manifest")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != MARKET_STATE_V1_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Market-State universe manifest schema_version {schema_version!r}; "
            f"expected {MARKET_STATE_V1_MANIFEST_SCHEMA_VERSION}"
        )
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Market-State universe manifest production_enabled must be false for v1 feature foundation")
    for field_name in (
        "anchors",
        "core_basket",
        "effective_core",
        "broad_universe",
        "stable_peg_panel",
        "speculative_satellite",
        "excluded",
        "needs_review",
        "not_selected_v1",
        "deferred_low_priority",
    ):
        _required_sequence(payload, field_name)
    broad_policy = _required_mapping(payload, "broad_policy_views")
    for field_name in ("clean_risk_members", "with_satellites_members", "speculative_satellite_members"):
        _required_sequence(broad_policy, field_name)


def _validate_expected_counts(
    views: Mapping[str, MarketStateUniverseView],
    raw_counts: Mapping[str, int],
) -> None:
    actual = {
        "anchors": views["anchors"].count,
        "core_basket": views["core_basket"].count,
        "effective_core": views["effective_core"].count,
        "stable_peg_panel": views["stable_peg_panel"].count,
        "speculative_satellite": views["speculative_satellite"].count,
        "broad_clean_risk": views["broad_clean_risk"].count,
        "broad_with_satellites": views["broad_with_satellites"].count,
    }
    errors = [
        f"{name}: expected {expected}, got {actual.get(name)}"
        for name, expected in MARKET_STATE_V1_EXPECTED_COUNTS.items()
        if actual.get(name) != expected
    ]
    if errors:
        raise ValueError("Market-State universe manifest v1.3 count validation failed: " + "; ".join(errors))
    raw_expected = {
        "anchors": 2,
        "core_basket": 13,
        "effective_core": 15,
        "stable_peg_panel": 11,
        "speculative_satellite": 22,
        "broad_universe": 128,
    }
    raw_errors = [
        f"counts.{name}: expected {expected}, got {raw_counts.get(name)}"
        for name, expected in raw_expected.items()
        if name in raw_counts and raw_counts.get(name) != expected
    ]
    if raw_errors:
        raise ValueError("Market-State universe manifest raw count validation failed: " + "; ".join(raw_errors))


def _validate_usual_needs_review(views: Mapping[str, MarketStateUniverseView]) -> None:
    enabled_views = ("anchors", "core_basket", "effective_core", "broad_clean_risk", "broad_with_satellites", "speculative_satellite", "stable_peg_panel")
    misplaced = [name for name in enabled_views if "USUAL" in views[name].members]
    if misplaced:
        raise ValueError(f"USUAL must remain in needs_review for Market-State v1; found in {', '.join(misplaced)}")
    if "USUAL" not in views["needs_review"].members:
        raise ValueError("USUAL must remain in Market-State v1 needs_review unless explicitly overridden")


def _recommended_variant(payload: Mapping[str, Any], broad_policy_views: Mapping[str, Any]) -> str:
    value = str(broad_policy_views.get("recommended_variant") or "").strip()
    if value:
        return value
    recommended = payload.get("recommended_v1_3_policy")
    if isinstance(recommended, Mapping):
        value = str(recommended.get("policy") or "").strip()
        if value:
            return value
    return ""


def _entry_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for field_name in (
        "anchors",
        "core_basket",
        "effective_core",
        "broad_universe",
        "broad_universe_clean_risk",
        "broad_universe_with_satellites",
        "stable_peg_panel",
        "speculative_satellite",
        "excluded",
        "needs_review",
        "not_selected_v1",
        "deferred_low_priority",
    ):
        values = payload.get(field_name, ())
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for item in values:
                if isinstance(item, Mapping):
                    asset = _asset(item.get("asset") or item.get("base_asset") or item.get("local_asset_id") or "")
                    if asset:
                        index.setdefault(asset, item)
    return index


def _source_asset_for_member(member: str, entry_index: Mapping[str, Mapping[str, Any]]) -> str:
    entry = entry_index.get(_asset(member), {})
    local = str(entry.get("local_asset_id") or entry.get("canonical_pair") or "").strip().upper()
    return local or _asset(member)


def _members_from_entries(values: Sequence[Any]) -> tuple[str, ...]:
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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _members_from_entries(value)
    raise ValueError("Market-State universe manifest view members must be a sequence")


def _raw_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    raw = payload.get("counts")
    if isinstance(raw, Mapping):
        return {str(key): int(value) for key, value in raw.items() if _is_int_like(value)}
    return {}


def _required_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    if field_name not in payload:
        raise ValueError(f"Market-State universe manifest missing required bucket/field: {field_name}")
    value = payload[field_name]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Market-State universe manifest field {field_name} must be a sequence")
    return value


def _required_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if field_name not in payload:
        raise ValueError(f"Market-State universe manifest missing required object: {field_name}")
    value = payload[field_name]
    if not isinstance(value, Mapping):
        raise ValueError(f"Market-State universe manifest field {field_name} must be an object")
    return value


def _load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Market-State universe manifest JSON root must be an object")
    return payload


def _asset(value: object) -> str:
    text = str(value).strip().upper()
    for token in ("/", "-", "_", " "):
        text = text.replace(token, "")
    return text


def _is_int_like(value: object) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


_VIEW_NAMES: tuple[str, ...] = (
    "anchors",
    "core_basket",
    "effective_core",
    "broad_clean_risk",
    "broad_with_satellites",
    "speculative_satellite",
    "stable_peg_panel",
    "excluded",
    "needs_review",
)


__all__ = [
    "MARKET_STATE_V1_EXPECTED_COUNTS",
    "MARKET_STATE_V1_MANIFEST_SCHEMA_VERSION",
    "MARKET_STATE_V1_RECOMMENDED_VARIANTS",
    "MarketStateUniverseV1Views",
    "MarketStateUniverseView",
    "load_market_state_universe_v1_views",
]
