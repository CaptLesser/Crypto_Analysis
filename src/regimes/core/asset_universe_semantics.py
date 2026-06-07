from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MARKET_STATE_UNIVERSE_MANIFEST = Path("reports/regimes/foundation/market_state_universe/market_state_universe_manifest_v1_3.json")

SEMANTIC_CLASS_NONE = "none"
SEMANTIC_CLASS_STABLE_OR_PEGGED = "stable_or_pegged"
SEMANTIC_CLASS_PROXY_OR_WRAPPED = "proxy_or_wrapped"
SEMANTIC_CLASS_FIAT_OR_COMMODITY = "fiat_or_commodity"
SEMANTIC_CLASS_UNCERTAIN_NEEDS_REVIEW = "uncertain_needs_review"

_STABLE_OR_PEGGED_BASES = frozenset(
    {
        "DAI",
        "EUROP",
        "EURQ",
        "EURR",
        "PYUSD",
        "USDC",
        "USDG",
        "USDQ",
        "USDR",
        "USDS",
        "USDT",
    }
)
_FIAT_BASES = frozenset({"AUD", "EUR", "GBP"})
_COMMODITY_BASES = frozenset({"PAXG"})
_WRAPPED_OR_PROXY_BASES = frozenset({"TBTC", "WBTC"})
_NEEDS_REVIEW_BASES = frozenset({"USUAL"})


def load_market_state_manifest_asset_semantics(path: Path | str | None = None) -> dict[str, Mapping[str, Any]]:
    manifest_path = Path(path) if path is not None else DEFAULT_MARKET_STATE_UNIVERSE_MANIFEST
    if not manifest_path.exists() or not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    index: dict[str, Mapping[str, Any]] = {}
    for value in payload.values():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            local = str(item.get("local_asset_id") or item.get("canonical_pair") or "").strip().upper()
            if local:
                index.setdefault(local, item)
            canonical_asset = str(item.get("asset") or item.get("base_asset") or "").strip().upper()
            canonical_pair = f"{canonical_asset}USD" if canonical_asset and not canonical_asset.endswith("USD") else canonical_asset
            if canonical_pair:
                index.setdefault(canonical_pair, item)
    return index


def asset_semantic_metadata(asset: str, manifest_index: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    index = manifest_index or {}
    entry = dict(index.get(str(asset).upper(), {}))
    tags = entry.get("semantic_tags", ())
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        tags = ()
    base = str(entry.get("base_asset") or entry.get("asset") or base_asset_from_pair(asset)).upper()
    return {
        "base_asset": base,
        "bucket": entry.get("bucket"),
        "market_state_v1_role": entry.get("market_state_v1_role"),
        "semantic_tags": [str(tag) for tag in tags if str(tag).strip()],
        "requires_manual_review": bool(entry.get("requires_manual_review", False)),
        "semantic_source": "market_state_universe_manifest" if entry else "asset_symbol_fallback",
    }


def classify_asset_semantic_role(asset: str, manifest_index: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    metadata = asset_semantic_metadata(asset, manifest_index)
    base = str(metadata.get("base_asset") or base_asset_from_pair(asset)).upper()
    bucket = str(metadata.get("bucket") or "")
    tags = {str(tag) for tag in metadata.get("semantic_tags", ()) if str(tag).strip()}
    role = str(metadata.get("market_state_v1_role") or "")
    if base in _NEEDS_REVIEW_BASES or bucket == "needs_review" or bool(metadata.get("requires_manual_review")):
        return SEMANTIC_CLASS_UNCERTAIN_NEEDS_REVIEW, ("semantic_review_required",), metadata
    if base in _STABLE_OR_PEGGED_BASES or bucket == "stable_peg_panel" or "stable_peg" in tags or role == "stable_peg_panel":
        return SEMANTIC_CLASS_STABLE_OR_PEGGED, ("stable_or_pegged_asset",), metadata
    if base in _WRAPPED_OR_PROXY_BASES or "wrapped_duplicate" in tags:
        return SEMANTIC_CLASS_PROXY_OR_WRAPPED, ("proxy_or_wrapped_duplicate",), metadata
    if base in _FIAT_BASES:
        return SEMANTIC_CLASS_FIAT_OR_COMMODITY, ("fiat_pair_not_asset_state_candidate",), metadata
    if base in _COMMODITY_BASES or "commodity_backed" in tags:
        return SEMANTIC_CLASS_FIAT_OR_COMMODITY, ("commodity_or_non_crypto_proxy",), metadata
    return SEMANTIC_CLASS_NONE, (), metadata


def base_asset_from_pair(asset: str) -> str:
    text = str(asset).strip().upper()
    return text[:-3] if text.endswith("USD") else text


__all__ = [
    "DEFAULT_MARKET_STATE_UNIVERSE_MANIFEST",
    "SEMANTIC_CLASS_FIAT_OR_COMMODITY",
    "SEMANTIC_CLASS_NONE",
    "SEMANTIC_CLASS_PROXY_OR_WRAPPED",
    "SEMANTIC_CLASS_STABLE_OR_PEGGED",
    "SEMANTIC_CLASS_UNCERTAIN_NEEDS_REVIEW",
    "asset_semantic_metadata",
    "base_asset_from_pair",
    "classify_asset_semantic_role",
    "load_market_state_manifest_asset_semantics",
]
