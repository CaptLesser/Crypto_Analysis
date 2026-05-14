from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Type

from src.regimes.core.clusterer_base import (
    BaseClustererAdapter,
    ClustererCapabilities,
    DummyClustererAdapter,
)
from src.regimes.core.contracts import require_non_empty_string


@dataclass(frozen=True)
class ClustererRegistryEntry:
    family_name: str
    adapter_type: Type[BaseClustererAdapter]
    capabilities: ClustererCapabilities

    def __post_init__(self) -> None:
        family_name = require_non_empty_string(self.family_name, field_name="clusterer family name").lower()
        if not issubclass(self.adapter_type, BaseClustererAdapter):
            raise ValueError("Regime clusterer registry adapter_type must subclass BaseClustererAdapter")
        if family_name != self.capabilities.family_name:
            raise ValueError("Regime clusterer registry family_name must match capabilities.family_name")
        object.__setattr__(self, "family_name", family_name)

    def build(self, **hyperparameters: Any) -> BaseClustererAdapter:
        adapter = self.adapter_type(**hyperparameters)
        if adapter.report_capabilities().family_name != self.family_name:
            raise ValueError("Regime clusterer adapter capabilities do not match registry entry")
        return adapter

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "adapter_type": f"{self.adapter_type.__module__}.{self.adapter_type.__name__}",
            "capabilities": self.capabilities.as_dict(),
        }


@dataclass(frozen=True)
class ClustererFamilyRegistry:
    entries: Mapping[str, ClustererRegistryEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, ClustererRegistryEntry] = {}
        for key, entry in self.entries.items():
            if not isinstance(entry, ClustererRegistryEntry):
                raise ValueError("Regime clusterer registry entries must be ClustererRegistryEntry instances")
            normalized_key = require_non_empty_string(key, field_name="clusterer family name").lower()
            if normalized_key != entry.family_name:
                raise ValueError("Regime clusterer registry keys must match entry family names")
            normalized[normalized_key] = entry
        object.__setattr__(self, "entries", dict(sorted(normalized.items())))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def get_entry(self, family_name: str) -> ClustererRegistryEntry:
        key = require_non_empty_string(family_name, field_name="clusterer family name").lower()
        try:
            return self.entries[key]
        except KeyError as exc:
            valid = ", ".join(self.names)
            raise ValueError(f"Unsupported Regime clusterer family {key!r}; expected one of: {valid}") from exc

    def get_capabilities(self, family_name: str) -> ClustererCapabilities:
        return self.get_entry(family_name).capabilities

    def build(self, family_name: str, **hyperparameters: Any) -> BaseClustererAdapter:
        return self.get_entry(family_name).build(**hyperparameters)

    def register(
        self,
        adapter_type: Type[BaseClustererAdapter],
        *,
        replace: bool = False,
    ) -> "ClustererFamilyRegistry":
        capabilities = adapter_type.capabilities
        entry = ClustererRegistryEntry(
            family_name=capabilities.family_name,
            adapter_type=adapter_type,
            capabilities=capabilities,
        )
        if entry.family_name in self.entries and not replace:
            raise ValueError(f"Regime clusterer family {entry.family_name!r} is already registered")
        next_entries = dict(self.entries)
        next_entries[entry.family_name] = entry
        return ClustererFamilyRegistry(next_entries)

    def as_dict(self) -> dict[str, Any]:
        return {name: entry.as_dict() for name, entry in self.entries.items()}


def default_clusterer_registry() -> ClustererFamilyRegistry:
    from src.regimes.core.clusterer_adapters import shared_tier_a_clusterer_adapter_types

    registry = ClustererFamilyRegistry().register(DummyClustererAdapter)
    for adapter_type in shared_tier_a_clusterer_adapter_types():
        registry = registry.register(adapter_type)
    return registry


def build_clusterer_adapter(family_name: str, **hyperparameters: Any) -> BaseClustererAdapter:
    return default_clusterer_registry().build(family_name, **hyperparameters)


def clusterer_capabilities_registry() -> dict[str, ClustererCapabilities]:
    return {name: entry.capabilities for name, entry in default_clusterer_registry().entries.items()}


__all__ = [
    "ClustererFamilyRegistry",
    "ClustererRegistryEntry",
    "build_clusterer_adapter",
    "clusterer_capabilities_registry",
    "default_clusterer_registry",
]
