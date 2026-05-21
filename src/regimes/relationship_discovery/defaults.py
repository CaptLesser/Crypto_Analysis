from __future__ import annotations

from src.regimes.relationship_discovery.policy import RelationshipDiscoveryPolicy


def default_relationship_discovery_policy() -> RelationshipDiscoveryPolicy:
    return RelationshipDiscoveryPolicy()


__all__ = ["default_relationship_discovery_policy"]
