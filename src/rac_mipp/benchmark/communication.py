"""Passive communication instrumentation; hooks never alter environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


NOT_AVAILABLE = "NOT_AVAILABLE_YET"


class NullCommunicationObserver:
    enabled = False

    def observe_positions(self, positions: Sequence[Any], radius: float) -> None:
        del positions, radius

    def record_attempt(self, payload_bytes: int | None = None) -> None:
        del payload_bytes

    def record_delivery(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "instrumentation": "HOOK_OFF",
            "messages_attempted": NOT_AVAILABLE,
            "messages_delivered": NOT_AVAILABLE,
            "bytes": NOT_AVAILABLE,
            "normalized_load": NOT_AVAILABLE,
            "effective_neighbor_degree": NOT_AVAILABLE,
            "communication_radius": NOT_AVAILABLE,
            "delivery_ratio": NOT_AVAILABLE,
        }


@dataclass
class PassiveCommunicationObserver:
    """Record supplied events and proximity snapshots without returning controls."""

    attempted: int = 0
    delivered: int = 0
    payload_bytes: int = 0
    bytes_known: bool = True
    proximity_directed_links: int = 0
    position_snapshots: int = 0
    agent_count: int | None = None
    radii: set[float] = field(default_factory=set)
    enabled: bool = True

    def observe_positions(self, positions: Sequence[Any], radius: float) -> None:
        import numpy as np

        copied = [np.asarray(position, dtype=float).copy() for position in positions]
        self.agent_count = len(copied)
        self.radii.add(float(radius))
        self.position_snapshots += 1
        for i, own in enumerate(copied):
            for j, other in enumerate(copied):
                if i != j and float(np.linalg.norm(own - other)) <= radius:
                    self.proximity_directed_links += 1

    def record_attempt(self, payload_bytes: int | None = None) -> None:
        self.attempted += 1
        if payload_bytes is None:
            self.bytes_known = False
        else:
            self.payload_bytes += int(payload_bytes)

    def record_delivery(self) -> None:
        self.delivered += 1

    def snapshot(self) -> dict[str, Any]:
        degree = NOT_AVAILABLE
        if self.position_snapshots and self.agent_count:
            degree = self.proximity_directed_links / (
                self.position_snapshots * self.agent_count
            )
        ratio = self.delivered / self.attempted if self.attempted else NOT_AVAILABLE
        radius = next(iter(self.radii)) if len(self.radii) == 1 else NOT_AVAILABLE
        return {
            "instrumentation": "PASSIVE_OBSERVER",
            "messages_attempted": self.attempted or NOT_AVAILABLE,
            "messages_delivered": self.delivered or NOT_AVAILABLE,
            "bytes": self.payload_bytes if self.bytes_known and self.attempted else NOT_AVAILABLE,
            "normalized_load": NOT_AVAILABLE,
            "effective_neighbor_degree": degree,
            "communication_radius": radius,
            "delivery_ratio": ratio,
            "proximity_directed_links": self.proximity_directed_links,
        }
