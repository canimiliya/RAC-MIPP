"""Seeded, episode-scoped unreliable communication channel.

The channel is algorithm-neutral: callers submit immutable send-time payloads and
positions, then consume per-receiver deliveries.  It never reads environment
state after submission and owns a dedicated NumPy generator.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import heapq
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ChannelConfig:
    packet_drop_probability: float = 0.0
    delay_steps: int = 0
    communication_radius: float = 25.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.packet_drop_probability <= 1.0:
            raise ValueError("packet_drop_probability must be in [0, 1]")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if self.communication_radius < 0:
            raise ValueError("communication_radius must be non-negative")


@dataclass(frozen=True)
class Message:
    sender_id: int
    receiver_id: int
    send_step: int
    delivery_step: int
    sequence: int
    payload: Any


@dataclass(frozen=True)
class CommunicationEvent:
    step: int
    event: str
    sender_id: int
    receiver_id: int
    send_step: int
    delivery_step: int | None
    message_age: int | None
    distance: float


class ChannelModel:
    """Directed broadcast channel with send-time range checks and fixed delay.

    One attempt is one directed non-self sender/receiver pair per step. Range is
    checked at send time. Eligible packets are independently dropped, otherwise
    queued for delivery at ``send_step + delay_steps``. A dropped packet never
    reappears. Queue ordering is ``(delivery_step, send_step, sender, receiver,
    sequence)`` and queues are cleared by ``reset``.
    """

    def __init__(self, config: ChannelConfig, channel_seed: int):
        self.config = config
        self.channel_seed = int(channel_seed)
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.channel_seed)
        self._queue: list[tuple[tuple[int, int, int, int, int], Message]] = []
        self._sequence = 0
        self.events: list[CommunicationEvent] = []

    def exchange(
        self,
        step: int,
        payloads: Mapping[int, Any],
        positions: Sequence[Any],
    ) -> dict[int, dict[int, Any]]:
        """Attempt all directed links and return packets delivered at ``step``."""

        ids = sorted(payloads)
        if ids != list(range(len(positions))):
            raise ValueError("payload ids and position indexes must be contiguous")
        step = int(step)
        neighbor_mask = self.neighbor_mask(positions)
        for receiver in ids:
            for sender in ids:
                if sender == receiver:
                    continue
                distance = float(
                    np.linalg.norm(
                        np.asarray(positions[sender], dtype=float)
                        - np.asarray(positions[receiver], dtype=float)
                    )
                )
                eligible = bool(neighbor_mask[receiver, sender])
                self._record(step, "ATTEMPTED", sender, receiver, step, None, None, distance)
                if not eligible:
                    self._record(step, "OUT_OF_RANGE", sender, receiver, step, None, None, distance)
                    continue
                self._record(step, "RANGE_ELIGIBLE", sender, receiver, step, None, None, distance)
                if self._rng.random() < self.config.packet_drop_probability:
                    self._record(step, "DROPPED", sender, receiver, step, None, None, distance)
                    continue
                delivery_step = step + self.config.delay_steps
                message = Message(
                    sender_id=sender,
                    receiver_id=receiver,
                    send_step=step,
                    delivery_step=delivery_step,
                    sequence=self._sequence,
                    payload=copy.deepcopy(payloads[sender]),
                )
                self._sequence += 1
                key = (delivery_step, step, sender, receiver, message.sequence)
                heapq.heappush(self._queue, (key, message))
                self._record(
                    step,
                    "ENQUEUED" if self.config.delay_steps else "SCHEDULED_IMMEDIATE",
                    sender,
                    receiver,
                    step,
                    delivery_step,
                    self.config.delay_steps,
                    distance,
                )

        delivered: dict[int, dict[int, Any]] = {receiver: {} for receiver in ids}
        while self._queue and self._queue[0][0][0] <= step:
            _, message = heapq.heappop(self._queue)
            delivered[message.receiver_id][message.sender_id] = copy.deepcopy(message.payload)
            distance = float(
                np.linalg.norm(
                    np.asarray(positions[message.sender_id], dtype=float)
                    - np.asarray(positions[message.receiver_id], dtype=float)
                )
            )
            self._record(
                step,
                "DELIVERED",
                message.sender_id,
                message.receiver_id,
                message.send_step,
                message.delivery_step,
                step - message.send_step,
                distance,
            )
        return delivered

    def neighbor_mask(self, positions: Sequence[Any]) -> np.ndarray:
        """Return the directed send-time range mask with a false diagonal."""

        copied = [np.asarray(position, dtype=float).copy() for position in positions]
        mask = np.zeros((len(copied), len(copied)), dtype=bool)
        for receiver, own in enumerate(copied):
            for sender, other in enumerate(copied):
                if sender != receiver:
                    mask[receiver, sender] = bool(
                        np.linalg.norm(other - own) <= self.config.communication_radius
                    )
        return mask

    def _record(
        self,
        step: int,
        event: str,
        sender: int,
        receiver: int,
        send_step: int,
        delivery_step: int | None,
        age: int | None,
        distance: float,
    ) -> None:
        self.events.append(
            CommunicationEvent(
                step, event, sender, receiver, send_step, delivery_step, age, distance
            )
        )

    def event_records(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]

    def summary(self, *, n_agents: int, episode_steps: int) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.event] = counts.get(event.event, 0) + 1
        attempted = counts.get("ATTEMPTED", 0)
        eligible = counts.get("RANGE_ELIGIBLE", 0)
        delivered = counts.get("DELIVERED", 0)
        ages = [event.message_age for event in self.events if event.event == "DELIVERED"]
        return {
            "messages_attempted": attempted,
            "messages_range_eligible": eligible,
            "messages_dropped": counts.get("DROPPED", 0),
            "messages_delayed": counts.get("ENQUEUED", 0),
            "messages_delivered": delivered,
            "packet_delivery_ratio": delivered / eligible if eligible else 0.0,
            "effective_neighbor_degree": eligible / (episode_steps * n_agents),
            "message_age_mean": float(np.mean(ages)) if ages else 0.0,
            "message_age_max": max(ages) if ages else 0,
            "communication_radius": self.config.communication_radius,
            "normalized_communication_load": eligible / attempted if attempted else 0.0,
            "load_unit": "MESSAGE_UNITS",
            "pending_messages_discarded_at_episode_end": len(self._queue),
        }
