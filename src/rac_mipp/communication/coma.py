"""Project-side bridge from the generic channel to the pinned COMA environment."""

from __future__ import annotations

from contextlib import contextmanager
import types
from typing import Any, Iterator

import numpy as np

from .channel import ChannelConfig, ChannelModel


class _CompatibleLog:
    def __init__(self, channel: ChannelModel, timestep: int, n_agents: int):
        self.channel = channel
        self.timestep = int(timestep)
        self.n_agents = int(n_agents)
        self.global_log: dict[int, Any] = {}
        self._deliveries: dict[int, dict[int, Any]] | None = None

    def store_agent_message(self, message: dict[str, Any], agent_id: int):
        self.global_log[int(agent_id)] = message
        return self.global_log

    def finalize(self, positions: list[Any]) -> None:
        # The frozen upstream consumes one global NumPy draw for every receiver x
        # sender pair, including self. Preserve that environment RNG trajectory
        # for exact zero-noise parity; channel outcomes use only ChannelModel RNG.
        np.random.random_sample(self.n_agents * self.n_agents)
        self._deliveries = self.channel.exchange(self.timestep, self.global_log, positions)

    def get_messages(self, receiver_id: int) -> dict[int, Any]:
        if self._deliveries is None:
            raise RuntimeError("communication exchange was not finalized")
        # Own observation is local state, not a self-message, and is never dropped.
        result = {int(receiver_id): self.global_log[int(receiver_id)]}
        result.update(self._deliveries[int(receiver_id)])
        return result

    def get_global_positions(self):
        return [[self.global_log[agent_id]["position"]] for agent_id in self.global_log]


class COMAChannelBridge:
    def __init__(self, config: ChannelConfig, channel_seed: int):
        self.channel = ChannelModel(config, channel_seed)

    def build_observations(
        self, wrapper, mapping, agents, num_episode, t, params, batch_memory, mode
    ):
        from actor.transformations import get_network_input as get_actor_input

        log = _CompatibleLog(self.channel, t, wrapper.n_agents)
        positions = []
        global_information = {}
        for agent_id in range(wrapper.n_agents):
            global_information, _, position = agents[agent_id].communicate(
                t, num_episode, log, mode
            )
            positions.append(position)
        log.finalize(positions)

        observations = []
        for agent_id in range(wrapper.n_agents):
            local_information, fused_local_map = agents[agent_id].receive_messages(
                log, agent_id, t
            )
            observation = get_actor_input(
                local_information,
                fused_local_map,
                mapping.simulated_map,
                agent_id,
                t,
                params,
                batch_memory,
                wrapper.agent_state_space,
            )
            batch_memory.add(agent_id, observation=observation)
            observations.append(observation)
        return global_information, positions, observations

    @contextmanager
    def installed(self, wrapper) -> Iterator[None]:
        original = wrapper.build_observations

        def bound(wrapper_self, *args, **kwargs):
            return self.build_observations(wrapper_self, *args, **kwargs)

        wrapper.build_observations = types.MethodType(bound, wrapper)
        try:
            yield
        finally:
            wrapper.build_observations = original
