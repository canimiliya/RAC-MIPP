from __future__ import annotations

import sys
import types

import numpy as np

from rac_mipp.communication import ChannelConfig, COMAChannelBridge


class FakeMemory:
    def __init__(self):
        self.rows = []

    def add(self, agent_id, observation):
        self.rows.append((agent_id, observation))


class FakeMapping:
    simulated_map = np.zeros((1, 1))

    def fuse_map(self, local_map, received, agent_id, mode):
        del local_map, agent_id, mode
        return received


class FakeAgent:
    def __init__(self, agent_id, position):
        self.agent_id = agent_id
        self.position = np.asarray(position, dtype=float)
        self.received = None

    def communicate(self, t, num_episode, log, mode):
        del num_episode, mode
        message = {
            "value": 1000 + t if self.agent_id else t,
            "position": self.position.copy(),
        }
        global_log = log.store_agent_message(message, self.agent_id)
        return global_log, {}, self.position

    def receive_messages(self, log, agent_id, t):
        del t
        self.received = log.get_messages(self.agent_id)
        return self.received, FakeMapping().fuse_map({}, self.received, agent_id, "local")


def test_future_leakage_is_blocked_through_coma_observation_fusion(monkeypatch):
    transformations = types.ModuleType("actor.transformations")
    transformations.get_network_input = lambda local_information, *args: local_information
    actor = types.ModuleType("actor")
    monkeypatch.setitem(sys.modules, "actor", actor)
    monkeypatch.setitem(sys.modules, "actor.transformations", transformations)

    bridge = COMAChannelBridge(ChannelConfig(0, 3, 25), 77)
    wrapper = types.SimpleNamespace(n_agents=2, agent_state_space=object())
    agents = [FakeAgent(0, [0, 0, 5]), FakeAgent(1, [1, 0, 5])]
    memory = FakeMemory()
    for step in (10, 11, 12, 13):
        bridge.build_observations(
            wrapper, FakeMapping(), agents, 15001, step, {}, memory, "eval"
        )
    assert agents[1].received[0]["value"] == 10
    assert agents[1].received[0]["value"] != 13
    delivered = [
        event for event in bridge.channel.events
        if event.event == "DELIVERED" and event.sender_id == 0 and event.receiver_id == 1
    ]
    assert delivered[0].send_step == 10
    assert delivered[0].step == 13
    assert delivered[0].message_age == 3
