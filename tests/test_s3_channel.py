from __future__ import annotations

import copy

import numpy as np
import pytest

from rac_mipp.communication import ChannelConfig, ChannelModel
from rac_mipp.benchmark.schema import normalize_episode_metrics


def payload(value: int) -> dict:
    return {"value": value, "position": [float(value), 0.0, 5.0]}


def test_config_boundaries():
    ChannelConfig(0.0, 0, 0.0)
    ChannelConfig(1.0, 5, 25.0)
    with pytest.raises(ValueError):
        ChannelConfig(-0.1, 0, 25.0)
    with pytest.raises(ValueError):
        ChannelConfig(0.0, -1, 25.0)


@pytest.mark.parametrize(
    ("distance", "eligible"), [(24.999, True), (25.0, True), (25.001, False)]
)
def test_send_time_radius_boundary(distance, eligible):
    channel = ChannelModel(ChannelConfig(0, 0, 25), 7)
    positions = [np.array([0, 0, 5]), np.array([distance, 0, 5])]
    delivered = channel.exchange(0, {0: payload(0), 1: payload(1)}, positions)
    assert (1 in delivered[0]) is eligible
    assert channel.summary(n_agents=2, episode_steps=1)["messages_range_eligible"] == (
        2 if eligible else 0
    )


def test_neighbor_mask_is_directed_non_self_and_inclusive():
    channel = ChannelModel(ChannelConfig(0, 0, 25), 7)
    mask = channel.neighbor_mask(
        [np.array([0, 0, 5]), np.array([25, 0, 5]), np.array([30, 0, 5])]
    )
    assert mask.tolist() == [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]


def test_drop_zero_and_one_are_real_and_no_self_messages():
    positions = [np.array([0, 0, 5]), np.array([1, 0, 5])]
    good = ChannelModel(ChannelConfig(0, 0, 25), 1)
    bad = ChannelModel(ChannelConfig(1, 0, 25), 1)
    assert good.exchange(0, {0: payload(0), 1: payload(1)}, positions) == {
        0: {1: payload(1)},
        1: {0: payload(0)},
    }
    assert bad.exchange(0, {0: payload(0), 1: payload(1)}, positions) == {0: {}, 1: {}}
    assert all(event.sender_id != event.receiver_id for event in good.events)
    assert bad.summary(n_agents=2, episode_steps=1)["messages_dropped"] == 2


@pytest.mark.parametrize("delay", [0, 1, 5])
def test_delay_delivery_time_and_send_snapshot(delay):
    channel = ChannelModel(ChannelConfig(0, delay, 25), 2)
    positions = [np.array([0, 0, 5]), np.array([1, 0, 5])]
    first = payload(10)
    deliveries = channel.exchange(10, {0: first, 1: payload(20)}, positions)
    first["value"] = 999
    if delay:
        assert deliveries[1] == {}
        for step in range(11, 10 + delay):
            channel.exchange(step, {0: payload(step), 1: payload(20)}, positions)
        deliveries = channel.exchange(
            10 + delay, {0: payload(10 + delay), 1: payload(20)}, positions
        )
    assert deliveries[1][0]["value"] == 10


def test_future_information_leakage_regression_t10_to_t13():
    channel = ChannelModel(ChannelConfig(0, 3, 25), 3)
    positions = [np.array([0, 0, 5]), np.array([1, 0, 5])]
    channel.exchange(10, {0: payload(10), 1: payload(-1)}, positions)
    channel.exchange(11, {0: payload(11), 1: payload(-1)}, positions)
    channel.exchange(12, {0: payload(12), 1: payload(-1)}, positions)
    received = channel.exchange(13, {0: payload(13), 1: payload(-1)}, positions)
    assert received[1][0]["value"] == 10
    delivered = [e for e in channel.events if e.event == "DELIVERED" and e.receiver_id == 1]
    assert delivered[0].send_step == 10
    assert delivered[0].message_age == 3


def test_channel_determinism_and_seed_variation():
    config = ChannelConfig(0.5, 1, 25)
    positions = [np.array([0, 0, 5]), np.array([1, 0, 5]), np.array([2, 0, 5])]

    def run(seed):
        channel = ChannelModel(config, seed)
        for step in range(8):
            channel.exchange(step, {i: payload(step * 10 + i) for i in range(3)}, positions)
        return channel.event_records()

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_queue_reset_and_episode_isolation():
    channel = ChannelModel(ChannelConfig(0, 5, 25), 4)
    positions = [np.array([0, 0, 5]), np.array([1, 0, 5])]
    channel.exchange(0, {0: payload(0), 1: payload(1)}, positions)
    assert channel.summary(n_agents=2, episode_steps=1)[
        "pending_messages_discarded_at_episode_end"
    ] == 2
    channel.reset()
    assert channel.events == []
    assert channel.exchange(5, {0: payload(5), 1: payload(6)}, positions) == {0: {}, 1: {}}


def test_multiple_message_order_is_stable():
    channel = ChannelModel(ChannelConfig(0, 1, 25), 5)
    positions = [np.array([0, 0, 5]), np.array([1, 0, 5]), np.array([2, 0, 5])]
    payloads = {i: payload(i) for i in range(3)}
    channel.exchange(0, payloads, positions)
    delivered = channel.exchange(1, payloads, positions)
    assert list(delivered[2]) == [0, 1]
    delivery_events = [event for event in channel.events if event.event == "DELIVERED"]
    keys = [(e.delivery_step, e.send_step, e.sender_id, e.receiver_id) for e in delivery_events]
    assert keys == sorted(keys)


def test_channel_rng_never_changes_global_numpy_state():
    np.random.seed(123)
    before = copy.deepcopy(np.random.get_state())
    channel = ChannelModel(ChannelConfig(0.5, 1, 25), 99)
    channel.exchange(
        0,
        {0: payload(0), 1: payload(1)},
        [np.array([0, 0, 5]), np.array([1, 0, 5])],
    )
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_s2_metric_schema_exposes_real_channel_events():
    metrics = normalize_episode_metrics(
        {
            "final_entropy": 0.2, "F1": 0.8, "mission_return": 1.0,
            "episode_steps": 15, "measurement_count": 60, "path_length": 10.0,
            "normalized_communication_load": 0.75, "packet_delivery_ratio": 0.5,
            "messages_attempted": 12, "messages_range_eligible": 9,
            "messages_dropped": 3, "messages_delayed": 6, "messages_delivered": 6,
            "effective_neighbor_degree": 1.5, "message_age_mean": 3.0,
            "message_age_max": 3, "communication_radius": 25.0,
        }
    )
    assert metrics["communication_load"] == 0.75
    assert metrics["packet_delivery"] == 0.5
    assert metrics["messages_delivered"] == 6
    assert metrics["tail_risk"] == "NOT_AVAILABLE_YET"
