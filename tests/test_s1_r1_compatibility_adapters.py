from pathlib import Path

import numpy as np
import pytest

from scripts.s1.run_formal_reproduction import (
    configure_upstream,
    exact_fast_ground_truth,
    install_adapters,
    paper_fixed_state,
)


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".deps/ipp-marl"


def test_paper_fixed_positions_are_exact():
    expected = (
        (10, 10, 15),
        (40, 10, 15),
        (40, 40, 15),
        (10, 40, 15),
    )
    assert tuple(tuple(paper_fixed_state(agent_id)) for agent_id in range(4)) == expected


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="local-only frozen upstream absent")
def test_synthetic_fast_path_matches_upstream_output_and_rng_state():
    configure_upstream(UPSTREAM)
    from marl_framework.mapping import ground_truths

    original = ground_truths.gaussian_random_field
    for episode in (1, 17, 20001):
        expected = original(lambda k: k ** -5, 37, 41, episode)
        expected_state = np.random.get_state()
        actual = exact_fast_ground_truth(lambda k: k ** -5, 37, 41, episode)
        actual_state = np.random.get_state()
        assert np.array_equal(actual, expected)
        assert actual_state[0] == expected_state[0]
        assert np.array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="local-only frozen upstream absent")
def test_mask_adapter_returns_namedtuple_mask_field():
    pytest.importorskip("torch")
    configure_upstream(UPSTREAM)
    from marl_framework.batch_memory import BatchMemory
    from utils.utils import TransitionCOMA

    install_adapters()
    memory = object.__new__(BatchMemory)
    memory.transitions = {
        0: [TransitionCOMA(None, None, None, "MASK_SENTINEL", None, None, None, None)]
    }
    assert memory.get(0, 0, "mask") == "MASK_SENTINEL"
