import hashlib
import json
from pathlib import Path

import yaml

from scripts.s1.run_formal_reproduction import classify_parity


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_preregistration_hashes_and_budget():
    config_path = ROOT / "configs/s1/r1/formal.yaml"
    protocol_path = ROOT / "docs/S1_FORMAL_REPRO_PROTOCOL.md"
    prereg = json.loads(
        (ROOT / "artifacts/s1/r1/formal_repro_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert prereg["config_hash"] == digest(config_path)
    assert prereg["protocol_hash"] == digest(protocol_path)
    assert prereg["total_budget"]["agent_transitions"] == 4_500_000
    assert config["training"]["update_cycles"] == 1500
    assert config["evaluation"]["trials"] == 50


def test_no_test_leakage_and_fixed_final_checkpoint():
    prereg = json.loads(
        (ROOT / "artifacts/s1/r1/formal_repro_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    assert prereg["test_leakage_prohibited"] is True
    assert prereg["formal_checkpoint_selection_uses_test"] is False
    assert prereg["model_selection_rule"] == "FINAL_CHECKPOINT_ONLY"
    seeds = prereg["evaluation"]["trial_seeds"]
    assert seeds == {"start": 20001, "stop_inclusive": 20050}


def test_preregistered_parity_boundaries():
    assert classify_parity(1.0, 1.0) == "STRONG_PARITY"
    assert classify_parity(2.0, 2.0) == "ACCEPTABLE_PARITY"
    assert classify_parity(2.000001, 0.0) == "MAJOR_GAP"
    assert classify_parity(0.0, 2.000001) == "MAJOR_GAP"
