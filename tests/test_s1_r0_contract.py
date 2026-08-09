import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_s1_smoke_config_is_explicitly_non_formal():
    text = (ROOT / "configs/s1/r0/smoke.yaml").read_text(encoding="utf-8")
    assert "smoke_only: true" in text
    assert "not_paper_result: true" in text
    assert "budget: 1" in text


def test_s1_contract_artifact_has_provenance_classes():
    artifact = json.loads(
        (ROOT / "artifacts/s1/r0/original_repro_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["task_id"] == "S1-R0-UPSTREAM-REPRO-CONTRACT-ENV-SMOKE-R1"
    assert artifact["upstream_commit"] == "1e9bdc3ba90f707ce79797468f533f5733c65e4b"
    assert set(artifact["provenance_classes"]) == {
        "PAPER_STATED",
        "CODE_DERIVED",
        "INFERRED",
        "UNKNOWN",
    }
    assert artifact["success_definition"]["smoke_is_not_paper_result"] is True
