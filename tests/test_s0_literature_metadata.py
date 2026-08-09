import json
import subprocess
import sys
from pathlib import Path


def test_literature_manifest_matches_catalog_and_summary():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/audit_s0_literature.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)
    assert completed.returncode == 0, result
    assert result["status"] == "PASS"
    assert result["papers_checked"] == 16
    assert result["errors"] == []
