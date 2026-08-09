"""Recompute the local-only S0 literature integrity and Git safety gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts" / "s0" / "literature_manifest.json"
CATALOG_PATH = ROOT / "docs" / "LITERATURE_CATALOG.md"


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def audit_metadata(manifest: dict) -> list[str]:
    errors: list[str] = []
    papers = manifest.get("papers", [])
    expected_ids = [f"P{index:02d}" for index in range(1, 17)]
    actual_ids = [paper.get("id") for paper in papers]
    if actual_ids != expected_ids:
        errors.append(f"paper IDs differ: {actual_ids!r}")

    status_counts = {status: 0 for status in ("VERIFIED", "CORRECTED", "UNRESOLVED")}
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    for paper in papers:
        paper_id = paper.get("id", "UNKNOWN")
        status = paper.get("metadata_status")
        if status not in status_counts:
            errors.append(f"{paper_id}: invalid metadata_status={status!r}")
            continue
        status_counts[status] += 1
        if status == "CORRECTED" and not paper.get("candidate_metadata_note"):
            errors.append(f"{paper_id}: CORRECTED record lacks correction note")

        row = next(
            (line for line in catalog.splitlines() if line.startswith(f"| {paper_id} |")),
            None,
        )
        if row is None:
            errors.append(f"{paper_id}: catalog row missing")
            continue
        required_values = [
            paper["canonical_title"],
            str(paper["year"]),
            paper["venue"],
            paper["metadata_status"],
            paper["local_relative_path"],
            f"arXiv:{paper['arxiv_id']}",
        ]
        if paper.get("doi"):
            required_values.append(paper["doi"])
        for value in required_values:
            if value not in row:
                errors.append(f"{paper_id}: catalog row lacks {value!r}")

    expected_counts = {
        "verified_count": status_counts["VERIFIED"],
        "corrected_count": status_counts["CORRECTED"],
        "unresolved_count": status_counts["UNRESOLVED"],
        "total_candidate_papers": len(papers),
    }
    for field, expected in expected_counts.items():
        if manifest.get(field) != expected:
            errors.append(
                f"summary {field}={manifest.get(field)!r}, recomputed={expected!r}"
            )
    return errors


def audit_local_pdfs(manifest: dict) -> list[str]:
    errors: list[str] = []
    seen_hashes: set[str] = set()
    for paper in manifest["papers"]:
        paper_id = paper["id"]
        relative_path = paper["local_relative_path"]
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"{paper_id}: missing {relative_path}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        checks = {
            "non-empty": bool(data),
            "%PDF- header": data.startswith(b"%PDF-"),
            "%%EOF trailer": b"%%EOF" in data[-2048:],
            "manifest size": len(data) == paper["file_size_bytes"],
            "manifest SHA-256": digest == paper["sha256"],
            "filename": path.name == paper["filename"],
            "unique SHA-256": digest not in seen_hashes,
        }
        seen_hashes.add(digest)
        for label, passed in checks.items():
            if not passed:
                errors.append(f"{paper_id}: failed {label}")
    return errors


def audit_git_safety(manifest: dict) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    first_pdf = manifest["papers"][0]["local_relative_path"]
    for path in (first_pdf, ".deps/ipp-marl/README.md"):
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", path], cwd=ROOT, check=False
        )
        if completed.returncode != 0:
            errors.append(f"Git ignore check failed for {path}")

    tracked = git("ls-files").splitlines()
    counts = {
        "tracked_pdf_count": sum(path.lower().endswith(".pdf") for path in tracked),
        "tracked_papers_count": sum(path.startswith(".papers/") for path in tracked),
        "tracked_upstream_dependency_count": sum(
            path.startswith(".deps/") for path in tracked
        ),
    }
    for field, value in counts.items():
        if value != 0:
            errors.append(f"{field}={value}, expected 0")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-local-pdfs",
        action="store_true",
        help="also require and hash all ignored local PDF files",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    errors = audit_metadata(manifest)
    if args.require_local_pdfs:
        errors.extend(audit_local_pdfs(manifest))
    git_errors, counts = audit_git_safety(manifest)
    errors.extend(git_errors)

    result = {
        "status": "PASS" if not errors else "FAIL",
        "papers_checked": len(manifest["papers"]),
        "local_pdfs_checked": len(manifest["papers"]) if args.require_local_pdfs else 0,
        **counts,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
