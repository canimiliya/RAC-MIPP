"""Run the frozen S2 evaluator with an original COMA policy adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rac_mipp.benchmark.coma import OriginalCOMAAdapter
from rac_mipp.benchmark.evaluator import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("artifacts/s2/r0/benchmark_contract.json"))
    parser.add_argument("--upstream", type=Path, default=Path(".deps/ipp-marl"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:\AgentData\RAC-MIPP\S1-R1\runs\S1R1-COMA-SEED-20260809-R1\checkpoints\final_actor_state.pt"))
    parser.add_argument("--s1-config", type=Path, default=Path("configs/s1/r1/formal.yaml"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\AgentData\RAC-MIPP\S2-R0\evaluation"))
    parser.add_argument("--role", choices=("VALIDATION", "IID_TEST", "OOD_TEST"), default="IID_TEST")
    parser.add_argument("--purpose", default="final_reporting")
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--acknowledge-final-test", action="store_true")
    parser.add_argument("--communication-hook", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_root = args.output.resolve()
    temp = runtime_root / "tmp"
    cache = runtime_root / "cache"
    for path in (runtime_root, temp, cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "TEMP": str(temp),
        "TMP": str(temp),
        "TORCH_HOME": str(cache / "torch"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    sys.dont_write_bytecode = True
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    seeds = contract["seed_contract"][args.role]["seeds"]
    if args.seed_limit is not None:
        seeds = seeds[: args.seed_limit]
    adapter = OriginalCOMAAdapter(
        checkpoint=args.checkpoint,
        upstream=args.upstream,
        config=args.s1_config,
        log_dir=runtime_root / "tensorboard",
    )
    try:
        contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
        run_metadata = {
            "RUN_ID": f"S2R0-COMA-{args.role}-{len(seeds)}TRIAL",
            "TASK_ID": contract["task_id"],
            "GIT_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, encoding="utf-8").strip(),
            "UPSTREAM_COMMIT": contract["upstream_commit"],
            "CONFIG_HASH": contract_hash,
            "ALGORITHM": adapter.algorithm,
            "ENVIRONMENT": "IPP_MARL_PINNED_SYNTHETIC_MAP",
            "TEAM_SIZE": 4,
            "COMM_DROP": 0.0,
            "COMM_DELAY": "NOT_AVAILABLE_YET",
        }
        summary = evaluate_policy(
            adapter,
            seeds=seeds,
            role=args.role,
            purpose=args.purpose,
            output_dir=runtime_root,
            run_metadata=run_metadata,
            acknowledge_final_test=args.acknowledge_final_test,
            communication_hook=args.communication_hook,
        )
    finally:
        adapter.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
