"""Preregistered original ipp-marl/COMA formal training and Table I evaluation.

The upstream checkout is never edited. Large runtime products are kept below
D:\\AgentData; only small metrics, plots, and manifests are written to Git.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import yaml


TASK_ID = "S1-R1-ORIGINAL-COMA-FORMAL-REPRODUCTION-R1"
UPSTREAM_COMMIT = "1e9bdc3ba90f707ce79797468f533f5733c65e4b"
START_HEAD = "8b94812233cbd2f95005cd9c103d725d9683d0f8"
PAPER_FIXED_POSITIONS = (
    (10, 10, 15),
    (40, 10, 15),
    (40, 40, 15),
    (10, 40, 15),
)
_COLLECTOR: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "evaluate", "debug"))
    parser.add_argument("--upstream", type=Path, default=Path(".deps/ipp-marl"))
    parser.add_argument("--config", type=Path, default=Path("configs/s1/r1/formal.yaml"))
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("artifacts/s1/r1/formal_repro_preregistration.json"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/S1_FORMAL_REPRO_PROTOCOL.md")
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(r"D:\AgentData\RAC-MIPP\S1-R1"),
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/s1/r1")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-updates", type=int, default=1)
    parser.add_argument(
        "--environment-workers",
        type=int,
        default=None,
        help="Execution-only worker cap; scientific mission RNG/order stay frozen.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def configure_runtime(root: Path, run_id: str) -> dict[str, Path]:
    root = root.resolve()
    paths = {
        "tmp": root / "tmp",
        "cache_torch": root / "cache" / "torch",
        "cache_matplotlib": root / "cache" / "matplotlib",
        "run": root / "runs" / run_id,
    }
    paths["checkpoints"] = paths["run"] / "checkpoints"
    paths["tensorboard"] = paths["run"] / "tensorboard"
    paths["logs"] = paths["run"] / "logs"
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TEMP": str(paths["tmp"]),
            "TMP": str(paths["tmp"]),
            "TORCH_HOME": str(paths["cache_torch"]),
            "MPLCONFIGDIR": str(paths["cache_matplotlib"]),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    sys.dont_write_bytecode = True
    return paths


def configure_upstream(upstream: Path) -> Path:
    upstream = upstream.resolve()
    framework = upstream / "marl_framework"
    if not framework.is_dir():
        raise FileNotFoundError(framework)
    for path in (str(framework), str(upstream)):
        if path not in sys.path:
            sys.path.insert(0, path)
    actual = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit drift: {actual}")
    return framework


def load_and_verify_contract(args: argparse.Namespace) -> tuple[dict, dict]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if sha256(args.config) != prereg["config_hash"]:
        raise RuntimeError("CONFIG_HASH mismatch")
    if sha256(args.protocol) != prereg["protocol_hash"]:
        raise RuntimeError("PROTOCOL_HASH mismatch")
    if prereg["task_id"] != TASK_ID or config["task_id"] != TASK_ID:
        raise RuntimeError("task id mismatch")
    if not prereg["frozen_before_formal_training"]:
        raise RuntimeError("protocol is not frozen")
    return config, prereg


def set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paper_fixed_state(agent_id: int):
    import numpy as np

    return np.array(PAPER_FIXED_POSITIONS[agent_id])


def exact_fast_ground_truth(pk, x_dim: int, y_dim: int, episode: int):
    import numpy as np

    del pk
    # The checked-in generator computes a random field but returns only this
    # axis-aligned field. It resets NumPy immediately before these draws.
    np.random.seed(episode)
    split_idx = np.random.randint(4)
    percentage_idx = np.random.randint(30, 61)
    field = np.zeros((y_dim, x_dim))
    if split_idx == 0:
        field[: int((y_dim * percentage_idx) / 100), :] = 1
    elif split_idx == 1:
        field[int((y_dim * (1 - percentage_idx)) / 100) :, :] = 1
    elif split_idx == 2:
        field[:, : int((x_dim * percentage_idx) / 100)] = 1
    else:
        field[:, int((x_dim * (1 - percentage_idx)) / 100) :] = 1
    return field


def install_adapters() -> list[str]:
    import numpy as np
    import torch
    from marl_framework.agent.state_space import AgentStateSpace
    from marl_framework.batch_memory import BatchMemory
    from marl_framework.mapping import ground_truths
    from mapping import ground_truths as legacy_ground_truths

    original_get = BatchMemory.get

    def compatible_get(self, t: int, agent_id: int, argument: str):
        if argument == "mask":
            return self.transitions[agent_id][t].mask
        return original_get(self, t, agent_id, argument)

    BatchMemory.get = compatible_get

    def fixed_state(self, agent_id: int, episode: int):
        del self, episode
        return paper_fixed_state(agent_id)

    AgentStateSpace.get_random_agent_state = fixed_state

    original_ground_truth = ground_truths.gaussian_random_field

    # Prove exact output and NumPy post-state parity before installing fast path.
    for episode in (1, 17):
        expected = original_ground_truth(lambda k: k ** -5, 37, 41, episode)
        expected_state = np.random.get_state()
        actual = exact_fast_ground_truth(lambda k: k ** -5, 37, 41, episode)
        actual_state = np.random.get_state()
        if not np.array_equal(expected, actual):
            raise RuntimeError("synthetic-map fast path output parity failed")
        if expected_state[0] != actual_state[0] or not np.array_equal(
            expected_state[1], actual_state[1]
        ) or expected_state[2:] != actual_state[2:]:
            raise RuntimeError("synthetic-map fast path RNG parity failed")
    ground_truths.gaussian_random_field = exact_fast_ground_truth
    legacy_ground_truths.gaussian_random_field = exact_fast_ground_truth
    torch.autograd.set_detect_anomaly(False)
    return [
        "BatchMemory mask accessor typo repair",
        "paper-stated fixed four-corner initial positions",
        "exact-output/RNG-parity synthetic-map dead-computation fast path for both upstream import names",
        "TD targets use CriticLearner synchronized target critic",
        "autograd anomaly tracing disabled for formal-run performance",
    ]


def collector_worker_init(upstream: str, params: dict, training_seed: int) -> None:
    """Initialize one persistent CPU-only environment collector process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    upstream_path = Path(upstream).resolve()
    framework = upstream_path / "marl_framework"
    for path in (str(framework), str(upstream_path)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2
    import torch
    from marl_framework.coma_wrapper import COMAWrapper
    from marl_framework.mapping.grid_maps import GridMap
    from marl_framework.missions.episode_generator import EpisodeGenerator
    from marl_framework.sensors import Sensor
    from marl_framework.sensors.models import SensorModel

    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    install_adapters()
    torch.autograd.set_detect_anomaly(False)
    grid = GridMap(params)
    sensor = Sensor(SensorModel(), grid)
    wrapper = COMAWrapper(params, None)
    _COLLECTOR.update(
        {
            "params": params,
            "training_seed": int(training_seed),
            "grid": grid,
            "sensor": sensor,
            "wrapper": wrapper,
            "episode_type": EpisodeGenerator,
        }
    )


def collector_worker_run(task: tuple[list[int], str]) -> dict[str, Any]:
    """Collect a contiguous episode chunk and return CPU transitions."""
    import torch
    from marl_framework.batch_memory import BatchMemory

    episode_indices, actor_path = task
    wrapper = _COLLECTOR["wrapper"]
    wrapper.actor_network.load_state_dict(
        torch.load(actor_path, map_location="cpu", weights_only=True)
    )
    memory = BatchMemory(_COLLECTOR["params"], wrapper)
    generator = _COLLECTOR["episode_type"](
        _COLLECTOR["params"], None, _COLLECTOR["grid"], _COLLECTOR["sensor"]
    )
    episode_rows = []
    for episode_index in episode_indices:
        set_all_seeds((_COLLECTOR["training_seed"] + episode_index) % (2**32))
        result = generator.execute(episode_index, memory, wrapper, "train")
        episode_rows.append(
            {
                "episode": episode_index,
                "return": float(result[0]),
                "absolute_return": float(result[2]),
                "epsilon": float(result[7]),
            }
        )
    transitions = {}
    for agent_id, agent_transitions in memory.transitions.items():
        transitions[agent_id] = [
            (
                item.state.cpu(),
                item.observation.cpu(),
                item.action.cpu(),
                item.mask.cpu(),
                float(item.reward),
                bool(item.done),
                None,
                None,
            )
            for item in agent_transitions
        ]
    return {"episodes": episode_rows, "transitions": transitions}


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def classify_parity(entropy_gap: float, f1_gap: float) -> str:
    if entropy_gap <= 1.0 and f1_gap <= 1.0:
        return "STRONG_PARITY"
    if entropy_gap <= 2.0 and f1_gap <= 2.0:
        return "ACCEPTABLE_PARITY"
    return "MAJOR_GAP"


def build_td_targets_batched(memory, target_critic) -> None:
    """Equivalent TD(lambda) target calculation with batched target inference."""
    import numpy as np
    import torch

    target_critic.eval()
    selected_q: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for agent_id in range(memory.n_agents):
            states = torch.stack(
                [item.state for item in memory.transitions[agent_id]]
            ).squeeze()
            actions = torch.stack(
                [item.action for item in memory.transitions[agent_id]]
            ).long()
            q_values, _ = target_critic.forward(states.to(next(target_critic.parameters()).device))
            chosen = torch.gather(q_values, 1, actions.to(q_values.device)).squeeze(1)
            selected_q[agent_id] = [item.cpu() for item in chosen]

    for agent_id in range(memory.n_agents):
        count = len(memory.transitions[agent_id])
        for t in range(count):
            sum_n_step_returns = torch.tensor([0.0])
            discounted_return = torch.tensor([0.0])
            for n in range(1, count - t + 1):
                leave = False
                n_step_return: Any = 0
                discounted_return = torch.tensor([0.0])
                for offset in range(n):
                    index = t + offset
                    if (not memory.get(index - 1, agent_id, "done")) or index == 0:
                        reward = memory.get(index, agent_id, "reward")
                        n_step_return += np.power(memory.gamma, offset) * reward
                        discounted_return += np.power(memory.gamma, offset) * reward
                    else:
                        leave = True
                        break
                if leave:
                    sum_n_step_returns += np.power(memory.lam, n) * n_step_return
                    break
                if t + n < count and not (
                    memory.get(t + n, agent_id, "done") or t + n + 1 >= count
                ):
                    n_step_return += (
                        np.power(memory.gamma, n) * selected_q[agent_id][t + n]
                    )
                sum_n_step_returns += np.power(memory.lam, n - 1) * n_step_return
            memory.insert(
                t,
                agent_id,
                td_target=(1 - memory.lam) * sum_n_step_returns,
                discounted_return=discounted_return,
            )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(wrapper, completed_updates: int, next_episode: int, manifest: dict):
    import numpy as np
    import torch

    return {
        "task_id": TASK_ID,
        "run_id": manifest["run_id"],
        "git_head": manifest["git_head"],
        "upstream_commit": UPSTREAM_COMMIT,
        "config_hash": manifest["config_hash"],
        "protocol_hash": manifest["protocol_hash"],
        "completed_updates": completed_updates,
        "next_episode": next_episode,
        "actor": wrapper.actor_network.state_dict(),
        "critic": wrapper.critic_network.state_dict(),
        "target_critic": wrapper.critic_learner.target_critic.state_dict(),
        "actor_optimizer": wrapper.actor_learner.optimizer.state_dict(),
        "critic_optimizer": wrapper.critic_learner.optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_checkpoint(wrapper, path: Path, manifest: dict) -> tuple[int, int]:
    import numpy as np
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    for field in ("task_id", "run_id", "git_head", "upstream_commit", "config_hash", "protocol_hash"):
        expected = TASK_ID if field == "task_id" else manifest[field]
        if state[field] != expected:
            raise RuntimeError(f"resume checkpoint {field} mismatch")
    wrapper.actor_network.load_state_dict(state["actor"])
    wrapper.critic_network.load_state_dict(state["critic"])
    wrapper.critic_learner.target_critic.load_state_dict(state["target_critic"])
    wrapper.actor_learner.optimizer.load_state_dict(state["actor_optimizer"])
    wrapper.critic_learner.optimizer.load_state_dict(state["critic_optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["completed_updates"]), int(state["next_episode"])


def training(args, config: dict, prereg: dict, params: dict, paths: dict[str, Path], debug: bool) -> int:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from marl_framework.batch_memory import BatchMemory
    from marl_framework.coma_wrapper import COMAWrapper
    from marl_framework.mapping.grid_maps import GridMap
    from marl_framework.missions.episode_generator import EpisodeGenerator
    from marl_framework.sensors import Sensor
    from marl_framework.sensors.models import SensorModel
    from marl_framework.utils.utils import TransitionCOMA

    if not torch.cuda.is_available():
        raise RuntimeError("formal GPU training requires CUDA")
    # Importing the upstream network modules re-enables anomaly tracing at module
    # scope, so disable it again after all training imports are complete.
    torch.autograd.set_detect_anomaly(False)
    run_id = paths["run"].name
    manifest_path = paths["run"] / "run_manifest.json"
    latest_path = paths["checkpoints"] / "latest_state.pt"
    head = git_head()
    formal = not debug
    if formal and head == START_HEAD:
        raise RuntimeError("formal runner/preregistration must be committed before training")
    configured_workers = int(config["training"]["environment_workers"])
    worker_count = int(args.environment_workers or configured_workers)
    if worker_count < 1 or worker_count > configured_workers:
        raise ValueError(
            f"environment workers must be within [1,{configured_workers}]"
        )
    manifest = {
        "task_id": TASK_ID,
        "run_id": run_id,
        "formal_run": formal,
        "debug_run": debug,
        "git_head": head,
        "upstream_commit": UPSTREAM_COMMIT,
        "config_hash": prereg["config_hash"],
        "protocol_hash": prereg["protocol_hash"],
        "seed": int(config["training_seed"]),
        "start_time": utc_now(),
        "end_time": None,
        "status": "RUNNING",
        "runtime_root": str(paths["run"]),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "configured_environment_workers": configured_workers,
        "effective_environment_workers": worker_count,
        "execution_attempts": [],
    }
    prior_wall_clock_seconds = 0.0
    if args.resume:
        if not manifest_path.is_file() or not latest_path.is_file():
            raise RuntimeError("resume requested but manifest/checkpoint is missing")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest["config_hash"] != prereg["config_hash"] or old_manifest[
            "protocol_hash"
        ] != prereg["protocol_hash"]:
            raise RuntimeError("resume manifest scientific contract mismatch")
        previous_attempts = old_manifest.get("execution_attempts", [])
        if not previous_attempts:
            previous_attempts = [
                {
                    "attempt": 1,
                    "git_head": old_manifest["git_head"],
                    "environment_workers": old_manifest.get(
                        "effective_environment_workers", configured_workers
                    ),
                    "start_time": old_manifest["start_time"],
                    "end_time": old_manifest.get("end_time"),
                    "status": old_manifest.get("status"),
                    "error": old_manifest.get("error"),
                }
            ]
        if old_manifest.get("wall_clock_seconds") is not None:
            prior_wall_clock_seconds = float(old_manifest["wall_clock_seconds"])
        elif old_manifest.get("end_time"):
            prior_wall_clock_seconds = (
                datetime.fromisoformat(old_manifest["end_time"])
                - datetime.fromisoformat(old_manifest["start_time"])
            ).total_seconds()
        manifest["start_time"] = old_manifest["start_time"]
        manifest["resume_time"] = utc_now()
        manifest["git_head"] = old_manifest["git_head"]
        manifest["resume_git_head"] = head
        manifest["execution_attempts"] = previous_attempts
    elif manifest_path.exists():
        raise RuntimeError(f"run directory already exists: {paths['run']}")
    write_json(manifest_path, manifest)

    writer = SummaryWriter(str(paths["tensorboard"]))
    grid = GridMap(params)
    sensor = Sensor(SensorModel(), grid)
    wrapper = COMAWrapper(params, writer)
    memory = BatchMemory(params, wrapper)
    episode = EpisodeGenerator(params, writer, grid, sensor)
    total_updates = int(args.debug_updates if debug else config["training"]["update_cycles"])
    completed_updates = 0
    next_episode = 1
    set_all_seeds(int(config["training_seed"]) + (1 if debug else 0))
    if args.resume:
        completed_updates, next_episode = restore_checkpoint(wrapper, latest_path, manifest)

    attempt_number = len(manifest["execution_attempts"]) + 1
    manifest["execution_attempts"].append(
        {
            "attempt": attempt_number,
            "git_head": head,
            "environment_workers": worker_count,
            "start_time": utc_now(),
            "end_time": None,
            "status": "RUNNING",
            "error": None,
        }
    )
    write_json(manifest_path, manifest)
    metrics_name = (
        "training_metrics.jsonl"
        if attempt_number == 1
        else f"training_metrics_attempt_{attempt_number}.jsonl"
    )
    metrics_path = paths["logs"] / metrics_name
    training_started = time.time()
    collector_actor = paths["run"] / "collector_actor_state.pt"
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=collector_worker_init,
        initargs=(str(args.upstream.resolve()), params, int(config["training_seed"])),
    )
    try:
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics_stream:
            for update_idx in range(completed_updates, total_updates):
                cycle_started = time.time()
                collection_started = time.time()
                returns = []
                absolute_returns = []
                eps = None
                torch.save(wrapper.actor_network.state_dict(), collector_actor)
                episodes_in_collection = int(config["training"]["episodes_per_collection"])
                episode_indices = list(
                    range(next_episode, next_episode + episodes_in_collection)
                )
                quotient, remainder = divmod(episodes_in_collection, worker_count)
                chunk_sizes = [
                    quotient + (1 if worker_id < remainder else 0)
                    for worker_id in range(worker_count)
                ]
                chunks = []
                cursor = 0
                for chunk_size in chunk_sizes:
                    if chunk_size:
                        chunks.append(episode_indices[cursor : cursor + chunk_size])
                        cursor += chunk_size
                tasks = [(chunk, str(collector_actor)) for chunk in chunks]
                collected = list(executor.map(collector_worker_run, tasks))
                episode_rows = [
                    row for chunk_result in collected for row in chunk_result["episodes"]
                ]
                if [row["episode"] for row in episode_rows] != episode_indices:
                    raise RuntimeError("parallel collector episode ordering drift")
                for row in episode_rows:
                    returns.append(row["return"])
                    absolute_returns.append(row["absolute_return"])
                    eps = row["epsilon"]
                for chunk_result in collected:
                    for agent_id, agent_transitions in chunk_result["transitions"].items():
                        memory.transitions[int(agent_id)].extend(
                            TransitionCOMA(
                                fields[0],
                                fields[1],
                                fields[2].to(wrapper.actor_network.device),
                                fields[3].to(wrapper.actor_network.device),
                                fields[4],
                                fields[5],
                                fields[6],
                                fields[7],
                            )
                            for fields in agent_transitions
                        )
                next_episode += episodes_in_collection
                collection_seconds = time.time() - collection_started
                expected = int(config["training"]["transitions_per_collection"])
                if memory.size() != expected:
                    raise RuntimeError(f"collection size {memory.size()} != {expected}")

                target_started = time.time()
                if update_idx % int(params["networks"]["copy_rate"]) == 0:
                    wrapper.critic_learner.target_critic.load_state_dict(
                        wrapper.critic_network.state_dict()
                    )
                    wrapper.critic_learner.target_critic.eval()
                build_td_targets_batched(memory, wrapper.critic_learner.target_critic)
                target_seconds = time.time() - target_started

                optimization_started = time.time()
                critic_metrics = actor_metrics = None
                for data_pass in range(int(params["networks"]["data_passes"])):
                    batches = memory.build_batches()
                    q_values, critic_metrics = wrapper.critic_learner.learn(
                        update_idx, batches, data_pass
                    )
                    _, actor_metrics = wrapper.actor_learner.learn(batches, q_values, eps)
                optimization_seconds = time.time() - optimization_started
                memory.clear()
                completed_updates = update_idx + 1
                transitions = completed_updates * expected
                row = {
                    "update": completed_updates,
                    "agent_transitions": transitions,
                    "missions": next_episode - 1,
                    "epsilon": eps,
                    "return_mean": float(np.mean(returns)),
                    "return_std": float(np.std(returns, ddof=1)),
                    "absolute_return_mean": float(np.mean(absolute_returns)),
                    "critic_loss": scalar(critic_metrics[0]),
                    "actor_loss": scalar(actor_metrics[0]),
                    "collection_seconds": collection_seconds,
                    "td_target_seconds": target_seconds,
                    "optimization_seconds": optimization_seconds,
                    "cycle_seconds": time.time() - cycle_started,
                    "timestamp": utc_now(),
                    "execution_attempt": attempt_number,
                }
                metrics_stream.write(json.dumps(row, sort_keys=True) + "\n")
                writer.add_scalar("formal/train_return_mean", row["return_mean"], completed_updates)
                writer.add_scalar("formal/critic_loss", row["critic_loss"], completed_updates)
                writer.add_scalar("formal/actor_loss", row["actor_loss"], completed_updates)
                writer.flush()

                checkpoint_every = int(config["training"]["checkpoint_every_updates"])
                if completed_updates % checkpoint_every == 0 or completed_updates == total_updates:
                    atomic_torch_save(
                        checkpoint_payload(wrapper, completed_updates, next_episode, manifest),
                        latest_path,
                    )
                print(json.dumps(row, sort_keys=True), flush=True)

        final_actor = paths["checkpoints"] / "final_actor_state.pt"
        torch.save(wrapper.actor_network.state_dict(), final_actor)
        final_full = paths["checkpoints"] / "final_training_state.pt"
        atomic_torch_save(
            checkpoint_payload(wrapper, completed_updates, next_episode, manifest),
            final_full,
        )
        manifest.update(
            {
                "status": "COMPLETED",
                "end_time": utc_now(),
                "completed_updates": completed_updates,
                "agent_transitions": completed_updates
                * int(config["training"]["transitions_per_collection"]),
                "missions": next_episode - 1,
                "wall_clock_seconds": prior_wall_clock_seconds
                + time.time()
                - training_started,
                "final_actor": str(final_actor),
                "final_actor_sha256": sha256(final_actor),
                "final_actor_size": final_actor.stat().st_size,
                "final_training_state": str(final_full),
                "final_training_state_sha256": sha256(final_full),
                "formal_training_completed": formal and completed_updates == 1500,
            }
        )
        manifest["execution_attempts"][-1].update(
            {"status": "COMPLETED", "end_time": manifest["end_time"]}
        )
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update({"status": "FAILED", "end_time": utc_now(), "error": repr(exc)})
        manifest["execution_attempts"][-1].update(
            {"status": "FAILED", "end_time": manifest["end_time"], "error": repr(exc)}
        )
        manifest["wall_clock_seconds"] = (
            prior_wall_clock_seconds + time.time() - training_started
        )
        write_json(manifest_path, manifest)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        writer.flush()
        writer.close()
    return 0


def evaluate_episode(
    params: dict,
    wrapper,
    grid,
    sensor,
    trial_seed: int,
    communication_observer=None,
) -> dict[str, Any]:
    import numpy as np
    from marl_framework.batch_memory import BatchMemory
    from marl_framework.mapping.mappings import Mapping
    from marl_framework.missions.episode_generator import EpisodeGenerator
    from marl_framework.utils.state import get_w_entropy_map
    from marl_framework.utils.utils import get_wrmse

    set_all_seeds(trial_seed)
    episode = EpisodeGenerator(params, wrapper.actor_learner.writer, grid, sensor)
    mapping = Mapping(grid, sensor, params, trial_seed)
    agents = episode.init_agents(mapping, wrapper)
    memory = BatchMemory(params, wrapper)
    current_map = agents[0].local_map.copy()
    trajectory = []
    mission_return = 0.0
    absolute_return = 0.0
    communication_events = 0
    for timestep in range(int(params["experiment"]["constraints"]["budget"]) + 1):
        global_info, positions, _ = wrapper.build_observations(
            mapping, agents, trial_seed, timestep, params, memory, "eval"
        )
        if timestep == 0:
            trajectory.append([position.copy() for position in positions])
        if communication_observer is not None:
            communication_observer.observe_positions(positions, radius=25.0)
        for i, own in enumerate(positions):
            for j, other in enumerate(positions):
                if i != j and np.linalg.norm(own - other) <= 25:
                    communication_events += 1
        (
            memory,
            reward,
            absolute,
            _,
            new_positions,
            _,
            _,
            _,
            next_map,
        ) = wrapper.steps(
            mapping,
            timestep,
            agents,
            current_map,
            trial_seed,
            memory,
            global_info,
            mapping.simulated_map,
            params,
            "eval",
        )
        mission_return += float(reward)
        absolute_return += float(absolute)
        current_map = next_map.copy()
        trajectory.append([position.copy() for position in new_positions])

    entropy_map = get_w_entropy_map(
        None, current_map, mapping.simulated_map, "eval", wrapper.agent_state_space
    )[0]
    positive = mapping.simulated_map == 1
    final_entropy = float(np.sum(np.where(positive, entropy_map, 0)) / np.sum(positive))
    f1 = float(get_wrmse(current_map.copy(), mapping.simulated_map))
    path_length = 0.0
    for timestep in range(1, len(trajectory)):
        for agent_id in range(len(trajectory[timestep])):
            path_length += float(
                np.linalg.norm(
                    trajectory[timestep][agent_id] - trajectory[timestep - 1][agent_id]
                )
            )
    return {
        "seed": trial_seed,
        "final_entropy": final_entropy,
        "F1": f1,
        "mission_return": mission_return,
        "absolute_return": absolute_return,
        "path_length": path_length,
        "communication_events": communication_events,
        "episode_steps": int(params["experiment"]["constraints"]["budget"]) + 1,
        "measurement_count": (int(params["experiment"]["constraints"]["budget"]) + 1)
        * int(params["experiment"]["missions"]["n_agents"]),
        "trajectory": np.asarray(trajectory),
        "ground_truth": mapping.simulated_map,
        "final_belief": current_map,
    }


def make_qualitative_plot(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(result["ground_truth"], cmap="gray", vmin=0, vmax=1, origin="lower", extent=(0, 50, 0, 50))
    axes[0].set_title("Synthetic ground truth")
    axes[1].imshow(result["final_belief"], cmap="viridis", vmin=0, vmax=1, origin="lower", extent=(0, 50, 0, 50))
    axes[1].set_title("Final fused belief")
    axes[2].imshow(result["ground_truth"], cmap="Greys", alpha=0.35, origin="lower", extent=(0, 50, 0, 50))
    trajectory = result["trajectory"]
    for agent_id in range(trajectory.shape[1]):
        axes[2].plot(trajectory[:, agent_id, 0], trajectory[:, agent_id, 1], marker="o", markersize=2, label=f"UAV {agent_id + 1}")
    axes[2].set_title("Formal four-UAV rollout")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluation(args, config: dict, prereg: dict, params: dict, paths: dict[str, Path]) -> int:
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from marl_framework.coma_wrapper import COMAWrapper
    from marl_framework.mapping.grid_maps import GridMap
    from marl_framework.sensors import Sensor
    from marl_framework.sensors.models import SensorModel

    manifest_path = paths["run"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("formal_training_completed") or manifest["status"] != "COMPLETED":
        raise RuntimeError("formal training is not complete")
    if manifest["config_hash"] != prereg["config_hash"] or manifest["protocol_hash"] != prereg["protocol_hash"]:
        raise RuntimeError("training/evaluation contract mismatch")
    checkpoint = Path(manifest["final_actor"])
    if sha256(checkpoint) != manifest["final_actor_sha256"]:
        raise RuntimeError("final actor checkpoint hash mismatch")

    writer = SummaryWriter(str(paths["tensorboard"] / "evaluation"))
    wrapper = COMAWrapper(params, writer)
    wrapper.actor_network.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    wrapper.actor_network.eval()
    grid = GridMap(params)
    sensor = Sensor(SensorModel(), grid)
    start = int(config["evaluation"]["seeds"]["start"])
    stop = int(config["evaluation"]["seeds"]["stop_inclusive"])
    results = []
    qualitative = None
    evaluation_started = time.time()
    for trial_id, seed in enumerate(range(start, stop + 1), start=1):
        result = evaluate_episode(params, wrapper, grid, sensor, seed)
        if trial_id == int(config["evaluation"]["qualitative_trial_id"]):
            qualitative = result
        results.append({key: result[key] for key in ("seed", "final_entropy", "F1", "mission_return", "absolute_return", "path_length", "communication_events", "episode_steps")})
        print(json.dumps({"trial_id": trial_id, **results[-1]}, sort_keys=True), flush=True)
    writer.flush()
    writer.close()

    artifact_root = args.artifact_root.resolve()
    trials_path = artifact_root / "table1_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["trial_id", "seed", "final_entropy", "F1", "mission_return", "absolute_return", "path_length", "communication_events", "episode_steps"]
        csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
        csv_writer.writeheader()
        for trial_id, result in enumerate(results, start=1):
            csv_writer.writerow({"trial_id": trial_id, **result})

    entropies = np.asarray([row["final_entropy"] for row in results])
    f1_values = np.asarray([row["F1"] for row in results])
    targets = config["paper_targets"]
    entropy_gap = abs(float(np.mean(entropies)) - float(targets["entropy_mean"])) / float(targets["entropy_sd"])
    f1_gap = abs(float(np.mean(f1_values)) - float(targets["f1_mean"])) / float(targets["f1_sd"])
    parity = classify_parity(entropy_gap, f1_gap)
    summary = {
        "task_id": TASK_ID,
        "run_id": manifest["run_id"],
        "formal_run": True,
        "test_leakage": False,
        "checkpoint_selection": "FINAL_CHECKPOINT_ONLY",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["final_actor_sha256"],
        "n": len(results),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies, ddof=1)),
        "F1_mean": float(np.mean(f1_values)),
        "F1_std": float(np.std(f1_values, ddof=1)),
        "paper_entropy_mean": float(targets["entropy_mean"]),
        "paper_entropy_std": float(targets["entropy_sd"]),
        "paper_F1_mean": float(targets["f1_mean"]),
        "paper_F1_std": float(targets["f1_sd"]),
        "entropy_absolute_gap": abs(float(np.mean(entropies)) - float(targets["entropy_mean"])),
        "F1_absolute_gap": abs(float(np.mean(f1_values)) - float(targets["f1_mean"])),
        "entropy_standardized_gap": entropy_gap,
        "F1_standardized_gap": f1_gap,
        "quantitative_parity": parity,
        "evaluation_start_time": datetime.fromtimestamp(evaluation_started, timezone.utc).isoformat(),
        "evaluation_end_time": utc_now(),
        "evaluation_wall_clock_seconds": time.time() - evaluation_started,
    }
    write_json(artifact_root / "table1_summary.json", summary)
    plot_path = artifact_root / "small_plots" / "formal_qualitative_rollout.png"
    assert qualitative is not None
    make_qualitative_plot(qualitative, plot_path)
    qualitative_npz = paths["run"] / "qualitative_trial_1.npz"
    np.savez_compressed(
        qualitative_npz,
        ground_truth=qualitative["ground_truth"],
        final_belief=qualitative["final_belief"],
        trajectory=qualitative["trajectory"],
    )
    summary["qualitative_plot"] = str(plot_path)
    summary["qualitative_local_npz"] = str(qualitative_npz)
    summary["qualitative_local_npz_sha256"] = sha256(qualitative_npz)
    write_json(artifact_root / "table1_summary.json", summary)

    formal_summary = {
        "task_id": TASK_ID,
        "status": "COMPLETED" if parity != "MAJOR_GAP" else "BLOCKED_MAJOR_GAP",
        "formal_run_ids": [manifest["run_id"]],
        "git_head_at_training": manifest["git_head"],
        "upstream_commit": UPSTREAM_COMMIT,
        "config_hash": prereg["config_hash"],
        "protocol_hash": prereg["protocol_hash"],
        "training": {
            "seed": manifest["seed"],
            "agent_transitions": manifest["agent_transitions"],
            "missions": manifest["missions"],
            "wall_clock_seconds": manifest["wall_clock_seconds"],
            "gpu": manifest["gpu"],
            "checkpoint": manifest["final_actor"],
            "checkpoint_sha256": manifest["final_actor_sha256"],
            "checkpoint_size": manifest["final_actor_size"],
        },
        "table1": summary,
        "qualitative_reproduction": True,
        "no_algorithm_change": True,
        "test_leakage": False,
        "long_training_started": True,
        "formal_repro_training_started": True,
        "runtime_manifest": str(manifest_path),
    }
    write_json(artifact_root / "formal_repro_summary.json", formal_summary)
    return 0 if parity != "MAJOR_GAP" else 2


def main() -> int:
    args = parse_args()
    config, prereg = load_and_verify_contract(args)
    run_id = config["formal_run_id"]
    if args.mode == "debug":
        run_id = f"DEBUG-{run_id}-{args.debug_updates}UPDATES"
    paths = configure_runtime(args.runtime_root, run_id)
    framework = configure_upstream(args.upstream)
    adapters = install_adapters()
    upstream_params = yaml.safe_load((framework / "params.yaml").read_text(encoding="utf-8"))
    params = deep_update(copy.deepcopy(upstream_params), config["overrides"])
    adapter_evidence = paths["run"] / "compatibility_adapters.json"
    write_json(adapter_evidence, {"task_id": TASK_ID, "adapters": adapters})
    if args.mode == "train":
        return training(args, config, prereg, params, paths, debug=False)
    if args.mode == "debug":
        return training(args, config, prereg, params, paths, debug=True)
    return evaluation(args, config, prereg, params, paths)


if __name__ == "__main__":
    raise SystemExit(main())
