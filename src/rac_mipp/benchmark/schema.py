"""Strict validation for formal run manifests and metric records."""

from __future__ import annotations

import re
from typing import Any


REQUIRED_RUN_FIELDS = (
    "RUN_ID",
    "TASK_ID",
    "GIT_HEAD",
    "UPSTREAM_COMMIT",
    "CONFIG_HASH",
    "SEED",
    "ALGORITHM",
    "ENVIRONMENT",
    "TEAM_SIZE",
    "COMM_DROP",
    "COMM_DELAY",
    "START_TIME",
    "END_TIME",
    "STATUS",
    "PRIMARY_METRICS",
    "ARTIFACT_PATHS",
)

CORE_METRICS = (
    "entropy",
    "F1",
    "mission_return",
    "episode_length",
    "measurement_count",
    "path_length",
)

FUTURE_METRICS = (
    "communication_load",
    "packet_delivery",
    "tail_risk",
    "OOD_gap",
)


def validate_run_manifest(payload: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_RUN_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required run fields: {missing}")
    if payload["STATUS"] not in {"RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"}:
        raise ValueError("invalid STATUS")
    for field, length in (("GIT_HEAD", 40), ("UPSTREAM_COMMIT", 40), ("CONFIG_HASH", 64)):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", str(payload[field])):
            raise ValueError(f"invalid {field}")
    seeds = payload["SEED"] if isinstance(payload["SEED"], list) else [payload["SEED"]]
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise TypeError("SEED must be an integer or a non-empty integer list")
    if not isinstance(payload["TEAM_SIZE"], int) or payload["TEAM_SIZE"] < 1:
        raise TypeError("TEAM_SIZE must be a positive integer")
    if not isinstance(payload["PRIMARY_METRICS"], dict):
        raise TypeError("PRIMARY_METRICS must be an object")
    if not isinstance(payload["ARTIFACT_PATHS"], list):
        raise TypeError("ARTIFACT_PATHS must be a list")


def normalize_episode_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "entropy": "final_entropy",
        "F1": "F1",
        "mission_return": "mission_return",
        "episode_length": "episode_steps",
        "measurement_count": "measurement_count",
        "path_length": "path_length",
    }
    result = {name: raw[source] for name, source in aliases.items()}
    result.update({name: "NOT_AVAILABLE_YET" for name in FUTURE_METRICS})
    if "normalized_communication_load" in raw:
        result["communication_load"] = raw["normalized_communication_load"]
    if "packet_delivery_ratio" in raw:
        result["packet_delivery"] = raw["packet_delivery_ratio"]
    for name in (
        "messages_attempted",
        "messages_range_eligible",
        "messages_dropped",
        "messages_delayed",
        "messages_delivered",
        "effective_neighbor_degree",
        "message_age_mean",
        "message_age_max",
        "communication_radius",
    ):
        if name in raw:
            result[name] = raw[name]
    return result
