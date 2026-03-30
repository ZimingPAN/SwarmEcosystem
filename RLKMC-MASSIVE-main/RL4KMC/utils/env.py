"""Unified environment variable helpers.

This codebase historically reads env vars ad-hoc via os.environ.get() with
slightly different parsing rules across modules. This module centralizes:

- Presence checks ("explicit override" semantics)
- Bool/int/float parsing with safe defaults

Guiding principles:
- Never throw on bad env values in production paths; fall back to defaults.
- Treat empty strings as "unset".
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


class EnvKeys:
    """Centralized names for commonly used environment variables."""

    # Polling / scheduler tuning
    TASK_LEADER_POLL_INTERVAL_SEC = "TASK_LEADER_POLL_INTERVAL_SEC"
    TASK_QUEUE_POLL_INTERVAL_SEC = "TASK_QUEUE_POLL_INTERVAL_SEC"
    TASK_NODE_MP_EXIT_SCAN_SEC = "TASK_NODE_MP_EXIT_SCAN_SEC"
    TASK_LOCAL_NUM_WORKERS = "TASK_LOCAL_NUM_WORKERS"
    TASK_LOCAL_QUEUE_CAP = "TASK_LOCAL_QUEUE_CAP"
    TASK_LOCAL_RESULT_SLOT_BYTES = "TASK_LOCAL_RESULT_SLOT_BYTES"

    TASK_LOCAL_WORKER_ID = "TASK_LOCAL_WORKER_ID"
    TASK_WORKER_QUEUE_LOG_SEC = "TASK_WORKER_QUEUE_LOG_SEC"

    TASK_NODE_MP_TERMINATE_TIMEOUT_SEC = "TASK_NODE_MP_TERMINATE_TIMEOUT_SEC"
    TASK_NODE_MP_KILL_TIMEOUT_SEC = "TASK_NODE_MP_KILL_TIMEOUT_SEC"
    TASK_NODE_MP_JOIN_SEC = "TASK_NODE_MP_JOIN_SEC"
    TASK_LEADER_STALL_LOG_SEC = "TASK_LEADER_STALL_LOG_SEC"
    TASK_WORKER_PARENT_WATCH_SEC = "TASK_WORKER_PARENT_WATCH_SEC"

    # Affinity / CPU pinning debug
    TASK_AFFINITY_DEBUG_LOG = "TASK_AFFINITY_DEBUG_LOG"

    # Adaptive polling feature flags
    TASK_DISABLE_ADAPTIVE_POLLING = "TASK_DISABLE_ADAPTIVE_POLLING"
    TASK_DISABLE_ADAPTIVE_LEADER_POLLING = "TASK_DISABLE_ADAPTIVE_LEADER_POLLING"
    TASK_DISABLE_ADAPTIVE_WORKER_POLLING = "TASK_DISABLE_ADAPTIVE_WORKER_POLLING"
    TASK_ENABLE_ADAPTIVE_POLLING = "TASK_ENABLE_ADAPTIVE_POLLING"
    TASK_ENABLE_ADAPTIVE_LEADER_POLLING = "TASK_ENABLE_ADAPTIVE_LEADER_POLLING"
    TASK_ENABLE_ADAPTIVE_WORKER_POLLING = "TASK_ENABLE_ADAPTIVE_WORKER_POLLING"

    # Adaptive polling knobs
    TASK_ADAPTIVE_LEADER_POLL_REF_WORKERS = "TASK_ADAPTIVE_LEADER_POLL_REF_WORKERS"
    TASK_ADAPTIVE_LEADER_POLL_MAX_SEC = "TASK_ADAPTIVE_LEADER_POLL_MAX_SEC"
    TASK_ADAPTIVE_WORKER_POLL_REF_WORKERS = "TASK_ADAPTIVE_WORKER_POLL_REF_WORKERS"
    TASK_ADAPTIVE_WORKER_POLL_MAX_SEC = "TASK_ADAPTIVE_WORKER_POLL_MAX_SEC"

    # Logging / output
    KMC_LOG_LEVEL = "KMC_LOG_LEVEL"
    LOG_LEVEL = "LOG_LEVEL"
    KMC_LOG_DIR = "KMC_LOG_DIR"
    TASK_RUN_OUTPUT_DIR = "TASK_RUN_OUTPUT_DIR"
    KMC_LOG_FLUSH_EVERY = "KMC_LOG_FLUSH_EVERY"
    KMC_LOG_BUFFER_BYTES = "KMC_LOG_BUFFER_BYTES"

    # MPI / distributed
    RANK = "RANK"
    LOCAL_RANK = "LOCAL_RANK"
    WORLD_SIZE = "WORLD_SIZE"
    SLURM_JOB_NUM_NODES = "SLURM_JOB_NUM_NODES"
    DIST_BACKEND = "DIST_BACKEND"
    KMC_MPI_SANITY_CHECK = "KMC_MPI_SANITY_CHECK"
    TASK_FINALIZE_MPI = "TASK_FINALIZE_MPI"
    TASK_FINALIZE_BARRIER_TIMEOUT_SEC = "TASK_FINALIZE_BARRIER_TIMEOUT_SEC"
    TASK_FINALIZE_BARRIER_MODE = "TASK_FINALIZE_BARRIER_MODE"

    # Task scheduler run id
    TASK_RUN_ID = "TASK_RUN_ID"

    # mpi_rma params
    TASK_BATCH_SIZE = "TASK_BATCH_SIZE"
    TASK_LEASE_TIMEOUT_SEC = "TASK_LEASE_TIMEOUT_SEC"
    TASK_LEASE_RENEW_SEC = "TASK_LEASE_RENEW_SEC"
    TASK_HEARTBEAT_SEC = "TASK_HEARTBEAT_SEC"
    TASK_HEARTBEAT_TIMEOUT_SEC = "TASK_HEARTBEAT_TIMEOUT_SEC"
    TASK_RECLAIM_SCAN_INTERVAL_SEC = "TASK_RECLAIM_SCAN_INTERVAL_SEC"
    TASK_RMA_OWNER_RANK = "TASK_RMA_OWNER_RANK"
    TASK_RMA_DONE_CHECK_INTERVAL_SEC = "TASK_RMA_DONE_CHECK_INTERVAL_SEC"
    TASK_LOCAL_QUEUE_HI_WATERMARK_RATIO = "TASK_LOCAL_QUEUE_HI_WATERMARK_RATIO"

    # Local queue / chunk pipelining
    TASK_PREFETCH_CHUNKS = "TASK_PREFETCH_CHUNKS"

    # Debug / tracing
    TASK_SCHEDULER_TRACE = "TASK_SCHEDULER_TRACE"
    TASK_SCHEDULER_DEBUG = "TASK_SCHEDULER_DEBUG"

    # mpi_rma implementation toggles
    TASK_RMA_LOCK_ALL = "TASK_RMA_LOCK_ALL"

    # Misc config
    LATTICE_BASE = "LATTICE_BASE"
    TASK_NUM_GROUPS = "TASK_NUM_GROUPS"

    # Timing / diagnostics
    # Group size for hierarchical MPI timing summary aggregation.
    # Large jobs (e.g. ~20000 ranks) should avoid a flat gather to rank0.
    TIMING_SUMMARY_GROUP_SIZE = "TIMING_SUMMARY_GROUP_SIZE"

    # Embedding
    EMBED_SDPA_IMPL = "EMBED_SDPA_IMPL"
    EMBED_CHUNK_SIZE = "EMBED_CHUNK_SIZE"
    EMBED_CHUNK_AUTO = "EMBED_CHUNK_AUTO"


KNOWN_ENV_KEYS: tuple[str, ...] = (
    EnvKeys.TASK_LEADER_POLL_INTERVAL_SEC,
    EnvKeys.TASK_QUEUE_POLL_INTERVAL_SEC,
    EnvKeys.TASK_NODE_MP_EXIT_SCAN_SEC,
    EnvKeys.TASK_LOCAL_NUM_WORKERS,
    EnvKeys.TASK_LOCAL_QUEUE_CAP,
    EnvKeys.TASK_LOCAL_RESULT_SLOT_BYTES,
    EnvKeys.TASK_LOCAL_WORKER_ID,
    EnvKeys.TASK_WORKER_QUEUE_LOG_SEC,
    EnvKeys.TASK_NODE_MP_TERMINATE_TIMEOUT_SEC,
    EnvKeys.TASK_NODE_MP_KILL_TIMEOUT_SEC,
    EnvKeys.TASK_NODE_MP_JOIN_SEC,
    EnvKeys.TASK_LEADER_STALL_LOG_SEC,
    EnvKeys.TASK_WORKER_PARENT_WATCH_SEC,

    EnvKeys.TASK_AFFINITY_DEBUG_LOG,
    EnvKeys.TASK_DISABLE_ADAPTIVE_POLLING,
    EnvKeys.TASK_DISABLE_ADAPTIVE_LEADER_POLLING,
    EnvKeys.TASK_DISABLE_ADAPTIVE_WORKER_POLLING,
    EnvKeys.TASK_ENABLE_ADAPTIVE_POLLING,
    EnvKeys.TASK_ENABLE_ADAPTIVE_LEADER_POLLING,
    EnvKeys.TASK_ENABLE_ADAPTIVE_WORKER_POLLING,
    EnvKeys.TASK_ADAPTIVE_LEADER_POLL_REF_WORKERS,
    EnvKeys.TASK_ADAPTIVE_LEADER_POLL_MAX_SEC,
    EnvKeys.TASK_ADAPTIVE_WORKER_POLL_REF_WORKERS,
    EnvKeys.TASK_ADAPTIVE_WORKER_POLL_MAX_SEC,

    EnvKeys.KMC_LOG_LEVEL,
    EnvKeys.LOG_LEVEL,
    EnvKeys.KMC_LOG_DIR,
    EnvKeys.TASK_RUN_OUTPUT_DIR,
    EnvKeys.KMC_LOG_FLUSH_EVERY,
    EnvKeys.KMC_LOG_BUFFER_BYTES,

    EnvKeys.RANK,
    EnvKeys.LOCAL_RANK,
    EnvKeys.WORLD_SIZE,
    EnvKeys.SLURM_JOB_NUM_NODES,
    EnvKeys.DIST_BACKEND,
    EnvKeys.KMC_MPI_SANITY_CHECK,
    EnvKeys.TASK_FINALIZE_MPI,
    EnvKeys.TASK_FINALIZE_BARRIER_TIMEOUT_SEC,
    EnvKeys.TASK_FINALIZE_BARRIER_MODE,

    EnvKeys.TASK_RUN_ID,

    EnvKeys.TASK_BATCH_SIZE,
    EnvKeys.TASK_PREFETCH_CHUNKS,
    EnvKeys.TASK_LEASE_TIMEOUT_SEC,
    EnvKeys.TASK_LEASE_RENEW_SEC,
    EnvKeys.TASK_HEARTBEAT_SEC,
    EnvKeys.TASK_HEARTBEAT_TIMEOUT_SEC,
    EnvKeys.TASK_RECLAIM_SCAN_INTERVAL_SEC,
    EnvKeys.TASK_RMA_OWNER_RANK,
    EnvKeys.TASK_RMA_DONE_CHECK_INTERVAL_SEC,
    EnvKeys.TASK_LOCAL_QUEUE_HI_WATERMARK_RATIO,

    EnvKeys.TASK_SCHEDULER_TRACE,
    EnvKeys.TASK_SCHEDULER_DEBUG,
    EnvKeys.TASK_RMA_LOCK_ALL,

    EnvKeys.LATTICE_BASE,
    EnvKeys.TASK_NUM_GROUPS,

    EnvKeys.TIMING_SUMMARY_GROUP_SIZE,

    EnvKeys.EMBED_SDPA_IMPL,
    EnvKeys.EMBED_CHUNK_SIZE,
    EnvKeys.EMBED_CHUNK_AUTO,
)


def env_has(name: str) -> bool:
    """Return True if env var exists and is non-empty after stripping."""

    try:
        return str(os.environ.get(name, "") or "").strip() != ""
    except Exception:
        return False


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    if not env_has(name):
        return default
    try:
        return str(os.environ.get(name, "") or "").strip()
    except Exception:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var.

    Accepts: 1/0, true/false, yes/no, on/off (case-insensitive).
    Unknown values fall back to default.
    """

    raw = env_str(name, None)
    if raw is None:
        return bool(default)
    v = str(raw).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return bool(default)


def env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw = env_str(name, None)
    if raw is None:
        v = int(default)
    else:
        try:
            v = int(str(raw).strip())
        except Exception:
            v = int(default)
    if min_value is not None:
        v = max(int(min_value), int(v))
    if max_value is not None:
        v = min(int(max_value), int(v))
    return int(v)


def env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = env_str(name, None)
    if raw is None:
        v = float(default)
    else:
        try:
            v = float(str(raw).strip())
        except Exception:
            v = float(default)
    if min_value is not None:
        v = max(float(min_value), float(v))
    if max_value is not None:
        v = min(float(max_value), float(v))
    return float(v)


def dump_known_env(
    *,
    keys: Iterable[str] | None = None,
    include_unset: bool = False,
    include_values: bool = True,
    max_value_chars: int = 200,
) -> str:
    """Return a human-readable dump of known env vars.

    By default, only prints variables that are explicitly set (non-empty).
    """

    use_keys = list(keys) if keys is not None else list(KNOWN_ENV_KEYS)
    lines: list[str] = []
    for k in use_keys:
        is_set = env_has(str(k))
        if not include_unset and not is_set:
            continue
        if not include_values:
            lines.append(f"{str(k)}={'SET' if is_set else 'UNSET'}")
            continue

        raw = os.environ.get(str(k), "")
        try:
            s = str(raw)
        except Exception:
            s = "<unprintable>"
        s = s.strip()
        if len(s) > int(max_value_chars):
            s = s[: int(max_value_chars)] + "…"
        if not is_set:
            s = ""
        lines.append(f"{str(k)}={s}")
    return "\n".join(lines)


__all__ = [
    "EnvKeys",
    "KNOWN_ENV_KEYS",
    "env_has",
    "env_str",
    "env_flag",
    "env_int",
    "env_float",
    "dump_known_env",
]
