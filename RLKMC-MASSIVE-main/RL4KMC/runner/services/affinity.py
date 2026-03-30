from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, cast

_LOGGER = logging.getLogger(__name__)


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def parse_cpulist(cpulist: str) -> list[int]:
    """Parse Linux cpulist format, e.g. "0-3,8,10-12"."""

    s = str(cpulist).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except Exception:
                continue
            if end < start:
                start, end = end, start
            out.extend(list(range(start, end + 1)))
        else:
            try:
                out.append(int(part))
            except Exception:
                continue

    # de-dup while preserving order
    seen: set[int] = set()
    uniq: list[int] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(int(c))
    return uniq


def linux_numa_nodes() -> list[int]:
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return []
    nodes: list[int] = []
    for name in os.listdir(base):
        m = re.match(r"node(\d+)$", name)
        if not m:
            continue
        try:
            if not os.path.isdir(os.path.join(base, name)):
                continue
        except Exception:
            pass
        nodes.append(int(m.group(1)))
    return sorted(nodes)


def linux_cpus_of_numa(node_id: int) -> list[int]:
    path = f"/sys/devices/system/node/node{int(node_id)}/cpulist"
    txt = _read_text(path)
    if txt is None:
        return []
    return parse_cpulist(txt)


def linux_thread_siblings(cpu_id: int) -> list[int]:
    """Best-effort SMT thread siblings for a logical CPU (Linux).

    Returns a list of CPU IDs that share the same physical core.
    """

    path = f"/sys/devices/system/cpu/cpu{int(cpu_id)}/topology/thread_siblings_list"
    txt = _read_text(path)
    if not txt:
        return [int(cpu_id)]
    sibs = parse_cpulist(txt)
    return sibs or [int(cpu_id)]


def all_online_cpus() -> list[int]:
    n = os.cpu_count() or 1
    return list(range(int(n)))


def pin_current_process(cpus: Sequence[int]) -> None:
    """Pin current process to a CPU set (Linux)."""
    if not cpus:
        return
    setter = getattr(os, "sched_setaffinity", None)
    if not callable(setter):
        _LOGGER.warning(
            "==== Warning: os.sched_setaffinity not available, worker bind disabled ==="
        )
        return
    setter(0, set(int(c) for c in cpus))


@dataclass(frozen=True)
class PinPlan:
    rank_cpus: list[int]
    worker_cpu_sets: list[list[int]]


@dataclass(frozen=True)
class CpuTopology:
    """Best-effort CPU topology snapshot (Linux)."""

    numa_nodes: list[int]
    cpus_by_numa: dict[int, list[int]]
    online_cpus: list[int]
    total_logical_cpus: int


def read_linux_cpu_topology() -> CpuTopology:
    """Read NUMA topology and online CPU list from /sys (Linux best-effort)."""

    nodes = linux_numa_nodes()
    cpus_by_numa: dict[int, list[int]] = {}
    if nodes:
        for nid in nodes:
            cpus = linux_cpus_of_numa(int(nid))
            cpus_by_numa[int(nid)] = [int(c) for c in cpus]
    else:
        nodes = [0]
        cpus_by_numa[0] = [int(c) for c in all_online_cpus()]

    online = all_online_cpus()
    return CpuTopology(
        numa_nodes=[int(n) for n in nodes],
        cpus_by_numa={int(k): [int(c) for c in v] for k, v in cpus_by_numa.items()},
        online_cpus=[int(c) for c in online],
        total_logical_cpus=int(len(online)),
    )


def read_current_rank_affinity() -> list[int]:
    """Read current process CPU affinity (Linux best-effort)."""

    getter = getattr(os, "sched_getaffinity", None)
    if not callable(getter):
        return [int(c) for c in all_online_cpus()]
    try:
        raw = cast(Iterable[int], getter(0))
        cpus = sorted(int(c) for c in raw)
        if cpus:
            return cpus
    except Exception:
        pass
    return [int(c) for c in all_online_cpus()]


def build_pin_plan(
    *,
    workers_per_rank: int,
    cores_per_worker: int,
    pin_policy: str = "spread",
) -> PinPlan:
    """Build per-worker CPU sets from current rank affinity CPUs.
    pin_policy:
      - spread: spread workers evenly over `rank_cpus`
      - compact: contiguous chunks over `rank_cpus`
    """

    rank_cpus = read_current_rank_affinity()
    assert (
        workers_per_rank * cores_per_worker > 0
        and workers_per_rank * cores_per_worker <= len(rank_cpus)
    ), (
        f"not enough CPUs in rank_cpus for workers_per_rank={workers_per_rank} cores_per_worker={cores_per_worker} "
        f"rank_cpus={rank_cpus}"
    )
    # de-dup while preserving order
    rank_seq: list[int] = []
    seen: set[int] = set()
    for c in rank_cpus:
        ci = int(c)
        if ci in seen:
            continue
        seen.add(ci)
        rank_seq.append(ci)
    if not rank_seq:
        rank_seq = [int(c) for c in all_online_cpus()]
    if not rank_seq:
        rank_seq = [0]

    _LOGGER.debug(
        f"build_pin_plan inputs: workers_per_rank={workers_per_rank} pin_policy={pin_policy} "
        f"rank_cpus_n={len(rank_seq)}"
    )

    if pin_policy not in {"spread", "compact"}:
        raise ValueError(f"invalid pin_policy: {pin_policy}")

    worker_sets: list[list[int]] = []

    for w in range(workers_per_rank):
        # For spread, we want to distribute workers as evenly as possible across the available CPUs.
        # For compact, we simply take contiguous chunks of CPUs for each worker.
        # Example: if rank_seq=[0,1,2,3,4,5], workers_per_rank=2, cores_per_worker=2:
        # - spread: worker 0 gets [0,1], worker 1 gets [3,4]
        # - compact: worker 0 gets [0,1], worker 1 gets [2,3]
        if pin_policy == "spread":
            # Keep each worker's CPUs contiguous, but spread the start offsets
            # as evenly as possible across the full rank CPU range.
            start = (w * len(rank_seq)) // workers_per_rank
            max_start = len(rank_seq) - cores_per_worker
            if start > max_start:
                start = max_start
            end = start + cores_per_worker
            worker_cpus = [int(c) for c in rank_seq[start:end]]
        else:  # compact
            start = w * cores_per_worker
            end = start + cores_per_worker
            worker_cpus = [int(c) for c in rank_seq[start:end]]
        worker_sets.append(worker_cpus)

    return PinPlan(rank_cpus=list(rank_seq), worker_cpu_sets=worker_sets)
