from __future__ import annotations
import socket
import time
import logging
from typing import Any, Literal, Protocol

_LOGGER = logging.getLogger(__name__)
CommOp = Literal["sum"]


class CommBackend(Protocol):
    def init(self) -> None: ...

    def finalize(self) -> None: ...

    def is_initialized(self) -> bool: ...

    def is_finalized(self) -> bool: ...

    def rank(self) -> int: ...

    def world_size(self) -> int: ...

    def local_rank(self) -> int | None: ...

    def barrier(self) -> None: ...

    def broadcast(self, value: Any, root: int = 0) -> Any: ...

    def allreduce(self, value: Any, op: CommOp = "sum") -> Any: ...

    def allgather(self, value: Any) -> list[Any]: ...

    def gather(self, value: Any, root: int = 0) -> list[Any] | None: ...

    def send(self, value: Any, dest: int, tag: int = 0) -> None: ...

    def isend(self, value: Any, dest: int, tag: int = 0) -> Any: ...

    def recv(
        self,
        source: int | None = None,
        tag: int | None = None,
        status: Any | None = None,
    ) -> Any: ...

    def iprobe(
        self,
        source: int | None = None,
        tag: int | None = None,
        status: Any | None = None,
    ) -> bool: ...

    def status(self) -> Any: ...

    def any_source(self) -> int: ...

    @property
    def comm(self) -> Any: ...

def comm_sanity_check(comm_backend: Any, rank: int, world_size: int) -> None:
    """Best-effort MPI communication sanity check.

    Runs a tiny set of collectives to detect misconfigured launches early.
    """

    if world_size <= 1:
        return

    try:
        comm_backend.barrier()
    except Exception as exc:
        raise RuntimeError(f"MPI Barrier failed: {exc}") from exc

    # allreduce: sum(ranks) must match expected.
    try:
        got_sum = int(comm_backend.allreduce(int(rank), op="sum"))
        expected_sum = int(world_size * (world_size - 1) // 2)
        if got_sum != expected_sum:
            raise RuntimeError(
                f"MPI allreduce mismatch: got={got_sum} expected={expected_sum} "
                f"(rank={rank} world_size={world_size})"
            )
    except Exception as exc:
        raise RuntimeError(f"MPI allreduce check failed: {exc}") from exc

    # bcast: all ranks must receive the same token.
    try:
        token = None
        if rank == 0:
            token = f"mpi_sanity_{int(time.time() * 1000)}"
        token = comm_backend.broadcast(token, root=0)
        if not isinstance(token, str) or not token.startswith("mpi_sanity_"):
            raise RuntimeError(f"MPI bcast token invalid: {token!r}")
    except Exception as exc:
        raise RuntimeError(f"MPI bcast check failed: {exc}") from exc

    # allgather: helpful for debugging placement issues.
    try:
        host = socket.gethostname()
        hosts = comm_backend.allgather(str(host))
        if rank == 0:
            _LOGGER.info("MPI sanity hosts=%s", hosts)
    except Exception:
        # Non-fatal: only for diagnostics.
        pass
