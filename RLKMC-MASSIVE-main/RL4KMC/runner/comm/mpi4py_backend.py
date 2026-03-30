from __future__ import annotations

import os
from typing import Any

from .backend import CommOp

class Mpi4pyCommBackend:
    def __init__(self, *, mpi: Any, comm: Any) -> None:
        self._mpi = mpi
        self._comm = comm

    @property
    def comm(self) -> Any:
        return self._comm

    def init(self) -> None:
        if not self._mpi.Is_initialized():
            self._mpi.Init()

    def finalize(self) -> None:
        if self._mpi.Is_initialized() and not self._mpi.Is_finalized():
            self._mpi.Finalize()

    def is_initialized(self) -> bool:
        return bool(self._mpi.Is_initialized())

    def is_finalized(self) -> bool:
        return bool(self._mpi.Is_finalized())

    def rank(self) -> int:
        return int(self._comm.Get_rank())

    def world_size(self) -> int:
        return int(self._comm.Get_size())

    def local_rank(self) -> int | None:
        raw = os.environ.get("LOCAL_RANK", None)
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    def barrier(self) -> None:
        self._comm.Barrier()

    def broadcast(self, value: Any, root: int = 0) -> Any:
        return self._comm.bcast(value, root=int(root))

    def allreduce(self, value: Any, op: CommOp = "sum") -> Any:
        if str(op) != "sum":
            raise ValueError(f"Unsupported allreduce op: {op!r}")
        return self._comm.allreduce(value, op=self._mpi.SUM)

    def allgather(self, value: Any) -> list[Any]:
        return list(self._comm.allgather(value))

    def gather(self, value: Any, root: int = 0) -> list[Any] | None:
        out = self._comm.gather(value, root=int(root))
        if out is None:
            return None
        return list(out)

    def send(self, value: Any, dest: int, tag: int = 0) -> None:
        self._comm.send(value, dest=int(dest), tag=int(tag))

    def isend(self, value: Any, dest: int, tag: int = 0) -> Any:
        return self._comm.isend(value, dest=int(dest), tag=int(tag))

    def recv(
        self,
        source: int | None = None,
        tag: int | None = None,
        status: Any | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if source is not None:
            kwargs["source"] = int(source)
        if tag is not None:
            kwargs["tag"] = int(tag)
        if status is not None:
            kwargs["status"] = status
        return self._comm.recv(**kwargs)

    def iprobe(
        self,
        source: int | None = None,
        tag: int | None = None,
        status: Any | None = None,
    ) -> bool:
        kwargs: dict[str, Any] = {}
        if source is not None:
            kwargs["source"] = int(source)
        if tag is not None:
            kwargs["tag"] = int(tag)
        if status is not None:
            kwargs["status"] = status
        return bool(self._comm.Iprobe(**kwargs))

    def status(self) -> Any:
        return self._mpi.Status()

    def any_source(self) -> int:
        return int(self._mpi.ANY_SOURCE)


def init_mpi4py() -> Mpi4pyCommBackend:
    import mpi4py  # type: ignore

    mpi4py.rc.initialize = False  # type: ignore[attr-defined]
    mpi4py.rc.finalize = False

    from mpi4py import MPI  # type: ignore

    try:
        if not MPI.Is_initialized():
            MPI.Init()
    except Exception:
        pass

    return Mpi4pyCommBackend(mpi=MPI, comm=MPI.COMM_WORLD)


def get_mpi4py_world_backend() -> Mpi4pyCommBackend:
    from mpi4py import MPI  # type: ignore

    return Mpi4pyCommBackend(mpi=MPI, comm=MPI.COMM_WORLD)

