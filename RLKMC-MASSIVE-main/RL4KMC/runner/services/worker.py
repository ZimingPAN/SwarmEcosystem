from __future__ import annotations

# Worker 负责以下事情
# 1. 领取任务并执行，更新header
import os
import json
import multiprocessing as mp
import signal
import ctypes
import logging
import time
import faulthandler
import sys
import socket
import torch
import numpy as np
from typing import Any
from pydantic import BaseModel
from multiprocessing import shared_memory
from multiprocessing.synchronize import Lock
from dataclasses import dataclass

from RL4KMC.runner.services.affinity import pin_current_process
from RL4KMC.runner.services.engine_runtime import build_engine_runtime
from RL4KMC.config import CONFIG
from environ import ENVIRON

_LOGGER = logging.getLogger(__name__)


class _OnlineDurationStats:
    __slots__ = ("count", "mean", "m2", "total", "vmin", "vmax", "_p2")

    class _P2Quantile:
        __slots__ = ("p", "_init", "q", "n", "np", "dn")

        def __init__(self, p: float) -> None:
            self.p = float(p)
            self._init: list[float] = []
            self.q = [0.0] * 5
            self.n = [0] * 5
            self.np = [0.0] * 5
            p = float(self.p)
            self.dn = [0.0, 0.5 * p, p, 0.5 * (1.0 + p), 1.0]

        def add(self, x: float) -> None:
            x = float(x)
            if not (x == x):
                return
            if len(self._init) < 5:
                self._init.append(float(x))
                if len(self._init) == 5:
                    self._init.sort()
                    for i in range(5):
                        self.q[i] = float(self._init[i])
                        self.n[i] = int(i + 1)
                    p = float(self.p)
                    self.np[0] = 1.0
                    self.np[1] = 1.0 + 2.0 * p
                    self.np[2] = 1.0 + 4.0 * p
                    self.np[3] = 3.0 + 2.0 * p
                    self.np[4] = 5.0
                return

            k = 0
            if x < self.q[0]:
                self.q[0] = float(x)
                k = 0
            elif x < self.q[1]:
                k = 0
            elif x < self.q[2]:
                k = 1
            elif x < self.q[3]:
                k = 2
            elif x <= self.q[4]:
                k = 3
            else:
                self.q[4] = float(x)
                k = 3

            for i in range(k + 1, 5):
                self.n[i] += 1
            for i in range(5):
                self.np[i] += float(self.dn[i])

            for i in (1, 2, 3):
                d = float(self.np[i]) - float(self.n[i])
                if (d >= 1.0 and (self.n[i + 1] - self.n[i]) > 1) or (
                    d <= -1.0 and (self.n[i] - self.n[i - 1]) > 1
                ):
                    ds = 1 if d >= 1.0 else -1
                    n_im1 = float(self.n[i - 1])
                    n_i = float(self.n[i])
                    n_ip1 = float(self.n[i + 1])
                    q_im1 = float(self.q[i - 1])
                    q_i = float(self.q[i])
                    q_ip1 = float(self.q[i + 1])

                    try:
                        q_hat = q_i + float(ds) / float(n_ip1 - n_im1) * (
                            (n_i - n_im1 + float(ds))
                            * (q_ip1 - q_i)
                            / float(n_ip1 - n_i)
                            + (n_ip1 - n_i - float(ds))
                            * (q_i - q_im1)
                            / float(n_i - n_im1)
                        )
                    except Exception:
                        q_hat = q_i

                    if q_im1 < q_hat < q_ip1:
                        self.q[i] = float(q_hat)
                    else:
                        j = i + ds
                        try:
                            self.q[i] = q_i + float(ds) * (
                                float(self.q[j]) - q_i
                            ) / float(self.n[j] - self.n[i])
                        except Exception:
                            self.q[i] = q_i
                    self.n[i] += int(ds)

        def value(self) -> float | None:
            if len(self._init) == 0:
                return None
            if len(self._init) < 5:
                try:
                    s = sorted(self._init)
                    mid = int(len(s) // 2)
                    return float(s[mid])
                except Exception:
                    return None
            try:
                return float(self.q[2])
            except Exception:
                return None

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.total = 0.0
        self.vmin = float("inf")
        self.vmax = float("-inf")
        self._p2 = _OnlineDurationStats._P2Quantile(0.5)

    def add(self, x: float) -> None:
        try:
            x = float(x)
        except Exception:
            return
        if not (x == x):
            return
        self.count += 1
        self.total += float(x)
        try:
            self._p2.add(float(x))
        except Exception:
            pass
        if x < self.vmin:
            self.vmin = float(x)
        if x > self.vmax:
            self.vmax = float(x)
        delta = float(x) - float(self.mean)
        self.mean += float(delta) / float(self.count)
        delta2 = float(x) - float(self.mean)
        self.m2 += float(delta) * float(delta2)

    def to_report_dict(self) -> dict[str, Any]:
        if int(self.count) <= 0:
            return {
                "task_count": 0,
                "total_duration": 0.0,
                "mean_duration": float("nan"),
                "min_duration": float("nan"),
                "max_duration": float("nan"),
                "median_duration": float("nan"),
                "var_duration": float("nan"),
                "m2_duration": 0.0,
            }
        var = float(self.m2) / float(self.count)
        try:
            med = self._p2.value()
            med = float(med) if med is not None else float("nan")
        except Exception:
            med = float("nan")
        return {
            "task_count": int(self.count),
            "total_duration": float(self.total),
            "mean_duration": float(self.mean),
            "min_duration": float(self.vmin),
            "max_duration": float(self.vmax),
            "median_duration": float(med),
            "var_duration": float(var),
            "m2_duration": float(self.m2),
        }


@dataclass
class TaskWindow:
    """Represents the range of task indices assigned to a worker."""
    start: int
    end: int
PAYLOAD_DTYPE = np.dtype(
    [
        ("done", "i1"),
        ("worker_id", "i4"),
        ("pid", "i4"),
        ("total_steps", "i8"),
        ("total_compute_time", "f8"),
        ("total_tasks", "i4"),
        ("total_jumps", "i8"),
        ("first_task_start_ts", "f8"),
        ("last_task_end_ts", "f8"),
        ("json_blob", "S102400"),  # 预留 100KB 给 series 和 load_balance 的 JSON 字符串
    ]
)

TASK_DTYPE = np.dtype(
    [
        ("temp", "f8"),
        ("cu_density", "f8"),
        ("v_density", "f8"),
        ("time", "f8"),
    ]
)


def set_pdeathsig(sig=signal.SIGKILL):
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, sig)


class Worker:
    def __init__(
        self,
        args,
        worker_id: int,
        num_workers: int,
        num_tasks: int,
        task_start: Any,
        task_end: Any,
        is_drained: Any,
        cursur_lock: Lock,
        model: torch.nn.Module,
        shm_name: str,
        task_shm_name: str,
        cpu_list: list[int],
        embedding_device: torch.device,
    ) -> None:
        self.args = args
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.num_tasks = int(num_tasks)
        self.task_start = task_start
        self.task_end = task_end
        self.is_drained = is_drained
        self.cursur_lock = cursur_lock
        self.model = model
        self.task_window = TaskWindow(0, 0)
        self.cpu_list = cpu_list
        self.shm_name = shm_name
        self.task_shm_name = str(task_shm_name)
        self.embedding_device = embedding_device
        self._last_wait_log_ts = 0.0
        self._faulthandler_stream: Any | None = None
        self._task_shm: Any | None = None
        self._task_array: Any | None = None

    def _attach_task_shm(self) -> None:
        if self._task_array is not None:
            return
        task_shm = shared_memory.SharedMemory(name=self.task_shm_name)
        self._task_shm = task_shm
        self._task_array = np.ndarray(
            shape=(int(self.num_tasks),), dtype=TASK_DTYPE, buffer=task_shm.buf
        )

    def _close_task_shm(self) -> None:
        task_shm = self._task_shm
        self._task_array = None
        self._task_shm = None
        if task_shm is None:
            return
        try:
            task_shm.close()
        except Exception:
            pass

    def _read_task(self, task_idx: int) -> dict[str, float]:
        task_array = self._task_array
        if task_array is None:
            raise RuntimeError("task shared memory is not attached")
        if int(task_idx) < 0 or int(task_idx) >= int(self.num_tasks):
            raise IndexError(f"task_idx out of range: {task_idx}")
        row = task_array[int(task_idx)]
        return {
            "temp": float(row["temp"]),
            "cu_density": float(row["cu_density"]),
            "v_density": float(row["v_density"]),
            "time": float(row["time"]),
        }

    def _resolve_worker_log_path(self) -> str:
        output_dir = (
            str(getattr(self.args, "output_dir", "") or "").strip()
            or str(os.environ.get("TASK_RUN_OUTPUT_DIR", "") or "").strip()
            or "logs"
        )
        rank_hint = (
            str(os.environ.get("RANK", "") or "").strip()
            or str(os.environ.get("OMPI_COMM_WORLD_RANK", "") or "").strip()
            or str(os.environ.get("LOCAL_RANK", "") or "").strip()
            or "unknown"
        )
        host = socket.gethostname()
        pid = os.getpid()

        log_dir = os.path.join(output_dir, "worker-debug")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

        return os.path.join(
            log_dir,
            f"rank{rank_hint}_w{int(self.worker_id)}_{host}_pid{pid}.log",
        )

    def _setup_worker_logging(self) -> str | None:
        
        if not ENVIRON.enable_worker_debug_log:
            return None
        
        level_name = (
            str(os.environ.get("KMC_LOG_LEVEL", "") or "").strip()
            or str(os.environ.get("LOG_LEVEL", "") or "").strip()
            or "INFO"
        )
        log_level = getattr(logging, str(level_name).upper(), logging.INFO)

        root = logging.getLogger()
        root.setLevel(log_level)

        if not root.handlers:
            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(
                logging.Formatter(
                    fmt=f"%(levelname)s [Worker {self.worker_id} PID:%(process)d] %(filename)s:%(lineno)d: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(stream_handler)

        log_path = self._resolve_worker_log_path()
        has_target_handler = any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", ""))
            == os.path.abspath(log_path)
            for h in list(root.handlers)
        )
        if not has_target_handler:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt=f"%(levelname)s [Worker {self.worker_id} PID:%(process)d] %(filename)s:%(lineno)d: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(file_handler)

        return os.path.abspath(log_path)

    def claim_task(self) -> bool:
        """Claim the next task index. Returns False if no more tasks are available."""
        claim_size = CONFIG.runner.worker_claim_size
        with self.cursur_lock:
            start = int(self.task_start.value)
            task_end = int(self.task_end.value)
            drained = bool(self.is_drained.value)

            if start >= task_end:
                if drained:
                    return False
                self.task_window = TaskWindow(start, start)
            else:
                end = min(task_end, start + claim_size)
                self.task_start.value = end
                self.task_window = TaskWindow(start, end)
                return True

        # No work is currently available for this rank yet. Back off briefly to
        # avoid busy-spinning on the shared cursor lock while leader updates it.
        now = time.monotonic()
        if now - float(self._last_wait_log_ts) >= 5.0:
            _LOGGER.info(
                "Worker %s waiting for tasks: cursor=(start=%s,end=%s,drained=%s)",
                self.worker_id,
                start,
                task_end,
                drained,
            )
            self._last_wait_log_ts = float(now)
        time.sleep(max(0.0, CONFIG.runner.worker_idle_sleep_sec))
        return True

    def write_result(self, payload: dict[str, Any]) -> None:
        """Write the result payload to shared memory."""

        # 将 series 和 load_balance 的数据转换为 JSON 字符串，并存储在 json_blob 字段中
        json_blob = json.dumps(
            {
                "series": payload.pop("series", []),
                "load_balance": payload.pop("load_balance", {}),
            }
        ).encode("utf-8")[
            : PAYLOAD_DTYPE["json_blob"].itemsize
        ]  # 确保不超过预留大小

        # 构建完整的 payload 数据
        full_payload = np.array(
            (
                0,
                payload["worker_id"],
                payload["pid"],
                payload["total_steps"],
                payload["total_compute_time"],
                payload["total_tasks"],
                payload["total_jumps"],
                payload["first_task_start_ts"],
                payload["last_task_end_ts"],
                json_blob,
            ),
            dtype=PAYLOAD_DTYPE,
        )

        # 将数据写入共享内存
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        result_array = np.ndarray(
            shape=(self.num_workers,), dtype=PAYLOAD_DTYPE, buffer=existing_shm.buf
        )
        result_array[self.worker_id] = full_payload
        result_array[self.worker_id]["done"] = 1
        existing_shm.close()

    def run(self) -> None:
        # 2. 阻断 mpi4py 自动 Finalize
        # 如果 sys.modules 里已经有了，确保它不会在退出时去同步
        if 'mpi4py' in sys.modules:
            import mpi4py
            mpi4py.rc.finalize = False
        log_path = self._setup_worker_logging()
        if log_path is not None:
            _LOGGER.info(
                "Worker %s log file enabled: %s", self.worker_id, str(log_path)
            )

        if log_path is not None:
            try:
                self._faulthandler_stream = open(log_path, "a", encoding="utf-8")
            except Exception:
                self._faulthandler_stream = None

        pin_current_process(self.cpu_list)
        _LOGGER.info(
            "Worker %s start: pid=%s cpu_list=%s embedding_device=%s",
            self.worker_id,
            os.getpid(),
            list(self.cpu_list),
            str(self.embedding_device),
        )

        try:
            faulthandler.enable(
                file=(self._faulthandler_stream or sys.stderr), all_threads=True
            )
        except Exception:
            pass
        if hasattr(signal, "SIGUSR1"):
            try:
                faulthandler.register(
                    signal.SIGUSR1,
                    file=(self._faulthandler_stream or sys.stderr),
                    all_threads=True,
                    chain=False,
                )
            except Exception:
                pass

        try:
            set_pdeathsig()
            _LOGGER.debug("Worker %s set pdeathsig complete", self.worker_id)

            self._attach_task_shm()
            _LOGGER.info(
                "Worker %s attached task shared memory, total=%s",
                self.worker_id,
                int(self.num_tasks),
            )

            engine_runtime = build_engine_runtime(
                args=self.args,
                worker_id=self.worker_id,
                embed=self.model,
                embed_device=self.embedding_device,
            )
            _LOGGER.info("Worker %s engine runtime built", self.worker_id)

            bench_step = self.args.bench_step
            use_traditional_kmc = self.args.use_traditional_kmc
            total_tasks = 0
            total_jumps = 0
            lb_stats = _OnlineDurationStats()
            while self.claim_task():
                if self.task_window.end <= self.task_window.start:
                    continue
                # _LOGGER.debug(
                #     "Worker %s claimed tasks [%s, %s)",
                #     self.worker_id,
                #     self.task_window.start,
                #     self.task_window.end,
                # )
                for task_idx in range(self.task_window.start, self.task_window.end):
                    task = self._read_task(int(task_idx))
                    task_begin_ts = time.time()
                    total_jumps += engine_runtime.engine.run_one_task(
                        task,
                        bench_step=bench_step,
                        use_traditional_kmc=use_traditional_kmc,
                    )
                    task_end_ts = time.time()
                    try:
                        lb_stats.add(float(task_end_ts - task_begin_ts))
                    except Exception:
                        pass
                total_tasks += self.task_window.end - self.task_window.start

            _LOGGER.info(
                "Worker %s finished tasks: total_tasks=%s total_jumps=%s total_compute_time=%.2fs",
                self.worker_id,
                total_tasks,
                total_jumps,
                engine_runtime.timing_stats.total_compute_time,
            )

            payload: dict[str, Any] = {
                "worker_id": self.worker_id,
                "pid": os.getpid(),
                "total_tasks": total_tasks,
                "total_jumps": total_jumps,
                "total_steps": engine_runtime.timing_stats.total_steps,
                "total_compute_time": engine_runtime.timing_stats.total_compute_time,
                "first_task_start_ts": engine_runtime.timing_stats.first_task_start_ts,
                "last_task_end_ts": engine_runtime.timing_stats.last_task_end_ts,
                "series": engine_runtime.timing_stats.series,
                "load_balance": lb_stats.to_report_dict(),
            }

            self.write_result(payload)
            _LOGGER.info(
                "Worker %s wrote results to shared memory and is exiting",
                self.worker_id,
            )
        except Exception:
            _LOGGER.exception("Worker %s crashed", self.worker_id)
            raise
        finally:
            if self._faulthandler_stream is not None:
                try:
                    self._faulthandler_stream.flush()
                except Exception:
                    pass
                try:
                    self._faulthandler_stream.close()
                except Exception:
                    pass
            self._close_task_shm()
