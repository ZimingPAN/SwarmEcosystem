from __future__ import annotations

# Leader 负责以下事情
# 1. 全生命周期管理worker进程（在 Leader+Worker 模式下）
# 2. 维护任务窗口和cursur，worker通过shm领取任务并更新cursur
# 3. 静态划分任务或者动态调度（使用scheduler）
# 4. 结果统计、分析、汇总
# 5. 直接执行计算（纯 Leader 模式下）
# 6. 读取模型权重到shm，将模型对象直接作为参数传递给worker
# 7. 初始化通信

from dataclasses import dataclass
import json
import os
import time
import signal
import logging
import socket
import torch
import numpy as np
import multiprocessing as mp
from typing import Any, Iterable, cast
from multiprocessing import shared_memory
from multiprocessing.context import SpawnProcess

from RL4KMC.runner.scheduler.static_queue_scheduler import StaticQueueScheduler
from environ import ENVIRON
from .worker import Worker, PAYLOAD_DTYPE, TASK_DTYPE
from RL4KMC.config import CONFIG
from RL4KMC.runner.services.output_manager import LocalStepStats
from RL4KMC.runner.engine.task_manager import generate_kmc_tasks
from RL4KMC.runner.scheduler import StaticQueueScheduler
from RL4KMC.runner.services.affinity import build_pin_plan, read_current_rank_affinity
from RL4KMC.runner.services.output_runtime import resolve_output_runtime
from RL4KMC.runner.services.timing_reporter import RunFinalizer

_LOGGER = logging.getLogger(__name__)


def _set_safe_env_for_prefork() -> None:
    """Best-effort mitigations for OpenMP + fork() crashes.

    Must run before importing torch.
    """

    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


@dataclass
class LeaderStatus:
    model_built: bool = False
    result_shm_init: bool = False
    workers_spawned: bool = False
    comm_init: bool = False
    scheduler_init: bool = False


class Leader:
    def __init__(
        self, args, comm_backend_type: str, scheduler_type: str, device: torch.device
    ) -> None:
        self.args = args
        self.status = LeaderStatus()

        self.comm_backend_type = comm_backend_type
        self.scheduler_type = scheduler_type
        self.workers: list[SpawnProcess] = []
        self._worker_proc_map: dict[int, SpawnProcess] = {}
        self.expected_num_workers = args.workers_per_rank
        self.shm_name = ""
        self.task_shm_name = ""
        self.device = device
        self.selected_numa_nodes: list[int] = []
        self.num_tasks = int(getattr(self.args, "n_radial", 1)) * int(
            getattr(self.args, "n_axial", 1)
        )
        self._timing_stats = LocalStepStats()
        self._worker_load_balance: dict[str, dict[str, Any]] = {}
        self._output = resolve_output_runtime(args=self.args)
        try:
            self._timing_stats.set_metadata(
                lattice_size=tuple(getattr(self.args, "lattice_size", (0, 0, 0))),
                nodes=int(getattr(self.args, "nodes", 1) or 1),
                output_level=int(getattr(self.args, "output_level", 1) or 1),
            )
        except Exception:
            pass
        self._payload_reporter: WorkerPayloadReporter | None = None

        self._mp_ctx = mp.get_context("spawn")

        # 任务cursur
        self.task_start = self._mp_ctx.RawValue("q", 0)
        self.task_end = self._mp_ctx.RawValue("q", 0)
        self.is_drained = self._mp_ctx.RawValue("b", False)  # 是否所有任务都已调度完成

        self.cursur_lock = self._mp_ctx.Lock()  # 保护cursur的锁

    def build_model(self) -> None:
        from RL4KMC.embedding.model_builder import build_embed

        # NOTE: torch.nn.Module.share_memory() only works for CPU tensors.
        # If workers need CUDA, they should move/copy the weights per-process.
        shared_device = torch.device("cpu")
        embed_build_result = build_embed(self.args, shared_device)
        self.model = embed_build_result.embed
        self.model.share_memory()  # 将模型参数移动到共享内存，以便worker进程访问
        self.status.model_built = True

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def init_comm(self) -> None:
        assert not self.status.comm_init, "Communication already initialized"

        from RL4KMC.runner.comm.mpi4py_backend import init_mpi4py

        self.comm_backend = init_mpi4py()
        self.rank = self.comm_backend.rank()
        self.world_size = self.comm_backend.world_size()
        self._payload_reporter = WorkerPayloadReporter(
            rank=int(self.rank),
            expected_num_workers=int(self.expected_num_workers),
            timing_stats=self._timing_stats,
            output_runtime=self._output,
        )

        self.status.comm_init = True

    def init_result_shm(self) -> None:
        payload_size = PAYLOAD_DTYPE.itemsize
        total_size = payload_size * self.expected_num_workers
        self.shm = shared_memory.SharedMemory(create=True, size=total_size)
        self.result_array = np.ndarray(
            shape=(self.expected_num_workers,), dtype=PAYLOAD_DTYPE, buffer=self.shm.buf
        )
        self.result_array.fill(0)
        self.shm_name = self.shm.name
        self.status.result_shm_init = True

    def init_task_shm(self) -> None:
        tasks = generate_kmc_tasks(
            int(getattr(self.args, "n_radial", 1)),
            int(getattr(self.args, "n_axial", 1)),
            float(getattr(self.args, "rescaled_sim_time", 1.0)),
            float(getattr(self.args, "cu_density", 0.0)),
            float(getattr(self.args, "v_density", 1.0)),
        )

        if len(tasks) != int(self.num_tasks):
            raise RuntimeError(
                f"task count mismatch: generated={len(tasks)} expected={int(self.num_tasks)}"
            )

        total_size = TASK_DTYPE.itemsize * int(self.num_tasks)
        self.task_shm = shared_memory.SharedMemory(create=True, size=total_size)
        task_array = np.ndarray(
            shape=(int(self.num_tasks),), dtype=TASK_DTYPE, buffer=self.task_shm.buf
        )

        for idx, task in enumerate(tasks):
            task_array[int(idx)] = (
                float(task.get("temp", 0.0)),
                float(task.get("cu_density", 0.0)),
                float(task.get("v_density", 0.0)),
                float(task.get("time", 0.0)),
            )

        self.task_shm_name = self.task_shm.name

    def spawn_workers(self, model) -> None:
        _set_safe_env_for_prefork()

        plan = build_pin_plan(
            workers_per_rank=int(self.args.workers_per_rank),
            cores_per_worker=int(self.args.cores_per_worker),
            pin_policy=str(self.args.pin_policy),
        )
        # _LOGGER.debug(f"Pin Plan: {plan}")

        cpu_sets = plan.worker_cpu_sets

        if not self.status.result_shm_init:
            self.init_result_shm()
        if not getattr(self, "task_shm_name", ""):
            self.init_task_shm()

        self.workers = []
        self._worker_proc_map = {}
        _LOGGER.info(
            "Spawning workers: expected_num_workers=%s pin_policy=%s",
            int(self.expected_num_workers),
            str(getattr(self.args, "pin_policy", "spread")),
        )

        for worker_id, cpu_list in enumerate(cpu_sets):
            worker = Worker(
                args=self.args,
                worker_id=int(worker_id),
                num_workers=int(self.expected_num_workers),
                num_tasks=int(self.num_tasks),
                task_start=self.task_start,
                task_end=self.task_end,
                is_drained=self.is_drained,
                cursur_lock=self.cursur_lock,
                model=model,
                shm_name=self.shm_name,
                task_shm_name=self.task_shm_name,
                cpu_list=list(cpu_list),
                embedding_device=self.device,
            )
            proc = self._mp_ctx.Process(target=worker.run)
            proc.daemon = False
            proc.start()
            self.workers.append(proc)
            self._worker_proc_map[int(worker_id)] = proc
            _LOGGER.debug(
                "Spawned worker %s pid=%s cpus=%s",
                int(worker_id),
                int(proc.pid) if proc.pid is not None else -1,
                list(cpu_list),
            )

        self.status.workers_spawned = True

    def _detach_worker_proc(self, proc: SpawnProcess) -> None:
        try:
            if proc in self.workers:
                self.workers.remove(proc)
        except Exception:
            pass

        for wid, p in list(self._worker_proc_map.items()):
            if p is proc:
                try:
                    self._worker_proc_map.pop(int(wid), None)
                except Exception:
                    pass
                break

    def _reclaim_worker_if_done(self, worker_id: int, proc: SpawnProcess) -> None:
        result_array = getattr(self, "result_array", None)
        if result_array is None:
            return
        if int(worker_id) < 0 or int(worker_id) >= int(self.expected_num_workers):
            return

        try:
            done = int(result_array[int(worker_id)]["done"])
        except Exception:
            done = 0

        if int(done) != 1:
            return

        _LOGGER.debug(
            "Worker %s result ready in shared memory, reclaiming pid=%s",
            int(worker_id),
            int(proc.pid) if proc.pid is not None else -1,
        )

        join_grace_sec = float(
            getattr(CONFIG.runner, "worker_join_poll_interval_sec", 1.0) or 1.0
        )
        join_grace_sec = max(0.1, join_grace_sec)

        proc.join(timeout=join_grace_sec)
        killed_by_leader = False
        if proc.is_alive():
            _LOGGER.debug(
                "Worker %s still alive after result ready, killing pid=%s",
                int(worker_id),
                int(proc.pid) if proc.pid is not None else -1,
            )
            proc.kill()
            proc.join(timeout=0.5)
            killed_by_leader = True

        # if proc.is_alive():
        #     _LOGGER.error(
        #         "Worker %s still alive after kill attempt, keep tracking pid=%s",
        #         int(worker_id),
        #         int(proc.pid) if proc.pid is not None else -1,
        #     )
        #     return

        exitcode = proc.exitcode
        if (not killed_by_leader) and exitcode is not None and int(exitcode) != 0:
            raise RuntimeError(
                f"worker {int(worker_id)} exited with non-zero code after reclaim: {int(exitcode)}"
            )

        self._detach_worker_proc(proc)

    def _force_reap_tracked_workers(self) -> None:
        for worker_id, proc in list(self._worker_proc_map.items()):
            if not proc.is_alive():
                self._detach_worker_proc(proc)
                continue

            _LOGGER.warning(
                "Force reaping tracked worker %s pid=%s",
                int(worker_id),
                int(proc.pid) if proc.pid is not None else -1,
            )
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=0.5)
            except Exception:
                pass

            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.join(timeout=1.0)
                except Exception:
                    pass

            if not proc.is_alive():
                self._detach_worker_proc(proc)

    def _force_reap_active_children(self) -> None:
        try:
            children = list(mp.active_children())
        except Exception:
            children = []

        if not children:
            return

        _LOGGER.warning("Force reaping active children count=%s", len(children))
        for child in children:
            try:
                if not child.is_alive():
                    continue
            except Exception:
                continue

            try:
                _LOGGER.warning(
                    "Force reaping child pid=%s exitcode=%s",
                    int(child.pid) if child.pid is not None else -1,
                    child.exitcode,
                )
            except Exception:
                pass

            try:
                child.terminate()
            except Exception:
                pass
            try:
                child.join(timeout=0.5)
            except Exception:
                pass
            try:
                if child.is_alive():
                    child.kill()
                    child.join(timeout=1.0)
            except Exception:
                pass

    def _reclaim_done_workers(self) -> None:
        for worker_id, proc in list(self._worker_proc_map.items()):
            self._reclaim_worker_if_done(int(worker_id), proc)

    def join_workers(self) -> None:
        if not self.workers:
            return

        join_timeout_sec = float(
            getattr(CONFIG.runner, "worker_join_timeout_sec", 0.0) or 0.0
        )
        poll_interval_sec = float(
            getattr(CONFIG.runner, "worker_join_poll_interval_sec", 1.0) or 1.0
        )
        poll_interval_sec = max(0.1, poll_interval_sec)
        dump_stacks_on_timeout = bool(
            getattr(CONFIG.runner, "worker_dump_stacks_on_timeout", True)
        )

        deadline = (
            time.monotonic() + join_timeout_sec
            if float(join_timeout_sec) > 0.0
            else None
        )
        last_wait_log_ts = time.monotonic()
        exitcodes: list[int] = []
        while self.workers:
            self._reclaim_done_workers()

            for worker_id, proc in list(self._worker_proc_map.items()):
                if not proc.is_alive():
                    try:
                        if proc.exitcode is not None:
                            exitcodes.append(int(proc.exitcode))
                    except Exception:
                        pass
                    _LOGGER.debug(
                        "Worker %s exited before done-flag reclaim, exitcode=%s",
                        int(worker_id),
                        proc.exitcode,
                    )
                    self._detach_worker_proc(proc)

            now = time.monotonic()
            if now - float(last_wait_log_ts) >= 10.0:
                alive: list[str] = []
                for worker_id, proc in list(self._worker_proc_map.items()):
                    alive.append(
                        f"w{int(worker_id)}:pid={proc.pid},exitcode={proc.exitcode}"
                    )
                _LOGGER.info(
                    "Waiting workers to reclaim (rank=%s): alive=[%s], cursor=(start=%s,end=%s,drained=%s)",
                    getattr(self, "rank", "?"),
                    ", ".join(alive),
                    int(self.task_start.value),
                    int(self.task_end.value),
                    bool(self.is_drained.value),
                )
                last_wait_log_ts = now

            if not self.workers:
                break

            if deadline is not None and time.monotonic() >= deadline:
                alive: list[str] = []
                for worker_id, proc in list(self._worker_proc_map.items()):
                    try:
                        alive.append(
                            f"w{int(worker_id)}:pid={proc.pid},exitcode={proc.exitcode}"
                        )
                    except Exception:
                        alive.append(f"pid={getattr(proc, 'pid', None)}")

                _LOGGER.error(
                    "Worker join timeout after %.1fs (rank=%s). alive=[%s], cursor=(start=%s,end=%s,drained=%s)",
                    float(join_timeout_sec),
                    getattr(self, "rank", "?"),
                    ", ".join(alive),
                    int(self.task_start.value),
                    int(self.task_end.value),
                    bool(self.is_drained.value),
                )

                if dump_stacks_on_timeout and hasattr(signal, "SIGUSR1"):
                    for proc in list(self._worker_proc_map.values()):
                        try:
                            if proc.pid is not None and proc.is_alive():
                                os.kill(int(proc.pid), int(signal.SIGUSR1))
                        except Exception:
                            continue
                    time.sleep(1.0)

                raise TimeoutError(
                    f"worker join timeout after {join_timeout_sec:.1f}s; "
                    f"alive workers={len(self.workers)}"
                )

            time.sleep(poll_interval_sec)

        bad = [code for code in exitcodes if int(code) != 0]
        if bad:
            raise RuntimeError(f"worker exited with non-zero codes: {bad}")
        _LOGGER.debug("All workers joined")

    def kill_workers(self) -> None:
        for proc in list(self.workers):
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.join(timeout=0.5)
            except Exception:
                pass
            if not proc.is_alive():
                self._detach_worker_proc(proc)

    def init_scheduler(self) -> None:
        assert (
            self.status.workers_spawned and self.status.comm_init
        ), "Workers and communication must be initialized before scheduler"
        self.scheduler = StaticQueueScheduler(
            self.rank, self.world_size, num_tasks=self.num_tasks
        )
        self.status.scheduler_init = True

    def fetch_new_tasks(self) -> None:
        assert (
            self.status.scheduler_init
        ), "Scheduler must be initialized before fetching tasks"
        if not self.scheduler.is_drained():
            self.scheduler.update_task_window()
        with self.cursur_lock:
            self.task_start.value = (
                self.scheduler.task_window.start
            )  # 更新cursur到当前窗口的起始位置
            self.task_end.value = (
                self.scheduler.task_window.end
            )  # 更新cursur到当前窗口的结束位置
            self.is_drained.value = (
                self.scheduler.is_drained()
            )  # 更新是否所有任务都已调度完成

    def _finalize_run(self, start_global_ts: float, end_global_ts: float) -> float:
        finalizer = RunFinalizer(
            output_dir=self._output.output_dir,
            rank_dir=self._output.rank_dir,
            rank_dir_name=str(self._output.rank_dir_name),
            rank=int(self.rank),
            world_size=int(self.world_size),
            timing_stats=self._timing_stats,
            enable_rank_detail=bool(self._output.enable_rank_detail),
            enable_timing_io=bool(self._output.enable_timing_io),
            enable_timing_report=bool(self._output.enable_timing_report),
            enable_timing_log=bool(self._output.enable_timing_log),
        )
        return float(
            finalizer.finalize(
                start_global_ts=start_global_ts, end_global_ts=end_global_ts
            )
        )

    def _cleanup_result_shm(self) -> None:
        _LOGGER.debug("_cleanup_result_shm")
        shm = getattr(self, "shm", None)
        if shm is None:
            return

        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
        _LOGGER.debug("clean up shm complete")

    def _cleanup_task_shm(self) -> None:
        task_shm = getattr(self, "task_shm", None)
        if task_shm is None:
            return
        try:
            task_shm.close()
        except Exception:
            pass
        try:
            task_shm.unlink()
        except Exception:
            pass

    def summarize_and_report_results(
        self, start_global_ts: float, end_global_ts: float
    ) -> dict[str, Any]:
        if self._payload_reporter is None:
            raise RuntimeError("Payload reporter is not initialized")

        summary = self._payload_reporter.summarize_into_timing(
            result_shm_init=bool(self.status.result_shm_init),
            result_array=getattr(self, "result_array", None),
        )
        self._finalize_run(start_global_ts, end_global_ts)
        return dict(summary)

    def log_progress(self) -> None:
        start = self.scheduler.task_window.start
        end = self.scheduler.task_window.end
        cursur = self.task_start.value
        progress = (cursur - start) / max(1, end - start)

        bar_width = 40
        filled_width = int(progress * bar_width)
        bar = "=" * filled_width + "-" * (bar_width - filled_width)
        _LOGGER.info(
            "Progress: [%s] %6.2f%%",
            bar,
            progress * 100.0,
        )

    def _all_dispatched(self) -> bool:
        return self.task_start.value >= self.task_end.value

    def _pending_worker_ids_from_shm(self) -> list[int]:
        result_array = getattr(self, "result_array", None)
        if result_array is None:
            return [
                int(worker_id) for worker_id in range(int(self.expected_num_workers))
            ]

        pending: list[int] = []
        for worker_id in range(int(self.expected_num_workers)):
            try:
                done = int(result_array[int(worker_id)]["done"])
            except Exception:
                done = 0
            if int(done) != 1:
                pending.append(int(worker_id))
        return pending

    def _all_worker_results_written(self) -> bool:
        return len(self._pending_worker_ids_from_shm()) == 0

    def _is_worker_result_written(self, worker_id: int) -> bool:
        result_array = getattr(self, "result_array", None)
        if result_array is None:
            return False
        if int(worker_id) < 0 or int(worker_id) >= int(self.expected_num_workers):
            return False
        try:
            return int(result_array[int(worker_id)]["done"]) == 1
        except Exception:
            return False

    def run(self) -> None:
        try:
            self.build_model()
            self.spawn_workers(self.model)
            self.init_comm()
            self.init_scheduler()
            start_global_ts = time.time()
            pending_worker_log_interval_sec = float(
                getattr(CONFIG.runner, "worker_pending_log_interval_sec", 10.0) or 10.0
            )
            pending_worker_log_interval_sec = max(0.1, pending_worker_log_interval_sec)
            last_pending_worker_log_ts = 0.0
            while True:
                self._reclaim_done_workers()
                if self._all_dispatched():
                    if not self.scheduler.is_drained():
                        self.fetch_new_tasks()
                    else:
                        pending_worker_ids = self._pending_worker_ids_from_shm()
                        if not pending_worker_ids:
                            _LOGGER.info(
                                "All tasks dispatched and results written, finishing run"
                            )
                            break

                        now = time.monotonic()
                        if last_pending_worker_log_ts <= 0.0 or now - float(
                            last_pending_worker_log_ts
                        ) >= float(pending_worker_log_interval_sec):
                            _LOGGER.info(
                                "Scheduler drained and tasks dispatched, waiting for %s workers to write results: worker_ids=%s",
                                int(len(pending_worker_ids)),
                                pending_worker_ids,
                            )
                            last_pending_worker_log_ts = now
                if self.rank == 0:
                    self.log_progress()
                # worker health check
                allow_clean_exit = bool(self.scheduler.is_drained() and self._all_dispatched())
                for worker_id, proc in list(self._worker_proc_map.items()):
                    if proc.is_alive():
                        continue

                    exitcode = proc.exitcode
                    if exitcode is None:
                        raise RuntimeError(
                            "worker exited unexpectedly without exit code"
                        )

                    if int(exitcode) != 0:
                        raise RuntimeError(
                            f"worker process failed, pid={proc.pid}, exitcode={exitcode}"
                        )

                    if self._is_worker_result_written(int(worker_id)):
                        self._detach_worker_proc(proc)
                        continue

                    if not allow_clean_exit:
                        raise RuntimeError(
                            f"worker exited too early, pid={proc.pid}, exitcode={exitcode}"
                        )
                time.sleep(CONFIG.runner.leader_tick_interval)
            end_global_ts = time.time()
            self.join_workers()
            self.summarize_and_report_results(start_global_ts, end_global_ts)
        except Exception as exc:
            self.kill_workers()
            raise exc
        finally:
            self._force_reap_tracked_workers()
            self._force_reap_active_children()
            self._cleanup_result_shm()
            self._cleanup_task_shm()
            del self.model

            _LOGGER.debug("leader entering comm finalize")
            self.comm_backend.finalize()
            _LOGGER.debug("leader finalize complete. Exiting..")
            return


class WorkerPayloadReporter:
    def __init__(
        self,
        *,
        rank: int,
        expected_num_workers: int,
        timing_stats: LocalStepStats,
        output_runtime: Any,
    ) -> None:
        self.rank = int(rank)
        self.expected_num_workers = int(expected_num_workers)
        self.timing_stats = timing_stats
        self.output_runtime = output_runtime
        self.worker_load_balance: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _decode_json_blob(blob: Any) -> dict[str, Any]:
        try:
            raw = bytes(blob)
            raw = raw.split(b"\x00", 1)[0]
            if not raw:
                return {}
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def collect_payloads(
        self, *, result_shm_init: bool, result_array: Any
    ) -> list[dict[str, Any]]:
        if not bool(result_shm_init) or result_array is None:
            return []

        payloads: list[dict[str, Any]] = []
        for row in result_array:
            pid = self._as_int(row["pid"], default=0)
            total_tasks = self._as_int(row["total_tasks"], default=0)
            if pid <= 0 and total_tasks <= 0:
                continue

            extra = self._decode_json_blob(row["json_blob"])
            payload = {
                "worker_id": self._as_int(row["worker_id"]),
                "pid": pid,
                "total_steps": self._as_int(row["total_steps"]),
                "total_compute_time": self._as_float(row["total_compute_time"]),
                "total_tasks": total_tasks,
                "total_jumps": self._as_int(row["total_jumps"]),
                "first_task_start_ts": self._as_float(row["first_task_start_ts"]),
                "last_task_end_ts": self._as_float(row["last_task_end_ts"]),
                "series": extra.get("series", {}),
                "load_balance": extra.get("load_balance", {}),
            }
            payloads.append(payload)

        payloads.sort(key=lambda item: int(item.get("worker_id", -1)))
        return payloads

    def merge_payloads(self, payloads: list[dict[str, Any]]) -> None:
        for payload in payloads:
            if not isinstance(payload, dict):
                continue

            try:
                self.timing_stats._total_tasks += int(
                    payload.get("total_tasks", 0) or 0
                )
            except Exception:
                pass
            try:
                self.timing_stats._total_jumps += int(
                    payload.get("total_jumps", 0) or 0
                )
            except Exception:
                pass
            try:
                self.timing_stats._total_steps += int(
                    payload.get("total_steps", 0) or 0
                )
            except Exception:
                pass
            try:
                self.timing_stats._total_compute_time += float(
                    payload.get("total_compute_time", 0.0) or 0.0
                )
            except Exception:
                pass

            try:
                start_ts = payload.get("first_task_start_ts", None)
                end_ts = payload.get("last_task_end_ts", None)
                if start_ts is not None and end_ts is not None:
                    self.timing_stats.add_task_window(float(start_ts), float(end_ts))
            except Exception:
                pass

            try:
                series = payload.get("series", None)
                if isinstance(series, dict):
                    for label, values in series.items():
                        self.timing_stats._series.setdefault(str(label), []).extend(
                            list(values or [])
                        )
            except Exception:
                pass

            try:
                lb = payload.get("load_balance", None)
                if isinstance(lb, dict):
                    key = payload.get("worker_id", None)
                    key = (
                        str(key)
                        if key is not None
                        else str(len(self.worker_load_balance))
                    )
                    self.worker_load_balance[str(key)] = dict(lb)
            except Exception:
                pass

    def set_node_load_balance_metadata(self) -> None:
        try:
            from RL4KMC.runner.engine import (
                merge_load_balance_stats,
                write_node_load_balance_report,
            )

            node_stats = merge_load_balance_stats(
                [
                    dict(item)
                    for item in self.worker_load_balance.values()
                    if isinstance(item, dict)
                ]
            )
            node_summary = {
                "rank": int(self.rank),
                "host": str(socket.gethostname()),
                "rank_dir_name": str(getattr(self.output_runtime, "rank_dir_name", "")),
                **(node_stats or {}),
            }
            self.timing_stats.set_metadata(node_load_balance=dict(node_summary))

            output_level = int(getattr(self.output_runtime, "output_level", 1) or 1)
            if (
                output_level >= 2
                and self.output_runtime.output_dir is not None
                and self.output_runtime.rank_dir is not None
            ):
                write_node_load_balance_report(
                    rank_dir=str(self.output_runtime.rank_dir),
                    node_summary=dict(node_summary),
                    worker_details=dict(self.worker_load_balance),
                )
        except Exception:
            pass

    def summarize_into_timing(
        self, *, result_shm_init: bool, result_array: Any
    ) -> dict[str, Any]:
        payloads = self.collect_payloads(
            result_shm_init=bool(result_shm_init), result_array=result_array
        )
        try:
            self.timing_stats.set_metadata(workers_total=int(self.expected_num_workers))
        except Exception:
            pass
        self.merge_payloads(payloads)
        self.set_node_load_balance_metadata()

        return {
            "workers_reported": int(len(payloads)),
            "workers_expected": int(self.expected_num_workers),
            "total_tasks": int(self.timing_stats.total_tasks),
            "total_jumps": int(self.timing_stats.total_jumps),
            "total_steps": int(self.timing_stats.total_steps),
            "total_compute_time_seconds": float(self.timing_stats.total_compute_time),
        }
