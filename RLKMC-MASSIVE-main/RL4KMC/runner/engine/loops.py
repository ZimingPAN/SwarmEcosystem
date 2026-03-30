from __future__ import annotations

import logging
import os
import random
import time
from typing import Any


_LOGGER = logging.getLogger(__name__)


# Single place to tune the worker task loop.
TASK_LOOP = {
    # How many task ids to dequeue at once.
    "dequeue_batch": 2,
    # Only worker 0 prints stats, once per this interval.
    "stats_interval_sec": 5.0,
    # Upper bound on idle sleep between empty polls.
    "max_sleep_sec": 0.2,
    # Add small jitter to avoid synchronized wakeups.
    "sleep_jitter_ratio": 0.1,
}


def _local_worker_id() -> int:
    try:
        return int(os.environ.get("TASK_LOCAL_WORKER_ID", "-1"))
    except Exception:
        return -1


class InProcessTaskQueue:
    """Single-process task id queue.

    This is used to keep all code paths queue-driven even when running
    without /dev/shm or multiprocessing.
    """

    def __init__(self, task_ids: list[int]) -> None:
        self._task_ids = [int(x) for x in task_ids]
        self._head = 0
        self._done = False
        self._enqueued_total = len(self._task_ids)
        self._completed_total = 0

    def try_dequeue(self) -> int | None:
        if self._head >= len(self._task_ids):
            return None
        v = int(self._task_ids[self._head])
        self._head += 1
        return int(v)

    def try_dequeue_many(self, max_n: int) -> list[int]:
        try:
            n_req = int(max_n)
        except Exception:
            n_req = 1
        if n_req <= 0:
            return []

        if self._head >= len(self._task_ids):
            return []
        end = int(min(len(self._task_ids), int(self._head + n_req)))
        out = [int(x) for x in self._task_ids[self._head : end]]
        self._head = int(end)
        return list(out)

    def mark_task_completed(self, n: int = 1) -> None:
        self._completed_total += int(n)
        if self._completed_total >= self._enqueued_total:
            self._done = True

    def is_done(self) -> bool:
        if self._enqueued_total == 0:
            return True
        return bool(self._done)

    def occupancy(self) -> int:
        return int(max(0, len(self._task_ids) - self._head))

    def get_totals(self) -> tuple[int, int]:
        return int(self._enqueued_total), int(self._completed_total)


def run_task_dispatch_loop(
    *,
    engine: Any,
    task_manager: Any,
    use_dist_tasks: bool,
    tasks: list[dict],
    store: Any | None,
    output_dir: str | None,
    enable_rank_detail: bool,
    rank_dir_name: str,
    enable_behavior_log: bool,
    enable_cu_cluster_sampling: bool,
    bench_step: int,
    use_traditional_kmc: bool,
    energy_series_interval_steps: int,
    poll_interval_sec: float = 0.01,
) -> tuple[int, int]:
    raise RuntimeError(
        "Legacy dispatch-loop scheduling has been removed. "
        "Use queue-driven scheduling (run_task_id_queue_loop) with a scheduler store that provides a queue."
    )


def run_task_id_queue_loop(
    *,
    engine: Any,
    tasks: list[dict],
    queue: Any,
    leader_driver: Any | None = None,
    output_dir: str | None,
    enable_rank_detail: bool,
    rank_dir_name: str,
    enable_behavior_log: bool,
    enable_cu_cluster_sampling: bool,
    bench_step: int,
    use_traditional_kmc: bool,
    energy_series_interval_steps: int,
    poll_interval_sec: float = 0.01,
) -> tuple[int, int]:
    """Run a local queue-driven task loop (used by node_mp workers).

    The queue is expected to support: try_dequeue(), is_done(), mark_task_completed(n).
    Returns (tasks_completed, jumps_completed).

    This module intentionally keeps the queue polling loop out of the compute engine.
    """

    tasks_completed_local = 0
    jumps_completed_local = 0

    # Best-effort wall-clock compute window for this worker process.
    # This is intentionally recorded at the orchestration layer (queue loop)
    # so it remains available even if the engine implementation changes.
    task_window_start_ts: float | None = None
    task_window_end_ts: float | None = None

    # Base poll sleep; keep a small floor to avoid accidental busy-polling.
    poll = float(max(0.0005, float(poll_interval_sec)))
    max_sleep = float(TASK_LOOP["max_sleep_sec"])

    local_worker_id = int(_local_worker_id())
    log_stats = int(local_worker_id) == 0

    dequeue_batch = int(max(1, min(256, int(TASK_LOOP["dequeue_batch"])) ))
    stats_interval = float(TASK_LOOP["stats_interval_sec"])
    jitter_ratio = float(max(0.0, float(TASK_LOOP["sleep_jitter_ratio"])))

    _rng = random.Random(int(os.getpid()) ^ int(time.time() * 1e6))

    idle_backoff = 0
    last_stats = time.time()

    # Window stats (reset after each stats log).
    win_loops = 0
    win_empty_loops = 0
    win_batches = 0
    win_tasks = 0

    while True:
        win_loops += 1

        if leader_driver is not None:
            # Best-effort: keep producer progressing when co-located.
            try:
                leader_driver.tick()
            except Exception:
                pass

        got = queue.try_dequeue_many(int(dequeue_batch))
        tids = [int(x) for x in (got or [])]

        if tids:
            win_batches += 1
            win_tasks += int(len(tids))
            idle_backoff = 0

            processed = 0
            executed_ids: list[int] = []
            for tid_i in list(tids):
                try:
                    tid_i = int(tid_i)
                except Exception:
                    continue
                if tid_i < 0 or tid_i >= len(tasks):
                    raise RuntimeError(
                        f"Bad task id dequeued: tid={tid_i} len(tasks)={len(tasks)} "
                        f"pid={os.getpid()} worker_id={local_worker_id}"
                    )

                t0 = time.time()
                jump_counter = engine.run_one_task(
                    dict(tasks[tid_i]),
                    output_dir=output_dir,
                    enable_rank_detail=bool(enable_rank_detail),
                    rank_dir_name=str(rank_dir_name),
                    enable_behavior_log=bool(enable_behavior_log),
                    enable_cu_cluster_sampling=bool(enable_cu_cluster_sampling),
                    bench_step=int(bench_step),
                    use_traditional_kmc=bool(use_traditional_kmc),
                    energy_series_interval_steps=int(energy_series_interval_steps),
                )
                t1 = time.time()
                if task_window_start_ts is None or float(t0) < float(task_window_start_ts):
                    task_window_start_ts = float(t0)
                if task_window_end_ts is None or float(t1) > float(task_window_end_ts):
                    task_window_end_ts = float(t1)
                try:
                    timing_stats = getattr(engine, "timing_stats", None)
                    if (
                        timing_stats is not None
                        and task_window_start_ts is not None
                        and task_window_end_ts is not None
                        and hasattr(timing_stats, "add_task_window")
                    ):
                        timing_stats.add_task_window(float(t0), float(t1))
                except Exception:
                    pass
                tasks_completed_local += 1
                jumps_completed_local += int(jump_counter)
                processed += 1
                executed_ids.append(int(tid_i))

            if processed > 0:
                queue.mark_task_completed(int(processed))
                # Optional: per-chunk completion tracking for pipelined schedulers.
                try:
                    m = getattr(queue, "mark_task_ids_completed", None)
                    if callable(m):
                        m(list(executed_ids))
                except Exception:
                    pass
            if log_stats:
                now = time.time()
                if float(now - last_stats) >= float(stats_interval):
                    last_stats = float(now)
                    occ = -1
                    try:
                        if hasattr(queue, "occupancy"):
                            occ = int(queue.occupancy())
                    except Exception:
                        occ = -1
                    loops = int(max(1, int(win_loops)))
                    empty_ratio = float(win_empty_loops) / float(loops)
                    avg_batch = float(win_tasks) / float(max(1, int(win_batches)))
                    _LOGGER.debug(
                        "worker queue stats rank=%s worker=%s occ=%s loops=%s empty_ratio=%.3f batches=%s avg_batch=%.2f",
                        int(getattr(engine, "rank", -1)),
                        int(local_worker_id),
                        int(occ),
                        int(win_loops),
                        float(empty_ratio),
                        int(win_batches),
                        float(avg_batch),
                    )
                    win_loops = 0
                    win_empty_loops = 0
                    win_batches = 0
                    win_tasks = 0
            continue

        if bool(queue.is_done()):
            break
        win_empty_loops += 1

        if log_stats:
            now = time.time()
            if float(now - last_stats) >= float(stats_interval):
                last_stats = float(now)
                occ = -1
                try:
                    if hasattr(queue, "occupancy"):
                        occ = int(queue.occupancy())
                except Exception:
                    occ = -1
                loops = int(max(1, int(win_loops)))
                empty_ratio = float(win_empty_loops) / float(loops)
                avg_batch = float(win_tasks) / float(max(1, int(win_batches)))
                _LOGGER.debug(
                    "worker queue stats rank=%s worker=%s occ=%s loops=%s empty_ratio=%.3f batches=%s avg_batch=%.2f",
                    int(getattr(engine, "rank", -1)),
                    int(local_worker_id),
                    int(occ),
                    int(win_loops),
                    float(empty_ratio),
                    int(win_batches),
                    float(avg_batch),
                )
                win_loops = 0
                win_empty_loops = 0
                win_batches = 0
                win_tasks = 0

        # IMPORTANT: avoid busy-polling when the local queue is empty.
        # Busy loops across many worker processes cause excessive context switching
        # and can show up as high system CPU usage.
        # Exponential backoff when idle.
        idle_backoff = int(min(8, idle_backoff + 1))
        sleep_sec = float(min(float(max_sleep), float(poll) * float(2**idle_backoff)))
        if jitter_ratio > 0:
            # jitter in [1-j, 1+j]
            sleep_sec *= float(1.0 + (2.0 * float(_rng.random()) - 1.0) * float(jitter_ratio))
        time.sleep(float(max(0.0, sleep_sec)))

    return int(tasks_completed_local), int(jumps_completed_local)
