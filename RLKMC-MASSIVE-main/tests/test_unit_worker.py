import json
import multiprocessing as mp
import sys
import types
from multiprocessing import shared_memory

import numpy as np
import torch


def _ensure_scheduler_stub():
    if "RL4KMC.runner.scheduler" in sys.modules:
        return

    class _TaskWindow:
        def __init__(self, start: int, end: int):
            self.start = int(start)
            self.end = int(end)

    class _StaticQueueScheduler:
        def __init__(self, *args, **kwargs):
            self.task_window = _TaskWindow(0, 0)

        def is_drained(self):
            return True

        def update_task_window(self):
            return None

    stub = types.ModuleType("RL4KMC.runner.scheduler")
    stub.TaskWindow = _TaskWindow
    stub.StaticQueueScheduler = _StaticQueueScheduler
    sys.modules["RL4KMC.runner.scheduler"] = stub


def _cleanup_worker(worker):
    task_shm = getattr(worker, "_test_task_shm", None)
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


def _build_worker(*, worker_id: int = 0, num_workers: int = 2, shm_name: str = "dummy"):
    _ensure_scheduler_stub()
    from RL4KMC.runner.services.worker import TASK_DTYPE, Worker

    args = types.SimpleNamespace(
        n_radial=1,
        n_axial=1,
        rescaled_sim_time=1.0,
        cu_density=0.0,
        v_density=1.0,
        bench_step=4,
        use_traditional_kmc=False,
    )
    num_tasks = 5
    task_shm = shared_memory.SharedMemory(create=True, size=TASK_DTYPE.itemsize * num_tasks)
    np.ndarray(shape=(num_tasks,), dtype=TASK_DTYPE, buffer=task_shm.buf)[:] = [
        (400.0, 0.0, 1.0, 1.0)
    ] * num_tasks

    worker = Worker(
        args=args,
        worker_id=worker_id,
        num_workers=num_workers,
        num_tasks=num_tasks,
        task_start=mp.Value("i", 0),
        task_end=mp.Value("i", 5),
        is_drained=mp.Value("b", False),
        cursur_lock=mp.Lock(),
        model=torch.nn.Identity(),
        shm_name=shm_name,
        task_shm_name=task_shm.name,
        cpu_list=[0],
        embedding_device=torch.device("cpu"),
    )
    worker._test_task_shm = task_shm
    return worker


def test_worker_claim_task_advances_window_and_stops_when_drained():
    from RL4KMC.config import CONFIG

    worker = _build_worker()
    try:
        claim_size = int(CONFIG.runner.worker_claim_size)
        assert worker.claim_task() is True
        assert (worker.task_window.start, worker.task_window.end) == (0, min(5, claim_size))

        assert worker.claim_task() is True
        second_start = min(5, claim_size)
        second_end = min(5, second_start + claim_size)
        assert (worker.task_window.start, worker.task_window.end) == (second_start, second_end)

        assert worker.claim_task() is True
        third_start = second_end
        third_end = min(5, third_start + claim_size)
        assert (worker.task_window.start, worker.task_window.end) == (third_start, third_end)

        worker.is_drained.value = True
        while third_end < 5:
            assert worker.claim_task() is True
            third_start = third_end
            third_end = min(5, third_start + claim_size)
            assert (worker.task_window.start, worker.task_window.end) == (third_start, third_end)
        assert worker.claim_task() is False
    finally:
        _cleanup_worker(worker)


def test_worker_claim_task_briefly_sleeps_when_temporarily_empty(monkeypatch):
    _ensure_scheduler_stub()
    from RL4KMC.runner.services import worker as worker_mod

    worker = _build_worker()
    try:
        worker.task_start.value = 5
        worker.task_end.value = 5
        worker.is_drained.value = False

        sleep_calls = []
        monkeypatch.setattr(worker_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)), raising=True)

        assert worker.claim_task() is True
        assert (worker.task_window.start, worker.task_window.end) == (5, 5)
        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 0.0

        worker.is_drained.value = True
        assert worker.claim_task() is False
        assert len(sleep_calls) == 1
    finally:
        _cleanup_worker(worker)


def test_worker_write_result_serializes_payload_to_shared_memory():
    _ensure_scheduler_stub()
    from RL4KMC.runner.services.worker import PAYLOAD_DTYPE

    num_workers = 2
    shm = shared_memory.SharedMemory(create=True, size=PAYLOAD_DTYPE.itemsize * num_workers)
    try:
        worker = _build_worker(worker_id=1, num_workers=num_workers, shm_name=shm.name)
        try:
            payload = {
                "worker_id": 1,
                "pid": 1234,
                "total_steps": 20,
                "total_compute_time": 1.25,
                "total_tasks": 3,
                "total_jumps": 9,
                "first_task_start_ts": 10.0,
                "last_task_end_ts": 12.0,
                "series": [{"step": 1, "t": 0.1}],
                "load_balance": {"rank": 0},
            }
            worker.write_result(dict(payload))

            arr = np.ndarray(shape=(num_workers,), dtype=PAYLOAD_DTYPE, buffer=shm.buf)
            row = arr[1]
            assert int(row["worker_id"]) == 1
            assert int(row["pid"]) == 1234
            assert int(row["total_tasks"]) == 3
            assert int(row["total_jumps"]) == 9

            blob = bytes(row["json_blob"]).rstrip(b"\x00")
            decoded = json.loads(blob.decode("utf-8"))
            assert decoded["series"] == [{"step": 1, "t": 0.1}]
            assert decoded["load_balance"] == {"rank": 0}
        finally:
            _cleanup_worker(worker)
    finally:
        shm.close()
        shm.unlink()


def test_worker_run_aggregates_tasks_and_emits_payload(monkeypatch):
    _ensure_scheduler_stub()
    from RL4KMC.runner.services import worker as worker_mod

    worker = _build_worker()
    try:
        worker.task_end.value = 3

        pin_called = {"cpu": None}
        monkeypatch.setattr(worker_mod, "pin_current_process", lambda cpus: pin_called.update(cpu=cpus), raising=True)
        monkeypatch.setattr(worker_mod, "set_pdeathsig", lambda *_a, **_k: None, raising=True)

        class FakeEngine:
            def run_one_task(self, task, **kwargs):
                assert "bench_step" in kwargs and "use_traditional_kmc" in kwargs
                return int(task["time"])

        class FakeTimingStats:
            total_steps = 99
            total_compute_time = 3.5
            first_task_start_ts = 1.0
            last_task_end_ts = 4.0
            series = [{"step": 0, "sec": 0.0}]

        fake_runtime = types.SimpleNamespace(engine=FakeEngine(), timing_stats=FakeTimingStats())
        monkeypatch.setattr(worker_mod, "build_engine_runtime", lambda **_k: fake_runtime, raising=True)

        windows = [types.SimpleNamespace(start=0, end=2), types.SimpleNamespace(start=2, end=3)]

        def fake_claim_task():
            if not windows:
                return False
            worker.task_window = windows.pop(0)
            return True

        monkeypatch.setattr(worker, "claim_task", fake_claim_task)

        captured = {}
        monkeypatch.setattr(worker, "write_result", lambda payload: captured.update(payload=payload))

        worker.run()

        assert pin_called["cpu"] == [0]
        assert captured["payload"]["worker_id"] == 0
        assert captured["payload"]["total_tasks"] == 3
        assert captured["payload"]["total_jumps"] == 3
        assert captured["payload"]["total_steps"] == 99
    finally:
        _cleanup_worker(worker)
