from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from RL4KMC.utils.util import StepCSVLogger, StepTimerController, TaskStepTimer

class _NoopStepTimer:
    @staticmethod
    def start_step(step_idx: int) -> None:
        return None

    @staticmethod
    def mark(tag: str) -> None:
        return None

    @staticmethod
    def end_step() -> None:
        return None

class _LocalStepStats:
    """Collect step timing in-memory to avoid per-step IO."""

    def __init__(self) -> None:
        self._series = {}
        self._total_steps = 0
        self._total_compute_time = 0.0
        self._total_tasks = 0
        self._total_jumps = 0
        self._first_task_start_ts = None
        self._last_task_end_ts = None
        self._metadata = {}

    def add_task_window(self, start_ts: float, end_ts: float) -> None:
        """Track the wall-clock compute window covered by tasks on this rank."""

        try:
            s = float(start_ts)
            e = float(end_ts)
        except Exception:
            return

        if self._first_task_start_ts is None or float(s) < float(self._first_task_start_ts):
            self._first_task_start_ts = float(s)
        if self._last_task_end_ts is None or float(e) > float(self._last_task_end_ts):
            self._last_task_end_ts = float(e)

    def add_step(self, durations: dict, total_time: float) -> None:
        self._total_steps += 1
        self._total_compute_time += float(total_time)
        for label, dt in durations.items():
            self._series.setdefault(str(label), []).append(float(dt))

    def add_task(self, count: int = 1) -> None:
        try:
            self._total_tasks += int(count)
        except Exception:
            pass

    def add_jumps(self, count: int = 1) -> None:
        try:
            self._total_jumps += int(count)
        except Exception:
            pass

    def set_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            self._metadata[str(key)] = value

    @property
    def total_steps(self) -> int:
        return int(self._total_steps)

    @property
    def total_compute_time(self) -> float:
        return float(self._total_compute_time)

    @property
    def total_tasks(self) -> int:
        return int(self._total_tasks)

    @property
    def total_jumps(self) -> int:
        return int(self._total_jumps)

    @property
    def first_task_start_ts(self) -> float | None:
        try:
            return None if self._first_task_start_ts is None else float(self._first_task_start_ts)
        except Exception:
            return None

    @property
    def last_task_end_ts(self) -> float | None:
        try:
            return None if self._last_task_end_ts is None else float(self._last_task_end_ts)
        except Exception:
            return None

    @property
    def series(self) -> dict:
        return self._series

    @property
    def metadata(self) -> dict:
        return self._metadata

    def get_metadata(self, key: str, default: Any = None) -> Any:
        try:
            return self._metadata.get(str(key), default)
        except Exception:
            return default


class _LocalStepTimer:
    """Step timer that records timings into _LocalStepStats."""

    def __init__(self, stats: _LocalStepStats, enabled: bool = True) -> None:
        self._stats = stats
        self.enabled = bool(enabled)
        self._t0 = None
        self._checkpoints = []

    def start_step(self, step_idx):
        if not self.enabled:
            return
        self._t0 = time.perf_counter()
        self._checkpoints = []

    def mark(self, label):
        if not self.enabled or self._t0 is None:
            return
        self._checkpoints.append((str(label), time.perf_counter()))

    def end_step(self):
        if not self.enabled or self._t0 is None:
            return
        t_end = time.perf_counter()
        total = float(t_end - self._t0)
        labels = []
        dts = []
        prev = self._t0
        for lab, ts in self._checkpoints:
            labels.append(lab)
            dts.append(float(ts - prev))
            prev = ts
        labels.append("other")
        dts.append(float(t_end - prev))
        durations = {lab: dt for lab, dt in zip(labels, dts)}
        self._stats.add_step(durations, total)
        self._t0 = None
        self._checkpoints = []


class _CompositeStepTimer:
    """Forward step timing to multiple timers (e.g., file + in-memory)."""

    def __init__(self, timers):
        self._timers = [t for t in timers if t is not None]

    def start_step(self, step_idx):
        for t in self._timers:
            t.start_step(step_idx)

    def mark(self, label):
        for t in self._timers:
            t.mark(label)

    def end_step(self):
        for t in self._timers:
            t.end_step()


@dataclass
class RunOutputPaths:
    rank_dir: str
    detail_fp: str
    comm_fp: str
    global_fp: str
    init_fp: str
    cu_moves_fp: str
    energy_drops_fp: str
    energy_series_fp: str


@dataclass
class TaskLogPaths:
    task_timing: str
    env_apply: str
    env_rl_select: str
    env_update: str


class RunOutputManager:
    """Manage per-rank output directory creation and file paths."""

    def __init__(self, enable_rank_detail: bool) -> None:
        self.enable_rank_detail = bool(enable_rank_detail)

    def set_run_output_paths(self, dirpath: Optional[str]) -> Optional[RunOutputPaths]:
        if dirpath is None or not self.enable_rank_detail:
            return None
        os.makedirs(dirpath, exist_ok=True)
        rank_dir = str(dirpath)
        return RunOutputPaths(
            rank_dir=rank_dir,
            detail_fp=os.path.join(rank_dir, "detail.out"),
            comm_fp=os.path.join(rank_dir, "comm.out"),
            global_fp=os.path.join(rank_dir, "global_status.out"),
            init_fp=os.path.join(rank_dir, "initial.out"),
            cu_moves_fp=os.path.join(rank_dir, "cu_moves.csv"),
            energy_drops_fp=os.path.join(rank_dir, "energy_drops.csv"),
            energy_series_fp=os.path.join(rank_dir, "energy_series.csv"),
        )


class TaskLoggerFactory:
    """Build per-task log paths and attach CSV/timing loggers to env."""

    def __init__(
        self,
        output_level: int,
        enable_timing_log: bool,
        enable_step_timer: bool,
        output_to_terminal: bool,
        enable_rank_detail: bool,
    ) -> None:
        self.output_level = int(output_level)
        self.enable_timing_log = bool(enable_timing_log)
        self.enable_step_timer = bool(enable_step_timer)
        self.output_to_terminal = bool(output_to_terminal)
        self.enable_rank_detail = bool(enable_rank_detail)
        self.enable_timing_io = (
            self.output_level >= 2
            and self.enable_timing_log
            and self.enable_step_timer
            and not self.output_to_terminal
        )

    def update_enable_rank_detail(self, enable_rank_detail: bool) -> None:
        self.enable_rank_detail = bool(enable_rank_detail)

    def format_task_val(self, v: Any) -> str:
        if v is None:
            return "na"
        try:
            s = f"{float(v):g}"
        except Exception:
            s = str(v)
        return s.replace("-", "m").replace(".", "p").replace("+", "")

    def build_task_log_paths(
        self,
        run_paths: RunOutputPaths,
        assigned_time: Any,
        assigned_temp: Any,
        assigned_cu: Any,
        assigned_vac: Any,
    ) -> TaskLogPaths:
        suffix = (
            f"_time{self.format_task_val(assigned_time)}"
            f"_temp{self.format_task_val(assigned_temp)}"
            f"_cu{self.format_task_val(assigned_cu)}"
            f"_vac{self.format_task_val(assigned_vac)}.csv"
        )
        base = os.path.dirname(run_paths.cu_moves_fp)
        return TaskLogPaths(
            task_timing=os.path.join(base, f"task_step_timing{suffix}"),
            env_apply=os.path.join(base, f"task_env_apply_timing{suffix}"),
            env_rl_select=os.path.join(base, f"task_env_rl_select_timing{suffix}"),
            env_update=os.path.join(base, f"task_env_update_pipeline_timing{suffix}"),
        )

    def setup_task_loggers(
        self,
        env: Any,
        timing_stats: _LocalStepStats,
        assigned_time: Any,
        assigned_temp: Any,
        assigned_cu: Any,
        assigned_vac: Any,
        run_paths: Optional[RunOutputPaths],
    ):
        if run_paths is None or not self.enable_rank_detail:
            local_timer = _LocalStepTimer(timing_stats, enabled=True)
            env._bench_apply_logger = None
            env._bench_rl_select_logger = None
            env._bench_update_logger = None
            return local_timer
        paths = self.build_task_log_paths(
            run_paths, assigned_time, assigned_temp, assigned_cu, assigned_vac
        )
        local_timer = _LocalStepTimer(timing_stats, enabled=True)
        step_timer = local_timer
        if self.enable_timing_io:
            file_timer = StepTimerController(
                TaskStepTimer(paths.task_timing, max_steps=20),
                enabled=True,
            )
            step_timer = _CompositeStepTimer([local_timer, file_timer])

        env._bench_step_limit = 10
        env._bench_apply_logger = (
            StepCSVLogger(
                paths.env_apply,
                columns=[
                    "method",
                    "t_select",
                    "t_pos",
                    "t_energy",
                    "t_energy_pre",
                    "t_energy_post",
                    "t_energy_delta",
                    "t_move",
                    "t_topk",
                    "t_ratei",
                    "t_rateu",
                    "t_feat",
                    "t_total",
                    "vac_id",
                    "dir_idx",
                    "delta_t",
                    "delta_E",
                ],
            )
            if self.enable_timing_io
            else None
        )
        env._bench_rl_select_logger = (
            StepCSVLogger(
                paths.env_rl_select,
                columns=[
                    "t_mask_sample",
                    "t_rate_sum",
                    "t_delta_t",
                    "t_total",
                    "total_rate",
                    "chosen_idx",
                    "masked_total",
                    "chosen_is_masked",
                    "vac_id",
                    "dir_idx",
                    "delta_t",
                ],
            )
            if self.enable_timing_io
            else None
        )
        env._bench_update_logger = (
            StepCSVLogger(
                paths.env_update,
                columns=[
                    "t_move_vacancy",
                    "t_move_cu",
                    "t_update_local_env",
                    "t_update_system",
                    "t_total",
                    "vac_id",
                    "moving_type",
                    "cu_id",
                    "cu_topk_id",
                ],
            )
            if self.enable_timing_io
            else None
        )
        return step_timer


def build_output_managers(
    *,
    output_level: int,
    enable_timing_log: bool,
    enable_step_timer: bool,
    output_to_terminal: bool,
    enable_rank_detail: bool,
) -> tuple[RunOutputManager, TaskLoggerFactory]:
    """Factory used by the runner to keep its __init__ smaller."""

    run_output_manager = RunOutputManager(
        enable_rank_detail=bool(enable_rank_detail)
    )
    task_logger_factory = TaskLoggerFactory(
        output_level=int(output_level),
        enable_timing_log=bool(enable_timing_log),
        enable_step_timer=bool(enable_step_timer),
        output_to_terminal=bool(output_to_terminal),
        enable_rank_detail=bool(enable_rank_detail),
    )
    return run_output_manager, task_logger_factory


# Public aliases (avoid importing underscore-prefixed names across modules)
LocalStepStats = _LocalStepStats
LocalStepTimer = _LocalStepTimer
CompositeStepTimer = _CompositeStepTimer
