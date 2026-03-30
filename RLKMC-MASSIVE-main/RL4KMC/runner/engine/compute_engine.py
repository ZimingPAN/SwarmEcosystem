from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, cast
import numpy as np
import torch

from RL4KMC.envs.distributed_kmc import DistributedKMCEnv, KMCObs
from RL4KMC.runner.services.output_manager import (
    LocalStepStats,
    RunOutputManager,
    RunOutputPaths,
    TaskLoggerFactory,
    _NoopStepTimer,
)
from RL4KMC.utils.util import SSAEvalFileLogger
from RL4KMC.runner.engine import TaskTimeRecord, record_task_time, reinit_env_for_task


_LOGGER = logging.getLogger(__name__)


class KMCComputeEngine:
    """Simplified decoupled compute engine.

    This engine keeps only compute-state and core compute operations:
    - env binding/reinit
    - TopK cache updates
    - policy logits inference (full / incremental)
    - single-step KMC execution (traditional / RL)
    """

    def __init__(
        self,
        *,
        args: Any,
        embed: torch.nn.Module,
        embed_device: torch.device,
        worker_id: int,
        enable_incremental_policy: bool = False,
        output_manager: RunOutputManager | None = None,
        task_logger_factory: TaskLoggerFactory | None = None,
        timing_stats: LocalStepStats | None = None,
    ) -> None:
        self.args = args
        self.embed = embed
        self.embed_device = embed_device
        self.worker_id = worker_id
        self.env = DistributedKMCEnv(args)
        self.enable_incremental_policy = bool(enable_incremental_policy)

        # 注入的服务，由调用者负责提供
        self.output_manager = output_manager
        self.task_logger_factory = task_logger_factory
        self.timing_stats = timing_stats

        self.diff_k_cache: torch.Tensor | None = None
        self.dist_k_cache: torch.Tensor | None = None
        self.logits_cache: torch.Tensor | None = None
        self.changed_vids_global: set[int] = set()

    @staticmethod
    def _timer_mark(step_timer: Any, tag: str) -> None:
        try:
            if step_timer is not None and hasattr(step_timer, "mark"):
                step_timer.mark(str(tag))
        except Exception:
            pass

    def bind_env(
        self,
        *,
        env: Any,
        diff_k_cache: torch.Tensor | None = None,
        dist_k_cache: torch.Tensor | None = None,
    ) -> None:
        self.env = env
        self.diff_k_cache = diff_k_cache
        self.dist_k_cache = dist_k_cache
        self.logits_cache = None
        self.changed_vids_global = set()

    def reinit_env_for_task(self, task: dict) -> None:
        env, diff_k_cache, dist_k_cache = reinit_env_for_task(
            task,
            self.args,
            self.embed_device,
        )
        self.bind_env(env=env, diff_k_cache=diff_k_cache, dist_k_cache=dist_k_cache)

    def update_topk_cache_from_obs(self, obs: KMCObs | None) -> None:
        if obs is None:
            return
        if not isinstance(obs, KMCObs):
            raise TypeError(f"expected KMCObs, got {type(obs)}")

        tk = obs.topk_update_info
        if not isinstance(tk, dict):
            return

        vid_list = tk.get("vid_list", [])
        diff_k = tk.get("diff_k")
        dist_k = tk.get("dist_k")
        if (
            diff_k is None
            or dist_k is None
            or not vid_list
            or self.diff_k_cache is None
            or self.dist_k_cache is None
        ):
            return

        device = self.diff_k_cache.device
        diff_k = diff_k.to(device)
        dist_k = dist_k.to(device)
        for i, vid in enumerate(vid_list):
            try:
                vid_i = int(vid)
            except Exception:
                continue
            self.diff_k_cache[vid_i] = diff_k[i]
            self.dist_k_cache[vid_i] = dist_k[i]

    def get_logits(self, V_feat: torch.Tensor) -> torch.Tensor:
        if self.diff_k_cache is None or self.dist_k_cache is None:
            try:
                topk_all = self.env.topk_sys.get_all_topk_tensors()
                device0 = self.embed_device
                self.diff_k_cache = topk_all["diff_k"].to(device0)
                self.dist_k_cache = topk_all["dist_k"].to(device0)
            except Exception:
                raise RuntimeError("topk cache not initialized")

        diff_k_cache = self.diff_k_cache
        dist_k_cache = self.dist_k_cache
        if diff_k_cache is None or dist_k_cache is None:
            raise RuntimeError("topk cache not initialized")

        device = diff_k_cache.device
        Nv_local = int(V_feat.shape[0])
        local_indices = list(range(Nv_local))

        if not self.enable_incremental_policy:
            sel_idx = torch.as_tensor(local_indices, dtype=torch.long, device=device)
            diff_k = diff_k_cache[sel_idx]
            dist_k = dist_k_cache[sel_idx]
            logits_full = self.embed(V_feat, diff_k, dist_k)
            self.logits_cache = logits_full.detach()
            self.changed_vids_global = set()
            return cast(torch.Tensor, self.logits_cache)

        if self.logits_cache is None:
            sel_idx = torch.as_tensor(local_indices, dtype=torch.long, device=device)
            diff_k = diff_k_cache[sel_idx]
            dist_k = dist_k_cache[sel_idx]
            logits_full = self.embed(V_feat, diff_k, dist_k)
            self.logits_cache = logits_full.detach()
            self.changed_vids_global = set()
            return cast(torch.Tensor, self.logits_cache)

        if self.changed_vids_global:
            sel_idx = torch.as_tensor(
                list(self.changed_vids_global), dtype=torch.long, device=device
            )
            diff_k = diff_k_cache[sel_idx]
            dist_k = dist_k_cache[sel_idx]
            V_subset = V_feat[sel_idx]
            logits_subset = self.embed(V_subset, diff_k, dist_k)
            self.logits_cache.index_copy_(
                0,
                sel_idx.to(device=self.logits_cache.device),
                logits_subset.detach().to(
                    device=self.logits_cache.device, dtype=self.logits_cache.dtype
                ),
            )
            self.changed_vids_global = set()

        return cast(torch.Tensor, self.logits_cache)

    def step_traditional_kmc(
        self,
        step_timer: Any | None = None,
    ) -> KMCObs | None:
        features = self.env.get_vacancy_neighbor_features(as_torch=False)
        if features.shape[0] == 0:
            return None

        self._timer_mark(step_timer, "get_feat")
        self._timer_mark(step_timer, "inference")
        obs = self.env.apply_fast_jump(
            None,
            None,
            None,
            method="traditional",
            logits=None,
            features=features,
        )
        self._timer_mark(step_timer, "kmc_step")
        if not isinstance(obs, KMCObs):
            raise TypeError(f"expected KMCObs, got {type(obs)}")
        return obs

    def step_rl_kmc(
        self, step_timer: Any | None = None, benchmark: bool = False
    ) -> KMCObs | None:
        device_logits = self.embed_device
        V_feat = self.env.get_vacancy_neighbor_features(
            as_torch=True, device=device_logits, dtype=torch.float32
        )
        if not isinstance(V_feat, torch.Tensor):
            V_feat = torch.as_tensor(V_feat, device=device_logits, dtype=torch.float32)
        if int(V_feat.shape[0]) == 0:
            return None

        self._timer_mark(step_timer, "get_feat")
        with torch.inference_mode():
            logits = self.get_logits(V_feat)
        self._timer_mark(step_timer, "inference")

        if benchmark:
            obs = self.env.bench_fast_jump(logits=logits)
        else:
            obs = self.env.apply_fast_jump(
                None,
                None,
                None,
                method="rl",
                logits=logits,
                features=None,
            )
        self._timer_mark(step_timer, "kmc_step")

        if not isinstance(obs, KMCObs):
            raise TypeError(f"expected KMCObs, got {type(obs)}")
        self.changed_vids_global = set(obs.changed_vids) if obs.changed_vids else set()
        return obs

    def run_one_task(
        self,
        current_task: dict,
        *,
        bench_step: int,
        use_traditional_kmc: bool,
    ) -> int:
        """Run one task with minimal orchestration and no output IO.

        This method intentionally keeps only compute-related behavior.
        Returns jump count executed for this task.
        """

        assigned_temp = current_task.get("temp")
        assigned_time = current_task.get("time")
        assigned_cu = current_task.get("cu_density")
        assigned_vac = current_task.get("v_density")

        if assigned_temp is None:
            return 0
        try:
            if int(assigned_temp) < 0:
                return 0
        except Exception:
            pass

        task_start_ts = time.time()
        assigned_time_f = (
            float(assigned_time) if assigned_time is not None else float("inf")
        )
        self.reinit_env_for_task(
            {
                "temp": assigned_temp,
                "cu_density": assigned_cu,
                "v_density": assigned_vac,
            }
        )

        step_timer: Any = _NoopStepTimer()
        if self.task_logger_factory is not None and self.timing_stats is not None:
            try:
                step_timer = self.task_logger_factory.setup_task_loggers(
                    env=self.env,
                    timing_stats=self.timing_stats,
                    assigned_time=assigned_time,
                    assigned_temp=assigned_temp,
                    assigned_cu=assigned_cu,
                    assigned_vac=assigned_vac,
                    run_paths=None,
                )
            except Exception:
                step_timer = _NoopStepTimer()

        jump_counter = 0

        if int(bench_step) > 0:
            while jump_counter < int(bench_step):
                step_timer.start_step(int(jump_counter))
                setattr(self.env, "_bench_step_idx", int(jump_counter))

                obs = (
                    self.step_traditional_kmc(step_timer)
                    if bool(use_traditional_kmc)
                    else self.step_rl_kmc(step_timer, benchmark=True)
                )
                if obs is None:
                    break

                self.update_topk_cache_from_obs(obs)
                step_timer.mark("cache_update")
                step_timer.end_step()
                jump_counter += 1
        else:
            while float(self.env.time) < float(assigned_time_f):
                step_timer.start_step(int(jump_counter))
                setattr(self.env, "_bench_step_idx", int(jump_counter))

                obs = (
                    self.step_traditional_kmc(step_timer)
                    if bool(use_traditional_kmc)
                    else self.step_rl_kmc(step_timer, benchmark=False)
                )
                if obs is None:
                    break

                self.update_topk_cache_from_obs(obs)
                step_timer.mark("cache_update")
                step_timer.end_step()
                jump_counter += 1

        task_end_ts = time.time()

        try:
            if self.timing_stats is not None and hasattr(
                self.timing_stats, "add_task_window"
            ):
                self.timing_stats.add_task_window(
                    float(task_start_ts), float(task_end_ts)
                )
        except Exception:
            pass

        try:
            if self.timing_stats is not None:
                if hasattr(self.timing_stats, "add_task"):
                    self.timing_stats.add_task(1)
                if hasattr(self.timing_stats, "add_jumps"):
                    self.timing_stats.add_jumps(int(jump_counter))
        except Exception:
            pass

        return int(jump_counter)
