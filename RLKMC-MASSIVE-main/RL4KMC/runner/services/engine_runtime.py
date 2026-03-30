from __future__ import annotations

import logging
import torch
from dataclasses import dataclass
from typing import Any

from RL4KMC.envs.distributed_kmc import DistributedKMCEnv
from RL4KMC.runner.services.output_manager import LocalStepStats, build_output_managers
from RL4KMC.runner.engine.compute_engine import KMCComputeEngine
from RL4KMC.runner.services.output_runtime import OutputRuntime, resolve_output_runtime

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineRuntime:
    engine: KMCComputeEngine
    timing_stats: LocalStepStats
    output: OutputRuntime


def build_engine_runtime(
    *, args: Any, worker_id: int, embed: torch.nn.Module, embed_device: torch.device
) -> EngineRuntime:
    """Build KMCComputeEngine + its dependencies for services/workers."""

    output = resolve_output_runtime(args=args, worker_id=worker_id)

    run_output_manager, task_logger_factory = build_output_managers(
        output_level=int(output.output_level),
        enable_timing_log=bool(output.enable_timing_log),
        enable_step_timer=bool(output.enable_step_timer),
        output_to_terminal=bool(output.output_to_terminal),
        enable_rank_detail=bool(output.enable_rank_detail),
    )

    timing_stats = LocalStepStats()

    engine = KMCComputeEngine(
        args=args,
        embed=embed,
        embed_device=embed_device,
        worker_id=worker_id,
        enable_incremental_policy=args.enable_incremental_policy,
        task_logger_factory=task_logger_factory,
        timing_stats=timing_stats,
    )

    return EngineRuntime(
        engine=engine,
        timing_stats=timing_stats,
        output=output,
    )
