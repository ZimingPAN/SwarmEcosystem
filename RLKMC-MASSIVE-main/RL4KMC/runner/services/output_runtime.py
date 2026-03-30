from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class OutputRuntime:
    output_level: int
    output_dir: str | None
    output_to_terminal: bool
    enable_rank_detail: bool
    rank_dir: str | None
    rank_dir_name: str
    enable_timing_log: bool
    enable_step_timer: bool
    enable_timing_report: bool
    enable_behavior_log: bool
    enable_timing_io: bool


def _clamp_output_level(v: Any) -> int:
    try:
        out = int(v)
    except Exception:
        out = 1
    if out < 0:
        out = 0
    if out > 2:
        out = 2
    return int(out)

def resolve_output_runtime(
    *,
    args: Any,
    worker_id: int | None = None,
    process_name: str | None = None,
) -> OutputRuntime:
    """Resolve output paths/flags without importing torch.

    Note:
    - Uses env var `TASK_RUN_OUTPUT_DIR` if set, to keep node_mp workers consistent.
    - Otherwise matches runner's default run-dir structure.
    """
    nnodes = int(getattr(args, "nodes", 1) or 1)
    output_level = _clamp_output_level(getattr(args, "output_level", 1))

    enable_timing_log = bool(getattr(args, "enable_timing_log", True))
    enable_step_timer = bool(getattr(args, "enable_step_timer", True))
    enable_timing_report = bool(getattr(args, "enable_timing_report", True))
    enable_behavior_log = bool(getattr(args, "enable_behavior_log", False))

    raw_output_dir = getattr(args, "output_dir", None)
    if raw_output_dir is None or str(raw_output_dir).strip() == "":
        output_dir = None
    else:
        output_dir = str(raw_output_dir)

    output_to_terminal = output_dir is None

    enable_rank_detail = bool(output_level >= 2) and (output_dir is not None)

    if process_name is None or str(process_name).strip() == "":
        rank_dir_name = f"host{socket.gethostname()}_pid{os.getpid()}"
        
    else:
        rank_dir_name = str(process_name).strip()

    rank_dir = None
    if enable_rank_detail and output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass
        rank_base_dir = os.path.join(output_dir, "rank-detail")
        try:
            os.makedirs(rank_base_dir, exist_ok=True)
        except Exception:
            pass
        rank_dir = os.path.join(rank_base_dir, str(rank_dir_name))
        if worker_id is not None:
            rank_dir = os.path.join(rank_dir, f"w{worker_id}")
        try:
            os.makedirs(rank_dir, exist_ok=True)
        except Exception:
            pass

    enable_timing_io = (
        output_level >= 2
        and enable_timing_log
        and enable_step_timer
        and (not output_to_terminal)
    )

    return OutputRuntime(
        output_level=int(output_level),
        output_dir=output_dir,
        output_to_terminal=bool(output_to_terminal),
        enable_rank_detail=bool(enable_rank_detail),
        rank_dir=rank_dir,
        rank_dir_name=str(rank_dir_name),
        enable_timing_log=bool(enable_timing_log),
        enable_step_timer=bool(enable_step_timer),
        enable_timing_report=bool(enable_timing_report),
        enable_behavior_log=bool(enable_behavior_log),
        enable_timing_io=bool(enable_timing_io),
    )
