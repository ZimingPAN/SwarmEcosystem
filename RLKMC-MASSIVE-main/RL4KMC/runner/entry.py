from __future__ import annotations

import logging
import os
import random
import sys
import time
import socket
from typing import Any


from RL4KMC.config import CONFIG
from RL4KMC.runner.services import Leader

_LOGGER = logging.getLogger(__name__)


def setup_logging(level: str | None = None) -> None:
    level_name = (
        level
        or os.environ.get("KMC_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL")
        or "INFO"
    )
    log_level = getattr(logging, str(level_name).upper(), logging.INFO)

    hostname = socket.gethostname()

    fmt = (
        f"[hostname {hostname} pid {os.getpid()} ] %(levelname)s "
        "%(filename)s:%(lineno)d: %(message)s"
    )

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=log_level, format=fmt, stream=sys.stdout)
    else:
        root.setLevel(log_level)
        for handler in root.handlers:
            try:
                handler.setFormatter(logging.Formatter(fmt))
            except Exception:
                pass


def eval_ssa(args: Any) -> None:
    setup_logging("DEBUG" if bool(getattr(args, "debug", False)) else None)

    # 简单的打散启动压力
    time.sleep(random.random() * 5.0)
    import torch

    device = torch.device("cpu")

    leader = Leader(
        args=args,
        comm_backend_type=CONFIG.runner.comm_backend,
        scheduler_type=CONFIG.runner.scheduler_type,
        device=device,
    )

    leader.run()
    return
