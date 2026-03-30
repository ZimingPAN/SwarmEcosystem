from __future__ import annotations

import logging
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from RL4KMC.utils.env import EnvKeys, env_str


def _get_first_int_env(names: Iterable[str]) -> Optional[int]:
    for name in names:
        try:
            s = env_str(str(name), None)
            if s is None or str(s).strip() == "":
                continue
            return int(str(s).strip())
        except Exception:
            continue
    return None

@dataclass(frozen=True)
class RankLoggingConfig:
    log_dir: str = "logs"
    run_id: str | None = None
    rank: int = 0
    world_size: int = 1
    output_level: int = 1
    rank_dir_name: str | None = None
    level: int = logging.INFO
    console: bool = True
    console_only_rank0: bool = True
    filename_prefix: str = "kmc"


def _default_rank_dir_name() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    base = f"host{host}_pid{pid}"
    # Keep consistent with node_mp worker naming in output_runtime.
    try:
        raw = os.environ.get(str(EnvKeys.TASK_LOCAL_WORKER_ID), None)
        if raw is not None and str(raw).strip() != "":
            base = f"{base}_w{int(str(raw).strip())}"
    except Exception:
        pass
    return str(base)


def _resolve_log_dir(cfg: RankLoggingConfig) -> str:
    # Highest priority: explicit log dir override.
    kmc_log_dir = str(env_str(EnvKeys.KMC_LOG_DIR, "") or "").strip()

    # Only route logs into rank-detail when output_level>=2.
    out_dir = str(env_str(EnvKeys.TASK_RUN_OUTPUT_DIR, "") or "").strip()
    try:
        output_level = int(getattr(cfg, "output_level", 1) or 1)
    except Exception:
        output_level = 1

    if kmc_log_dir:
        base = kmc_log_dir
    elif output_level >= 2 and out_dir:
        rank_dir_name = (
            str(cfg.rank_dir_name).strip()
            if cfg.rank_dir_name is not None and str(cfg.rank_dir_name).strip() != ""
            else _default_rank_dir_name()
        )
        base = os.path.join(out_dir, "rank-detail", str(rank_dir_name))
    else:
        base = str(cfg.log_dir or "logs")

    base = os.path.abspath(base)
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = os.getcwd()
    return base


def _handler_already_points_to(root: logging.Logger, path: str) -> bool:
    for h in list(getattr(root, "handlers", []) or []):
        try:
            if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(path):
                return True
        except Exception:
            continue
    return False


def _remove_marked_console_handlers(root: logging.Logger) -> None:
    for h in list(getattr(root, "handlers", []) or []):
        try:
            if getattr(h, "_kmc_console", False) is True:
                root.removeHandler(h)
        except Exception:
            continue


def setup_rank_file_logging(cfg: RankLoggingConfig) -> str:
    """Attach a per-rank FileHandler to the *root* logger.

    Returns the absolute log file path.

    Notes:
    - Idempotent: won't add duplicate FileHandlers for the same path.
    - Safe to call multiple times (e.g. before and after MPI init).
    """

    root = logging.getLogger()
    if not root.handlers:
        # Ensure something exists so libraries using logging don't drop messages.
        # We'll add our own handlers below.
        root.addHandler(logging.NullHandler())

    # Resolve level.
    try:
        level = int(cfg.level)
    except Exception:
        level = logging.INFO
    root.setLevel(level)

    host = socket.gethostname()
    pid = os.getpid()

    log_dir = _resolve_log_dir(cfg)

    run_id = (str(cfg.run_id).strip() if cfg.run_id is not None else "")

    # In this project’s current design, each node runs a single MPI rank.
    # Use hostname (and pid) to distinguish log files; do not require rank/world_size.
    if run_id:
        filename = f"{cfg.filename_prefix}.{run_id}.host{host}.pid{pid}.log"
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{cfg.filename_prefix}.host{host}.pid{pid}.{ts}.log"
    log_path = os.path.join(log_dir, filename)

    # Only enable per-rank file logs when output_level>=2, unless KMC_LOG_DIR is explicitly set.
    enable_file = False
    try:
        enable_file = bool(int(getattr(cfg, "output_level", 1) or 1) >= 2)
    except Exception:
        enable_file = False
    if not enable_file:
        try:
            enable_file = bool(str(env_str(EnvKeys.KMC_LOG_DIR, "") or "").strip() != "")
        except Exception:
            enable_file = False

    if enable_file and (not _handler_already_points_to(root, log_path)):
        try:
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)s [rank %(rank)s] %(process)d %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            setattr(fh, "_kmc_rank_file", True)
            root.addHandler(fh)
        except Exception:
            # If file logging fails, fall back silently.
            return os.path.abspath(log_path)

    # Console handler policy.
    if cfg.console:
        want_console = True
        if cfg.console_only_rank0 and int(cfg.rank) != 0:
            want_console = False

        if not want_console:
            _remove_marked_console_handlers(root)
        else:
            # Ensure at most one *marked* StreamHandler is present.
            has_marked_stream = any(
                isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
                and getattr(h, "_kmc_console", False) is True
                for h in list(root.handlers)
            )
            if not has_marked_stream:
                sh = logging.StreamHandler(stream=sys.stdout)
                sh.setLevel(level)
                sh.setFormatter(
                    logging.Formatter(
                        fmt=f"%(levelname)s [{host} pid{pid}] %(filename)s:%(lineno)d: %(message)s"
                    )
                )
                setattr(sh, "_kmc_console", True)
                root.addHandler(sh)

    # Inject rank into LogRecord (so formatter can show it).
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        record = old_factory(*args, **kwargs)
        try:
            setattr(record, "rank", int(cfg.rank))
        except Exception:
            pass
        try:
            setattr(record, "host", str(host))
        except Exception:
            pass
        return record

    logging.setLogRecordFactory(record_factory)

    return os.path.abspath(log_path)
