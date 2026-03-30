#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket
import logging

# third-party packages
import numpy as np
import torch
try:
    import setproctitle
except ImportError:  # pragma: no cover - optional runtime dependency
    setproctitle = None

from RL4KMC.envs.kmc_env import KMCEnvWrap as KMCEnv
from RL4KMC.parser.parser import get_config

# code repository sub-packages
# from onpolicy.config import get_config
# from onpolicy.envs.football.Football_Env import FootballEnv
# from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    env = KMCEnv(all_args)
    env.seed(all_args.seed)
    return env

def make_eval_env(all_args):
    env = KMCEnv(all_args)
    env.seed(all_args.seed * 50000)
    return env


def setup_logging(level: str | None = None):
    level_name = (
        level
        or os.environ.get("KMC_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL")
        or "INFO"
    )
    lvl = getattr(logging, str(level_name).upper(), logging.INFO)

    # Per-process log file (distinguish by hostname/pid; no need to guess rank/world).
    try:
        from RL4KMC.utils.rank_logging import RankLoggingConfig, setup_rank_file_logging

        host = socket.gethostname()
        pid = os.getpid()
        run_id = f"train_{host}_{pid}"

        setup_rank_file_logging(
            RankLoggingConfig(
                log_dir=str(os.environ.get("KMC_LOG_DIR", "logs") or "logs"),
                run_id=str(run_id),
                rank=0,
                world_size=1,
                level=int(lvl),
                console=True,
                console_only_rank0=False,
                filename_prefix="train",
            )
        )
    except Exception:
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        fmt = f"%(levelname)s [rank {rank}] %(filename)s:%(lineno)d: %(message)s"
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=lvl, format=fmt, stream=sys.stdout)
        else:
            root.setLevel(lvl)


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]

    setup_logging("DEBUG" if bool(getattr(all_args, "debug", False)) else None)
    logger = logging.getLogger(__name__)

    if all_args.algorithm_name == "rmappo":
        logger.info(
            "Choosing rmappo: set use_recurrent_policy=True use_naive_recurrent_policy=False"
        )
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        logger.info(
            "Choosing mappo: set use_recurrent_policy=False use_naive_recurrent_policy=False"
        )
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        logger.info(
            "Choosing ippo: set use_centralized_V=False (note: GRF fully observed => ippo is rmappo)"
        )
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    if all_args.cuda and torch.cuda.is_available():
        logger.info("choose to use gpu...")
        device = torch.device("cuda:0")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logger.info("choose to use cpu...")
        device = torch.device("cpu")

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # setproctitle.setproctitle("-".join([
    #     all_args.env_name, 
    #     all_args.scenario_name, 
    #     all_args.algorithm_name, 
    #     all_args.experiment_name
    # ]) + "@" + all_args.user_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    if hasattr(envs, "env"):
        all_args.lattice_v_nums = int(getattr(envs.env, "V_nums", getattr(all_args, "lattice_v_nums", 0)))
        all_args.lattice_cu_nums = int(getattr(envs.env, "Cu_nums", getattr(all_args, "lattice_cu_nums", 0)))
    if eval_envs is not None and hasattr(eval_envs, "env"):
        eval_envs.env.args.lattice_v_nums = int(all_args.lattice_v_nums)
        eval_envs.env.args.lattice_cu_nums = int(all_args.lattice_cu_nums)


    from RL4KMC.runner.kmc_runner import KMCRunner as Runner

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(all_args, config)
    runner.run()
    

if __name__ == "__main__":
    main(sys.argv[1:])
