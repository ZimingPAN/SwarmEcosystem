import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("MPI4PY_RC_FINALIZE", "0")

from RL4KMC.parser.parser import get_config


def main():
    from RL4KMC.runner.entry import eval_ssa

    cfg_parser = get_config()
    all_args = cfg_parser.parse_args()
    try:
        eval_ssa(all_args)
    except Exception as exc:
        raise exc
        

if __name__ == "__main__":
    main()