import argparse
import os
import sys
from RL4KMC.utils.env import EnvKeys, env_int
from pydantic import BaseModel, Field


class KMCArgs(BaseModel):
    # WIP
    seed: int = Field(777, description="Random seed for numpy/torch")
    nodes: int = Field(1, description="Number of nodes for distributed training")
    base: int = Field(10000, description="lattice side length")
    numa_per_rank: int = Field(1, description="Number of NUMA nodes per rank ")
    cores_per_numa: int = Field(1, description="Number of CPU cores per NUMA node")
    workers_per_rank: int = Field(
        0,
        description="Number of worker processes per rank (0 means fallback to workers_per_numa)",
    )
    workers_per_numa: int = Field(
        1, description="DEPRECATED fallback when workers_per_rank is 0"
    )
    enable_incremental_policy: bool = Field(
        False, description="Whether to enable incremental policy"
    )


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="mappo",
        choices=["rmappo", "mappo", "happo", "hatrpo", "mat", "mat_dec"],
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.",
    )

    # env parameters
    parser.add_argument(
        "--env_name", type=str, default="KMCEnv", help="specify the name of environment"
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=200, help="Max length for any episode"
    )

    # parser.add_argument("--proc_grid_x", type=int, default=None)
    # parser.add_argument("--proc_grid_y", type=int, default=None)
    # parser.add_argument("--proc_grid_z", type=int, default=None)
    # parser.add_argument("--processor_dim_x", type=int, default=None)
    # parser.add_argument("--processor_dim_y", type=int, default=None)
    # parser.add_argument("--processor_dim_z", type=int, default=None)
    # parser.add_argument("--proc_grid_dim", type=int, nargs=3, default=None)
    # parser.add_argument("--processor_dim", type=int, nargs=3, default=None)
    # parser.add_argument("--sub_block_grid_dim", type=int, nargs=3, default=None)
    # parser.add_argument("--halo_depth", type=int, default=2)
    # parser.add_argument("--sub_block_grid_x", type=int, default=2)
    # parser.add_argument("--sub_block_grid_y", type=int, default=2)
    # parser.add_argument("--sub_block_grid_z", type=int, default=2)
    # parser.add_argument("--mode", choices=["strong", "weak"], default="strong")
    # parser.add_argument(
    #     "--gpus", type=int, default=env_int(EnvKeys.WORLD_SIZE, 1, min_value=1)
    # )
    parser.add_argument(
        "--nodes",
        type=int,
        default=env_int(EnvKeys.SLURM_JOB_NUM_NODES, 1, min_value=1),
    )
    parser.add_argument(
        "--base", type=int, default=env_int(EnvKeys.LATTICE_BASE, 10000, min_value=1)
    )

    # parser.add_argument(
    #     "--sub_block_grid",
    #     type=int,
    #     nargs=3,
    #     default=[
    #         env_int(EnvKeys.SUB_BLOCK_GRID_X, 2, min_value=1),
    #         env_int(EnvKeys.SUB_BLOCK_GRID_Y, 2, min_value=1),
    #         env_int(EnvKeys.SUB_BLOCK_GRID_Z, 2, min_value=1),
    #     ],
    # )
    parser.add_argument("--dist_backend", type=str, default=None)
    parser.add_argument(
        "--task_num_groups",
        type=int,
        default=env_int(EnvKeys.TASK_NUM_GROUPS, 1, min_value=1),
        help="DEPRECATED (ignored). Task grouping/TCPStore scheduling has been removed; use --task_scheduler_mode=mpi_rma or static.",
    )
    parser.add_argument(
        "--workers_per_rank",
        type=int,
        default=1,
        help="Number of worker processes per rank.",
    )
    parser.add_argument(
        "--cores_per_worker",
        type=int,
        default=1,
        help="Number of CPU cores to bind per worker process (default: 1). Note: this is not a hard limit, but will be used as a hint for affinity planning.",
    )

    parser.add_argument(
        "--pin_policy", type=str, default="spread", choices=["spread", "compact"]
    )
    parser.add_argument(
        "--enable_affinity_debug_log",
        action="store_true",
        default=False,
        help="Write detailed CPU topology + pin-plan debug info to core_bind.plan.log.",
    )
    parser.add_argument("--max_ssa_rounds", type=int, default=sys.maxsize)
    parser.add_argument("--enable_detail_log", action="store_true", default=False)
    parser.add_argument(
        "--enable_worker_debug_log",
        action="store_true",
        default=False,
        help="Write worker (node_mp) debug logs to a per-worker file.",
    )
    # parser.add_argument("--enable_comm_log", action="store_true", default=True)
    # parser.add_argument(
    #     "--disable_comm_log", action="store_false", dest="enable_comm_log"
    # )
    # parser.add_argument(
    #     "--enable_global_status_log", action="store_true", default=False
    # )
    # parser.add_argument("--enable_init_log", action="store_true", default=True)
    # parser.add_argument(
    #     "--disable_init_log", action="store_false", dest="enable_init_log"
    # )
    # parser.add_argument("--enable_end_step_compute", action="store_true", default=False)
    parser.add_argument(
        "--enable_incremental_policy", action="store_true", default=False
    )
    parser.add_argument("--self_generate_local", action="store_true", default=False)
    # parser.add_argument(
    #     "--same_init_positions_across_ranks", action="store_true", default=False
    # )

    # network parameters
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_ReLU", action="store_false", default=True, help="Whether to use ReLU"
    )
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain", type=float, default=0.01, help="The gain # of last action layer"
    )

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--recurrent_N", type=int, default=1, help="The number of recurrent layers."
    )
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-4,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=0.01,
        help="the threshold of kl-divergence (default: 0.01)",
    )
    parser.add_argument(
        "--ls_step", type=int, default=10, help="number of line search (default: 10)"
    )
    parser.add_argument(
        "--accept_ratio",
        type=float,
        default=0.5,
        help="accept ratio of loss improve (default: 0.5)",
    )

    # ppo parameters
    parser.add_argument(
        "--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)"
    )
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=10.0, help=" coefficience of huber loss."
    )

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )
    parser.add_argument(
        "--only_eval",
        action="store_true",
        default=False,
        help="run only evaluation without training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="enable verbose debug logging",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    # add for transformer
    parser.add_argument("--encode_state", action="store_true", default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action="store_true", default=False)
    parser.add_argument("--share_actor", action="store_true", default=False)

    # add for online multi-task
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="academy_3_vs_1_with_keeper",
        help="which scenario to run on.",
    )

    parser.add_argument(
        "--gnn_attention_heads",
        type=int,
        default=4,
        help="Number of attention heads in GNN.",
    )
    parser.add_argument(
        "--gnn_attention_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for attention in GNN.",
    )
    parser.add_argument(
        "--gnn_layers", type=int, default=3, help="Number of GNN layers."
    )

    parser.add_argument(
        "--temperature", type=float, default=300.0, help="Temperature for KMC."
    )

    parser.add_argument(
        "--lattice_size",
        type=int,  # 关键：将每个字符串参数转换为整数
        nargs=3,  # 关键：告诉argparse接收三个参数
        default=[20, 20, 20],
    )
    # parser.add_argument("--lattice_size", type=tuple, default=(10, 10, 10),
    #                     help="Size of the lattice.")
    parser.add_argument(
        "--neighbor_radius", type=float, default=5.0, help="Neighbor radius for GNN."
    )
    parser.add_argument(
        "--max_num_neighbors",
        type=int,
        default=10,
        help="Max number of neighbors for GNN.",
    )
    parser.add_argument(
        "--lattice_cu_nums",
        type=int,
        default=0,
        help="Number of Cu atoms in the lattice.",
    )
    parser.add_argument(
        "--lattice_v_nums", type=int, default=0, help="Number of V in the lattice."
    )
    parser.add_argument(
        "--cu_density", type=float, default=None, help="Cu density override"
    )
    parser.add_argument(
        "--v_density", type=float, default=None, help="Vacancy density override"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )

    parser.add_argument(
        "--layer_N",
        type=int,
        default=10,
        help="Number of layers for embed/actor/critic networks",
    )
    parser.add_argument(
        "--reward_scale", type=float, default=1.0, help="Scale factor for rewards."
    )

    parser.add_argument("--topk", type=int, default=16, help="TopK for local obs.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="embedding",
        choices=["embedding", "embedding2"],
        help="Select embedding model implementation",
    )
    parser.add_argument(
        "--use_traditional_kmc",
        action="store_true",
        default=False,
        help="use traditional KMC event selection by diffusion rates",
    )
    parser.add_argument(
        "--prevent_backjump",
        action="store_true",
        default=True,
        help="Prevent immediate backjump for each vacancy agent.",
    )
    parser.add_argument(
        "--n_radial", type=int, default=1, help="Number of radial directions for RPV."
    )
    parser.add_argument(
        "--n_axial", type=int, default=1, help="Number of axial directions for RPV."
    )
    parser.add_argument(
        "--rescaled_sim_time",
        type=float,
        default=None,
        help="Rescaled simulation time.",
    )

    parser.add_argument(
        "--bench_step",
        type=int,
        default=-1,
        help="If >0, enable benchmark mode with given step interval.",
    )
    parser.add_argument("--output_level", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory to save outputs."
    )
    return parser
