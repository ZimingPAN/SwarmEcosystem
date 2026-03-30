from easydict import EasyDict

collector_env_num = 2
evaluator_env_num = 2
n_episode = 2
num_simulations = 32
update_per_collect = 50
batch_size = 64
max_env_step = int(1e5)

max_vacancies = 32
max_defects = 384
max_shells = 16
neighbor_order = "2NN"
node_feat_dim = 4
stats_dim = 10
observation_shape = max_vacancies * max_defects * node_feat_dim + max_vacancies * max_defects + stats_dim
action_space_size = max_vacancies * 8

kmc_gumbel_muzero_config = dict(
    exp_name="data_muzero/kmc_gumbel_muzero_graph_seed0",
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        lattice_size=(40, 40, 40),
        max_episode_steps=200,
        max_vacancies=max_vacancies,
        max_defects=max_defects,
        max_shells=max_shells,
        node_feat_dim=node_feat_dim,
        stats_dim=stats_dim,
        temperature=300.0,
        reward_scale=1.0,
        cu_density=0.05,
        v_density=0.0002,
        lattice_cu_nums=0,
        lattice_v_nums=0,
        rlkmc_topk=16,
        neighbor_order=neighbor_order,
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            model_type="kmc_graph",
            latent_state_dim=256,
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=node_feat_dim,
            stats_dim=stats_dim,
            graph_hidden_size=128,
            per_vacancy_latent_dim=16,
            lattice_size=(40, 40, 40),
            neighbor_order=neighbor_order,
            self_supervised_learning_loss=False,
            discrete_action_encoding_type="one_hot",
            norm_type="BN",
        ),
        model_path=None,
        cuda=True,
        env_type="not_board_games",
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type="Adam",
        max_num_considered_actions=16,
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.001,
        ssl_loss_weight=0,
        num_simulations=num_simulations,
        reanalyze_ratio=0,
        n_episode=n_episode,
        eval_freq=500,
        replay_buffer_size=int(5e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        use_physics_rate_search=True,
        physics_rate_search_use_completed_q=True,
    ),
)

kmc_gumbel_muzero_config = EasyDict(kmc_gumbel_muzero_config)
main_config = kmc_gumbel_muzero_config

kmc_gumbel_muzero_create_config = dict(
    env=dict(
        type="kmc_lightzero",
        import_names=["zoo.kmc.envs.kmc_lightzero_env"],
    ),
    env_manager=dict(type="base"),
    policy=dict(
        type="gumbel_muzero",
        import_names=["lzero.policy.gumbel_muzero"],
    ),
)
kmc_gumbel_muzero_create_config = EasyDict(kmc_gumbel_muzero_create_config)
create_config = kmc_gumbel_muzero_create_config


if __name__ == "__main__":
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
