
import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from RL4KMC.utils.separated_buffer import SeparatedReplayBuffer
from RL4KMC.utils.util import update_linear_schedule
from RL4KMC.envs.kmc_env import KMCEnvWrap

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        print("Runner(object) 20")

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        print("Runner(object) 25")

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        # self.n_rollout_threads = self.all_args.n_rollout_threads
        # self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = False
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.num_agents = self.all_args.lattice_v_nums

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        print("Runner(object) 47")
        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        from RL4KMC.PPO.r_mappo import R_MAPPO as TrainAlgo
        from RL4KMC.PPO.rMAPPOPolicy import R_MAPPOPolicy as Policy

        # print("share_observation_space: ", self.envs.share_observation_space)
        # print("observation_space: ", self.envs.observation_space)
        # print("action_space: ", self.envs.action_space)
        print("Runner(object) 78")

        self.policy = Policy(self.all_args,
                            self.envs.action_space,
                            device = self.device)

        # if self.model_dir is not None:
            # self.restore()
        print("Runner(object) 86")

        if self.model_dir is not None:
            if self.use_eval:
                self.restore4eval()
            else:   
                self.restore()
        print("Runner(object) 93")

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        self.buffer = SeparatedReplayBuffer(self.all_args, self.envs.action_space)
        print("Runner(object) 96")

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_value = self.trainer.policy.get_values(self.buffer.share_obs[-1])
        next_value = _t2n(next_value)
        self.buffer.compute_returns(next_value, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_info

    def save(self):
        save_dir = "models/"
        print("save model to: ", save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(save_dir) + "/actor_agent" + ".pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        if self.trainer._use_valuenorm:
            policy_vnrom = self.trainer.value_normalizer
            torch.save(policy_vnrom.state_dict(), str(save_dir) + "/vnrom_agent" + ".pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + '.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + '.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)
        if self.trainer._use_valuenorm:
            policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent'+ '.pt')
            self.trainer.value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def restore4eval(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + '.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for k, v in train_infos[0].items():
            self.writter.add_scalars("reward", {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
