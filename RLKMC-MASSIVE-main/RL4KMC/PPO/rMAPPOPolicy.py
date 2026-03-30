import torch
from RL4KMC.PPO.r_actor_critic import R_Actor, R_Critic
from RL4KMC.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    def __init__(self, args, act_space, device=torch.device("cuda")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.act_space = act_space

        self.actor = R_Actor(args, self.act_space, self.device)
        self.critic = R_Critic(args, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, sys_obs, obs, deterministic=False):
        actions, action_log_probs = self.actor(obs, deterministic=deterministic)
        values = self.critic(sys_obs)
        return values, actions, action_log_probs

    def get_values(self, sys_obs):
        values = self.critic(sys_obs)
        return values

    def evaluate_actions(self, sys_obs, obs, action):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action)
        values = self.critic(sys_obs)
        return values, action_log_probs, dist_entropy

    def act(self, obs, deterministic=False):
        actions, dist_entropy  = self.actor(obs, deterministic)
        return actions, dist_entropy
