import numpy as np
import torch
import torch.nn as nn
from RL4KMC.utils.util import get_gard_norm, huber_loss, mse_loss
from RL4KMC.utils.valuenorm import ValueNorm
from RL4KMC.PPO.utils.util import check

class R_MAPPO():
    def __init__(self, args, policy, device=torch.device("cuda")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        value_loss = value_loss.mean()
        return value_loss

    def ppo_update(self, sample, update_actor=True):
        if len(sample) == 7:
            sys_obs_batch, obs_batch,  actions_batch, \
            value_preds_batch, return_batch, old_action_log_probs_batch, \
            adv_targ = sample
        else:
            sys_obs_batch, obs_batch,  actions_batch, \
            value_preds_batch, return_batch, old_action_log_probs_batch, \
            adv_targ, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=old_action_log_probs_batch.device)
        old_action_log_probs_batch = torch.gather(old_action_log_probs_batch, dim=1, index=actions_batch)
        # print("old_action_log_probs_batch.shape: ", old_action_log_probs_batch.shape)
        # print("actions_batch.shape: ", actions_batch.shape)
        # print("sys_obs.shape: ", sys_obs_batch.shape)
        # print("obs.shape: ", obs_batch.shape)
        values = []
        action_log_probs = []
        dist_entropies = []
        for sys_obs, obs, actions in zip(sys_obs_batch, obs_batch, actions_batch):
            value, action_log_prob, dist_entropy = self.policy.evaluate_actions(sys_obs, obs, actions)
            values.append(value)
            action_log_probs.append(action_log_prob)
            dist_entropies.append(dist_entropy)
        values = torch.stack(values)
        action_log_probs = torch.stack(action_log_probs)
        dist_entropies = torch.stack(dist_entropies)
        # print("values.shape: ", values.shape)
        # print("action_log_probs.shape: ", action_log_probs.shape)
        # print("dist_entropies.shape: ", dist_entropies.shape)
        
        # print("actions_log_probs.shape: ", action_log_probs.shape)
        # print("old_action_log_probs_batch: ", old_action_log_probs_batch)
        # print("actions_log_probs: ", action_log_probs)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # print("imp_weights: ", imp_weights)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info['mean_advantages'] = mean_advantages
        train_info['std_advantages'] = std_advantages
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
