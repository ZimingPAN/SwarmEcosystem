"""
Standalone DreamerV4+GNN training for KMC task.
Online RL with world model imagination.
No DI-engine dependency.
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- path setup ----
ROOT = Path(__file__).resolve().parents[0]
RLKMC = ROOT.parent / "RLKMC-MASSIVE-main"
LIGHTZERO = ROOT.parent / "LightZero-main"
for p in [str(ROOT), str(LIGHTZERO), str(RLKMC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from RL4KMC.envs.kmc import KMCEnv
from RL4KMC.world_models import (
    DefectGraphObservationShape,
    build_defect_graph_observation,
    build_kmc_action_mask,
    unflatten_defect_graph_observation,
)
from dreamer4.kmc import KMCGraphEncoder


# ============================================================
# Data
# ============================================================
@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    action_mask: np.ndarray
    done: bool
    delta_t: float
    delta_E: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, n: int) -> List[Transition]:
        return random.choices(list(self.buf), k=n)

    def __len__(self):
        return len(self.buf)


# ============================================================
# Environment wrapper
# ============================================================
class KMCEnvWrapper:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.shape = DefectGraphObservationShape(
            max_vacancies=cfg["max_vacancies"],
            max_defects=cfg["max_defects"],
            max_shells=cfg["max_shells"],
            node_feat_dim=4,
            stats_dim=cfg.get("stats_dim", 10),
        )
        self.env: Optional[KMCEnv] = None
        self.timestep = 0
        self.max_steps = cfg["max_episode_steps"]

    def _build_args(self):
        from RL4KMC.parser.parser import get_config
        parser = get_config()
        args = parser.parse_known_args([])[0]
        total = int(np.prod(self.cfg["lattice_size"]) * 2)
        args.lattice_size = list(self.cfg["lattice_size"])
        args.temperature = self.cfg.get("temperature", 300.0)
        args.reward_scale = self.cfg.get("reward_scale", 1.0)
        args.topk = self.cfg.get("rlkmc_topk", 16)
        args.device = "cpu"
        args.cu_density = self.cfg["cu_density"]
        args.v_density = self.cfg["v_density"]
        args.lattice_cu_nums = int(round(self.cfg["cu_density"] * total))
        args.lattice_v_nums = max(int(round(self.cfg["v_density"] * total)), 1)
        args.compute_global_static_env_reset = True
        args.skip_stats = True
        args.skip_global_diffusion_reset = False
        args.max_ssa_rounds = self.max_steps
        args.neighbor_order = self.cfg.get("neighbor_order", "2NN")
        return args

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        if self.env is None:
            self.env = KMCEnv(self._build_args())
        self.env.reset()
        self.timestep = 0
        return self._obs(), self._mask()

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        n_vac = self.env.V_nums
        vac_idx = action // 8
        if vac_idx >= n_vac:
            vac_idx = vac_idx % max(n_vac, 1)
            action = vac_idx * 8 + (action % 8)

        self.env._ensure_diffusion_rates()
        flat_rates = [r for vr in self.env.diffusion_rates for r in vr if r > 0]
        total_rate = float(np.sum(flat_rates)) if flat_rates else 0.0
        self.env.step_fast(int(action), self.timestep)

        delta_t = -np.log(np.random.rand()) / total_rate if total_rate > 0 else 0.0
        self.env.time += delta_t
        self.env.time_history.append(self.env.time)
        energy_after = self.env.calculate_system_energy()
        delta_E = self.env.energy_last - energy_after
        reward = float(delta_E * self.env.args.reward_scale)
        self.env.energy_last = energy_after
        self.env.energy_history.append(energy_after)
        self.timestep += 1
        done = self.timestep >= self.max_steps
        return self._obs(), self._mask(), reward, done, {"delta_t": delta_t, "delta_E": delta_E}

    def _obs(self) -> np.ndarray:
        share_obs = np.zeros(self.shape.stats_dim, dtype=np.float32)
        # Feature 3: inject physics parameters into stats vector
        share_obs[0] = self.cfg.get("temperature", 300.0) / 1000.0  # normalized
        share_obs[1] = self.cfg.get("cu_density", 0.05)
        share_obs[2] = self.cfg.get("v_density", 0.0002)
        return build_defect_graph_observation(self.env, shape=self.shape, share_obs=share_obs).astype(np.float32)

    def _mask(self) -> np.ndarray:
        return build_kmc_action_mask(self.env, max_vacancies=self.shape.max_vacancies)


# ============================================================
# Dreamer Actor-Critic model wrapping KMCDynamicsWorldModel
# ============================================================
class DreamerKMCAgent(nn.Module):
    def __init__(self, *, dim_latent, max_vacancies, max_defects, max_shells,
                 stats_dim, lattice_size, neighbor_order, action_space_size,
                 graph_hidden_size=32,
                 use_topology_head: bool = False,
                 use_shortcut_forcing: bool = False):
        super().__init__()
        self.max_vacancies = max_vacancies
        self.dim_latent = dim_latent
        self.action_space_size = action_space_size
        self.use_topology_head = use_topology_head
        self.use_shortcut_forcing = use_shortcut_forcing
        latent_flat = max_vacancies * dim_latent

        # GNN encoder (directly, no heavy DynamicsWorldModel parent)
        self.graph_encoder = KMCGraphEncoder(
            max_vacancies=max_vacancies, max_defects=max_defects,
            max_shells=max_shells, node_feat_dim=4, stats_dim=stats_dim,
            hidden_size=graph_hidden_size, dim_latent=dim_latent,
            lattice_size=lattice_size, neighbor_order=neighbor_order,
        )

        # Dynamics transition network
        if use_shortcut_forcing:
            # Feature 5: shortcut forcing — add horizon embedding
            self.max_horizon = 8
            self.horizon_embed = nn.Embedding(self.max_horizon, 32)
            self.dynamics_net = nn.Sequential(
                nn.Linear(latent_flat + action_space_size + 32, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, latent_flat),
            )
        else:
            self.max_horizon = 1
            self.dynamics_net = nn.Sequential(
                nn.Linear(latent_flat + action_space_size, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, latent_flat),
            )

        # Actor (policy head)
        self.actor = nn.Sequential(
            nn.LayerNorm(latent_flat),
            nn.Linear(latent_flat, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_space_size),
        )

        # Critic (value head)
        self.critic = nn.Sequential(
            nn.LayerNorm(latent_flat),
            nn.Linear(latent_flat, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

        # Reward head — conditioned on action for accurate world model prediction
        self.reward_head = nn.Sequential(
            nn.LayerNorm(latent_flat + action_space_size),
            nn.Linear(latent_flat + action_space_size, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Physics time head — state-only (Δt depends on Γ_tot of current config,
        # NOT on which action is taken — this is a KMC Poisson process property)
        self.time_head = nn.Sequential(
            nn.LayerNorm(latent_flat),
            nn.Linear(latent_flat, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Initialize time_head output bias to log(~3e-5) so predictions
        # start at the correct physical Δt scale (Poisson process)
        with torch.no_grad():
            nn.init.constant_(self.time_head[-1].bias, -10.0)  # exp(-10) ≈ 4.5e-5

        # Energy prediction head — conditioned on action
        self.energy_head = nn.Sequential(
            nn.LayerNorm(latent_flat + action_space_size),
            nn.Linear(latent_flat + action_space_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Feature 4: topology reconstruction head
        if use_topology_head:
            node_feat_dim = 4  # offset_x, offset_y, offset_z, type
            self.max_defects = max_defects
            self.node_feat_dim = node_feat_dim
            self.topology_head = nn.Sequential(
                nn.LayerNorm(latent_flat),
                nn.Linear(latent_flat, 256),
                nn.SiLU(),
                nn.Linear(256, max_defects * (node_feat_dim + 1)),
            )

    def encode(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Encode flat observation into latent state."""
        latent = self.graph_encoder(obs_flat)  # (B, V, D)
        return latent.reshape(latent.shape[0], -1)  # (B, V*D)

    def dynamics(self, latent_flat: torch.Tensor, action: torch.Tensor,
                 horizon: int = 1) -> torch.Tensor:
        """Predict next latent state using learned transition."""
        action_onehot = F.one_hot(action.long(), self.action_space_size).float()
        if self.use_shortcut_forcing:
            horizon_t = torch.clamp(
                torch.tensor([horizon - 1], device=latent_flat.device), 0, self.max_horizon - 1
            )
            horizon_emb = self.horizon_embed(horizon_t).expand(latent_flat.shape[0], -1)
            combined = torch.cat([latent_flat, action_onehot, horizon_emb], dim=-1)
        else:
            combined = torch.cat([latent_flat, action_onehot], dim=-1)
        delta = self.dynamics_net(combined)
        return latent_flat + 0.1 * delta  # residual connection

    def forward_policy(self, latent_flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.actor(latent_flat)
        logits[~mask] = -1e9
        return logits

    def forward_value(self, latent_flat: torch.Tensor) -> torch.Tensor:
        return self.critic(latent_flat).squeeze(-1)

    def forward_reward(self, latent_flat: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        if action is not None:
            action_onehot = F.one_hot(action.long(), self.action_space_size).float()
            inp = torch.cat([latent_flat, action_onehot], dim=-1)
        else:
            inp = torch.cat([latent_flat, torch.zeros(latent_flat.shape[0], self.action_space_size, device=latent_flat.device)], dim=-1)
        return self.reward_head(inp).squeeze(-1)

    def forward_time(self, latent_flat: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """Return predicted Δt (exponentiated from log-space).
        Δt is a state property (Poisson process), action arg is ignored for compatibility."""
        return torch.exp(self.time_head(latent_flat).squeeze(-1))

    def forward_log_time(self, latent_flat: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """Return raw log-space prediction for training loss.
        Δt is a state property (Poisson process), action arg is ignored for compatibility."""
        return self.time_head(latent_flat).squeeze(-1)

    def reconstruct_topology(self, latent_flat: torch.Tensor):
        """Feature 4: Reconstruct node attributes and mask for self-supervised learning."""
        topo = self.topology_head(latent_flat)
        B = topo.shape[0]
        topo = topo.view(B, self.max_defects, self.node_feat_dim + 1)
        node_attr_pred = topo[..., :self.node_feat_dim]
        node_mask_logits = topo[..., self.node_feat_dim]
        return node_attr_pred, node_mask_logits


# ============================================================
# Training
# ============================================================
def collect_episode(env: KMCEnvWrapper, agent: DreamerKMCAgent,
                    buffer: ReplayBuffer, device: str,
                    epsilon: float = 0.1) -> tuple[float, int]:
    obs, mask = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
            latent = agent.encode(obs_t)
            logits = agent.forward_policy(latent, mask_t)
            probs = F.softmax(logits[0], dim=-1)

        if random.random() < epsilon:
            valid = np.flatnonzero(mask)
            action = int(np.random.choice(valid)) if len(valid) > 0 else 0
        else:
            action = int(torch.multinomial(probs, 1).item())

        next_obs, next_mask, reward, done, info = env.step(action)

        buffer.push(Transition(
            obs=obs, action=action, reward=reward,
            next_obs=next_obs, action_mask=mask, done=done,
            delta_t=info["delta_t"], delta_E=info["delta_E"],
        ))

        obs, mask = next_obs, next_mask
        total_reward += reward
        steps += 1

    return total_reward, steps


def train_step(agent: DreamerKMCAgent, optimizer: torch.optim.Optimizer,
               batch: List[Transition], device: str, discount: float,
               use_physics_discount: bool = False, time_scale_tau: float = 1.0) -> dict:
    obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device)
    masks = torch.tensor(np.stack([t.action_mask for t in batch]), dtype=torch.bool, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
    delta_ts = torch.tensor([t.delta_t for t in batch], dtype=torch.float32, device=device)
    delta_Es = torch.tensor([t.delta_E for t in batch], dtype=torch.float32, device=device)

    B = len(batch)

    # Encode
    latent = agent.encode(obs)
    next_latent = agent.encode(next_obs)

    # Policy loss (actor-critic)
    logits = agent.forward_policy(latent, masks)
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_v = agent.forward_value(next_latent.detach())
        # Physics-time discount: γ = exp(-Δt/τ) per step
        if use_physics_discount:
            physics_gamma = torch.exp(-delta_ts / time_scale_tau).clamp(0.01, 1.0)
        else:
            physics_gamma = torch.full_like(rewards, discount)
        target_v = rewards + physics_gamma * (1 - dones) * next_v
        value = agent.forward_value(latent.detach())
        advantage = target_v - value
        # Normalize advantages to prevent policy gradient explosion
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # Clipped policy loss (PPO-style stability)
    raw_pg = -(action_log_probs * advantage.detach())
    clipped_pg = raw_pg.clamp(-5.0, 5.0)  # prevent extreme gradients
    policy_loss = clipped_pg.mean()
    entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
    policy_loss = policy_loss - 0.003 * entropy

    # Value loss
    value_pred = agent.forward_value(latent)
    value_loss = F.mse_loss(value_pred, target_v.detach())

    # Reward prediction loss — conditioned on action
    reward_pred = agent.forward_reward(latent, actions)
    reward_loss = F.mse_loss(reward_pred, rewards)

    # Time prediction loss — train in log-space, detach latent so time head
    # gets strong gradients without interfering with policy/reward backbone
    log_time_pred = agent.forward_log_time(latent.detach())
    log_time_target = torch.log(delta_ts.clamp(min=1e-10))
    time_loss = F.mse_loss(log_time_pred, log_time_target)

    # Energy prediction loss (physics head) — conditioned on action
    action_onehot = F.one_hot(actions.long(), agent.action_space_size).float()
    energy_input = torch.cat([latent, action_onehot], dim=-1)
    energy_pred = agent.energy_head(energy_input).squeeze(-1)
    energy_loss = F.mse_loss(energy_pred, delta_Es)

    # Total loss — scale policy loss down to prevent it from dominating world model learning
    loss = 0.1 * policy_loss + 0.5 * value_loss + 1.0 * reward_loss + 1.0 * time_loss + 0.2 * energy_loss

    result = {
        "loss": 0.0,
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "reward_loss": reward_loss.item(),
        "time_loss": time_loss.item(),
        "energy_loss": energy_loss.item(),
        "entropy": entropy.item(),
    }

    # Feature 4: topology reconstruction loss
    topology_loss_val = 0.0
    if agent.use_topology_head:
        shape = agent.graph_encoder.shape
        node_attr_gt, node_mask_gt, _ = unflatten_defect_graph_observation(
            obs[:, :shape.flat_dim] if obs.shape[-1] > shape.flat_dim else obs, shape=shape
        )
        node_attr_gt = node_attr_gt.to(device)
        node_mask_gt = node_mask_gt.to(device).float()
        node_attr_pred, node_mask_logits = agent.reconstruct_topology(latent)
        # Average over vacancies for ground truth
        node_attr_gt_avg = node_attr_gt.reshape(B, shape.max_vacancies, shape.max_defects, shape.node_feat_dim).mean(dim=1)
        node_mask_gt_avg = node_mask_gt.reshape(B, shape.max_vacancies, shape.max_defects).mean(dim=1)
        topology_loss = F.mse_loss(node_attr_pred, node_attr_gt_avg) + F.binary_cross_entropy_with_logits(node_mask_logits, node_mask_gt_avg)
        loss = loss + 0.1 * topology_loss
        topology_loss_val = topology_loss.item()
    result["topology_loss"] = topology_loss_val

    # Feature 5: shortcut forcing consistency loss
    shortcut_loss_val = 0.0
    if agent.use_shortcut_forcing and random.random() < 0.2 and len(batch) > 2:
        next_latent_chain = agent.dynamics(latent.detach(), actions, horizon=1)
        next_next_latent_chain = agent.dynamics(next_latent_chain.detach(), actions, horizon=1)
        next_next_latent_direct = agent.dynamics(latent.detach(), actions, horizon=2)
        shortcut_loss = F.mse_loss(next_next_latent_direct, next_next_latent_chain.detach())
        loss = loss + 0.05 * shortcut_loss
        shortcut_loss_val = shortcut_loss.item()
    result["shortcut_loss"] = shortcut_loss_val

    optimizer.zero_grad()
    # Two-pass backward: backbone and time_head are decoupled (latent.detach())
    # Compute backbone_loss explicitly to avoid shared computation graph issues
    backbone_loss = 0.1 * policy_loss + 0.5 * value_loss + 1.0 * reward_loss + 0.2 * energy_loss
    if agent.use_topology_head and topology_loss_val > 0:
        backbone_loss = backbone_loss + 0.1 * topology_loss
    if agent.use_shortcut_forcing and shortcut_loss_val > 0:
        backbone_loss = backbone_loss + 0.05 * shortcut_loss
    backbone_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
    time_loss.backward()
    optimizer.step()

    result["loss"] = loss.item()
    return result


def evaluate(env_cfg: dict, agent: DreamerKMCAgent, device: str,
             n_episodes: int = 5) -> dict:
    total_rewards = []
    total_positive = []

    for _ in range(n_episodes):
        eval_env = KMCEnvWrapper(env_cfg)
        obs, mask = eval_env.reset()
        ep_reward = 0.0
        ep_pos = 0
        done = False
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
                latent = agent.encode(obs_t)
                logits = agent.forward_policy(latent, mask_t)
                action = int(logits[0].argmax().item())
            obs, mask, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            if reward > 0:
                ep_pos += 1
        total_rewards.append(ep_reward)
        total_positive.append(ep_pos)

    return {
        "eval_mean_reward": np.mean(total_rewards),
        "eval_mean_positive_steps": np.mean(total_positive),
        "eval_rewards": total_rewards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--cu_density", type=float, default=0.05)
    parser.add_argument("--v_density", type=float, default=0.0002)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--dim_latent", type=int, default=16)

    # Training
    parser.add_argument("--collect_episodes", type=int, default=4)
    parser.add_argument("--train_steps_per_collect", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--total_iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_cu_density", type=float, default=None,
                        help="Cu density for eval (default: same as training)")
    parser.add_argument("--eval_v_density", type=float, default=None,
                        help="V density for eval (default: same as training)")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="dreamer_kmc_results")
    # Feature flags
    parser.add_argument("--use_topology_head", action="store_true", default=False,
                        help="Feature 4: topology reconstruction self-supervised loss")
    parser.add_argument("--use_shortcut_forcing", action="store_true", default=False,
                        help="Feature 5: variable horizon shortcut forcing")
    parser.add_argument("--use_physics_discount", action="store_true", default=False,
                        help="Feature 1: use physics-time discount γ=exp(-Δt/τ)")
    parser.add_argument("--time_scale_tau", type=float, default=1.0,
                        help="Time scale τ for physics discount γ=exp(-Δt/τ)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = {
        "lattice_size": tuple(args.lattice_size),
        "max_episode_steps": args.max_episode_steps,
        "max_vacancies": args.max_vacancies,
        "max_defects": args.max_defects,
        "max_shells": args.max_shells,
        "stats_dim": 10,
        "temperature": args.temperature,
        "reward_scale": args.reward_scale,
        "cu_density": args.cu_density,
        "v_density": args.v_density,
        "rlkmc_topk": 16,
        "neighbor_order": args.neighbor_order,
    }

    env = KMCEnvWrapper(env_cfg)

    # Build eval config (may differ from training config for low-density testing)
    eval_cfg = dict(env_cfg)
    if args.eval_cu_density is not None:
        eval_cfg["cu_density"] = args.eval_cu_density
    if args.eval_v_density is not None:
        eval_cfg["v_density"] = args.eval_v_density

    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    action_dim = mask.shape[0]
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Cu: {env.env.Cu_nums}, V: {env.env.V_nums}")

    agent = DreamerKMCAgent(
        dim_latent=args.dim_latent,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=10,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        action_space_size=action_dim,
        graph_hidden_size=32,
        use_topology_head=args.use_topology_head,
        use_shortcut_forcing=args.use_shortcut_forcing,
    ).to(args.device)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, weight_decay=1e-5)
    buffer = ReplayBuffer(args.buffer_size)

    log_file = save_dir / "training_log.txt"
    best_eval_reward = -float("inf")
    start_iteration = 1

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        print(f"Resumed from {args.resume}, starting at iteration {start_iteration}")

    print(f"\n{'='*60}")
    print(f"DreamerV4+GNN KMC Training")
    print(f"Cu: {args.cu_density}, V: {args.v_density}, Lattice: {args.lattice_size}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    for iteration in range(start_iteration, args.total_iterations + 1):
        t0 = time.time()
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * min(1.0, iteration / (args.total_iterations * 0.7))

        # Collect
        collect_rewards = []
        for _ in range(args.collect_episodes):
            ep_r, ep_s = collect_episode(env, agent, buffer, args.device, epsilon)
            collect_rewards.append(ep_r)

        # Train
        train_losses = []
        if len(buffer) >= args.batch_size:
            for _ in range(args.train_steps_per_collect):
                batch = buffer.sample(args.batch_size)
                losses = train_step(agent, optimizer, batch, args.device, args.discount,
                                    use_physics_discount=args.use_physics_discount,
                                    time_scale_tau=args.time_scale_tau)
                train_losses.append(losses)

        elapsed = time.time() - t0
        avg_loss = np.mean([l["loss"] for l in train_losses]) if train_losses else 0
        avg_rl = np.mean([l["reward_loss"] for l in train_losses]) if train_losses else 0
        avg_topo = np.mean([l["topology_loss"] for l in train_losses]) if train_losses else 0
        avg_shortcut = np.mean([l["shortcut_loss"] for l in train_losses]) if train_losses else 0

        log_msg = (
            f"[Iter {iteration:4d}/{args.total_iterations}] "
            f"collect_reward={np.mean(collect_rewards):+.4f} "
            f"loss={avg_loss:.4f} rew_loss={avg_rl:.4f} "
            f"topo={avg_topo:.4f} shortcut={avg_shortcut:.4f} "
            f"eps={epsilon:.3f} buf={len(buffer)} time={elapsed:.1f}s"
        )
        print(log_msg)

        # Evaluate
        if iteration % args.eval_freq == 0:
            eval_results = evaluate(eval_cfg, agent, args.device, args.eval_episodes)
            eval_msg = (
                f"  >>> EVAL: mean_reward={eval_results['eval_mean_reward']:+.4f} "
                f"positive_steps={eval_results['eval_mean_positive_steps']:.1f} "
                f"rewards={[f'{r:+.3f}' for r in eval_results['eval_rewards']]}"
            )
            print(eval_msg)

            with open(log_file, "a") as f:
                f.write(f"{log_msg}\n{eval_msg}\n")

            if eval_results["eval_mean_reward"] > best_eval_reward:
                best_eval_reward = eval_results["eval_mean_reward"]
                torch.save(agent.state_dict(), save_dir / "best_model.pt")
                print(f"  >>> New best model! reward={best_eval_reward:+.4f}")

        if iteration % args.save_freq == 0:
            torch.save({
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
            }, save_dir / f"checkpoint_{iteration}.pt")

    # Final evaluation
    print(f"\n{'='*60}")
    final_eval = evaluate(eval_cfg, agent, args.device, n_episodes=10)
    print(f"Final mean reward: {final_eval['eval_mean_reward']:+.4f}")
    print(f"Final rewards: {final_eval['eval_rewards']}")
    print(f"Best eval reward: {best_eval_reward:+.4f}")
    torch.save(agent.state_dict(), save_dir / "final_model.pt")


if __name__ == "__main__":
    main()
