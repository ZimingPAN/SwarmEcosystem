"""
Standalone Gumbel MuZero training for KMC task.
No DI-engine dependency — uses KMC env + GNN MuZero model directly.
Implements: online MCTS collect → replay buffer → train loop.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- path setup ----
ROOT = Path(__file__).resolve().parents[2]
RLKMC = ROOT.parent / "RLKMC-MASSIVE-main"
for p in [str(ROOT), str(RLKMC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from RL4KMC.envs.kmc import KMCEnv
from RL4KMC.world_models import (
    DefectGraphObservationShape,
    build_defect_graph_observation,
    build_kmc_action_mask,
)
from lzero.model.kmc_graph_muzero_model import KMCGraphMuZeroModel


# ============================================================
# Data structures
# ============================================================
@dataclass
class Transition:
    obs: np.ndarray          # flat graph obs
    action: int
    reward: float
    next_obs: np.ndarray
    action_mask: np.ndarray  # bool
    done: bool
    delta_t: float = 0.0     # physical time step from KMC Poisson process


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
# Environment wrapper (thin, no DI-engine)
# ============================================================
class KMCEnvWrapper:
    """Wraps KMCEnv to produce flat graph observations + action masks."""

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
        obs = self._obs()
        mask = self._mask()
        return obs, mask

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        # Clamp action to valid vacancy range
        n_vac = self.env.V_nums
        vac_idx = action // 8
        if vac_idx >= n_vac:
            vac_idx = vac_idx % max(n_vac, 1)
            action = vac_idx * 8 + (action % 8)

        self.env._ensure_diffusion_rates()
        flat_rates = [r for vr in self.env.diffusion_rates for r in vr if r > 0]
        total_rate = float(np.sum(flat_rates)) if flat_rates else 0.0

        self.env.step_fast(int(action), self.timestep)

        if total_rate > 0:
            delta_t = -np.log(np.random.rand()) / total_rate
        else:
            delta_t = 0.0
        self.env.time += delta_t
        self.env.time_history.append(self.env.time)

        energy_after = self.env.calculate_system_energy()
        delta_E = self.env.energy_last - energy_after
        reward = float(delta_E * self.env.args.reward_scale)
        self.env.energy_last = energy_after
        self.env.energy_history.append(energy_after)

        self.timestep += 1
        done = self.timestep >= self.max_steps

        obs = self._obs()
        mask = self._mask()
        info = {"delta_t": delta_t, "delta_E": delta_E}
        return obs, mask, reward, done, info

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
# Simplified MCTS (no C++ tree, pure Python)
# ============================================================
class SimpleMCTS:
    """Lightweight MCTS using the learned world model."""

    def __init__(self, model: KMCGraphMuZeroModel, num_simulations: int,
                 discount: float, c_puct: float, device: str,
                 use_physics_discount: bool = False,
                 time_scale_tau: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.discount = discount
        self.c_puct = c_puct
        self.device = device
        self.use_physics_discount = use_physics_discount
        self.time_scale_tau = time_scale_tau

    @torch.no_grad()
    def search(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Run MCTS and return improved policy (action probs)."""
        self.model.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        out = self.model.initial_inference(obs_t)
        latent = out.latent_state  # (1, D)

        # Get prior policy from model
        if out.policy_logits.shape[-1] != action_mask.shape[0]:
            # Truncate or pad
            logits = torch.zeros(1, action_mask.shape[0], device=self.device)
            n = min(out.policy_logits.shape[-1], action_mask.shape[0])
            logits[:, :n] = out.policy_logits[:, :n]
        else:
            logits = out.policy_logits

        mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        logits[0, ~mask_t] = -1e9
        prior = F.softmax(logits[0], dim=-1).cpu().numpy()

        n_actions = len(action_mask)
        visit_count = np.zeros(n_actions, dtype=np.float32)
        total_value = np.zeros(n_actions, dtype=np.float32)
        mean_value = np.zeros(n_actions, dtype=np.float32)

        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            return np.ones(n_actions) / n_actions

        # Physics-time discount: compute γ from ROOT state ONCE (Δt is a state
        # property — same for all actions, determined by Γ_tot of current config)
        if self.use_physics_discount and hasattr(self.model, 'latest_time_delta'):
            root_dt = max(self.model.latest_time_delta[0].item(), 1e-8)
            root_gamma = math.exp(-root_dt / self.time_scale_tau)
        else:
            root_gamma = self.discount

        for _ in range(self.num_simulations):
            # UCB selection
            total_visits = visit_count.sum() + 1
            exploration = self.c_puct * prior * np.sqrt(total_visits) / (1 + visit_count)
            ucb = mean_value + exploration
            ucb[~action_mask.astype(bool)] = -1e9
            action = int(np.argmax(ucb))

            # Simulate one step with dynamics model
            action_t = torch.tensor([action], device=self.device)
            dyn_out = self.model.recurrent_inference(latent, action_t)

            # Decode reward (categorical → scalar)
            if self.model.categorical_distribution:
                reward_probs = F.softmax(dyn_out.reward[0], dim=-1)
                support = torch.arange(
                    -300, 301, 1.0, device=self.device
                )[:reward_probs.shape[0]]
                r = (reward_probs * support).sum().item()
            else:
                r = dyn_out.reward[0].item()

            # Decode value
            if self.model.categorical_distribution:
                value_probs = F.softmax(dyn_out.value[0], dim=-1)
                support_v = torch.arange(
                    -300, 301, 1.0, device=self.device
                )[:value_probs.shape[0]]
                v = (value_probs * support_v).sum().item()
            else:
                v = dyn_out.value[0].item()

            # Use root_gamma (action-independent) for backup
            q = r + root_gamma * v
            visit_count[action] += 1
            total_value[action] += q
            mean_value[action] = total_value[action] / visit_count[action]

        # Policy from visit counts (temperature=1)
        if visit_count.sum() == 0:
            policy = np.ones(n_actions) / n_actions
        else:
            policy = visit_count / visit_count.sum()
        self.model.train()
        return policy


# ============================================================
# Training
# ============================================================
def collect_episode(env: KMCEnvWrapper, model: KMCGraphMuZeroModel,
                    mcts: SimpleMCTS, buffer: ReplayBuffer,
                    epsilon: float = 0.1) -> tuple[float, int]:
    """Collect one episode using MCTS policy. Returns (total_reward, steps)."""
    obs, mask = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        # MCTS search
        policy = mcts.search(obs, mask)

        # ε-greedy exploration
        if random.random() < epsilon:
            valid = np.flatnonzero(mask)
            action = int(np.random.choice(valid)) if len(valid) > 0 else 0
        else:
            action = int(np.random.choice(len(policy), p=policy))

        next_obs, next_mask, reward, done, info = env.step(action)

        buffer.push(Transition(
            obs=obs, action=action, reward=reward,
            next_obs=next_obs, action_mask=mask, done=done,
            delta_t=info.get("delta_t", 0.0),
        ))

        obs, mask = next_obs, next_mask
        total_reward += reward
        steps += 1

    return total_reward, steps


def train_step(model: KMCGraphMuZeroModel, optimizer: torch.optim.Optimizer,
               batch: List[Transition], device: str, discount: float,
               temperature: float = 300.0, use_entropy_reg: bool = False,
               use_physics_discount: bool = False, time_scale_tau: float = 1.0) -> dict:
    """One training step on a batch of transitions."""
    obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device)
    masks = torch.tensor(np.stack([t.action_mask for t in batch]), dtype=torch.bool, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    B = len(batch)

    # 1. Representation
    init_out = model.initial_inference(obs)
    latent = init_out.latent_state

    # 2. Dynamics (predict reward from action)
    dyn_out = model.recurrent_inference(latent, actions)

    # 3. Policy loss (cross-entropy with MCTS-like target = action taken)
    # Simple: use one-hot of taken action as target (behavior cloning component)
    # Plus exploration entropy bonus
    policy_logits = init_out.policy_logits
    if policy_logits.shape[-1] != masks.shape[-1]:
        padded = torch.zeros(B, masks.shape[-1], device=device)
        n = min(policy_logits.shape[-1], masks.shape[-1])
        padded[:, :n] = policy_logits[:, :n]
        policy_logits = padded

    # Mask invalid actions
    policy_logits[~masks] = -1e9
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = F.nll_loss(log_probs, actions)

    # 4. Reward loss
    if model.categorical_distribution:
        # Convert reward to categorical target
        support = torch.arange(-300, 301, 1.0, device=device)
        n_bins = len(support)
        # Clamp reward to support range
        r_clamped = rewards.clamp(-300, 300)
        # Distribute probability to nearest bins
        idx_f = (r_clamped - support[0])
        idx_low = idx_f.long().clamp(0, n_bins - 2)
        idx_high = (idx_low + 1).clamp(max=n_bins - 1)
        frac = idx_f - idx_low.float()
        target_dist = torch.zeros(B, n_bins, device=device)
        target_dist.scatter_(1, idx_low.unsqueeze(1), (1 - frac).unsqueeze(1))
        target_dist.scatter_add_(1, idx_high.unsqueeze(1), frac.unsqueeze(1))
        reward_pred = dyn_out.reward
        if reward_pred.shape[-1] != n_bins:
            n = min(reward_pred.shape[-1], n_bins)
            target_dist = target_dist[:, :n]
            reward_pred = reward_pred[:, :n]
        reward_loss = F.cross_entropy(reward_pred, target_dist)
    else:
        reward_loss = F.mse_loss(dyn_out.reward.squeeze(-1), rewards)

    # 5. Value loss (bootstrap target)
    delta_ts = torch.tensor([t.delta_t for t in batch], dtype=torch.float32, device=device)
    with torch.no_grad():
        next_out = model.initial_inference(next_obs)
        if model.categorical_distribution:
            support_v = torch.arange(-300, 301, 1.0, device=device)
            n_v = min(next_out.value.shape[-1], len(support_v))
            next_v_probs = F.softmax(next_out.value[:, :n_v], dim=-1)
            next_v = (next_v_probs * support_v[:n_v]).sum(dim=-1)
        else:
            next_v = next_out.value.squeeze(-1)
        # Physics-time discount: γ = exp(-Δt/τ) per step
        if use_physics_discount:
            physics_gamma = torch.exp(-delta_ts / time_scale_tau).clamp(0.01, 1.0)
        else:
            physics_gamma = torch.full_like(rewards, discount)
        target_v = rewards + physics_gamma * (1 - dones) * next_v

    if model.categorical_distribution:
        support = torch.arange(-300, 301, 1.0, device=device)
        n_bins = len(support)
        v_clamped = target_v.clamp(-300, 300)
        idx_f = v_clamped - support[0]
        idx_low = idx_f.long().clamp(0, n_bins - 2)
        idx_high = (idx_low + 1).clamp(max=n_bins - 1)
        frac = idx_f - idx_low.float()
        v_target_dist = torch.zeros(B, n_bins, device=device)
        v_target_dist.scatter_(1, idx_low.unsqueeze(1), (1 - frac).unsqueeze(1))
        v_target_dist.scatter_add_(1, idx_high.unsqueeze(1), frac.unsqueeze(1))
        value_pred = init_out.value
        if value_pred.shape[-1] != n_bins:
            n = min(value_pred.shape[-1], n_bins)
            v_target_dist = v_target_dist[:, :n]
            value_pred = value_pred[:, :n]
        value_loss = F.cross_entropy(value_pred, v_target_dist)
    else:
        value_loss = F.mse_loss(init_out.value.squeeze(-1), target_v)

    # Feature 2: Temperature-based entropy regularization
    probs = F.softmax(policy_logits, dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    if use_entropy_reg:
        kT = 8.617e-5 * temperature  # eV
        entropy_coeff = 0.01 * kT
    else:
        entropy_coeff = 0.0

    # Total loss
    # Time supervision loss — train in log-space, detach latent so time head
    # gets strong gradients without interfering with policy/reward backbone
    log_time_pred = model.predict_log_time_delta(latent.detach())
    log_time_target = torch.log(delta_ts.clamp(min=1e-10))
    time_loss = F.mse_loss(log_time_pred, log_time_target)
    loss = policy_loss + reward_loss + 0.25 * value_loss + 1.0 * time_loss - entropy_coeff * entropy

    optimizer.zero_grad()
    # Two-pass backward: backbone and time_head are decoupled (latent.detach())
    # so we backward them separately to avoid time_head's large gradients
    # inflating the total norm and starving backbone gradients during clipping.
    backbone_loss = policy_loss + reward_loss + 0.25 * value_loss - entropy_coeff * entropy
    backbone_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # time_loss only touches time_head (latent detached), separate graph
    time_loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "reward_loss": reward_loss.item(),
        "value_loss": value_loss.item(),
        "time_loss": time_loss.item(),
        "entropy": entropy.item(),
    }


def evaluate(env_cfg: dict, model: KMCGraphMuZeroModel,
             mcts: SimpleMCTS, n_episodes: int = 5) -> dict:
    """Evaluate model with greedy MCTS policy on fresh environments."""
    total_rewards = []
    total_positive = []
    total_steps = []

    for _ in range(n_episodes):
        eval_env = KMCEnvWrapper(env_cfg)
        obs, mask = eval_env.reset()
        ep_reward = 0.0
        ep_pos = 0
        done = False
        steps = 0
        while not done:
            policy = mcts.search(obs, mask)
            action = int(np.argmax(policy))
            obs, mask, reward, done, info = eval_env.step(action)
            ep_reward += reward
            if reward > 0:
                ep_pos += 1
            steps += 1
        total_rewards.append(ep_reward)
        total_positive.append(ep_pos)
        total_steps.append(steps)

    return {
        "eval_mean_reward": np.mean(total_rewards),
        "eval_mean_positive_steps": np.mean(total_positive),
        "eval_mean_episode_steps": np.mean(total_steps),
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

    # Training
    parser.add_argument("--collect_episodes", type=int, default=4)
    parser.add_argument("--train_steps_per_collect", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--total_iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument("--c_puct", type=float, default=1.25)
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_cu_density", type=float, default=None,
                        help="Cu density for eval (default: same as training)")
    parser.add_argument("--eval_v_density", type=float, default=None,
                        help="V density for eval (default: same as training)")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="muzero_kmc_results")
    # Feature flags
    parser.add_argument("--use_physics_discount", action="store_true", default=False,
                        help="Feature 1: use physics-time discount γ=exp(-Δt/τ) in MCTS and training")
    parser.add_argument("--time_scale_tau", type=float, default=1.0,
                        help="Time scale τ for physics discount γ=exp(-Δt/τ)")
    parser.add_argument("--use_entropy_reg", action="store_true", default=False,
                        help="Feature 2: temperature-based entropy regularization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Environment config
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

    # Test reset
    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    action_dim = mask.shape[0]
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Cu: {env.env.Cu_nums}, V: {env.env.V_nums}")
    print(f"Valid actions: {mask.sum()}")

    # Model
    model = KMCGraphMuZeroModel(
        observation_shape=obs_dim,
        action_space_size=action_dim,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        latent_state_dim=128,
        graph_hidden_size=32,
        per_vacancy_latent_dim=8,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        categorical_distribution=False,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    buffer = ReplayBuffer(args.buffer_size)
    mcts = SimpleMCTS(model, args.num_simulations, args.discount, args.c_puct, args.device,
                      use_physics_discount=args.use_physics_discount,
                      time_scale_tau=args.time_scale_tau)

    # Logging
    log_file = save_dir / "training_log.txt"
    best_eval_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"MuZero KMC Training")
    print(f"Cu density: {args.cu_density}, V density: {args.v_density}")
    print(f"Lattice: {args.lattice_size}, Episode length: {args.max_episode_steps}")
    print(f"MCTS sims: {args.num_simulations}, Discount: {args.discount}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    for iteration in range(1, args.total_iterations + 1):
        t0 = time.time()
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * min(1.0, iteration / (args.total_iterations * 0.7))

        # Collect
        collect_rewards = []
        collect_steps = []
        for _ in range(args.collect_episodes):
            ep_r, ep_s = collect_episode(env, model, mcts, buffer, epsilon)
            collect_rewards.append(ep_r)
            collect_steps.append(ep_s)

        # Train
        train_losses = []
        if len(buffer) >= args.batch_size:
            for _ in range(args.train_steps_per_collect):
                batch = buffer.sample(args.batch_size)
                losses = train_step(model, optimizer, batch, args.device, args.discount,
                                    temperature=args.temperature,
                                    use_entropy_reg=args.use_entropy_reg,
                                    use_physics_discount=args.use_physics_discount,
                                    time_scale_tau=args.time_scale_tau)
                train_losses.append(losses)

        collect_time = time.time() - t0
        avg_loss = np.mean([l["loss"] for l in train_losses]) if train_losses else 0
        avg_policy_loss = np.mean([l["policy_loss"] for l in train_losses]) if train_losses else 0
        avg_reward_loss = np.mean([l["reward_loss"] for l in train_losses]) if train_losses else 0

        log_msg = (
            f"[Iter {iteration:4d}/{args.total_iterations}] "
            f"collect_reward={np.mean(collect_rewards):+.4f} "
            f"loss={avg_loss:.4f} (pol={avg_policy_loss:.4f} rew={avg_reward_loss:.4f}) "
            f"eps={epsilon:.3f} buf={len(buffer)} time={collect_time:.1f}s"
        )
        print(log_msg)

        # Evaluate
        if iteration % args.eval_freq == 0:
            eval_results = evaluate(eval_cfg, model, mcts, args.eval_episodes)
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
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                print(f"  >>> New best model! reward={best_eval_reward:+.4f}")

        # Save checkpoint
        if iteration % args.save_freq == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_eval_reward,
            }, save_dir / f"checkpoint_{iteration}.pt")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation...")
    final_eval = evaluate(eval_cfg, model, mcts, n_episodes=10)
    print(f"Final mean reward: {final_eval['eval_mean_reward']:+.4f}")
    print(f"Final rewards: {final_eval['eval_rewards']}")
    print(f"Best eval reward: {best_eval_reward:+.4f}")

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
