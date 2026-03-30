"""Standalone PPO training with the SAME GNN encoder and environment wrapper
used by the MuZero / Dreamer world-model baselines, for fair comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
LIGHTZERO = ROOT.parent / "LightZero-main"
for p in [str(ROOT), str(LIGHTZERO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from RL4KMC.envs.kmc import KMCEnv
from RL4KMC.world_models import (
    DefectGraphEncoder,
    DefectGraphObservationShape,
    build_defect_graph_observation,
    build_kmc_action_mask,
    unflatten_defect_graph_observation,
)


# ── KMCGraphEncoder (copied verbatim from dreamer4/kmc.py to avoid
#    pulling in the full dreamer4 dependency tree) ──────────────────────
class KMCGraphEncoder(nn.Module):
    """Wraps DefectGraphEncoder with stats-conditioned FiLM modulation.

    Identical to ``dreamer4.kmc.KMCGraphEncoder``.
    """

    def __init__(
        self,
        *,
        max_vacancies: int,
        max_defects: int,
        max_shells: int = 16,
        node_feat_dim: int = 4,
        stats_dim: int = 10,
        hidden_size: int = 128,
        dim_latent: int = 32,
        lattice_size=(40, 40, 40),
        neighbor_order: str | int | None = "2NN",
    ) -> None:
        super().__init__()
        self.shape = DefectGraphObservationShape(
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=node_feat_dim,
            stats_dim=stats_dim,
        )
        self.graph_encoder = DefectGraphEncoder(
            hidden_size=hidden_size,
            output_dim=dim_latent,
            neighbor_order=neighbor_order,
            lattice_size=tuple(int(v) for v in lattice_size),
        )
        self.stats_to_token = nn.Sequential(
            nn.Linear(stats_dim, dim_latent),
            nn.LayerNorm(dim_latent),
            nn.SiLU(),
        )
        self.stats_to_scale_shift = nn.Sequential(
            nn.Linear(stats_dim, dim_latent * 2),
            nn.SiLU(),
            nn.Linear(dim_latent * 2, dim_latent * 2),
        )

    def forward(self, flat_observation: torch.Tensor) -> torch.Tensor:
        squeeze_time = False
        if flat_observation.ndim == 2:
            flat_observation = flat_observation.unsqueeze(1)
            squeeze_time = True
        if flat_observation.ndim != 3:
            raise ValueError(
                f"expected [batch, time, dim] or [batch, dim], got {tuple(flat_observation.shape)}"
            )
        batch, time, _ = flat_observation.shape
        flat_bt = flat_observation.reshape(batch * time, -1)
        node_attr, node_mask, stats = unflatten_defect_graph_observation(flat_bt, shape=self.shape)
        device = flat_observation.device
        self.graph_encoder = self.graph_encoder.to(device)
        node_latents = self.graph_encoder(
            node_attr.to(device=device, dtype=torch.float32),
            node_mask.to(device=device, dtype=torch.float32),
        )
        stats = stats.to(device=device, dtype=torch.float32)
        stats_token = self.stats_to_token(stats).unsqueeze(1)
        stats_scale, stats_shift = self.stats_to_scale_shift(stats).chunk(2, dim=-1)
        latents = (
            node_latents * (1.0 + stats_scale.unsqueeze(1))
            + stats_shift.unsqueeze(1)
            + stats_token
        )
        latents = latents.reshape(batch, time, self.shape.max_vacancies, -1)
        if squeeze_time:
            return latents[:, 0]
        return latents

# ── Environment wrapper (identical to train_dreamer_standalone.py) ─────
class KMCEnvWrapper:
    """Thin wrapper that returns flat graph observations + action masks."""

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
        share_obs[0] = self.cfg.get("temperature", 300.0) / 1000.0
        share_obs[1] = self.cfg.get("cu_density", 0.05)
        share_obs[2] = self.cfg.get("v_density", 0.0002)
        return build_defect_graph_observation(
            self.env, shape=self.shape, share_obs=share_obs,
        ).astype(np.float32)

    def _mask(self) -> np.ndarray:
        return build_kmc_action_mask(
            self.env, max_vacancies=self.shape.max_vacancies,
        )


# ── PPO Agent ──────────────────────────────────────────────────────────
class PPOGNNAgent(nn.Module):
    """Actor-Critic with the same GNN encoder used by Dreamer / MuZero."""

    def __init__(
        self,
        *,
        max_vacancies: int,
        max_defects: int,
        max_shells: int,
        stats_dim: int,
        lattice_size: tuple[int, ...],
        neighbor_order: str,
        action_space_size: int,
        graph_hidden_size: int = 32,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.graph_encoder = KMCGraphEncoder(
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=4,
            stats_dim=stats_dim,
            hidden_size=graph_hidden_size,
            dim_latent=latent_dim,
            lattice_size=lattice_size,
            neighbor_order=neighbor_order,
        )

        latent_flat = max_vacancies * latent_dim

        self.actor = nn.Sequential(
            nn.LayerNorm(latent_flat),
            nn.Linear(latent_flat, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_space_size),
        )

        self.critic = nn.Sequential(
            nn.LayerNorm(latent_flat),
            nn.Linear(latent_flat, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def encode(self, obs_flat: torch.Tensor) -> torch.Tensor:
        latent = self.graph_encoder(obs_flat)          # (B, V, D)
        return latent.reshape(latent.shape[0], -1)     # (B, V*D)

    def get_action_and_value(
        self,
        obs_flat: torch.Tensor,
        mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        latent = self.encode(obs_flat)
        logits = self.actor(latent)
        logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)

        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(latent).squeeze(-1)
        return action, log_prob, entropy, value


# ── Rollout buffer ─────────────────────────────────────────────────────
@dataclass
class RolloutBuffer:
    obs: list          # list of np.ndarray
    actions: list      # list of int
    log_probs: list    # list of float
    rewards: list      # list of float
    dones: list        # list of bool
    values: list       # list of float
    masks: list        # list of np.ndarray

    @staticmethod
    def empty() -> "RolloutBuffer":
        return RolloutBuffer([], [], [], [], [], [], [])

    def append(self, obs, action, log_prob, reward, done, value, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(mask)

    def __len__(self):
        return len(self.rewards)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (advantages, returns) using Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ── PPO update ─────────────────────────────────────────────────────────
def ppo_update(
    agent: PPOGNNAgent,
    optimizer: torch.optim.Optimizer,
    buf: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    *,
    ppo_epochs: int,
    clip_param: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    obs_t = torch.as_tensor(np.stack(buf.obs), dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(buf.actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(np.stack(buf.masks), dtype=torch.bool, device=device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    # Normalise advantages
    if advantages_t.numel() > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for _ in range(ppo_epochs):
        _, new_log_probs, entropy, values = agent.get_action_and_value(
            obs_t, masks_t, action=actions_t,
        )
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns_t)
        ent_loss = entropy.mean()

        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * ent_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += ent_loss.item()

    n = ppo_epochs
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
    }


# ── Evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    agent: PPOGNNAgent,
    env_cfg: dict,
    n_episodes: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    agent.eval()
    rewards_list: list[float] = []
    for _ in range(n_episodes):
        env = KMCEnvWrapper(env_cfg)
        obs, mask = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs_t, mask_t, deterministic=True)
            obs, mask, reward, done, _ = env.step(action.item())
            ep_reward += reward
        rewards_list.append(ep_reward)
    agent.train()
    return float(np.mean(rewards_list)), rewards_list


# ── Main ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO + GNN baseline for KMC")
    # Environment
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    p.add_argument("--cu_density", type=float, default=0.08)
    p.add_argument("--v_density", type=float, default=0.0002)
    p.add_argument("--max_episode_steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--reward_scale", type=float, default=10.0)
    p.add_argument("--neighbor_order", type=str, default="2NN")
    p.add_argument("--max_vacancies", type=int, default=32)
    p.add_argument("--max_defects", type=int, default=64)
    p.add_argument("--max_shells", type=int, default=16)
    # PPO hyper-parameters
    p.add_argument("--total_episodes", type=int, default=500)
    p.add_argument("--episode_length", type=int, default=None,
                    help="Rollout length per episode (defaults to max_episode_steps)")
    p.add_argument("--ppo_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_param", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_loss_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    # Eval / checkpointing
    p.add_argument("--eval_freq", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_cu_density", type=float, default=None,
                    help="Cu density for eval (default: same as training)")
    p.add_argument("--eval_v_density", type=float, default=None,
                    help="V density for eval (default: same as training)")
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="ppo_gnn_results")
    # GNN encoder
    p.add_argument("--graph_hidden_size", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Default episode_length to max_episode_steps
    if args.episode_length is None:
        args.episode_length = args.max_episode_steps

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Environment config dict (shared with eval)
    stats_dim = 10
    env_cfg = dict(
        lattice_size=tuple(args.lattice_size),
        cu_density=args.cu_density,
        v_density=args.v_density,
        max_episode_steps=args.max_episode_steps,
        temperature=args.temperature,
        reward_scale=args.reward_scale,
        neighbor_order=args.neighbor_order,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=stats_dim,
    )

    action_space_size = args.max_vacancies * 8

    # Build eval config (may differ from training for low-density testing)
    eval_cfg = dict(env_cfg)
    if args.eval_cu_density is not None:
        eval_cfg["cu_density"] = args.eval_cu_density
    if args.eval_v_density is not None:
        eval_cfg["v_density"] = args.eval_v_density

    # Build agent
    agent = PPOGNNAgent(
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=stats_dim,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        action_space_size=action_space_size,
        graph_hidden_size=args.graph_hidden_size,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    print(f"PPO-GNN agent  |  params: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"Device: {device}  |  Action space: {action_space_size}")
    print(f"Env: lattice={args.lattice_size}  cu={args.cu_density}  v={args.v_density}  T={args.temperature}")
    print("-" * 72)

    best_eval_reward = -float("inf")
    env = KMCEnvWrapper(env_cfg)

    for episode_idx in range(1, args.total_episodes + 1):
        t0 = time.time()
        agent.train()

        obs, mask = env.reset()
        buf = RolloutBuffer.empty()
        ep_reward = 0.0

        for _step in range(args.episode_length):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_t, mask_t)

            next_obs, next_mask, reward, done, info = env.step(action.item())
            buf.append(obs, action.item(), log_prob.item(), reward, done, value.item(), mask)
            ep_reward += reward
            obs, mask = next_obs, next_mask

            if done:
                break

        # Bootstrap value for last state if not terminal
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            _, _, _, last_value = agent.get_action_and_value(obs_t, mask_t)
            last_value = last_value.item()
        if done:
            last_value = 0.0

        # GAE
        rewards_np = np.array(buf.rewards, dtype=np.float32)
        values_np = np.array(buf.values, dtype=np.float32)
        dones_np = np.array(buf.dones, dtype=np.float32)
        advantages, returns = compute_gae(
            rewards_np, values_np, dones_np, last_value,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
        )

        # PPO update
        stats = ppo_update(
            agent, optimizer, buf, advantages, returns,
            ppo_epochs=args.ppo_epochs,
            clip_param=args.clip_param,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )

        elapsed = time.time() - t0
        print(
            f"[Episode {episode_idx:>4d}/{args.total_episodes}]  "
            f"reward={ep_reward:+.4f}  "
            f"p_loss={stats['policy_loss']:.4f}  "
            f"v_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # ── Evaluation ─────────────────────────────────────────────
        if episode_idx % args.eval_freq == 0:
            mean_r, rewards_list = evaluate(agent, eval_cfg, args.eval_episodes, device)
            rstr = ", ".join(f"{r:+.4f}" for r in rewards_list)
            print(f">>> EVAL: mean_reward={mean_r:+.4f}  rewards=[{rstr}]")
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                torch.save(agent.state_dict(), save_dir / "best_model.pt")
                print(f"    ★ New best model saved (reward={mean_r:+.4f})")

        # ── Checkpoint ─────────────────────────────────────────────
        if episode_idx % args.save_freq == 0:
            torch.save(
                {"episode": episode_idx, "model": agent.state_dict(),
                 "optimizer": optimizer.state_dict()},
                save_dir / f"checkpoint_{episode_idx}.pt",
            )
            print(f"    Checkpoint saved: checkpoint_{episode_idx}.pt")

    # Final save
    torch.save(agent.state_dict(), save_dir / "final_model.pt")
    print("-" * 72)
    print(f"Training complete. Best eval reward: {best_eval_reward:+.4f}")
    print(f"Models saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
