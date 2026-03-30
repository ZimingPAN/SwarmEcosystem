from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_roots() -> None:
    root = Path(__file__).resolve().parents[0]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    lightzero_root = root.parent / "LightZero-main"
    if str(lightzero_root) not in sys.path:
        sys.path.insert(0, str(lightzero_root))


_ensure_roots()

from dreamer4.kmc import KMCDynamicsWorldModel
from zoo.kmc.envs.kmc_lightzero_env import KMCLightZeroEnv


@dataclass
class Transition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    delta_t: float


class DreamerKMCPolicy(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_latent: int,
        max_vacancies: int,
        max_defects: int,
        max_shells: int,
        stats_dim: int,
        lattice_size: tuple[int, int, int],
        neighbor_order: str | int | None,
    ) -> None:
        super().__init__()
        self.max_vacancies = int(max_vacancies)
        self.dim_latent = int(dim_latent)
        self.world_model = KMCDynamicsWorldModel(
            dim=dim,
            dim_latent=dim_latent,
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=4,
            stats_dim=stats_dim,
            graph_hidden_size=128,
            lattice_size=lattice_size,
            neighbor_order=neighbor_order,
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(max_vacancies * dim_latent),
            nn.Linear(max_vacancies * dim_latent, max_vacancies * dim_latent),
            nn.SiLU(),
            nn.Linear(max_vacancies * dim_latent, max_vacancies * 8),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        latents = self.world_model.encode_observation(obs)
        logits = self.policy_head(latents[:, 0].reshape(latents.shape[0], -1))
        time_delta = self.world_model.predict_time_delta(latents).reshape(latents.shape[0], -1).mean(dim=-1)
        return logits, time_delta


def build_env(args: argparse.Namespace) -> KMCLightZeroEnv:
    return KMCLightZeroEnv(
        {
            "device": "cpu",
            "max_episode_steps": int(args.rollout_steps),
            "use_system_stats": False,
            "lattice_size": (args.lattice_x, args.lattice_y, args.lattice_z),
            "max_vacancies": int(args.max_vacancies),
            "max_defects": int(args.max_defects),
            "max_shells": int(args.max_shells),
            "neighbor_order": args.neighbor_order,
            "cu_density": args.cu_density,
            "v_density": args.v_density,
            "reward_scale": args.reward_scale,
            "seed": args.seed,
        }
    )


def oracle_best_action(env: KMCLightZeroEnv, action_mask: np.ndarray) -> tuple[int, float, float]:
    valid_actions = np.flatnonzero(action_mask)
    if len(valid_actions) == 0:
        return 0, 0.0, 0.0
    best_action = int(valid_actions[0])
    best_reward = -float("inf")
    best_dt = 0.0
    for action in valid_actions.tolist():
        probe_env = copy.deepcopy(env._env)
        _obs, _full_obs, _positions, reward, _done, info = probe_env.step_with_stats(int(action), env._timestep)
        reward = float(reward)
        if reward > best_reward:
            best_reward = reward
            best_action = int(action)
            best_dt = float(info.get("delta_t", 0.0))
    return best_action, best_reward, best_dt


def collect_dataset(env_prototype: KMCLightZeroEnv, episodes: int) -> tuple[list[Transition], float]:
    dataset: list[Transition] = []
    total_reward = 0.0
    for _ in range(episodes):
        env = copy.deepcopy(env_prototype)
        obs = env.reset()
        done = False
        while not done:
            action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
            action, reward, delta_t = oracle_best_action(env, action_mask)
            dataset.append(
                Transition(
                    observation=np.asarray(obs["observation"], dtype=np.float32),
                    action_mask=action_mask.copy(),
                    action=action,
                    reward=reward,
                    delta_t=delta_t,
                )
            )
            timestep = env.step(action)
            total_reward += float(timestep.reward)
            obs = timestep.obs
            done = bool(timestep.done)
    return dataset, total_reward / max(len(dataset), 1)


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask <= 0, -1e9)


def train_bc(
    model: DreamerKMCPolicy,
    dataset: list[Transition],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    positive_weight: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        random.shuffle(dataset)
        ce_running = 0.0
        dt_running = 0.0
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            obs = torch.tensor(np.stack([x.observation for x in batch]), dtype=torch.float32, device=device)
            action_mask = torch.tensor(np.stack([x.action_mask for x in batch]), dtype=torch.float32, device=device)
            actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device)
            rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device)
            delta_t = torch.tensor([x.delta_t for x in batch], dtype=torch.float32, device=device)

            logits, pred_dt = model(obs)
            logits = masked_logits(logits, action_mask)
            ce_raw = F.cross_entropy(logits, actions, reduction="none")
            sample_weight = torch.ones_like(ce_raw) + (rewards > 0).float() * float(positive_weight)
            ce_loss = (ce_raw * sample_weight).mean()
            dt_loss = F.smooth_l1_loss(pred_dt, delta_t)
            loss = ce_loss + 0.1 * dt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ce_running += float(ce_loss.item()) * len(batch)
            dt_running += float(dt_loss.item()) * len(batch)
        print(
            f"[dreamer-oracle] epoch={epoch} ce_loss={ce_running / max(len(dataset), 1):.6f} "
            f"dt_loss={dt_running / max(len(dataset), 1):.6f}"
        )


def evaluate_policy(
    model: DreamerKMCPolicy,
    env_prototype: KMCLightZeroEnv,
    *,
    device: torch.device,
    episodes: int,
) -> dict[str, float]:
    model.eval()
    total_reward = 0.0
    total_steps = 0
    positive_steps = 0
    with torch.no_grad():
        for _ in range(episodes):
            env = copy.deepcopy(env_prototype)
            obs = env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                action_mask = torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                logits, _pred_dt = model(obs_tensor)
                action = int(torch.argmax(masked_logits(logits, action_mask), dim=-1).item())
                timestep = env.step(action)
                reward = float(timestep.reward)
                total_reward += reward
                positive_steps += int(reward > 0.0)
                total_steps += 1
                obs = timestep.obs
                done = bool(timestep.done)
    return {
        "avg_reward": total_reward / max(total_steps, 1),
        "positive_step_ratio": positive_steps / max(total_steps, 1),
        "steps": float(total_steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-greedy BC probe for Dreamer4 KMC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lattice_x", type=int, default=20)
    parser.add_argument("--lattice_y", type=int, default=20)
    parser.add_argument("--lattice_z", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--collect_episodes", type=int, default=16)
    parser.add_argument("--eval_episodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--positive_weight", type=float, default=16.0)
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=2e-4)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_vacancies", type=int, default=4)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--max_defects", type=int, default=96)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim_latent", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    env_prototype = build_env(args)
    env_prototype.reset()
    model = DreamerKMCPolicy(
        dim=args.dim,
        dim_latent=args.dim_latent,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=10,
        lattice_size=(args.lattice_x, args.lattice_y, args.lattice_z),
        neighbor_order=args.neighbor_order,
    ).to(device)

    dataset, oracle_avg_reward = collect_dataset(env_prototype, episodes=args.collect_episodes)
    print(
        f"[dreamer-oracle] dataset_size={len(dataset)} oracle_avg_reward={oracle_avg_reward:.6f} "
        f"positive_ratio={sum(1 for x in dataset if x.reward > 0) / max(len(dataset), 1):.6f} "
        f"vacancies={env_prototype._env.V_nums} cu_atoms={env_prototype._env.Cu_nums}"
    )

    train_bc(
        model,
        dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight=args.positive_weight,
    )
    metrics = evaluate_policy(model, env_prototype, device=device, episodes=args.eval_episodes)
    print(
        f"[dreamer-oracle] eval_avg_reward={metrics['avg_reward']:.6f} "
        f"positive_step_ratio={metrics['positive_step_ratio']:.6f} steps={int(metrics['steps'])}"
    )


if __name__ == "__main__":
    main()
