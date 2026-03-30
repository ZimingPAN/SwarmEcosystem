from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_project_root() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    rlkmc_root = root.parent / "RLKMC-MASSIVE-main"
    if str(rlkmc_root) not in sys.path:
        sys.path.insert(0, str(rlkmc_root))


_ensure_project_root()

from lzero.model.kmc_graph_muzero_model import KMCGraphMuZeroModel
from zoo.kmc.envs.kmc_lightzero_env import KMCLightZeroEnv


@dataclass
class Transition:
    observation: np.ndarray
    next_observation: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    energy_delta: float
    delta_t: float
    rate_target: float


def build_env(args: argparse.Namespace) -> KMCLightZeroEnv:
    return KMCLightZeroEnv(
        {
            "device": "cpu",
            "max_episode_steps": int(args.rollout_steps),
            "use_system_stats": bool(args.use_system_stats),
            "lattice_size": (args.lattice_x, args.lattice_y, args.lattice_z),
            "max_vacancies": int(args.max_vacancies),
            "neighbor_order": args.neighbor_order,
            "max_shells": args.max_shells,
            "max_defects": args.max_defects,
            "cu_density": args.cu_density,
            "v_density": args.v_density,
            "reward_scale": args.reward_scale,
            "seed": args.seed,
        }
    )


def oracle_best_action(env: KMCLightZeroEnv, action_mask: np.ndarray) -> tuple[int, float, float, float]:
    valid_actions = np.flatnonzero(action_mask)
    if len(valid_actions) == 0:
        return 0, 0.0, 0.0, 0.0
    best_action = int(valid_actions[0])
    best_reward = -float("inf")
    best_dt = 0.0
    best_energy_delta = 0.0
    for action in valid_actions.tolist():
        probe_env = copy.deepcopy(env._env)
        _obs, _full_obs, _positions, reward, _done, info = probe_env.step_with_stats(int(action), env._timestep)
        reward = float(reward)
        if reward > best_reward:
            best_reward = reward
            best_action = int(action)
            best_dt = float(info.get("delta_t", 0.0))
            best_energy_delta = float(info.get("energy_change", reward))
    return best_action, best_reward, best_dt, best_energy_delta


def collect_dataset(env_prototype: KMCLightZeroEnv, episodes: int) -> tuple[list[Transition], float]:
    dataset: list[Transition] = []
    total_oracle_reward = 0.0
    for _ in range(episodes):
        env = copy.deepcopy(env_prototype)
        obs = env.reset()
        done = False
        while not done:
            action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
            action, reward, delta_t, energy_delta = oracle_best_action(env, action_mask)
            timestep = env.step(action)
            rate_target = float(reward / max(delta_t, 1e-8))
            dataset.append(
                Transition(
                    observation=np.asarray(obs["observation"], dtype=np.float32),
                    next_observation=np.asarray(timestep.obs["observation"], dtype=np.float32),
                    action_mask=action_mask.copy(),
                    action=action,
                    reward=reward,
                    energy_delta=energy_delta,
                    delta_t=delta_t,
                    rate_target=rate_target,
                )
            )
            total_oracle_reward += float(timestep.reward)
            obs = timestep.obs
            done = bool(timestep.done)
    return dataset, total_oracle_reward / max(len(dataset), 1)


def save_dataset(dataset: list[Transition], path: str) -> None:
    payload = [x.__dict__ for x in dataset]
    torch.save(payload, path)


def load_dataset(path: str) -> list[Transition]:
    payload = torch.load(path, map_location="cpu")
    return [Transition(**item) for item in payload]


def masked_policy_logits(policy_logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return policy_logits.masked_fill(action_mask <= 0, -1e9)


def train_pretrain(
    model: KMCGraphMuZeroModel,
    dataset: list[Transition],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    positive_weight: float,
    dt_loss_weight: float,
    latent_loss_weight: float,
    rate_loss_weight: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        random.shuffle(dataset)
        logs = {
            "policy": 0.0,
            "reward": 0.0,
            "delta_t": 0.0,
            "latent": 0.0,
            "rate": 0.0,
        }
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            obs = torch.tensor(np.stack([x.observation for x in batch]), dtype=torch.float32, device=device)
            next_obs = torch.tensor(np.stack([x.next_observation for x in batch]), dtype=torch.float32, device=device)
            action_mask = torch.tensor(np.stack([x.action_mask for x in batch]), dtype=torch.float32, device=device)
            actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device).unsqueeze(-1)
            rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device)
            delta_t = torch.tensor([x.delta_t for x in batch], dtype=torch.float32, device=device)
            rate_target = torch.tensor([x.rate_target for x in batch], dtype=torch.float32, device=device)

            init_output = model.initial_inference(obs)
            init_logits = masked_policy_logits(init_output.policy_logits.float(), action_mask)
            ce_raw = F.cross_entropy(init_logits, actions.squeeze(-1), reduction="none")
            sample_weight = torch.ones_like(ce_raw) + (rewards > 0).float() * float(positive_weight)
            policy_loss = (ce_raw * sample_weight).mean()

            next_output = model.recurrent_inference(init_output.latent_state, actions)
            pred_reward = next_output.reward.float().squeeze(-1)
            pred_dt = model.latest_time_delta.float()
            reward_loss = F.smooth_l1_loss(pred_reward, rewards)
            dt_loss = F.smooth_l1_loss(pred_dt, delta_t)
            pred_rate = pred_reward / pred_dt.clamp(min=1e-8)
            rate_loss = F.smooth_l1_loss(pred_rate, rate_target)

            with torch.no_grad():
                next_target = model.initial_inference(next_obs).latent_state.detach()
            latent_loss = F.smooth_l1_loss(next_output.latent_state.float(), next_target.float())

            loss = (
                policy_loss
                + reward_loss
                + float(dt_loss_weight) * dt_loss
                + float(latent_loss_weight) * latent_loss
                + float(rate_loss_weight) * rate_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs["policy"] += float(policy_loss.item()) * len(batch)
            logs["reward"] += float(reward_loss.item()) * len(batch)
            logs["delta_t"] += float(dt_loss.item()) * len(batch)
            logs["latent"] += float(latent_loss.item()) * len(batch)
            logs["rate"] += float(rate_loss.item()) * len(batch)

        denom = max(len(dataset), 1)
        print(
            f"[muzero-physics] epoch={epoch} "
            f"policy_loss={logs['policy'] / denom:.6f} "
            f"reward_loss={logs['reward'] / denom:.6f} "
            f"delta_t_loss={logs['delta_t'] / denom:.6f} "
            f"latent_loss={logs['latent'] / denom:.6f} "
            f"rate_loss={logs['rate'] / denom:.6f}"
        )


def evaluate_policy(
    model: KMCGraphMuZeroModel,
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
                output = model.initial_inference(obs_tensor)
                logits = masked_policy_logits(output.policy_logits.float(), action_mask)
                action = int(torch.argmax(logits, dim=-1).item())
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
    parser = argparse.ArgumentParser(description="Physics-aware MuZero pretraining on A100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lattice_x", type=int, default=40)
    parser.add_argument("--lattice_y", type=int, default=40)
    parser.add_argument("--lattice_z", type=int, default=40)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--collect_episodes", type=int, default=8)
    parser.add_argument("--eval_episodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--positive_weight", type=float, default=8.0)
    parser.add_argument("--dt_loss_weight", type=float, default=0.2)
    parser.add_argument("--latent_loss_weight", type=float, default=0.5)
    parser.add_argument("--rate_loss_weight", type=float, default=0.2)
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=2e-4)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_vacancies", type=int, default=4)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--max_defects", type=int, default=96)
    parser.add_argument("--use_system_stats", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_path", type=str, default="")
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
    first_obs = env_prototype.reset()
    model = KMCGraphMuZeroModel(
        observation_shape=first_obs["observation"].shape[0],
        action_space_size=env_prototype.action_space.n,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        lattice_size=(args.lattice_x, args.lattice_y, args.lattice_z),
        neighbor_order=args.neighbor_order,
        categorical_distribution=False,
    ).to(device)

    if args.dataset_path and Path(args.dataset_path).exists():
        dataset = load_dataset(args.dataset_path)
        oracle_avg_reward = float(np.mean([x.reward for x in dataset])) if dataset else 0.0
    else:
        dataset, oracle_avg_reward = collect_dataset(env_prototype, episodes=args.collect_episodes)
        if args.dataset_path:
            Path(args.dataset_path).parent.mkdir(parents=True, exist_ok=True)
            save_dataset(dataset, args.dataset_path)
    print(
        f"[muzero-physics] dataset_size={len(dataset)} oracle_avg_reward={oracle_avg_reward:.6f} "
        f"positive_ratio={sum(1 for x in dataset if x.reward > 0) / max(len(dataset), 1):.6f} "
        f"vacancies={env_prototype._env.V_nums} cu_atoms={env_prototype._env.Cu_nums}"
    )

    train_pretrain(
        model,
        dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight=args.positive_weight,
        dt_loss_weight=args.dt_loss_weight,
        latent_loss_weight=args.latent_loss_weight,
        rate_loss_weight=args.rate_loss_weight,
    )
    metrics = evaluate_policy(model, env_prototype, device=device, episodes=args.eval_episodes)
    print(
        f"[muzero-physics] eval_avg_reward={metrics['avg_reward']:.6f} "
        f"positive_step_ratio={metrics['positive_step_ratio']:.6f} steps={int(metrics['steps'])}"
    )


if __name__ == "__main__":
    main()
