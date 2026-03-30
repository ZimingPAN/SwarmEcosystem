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
    rlkmc_root = root.parent / "RLKMC-MASSIVE-main"
    if str(rlkmc_root) not in sys.path:
        sys.path.insert(0, str(rlkmc_root))


_ensure_roots()

from dreamer4.kmc import KMCDynamicsWorldModel
from zoo.kmc.envs.kmc_lightzero_env import KMCLightZeroEnv
from RL4KMC.world_models import DefectGraphObservationShape, unflatten_defect_graph_observation


@dataclass
class Transition:
    observation: np.ndarray
    next_observation: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    energy_delta: float
    delta_t: float


class DreamerKMCPhysicsModel(nn.Module):
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
        self.action_space_size = int(max_vacancies) * 8
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
            nn.Linear(max_vacancies * dim_latent, self.action_space_size),
        )
        self.action_embed = nn.Embedding(self.action_space_size, dim_latent)
        self.transition_head = nn.Sequential(
            nn.LayerNorm(dim_latent * 2),
            nn.Linear(dim_latent * 2, dim_latent * 2),
            nn.SiLU(),
            nn.Linear(dim_latent * 2, dim_latent),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self.world_model.encode_observation(obs)
        policy_logits = self.policy_head(latents.reshape(latents.shape[0], -1))
        energy_delta = self.world_model.predict_energy_delta(latents)
        time_delta = self.world_model.predict_time_delta(latents)
        topo_attr, topo_mask_logits = self.world_model.reconstruct_topology(latents)

        action_embed = self.action_embed(actions).unsqueeze(1).expand(-1, latents.shape[1], -1)
        next_latents = self.transition_head(torch.cat([latents, action_embed], dim=-1))
        return {
            "latents": latents,
            "policy_logits": policy_logits,
            "energy_delta": energy_delta,
            "time_delta": time_delta,
            "topology_attr": topo_attr,
            "topology_mask_logits": topo_mask_logits,
            "next_latents": next_latents,
        }


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
    total_reward = 0.0
    for _ in range(episodes):
        env = copy.deepcopy(env_prototype)
        obs = env.reset()
        done = False
        while not done:
            action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
            action, reward, delta_t, energy_delta = oracle_best_action(env, action_mask)
            timestep = env.step(action)
            dataset.append(
                Transition(
                    observation=np.asarray(obs["observation"], dtype=np.float32),
                    next_observation=np.asarray(timestep.obs["observation"], dtype=np.float32),
                    action_mask=action_mask.copy(),
                    action=action,
                    reward=reward,
                    energy_delta=energy_delta,
                    delta_t=delta_t,
                )
            )
            total_reward += float(timestep.reward)
            obs = timestep.obs
            done = bool(timestep.done)
    return dataset, total_reward / max(len(dataset), 1)


def save_dataset(dataset: list[Transition], path: str) -> None:
    payload = [x.__dict__ for x in dataset]
    torch.save(payload, path)


def load_dataset(path: str) -> list[Transition]:
    payload = torch.load(path, map_location="cpu")
    return [Transition(**item) for item in payload]


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask <= 0, -1e9)


def build_topology_targets(obs: torch.Tensor, shape: DefectGraphObservationShape) -> tuple[torch.Tensor, torch.Tensor]:
    node_attr, node_mask, _stats = unflatten_defect_graph_observation(obs, shape=shape)
    return node_attr, node_mask


def train(
    model: DreamerKMCPhysicsModel,
    dataset: list[Transition],
    *,
    device: torch.device,
    observation_shape: DefectGraphObservationShape,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    positive_weight: float,
    latent_loss_weight: float,
    topology_loss_weight: float,
    energy_loss_weight: float,
    delta_t_loss_weight: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        random.shuffle(dataset)
        logs = {
            "policy": 0.0,
            "delta_t": 0.0,
            "energy": 0.0,
            "topology": 0.0,
            "latent": 0.0,
        }
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            obs = torch.tensor(np.stack([x.observation for x in batch]), dtype=torch.float32, device=device)
            next_obs = torch.tensor(np.stack([x.next_observation for x in batch]), dtype=torch.float32, device=device)
            action_mask = torch.tensor(np.stack([x.action_mask for x in batch]), dtype=torch.float32, device=device)
            actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device)
            rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device)
            energy_delta = torch.tensor([x.energy_delta for x in batch], dtype=torch.float32, device=device)
            delta_t = torch.tensor([x.delta_t for x in batch], dtype=torch.float32, device=device)

            out = model(obs, actions)
            logits = masked_logits(out["policy_logits"], action_mask)
            ce_raw = F.cross_entropy(logits, actions, reduction="none")
            sample_weight = torch.ones_like(ce_raw) + (rewards > 0).float() * float(positive_weight)
            policy_loss = (ce_raw * sample_weight).mean()

            delta_t_loss = F.smooth_l1_loss(out["time_delta"], delta_t)
            energy_loss = F.smooth_l1_loss(out["energy_delta"], energy_delta)

            gt_node_attr, gt_node_mask = build_topology_targets(obs, observation_shape)
            pred_node_attr = out["topology_attr"]
            pred_node_mask_logits = out["topology_mask_logits"]
            mask_loss = F.binary_cross_entropy_with_logits(pred_node_mask_logits, gt_node_mask)
            attr_weight = gt_node_mask.unsqueeze(-1)
            attr_loss = (
                F.smooth_l1_loss(pred_node_attr * attr_weight, gt_node_attr * attr_weight, reduction="sum")
                / attr_weight.sum().clamp(min=1.0)
            )
            topology_loss = mask_loss + attr_loss

            with torch.no_grad():
                next_latent_target = model.world_model.encode_observation(next_obs)
            latent_loss = F.smooth_l1_loss(out["next_latents"], next_latent_target)

            loss = (
                policy_loss
                + float(delta_t_loss_weight) * delta_t_loss
                + float(energy_loss_weight) * energy_loss
                + float(topology_loss_weight) * topology_loss
                + float(latent_loss_weight) * latent_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs["policy"] += float(policy_loss.item()) * len(batch)
            logs["delta_t"] += float(delta_t_loss.item()) * len(batch)
            logs["energy"] += float(energy_loss.item()) * len(batch)
            logs["topology"] += float(topology_loss.item()) * len(batch)
            logs["latent"] += float(latent_loss.item()) * len(batch)

        denom = max(len(dataset), 1)
        print(
            f"[dreamer-physics] epoch={epoch} "
            f"policy_loss={logs['policy'] / denom:.6f} "
            f"delta_t_loss={logs['delta_t'] / denom:.6f} "
            f"energy_loss={logs['energy'] / denom:.6f} "
            f"topology_loss={logs['topology'] / denom:.6f} "
            f"latent_loss={logs['latent'] / denom:.6f}"
        )


def evaluate(
    model: DreamerKMCPhysicsModel,
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
                actions = torch.zeros((1,), dtype=torch.long, device=device)
                logits = masked_logits(model(obs_tensor, actions)["policy_logits"], action_mask)
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
    parser = argparse.ArgumentParser(description="Dreamer4 KMC physics pretraining on A100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lattice_x", type=int, default=40)
    parser.add_argument("--lattice_y", type=int, default=40)
    parser.add_argument("--lattice_z", type=int, default=40)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--collect_episodes", type=int, default=8)
    parser.add_argument("--eval_episodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--positive_weight", type=float, default=16.0)
    parser.add_argument("--latent_loss_weight", type=float, default=0.5)
    parser.add_argument("--topology_loss_weight", type=float, default=0.2)
    parser.add_argument("--energy_loss_weight", type=float, default=0.2)
    parser.add_argument("--delta_t_loss_weight", type=float, default=0.2)
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=2e-4)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_vacancies", type=int, default=4)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--max_defects", type=int, default=96)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim_latent", type=int, default=32)
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
    shape = DefectGraphObservationShape(
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        node_feat_dim=4,
        stats_dim=10,
    )
    model = DreamerKMCPhysicsModel(
        dim=args.dim,
        dim_latent=args.dim_latent,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=10,
        lattice_size=(args.lattice_x, args.lattice_y, args.lattice_z),
        neighbor_order=args.neighbor_order,
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
        f"[dreamer-physics] dataset_size={len(dataset)} oracle_avg_reward={oracle_avg_reward:.6f} "
        f"positive_ratio={sum(1 for x in dataset if x.reward > 0) / max(len(dataset), 1):.6f} "
        f"vacancies={env_prototype._env.V_nums} cu_atoms={env_prototype._env.Cu_nums}"
    )
    train(
        model,
        dataset,
        device=device,
        observation_shape=shape,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight=args.positive_weight,
        latent_loss_weight=args.latent_loss_weight,
        topology_loss_weight=args.topology_loss_weight,
        energy_loss_weight=args.energy_loss_weight,
        delta_t_loss_weight=args.delta_t_loss_weight,
    )
    metrics = evaluate(model, env_prototype, device=device, episodes=args.eval_episodes)
    print(
        f"[dreamer-physics] eval_avg_reward={metrics['avg_reward']:.6f} "
        f"positive_step_ratio={metrics['positive_step_ratio']:.6f} steps={int(metrics['steps'])}"
    )


if __name__ == "__main__":
    main()
