from __future__ import annotations

import random

import numpy as np
import torch

from lzero.model.kmc_graph_muzero_model import KMCGraphMuZeroModel
from zoo.kmc.envs.kmc_lightzero_env import KMCLightZeroEnv


def main() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = KMCLightZeroEnv(
        {
            "device": "cpu",
            "max_episode_steps": 8,
            "use_system_stats": False,
        }
    )
    obs = env.reset()
    model = KMCGraphMuZeroModel(
        observation_shape=obs["observation"].shape[0],
        action_space_size=env.action_space.n,
        neighbor_order=env._cfg.neighbor_order,
        lattice_size=env._cfg.lattice_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    rewards = []
    for step in range(3):
        valid = np.flatnonzero(obs["action_mask"])
        action = int(valid[0]) if len(valid) else 0
        timestep = env.step(action)
        rewards.append(float(timestep.reward))

        batch_obs = torch.tensor(timestep.obs["observation"]).unsqueeze(0).repeat(2, 1).float()
        output = model.initial_inference(batch_obs)
        loss = (
            output.value.float().mean()
            + output.policy_logits.float().mean()
            + model.latest_time_delta.float().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"[lightzero-smoke] step={step} action={action} reward={float(timestep.reward):.6f} "
            f"raw_reward={float(timestep.info.get('raw_reward', timestep.reward)):.6f} loss={float(loss):.6f}"
        )
        obs = timestep.obs

    print(
        f"[lightzero-smoke] done min_reward={min(rewards):.6f} "
        f"max_reward={max(rewards):.6f} total_reward={sum(rewards):.6f}"
    )


if __name__ == "__main__":
    main()
