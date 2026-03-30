from __future__ import annotations

import random

import numpy as np
import torch

from dreamer4.kmc import KMCDynamicsWorldModel
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

    model = KMCDynamicsWorldModel(
        dim=64,
        dim_latent=32,
        max_vacancies=32,
        max_defects=384,
        max_shells=16,
        node_feat_dim=4,
        stats_dim=10,
        graph_hidden_size=128,
        lattice_size=(40, 40, 40),
        neighbor_order=env._cfg.neighbor_order,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(3):
        batch_obs = torch.tensor(obs["observation"]).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1).float()
        latents = model.encode_observation(batch_obs)
        time_delta = model.predict_time_delta(latents)
        target = torch.full_like(time_delta, 0.5 + step * 0.1)
        loss = torch.nn.functional.mse_loss(time_delta, target) + latents.float().mean() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid = np.flatnonzero(obs["action_mask"])
        action = int(valid[0]) if len(valid) else 0
        timestep = env.step(action)
        print(
            f"[dreamer-smoke] step={step} action={action} reward={float(timestep.reward):.6f} "
            f"raw_reward={float(timestep.info.get('raw_reward', timestep.reward)):.6f} "
            f"dt_mean={float(time_delta.mean()):.6f} loss={float(loss):.6f}"
        )
        obs = timestep.obs

    print("[dreamer-smoke] done")


if __name__ == "__main__":
    main()
