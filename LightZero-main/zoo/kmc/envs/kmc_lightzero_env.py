from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, Union

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


def _ensure_rlkmc_path() -> None:
    root = Path(__file__).resolve().parents[4] / "RLKMC-MASSIVE-main"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_rlkmc_path()

from RL4KMC.envs.kmc import KMCEnv  # noqa: E402
from RL4KMC.parser.parser import get_config  # noqa: E402
from RL4KMC.world_models import (  # noqa: E402
    DefectGraphObservationShape,
    build_kmc_action_mask,
    build_defect_graph_observation,
)


@ENV_REGISTRY.register("kmc_lightzero")
class KMCLightZeroEnv(BaseEnv):
    config = dict(
        lattice_size=(40, 40, 40),
        max_episode_steps=200,
        max_vacancies=32,
        max_defects=384,
        max_shells=16,
        node_feat_dim=4,
        stats_dim=10,
        temperature=300.0,
        reward_scale=1.0,
        cu_density=0.05,
        v_density=0.0002,
        lattice_cu_nums=0,
        lattice_v_nums=0,
        rlkmc_topk=16,
        seed=0,
        device="cpu",
        neighbor_order="2NN",
        use_system_stats=False,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = EasyDict(copy.deepcopy(self.config))
        self._cfg.update(cfg)
        self._shape = DefectGraphObservationShape(
            max_vacancies=int(self._cfg.max_vacancies),
            max_defects=int(self._cfg.max_defects),
            max_shells=int(self._cfg.max_shells),
            node_feat_dim=int(self._cfg.node_feat_dim),
            stats_dim=int(self._cfg.stats_dim),
        )
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._shape.flat_dim,),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Discrete(int(self._cfg.max_vacancies) * 8)
        self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self._eval_episode_return = 0.0
        self._init_flag = False
        self._timestep = 0

    def _build_args(self):
        parser = get_config()
        args = parser.parse_known_args([])[0]
        lattice_size = list(self._cfg.lattice_size)
        total_half_sites = int(np.prod(lattice_size) * 2)
        cu_count = int(round(float(self._cfg.cu_density) * total_half_sites))
        if self._cfg.v_density is not None:
            vac_count = int(round(float(self._cfg.v_density) * total_half_sites))
        else:
            vac_count = int(self._cfg.lattice_v_nums)
        vac_count = max(vac_count, 1)
        args.lattice_size = list(self._cfg.lattice_size)
        args.temperature = float(self._cfg.temperature)
        args.reward_scale = float(self._cfg.reward_scale)
        args.topk = int(self._cfg.rlkmc_topk)
        args.device = str(self._cfg.device)
        args.cu_density = float(self._cfg.cu_density)
        args.v_density = float(self._cfg.v_density)
        args.lattice_cu_nums = cu_count
        args.lattice_v_nums = vac_count
        args.compute_global_static_env_reset = True
        args.skip_stats = not bool(self._cfg.use_system_stats)
        args.skip_global_diffusion_reset = False
        args.max_ssa_rounds = int(self._cfg.max_episode_steps)
        args.neighbor_order = str(self._cfg.neighbor_order)
        return args

    def _get_share_obs(self) -> np.ndarray:
        if bool(self._cfg.use_system_stats):
            return np.asarray(self._env.get_system_stats(), dtype=np.float32)
        return np.zeros((self._shape.stats_dim,), dtype=np.float32)

    def _pack_obs(self, obs: dict, share_obs: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        action_mask = build_kmc_action_mask(self._env, max_vacancies=self._shape.max_vacancies)
        flat_obs = build_defect_graph_observation(self._env, shape=self._shape, share_obs=share_obs)
        return {
            "observation": to_ndarray(flat_obs, dtype=np.float32),
            "action_mask": action_mask,
            "to_play": -1,
            "timestep": self._timestep,
        }

    def reset(self) -> Dict[str, np.ndarray]:
        if not self._init_flag:
            self._env = KMCEnv(self._build_args())
            if not bool(self._cfg.use_system_stats):
                self._env.get_system_stats = lambda: np.zeros((self._shape.stats_dim,), dtype=np.float32)
            self._init_flag = True
        obs, _full_obs = self._env.reset()
        self._eval_episode_return = 0.0
        self._timestep = 0
        share_obs = self._get_share_obs()
        return self._pack_obs(obs, share_obs=share_obs)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray):
            action = int(np.asarray(action).reshape(-1)[0])
        action = int(action)
        # Clamp action to valid vacancy range to prevent crashes
        n_vac = self._env.V_nums
        vac_idx = action // 8
        dir_idx = action % 8
        if vac_idx >= n_vac:
            vac_idx = vac_idx % max(n_vac, 1)
            action = vac_idx * 8 + dir_idx
        if bool(self._cfg.use_system_stats):
            obs, _full_obs, _positions, reward, _done, info = self._env.step(int(action), self._timestep)
        else:
            self._env._ensure_diffusion_rates()
            flat_rates = [rate for vac_rates in self._env.diffusion_rates for rate in vac_rates if rate > 0]
            total_rate = float(np.sum(flat_rates)) if flat_rates else 0.0
            obs = self._env.step_fast(int(action), self._timestep)
            if total_rate > 0.0:
                delta_t = -np.log(np.random.rand()) / total_rate
            else:
                delta_t = 0.0
            self._env.time += delta_t
            self._env.time_history.append(self._env.time)
            energy_after = self._env.calculate_system_energy()
            delta_E = self._env.energy_last - energy_after
            reward = float(delta_E * self._env.args.reward_scale)
            self._env.energy_last = energy_after
            self._env.energy_history.append(energy_after)
            info = {
                "individual_reward": reward,
                "energy_change": float(delta_E),
                "time": float(self._env.time),
                "delta_t": float(delta_t),
            }
        info["raw_reward"] = float(reward)
        self._eval_episode_return += float(reward)
        self._timestep += 1
        done = self._timestep >= int(self._cfg.max_episode_steps)
        if done:
            info["eval_episode_return"] = float(self._eval_episode_return)
        packed_obs = self._pack_obs(obs, share_obs=self._get_share_obs())
        return BaseEnvTimestep(packed_obs, np.float32(reward), done, info)

    def close(self) -> None:
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = int(seed)
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> np.ndarray:
        return to_ndarray([self.action_space.sample()], dtype=np.int64)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero KMC Env"
