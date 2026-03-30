from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class KMCObservationShape:
    max_vacancies: int
    top_k: int
    node_feat_dim: int = 14
    stats_dim: int = 10

    @property
    def flat_dim(self) -> int:
        return self.max_vacancies * self.node_feat_dim + self.max_vacancies * self.top_k * 4 + self.stats_dim


def _to_numpy(array_like: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if array_like is None:
        return np.zeros((0,), dtype=dtype)
    if isinstance(array_like, np.ndarray):
        return array_like.astype(dtype, copy=False)
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(array_like, dtype=dtype)


def _pad_or_truncate(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(target_shape, dtype=np.float32)
    if array.size == 0:
        return out
    slices = tuple(slice(0, min(src, dst)) for src, dst in zip(array.shape, target_shape))
    out[slices] = array[slices]
    return out


def flatten_kmc_observation(
    obs: dict[str, Any],
    *,
    shape: KMCObservationShape,
    share_obs: Any | None = None,
) -> np.ndarray:
    v_feat = _to_numpy(obs.get("V_features_local"), dtype=np.float32)
    topk = obs.get("topk_update_info") or {}
    diff_k = _to_numpy(topk.get("diff_k"), dtype=np.float32)
    dist_k = _to_numpy(topk.get("dist_k"), dtype=np.float32)
    share_obs_np = _to_numpy(share_obs, dtype=np.float32)

    v_feat = _pad_or_truncate(v_feat.reshape(-1, shape.node_feat_dim), (shape.max_vacancies, shape.node_feat_dim))
    diff_k = _pad_or_truncate(diff_k.reshape(-1, shape.top_k, 3), (shape.max_vacancies, shape.top_k, 3))
    dist_k = _pad_or_truncate(dist_k.reshape(-1, shape.top_k), (shape.max_vacancies, shape.top_k))

    stats = np.zeros((shape.stats_dim,), dtype=np.float32)
    if share_obs_np.size > 0:
        stats[: min(shape.stats_dim, share_obs_np.size)] = share_obs_np.reshape(-1)[: shape.stats_dim]

    flat = np.concatenate(
        [
            v_feat.reshape(-1),
            diff_k.reshape(-1),
            dist_k.reshape(-1),
            stats,
        ],
        axis=0,
    )
    return flat.astype(np.float32, copy=False)


def unflatten_kmc_observation(
    flat_obs: torch.Tensor,
    *,
    shape: KMCObservationShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if flat_obs.ndim == 1:
        flat_obs = flat_obs.unsqueeze(0)
    batch = flat_obs.shape[0]
    expected = shape.flat_dim
    if flat_obs.shape[-1] != expected:
        raise ValueError(f"expected flat observation dim {expected}, got {flat_obs.shape[-1]}")

    idx = 0
    v_feat_size = shape.max_vacancies * shape.node_feat_dim
    diff_size = shape.max_vacancies * shape.top_k * 3
    dist_size = shape.max_vacancies * shape.top_k

    v_feat = flat_obs[:, idx:idx + v_feat_size].reshape(batch, shape.max_vacancies, shape.node_feat_dim)
    idx += v_feat_size
    diff_k = flat_obs[:, idx:idx + diff_size].reshape(batch, shape.max_vacancies, shape.top_k, 3)
    idx += diff_size
    dist_k = flat_obs[:, idx:idx + dist_size].reshape(batch, shape.max_vacancies, shape.top_k)
    idx += dist_size
    stats = flat_obs[:, idx:idx + shape.stats_dim]
    return v_feat, diff_k, dist_k, stats


def build_kmc_action_mask(env: Any, *, max_vacancies: int) -> np.ndarray:
    mask = np.zeros((max_vacancies * 8,), dtype=np.int8)
    try:
        vacancy_count = int(len(env.get_vacancy_array()))
    except Exception:
        vacancy_count = 0

    actual = min(max_vacancies, vacancy_count)
    for vac_idx in range(actual):
        for dir_idx in range(8):
            action = vac_idx * 8 + dir_idx
            try:
                _, _, _, _, moving_type = env._decode_action(action)
                mask[action] = 0 if int(moving_type) == int(env.V_TYPE) else 1
            except Exception:
                mask[action] = 0
    return mask
