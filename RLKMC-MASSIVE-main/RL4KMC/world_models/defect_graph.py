from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DefectGraphObservationShape:
    max_vacancies: int
    max_defects: int
    max_shells: int = 16
    node_feat_dim: int = 4
    stats_dim: int = 10

    @property
    def flat_dim(self) -> int:
        return self.max_vacancies * self.max_defects * self.node_feat_dim + self.max_vacancies * self.max_defects + self.stats_dim


def _to_numpy(array_like: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if array_like is None:
        return np.zeros((0,), dtype=dtype)
    if isinstance(array_like, np.ndarray):
        return array_like.astype(dtype, copy=False)
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(array_like, dtype=dtype)


def _pbc_offsets(points: np.ndarray, center: np.ndarray, box: np.ndarray) -> np.ndarray:
    offsets = points - center[None, :]
    return offsets - np.round(offsets / box[None, :]) * box[None, :]


def _is_bcc_offset(x: int, y: int, z: int) -> bool:
    return (x & 1) == (y & 1) == (z & 1)


@lru_cache(maxsize=32)
def bcc_shell_squared_distances(box_dims: tuple[int, int, int]) -> tuple[int, ...]:
    half_box = tuple(int(dim) // 2 for dim in box_dims)
    shell_sq = set()
    for x in range(-half_box[0], half_box[0] + 1):
        for y in range(-half_box[1], half_box[1] + 1):
            for z in range(-half_box[2], half_box[2] + 1):
                if x == 0 and y == 0 and z == 0:
                    continue
                if _is_bcc_offset(x, y, z):
                    shell_sq.add(x * x + y * y + z * z)
    return tuple(sorted(shell_sq))


def bcc_shell_cutoff_sq(max_shells: int, box_dims: tuple[int, int, int]) -> int:
    shell_sq = bcc_shell_squared_distances(box_dims)
    if max_shells <= 0:
        raise ValueError(f"max_shells must be positive, got {max_shells}")
    if len(shell_sq) < max_shells:
        raise ValueError(f"box {box_dims} only exposes {len(shell_sq)} BCC shells, need {max_shells}")
    return int(shell_sq[max_shells - 1])


def parse_neighbor_order(neighbor_order: Any) -> int | None:
    if neighbor_order is None:
        return None
    if isinstance(neighbor_order, (int, np.integer)):
        order = int(neighbor_order)
    else:
        token = str(neighbor_order).strip().upper()
        if token in {"FULL", "ALL", "NONE"}:
            return None
        if token.endswith("NN"):
            token = token[:-2]
        order = int(token)
    if order <= 0:
        raise ValueError(f"neighbor_order must be positive, got {neighbor_order}")
    return order


def build_defect_graph_observation(
    env: Any,
    *,
    shape: DefectGraphObservationShape,
    share_obs: Any | None = None,
) -> np.ndarray:
    vacancies = _to_numpy(getattr(env, "get_vacancy_array")(), dtype=np.float32).reshape(-1, 3)
    cu_atoms = _to_numpy(getattr(env, "get_cu_array")(), dtype=np.float32).reshape(-1, 3)
    box_int = _to_numpy(getattr(env, "dims"), dtype=np.int32).reshape(3)
    box = box_int.astype(np.float32, copy=False)
    cutoff_sq = float(bcc_shell_cutoff_sq(shape.max_shells, tuple(int(v) for v in box_int.tolist())))

    vacancy_types = np.full((len(vacancies),), float(getattr(env, "V_TYPE", 2)), dtype=np.float32)
    cu_types = np.full((len(cu_atoms),), float(getattr(env, "CU_TYPE", 1)), dtype=np.float32)

    all_pos = np.concatenate([vacancies, cu_atoms], axis=0) if len(cu_atoms) > 0 else vacancies.copy()
    all_types = np.concatenate([vacancy_types, cu_types], axis=0) if len(cu_atoms) > 0 else vacancy_types.copy()

    node_attr = np.zeros((shape.max_vacancies, shape.max_defects, shape.node_feat_dim), dtype=np.float32)
    node_mask = np.zeros((shape.max_vacancies, shape.max_defects), dtype=np.float32)

    actual_vacancies = min(shape.max_vacancies, len(vacancies))
    if all_pos.size > 0:
        for vac_idx in range(actual_vacancies):
            center = vacancies[vac_idx]
            offsets = np.rint(_pbc_offsets(all_pos, center, box)).astype(np.float32, copy=False)
            dist_sq = np.sum(offsets * offsets, axis=-1)
            select = np.flatnonzero(dist_sq <= cutoff_sq + 1e-6)
            if select.size == 0:
                continue
            order = np.argsort(dist_sq[select], kind="stable")
            select = select[order]
            if select.size > shape.max_defects:
                select = select[: shape.max_defects]
            selected_offsets = offsets[select]
            selected_types = all_types[select]
            count = len(select)
            node_attr[vac_idx, :count, :3] = selected_offsets
            node_attr[vac_idx, :count, 3] = selected_types
            node_mask[vac_idx, :count] = 1.0

    stats = np.zeros((shape.stats_dim,), dtype=np.float32)
    share_obs_np = _to_numpy(share_obs, dtype=np.float32).reshape(-1)
    if share_obs_np.size > 0:
        stats[: min(shape.stats_dim, share_obs_np.size)] = share_obs_np[: shape.stats_dim]

    flat = np.concatenate([node_attr.reshape(-1), node_mask.reshape(-1), stats], axis=0)
    return flat.astype(np.float32, copy=False)


def unflatten_defect_graph_observation(
    flat_obs: torch.Tensor,
    *,
    shape: DefectGraphObservationShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if flat_obs.ndim == 1:
        flat_obs = flat_obs.unsqueeze(0)
    batch = flat_obs.shape[0]
    expected = shape.flat_dim
    if flat_obs.shape[-1] != expected:
        raise ValueError(f"expected flat observation dim {expected}, got {flat_obs.shape[-1]}")

    idx = 0
    node_size = shape.max_vacancies * shape.max_defects * shape.node_feat_dim
    mask_size = shape.max_vacancies * shape.max_defects
    node_attr = flat_obs[:, idx:idx + node_size].reshape(batch, shape.max_vacancies, shape.max_defects, shape.node_feat_dim)
    idx += node_size
    node_mask = flat_obs[:, idx:idx + mask_size].reshape(batch, shape.max_vacancies, shape.max_defects)
    idx += mask_size
    stats = flat_obs[:, idx:idx + shape.stats_dim]
    return node_attr, node_mask, stats


class _EdgeAwareBlock(nn.Module):
    def __init__(self, hidden_size: int, edge_cutoff_sq: float | None = None) -> None:
        super().__init__()
        self.edge_cutoff_sq = edge_cutoff_sq
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_proj = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, node_h: torch.Tensor, offsets: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(node_h)
        k = self.k_proj(node_h)
        v = self.v_proj(node_h)
        edge_offset = offsets.unsqueeze(-2) - offsets.unsqueeze(-3)
        edge_bias = self.edge_proj(edge_offset).squeeze(-1)
        scores = torch.einsum("...ih,...jh->...ij", q, k) / (q.shape[-1] ** 0.5)
        scores = scores + edge_bias
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        if self.edge_cutoff_sq is not None:
            edge_dist_sq = (edge_offset * edge_offset).sum(dim=-1)
            pair_mask = pair_mask * (edge_dist_sq <= float(self.edge_cutoff_sq) + 1e-6).to(pair_mask.dtype)
        scores = scores.masked_fill(pair_mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1) * pair_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        update = torch.einsum("...ij,...jh->...ih", attn, v)
        node_h = self.norm1(node_h + self.out_proj(update))
        node_h = self.norm2(node_h + self.ff(node_h))
        return node_h * node_mask.unsqueeze(-1)


class DefectGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 2,
        num_types: int = 3,
        neighbor_order: Any = "2NN",
        lattice_size: tuple[int, int, int] = (40, 40, 40),
    ) -> None:
        super().__init__()
        parsed_neighbor_order = parse_neighbor_order(neighbor_order)
        doubled_lattice = tuple(max(int(dim) * 2, 2) for dim in lattice_size)
        edge_cutoff_sq = (
            float(bcc_shell_cutoff_sq(parsed_neighbor_order, doubled_lattice))
            if parsed_neighbor_order is not None
            else None
        )
        type_dim = max(hidden_size // 4, 8)
        self.type_embedding = nn.Embedding(num_types, type_dim)
        self.offset_proj = nn.Sequential(
            nn.Linear(3, hidden_size - type_dim),
            nn.SiLU(),
            nn.Linear(hidden_size - type_dim, hidden_size - type_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [_EdgeAwareBlock(hidden_size, edge_cutoff_sq=edge_cutoff_sq) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, node_attr: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        offsets = node_attr[..., :3].to(dtype=torch.float32)
        # Normalize integer BCC offsets to [-1, 1] for better MLP conditioning
        # Max offset for 16 shells is ~sqrt(44) ≈ 6.6; use 8.0 as normalization constant
        offsets_norm = offsets / 8.0
        defect_types = node_attr[..., 3].long().clamp(min=0, max=self.type_embedding.num_embeddings - 1)
        offset_h = self.offset_proj(offsets_norm)
        type_h = self.type_embedding(defect_types)
        node_h = self.input_proj(torch.cat([offset_h, type_h], dim=-1))
        node_h = node_h * node_mask.unsqueeze(-1)
        for block in self.blocks:
            node_h = block(node_h, offsets_norm, node_mask)
        pooled = (node_h * node_mask.unsqueeze(-1)).sum(dim=-2) / node_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return self.output_proj(pooled)
