from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2] / "RLKMC-MASSIVE-main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL4KMC.world_models import DefectGraphEncoder, DefectGraphObservationShape, unflatten_defect_graph_observation

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency fallback
    linear_sum_assignment = None


NUM_SITE_TYPES = 3
CU_SITE_TYPE = 1
VACANCY_SITE_TYPE = 2
BASE_TEACHER_PATH_SUMMARY_DIM = 18


def teacher_path_summary_dim(horizon_k: int, include_stepwise_features: bool = True) -> int:
    if not include_stepwise_features:
        return BASE_TEACHER_PATH_SUMMARY_DIM
    return BASE_TEACHER_PATH_SUMMARY_DIM + 2 * int(horizon_k)


def macro_duration_baseline_log_tau(global_summary: torch.Tensor, horizon_k: torch.Tensor) -> torch.Tensor:
    horizon = horizon_k.to(device=global_summary.device, dtype=global_summary.dtype).clamp(min=1.0)
    return torch.log(horizon) - global_summary[..., 10]


def _periodic_edge_offsets(positions: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    delta = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    if box.ndim == 1:
        box = box.view(1, 1, 1, 3)
    elif box.ndim == 2:
        box = box.view(box.shape[0], 1, 1, 3)
    else:
        raise ValueError(f"expected box dims with shape [3] or [batch, 3], got {tuple(box.shape)}")
    return delta - torch.round(delta / box) * box


class _PatchEdgeBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
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

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, node_mask: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        offsets = _periodic_edge_offsets(positions, box)
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        scores = torch.einsum("bih,bjh->bij", q, k) / (q.shape[-1] ** 0.5)
        scores = scores + self.edge_proj(offsets).squeeze(-1)
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        scores = scores.masked_fill(pair_mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1) * pair_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        update = torch.einsum("bij,bjh->bih", attn, v)
        hidden = self.norm1(hidden + self.out_proj(update))
        hidden = self.norm2(hidden + self.ff(hidden))
        return hidden * node_mask.unsqueeze(-1)


class KMCGraphEncoder(nn.Module):
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
            raise ValueError(f"expected [batch, time, dim] or [batch, dim], got {tuple(flat_observation.shape)}")

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
        latents = node_latents * (1.0 + stats_scale.unsqueeze(1)) + stats_shift.unsqueeze(1) + stats_token
        latents = latents.reshape(batch, time, self.shape.max_vacancies, -1)
        if squeeze_time:
            return latents[:, 0]
        return latents


class ActivePatchEncoder(nn.Module):
    def __init__(self, *, hidden_size: int, output_dim: int, global_summary_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        type_dim = max(hidden_size // 4, 8)
        numeric_dim = hidden_size - type_dim
        self.type_embedding = nn.Embedding(NUM_SITE_TYPES, type_dim)
        self.numeric_proj = nn.Sequential(
            nn.Linear(5, numeric_dim),
            nn.SiLU(),
            nn.Linear(numeric_dim, numeric_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([_PatchEdgeBlock(hidden_size) for _ in range(num_layers)])
        self.global_proj = nn.Sequential(
            nn.Linear(global_summary_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.site_out = nn.Linear(hidden_size, output_dim)
        self.patch_out = nn.Sequential(
            nn.LayerNorm(output_dim + hidden_size),
            nn.Linear(output_dim + hidden_size, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        *,
        positions: torch.Tensor,
        nearest_vacancy_offset: torch.Tensor,
        reach_depth: torch.Tensor,
        is_start_vacancy: torch.Tensor,
        type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        global_summary: torch.Tensor,
        box_dims: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numeric = torch.cat(
            [
                nearest_vacancy_offset / 8.0,
                reach_depth.unsqueeze(-1),
                is_start_vacancy.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = torch.cat([self.numeric_proj(numeric), self.type_embedding(type_ids)], dim=-1)
        hidden = self.input_proj(hidden) * node_mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, positions, node_mask, box_dims)
        site_latent = self.site_out(hidden) * node_mask.unsqueeze(-1)
        pooled = (site_latent * node_mask.unsqueeze(-1)).sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_token = self.global_proj(global_summary)
        patch_latent = self.patch_out(torch.cat([pooled, global_token], dim=-1))
        return site_latent, patch_latent


class GaussianPathHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, max(input_dim, latent_dim * 2)),
            nn.SiLU(),
            nn.Linear(max(input_dim, latent_dim * 2), latent_dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(x).chunk(2, dim=-1)
        logvar = logvar.clamp(min=-8.0, max=4.0)
        return mu, logvar


class MacroDreamerEditModel(nn.Module):
    def __init__(
        self,
        *,
        max_vacancies: int,
        max_defects: int,
        max_shells: int,
        stats_dim: int,
        lattice_size: tuple[int, int, int],
        neighbor_order: str | int | None,
        dim_latent: int,
        graph_hidden_size: int,
        patch_hidden_size: int,
        patch_latent_dim: int,
        path_latent_dim: int,
        global_summary_dim: int,
        teacher_path_summary_dim: int,
        max_macro_k: int,
    ) -> None:
        super().__init__()
        self.dim_latent = int(dim_latent)
        self.max_vacancies = int(max_vacancies)
        self.max_macro_k = int(max_macro_k)
        self.global_latent_dim = self.max_vacancies * self.dim_latent
        self.global_summary_dim = int(global_summary_dim)
        self.teacher_path_summary_dim = int(teacher_path_summary_dim)
        self.path_latent_dim = int(path_latent_dim)
        self.k_embed = nn.Embedding(self.max_macro_k + 1, 32)
        self.global_encoder = KMCGraphEncoder(
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=4,
            stats_dim=stats_dim,
            hidden_size=graph_hidden_size,
            dim_latent=dim_latent,
            lattice_size=lattice_size,
            neighbor_order=neighbor_order,
        )
        self.patch_encoder = ActivePatchEncoder(
            hidden_size=patch_hidden_size,
            output_dim=patch_latent_dim,
            global_summary_dim=global_summary_dim,
        )
        prior_in = self.global_latent_dim + global_summary_dim + 32
        post_in = self.global_latent_dim * 2 + teacher_path_summary_dim + 32
        self.path_prior = GaussianPathHead(prior_in, path_latent_dim)
        self.path_posterior = GaussianPathHead(post_in, path_latent_dim)
        self.macro_dynamics = nn.Sequential(
            nn.LayerNorm(self.global_latent_dim + path_latent_dim + 32),
            nn.Linear(self.global_latent_dim + path_latent_dim + 32, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, self.global_latent_dim),
        )
        decoder_in = patch_latent_dim + patch_latent_dim + self.global_latent_dim + path_latent_dim + 32
        self.edit_decoder = nn.Sequential(
            nn.LayerNorm(decoder_in),
            nn.Linear(decoder_in, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.change_head = nn.Linear(256, 1)
        self.type_head = nn.Linear(256, NUM_SITE_TYPES)
        self.type_copy_bias = 2.0
        reward_time_in = self.global_latent_dim * 2 + path_latent_dim + global_summary_dim + 32
        self.reward_head = nn.Sequential(
            nn.LayerNorm(reward_time_in),
            nn.Linear(reward_time_in, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.duration_head = nn.Sequential(
            nn.LayerNorm(reward_time_in),
            nn.Linear(reward_time_in, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )
        with torch.no_grad():
            # The duration head predicts a residual around the physics baseline k / Gamma_tot(start).
            self.duration_head[-1].bias[0] = 0.0
            self.duration_head[-1].bias[1] = -2.0

    def encode_global(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.global_encoder(obs)
        return latent.reshape(latent.shape[0], -1)

    def encode_patch(
        self,
        *,
        positions: torch.Tensor,
        nearest_vacancy_offset: torch.Tensor,
        reach_depth: torch.Tensor,
        is_start_vacancy: torch.Tensor,
        type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        global_summary: torch.Tensor,
        box_dims: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patch_encoder(
            positions=positions,
            nearest_vacancy_offset=nearest_vacancy_offset,
            reach_depth=reach_depth,
            is_start_vacancy=is_start_vacancy,
            type_ids=type_ids,
            node_mask=node_mask,
            global_summary=global_summary,
            box_dims=box_dims,
        )

    def _k_embedding(self, horizon_k: torch.Tensor) -> torch.Tensor:
        return self.k_embed(horizon_k.clamp(min=0, max=self.max_macro_k))

    def prior_stats(self, global_latent: torch.Tensor, global_summary: torch.Tensor, horizon_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_emb = self._k_embedding(horizon_k)
        return self.path_prior(torch.cat([global_latent, global_summary, k_emb], dim=-1))

    def posterior_stats(
        self,
        global_latent: torch.Tensor,
        next_global_latent: torch.Tensor,
        teacher_path_summary: torch.Tensor,
        horizon_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_emb = self._k_embedding(horizon_k)
        return self.path_posterior(torch.cat([global_latent, next_global_latent, teacher_path_summary, k_emb], dim=-1))

    def sample_path_latent(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def predict_next_global(self, global_latent: torch.Tensor, path_latent: torch.Tensor, horizon_k: torch.Tensor) -> torch.Tensor:
        k_emb = self._k_embedding(horizon_k)
        delta = self.macro_dynamics(torch.cat([global_latent, path_latent, k_emb], dim=-1))
        return global_latent + 0.1 * delta

    def decode_edit(
        self,
        *,
        site_latent: torch.Tensor,
        patch_latent: torch.Tensor,
        predicted_next_global: torch.Tensor,
        path_latent: torch.Tensor,
        horizon_k: torch.Tensor,
        current_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_emb = self._k_embedding(horizon_k)
        batch, sites, _ = site_latent.shape
        context = torch.cat(
            [
                site_latent,
                patch_latent.unsqueeze(1).expand(batch, sites, -1),
                predicted_next_global.unsqueeze(1).expand(batch, sites, -1),
                path_latent.unsqueeze(1).expand(batch, sites, -1),
                k_emb.unsqueeze(1).expand(batch, sites, -1),
            ],
            dim=-1,
        )
        hidden = self.edit_decoder(context)
        raw_type_logits = self.type_head(hidden)
        return self.change_head(hidden).squeeze(-1), raw_type_logits

    def apply_type_copy_bias(self, raw_type_logits: torch.Tensor, current_types: torch.Tensor) -> torch.Tensor:
        copy_prior = F.one_hot(current_types, num_classes=NUM_SITE_TYPES).float() * self.type_copy_bias
        return raw_type_logits + copy_prior

    def predict_reward_and_duration(
        self,
        global_latent: torch.Tensor,
        predicted_next_global: torch.Tensor,
        path_latent: torch.Tensor,
        global_summary: torch.Tensor,
        horizon_k: torch.Tensor,
        *,
        detach_duration_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_emb = self._k_embedding(horizon_k)
        reward_hidden = torch.cat([global_latent, predicted_next_global, path_latent, global_summary, k_emb], dim=-1)
        reward = self.reward_head(reward_hidden).squeeze(-1)

        if detach_duration_inputs:
            duration_inputs = [
                global_latent.detach(),
                predicted_next_global.detach(),
                path_latent.detach(),
                global_summary,
                k_emb.detach(),
            ]
        else:
            duration_inputs = [global_latent, predicted_next_global, path_latent, global_summary, k_emb]
        duration_hidden = torch.cat(duration_inputs, dim=-1)
        residual_mu, log_sigma = self.duration_head(duration_hidden).chunk(2, dim=-1)
        mu = macro_duration_baseline_log_tau(global_summary, horizon_k).unsqueeze(-1) + residual_mu
        return reward, mu.squeeze(-1), log_sigma.squeeze(-1).clamp(min=-6.0, max=2.0)


def _assignment_slots(quotas: Iterable[int]) -> list[int]:
    slots: list[int] = []
    for type_id, count in enumerate(quotas):
        slots.extend([int(type_id)] * int(count))
    return slots


def _solve_assignment(scores: torch.Tensor, quotas: torch.Tensor) -> torch.Tensor:
    slots = _assignment_slots(quotas.tolist())
    count = len(slots)
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    cost = -scores.detach().cpu().numpy()[:, slots]
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = torch.empty((count,), dtype=torch.long)
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            assigned[row] = int(slots[col])
        return assigned.to(device=scores.device)

    remaining_rows = set(range(count))
    remaining_cols = set(range(count))
    assigned = torch.empty((count,), dtype=torch.long)
    while remaining_rows:
        best_score = None
        best_row = None
        best_col = None
        for row in remaining_rows:
            row_scores = scores[row].detach().cpu()
            for col in remaining_cols:
                score = float(row_scores[slots[col]].item())
                if best_score is None or score > best_score:
                    best_score = score
                    best_row = row
                    best_col = col
        assigned[int(best_row)] = int(slots[int(best_col)])
        remaining_rows.remove(int(best_row))
        remaining_cols.remove(int(best_col))
    return assigned.to(device=scores.device)


def _bcc_hop_distance(pos_a: torch.Tensor, pos_b: torch.Tensor, box: torch.Tensor) -> int:
    delta = pos_a - pos_b
    delta = delta - torch.round(delta / box) * box
    return int(torch.abs(delta).max().item())


def _bcc_hop_distance_matrix(pos_a: torch.Tensor, pos_b: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Vectorized all-pairs BCC hop distance (Chebyshev on PBC lattice).

    Args:
        pos_a: (N, 3) positions of first set.
        pos_b: (M, 3) positions of second set.
        box: (3,) box dimensions.

    Returns:
        (N, M) integer tensor of hop distances.
    """
    delta = pos_a.unsqueeze(1) - pos_b.unsqueeze(0)  # (N, M, 3)
    delta = delta - torch.round(delta / box) * box
    return torch.abs(delta).amax(dim=-1).to(torch.int32)  # (N, M)


def _type_transport_cost(
    *,
    positions: torch.Tensor,
    current_types: torch.Tensor,
    final_types: torch.Tensor,
    node_mask: torch.Tensor,
    box_dims: torch.Tensor,
    tracked_type: int,
) -> float:
    valid_idx = torch.nonzero(node_mask > 0, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return 0.0
    current_type_mask = current_types[valid_idx] == tracked_type
    final_type_mask = final_types[valid_idx] == tracked_type
    removed_nodes = valid_idx[current_type_mask & ~final_type_mask]
    added_nodes = valid_idx[~current_type_mask & final_type_mask]
    if removed_nodes.numel() != added_nodes.numel():
        return float("inf")
    if removed_nodes.numel() == 0:
        return 0.0
    costs = _bcc_hop_distance_matrix(
        positions[removed_nodes], positions[added_nodes], box_dims
    ).float()
    quotas = torch.ones((added_nodes.numel(),), dtype=torch.long)
    assigned = _solve_assignment(-costs, quotas)
    total = 0.0
    for row, col in enumerate(assigned.tolist()):
        total += float(costs[row, col].item())
    return total


@torch.no_grad()
def project_types_by_inventory(
    *,
    current_types: torch.Tensor,
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    node_mask: torch.Tensor,
    positions: torch.Tensor,
    box_dims: torch.Tensor,
    horizon_k: torch.Tensor,
    max_changed_sites: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    final_types = current_types.clone()
    changed_mask = torch.zeros_like(node_mask, dtype=torch.float32)
    transport_costs = torch.zeros((current_types.shape[0],), dtype=torch.float32, device=current_types.device)
    violations = torch.zeros((current_types.shape[0],), dtype=torch.float32, device=current_types.device)
    batch = current_types.shape[0]
    for batch_idx in range(batch):
        valid_idx = torch.nonzero(node_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        probs = torch.sigmoid(change_logits[batch_idx, valid_idx])
        local_current_types = current_types[batch_idx, valid_idx]
        local_type_probs = torch.softmax(type_logits[batch_idx, valid_idx], dim=-1)
        predicted_types = type_logits[batch_idx, valid_idx].argmax(dim=-1)
        local_current_conf = local_type_probs.gather(1, local_current_types.unsqueeze(-1)).squeeze(-1)
        type_change_score = 1.0 - local_current_conf
        typed_change_mass = probs * type_change_score
        combined_scores = 0.5 * probs + 0.5 * type_change_score
        predicted_type_change_count = int((predicted_types != local_current_types).sum().item())

        # Pre-compute pair candidates and distance matrices for each atom type
        pair_data: dict[int, tuple[list[int], list[int], torch.Tensor | None]] = {}
        available_pair_budget = 0
        box_b = box_dims[batch_idx]
        for atom_type in (0, CU_SITE_TYPE):
            vacancy_candidates = torch.nonzero(
                (local_current_types == VACANCY_SITE_TYPE) & (predicted_types == atom_type), as_tuple=False
            ).squeeze(-1)
            atom_candidates = torch.nonzero(
                (local_current_types == atom_type) & (predicted_types == VACANCY_SITE_TYPE), as_tuple=False
            ).squeeze(-1)
            n_vac_full = vacancy_candidates.numel()
            n_atom_full = atom_candidates.numel()
            available_pair_budget += 2 * min(n_vac_full, n_atom_full)
            vac_sorted = sorted(vacancy_candidates.cpu().tolist(), key=lambda idx: float(combined_scores[idx].item()), reverse=True)[:16]
            atom_sorted = sorted(atom_candidates.cpu().tolist(), key=lambda idx: float(combined_scores[idx].item()), reverse=True)[:64]
            # Pre-compute distance matrix (V, A) for this atom type
            if vac_sorted and atom_sorted:
                vac_global = valid_idx[torch.tensor(vac_sorted, dtype=torch.long)]
                atom_global = valid_idx[torch.tensor(atom_sorted, dtype=torch.long)]
                dm = _bcc_hop_distance_matrix(
                    positions[batch_idx, vac_global],
                    positions[batch_idx, atom_global],
                    box_b,
                )
                # Pre-compute score matrix (V, A)
                vac_scores = combined_scores[torch.tensor(vac_sorted)]  # (V,)
                atom_scores_t = combined_scores[torch.tensor(atom_sorted)]  # (A,)
                score_mat = vac_scores.unsqueeze(1) + atom_scores_t.unsqueeze(0) - 0.05 * dm.float()  # (V, A)
            else:
                dm = None
                score_mat = None
            pair_data[atom_type] = (vac_sorted, atom_sorted, dm, score_mat)

        budget = max(int(torch.ceil(typed_change_mass.sum()).item()), predicted_type_change_count)
        if budget <= 0 and float(typed_change_mass.max().item()) > 0.25:
            budget = 1
        if budget == 1 and float(typed_change_mass.max().item()) > 0.25:
            budget = 2
        budget = max(0, min(int(max_changed_sites), int(valid_idx.numel()), int(available_pair_budget), budget))
        if budget % 2 == 1:
            budget -= 1
        initial_budget = budget
        if budget <= 0:
            continue
        accepted = False
        while budget > 0:
            remaining_budget = budget
            remaining_transport_budget = max(1, int(float(horizon_k[batch_idx].item())))
            used_vac: dict[int, set[int]] = {at: set() for at in pair_data}
            used_atom: dict[int, set[int]] = {at: set() for at in pair_data}
            chosen_pairs: list[tuple[int, int, int]] = []
            while remaining_budget >= 2 and remaining_transport_budget > 0:
                best_pair_score = None
                best_pair_cost = None
                best_type = None
                for atom_type, (vac_sorted, atom_sorted, dm, score_mat) in pair_data.items():
                    if dm is None:
                        continue
                    # Build a masked copy of score_mat for valid candidates
                    masked_scores = score_mat.clone()
                    for vi in used_vac[atom_type]:
                        masked_scores[vi, :] = -float('inf')
                    for ai in used_atom[atom_type]:
                        masked_scores[:, ai] = -float('inf')
                    # Mask out pairs exceeding transport budget or with zero cost
                    invalid = (dm <= 0) | (dm > remaining_transport_budget)
                    masked_scores[invalid] = -float('inf')
                    # Find best pair in this atom_type
                    best_flat = masked_scores.argmax().item()
                    best_vi = best_flat // masked_scores.shape[1]
                    best_ai = best_flat % masked_scores.shape[1]
                    pair_score = float(masked_scores[best_vi, best_ai].item())
                    if pair_score == -float('inf'):
                        continue
                    pair_cost = int(dm[best_vi, best_ai].item())
                    if (
                        best_pair_score is None
                        or pair_score > best_pair_score
                        or (
                            abs(pair_score - best_pair_score) < 1e-6
                            and best_pair_cost is not None
                            and pair_cost < best_pair_cost
                        )
                    ):
                        best_pair_score = pair_score
                        best_pair_cost = pair_cost
                        best_type = (best_vi, best_ai, atom_type, pair_cost, vac_sorted[best_vi], atom_sorted[best_ai])
                if best_type is None:
                    break
                vi_idx, ai_idx, atom_type, pair_cost, vac_choice, atom_choice = best_type
                used_vac[atom_type].add(vi_idx)
                used_atom[atom_type].add(ai_idx)
                chosen_pairs.append((vac_choice, atom_choice, atom_type))
                remaining_budget -= 2
                remaining_transport_budget -= pair_cost
            proposed = current_types[batch_idx].clone()
            for vac_choice, atom_choice, atom_type in chosen_pairs:
                vac_idx = int(valid_idx[vac_choice].item())
                atom_idx = int(valid_idx[atom_choice].item())
                proposed[vac_idx] = int(atom_type)
                proposed[atom_idx] = VACANCY_SITE_TYPE
            if not chosen_pairs:
                budget -= 2
                continue
            vacancy_transport = _type_transport_cost(
                positions=positions[batch_idx],
                current_types=current_types[batch_idx],
                final_types=proposed,
                node_mask=node_mask[batch_idx],
                box_dims=box_dims[batch_idx],
                tracked_type=VACANCY_SITE_TYPE,
            )
            cu_transport = _type_transport_cost(
                positions=positions[batch_idx],
                current_types=current_types[batch_idx],
                final_types=proposed,
                node_mask=node_mask[batch_idx],
                box_dims=box_dims[batch_idx],
                tracked_type=CU_SITE_TYPE,
            )
            horizon = float(horizon_k[batch_idx].item())
            total_transport = vacancy_transport + cu_transport
            if vacancy_transport <= horizon and cu_transport <= horizon and total_transport <= 2.0 * horizon:
                final_types[batch_idx] = proposed
                changed_mask[batch_idx] = (proposed != current_types[batch_idx]).float() * node_mask[batch_idx]
                transport_costs[batch_idx] = float(total_transport)
                accepted = True
                break
            budget -= 1
        if not accepted:
            transport_costs[batch_idx] = _type_transport_cost(
                positions=positions[batch_idx],
                current_types=current_types[batch_idx],
                final_types=current_types[batch_idx],
                node_mask=node_mask[batch_idx],
                box_dims=box_dims[batch_idx],
                tracked_type=VACANCY_SITE_TYPE,
            )
            if initial_budget > 0:
                violations[batch_idx] = 1.0
    return final_types, changed_mask, transport_costs, violations


def kl_divergence_diag_gaussian(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
    ).sum(dim=-1)


def lognormal_nll(target: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    safe_target = target.clamp(min=1e-10)
    sigma = torch.exp(log_sigma)
    log_target = torch.log(safe_target)
    return log_sigma + 0.5 * ((log_target - mu) / sigma) ** 2