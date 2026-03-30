from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.dreamer4 import DynamicsWorldModel


def _ensure_rlkmc_path() -> None:
    root = Path(__file__).resolve().parents[2] / "RLKMC-MASSIVE-main"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_rlkmc_path()

from RL4KMC.world_models import (  # noqa: E402
    DefectGraphEncoder,
    DefectGraphObservationShape,
    build_defect_graph_observation,
    unflatten_defect_graph_observation,
)


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


class KMCDynamicsWorldModel(DynamicsWorldModel):
    def __init__(
        self,
        *,
        dim: int,
        dim_latent: int,
        max_vacancies: int = 32,
        max_defects: int = 384,
        max_shells: int = 16,
        node_feat_dim: int = 4,
        stats_dim: int = 10,
        graph_hidden_size: int = 128,
        lattice_size=(40, 40, 40),
        neighbor_order: str | int | None = "2NN",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_latent_tokens", max_vacancies)
        super().__init__(dim=dim, dim_latent=dim_latent, video_tokenizer=None, **kwargs)
        self.max_vacancies = int(max_vacancies)
        self.max_defects = int(max_defects)
        self.node_feat_dim = int(node_feat_dim)
        self.stats_dim = int(stats_dim)
        self.kmc_graph_encoder = KMCGraphEncoder(
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=node_feat_dim,
            stats_dim=stats_dim,
            hidden_size=graph_hidden_size,
            dim_latent=dim_latent,
            lattice_size=lattice_size,
            neighbor_order=neighbor_order,
        )
        self.time_head = nn.Sequential(
            nn.LayerNorm(dim_latent),
            nn.Linear(dim_latent, dim_latent),
            nn.SiLU(),
            nn.Linear(dim_latent, 1),
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(dim_latent),
            nn.Linear(dim_latent, dim_latent),
            nn.SiLU(),
            nn.Linear(dim_latent, 1),
        )
        self.topology_head = nn.Sequential(
            nn.LayerNorm(dim_latent),
            nn.Linear(dim_latent, dim_latent * 2),
            nn.SiLU(),
            nn.Linear(dim_latent * 2, max_defects * (node_feat_dim + 1)),
        )

    def encode_observation(self, observations: torch.Tensor) -> torch.Tensor:
        return self.kmc_graph_encoder(observations)

    def predict_time_delta(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            latents = latents.mean(dim=2)
        if latents.ndim == 4:
            latents = latents.mean(dim=2)
        if latents.ndim == 3:
            latents = latents.mean(dim=1)
        return F.softplus(self.time_head(latents)).squeeze(-1)

    def predict_energy_delta(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            latents = latents.mean(dim=2)
        if latents.ndim == 4:
            latents = latents.mean(dim=2)
        if latents.ndim == 3:
            latents = latents.mean(dim=1)
        return self.energy_head(latents).squeeze(-1)

    def reconstruct_topology(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze_time = False
        if latents.ndim == 3:
            latents = latents.unsqueeze(1)
            squeeze_time = True
        if latents.ndim != 4:
            raise ValueError(f"expected [batch, time, vacancy, dim] or [batch, vacancy, dim], got {tuple(latents.shape)}")
        topo = self.topology_head(latents)
        topo = topo.view(
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            self.max_defects,
            self.node_feat_dim + 1,
        )
        node_attr = topo[..., :self.node_feat_dim]
        node_mask_logits = topo[..., self.node_feat_dim]
        if squeeze_time:
            return node_attr[:, 0], node_mask_logits[:, 0]
        return node_attr, node_mask_logits

    def predict_physics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "delta_t": self.predict_time_delta(latents),
            "energy_delta": self.predict_energy_delta(latents),
            "topology": self.reconstruct_topology(latents),
        }

    def forward(
        self,
        *args: Any,
        observations: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if observations is not None:
            if latents is not None:
                raise ValueError("pass either observations or latents, not both")
            latents = self.encode_observation(observations)
        return super().forward(*args, latents=latents, **kwargs)
