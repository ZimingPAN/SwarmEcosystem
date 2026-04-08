from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, PredictionNetworkMLP, MLP_V2
from .muzero_model_mlp import DynamicsNetwork
from .utils import get_params_mean, renormalize


def _ensure_rlkmc_path() -> None:
    root = Path(__file__).resolve().parents[3] / "RLKMC-MASSIVE-main"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_rlkmc_path()

from RL4KMC.world_models import (  # noqa: E402
    DefectGraphEncoder,
    DefectGraphObservationShape,
    unflatten_defect_graph_observation,
)


class KMCGraphRepresentationNetwork(nn.Module):
    def __init__(
        self,
        *,
        observation_shape: DefectGraphObservationShape,
        latent_state_dim: int,
        graph_hidden_size: int,
        per_vacancy_latent_dim: int,
        neighbor_order: str | int | None,
        lattice_size: SequenceType,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.graph_encoder = DefectGraphEncoder(
            hidden_size=graph_hidden_size,
            output_dim=per_vacancy_latent_dim,
            neighbor_order=neighbor_order,
            lattice_size=tuple(int(v) for v in lattice_size),
        )
        raw_latent_dim = observation_shape.max_vacancies * per_vacancy_latent_dim + observation_shape.stats_dim
        if raw_latent_dim == latent_state_dim:
            self.project = nn.Identity()
        else:
            self.project = nn.Sequential(
                nn.Linear(raw_latent_dim, latent_state_dim),
                nn.LayerNorm(latent_state_dim),
                activation,
            )

    def forward(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent_state, log_total_rate) where log_total_rate = stats[3]."""
        node_attr, node_mask, stats = unflatten_defect_graph_observation(flat_obs, shape=self.observation_shape)
        device = flat_obs.device
        self.graph_encoder = self.graph_encoder.to(device)
        vacancy_latents = self.graph_encoder(
            node_attr.to(device=device, dtype=torch.float32),
            node_mask.to(device=device, dtype=torch.float32),
        )
        flat_latents = vacancy_latents.reshape(vacancy_latents.shape[0], -1)
        stats_f = stats.to(device=device, dtype=torch.float32)
        combined = torch.cat([flat_latents, stats_f], dim=-1)
        latent = self.project(combined)
        log_total_rate = stats_f[:, 3]  # log(Γ_tot) injected by env wrapper
        return latent, log_total_rate


@MODEL_REGISTRY.register("KMCGraphMuZeroModel")
class KMCGraphMuZeroModel(nn.Module):
    def __init__(
        self,
        observation_shape: int,
        action_space_size: int,
        max_vacancies: int = 32,
        max_defects: int = 384,
        max_shells: int = 16,
        node_feat_dim: int = 4,
        stats_dim: int = 10,
        latent_state_dim: int = 256,
        graph_hidden_size: int = 128,
        per_vacancy_latent_dim: int = 16,
        lattice_size: SequenceType = (40, 40, 40),
        neighbor_order: str | int | None = "2NN",
        reward_head_hidden_channels: SequenceType = (128, 128),
        value_head_hidden_channels: SequenceType = (128, 128),
        policy_head_hidden_channels: SequenceType = (128, 128),
        time_head_hidden_channels: SequenceType = (64, 64),
        reward_support_range: SequenceType = (-300.0, 301.0, 1.0),
        value_support_range: SequenceType = (-300.0, 301.0, 1.0),
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        self_supervised_learning_loss: bool = False,
        categorical_distribution: bool = True,
        activation: Optional[nn.Module] = None,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        discrete_action_encoding_type: str = "one_hot",
        norm_type: Optional[str] = "LN",
        res_connection_in_dynamics: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        activation = activation or nn.ReLU(inplace=True)
        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = len(torch.arange(*reward_support_range))
            self.value_support_size = len(torch.arange(*value_support_range))
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

        self.action_space_size = action_space_size
        self.continuous_action_space = False
        self.action_space_dim = 1
        if discrete_action_encoding_type not in ["one_hot", "not_one_hot"]:
            raise ValueError(f"unsupported discrete_action_encoding_type: {discrete_action_encoding_type}")
        self.discrete_action_encoding_type = discrete_action_encoding_type
        self.action_encoding_dim = action_space_size if discrete_action_encoding_type == "one_hot" else 1
        self.latent_state_dim = latent_state_dim
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.res_connection_in_dynamics = res_connection_in_dynamics
        self.observation_shape = DefectGraphObservationShape(
            max_vacancies=max_vacancies,
            max_defects=max_defects,
            max_shells=max_shells,
            node_feat_dim=node_feat_dim,
            stats_dim=stats_dim,
        )
        if int(observation_shape) != self.observation_shape.flat_dim:
            raise ValueError(
                f"observation_shape={observation_shape} does not match flattened KMC dim {self.observation_shape.flat_dim}"
            )

        self.representation_network = KMCGraphRepresentationNetwork(
            observation_shape=self.observation_shape,
            latent_state_dim=latent_state_dim,
            graph_hidden_size=graph_hidden_size,
            per_vacancy_latent_dim=per_vacancy_latent_dim,
            neighbor_order=neighbor_order,
            lattice_size=lattice_size,
            activation=activation,
        )
        self.dynamics_network = DynamicsNetwork(
            action_encoding_dim=self.action_encoding_dim,
            num_channels=self.latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            reward_head_hidden_channels=list(reward_head_hidden_channels),
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type,
            activation=activation,
            res_connection_in_dynamics=self.res_connection_in_dynamics,
        )
        self.prediction_network = PredictionNetworkMLP(
            action_space_size=action_space_size,
            num_channels=latent_state_dim,
            value_head_hidden_channels=list(value_head_hidden_channels),
            policy_head_hidden_channels=list(policy_head_hidden_channels),
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type,
        )
        self.time_head = MLP_V2(
            in_channels=latent_state_dim,
            hidden_channels=list(time_head_hidden_channels),
            out_channels=1,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
        )
        self.latest_time_delta: torch.Tensor | None = None
        self.latest_log_total_rate: torch.Tensor | None = None

        # Zero-init time_head for residual mode:
        # log(E[Δt|s]) = -log(Γ_tot) + residual, where residual starts at 0
        # so initial prediction = 1/Γ_tot (physically correct baseline)
        with torch.no_grad():
            for m in reversed(list(self.time_head.modules())):
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    nn.init.constant_(m.bias, 0.0)
                    nn.init.zeros_(m.weight)
                    break

        if self.self_supervised_learning_loss:
            self.projection_input_dim = latent_state_dim
            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid),
                nn.LayerNorm(self.proj_hid),
                activation,
                nn.Linear(self.proj_hid, self.proj_hid),
                nn.LayerNorm(self.proj_hid),
                activation,
                nn.Linear(self.proj_hid, self.proj_out),
                nn.LayerNorm(self.proj_out),
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.LayerNorm(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        batch_size = obs.size(0)
        latent_state, log_total_rate = self._representation(obs)
        self.latest_log_total_rate = log_total_rate
        policy_logits, value = self._prediction(latent_state)
        # Residual time prediction with gradient isolation:
        # log(E[Δt|s]) = -log(Γ_tot) + residual(latent.detach())
        log_baseline = -log_total_rate  # log(1/Γ_tot)
        residual = self.time_head(latent_state.detach()).squeeze(-1)
        log_time = log_baseline + residual
        self.latest_time_delta = torch.exp(log_time)
        return MZNetworkOutput(
            value=value,
            reward=[0.0 for _ in range(batch_size)],
            policy_logits=policy_logits,
            latent_state=latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
        next_latent_state, reward = self._dynamics(latent_state, action)
        policy_logits, value = self._prediction(next_latent_state)
        # Residual time prediction using stored log_total_rate from initial_inference
        if self.latest_log_total_rate is not None:
            log_baseline = -self.latest_log_total_rate
        else:
            log_baseline = torch.zeros(next_latent_state.shape[0], device=next_latent_state.device)
        residual = self.time_head(next_latent_state.detach()).squeeze(-1)
        log_time = log_baseline + residual
        self.latest_time_delta = torch.exp(log_time)
        return MZNetworkOutput(value=value, reward=reward, policy_logits=policy_logits, latent_state=next_latent_state)

    def predict_time_delta(self, latent_state: torch.Tensor, log_total_rate: torch.Tensor | None = None) -> torch.Tensor:
        """Predict E[Δt | s] using residual mode: exp(-log(Γ_tot) + residual)."""
        residual = self.time_head(latent_state.detach()).squeeze(-1)
        if log_total_rate is not None:
            log_time = -log_total_rate + residual
        elif self.latest_log_total_rate is not None:
            log_time = -self.latest_log_total_rate + residual
        else:
            log_time = residual
        return torch.exp(log_time)

    def predict_log_time_delta(self, latent_state: torch.Tensor, log_total_rate: torch.Tensor | None = None) -> torch.Tensor:
        """Return log-space prediction: -log(Γ_tot) + residual."""
        residual = self.time_head(latent_state.detach()).squeeze(-1)
        if log_total_rate is not None:
            return -log_total_rate + residual
        elif self.latest_log_total_rate is not None:
            return -self.latest_log_total_rate + residual
        return residual

    def _representation(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_state, log_total_rate = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state, log_total_rate

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value = self.prediction_network(latent_state)
        return policy_logits, value

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.continuous_action_space:
            if self.discrete_action_encoding_type == "one_hot":
                action_ids = action.long().unsqueeze(-1) if action.ndim == 1 else action.long()
                action = (
                    torch.zeros(action_ids.shape[0], self.action_space_size, device=action_ids.device).scatter(
                        1, action_ids, 1.0
                    )
                )
            else:
                action = action / self.action_space_size
        state_action_encoding = torch.cat([latent_state, action], dim=1)
        next_latent_state, reward = self.dynamics_network(state_action_encoding)
        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, reward

    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        proj = self.projection(latent_state)
        if with_grad:
            return self.prediction_head(proj)
        return self.prediction_head(proj.detach())

    def get_params_mean(self) -> float:
        return get_params_mean(self)
