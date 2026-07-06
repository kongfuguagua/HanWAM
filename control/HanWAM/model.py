"""HanWAM: strict two-stage SkyJEPA-style world model for HVAC MPC."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def mlp(sizes: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(din, dout))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


@dataclass
class SequenceLossBreakdown:
    total: torch.Tensor
    latent: torch.Tensor
    prober: torch.Tensor
    regularizer: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor


class TemporalConvEncoder(nn.Module):
    """Small TCN-style encoder for state or action history windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        padding = kernel_size // 2
        convs: list[nn.Module] = []
        for _ in range(max(1, int(layers))):
            convs.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                    nn.SiLU(),
                ]
            )
        self.tcn = nn.Sequential(*convs)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(1)
        if sequence.ndim != 3:
            raise ValueError(f"sequence must have shape [B,H,D], got {tuple(sequence.shape)}")
        x = self.input(sequence)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.output(x[:, -1])


class HanWAM(nn.Module):
    """Strict two-stage WAM.

    Stage 1 trains the latent world model only:
      state history encoder + action encoder + GRU predictor.

    Stage 2 freezes the latent world model and trains only the physical prober.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        physical_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        action_latent_dim: int | None = None,
        tcn_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.physical_dim = int(physical_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_latent_dim = int(action_latent_dim or max(16, latent_dim // 2))

        self.state_encoder = TemporalConvEncoder(self.obs_dim, self.hidden_dim, self.latent_dim, layers=tcn_layers)
        self.action_encoder = mlp([self.action_dim, self.hidden_dim, self.action_latent_dim])
        self.predictor = nn.GRUCell(self.action_latent_dim, self.latent_dim)
        self.prober = mlp([self.latent_dim, self.hidden_dim, self.physical_dim])

    def world_model_parameters(self):
        yield from self.state_encoder.parameters()
        yield from self.action_encoder.parameters()
        yield from self.predictor.parameters()

    def prober_parameters(self):
        yield from self.prober.parameters()

    def freeze_world_model(self) -> None:
        for param in self.world_model_parameters():
            param.requires_grad_(False)
        for param in self.prober_parameters():
            param.requires_grad_(True)

    def freeze_prober(self) -> None:
        for param in self.prober_parameters():
            param.requires_grad_(False)
        for param in self.world_model_parameters():
            param.requires_grad_(True)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs_seq = obs.unsqueeze(1) if obs.ndim == 2 else obs
        if obs_seq.ndim != 3:
            raise ValueError(f"obs must have shape [B,D] or [B,H,D], got {tuple(obs.shape)}")
        return self.state_encoder(obs_seq)

    def predict_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.action_encoder(action), latent)

    def probe(self, latent: torch.Tensor) -> torch.Tensor:
        return self.prober(latent)

    def rollout_latents_from_latent(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        latents = []
        z = latent
        for step in range(actions.shape[1]):
            z = self.predict_latent(z, actions[:, step])
            latents.append(z)
        return torch.stack(latents, dim=1)

    def rollout_from_latent(self, latent: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.rollout_latents_from_latent(latent, actions)
        physical = self.probe(latents.reshape(-1, self.latent_dim)).reshape(
            actions.shape[0],
            actions.shape[1],
            self.physical_dim,
        )
        return latents, physical

    def rollout(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rollout_from_latent(self.encode(obs), actions)

    @staticmethod
    def vicreg_loss(
        latents: torch.Tensor,
        variance_target: float = 1.0,
        variance_eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents.ndim > 2:
            latents = latents.reshape(-1, latents.shape[-1])
        if latents.shape[0] < 2:
            zero = latents.new_tensor(0.0)
            return zero, zero, zero
        centered = latents - latents.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + variance_eps)
        variance = torch.mean(F.relu(float(variance_target) - std))
        cov = centered.T @ centered / max(1, latents.shape[0] - 1)
        cov = cov - torch.diag(torch.diag(cov))
        covariance = (cov * cov).sum() / latents.shape[-1]
        return variance + covariance, variance, covariance

    @staticmethod
    def sigreg_loss(
        latents: torch.Tensor,
        num_projections: int = 64,
        target_std: float = 1.0,
        mean_weight: float = 1.0,
        eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents.ndim > 2:
            latents = latents.reshape(-1, latents.shape[-1])
        if latents.shape[0] < 2:
            zero = latents.new_tensor(0.0)
            return zero, zero, zero
        centered = latents - latents.mean(dim=0, keepdim=True)
        directions = torch.randn(
            latents.shape[-1],
            int(num_projections),
            dtype=latents.dtype,
            device=latents.device,
        )
        directions = F.normalize(directions, dim=0)
        projected = centered @ directions
        std = torch.sqrt(projected.var(dim=0, unbiased=False) + eps)
        projection = torch.mean((std - float(target_std)).pow(2))
        mean = torch.mean(latents.mean(dim=0).pow(2))
        total = projection + float(mean_weight) * mean
        return total, projection, mean

    def latent_sequence_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        future_obs: torch.Tensor,
        horizon_gamma: float = 0.98,
        latent_weight: float = 1.0,
        sigreg_weight: float = 0.05,
        variance_target: float = 1.0,
        sigreg_num_projections: int = 64,
        sigreg_mean_weight: float = 1.0,
    ) -> SequenceLossBreakdown:
        z0 = self.encode(obs)
        pred_latents = self.rollout_latents_from_latent(z0, actions)
        if future_obs.ndim == 4:
            bsz, horizon, history_steps, obs_dim = future_obs.shape
            future_flat = future_obs.reshape(bsz * horizon, history_steps, obs_dim)
        elif future_obs.ndim == 3:
            bsz, horizon, obs_dim = future_obs.shape
            future_flat = future_obs.reshape(bsz * horizon, obs_dim)
        else:
            raise ValueError(f"future_obs must have shape [B,T,D] or [B,T,H,D], got {tuple(future_obs.shape)}")
        target_latents = self.encode(future_flat).reshape(bsz, horizon, self.latent_dim)
        weights = torch.pow(
            torch.as_tensor(float(horizon_gamma), dtype=obs.dtype, device=obs.device),
            torch.arange(horizon, dtype=obs.dtype, device=obs.device),
        ).view(1, horizon, 1)
        weights = weights / weights.mean().clamp_min(1e-6)
        latent = ((pred_latents - target_latents).pow(2) * weights).mean()
        reg_latents = torch.cat(
            [
                z0,
                pred_latents.reshape(bsz * horizon, self.latent_dim),
                target_latents.reshape(bsz * horizon, self.latent_dim),
            ],
            dim=0,
        )
        reg, projection, mean = self.sigreg_loss(
            reg_latents,
            num_projections=sigreg_num_projections,
            target_std=variance_target,
            mean_weight=sigreg_mean_weight,
        )
        total = float(latent_weight) * latent + float(sigreg_weight) * reg
        zero = obs.new_tensor(0.0)
        return SequenceLossBreakdown(total, latent, zero, reg, projection, mean)

    def prober_sequence_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        physical_targets: torch.Tensor,
        physical_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            latents = self.rollout_latents_from_latent(self.encode(obs), actions)
        pred = self.probe(latents.reshape(-1, self.latent_dim)).reshape_as(physical_targets)
        err = (pred - physical_targets).pow(2)
        if physical_weights is not None:
            err = err * physical_weights.view(1, 1, -1)
        return err.mean(), pred


def build_wam_model(model_config: dict) -> HanWAM:
    config = dict(model_config)
    class_name = str(config.pop("class_name", config.pop("model_class", "HanWAM")))
    if class_name != "HanWAM":
        raise ValueError(f"HanWAM checkpoint expected class_name=HanWAM, got {class_name}")
    return HanWAM(**config)
