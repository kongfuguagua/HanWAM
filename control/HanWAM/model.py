"""HanWAM E007 block world model and controller prober."""
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
    """Small TCN-style encoder for a time sequence."""

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
        if sequence.ndim == 4:
            bsz, blocks, frames, dim = sequence.shape
            sequence = sequence.reshape(bsz, blocks * frames, dim)
        if sequence.ndim != 3:
            raise ValueError(f"sequence must have shape [B,T,D] or [B,L,T,D], got {tuple(sequence.shape)}")
        x = self.input(sequence)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.output(x[:, -1])


class HanWAMWorldModel(nn.Module):
    """Stage-1-only latent dynamics model with no physical prober."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        action_latent_dim: int | None = None,
        tcn_layers: int = 2,
        frames_per_block: int = 12,
        history_blocks: int = 3,
        future_blocks: int = 5,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_latent_dim = int(action_latent_dim or max(16, latent_dim // 2))
        self.frames_per_block = int(frames_per_block)
        self.history_blocks = int(history_blocks)
        self.future_blocks = int(future_blocks)

        self.obs_encoder = TemporalConvEncoder(self.obs_dim, self.hidden_dim, self.latent_dim, layers=tcn_layers)
        self.act_history_encoder = TemporalConvEncoder(
            self.action_dim,
            self.hidden_dim,
            self.action_latent_dim,
            layers=tcn_layers,
        )
        self.context_fusion = mlp([self.latent_dim + self.action_latent_dim, self.hidden_dim, self.latent_dim])
        self.action_block_encoder = TemporalConvEncoder(
            self.action_dim,
            self.hidden_dim,
            self.action_latent_dim,
            layers=tcn_layers,
        )
        self.predictor = nn.GRUCell(self.action_latent_dim, self.latent_dim)

    def world_model_parameters(self):
        yield from self.parameters()

    def encode_obs_block(self, obs_block: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(obs_block)

    def encode_context(self, obs_history_blocks: torch.Tensor, act_history_blocks: torch.Tensor) -> torch.Tensor:
        z_obs = self.obs_encoder(obs_history_blocks)
        z_act = self.act_history_encoder(act_history_blocks)
        return self.context_fusion(torch.cat([z_obs, z_act], dim=-1))

    def rollout_latents_from_context(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode_context(obs_history_blocks, act_history_blocks)
        latents = []
        for block_idx in range(future_act_blocks.shape[1]):
            action_latent = self.action_block_encoder(future_act_blocks[:, block_idx])
            z = self.predictor(action_latent, z)
            latents.append(z)
        return torch.stack(latents, dim=1)

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
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
        target_obs_blocks: torch.Tensor,
        horizon_gamma: float = 0.98,
        latent_weight: float = 1.0,
        sigreg_weight: float = 0.05,
        variance_target: float = 1.0,
        sigreg_num_projections: int = 64,
        sigreg_mean_weight: float = 1.0,
    ) -> SequenceLossBreakdown:
        z0 = self.encode_context(obs_history_blocks, act_history_blocks)
        pred_latents = self.rollout_latents_from_context(obs_history_blocks, act_history_blocks, future_act_blocks)
        bsz, future_blocks, frames, obs_dim = target_obs_blocks.shape
        target_latents = self.encode_obs_block(target_obs_blocks.reshape(bsz * future_blocks, frames, obs_dim)).reshape(
            bsz,
            future_blocks,
            self.latent_dim,
        )
        weights = torch.pow(
            torch.as_tensor(float(horizon_gamma), dtype=obs_history_blocks.dtype, device=obs_history_blocks.device),
            torch.arange(future_blocks, dtype=obs_history_blocks.dtype, device=obs_history_blocks.device),
        ).view(1, future_blocks, 1)
        weights = weights / weights.mean().clamp_min(1e-6)
        latent = ((pred_latents - target_latents).pow(2) * weights).mean()
        reg_latents = torch.cat(
            [
                z0,
                pred_latents.reshape(bsz * future_blocks, self.latent_dim),
                target_latents.reshape(bsz * future_blocks, self.latent_dim),
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
        zero = obs_history_blocks.new_tensor(0.0)
        return SequenceLossBreakdown(total, latent, zero, reg, projection, mean)


class HanWAMControllerModel(nn.Module):
    """Stage-2 controller model: frozen world model plus physical prober."""

    def __init__(
        self,
        world_model: HanWAMWorldModel,
        physical_dim: int = 2,
        hidden_dim: int | None = None,
        prober_action_scale: float = 1.0,
    ):
        super().__init__()
        self.world_model = world_model
        self.physical_dim = int(physical_dim)
        self.prober_action_scale = float(prober_action_scale)
        hidden = int(hidden_dim or world_model.hidden_dim)
        self.prober = mlp([world_model.latent_dim + world_model.action_dim, hidden, hidden, self.physical_dim])

    @property
    def obs_dim(self) -> int:
        return self.world_model.obs_dim

    @property
    def action_dim(self) -> int:
        return self.world_model.action_dim

    @property
    def latent_dim(self) -> int:
        return self.world_model.latent_dim

    @property
    def frames_per_block(self) -> int:
        return self.world_model.frames_per_block

    @property
    def history_blocks(self) -> int:
        return self.world_model.history_blocks

    @property
    def future_blocks(self) -> int:
        return self.world_model.future_blocks

    def world_model_parameters(self):
        yield from self.world_model.parameters()

    def prober_parameters(self):
        yield from self.prober.parameters()

    def freeze_world_model(self) -> None:
        for param in self.world_model_parameters():
            param.requires_grad_(False)
        for param in self.prober_parameters():
            param.requires_grad_(True)

    def encode_context(self, obs_history_blocks: torch.Tensor, act_history_blocks: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode_context(obs_history_blocks, act_history_blocks)

    def rollout_latents_from_context(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> torch.Tensor:
        return self.world_model.rollout_latents_from_context(obs_history_blocks, act_history_blocks, future_act_blocks)

    def probe_blocks(self, block_latents: torch.Tensor, future_act_blocks: torch.Tensor) -> torch.Tensor:
        bsz, future_blocks, frames, action_dim = future_act_blocks.shape
        latent_expanded = block_latents.unsqueeze(2).expand(bsz, future_blocks, frames, self.latent_dim)
        action_inputs = future_act_blocks * float(self.prober_action_scale)
        inputs = torch.cat([latent_expanded, action_inputs], dim=-1)
        return self.prober(inputs.reshape(bsz * future_blocks * frames, self.latent_dim + action_dim)).reshape(
            bsz,
            future_blocks * frames,
            self.physical_dim,
        )

    def rollout(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.rollout_latents_from_context(obs_history_blocks, act_history_blocks, future_act_blocks)
        physical = self.probe_blocks(latents, future_act_blocks)
        return latents, physical

    def prober_sequence_loss(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
        physical_targets: torch.Tensor,
        physical_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            latents = self.rollout_latents_from_context(obs_history_blocks, act_history_blocks, future_act_blocks)
        pred = self.probe_blocks(latents, future_act_blocks)
        err = (pred - physical_targets).pow(2)
        if physical_weights is not None:
            err = err * physical_weights.view(1, 1, -1)
        return err.mean(), pred


class HanWAMPhysicsGuidedControllerModel(HanWAMControllerModel):
    """Stage-2 prober with a small differentiable cooling/power mechanism."""

    def __init__(
        self,
        world_model: HanWAMWorldModel,
        physical_dim: int = 2,
        hidden_dim: int | None = None,
        prober_action_scale: float = 1.0,
        action_mean: list[float] | None = None,
        action_std: list[float] | None = None,
        physical_mean: list[float] | None = None,
        physical_std: list[float] | None = None,
        freq_on_threshold_hz: float = 15.0,
        freq_max_hz: float = 80.0,
        fan_max: float = 900.0,
        eev_min: float = 69.0,
        eev_max: float = 480.0,
        eev_width_min: float = 45.0,
        eev_width_max: float = 260.0,
        eev_effect_floor: float = 0.55,
        passive_delta_scale: float = 0.035,
        cooling_delta_scale: float = 0.055,
        residual_delta_scale: float = 0.006,
        compressor_energy_scale: float = 0.0030,
        fan_energy_scale: float = 0.00045,
        residual_energy_scale: float = 0.00008,
        compressor_transition_hz: float = 4.0,
        fan_effect_floor: float = 0.20,
        freq_effect_floor: float = 0.08,
        mechanism_context_blend: float = 0.0,
        passive_nonnegative: bool = False,
    ):
        nn.Module.__init__(self)
        self.world_model = world_model
        self.physical_dim = int(physical_dim)
        if self.physical_dim != 2:
            raise ValueError("HanWAMPhysicsGuidedControllerModel currently expects physical_dim=2")
        self.prober_action_scale = float(prober_action_scale)
        hidden = int(hidden_dim or world_model.hidden_dim)
        self.prober = nn.ModuleDict(
            {
                "mechanism": mlp([world_model.latent_dim, hidden, hidden, 6]),
                "residual": mlp([world_model.latent_dim + world_model.action_dim, hidden, hidden, self.physical_dim]),
            }
        )
        action_mean_t = torch.as_tensor(action_mean if action_mean is not None else [0.0, 0.0, 0.0], dtype=torch.float32)
        action_std_t = torch.as_tensor(action_std if action_std is not None else [1.0, 1.0, 1.0], dtype=torch.float32)
        physical_mean_t = torch.as_tensor(physical_mean if physical_mean is not None else [0.0, 0.0], dtype=torch.float32)
        physical_std_t = torch.as_tensor(physical_std if physical_std is not None else [1.0, 1.0], dtype=torch.float32)
        self.register_buffer("action_mean", action_mean_t.view(1, 1, -1))
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6).view(1, 1, -1))
        self.register_buffer("physical_mean", physical_mean_t.view(1, 1, -1))
        self.register_buffer("physical_std", physical_std_t.clamp_min(1e-6).view(1, 1, -1))
        self.freq_on_threshold_hz = float(freq_on_threshold_hz)
        self.freq_max_hz = float(freq_max_hz)
        self.fan_max = float(fan_max)
        self.eev_min = float(eev_min)
        self.eev_max = float(eev_max)
        self.eev_width_min = float(eev_width_min)
        self.eev_width_max = float(eev_width_max)
        self.eev_effect_floor = float(eev_effect_floor)
        self.passive_delta_scale = float(passive_delta_scale)
        self.cooling_delta_scale = float(cooling_delta_scale)
        self.residual_delta_scale = float(residual_delta_scale)
        self.compressor_energy_scale = float(compressor_energy_scale)
        self.fan_energy_scale = float(fan_energy_scale)
        self.residual_energy_scale = float(residual_energy_scale)
        self.compressor_transition_hz = float(compressor_transition_hz)
        self.fan_effect_floor = float(fan_effect_floor)
        self.freq_effect_floor = float(freq_effect_floor)
        self.mechanism_context_blend = min(max(float(mechanism_context_blend), 0.0), 1.0)
        self.passive_nonnegative = bool(passive_nonnegative)

    def _probe_blocks_impl(
        self,
        block_latents: torch.Tensor,
        future_act_blocks: torch.Tensor,
        context_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, future_blocks, frames, action_dim = future_act_blocks.shape
        latent_expanded = block_latents.unsqueeze(2).expand(bsz, future_blocks, frames, self.latent_dim)
        latent_flat = latent_expanded.reshape(bsz * future_blocks * frames, self.latent_dim)
        mechanism_latent = latent_expanded
        if context_latent is not None and self.mechanism_context_blend > 0.0:
            context_expanded = context_latent.view(bsz, 1, 1, self.latent_dim).expand_as(latent_expanded)
            blend = float(self.mechanism_context_blend)
            mechanism_latent = (1.0 - blend) * latent_expanded + blend * context_expanded
        mechanism_flat = mechanism_latent.reshape(bsz * future_blocks * frames, self.latent_dim)
        action_n = future_act_blocks.reshape(bsz * future_blocks * frames, action_dim)
        action_raw = action_n.view(bsz, future_blocks * frames, action_dim) * self.action_std + self.action_mean
        action_raw = action_raw.reshape(bsz * future_blocks * frames, action_dim)
        freq = action_raw[:, 0].clamp_min(0.0)
        eev = action_raw[:, 1]
        fan = action_raw[:, 2].clamp_min(0.0)

        mech = self.prober["mechanism"](mechanism_flat)
        residual_in = torch.cat([latent_flat, action_n * float(self.prober_action_scale)], dim=-1)
        residual = torch.tanh(self.prober["residual"](residual_in))

        threshold = freq.new_tensor(self.freq_on_threshold_hz)
        freq_span = max(self.freq_max_hz - self.freq_on_threshold_hz, 1e-6)
        compressor_on = torch.sigmoid((freq - threshold) / max(self.compressor_transition_hz, 1e-6))
        freq_level = ((freq - threshold) / freq_span).clamp(0.0, 1.0)
        freq_effect = self.freq_effect_floor + (1.0 - self.freq_effect_floor) * freq_level.pow(1.25)
        fan_level = (fan / max(self.fan_max, 1e-6)).clamp(0.0, 1.0)
        fan_effect = self.fan_effect_floor + (1.0 - self.fan_effect_floor) * fan_level.pow(0.75)

        eev_center = self.eev_min + (self.eev_max - self.eev_min) * torch.sigmoid(mech[:, 2])
        eev_width = self.eev_width_min + (self.eev_width_max - self.eev_width_min) * torch.sigmoid(mech[:, 3])
        eev_bell = torch.exp(-0.5 * ((eev - eev_center) / eev_width.clamp_min(1.0)).pow(2))
        eev_effect = self.eev_effect_floor + (1.0 - self.eev_effect_floor) * eev_bell

        if self.passive_nonnegative:
            passive_delta = self.passive_delta_scale * torch.sigmoid(mech[:, 0])
        else:
            passive_delta = self.passive_delta_scale * torch.tanh(mech[:, 0])
        cooling_capacity = self.cooling_delta_scale * torch.sigmoid(mech[:, 1])
        cooling_delta = compressor_on * cooling_capacity * freq_effect * fan_effect * eev_effect
        temp_delta = passive_delta - cooling_delta + self.residual_delta_scale * residual[:, 0]

        comp_mag = self.compressor_energy_scale * torch.nn.functional.softplus(mech[:, 4])
        fan_mag = self.fan_energy_scale * torch.nn.functional.softplus(mech[:, 5])
        compressor_energy = compressor_on * comp_mag * freq_effect
        fan_energy = fan_mag * fan_level.pow(2.0)
        energy_delta = torch.clamp(
            compressor_energy + fan_energy + self.residual_energy_scale * residual[:, 1],
            min=0.0,
        )
        physical_raw = torch.stack([temp_delta, energy_delta], dim=-1).view(bsz, future_blocks * frames, self.physical_dim)
        return (physical_raw - self.physical_mean) / self.physical_std

    def probe_blocks(self, block_latents: torch.Tensor, future_act_blocks: torch.Tensor) -> torch.Tensor:
        return self._probe_blocks_impl(block_latents, future_act_blocks)

    def rollout(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_latent = self.encode_context(obs_history_blocks, act_history_blocks)
        z = context_latent
        latents = []
        for block_idx in range(future_act_blocks.shape[1]):
            action_latent = self.world_model.action_block_encoder(future_act_blocks[:, block_idx])
            z = self.world_model.predictor(action_latent, z)
            latents.append(z)
        block_latents = torch.stack(latents, dim=1)
        physical = self._probe_blocks_impl(block_latents, future_act_blocks, context_latent=context_latent)
        return block_latents, physical

    def prober_sequence_loss(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
        physical_targets: torch.Tensor,
        physical_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            context_latent = self.encode_context(obs_history_blocks, act_history_blocks)
            z = context_latent
            latents = []
            for block_idx in range(future_act_blocks.shape[1]):
                action_latent = self.world_model.action_block_encoder(future_act_blocks[:, block_idx])
                z = self.world_model.predictor(action_latent, z)
                latents.append(z)
            block_latents = torch.stack(latents, dim=1)
        pred = self._probe_blocks_impl(block_latents, future_act_blocks, context_latent=context_latent)
        err = (pred - physical_targets).pow(2)
        if physical_weights is not None:
            err = err * physical_weights.view(1, 1, -1)
        return err.mean(), pred


class HanWAMHardMechanismControllerModel(HanWAMControllerModel):
    """Stage-2 state prober whose differentiable heat-balance mechanism computes physical deltas."""

    def __init__(
        self,
        world_model: HanWAMWorldModel,
        physical_dim: int = 2,
        hidden_dim: int | None = None,
        prober_action_scale: float = 0.0,
        action_mean: list[float] | None = None,
        action_std: list[float] | None = None,
        physical_mean: list[float] | None = None,
        physical_std: list[float] | None = None,
        obs_mean: list[float] | None = None,
        obs_std: list[float] | None = None,
        t_in_obs_index: int = 7,
        t_out_obs_index: int = 8,
        freq_on_threshold_hz: float = 15.0,
        freq_max_hz: float = 80.0,
        fan_max: float = 900.0,
        eev_min: float = 69.0,
        eev_max: float = 480.0,
        eev_width_min: float = 55.0,
        eev_width_max: float = 260.0,
        eev_effect_floor: float = 0.75,
        ua_delta_scale: float = 0.0012,
        internal_delta_scale: float = 0.0020,
        cooling_delta_scale: float = 0.070,
        compressor_energy_scale: float = 1.0,
        fan_energy_scale: float = 0.00045,
        energy_model: str = "cooling_cop",
        energy_step_seconds: float = 5.0,
        energy_power_intercept_w: float = 329.87,
        energy_power_linear_w_per_hz: float = 9.0034,
        energy_power_quadratic_w_per_hz2: float = 0.04494,
        compressor_transition_hz: float = 2.0,
        fan_effect_floor: float = 0.05,
        freq_effect_floor: float = 0.08,
        cop_min: float = 1.5,
        cop_max: float = 5.0,
        mechanism_context_blend: float = 1.0,
        temperature_mechanism: str = "heat_balance",
        baseline_drift_scale_c: float = 0.04,
        cooling_history_seconds: float = 180.0,
        cooling_freq_exponent: float = 1.10,
        cooling_fan_exponent: float = 0.60,
        cooling_fan_reference: float = 750.0,
        cooling_eev_reference: float = 240.0,
        cooling_eev_range: float = 170.0,
        cooling_eev_gain_scale: float = 0.12,
        cooling_eev_effect_min: float = 0.85,
        cooling_eev_effect_max: float = 1.15,
        cooling_lag_enabled: bool = False,
        cooling_lag_alpha_min: float = 0.70,
        cooling_lag_alpha_max: float = 0.985,
        energy_eev_correction_mode: str = "signed_linear",
        energy_eev_correction_scale_w: float = 0.0,
        energy_eev_correction_min_w: float = 50.0,
        energy_eev_correction_max_w: float = 150.0,
        energy_eev_anchor: float = 240.0,
        energy_eev_range: float = 170.0,
    ):
        nn.Module.__init__(self)
        self.world_model = world_model
        self.physical_dim = int(physical_dim)
        if self.physical_dim != 2:
            raise ValueError("HanWAMHardMechanismControllerModel currently expects physical_dim=2")
        self.prober_action_scale = float(prober_action_scale)
        hidden = int(hidden_dim or world_model.hidden_dim)
        self.temperature_mechanism = str(temperature_mechanism)
        if self.temperature_mechanism not in {"heat_balance", "lagged_incremental_cooling"}:
            raise ValueError(
                "temperature_mechanism must be 'heat_balance' or "
                f"'lagged_incremental_cooling', got {self.temperature_mechanism!r}"
            )
        base_mechanism_dim = 6 if self.temperature_mechanism == "heat_balance" else 3
        self.cooling_lag_enabled = bool(cooling_lag_enabled)
        self.energy_eev_correction_mode = str(energy_eev_correction_mode)
        if self.energy_eev_correction_mode not in {"signed_linear", "positive_offset"}:
            raise ValueError(
                "energy_eev_correction_mode must be 'signed_linear' or "
                f"'positive_offset', got {self.energy_eev_correction_mode!r}"
            )
        self.energy_eev_correction_scale_w = float(energy_eev_correction_scale_w)
        self.energy_eev_correction_enabled = (
            self.energy_eev_correction_mode == "positive_offset"
            or abs(self.energy_eev_correction_scale_w) > 0.0
        )
        self.cooling_lag_index = (
            base_mechanism_dim if self.cooling_lag_enabled and self.temperature_mechanism == "heat_balance" else None
        )
        self.energy_eev_correction_index = (
            base_mechanism_dim + int(self.cooling_lag_index is not None)
            if self.energy_eev_correction_enabled
            else None
        )
        self.mechanism_dim = (
            base_mechanism_dim
            + int(self.cooling_lag_index is not None)
            + int(self.energy_eev_correction_enabled)
        )
        self.prober = nn.ModuleDict({"mechanism": mlp([world_model.latent_dim, hidden, hidden, self.mechanism_dim])})

        action_mean_t = torch.as_tensor(action_mean if action_mean is not None else [0.0, 0.0, 0.0], dtype=torch.float32)
        action_std_t = torch.as_tensor(action_std if action_std is not None else [1.0, 1.0, 1.0], dtype=torch.float32)
        physical_mean_t = torch.as_tensor(physical_mean if physical_mean is not None else [0.0, 0.0], dtype=torch.float32)
        physical_std_t = torch.as_tensor(physical_std if physical_std is not None else [1.0, 1.0], dtype=torch.float32)
        obs_mean_t = torch.as_tensor(
            obs_mean if obs_mean is not None else [0.0] * world_model.obs_dim,
            dtype=torch.float32,
        )
        obs_std_t = torch.as_tensor(
            obs_std if obs_std is not None else [1.0] * world_model.obs_dim,
            dtype=torch.float32,
        )
        self.register_buffer("action_mean", action_mean_t.view(1, 1, -1))
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6).view(1, 1, -1))
        self.register_buffer("physical_mean", physical_mean_t.view(1, 1, -1))
        self.register_buffer("physical_std", physical_std_t.clamp_min(1e-6).view(1, 1, -1))
        self.register_buffer("obs_mean", obs_mean_t.view(1, -1))
        self.register_buffer("obs_std", obs_std_t.clamp_min(1e-6).view(1, -1))

        self.t_in_obs_index = int(t_in_obs_index)
        self.t_out_obs_index = int(t_out_obs_index)
        self.freq_on_threshold_hz = float(freq_on_threshold_hz)
        self.freq_max_hz = float(freq_max_hz)
        self.fan_max = float(fan_max)
        self.eev_min = float(eev_min)
        self.eev_max = float(eev_max)
        self.eev_width_min = float(eev_width_min)
        self.eev_width_max = float(eev_width_max)
        self.eev_effect_floor = float(eev_effect_floor)
        self.ua_delta_scale = float(ua_delta_scale)
        self.internal_delta_scale = float(internal_delta_scale)
        self.cooling_delta_scale = float(cooling_delta_scale)
        self.compressor_energy_scale = float(compressor_energy_scale)
        self.fan_energy_scale = float(fan_energy_scale)
        self.energy_model = str(energy_model)
        if self.energy_model not in {"cooling_cop", "frequency_polynomial"}:
            raise ValueError(
                "energy_model must be 'cooling_cop' or 'frequency_polynomial', "
                f"got {self.energy_model!r}"
            )
        self.energy_step_seconds = float(energy_step_seconds)
        self.energy_power_intercept_w = float(energy_power_intercept_w)
        self.energy_power_linear_w_per_hz = float(energy_power_linear_w_per_hz)
        self.energy_power_quadratic_w_per_hz2 = float(energy_power_quadratic_w_per_hz2)
        self.compressor_transition_hz = float(compressor_transition_hz)
        self.fan_effect_floor = float(fan_effect_floor)
        self.freq_effect_floor = float(freq_effect_floor)
        self.cop_min = float(cop_min)
        self.cop_max = float(cop_max)
        self.mechanism_context_blend = min(max(float(mechanism_context_blend), 0.0), 1.0)
        self.baseline_drift_scale_c = max(float(baseline_drift_scale_c), 0.0)
        self.cooling_history_seconds = max(float(cooling_history_seconds), float(energy_step_seconds))
        self.cooling_freq_exponent = max(float(cooling_freq_exponent), 1e-6)
        self.cooling_fan_exponent = max(float(cooling_fan_exponent), 1e-6)
        self.cooling_fan_reference = max(float(cooling_fan_reference), 1e-6)
        self.cooling_eev_reference = float(cooling_eev_reference)
        self.cooling_eev_range = max(abs(float(cooling_eev_range)), 1e-6)
        self.cooling_eev_gain_scale = float(cooling_eev_gain_scale)
        self.cooling_eev_effect_min = float(cooling_eev_effect_min)
        self.cooling_eev_effect_max = max(float(cooling_eev_effect_max), self.cooling_eev_effect_min)
        lag_alpha_min = float(cooling_lag_alpha_min)
        lag_alpha_max = float(cooling_lag_alpha_max)
        self.cooling_lag_alpha_min = min(max(lag_alpha_min, 0.0), 0.999)
        self.cooling_lag_alpha_max = min(max(lag_alpha_max, self.cooling_lag_alpha_min), 0.999)
        self.energy_eev_correction_min_w = float(energy_eev_correction_min_w)
        self.energy_eev_correction_max_w = max(
            float(energy_eev_correction_max_w),
            self.energy_eev_correction_min_w,
        )
        self.energy_eev_anchor = float(energy_eev_anchor)
        self.energy_eev_range = max(abs(float(energy_eev_range)), 1e-6)

    def _latest_raw_temperatures(self, obs_history_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latest_obs = obs_history_blocks[:, -1, -1, :] * self.obs_std + self.obs_mean
        return latest_obs[:, self.t_in_obs_index], latest_obs[:, self.t_out_obs_index]

    def _rollout_latents_with_context(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_latent = self.encode_context(obs_history_blocks, act_history_blocks)
        z = context_latent
        latents = []
        for block_idx in range(future_act_blocks.shape[1]):
            action_latent = self.world_model.action_block_encoder(future_act_blocks[:, block_idx])
            z = self.world_model.predictor(action_latent, z)
            latents.append(z)
        return context_latent, torch.stack(latents, dim=1)

    def _lagged_incremental_cooling_command(
        self,
        freq: torch.Tensor,
        eev: torch.Tensor,
        fan: torch.Tensor,
        cooling_scale: torch.Tensor,
    ) -> torch.Tensor:
        threshold = freq.new_tensor(self.freq_on_threshold_hz)
        freq_span = max(self.freq_max_hz - self.freq_on_threshold_hz, 1e-6)
        compressor_on = torch.sigmoid((freq - threshold) / max(self.compressor_transition_hz, 1e-6))
        freq_level = ((freq - threshold) / freq_span).clamp(0.0, 1.0)
        freq_effect = freq_level.pow(self.cooling_freq_exponent)
        fan_effect = (fan.clamp_min(0.0) / self.cooling_fan_reference).clamp(0.50, 1.30).pow(
            self.cooling_fan_exponent
        )
        eev_norm = (eev - self.cooling_eev_reference) / self.cooling_eev_range
        eev_effect = (1.0 + self.cooling_eev_gain_scale * eev_norm).clamp(
            self.cooling_eev_effect_min,
            self.cooling_eev_effect_max,
        )
        return compressor_on * cooling_scale * freq_effect * fan_effect * eev_effect

    def _energy_eev_correction_coeff(self, mech: torch.Tensor) -> torch.Tensor | None:
        if self.energy_eev_correction_index is None:
            return None
        raw = mech[:, :, self.energy_eev_correction_index]
        if self.energy_eev_correction_mode == "positive_offset":
            span = self.energy_eev_correction_max_w - self.energy_eev_correction_min_w
            return self.energy_eev_correction_min_w + span * torch.sigmoid(raw)
        return self.energy_eev_correction_scale_w * torch.tanh(raw)

    def _energy_eev_factor(self, eev: torch.Tensor) -> torch.Tensor:
        return (eev - self.energy_eev_anchor) / self.energy_eev_range

    def _probe_blocks_lagged_incremental_impl(
        self,
        future_act_blocks: torch.Tensor,
        mech: torch.Tensor,
        obs_history_blocks: torch.Tensor | None,
        act_history_blocks: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, future_blocks, frames, action_dim = future_act_blocks.shape
        steps = future_blocks * frames
        action_n = future_act_blocks.reshape(bsz, steps, action_dim)
        action_raw = action_n * self.action_std + self.action_mean
        freq = action_raw[:, :, 0].clamp_min(0.0)
        eev = action_raw[:, :, 1]
        fan = action_raw[:, :, 2].clamp_min(0.0)

        baseline_drift = self.baseline_drift_scale_c * torch.tanh(mech[:, :, 0])
        cooling_scale = self.cooling_delta_scale * torch.sigmoid(mech[:, :, 1])
        alpha = self.cooling_lag_alpha_min + (self.cooling_lag_alpha_max - self.cooling_lag_alpha_min) * torch.sigmoid(
            mech[:, :, 2]
        )
        q_ref = future_act_blocks.new_zeros(bsz)
        q_eff = future_act_blocks.new_zeros(bsz)
        if act_history_blocks is not None:
            act_raw = act_history_blocks.reshape(bsz, -1, action_dim) * self.action_std + self.action_mean
            hist_steps = max(1, int(round(self.cooling_history_seconds / max(float(self.energy_step_seconds), 1e-6))))
            hist_steps = min(hist_steps, act_raw.shape[1])
            act_window = act_raw[:, -hist_steps:]
            hist_scale = cooling_scale[:, :1].expand(-1, hist_steps)
            hist_q = self._lagged_incremental_cooling_command(
                act_window[:, :, 0].clamp_min(0.0),
                act_window[:, :, 1],
                act_window[:, :, 2].clamp_min(0.0),
                hist_scale,
            )
            hist_alpha = alpha[:, :1].expand(-1, hist_steps)
            q_eff = hist_q[:, 0]
            for hist_idx in range(hist_steps):
                q_eff = hist_alpha[:, hist_idx] * q_eff + (1.0 - hist_alpha[:, hist_idx]) * hist_q[:, hist_idx]
            q_ref = q_eff

        q_command = self._lagged_incremental_cooling_command(freq, eev, fan, cooling_scale)
        energy_eev_coeff = self._energy_eev_correction_coeff(mech)

        temp_deltas = []
        energy_deltas = []
        for step_idx in range(steps):
            q_eff = alpha[:, step_idx] * q_eff + (1.0 - alpha[:, step_idx]) * q_command[:, step_idx]
            temp_delta = baseline_drift[:, step_idx] - (q_eff - q_ref)
            step_freq = freq[:, step_idx]
            power_w = (
                self.energy_power_intercept_w
                + self.energy_power_linear_w_per_hz * step_freq
                + self.energy_power_quadratic_w_per_hz2 * step_freq.pow(2.0)
            )
            if energy_eev_coeff is not None:
                eev_norm = self._energy_eev_factor(eev[:, step_idx])
                power_w = power_w + energy_eev_coeff[:, step_idx] * eev_norm
            energy_delta = torch.clamp(self.energy_step_seconds * power_w / 3.6e6, min=0.0)
            temp_deltas.append(temp_delta)
            energy_deltas.append(energy_delta)

        physical_raw = torch.stack(
            [torch.stack(temp_deltas, dim=1), torch.stack(energy_deltas, dim=1)],
            dim=-1,
        )
        return (physical_raw - self.physical_mean) / self.physical_std

    def _probe_blocks_impl(
        self,
        block_latents: torch.Tensor,
        future_act_blocks: torch.Tensor,
        initial_t_in: torch.Tensor,
        outdoor_t: torch.Tensor,
        context_latent: torch.Tensor | None = None,
        latest_action_n: torch.Tensor | None = None,
        obs_history_blocks: torch.Tensor | None = None,
        act_history_blocks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, future_blocks, frames, action_dim = future_act_blocks.shape
        steps = future_blocks * frames
        latent_expanded = block_latents.unsqueeze(2).expand(bsz, future_blocks, frames, self.latent_dim)
        mechanism_latent = latent_expanded
        if context_latent is not None and self.mechanism_context_blend > 0.0:
            context_expanded = context_latent.view(bsz, 1, 1, self.latent_dim).expand_as(latent_expanded)
            blend = float(self.mechanism_context_blend)
            mechanism_latent = (1.0 - blend) * latent_expanded + blend * context_expanded
        mechanism_flat = mechanism_latent.reshape(bsz * steps, self.latent_dim)
        mech = self.prober["mechanism"](mechanism_flat).reshape(bsz, steps, self.mechanism_dim)
        if self.temperature_mechanism == "lagged_incremental_cooling":
            return self._probe_blocks_lagged_incremental_impl(
                future_act_blocks,
                mech,
                obs_history_blocks,
                act_history_blocks,
            )

        action_n = future_act_blocks.reshape(bsz, steps, action_dim)
        action_raw = action_n * self.action_std + self.action_mean
        freq = action_raw[:, :, 0].clamp_min(0.0)
        eev = action_raw[:, :, 1]
        fan = action_raw[:, :, 2].clamp_min(0.0)

        threshold = freq.new_tensor(self.freq_on_threshold_hz)
        freq_span = max(self.freq_max_hz - self.freq_on_threshold_hz, 1e-6)
        compressor_on = torch.sigmoid((freq - threshold) / max(self.compressor_transition_hz, 1e-6))
        freq_level = ((freq - threshold) / freq_span).clamp(0.0, 1.0)
        freq_effect = self.freq_effect_floor + (1.0 - self.freq_effect_floor) * freq_level.pow(1.25)
        fan_level = (fan / max(self.fan_max, 1e-6)).clamp(0.0, 1.0)
        fan_effect = self.fan_effect_floor + (1.0 - self.fan_effect_floor) * fan_level.pow(0.75)

        eev_center = self.eev_min + (self.eev_max - self.eev_min) * torch.sigmoid(mech[:, :, 4])
        eev_width = self.eev_width_min + (self.eev_width_max - self.eev_width_min) * torch.sigmoid(mech[:, :, 5])
        eev_bell = torch.exp(-0.5 * ((eev - eev_center) / eev_width.clamp_min(1.0)).pow(2))
        eev_effect = self.eev_effect_floor + (1.0 - self.eev_effect_floor) * eev_bell

        ua_gain = self.ua_delta_scale * torch.sigmoid(mech[:, :, 0])
        internal_load = self.internal_delta_scale * torch.sigmoid(mech[:, :, 1])
        cooling_capacity = self.cooling_delta_scale * torch.sigmoid(mech[:, :, 2])
        cop = self.cop_min + (self.cop_max - self.cop_min) * torch.sigmoid(mech[:, :, 3])
        cooling_lag_alpha = None
        if self.cooling_lag_index is not None:
            lag_raw = mech[:, :, self.cooling_lag_index]
            cooling_lag_alpha = self.cooling_lag_alpha_min + (
                self.cooling_lag_alpha_max - self.cooling_lag_alpha_min
            ) * torch.sigmoid(lag_raw)
        energy_eev_coeff = self._energy_eev_correction_coeff(mech)

        temp_deltas = []
        energy_deltas = []
        t_state = initial_t_in
        cooling_state = None
        if cooling_lag_alpha is not None and latest_action_n is not None:
            latest_action_raw = latest_action_n * self.action_std.view(1, -1) + self.action_mean.view(1, -1)
            latest_freq = latest_action_raw[:, 0].clamp_min(0.0)
            latest_eev = latest_action_raw[:, 1]
            latest_fan = latest_action_raw[:, 2].clamp_min(0.0)
            latest_compressor_on = torch.sigmoid(
                (latest_freq - threshold) / max(self.compressor_transition_hz, 1e-6)
            )
            latest_freq_level = ((latest_freq - threshold) / freq_span).clamp(0.0, 1.0)
            latest_freq_effect = self.freq_effect_floor + (1.0 - self.freq_effect_floor) * latest_freq_level.pow(1.25)
            latest_fan_level = (latest_fan / max(self.fan_max, 1e-6)).clamp(0.0, 1.0)
            latest_fan_effect = self.fan_effect_floor + (1.0 - self.fan_effect_floor) * latest_fan_level.pow(0.75)
            latest_eev_bell = torch.exp(
                -0.5 * ((latest_eev - eev_center[:, 0]) / eev_width[:, 0].clamp_min(1.0)).pow(2)
            )
            latest_eev_effect = self.eev_effect_floor + (1.0 - self.eev_effect_floor) * latest_eev_bell
            cooling_state = (
                latest_compressor_on
                * cooling_capacity[:, 0]
                * latest_freq_effect
                * latest_fan_effect
                * latest_eev_effect
            )
        for step_idx in range(steps):
            env_delta = ua_gain[:, step_idx] * torch.relu(outdoor_t - t_state)
            load_delta = env_delta + internal_load[:, step_idx]
            cooling_command = (
                compressor_on[:, step_idx]
                * cooling_capacity[:, step_idx]
                * freq_effect[:, step_idx]
                * fan_effect[:, step_idx]
                * eev_effect[:, step_idx]
            )
            if cooling_lag_alpha is not None:
                if cooling_state is None:
                    cooling_state = cooling_command
                else:
                    alpha = cooling_lag_alpha[:, step_idx]
                    cooling_state = alpha * cooling_state + (1.0 - alpha) * cooling_command
                cooling_delta = cooling_state
            else:
                cooling_delta = cooling_command
            temp_delta = load_delta - cooling_delta
            if self.energy_model == "frequency_polynomial":
                step_freq = freq[:, step_idx]
                power_w = (
                    self.energy_power_intercept_w
                    + self.energy_power_linear_w_per_hz * step_freq
                    + self.energy_power_quadratic_w_per_hz2 * step_freq.pow(2.0)
                )
                if energy_eev_coeff is not None:
                    eev_norm = self._energy_eev_factor(eev[:, step_idx])
                    power_w = power_w + energy_eev_coeff[:, step_idx] * eev_norm
                energy_delta = torch.clamp(
                    self.energy_step_seconds * power_w / 3.6e6,
                    min=0.0,
                )
            else:
                compressor_energy = self.compressor_energy_scale * cooling_delta / cop[:, step_idx].clamp_min(0.5)
                fan_energy = self.fan_energy_scale * fan_level[:, step_idx].pow(2.0)
                energy_delta = torch.clamp(compressor_energy + fan_energy, min=0.0)
            temp_deltas.append(temp_delta)
            energy_deltas.append(energy_delta)
            t_state = t_state + temp_delta

        physical_raw = torch.stack(
            [torch.stack(temp_deltas, dim=1), torch.stack(energy_deltas, dim=1)],
            dim=-1,
        )
        return (physical_raw - self.physical_mean) / self.physical_std

    def probe_blocks(self, block_latents: torch.Tensor, future_act_blocks: torch.Tensor) -> torch.Tensor:
        bsz = block_latents.shape[0]
        initial_t_in = block_latents.new_zeros(bsz)
        outdoor_t = block_latents.new_zeros(bsz)
        return self._probe_blocks_impl(block_latents, future_act_blocks, initial_t_in, outdoor_t)

    def rollout(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_latent, block_latents = self._rollout_latents_with_context(
            obs_history_blocks,
            act_history_blocks,
            future_act_blocks,
        )
        initial_t_in, outdoor_t = self._latest_raw_temperatures(obs_history_blocks)
        physical = self._probe_blocks_impl(
            block_latents,
            future_act_blocks,
            initial_t_in,
            outdoor_t,
            context_latent=context_latent,
            latest_action_n=act_history_blocks[:, -1, -1, :],
            obs_history_blocks=obs_history_blocks,
            act_history_blocks=act_history_blocks,
        )
        return block_latents, physical

    def prober_sequence_loss(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
        physical_targets: torch.Tensor,
        physical_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            context_latent, block_latents = self._rollout_latents_with_context(
                obs_history_blocks,
                act_history_blocks,
                future_act_blocks,
            )
            initial_t_in, outdoor_t = self._latest_raw_temperatures(obs_history_blocks)
        pred = self._probe_blocks_impl(
            block_latents,
            future_act_blocks,
            initial_t_in,
            outdoor_t,
            context_latent=context_latent,
            latest_action_n=act_history_blocks[:, -1, -1, :],
            obs_history_blocks=obs_history_blocks,
            act_history_blocks=act_history_blocks,
        )
        err = (pred - physical_targets).pow(2)
        if physical_weights is not None:
            err = err * physical_weights.view(1, 1, -1)
        return err.mean(), pred


HanWAM = HanWAMControllerModel


def build_world_model(model_config: dict) -> HanWAMWorldModel:
    config = dict(model_config)
    class_name = str(config.pop("class_name", "HanWAMWorldModel"))
    if class_name not in {"HanWAMWorldModel", "HanWAM"}:
        raise ValueError(f"Expected HanWAMWorldModel config, got class_name={class_name}")
    config.pop("physical_dim", None)
    return HanWAMWorldModel(**config)


def build_controller_model(model_config: dict) -> HanWAMControllerModel:
    config = dict(model_config)
    class_name = str(config.pop("class_name", "HanWAMControllerModel"))
    if class_name not in {
        "HanWAMControllerModel",
        "HanWAMPhysicsGuidedControllerModel",
        "HanWAMHardMechanismControllerModel",
        "HanWAM",
    }:
        raise ValueError(f"Expected HanWAMControllerModel config, got class_name={class_name}")
    physical_dim = int(config.pop("physical_dim", 2))
    hidden_dim = int(config.pop("prober_hidden_dim", config.get("hidden_dim", 128)))
    prober_action_scale = float(config.pop("prober_action_scale", 1.0))
    world_cfg = config.pop("world_model_config", None)
    if world_cfg is None:
        world_cfg = {"class_name": "HanWAMWorldModel", **config}
    world = build_world_model(world_cfg)
    if class_name == "HanWAMPhysicsGuidedControllerModel":
        return HanWAMPhysicsGuidedControllerModel(
            world,
            physical_dim=physical_dim,
            hidden_dim=hidden_dim,
            prober_action_scale=prober_action_scale,
            **config,
        )
    if class_name == "HanWAMHardMechanismControllerModel":
        return HanWAMHardMechanismControllerModel(
            world,
            physical_dim=physical_dim,
            hidden_dim=hidden_dim,
            prober_action_scale=prober_action_scale,
            **config,
        )
    return HanWAMControllerModel(
        world,
        physical_dim=physical_dim,
        hidden_dim=hidden_dim,
        prober_action_scale=prober_action_scale,
    )


def build_controller_model_from_world_model(
    world_model: HanWAMWorldModel,
    model_config: dict,
) -> HanWAMControllerModel:
    config = dict(model_config)
    class_name = str(config.pop("class_name", "HanWAMControllerModel"))
    if class_name not in {
        "HanWAMControllerModel",
        "HanWAMPhysicsGuidedControllerModel",
        "HanWAMHardMechanismControllerModel",
        "HanWAM",
    }:
        raise ValueError(f"Expected HanWAMControllerModel config, got class_name={class_name}")
    physical_dim = int(config.pop("physical_dim", 2))
    hidden_dim = int(config.pop("prober_hidden_dim", config.get("hidden_dim", world_model.hidden_dim)))
    prober_action_scale = float(config.pop("prober_action_scale", 1.0))
    config.pop("world_model_config", None)
    if class_name == "HanWAMPhysicsGuidedControllerModel":
        return HanWAMPhysicsGuidedControllerModel(
            world_model,
            physical_dim=physical_dim,
            hidden_dim=hidden_dim,
            prober_action_scale=prober_action_scale,
            **config,
        )
    if class_name == "HanWAMHardMechanismControllerModel":
        return HanWAMHardMechanismControllerModel(
            world_model,
            physical_dim=physical_dim,
            hidden_dim=hidden_dim,
            prober_action_scale=prober_action_scale,
            **config,
        )
    return HanWAMControllerModel(
        world_model,
        physical_dim=physical_dim,
        hidden_dim=hidden_dim,
        prober_action_scale=prober_action_scale,
    )


def build_wam_model(model_config: dict) -> nn.Module:
    class_name = str((model_config or {}).get("class_name", "HanWAMControllerModel"))
    if class_name == "HanWAMWorldModel":
        return build_world_model(model_config)
    return build_controller_model(model_config)
