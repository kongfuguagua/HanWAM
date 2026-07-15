"""HanWAM latent world model and hard-mechanism controller prober."""
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


@dataclass(frozen=True)
class PreparedBlockContext:
    """History-dependent state shared by every candidate in one replan."""

    state_latent: torch.Tensor
    initial_t_in: torch.Tensor
    outdoor_t: torch.Tensor
    initial_cooling: torch.Tensor


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
        return self.rollout_latents(z, future_act_blocks)

    def rollout_latents(
        self,
        state_latent: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out candidate-specific block latents from a precomputed state."""

        z = state_latent
        if z.shape[0] == 1 and future_act_blocks.shape[0] != 1:
            z = z.expand(future_act_blocks.shape[0], -1)
        elif z.shape[0] != future_act_blocks.shape[0]:
            raise ValueError(
                "state_latent batch must be 1 or match future actions: "
                f"{z.shape[0]} vs {future_act_blocks.shape[0]}"
            )
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
        quadrature_points: int = 17,
        integration_limit: float = 5.0,
        kernel_sigma: float = 1.0,
        projection_chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable Epps-Pulley SIGReg on predicted rollout latents.

        ``latents`` is [batch, horizon, dim].  For each horizon and random
        one-dimensional projection, the empirical characteristic function
        across the batch is matched to N(0, 1), then averaged over time and
        projections.  The statistic intentionally does not standardize the
        projected samples because their mean and scale are part of the
        isotropic-Gaussian constraint.
        """

        if latents.ndim == 2:
            latents = latents.unsqueeze(1)
        if latents.ndim != 3:
            raise ValueError(f"SIGReg expects [B,T,D] latents, got {tuple(latents.shape)}")
        if latents.shape[0] < 2:
            zero = latents.new_tensor(0.0)
            return zero, zero, zero

        work = latents.float()
        projections = max(1, int(num_projections))
        directions = torch.randn(
            work.shape[-1],
            projections,
            dtype=work.dtype,
            device=work.device,
        )
        directions = F.normalize(directions, dim=0)
        projected = torch.einsum("btd,dm->tbm", work, directions)

        points = max(3, int(quadrature_points))
        limit = max(float(integration_limit), 1e-3)
        sigma = max(float(kernel_sigma), 1e-3)
        grid = torch.linspace(-limit, limit, points, dtype=work.dtype, device=work.device)
        target_real = torch.exp(-0.5 * grid.pow(2)).view(1, 1, points)
        weight = torch.exp(-0.5 * (grid / sigma).pow(2)).view(1, 1, points)

        statistics = []
        chunk_size = max(1, int(projection_chunk_size))
        for start in range(0, projections, chunk_size):
            values = projected[:, :, start : start + chunk_size]
            phase = values.unsqueeze(-1) * grid.view(1, 1, 1, points)
            empirical_real = torch.cos(phase).mean(dim=1)
            empirical_imag = torch.sin(phase).mean(dim=1)
            discrepancy = (
                (empirical_real - target_real).pow(2) + empirical_imag.pow(2)
            ) * weight
            statistics.append(torch.trapezoid(discrepancy, grid, dim=-1))
        epps_pulley = torch.cat(statistics, dim=1).mean()

        projected_mean = projected.mean(dim=1).pow(2).mean()
        projected_std = torch.sqrt(projected.var(dim=1, unbiased=False) + 1e-6)
        projected_scale = (projected_std - 1.0).pow(2).mean()
        return epps_pulley.to(latents.dtype), projected_scale.to(latents.dtype), projected_mean.to(latents.dtype)

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
        sigreg_quadrature_points: int = 17,
        sigreg_integration_limit: float = 5.0,
        sigreg_kernel_sigma: float = 1.0,
        sigreg_projection_chunk_size: int = 16,
    ) -> SequenceLossBreakdown:
        z0 = self.encode_context(obs_history_blocks, act_history_blocks)
        pred_latents = self.rollout_latents(z0, future_act_blocks)
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
        reg, projection, mean = self.sigreg_loss(
            pred_latents,
            num_projections=sigreg_num_projections,
            quadrature_points=sigreg_quadrature_points,
            integration_limit=sigreg_integration_limit,
            kernel_sigma=sigreg_kernel_sigma,
            projection_chunk_size=sigreg_projection_chunk_size,
        )
        total = float(latent_weight) * latent + float(sigreg_weight) * reg
        zero = obs_history_blocks.new_tensor(0.0)
        return SequenceLossBreakdown(total, latent, zero, reg, projection, mean)


class HanWAMControllerModel(nn.Module):
    """Shared interface for the current Stage-II controller model."""

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


class HanWAMBlockControllerModel(HanWAMControllerModel):
    """Future-latent-only block prober with an explicit thermal mechanism.

    The learned prober runs once per 60-second block.  The deterministic
    mechanism may integrate the block at the native five-second data rate,
    but exposes one temperature/energy delta per block to MPPI.
    """

    physical_is_block = True
    architecture_version = "hanwam_block_v1"

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
        step_seconds: float = 5.0,
        freq_on_threshold_hz: float = 5.0,
        freq_max_hz: float = 80.0,
        compressor_transition_hz: float = 2.0,
        cooling_freq_exponent: float = 1.10,
        cooling_fan_exponent: float = 0.60,
        cooling_fan_reference: float = 750.0,
        cooling_eev_reference: float = 240.0,
        cooling_eev_range: float = 170.0,
        cooling_eev_gain_scale: float = 0.12,
        cooling_eev_effect_min: float = 0.85,
        cooling_eev_effect_max: float = 1.15,
        ua_delta_max_c: float = 0.0012,
        internal_delta_max_c: float = 0.0040,
        cooling_delta_max_c: float = 0.10,
        cooling_lag_alpha_min: float = 0.70,
        cooling_lag_alpha_max: float = 0.995,
        nominal_cooling_delta_c: float = 0.070,
        nominal_cooling_lag_alpha: float = 0.92,
        energy_power_intercept_w: float = 279.87,
        energy_power_linear_w_per_hz: float = 9.0034,
        energy_power_quadratic_w_per_hz2: float = 0.04494,
        energy_eev_anchor: float = 100.0,
        energy_eev_range: float = 170.0,
        energy_eev_correction_max_w: float = 150.0,
    ):
        nn.Module.__init__(self)
        self.world_model = world_model
        self.physical_dim = int(physical_dim)
        if self.physical_dim != 2:
            raise ValueError("HanWAMBlockControllerModel expects physical_dim=2")
        if float(prober_action_scale) != 0.0:
            raise ValueError("HanWAM block prober is future-latent-only; prober_action_scale must be 0")
        self.prober_action_scale = 0.0
        hidden = int(hidden_dim or world_model.hidden_dim)
        self.prober = mlp([world_model.latent_dim, hidden, hidden, 5])

        action_mean_t = torch.as_tensor(action_mean if action_mean is not None else [0.0, 0.0, 0.0], dtype=torch.float32)
        action_std_t = torch.as_tensor(action_std if action_std is not None else [1.0, 1.0, 1.0], dtype=torch.float32)
        physical_mean_t = torch.as_tensor(physical_mean if physical_mean is not None else [0.0, 0.0], dtype=torch.float32)
        physical_std_t = torch.as_tensor(physical_std if physical_std is not None else [1.0, 1.0], dtype=torch.float32)
        obs_mean_t = torch.as_tensor(obs_mean if obs_mean is not None else [0.0] * world_model.obs_dim, dtype=torch.float32)
        obs_std_t = torch.as_tensor(obs_std if obs_std is not None else [1.0] * world_model.obs_dim, dtype=torch.float32)
        self.register_buffer("action_mean", action_mean_t.view(1, 1, 1, -1))
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6).view(1, 1, 1, -1))
        self.register_buffer("physical_mean", physical_mean_t.view(1, 1, -1))
        self.register_buffer("physical_std", physical_std_t.clamp_min(1e-6).view(1, 1, -1))
        self.register_buffer("obs_mean", obs_mean_t.view(1, -1))
        self.register_buffer("obs_std", obs_std_t.clamp_min(1e-6).view(1, -1))

        self.t_in_obs_index = int(t_in_obs_index)
        self.t_out_obs_index = int(t_out_obs_index)
        self.step_seconds = float(step_seconds)
        self.freq_on_threshold_hz = float(freq_on_threshold_hz)
        self.freq_max_hz = float(freq_max_hz)
        self.compressor_transition_hz = max(float(compressor_transition_hz), 1e-6)
        self.cooling_freq_exponent = max(float(cooling_freq_exponent), 1e-6)
        self.cooling_fan_exponent = max(float(cooling_fan_exponent), 1e-6)
        self.cooling_fan_reference = max(float(cooling_fan_reference), 1e-6)
        self.cooling_eev_reference = float(cooling_eev_reference)
        self.cooling_eev_range = max(abs(float(cooling_eev_range)), 1e-6)
        self.cooling_eev_gain_scale = float(cooling_eev_gain_scale)
        self.cooling_eev_effect_min = float(cooling_eev_effect_min)
        self.cooling_eev_effect_max = max(float(cooling_eev_effect_max), self.cooling_eev_effect_min)

        self.ua_delta_max_c = max(float(ua_delta_max_c), 0.0)
        self.internal_delta_max_c = max(float(internal_delta_max_c), 0.0)
        self.cooling_delta_max_c = max(float(cooling_delta_max_c), 1e-6)
        self.cooling_lag_alpha_min = min(max(float(cooling_lag_alpha_min), 0.0), 0.999)
        self.cooling_lag_alpha_max = min(max(float(cooling_lag_alpha_max), self.cooling_lag_alpha_min), 0.999)
        self.nominal_cooling_delta_c = max(float(nominal_cooling_delta_c), 1e-6)
        self.nominal_cooling_lag_alpha = min(
            max(float(nominal_cooling_lag_alpha), self.cooling_lag_alpha_min),
            self.cooling_lag_alpha_max,
        )
        self.energy_power_intercept_w = float(energy_power_intercept_w)
        self.energy_power_linear_w_per_hz = float(energy_power_linear_w_per_hz)
        self.energy_power_quadratic_w_per_hz2 = float(energy_power_quadratic_w_per_hz2)
        self.energy_eev_anchor = float(energy_eev_anchor)
        self.energy_eev_range = max(abs(float(energy_eev_range)), 1e-6)
        self.energy_eev_correction_max_w = max(float(energy_eev_correction_max_w), 0.0)

    def _decode_actions(self, actions_n: torch.Tensor) -> torch.Tensor:
        return actions_n * self.action_std + self.action_mean

    def _cooling_command(
        self,
        freq: torch.Tensor,
        eev: torch.Tensor,
        fan: torch.Tensor,
        cooling_scale: torch.Tensor,
    ) -> torch.Tensor:
        threshold = freq.new_tensor(self.freq_on_threshold_hz)
        span = max(self.freq_max_hz - self.freq_on_threshold_hz, 1e-6)
        compressor_on = torch.sigmoid((freq - threshold) / self.compressor_transition_hz)
        freq_level = ((freq - threshold) / span).clamp(0.0, 1.0)
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

    def _initial_cooling_state(self, act_history_blocks: torch.Tensor) -> torch.Tensor:
        raw = self._decode_actions(act_history_blocks).reshape(act_history_blocks.shape[0], -1, self.action_dim)
        scale = raw.new_full(raw.shape[:2], self.nominal_cooling_delta_c)
        commands = self._cooling_command(raw[..., 0], raw[..., 1], raw[..., 2], scale)
        alpha = float(self.nominal_cooling_lag_alpha)
        q_state = commands[:, 0]
        for step_idx in range(1, commands.shape[1]):
            q_state = alpha * q_state + (1.0 - alpha) * commands[:, step_idx]
        return q_state

    def prepare_context(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
    ) -> PreparedBlockContext:
        state_latent = self.world_model.encode_context(obs_history_blocks, act_history_blocks)
        raw_obs = obs_history_blocks * self.obs_std.view(1, 1, 1, -1) + self.obs_mean.view(1, 1, 1, -1)
        latest_obs = raw_obs[:, -1, -1, :]
        initial_cooling = self._initial_cooling_state(act_history_blocks)
        return PreparedBlockContext(
            state_latent=state_latent,
            initial_t_in=latest_obs[:, self.t_in_obs_index],
            outdoor_t=latest_obs[:, self.t_out_obs_index],
            initial_cooling=initial_cooling,
        )

    @staticmethod
    def _expand_context_value(value: torch.Tensor, batch: int) -> torch.Tensor:
        if value.shape[0] == batch:
            return value
        if value.shape[0] == 1:
            return value.expand(batch, *value.shape[1:])
        raise ValueError(f"prepared context batch {value.shape[0]} does not match candidate batch {batch}")

    def _mechanism_parameters(
        self,
        block_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.prober(block_latents)
        ua = self.ua_delta_max_c * torch.sigmoid(raw[..., 0])
        internal = self.internal_delta_max_c * torch.sigmoid(raw[..., 1])
        cooling = self.cooling_delta_max_c * torch.sigmoid(raw[..., 2])
        alpha = self.cooling_lag_alpha_min + (
            self.cooling_lag_alpha_max - self.cooling_lag_alpha_min
        ) * torch.sigmoid(raw[..., 3])
        energy_eev = self.energy_eev_correction_max_w * torch.sigmoid(raw[..., 4])
        return ua, internal, cooling, alpha, energy_eev

    def probe_prepared(
        self,
        context: PreparedBlockContext,
        block_latents: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> torch.Tensor:
        batch, future_blocks, frames, _ = future_act_blocks.shape
        if block_latents.shape[:2] != (batch, future_blocks):
            raise ValueError("block latents and future actions must share [B,K]")
        raw_actions = self._decode_actions(future_act_blocks)
        ua, internal, cooling, alpha, energy_eev = self._mechanism_parameters(block_latents)
        t_state = self._expand_context_value(context.initial_t_in, batch)
        outdoor = self._expand_context_value(context.outdoor_t, batch)
        q_eff = self._expand_context_value(context.initial_cooling, batch)
        block_temp_delta = []
        block_energy = []

        for block_idx in range(future_blocks):
            t_start = t_state
            energy_sum = t_state.new_zeros(batch)
            for frame_idx in range(frames):
                action = raw_actions[:, block_idx, frame_idx]
                freq, eev, fan = action[:, 0], action[:, 1], action[:, 2]
                q_cmd = self._cooling_command(freq, eev, fan, cooling[:, block_idx])
                q_eff = alpha[:, block_idx] * q_eff + (1.0 - alpha[:, block_idx]) * q_cmd
                heat = ua[:, block_idx] * torch.relu(outdoor - t_state) + internal[:, block_idx]
                t_state = t_state + heat - q_eff

                power_w = (
                    self.energy_power_intercept_w
                    + self.energy_power_linear_w_per_hz * freq
                    + self.energy_power_quadratic_w_per_hz2 * freq.pow(2.0)
                )
                eev_factor = (eev - self.energy_eev_anchor) / self.energy_eev_range
                power_w = power_w + energy_eev[:, block_idx] * eev_factor
                # The polynomial was fitted on compressor-on data.  When MPPI
                # selects the actuator deadband/off state it must not inherit
                # the fitted on-state intercept power.
                power_w = torch.where(
                    freq >= self.freq_on_threshold_hz,
                    power_w,
                    torch.zeros_like(power_w),
                )
                energy_sum = energy_sum + torch.clamp(
                    self.step_seconds * power_w / 3.6e6,
                    min=0.0,
                )
            block_temp_delta.append(t_state - t_start)
            block_energy.append(energy_sum)

        physical_raw = torch.stack(
            [torch.stack(block_temp_delta, dim=1), torch.stack(block_energy, dim=1)],
            dim=-1,
        )
        return (physical_raw - self.physical_mean) / self.physical_std

    def rollout_prepared(
        self,
        context: PreparedBlockContext,
        future_act_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_latents = self.world_model.rollout_latents(context.state_latent, future_act_blocks)
        physical = self.probe_prepared(context, block_latents, future_act_blocks)
        return block_latents, physical

    def probe_blocks(self, block_latents: torch.Tensor, future_act_blocks: torch.Tensor) -> torch.Tensor:
        batch = block_latents.shape[0]
        zero = block_latents.new_zeros(batch)
        context = PreparedBlockContext(block_latents[:, 0], zero, zero, zero)
        return self.probe_prepared(context, block_latents, future_act_blocks)

    def rollout(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.prepare_context(obs_history_blocks, act_history_blocks)
        return self.rollout_prepared(context, future_act_blocks)

    def prober_sequence_loss(
        self,
        obs_history_blocks: torch.Tensor,
        act_history_blocks: torch.Tensor,
        future_act_blocks: torch.Tensor,
        physical_targets: torch.Tensor,
        physical_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            context = self.prepare_context(obs_history_blocks, act_history_blocks)
            block_latents = self.world_model.rollout_latents(context.state_latent, future_act_blocks)
        pred = self.probe_prepared(context, block_latents.detach(), future_act_blocks)
        err = (pred - physical_targets).pow(2)
        if physical_weights is not None:
            err = err * physical_weights.view(1, 1, -1)
        return err.mean(), pred


def build_world_model(model_config: dict) -> HanWAMWorldModel:
    config = dict(model_config)
    class_name = str(config.pop("class_name", "HanWAMWorldModel"))
    if class_name != "HanWAMWorldModel":
        raise ValueError(f"Expected HanWAMWorldModel config, got class_name={class_name}")
    config.pop("physical_dim", None)
    return HanWAMWorldModel(**config)


def build_controller_model(model_config: dict) -> HanWAMControllerModel:
    config = dict(model_config)
    class_name = str(config.pop("class_name", "HanWAMBlockControllerModel"))
    if class_name != "HanWAMBlockControllerModel":
        raise ValueError(f"Unsupported HanWAM controller class: {class_name}")
    physical_dim = int(config.pop("physical_dim", 2))
    hidden_dim = int(config.pop("prober_hidden_dim", config.get("hidden_dim", 128)))
    prober_action_scale = float(config.pop("prober_action_scale", 1.0))
    world_cfg = config.pop("world_model_config", None)
    if world_cfg is None:
        world_cfg = {"class_name": "HanWAMWorldModel", **config}
    world = build_world_model(world_cfg)
    return HanWAMBlockControllerModel(
        world,
        physical_dim=physical_dim,
        hidden_dim=hidden_dim,
        prober_action_scale=prober_action_scale,
        **config,
    )


def build_controller_model_from_world_model(
    world_model: HanWAMWorldModel,
    model_config: dict,
) -> HanWAMControllerModel:
    config = dict(model_config)
    class_name = str(config.pop("class_name", "HanWAMBlockControllerModel"))
    if class_name != "HanWAMBlockControllerModel":
        raise ValueError(f"Unsupported HanWAM controller class: {class_name}")
    physical_dim = int(config.pop("physical_dim", 2))
    hidden_dim = int(config.pop("prober_hidden_dim", config.get("hidden_dim", world_model.hidden_dim)))
    prober_action_scale = float(config.pop("prober_action_scale", 1.0))
    config.pop("world_model_config", None)
    return HanWAMBlockControllerModel(
        world_model,
        physical_dim=physical_dim,
        hidden_dim=hidden_dim,
        prober_action_scale=prober_action_scale,
        **config,
    )


def build_wam_model(model_config: dict) -> nn.Module:
    class_name = str((model_config or {}).get("class_name", "HanWAMBlockControllerModel"))
    if class_name == "HanWAMWorldModel":
        return build_world_model(model_config)
    return build_controller_model(model_config)
