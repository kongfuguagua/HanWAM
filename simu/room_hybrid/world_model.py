"""Unified latent room/AC world model."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .api import ACTION_COLUMNS, DT_SECONDS, RESET_COLUMNS
from .data import OUTPUT_COLUMNS


ROOM_HEAT_CAPACITY_SECONDS = 125.0


@dataclass
class RoomWorldModelConfig:
    hidden_size: int = 96
    context_size: int = 32
    load_coefficient_min: float = 0.030
    load_coefficient_max: float = 0.085
    cooling_capacity_max: float = 2.4
    cooling_tau_min_seconds: float = 20.0
    cooling_tau_max_seconds: float = 1500.0
    action_lag_tau_seconds: float = 45.0
    room_tau_min_seconds: float = 10.0
    room_tau_max_seconds: float = 420.0
    max_room_rate_c_per_min: float = 1.0
    max_ac_rate_c_per_min: float = 8.0
    room_residual_rate_c_per_min: float = 0.20
    cap_room_at_initial_temperature: bool = True
    room_bias_max_c: float = 0.0
    room_bias_tau_min_seconds: float = 300.0
    room_bias_tau_max_seconds: float = 7200.0
    context_gain_log_range: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RoomWorldModel(nn.Module):
    def __init__(
        self,
        reset_mean,
        reset_scale,
        action_mean,
        action_scale,
        output_mean,
        output_scale,
        config: RoomWorldModelConfig | None = None,
    ):
        super().__init__()
        self.config = config or RoomWorldModelConfig()
        self.register_buffer("reset_mean", torch.as_tensor(reset_mean, dtype=torch.float32))
        self.register_buffer("reset_scale", torch.as_tensor(reset_scale, dtype=torch.float32))
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("output_mean", torch.as_tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_scale", torch.as_tensor(output_scale, dtype=torch.float32))

        n_reset = len(RESET_COLUMNS)
        n_action = len(ACTION_COLUMNS)
        h = self.config.hidden_size
        c = self.config.context_size

        self.context_encoder = nn.Sequential(
            nn.Linear(n_reset, 96), nn.SiLU(),
            nn.Linear(96, c), nn.Tanh(),
        )
        self.hidden_encoder = nn.Sequential(
            nn.Linear(n_reset + c, 128), nn.SiLU(),
            nn.Linear(128, h), nn.Tanh(),
        )
        self.load_head = nn.Linear(c, 1)
        self.initial_cooling_head = nn.Linear(n_reset + c, 1)
        if self.config.context_gain_log_range > 0:
            self.context_gain_head = nn.Sequential(
                nn.Linear(n_reset + c, 64), nn.SiLU(), nn.Linear(64, 4),
            )
        if self.config.room_bias_max_c > 0:
            self.room_bias_head = nn.Sequential(
                nn.Linear(n_reset + c, 64), nn.SiLU(), nn.Linear(64, 1),
            )
            self.room_bias_tau_head = nn.Linear(n_reset + c, 1)

        n_output = len(OUTPUT_COLUMNS)
        dynamic_size = n_action + n_output + 3 + c
        self.cell = nn.GRUCell(dynamic_size, h)
        self.capacity_head = nn.Sequential(
            nn.Linear(h + dynamic_size, 96), nn.SiLU(), nn.Linear(96, 1),
        )
        self.cooling_tau_head = nn.Linear(h, 1)
        self.room_tau_head = nn.Linear(h, 1)
        self.room_residual_head = nn.Sequential(
            nn.Linear(h + dynamic_size, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.ac_rate_head = nn.Sequential(
            nn.Linear(h + dynamic_size, 96), nn.SiLU(), nn.Linear(96, 3),
        )

        self.t_out_index = RESET_COLUMNS.index("T_out")
        self.t_in_index = OUTPUT_COLUMNS.index("T_in")

    def initialize(self, reset: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = (reset - self.reset_mean) / self.reset_scale
        context = self.context_encoder(normalized)
        encoded = torch.cat([normalized, context], dim=-1)
        hidden = self.hidden_encoder(encoded)
        load_fraction = torch.sigmoid(self.load_head(context)).squeeze(-1)
        load_coefficient = (
            self.config.load_coefficient_min
            + (self.config.load_coefficient_max - self.config.load_coefficient_min)
            * load_fraction
        )
        cooling = self.config.cooling_capacity_max * torch.sigmoid(
            self.initial_cooling_head(encoded).squeeze(-1)
        )
        frequency = reset[:, RESET_COLUMNS.index("compressor_frequency")]
        current = reset[:, RESET_COLUMNS.index("I_comp")]
        cooling = cooling * torch.sigmoid((frequency - 1.0) / 1.5)
        cooling = cooling * torch.sigmoid((current - 0.1) / 0.2)
        if self.config.context_gain_log_range > 0:
            gain_logits = torch.tanh(self.context_gain_head(encoded))
            gains = torch.exp(self.config.context_gain_log_range * gain_logits)
            load_gain = gains[:, 0]
            capacity_gain = gains[:, 1]
            room_tau_gain = gains[:, 2]
            cooling_tau_gain = gains[:, 3]
        else:
            load_gain = torch.ones_like(cooling)
            capacity_gain = torch.ones_like(cooling)
            room_tau_gain = torch.ones_like(cooling)
            cooling_tau_gain = torch.ones_like(cooling)
        if self.config.room_bias_max_c > 0:
            room_bias_target = self.config.room_bias_max_c * torch.tanh(
                self.room_bias_head(encoded).squeeze(-1)
            )
            tau_fraction = torch.sigmoid(self.room_bias_tau_head(encoded).squeeze(-1))
            room_bias_tau = (
                self.config.room_bias_tau_min_seconds
                + (self.config.room_bias_tau_max_seconds - self.config.room_bias_tau_min_seconds)
                * tau_fraction
            )
        else:
            room_bias_target = torch.zeros_like(cooling)
            room_bias_tau = torch.ones_like(cooling)

        outputs = torch.stack([
            reset[:, RESET_COLUMNS.index("T_in")],
            reset[:, RESET_COLUMNS.index("T_in_coil")],
            reset[:, RESET_COLUMNS.index("T_out_coil")],
            reset[:, RESET_COLUMNS.index("T_out_discharge")],
        ], dim=-1)
        action = torch.stack([
            reset[:, RESET_COLUMNS.index("compressor_frequency")],
            reset[:, RESET_COLUMNS.index("eev_opening")],
            reset[:, RESET_COLUMNS.index("outdoor_fan_speed")],
        ], dim=-1)
        return {
            "context": context,
            "hidden": hidden,
            "outputs": outputs,
            "room_core": outputs[:, self.t_in_index],
            "initial_room": outputs[:, self.t_in_index],
            "outdoor_temperature": reset[:, self.t_out_index],
            "load_coefficient": load_coefficient,
            "load_gain": load_gain,
            "capacity_gain": capacity_gain,
            "room_tau_gain": room_tau_gain,
            "cooling_tau_gain": cooling_tau_gain,
            "cooling": cooling,
            "room_bias": torch.zeros_like(outputs[:, self.t_in_index]),
            "room_bias_target": room_bias_target,
            "room_bias_tau_seconds": room_bias_tau,
            "filtered_action": action,
            "elapsed_steps": torch.zeros_like(outputs[:, self.t_in_index]),
        }

    def deterministic_load(
        self,
        room_temperature: torch.Tensor,
        outdoor_temperature: torch.Tensor,
        load_coefficient: torch.Tensor,
    ) -> torch.Tensor:
        drive = torch.clamp(outdoor_temperature - room_temperature + 4.0, min=0.0)
        return load_coefficient * drive

    def step_state(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        tau = max(float(self.config.action_lag_tau_seconds), 1e-6)
        action_alpha = 1.0 - torch.exp(torch.as_tensor(-DT_SECONDS / tau, device=action.device))
        filtered_action = state["filtered_action"] + action_alpha * (action - state["filtered_action"])

        normalized_action = (filtered_action - self.action_mean) / self.action_scale
        normalized_outputs = (state["outputs"] - self.output_mean) / self.output_scale
        room = state["outputs"][:, self.t_in_index]
        dynamic = torch.cat([
            normalized_action,
            normalized_outputs,
            ((room - state["initial_room"]) / 8.0).unsqueeze(-1),
            ((state["outdoor_temperature"] - room) / 20.0).unsqueeze(-1),
            torch.log1p(state["elapsed_steps"] / 120.0).unsqueeze(-1),
            state["context"],
        ], dim=-1)
        hidden = self.cell(dynamic, state["hidden"])
        head_input = torch.cat([hidden, dynamic], dim=-1)

        steady_capacity = self.config.cooling_capacity_max * torch.sigmoid(
            self.capacity_head(head_input).squeeze(-1)
        )
        steady_capacity = steady_capacity * torch.sigmoid((filtered_action[:, 0] - 1.0) / 1.5)
        steady_capacity = steady_capacity * state["capacity_gain"]
        tau_fraction = torch.sigmoid(self.cooling_tau_head(hidden).squeeze(-1))
        cooling_tau = (
            self.config.cooling_tau_min_seconds
            + (self.config.cooling_tau_max_seconds - self.config.cooling_tau_min_seconds)
            * tau_fraction
        )
        cooling_tau = torch.clamp(
            cooling_tau * state["cooling_tau_gain"],
            min=self.config.cooling_tau_min_seconds,
            max=self.config.cooling_tau_max_seconds,
        )
        cooling_alpha = 1.0 - torch.exp(torch.as_tensor(-DT_SECONDS, device=action.device) / cooling_tau)
        cooling = state["cooling"] + cooling_alpha * (steady_capacity - state["cooling"])

        load = self.deterministic_load(
            state["room_core"], state["outdoor_temperature"], state["load_coefficient"],
        )
        load = load * state["load_gain"]
        room_delta = DT_SECONDS / ROOM_HEAT_CAPACITY_SECONDS * (load - cooling)
        residual_rate = self.config.room_residual_rate_c_per_min * torch.tanh(
            self.room_residual_head(head_input).squeeze(-1)
        )
        room_delta = room_delta + residual_rate * DT_SECONDS / 60.0
        max_room_delta = self.config.max_room_rate_c_per_min * DT_SECONDS / 60.0
        room_delta = max_room_delta * torch.tanh(room_delta / max_room_delta)
        room_core = state["room_core"] + room_delta
        if self.config.cap_room_at_initial_temperature:
            room_core = torch.minimum(room_core, state["initial_room"])

        room_tau_fraction = torch.sigmoid(self.room_tau_head(hidden).squeeze(-1))
        room_tau = (
            self.config.room_tau_min_seconds
            + (self.config.room_tau_max_seconds - self.config.room_tau_min_seconds)
            * room_tau_fraction
        )
        room_tau = torch.clamp(
            room_tau * state["room_tau_gain"],
            min=self.config.room_tau_min_seconds,
            max=self.config.room_tau_max_seconds,
        )
        room_alpha = 1.0 - torch.exp(torch.as_tensor(-DT_SECONDS, device=action.device) / room_tau)
        room_output = room + room_alpha * (room_core - room)
        if self.config.cap_room_at_initial_temperature:
            room_output = torch.minimum(room_output, state["initial_room"])
        if self.config.room_bias_max_c > 0:
            bias_alpha = 1.0 - torch.exp(
                torch.as_tensor(-DT_SECONDS, device=action.device)
                / state["room_bias_tau_seconds"].clamp_min(1.0)
            )
            room_bias = state["room_bias"] + bias_alpha * (
                state["room_bias_target"] - state["room_bias"]
            )
            room_output = room_output + room_bias
        else:
            room_bias = state["room_bias"]

        ac_rates = self.config.max_ac_rate_c_per_min * torch.tanh(self.ac_rate_head(head_input))
        ac_outputs = state["outputs"][:, 1:] + ac_rates * DT_SECONDS / 60.0
        outputs = torch.cat([room_output.unsqueeze(-1), ac_outputs], dim=-1)
        return {
            **state,
            "hidden": hidden,
            "outputs": outputs,
            "room_core": room_core,
            "cooling": cooling,
            "room_bias": room_bias,
            "filtered_action": filtered_action,
            "elapsed_steps": state["elapsed_steps"] + 1.0,
            "last_load": load,
            "last_steady_capacity": steady_capacity,
            "last_residual_rate_c_per_min": residual_rate,
            "last_room_tau_seconds": room_tau,
            "last_cooling_tau_seconds": cooling_tau,
            "last_room_bias": room_bias,
            "last_load_gain": state["load_gain"],
            "last_capacity_gain": state["capacity_gain"],
            "last_room_tau_gain": state["room_tau_gain"],
            "last_cooling_tau_gain": state["cooling_tau_gain"],
        }

    def rollout(
        self,
        reset: torch.Tensor,
        actions: torch.Tensor,
        truth: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        state = self.initialize(reset)
        outputs = [state["outputs"]]
        loads = []
        cooling = []
        for index in range(actions.shape[1]):
            if truth is not None and teacher_forcing > 0:
                state = {
                    **state,
                    "outputs": (
                        (1.0 - teacher_forcing) * state["outputs"]
                        + teacher_forcing * truth[:, index]
                    ),
                    "room_core": (
                        (1.0 - teacher_forcing) * state["room_core"]
                        + teacher_forcing * truth[:, index, self.t_in_index]
                    ),
                }
            state = self.step_state(state, actions[:, index])
            outputs.append(state["outputs"])
            loads.append(state["last_load"])
            cooling.append(state["cooling"])
        return {
            "outputs": torch.stack(outputs, dim=1),
            "load": torch.stack(loads, dim=1),
            "cooling": torch.stack(cooling, dim=1),
            "load_coefficient": state["load_coefficient"],
        }
