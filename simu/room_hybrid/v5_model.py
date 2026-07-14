"""V5 hybrid virtual-room dynamics.

The room-load term follows the deterministic temperature-difference law from
Appendix C of the dynamic room-air-conditioner test procedure.  A compact
recurrent state represents the unmeasured refrigerant/capacity dynamics.  No
setpoint, test-condition identifier, or future measurement is used after
reset.

V5 adds two slow states for the new measured behavior:

* actuator commands are filtered before they drive the learned refrigerant
  dynamics, representing command-to-plant delay;
* the physical room temperature is exposed through a learned first-order
  sensor/air-mixing lag, reducing unrealistically fast steady-state ripples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


DT_SECONDS = 5.0
SR = 1.33
REFERENCE_TEMPERATURE_SPAN_C = 35.0 - 23.0
NORMALIZED_ROOM_HEAT_CAPACITY_SECONDS = 125.0

CONTROL_COLUMNS = [
    "control_frequency", "control_eev", "control_fan_out",
]

# Physical observations that may initialize the autonomous plant state.
# Controller targets, setpoints, inference outputs, and condition labels are
# deliberately absent.  The three current actuator values initialize the
# refrigerant state and subsequently arrive through ``step`` only.
INITIAL_COLUMNS = [
    "T_out", "T_out_coil", "T_out_discharge",
    "compressor_frequency", "eev_opening", "outdoor_fan_speed",
    "I_comp", "T_in", "T_in_coil", "RH_in", "mode", "energy_cum",
    "fault", "swing",
]


@dataclass
class V5Config:
    hidden_size: int = 48
    context_size: int = 16
    load_coefficient_min: float = 0.035
    load_coefficient_max: float = 0.075
    cooling_capacity_max: float = 2.0
    cooling_tau_min_seconds: float = 45.0
    cooling_tau_max_seconds: float = 1800.0
    control_lag_tau_seconds: float = 90.0
    room_tau_min_seconds: float = 30.0
    room_tau_max_seconds: float = 600.0
    fast_capacity_fraction: float = 0.0
    residual_rate_max_c_per_min: float = 0.0
    max_temperature_rate_c_per_min: float = 0.6
    cap_at_initial_temperature: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class HybridRoomV5(nn.Module):
    """Deterministic load balance with learned recurrent cooling dynamics."""

    def __init__(
        self,
        initial_mean,
        initial_scale,
        control_mean,
        control_scale,
        config: V5Config | None = None,
    ):
        super().__init__()
        self.config = config or V5Config()
        self.register_buffer("initial_mean", torch.as_tensor(initial_mean, dtype=torch.float32))
        self.register_buffer("initial_scale", torch.as_tensor(initial_scale, dtype=torch.float32))
        self.register_buffer("control_mean", torch.as_tensor(control_mean, dtype=torch.float32))
        self.register_buffer("control_scale", torch.as_tensor(control_scale, dtype=torch.float32))

        n_initial = len(INITIAL_COLUMNS)
        h = self.config.hidden_size
        c = self.config.context_size
        self.context_encoder = nn.Sequential(
            nn.Linear(n_initial, 64), nn.SiLU(),
            nn.Linear(64, c), nn.Tanh(),
        )
        self.hidden_encoder = nn.Sequential(
            nn.Linear(n_initial + c, 64), nn.SiLU(),
            nn.Linear(64, h), nn.Tanh(),
        )
        self.load_head = nn.Linear(c, 1)
        self.initial_cooling_head = nn.Linear(n_initial + c, 1)

        # normalized controls + temperature relative to reset + outdoor-room
        # difference + normalized elapsed time + fixed context
        dynamic_size = len(CONTROL_COLUMNS) + 3 + c
        self.cell = nn.GRUCell(dynamic_size, h)
        self.capacity_head = nn.Sequential(
            nn.Linear(h + dynamic_size, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.tau_head = nn.Linear(h, 1)
        self.room_tau_head = nn.Linear(h, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(h + dynamic_size, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        # Backward-compatible zero residual for models trained before this
        # branch existed.  Fine-tuning can then add fast bidirectional motion
        # without perturbing the established slow thermal trajectory at load.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.t_in_index = INITIAL_COLUMNS.index("T_in")
        self.t_out_index = INITIAL_COLUMNS.index("T_out")

    def normalize_initial(self, initial: torch.Tensor) -> torch.Tensor:
        return (initial - self.initial_mean) / self.initial_scale

    def initialize(self, initial: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = self.normalize_initial(initial)
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
        initial_frequency = initial[:, INITIAL_COLUMNS.index("compressor_frequency")]
        initial_current = initial[:, INITIAL_COLUMNS.index("I_comp")]
        initial_gate = torch.sigmoid((initial_frequency - 1.0) / 1.5)
        initial_gate = initial_gate * torch.sigmoid((initial_current - 0.1) / 0.2)
        cooling = cooling * initial_gate
        temperature = initial[:, self.t_in_index]
        filtered_control = initial[:, [
            INITIAL_COLUMNS.index("compressor_frequency"),
            INITIAL_COLUMNS.index("eev_opening"),
            INITIAL_COLUMNS.index("outdoor_fan_speed"),
        ]]
        return {
            "context": context,
            "hidden": hidden,
            "cooling": cooling,
            "plant_temperature": temperature,
            "temperature": temperature,
            "initial_temperature": temperature,
            "outdoor_temperature": initial[:, self.t_out_index],
            "load_coefficient": load_coefficient,
            "filtered_control": filtered_control,
            "elapsed_steps": torch.zeros_like(temperature),
        }

    def deterministic_load(
        self, temperature: torch.Tensor, outdoor_temperature: torch.Tensor,
        load_coefficient: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized sensible load; monotone in outdoor-room difference."""
        temperature_drive = torch.clamp(
            outdoor_temperature - temperature + 4.0, min=0.0,
        )
        return load_coefficient * temperature_drive

    def step_state(self, state: dict[str, torch.Tensor], control: torch.Tensor) -> dict:
        control_lag_tau = max(float(self.config.control_lag_tau_seconds), 1e-6)
        control_alpha = 1.0 - torch.exp(
            torch.as_tensor(-DT_SECONDS / control_lag_tau, device=control.device)
        )
        filtered_control = state["filtered_control"] + control_alpha * (
            control - state["filtered_control"]
        )
        normalized_control = (filtered_control - self.control_mean) / self.control_scale
        temperature = state["temperature"]
        plant_temperature = state["plant_temperature"]
        initial_temperature = state["initial_temperature"]
        outdoor_temperature = state["outdoor_temperature"]
        dynamic = torch.cat([
            normalized_control,
            ((plant_temperature - initial_temperature) / 8.0).unsqueeze(-1),
            ((outdoor_temperature - plant_temperature) / 20.0).unsqueeze(-1),
            torch.log1p(state["elapsed_steps"] / 120.0).unsqueeze(-1),
            state["context"],
        ], dim=-1)
        hidden = self.cell(dynamic, state["hidden"])
        capacity_input = torch.cat([hidden, dynamic], dim=-1)
        steady_capacity = self.config.cooling_capacity_max * torch.sigmoid(
            self.capacity_head(capacity_input).squeeze(-1)
        )

        # A stopped compressor has zero steady-state capacity, while the
        # first-order capacity state preserves physically plausible coast-down.
        compressor_gate = torch.sigmoid((filtered_control[:, 0] - 1.0) / 1.5)
        steady_capacity = steady_capacity * compressor_gate
        tau_fraction = torch.sigmoid(self.tau_head(hidden).squeeze(-1))
        tau = (
            self.config.cooling_tau_min_seconds
            + (self.config.cooling_tau_max_seconds - self.config.cooling_tau_min_seconds)
            * tau_fraction
        )
        alpha = 1.0 - torch.exp(torch.as_tensor(-DT_SECONDS, device=tau.device) / tau)
        cooling = state["cooling"] + alpha * (steady_capacity - state["cooling"])
        effective_cooling = cooling + self.config.fast_capacity_fraction * (
            steady_capacity - cooling
        )
        effective_cooling = torch.clamp(
            effective_cooling, min=0.0, max=self.config.cooling_capacity_max,
        )

        load = self.deterministic_load(
            plant_temperature, outdoor_temperature, state["load_coefficient"],
        )
        delta = DT_SECONDS / NORMALIZED_ROOM_HEAT_CAPACITY_SECONDS * (
            load - effective_cooling
        )
        residual_rate = self.config.residual_rate_max_c_per_min * torch.tanh(
            self.residual_head(capacity_input).squeeze(-1)
        )
        delta = delta + residual_rate * DT_SECONDS / 60.0
        max_delta = self.config.max_temperature_rate_c_per_min * DT_SECONDS / 60.0
        delta = max_delta * torch.tanh(delta / max_delta)
        new_plant_temperature = plant_temperature + delta
        if self.config.cap_at_initial_temperature:
            new_plant_temperature = torch.minimum(new_plant_temperature, initial_temperature)

        room_tau_fraction = torch.sigmoid(self.room_tau_head(hidden).squeeze(-1))
        room_tau = (
            self.config.room_tau_min_seconds
            + (self.config.room_tau_max_seconds - self.config.room_tau_min_seconds)
            * room_tau_fraction
        )
        room_alpha = 1.0 - torch.exp(
            torch.as_tensor(-DT_SECONDS, device=room_tau.device) / room_tau
        )
        new_temperature = temperature + room_alpha * (new_plant_temperature - temperature)
        if self.config.cap_at_initial_temperature:
            new_temperature = torch.minimum(new_temperature, initial_temperature)

        return {
            **state,
            "hidden": hidden,
            "cooling": cooling,
            "effective_cooling": effective_cooling,
            "filtered_control": filtered_control,
            "plant_temperature": new_plant_temperature,
            "temperature": new_temperature,
            "elapsed_steps": state["elapsed_steps"] + 1.0,
            "last_load": load,
            "last_steady_capacity": steady_capacity,
            "last_tau_seconds": tau,
            "last_room_tau_seconds": room_tau,
            "last_residual_rate_c_per_min": residual_rate,
        }

    def rollout(self, initial: torch.Tensor, controls: torch.Tensor) -> dict[str, torch.Tensor]:
        """Roll out ``[batch, time, 3]`` controls without measurement feedback."""
        state = self.initialize(initial)
        temperatures = [state["temperature"]]
        loads, capacities = [], []
        for index in range(controls.shape[1]):
            state = self.step_state(state, controls[:, index])
            temperatures.append(state["temperature"])
            loads.append(state["last_load"])
            capacities.append(state["cooling"])
        return {
            "temperature": torch.stack(temperatures, dim=1),
            "load": torch.stack(loads, dim=1),
            "cooling": torch.stack(capacities, dim=1),
            "load_coefficient": state["load_coefficient"],
        }
