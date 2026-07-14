"""V3 continuous cooling-room feature and heat-state definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DT_SECONDS = 5.0
RAW_COLUMNS = [
    "ts", "T_out", "T_out_coil", "T_out_discharge",
    "compressor_frequency", "eev_opening", "outdoor_fan_speed",
    "I_comp", "T_in", "T_in_coil", "indoor_fan_target", "RH_target",
    "RH_in", "T_set", "mode", "energy_cum", "inference_freq",
    "inference_eev", "inference_fan_out", "inference_fan_in", "fault",
    "swing", "pid_freq", "pid_target",
]
CONTROL_COLUMNS = ["compressor_frequency", "eev_opening", "outdoor_fan_speed"]
STATE_COLUMNS = [c for c in RAW_COLUMNS if c not in {"ts", *CONTROL_COLUMNS}]
STATE_INDEX = {name: index for index, name in enumerate(STATE_COLUMNS)}

# Plant-model inputs: actual measurements and controls only.
# Target / setpoint variables (T_set, indoor_fan_target, RH_target, pid_*)
# are intentionally excluded so the learned simulator remains a pure plant
# model and does not inherit controller intent.
PLANT_STATE_COLUMNS = [
    "T_out", "T_out_coil", "T_out_discharge",
    "T_in", "T_in_coil", "RH_in",
    "mode", "energy_cum", "fault", "swing",
    "inference_freq", "inference_eev", "inference_fan_out", "inference_fan_in",
]
PLANT_STATE_INDEX = {name: index for index, name in enumerate(PLANT_STATE_COLUMNS)}


@dataclass
class HeatLoadServoConfig:
    tracking_tau_seconds: float = 60.0
    tracking_band_c: float = 0.5
    cooling_balance_offset_c: float = 4.0
    heating_balance_offset_c: float = 5.0
    load_filter_tau_seconds: float = 300.0


class StandardHeatLoadServo:
    """Constrained heat-load diagnostic state; not an estimator input."""

    def __init__(self, config: HeatLoadServoConfig | None = None):
        self.config = config or HeatLoadServoConfig()

    def reset(self, room_temperature: float, outdoor_temperature: float, mode: int) -> None:
        self.target = float(room_temperature)
        self.actual = float(room_temperature)
        self.outdoor = float(outdoor_temperature)
        self.mode = int(mode)
        self.load_proxy = self._load_proxy()
        self.filtered_load = self.load_proxy
        self.mean_load = self.load_proxy
        self.steps = 0
        self.saturated_steps = 0

    def _load_proxy(self) -> float:
        if self.mode == 3:
            return max(0.0, self.actual - self.config.heating_balance_offset_c - self.outdoor)
        return max(0.0, self.outdoor - (self.actual - self.config.cooling_balance_offset_c))

    def features(self) -> np.ndarray:
        return np.asarray([
            self.target, self.actual, self.target - self.actual, self.load_proxy,
            self.filtered_load, self.mean_load,
            self.saturated_steps / max(1, self.steps),
        ], np.float32)

    def update(self, room_temperature: float, dt: float = DT_SECONDS) -> None:
        self.target = float(room_temperature)
        alpha = 1.0 - np.exp(-dt / max(self.config.tracking_tau_seconds, 1e-6))
        candidate = self.actual + alpha * (self.target - self.actual)
        projected = float(np.clip(
            candidate,
            self.target - self.config.tracking_band_c,
            self.target + self.config.tracking_band_c,
        ))
        if projected != candidate:
            self.saturated_steps += 1
        self.actual = projected
        self.load_proxy = self._load_proxy()
        load_alpha = 1.0 - np.exp(-dt / max(self.config.load_filter_tau_seconds, 1e-6))
        self.filtered_load += load_alpha * (self.load_proxy - self.filtered_load)
        self.steps += 1
        self.mean_load += (self.load_proxy - self.mean_load) / self.steps


@dataclass
class FeatureConfig:
    """Which physics-aware feature batches are enabled.

    Features are grouped so they can be ablated during training.
    Energy-derived features are intentionally excluded.
    """

    batch1_control_memory: bool = True  # freq integrals, duty cycle, freq EWM
    batch2_approach_temps: bool = True  # T_in-T_in_coil / T_out-T_out_coil EWMs
    batch3_interactions: bool = True  # (T_in - T_out) * freq, RH_in * approach

    def as_tuple(self) -> tuple[bool, bool, bool]:
        return (self.batch1_control_memory, self.batch2_approach_temps, self.batch3_interactions)


class ControlPrefixFeatures:
    def __init__(self, dt: float = DT_SECONDS):
        self.dt = float(dt)
        self.n = 0

    def update(self, control: Sequence[float]) -> None:
        u = np.asarray(control, np.float64)
        if u.shape != (3,) or not np.isfinite(u).all():
            raise ValueError("control must contain finite [frequency, eev, outdoor_fan]")
        if self.n == 0:
            self.first = u.copy(); self.last = u.copy()
            self.total = u.copy(); self.total_sq = u * u
            self.minimum = u.copy(); self.maximum = u.copy()
            self.ewm_fast = u.copy(); self.ewm_slow = u.copy()
            self.abs_delta_total = np.zeros(3)
            self.on_steps = float(u[0] > 1.0)
            # Time-decaying integrals for compressor frequency (index 0).
            self.freq_integral_60 = 0.0
            self.freq_integral_300 = 0.0
            self.freq_integral_600 = 0.0
        else:
            delta = u - self.last
            self.abs_delta_total += np.abs(delta); self.last = u.copy()
            self.total += u; self.total_sq += u * u
            self.minimum = np.minimum(self.minimum, u)
            self.maximum = np.maximum(self.maximum, u)
            a_fast = 1.0 - np.exp(-self.dt / 30.0)
            a_slow = 1.0 - np.exp(-self.dt / 300.0)
            self.ewm_fast += a_fast * (u - self.ewm_fast)
            self.ewm_slow += a_slow * (u - self.ewm_slow)
            self.on_steps += float(u[0] > 1.0)
            # Decaying integrals: sum of freq * dt with exponential forgetting.
            for tau, attr in ((60.0, "freq_integral_60"),
                              (300.0, "freq_integral_300"),
                              (600.0, "freq_integral_600")):
                alpha = 1.0 - np.exp(-self.dt / tau)
                current = getattr(self, attr)
                setattr(self, attr, current + alpha * (u[0] - current))
        self.n += 1

    def features(self, config: FeatureConfig | None = None) -> np.ndarray:
        if self.n == 0:
            raise RuntimeError("update a control before requesting features")
        config = config or FeatureConfig()
        mean = self.total / self.n
        variance = np.maximum(0.0, self.total_sq / self.n - mean * mean)
        interactions = np.asarray([
            mean[0] * mean[2], mean[0] / (abs(mean[1]) + 20.0),
            self.on_steps / self.n,
        ])
        base = [
            self.last, self.first, mean, np.sqrt(variance), self.minimum,
            self.maximum, self.ewm_fast, self.ewm_slow,
            self.abs_delta_total / max(1, self.n - 1), interactions,
        ]
        if config.batch1_control_memory:
            base.extend([
                np.asarray([
                    self.freq_integral_60,
                    self.freq_integral_300,
                    self.freq_integral_600,
                    self.on_steps / self.n,
                    self.ewm_fast[0],
                    self.ewm_slow[0],
                ], np.float64),
            ])
        return np.concatenate(base).astype(np.float32)


def base_feature(initial_state: np.ndarray, prefix: ControlPrefixFeatures,
                 config: FeatureConfig | None = None) -> np.ndarray:
    elapsed = prefix.n * prefix.dt
    return np.concatenate([
        initial_state.astype(np.float32),
        np.asarray([elapsed / 3600.0, np.log1p(elapsed / 60.0)], np.float32),
        prefix.features(config),
    ])


class ContinuousFeatureState:
    """Validated 61-feature V3 thermal-inertia state."""

    def __init__(self, initial_state: np.ndarray):
        self.initial_state = np.asarray(initial_state, np.float32)
        self.base = ControlPrefixFeatures(DT_SECONDS)

    def update(self, control: Sequence[float]) -> None:
        self.base.update(control)

    def update_temperature(self, t_in: float) -> None:
        """Update the cached initial T_in (for stateful inference variants)."""
        self.initial_state[STATE_INDEX["T_in"]] = float(t_in)

    def feature(self) -> np.ndarray:
        state = self.initial_state
        elapsed = self.base.n * DT_SECONDS
        t_in0 = float(state[STATE_INDEX["T_in"]])
        t_set = float(state[STATE_INDEX["T_set"]])
        t_out = float(state[STATE_INDEX["T_out"]])
        t_coil = float(state[STATE_INDEX["T_in_coil"]])
        inertia = []
        for tau in (600.0, 1800.0, 3600.0):
            room_proxy = t_set + (t_in0 - t_set) * np.exp(-elapsed / tau)
            inertia.extend([room_proxy, t_out - room_proxy, room_proxy - t_coil])
        return np.concatenate([
            base_feature(state, self.base), np.asarray(inertia, np.float32),
        ]).astype(np.float32)


class AutoregressiveFeatureState:
    """Stateful plant-model feature builder.

    Uses only actual plant state and control history. Setpoints/targets
    (``T_set``, ``indoor_fan_target``, ``RH_target``, ``pid_*``) are
    intentionally excluded so the learned simulator remains a pure plant
    model and responds only to the three physical controls.
    """

    def __init__(self, initial_state: np.ndarray,
                 config: FeatureConfig | None = None):
        self.initial_state = np.asarray(initial_state, np.float32)
        self.current_state = self.initial_state.copy()
        self.base = ControlPrefixFeatures(DT_SECONDS)
        self.config = config or FeatureConfig()
        # Approach-temperature EWMs (batch 2).
        self.approach_in_coil_ewm = 0.0
        self.approach_out_coil_ewm = 0.0

    def update_control(self, control: Sequence[float]) -> None:
        self.base.update(control)

    def update(self, control: Sequence[float]) -> None:
        """Alias for ``update_control`` for API compatibility."""
        self.update_control(control)

    def update_temperature(self, t_in: float) -> None:
        self.current_state[PLANT_STATE_INDEX["T_in"]] = float(t_in)

    def feature(self) -> np.ndarray:
        state = self.current_state
        t_in = float(state[PLANT_STATE_INDEX["T_in"]])
        t_out = float(state[PLANT_STATE_INDEX["T_out"]])
        t_coil_in = float(state[PLANT_STATE_INDEX["T_in_coil"]])
        t_coil_out = float(state[PLANT_STATE_INDEX["T_out_coil"]])
        rh_in = float(state[PLANT_STATE_INDEX["RH_in"]])

        # Update approach-temperature EWMs when temperature is refreshed.
        approach_in = t_in - t_coil_in
        approach_out = t_out - t_coil_out
        a_fast = 1.0 - np.exp(-self.base.dt / 30.0)
        a_slow = 1.0 - np.exp(-self.base.dt / 300.0)
        if self.base.n > 0:
            self.approach_in_coil_ewm += a_fast * (approach_in - self.approach_in_coil_ewm)
            self.approach_out_coil_ewm += a_slow * (approach_out - self.approach_out_coil_ewm)

        deltas = np.asarray([
            t_in - t_out,
            t_in - t_coil_in,
            t_in - t_coil_out,
            t_out - t_coil_out,
        ], np.float32)
        elapsed = self.base.n * self.base.dt
        features = [
            state.astype(np.float32),
            deltas,
            np.asarray([elapsed / 3600.0, np.log1p(elapsed / 60.0)], np.float32),
            self.base.features(self.config),
        ]
        if self.config.batch2_approach_temps:
            features.append(np.asarray([
                self.approach_in_coil_ewm,
                self.approach_out_coil_ewm,
            ], np.float32))
        if self.config.batch3_interactions:
            # Interactions use current state and latest control.
            freq = float(self.base.last[0]) if self.base.n > 0 else 0.0
            features.append(np.asarray([
                (t_in - t_out) * freq,
                rh_in * approach_in,
            ], np.float32))
        return np.concatenate(features).astype(np.float32)


# Backward-compatible internal name while V3 files are migrated.
V3FeatureState = ContinuousFeatureState
