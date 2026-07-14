"""Air-conditioner actuator simulator.

The controller commands target actions [freq_target, eev, fan_out].  This
simulator turns them into actual compressor frequency and electric power.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simu.energy.model import EnergyModel
from simu.frequency.freq_response_model import (
    FREQ_MAX,
    FREQ_MIN,
    ON_THRESHOLD,
    classify_state,
    simulate_freq,
)


TARGET_ACTION_COLS = ["freq_target", "eev", "fan_out"]
ACTUAL_ACTION_COLS = ["freq", "eev", "fan_out"]


@dataclass
class AirConditionerState:
    elapsed_seconds: float
    freq_target: float
    freq: float
    eev: float
    fan_out: float
    power_w: float
    electric_kwh: float

    def as_dict(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "freq_target": self.freq_target,
            "freq": self.freq,
            "eev": self.eev,
            "fan_out": self.fan_out,
            "power_w": self.power_w,
            "electric_kwh": self.electric_kwh,
        }


class AirConditionerSimulator:
    """Stateful AC simulator for one 5-second control loop."""

    def __init__(
        self,
        mode: int | str = 1,
        step_seconds: float = 5.0,
        freq_cap: float | None = 80.0,
        energy_model: EnergyModel | None = None,
        freq_params: dict | None = None,
    ):
        self.mode = mode
        self.step_seconds = float(step_seconds)
        self.freq_cap = freq_cap
        self.energy_model = energy_model or EnergyModel(mode=mode, step_seconds=step_seconds)
        self.freq_params = freq_params or {}
        self.reset()

    def reset(self, freq0: float = 0.0, target0: float | None = None) -> dict:
        self.elapsed_seconds = 0.0
        self.freq0 = float(freq0)
        self.target0 = float(freq0 if target0 is None else target0)
        self.target_history: list[float] = []
        self.last_freq = self.freq0
        self.last_target = self.target0
        self.dead_cold_remaining = 0
        self.dead_steady_remaining = 0
        # Reset the energy model's internal low-frequency EMA state so each
        # trajectory starts fresh.
        try:
            self.energy_model.reset()
        except AttributeError:
            pass
        self.last_output = AirConditionerState(
            elapsed_seconds=0.0,
            freq_target=self.target0,
            freq=self.freq0,
            eev=0.0,
            fan_out=0.0,
            power_w=0.0,
            electric_kwh=0.0,
        )
        return self.last_output.as_dict()

    def _actual_freq(self, target_freq: float) -> float:
        target = float(target_freq)
        params = {
            "tau_cold": 6.7,
            "tau_up": 15.0,
            "tau_down": 10.0,
            "tau_clamp": 1000.0,
            "I_max": 3.0,
            "on_threshold_hz": ON_THRESHOLD,
            "t_dead_cold_sec": 5.0,
            "t_dead_steady_sec": 0.0,
            "freq_max": FREQ_MAX,
        }
        params.update(self.freq_params)
        dt = self.step_seconds
        on_threshold = float(params["on_threshold_hz"])
        is_off_to_on = self.last_target < on_threshold and target >= on_threshold
        is_on_to_off = self.last_target >= on_threshold and target < on_threshold
        is_jump = abs(target - self.last_target) > 5.0
        if is_off_to_on:
            self.dead_cold_remaining = int(float(params["t_dead_cold_sec"]) / dt)
        elif is_jump and not is_on_to_off:
            self.dead_steady_remaining = int(float(params["t_dead_steady_sec"]) / dt)

        if self.dead_cold_remaining > 0:
            freq = self.last_freq
            self.dead_cold_remaining -= 1
        elif self.dead_steady_remaining > 0:
            freq = self.last_freq
            self.dead_steady_remaining -= 1
        else:
            state = classify_state(
                self.last_freq,
                self.last_target,
                target,
                I_comp=0.0,
                I_max=float(params["I_max"]),
                freq_cap=self.freq_cap,
                on_threshold=on_threshold,
            )
            if state == "off":
                freq = 0.0
            elif state == "cold_start":
                tau = float(params["tau_cold"])
                freq = self.last_freq + dt / tau * (target - self.last_freq)
            elif state == "up":
                tau = float(params["tau_up"])
                freq = self.last_freq + dt / tau * (target - self.last_freq)
            elif state == "down":
                tau = float(params["tau_down"])
                freq = self.last_freq + dt / tau * (target - self.last_freq)
            elif state == "clamped":
                freq = min(self.last_freq, self.freq_cap - 0.5) if self.freq_cap else self.last_freq
            else:
                freq = self.last_freq
            freq_max = float(params["freq_max"])
            freq = max(FREQ_MIN, min(freq_max, freq))
        self.target_history.append(target)
        self.last_target = target
        self.last_freq = float(freq)
        return self.last_freq

    def step(self, action: dict | list[float] | np.ndarray) -> dict:
        if isinstance(action, dict):
            freq_target = float(action.get("freq_target", action.get("freq_in_tgt")))
            eev = float(action["eev"])
            fan_out = float(action["fan_out"])
        else:
            freq_target, eev, fan_out = np.asarray(action, dtype=np.float32).tolist()

        freq = self._actual_freq(freq_target)
        actual_action = np.asarray([freq, eev, fan_out], dtype=np.float32)
        power_w = float(self.energy_model.predict_power(actual_action)[0])
        electric_kwh = power_w * self.step_seconds / 3600_000.0
        self.elapsed_seconds += self.step_seconds
        self.last_output = AirConditionerState(
            elapsed_seconds=self.elapsed_seconds,
            freq_target=float(freq_target),
            freq=freq,
            eev=float(eev),
            fan_out=float(fan_out),
            power_w=power_w,
            electric_kwh=electric_kwh,
        )
        return self.last_output.as_dict()

    def simulate(self, actions: np.ndarray, freq0: float = 0.0, target0: float | None = None) -> pd.DataFrame:
        self.reset(freq0=freq0, target0=target0)
        rows = [self.step(action) for action in np.asarray(actions, dtype=np.float32)]
        return pd.DataFrame(rows)
