"""Room/environment simulator with enthalpy-proxy thermal capacity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simu.temperature.simulator import OnlineEnthalpyRoomEnv

try:
    from simu.room.simulator import ContinuousEnthalpyRoomEnv
except Exception:  # pragma: no cover - V3 model artifact may be absent in minimal installs.
    ContinuousEnthalpyRoomEnv = None

try:
    from simu.room_hybrid import HybridRoomEnv
except Exception:  # pragma: no cover - hybrid artifacts may be absent in minimal installs.
    HybridRoomEnv = None


ENV_OBS_COLS = ["T_out", "T_out_coil", "T_out_discharge", "T_in", "T_in_coil", "RH_in", "fan_in"]


@dataclass
class EnvironmentState:
    elapsed_seconds: float
    T_out: float
    T_out_coil: float
    T_out_discharge: float
    T_in: float
    T_in_coil: float
    RH_in: float
    fan_in: float
    thermal_power_w: float
    thermal_kwh: float
    model_spread: float

    def as_dict(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "T_out": self.T_out,
            "T_out_coil": self.T_out_coil,
            "T_out_discharge": self.T_out_discharge,
            "T_in": self.T_in,
            "T_in_coil": self.T_in_coil,
            "RH_in": self.RH_in,
            "fan_in": self.fan_in,
            "thermal_power_w": self.thermal_power_w,
            "thermal_kwh": self.thermal_kwh,
            "model_spread": self.model_spread,
        }


def _is_heating(mode: int | str) -> bool:
    return str(mode) in {"3", "3.0", "制热", "heat", "heating"}


def _humidity_ratio(temp_c: float, rh: float, pressure_pa: float = 101_325.0) -> float:
    rh = float(rh)
    if rh > 1.5:
        rh /= 100.0
    rh = float(np.clip(rh, 0.01, 1.0))
    temp_c = float(temp_c)
    p_ws = 610.94 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
    p_w = np.clip(rh * p_ws, 1.0, pressure_pa * 0.95)
    return float(0.62198 * p_w / (pressure_pa - p_w))


def moist_air_enthalpy_kj_per_kg(temp_c: float, rh: float) -> float:
    w = _humidity_ratio(temp_c, rh)
    return float(1.006 * temp_c + w * (2501.0 + 1.86 * temp_c))


class EnvironmentSimulator:
    """Stateful environment simulator.

    It reuses the validated online temperature model for T_in and estimates
    thermal output from room/coil enthalpy difference.  The coil temperature is
    the measured initial coil state proxy because the current temperature model
    only predicts room temperature.
    """

    def __init__(
        self,
        mode: int | str = 1,
        step_seconds: float = 5.0,
        max_airflow_m3h: float = 650.0,
        rated_fan_rpm: float = 1050.0,
        air_density_kg_m3: float = 1.2,
        room_heat_capacity_kj_per_c: float = 90.0,
        passive_heat_tau_seconds: float = 14_400.0,
        temperature_model: str | None = None,
        room_hybrid_kwargs: dict | None = None,
        temperature_env: object | None = None,
    ):
        self.mode = mode
        self.step_seconds = float(step_seconds)
        self.max_airflow_m3h = float(max_airflow_m3h)
        self.rated_fan_rpm = float(rated_fan_rpm)
        self.air_density_kg_m3 = float(air_density_kg_m3)
        self.room_heat_capacity_kj_per_c = float(room_heat_capacity_kj_per_c)
        self.passive_heat_tau_seconds = float(passive_heat_tau_seconds)
        requested_temperature_model = str(temperature_model or "auto")
        if temperature_env is not None:
            self.temperature_env = temperature_env
            self.temperature_model = "custom"
        elif requested_temperature_model == "room_hybrid":
            if HybridRoomEnv is None:
                raise RuntimeError("simulator.environment.temperature_model=room_hybrid requires simu.room_hybrid artifacts")
            self.temperature_env = HybridRoomEnv(**(room_hybrid_kwargs or {}))
            self.temperature_model = "room_hybrid"
        elif str(mode) in {"1", "1.0"} and ContinuousEnthalpyRoomEnv is not None:
            self.temperature_env = ContinuousEnthalpyRoomEnv(
                passive_heat_tau_seconds=self.passive_heat_tau_seconds
            )
            self.temperature_model = "room_v3"
        else:
            self.temperature_env = OnlineEnthalpyRoomEnv()
            self.temperature_model = "enthalpy_v2"
        self._ready = False

    def reset(self, initial_state: dict | list[float] | np.ndarray) -> dict:
        if isinstance(initial_state, dict):
            self.T_out = float(initial_state["T_out"])
            self.T_in = float(initial_state["T_in"])
            self.T_out_coil = float(initial_state["T_out_coil"])
            self.T_out_discharge = float(initial_state.get("T_out_discharge", self.T_out))
            self.T_in_coil = float(initial_state["T_in_coil"])
            self.RH_in = float(initial_state.get("RH_in", 0.6))
            self.fan_in = float(initial_state.get("fan_in", self.rated_fan_rpm))
        else:
            raw = np.asarray(initial_state, dtype=np.float32)
            if raw.shape[0] < 4:
                raise ValueError("initial_state must contain at least T_out, T_in, T_out_coil, T_in_coil")
            self.T_out, self.T_in, self.T_out_coil, self.T_in_coil = raw[:4].tolist()
            self.T_out_discharge = self.T_out
            self.RH_in = float(raw[4]) if raw.shape[0] > 4 else 0.6
            self.fan_in = float(raw[5]) if raw.shape[0] > 5 else self.rated_fan_rpm

        if not np.isfinite(self.fan_in) or self.fan_in <= 1.0:
            self.fan_in = self.rated_fan_rpm

        self.elapsed_seconds = 0.0
        if self.temperature_model == "room_hybrid":
            if isinstance(initial_state, dict):
                temp = self.temperature_env.reset(initial_observation=initial_state)
            else:
                temp = self.temperature_env.reset(
                    initial_observation={
                        "T_out": self.T_out,
                        "T_in": self.T_in,
                        "T_out_coil": self.T_out_coil,
                        "T_out_discharge": self.T_out_discharge,
                        "T_in_coil": self.T_in_coil,
                        "RH_in": self.RH_in,
                        "fan_in": self.fan_in,
                        "mode": float(self.mode),
                    }
                )
            self.T_in = float(temp.get("T_in", self.T_in))
            self.T_out = float(temp.get("T_out", self.T_out))
            self.T_out_coil = float(temp.get("T_out_coil", self.T_out_coil))
            self.T_out_discharge = float(temp.get("T_out_discharge", self.T_out_discharge))
            self.T_in_coil = float(temp.get("T_in_coil", self.T_in_coil))
        elif self.temperature_model == "room_v3":
            if isinstance(initial_state, dict):
                self.temperature_env.reset(initial_observation=initial_state)
            else:
                self.temperature_env.reset(self.T_out, self.T_in, self.T_out_coil, self.T_in_coil, mode=1)
        else:
            self.temperature_env.reset(self.T_out, self.T_in, self.T_out_coil, self.T_in_coil)
        self._ready = True
        state = EnvironmentState(
            elapsed_seconds=0.0,
            T_out=self.T_out,
            T_out_coil=self.T_out_coil,
            T_out_discharge=self.T_out_discharge,
            T_in=self.T_in,
            T_in_coil=self.T_in_coil,
            RH_in=self.RH_in,
            fan_in=self.fan_in,
            thermal_power_w=0.0,
            thermal_kwh=0.0,
            model_spread=0.0,
        )
        return state.as_dict()

    def _thermal_power(
        self,
        T_in: float,
        previous_T_in: float | None = None,
        compressor_freq: float | None = None,
        electric_power_w: float | None = None,
    ) -> float:
        if compressor_freq is not None and electric_power_w is not None:
            if float(compressor_freq) <= 1.0 and float(electric_power_w) <= 1.0:
                return 0.0
        fan_ratio = np.clip(self.fan_in / self.rated_fan_rpm, 0.0, 1.25)
        airflow_m3s = self.max_airflow_m3h * fan_ratio / 3600.0
        mass_flow = airflow_m3s * self.air_density_kg_m3
        room_h = moist_air_enthalpy_kj_per_kg(T_in, self.RH_in)
        coil_h = moist_air_enthalpy_kj_per_kg(self.T_in_coil, self.RH_in)
        if _is_heating(self.mode):
            thermal_kw = max(coil_h - room_h, 0.0) * mass_flow
            if previous_T_in is not None:
                delta_kw = max(T_in - previous_T_in, 0.0) * self.room_heat_capacity_kj_per_c / self.step_seconds
                thermal_kw = max(thermal_kw, delta_kw)
        else:
            thermal_kw = max(room_h - coil_h, 0.0) * mass_flow
            if previous_T_in is not None:
                delta_kw = max(previous_T_in - T_in, 0.0) * self.room_heat_capacity_kj_per_c / self.step_seconds
                thermal_kw = max(thermal_kw, delta_kw)
        return float(thermal_kw * 1000.0)

    def step(self, ac_output: dict) -> dict:
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        previous_T_in = self.T_in
        if self.temperature_model == "room_hybrid":
            temp = self.temperature_env.step(
                float(ac_output["freq"]),
                float(ac_output["eev"]),
                float(ac_output["fan_out"]),
            )
        else:
            temp = self.temperature_env.step(
                freq=float(ac_output["freq"]),
                eev=float(ac_output["eev"]),
                fan_out=float(ac_output["fan_out"]),
            )
        self.elapsed_seconds = float(temp["elapsed_seconds"])
        self.T_in = float(temp["T_in"])
        self.T_out = float(temp.get("T_out", self.T_out))
        self.T_out_coil = float(temp.get("T_out_coil", self.T_out_coil))
        self.T_in_coil = float(temp.get("T_in_coil", self.T_in_coil))
        freq = max(float(ac_output.get("freq", 0.0)), 0.0)
        if "T_out_discharge" in temp:
            self.T_out_discharge = float(temp["T_out_discharge"])
        else:
            target_discharge = self.T_out + 0.45 * freq
            alpha = np.clip(self.step_seconds / 120.0, 0.0, 1.0)
            self.T_out_discharge = float(self.T_out_discharge + alpha * (target_discharge - self.T_out_discharge))
        thermal_power_w = self._thermal_power(
            self.T_in,
            previous_T_in=previous_T_in,
            compressor_freq=float(ac_output.get("freq", np.nan)),
            electric_power_w=float(ac_output.get("power_w", np.nan)),
        )
        thermal_kwh = thermal_power_w * self.step_seconds / 3600_000.0
        state = EnvironmentState(
            elapsed_seconds=self.elapsed_seconds,
            T_out=self.T_out,
            T_out_coil=self.T_out_coil,
            T_out_discharge=self.T_out_discharge,
            T_in=self.T_in,
            T_in_coil=self.T_in_coil,
            RH_in=self.RH_in,
            fan_in=self.fan_in,
            thermal_power_w=thermal_power_w,
            thermal_kwh=thermal_kwh,
            model_spread=float(temp.get("model_spread", 0.0)),
        )
        return state.as_dict()

    def simulate(self, ac_outputs: list[dict] | pd.DataFrame, initial_state: dict | list[float] | np.ndarray) -> pd.DataFrame:
        self.reset(initial_state)
        rows = []
        records = ac_outputs.to_dict("records") if isinstance(ac_outputs, pd.DataFrame) else ac_outputs
        for ac_output in records:
            rows.append(self.step(ac_output))
        return pd.DataFrame(rows)
