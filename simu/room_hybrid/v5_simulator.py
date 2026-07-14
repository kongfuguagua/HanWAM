"""Online V5 room simulator: reset once, then only three actuator inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .v5_model import CONTROL_COLUMNS, INITIAL_COLUMNS, HybridRoomV5, V5Config


HERE = Path(__file__).resolve().parent
DEFAULT_V5_MODEL_PATH = HERE / "models" / "room_v5_model.pt"
IGNORED_TARGETS = {
    "T_set", "RH_target", "indoor_fan_target", "pid_freq", "pid_target",
    "inference_freq", "inference_eev", "inference_fan_out", "inference_fan_in",
}


def load_v5_model(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "haier_room_v5_torch_state_dict":
        raise ValueError("不是受支持的 V5 模型文件")
    if payload.get("initial_columns") != INITIAL_COLUMNS:
        raise ValueError("V5 初始化列与当前代码不一致")
    if payload.get("control_columns") != CONTROL_COLUMNS:
        raise ValueError("V5 控制列与当前代码不一致")
    norm = {key: np.asarray(value, np.float32)
            for key, value in payload["normalization"].items()}
    model = HybridRoomV5(
        **norm, config=V5Config(**payload["config"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, payload


class HybridRoomV5Env:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_V5_MODEL_PATH,
        device: str | torch.device = "cpu",
        control_delay_seconds: float = 0.0,
    ):
        self.device = torch.device(device)
        self.model, self.payload = load_v5_model(model_path, self.device)
        self.control_delay_seconds = float(control_delay_seconds)
        if self.control_delay_seconds < 0:
            raise ValueError("control_delay_seconds 必须 >= 0")
        self._delay_steps = int(round(self.control_delay_seconds / 5.0))
        self._ready = False

    def reset(
        self,
        initial_observation: Mapping[str, float] | pd.Series | None = None,
        **initial_values: float,
    ) -> dict:
        observation = dict(initial_observation) if initial_observation is not None else {}
        observation.update(initial_values)
        if "T_out" not in observation or "T_in" not in observation:
            raise ValueError("V5 初始化至少需要 T_out 和 T_in")
        medians = self.model.initial_mean.detach().cpu().numpy()
        values = np.asarray([
            observation.get(name, medians[index])
            for index, name in enumerate(INITIAL_COLUMNS)
        ], np.float32)
        values = np.where(np.isfinite(values), values, medians).astype(np.float32)
        self.initial_values = values
        self._delay_queue = [
            self._initial_control(observation, values)
            for _ in range(self._delay_steps)
        ]
        initial = torch.tensor(values[None], device=self.device)
        with torch.no_grad():
            self.state = self.model.initialize(initial)
        self._ready = True
        return self._observation()

    def step(self, freq: float, eev: float, fan_out: float) -> dict:
        if not self._ready:
            raise RuntimeError("请先调用 reset()")
        control = np.asarray([freq, eev, fan_out], np.float32)
        if not np.isfinite(control).all():
            raise ValueError("控制量含 NaN/Inf")
        applied_control = self._delayed_control(control)
        with torch.no_grad():
            tensor = torch.tensor(applied_control[None], device=self.device)
            self.state = self.model.step_state(self.state, tensor)
        return self._observation()

    def _initial_control(self, observation: Mapping[str, float], values: np.ndarray) -> np.ndarray:
        actual_columns = ["compressor_frequency", "eev_opening", "outdoor_fan_speed"]
        result = []
        for index, (control_name, actual_name) in enumerate(zip(CONTROL_COLUMNS, actual_columns)):
            actual_value = values[INITIAL_COLUMNS.index(actual_name)]
            value = observation.get(control_name, actual_value)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float(actual_value)
            if not np.isfinite(value):
                value = float(actual_value)
            result.append(value)
        return np.asarray(result, np.float32)

    def _delayed_control(self, control: np.ndarray) -> np.ndarray:
        if self._delay_steps <= 0:
            return control
        self._delay_queue.append(control.copy())
        return self._delay_queue.pop(0)

    def _scalar(self, key: str) -> float:
        return float(self.state[key].detach().cpu().item())

    def _observation(self) -> dict:
        result = {
            "elapsed_seconds": self._scalar("elapsed_steps") * 5.0,
            "T_in": self._scalar("temperature"),
            "T_in_plant": self._scalar("plant_temperature"),
            "T_out": self._scalar("outdoor_temperature"),
            "load_coefficient": self._scalar("load_coefficient"),
            "normalized_cooling_capacity": self._scalar("cooling"),
            "control_delay_seconds": self.control_delay_seconds,
        }
        if "last_load" in self.state:
            result["normalized_sensible_load"] = self._scalar("last_load")
            result["cooling_tau_seconds"] = self._scalar("last_tau_seconds")
            result["room_tau_seconds"] = self._scalar("last_room_tau_seconds")
            result["residual_rate_c_per_min"] = self._scalar(
                "last_residual_rate_c_per_min"
            )
        return result

    def simulate(self, initial_observation, controls) -> pd.DataFrame:
        rows = [self.reset(initial_observation)]
        for freq, eev, fan_out in np.asarray(controls, np.float32):
            rows.append(self.step(float(freq), float(eev), float(fan_out)))
        return pd.DataFrame(rows)


# Backward-compatible aliases for callers that prefer older names.
RoomV5Env = HybridRoomV5Env
HybridRoomV4Env = HybridRoomV5Env
RoomV4Env = HybridRoomV5Env
DEFAULT_V4_MODEL_PATH = DEFAULT_V5_MODEL_PATH
load_v4_model = load_v5_model
