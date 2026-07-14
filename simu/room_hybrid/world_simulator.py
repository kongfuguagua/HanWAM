"""Open-loop online simulator for the room world model."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .api import ACTION_COLUMNS, RESET_COLUMNS
from .data import OUTPUT_COLUMNS
from .world_model import RoomWorldModel, RoomWorldModelConfig


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "models" / "room_world_aux_v6_context.pt"


def load_room_world_model(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "haier_room_world_v1":
        raise ValueError("not a supported room world model file")
    if payload.get("reset_columns") != RESET_COLUMNS:
        raise ValueError("reset columns do not match current code")
    if payload.get("action_columns") != ACTION_COLUMNS:
        raise ValueError("action columns do not match current code")
    if payload.get("output_columns") != OUTPUT_COLUMNS:
        raise ValueError("output columns do not match current code")
    norm = {
        key: np.asarray(value, np.float32)
        for key, value in payload["normalization"].items()
    }
    model = RoomWorldModel(
        **norm, config=RoomWorldModelConfig(**payload["config"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, payload


class RoomWorldEnv:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model, self.payload = load_room_world_model(model_path, self.device)
        self._ready = False

    def reset(
        self,
        initial_observation: Mapping[str, float] | pd.Series | None = None,
        **initial_values: float,
    ) -> dict:
        observation = dict(initial_observation) if initial_observation is not None else {}
        observation.update(initial_values)
        medians = self.model.reset_mean.detach().cpu().numpy()
        values = np.asarray([
            observation.get(name, medians[index])
            for index, name in enumerate(RESET_COLUMNS)
        ], np.float32)
        values = np.where(np.isfinite(values), values, medians).astype(np.float32)
        with torch.no_grad():
            self.state = self.model.initialize(torch.tensor(values[None], device=self.device))
        self._ready = True
        return self._observation()

    def step(
        self,
        compressor_frequency: float,
        eev_opening: float,
        outdoor_fan_speed: float,
    ) -> dict:
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        action = np.asarray([compressor_frequency, eev_opening, outdoor_fan_speed], np.float32)
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN/Inf")
        with torch.no_grad():
            self.state = self.model.step_state(
                self.state, torch.tensor(action[None], device=self.device),
            )
        return self._observation()

    def _scalar(self, key: str) -> float:
        return float(self.state[key].detach().cpu().item())

    def _observation(self) -> dict:
        outputs = self.state["outputs"].detach().cpu().numpy()[0]
        result = {
            "elapsed_seconds": self._scalar("elapsed_steps") * 5.0,
            "T_out": self._scalar("outdoor_temperature"),
        }
        result.update({
            name: float(outputs[index])
            for index, name in enumerate(OUTPUT_COLUMNS)
        })
        if "last_load" in self.state:
            result.update({
                "normalized_sensible_load": self._scalar("last_load"),
                "normalized_cooling_capacity": self._scalar("cooling"),
                "room_tau_seconds": self._scalar("last_room_tau_seconds"),
                "cooling_tau_seconds": self._scalar("last_cooling_tau_seconds"),
                "room_residual_rate_c_per_min": self._scalar("last_residual_rate_c_per_min"),
                "load_gain": self._scalar("last_load_gain"),
                "capacity_gain": self._scalar("last_capacity_gain"),
                "room_tau_gain": self._scalar("last_room_tau_gain"),
                "cooling_tau_gain": self._scalar("last_cooling_tau_gain"),
            })
            if "last_room_bias" in self.state:
                result["room_bias"] = self._scalar("last_room_bias")
                result["room_bias_target"] = self._scalar("room_bias_target")
                result["room_bias_tau_seconds"] = self._scalar("room_bias_tau_seconds")
        return result

    def simulate(self, initial_observation, actions) -> pd.DataFrame:
        rows = [self.reset(initial_observation)]
        for freq, eev, fan in np.asarray(actions, np.float32):
            rows.append(self.step(float(freq), float(eev), float(fan)))
        return pd.DataFrame(rows)
