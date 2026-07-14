"""Hybrid room simulator: reset once, then drive with three actuator inputs.

This package keeps the same online usage style as ``simu.room`` and
the earlier room V5 work.  The room-temperature output uses the strongest verified
strict-open-loop backbone:

    V5 slow grey-box room model + causal residual TCN

Auxiliary AC temperatures are provided by a local copied world model so callers can still
inspect indoor coil, outdoor coil and discharge temperatures.  After ``reset``,
the simulator only consumes compressor frequency, EEV opening and outdoor fan
speed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .world_simulator import DEFAULT_MODEL_PATH as ROOM_WORLD_DEFAULT_MODEL
from .world_simulator import RoomWorldEnv
from .residual_simulator import (
    DEFAULT_RESIDUAL_TCN_PATH,
    ResidualTCNRoomV5Env,
    residual_gate_from_controls,
)
from .v5_model import CONTROL_COLUMNS
from .v5_simulator import DEFAULT_V5_MODEL_PATH
from .residual_features import build_features


HERE = Path(__file__).resolve().parent
DEFAULT_ROOM_MODEL_PATH = DEFAULT_V5_MODEL_PATH
DEFAULT_RESIDUAL_MODEL_PATH = DEFAULT_RESIDUAL_TCN_PATH
DEFAULT_AUX_MODEL_PATH = HERE / "models" / "room_world_aux_v6_context.pt"


class HybridRoomEnv:
    """Online hybrid room environment prioritizing ``T_in`` accuracy."""

    def __init__(
        self,
        room_model_path: str | Path = DEFAULT_ROOM_MODEL_PATH,
        residual_model_path: str | Path = DEFAULT_RESIDUAL_MODEL_PATH,
        aux_model_path: str | Path = DEFAULT_AUX_MODEL_PATH,
        device: str = "cpu",
        control_delay_seconds: float = 0.0,
        residual_base_weight: float = 1.0,
    ):
        self.room = ResidualTCNRoomV5Env(
            slow_model_path=room_model_path,
            residual_model_path=residual_model_path,
            device=device,
            control_delay_seconds=control_delay_seconds,
            residual_base_weight=residual_base_weight,
        )
        aux_path = Path(aux_model_path)
        if not aux_path.exists():
            aux_path = ROOM_WORLD_DEFAULT_MODEL
        self.aux = RoomWorldEnv(aux_path, device=device)
        self.room_model_path = Path(room_model_path)
        self.residual_model_path = Path(residual_model_path)
        self.aux_model_path = aux_path
        self.control_delay_seconds = float(control_delay_seconds)
        self.residual_base_weight = float(residual_base_weight)
        self._ready = False

    def reset(
        self,
        initial_observation: Mapping[str, float] | pd.Series | None = None,
        **initial_values: float,
    ) -> dict:
        """Initialize from the first data row.

        ``T_set`` and other target-like fields may be present in the row, but
        they are not used as future inputs.  Missing optional reset features are
        filled by each underlying model's training medians.
        """
        observation = dict(initial_observation) if initial_observation is not None else {}
        observation.update(initial_values)
        room = self.room.reset(observation)
        aux = self.aux.reset(observation)
        self._ready = True
        return self._merge(room, aux)

    def step(
        self,
        compressor_frequency: float,
        eev_opening: float,
        outdoor_fan_speed: float,
    ) -> dict:
        """Advance one 5-second step with only the three actuator inputs."""
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        room = self.room.step(compressor_frequency, eev_opening, outdoor_fan_speed)
        aux = self.aux.step(compressor_frequency, eev_opening, outdoor_fan_speed)
        return self._merge(room, aux)

    def _merge(self, room: dict, aux: dict) -> dict:
        merged = dict(aux)
        merged.update({
            "T_in": float(room["T_in"]),
            "T_in_room_backbone": float(room["T_in"]),
            "T_in_aux_backbone": float(aux["T_in"]),
            "T_in_slow": float(room.get("T_in_slow", room["T_in"])),
            "T_in_residual": float(room.get("T_in_residual", 0.0)),
            "T_in_residual_raw": float(room.get("T_in_residual_raw", 0.0)),
            "residual_weight": float(room.get("residual_weight", 1.0)),
            "room_control_delay_seconds": self.control_delay_seconds,
            "residual_base_weight": self.residual_base_weight,
        })
        return merged

    def simulate(self, initial_observation, controls) -> pd.DataFrame:
        """Roll out a complete action sequence.

        ``controls`` must be ordered as ``freq, eev, fan_out``.
        """
        observation = dict(initial_observation)
        actions = np.asarray(controls, np.float32)
        slow_result = self.room.slow.simulate(observation, actions)
        aux_result = self.aux.simulate(observation, actions)
        n = min(len(slow_result), len(aux_result))
        control_frame = self._control_frame(observation, actions, n)
        control_mean = self.room.slow.model.control_mean.detach().cpu().numpy().astype(np.float32)
        control_scale = self.room.slow.model.control_scale.detach().cpu().numpy().astype(np.float32)
        features = build_features(
            control_frame.iloc[:n],
            slow_result.iloc[:n],
            control_mean,
            control_scale,
        )
        with torch.no_grad():
            tensor = torch.tensor(features[None], device=self.room.device)
            residual_raw = self.room.residual_model(tensor)[0].detach().cpu().numpy().astype(np.float32)
        weights = residual_gate_from_controls(
            control_frame.iloc[:n],
            self.residual_base_weight,
        )
        result = aux_result.iloc[:n].copy()
        slow_temperature = slow_result["T_in"].to_numpy(np.float32)[:n]
        residual = residual_raw[:n] * weights[:n]
        result["T_in"] = slow_temperature + residual
        result["T_in_room_backbone"] = result["T_in"]
        result["T_in_aux_backbone"] = aux_result["T_in"].to_numpy(np.float32)[:n]
        result["T_in_slow"] = slow_temperature
        result["T_in_residual"] = residual
        result["T_in_residual_raw"] = residual_raw[:n]
        result["residual_weight"] = weights[:n]
        result["room_control_delay_seconds"] = self.control_delay_seconds
        result["residual_base_weight"] = self.residual_base_weight
        return result.reset_index(drop=True)

    def _control_frame(
        self,
        initial_observation: Mapping[str, float],
        actions: np.ndarray,
        length: int,
    ) -> pd.DataFrame:
        if len(actions) > 0:
            controls = actions.astype(np.float32, copy=True)
            if len(controls) < length:
                controls = np.vstack([controls, controls[-1:]])
            controls = controls[:length]
        else:
            fallback = {
                "control_frequency": initial_observation.get("compressor_frequency", 0.0),
                "control_eev": initial_observation.get("eev_opening", 0.0),
                "control_fan_out": initial_observation.get("outdoor_fan_speed", 0.0),
            }
            initial = []
            for name in CONTROL_COLUMNS:
                value = initial_observation.get(name, fallback[name])
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float(fallback[name])
                initial.append(value if np.isfinite(value) else float(fallback[name]))
            controls = np.repeat(np.asarray(initial, np.float32)[None], length, axis=0)
        return pd.DataFrame(controls, columns=CONTROL_COLUMNS)


RoomHybridEnv = HybridRoomEnv
