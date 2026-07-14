"""Online simulator that adds a causal residual TCN to the V5 slow branch."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .v5_model import CONTROL_COLUMNS
from .residual_tcn import ResidualTCN, ResidualTCNConfig
from .v5_simulator import DEFAULT_V5_MODEL_PATH, HybridRoomV5Env
from .residual_features import FEATURE_COLUMNS, build_features


HERE = Path(__file__).resolve().parent
DEFAULT_RESIDUAL_TCN_PATH = HERE / "models" / "room_v5_residual_tcn.pt"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def residual_gate_from_controls(
    control_frame: pd.DataFrame,
    base_weight: float = 1.0,
) -> np.ndarray:
    """Down-weight residuals in steady high-load regions.

    The residual TCN is most useful in low-load/PID maintenance.  Stable
    high-frequency regions can be handled better by the slow branch, so this
    gate reduces residual authority when compressor frequency is high, its
    short-term variance is low, and outdoor fan speed is not high enough to
    resemble the pidA high-airflow condition.
    """
    frequency = control_frame["control_frequency"].rolling(
        121, min_periods=1,
    ).mean().to_numpy(np.float32)
    fan = control_frame["control_fan_out"].rolling(
        121, min_periods=1,
    ).mean().to_numpy(np.float32)
    frequency_std = control_frame["control_frequency"].rolling(
        121, min_periods=2,
    ).std().fillna(0.0).to_numpy(np.float32)
    penalty = (
        _sigmoid((frequency - 22.0) / 4.0)
        * _sigmoid((750.0 - fan) / 40.0)
        * _sigmoid((7.0 - frequency_std) / 1.5)
    )
    weight = base_weight * (1.0 - penalty)
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def load_residual_tcn(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "haier_room_v5_residual_tcn":
        raise ValueError("not a supported V5 residual TCN model file")
    if payload.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("residual TCN feature columns do not match current code")
    model = ResidualTCN(ResidualTCNConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


class ResidualTCNRoomV5Env:
    """Strict open-loop slow V5 plus causal residual correction."""

    def __init__(
        self,
        slow_model_path: str | Path = DEFAULT_V5_MODEL_PATH,
        residual_model_path: str | Path = DEFAULT_RESIDUAL_TCN_PATH,
        device: str | torch.device = "cpu",
        control_delay_seconds: float = 0.0,
        residual_base_weight: float = 1.0,
    ):
        self.device = torch.device(device)
        self.slow = HybridRoomV5Env(
            slow_model_path,
            device=self.device,
            control_delay_seconds=control_delay_seconds,
        )
        self.residual_model, self.residual_payload = load_residual_tcn(
            residual_model_path, self.device,
        )
        self.control_delay_seconds = float(control_delay_seconds)
        self.residual_base_weight = float(residual_base_weight)
        self._ready = False

    def reset(
        self,
        initial_observation: Mapping[str, float] | pd.Series | None = None,
        **initial_values: float,
    ) -> dict:
        observation = dict(initial_observation) if initial_observation is not None else {}
        observation.update(initial_values)
        self._slow_rows = [self.slow.reset(observation)]
        self._controls = [self._initial_control(observation)]
        self._ready = True
        return self._observation()

    def step(self, freq: float, eev: float, fan_out: float) -> dict:
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        self._slow_rows.append(self.slow.step(freq, eev, fan_out))
        self._controls.append(np.asarray([freq, eev, fan_out], np.float32))
        return self._observation()

    def _initial_control(self, observation: Mapping[str, float]) -> np.ndarray:
        fallback = {
            "control_frequency": observation.get("compressor_frequency", 0.0),
            "control_eev": observation.get("eev_opening", 0.0),
            "control_fan_out": observation.get("outdoor_fan_speed", 0.0),
        }
        values = []
        for name in CONTROL_COLUMNS:
            value = observation.get(name, fallback[name])
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float(fallback[name])
            values.append(value if np.isfinite(value) else float(fallback[name]))
        return np.asarray(values, np.float32)

    def _feature_frame(self) -> pd.DataFrame:
        rows = []
        for control in self._controls:
            rows.append({
                name: float(control[index])
                for index, name in enumerate(CONTROL_COLUMNS)
            })
        return pd.DataFrame(rows)

    def _residual_sequence(self, slow_frame: pd.DataFrame) -> np.ndarray:
        control_mean = self.slow.model.control_mean.detach().cpu().numpy().astype(np.float32)
        control_scale = self.slow.model.control_scale.detach().cpu().numpy().astype(np.float32)
        features = build_features(
            self._feature_frame(), slow_frame, control_mean, control_scale,
        )
        with torch.no_grad():
            tensor = torch.tensor(features[None], device=self.device)
            residual = self.residual_model(tensor)[0].detach().cpu().numpy()
        return residual

    def _observation(self) -> dict:
        slow_frame = pd.DataFrame(self._slow_rows)
        residual = self._residual_sequence(slow_frame)
        weights = residual_gate_from_controls(
            self._feature_frame(), self.residual_base_weight,
        )
        slow_temperature = float(slow_frame["T_in"].iloc[-1])
        residual_temperature = float(residual[-1] * weights[-1])
        return {
            **self._slow_rows[-1],
            "T_in_slow": slow_temperature,
            "T_in_residual": residual_temperature,
            "T_in_residual_raw": float(residual[-1]),
            "residual_weight": float(weights[-1]),
            "T_in": slow_temperature + residual_temperature,
            "control_delay_seconds": self.control_delay_seconds,
        }

    def simulate(self, initial_observation, controls) -> pd.DataFrame:
        rows = [self.reset(initial_observation)]
        for freq, eev, fan_out in np.asarray(controls, np.float32):
            rows.append(self.step(float(freq), float(eev), float(fan_out)))
        return pd.DataFrame(rows)
