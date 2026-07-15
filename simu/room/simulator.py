"""焓差室制冷 V3 在线环境：reset 一次，随后每 5 秒调用 step。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from .continuous_model import (
        CONTROL_COLUMNS, DT_SECONDS, PLANT_STATE_COLUMNS, STATE_COLUMNS,
        AutoregressiveFeatureState, ContinuousFeatureState,
        FeatureConfig, HeatLoadServoConfig, StandardHeatLoadServo,
    )
except ImportError:  # direct script execution
    from continuous_model import (  # type: ignore
        CONTROL_COLUMNS, DT_SECONDS, PLANT_STATE_COLUMNS, STATE_COLUMNS,
        AutoregressiveFeatureState, ContinuousFeatureState,
        FeatureConfig, HeatLoadServoConfig, StandardHeatLoadServo,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "continuous_cooling_model.joblib"


def _install_sklearn_loss_compatibility() -> None:
    if "_loss" in sys.modules:
        return
    try:
        import sklearn._loss.loss as sklearn_loss
    except ImportError:
        return
    sys.modules["_loss"] = sklearn_loss


class ContinuousEnthalpyRoomEnv:
    """与 ``simu.temperature.OnlineEnthalpyRoomEnv`` 类似的 5 秒在线环境。

    从 V3 开始，该环境是一个纯 plant 模型：只根据当前热状态
    （室内外温度、盘管温度、湿度等）和三个控制量预测下一时刻室内温度。
    目标温度 ``T_set`` 不再作为模型输入传入，仅用于上层控制器。
    """

    # Hybrid off-cycle physics fallback parameters, calibrated from
    # data/dataset off-cycle segments. These are constants, not user-tunable.
    OFFCYCLE_B = 0.011124  # °C/min/°C
    OFFCYCLE_C = 0.04      # °C/min
    FREQ_ZERO_THRESHOLD = 0.5  # Hz

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        passive_heat_tau_seconds: float | None = None,
        active_cooling_gain: float = 0.0,
        active_cooling_band: float = 0.0,
        use_physics_offcycle: bool = True,
    ):
        _install_sklearn_loss_compatibility()
        payload = joblib.load(model_path)
        if int(payload.get("supported_mode", 1)) != 1:
            raise ValueError("V3 仅支持制冷 mode=1")
        self.model = payload["model"]
        self.feature_variant = str(payload.get("feature_variant", "thermal_inertia"))
        self.state_columns = list(payload.get("state_columns", STATE_COLUMNS))
        self.state_medians = np.asarray(payload["state_medians"], np.float32)
        self.dt = float(payload.get("dt_seconds", DT_SECONDS))
        continuity = payload.get("continuity", {})
        self.continuity_tau_seconds = float(continuity.get("tau_seconds", 90.0))
        rate = continuity.get("max_rate_c_per_min", 0.5)
        self.max_rate_c_per_min = None if rate is None else float(rate)
        passive_tau = payload.get("passive_heat_tau_seconds", 14_400.0)
        self.passive_heat_tau_seconds = float(
            passive_tau if passive_heat_tau_seconds is None else passive_heat_tau_seconds
        )
        self.metadata = payload.get("metadata", {})
        offcycle = payload.get("offcycle", {})
        self.use_physics_offcycle = bool(payload.get("use_physics_offcycle", use_physics_offcycle))
        self.freq_zero_threshold = float(payload.get("freq_zero_threshold", self.FREQ_ZERO_THRESHOLD))
        self.offcycle_b = float(payload.get("offcycle_b", offcycle.get("b", self.OFFCYCLE_B)))
        self.offcycle_c = float(payload.get("offcycle_c", offcycle.get("c", self.OFFCYCLE_C)))
        # Feature configuration for autoregressive variants. Missing config means
        # an older model trained before physics-aware batches were added; default
        # all batches to False to preserve exact feature dimensions.
        feature_cfg = payload.get("feature_config")
        if feature_cfg is None:
            self.feature_config = FeatureConfig(
                batch1_control_memory=False,
                batch2_approach_temps=False,
                batch3_interactions=False,
            )
        else:
            self.feature_config = FeatureConfig(
                batch1_control_memory=bool(feature_cfg.get("batch1_control_memory", True)),
                batch2_approach_temps=bool(feature_cfg.get("batch2_approach_temps", True)),
                batch3_interactions=bool(feature_cfg.get("batch3_interactions", True)),
            )
        self.heat_load = StandardHeatLoadServo(
            HeatLoadServoConfig(**payload.get("heat_load_config", {}))
        )
        self.compressor_off_freq_hz = float(payload.get("compressor_off_freq_hz", 1.0))
        self.autoregressive_horizon = int(payload.get("autoregressive_horizon", 1))

        # These parameters are kept for API compatibility but are no longer used
        # by the plant model; the trained model itself now encodes the cooling
        # response without explicit setpoint knowledge.
        self.active_cooling_gain = float(active_cooling_gain)
        self.active_cooling_band = float(active_cooling_band)
        self._ready = False

    def reset(
        self,
        T_out: float | Mapping[str, float] | pd.Series | None = None,
        T_in: float | None = None,
        T_out_coil: float | None = None,
        T_in_coil: float | None = None,
        *,
        initial_observation: Mapping[str, float] | pd.Series | None = None,
        **state_overrides: float,
    ) -> dict:
        """重置环境。

        可像 temperature 模块一样传四个温度，也可传完整观测 ``Series/dict``：
        ``reset(initial_observation=df.iloc[0])``。``T_set`` 可出现在观测中，
        但会被忽略——本环境是纯 plant 模型，不应知道控制目标。
        """
        if isinstance(T_out, (Mapping, pd.Series)) and initial_observation is None:
            initial_observation = T_out
            T_out = None
        observation = dict(initial_observation) if initial_observation is not None else {}
        explicit = {
            "T_out": T_out, "T_in": T_in,
            "T_out_coil": T_out_coil, "T_in_coil": T_in_coil,
        }
        observation.update({key: value for key, value in explicit.items() if value is not None})
        observation.update(state_overrides)
        state = np.asarray([
            observation.get(name, np.nan) for name in self.state_columns
        ], np.float32)
        self.initial_state = np.where(
            np.isfinite(state), state, self.state_medians,
        ).astype(np.float32)
        index = {name: i for i, name in enumerate(self.state_columns)}
        if any(name not in observation for name in ("T_out", "T_in", "T_out_coil", "T_in_coil")):
            missing = [name for name in ("T_out", "T_in", "T_out_coil", "T_in_coil")
                       if name not in observation]
            raise ValueError(f"初始温度缺失: {missing}")
        mode = int(round(float(self.initial_state[index["mode"]])))
        if mode != 1:
            raise ValueError(f"V3 制冷模型要求 mode=1，当前 mode={mode}")
        self.temperature = float(self.initial_state[index["T_in"]])
        self.outdoor_temperature = float(self.initial_state[index["T_out"]])
        self._outdoor_temperature = float(self.initial_state[index["T_out"]])
        # Legacy thermal_inertia models need T_set as a reference; plant models
        # do not use it. Keep the attribute for backward compatibility.
        if "T_set" in index:
            self.reference_temperature = float(self.initial_state[index["T_set"]])
        else:
            self.reference_temperature = float(self.temperature)

        if self.feature_variant.startswith("autoregressive"):
            self.prefix = AutoregressiveFeatureState(self.initial_state, self.feature_config)
        else:
            self.prefix = ContinuousFeatureState(self.initial_state)
        self.heat_load.reset(
            self.temperature, float(self.initial_state[index["T_out"]]), mode,
        )
        self.last_direct_target = self.temperature
        self._ready = True
        return self._observation()

    def step(self, freq: float, eev: float, fan_out: float) -> dict:
        if not self._ready:
            raise RuntimeError("请先调用 reset()")
        control = np.asarray([freq, eev, fan_out], np.float32)
        if not np.isfinite(control).all():
            raise ValueError("控制量含 NaN/Inf")

        if self.feature_variant.startswith("autoregressive"):
            self._step_autoregressive(control)
        else:
            self._step_thermal_inertia(control)

        self.heat_load.update(self.temperature, self.dt)
        return self._observation()

    def _step_autoregressive(self, control: np.ndarray) -> None:
        """Stateful plant-model step: predict absolute next T_in."""
        freq = float(control[0])
        self.prefix.update_temperature(self.temperature)
        self.prefix.update_control(control)

        if self.use_physics_offcycle and freq <= self.freq_zero_threshold:
            dT = (
                self.offcycle_b * (self._outdoor_temperature - self.temperature)
                + self.offcycle_c
            ) * (self.dt / 60.0)
            self.temperature = float(np.clip(self.temperature + dT, -30.0, 65.0))
        else:
            horizon = max(1, self.autoregressive_horizon)
            target = float(self.model.predict(self.prefix.feature()[None])[0])
            delta = (target - self.temperature) / horizon
            if self.max_rate_c_per_min is not None:
                max_step = self.max_rate_c_per_min * self.dt / 60.0
                delta = float(np.clip(delta, -max_step, max_step))
            self.temperature = float(np.clip(self.temperature + delta, -30.0, 65.0))

        self.last_direct_target = self.temperature

    def _step_thermal_inertia(self, control: np.ndarray) -> None:
        """Original open-loop target-offset prediction (kept for old models)."""
        self.prefix.update(control)
        offset = float(self.model.predict(self.prefix.feature()[None])[0])
        self.last_direct_target = self._reference_temperature + offset
        freq = float(control[0])

        if self.use_physics_offcycle and freq <= self.freq_zero_threshold:
            dT = (
                self.offcycle_b * (self._outdoor_temperature - self.temperature)
                + self.offcycle_c
            ) * (self.dt / 60.0)
            self.temperature = float(np.clip(self.temperature + dT, -30.0, 65.0))
            self.last_direct_target = self.temperature
        else:
            self.last_direct_target = self._apply_compressor_physics(control, self.last_direct_target)
            self._advance_temperature(self.last_direct_target)

        # Compatibility no-op: active_cooling_gain is ignored for the plant model.

    @property
    def _reference_temperature(self) -> float:
        """Return the old T_set reference for legacy thermal_inertia models."""
        return getattr(self, "reference_temperature", 0.0)

    def _apply_compressor_physics(self, control: np.ndarray, direct_target: float) -> float:
        freq = float(control[0])
        if freq <= self.compressor_off_freq_hz and direct_target < self.temperature:
            direct_target = self.temperature
        if freq <= self.compressor_off_freq_hz and self.outdoor_temperature > self.temperature:
            model_alpha = 1.0 - np.exp(-self.dt / self.continuity_tau_seconds)
            passive_alpha = 1.0 - np.exp(-self.dt / max(self.passive_heat_tau_seconds, 1e-6))
            passive_delta = passive_alpha * (self.outdoor_temperature - self.temperature)
            passive_target = self.temperature + passive_delta / max(model_alpha, 1e-6)
            direct_target = max(float(direct_target), float(passive_target))
        return float(direct_target)

    def _advance_temperature(self, direct_target: float) -> None:
        alpha = 1.0 - np.exp(-self.dt / self.continuity_tau_seconds)
        delta = alpha * (float(direct_target) - self.temperature)
        if self.max_rate_c_per_min is not None:
            max_step = self.max_rate_c_per_min * self.dt / 60.0
            delta = float(np.clip(delta, -max_step, max_step))
        self.temperature = float(np.clip(self.temperature + delta, -30.0, 65.0))

    def _observation(self) -> dict:
        heat = self.heat_load.features()
        elapsed = 0.0 if not hasattr(self, "prefix") else self.prefix.base.n * self.dt
        return {
            "elapsed_seconds": elapsed,
            "T_in": float(self.temperature),
            "T_in_raw": float(self.last_direct_target),
            "model_spread": 0.0,
            "heat_load_target": float(heat[0]),
            "heat_load_temperature": float(heat[1]),
            "heat_load_tracking_error": float(heat[2]),
            "heat_load_proxy": float(heat[3]),
        }

    def simulate(self, initial_state, controls: np.ndarray) -> pd.DataFrame:
        if isinstance(initial_state, (Mapping, pd.Series)):
            rows = [self.reset(initial_observation=initial_state)]
        else:
            values = list(initial_state)
            if len(values) == 4:
                rows = [self.reset(*values)]
            elif len(values) == len(self.state_columns):
                rows = [self.reset(initial_observation=dict(zip(self.state_columns, values)))]
            else:
                raise ValueError("initial_state 需为完整观测，或 [T_out,T_in,T_out_coil,T_in_coil]")
        for freq, eev, fan_out in np.asarray(controls):
            rows.append(self.step(freq, eev, fan_out))
        return pd.DataFrame(rows)


# 兼容整理前的 V3 类名。
MechanisticCoolingSimulator = ContinuousEnthalpyRoomEnv


def read_controls(path: Path) -> np.ndarray:
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    aliases = [
        ("freq", "eev", "fan_out"),
        ("compressor_frequency", "eev_opening", "outdoor_fan_speed"),
    ]
    for columns in aliases:
        if set(columns).issubset(frame.columns):
            return frame[list(columns)].to_numpy(np.float32)
    if frame.shape[1] >= 7:
        return frame.iloc[:, [4, 5, 6]].to_numpy(np.float32)
    raise ValueError("CSV 需含三项控制列，或采用原始训练 CSV 布局")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="焓差室 V3 连续制冷仿真")
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--initial", nargs=4, type=float, required=True,
                        metavar=("T_OUT", "T_IN", "T_OUT_COIL", "T_IN_COIL"))
    parser.add_argument("--output", type=Path,
                        default=HERE / "continuous_simulation_output.csv")
    args = parser.parse_args()
    result = ContinuousEnthalpyRoomEnv().simulate(args.initial, read_controls(args.controls))
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(args.output)
