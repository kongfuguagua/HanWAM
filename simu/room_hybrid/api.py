"""统一房间/空调世界模型的接口草案。

这里先只保留接口约定和列定义。等 MODEL_SPEC.md 里的建模边界讨论清楚后，
再补具体的 PyTorch 实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


DT_SECONDS = 5.0

ACTION_COLUMNS = [
    "control_frequency",
    "control_eev",
    "control_fan_out",
]

RESET_COLUMNS = [
    "T_out",
    "T_out_coil",
    "T_out_discharge",
    "compressor_frequency",
    "eev_opening",
    "outdoor_fan_speed",
    "I_comp",
    "T_in",
    "T_in_coil",
    "RH_in",
    "mode",
    "energy_cum",
    "fault",
    "swing",
]

TARGET_LIKE_COLUMNS = {
    "T_set",
    "RH_target",
    "indoor_fan_target",
    "pid_freq",
    "pid_target",
    "inference_freq",
    "inference_eev",
    "inference_fan_out",
    "inference_fan_in",
}

ROOM_OUTPUT_COLUMNS = [
    "T_in",
]

AC_OUTPUT_COLUMNS = [
    "T_in_coil",
    "T_out_coil",
    "T_out_discharge",
]


@dataclass(frozen=True)
class RoomWorldConfig:
    hidden_size: int = 96
    context_size: int = 32
    action_lag_size: int = 4
    predict_ac_outputs: bool = True
    use_grey_box_heat_balance: bool = True
    max_room_rate_c_per_min: float = 1.0


class RoomWorldEnvProtocol:
    """在线仿真器期望实现的接口。"""

    def reset(self, initial_observation: Mapping[str, float]) -> Mapping[str, float]:
        """使用第一行实测观测初始化隐藏状态。"""
        raise NotImplementedError

    def step(
        self,
        compressor_frequency: float,
        eev_opening: float,
        outdoor_fan_speed: float,
    ) -> Mapping[str, float]:
        """只使用三路执行量推进一个 5 秒步长。"""
        raise NotImplementedError

    def simulate(
        self,
        initial_observation: Mapping[str, float],
        actions: Sequence[Sequence[float]],
    ) -> Sequence[Mapping[str, float]]:
        """滚动仿真完整动作序列，并返回预测观测。"""
        rows = [self.reset(initial_observation)]
        for compressor_frequency, eev_opening, outdoor_fan_speed in actions:
            rows.append(self.step(
                float(compressor_frequency),
                float(eev_opening),
                float(outdoor_fan_speed),
            ))
        return rows
