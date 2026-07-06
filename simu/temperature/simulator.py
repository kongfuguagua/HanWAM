"""焓差室5秒在线环境：reset一次，随后每个控制步调用step一次。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

try:
    from .online_history_model import prefix_feature
except ImportError:  # direct script execution
    from online_history_model import prefix_feature


HERE = Path(__file__).resolve().parent
DT_SECONDS = 5.0
VALIDATED_STEPS = 720


class OnlineEnthalpyRoomEnv:
    """状态化在线环境；第n次step只使用前n次控制输入。"""

    def __init__(self, model_path: str | Path = HERE / "online_history_ensemble.joblib"):
        payload = joblib.load(model_path)
        self.members = payload["members"]
        self.blocks = int(payload["blocks"])
        self.configs = payload.get("configs", [])
        self._ready = False

    def reset(self, T_out: float, T_in: float, T_out_coil: float, T_in_coil: float) -> dict:
        self.state0 = np.asarray([T_out, T_out_coil, T_in, T_in_coil], np.float32)
        if not np.isfinite(self.state0).all():
            raise ValueError("初始温度含NaN/Inf")
        self.controls: list[np.ndarray] = []
        self.last_prediction = float(T_in)
        self._ready = True
        return {"elapsed_seconds": 0.0, "T_in": float(T_in), "T_in_raw": float(T_in), "model_spread": 0.0}

    def step(self, freq: float, eev: float, fan_out: float) -> dict:
        if not self._ready:
            raise RuntimeError("请先调用 reset()")
        control = np.asarray([freq, eev, fan_out], np.float32)
        if not np.isfinite(control).all():
            raise ValueError("控制量含NaN/Inf")
        self.controls.append(control)
        step = len(self.controls)
        prefix = np.asarray(self.controls, np.float32)
        feature = prefix_feature(self.state0, prefix, step, VALIDATED_STEPS, self.blocks)[None]
        member_delta = np.asarray([float(model.predict(feature)[0]) for model in self.members])
        raw_prediction = float(self.state0[2] + member_delta.mean())
        alpha = DT_SECONDS / (30.0 + DT_SECONDS)
        self.last_prediction = self.last_prediction + alpha * (raw_prediction - self.last_prediction)
        return {
            "elapsed_seconds": step * DT_SECONDS,
            "T_in": float(self.last_prediction),
            "T_in_raw": raw_prediction,
            "model_spread": float(member_delta.std()),
        }

    def simulate(self, initial_state, controls: np.ndarray) -> pd.DataFrame:
        """便利批量接口；内部逐行step，结果与实时调用完全相同。"""
        T_out, T_in, T_out_coil, T_in_coil = initial_state
        rows = [self.reset(T_out, T_in, T_out_coil, T_in_coil)]
        for freq, eev, fan_out in np.asarray(controls):
            rows.append(self.step(freq, eev, fan_out))
        return pd.DataFrame(rows)


def read_controls(path: Path) -> np.ndarray:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            frame = pd.read_csv(path, encoding=enc); break
        except UnicodeDecodeError:
            continue
    if {"freq", "eev", "fan_out"}.issubset(frame.columns):
        return frame[["freq", "eev", "fan_out"]].to_numpy(np.float32)
    if frame.shape[1] >= 7:
        return frame.iloc[:, [4, 5, 6]].to_numpy(np.float32)
    raise ValueError("CSV需含freq/eev/fan_out列，或采用原始训练CSV列布局")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description="焓差室5秒在线温度仿真")
    p.add_argument("--controls",type=Path,required=True);p.add_argument("--initial",nargs=4,type=float,required=True,metavar=("T_OUT","T_IN","T_OUT_COIL","T_IN_COIL"));p.add_argument("--output",type=Path,default=HERE/"online_simulation_output.csv")
    a=p.parse_args();result=OnlineEnthalpyRoomEnv().simulate(a.initial,read_controls(a.controls));result.to_csv(a.output,index=False,encoding="utf-8-sig");print(a.output)
