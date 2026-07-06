"""PID controller for closed-loop AC experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from control.MiniController.base import BaseController, ControlPlan, ControllerContext
from control.MiniController.config import fixed_action_for_mode, pid_config_for_mode, signed_temperature_error
from control.MiniController.schemas import TARGET_ACTION_COLS


class PIDController(BaseController):
    policy_name = "pid"

    def __init__(
        self,
        mode: int | str,
        config: dict,
        action_bounds: dict[str, tuple[float, float]],
        step_seconds: int = 5,
    ):
        self.mode = mode
        self.step_seconds = int(step_seconds)
        self.fixed_action = fixed_action_for_mode(config, mode)
        pid = pid_config_for_mode(config, mode)
        self.kp = float(pid["kp"])
        self.ki = float(pid["ki"])
        self.kd = float(pid["kd"])
        self.integral_limit = abs(float(pid.get("integral_limit", 300.0)))
        self.action_bounds = action_bounds
        self.integral = 0.0
        self.prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def plan(
        self,
        observation: dict | np.ndarray,
        target: float,
        context: ControllerContext | None = None,
    ) -> ControlPlan:
        step_seconds = int(context.step_seconds if context else self.step_seconds)
        horizon_seconds = int(context.horizon_seconds if context and context.horizon_seconds else step_seconds)
        if isinstance(observation, dict):
            T_in = float(observation["T_in"])
        else:
            T_in = float(np.asarray(observation, dtype=np.float32)[2])
        error = signed_temperature_error(self.mode, T_in, target)
        self.integral = float(np.clip(self.integral + error * step_seconds, -self.integral_limit, self.integral_limit))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / step_seconds
        self.prev_error = error

        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        raw_freq = float(self.fixed_action[0] + p_term + i_term + d_term)
        lo, hi = self.action_bounds["freq_target"]
        clipped_freq = float(np.clip(raw_freq, lo, hi))
        action = self.fixed_action.copy()
        action[0] = clipped_freq

        steps = max(1, horizon_seconds // step_seconds)
        actions = np.repeat(action[None], steps, axis=0)
        plan = pd.DataFrame(actions, columns=TARGET_ACTION_COLS)
        plan["elapsed_seconds"] = (np.arange(len(plan)) + 1) * step_seconds
        debug = {
            "controller": self.policy_name,
            "p_term": float(p_term),
            "i_term": float(i_term),
            "d_term": float(d_term),
            "raw_freq_target": raw_freq,
            "clipped_freq_target": clipped_freq,
        }
        return ControlPlan(action=action, target_plan=plan, debug=debug)
