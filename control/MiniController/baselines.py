"""Simple baseline controllers shared across experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseController, ControlPlan, ControllerContext
from .config import fixed_action_for_mode
from .schemas import TARGET_ACTION_COLS


class FixedActionController(BaseController):
    policy_name = "fixed"

    def __init__(
        self,
        mode: int | str,
        config: dict,
        step_seconds: int = 5,
    ):
        self.mode = mode
        self.step_seconds = int(step_seconds)
        self.action = fixed_action_for_mode(config, mode)

    def plan(
        self,
        observation: dict | np.ndarray,
        target: float,
        context: ControllerContext | None = None,
    ) -> ControlPlan:
        step_seconds = int(context.step_seconds if context else self.step_seconds)
        horizon_seconds = int(context.horizon_seconds if context and context.horizon_seconds else step_seconds)
        steps = max(1, horizon_seconds // step_seconds)
        actions = np.repeat(self.action[None], steps, axis=0)
        plan = pd.DataFrame(actions, columns=TARGET_ACTION_COLS)
        plan["elapsed_seconds"] = (np.arange(len(plan)) + 1) * step_seconds
        return ControlPlan(
            action=self.action.copy(),
            target_plan=plan,
            debug={"controller": self.policy_name},
        )

