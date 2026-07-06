"""HanWAM planner controller for closed-loop experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from control.MiniController.base import BaseController, ControlPlan, ControllerContext
from .config import DEFAULT_CHECKPOINT
from .dataloader import Normalizer
from .model import build_wam_model
from .planner import CEMPlanner
from .type import WAM_ACTION_COLS


class HanWAMController(BaseController):
    policy_name = "hanwam"

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        action_bounds: dict[str, tuple[float, float]] | None = None,
        planner_config: dict | None = None,
        device: str | None = None,
        mode: int | str = 1,
        step_seconds: int = 5,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.obs_cols = checkpoint["obs_cols"]
        self.target_action_cols = checkpoint["target_action_cols"]
        self.physical_cols = checkpoint["physical_cols"]
        self.mode = mode
        self.step_seconds = int(step_seconds)
        self.obs_norm = Normalizer.from_dict(checkpoint["obs_norm"])
        self.target_action_norm = Normalizer.from_dict(checkpoint["target_action_norm"])
        self.physical_norm = Normalizer.from_dict(checkpoint["physical_norm"])
        self.action_bounds = action_bounds or {
            col: tuple(map(float, bounds)) for col, bounds in checkpoint["target_action_bounds"].items()
        }
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = build_wam_model(checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        resolved_planner_config = dict(checkpoint.get("planner_config") or {})
        if planner_config:
            resolved_planner_config.update(planner_config)
            if "cost_weights" in planner_config:
                merged_weights = dict((checkpoint.get("planner_config") or {}).get("cost_weights") or {})
                merged_weights.update(planner_config.get("cost_weights") or {})
                resolved_planner_config["cost_weights"] = merged_weights
        self.control_interval_steps = max(1, int(resolved_planner_config.get("control_interval_steps", 1)))
        self.history_steps = max(
            1,
            int(
                resolved_planner_config.get(
                    "history_steps",
                    checkpoint.get("history_steps", 1),
                )
            ),
        )
        self._cached_actions: list[np.ndarray] = []
        self._obs_history: list[np.ndarray] = []
        self.planner = CEMPlanner(
            model=self.model,
            obs_norm=self.obs_norm,
            action_norm=self.target_action_norm,
            physical_norm=self.physical_norm,
            obs_cols=self.obs_cols,
            physical_cols=self.physical_cols,
            action_bounds=self.action_bounds,
            config=resolved_planner_config,
            device=self.device,
        )

    def reset(self) -> None:
        self._cached_actions = []
        self._obs_history = []
        if hasattr(self.planner, "reset"):
            self.planner.reset()

    def _obs_array(self, observation: dict | np.ndarray) -> np.ndarray:
        if isinstance(observation, dict):
            missing = [col for col in self.obs_cols if col not in observation]
            if missing:
                raise ValueError(f"observation is missing required WAM columns: {missing}")
            return np.asarray([observation[col] for col in self.obs_cols], dtype=np.float32)
        raw = np.asarray(observation, dtype=np.float32)
        if raw.shape[0] == len(self.obs_cols):
            return raw
        raise ValueError(f"observation must have {len(self.obs_cols)} values")

    def _append_history(self, obs: np.ndarray) -> None:
        self._obs_history.append(np.asarray(obs, dtype=np.float32).copy())
        if len(self._obs_history) > self.history_steps:
            self._obs_history = self._obs_history[-self.history_steps :]

    def _history_array(self) -> np.ndarray:
        if not self._obs_history:
            raise RuntimeError("HanWAM history is empty")
        obs = list(self._obs_history)
        while len(obs) < self.history_steps:
            obs.insert(0, obs[0].copy())
        return np.stack(obs[-self.history_steps :], axis=0)

    def _remaining_seconds(self, context: ControllerContext | None) -> float | None:
        if context is None:
            return None
        ddl = context.metadata.get("ddl_seconds") or context.metadata.get("experiment_horizon_seconds")
        if ddl is None:
            return None
        return max(0.0, float(ddl) - float(context.elapsed_seconds))

    @torch.no_grad()
    def plan(
        self,
        observation: dict | np.ndarray,
        target: float,
        context: ControllerContext | None = None,
    ) -> ControlPlan:
        step_seconds = int(context.step_seconds if context else self.step_seconds)
        horizon_seconds = int(context.horizon_seconds if context and context.horizon_seconds else step_seconds)
        obs = self._obs_array(observation)
        self._append_history(obs)
        obs_history = self._history_array()
        if self._cached_actions:
            action = np.asarray(self._cached_actions.pop(0), dtype=np.float32)
            steps = max(1, horizon_seconds // step_seconds)
            actions = np.repeat(action[None], steps, axis=0)
            plan = pd.DataFrame(actions, columns=WAM_ACTION_COLS)
            plan["elapsed_seconds"] = (np.arange(len(plan)) + 1) * step_seconds
            debug = {
                "controller": "hanwam",
                "hanwam_planner": "cem",
                "hanwam_reused_plan": True,
                "hanwam_control_interval_steps": self.control_interval_steps,
                "hanwam_history_steps": self.history_steps,
                "hanwam_clipped_freq_target": float(action[0]),
                "hanwam_clipped_eev": float(action[1]),
                "hanwam_clipped_fan_out": float(action[2]),
            }
            return ControlPlan(action=action, target_plan=plan, debug=debug, history=pd.DataFrame())
        result = self.planner.plan(
            obs,
            target=float(target),
            remaining_seconds=self._remaining_seconds(context),
            obs_history=obs_history,
        )
        if self.control_interval_steps > 1:
            self._cached_actions = [
                np.asarray(action, dtype=np.float32)
                for action in result.best_sequence[1 : self.control_interval_steps]
            ]
        steps = max(1, horizon_seconds // step_seconds)
        actions = np.repeat(result.action[None], steps, axis=0)
        plan = pd.DataFrame(actions, columns=WAM_ACTION_COLS)
        plan["elapsed_seconds"] = (np.arange(len(plan)) + 1) * step_seconds
        history = result.history.copy()
        if not history.empty:
            history["first_freq_target"] = float(result.action[0])
            history["first_eev"] = float(result.action[1])
            history["first_fan_out"] = float(result.action[2])
            history["control_interval_steps"] = self.control_interval_steps
            history["history_steps"] = self.history_steps
        result.debug["hanwam_history_steps"] = self.history_steps
        return ControlPlan(action=result.action, target_plan=plan, debug=result.debug, history=history)
