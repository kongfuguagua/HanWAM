from __future__ import annotations

import sys
import tempfile
import types
import unittest

import numpy as np
import torch

from control.PID.controller import PIDController
from control.MiniController.baselines import FixedActionController
from control.MiniController.base import ControlPlan
from control.MiniController.config import (
    action_bounds_for_mode,
    load_controller_config,
)
from control.MiniController.config_schema import expand_modes
from control.MiniController.tracking import SwanLabTracker
from control.HanWAM.model import HanWAM
from control.HanWAM.controller import HanWAMController
from control.HanWAM.dataloader import Normalizer
from control.HanWAM.planner import CEMPlanner
from control.HanWAM.type import WAM_ACTION_COLS, WAM_OBS_COLS, WAM_PHYSICAL_COLS


class BaselineControllerTest(unittest.TestCase):
    def setUp(self):
        self.fixed_config = load_controller_config("control/config/experiments/fixed.yml")
        self.pid_config = load_controller_config("control/config/experiments/pid.yml")
        self.bounds1 = action_bounds_for_mode(
            self.pid_config,
            1,
            {"freq_target": (0, 90), "eev": (0, 480), "fan_out": (0, 850)},
        )
        self.bounds3 = action_bounds_for_mode(
            self.pid_config,
            3,
            {"freq_target": (0, 110), "eev": (0, 480), "fan_out": (0, 800)},
        )

    def test_fixed_action_controller_uses_config(self):
        controller = FixedActionController(mode=1, config=self.fixed_config)
        plan = controller.plan({"T_in": 30}, target=27)
        np.testing.assert_allclose(plan.action, [40.2, 165.0, 750.0], rtol=1e-5)
        self.assertEqual(len(plan.target_plan), 1)

    def test_config_uses_method_action_space(self):
        self.assertEqual(expand_modes("all"), [1, 3])
        self.assertEqual(self.bounds1["freq_target"], (0.0, 90.0))
        self.assertEqual(self.bounds3["fan_out"], (0.0, 800.0))

    def test_pid_cooling_direction(self):
        controller = PIDController(mode=1, config=self.pid_config, action_bounds=self.bounds1)
        high = controller.plan({"T_in": 30.0}, target=27.0).action[0]
        controller.reset()
        low = controller.plan({"T_in": 25.0}, target=27.0).action[0]
        self.assertGreater(high, 40.2)
        self.assertLess(low, 40.2)

    def test_pid_heating_direction(self):
        controller = PIDController(mode=3, config=self.pid_config, action_bounds=self.bounds3)
        low = controller.plan({"T_in": 15.0}, target=20.0).action[0]
        controller.reset()
        high = controller.plan({"T_in": 23.0}, target=20.0).action[0]
        self.assertGreater(low, 22.0)
        self.assertLess(high, 22.0)

    def test_control_plan_as_dict(self):
        plan = ControlPlan(action=np.asarray([1.0, 2.0, 3.0]))
        payload = plan.as_dict()
        np.testing.assert_allclose(payload["action"], [1.0, 2.0, 3.0])

    def test_hanwam_rollout_shape(self):
        model = HanWAM(obs_dim=6, action_dim=3, physical_dim=4, latent_dim=8, hidden_dim=16)
        latent, physical = model.rollout(torch.zeros(5, 6), torch.zeros(5, 7, 3))
        self.assertEqual(tuple(latent.shape), (5, 7, 8))
        self.assertEqual(tuple(physical.shape), (5, 7, 4))

    def test_hanwam_state_encoder_uses_observation_history(self):
        model = HanWAM(obs_dim=6, action_dim=3, physical_dim=4, latent_dim=8, hidden_dim=16)
        obs_a = torch.zeros(5, 4, 6)
        obs_b = obs_a.clone()
        obs_b[:, 1, :] = 1.0
        delta = (model.encode(obs_a) - model.encode(obs_b)).detach().abs().max()
        self.assertGreater(float(delta), 0.0)

    def test_hanwam_controller_uses_runtime_planner_overrides(self):
        model_config = {
            "class_name": "HanWAM",
            "obs_dim": len(WAM_OBS_COLS),
            "action_dim": len(WAM_ACTION_COLS),
            "physical_dim": len(WAM_PHYSICAL_COLS),
            "latent_dim": 8,
            "hidden_dim": 16,
        }
        model = HanWAM(**{key: value for key, value in model_config.items() if key != "class_name"})
        checkpoint = {
            "obs_cols": WAM_OBS_COLS,
            "target_action_cols": WAM_ACTION_COLS,
            "physical_cols": WAM_PHYSICAL_COLS,
            "obs_norm": {"mean": [0.0] * len(WAM_OBS_COLS), "std": [1.0] * len(WAM_OBS_COLS)},
            "target_action_norm": {"mean": [0.0] * len(WAM_ACTION_COLS), "std": [1.0] * len(WAM_ACTION_COLS)},
            "physical_norm": {"mean": [0.0] * len(WAM_PHYSICAL_COLS), "std": [1.0] * len(WAM_PHYSICAL_COLS)},
            "target_action_bounds": {
                "freq_target": [0.0, 90.0],
                "eev": [69.0, 480.0],
                "fan_out": [0.0, 850.0],
            },
            "model_config": model_config,
            "model": model.state_dict(),
            "planner_config": {"horizon_steps": 24, "cost_weights": {"energy": 2.0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/hanwam.pt"
            torch.save(checkpoint, path)
            controller = HanWAMController(
                path,
                mode=1,
                device="cpu",
                planner_config={"horizon_steps": 12, "cost_weights": {"action_smooth": 0.05}},
            )
        self.assertEqual(controller.planner.horizon_steps, 12)
        self.assertEqual(controller.planner.cost_weights["energy"], 2.0)
        self.assertEqual(controller.planner.cost_weights["action_smooth"], 0.05)

    def test_hanwam_off_fan_cost_penalizes_high_fan_when_compressor_off(self):
        model = HanWAM(obs_dim=len(WAM_OBS_COLS), action_dim=len(WAM_ACTION_COLS), physical_dim=len(WAM_PHYSICAL_COLS), latent_dim=8, hidden_dim=16)
        norm_obs = Normalizer(mean=np.zeros(len(WAM_OBS_COLS), dtype=np.float32), std=np.ones(len(WAM_OBS_COLS), dtype=np.float32))
        norm_action = Normalizer(mean=np.zeros(len(WAM_ACTION_COLS), dtype=np.float32), std=np.ones(len(WAM_ACTION_COLS), dtype=np.float32))
        norm_phys = Normalizer(mean=np.zeros(len(WAM_PHYSICAL_COLS), dtype=np.float32), std=np.ones(len(WAM_PHYSICAL_COLS), dtype=np.float32))
        planner = CEMPlanner(
            model=model,
            obs_norm=norm_obs,
            action_norm=norm_action,
            physical_norm=norm_phys,
            obs_cols=WAM_OBS_COLS,
            physical_cols=WAM_PHYSICAL_COLS,
            action_bounds={"freq_target": (0.0, 90.0), "eev": (69.0, 480.0), "fan_out": (0.0, 850.0)},
            config={
                "horizon_steps": 2,
                "chunk_steps": 1,
                "cost_weights": {"tracking": 0.0, "energy": 0.0, "action_smooth": 0.0, "off_fan": 1.0},
            },
            device=torch.device("cpu"),
        )
        physical = torch.zeros(2, 2, len(WAM_PHYSICAL_COLS))
        low_fan = torch.tensor([[[0.0, 165.0, 0.0], [0.0, 165.0, 0.0]]])
        high_fan = torch.tensor([[[0.0, 165.0, 850.0], [0.0, 165.0, 850.0]]])
        actions = torch.cat([low_fan, high_fan], dim=0)
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0
        cost, parts = planner._cost(physical, actions, target=27.0, initial_t_in=27.0, remaining_seconds=None, observation=observation)
        self.assertIn("off_fan", parts)
        self.assertLess(float(cost[0]), float(cost[1]))

    def test_hanwam_action_prior_penalizes_action_z_outliers(self):
        model = HanWAM(obs_dim=len(WAM_OBS_COLS), action_dim=len(WAM_ACTION_COLS), physical_dim=len(WAM_PHYSICAL_COLS), latent_dim=8, hidden_dim=16)
        norm_obs = Normalizer(mean=np.zeros(len(WAM_OBS_COLS), dtype=np.float32), std=np.ones(len(WAM_OBS_COLS), dtype=np.float32))
        norm_action = Normalizer(
            mean=np.asarray([40.0, 165.0, 750.0], dtype=np.float32),
            std=np.asarray([10.0, 30.0, 50.0], dtype=np.float32),
        )
        norm_phys = Normalizer(mean=np.zeros(len(WAM_PHYSICAL_COLS), dtype=np.float32), std=np.ones(len(WAM_PHYSICAL_COLS), dtype=np.float32))
        planner = CEMPlanner(
            model=model,
            obs_norm=norm_obs,
            action_norm=norm_action,
            physical_norm=norm_phys,
            obs_cols=WAM_OBS_COLS,
            physical_cols=WAM_PHYSICAL_COLS,
            action_bounds={"freq_target": (0.0, 90.0), "eev": (69.0, 480.0), "fan_out": (0.0, 850.0)},
            config={
                "horizon_steps": 2,
                "chunk_steps": 1,
                "cost_weights": {"tracking": 0.0, "energy": 0.0, "action_smooth": 0.0, "action_prior": 1.0},
            },
            device=torch.device("cpu"),
        )
        physical = torch.zeros(2, 2, len(WAM_PHYSICAL_COLS))
        typical = torch.tensor([[[40.0, 165.0, 750.0], [40.0, 165.0, 750.0]]])
        outlier = torch.tensor([[[90.0, 480.0, 0.0], [90.0, 480.0, 0.0]]])
        actions = torch.cat([typical, outlier], dim=0)
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0
        cost, parts = planner._cost(physical, actions, target=27.0, initial_t_in=27.0, remaining_seconds=None, observation=observation)
        self.assertIn("action_prior", parts)
        self.assertLess(float(cost[0]), float(cost[1]))

    def test_vicreg_penalizes_collapsed_latents(self):
        collapsed = torch.zeros(16, 8)
        varied = torch.randn(16, 8)
        collapsed_loss = HanWAM.vicreg_loss(collapsed)[0]
        varied_loss = HanWAM.vicreg_loss(varied)[0]
        self.assertGreater(float(collapsed_loss), float(varied_loss))

    def test_sigreg_penalizes_collapsed_latents(self):
        collapsed = torch.zeros(16, 8)
        varied = torch.randn(16, 8)
        collapsed_loss = HanWAM.sigreg_loss(collapsed, num_projections=16)[0]
        varied_loss = HanWAM.sigreg_loss(varied, num_projections=16)[0]
        self.assertGreater(float(collapsed_loss), float(varied_loss))

    def test_swanlab_disabled_does_not_import(self):
        before = dict(sys.modules)
        tracker = SwanLabTracker(enabled=False)
        tracker.log({"x": 1})
        tracker.finish()
        self.assertEqual(set(before), set(sys.modules))

    def test_swanlab_mock(self):
        calls = []

        class FakeRun:
            def log(self, payload, step=None):
                calls.append(("run.log", payload, step))

        fake = types.SimpleNamespace(
            init=lambda **kwargs: calls.append(("init", kwargs, None)) or FakeRun(),
            Image=lambda path, caption=None: {"path": path, "caption": caption},
            finish=lambda: calls.append(("finish", {}, None)),
        )
        old = sys.modules.get("swanlab")
        sys.modules["swanlab"] = fake
        try:
            tracker = SwanLabTracker(enabled=True, project="p", experiment_name="e", config={"a": 1})
            tracker.log({"metric": 1.0}, step=2)
            tracker.log_image("img", "x.png", caption="x")
            tracker.finish()
        finally:
            if old is None:
                sys.modules.pop("swanlab", None)
            else:
                sys.modules["swanlab"] = old
        self.assertEqual(calls[0][0], "init")
        self.assertTrue(any(call[0] == "run.log" for call in calls))
        self.assertEqual(calls[-1][0], "finish")


if __name__ == "__main__":
    unittest.main()
