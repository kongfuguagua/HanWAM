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
from control.HanWAM.model import HanWAMBlockControllerModel, HanWAMWorldModel
from control.HanWAM.controller import HanWAMController
from control.HanWAM.dataloader import Normalizer
from control.HanWAM.planner import MPPIPlanner
from control.HanWAM.type import WAM_ACTION_COLS, WAM_OBS_COLS, WAM_PHYSICAL_COLS


def _tiny_hanwam(
    obs_dim: int = len(WAM_OBS_COLS),
    action_dim: int = len(WAM_ACTION_COLS),
    physical_dim: int = len(WAM_PHYSICAL_COLS),
    latent_dim: int = 8,
    hidden_dim: int = 16,
    frames_per_block: int = 2,
    history_blocks: int = 2,
    future_blocks: int = 2,
) -> HanWAMBlockControllerModel:
    world = HanWAMWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_latent_dim=4,
        frames_per_block=frames_per_block,
        history_blocks=history_blocks,
        future_blocks=future_blocks,
    )
    if physical_dim != 2:
        raise ValueError("current HanWAM physical trajectory has exactly two channels")
    return HanWAMBlockControllerModel(
        world,
        physical_dim=physical_dim,
        hidden_dim=hidden_dim,
        prober_action_scale=0.0,
        action_mean=[0.0, 0.0, 0.0],
        action_std=[1.0, 1.0, 1.0],
        physical_mean=[0.0, 0.0],
        physical_std=[1.0, 1.0],
        obs_mean=[0.0] * obs_dim,
        obs_std=[1.0] * obs_dim,
        t_in_obs_index=0,
        t_out_obs_index=min(1, obs_dim - 1),
    )


def _tiny_mppi_planner(config: dict, model: HanWAMBlockControllerModel | None = None) -> MPPIPlanner:
    model = model or _tiny_hanwam()
    norm_obs = Normalizer(mean=np.zeros(len(WAM_OBS_COLS), dtype=np.float32), std=np.ones(len(WAM_OBS_COLS), dtype=np.float32))
    norm_action = Normalizer(mean=np.zeros(len(WAM_ACTION_COLS), dtype=np.float32), std=np.ones(len(WAM_ACTION_COLS), dtype=np.float32))
    norm_phys = Normalizer(mean=np.zeros(len(WAM_PHYSICAL_COLS), dtype=np.float32), std=np.ones(len(WAM_PHYSICAL_COLS), dtype=np.float32))
    return MPPIPlanner(
        model=model,
        obs_norm=norm_obs,
        action_norm=norm_action,
        physical_norm=norm_phys,
        obs_cols=WAM_OBS_COLS,
        physical_cols=WAM_PHYSICAL_COLS,
        action_bounds={"freq_target": (0.0, 80.0), "eev": (69.0, 480.0), "fan_out": (0.0, 900.0)},
        config=config,
        device=torch.device("cpu"),
    )


class BaselineControllerTest(unittest.TestCase):
    def setUp(self):
        self.fixed_config = load_controller_config("control/config/experiments/fixed.yml")
        self.pid_config = load_controller_config("control/config/experiments/pid.yml")
        self.bounds1 = action_bounds_for_mode(
            self.pid_config,
            1,
            {"freq_target": (0, 80), "eev": (0, 480), "fan_out": (0, 900)},
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
        self.assertEqual(self.bounds1["freq_target"], (0.0, 80.0))
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
        model = _tiny_hanwam(obs_dim=6, frames_per_block=2, history_blocks=2, future_blocks=3)
        obs_history = torch.zeros(5, 2, 2, 6)
        act_history = torch.zeros(5, 2, 2, 3)
        future_act = torch.zeros(5, 3, 2, 3)
        latent, physical = model.rollout(obs_history, act_history, future_act)
        self.assertEqual(tuple(latent.shape), (5, 3, 8))
        self.assertEqual(tuple(physical.shape), (5, 3, 2))

    def test_hanwam_context_encoder_uses_observation_and_action_history(self):
        world = HanWAMWorldModel(obs_dim=6, action_dim=3, latent_dim=8, hidden_dim=16, action_latent_dim=4, frames_per_block=2, history_blocks=2, future_blocks=2)
        obs_a = torch.zeros(5, 2, 2, 6)
        obs_b = obs_a.clone()
        obs_b[:, 1, :, :] = 1.0
        act_a = torch.zeros(5, 2, 2, 3)
        act_b = act_a.clone()
        act_b[:, 1, :, 0] = 1.0
        obs_delta = (world.encode_context(obs_a, act_a) - world.encode_context(obs_b, act_a)).detach().abs().max()
        act_delta = (world.encode_context(obs_a, act_a) - world.encode_context(obs_a, act_b)).detach().abs().max()
        self.assertGreater(float(obs_delta), 0.0)
        self.assertGreater(float(act_delta), 0.0)

    def test_hanwam_controller_uses_runtime_planner_overrides(self):
        model = _tiny_hanwam(frames_per_block=2, history_blocks=2, future_blocks=1)
        checkpoint = {
            "obs_cols": WAM_OBS_COLS,
            "target_action_cols": WAM_ACTION_COLS,
            "physical_cols": WAM_PHYSICAL_COLS,
            "obs_norm": {"mean": [0.0] * len(WAM_OBS_COLS), "std": [1.0] * len(WAM_OBS_COLS)},
            "target_action_norm": {"mean": [0.0] * len(WAM_ACTION_COLS), "std": [1.0] * len(WAM_ACTION_COLS)},
            "physical_norm": {"mean": [0.0] * len(WAM_PHYSICAL_COLS), "std": [1.0] * len(WAM_PHYSICAL_COLS)},
            "target_action_bounds": {
                "freq_target": [0.0, 80.0],
                "eev": [69.0, 480.0],
                "fan_out": [0.0, 900.0],
            },
            "model_config": {
                "class_name": "HanWAMBlockControllerModel",
                "physical_dim": len(WAM_PHYSICAL_COLS),
                "prober_hidden_dim": 16,
                "prober_action_scale": 0.0,
                "action_mean": [0.0] * len(WAM_ACTION_COLS),
                "action_std": [1.0] * len(WAM_ACTION_COLS),
                "physical_mean": [0.0] * len(WAM_PHYSICAL_COLS),
                "physical_std": [1.0] * len(WAM_PHYSICAL_COLS),
                "obs_mean": [0.0] * len(WAM_OBS_COLS),
                "obs_std": [1.0] * len(WAM_OBS_COLS),
                "t_in_obs_index": WAM_OBS_COLS.index("T_in"),
                "t_out_obs_index": WAM_OBS_COLS.index("T_out"),
                "world_model_config": {
                    "class_name": "HanWAMWorldModel",
                    "obs_dim": len(WAM_OBS_COLS),
                    "action_dim": len(WAM_ACTION_COLS),
                    "latent_dim": 8,
                    "hidden_dim": 16,
                    "action_latent_dim": 4,
                    "frames_per_block": 2,
                    "history_blocks": 2,
                    "future_blocks": 1,
                },
            },
            "model": model.state_dict(),
            "architecture_version": "hanwam_block_v1",
            "planner_config": {"algorithm": "mppi", "horizon_steps": 2, "frames_per_block": 2, "future_blocks": 1, "cost_weights": {"energy": 2.0}},
            "frames_per_block": 2,
            "history_blocks": 2,
            "future_blocks": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/hanwam.pt"
            torch.save(checkpoint, path)
            controller = HanWAMController(
                path,
                mode=1,
                device="cpu",
                planner_config={"horizon_steps": 2, "cost_weights": {"action_smooth": 0.05}},
            )
        self.assertEqual(controller.planner.horizon_steps, 2)
        self.assertEqual(controller.planner.cost_weights["energy"], 2.0)
        self.assertEqual(controller.planner.cost_weights["action_smooth"], 0.05)

    def test_phase_clamp_cost_parts_are_current_objective(self):
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "horizon_steps": 4,
                "chunk_steps": 1,
                "step_seconds": 5,
                "first_principles": {
                    "phase": {"clamp_center_weight": 0.0},
                    "weights": {"reach": 1.0, "clamp": 1.0, "energy_post": 1.0, "action_post": 0.0},
                },
            }
        )
        idx = {name: i for i, name in enumerate(WAM_PHYSICAL_COLS)}
        physical = torch.zeros(1, 4, len(WAM_PHYSICAL_COLS))
        physical[0, :, idx["electric_kwh_delta"]] = 0.001
        actions = torch.zeros(1, 4, len(WAM_ACTION_COLS))
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0

        _, parts = planner._cost(
            physical,
            actions,
            target=27.0,
            initial_t_in=27.0,
            remaining_seconds=0.0,
            observation=observation,
        )

        self.assertEqual(set(parts), {"reach", "clamp", "energy", "action"})

    def test_phase_clamp_penalizes_post_deadline_high_and_low_temperature(self):
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "horizon_steps": 2,
                "chunk_steps": 1,
                "first_principles": {
                    "phase": {
                        "clamp_upper_c": 0.35,
                        "clamp_lower_c": 0.35,
                        "clamp_center_weight": 0.0,
                    },
                    "weights": {"reach": 0.0, "clamp": 1.0, "energy_post": 0.0, "action_post": 0.0},
                },
            }
        )
        idx = {name: i for i, name in enumerate(WAM_PHYSICAL_COLS)}
        physical = torch.zeros(3, 2, len(WAM_PHYSICAL_COLS))
        physical[1, :, idx["T_in_delta"]] = torch.tensor([0.50, 0.0])
        physical[2, :, idx["T_in_delta"]] = torch.tensor([-0.50, 0.0])
        actions = torch.zeros(3, 2, len(WAM_ACTION_COLS))
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0

        cost, parts = planner._cost(
            physical,
            actions,
            target=27.0,
            initial_t_in=27.0,
            remaining_seconds=0.0,
            observation=observation,
        )

        self.assertAlmostEqual(float(parts["clamp"][0]), 0.0)
        self.assertGreater(float(parts["clamp"][1]), 0.0)
        self.assertGreater(float(parts["clamp"][2]), 0.0)
        self.assertLess(float(cost[0]), float(cost[1]))
        self.assertLess(float(cost[0]), float(cost[2]))

    def test_phase_clamp_switches_to_clamp_after_predicted_reach(self):
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "horizon_steps": 4,
                "chunk_steps": 1,
                "step_seconds": 5,
                "first_principles": {
                    "phase": {
                        "reach_deadline_fraction": 1.0,
                        "reach_band_c": 0.45,
                        "hold_activation_c": 0.50,
                        "clamp_upper_c": 0.20,
                        "clamp_lower_c": 0.25,
                        "clamp_center_weight": 0.0,
                    },
                    "weights": {"reach": 0.0, "clamp": 1.0, "energy_pre": 0.0, "action_pre": 0.0},
                },
            }
        )
        idx = {name: i for i, name in enumerate(WAM_PHYSICAL_COLS)}
        physical = torch.zeros(2, 4, len(WAM_PHYSICAL_COLS))
        physical[0, :, idx["T_in_delta"]] = torch.tensor([-0.10, -0.10, -0.10, -0.10])
        physical[1, :, idx["T_in_delta"]] = torch.tensor([-0.60, -0.60, -0.60, -0.60])
        actions = torch.zeros(2, 4, len(WAM_ACTION_COLS))
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 28.0

        _, parts = planner._cost(
            physical,
            actions,
            target=27.0,
            initial_t_in=28.0,
            remaining_seconds=600.0,
            observation=observation,
        )

        self.assertAlmostEqual(float(parts["clamp"][0]), 0.0)
        self.assertGreater(float(parts["clamp"][1]), 0.0)

    def test_phase_clamp_chunk_action_cost_is_not_diluted_by_repeated_steps(self):
        model = _tiny_hanwam(frames_per_block=12, history_blocks=1, future_blocks=2)
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "frames_per_block": 12,
                "history_blocks": 1,
                "future_blocks": 2,
                "horizon_steps": 24,
                "chunk_steps": 12,
                "first_principles": {
                    "phase": {"clamp_center_weight": 0.0},
                    "action": {
                        "reference": "last_issued_command",
                        "loss": "mse",
                        "post_first_weight": 0.0,
                        "post_sequence_weight": 1.0,
                    },
                    "weights": {"reach": 0.0, "clamp": 0.0, "energy_post": 0.0, "action_post": 1.0},
                },
            },
            model=model,
        )
        physical = torch.zeros(2, 24, len(WAM_PHYSICAL_COLS))
        steady = torch.tensor([[0.0, 165.0, 0.0]] * 24)
        jump = steady.clone()
        jump[12:, 0] = 80.0
        actions = torch.stack([steady, jump], dim=0)
        history = np.asarray([[[0.0, 165.0, 0.0]] * 12], dtype=np.float32)
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0

        _, parts = planner._cost(
            physical,
            actions,
            target=27.0,
            initial_t_in=27.0,
            remaining_seconds=None,
            observation=observation,
            current_action=np.asarray([0.0, 165.0, 0.0], dtype=np.float32),
            act_history_blocks=history,
        )

        self.assertAlmostEqual(float(parts["action"][0]), 0.0)
        self.assertAlmostEqual(float(parts["action"][1]), 1.0 / 3.0, places=6)

    def test_phase_clamp_recent_action_reference_can_use_history_ema(self):
        history = np.asarray(
            [[[0.0, 165.0, 0.0]] * 11 + [[80.0, 165.0, 0.0]]],
            dtype=np.float32,
        )
        mean_planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "frames_per_block": 12,
                "history_blocks": 1,
                "future_blocks": 1,
                "horizon_steps": 12,
                "chunk_steps": 12,
            },
            model=_tiny_hanwam(frames_per_block=12, history_blocks=1, future_blocks=1),
        )
        ema_planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "frames_per_block": 12,
                "history_blocks": 1,
                "future_blocks": 1,
                "horizon_steps": 12,
                "chunk_steps": 12,
                "action_history_ema_tau_seconds": 10.0,
            },
            model=_tiny_hanwam(frames_per_block=12, history_blocks=1, future_blocks=1),
        )

        mean_reference = mean_planner._recent_action_reference(history, None)
        ema_reference = ema_planner._recent_action_reference(history, None)

        self.assertAlmostEqual(float(mean_reference[0]), 80.0 / 12.0, places=5)
        self.assertGreater(float(ema_reference[0]), float(mean_reference[0]))
        self.assertLess(float(ema_reference[0]), 80.0)
        self.assertAlmostEqual(float(ema_reference[1]), 165.0, places=5)

    def test_phase_clamp_energy_normalization_uses_horizon_reference(self):
        model = _tiny_hanwam(frames_per_block=12, history_blocks=1, future_blocks=8)
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "frames_per_block": 12,
                "history_blocks": 1,
                "future_blocks": 8,
                "horizon_steps": 96,
                "chunk_steps": 12,
                "step_seconds": 5,
                "first_principles": {
                    "energy": {"reference_kwh_per_hour": 0.6},
                    "phase": {"clamp_center_weight": 0.0},
                    "weights": {"reach": 0.0, "clamp": 0.0, "energy_post": 1.0, "action_post": 0.0},
                },
            },
            model=model,
        )
        idx = {name: i for i, name in enumerate(WAM_PHYSICAL_COLS)}
        physical = torch.zeros(1, 8, len(WAM_PHYSICAL_COLS))
        physical[0, :, idx["electric_kwh_delta"]] = 0.08 / 8.0
        actions = torch.zeros(1, 96, len(WAM_ACTION_COLS))
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0

        _, parts = planner._cost(
            physical,
            actions,
            target=27.0,
            initial_t_in=27.0,
            remaining_seconds=None,
            observation=observation,
        )

        self.assertAlmostEqual(float(parts["energy"][0]), 1.0, places=6)

    def test_phase_clamp_action_reference_uses_last_issued_command(self):
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "horizon_steps": 2,
                "chunk_steps": 1,
                "first_principles": {
                    "action": {
                        "reference": "last_issued_command",
                        "loss": "mse",
                        "post_first_weight": 1.0,
                        "post_sequence_weight": 0.0,
                    },
                    "phase": {"clamp_center_weight": 0.0},
                    "weights": {"reach": 0.0, "clamp": 0.0, "energy_post": 0.0, "action_post": 1.0},
                },
            }
        )
        physical = torch.zeros(1, 2, len(WAM_PHYSICAL_COLS))
        actions = torch.tensor([[[20.0, 165.0, 300.0], [20.0, 165.0, 300.0]]])
        history = np.asarray([[[80.0, 300.0, 900.0], [80.0, 300.0, 900.0]]], dtype=np.float32)
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0

        _, parts = planner._cost(
            physical,
            actions,
            target=27.0,
            initial_t_in=27.0,
            remaining_seconds=0.0,
            observation=observation,
            current_action=np.asarray([20.0, 165.0, 300.0], dtype=np.float32),
            act_history_blocks=history,
        )

        self.assertAlmostEqual(float(parts["action"][0]), 0.0)

    def test_phase_clamp_slew_projection_limits_first_and_future_chunks(self):
        model = _tiny_hanwam(frames_per_block=12, history_blocks=1, future_blocks=2)
        planner = _tiny_mppi_planner(
            {
                "objective": "phase_energy_clamp",
                "frames_per_block": 12,
                "history_blocks": 1,
                "future_blocks": 2,
                "horizon_steps": 24,
                "chunk_steps": 12,
                "control_interval_steps": 2,
                "step_seconds": 5,
                "first_principles": {
                    "slew": {
                        "pre_deadline": {"freq": 8, "eev": 20, "fan_out": 100},
                        "post_deadline": {"freq": 4, "eev": 10, "fan_out": 50},
                    }
                },
            },
            model=model,
        )

        chunks = torch.tensor([[[80.0, 300.0, 900.0], [80.0, 300.0, 900.0]]])
        projected = planner._project_action_chunks(
            chunks,
            current_action=np.asarray([0.0, 165.0, 0.0], dtype=np.float32),
            remaining_seconds=2400.0,
        )[0]

        self.assertAlmostEqual(float(projected[0, 0]), 15.0, places=5)
        self.assertAlmostEqual(float(projected[0, 1]), 185.0, places=5)
        self.assertAlmostEqual(float(projected[0, 2]), 100.0, places=5)
        self.assertAlmostEqual(float(projected[1, 0]), 63.0, places=5)
        self.assertAlmostEqual(float(projected[1, 1]), 300.0, places=5)
        self.assertAlmostEqual(float(projected[1, 2]), 700.0, places=5)

        post_chunks = torch.tensor([[[80.0, 300.0, 900.0], [0.0, 69.0, 0.0]]])
        post_projected = planner._project_action_chunks(
            post_chunks,
            current_action=np.asarray([40.0, 200.0, 500.0], dtype=np.float32),
            remaining_seconds=0.0,
        )[0]

        self.assertAlmostEqual(float(post_projected[0, 0]), 44.0, places=5)
        self.assertAlmostEqual(float(post_projected[0, 1]), 210.0, places=5)
        self.assertAlmostEqual(float(post_projected[0, 2]), 550.0, places=5)
        self.assertAlmostEqual(float(post_projected[1, 0]), 20.0, places=5)
        self.assertAlmostEqual(float(post_projected[1, 1]), 150.0, places=5)
        self.assertAlmostEqual(float(post_projected[1, 2]), 250.0, places=5)

    def test_sigreg_penalizes_collapsed_latents(self):
        collapsed = torch.zeros(16, 8)
        varied = torch.randn(16, 8)
        collapsed_loss = HanWAMWorldModel.sigreg_loss(collapsed, num_projections=16)[0]
        varied_loss = HanWAMWorldModel.sigreg_loss(varied, num_projections=16)[0]
        self.assertGreater(float(collapsed_loss), float(varied_loss))

    def test_hanwam_stage1_state_dict_has_no_prober_and_stage2_freezes_world_model(self):
        world = HanWAMWorldModel(
            obs_dim=len(WAM_OBS_COLS),
            action_dim=len(WAM_ACTION_COLS),
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=4,
            frames_per_block=2,
            history_blocks=2,
            future_blocks=2,
        )
        self.assertFalse(any(key.startswith("prober") for key in world.state_dict()))
        controller = _tiny_hanwam(frames_per_block=2, history_blocks=2, future_blocks=2)
        controller.freeze_world_model()
        self.assertFalse(any(param.requires_grad for param in controller.world_model_parameters()))
        self.assertTrue(all(param.requires_grad for param in controller.prober_parameters()))

    def test_mppi_softmax_planner_outputs_bounded_deadband_action(self):
        model = _tiny_hanwam(frames_per_block=2, history_blocks=2, future_blocks=1)
        norm_obs = Normalizer(mean=np.zeros(len(WAM_OBS_COLS), dtype=np.float32), std=np.ones(len(WAM_OBS_COLS), dtype=np.float32))
        norm_action = Normalizer(mean=np.zeros(len(WAM_ACTION_COLS), dtype=np.float32), std=np.ones(len(WAM_ACTION_COLS), dtype=np.float32))
        norm_phys = Normalizer(mean=np.zeros(len(WAM_PHYSICAL_COLS), dtype=np.float32), std=np.ones(len(WAM_PHYSICAL_COLS), dtype=np.float32))
        planner = MPPIPlanner(
            model=model,
            obs_norm=norm_obs,
            action_norm=norm_action,
            physical_norm=norm_phys,
            obs_cols=WAM_OBS_COLS,
            physical_cols=WAM_PHYSICAL_COLS,
            action_bounds={"freq_target": (0.0, 80.0), "eev": (69.0, 480.0), "fan_out": (0.0, 900.0)},
            config={
                "horizon_steps": 2,
                "frames_per_block": 2,
                "future_blocks": 1,
                "history_blocks": 2,
                "chunk_steps": 1,
                "num_samples": 8,
                "num_iterations": 1,
                "temperature": 1.0,
                "cost_weights": {"energy": 0.0, "action_smooth": 0.0},
            },
            device=torch.device("cpu"),
        )
        observation = np.zeros(len(WAM_OBS_COLS), dtype=np.float32)
        observation[WAM_OBS_COLS.index("T_in")] = 27.0
        obs_hist = np.zeros((2, 2, len(WAM_OBS_COLS)), dtype=np.float32)
        act_hist = np.zeros((2, 2, len(WAM_ACTION_COLS)), dtype=np.float32)
        result = planner.plan(
            observation,
            target=27.0,
            obs_history_blocks=obs_hist,
            act_history_blocks=act_hist,
            current_action=np.asarray([10.0, 165.0, 750.0], dtype=np.float32),
        )
        self.assertEqual(result.debug["hanwam_planner"], "mppi")
        self.assertGreaterEqual(float(result.action[0]), 0.0)
        self.assertLessEqual(float(result.action[0]), 80.0)
        self.assertTrue(float(result.action[0]) == 0.0 or float(result.action[0]) >= 15.0)

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
