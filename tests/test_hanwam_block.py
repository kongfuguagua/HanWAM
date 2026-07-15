from __future__ import annotations

import unittest

import numpy as np
import torch

from control.HanWAM.dataloader import Normalizer
from control.HanWAM.model import (
    HanWAMBlockControllerModel,
    HanWAMWorldModel,
)
from control.HanWAM.planner import MPPIPlanner
from control.HanWAM.type import WAM_ACTION_COLS, WAM_OBS_COLS, WAM_PHYSICAL_COLS
from control.HanWAM.train import _controller_model_config
from control.MiniController.config_schema import load_config


def build_model() -> HanWAMBlockControllerModel:
    world = HanWAMWorldModel(
        obs_dim=len(WAM_OBS_COLS),
        action_dim=len(WAM_ACTION_COLS),
        latent_dim=12,
        hidden_dim=20,
        action_latent_dim=8,
        tcn_layers=1,
        frames_per_block=2,
        history_blocks=2,
        future_blocks=2,
    )
    return HanWAMBlockControllerModel(
        world,
        hidden_dim=20,
        action_mean=[40.0, 185.0, 750.0],
        action_std=[20.0, 70.0, 1.0],
        physical_mean=[0.0, 0.0],
        physical_std=[0.2, 0.01],
        obs_mean=[0.0] * len(WAM_OBS_COLS),
        obs_std=[1.0] * len(WAM_OBS_COLS),
        t_in_obs_index=list(WAM_OBS_COLS).index("T_in"),
        t_out_obs_index=list(WAM_OBS_COLS).index("T_out"),
        step_seconds=5.0,
    )


class HanWAMBlockWorldModelTest(unittest.TestCase):
    def test_epps_pulley_sigreg_prefers_gaussian_over_collapse(self):
        torch.manual_seed(17)
        gaussian = torch.randn(512, 4, 16)
        collapsed = torch.zeros_like(gaussian)
        gaussian_loss, _, _ = HanWAMWorldModel.sigreg_loss(
            gaussian,
            num_projections=32,
            quadrature_points=17,
        )
        torch.manual_seed(17)
        collapsed_loss, _, _ = HanWAMWorldModel.sigreg_loss(
            collapsed,
            num_projections=32,
            quadrature_points=17,
        )
        self.assertTrue(torch.isfinite(gaussian_loss))
        self.assertLess(float(gaussian_loss), float(collapsed_loss))

    def test_stage1_encodes_history_once_and_regularizes_predicted_rollout(self):
        torch.manual_seed(19)
        model = build_model().world_model
        calls = {"obs": 0, "act": 0}
        obs_hook = model.obs_encoder.register_forward_hook(
            lambda *_: calls.__setitem__("obs", calls["obs"] + 1)
        )
        act_hook = model.act_history_encoder.register_forward_hook(
            lambda *_: calls.__setitem__("act", calls["act"] + 1)
        )
        try:
            batch = 8
            breakdown = model.latent_sequence_loss(
                torch.randn(batch, 2, 2, len(WAM_OBS_COLS)),
                torch.randn(batch, 2, 2, len(WAM_ACTION_COLS)),
                torch.randn(batch, 2, 2, len(WAM_ACTION_COLS)),
                torch.randn(batch, 2, 2, len(WAM_OBS_COLS)),
                sigreg_num_projections=8,
                sigreg_quadrature_points=9,
            )
        finally:
            obs_hook.remove()
            act_hook.remove()
        # One history obs encode plus one vectorized future-target encode.
        self.assertEqual(calls["obs"], 2)
        self.assertEqual(calls["act"], 1)
        self.assertTrue(torch.isfinite(breakdown.total))
        self.assertGreaterEqual(float(breakdown.regularizer.detach()), 0.0)


class HanWAMBlockControllerTest(unittest.TestCase):
    def _inputs(self, batch: int = 1):
        obs = torch.zeros(batch, 2, 2, len(WAM_OBS_COLS))
        obs[..., list(WAM_OBS_COLS).index("T_in")] = 30.0
        obs[..., list(WAM_OBS_COLS).index("T_out")] = 35.0
        act = torch.zeros(batch, 2, 2, len(WAM_ACTION_COLS))
        future = torch.zeros(batch, 2, 2, len(WAM_ACTION_COLS))
        return obs, act, future

    def test_parameter_prober_returns_block_trajectory(self):
        model = build_model()
        obs, act, future = self._inputs(batch=3)
        latents, physical = model.rollout(obs, act, future)
        self.assertEqual(tuple(latents.shape), (3, 2, model.latent_dim))
        self.assertEqual(tuple(physical.shape), (3, 2, 2))
        self.assertTrue(torch.isfinite(physical).all())

    def test_stage2_gradients_do_not_enter_world_model(self):
        torch.manual_seed(23)
        model = build_model()
        model.freeze_world_model()
        obs, act, future = self._inputs(batch=4)
        target = torch.ones(4, 2, 2)
        loss, _ = model.prober_sequence_loss(obs, act, future, target)
        loss.backward()
        prober_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.prober_parameters()
            if parameter.grad is not None
        )
        world_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.world_model_parameters()
            if parameter.grad is not None
        )
        self.assertGreater(prober_grad, 0.0)
        self.assertEqual(world_grad, 0.0)

    def test_future_latent_changes_physical_output(self):
        torch.manual_seed(29)
        model = build_model()
        obs, act, future = self._inputs()
        context = model.prepare_context(obs, act)
        latents = model.world_model.rollout_latents(context.state_latent, future)
        normal = model.probe_prepared(context, latents, future)
        perturbed = model.probe_prepared(context, latents + 2.0, future)
        self.assertGreater(float((normal - perturbed).abs().max().detach()), 1e-6)

    def test_off_action_has_zero_mechanism_energy(self):
        model = build_model()
        obs, act, future = self._inputs()
        # Normalized values decoding to raw [0 Hz, 100 EEV, 750 rpm].
        future[..., 0] = -2.0
        future[..., 1] = (100.0 - 185.0) / 70.0
        future[..., 2] = 0.0
        context = model.prepare_context(obs, act)
        _, physical_n = model.rollout_prepared(context, future)
        physical = physical_n * model.physical_std + model.physical_mean
        self.assertTrue(torch.equal(physical[..., 1], torch.zeros_like(physical[..., 1])))

    def test_planner_encodes_history_once_and_keeps_four_cost_parts(self):
        torch.manual_seed(31)
        model = build_model()
        obs_calls = {"count": 0}
        act_calls = {"count": 0}
        obs_hook = model.world_model.obs_encoder.register_forward_hook(
            lambda *_: obs_calls.__setitem__("count", obs_calls["count"] + 1)
        )
        act_hook = model.world_model.act_history_encoder.register_forward_hook(
            lambda *_: act_calls.__setitem__("count", act_calls["count"] + 1)
        )
        obs_norm = Normalizer(np.zeros(len(WAM_OBS_COLS), np.float32), np.ones(len(WAM_OBS_COLS), np.float32))
        action_norm = Normalizer(
            np.asarray([40.0, 185.0, 750.0], np.float32),
            np.asarray([20.0, 70.0, 1.0], np.float32),
        )
        physical_norm = Normalizer(np.zeros(2, np.float32), np.asarray([0.2, 0.01], np.float32))
        planner = MPPIPlanner(
            model=model,
            obs_norm=obs_norm,
            action_norm=action_norm,
            physical_norm=physical_norm,
            obs_cols=list(WAM_OBS_COLS),
            physical_cols=list(WAM_PHYSICAL_COLS),
            action_bounds={
                "freq_target": (10.0, 80.0),
                "eev": (100.0, 270.0),
                "fan_out": (750.0, 750.0),
            },
            config={
                "timing": {
                    "step_seconds": 5,
                    "frames_per_block": 2,
                    "history_blocks": 2,
                    "future_blocks": 2,
                    "chunk_steps": 2,
                    "horizon_steps": 4,
                    "control_interval_steps": 1,
                },
                "sampling": {"num_samples": 8, "num_iterations": 2, "seed": 9},
                "first_principles": {
                    "weights": {
                        "reach": 1.0,
                        "clamp": 1.0,
                        "energy_pre": 1.0,
                        "action_pre": 1.0,
                    }
                },
            },
            device=torch.device("cpu"),
        )
        observation = np.zeros(len(WAM_OBS_COLS), np.float32)
        observation[list(WAM_OBS_COLS).index("T_in")] = 30.0
        observation[list(WAM_OBS_COLS).index("T_out")] = 35.0
        current = np.asarray([40.0, 185.0, 750.0], np.float32)
        obs_history = np.repeat(observation[None], 4, axis=0).reshape(2, 2, -1)
        act_history = np.repeat(current[None], 4, axis=0).reshape(2, 2, -1)
        try:
            result = planner.plan(
                observation,
                target=27.0,
                remaining_seconds=600.0,
                obs_history_blocks=obs_history,
                act_history_blocks=act_history,
                current_action=current,
            )
        finally:
            obs_hook.remove()
            act_hook.remove()
        self.assertEqual(obs_calls["count"], 1)
        self.assertEqual(act_calls["count"], 1)
        self.assertEqual(result.debug["hanwam_history_encode_count"], 1)
        self.assertEqual(result.debug["hanwam_physical_horizon_nodes"], 2)
        for part in ("reach", "clamp", "energy", "action"):
            self.assertIn(f"hanwam_cost_{part}", result.debug)

    def test_abc_acceptance_matches_dataset_readme(self):
        config = load_config("control/HanWAM/config/hanwam.yml")
        scenarios = config["eval"]["scenarios"]
        self.assertEqual([row["condition"]["name"] for row in scenarios], ["A", "B", "C"])
        self.assertEqual([row["reach_deadline_seconds"] for row in scenarios], [2400, 2400, 2400])
        self.assertEqual(scenarios[1]["condition"]["indoor_initial_wet_bulb_c"], 24)
        self.assertEqual(float(config["eval"]["comfort_lower_band_c"]), 1.0)
        self.assertEqual(float(config["eval"]["comfort_upper_band_c"]), 0.5)
        self.assertEqual(
            [row["success_requires_post_deadline_band"] for row in scenarios],
            [True, True, False],
        )
        self.assertFalse(scenarios[2]["success_requires_final_band"])
        model_config = _controller_model_config(config)
        self.assertNotIn("cooling_state_order", model_config)


if __name__ == "__main__":
    unittest.main()
