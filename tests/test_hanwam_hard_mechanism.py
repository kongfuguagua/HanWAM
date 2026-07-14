from __future__ import annotations

import unittest
from pathlib import Path

import torch

from control.HanWAM.dataloader import action_bounds
from control.HanWAM.model import HanWAMHardMechanismControllerModel, HanWAMWorldModel
from control.MiniController.config_schema import load_config


class HanWAMHardMechanismGradientTest(unittest.TestCase):
    def test_stage2_loss_updates_only_state_prober(self):
        torch.manual_seed(20260711)
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=2,
            history_blocks=2,
            future_blocks=2,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[40.0, 165.0, 450.0],
            action_std=[20.0, 80.0, 300.0],
            physical_mean=[0.0, 0.0],
            physical_std=[0.02, 0.001],
            obs_mean=[0.0] * 12,
            obs_std=[1.0] * 12,
            t_in_obs_index=7,
            t_out_obs_index=8,
        )
        model.freeze_world_model()

        obs_history = torch.zeros(3, 2, 2, 12)
        obs_history[:, -1, -1, 7] = 28.0
        obs_history[:, -1, -1, 8] = 35.0
        act_history = torch.zeros(3, 2, 2, 3)
        future_actions = torch.zeros(3, 2, 2, 3)
        future_actions[..., 0] = 1.0
        future_actions[..., 1] = 0.5
        future_actions[..., 2] = 1.0
        physical_targets = torch.ones(3, 4, 2)

        loss, prediction = model.prober_sequence_loss(
            obs_history,
            act_history,
            future_actions,
            physical_targets,
            physical_weights=torch.tensor([2.0, 1.0]),
        )
        self.assertEqual(tuple(prediction.shape), (3, 4, 2))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        prober_grad = sum(
            float(param.grad.detach().abs().sum())
            for param in model.prober_parameters()
            if param.grad is not None
        )
        world_grad = sum(
            float(param.grad.detach().abs().sum())
            for param in model.world_model_parameters()
            if param.grad is not None
        )
        self.assertGreater(prober_grad, 0.0)
        self.assertEqual(world_grad, 0.0)

    def test_frequency_polynomial_energy_matches_five_second_mechanism(self):
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=2,
            history_blocks=2,
            future_blocks=2,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 1.0, 1.0],
            physical_mean=[0.0, 0.0],
            physical_std=[1.0, 1.0],
            energy_model="frequency_polynomial",
            energy_step_seconds=5.0,
        )
        block_latents = torch.zeros(1, 2, 8)
        future_actions = torch.zeros(1, 2, 2, 3)
        frequencies = torch.tensor([0.0, 20.0, 40.0, 80.0])
        future_actions[..., 0] = frequencies.reshape(1, 2, 2)

        prediction = model.probe_blocks(block_latents, future_actions)
        expected = 5.0 / 3.6e6 * (
            329.87 + 9.0034 * frequencies + 0.04494 * frequencies.pow(2.0)
        )

        torch.testing.assert_close(prediction[0, :, 1], expected)

    def test_cooling_lag_uses_latest_history_action_as_initial_state(self):
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=2,
            history_blocks=2,
            future_blocks=1,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 1.0, 1.0],
            physical_mean=[0.0, 0.0],
            physical_std=[1.0, 1.0],
            obs_mean=[0.0] * 12,
            obs_std=[1.0] * 12,
            t_in_obs_index=7,
            t_out_obs_index=8,
            ua_delta_scale=0.0,
            internal_delta_scale=0.0,
            cooling_delta_scale=0.10,
            cooling_lag_enabled=True,
            cooling_lag_alpha_min=0.80,
            cooling_lag_alpha_max=0.80,
        )
        for param in model.prober["mechanism"].parameters():
            param.data.zero_()

        obs_history = torch.zeros(1, 2, 2, 12)
        obs_history[:, -1, -1, 7] = 27.0
        obs_history[:, -1, -1, 8] = 27.0
        future_actions = torch.zeros(1, 1, 2, 3)
        future_actions[..., 0] = 20.0
        future_actions[..., 1] = 240.0
        future_actions[..., 2] = 750.0

        low_history = torch.zeros(1, 2, 2, 3)
        low_history[..., 0] = 20.0
        low_history[..., 1] = 240.0
        low_history[..., 2] = 750.0
        high_history = low_history.clone()
        high_history[:, -1, -1, 0] = 80.0

        _, pred_low = model.rollout(obs_history, low_history, future_actions)
        _, pred_high = model.rollout(obs_history, high_history, future_actions)

        self.assertLess(float(pred_high[0, 0, 0].detach()), float(pred_low[0, 0, 0].detach()))
        self.assertLess(float(pred_high[0, 1, 0].detach()), float(pred_low[0, 1, 0].detach()))

    def test_frequency_polynomial_energy_accepts_linear_eev_correction(self):
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=2,
            history_blocks=2,
            future_blocks=1,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 1.0, 1.0],
            physical_mean=[0.0, 0.0],
            physical_std=[1.0, 1.0],
            energy_model="frequency_polynomial",
            energy_step_seconds=5.0,
            energy_eev_correction_scale_w=120.0,
            energy_eev_anchor=240.0,
            energy_eev_range=170.0,
        )
        for param in model.prober["mechanism"].parameters():
            param.data.zero_()
        final_linear = model.prober["mechanism"][-1]
        final_linear.bias.data[model.energy_eev_correction_index] = 1.0

        block_latents = torch.zeros(1, 1, 8)
        future_actions = torch.zeros(1, 1, 2, 3)
        future_actions[..., 0] = 40.0
        future_actions[0, 0, 0, 1] = 240.0
        future_actions[0, 0, 1, 1] = 270.0
        future_actions[..., 2] = 750.0

        prediction = model.probe_blocks(block_latents, future_actions)
        expected_extra = 5.0 / 3.6e6 * 120.0 * torch.tanh(torch.tensor(1.0)) * (30.0 / 170.0)

        torch.testing.assert_close(prediction[0, 1, 1] - prediction[0, 0, 1], expected_extra)

    def test_positive_offset_eev_energy_is_monotonic_and_bounded(self):
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=3,
            history_blocks=2,
            future_blocks=1,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 1.0, 1.0],
            physical_mean=[0.0, 0.0],
            physical_std=[1.0, 1.0],
            energy_model="frequency_polynomial",
            energy_step_seconds=5.0,
            energy_power_intercept_w=279.87,
            energy_eev_correction_mode="positive_offset",
            energy_eev_correction_min_w=50.0,
            energy_eev_correction_max_w=150.0,
            energy_eev_anchor=100.0,
            energy_eev_range=170.0,
        )
        for param in model.prober["mechanism"].parameters():
            param.data.zero_()

        block_latents = torch.zeros(1, 1, 8)
        future_actions = torch.zeros(1, 1, 3, 3)
        future_actions[..., 0] = 40.0
        future_actions[0, 0, :, 1] = torch.tensor([100.0, 240.0, 270.0])
        future_actions[..., 2] = 750.0

        prediction = model.probe_blocks(block_latents, future_actions)
        energy = prediction[0, :, 1].detach()
        coeff_w = 100.0
        expected_extra_270 = 5.0 / 3.6e6 * coeff_w

        self.assertLess(float(energy[0]), float(energy[1]))
        self.assertLess(float(energy[1]), float(energy[2]))
        torch.testing.assert_close(energy[2] - energy[0], torch.tensor(expected_extra_270))

    def test_lagged_incremental_cooling_preserves_baseline_when_future_matches_history(self):
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=3,
            history_blocks=2,
            future_blocks=1,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 1.0, 1.0],
            physical_mean=[0.0, 0.0],
            physical_std=[1.0, 1.0],
            obs_mean=[0.0] * 12,
            obs_std=[1.0] * 12,
            t_in_obs_index=7,
            t_out_obs_index=8,
            energy_model="frequency_polynomial",
            temperature_mechanism="lagged_incremental_cooling",
            baseline_drift_scale_c=0.0,
            cooling_delta_scale=0.08,
            cooling_lag_alpha_min=0.80,
            cooling_lag_alpha_max=0.80,
        )
        for param in model.prober["mechanism"].parameters():
            param.data.zero_()

        obs_history = torch.zeros(1, 2, 3, 12)
        obs_history[:, :, :, 7] = 27.0
        obs_history[:, :, :, 8] = 35.0
        act_history = torch.zeros(1, 2, 3, 3)
        act_history[..., 0] = 40.0
        act_history[..., 1] = 240.0
        act_history[..., 2] = 750.0
        future_actions = torch.zeros(1, 1, 3, 3)
        future_actions[..., 0] = 40.0
        future_actions[..., 1] = 240.0
        future_actions[..., 2] = 750.0

        _, prediction = model.rollout(obs_history, act_history, future_actions)

        torch.testing.assert_close(prediction[0, :, 0], torch.zeros(3), atol=1e-5, rtol=1e-5)

    def test_lagged_incremental_cooling_cools_more_when_future_frequency_exceeds_history(self):
        world = HanWAMWorldModel(
            obs_dim=12,
            action_dim=3,
            latent_dim=8,
            hidden_dim=16,
            action_latent_dim=6,
            tcn_layers=1,
            frames_per_block=3,
            history_blocks=2,
            future_blocks=1,
        )
        model = HanWAMHardMechanismControllerModel(
            world,
            hidden_dim=16,
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 1.0, 1.0],
            physical_mean=[0.0, 0.0],
            physical_std=[1.0, 1.0],
            obs_mean=[0.0] * 12,
            obs_std=[1.0] * 12,
            t_in_obs_index=7,
            t_out_obs_index=8,
            energy_model="frequency_polynomial",
            temperature_mechanism="lagged_incremental_cooling",
            baseline_drift_scale_c=0.0,
            cooling_delta_scale=0.08,
            cooling_lag_alpha_min=0.80,
            cooling_lag_alpha_max=0.80,
        )
        for param in model.prober["mechanism"].parameters():
            param.data.zero_()

        obs_history = torch.zeros(1, 2, 3, 12)
        obs_history[:, :, :, 7] = 27.0
        obs_history[:, :, :, 8] = 35.0
        act_history = torch.zeros(1, 2, 3, 3)
        act_history[..., 0] = 20.0
        act_history[..., 1] = 240.0
        act_history[..., 2] = 750.0
        baseline_actions = torch.zeros(1, 1, 3, 3)
        baseline_actions[..., 0] = 20.0
        baseline_actions[..., 1] = 240.0
        baseline_actions[..., 2] = 750.0
        higher_actions = baseline_actions.clone()
        higher_actions[..., 0] = 60.0

        _, pred_base = model.rollout(obs_history, act_history, baseline_actions)
        _, pred_high = model.rollout(obs_history, act_history, higher_actions)

        self.assertTrue(torch.all(pred_high[0, :, 0] < pred_base[0, :, 0]))

    def test_e056_action_space_is_frequency_only(self):
        config_path = Path("control/HanWAM/config/hanwam_e056_h4_f8_freq_only_poly_energy_v1.yml")
        config = load_config(config_path)

        bounds = action_bounds([], config=config)

        self.assertEqual(bounds["freq_target"], (0.0, 80.0))
        self.assertEqual(bounds["eev"], (240.0, 240.0))
        self.assertEqual(bounds["fan_out"], (750.0, 750.0))
        self.assertEqual(config["method"]["model"]["energy_model"], "frequency_polynomial")

    def test_e060_uses_e056_checkpoint_with_narrow_frequency_eev_action_space(self):
        config_path = Path("control/HanWAM/config/hanwam_e060_e56wm_phase_energy_clamp_v1.yml")
        config = load_config(config_path)

        bounds = action_bounds([], config=config)
        anchors = config["method"]["planner"]["sampling"]["proposal_action_anchors"]

        self.assertEqual(config["method"]["planner"]["objective"], "hanwam_e060_phase_energy_clamp_v1")
        self.assertIn("hanwam_e056_h4_f8_freq_only_poly_energy_v1", config["method"]["checkpoint_template"])
        self.assertEqual(bounds["freq_target"], (20.0, 80.0))
        self.assertEqual(bounds["eev"], (100.0, 270.0))
        self.assertEqual(bounds["fan_out"], (750.0, 750.0))
        self.assertEqual([row[0] for row in anchors], list(range(20, 81, 5)))
        self.assertEqual([row[1] for row in anchors], [240.0] * len(anchors))
        self.assertEqual([row[2] for row in anchors], [750.0] * len(anchors))
        self.assertEqual(config["method"]["planner"]["first_principles"]["action"]["anchor_gate_c"], 0.6)

    def test_e062_enables_lagged_mechanism_without_changing_h4_f8_shape(self):
        config_path = Path("control/HanWAM/config/hanwam_e062_h4_f8_lagged_energy_v1.yml")
        config = load_config(config_path)
        model_cfg = config["method"]["model"]

        self.assertEqual(config["train"]["stage"], "both")
        self.assertEqual(config["train"]["history_blocks"], 4)
        self.assertEqual(config["train"]["future_blocks"], 8)
        self.assertTrue(model_cfg["cooling_lag_enabled"])
        self.assertEqual(model_cfg["energy_model"], "frequency_polynomial")
        self.assertGreater(model_cfg["energy_eev_correction_scale_w"], 0.0)
        self.assertEqual(config["method"]["planner"]["objective"], "hanwam_e060_phase_energy_clamp_v1")
        weights = config["method"]["planner"]["first_principles"]["weights"]
        action = config["method"]["planner"]["first_principles"]["action"]
        self.assertEqual(weights["reach"], 80.0)
        self.assertEqual(weights["clamp"], 140.0)
        self.assertEqual(weights["action_post"], 64.0)
        self.assertEqual(action["post_anchor_weight"], 80.0)

    def test_e064_uses_lagged_incremental_cooling_with_e062_mpc_weights(self):
        config_path = Path("control/HanWAM/config/hanwam_e064_lagged_incremental_cooling_v1.yml")
        config = load_config(config_path)
        model_cfg = config["method"]["model"]
        weights = config["method"]["planner"]["first_principles"]["weights"]

        self.assertEqual(config["train"]["stage"], "stage2")
        self.assertEqual(config["train"]["history_blocks"], 4)
        self.assertEqual(config["train"]["future_blocks"], 8)
        self.assertEqual(model_cfg["temperature_mechanism"], "lagged_incremental_cooling")
        self.assertEqual(model_cfg["cooling_history_seconds"], 180.0)
        self.assertGreater(model_cfg["energy_eev_correction_scale_w"], 0.0)
        self.assertEqual(weights["reach"], 80.0)
        self.assertEqual(weights["clamp"], 140.0)
        self.assertEqual(weights["action_post"], 64.0)

    def test_e065_uses_positive_offset_eev_energy(self):
        config_path = Path("control/HanWAM/config/hanwam_e065_positive_eev_energy_v1.yml")
        config = load_config(config_path)
        model_cfg = config["method"]["model"]
        weights = config["method"]["planner"]["first_principles"]["weights"]

        self.assertEqual(config["train"]["stage"], "stage2")
        self.assertEqual(config["train"]["history_blocks"], 4)
        self.assertEqual(config["train"]["future_blocks"], 8)
        self.assertEqual(model_cfg["temperature_mechanism"], "lagged_incremental_cooling")
        self.assertEqual(model_cfg["energy_model"], "frequency_polynomial")
        self.assertEqual(model_cfg["energy_eev_correction_mode"], "positive_offset")
        self.assertEqual(model_cfg["energy_power_intercept_w"], 279.87)
        self.assertEqual(model_cfg["energy_eev_correction_min_w"], 50.0)
        self.assertEqual(model_cfg["energy_eev_correction_max_w"], 150.0)
        self.assertEqual(model_cfg["energy_eev_anchor"], 100.0)
        self.assertEqual(model_cfg["energy_eev_range"], 170.0)
        self.assertEqual(weights["reach"], 80.0)
        self.assertEqual(weights["clamp"], 140.0)
        self.assertEqual(weights["action_post"], 64.0)

    def test_e065_lowfreq10_keeps_checkpoint_and_uses_low_load_planner_space(self):
        config_path = Path("control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml")
        config = load_config(config_path)
        bounds = action_bounds([], config=config)
        model_cfg = config["method"]["model"]
        planner = config["method"]["planner"]
        action = planner["first_principles"]["action"]
        anchors = planner["sampling"]["proposal_action_anchors"]

        self.assertEqual(bounds["freq_target"], (10.0, 80.0))
        self.assertEqual(model_cfg["freq_on_threshold_hz"], 5.0)
        self.assertNotIn("model_config_override", planner)
        self.assertEqual(planner["actuator"]["compressor_on_threshold_hz"], 10.0)
        self.assertEqual(config["simulator"]["air_conditioner"]["freq_params"]["on_threshold_hz"], 10.0)
        self.assertEqual([row[0] for row in anchors[:3]], [10.0, 15.0, 20.0])
        self.assertEqual([row[1] for row in anchors[:3]], [100.0, 100.0, 100.0])
        self.assertEqual(action["anchor_freq"], 10.0)
        self.assertEqual(action["anchor_eev"], 100.0)
        self.assertEqual(action["anchor_freq_deadband"], 5.0)
        self.assertEqual(action["anchor_eev_weight"], 0.05)


if __name__ == "__main__":
    unittest.main()
