from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from control.MiniController.config_schema import load_config
from control.HanWAM.dataloader import Normalizer, block_sequence_arrays, load_all_runs
from control.HanWAM.type import WAM_ACTION_COLS, WAM_OBS_COLS, WAM_PHYSICAL_COLS
from control.MiniController.experiment import (
    relative_humidity_from_wet_bulb,
    run_closed_loop,
    standard_condition_frame,
)
from control.MiniController.metrics import summarize_closed_loop
from data.io import discover_csvs, load_all_raw
from simu.air_conditioner import AirConditionerSimulator
from simu.environment import EnvironmentSimulator


def _write_status_csv(path: Path, rows: int = 170, mode: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(np.zeros((rows, 24), dtype=object), columns=[f"c{i}" for i in range(24)])
    raw.iloc[:, 0] = list(pd.date_range("2026-01-01", periods=rows, freq="5s").astype(str))
    raw.iloc[:, 1] = 35.0
    raw.iloc[:, 2] = 36.0
    raw.iloc[:, 3] = 45.0
    raw.iloc[:, 4] = 30.0
    raw.iloc[:, 5] = 180.0
    raw.iloc[:, 6] = 750.0
    raw.iloc[:, 7] = 4.0
    raw.iloc[:, 8] = np.linspace(30.0, 27.0, rows)
    raw.iloc[:, 9] = 24.0
    raw.iloc[:, 10] = 900.0
    raw.iloc[:, 12] = 0.6
    raw.iloc[:, 13] = 27.0
    raw.iloc[:, 14] = float(mode)
    raw.iloc[:, 15] = np.arange(rows, dtype=float) * 0.001
    raw.iloc[:, 23] = 30.0
    raw.to_csv(path, index=False, encoding="utf-8")


def _mock_data_config(root: Path) -> dict:
    config = load_config("control/HanWAM/config/hanwam.yml")
    config["data"] = {
        "root": str(root),
        "manifest": None,
        "group_dirs": {"include": [], "exclude": []},
        "split": {
            "train": {"source_contains": ["train_group"]},
            "val": {"source_contains": ["val_group"]},
            "test": {"source_contains": ["test_group"]},
        },
        "sampling": {"step_seconds": 5, "interpolation_limit": 2, "min_run_steps": 3},
    }
    return config


def _write_mock_dataset(root: Path) -> dict:
    for group in ("train_group", "val_group", "test_group"):
        _write_status_csv(root / group / f"status_data_{group}.csv")
    return _mock_data_config(root)


class _StepTemperatureEnv:
    def reset(self, T_out: float, T_in: float, T_out_coil: float, T_in_coil: float) -> dict:
        self.elapsed_seconds = 0.0
        self.temperature = float(T_in)
        return {"elapsed_seconds": 0.0, "T_in": self.temperature, "model_spread": 0.0}

    def step(self, *, freq: float, eev: float, fan_out: float) -> dict:
        self.elapsed_seconds += 5.0
        self.temperature -= 0.02 if freq > 1.0 else -0.01
        return {"elapsed_seconds": self.elapsed_seconds, "T_in": self.temperature, "model_spread": 0.0}


class _ZeroEnergyModel:
    def reset(self) -> None:
        pass

    def predict_power(self, controls):
        return [0.0]


class ClosedLoopInterfaceTest(unittest.TestCase):
    def test_raw_data_schema_has_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "raw"
            _write_status_csv(root / "cooling" / "status_data_cooling.csv", mode=1)
            _write_status_csv(root / "heating" / "status_data_heating.csv", mode=3)
            self.assertEqual(len(discover_csvs(root)), 2)
            frame = load_all_raw(root, verbose=False)

        for col in ["fan_in", "RH_in", "T_set", "energy_cum", "mode", "freq_in_tgt", "T_out_discharge"]:
            self.assertIn(col, frame.columns)
            self.assertGreater(frame[col].notna().sum(), 0)
        self.assertTrue({1, 3}.issubset(set(frame["mode"].dropna().astype(int).unique())))

    def test_air_conditioner_frequency_lag_and_power(self):
        ac = AirConditionerSimulator(mode=1, energy_model=_ZeroEnergyModel())
        ac.reset(freq0=0.0, target0=0.0)
        first = ac.step([45.0, 180.0, 750.0])
        second = ac.step([45.0, 180.0, 750.0])
        self.assertLess(first["freq"], 45.0)
        self.assertGreater(second["freq"], first["freq"])
        self.assertGreaterEqual(second["power_w"], 0.0)

    def test_environment_thermal_proxy_sign(self):
        cooling = EnvironmentSimulator(mode=1, temperature_env=_StepTemperatureEnv())
        cooling.reset({"T_out": 35, "T_in": 30, "T_out_coil": 37, "T_in_coil": 24, "RH_in": 0.6, "fan_in": 900})
        cool_row = cooling.step({"freq": 35, "eev": 180, "fan_out": 750})
        self.assertGreater(cool_row["thermal_power_w"], 0.0)

        heating = EnvironmentSimulator(mode=3, temperature_env=_StepTemperatureEnv())
        heating.reset({"T_out": 5, "T_in": 15, "T_out_coil": 0, "T_in_coil": 35, "RH_in": 0.5, "fan_in": 900})
        heat_row = heating.step({"freq": 35, "eev": 180, "fan_out": 750})
        self.assertGreater(heat_row["thermal_power_w"], 0.0)

    def test_metrics(self):
        frame = pd.DataFrame(
            {
                "elapsed_seconds": [0.0, 5.0, 10.0],
                "T_in": [30.0, 27.6, 27.4],
                "electric_kwh": [0.0, 0.1, 0.1],
                "thermal_kwh": [0.0, 0.2, 0.2],
            }
        )
        summary = summarize_closed_loop(frame, target=27.0, comfort_band_c=0.5)
        self.assertEqual(summary["reach_time_s"], 10.0)
        self.assertEqual(summary["final_T_in_c"], 27.4)
        self.assertAlmostEqual(summary["energy_efficiency"], 2.0)
        deadline_summary = summarize_closed_loop(frame, target=27.0, comfort_band_c=0.5, ddl_seconds=10.0)
        self.assertTrue(deadline_summary["success"])
        held = summarize_closed_loop(
            frame,
            target=27.0,
            comfort_band_c=0.5,
            ddl_seconds=10.0,
            reach_deadline_seconds=10.0,
            require_post_reach_band=True,
        )
        self.assertTrue(held["success"])
        drift = pd.DataFrame(
            {
                "elapsed_seconds": [0.0, 5.0, 10.0, 15.0],
                "T_in": [30.0, 27.4, 27.6, 27.7],
                "electric_kwh": [0.0, 0.1, 0.1, 0.1],
                "thermal_kwh": [0.0, 0.2, 0.2, 0.2],
            }
        )
        drift_summary = summarize_closed_loop(
            drift,
            target=27.0,
            comfort_band_c=0.5,
            ddl_seconds=15.0,
            reach_deadline_seconds=5.0,
            require_post_reach_band=True,
        )
        self.assertFalse(drift_summary["success"])
        self.assertGreater(drift_summary["post_reach_band_violation_count"], 0)
        self.assertGreater(drift_summary["post_deadline_band_violation_count"], 0)
        self.assertGreater(drift_summary["post_deadline_band_violation_ratio"], 0.0)

    def test_metrics_support_asymmetric_post_deadline_band(self):
        frame = pd.DataFrame(
            {
                "elapsed_seconds": [0.0, 5.0, 10.0, 15.0],
                "T_in": [28.0, 26.2, 26.0, 27.5],
                "freq_target": [40.0, 30.0, 20.0, 21.0],
                "eev": [180.0, 160.0, 140.0, 141.0],
                "fan_out": [750.0] * 4,
                "electric_kwh": [0.0, 0.1, 0.1, 0.1],
                "thermal_kwh": [0.0, 0.2, 0.2, 0.2],
            }
        )
        accepted = summarize_closed_loop(
            frame,
            target=27.0,
            comfort_lower_band_c=1.0,
            comfort_upper_band_c=0.5,
            reach_deadline_seconds=10.0,
            require_post_deadline_band=True,
        )
        self.assertTrue(accepted["success"])
        self.assertEqual(accepted["post_deadline_band_violation_count"], 0)
        self.assertAlmostEqual(accepted["post_deadline_temp_error_min_c"], -1.0)
        self.assertIn("post_deadline_freq_target_mean_abs_step", accepted)

        rejected = frame.copy()
        rejected.loc[3, "T_in"] = 27.51
        summary = summarize_closed_loop(
            rejected,
            target=27.0,
            comfort_lower_band_c=1.0,
            comfort_upper_band_c=0.5,
            reach_deadline_seconds=10.0,
            require_post_deadline_band=True,
        )
        self.assertFalse(summary["success"])
        self.assertEqual(summary["post_deadline_band_violation_count"], 1)

    def test_standard_condition_frame_from_dry_wet_bulb(self):
        rh = relative_humidity_from_wet_bulb(32.0, 28.0)
        self.assertGreater(rh, 0.0)
        self.assertLessEqual(rh, 1.0)
        frame = standard_condition_frame(
            {
                "name": "A",
                "indoor_initial_dry_bulb_c": 32.0,
                "indoor_initial_wet_bulb_c": 28.0,
                "indoor_target_dry_bulb_c": 27.0,
                "outdoor_dry_bulb_c": 35.0,
                "outdoor_wet_bulb_c": 24.0,
            },
            mode=1,
            horizon_seconds=600,
            step_seconds=5,
            name="condition_A",
        )
        self.assertEqual(len(frame), 121)
        self.assertEqual(float(frame["T_in"].iloc[0]), 32.0)
        self.assertEqual(float(frame["T_set"].iloc[0]), 27.0)
        self.assertEqual(float(frame["T_out"].iloc[0]), 35.0)
        self.assertEqual(float(frame["T_out_discharge"].iloc[0]), 35.0)
        self.assertEqual(str(frame["condition"].iloc[0]), "A")

    def test_hanwam_block_sequence_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _write_mock_dataset(Path(tmpdir) / "dataset")
            runs = load_all_runs(mode=1, config=config)
            obs_history, act_history, future_act, target_obs, physical, keys = block_sequence_arrays(
                runs,
                "train",
                config=config,
                limit=2,
                seed=7,
            )
        self.assertEqual(tuple(obs_history.shape[1:]), (4, 12, len(WAM_OBS_COLS)))
        self.assertEqual(tuple(act_history.shape[1:]), (4, 12, len(WAM_ACTION_COLS)))
        self.assertEqual(tuple(future_act.shape[1:]), (8, 12, len(WAM_ACTION_COLS)))
        self.assertEqual(tuple(target_obs.shape[1:]), (8, 12, len(WAM_OBS_COLS)))
        self.assertEqual(tuple(physical.shape[1:]), (96, len(WAM_PHYSICAL_COLS)))
        self.assertEqual(len(keys), obs_history.shape[0])

    def test_normalizer_round_trip(self):
        norm = Normalizer(mean=pd.Series([1.0, 2.0]).to_numpy(), std=pd.Series([2.0, 4.0]).to_numpy())
        values = pd.DataFrame([[3.0, 10.0]]).to_numpy()
        decoded = norm.decode(norm.encode(values))
        self.assertTrue((decoded == values).all())

    def test_historical_closed_loop_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _write_mock_dataset(Path(tmpdir) / "dataset")
            runs = load_all_runs(mode=1, config=config)
            run = next(r for r in runs if r.split == "test" and len(r.frame) >= 7)
            trajectory, controller_log, history_log = run_closed_loop(
                frame=run.frame.iloc[:7].reset_index(drop=True),
                policy="historical",
                mode=1,
                target=float(run.frame["T_set"].iloc[0]),
                controller=None,
                horizon_seconds=30,
                simulator_config={
                    "air_conditioner": {"energy_model": _ZeroEnergyModel()},
                    "environment": {"temperature_env": _StepTemperatureEnv()},
                },
            )
        self.assertEqual(len(trajectory), 7)
        for col in ["freq_target", "freq", "power_w", "thermal_power_w", "electric_kwh", "thermal_kwh"]:
            self.assertIn(col, trajectory.columns)
        self.assertEqual(len(controller_log), 6)
        self.assertTrue(history_log.empty)

    def test_grouped_dataset_manifest_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _write_mock_dataset(Path(tmpdir) / "dataset")
            runs = load_all_runs(mode=1, config=config)
        splits = {run.split for run in runs}
        self.assertTrue({"train", "val", "test"}.issubset(splits))
        self.assertTrue(all("/dataset/" in str(run.path) for run in runs))


if __name__ == "__main__":
    unittest.main()
