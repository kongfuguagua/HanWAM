from __future__ import annotations

import unittest

import pandas as pd

from control.MiniController.config import load_controller_config
from control.HanWAM.dataloader import Normalizer, load_all_runs
from control.MiniController.experiment import run_closed_loop
from control.MiniController.metrics import summarize_closed_loop
from data.io import discover_csvs, load_all_raw
from simu.air_conditioner import AirConditionerSimulator
from simu.environment import EnvironmentSimulator


class ClosedLoopInterfaceTest(unittest.TestCase):
    def test_raw_data_schema_has_evaluation_fields(self):
        self.assertGreaterEqual(len(discover_csvs()), 53)
        frame = load_all_raw(verbose=False)
        for col in ["fan_in", "RH_in", "T_set", "energy_cum", "mode", "freq_in_tgt"]:
            self.assertIn(col, frame.columns)
            self.assertGreater(frame[col].notna().sum(), 0)
        self.assertTrue({1, 3}.issubset(set(frame["mode"].dropna().astype(int).unique())))

    def test_air_conditioner_frequency_lag_and_power(self):
        ac = AirConditionerSimulator(mode=1)
        ac.reset(freq0=0.0, target0=0.0)
        first = ac.step([45.0, 180.0, 750.0])
        second = ac.step([45.0, 180.0, 750.0])
        self.assertLess(first["freq"], 45.0)
        self.assertGreater(second["freq"], first["freq"])
        self.assertGreaterEqual(second["power_w"], 0.0)

    def test_environment_thermal_proxy_sign(self):
        cooling = EnvironmentSimulator(mode=1)
        cooling.reset({"T_out": 35, "T_in": 30, "T_out_coil": 37, "T_in_coil": 24, "RH_in": 0.6, "fan_in": 900})
        cool_row = cooling.step({"freq": 35, "eev": 180, "fan_out": 750})
        self.assertGreater(cool_row["thermal_power_w"], 0.0)

        heating = EnvironmentSimulator(mode=3)
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

    def test_normalizer_round_trip(self):
        norm = Normalizer(mean=pd.Series([1.0, 2.0]).to_numpy(), std=pd.Series([2.0, 4.0]).to_numpy())
        values = pd.DataFrame([[3.0, 10.0]]).to_numpy()
        decoded = norm.decode(norm.encode(values))
        self.assertTrue((decoded == values).all())

    def test_historical_closed_loop_smoke(self):
        runs = load_all_runs(mode=1)
        run = next(r for r in runs if r.split == "test" and len(r.frame) >= 7)
        trajectory, controller_log, history_log = run_closed_loop(
            frame=run.frame.iloc[:7].reset_index(drop=True),
            policy="historical",
            mode=1,
            target=float(run.frame["T_set"].iloc[0]),
            controller=None,
            horizon_seconds=30,
        )
        self.assertEqual(len(trajectory), 7)
        for col in ["freq_target", "freq", "power_w", "thermal_power_w", "electric_kwh", "thermal_kwh"]:
            self.assertIn(col, trajectory.columns)
        self.assertEqual(len(controller_log), 6)
        self.assertTrue(history_log.empty)

    def test_grouped_dataset_manifest_splits(self):
        config = load_controller_config()
        runs = load_all_runs(mode=1, config=config)
        splits = {run.split for run in runs}
        self.assertTrue({"train", "val", "test"}.issubset(splits))
        self.assertTrue(all("/dataset/" in str(run.path) for run in runs))


if __name__ == "__main__":
    unittest.main()
