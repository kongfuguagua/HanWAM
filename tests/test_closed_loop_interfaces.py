from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from control.MiniController.config import load_controller_config
from control.MiniController.config_schema import load_config
from control.HanWAM.dataloader import Normalizer, Run, block_sequence_arrays, load_all_runs
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


class _FlatEnergyModel:
    def reset(self) -> None:
        pass

    def predict_power(self, controls):
        freq = float(np.asarray(controls, dtype=np.float32)[0])
        return [max(freq, 0.0) * 10.0]


class _StepTemperatureEnv:
    def __init__(self, delta_c: float):
        self.delta_c = float(delta_c)

    def reset(self, T_out: float, T_in: float, T_out_coil: float, T_in_coil: float, **kwargs):
        self.elapsed_seconds = 0.0
        self.temperature = float(T_in)
        return {"elapsed_seconds": 0.0, "T_in": self.temperature, "model_spread": 0.0}

    def step(self, freq: float, eev: float, fan_out: float):
        self.elapsed_seconds += 5.0
        self.temperature += self.delta_c
        return {"elapsed_seconds": self.elapsed_seconds, "T_in": self.temperature, "model_spread": 0.0}


def _raw_status_frame(rows: int = 80, mode: int = 1) -> pd.DataFrame:
    values: dict[int, object] = {
        0: pd.date_range("2026-01-01", periods=rows, freq="5s").astype(str),
        1: np.full(rows, 35.0),
        2: np.full(rows, 34.0),
        3: np.full(rows, 50.0),
        4: np.full(rows, 40.0),
        5: np.full(rows, 165.0),
        6: np.full(rows, 750.0),
        7: np.full(rows, 2.0),
        8: np.linspace(30.0, 27.0, rows),
        9: np.full(rows, 25.0),
        10: np.full(rows, 900.0),
        11: np.zeros(rows),
        12: np.full(rows, 55.0),
        13: np.full(rows, 27.0),
        14: np.full(rows, float(mode)),
        15: np.linspace(0.0, 0.2, rows),
        16: np.zeros(rows),
        17: np.zeros(rows),
        18: np.zeros(rows),
        19: np.zeros(rows),
        20: np.zeros(rows),
        21: np.zeros(rows),
        22: np.zeros(rows),
        23: np.full(rows, 40.0),
    }
    return pd.DataFrame({f"c{i}": values.get(i, np.zeros(rows)) for i in range(24)})


def _write_raw_csv(path: Path, rows: int = 80, mode: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _raw_status_frame(rows=rows, mode=mode).to_csv(path, index=False, encoding="utf-8")


def _mock_wam_frame(rows: int = 120, mode: int = 1) -> pd.DataFrame:
    elapsed = np.arange(rows, dtype=np.float32) * 5.0
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=rows, freq="5s"),
            "elapsed_seconds": elapsed,
            "freq": np.full(rows, 40.0),
            "freq_target": np.full(rows, 40.0),
            "fan_out": np.full(rows, 750.0),
            "fan_in": np.full(rows, 900.0),
            "eev": np.full(rows, 165.0),
            "T_out_coil": np.full(rows, 34.0),
            "T_in_coil": np.full(rows, 25.0),
            "T_out_discharge": np.full(rows, 50.0),
            "T_in": np.linspace(30.0, 27.0, rows),
            "T_out": np.full(rows, 35.0),
            "energy_cum": np.linspace(0.0, 0.3, rows),
            "T_set": np.full(rows, 27.0),
            "mode": np.full(rows, float(mode)),
        }
    )


class ClosedLoopInterfaceTest(unittest.TestCase):
    def test_raw_data_schema_has_evaluation_fields(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_raw_csv(root / "1轮" / "X1" / "status_data_train.csv", mode=1)
            _write_raw_csv(root / "2轮" / "X1" / "status_data_val.csv", mode=3)
            self.assertEqual(len(discover_csvs(root)), 2)
            frame = load_all_raw(root, verbose=False)
        for col in ["fan_in", "RH_in", "T_set", "energy_cum", "mode", "freq_in_tgt", "T_out_discharge"]:
            self.assertIn(col, frame.columns)
            self.assertGreater(frame[col].notna().sum(), 0)
        self.assertTrue({1, 3}.issubset(set(frame["mode"].dropna().astype(int).unique())))

    def test_air_conditioner_frequency_lag_and_power(self):
        ac = AirConditionerSimulator(mode=1, energy_model=_FlatEnergyModel())
        ac.reset(freq0=0.0, target0=0.0)
        first = ac.step([45.0, 180.0, 750.0])
        second = ac.step([45.0, 180.0, 750.0])
        self.assertLess(first["freq"], 45.0)
        self.assertGreater(second["freq"], first["freq"])
        self.assertGreaterEqual(second["power_w"], 0.0)

    def test_environment_thermal_proxy_sign(self):
        cooling = EnvironmentSimulator(mode=1, temperature_env=_StepTemperatureEnv(delta_c=-0.1))
        cooling.reset({"T_out": 35, "T_in": 30, "T_out_coil": 37, "T_in_coil": 24, "RH_in": 0.6, "fan_in": 900})
        cool_row = cooling.step({"freq": 35, "eev": 180, "fan_out": 750})
        self.assertGreater(cool_row["thermal_power_w"], 0.0)

        heating = EnvironmentSimulator(mode=3, temperature_env=_StepTemperatureEnv(delta_c=0.1))
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
        config = load_config("control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml")
        runs = [Run("mock_train", Path("mock_train.csv"), "train", _mock_wam_frame(rows=180))]
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
        run = Run("mock_test", Path("mock_test.csv"), "test", _mock_wam_frame(rows=7))
        with (
            patch(
                "control.MiniController.experiment.AirConditionerSimulator",
                lambda **kwargs: AirConditionerSimulator(**kwargs, energy_model=_FlatEnergyModel()),
            ),
            patch(
                "control.MiniController.experiment.EnvironmentSimulator",
                lambda **kwargs: EnvironmentSimulator(
                    **kwargs,
                    temperature_env=_StepTemperatureEnv(delta_c=-0.1),
                ),
            ),
        ):
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
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = tmp / "dataset"
            manifest = tmp / "split_manifest.csv"
            rows = []
            for group, source_marker, filename in (
                ("X1", "1轮", "status_data_train.csv"),
                ("X2", "2轮", "status_data_val.csv"),
                ("X3", "3轮", "status_data_test.csv"),
            ):
                path = root / group / filename
                _write_raw_csv(path, rows=80, mode=1)
                rows.append({"destination": str(path.relative_to(root)), "source": f"{source_marker}/{filename}"})
            pd.DataFrame(rows).to_csv(manifest, index=False)
            config = load_controller_config()
            config["data"]["root"] = str(root)
            config["data"]["manifest"] = str(manifest)
            config["data"]["group_dirs"] = {"include": ["X1", "X2", "X3"], "exclude": []}
            runs = load_all_runs(mode=1, config=config)
        splits = {run.split for run in runs}
        self.assertTrue({"train", "val", "test"}.issubset(splits))
        self.assertTrue(all(root in run.path.parents for run in runs))


if __name__ == "__main__":
    unittest.main()
