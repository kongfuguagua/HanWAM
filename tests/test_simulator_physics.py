from __future__ import annotations

import unittest

from simu.environment import EnvironmentSimulator
from simu.air_conditioner import AirConditionerSimulator
from simu.frequency.freq_response_model import simulate_freq
from simu.room.simulator import ContinuousEnthalpyRoomEnv


class _ZeroEnergyModel:
    def predict_power(self, controls):
        return [0.0]


class _CoolingFakeEnv:
    def reset(self, *args, **kwargs):
        self.elapsed_seconds = 0.0
        self.temperature = 30.0
        return {"elapsed_seconds": 0.0, "T_in": self.temperature, "model_spread": 0.0}

    def step(self, freq: float, eev: float, fan_out: float):
        self.elapsed_seconds += 5.0
        self.temperature -= 0.1
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "T_in": self.temperature,
            "model_spread": 0.0,
        }


class SimulatorPhysicsTest(unittest.TestCase):
    def test_v3_fan_only_warms_with_outdoor_heat_load(self):
        env = ContinuousEnthalpyRoomEnv()
        obs = {
            "T_out": 35.0,
            "T_out_coil": 35.0,
            "T_in": 30.0,
            "T_in_coil": 30.0,
            "RH_in": 0.6,
            "fan_in": 900.0,
            "T_set": 26.0,
            "mode": 1,
            "energy_cum": 0.0,
        }
        start = env.reset(initial_observation=obs)["T_in"]
        for _ in range(120):
            current = env.step(freq=0.0, eev=165.0, fan_out=850.0)["T_in"]
        self.assertGreater(current, start)

    def test_compressor_off_has_zero_hvac_thermal_output(self):
        env = EnvironmentSimulator(mode=1, temperature_env=_CoolingFakeEnv())
        env.reset(
            {
                "T_out": 35.0,
                "T_in": 30.0,
                "T_out_coil": 35.0,
                "T_in_coil": 20.0,
                "RH_in": 0.6,
                "fan_in": 900.0,
            }
        )
        state = env.step({"freq": 0.0, "eev": 165.0, "fan_out": 850.0, "power_w": 0.0})
        self.assertEqual(state["thermal_power_w"], 0.0)
        self.assertEqual(state["thermal_kwh"], 0.0)

    def test_incremental_frequency_matches_batch_simulator(self):
        target0 = 0.0
        targets = [0.0, 40.0, 40.0, 70.0, 70.0, 30.0, 30.0, 0.0, 0.0, 50.0]
        expected = simulate_freq([target0, *targets], freq0=0.0, freq_cap=90.0)[1:]
        ac = AirConditionerSimulator(mode=1, energy_model=_ZeroEnergyModel(), freq_cap=90.0)
        ac.reset(freq0=0.0, target0=target0)
        actual = [ac.step([target, 165.0, 750.0])["freq"] for target in targets]
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=6)


if __name__ == "__main__":
    unittest.main()
