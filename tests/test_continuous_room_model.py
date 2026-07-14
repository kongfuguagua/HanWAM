from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from simu.room.continuous_model import (
    CONTROL_COLUMNS, HeatLoadServoConfig, PLANT_STATE_COLUMNS, StandardHeatLoadServo,
)
from simu.room.simulator import ContinuousEnthalpyRoomEnv

class _ConstantRoomModel:
    def predict(self, features):
        return np.full(len(features), 26.5, dtype=np.float32)


def _mock_room_payload() -> dict:
    medians = {
        "T_out": 35.0,
        "T_out_coil": 34.0,
        "T_out_discharge": 50.0,
        "T_in": 27.0,
        "T_in_coil": 25.0,
        "RH_in": 55.0,
        "mode": 1.0,
        "energy_cum": 0.0,
        "fault": 0.0,
        "swing": 0.0,
        "inference_freq": 40.0,
        "inference_eev": 165.0,
        "inference_fan_out": 750.0,
        "inference_fan_in": 900.0,
    }
    return {
        "supported_mode": 1,
        "model": _ConstantRoomModel(),
        "feature_variant": "autoregressive_mock",
        "state_columns": list(PLANT_STATE_COLUMNS),
        "state_medians": np.asarray([medians[col] for col in PLANT_STATE_COLUMNS], dtype=np.float32),
        "dt_seconds": 5.0,
        "continuity": {"tau_seconds": 90.0, "max_rate_c_per_min": 0.5},
        "feature_config": {
            "batch1_control_memory": True,
            "batch2_approach_temps": True,
            "batch3_interactions": True,
        },
        "heat_load_config": {},
        "autoregressive_horizon": 1,
    }


def _mock_room_frame(rows: int = 301) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "T_out": np.full(rows, 35.0),
            "T_out_coil": np.full(rows, 34.0),
            "T_out_discharge": np.full(rows, 50.0),
            "T_in": np.linspace(27.0, 26.8, rows),
            "T_in_coil": np.full(rows, 25.0),
            "RH_in": np.full(rows, 55.0),
            "mode": np.full(rows, 1.0),
            "energy_cum": np.linspace(0.0, 0.3, rows),
            "fault": np.zeros(rows),
            "swing": np.zeros(rows),
            "inference_freq": np.full(rows, 40.0),
            "inference_eev": np.full(rows, 165.0),
            "inference_fan_out": np.full(rows, 750.0),
            "inference_fan_in": np.full(rows, 900.0),
            "compressor_frequency": np.full(rows, 40.0),
            "eev_opening": np.full(rows, 165.0),
            "outdoor_fan_speed": np.full(rows, 750.0),
        }
    )


def test_heat_load_servo_respects_tracking_band():
    servo = StandardHeatLoadServo(HeatLoadServoConfig(tracking_tau_seconds=120.0))
    servo.reset(room_temperature=27.0, outdoor_temperature=35.0, mode=1)
    for room_temperature in np.linspace(27.0, 20.0, 200):
        servo.update(float(room_temperature), dt=5.0)
        assert abs(servo.target - servo.actual) <= 0.5 + 1e-9


def test_v3_model_is_continuous_and_finite():
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "continuous_cooling_model.joblib"
        joblib.dump(_mock_room_payload(), model_path)
        frame = _mock_room_frame()
        simulator = ContinuousEnthalpyRoomEnv(model_path)
        output = simulator.simulate(frame.iloc[0], frame[CONTROL_COLUMNS].to_numpy()[:300])
    assert len(output) == 301
    assert np.isfinite(output["T_in"]).all()
    assert output["T_in"].diff().abs().max() <= 0.5 * 5.0 / 60.0 + 1e-9
    assert output["heat_load_tracking_error"].abs().max() <= 0.5 + 1e-6
