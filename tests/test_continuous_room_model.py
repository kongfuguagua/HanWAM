from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from simu.room.continuous_model import (
    CONTROL_COLUMNS,
    HeatLoadServoConfig,
    STATE_COLUMNS,
    StandardHeatLoadServo,
)
from simu.room.simulator import ContinuousEnthalpyRoomEnv


class _MockCoolingModel:
    def predict(self, values):
        return np.full(len(values), -0.5, dtype=np.float32)


def _write_mock_model(path: Path) -> None:
    medians = pd.Series(0.0, index=STATE_COLUMNS, dtype=float)
    medians["mode"] = 1.0
    medians["T_set"] = 27.0
    payload = {
        "supported_mode": 1,
        "model": _MockCoolingModel(),
        "feature_variant": "thermal_inertia",
        "state_columns": list(STATE_COLUMNS),
        "state_medians": medians.to_numpy(np.float32),
        "dt_seconds": 5.0,
        "continuity": {"tau_seconds": 90.0, "max_rate_c_per_min": 0.5},
        "passive_heat_tau_seconds": 14_400.0,
        "use_physics_offcycle": True,
    }
    joblib.dump(payload, path)


def _initial_frame() -> pd.DataFrame:
    row = {name: 0.0 for name in STATE_COLUMNS}
    row.update(
        {
            "T_out": 35.0,
            "T_out_coil": 36.0,
            "T_out_discharge": 45.0,
            "T_in": 30.0,
            "T_in_coil": 24.0,
            "RH_in": 0.6,
            "T_set": 27.0,
            "mode": 1.0,
            "energy_cum": 0.0,
        }
    )
    return pd.DataFrame([row])


def test_heat_load_servo_respects_tracking_band():
    servo = StandardHeatLoadServo(HeatLoadServoConfig(tracking_tau_seconds=120.0))
    servo.reset(room_temperature=27.0, outdoor_temperature=35.0, mode=1)
    for room_temperature in np.linspace(27.0, 20.0, 200):
        servo.update(float(room_temperature), dt=5.0)
        assert abs(servo.target - servo.actual) <= 0.5 + 1e-9


def test_v3_model_is_continuous_and_finite():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "continuous_cooling_model.joblib"
        _write_mock_model(model_path)
        frame = _initial_frame()
        controls = np.tile(np.asarray([[40.0, 180.0, 750.0]], dtype=np.float32), (300, 1))
        simulator = ContinuousEnthalpyRoomEnv(model_path)
        output = simulator.simulate(frame.iloc[0], controls)

    assert len(output) == 301
    assert np.isfinite(output["T_in"]).all()
    assert output["T_in"].diff().abs().max() <= 0.5 * 5.0 / 60.0 + 1e-9
    assert output["heat_load_tracking_error"].abs().max() <= 0.5 + 1e-6
    assert list(CONTROL_COLUMNS) == ["compressor_frequency", "eev_opening", "outdoor_fan_speed"]
