from pathlib import Path

import numpy as np

from simu.room.continuous_model import (
    CONTROL_COLUMNS, HeatLoadServoConfig, StandardHeatLoadServo,
)
from simu.room.simulator import ContinuousEnthalpyRoomEnv
from simu.room.tools.data_pipeline import load_status_csv


ROOT = Path(__file__).resolve().parents[1]


def test_heat_load_servo_respects_tracking_band():
    servo = StandardHeatLoadServo(HeatLoadServoConfig(tracking_tau_seconds=120.0))
    servo.reset(room_temperature=27.0, outdoor_temperature=35.0, mode=1)
    for room_temperature in np.linspace(27.0, 20.0, 200):
        servo.update(float(room_temperature), dt=5.0)
        assert abs(servo.target - servo.actual) <= 0.5 + 1e-9


def test_v3_model_is_continuous_and_finite():
    model_path = ROOT / "simu" / "room" / "continuous_cooling_model.joblib"
    csv_path = next((ROOT / "data" / "dataset" / "X1").glob("*.csv"))
    frame = load_status_csv(csv_path)
    simulator = ContinuousEnthalpyRoomEnv(model_path)
    output = simulator.simulate(frame.iloc[0], frame[CONTROL_COLUMNS].to_numpy()[:300])
    assert len(output) == 301
    assert np.isfinite(output["T_in"]).all()
    assert output["T_in"].diff().abs().max() <= 0.5 * 5.0 / 60.0 + 1e-9
    assert output["heat_load_tracking_error"].abs().max() <= 0.5 + 1e-6
