"""Shared schemas for controller and simulator interaction."""
from __future__ import annotations

TEMPERATURE_OBS_COLS = ["T_out", "T_out_coil", "T_out_discharge", "T_in", "T_in_coil"]
ACTUATOR_OBS_COLS = ["freq_target", "freq", "eev", "fan_out"]
INDOOR_ENV_OBS_COLS = ["fan_in", "RH_in"]
SETPOINT_OBS_COLS = ["T_set", "mode"]
ENERGY_OBS_COLS = ["energy_cum"]
TIME_OBS_COLS = ["elapsed_seconds"]

# Features observable by the controller/AC at each closed-loop decision point.
OBS_COLS = (
    TEMPERATURE_OBS_COLS
    + ACTUATOR_OBS_COLS
    + INDOOR_ENV_OBS_COLS
    + SETPOINT_OBS_COLS
    + ENERGY_OBS_COLS
    + TIME_OBS_COLS
)
STATE_COLS = TEMPERATURE_OBS_COLS
ACTUAL_ACTION_COLS = ["freq", "eev", "fan_out"]
TARGET_ACTION_COLS = ["freq_target", "eev", "fan_out"]
CONTROL_COLS = ACTUAL_ACTION_COLS
ENV_COLS = ["fan_in", "RH_in", "T_set", "energy_cum", "mode"]
MODE_TO_NAME = {1: "制冷", 3: "制热"}
