"""Static HanWAM schemas and typed payloads."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Online-observable inputs confirmed for AC deployment.
WAM_OBS_COLS = [
    "freq",
    "fan_out",
    "fan_in",
    "eev",
    "T_out_coil",
    "T_in_coil",
    "T_out_discharge",
    "T_in",
    "T_out",
    "energy_cum",
    "T_set",
    "mode",
]

# Controls that the HanWAM planner is allowed to optimize.
WAM_ACTION_COLS = ["freq_target", "eev", "fan_out"]

# Physical quantities decoded from latent state for planning costs.
WAM_PHYSICAL_COLS = ["T_in_delta", "electric_kwh_delta"]

# Extra columns needed to build trajectories, simulator initial state, and targets.
WAM_REQUIRED_RAW_COLS = sorted(
    set(
        [
            "ts",
            "T_out",
            "T_out_coil",
            "T_out_discharge",
            "T_in",
            "T_in_coil",
            "freq",
            "freq_in_tgt",
            "eev",
            "fan_out",
            "I_comp",
            "fan_in",
            "RH_in",
            "T_set",
            "energy_cum",
            "mode",
        ]
    )
)


@dataclass
class HanWAMBlockSequenceArrays:
    obs_history_blocks: np.ndarray
    act_history_blocks: np.ndarray
    future_act_blocks: np.ndarray
    target_obs_blocks: np.ndarray
    physical: np.ndarray
    keys: list[str]
WAMBlockSequenceArrays = HanWAMBlockSequenceArrays
