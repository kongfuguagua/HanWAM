"""Static HanWAM schemas and typed payloads."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Online-observable inputs confirmed for AC deployment.
WAM_OBS_COLS = [
    "T_in",
    "T_set",
    "mode",
    "freq_target",
    "freq",
    "eev",
    "fan_out",
    "elapsed_seconds",
    "T_out",
    "energy_cum",
]

# Controls that the HanWAM planner is allowed to optimize.
WAM_ACTION_COLS = ["freq_target", "eev", "fan_out"]

# Physical quantities decoded from latent state for planning costs.
WAM_PHYSICAL_COLS = ["T_in", "freq", "T_in_delta", "electric_kwh_delta"]

# Extra columns needed to build trajectories, simulator initial state, and targets.
WAM_REQUIRED_RAW_COLS = sorted(
    set(
        [
            "ts",
            "T_out",
            "T_out_coil",
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
class HanWAMSequenceArrays:
    obs_history: np.ndarray
    actions: np.ndarray
    future_obs_history: np.ndarray
    physical: np.ndarray
    keys: list[str]


WAMSequenceArrays = HanWAMSequenceArrays
