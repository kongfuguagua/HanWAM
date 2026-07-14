"""Feature builder for the local residual TCN branch."""
from __future__ import annotations

import numpy as np

from .v5_model import CONTROL_COLUMNS


FEATURE_COLUMNS = [
    "control_frequency_norm",
    "control_eev_norm",
    "control_fan_out_norm",
    "delta_frequency_norm",
    "delta_eev_norm",
    "delta_fan_out_norm",
    "slow_minus_initial",
    "outdoor_minus_slow",
    "slow_rate_c_per_min",
    "elapsed_log",
    "elapsed_sin_10min",
    "elapsed_cos_10min",
    "compressor_on",
    "load_coefficient",
    "cooling_capacity",
    "sensible_load",
]


def _series_or_zeros(frame, name: str, length: int) -> np.ndarray:
    if name not in frame:
        return np.zeros(length, dtype=np.float32)
    return np.nan_to_num(
        frame[name].to_numpy(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_features(frame, slow_frame, control_mean, control_scale) -> np.ndarray:
    n = len(slow_frame)
    controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)[:n].copy()
    control_scale = np.where(control_scale < 1e-4, 1.0, control_scale).astype(np.float32)
    control_norm = (controls - control_mean) / control_scale
    delta = np.diff(controls, axis=0, prepend=controls[:1]) / control_scale
    slow = slow_frame["T_in"].to_numpy(np.float32)
    t_out = slow_frame["T_out"].to_numpy(np.float32)
    elapsed = slow_frame["elapsed_seconds"].to_numpy(np.float32)
    slow_rate = np.diff(slow, prepend=slow[:1]) * 60.0 / 5.0
    load_coefficient = _series_or_zeros(slow_frame, "load_coefficient", n)
    cooling = _series_or_zeros(slow_frame, "normalized_cooling_capacity", n)
    sensible_load = _series_or_zeros(slow_frame, "normalized_sensible_load", n)
    features = np.column_stack([
        control_norm,
        delta,
        (slow - slow[0]) / 4.0,
        (t_out - slow) / 10.0,
        slow_rate / 0.2,
        np.log1p(elapsed / 60.0),
        np.sin(2.0 * np.pi * elapsed / 600.0),
        np.cos(2.0 * np.pi * elapsed / 600.0),
        (controls[:, 0] > 1.0).astype(np.float32),
        load_coefficient,
        cooling,
        sensible_load,
    ]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
