"""Small HanWAM utility helpers."""
from __future__ import annotations

import numpy as np
import torch


def cooling_time_minutes(t_in: float, t_out: float, t_target: float) -> float:
    """Estimate cooling reach time in minutes with the piecewise linear model."""
    base = 5.0 * (float(t_in) - float(t_target))
    penalty = (10.0 / 3.0) * np.maximum(0.0, float(t_out) - float(t_in))
    return float(base + penalty + 5.0)


def cooling_ddl_seconds(t_in: float, t_out: float, t_target: float) -> float:
    """Estimate non-negative cooling deadline in seconds."""
    return max(0.0, cooling_time_minutes(t_in, t_out, t_target) * 60.0)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def maybe_limit(*arrays, limit: int, seed: int):
    if not limit or len(arrays[0]) <= limit:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arrays[0]), size=limit, replace=False)
    return tuple(arr[idx] for arr in arrays)
