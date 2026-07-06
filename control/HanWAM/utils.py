"""Small HanWAM utility helpers."""
from __future__ import annotations

import numpy as np
import torch


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
