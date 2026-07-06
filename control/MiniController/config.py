"""Controller configuration helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config_schema import CONFIG_PATH, action_space_for_method, load_config, mode_key
from .schemas import TARGET_ACTION_COLS


def load_controller_config(path: str | Path = CONFIG_PATH) -> dict:
    return load_config(path)


def action_bounds_for_mode(
    config: dict,
    mode: int | str,
    fallback: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    try:
        return action_space_for_method(config, mode)
    except KeyError:
        if fallback is None:
            raise
        return {col: tuple(map(float, fallback[col])) for col in TARGET_ACTION_COLS}


def fixed_action_for_mode(config: dict, mode: int | str) -> np.ndarray:
    method = config["method"]
    payload = (method.get("actions_by_mode") or {}).get(mode_key(mode))
    if payload is None:
        payload = (method.get("base_action_by_mode") or {}).get(mode_key(mode))
    if payload is None:
        raise KeyError(f"Missing method actions/base_action for mode={mode}")
    return np.asarray([payload[col] for col in TARGET_ACTION_COLS], dtype=np.float32)


def pid_config_for_mode(config: dict, mode: int | str) -> dict:
    payload = (config["method"].get("gains_by_mode") or {}).get(mode_key(mode))
    if payload is None:
        raise KeyError(f"Missing method.gains_by_mode for mode={mode}")
    return payload


def signed_temperature_error(mode: int | str, T_in: float, target: float) -> float:
    key = mode_key(mode)
    if key == "3":
        return float(target - T_in)
    return float(T_in - target)
