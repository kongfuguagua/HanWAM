"""Closed-loop control metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _settled_time(elapsed: np.ndarray, within_band: np.ndarray, hold_seconds: float = 300.0) -> float:
    if len(elapsed) < 2 or not within_band.any():
        return np.nan
    step_seconds = float(np.median(np.diff(elapsed)))
    hold_steps = max(1, int(np.ceil(float(hold_seconds) / step_seconds)))
    for idx in np.flatnonzero(within_band):
        end = idx + hold_steps
        if end <= len(within_band) and bool(within_band[idx:end].all()):
            return float(elapsed[idx])
    return np.nan


def summarize_closed_loop(
    frame: pd.DataFrame,
    target: float,
    comfort_band_c: float = 0.5,
    settle_hold_seconds: float = 300.0,
    ddl_seconds: float | None = None,
) -> dict:
    if frame.empty:
        raise ValueError("Cannot summarize an empty trajectory")
    temp = frame["T_in"].to_numpy(dtype=float)
    elapsed = frame["elapsed_seconds"].to_numpy(dtype=float)
    error = temp - float(target)
    abs_error = np.abs(error)
    within_band = abs_error <= comfort_band_c
    reached = np.flatnonzero(within_band)
    best_idx = int(np.nanargmin(abs_error))
    electric_kwh = float(frame["electric_kwh"].sum()) if "electric_kwh" in frame else 0.0
    thermal_kwh = float(frame["thermal_kwh"].sum()) if "thermal_kwh" in frame else 0.0
    reach_time = float(elapsed[reached[0]]) if len(reached) else np.nan
    final_error = float(error[-1])
    ddl = float(ddl_seconds) if ddl_seconds is not None else np.inf
    success = bool(len(reached)) and bool(reach_time < ddl) and bool(abs(final_error) <= comfort_band_c)
    return {
        "reach_time_s": reach_time,
        "settle_time_s": _settled_time(elapsed, within_band, hold_seconds=settle_hold_seconds),
        "reached": bool(len(reached)),
        "success": success,
        "ddl_seconds": float(ddl) if np.isfinite(ddl) else np.nan,
        "initial_error_c": float(error[0]),
        "final_error_c": final_error,
        "min_abs_error_c": float(abs_error[best_idx]),
        "best_time_s": float(elapsed[best_idx]),
        "final_T_in_c": float(temp[-1]),
        "electric_kwh": electric_kwh,
        "thermal_kwh": thermal_kwh,
        "energy_efficiency": thermal_kwh / electric_kwh if electric_kwh > 0 else np.nan,
    }
