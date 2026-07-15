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
    comfort_lower_band_c: float | None = None,
    comfort_upper_band_c: float | None = None,
    settle_hold_seconds: float = 300.0,
    ddl_seconds: float | None = None,
    require_final_in_band: bool = True,
    reach_deadline_seconds: float | None = None,
    require_post_reach_band: bool = False,
    require_post_deadline_band: bool = False,
) -> dict:
    if frame.empty:
        raise ValueError("Cannot summarize an empty trajectory")
    temp = frame["T_in"].to_numpy(dtype=float)
    elapsed = frame["elapsed_seconds"].to_numpy(dtype=float)
    error = temp - float(target)
    abs_error = np.abs(error)
    lower_band = float(comfort_band_c if comfort_lower_band_c is None else comfort_lower_band_c)
    upper_band = float(comfort_band_c if comfort_upper_band_c is None else comfort_upper_band_c)
    if lower_band < 0.0 or upper_band < 0.0:
        raise ValueError("comfort bands must be non-negative magnitudes")
    within_band = (error >= -lower_band) & (error <= upper_band)
    reached = np.flatnonzero(within_band)
    best_idx = int(np.nanargmin(abs_error))
    electric_kwh = float(frame["electric_kwh"].sum()) if "electric_kwh" in frame else 0.0
    thermal_kwh = float(frame["thermal_kwh"].sum()) if "thermal_kwh" in frame else 0.0
    reach_time = float(elapsed[reached[0]]) if len(reached) else np.nan
    sustained_reach_time = np.nan
    post_reach_violation_count = np.nan
    post_deadline_violation_count = np.nan
    post_deadline_violation_ratio = np.nan
    if len(reached):
        for idx in reached:
            if bool(within_band[idx:].all()):
                sustained_reach_time = float(elapsed[idx])
                post_reach_violation_count = 0
                break
        if not np.isfinite(sustained_reach_time):
            first = int(reached[0])
            post_reach_violation_count = int((~within_band[first:]).sum())
    final_error = float(error[-1])
    ddl = float(ddl_seconds) if ddl_seconds is not None else np.inf
    reach_deadline = float(reach_deadline_seconds) if reach_deadline_seconds is not None else ddl
    if np.isfinite(reach_deadline):
        after_deadline = elapsed >= reach_deadline
        if after_deadline.any():
            post_deadline_violation_count = int((~within_band[after_deadline]).sum())
            post_deadline_violation_ratio = float(post_deadline_violation_count / after_deadline.sum())
    if require_post_reach_band:
        success = bool(np.isfinite(sustained_reach_time)) and bool(sustained_reach_time <= reach_deadline)
    else:
        success = bool(len(reached)) and bool(reach_time <= reach_deadline)
    if require_post_deadline_band and np.isfinite(reach_deadline):
        success = success and bool(post_deadline_violation_count == 0)
    if require_final_in_band:
        success = success and bool(-lower_band <= final_error <= upper_band)
    result = {
        "reach_time_s": reach_time,
        "sustained_reach_time_s": sustained_reach_time,
        "reach_slack_s": float(reach_deadline - reach_time) if np.isfinite(reach_time) else np.nan,
        "sustained_reach_slack_s": (
            float(reach_deadline - sustained_reach_time) if np.isfinite(sustained_reach_time) else np.nan
        ),
        "settle_time_s": _settled_time(elapsed, within_band, hold_seconds=settle_hold_seconds),
        "reached": bool(len(reached)),
        "success": success,
        "ddl_seconds": float(ddl) if np.isfinite(ddl) else np.nan,
        "reach_deadline_seconds": float(reach_deadline) if np.isfinite(reach_deadline) else np.nan,
        "post_reach_band_violation_count": post_reach_violation_count,
        "post_deadline_band_violation_count": post_deadline_violation_count,
        "post_deadline_band_violation_ratio": post_deadline_violation_ratio,
        "initial_error_c": float(error[0]),
        "final_error_c": final_error,
        "min_abs_error_c": float(abs_error[best_idx]),
        "best_time_s": float(elapsed[best_idx]),
        "final_T_in_c": float(temp[-1]),
        "electric_kwh": electric_kwh,
        "thermal_kwh": thermal_kwh,
        "energy_efficiency": thermal_kwh / electric_kwh if electric_kwh > 0 else np.nan,
        "comfort_lower_band_c": lower_band,
        "comfort_upper_band_c": upper_band,
    }
    if np.isfinite(reach_deadline):
        after_deadline = elapsed >= reach_deadline
        if after_deadline.any():
            post_error = error[after_deadline]
            result.update(
                {
                    "post_deadline_temp_error_mean_c": float(np.mean(post_error)),
                    "post_deadline_temp_error_std_c": float(np.std(post_error)),
                    "post_deadline_temp_error_min_c": float(np.min(post_error)),
                    "post_deadline_temp_error_max_c": float(np.max(post_error)),
                    "post_deadline_temp_range_c": float(np.ptp(post_error)),
                }
            )
            for action_col in ("freq_target", "eev", "fan_out"):
                if action_col not in frame:
                    continue
                values = frame.loc[after_deadline, action_col].to_numpy(dtype=float)
                deltas = np.abs(np.diff(values))
                result.update(
                    {
                        f"post_deadline_{action_col}_mean": float(np.mean(values)),
                        f"post_deadline_{action_col}_std": float(np.std(values)),
                        f"post_deadline_{action_col}_range": float(np.ptp(values)),
                        f"post_deadline_{action_col}_mean_abs_step": (
                            float(np.mean(deltas)) if len(deltas) else 0.0
                        ),
                        f"post_deadline_{action_col}_max_abs_step": (
                            float(np.max(deltas)) if len(deltas) else 0.0
                        ),
                    }
                )
    return result
