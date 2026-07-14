"""Dataset utilities for V4 whole-run open-loop identification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simu.room.tools.data_pipeline import load_status_csv

from .model import CONTROL_COLUMNS, INITIAL_COLUMNS


TARGET_LIKE_COLUMNS = {
    "T_set", "RH_target", "indoor_fan_target", "pid_freq", "pid_target",
    "inference_freq", "inference_eev", "inference_fan_out", "inference_fan_in",
}
assert not TARGET_LIKE_COLUMNS.intersection(INITIAL_COLUMNS)


def discover_v4_cooling_runs(data_dir: str | Path) -> list[dict]:
    """Load runs that are predominantly cooling, including mode-4 startup rows."""
    root = Path(data_dir)
    runs = []
    for path in sorted(root.rglob("*.csv")):
        frame = load_status_csv(path)
        if frame.empty:
            continue
        cooling_fraction = float((frame["mode"].round() == 1).mean())
        if cooling_fraction < 0.90:
            continue
        runs.append({
            "path": path,
            "name": str(path.relative_to(root)),
            "group": path.parent.name,
            "frame": frame,
            "cooling_fraction": cooling_fraction,
        })
    return runs


def assign_v4_splits(runs: list[dict]) -> None:
    """Hold out the newest complete run in every repeated condition group."""
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["group"], []).append(run)
    for group_runs in groups.values():
        ordered = sorted(group_runs, key=lambda run: run["frame"]["ts"].iloc[0])
        for run in ordered:
            run["split"] = "train"
        if len(ordered) >= 2:
            ordered[-1]["split"] = "test"


def compute_normalization(runs: list[dict]) -> dict[str, np.ndarray]:
    initial = np.stack([
        run["frame"].iloc[0][INITIAL_COLUMNS].to_numpy(np.float32) for run in runs
    ])
    controls = np.concatenate([
        run["frame"][CONTROL_COLUMNS].to_numpy(np.float32) for run in runs
    ])
    initial_mean = np.nanmedian(initial, axis=0).astype(np.float32)
    initial = np.where(np.isfinite(initial), initial, initial_mean)
    initial_scale = np.nanstd(initial, axis=0).astype(np.float32)
    control_mean = np.nanmean(controls, axis=0).astype(np.float32)
    control_scale = np.nanstd(controls, axis=0).astype(np.float32)
    initial_scale = np.where(initial_scale < 1e-3, 1.0, initial_scale).astype(np.float32)
    control_scale = np.where(control_scale < 1e-3, 1.0, control_scale).astype(np.float32)
    return {
        "initial_mean": initial_mean,
        "initial_scale": initial_scale,
        "control_mean": control_mean,
        "control_scale": control_scale,
    }


def run_arrays(run: dict, initial_median: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = run["frame"]
    initial = frame.iloc[0][INITIAL_COLUMNS].to_numpy(np.float32)
    initial = np.where(np.isfinite(initial), initial, initial_median).astype(np.float32)
    # control at k advances measured temperature k -> k+1
    controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)[:-1]
    truth = frame["T_in"].to_numpy(np.float32)
    return initial, controls, truth


def describe_runs(runs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "run": run["name"], "group": run["group"], "split": run.get("split", ""),
        "samples": len(run["frame"]),
        "duration_h": (len(run["frame"]) - 1) * 5.0 / 3600.0,
        "cooling_fraction": run["cooling_fraction"],
    } for run in runs])
