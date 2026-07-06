"""V3 CSV loading and whole-run split utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..continuous_model import CONTROL_COLUMNS, DT_SECONDS, RAW_COLUMNS


def load_status_csv(path: str | Path, repair_coarse_timestamps: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="gbk")
    if len(frame.columns) != len(RAW_COLUMNS):
        raise ValueError(f"{path}: expected {len(RAW_COLUMNS)} columns, got {len(frame.columns)}")
    frame.columns = RAW_COLUMNS
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    for column in RAW_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ts", "T_in", *CONTROL_COLUMNS]).reset_index(drop=True)
    unique_ratio = frame["ts"].nunique() / max(1, len(frame))
    reconstructed = bool(repair_coarse_timestamps and unique_ratio < 0.5)
    if reconstructed:
        frame["ts"] = frame["ts"].iloc[0] + pd.to_timedelta(
            np.arange(len(frame), dtype=float) * DT_SECONDS, unit="s",
        )
    else:
        frame = frame.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    frame[RAW_COLUMNS[1:]] = frame[RAW_COLUMNS[1:]].interpolate(
        limit=2, limit_direction="both",
    )
    frame = frame.dropna(subset=["T_in", *CONTROL_COLUMNS]).reset_index(drop=True)
    frame.attrs["timestamp_reconstructed"] = reconstructed
    frame.attrs["original_timestamp_unique_ratio"] = float(unique_ratio)
    return frame


def discover_cooling_runs(data_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(data_dir.rglob("*.csv")):
        frame = load_status_csv(path)
        if frame.empty or int(round(float(frame["mode"].iloc[0]))) != 1:
            continue
        runs.append({
            "path": path, "name": str(path.relative_to(data_dir)),
            "group": path.parent.name, "frame": frame,
        })
    return runs


def assign_groupwise_splits(runs: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["group"], []).append(run)
    for group_runs in groups.values():
        ordered = sorted(group_runs, key=lambda run: run["frame"]["ts"].iloc[0])
        for run in ordered:
            run["split"] = "train"
        if len(ordered) >= 2:
            ordered[-1]["split"] = "test"

