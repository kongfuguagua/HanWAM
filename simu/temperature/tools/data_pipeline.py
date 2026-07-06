"""焓差室 CSV 数据读取、清洗、重采样与严格的按实验轮次切分。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.io import DATA_DIR, discover_csvs, infer_split, read_raw_csv, safe_source_name


PROJECT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_DIR.parents[1]

STATE_COLS = ["T_out", "T_out_coil", "T_in", "T_in_coil"]
CONTROL_COLS = ["freq", "eev", "fan_out"]

@dataclass
class Run:
    name: str
    path: Path
    split: str
    frame: pd.DataFrame

def _read_csv(path: Path) -> pd.DataFrame:
    return read_raw_csv(path, ["ts", *STATE_COLS, *CONTROL_COLS, "mode"])


def load_run(path: Path, step_seconds: int = 30) -> Run:
    frame = _read_csv(path)
    frame = frame.dropna(subset=["ts"] + STATE_COLS + CONTROL_COLS)
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last")

    # 原数据以 5 秒为主。30 秒均值抑制 0.1°C 量化噪声，同时保留控制时滞。
    numeric = STATE_COLS + CONTROL_COLS + ["mode"]
    frame = (
        frame.set_index("ts")[numeric]
        .resample(f"{step_seconds}s")
        .mean()
        .interpolate(method="time", limit=2, limit_direction="both")
        .dropna()
        .reset_index()
    )
    return Run(name=safe_source_name(path), path=path, split=infer_split(path), frame=frame)


def load_all_runs(step_seconds: int = 30) -> list[Run]:
    runs = [load_run(path, step_seconds=step_seconds) for path in discover_csvs()]
    too_short = [r.path for r in runs if len(r.frame) < 121]
    if too_short:
        raise ValueError(f"以下实验不足1小时: {too_short}")
    return runs


def describe_runs(runs: list[Run], step_seconds: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        f = run.frame
        rows.append(
            {
                "run": run.name,
                "split": run.split,
                "rows": len(f),
                "duration_h": (len(f) - 1) * step_seconds / 3600,
                "T_out_min": f.T_out.min(),
                "T_out_max": f.T_out.max(),
                "T_in_start": f.T_in.iloc[0],
                "T_in_end": f.T_in.iloc[-1],
                "freq_max": f.freq.max(),
            }
        )
    return pd.DataFrame(rows)


def make_windows(
    runs: list[Run],
    split: str,
    horizon: int = 120,
    stride: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """返回 state[样本,horizon+1,4]、controls[样本,horizon,3]。"""
    states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []
    for run in runs:
        if run.split != split:
            continue
        s = run.frame[STATE_COLS].to_numpy(np.float32)
        u = run.frame[CONTROL_COLS].to_numpy(np.float32)
        starts = list(range(0, len(run.frame) - horizon, stride))
        last = len(run.frame) - horizon - 1
        if last >= 0 and (not starts or starts[-1] != last):
            starts.append(last)
        for start in starts:
            states.append(s[start : start + horizon + 1])
            # u[t] drives state t -> t+1.
            controls.append(u[start : start + horizon])
            keys.append((run.name, start))
    return np.stack(states), np.stack(controls), keys


if __name__ == "__main__":
    runs = load_all_runs()
    summary = describe_runs(runs, 30)
    print(summary.groupby("split").agg(runs=("run", "count"), rows=("rows", "sum")))
