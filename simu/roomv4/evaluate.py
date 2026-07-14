"""Evaluate V4 with strict whole-run open-loop simulation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import assign_v4_splits, discover_v4_cooling_runs
from .model import CONTROL_COLUMNS
from .simulator import HybridRoomV4Env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset_full"))
    parser.add_argument("--model", type=Path,
                        default=Path("simu/roomv4/room_v4_model.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("simu/roomv4/output"))
    args = parser.parse_args()
    runs = discover_v4_cooling_runs(args.data_dir)
    assign_v4_splits(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = HybridRoomV4Env(args.model)
    rows = []
    for run in runs:
        frame = run["frame"]
        controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)[:-1]
        prediction = env.simulate(frame.iloc[0], controls)["T_in"].to_numpy()
        truth = frame["T_in"].to_numpy()[:len(prediction)]
        error = prediction - truth
        rows.append({
            "run": run["name"], "group": run["group"], "split": run["split"],
            "samples": len(error), "duration_h": (len(error) - 1) * 5.0 / 3600.0,
            "mae_c": float(np.mean(np.abs(error))),
            "rmse_c": float(np.sqrt(np.mean(error ** 2))),
            "bias_c": float(np.mean(error)),
            "max_abs_c": float(np.max(np.abs(error))),
            "end_error_c": float(error[-1]),
            "max_step_c": float(np.max(np.abs(np.diff(prediction)))),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics_all_runs.csv", index=False,
                   encoding="utf-8-sig")
    summary = metrics.groupby("split").agg(
        runs=("run", "count"), mae_c=("mae_c", "mean"),
        rmse_c=("rmse_c", "mean"), max_abs_c=("max_abs_c", "max"),
        end_abs_c=("end_error_c", lambda values: float(np.mean(np.abs(values)))),
    )
    summary.to_csv(args.output_dir / "summary.csv", encoding="utf-8-sig")
    print(summary.to_string())


if __name__ == "__main__":
    main()
