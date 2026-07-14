"""Evaluate ``room_hybrid`` with strict open-loop rollout."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .api import ACTION_COLUMNS
from .data import OUTPUT_COLUMNS, assign_splits, discover_cooling_runs, load_status_csv
from .simulator import (
    DEFAULT_AUX_MODEL_PATH,
    DEFAULT_RESIDUAL_MODEL_PATH,
    DEFAULT_ROOM_MODEL_PATH,
    HybridRoomEnv,
)


def safe_curve_name(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset_eval"))
    parser.add_argument("--room-model", type=Path, default=DEFAULT_ROOM_MODEL_PATH)
    parser.add_argument("--residual-model", type=Path, default=DEFAULT_RESIDUAL_MODEL_PATH)
    parser.add_argument("--aux-model", type=Path, default=DEFAULT_AUX_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("simu/room_hybrid/output/dataset_eval"))
    parser.add_argument("--extra-run", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    runs = discover_cooling_runs(args.data_dir)
    assign_splits(runs)
    for path in args.extra_run:
        frame = load_status_csv(path)
        runs.append({
            "path": path,
            "name": str(path),
            "group": "extra",
            "split": "extra",
            "frame": frame,
            "cooling_fraction": float((frame["mode"].round() == 1).mean()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = args.output_dir / "curves"
    curves.mkdir(parents=True, exist_ok=True)

    env = HybridRoomEnv(
        room_model_path=args.room_model,
        residual_model_path=args.residual_model,
        aux_model_path=args.aux_model,
        device=args.device,
    )
    rows = []
    for run in runs:
        frame = run["frame"]
        actions = frame[ACTION_COLUMNS].to_numpy(np.float32)[:-1]
        result = env.simulate(frame.iloc[0], actions)
        n = min(len(frame), len(result))
        truth = frame[OUTPUT_COLUMNS].to_numpy(np.float32)[:n]
        pred = result[OUTPUT_COLUMNS].to_numpy(np.float32)[:n]
        error = pred - truth
        curve_name = safe_curve_name(run["name"])
        pd.DataFrame({
            "elapsed_min": np.arange(n) * 5.0 / 60.0,
            **{f"truth_{name}": truth[:, index] for index, name in enumerate(OUTPUT_COLUMNS)},
            **{f"pred_{name}": pred[:, index] for index, name in enumerate(OUTPUT_COLUMNS)},
            **{f"error_{name}": error[:, index] for index, name in enumerate(OUTPUT_COLUMNS)},
            "T_in_aux_backbone": result["T_in_aux_backbone"].to_numpy(np.float32)[:n],
            "T_in_slow": result["T_in_slow"].to_numpy(np.float32)[:n],
            "T_in_residual": result["T_in_residual"].to_numpy(np.float32)[:n],
            "T_in_residual_raw": result["T_in_residual_raw"].to_numpy(np.float32)[:n],
            "residual_weight": result["residual_weight"].to_numpy(np.float32)[:n],
        }).to_csv(curves / f"{curve_name}.csv", index=False, encoding="utf-8-sig")
        rows.append({
            "run": run["name"],
            "group": run["group"],
            "split": run["split"],
            "samples": n,
            "duration_h": (n - 1) * 5.0 / 3600.0,
            "T_in_mae_c": float(np.mean(np.abs(error[:, 0]))),
            "T_in_rmse_c": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "T_in_bias_c": float(np.mean(error[:, 0])),
            "T_in_end_error_c": float(error[-1, 0]),
            "T_in_coil_mae_c": float(np.mean(np.abs(error[:, 1]))),
            "T_out_coil_mae_c": float(np.mean(np.abs(error[:, 2]))),
            "T_out_discharge_mae_c": float(np.mean(np.abs(error[:, 3]))),
            "all_aux_mae_c": float(np.mean(np.abs(error[:, 1:]))),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = metrics.groupby("split").agg(
        runs=("run", "count"),
        T_in_mae_c=("T_in_mae_c", "mean"),
        T_in_rmse_c=("T_in_rmse_c", "mean"),
        T_in_end_abs_c=("T_in_end_error_c", lambda values: float(np.mean(np.abs(values)))),
        aux_mae_c=("all_aux_mae_c", "mean"),
    )
    summary.to_csv(args.output_dir / "summary.csv", encoding="utf-8-sig")
    print(summary.to_string())


if __name__ == "__main__":
    main()
