"""Plot strict open-loop V4 temperature curves for complete experiment runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import assign_v4_splits, discover_v4_cooling_runs
from .following_simulator import HybridRoomV4FollowingEnv
from .model import CONTROL_COLUMNS
from .simulator import HybridRoomV4Env


_WORKER_ENV = None


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "__", value).strip("_")


def simulate_run(env: HybridRoomV4Env, run: dict) -> dict:
    frame = run["frame"]
    controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)[:-1]
    prediction = env.simulate(frame.iloc[0], controls)["T_in"].to_numpy()
    truth = frame["T_in"].to_numpy()[:len(prediction)]
    hours = np.arange(len(truth), dtype=float) * 5.0 / 3600.0
    error = prediction - truth
    result = {
        "hours": hours, "truth": truth, "prediction": prediction, "error": error,
        "mae_c": float(np.mean(np.abs(error))),
        "rmse_c": float(np.sqrt(np.mean(error ** 2))),
        "bias_c": float(np.mean(error)),
        "max_abs_c": float(np.max(np.abs(error))),
    }
    for horizon, label in ((12, "60s"), (60, "300s")):
        measured_change = truth[horizon:] - truth[:-horizon]
        predicted_change = prediction[horizon:] - prediction[:-horizon]
        result[f"delta_{label}_corr"] = float(np.corrcoef(
            measured_change, predicted_change,
        )[0, 1])
        result[f"delta_{label}_std_ratio"] = float(
            np.std(predicted_change) / max(np.std(measured_change), 1e-8)
        )
    return result


def _init_worker(model_path: str, following_weight: float | None) -> None:
    global _WORKER_ENV
    _limit_numeric_threads()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER_ENV = (
        HybridRoomV4Env(model_path)
        if following_weight is None
        else HybridRoomV4FollowingEnv(
            slow_model_path=model_path,
            fast_weight=following_weight,
        )
    )


def _limit_numeric_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def _simulate_run_worker(run: dict) -> tuple[dict, dict]:
    if _WORKER_ENV is None:
        raise RuntimeError("worker environment has not been initialized")
    return run, simulate_run(_WORKER_ENV, run)


def plot_single(run: dict, result: dict, output: Path) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(10.5, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]}, constrained_layout=True,
    )
    axes[0].plot(result["hours"], result["truth"], color="black", linewidth=1.35,
                 label="Measured T_in")
    axes[0].plot(result["hours"], result["prediction"], color="#1f77b4", linewidth=1.25,
                 label="V4 open-loop T_in")
    axes[0].set_ylabel("Temperature (degC)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[0].set_title(
        f"{run['name']}  |  MAE={result['mae_c']:.3f} C, "
        f"RMSE={result['rmse_c']:.3f} C"
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(result["hours"], result["error"], color="#d62728", linewidth=1.0)
    axes[1].set_xlabel("Elapsed time (h)")
    axes[1].set_ylabel("Error (C)")
    axes[1].grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_overview(items: list[tuple[dict, dict]], output: Path) -> None:
    columns = 3 if len(items) <= 12 else 4
    rows = math.ceil(len(items) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 2.35 * rows), squeeze=False,
        constrained_layout=True,
    )
    for axis, (run, result) in zip(axes.flat, items):
        axis.plot(result["hours"], result["truth"], color="black", linewidth=0.9)
        axis.plot(result["hours"], result["prediction"], color="#1f77b4", linewidth=0.9)
        axis.set_title(f"{run['name']}\nMAE {result['mae_c']:.3f} C", fontsize=8)
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(items):]:
        axis.set_visible(False)
    figure.supxlabel("Elapsed time (h)")
    figure.supylabel("T_in (degC)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("simu/roomv4/room_v4_model.pt"))
    parser.add_argument("--subset", choices=["train", "test", "all"], default="all")
    parser.add_argument("--following-weight", type=float, default=None,
                        help="Use the V4.1 fast/slow following blend with this fast weight.")
    parser.add_argument("--overview-only", action="store_true",
                        help="Only write metrics.csv and the overview figure; skip per-run PNGs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-run progress output.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of worker processes used for run-level simulation.")
    args = parser.parse_args()

    runs = discover_v4_cooling_runs(args.data_dir)
    if args.subset != "all":
        assign_v4_splits(runs)
        runs = [run for run in runs if run["split"] == args.subset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _limit_numeric_threads()
    if args.jobs < 1:
        raise ValueError("--jobs 必须 >= 1")
    env = None
    if args.jobs == 1:
        env = (
            HybridRoomV4Env(args.model)
            if args.following_weight is None
            else HybridRoomV4FollowingEnv(
                slow_model_path=args.model, fast_weight=args.following_weight,
            )
        )
    rows = []
    items = []
    if args.jobs == 1:
        iterator = ((run, simulate_run(env, run)) for run in runs)
    else:
        pool = ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(str(args.model), args.following_weight),
        )
        iterator = pool.map(_simulate_run_worker, runs)
    try:
        for index, (run, result) in enumerate(iterator, 1):
            items.append((run, result))
            filename = ""
            if not args.overview_only:
                filename = safe_name(run["name"]) + ".png"
                plot_single(run, result, args.output_dir / filename)
            rows.append({
                "run": run["name"], "samples": len(result["truth"]),
                "duration_h": result["hours"][-1],
                "mae_c": result["mae_c"], "rmse_c": result["rmse_c"],
                "bias_c": result["bias_c"], "max_abs_c": result["max_abs_c"],
                "delta_60s_corr": result["delta_60s_corr"],
                "delta_60s_std_ratio": result["delta_60s_std_ratio"],
                "delta_300s_corr": result["delta_300s_corr"],
                "delta_300s_std_ratio": result["delta_300s_std_ratio"],
                "image": filename,
            })
            if not args.quiet:
                print(f"[{index:02d}/{len(runs):02d}] {run['name']} MAE={result['mae_c']:.4f} C")
    finally:
        if args.jobs != 1:
            pool.shutdown(wait=True, cancel_futures=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    plot_overview(items, args.output_dir / "temperature_curves_overview.png")
    print(f"runs={len(metrics)} mean_mae={metrics['mae_c'].mean():.4f} "
          f"mean_rmse={metrics['rmse_c'].mean():.4f}")


if __name__ == "__main__":
    main()
