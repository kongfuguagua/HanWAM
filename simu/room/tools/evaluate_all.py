"""Evaluate the cooling simulator on every cooling CSV and save one plot per run.

Example
-------
python -m simu.room.tools.evaluate_all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..continuous_model import CONTROL_COLUMNS
from ..simulator import ContinuousEnthalpyRoomEnv
from .data_pipeline import assign_groupwise_splits, discover_cooling_runs


def simulate_all_batched(model_path: Path, runs: list[dict], use_physics_offcycle: bool = True) -> list[dict]:
    """Closed-loop simulation with one batched model call per 5 s clock tick."""
    payload = joblib.load(model_path)
    model = payload["model"]
    if int(payload.get("supported_mode", 1)) != 1:
        raise ValueError(f"{model_path} is not a cooling model")

    feature_variant = str(payload.get("feature_variant", "thermal_inertia"))

    contexts: list[dict] = []
    for run in runs:
        simulator = ContinuousEnthalpyRoomEnv(
            model_path, use_physics_offcycle=use_physics_offcycle,
        )
        simulator.reset(run["frame"].iloc[0])
        contexts.append({
            "run": run,
            "simulator": simulator,
            "controls": run["frame"][CONTROL_COLUMNS].to_numpy(np.float32)[:-1],
            "temperature": [simulator.temperature],
            "tracking_error": [0.0],
            "load_proxy": [simulator.heat_load.load_proxy],
        })

    max_steps = max(len(c["controls"]) for c in contexts)
    for step in range(max_steps):
        active: list[dict] = []
        features: list[np.ndarray] = []
        for context in contexts:
            if step >= len(context["controls"]):
                continue
            simulator = context["simulator"]
            if feature_variant.startswith("autoregressive"):
                simulator.prefix.update_temperature(simulator.temperature)
            simulator.prefix.update(context["controls"][step])
            feature = simulator.prefix.feature()
            features.append(feature)
            active.append(context)

        predictions = model.predict(np.asarray(features, dtype=np.float32))
        for context, prediction in zip(active, predictions):
            simulator = context["simulator"]
            if feature_variant.startswith("autoregressive"):
                freq = float(context["controls"][step][0])
                horizon = max(1, simulator.autoregressive_horizon)
                if simulator.use_physics_offcycle and freq <= simulator.freq_zero_threshold:
                    dT = (
                        simulator.offcycle_b * (simulator._outdoor_temperature - simulator.temperature)
                        + simulator.offcycle_c
                    ) * (simulator.dt / 60.0)
                    simulator.temperature = float(np.clip(simulator.temperature + dT, -30.0, 65.0))
                elif "_delta" in feature_variant:
                    dT = float(prediction) / horizon
                    if simulator.max_rate_c_per_min is not None:
                        max_step = simulator.max_rate_c_per_min * simulator.dt / 60.0
                        dT = float(np.clip(dT, -max_step, max_step))
                    simulator.temperature = float(np.clip(simulator.temperature + dT, -30.0, 65.0))
                else:
                    target = float(prediction)
                    delta = (target - simulator.temperature) / horizon
                    if simulator.max_rate_c_per_min is not None:
                        max_step = simulator.max_rate_c_per_min * simulator.dt / 60.0
                        delta = float(np.clip(delta, -max_step, max_step))
                    simulator.temperature = float(np.clip(simulator.temperature + delta, -30.0, 65.0))
            else:
                direct_target = simulator.reference_temperature + float(prediction)
                simulator._advance_temperature(direct_target)
            simulator.heat_load.update(simulator.temperature, simulator.dt)
            heat_features = simulator.heat_load.features()
            context["temperature"].append(simulator.temperature)
            context["tracking_error"].append(float(heat_features[2]))
            context["load_proxy"].append(float(heat_features[3]))
    return contexts


def safe_stem(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "__", name).strip("_")


def save_run_plot(context: dict, output_dir: Path) -> dict:
    run = context["run"]
    frame = run["frame"]
    predicted = np.asarray(context["temperature"], dtype=float)
    measured = frame["T_in"].to_numpy(dtype=float)[:len(predicted)]
    tracking_error = np.asarray(context["tracking_error"], dtype=float)
    load_proxy = np.asarray(context["load_proxy"], dtype=float)
    error = predicted - measured
    hours = np.arange(len(predicted)) * 5.0 / 3600.0

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    max_abs = float(np.max(np.abs(error)))

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )
    axes[0].plot(hours, measured, label="measured", linewidth=1.5)
    axes[0].plot(hours, predicted, label="simulated", linewidth=1.25)
    axes[0].set_ylabel("indoor temperature (degC)")
    axes[0].set_title(
        f"{run['name']} [{run['split']}]  MAE={mae:.3f} degC  RMSE={rmse:.3f} degC"
    )
    axes[0].legend()

    axes[1].plot(hours, error, color="tab:red", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhspan(-0.5, 0.5, color="tab:green", alpha=0.12, label="+/-0.5 degC")
    axes[1].set_ylabel("simulation error (degC)")
    axes[1].legend(loc="upper right")

    axes[2].plot(hours, tracking_error, label="heat-load tracking error", linewidth=1.0)
    axes[2].axhline(0.5, color="tab:red", linestyle="--", linewidth=0.8)
    axes[2].axhline(-0.5, color="tab:red", linestyle="--", linewidth=0.8)
    proxy_axis = axes[2].twinx()
    proxy_axis.plot(hours, load_proxy, color="tab:orange", alpha=0.55,
                    label="normalized load proxy")
    axes[2].set_ylabel("tracking error (degC)")
    proxy_axis.set_ylabel("load proxy")
    axes[2].set_xlabel("time (h)")

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    image_path = output_dir / f"{safe_stem(run['name'])}.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    return {
        "run": run["name"], "group": run["group"], "split": run["split"],
        "samples": len(predicted), "duration_h": (len(predicted) - 1) * 5.0 / 3600.0,
        "mae_c": mae, "rmse_c": rmse, "bias_c": bias, "max_abs_c": max_abs,
        "max_sim_step_c": float(np.max(np.abs(np.diff(predicted)))),
        "heat_tracking_max_abs_c": float(np.max(np.abs(tracking_error))),
        "image": image_path.name,
    }


def save_summary_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    ordered = metrics.sort_values("mae_c", ascending=True).reset_index(drop=True)
    colors = np.where(ordered["split"].eq("test"), "tab:orange", "tab:blue")
    height = max(8.0, 0.24 * len(ordered))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(ordered["run"], ordered["mae_c"], color=colors)
    ax.axvline(0.5, color="tab:red", linestyle="--", label="0.5 degC")
    ax.set(xlabel="MAE (degC)", title="Cooling simulation error for every run")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "summary_mae.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--model", type=Path,
        default=Path("simu/room/continuous_cooling_model.joblib"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("simu/room/output/cooling_v3_all_runs"),
    )
    parser.add_argument(
        "--no-physics-offcycle", action="store_false", dest="use_physics_offcycle",
        help="Disable the off-cycle physics fallback and use the ML model only.",
    )
    parser.set_defaults(use_physics_offcycle=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_cooling_runs(args.data_dir)
    if not runs:
        raise ValueError(f"no cooling runs under {args.data_dir}")
    if args.data_dir.name.lower().endswith("eval"):
        for run in runs:
            run["split"] = "eval"
    else:
        assign_groupwise_splits(runs)
    print(f"cooling runs: {len(runs)}")

    contexts = simulate_all_batched(args.model, runs, args.use_physics_offcycle)
    rows = [save_run_plot(context, args.output_dir) for context in contexts]
    metrics = pd.DataFrame(rows).sort_values(["split", "group", "run"])
    metrics.to_csv(args.output_dir / "metrics_all_cooling_runs.csv",
                   index=False, encoding="utf-8-sig")
    save_summary_plot(metrics, args.output_dir)

    summary = metrics.groupby("split").agg(
        runs=("run", "size"), mae_c=("mae_c", "mean"),
        rmse_c=("rmse_c", "mean"), max_abs_c=("max_abs_c", "max"),
    )
    print(summary.round(4).to_string())
    print(f"images: {len(rows)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
