"""Visualization helpers for closed-loop control experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_run_debug(trajectory: pd.DataFrame, path: str | Path, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = trajectory.copy()
    minutes = frame["elapsed_seconds"].to_numpy(dtype=float) / 60.0
    electric_cum = frame["electric_kwh"].cumsum()
    thermal_cum = frame["thermal_kwh"].cumsum()
    efficiency = np.divide(
        thermal_cum,
        electric_cum,
        out=np.zeros_like(thermal_cum, dtype=float),
        where=electric_cum.to_numpy(dtype=float) > 0,
    )

    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    axes[0].plot(minutes, frame["T_in"], label="T_in", lw=1.7)
    axes[0].plot(minutes, frame["target_T_in"], label="target", lw=1.2, ls="--")
    axes[0].set_ylabel("Temp (C)")
    axes[0].legend()

    axes[1].plot(minutes, frame["freq_target"], label="freq_target", lw=1.5)
    axes[1].plot(minutes, frame["freq"], label="actual_freq", lw=1.5)
    axes[1].set_ylabel("Hz")
    axes[1].legend()

    axes[2].plot(minutes, frame["power_w"], label="power_w", lw=1.5)
    axes[2].plot(minutes, frame["thermal_power_w"], label="thermal_power_w", lw=1.5)
    axes[2].set_ylabel("W")
    axes[2].legend()

    axes[3].plot(minutes, electric_cum, label="electric_kwh", lw=1.5)
    axes[3].plot(minutes, thermal_cum, label="thermal_kwh", lw=1.5)
    axes[3].set_ylabel("kWh")
    axes[3].legend()

    axes[4].plot(minutes, efficiency, label="thermal/electric", lw=1.5)
    axes[4].set_ylabel("Efficiency")
    axes[4].set_xlabel("Minutes")
    axes[4].legend()

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_summary(summary: pd.DataFrame, path: str | Path, trajectories: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if trajectories.empty:
        raise ValueError("Cannot plot trajectory comparison from empty trajectories")

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    grouped = trajectories.groupby(["policy", "run"], sort=False)
    for (policy, run), frame in grouped:
        frame = frame.sort_values("elapsed_seconds")
        minutes = frame["elapsed_seconds"].to_numpy(dtype=float) / 60.0
        label = f"{policy}/{run}"
        axes[0].plot(minutes, frame["T_in"], lw=1.4, label=label)
        if "target_T_in" in frame:
            axes[0].plot(minutes, frame["target_T_in"], lw=0.9, ls="--", alpha=0.6)
        axes[1].plot(minutes, frame["freq_target"], lw=1.2, label=f"{label} target")
        axes[1].plot(minutes, frame["freq"], lw=1.2, ls="--", label=f"{label} actual")
        axes[2].plot(minutes, frame["power_w"], lw=1.2, label=f"{label} electric")
        axes[2].plot(minutes, frame["thermal_power_w"], lw=1.2, ls="--", label=f"{label} thermal")
        axes[3].plot(minutes, frame["electric_kwh"].cumsum(), lw=1.2, label=f"{label} electric")
        axes[3].plot(minutes, frame["thermal_kwh"].cumsum(), lw=1.2, ls="--", label=f"{label} thermal")

    axes[0].set_ylabel("Temp (C)")
    axes[1].set_ylabel("Hz")
    axes[2].set_ylabel("W")
    axes[3].set_ylabel("kWh")
    axes[3].set_xlabel("Minutes")
    axes[0].set_title("Closed-loop trajectory comparison")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
