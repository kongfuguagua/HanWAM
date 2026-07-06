"""Generate HanWAM rollout visualizations from saved CSV artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENTS = [
    "hanwam_design_validation",
    "hanwam_large_v1",
    "hanwam_large_v1_offfan003_eval",
    "hanwam_large_v1_actionprior005_eval",
    "hanwam_large_v1_actionprior003_fan0_eval",
    "hanwam_medium_v1",
    "hanwam_design_validation_deploycost_eval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save visual debug plots for HanWAM rollout directories")
    parser.add_argument("--outputs-root", type=Path, default=Path("control/outputs"))
    parser.add_argument("--experiment", action="append", default=None, help="Experiment name. May be repeated.")
    parser.add_argument("--scenario", default="one_four_hour")
    parser.add_argument("--mode", default="mode1")
    return parser.parse_args()


def _safe_cumsum(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame:
        return np.zeros(len(frame), dtype=float)
    return frame[col].to_numpy(dtype=float).cumsum()


def _plot_optional(ax, minutes: np.ndarray, frame: pd.DataFrame, col: str, *, label: str | None = None, **kwargs) -> None:
    if col in frame:
        ax.plot(minutes, frame[col].to_numpy(dtype=float), label=label or col, **kwargs)


def plot_rollout(rollout_dir: Path) -> Path | None:
    trajectory_path = rollout_dir / "trajectories.csv"
    controller_path = rollout_dir / "controller_log.csv"
    if not trajectory_path.exists():
        return None
    trajectory = pd.read_csv(trajectory_path, encoding="utf-8-sig").sort_values("elapsed_seconds")
    controller = pd.read_csv(controller_path, encoding="utf-8-sig").sort_values("elapsed_seconds") if controller_path.exists() else pd.DataFrame()
    minutes = trajectory["elapsed_seconds"].to_numpy(dtype=float) / 60.0
    electric_cum = _safe_cumsum(trajectory, "electric_kwh")
    thermal_cum = _safe_cumsum(trajectory, "thermal_kwh")

    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    title = "/".join(rollout_dir.parts[-4:])

    _plot_optional(axes[0], minutes, trajectory, "T_in", label="T_in", lw=1.7)
    _plot_optional(axes[0], minutes, trajectory, "target_T_in", label="target", lw=1.2, ls="--")
    _plot_optional(axes[0], minutes, trajectory, "T_out", label="T_out", lw=1.2, alpha=0.75)
    _plot_optional(axes[0], minutes, trajectory, "T_in_coil", label="T_in_coil", lw=1.0, alpha=0.6)
    axes[0].set_ylabel("Temperature (C)")

    _plot_optional(axes[1], minutes, trajectory, "T_out", label="T_out", lw=1.5)
    _plot_optional(axes[1], minutes, trajectory, "T_out_coil", label="T_out_coil", lw=1.2, alpha=0.75)
    if "RH_in" in trajectory:
        ax_rh = axes[1].twinx()
        ax_rh.plot(minutes, trajectory["RH_in"].to_numpy(dtype=float), label="RH_in", lw=1.0, color="tab:green", alpha=0.55)
        ax_rh.set_ylabel("RH")
        ax_rh.legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Outdoor / Coil (C)")

    _plot_optional(axes[2], minutes, trajectory, "freq_target", label="freq_target", lw=1.4)
    _plot_optional(axes[2], minutes, trajectory, "freq", label="actual_freq", lw=1.4)
    axes[2].axhline(15.0, color="gray", lw=0.9, ls=":", label="15Hz threshold")
    axes[2].set_ylabel("Compressor (Hz)")

    _plot_optional(axes[3], minutes, trajectory, "fan_out", label="fan_out", lw=1.4)
    _plot_optional(axes[3], minutes, trajectory, "fan_in", label="fan_in", lw=1.0, alpha=0.6)
    axes[3].set_ylabel("Fan")

    _plot_optional(axes[4], minutes, trajectory, "eev", label="eev", lw=1.4)
    axes[4].set_ylabel("EEV")

    _plot_optional(axes[5], minutes, trajectory, "power_w", label="electric_power_w", lw=1.4)
    _plot_optional(axes[5], minutes, trajectory, "thermal_power_w", label="thermal_power_w", lw=1.2, alpha=0.75)
    axes[5].set_ylabel("Power (W)")

    axes[6].plot(minutes, electric_cum, label="cumulative electric_kwh", lw=1.5)
    axes[6].plot(minutes, thermal_cum, label="cumulative thermal_kwh", lw=1.2)
    if len(electric_cum) and electric_cum[-1] > 0:
        axes[6].text(
            0.01,
            0.92,
            f"final electric={electric_cum[-1]:.3f} kWh",
            transform=axes[6].transAxes,
            fontsize=10,
            va="top",
        )
    axes[6].set_ylabel("Energy (kWh)")
    axes[6].set_xlabel("Minutes")

    if not controller.empty:
        call_count = int((controller.get("hanwam_reused_plan", pd.Series([False] * len(controller))) != True).sum())
        axes[2].text(
            0.01,
            0.92,
            f"planner calls={call_count}, controller rows={len(controller)}",
            transform=axes[2].transAxes,
            fontsize=9,
            va="top",
        )

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = rollout_dir / "rollout_debug.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    experiments = args.experiment or DEFAULT_EXPERIMENTS
    for experiment in experiments:
        rollout_dir = args.outputs_root / experiment / args.scenario / args.mode
        image = plot_rollout(rollout_dir)
        if image is None:
            print(f"[skip] missing trajectories.csv: {rollout_dir}")
            continue
        print(f"[ok] {image}")


if __name__ == "__main__":
    main()
