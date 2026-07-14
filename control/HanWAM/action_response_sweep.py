"""Visualize HanWAM prober response to single-action and combined controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from control.MiniController.config_schema import CONFIG_PATH, checkpoint_path, load_config, output_root

from .dataloader import Normalizer, PHYSICAL_COLS, action_bounds, block_sequence_arrays, load_all_runs
from .model import build_wam_model
from .utils import choose_device


ACTION_COLS = ("freq_target", "eev", "fan_out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep HanWAM action response for OAT and combined controls")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--horizons-seconds", default="300,600,1800")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _horizons(args: argparse.Namespace, step_seconds: int) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for raw in str(args.horizons_seconds).split(","):
        raw = raw.strip()
        if not raw:
            continue
        seconds = int(float(raw))
        values.append((seconds, max(1, int(round(seconds / float(step_seconds))))))
    return values


def _base_action(config: dict, bounds: dict[str, tuple[float, float]], mode: int) -> np.ndarray:
    base_cfg = ((config.get("method") or {}).get("base_action_by_mode") or {}).get(str(mode)) or {}
    base = np.asarray(
        [
            float(base_cfg.get("freq_target", 40.0)),
            float(base_cfg.get("eev", 165.0)),
            float(base_cfg.get("fan_out", 750.0)),
        ],
        dtype=np.float32,
    )
    lo = np.asarray([bounds[col][0] for col in ACTION_COLS], dtype=np.float32)
    hi = np.asarray([bounds[col][1] for col in ACTION_COLS], dtype=np.float32)
    return np.clip(base, lo, hi)


def _levels(bounds: dict[str, tuple[float, float]], base: np.ndarray) -> dict[str, list[float]]:
    freq_lo, freq_hi = bounds["freq_target"]
    eev_lo, eev_hi = bounds["eev"]
    fan_lo, fan_hi = bounds["fan_out"]
    return {
        "freq_target": sorted({float(freq_lo), 20.0, float(base[0]), 60.0, float(freq_hi)}),
        "eev": sorted({float(eev_lo), 120.0, float(base[1]), 240.0, 300.0, float(eev_hi)}),
        "fan_out": sorted({float(fan_lo), 300.0, 600.0, float(base[2]), float(fan_hi)}),
    }


def _oat_profiles(bounds: dict[str, tuple[float, float]], base: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    levels = _levels(bounds, base)
    for dim_idx, dim in enumerate(ACTION_COLS):
        for value in levels[dim]:
            action = base.copy()
            action[dim_idx] = np.float32(value)
            rows.append(
                {
                    "profile": f"oat_{dim}_{value:g}",
                    "sweep": dim,
                    "sweep_value": float(value),
                    "freq_target": float(action[0]),
                    "eev": float(action[1]),
                    "fan_out": float(action[2]),
                }
            )
    return rows


def _combo_profiles(bounds: dict[str, tuple[float, float]], base: np.ndarray) -> list[dict]:
    lo = {col: float(bounds[col][0]) for col in ACTION_COLS}
    hi = {col: float(bounds[col][1]) for col in ACTION_COLS}
    low_freq = max(15.0, float(base[0]) * 0.5)
    raw = [
        ("off_all", [0.0, base[1], lo["fan_out"]]),
        ("low_freq_base_fan", [low_freq, base[1], base[2]]),
        ("base", base.tolist()),
        ("high_freq_base_fan", [hi["freq_target"], base[1], base[2]]),
        ("high_freq_high_fan", [hi["freq_target"], base[1], hi["fan_out"]]),
        ("high_freq_low_fan", [hi["freq_target"], base[1], lo["fan_out"]]),
        ("low_freq_high_fan", [low_freq, base[1], hi["fan_out"]]),
        ("high_freq_low_eev_high_fan", [hi["freq_target"], lo["eev"], hi["fan_out"]]),
        ("high_freq_mid_eev_high_fan", [hi["freq_target"], 300.0, hi["fan_out"]]),
        ("high_freq_max_eev_high_fan", [hi["freq_target"], hi["eev"], hi["fan_out"]]),
    ]
    rows: list[dict] = []
    for name, action in raw:
        clipped = np.asarray(action, dtype=np.float32)
        clipped[0] = np.clip(clipped[0], lo["freq_target"], hi["freq_target"])
        clipped[1] = np.clip(clipped[1], lo["eev"], hi["eev"])
        clipped[2] = np.clip(clipped[2], lo["fan_out"], hi["fan_out"])
        rows.append(
            {
                "profile": name,
                "sweep": "combined",
                "sweep_value": np.nan,
                "freq_target": float(clipped[0]),
                "eev": float(clipped[1]),
                "fan_out": float(clipped[2]),
            }
        )
    return rows


@torch.no_grad()
def _predict_profiles(
    profile_rows: list[dict],
    *,
    model,
    obs_norm: Normalizer,
    action_norm: Normalizer,
    physical_norm: Normalizer,
    obs_history_blocks: np.ndarray,
    act_history_blocks: np.ndarray,
    frames_per_block: int,
    horizon_steps: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    future_blocks = int(np.ceil(horizon_steps / float(frames_per_block)))
    rollout_steps = future_blocks * frames_per_block
    obs_n = torch.as_tensor(obs_norm.encode(obs_history_blocks), dtype=torch.float32, device=device)
    act_hist_n = torch.as_tensor(action_norm.encode(act_history_blocks), dtype=torch.float32, device=device)
    phys_idx = {name: idx for idx, name in enumerate(PHYSICAL_COLS)}
    summary_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    for row in profile_rows:
        action = np.asarray([row[col] for col in ACTION_COLS], dtype=np.float32)
        actions = np.tile(action.reshape(1, 1, 1, -1), (len(obs_history_blocks), future_blocks, frames_per_block, 1))
        actions_n = torch.as_tensor(action_norm.encode(actions), dtype=torch.float32, device=device)
        _, physical_n = model.rollout(obs_n, act_hist_n, actions_n)
        physical = physical_norm.decode(physical_n.detach().cpu().numpy().reshape(-1, len(PHYSICAL_COLS))).reshape(
            len(obs_history_blocks),
            rollout_steps,
            len(PHYSICAL_COLS),
        )[:, :horizon_steps]
        temp_delta = physical[:, :, phys_idx["T_in_delta"]]
        energy_delta = np.clip(physical[:, :, phys_idx["electric_kwh_delta"]], 0.0, None)
        mean_temp_cum = np.cumsum(temp_delta, axis=1).mean(axis=0)
        mean_energy_cum = np.cumsum(energy_delta, axis=1).mean(axis=0)
        for step in range(horizon_steps):
            trajectory_rows.append(
                {
                    **row,
                    "step": int(step + 1),
                    "mean_cum_T_in_delta_c": float(mean_temp_cum[step]),
                    "mean_cum_electric_kwh": float(mean_energy_cum[step]),
                }
            )
        for horizon_seconds, steps in _GLOBAL_HORIZONS:
            use_steps = min(int(steps), horizon_steps)
            summary_rows.append(
                {
                    **row,
                    "windows": int(len(obs_history_blocks)),
                    "horizon_seconds": int(horizon_seconds),
                    "horizon_steps": int(use_steps),
                    "mean_cum_T_in_delta_c": float(temp_delta[:, :use_steps].sum(axis=1).mean()),
                    "mean_cum_electric_kwh": float(energy_delta[:, :use_steps].sum(axis=1).mean()),
                    "mean_step_T_in_delta_c": float(temp_delta[:, :use_steps].mean()),
                    "mean_step_electric_kwh": float(energy_delta[:, :use_steps].mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(trajectory_rows)


def _plot_oat(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=False)
    horizons = sorted(summary["horizon_seconds"].unique())
    for col_idx, sweep in enumerate(ACTION_COLS):
        frame = summary[summary["sweep"] == sweep].sort_values(["horizon_seconds", "sweep_value"])
        for horizon in horizons:
            sub = frame[frame["horizon_seconds"] == horizon]
            minutes = int(horizon // 60)
            axes[0, col_idx].plot(sub["sweep_value"], sub["mean_cum_T_in_delta_c"], marker="o", label=f"{minutes}min")
            axes[1, col_idx].plot(sub["sweep_value"], sub["mean_cum_electric_kwh"], marker="o", label=f"{minutes}min")
        axes[0, col_idx].axhline(0.0, color="gray", lw=0.8, ls=":")
        axes[0, col_idx].set_title(f"Vary {sweep}")
        axes[1, col_idx].set_xlabel(sweep)
        axes[0, col_idx].grid(alpha=0.25)
        axes[1, col_idx].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Cumulative T_in delta (C)")
    axes[1, 0].set_ylabel("Cumulative electric (kWh)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("HanWAM one-action-at-a-time response")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_combo(trajectory: pd.DataFrame, output: Path, step_seconds: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    order = [
        "off_all",
        "low_freq_base_fan",
        "base",
        "high_freq_base_fan",
        "high_freq_high_fan",
        "high_freq_low_fan",
        "low_freq_high_fan",
        "high_freq_low_eev_high_fan",
        "high_freq_mid_eev_high_fan",
        "high_freq_max_eev_high_fan",
    ]
    for profile in order:
        frame = trajectory[trajectory["profile"] == profile]
        if frame.empty:
            continue
        minutes = frame["step"].to_numpy(dtype=float) * float(step_seconds) / 60.0
        axes[0].plot(minutes, frame["mean_cum_T_in_delta_c"], label=profile)
        axes[1].plot(minutes, frame["mean_cum_electric_kwh"], label=profile)
    axes[0].axhline(0.0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("Cumulative T_in delta (C)")
    axes[1].set_ylabel("Cumulative electric (kWh)")
    axes[1].set_xlabel("Minutes")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("HanWAM combined-action response")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


_GLOBAL_HORIZONS: list[tuple[int, int]] = []


def main() -> None:
    global _GLOBAL_HORIZONS
    args = parse_args()
    config = load_config(args.config)
    mode = int(args.mode if args.mode is not None else config["eval"]["mode"])
    ckpt_path = args.checkpoint or checkpoint_path(config, mode)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = choose_device(args.device or (config.get("train") or {}).get("device", "auto"))
    model = build_wam_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    obs_norm = Normalizer.from_dict(checkpoint["obs_norm"])
    action_norm = Normalizer.from_dict(checkpoint["target_action_norm"])
    physical_norm = Normalizer.from_dict(checkpoint["physical_norm"])
    step_seconds = int(
        (checkpoint.get("planner_config") or {}).get(
            "step_seconds",
            (config.get("data") or {}).get("sampling", {}).get("step_seconds", 5),
        )
    )
    frames_per_block = int(
        checkpoint.get(
            "frames_per_block",
            (checkpoint.get("planner_config") or {}).get("frames_per_block", getattr(model, "frames_per_block", 12)),
        )
    )
    _GLOBAL_HORIZONS = _horizons(args, step_seconds)
    horizon_steps = max(step for _, step in _GLOBAL_HORIZONS)

    runs = load_all_runs(mode=mode, config=config)
    obs_history_blocks, act_history_blocks, _, _, _, _ = block_sequence_arrays(
        runs,
        args.split,
        config=config,
        limit=int(args.limit),
        seed=int(args.seed),
    )
    bounds = action_bounds(runs, config=config)
    base = _base_action(config, bounds, mode)
    oat_rows = _oat_profiles(bounds, base)
    combo_rows = _combo_profiles(bounds, base)

    oat_summary, oat_trajectory = _predict_profiles(
        oat_rows,
        model=model,
        obs_norm=obs_norm,
        action_norm=action_norm,
        physical_norm=physical_norm,
        obs_history_blocks=obs_history_blocks,
        act_history_blocks=act_history_blocks,
        frames_per_block=frames_per_block,
        horizon_steps=horizon_steps,
        device=device,
    )
    combo_summary, combo_trajectory = _predict_profiles(
        combo_rows,
        model=model,
        obs_norm=obs_norm,
        action_norm=action_norm,
        physical_norm=physical_norm,
        obs_history_blocks=obs_history_blocks,
        act_history_blocks=act_history_blocks,
        frames_per_block=frames_per_block,
        horizon_steps=horizon_steps,
        device=device,
    )

    out_dir = args.output_dir or (output_root(config) / config["experiment"]["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    oat_summary.to_csv(out_dir / "action_response_oat_summary.csv", index=False, encoding="utf-8-sig")
    oat_trajectory.to_csv(out_dir / "action_response_oat_trajectory.csv", index=False, encoding="utf-8-sig")
    combo_summary.to_csv(out_dir / "action_response_combo_summary.csv", index=False, encoding="utf-8-sig")
    combo_trajectory.to_csv(out_dir / "action_response_combo_trajectory.csv", index=False, encoding="utf-8-sig")
    _plot_oat(oat_summary, out_dir / "action_response_oat.png")
    _plot_combo(combo_trajectory, out_dir / "action_response_combo.png", step_seconds)

    print("OAT 30min response:")
    print(oat_summary[oat_summary["horizon_seconds"] == max(h for h, _ in _GLOBAL_HORIZONS)].to_string(index=False))
    print("Combined 30min response:")
    print(combo_summary[combo_summary["horizon_seconds"] == max(h for h, _ in _GLOBAL_HORIZONS)].to_string(index=False))


if __name__ == "__main__":
    main()
