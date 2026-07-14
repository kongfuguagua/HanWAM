"""Open-loop HanWAM action-sensitivity diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from control.MiniController.config_schema import CONFIG_PATH, checkpoint_path, load_config

from .dataloader import Normalizer, PHYSICAL_COLS, action_bounds, block_sequence_arrays, load_all_runs
from .model import build_wam_model
from .utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prober rollouts under fixed action profiles")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--horizons-seconds", default="300,600,1800", help="Comma-separated report horizons.")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    return parser.parse_args()


def _profile_actions(config: dict, bounds: dict[str, tuple[float, float]], mode: int) -> dict[str, np.ndarray]:
    base_cfg = ((config.get("method") or {}).get("base_action_by_mode") or {}).get(str(mode)) or {}
    base = np.asarray(
        [
            float(base_cfg.get("freq_target", 40.0)),
            float(base_cfg.get("eev", 165.0)),
            float(base_cfg.get("fan_out", 750.0)),
        ],
        dtype=np.float32,
    )
    lo = np.asarray([bounds[col][0] for col in ("freq_target", "eev", "fan_out")], dtype=np.float32)
    hi = np.asarray([bounds[col][1] for col in ("freq_target", "eev", "fan_out")], dtype=np.float32)
    low_freq = max(15.0, float(base[0]) * 0.5)
    return {
        "A_high_freq": np.asarray([hi[0], base[1], base[2]], dtype=np.float32),
        "B_low_freq": np.asarray([low_freq, base[1], base[2]], dtype=np.float32),
        "C_off": np.asarray([0.0, base[1], lo[2]], dtype=np.float32),
        "base": base,
        "high": np.asarray([hi[0], base[1], hi[2]], dtype=np.float32),
        "high_freq_low_fan": np.asarray([hi[0], base[1], lo[2]], dtype=np.float32),
        "low_freq_high_fan": np.asarray([low_freq, base[1], hi[2]], dtype=np.float32),
    }


def _report_horizons(args: argparse.Namespace, step_seconds: int) -> list[tuple[int, int]]:
    values = []
    for raw in str(args.horizons_seconds).split(","):
        raw = raw.strip()
        if not raw:
            continue
        seconds = int(float(raw))
        values.append((seconds, max(1, int(round(seconds / float(step_seconds))))))
    if args.horizon_steps is not None:
        total = int(args.horizon_steps)
    else:
        total = max(step for _, step in values)
    return [(seconds, min(step, total)) for seconds, step in values]


def _plot_result(result: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    abc = result[result["profile"].isin(["A_high_freq", "B_low_freq", "C_off"])].copy()
    if abc.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for profile, frame in abc.groupby("profile", sort=False):
        frame = frame.sort_values("horizon_seconds")
        minutes = frame["horizon_seconds"].to_numpy(dtype=float) / 60.0
        axes[0].plot(minutes, frame["mean_cum_T_in_delta_c"], marker="o", label=profile)
        axes[1].plot(minutes, frame["mean_cum_electric_kwh"], marker="o", label=profile)
    axes[0].axhline(0.0, color="gray", lw=0.9, ls=":")
    axes[0].set_ylabel("Predicted cumulative T_in delta (C)")
    axes[1].set_ylabel("Predicted cumulative electric (kWh)")
    axes[1].set_xlabel("Horizon (minutes)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("HanWAM open-loop action sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_diagnostic(args: argparse.Namespace) -> pd.DataFrame:
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
    step_seconds = int((checkpoint.get("planner_config") or {}).get("step_seconds", (config.get("data") or {}).get("sampling", {}).get("step_seconds", 5)))
    frames_per_block = int(checkpoint.get("frames_per_block", (checkpoint.get("planner_config") or {}).get("frames_per_block", getattr(model, "frames_per_block", 12))))
    report_horizons = _report_horizons(args, step_seconds)
    horizon_steps = int(args.horizon_steps or max(step for _, step in report_horizons))
    future_blocks = int(np.ceil(horizon_steps / float(frames_per_block)))
    rollout_steps = future_blocks * frames_per_block
    runs = load_all_runs(mode=mode, config=config)
    obs_history_blocks, act_history_blocks, _, _, _, keys = block_sequence_arrays(
        runs,
        args.split,
        config=config,
        limit=int(args.limit),
        seed=int(args.seed),
    )
    obs_n = torch.as_tensor(obs_norm.encode(obs_history_blocks), dtype=torch.float32, device=device)
    act_hist_n = torch.as_tensor(action_norm.encode(act_history_blocks), dtype=torch.float32, device=device)

    bounds = action_bounds(runs, config=config)
    profiles = _profile_actions(config, bounds, mode)
    phys_idx = {name: idx for idx, name in enumerate(PHYSICAL_COLS)}
    rows: list[dict] = []
    for name, action in profiles.items():
        actions = np.tile(action.reshape(1, 1, 1, -1), (len(obs_history_blocks), future_blocks, frames_per_block, 1)).astype(np.float32)
        actions_n = torch.as_tensor(action_norm.encode(actions), dtype=torch.float32, device=device)
        _, physical_n = model.rollout(obs_n, act_hist_n, actions_n)
        physical = physical_norm.decode(physical_n.detach().cpu().numpy().reshape(-1, len(PHYSICAL_COLS))).reshape(
            len(obs_history_blocks),
            rollout_steps,
            len(PHYSICAL_COLS),
        )[:, :horizon_steps]
        temp_delta = physical[:, :, phys_idx["T_in_delta"]]
        energy_delta = np.clip(physical[:, :, phys_idx["electric_kwh_delta"]], 0.0, None)
        for horizon_seconds, steps in report_horizons:
            rows.append(
                {
                    "profile": name,
                    "windows": int(len(keys)),
                    "horizon_seconds": int(horizon_seconds),
                    "horizon_steps": int(steps),
                    "freq_target": float(action[0]),
                    "eev": float(action[1]),
                    "fan_out": float(action[2]),
                    "mean_cum_T_in_delta_c": float(temp_delta[:, :steps].sum(axis=1).mean()),
                    "mean_cum_electric_kwh": float(energy_delta[:, :steps].sum(axis=1).mean()),
                    "mean_step_T_in_delta_c": float(temp_delta[:, :steps].mean()),
                    "mean_step_electric_kwh": float(energy_delta[:, :steps].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    result = run_diagnostic(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.plot is not None:
        _plot_result(result, args.plot)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
