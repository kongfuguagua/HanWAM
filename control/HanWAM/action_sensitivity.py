"""Open-loop HanWAM action-sensitivity diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from control.MiniController.config_schema import CONFIG_PATH, checkpoint_path, load_config

from .dataloader import Normalizer, PHYSICAL_COLS, action_bounds, load_all_runs, sequence_arrays
from .model import build_wam_model
from .utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prober rollouts under fixed action profiles")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
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
    return {
        "off": np.asarray([0.0, base[1], lo[2]], dtype=np.float32),
        "base": base,
        "high": np.asarray([hi[0], base[1], hi[2]], dtype=np.float32),
        "high_freq_low_fan": np.asarray([hi[0], base[1], lo[2]], dtype=np.float32),
        "low_freq_high_fan": np.asarray([max(15.0, base[0] * 0.5), base[1], hi[2]], dtype=np.float32),
    }


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
    horizon_steps = int(args.horizon_steps or (checkpoint.get("planner_config") or {}).get("horizon_steps", 24))
    runs = load_all_runs(mode=mode, config=config)
    obs_history, _, _, _, keys = sequence_arrays(
        runs,
        args.split,
        horizon_steps=horizon_steps,
        config=config,
        limit=int(args.limit),
        seed=int(args.seed),
    )
    obs_n = torch.as_tensor(obs_norm.encode(obs_history), dtype=torch.float32, device=device)
    latent = model.encode(obs_n)

    bounds = action_bounds(runs, config=config)
    profiles = _profile_actions(config, bounds, mode)
    phys_idx = {name: idx for idx, name in enumerate(PHYSICAL_COLS)}
    rows: list[dict] = []
    for name, action in profiles.items():
        actions = np.tile(action.reshape(1, 1, -1), (len(obs_history), horizon_steps, 1)).astype(np.float32)
        actions_n = torch.as_tensor(action_norm.encode(actions.reshape(-1, actions.shape[-1])).reshape(actions.shape), dtype=torch.float32, device=device)
        _, physical_n = model.rollout_from_latent(latent, actions_n)
        physical = physical_norm.decode(physical_n.detach().cpu().numpy().reshape(-1, len(PHYSICAL_COLS))).reshape(
            len(obs_history),
            horizon_steps,
            len(PHYSICAL_COLS),
        )
        temp_delta = physical[:, :, phys_idx["T_in_delta"]]
        energy_delta = np.clip(physical[:, :, phys_idx["electric_kwh_delta"]], 0.0, None)
        rows.append(
            {
                "profile": name,
                "windows": int(len(keys)),
                "horizon_steps": int(horizon_steps),
                "freq_target": float(action[0]),
                "eev": float(action[1]),
                "fan_out": float(action[2]),
                "mean_final_cum_T_in_delta_c": float(temp_delta.sum(axis=1).mean()),
                "mean_cum_electric_kwh": float(energy_delta.sum(axis=1).mean()),
                "mean_step_T_in_delta_c": float(temp_delta.mean()),
                "mean_step_electric_kwh": float(energy_delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    result = run_diagnostic(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
