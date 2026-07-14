"""Strict two-stage HanWAM E007 training.

Stage 1 trains only the latent block world model with latent MSE + SIGReg.
Stage 2 freezes that world model and trains only the physical prober.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from control.MiniController.config_schema import CONFIG_PATH, checkpoint_path, expand_modes, load_config, output_root
from control.MiniController.tracking import SwanLabTracker

from .dataloader import (
    PHYSICAL_COLS,
    action_bounds,
    block_sequence_arrays,
    columns_from_config,
    describe_runs,
    fit_normalizer,
    load_all_runs,
)
from .model import HanWAMControllerModel, HanWAMWorldModel, build_controller_model_from_world_model, build_world_model
from .utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict two-stage HanWAM")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit-transitions", type=int, default=None)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--swanlab", action="store_true", help="Enable SwanLab for this run.")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab for this run.")
    return parser.parse_args()


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.mode is not None:
        config["train"]["mode"] = args.mode
    if args.stage is not None:
        config["train"]["stage"] = args.stage
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.limit_transitions is not None:
        config["train"]["limit_transitions"] = args.limit_transitions
    if args.stage1_checkpoint is not None:
        config.setdefault("train", {}).setdefault("stage2", {})["stage1_checkpoint"] = str(args.stage1_checkpoint)
    if args.swanlab or args.no_swanlab:
        config.setdefault("experiment", {}).setdefault("tracking", {}).setdefault("swanlab", {})["enabled"] = bool(args.swanlab)
    return config


def _stage_cfg(config: dict, stage: str) -> dict:
    train_cfg = dict(config["train"])
    stage_cfg = dict(train_cfg.get(stage) or {})
    merged = {
        "epochs": int(stage_cfg.get("epochs", train_cfg.get("epochs", 50))),
        "batch_size": int(stage_cfg.get("batch_size", train_cfg.get("batch_size", 4096))),
        "lr": float(stage_cfg.get("lr", train_cfg.get("lr", 1e-3))),
        "weight_decay": float(stage_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-4))),
    }
    if "loss" in stage_cfg:
        merged["loss"] = dict(stage_cfg["loss"])
    if "checkpoint_epochs" in stage_cfg:
        merged["checkpoint_epochs"] = list(stage_cfg["checkpoint_epochs"] or [])
    if "checkpoint_every_epochs" in stage_cfg:
        merged["checkpoint_every_epochs"] = int(stage_cfg["checkpoint_every_epochs"] or 0)
    return merged


def _block_params(config: dict) -> tuple[int, int, int]:
    train_cfg = config.get("train") or {}
    frames_per_block = int(train_cfg.get("frames_per_block", 12))
    history_blocks = int(train_cfg.get("history_blocks", 3))
    future_blocks = int(train_cfg.get("future_blocks", 5))
    return frames_per_block, history_blocks, future_blocks


def _stage1_checkpoint_path(config: dict, mode: int) -> Path:
    final = checkpoint_path(config, mode)
    return final.with_name(f"{final.stem}_stage1{final.suffix}")


def _epoch_checkpoint_path(config: dict, mode: int, stage: str, epoch: int) -> Path:
    final = checkpoint_path(config, mode)
    return final.with_name(f"{final.stem}_{stage}_epoch{int(epoch):04d}{final.suffix}")


def _checkpoint_epochs(stage_cfg: dict, total_epochs: int) -> set[int]:
    selected = {int(epoch) for epoch in (stage_cfg.get("checkpoint_epochs") or [])}
    every = int(stage_cfg.get("checkpoint_every_epochs", 0) or 0)
    if every > 0:
        selected.update(range(every, int(total_epochs) + 1, every))
    return {epoch for epoch in selected if 1 <= epoch <= int(total_epochs)}


def _planner_config(config: dict, mode: int) -> dict:
    frames_per_block, history_blocks, future_blocks = _block_params(config)
    planner = dict((config.get("method") or {}).get("planner") or {})
    planner.setdefault("algorithm", "mppi")
    planner.setdefault("objective", "hanwam_e007")
    planner.setdefault("horizon_steps", frames_per_block * future_blocks)
    planner.setdefault("frames_per_block", frames_per_block)
    planner.setdefault("future_blocks", future_blocks)
    planner.setdefault("chunk_steps", frames_per_block)
    planner.setdefault("control_interval_steps", 6)
    planner.setdefault("num_samples", 512)
    planner.setdefault("num_iterations", 3)
    planner.setdefault("temperature", 1.0)
    planner.setdefault("step_seconds", int(config["data"]["sampling"]["step_seconds"]))
    planner.setdefault("reference_schedule", "deadline_linear")
    planner.setdefault("comfort_band_reference", "schedule_then_target")
    planner.setdefault("comfort_band_c", 0.5)
    planner.setdefault("target_band_margin_seconds", 600.0)
    planner.setdefault("deadline_comfort_band_c", 0.5)
    planner.setdefault("target_margin_comfort_band_c", 0.5)
    planner.setdefault("compressor_on_threshold_hz", 15.0)
    planner.setdefault("snap_deadband_freq", True)
    planner.setdefault(
        "cost_weights",
        {
            "comfort_band_violation": 4.0,
            "target_margin_band_violation": 12.0,
            "deadline_band_violation": 40.0,
            "energy": 2.0,
            "action_smooth": 0.20,
        },
    )
    planner.setdefault("history_blocks", history_blocks)
    planner["mode"] = int(mode)
    return planner


def _tracker(config: dict, mode: int, stage: str) -> SwanLabTracker:
    tracker_cfg = ((config.get("experiment") or {}).get("tracking") or {}).get("swanlab") or {}
    experiment_name = tracker_cfg.get("experiment") or f"{config['experiment']['name']}_mode{mode}_{stage}"
    return SwanLabTracker(
        enabled=bool(tracker_cfg.get("enabled", False)),
        project=str(tracker_cfg.get("project", "haier-jk-control")),
        experiment_name=experiment_name,
        config={"stage": stage, "mode": mode, **config},
    )


def _prepare_data(config: dict, mode: int):
    train_cfg = config["train"]
    seed = int(config["experiment"]["seed"])
    frames_per_block, history_blocks, future_blocks = _block_params(config)
    runs = load_all_runs(mode=mode, config=config)
    limit = int(train_cfg.get("limit_transitions", 0))
    val_limit = int(train_cfg.get("val_limit_transitions", min(limit, 8192) if limit > 0 else 0))
    train_arrays = block_sequence_arrays(runs, "train", config=config, limit=limit, seed=seed)
    val_arrays = block_sequence_arrays(runs, "val", config=config, limit=val_limit, seed=seed + 1)
    train_obs_history, train_act_history, train_future_act, train_target_obs, train_physical, _ = train_arrays
    obs_norm = fit_normalizer(
        np.concatenate(
            [
                train_obs_history.reshape(-1, train_obs_history.shape[-1]),
                train_target_obs.reshape(-1, train_target_obs.shape[-1]),
            ],
            axis=0,
        )
    )
    action_norm = fit_normalizer(
        np.concatenate(
            [
                train_act_history.reshape(-1, train_act_history.shape[-1]),
                train_future_act.reshape(-1, train_future_act.shape[-1]),
            ],
            axis=0,
        )
    )
    physical_norm = fit_normalizer(train_physical.reshape(-1, train_physical.shape[-1]))
    return {
        "runs": runs,
        "train": train_arrays[:5],
        "val": val_arrays[:5],
        "train_keys": train_arrays[5],
        "val_keys": val_arrays[5],
        "obs_norm": obs_norm,
        "action_norm": action_norm,
        "physical_norm": physical_norm,
        "frames_per_block": frames_per_block,
        "history_blocks": history_blocks,
        "future_blocks": future_blocks,
        "history_steps": frames_per_block * history_blocks,
        "horizon_steps": frames_per_block * future_blocks,
    }


def _loader(arrays, obs_norm, action_norm, physical_norm, batch_size: int) -> DataLoader:
    obs_history, act_history, future_act, target_obs, physical = arrays
    return DataLoader(
        TensorDataset(
            torch.as_tensor(obs_norm.encode(obs_history), dtype=torch.float32),
            torch.as_tensor(action_norm.encode(act_history), dtype=torch.float32),
            torch.as_tensor(action_norm.encode(future_act), dtype=torch.float32),
            torch.as_tensor(obs_norm.encode(target_obs), dtype=torch.float32),
            torch.as_tensor(physical_norm.encode(physical), dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


@torch.no_grad()
def _eval_stage1(model: HanWAMWorldModel, loader: DataLoader, device: torch.device, loss_cfg: dict) -> dict:
    model.eval()
    rows = []
    for obs_b, act_hist_b, future_act_b, target_obs_b, _ in loader:
        obs_b = obs_b.to(device)
        act_hist_b = act_hist_b.to(device)
        future_act_b = future_act_b.to(device)
        target_obs_b = target_obs_b.to(device)
        breakdown = model.latent_sequence_loss(
            obs_b,
            act_hist_b,
            future_act_b,
            target_obs_b,
            **loss_cfg,
        )
        rows.append([float(breakdown.total.cpu()), float(breakdown.latent.cpu()), float(breakdown.regularizer.cpu())])
    return {
        "val_stage1_loss": float(np.mean([r[0] for r in rows])),
        "val_latent_loss": float(np.mean([r[1] for r in rows])),
        "val_sigreg_loss": float(np.mean([r[2] for r in rows])),
    }


@torch.no_grad()
def _eval_stage2(
    model: HanWAMControllerModel,
    loader: DataLoader,
    physical_norm,
    device: torch.device,
    physical_weights: torch.Tensor | None,
) -> dict:
    model.eval()
    pred_chunks = []
    target_chunks = []
    losses = []
    horizon = None
    for obs_b, act_hist_b, future_act_b, _, physical_b in loader:
        obs_b = obs_b.to(device)
        act_hist_b = act_hist_b.to(device)
        future_act_b = future_act_b.to(device)
        physical_b = physical_b.to(device)
        loss, pred_n = model.prober_sequence_loss(
            obs_b,
            act_hist_b,
            future_act_b,
            physical_b,
            physical_weights=physical_weights,
        )
        losses.append(float(loss.cpu()))
        pred_chunks.append(pred_n.cpu().numpy())
        target_chunks.append(physical_b.cpu().numpy())
        horizon = physical_b.shape[1]
    if horizon is None:
        raise ValueError("empty loader")
    pred = physical_norm.decode(np.concatenate(pred_chunks).reshape(-1, len(PHYSICAL_COLS)))
    target = physical_norm.decode(np.concatenate(target_chunks).reshape(-1, len(PHYSICAL_COLS)))
    err = pred - target
    idx = {name: i for i, name in enumerate(PHYSICAL_COLS)}
    pred_delta = pred[:, idx["T_in_delta"]].reshape(-1, horizon)
    target_delta = target[:, idx["T_in_delta"]].reshape(-1, horizon)
    cum_err = np.cumsum(pred_delta, axis=1) - np.cumsum(target_delta, axis=1)
    return {
        "val_stage2_loss": float(np.mean(losses)),
        "physical_rmse": float(np.sqrt(np.mean(err * err))),
        "T_in_delta_mae_c": float(np.mean(np.abs(err[:, idx["T_in_delta"]]))),
        "electric_delta_mae_kwh": float(np.mean(np.abs(err[:, idx["electric_kwh_delta"]]))),
        "T_delta_cum_mae_c": float(np.mean(np.abs(cum_err))),
        "T_delta_cum_final_mae_c": float(np.mean(np.abs(cum_err[:, -1]))),
    }


def _physical_weights(config: dict, device: torch.device) -> torch.Tensor | None:
    weights_cfg = ((config["train"].get("stage2") or {}).get("loss") or {}).get("physical_weights")
    if not weights_cfg:
        return None
    values = [float(weights_cfg.get(name, 1.0)) for name in PHYSICAL_COLS]
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _stage2_cumulative_weights(config: dict) -> dict[str, float]:
    loss_cfg = ((config["train"].get("stage2") or {}).get("loss") or {})
    return {
        "T_in_delta": float(loss_cfg.get("cumulative_T_in_delta_weight", 0.0)),
        "electric_kwh_delta": float(loss_cfg.get("cumulative_electric_kwh_delta_weight", 0.0)),
    }


def _stage2_loss(
    model: HanWAMControllerModel,
    obs_b: torch.Tensor,
    act_hist_b: torch.Tensor,
    future_act_b: torch.Tensor,
    physical_b: torch.Tensor,
    physical_weights: torch.Tensor | None,
    physical_std: torch.Tensor,
    cumulative_weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    base_loss, pred_n = model.prober_sequence_loss(
        obs_b,
        act_hist_b,
        future_act_b,
        physical_b,
        physical_weights=physical_weights,
    )
    err_phys = (pred_n - physical_b) * physical_std.view(1, 1, -1)
    extra = pred_n.new_tensor(0.0)
    parts = {"base_loss": float(base_loss.detach().cpu())}
    idx = {name: i for i, name in enumerate(PHYSICAL_COLS)}
    weight = float(cumulative_weights.get("T_in_delta", 0.0))
    if weight > 0.0 and "T_in_delta" in idx:
        cum_t = torch.cumsum(err_phys[:, :, idx["T_in_delta"]], dim=1).pow(2).mean()
        extra = extra + weight * cum_t
        parts["cum_T_in_delta_loss"] = float(cum_t.detach().cpu())
    weight = float(cumulative_weights.get("electric_kwh_delta", 0.0))
    if weight > 0.0 and "electric_kwh_delta" in idx:
        cum_e = torch.cumsum(err_phys[:, :, idx["electric_kwh_delta"]], dim=1).pow(2).mean()
        extra = extra + weight * cum_e
        parts["cum_electric_kwh_delta_loss"] = float(cum_e.detach().cpu())
    total = base_loss + extra
    parts["total_loss"] = float(total.detach().cpu())
    return total, pred_n, parts


def _world_model_config(config: dict) -> dict:
    model_cfg = dict(((config.get("method") or {}).get("model") or {}))
    columns = columns_from_config(config)
    frames_per_block, history_blocks, future_blocks = _block_params(config)
    latent_dim = int(model_cfg.get("latent_dim", 64))
    return {
        "class_name": "HanWAMWorldModel",
        "obs_dim": len(columns["observation"]),
        "action_dim": len(columns["target_action"]),
        "latent_dim": latent_dim,
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "action_latent_dim": int(model_cfg.get("action_latent_dim", max(16, latent_dim // 2))),
        "tcn_layers": int(model_cfg.get("tcn_layers", 2)),
        "frames_per_block": frames_per_block,
        "history_blocks": history_blocks,
        "future_blocks": future_blocks,
    }


def _controller_model_config(config: dict, data: dict | None = None) -> dict:
    model_cfg = dict(((config.get("method") or {}).get("model") or {}))
    class_name = str(model_cfg.get("class_name", "HanWAMControllerModel"))
    payload = {
        "class_name": class_name,
        "physical_dim": len(PHYSICAL_COLS),
        "prober_hidden_dim": int(model_cfg.get("prober_hidden_dim", model_cfg.get("hidden_dim", 128))),
        "prober_action_scale": float(model_cfg.get("prober_action_scale", 1.0)),
        "world_model_config": _world_model_config(config),
    }
    if class_name in {"HanWAMPhysicsGuidedControllerModel", "HanWAMHardMechanismControllerModel"}:
        common_physics_keys = {
            "freq_on_threshold_hz",
            "freq_max_hz",
            "fan_max",
            "eev_min",
            "eev_max",
            "eev_width_min",
            "eev_width_max",
            "eev_effect_floor",
            "compressor_energy_scale",
            "fan_energy_scale",
            "energy_model",
            "energy_step_seconds",
            "energy_power_intercept_w",
            "energy_power_linear_w_per_hz",
            "energy_power_quadratic_w_per_hz2",
            "compressor_transition_hz",
            "fan_effect_floor",
            "freq_effect_floor",
            "mechanism_context_blend",
            "temperature_mechanism",
            "baseline_drift_scale_c",
            "cooling_history_seconds",
            "cooling_freq_exponent",
            "cooling_fan_exponent",
            "cooling_fan_reference",
            "cooling_eev_reference",
            "cooling_eev_range",
            "cooling_eev_gain_scale",
            "cooling_eev_effect_min",
            "cooling_eev_effect_max",
            "cooling_lag_enabled",
            "cooling_lag_alpha_min",
            "cooling_lag_alpha_max",
            "energy_eev_correction_mode",
            "energy_eev_correction_scale_w",
            "energy_eev_correction_min_w",
            "energy_eev_correction_max_w",
            "energy_eev_anchor",
            "energy_eev_range",
        }
        soft_physics_keys = {
            "passive_delta_scale",
            "cooling_delta_scale",
            "residual_delta_scale",
            "residual_energy_scale",
            "passive_nonnegative",
        }
        hard_physics_keys = {
            "ua_delta_scale",
            "internal_delta_scale",
            "cooling_delta_scale",
            "cop_min",
            "cop_max",
            "t_in_obs_index",
            "t_out_obs_index",
        }
        physics_keys = set(common_physics_keys)
        if class_name == "HanWAMPhysicsGuidedControllerModel":
            physics_keys.update(soft_physics_keys)
        if class_name == "HanWAMHardMechanismControllerModel":
            physics_keys.update(hard_physics_keys)
        for key in physics_keys:
            if key in model_cfg:
                payload[key] = model_cfg[key]
        if data is not None:
            payload["action_mean"] = data["action_norm"].mean.tolist()
            payload["action_std"] = data["action_norm"].std.tolist()
            payload["physical_mean"] = data["physical_norm"].mean.tolist()
            payload["physical_std"] = data["physical_norm"].std.tolist()
            payload["obs_mean"] = data["obs_norm"].mean.tolist()
            payload["obs_std"] = data["obs_norm"].std.tolist()
            columns = columns_from_config(config)["observation"]
            payload["t_in_obs_index"] = int(columns.index("T_in"))
            payload["t_out_obs_index"] = int(columns.index("T_out"))
    return payload


def _base_payload(config: dict, mode: int, data: dict, result_dir: Path) -> dict:
    runs = data["runs"]
    return {
        "obs_cols": columns_from_config(config)["observation"],
        "target_action_cols": columns_from_config(config)["target_action"],
        "physical_cols": PHYSICAL_COLS,
        "mode": mode,
        "obs_norm": data["obs_norm"].to_dict(),
        "target_action_norm": data["action_norm"].to_dict(),
        "physical_norm": data["physical_norm"].to_dict(),
        "target_action_bounds": action_bounds(runs, config=config),
        "planner_config": _planner_config(config, mode),
        "frames_per_block": int(data["frames_per_block"]),
        "history_blocks": int(data["history_blocks"]),
        "future_blocks": int(data["future_blocks"]),
        "history_steps": int(data["history_steps"]),
        "horizon_steps": int(data["horizon_steps"]),
        "resolved_config": config,
        "result_dir": str(result_dir),
    }


def _save_checkpoint(path: Path, payload: dict, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = None
    payload["model"] = model.cpu().state_dict()
    torch.save(payload, path)
    if original_device is not None:
        model.to(original_device)


def train_stage1(
    config: dict,
    mode: int,
    data: dict,
    device: torch.device,
    result_dir: Path,
) -> tuple[HanWAMWorldModel, Path, list[dict]]:
    stage_cfg = _stage_cfg(config, "stage1")
    loss_cfg = dict(stage_cfg.get("loss") or {})
    loss_cfg = {
        "horizon_gamma": float(loss_cfg.get("horizon_gamma", 0.98)),
        "latent_weight": float(loss_cfg.get("latent_weight", 1.0)),
        "sigreg_weight": float(loss_cfg.get("sigreg_weight", loss_cfg.get("vicreg_weight", 0.05))),
        "variance_target": float(loss_cfg.get("variance_target", 1.0)),
        "sigreg_num_projections": int(loss_cfg.get("sigreg_num_projections", 64)),
        "sigreg_mean_weight": float(loss_cfg.get("sigreg_mean_weight", 1.0)),
    }
    model = build_world_model(_world_model_config(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg["lr"]),
        weight_decay=float(stage_cfg["weight_decay"]),
    )
    train_loader = _loader(data["train"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    val_loader = _loader(data["val"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    tracker = _tracker(config, mode, "stage1")
    history = []
    total_epochs = int(stage_cfg["epochs"])
    checkpoint_epochs = _checkpoint_epochs(stage_cfg, total_epochs)
    for epoch in range(1, total_epochs + 1):
        model.train()
        rows = []
        for obs_b, act_hist_b, future_act_b, target_obs_b, _ in train_loader:
            obs_b = obs_b.to(device)
            act_hist_b = act_hist_b.to(device)
            future_act_b = future_act_b.to(device)
            target_obs_b = target_obs_b.to(device)
            breakdown = model.latent_sequence_loss(
                obs_b,
                act_hist_b,
                future_act_b,
                target_obs_b,
                **loss_cfg,
            )
            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            rows.append([float(breakdown.total.detach().cpu()), float(breakdown.latent.detach().cpu()), float(breakdown.regularizer.detach().cpu())])
        row = {
            "stage": "stage1",
            "epoch": epoch,
            "loss": float(np.mean([r[0] for r in rows])),
            "latent_loss": float(np.mean([r[1] for r in rows])),
            "sigreg_loss": float(np.mean([r[2] for r in rows])),
        }
        if epoch == total_epochs or epoch % max(1, total_epochs // 5) == 0:
            row.update(_eval_stage1(model, val_loader, device, loss_cfg))
        history.append(row)
        tracker.log({f"stage1/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in checkpoint_epochs:
            payload = _base_payload(config, mode, data, result_dir)
            payload.update(
                {
                    "model_config": _world_model_config(config),
                    "training_phase": "stage1_world_model",
                    "stage1_config": stage_cfg,
                    "history": history,
                    "epoch": epoch,
                }
            )
            _save_checkpoint(_epoch_checkpoint_path(config, mode, "stage1", epoch), payload, model)
    tracker.finish()
    pd.DataFrame(history).to_csv(result_dir / "hanwam_stage1_history.csv", index=False, encoding="utf-8-sig")
    payload = _base_payload(config, mode, data, result_dir)
    payload.update(
        {
            "model_config": _world_model_config(config),
            "training_phase": "stage1_world_model",
            "stage1_config": stage_cfg,
            "history": history,
        }
    )
    stage1_path = _stage1_checkpoint_path(config, mode)
    _save_checkpoint(stage1_path, payload, model)
    model.to(device)
    return model, stage1_path, history


def _load_stage1_world_model(path: Path) -> HanWAMWorldModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_world_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model


def train_stage2(
    config: dict,
    mode: int,
    data: dict,
    device: torch.device,
    result_dir: Path,
    world_model: HanWAMWorldModel | None = None,
) -> tuple[HanWAMControllerModel, Path, list[dict]]:
    stage_cfg = _stage_cfg(config, "stage2")
    stage1_path = Path((config["train"].get("stage2") or {}).get("stage1_checkpoint") or _stage1_checkpoint_path(config, mode))
    if world_model is None:
        world_model = _load_stage1_world_model(stage1_path)
    model_cfg = _controller_model_config(config, data)
    model = build_controller_model_from_world_model(world_model, model_cfg).to(device)
    model.freeze_world_model()
    optimizer = torch.optim.AdamW(
        [param for param in model.prober_parameters() if param.requires_grad],
        lr=float(stage_cfg["lr"]),
        weight_decay=float(stage_cfg["weight_decay"]),
    )
    train_loader = _loader(data["train"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    val_loader = _loader(data["val"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    physical_weights = _physical_weights(config, device)
    cumulative_weights = _stage2_cumulative_weights(config)
    physical_std = torch.as_tensor(data["physical_norm"].std, dtype=torch.float32, device=device)
    tracker = _tracker(config, mode, "stage2")
    history = []
    total_epochs = int(stage_cfg["epochs"])
    checkpoint_epochs = _checkpoint_epochs(stage_cfg, total_epochs)
    for epoch in range(1, total_epochs + 1):
        model.train()
        rows = []
        part_rows: dict[str, list[float]] = {}
        for obs_b, act_hist_b, future_act_b, _, physical_b in train_loader:
            obs_b = obs_b.to(device)
            act_hist_b = act_hist_b.to(device)
            future_act_b = future_act_b.to(device)
            physical_b = physical_b.to(device)
            loss, _, parts = _stage2_loss(
                model,
                obs_b,
                act_hist_b,
                future_act_b,
                physical_b,
                physical_weights,
                physical_std,
                cumulative_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prober.parameters(), 5.0)
            optimizer.step()
            rows.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                part_rows.setdefault(key, []).append(float(value))
        row = {
            "stage": "stage2",
            "epoch": epoch,
            "loss": float(np.mean(rows)),
        }
        for key, values in part_rows.items():
            row[key] = float(np.mean(values))
        if epoch == total_epochs or epoch % max(1, total_epochs // 5) == 0:
            row.update(_eval_stage2(model, val_loader, data["physical_norm"], device, physical_weights))
        history.append(row)
        tracker.log({f"stage2/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in checkpoint_epochs:
            payload = _base_payload(config, mode, data, result_dir)
            payload.update(
                {
                    "model_config": model_cfg,
                    "training_phase": "stage2_controller",
                    "stage1_checkpoint": str(stage1_path),
                    "stage2_config": stage_cfg,
                    "history": history,
                    "epoch": epoch,
                }
            )
            _save_checkpoint(_epoch_checkpoint_path(config, mode, "stage2", epoch), payload, model)
    tracker.finish()
    pd.DataFrame(history).to_csv(result_dir / "hanwam_stage2_history.csv", index=False, encoding="utf-8-sig")
    payload = _base_payload(config, mode, data, result_dir)
    payload.update(
        {
            "model_config": model_cfg,
            "training_phase": "stage2_controller",
            "stage1_checkpoint": str(stage1_path),
            "stage2_config": stage_cfg,
            "history": history,
        }
    )
    final_path = checkpoint_path(config, mode)
    _save_checkpoint(final_path, payload, model)
    model.to(device)
    return model, final_path, history


def train_one_mode(config: dict, mode: int) -> dict:
    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device(config["train"]["device"])
    result_dir = output_root(config) / config["experiment"]["name"] / f"train_mode{mode}"
    result_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(config, mode)
    describe_runs(data["runs"]).to_csv(result_dir / "hanwam_data_audit.csv", index=False, encoding="utf-8-sig")
    stage = str(config["train"].get("stage", "both"))
    result: dict = {}
    world_model = None
    if stage in {"stage1", "both"}:
        world_model, stage1_path, stage1_history = train_stage1(config, mode, data, device, result_dir)
        result["stage1_checkpoint"] = str(stage1_path)
        result["stage1_history"] = stage1_history
    if stage in {"stage2", "both"}:
        model, final_path, stage2_history = train_stage2(config, mode, data, device, result_dir, world_model=world_model)
        result["checkpoint"] = str(final_path)
        result["stage2_history"] = stage2_history
    return result


def main(config: dict | None = None) -> dict:
    if config is None:
        args = parse_args()
        config = _apply_cli_overrides(load_config(args.config), args)
    results = {}
    for mode in expand_modes(config["train"]["mode"]):
        results[f"mode{mode}"] = train_one_mode(config, int(mode))
    return results


if __name__ == "__main__":
    main()
