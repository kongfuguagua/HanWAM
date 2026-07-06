"""Strict two-stage HanWAM training.

Stage 1 trains only latent dynamics with latent MSE + SIGReg.
Stage 2 freezes the latent world model and trains only the physical prober.
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
    columns_from_config,
    describe_runs,
    fit_normalizer,
    load_all_runs,
    sequence_arrays,
)
from .model import HanWAM, build_wam_model
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
    planner = dict((config.get("method") or {}).get("planner") or {})
    planner.setdefault("objective", "hanwam_simple")
    planner.setdefault("horizon_steps", 24)
    planner.setdefault("chunk_steps", 6)
    planner.setdefault("num_samples", 128)
    planner.setdefault("num_iterations", 3)
    planner.setdefault("elite_ratio", 0.1)
    planner.setdefault("step_seconds", int(config["data"]["sampling"]["step_seconds"]))
    planner.setdefault("reference_schedule", "deadline_linear")
    planner.setdefault("compressor_on_threshold_hz", 15.0)
    planner.setdefault("snap_deadband_freq", True)
    planner.setdefault("cost_weights", {"tracking": 4.0, "energy": 2.0, "action_smooth": 0.05})
    planner.setdefault("history_steps", int((config.get("train") or {}).get("history_steps", 1)))
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
    horizon_steps = int(train_cfg.get("horizon_steps", 60))
    seed = int(config["experiment"]["seed"])
    runs = load_all_runs(mode=mode, config=config)
    limit = int(train_cfg.get("limit_transitions", 0))
    val_limit = int(train_cfg.get("val_limit_transitions", min(limit, 8192) if limit > 0 else 0))
    train_obs, train_actions, train_future, train_physical, _ = sequence_arrays(
        runs, "train", horizon_steps=horizon_steps, config=config, limit=limit, seed=seed
    )
    val_obs, val_actions, val_future, val_physical, _ = sequence_arrays(
        runs, "val", horizon_steps=horizon_steps, config=config, limit=val_limit, seed=seed + 1
    )
    obs_norm = fit_normalizer(
        np.concatenate(
            [
                train_obs.reshape(-1, train_obs.shape[-1]),
                train_future.reshape(-1, train_future.shape[-1]),
            ],
            axis=0,
        )
    )
    action_norm = fit_normalizer(
        train_actions.reshape(-1, train_actions.shape[-1])
    )
    physical_norm = fit_normalizer(train_physical.reshape(-1, train_physical.shape[-1]))
    return {
        "runs": runs,
        "train": (train_obs, train_actions, train_future, train_physical),
        "val": (val_obs, val_actions, val_future, val_physical),
        "obs_norm": obs_norm,
        "action_norm": action_norm,
        "physical_norm": physical_norm,
        "horizon_steps": horizon_steps,
        "history_steps": int(train_cfg.get("history_steps", 1)),
    }


def _loader(arrays, obs_norm, action_norm, physical_norm, batch_size: int) -> DataLoader:
    obs, actions, future, physical = arrays
    return DataLoader(
        TensorDataset(
            torch.as_tensor(obs_norm.encode(obs), dtype=torch.float32),
            torch.as_tensor(action_norm.encode(actions.reshape(-1, actions.shape[-1])).reshape(actions.shape), dtype=torch.float32),
            torch.as_tensor(obs_norm.encode(future.reshape(-1, future.shape[-1])).reshape(future.shape), dtype=torch.float32),
            torch.as_tensor(physical_norm.encode(physical.reshape(-1, physical.shape[-1])).reshape(physical.shape), dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


@torch.no_grad()
def _eval_stage1(model, loader: DataLoader, device: torch.device, loss_cfg: dict) -> dict:
    model.eval()
    rows = []
    for obs_b, actions_b, future_b, _ in loader:
        obs_b = obs_b.to(device)
        actions_b = actions_b.to(device)
        future_b = future_b.to(device)
        breakdown = model.latent_sequence_loss(
            obs_b,
            actions_b,
            future_b,
            **loss_cfg,
        )
        rows.append([float(breakdown.total.cpu()), float(breakdown.latent.cpu()), float(breakdown.regularizer.cpu())])
    return {
        "val_stage1_loss": float(np.mean([r[0] for r in rows])),
        "val_latent_loss": float(np.mean([r[1] for r in rows])),
        "val_sigreg_loss": float(np.mean([r[2] for r in rows])),
    }


@torch.no_grad()
def _eval_stage2(model, loader: DataLoader, physical_norm, device: torch.device, physical_weights: torch.Tensor | None) -> dict:
    model.eval()
    pred_chunks = []
    target_chunks = []
    losses = []
    horizon = None
    for obs_b, actions_b, _, physical_b in loader:
        obs_b = obs_b.to(device)
        actions_b = actions_b.to(device)
        physical_b = physical_b.to(device)
        loss, pred_n = model.prober_sequence_loss(
            obs_b,
            actions_b,
            physical_b,
            physical_weights=physical_weights,
        )
        losses.append(float(loss.cpu()))
        pred_chunks.append(pred_n.cpu().numpy())
        target_chunks.append(physical_b.cpu().numpy())
        horizon = physical_b.shape[1]
    pred = physical_norm.decode(np.concatenate(pred_chunks).reshape(-1, len(PHYSICAL_COLS)))
    target = physical_norm.decode(np.concatenate(target_chunks).reshape(-1, len(PHYSICAL_COLS)))
    err = pred - target
    idx = {name: i for i, name in enumerate(PHYSICAL_COLS)}
    if horizon is None:
        raise ValueError("empty loader")
    pred_delta = pred[:, idx["T_in_delta"]].reshape(-1, horizon)
    target_delta = target[:, idx["T_in_delta"]].reshape(-1, horizon)
    cum_err = np.cumsum(pred_delta, axis=1) - np.cumsum(target_delta, axis=1)
    return {
        "val_stage2_loss": float(np.mean(losses)),
        "physical_rmse": float(np.sqrt(np.mean(err * err))),
        "T_in_mae_c": float(np.mean(np.abs(err[:, idx["T_in"]]))),
        "freq_mae_hz": float(np.mean(np.abs(err[:, idx["freq"]]))),
        "electric_delta_mae_kwh": float(np.mean(np.abs(err[:, idx["electric_kwh_delta"]]))),
        "T_delta_cum_mae_c": float(np.mean(np.abs(cum_err))),
        "T_delta_cum_final_mae_c": float(np.mean(np.abs(cum_err[:, -1]))),
    }


def _physical_weights(config: dict, physical_norm, device: torch.device) -> torch.Tensor | None:
    weights_cfg = ((config["train"].get("stage2") or {}).get("loss") or {}).get("physical_weights")
    if not weights_cfg:
        return None
    values = [float(weights_cfg.get(name, 1.0)) for name in PHYSICAL_COLS]
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _base_payload(config: dict, mode: int, model, data: dict, result_dir: Path) -> dict:
    runs = data["runs"]
    return {
        "model_config": {
            "class_name": "HanWAM",
            **{
                key: value
                for key, value in model.__dict__.items()
                if key in set()
            },
        },
        "obs_cols": columns_from_config(config)["observation"],
        "target_action_cols": columns_from_config(config)["target_action"],
        "physical_cols": PHYSICAL_COLS,
        "mode": mode,
        "obs_norm": data["obs_norm"].to_dict(),
        "target_action_norm": data["action_norm"].to_dict(),
        "physical_norm": data["physical_norm"].to_dict(),
        "target_action_bounds": action_bounds(runs, config=config),
        "planner_config": _planner_config(config, mode),
        "history_steps": int(data.get("history_steps", (config.get("train") or {}).get("history_steps", 1))),
        "resolved_config": config,
        "result_dir": str(result_dir),
    }


def _model_config(config: dict) -> dict:
    method_cfg = config["method"]
    model_cfg = dict(method_cfg.get("model") or {})
    model_cfg.setdefault("class_name", "HanWAM")
    columns = columns_from_config(config)
    return {
        "class_name": model_cfg.get("class_name", "HanWAM"),
        "obs_dim": len(columns["observation"]),
        "action_dim": len(columns["target_action"]),
        "physical_dim": len(PHYSICAL_COLS),
        "latent_dim": int(model_cfg.get("latent_dim", 64)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "action_latent_dim": int(model_cfg.get("action_latent_dim", max(16, int(model_cfg.get("latent_dim", 64)) // 2))),
        "tcn_layers": int(model_cfg.get("tcn_layers", 2)),
    }


def _save_checkpoint(path: Path, payload: dict, model) -> None:
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


def train_stage1(config: dict, mode: int, data: dict, device: torch.device, result_dir: Path) -> tuple[HanWAM, Path, list[dict]]:
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
    model = build_wam_model(_model_config(config)).to(device)
    if not isinstance(model, HanWAM):
        raise TypeError("HanWAM train requires method.model.class_name=HanWAM")
    model.freeze_prober()
    optimizer = torch.optim.AdamW(
        [param for param in model.world_model_parameters() if param.requires_grad],
        lr=float(stage_cfg["lr"]),
        weight_decay=float(stage_cfg["weight_decay"]),
    )
    train_loader = _loader(data["train"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    val_loader = _loader(data["val"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    tracker = _tracker(config, mode, "stage1")
    history = []
    total_epochs = int(stage_cfg["epochs"])
    checkpoint_epochs = _checkpoint_epochs(stage_cfg, total_epochs)
    for epoch in range(1, int(stage_cfg["epochs"]) + 1):
        model.train()
        rows = []
        for obs_b, actions_b, future_b, _ in train_loader:
            obs_b = obs_b.to(device)
            actions_b = actions_b.to(device)
            future_b = future_b.to(device)
            breakdown = model.latent_sequence_loss(
                obs_b,
                actions_b,
                future_b,
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
        if epoch == int(stage_cfg["epochs"]) or epoch % max(1, int(stage_cfg["epochs"]) // 5) == 0:
            row.update(_eval_stage1(model, val_loader, device, loss_cfg))
        history.append(row)
        tracker.log({f"stage1/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in checkpoint_epochs:
            payload = _base_payload(config, mode, model, data, result_dir)
            payload.update(
                {
                    "model_config": _model_config(config),
                    "training_phase": "stage1",
                    "stage1_config": stage_cfg,
                    "history": history,
                    "epoch": epoch,
                }
            )
            _save_checkpoint(_epoch_checkpoint_path(config, mode, "stage1", epoch), payload, model)
    tracker.finish()
    pd.DataFrame(history).to_csv(result_dir / "hanwam_stage1_history.csv", index=False, encoding="utf-8-sig")
    payload = _base_payload(config, mode, model, data, result_dir)
    payload.update(
        {
            "model_config": _model_config(config),
            "training_phase": "stage1",
            "stage1_config": stage_cfg,
            "history": history,
        }
    )
    stage1_path = _stage1_checkpoint_path(config, mode)
    _save_checkpoint(stage1_path, payload, model)
    model.to(device)
    return model, stage1_path, history


def train_stage2(config: dict, mode: int, data: dict, device: torch.device, result_dir: Path, model: HanWAM | None = None) -> tuple[HanWAM, Path, list[dict]]:
    stage_cfg = _stage_cfg(config, "stage2")
    stage1_path = Path((config["train"].get("stage2") or {}).get("stage1_checkpoint") or _stage1_checkpoint_path(config, mode))
    if model is None:
        checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
        model = build_wam_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
    if not isinstance(model, HanWAM):
        raise TypeError("HanWAM stage2 requires a HanWAM checkpoint")
    model = model.to(device)
    model.freeze_world_model()
    optimizer = torch.optim.AdamW(
        [param for param in model.prober_parameters() if param.requires_grad],
        lr=float(stage_cfg["lr"]),
        weight_decay=float(stage_cfg["weight_decay"]),
    )
    train_loader = _loader(data["train"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    val_loader = _loader(data["val"], data["obs_norm"], data["action_norm"], data["physical_norm"], int(stage_cfg["batch_size"]))
    physical_weights = _physical_weights(config, data["physical_norm"], device)
    tracker = _tracker(config, mode, "stage2")
    history = []
    total_epochs = int(stage_cfg["epochs"])
    checkpoint_epochs = _checkpoint_epochs(stage_cfg, total_epochs)
    for epoch in range(1, int(stage_cfg["epochs"]) + 1):
        model.train()
        rows = []
        for obs_b, actions_b, _, physical_b in train_loader:
            obs_b = obs_b.to(device)
            actions_b = actions_b.to(device)
            physical_b = physical_b.to(device)
            loss, _ = model.prober_sequence_loss(
                obs_b,
                actions_b,
                physical_b,
                physical_weights=physical_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prober.parameters(), 5.0)
            optimizer.step()
            rows.append(float(loss.detach().cpu()))
        row = {
            "stage": "stage2",
            "epoch": epoch,
            "loss": float(np.mean(rows)),
        }
        if epoch == int(stage_cfg["epochs"]) or epoch % max(1, int(stage_cfg["epochs"]) // 5) == 0:
            row.update(_eval_stage2(model, val_loader, data["physical_norm"], device, physical_weights))
        history.append(row)
        tracker.log({f"stage2/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in checkpoint_epochs:
            payload = _base_payload(config, mode, model, data, result_dir)
            payload.update(
                {
                    "model_config": _model_config(config),
                    "training_phase": "stage2",
                    "stage1_checkpoint": str(stage1_path),
                    "stage2_config": stage_cfg,
                    "history": history,
                    "epoch": epoch,
                }
            )
            _save_checkpoint(_epoch_checkpoint_path(config, mode, "stage2", epoch), payload, model)
    tracker.finish()
    pd.DataFrame(history).to_csv(result_dir / "hanwam_stage2_history.csv", index=False, encoding="utf-8-sig")
    payload = _base_payload(config, mode, model, data, result_dir)
    payload.update(
        {
            "model_config": _model_config(config),
            "training_phase": "stage2",
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
    model = None
    if stage in {"stage1", "both"}:
        model, stage1_path, stage1_history = train_stage1(config, mode, data, device, result_dir)
        result["stage1_checkpoint"] = str(stage1_path)
        result["stage1_history"] = stage1_history
    if stage in {"stage2", "both"}:
        model, final_path, stage2_history = train_stage2(config, mode, data, device, result_dir, model=model)
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
